import numpy as np
import tifffile

from hoshicore.engine.build import ValidatedDag
from hoshicore.engine.preflight import PreflightReport, ResourceEstimate
from hoshicore.engine.runtime_plan import apply_runtime_plan, plan_runtime
import hoshicore.engine.runtime_plan as runtime_plan_module
from hoshicore.ops.base import BaseOp
from hoshicore.ops.sigma_clip_ops import MedianReduceOp


def _make_dag() -> ValidatedDag:
    return ValidatedDag(
        nodes={
            "median": {
                "op": "MedianReduceOp",
                "configs": {"chunk_rows": "configs.chunk_rows"},
                "outputs": {"result": {"type": "image"}},
            }
        },
        global_inputs={},
        global_configs={},
        output_links={},
        node_deps={},
        exec_order=["median"],
    )


def _make_no_chunk_dag() -> ValidatedDag:
    return ValidatedDag(
        nodes={
            "noop": {
                "op": "NoChunkOp",
                "configs": {},
                "outputs": {"result": {"type": "image"}},
            }
        },
        global_inputs={},
        global_configs={},
        output_links={},
        node_deps={},
        exec_order=["noop"],
    )


def _report(non_chunk_mem: int = 0) -> PreflightReport:
    return PreflightReport(
        estimate=ResourceEstimate(0, 0),
        available_memory_bytes=0,
        available_disk_bytes=0,
        non_chunk_mem=non_chunk_mem,
    )


def _mock_available_memory(monkeypatch, budget: int, non_chunk_mem: int = 0) -> None:
    available = int(
        (runtime_plan_module.MEMORY_FIXED_OVERHEAD + budget + non_chunk_mem) /
        runtime_plan_module.MEMORY_SAFETY_FACTOR) + 16

    class FakeVMem:
        pass

    FakeVMem.available = available
    monkeypatch.setattr(
        runtime_plan_module.psutil,
        "virtual_memory",
        lambda: FakeVMem())


class NoChunkOp(BaseOp):
    pass


def _median_registry() -> dict[str, type[BaseOp]]:
    return {"MedianReduceOp": MedianReduceOp}


def test_runtime_planner_disabled_returns_empty(tmp_path):
    path = tmp_path / "frame.tif"
    tifffile.imwrite(str(path), np.zeros((100, 20, 3), dtype=np.uint16))

    plan = plan_runtime(
        _make_dag(),
        {"runtime_planner": False},
        {"fnames": [str(path)] * 4},
        op_registry=_median_registry(),
    )

    assert plan.config_overrides == {}
    assert plan.decisions == []


def test_runtime_planner_sets_chunk_rows_when_enabled(tmp_path, monkeypatch):
    path = tmp_path / "frame.tif"
    tifffile.imwrite(str(path), np.zeros((512, 10), dtype=np.uint16))
    non_chunk_mem = 1000
    monkeypatch.setattr(
        runtime_plan_module.psutil,
        "virtual_memory",
        lambda: type("FakeVMem", (), {"available": 100_000})())

    plan = plan_runtime(
        _make_dag(),
        {
            "runtime_planner": True,
        },
        {"fnames": [str(path)] * 4},
        op_registry=_median_registry(),
        preflight_report=_report(non_chunk_mem),
    )

    assert plan.config_overrides["chunk_rows"] == 1
    assert plan.decisions[0].key == "chunk_rows"


def test_runtime_planner_uses_preflight_formula(tmp_path, monkeypatch):
    path = tmp_path / "frame.tif"
    tifffile.imwrite(str(path), np.zeros((512, 10), dtype=np.uint16))
    non_chunk_mem = 1000
    # 灰度图 row_bytes = 10 * 1 * 2，Median cost = (4 + 1) * 20 = 100 bytes/row.
    _mock_available_memory(monkeypatch, budget=12800, non_chunk_mem=non_chunk_mem)

    plan = plan_runtime(
        _make_dag(),
        {"runtime_planner": True},
        {"fnames": [str(path)] * 4},
        op_registry=_median_registry(),
        preflight_report=_report(non_chunk_mem),
    )

    assert plan.config_overrides["chunk_rows"] == 128


def test_runtime_planner_requires_preflight_report(tmp_path):
    path = tmp_path / "frame.tif"
    tifffile.imwrite(str(path), np.zeros((100, 20, 3), dtype=np.uint16))

    plan = plan_runtime(
        _make_dag(),
        {"runtime_planner": True},
        {"fnames": [str(path)] * 4},
        op_registry=_median_registry(),
    )

    assert plan.config_overrides == {}


def test_runtime_planner_keeps_explicit_chunk_rows(tmp_path):
    path = tmp_path / "frame.tif"
    tifffile.imwrite(str(path), np.zeros((100, 20, 3), dtype=np.uint16))

    plan = plan_runtime(
        _make_dag(),
        {"runtime_planner": True, "chunk_rows": 32},
        {"fnames": [str(path)] * 4},
        op_registry=_median_registry(),
        preflight_report=_report(),
        explicit_config_keys={"chunk_rows"},
    )

    assert plan.config_overrides == {}


def test_runtime_planner_allows_auto_chunk_rows(tmp_path, monkeypatch):
    path = tmp_path / "frame.tif"
    tifffile.imwrite(str(path), np.zeros((100, 20, 3), dtype=np.uint16))
    monkeypatch.setattr(
        runtime_plan_module.psutil,
        "virtual_memory",
        lambda: type("FakeVMem", (), {"available": 100_000})())

    plan = plan_runtime(
        _make_dag(),
        {
            "runtime_planner": True,
            "chunk_rows": "auto",
        },
        {"fnames": [str(path)] * 4},
        op_registry=_median_registry(),
        preflight_report=_report(),
        explicit_config_keys={"chunk_rows"},
    )

    configs = {"chunk_rows": "auto"}
    apply_runtime_plan(plan, configs)
    assert configs["chunk_rows"] == 1


def test_runtime_planner_no_chunk_planned_op_returns_empty(tmp_path):
    path = tmp_path / "frame.tif"
    tifffile.imwrite(str(path), np.zeros((100, 20, 3), dtype=np.uint16))

    plan = plan_runtime(
        _make_no_chunk_dag(),
        {"runtime_planner": True},
        {"fnames": [str(path)] * 4},
        {"NoChunkOp": NoChunkOp},
        preflight_report=_report(),
    )

    assert plan.config_overrides == {}


def test_runtime_planner_clamps_to_min_when_budget_is_exhausted(tmp_path, monkeypatch):
    path = tmp_path / "frame.tif"
    tifffile.imwrite(str(path), np.zeros((100, 20, 3), dtype=np.uint16))
    monkeypatch.setattr(
        runtime_plan_module.psutil,
        "virtual_memory",
        lambda: type("FakeVMem", (), {"available": 100_000})())

    plan = plan_runtime(
        _make_dag(),
        {"runtime_planner": True},
        {"fnames": [str(path)] * 4},
        op_registry=_median_registry(),
        preflight_report=_report(non_chunk_mem=10_000),
    )

    assert plan.config_overrides["chunk_rows"] == 1
