"""Runtime execution planning helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import psutil
from loguru import logger

from ..component.image_io import peek_shape
from ..ops.base import BaseOp
from .build import ValidatedDag
from .preflight import PreflightReport
from .registry import REGISTERED_OP

MEMORY_SAFETY_FACTOR = 0.7
MEMORY_FIXED_OVERHEAD = 200 * 1024 * 1024
DEFAULT_MIN_CHUNK_ROWS = 1
DEFAULT_MAX_CHUNK_ROWS = 512


@dataclass(frozen=True)
class RuntimePlanDecision:
    key: str
    value: Any
    reason: str


@dataclass(frozen=True)
class RuntimePlan:
    config_overrides: dict[str, Any] = field(default_factory=dict)
    decisions: list[RuntimePlanDecision] = field(default_factory=list)


def plan_runtime(
    dag: ValidatedDag,
    effective_configs: dict[str, Any],
    global_inputs: dict[str, Any],
    op_registry: Optional[dict[str, type[BaseOp]]] = None,
    *,
    preflight_report: Optional[PreflightReport] = None,
    explicit_config_keys: set[str] | None = None,
) -> RuntimePlan:
    if not _planner_enabled(effective_configs.get("runtime_planner", False)):
        return RuntimePlan()

    registry = op_registry or REGISTERED_OP
    chunk_ops = _find_chunk_planned_ops(dag, registry)
    if not chunk_ops:
        return RuntimePlan()

    explicit_keys = explicit_config_keys or set()
    current_chunk_rows = effective_configs.get("chunk_rows")
    if "chunk_rows" in explicit_keys and current_chunk_rows != "auto":
        return RuntimePlan()

    shape_info = _peek_input_shape(global_inputs)
    if shape_info is None:
        return RuntimePlan()
    shape, dtype_bytes, n_frames = shape_info
    chunk_rows = _plan_chunk_rows(
        shape, dtype_bytes, n_frames, effective_configs,
        chunk_ops, preflight_report)
    if chunk_rows is None:
        return RuntimePlan()

    non_chunk_mem = preflight_report.non_chunk_mem if preflight_report else 0
    reason = (
        f"planned chunk_rows={chunk_rows} for shape={shape}, "
        f"frames={n_frames}, dtype_bytes={dtype_bytes}, "
        f"non_chunk_mem={non_chunk_mem}"
    )
    return RuntimePlan(
        config_overrides={"chunk_rows": chunk_rows},
        decisions=[RuntimePlanDecision("chunk_rows", chunk_rows, reason)],
    )


def apply_runtime_plan(plan: RuntimePlan, effective_configs: dict[str, Any]) -> None:
    for key, value in plan.config_overrides.items():
        old_value = effective_configs.get(key)
        effective_configs[key] = value
        logger.info(f"[RuntimePlan] {key}: {old_value} -> {value}")
    for decision in plan.decisions:
        logger.info(f"[RuntimePlan] {decision.reason}")


def _planner_enabled(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "auto"}
    return bool(value)


def _find_chunk_planned_ops(
    dag: ValidatedDag,
    registry: dict[str, type[BaseOp]],
) -> list[type[BaseOp]]:
    chunk_ops: list[type[BaseOp]] = []
    for node_name in dag.exec_order:
        node_spec = dag.nodes[node_name]
        op_cls = registry.get(node_spec["op"])
        if op_cls is None:
            continue
        if getattr(op_cls, "CHUNK_PLANNED", False):
            chunk_ops.append(op_cls)
    return chunk_ops


def _peek_input_shape(global_inputs: dict[str, Any]) -> tuple[tuple[int, ...], int, int] | None:
    fnames = global_inputs.get("fnames")
    if not fnames or not isinstance(fnames, (list, tuple)):
        return None
    if len(fnames) == 0:
        return None
    try:
        shape, dtype_bytes = peek_shape(fnames[0])
    except (FileNotFoundError, ValueError, OSError) as exc:
        logger.warning(f"[RuntimePlan] cannot peek first frame: {exc}")
        return None
    return tuple(shape), int(dtype_bytes), len(fnames)


def _plan_chunk_rows(
    shape: tuple[int, ...],
    dtype_bytes: int,
    n_frames: int,
    configs: dict[str, Any],
    chunk_ops: list[type[BaseOp]],
    preflight_report: PreflightReport | None,
) -> int | None:
    if preflight_report is None:
        return None
    if len(shape) < 2 or n_frames <= 0 or dtype_bytes <= 0:
        return None

    height = int(shape[0])
    width = int(shape[1])
    channels = int(shape[2]) if len(shape) >= 3 else 1
    row_bytes = width * channels * dtype_bytes
    if height <= 0 or row_bytes <= 0:
        return None

    cost_per_row = sum(
        op_cls.chunk_cost_per_row(n_frames, row_bytes, dtype_bytes)
        for op_cls in chunk_ops
    )
    if cost_per_row <= 0:
        return None

    # preflight 负责估算非 chunk 常驻内存；planner 只拿剩余预算分配 chunk_rows。
    chunk_budget = _memory_budget_bytes() - int(preflight_report.non_chunk_mem)
    if chunk_budget <= 0:
        rows = DEFAULT_MIN_CHUNK_ROWS
    else:
        rows = max(DEFAULT_MIN_CHUNK_ROWS, chunk_budget // cost_per_row)
    rows = _round_down_to_multiple(int(rows), 16)
    min_rows = _positive_int(configs.get("runtime_planner_min_chunk_rows"), DEFAULT_MIN_CHUNK_ROWS)
    max_rows = _positive_int(configs.get("runtime_planner_max_chunk_rows"), DEFAULT_MAX_CHUNK_ROWS)
    rows = max(min_rows, min(rows, max_rows, height))
    return max(1, int(rows))


def _memory_budget_bytes() -> int:
    avail_mem = psutil.virtual_memory().available
    return int(avail_mem * MEMORY_SAFETY_FACTOR) - MEMORY_FIXED_OVERHEAD


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _round_down_to_multiple(value: int, multiple: int) -> int:
    if value < multiple:
        return value
    return max(multiple, (value // multiple) * multiple)
