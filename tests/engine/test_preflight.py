import os

import numpy as np
import pytest
import tifffile

from hoshicore.engine.preflight import (
    CheckResult,
    PreflightFix,
    PreflightIssue,
    PreflightReport,
    ResourceEstimate,
    apply_check_fixes,
    config_validity_check,
    run_preflight_checks,
    _resolve_node_configs,
)
from hoshicore.ops.base import BaseOp
from hoshicore.ops.sigma_clip_ops import DiskBufferWriterOp
from hoshicore.ops.trailstacker import MeanStackerOp


class _FixedMemOp(BaseOp):
    @classmethod
    def estimate_resources(cls, configs, frame_bytes, n_frames,
                           dtype_bytes=None):
        _ = configs, frame_bytes, n_frames, dtype_bytes
        return (1000, 0)


class _PlannedMemOp(BaseOp):
    CHUNK_PLANNED = True

    @classmethod
    def estimate_resources(cls, configs, frame_bytes, n_frames,
                           dtype_bytes=None):
        _ = configs, frame_bytes, n_frames, dtype_bytes
        return (5000, 0)


class _SequenceOutputOp(BaseOp):
    OUTPUTS = {"result": {"type": "sequence"}}


class TestEstimateResources:
    def test_base_op_default(self):
        assert BaseOp.estimate_resources({}, 1000, 10) == (0, 0)

    def test_disk_buffer_writer_disk_mode(self):
        mem, disk = DiskBufferWriterOp.estimate_resources(
            {"buffer_mode": "disk"}, 1000, 50)
        assert mem == 0
        assert disk == 50000

    def test_disk_buffer_writer_memory_mode(self):
        mem, disk = DiskBufferWriterOp.estimate_resources(
            {"buffer_mode": "memory"}, 2000, 30)
        assert mem == 60000
        assert disk == 0

    def test_disk_buffer_writer_replay_mode(self):
        mem, disk = DiskBufferWriterOp.estimate_resources(
            {"buffer_mode": "replay"}, 2000, 30)
        assert mem == 0
        assert disk == 0

    def test_disk_buffer_writer_none_frames(self):
        mem, disk = DiskBufferWriterOp.estimate_resources(
            {"buffer_mode": "disk"}, 1000, None)
        assert mem == 0
        assert disk == 0

    def test_mean_stacker_estimates_fgp_dtype_bytes(self):
        frame_bytes = 100 * 200 * 3 * 2
        # uint16 + int_weight=True: sum_mu uint64 + square_sum float64 + n uint32
        mem, disk = MeanStackerOp.estimate_resources(
            {"int_weight": True}, frame_bytes, 5, dtype_bytes=2)
        assert mem == 100 * 200 * 3 * (8 + 8 + 4)
        assert disk == 0


class TestResolveNodeConfigs:
    def test_configs_link(self):
        node_spec = {"configs": {"buffer_mode": "configs.buffer_mode"}}
        effective = {"buffer_mode": "memory"}
        result = _resolve_node_configs(node_spec, effective, DiskBufferWriterOp)
        assert result["buffer_mode"] == "memory"

    def test_literal_value(self):
        node_spec = {"configs": {"buffer_mode": "disk"}}
        result = _resolve_node_configs(node_spec, {}, DiskBufferWriterOp)
        assert result["buffer_mode"] == "disk"

    def test_cross_node_link_uses_default(self):
        node_spec = {"configs": {"buffer_mode": "other_node.output"}}
        result = _resolve_node_configs(node_spec, {}, DiskBufferWriterOp)
        # cross-node link skipped → falls back to Op default
        assert result["buffer_mode"] == "disk"

    def test_missing_key_uses_default(self):
        node_spec = {"configs": {}}
        result = _resolve_node_configs(node_spec, {}, DiskBufferWriterOp)
        assert result["buffer_mode"] == "disk"
        assert result["temp_path"] is None


class TestApplyCheckFixes:
    def test_applies_fixes(self):
        result = CheckResult(
            check_name="test",
            issues=[
                PreflightIssue(
                    severity="warning",
                    code="resource.memory",
                    message="内存不足",
                    fix=PreflightFix("buffer_mode", "memory", "disk", "内存不足"),
                )
            ],
        )
        configs = {"buffer_mode": "memory"}
        msgs = apply_check_fixes(result, configs)
        assert configs["buffer_mode"] == "disk"
        assert len(msgs) == 1
        assert "memory → disk" in msgs[0]

    def test_skips_issues_without_fix(self):
        result = CheckResult(
            check_name="test",
            issues=[
                PreflightIssue(severity="warning", code="x", message="纯警告"),
            ],
        )
        configs = {"buffer_mode": "disk"}
        msgs = apply_check_fixes(result, configs)
        assert configs["buffer_mode"] == "disk"
        assert msgs == []


class TestRunPreflightChecks:
    def _make_dag(self, nodes=None, exec_order=None):
        from hoshicore.engine.build import ValidatedDag
        return ValidatedDag(
            nodes=nodes or {},
            global_inputs={},
            global_configs={},
            output_links={},
            node_deps={},
            exec_order=exec_order or [],
        )

    def test_no_fnames_returns_empty(self):
        dag = self._make_dag()
        report, results = run_preflight_checks(dag, {}, {})
        assert report.estimate.peak_memory_bytes == 0
        assert report.estimate.peak_disk_bytes == 0
        assert results[0].issues == []  # resource check result

    def test_invalid_temp_path_does_not_crash(self, tmp_path):
        """temp_path 不存在时回退到系统默认路径计算磁盘空间，不抛异常。"""
        frame = tmp_path / "frame.tif"
        tifffile.imwrite(str(frame), np.zeros((10, 10), dtype=np.uint16))
        dag = self._make_dag()
        missing = str(tmp_path / "nonexistent_cache")
        report, results = run_preflight_checks(
            dag, {"buffer_mode": "disk", "temp_path": missing},
            {"fnames": [str(frame)]})
        assert report.available_disk_bytes > 0

    def test_estimates_with_tiff_input(self, tmp_path):
        for i in range(5):
            path = tmp_path / f"frame_{i:03d}.tif"
            tifffile.imwrite(str(path), np.zeros((100, 200, 3), dtype=np.uint16))

        fnames = sorted([str(p) for p in tmp_path.glob("*.tif")])

        dag = self._make_dag(
            nodes={
                "buffer": {
                    "op": "DiskBufferWriterOp",
                    "configs": {"buffer_mode": "configs.buffer_mode"},
                }
            },
            exec_order=["buffer"],
        )

        effective_configs = {"buffer_mode": "disk"}
        report, results = run_preflight_checks(
            dag, effective_configs, {"fnames": fnames})

        # 5 frames × 100×200×3 × 2 bytes = 600,000 bytes disk
        assert report.estimate.peak_disk_bytes == 600000
        assert report.estimate.peak_memory_bytes == 0
        assert report.non_chunk_mem == 0

    def test_memory_mode_triggers_issue_when_insufficient(self, tmp_path):
        from unittest.mock import patch

        path = tmp_path / "frame.tif"
        tifffile.imwrite(str(path), np.zeros((100, 100, 3), dtype=np.uint16))

        dag = self._make_dag(
            nodes={
                "buffer": {
                    "op": "DiskBufferWriterOp",
                    "configs": {"buffer_mode": "configs.buffer_mode"},
                }
            },
            exec_order=["buffer"],
        )

        class FakeVMem:
            available = 100_000  # 100KB — way too low

        with patch("hoshicore.engine.preflight.psutil") as mock_psutil:
            mock_psutil.virtual_memory.return_value = FakeVMem()
            effective_configs = {"buffer_mode": "memory"}
            report, results = run_preflight_checks(
                dag, effective_configs,
                {"fnames": [str(path)] * 10})

        resource_result = results[0]
        assert len(resource_result.issues) > 0
        fixes = [i.fix for i in resource_result.issues if i.fix]
        assert len(fixes) == 1
        assert fixes[0].proposed_value == "disk"

    def test_chunk_planned_op_excluded_from_non_chunk_mem(self, tmp_path):
        path = tmp_path / "frame.tif"
        tifffile.imwrite(str(path), np.zeros((10, 10), dtype=np.uint16))

        dag = self._make_dag(
            nodes={
                "fixed": {"op": "_FixedMemOp", "configs": {}},
                "planned": {"op": "_PlannedMemOp", "configs": {}},
            },
            exec_order=["fixed", "planned"],
        )

        report, results = run_preflight_checks(
            dag, {}, {"fnames": [str(path)]},
            op_registry={
                "_FixedMemOp": _FixedMemOp,
                "_PlannedMemOp": _PlannedMemOp,
            })

        assert report.estimate.peak_memory_bytes == 6000
        assert report.non_chunk_mem == 1000

    def test_queue_overhead_counts_as_non_chunk_mem(self, tmp_path):
        path = tmp_path / "frame.tif"
        tifffile.imwrite(str(path), np.zeros((10, 10), dtype=np.uint16))

        dag = self._make_dag(
            nodes={"seq": {"op": "_SequenceOutputOp", "configs": {}}},
            exec_order=["seq"],
        )

        report, results = run_preflight_checks(
            dag, {}, {"fnames": [str(path)]},
            op_registry={"_SequenceOutputOp": _SequenceOutputOp})

        assert report.estimate.peak_memory_bytes == 200
        assert report.non_chunk_mem == 200

    def test_post_fallback_non_chunk_mem_is_computed_but_not_applied(self, tmp_path):
        from unittest.mock import patch

        path = tmp_path / "frame.tif"
        tifffile.imwrite(str(path), np.zeros((10, 10), dtype=np.uint16))

        dag = self._make_dag(
            nodes={
                "buffer": {
                    "op": "DiskBufferWriterOp",
                    "configs": {"buffer_mode": "configs.buffer_mode"},
                }
            },
            exec_order=["buffer"],
        )

        class FakeVMem:
            available = 100_000

        with patch("hoshicore.engine.preflight.psutil") as mock_psutil:
            mock_psutil.virtual_memory.return_value = FakeVMem()
            report, results = run_preflight_checks(
                dag, {"buffer_mode": "memory"},
                {"fnames": [str(path)] * 10})

        resource_result = results[0]
        assert resource_result.has_fix
        assert report.non_chunk_mem == 2000
        assert report.post_fallback_non_chunk_mem == 0


class TestConfigValidityCheck:
    def _make_dag(self):
        from hoshicore.engine.build import ValidatedDag
        return ValidatedDag(
            nodes={}, global_inputs={}, global_configs={},
            output_links={}, node_deps={}, exec_order=[],
        )

    def test_no_temp_path_configured(self):
        dag = self._make_dag()
        result = config_validity_check(dag, {"buffer_mode": "disk"}, {})
        assert result.issues == []

    def test_memory_mode_skips_check(self, tmp_path):
        # even if temp_path is invalid, memory mode doesn't need it
        dag = self._make_dag()
        result = config_validity_check(
            dag, {"buffer_mode": "memory", "temp_path": "/nonexistent/path"}, {})
        assert result.issues == []

    def test_replay_mode_skips_check(self, tmp_path):
        dag = self._make_dag()
        result = config_validity_check(
            dag, {"buffer_mode": "replay", "temp_path": "/nonexistent/path"}, {})
        assert result.issues == []

    def test_missing_temp_path(self, tmp_path):
        dag = self._make_dag()
        missing = str(tmp_path / "nonexistent_dir")
        result = config_validity_check(
            dag, {"buffer_mode": "disk", "temp_path": missing}, {})
        assert len(result.issues) == 1
        assert result.issues[0].code == "config.cache_path.missing"
        assert result.issues[0].severity == "error"
        assert result.issues[0].fix is not None
        assert result.issues[0].fix.config_key == "temp_path"
        assert result.issues[0].fix.current_value == missing

    def test_not_writable_temp_path(self, tmp_path):
        dag = self._make_dag()
        locked = tmp_path / "locked_dir"
        locked.mkdir()
        original_mode = locked.stat().st_mode
        try:
            locked.chmod(0o444)
            # skip on Windows where chmod has no effect on directories
            if os.access(str(locked), os.W_OK):
                pytest.skip("chmod has no effect on this platform")
            result = config_validity_check(
                dag, {"buffer_mode": "disk", "temp_path": str(locked)}, {})
            assert len(result.issues) == 1
            assert result.issues[0].code == "config.cache_path.not_writable"
            assert result.issues[0].fix is not None
        finally:
            locked.chmod(original_mode)

    def test_valid_temp_path(self, tmp_path):
        dag = self._make_dag()
        result = config_validity_check(
            dag, {"buffer_mode": "disk", "temp_path": str(tmp_path)}, {})
        assert result.issues == []

    def test_fix_proposes_system_default(self, tmp_path):
        import tempfile
        dag = self._make_dag()
        missing = str(tmp_path / "nonexistent")
        result = config_validity_check(
            dag, {"buffer_mode": "disk", "temp_path": missing}, {})
        assert result.issues[0].fix.proposed_value == tempfile.gettempdir()
