import numpy as np
import pytest
import tifffile

from hoshicore.engine.preflight import (
    FallbackProposal,
    PreflightReport,
    ResourceEstimate,
    apply_fallbacks,
    preflight_check,
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


class TestApplyFallbacks:
    def test_applies_proposals(self):
        report = PreflightReport(
            estimate=ResourceEstimate(5000, 0),
            available_memory_bytes=3000,
            available_disk_bytes=10000,
            proposed_fallbacks=[
                FallbackProposal("buffer_mode", "memory", "disk", "内存不足")
            ],
        )
        configs = {"buffer_mode": "memory"}
        apply_fallbacks(report, configs)
        assert configs["buffer_mode"] == "disk"
        assert len(report.applied_fallbacks) == 1
        assert "memory → disk" in report.applied_fallbacks[0]


class TestPreflightCheck:
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
        report = preflight_check(dag, {}, {})
        assert report.estimate.peak_memory_bytes == 0
        assert report.estimate.peak_disk_bytes == 0
        assert report.warnings == []

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
        report = preflight_check(dag, effective_configs, {"fnames": fnames})

        # 5 frames × 100×200×3 × 2 bytes = 600,000 bytes disk
        assert report.estimate.peak_disk_bytes == 600000
        assert report.estimate.peak_memory_bytes == 0
        assert report.non_chunk_mem == 0

    def test_memory_mode_triggers_warning_when_insufficient(self, tmp_path):
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
            report = preflight_check(
                dag, effective_configs,
                {"fnames": [str(path)] * 10})

        assert len(report.warnings) > 0
        assert len(report.proposed_fallbacks) == 1
        assert report.proposed_fallbacks[0].proposed_value == "disk"

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

        report = preflight_check(
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

        report = preflight_check(
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
            report = preflight_check(
                dag, {"buffer_mode": "memory"},
                {"fnames": [str(path)] * 10})

        assert report.proposed_fallbacks
        assert report.non_chunk_mem == 2000
        assert report.post_fallback_non_chunk_mem == 0
