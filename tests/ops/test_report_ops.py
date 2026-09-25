import asyncio
import json
import os
import tempfile

import numpy as np
import pytest

from hoshicore.component.norma.bundle import (BAAlignmentPlan, FrameAlignment,
                                               FrameAlignmentStatus)
from hoshicore.component.norma.types import CameraModel, Distortion, Intrinsics
from hoshicore.component.queue import RichContextQueue
from hoshicore.ops.bundle_ops import (BundleWindowFrameReport,
                                      BundleWindowReport)
from hoshicore.ops.report_ops import (BundleFrameRotationExportOp,
                                      BundleFrameRotationExportSaveOp,
                                      build_bundle_frame_rotation_export,
                                      format_bundle_window_report)


def _frame(index, status, reason=None):
    return FrameAlignment(
        index=index,
        status=status,
        rotation_ref_to_src=None,
        pose_source="test",
        reason=reason,
    )


def test_format_bundle_window_report_derives_all_skip_reasons():
    plan = BAAlignmentPlan(
        reference_frame_index=0,
        shared_camera=CameraModel(
            Intrinsics(20.0, 36.0, 24.0, 120, 80), Distortion()),
        frames=(
            _frame(0, FrameAlignmentStatus.SOLVED),
            _frame(1, FrameAlignmentStatus.EXCLUDED, "weak outer field"),
            _frame(2, FrameAlignmentStatus.SOLVED),
            _frame(3, FrameAlignmentStatus.SOLVED),
        ),
        accepted_edge_count=2,
        rejected_edge_count=1,
        active_camera_parameter_count=0,
        observability_condition=None,
    )
    report = BundleWindowReport((
        BundleWindowFrameReport(0, (0, 2)),
        BundleWindowFrameReport(2, (0, 2, 3)),
    ))

    summary = format_bundle_window_report(plan, report)

    assert "4 input, 3 solved, 1 excluded" in summary
    assert "Shared camera:" in summary
    assert "Bundle edges: 2 accepted, 1 rejected" in summary
    assert "2 emitted, 1 insufficient contributors" in summary
    assert "min=2, median=2.5, max=3" in summary
    assert "1: weak outer field" in summary
    assert "3: insufficient contributors" in summary


def _rotation_frame(index, status, rotation=None, reason=None):
    return FrameAlignment(
        index=index,
        status=status,
        rotation_ref_to_src=rotation,
        pose_source="bundle" if rotation is not None else "none",
        residual_p90_rad=0.001 if rotation is not None else None,
        incident_edge_count=3 if rotation is not None else 0,
        reason=reason,
    )


def _rotation_plan():
    return BAAlignmentPlan(
        reference_frame_index=0,
        shared_camera=CameraModel(
            Intrinsics(20.0, 36.0, 24.0, 120, 80), Distortion(k1=0.01)),
        frames=(
            _rotation_frame(0, FrameAlignmentStatus.SOLVED, np.eye(3)),
            _rotation_frame(1, FrameAlignmentStatus.EXCLUDED,
                           reason="weak outer field"),
            _rotation_frame(2, FrameAlignmentStatus.SOLVED, np.eye(3) * 0.99),
        ),
        accepted_edge_count=2,
        rejected_edge_count=1,
        active_camera_parameter_count=0,
        observability_condition=1.5,
    )


def test_build_bundle_frame_rotation_export_covers_camera_and_frames():
    export = build_bundle_frame_rotation_export(_rotation_plan())

    assert export.camera.model_type == "CameraModel"
    assert export.camera.focal_length_mm == 20.0
    assert export.camera.accepted_edge_count == 2
    assert export.camera.observability_condition == 1.5

    assert [f.index for f in export.frames] == [0, 1, 2]
    solved, excluded, _ = export.frames
    assert solved.status == "solved"
    assert solved.rotation_ref_to_src is not None
    assert excluded.status == "excluded"
    assert excluded.rotation_ref_to_src is None
    assert excluded.reason == "weak outer field"


def test_build_bundle_frame_rotation_export_rotation_is_read_only():
    export = build_bundle_frame_rotation_export(_rotation_plan())

    with pytest.raises(ValueError):
        export.frames[0].rotation_ref_to_src[0, 0] = 5.0


@pytest.mark.asyncio
async def test_bundle_frame_rotation_export_op_broadcasts_export():
    op = BundleFrameRotationExportOp("frame_rotation_export")
    sink = RichContextQueue(maxsize=1)
    op.outputs["export"].append(sink)
    await op.config["alignment_plan"].put(_rotation_plan())

    task = asyncio.create_task(op.execute())
    export = await asyncio.wait_for(sink.get(), timeout=0.1)
    await asyncio.wait_for(task, timeout=0.1)

    assert len(export.frames) == 3
    assert export.camera.model_type == "CameraModel"


@pytest.mark.asyncio
async def test_bundle_frame_rotation_export_save_op_writes_json():
    export = build_bundle_frame_rotation_export(_rotation_plan())

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = os.path.join(tmp_dir, "nested", "rotations.json")

        op = BundleFrameRotationExportSaveOp("frame_rotation_export_save")
        sink = RichContextQueue(maxsize=1)
        op.outputs["result"].append(sink)
        await op.config["export"].put(export)
        await op.config["output_path"].put(output_path)

        task = asyncio.create_task(op.execute())
        result_path = await asyncio.wait_for(sink.get(), timeout=0.1)
        await asyncio.wait_for(task, timeout=0.1)

        assert result_path == output_path
        with open(output_path, encoding="utf-8") as f:
            payload = json.load(f)

    assert payload["camera"]["focal_length_mm"] == 20.0
    assert payload["frames"][0]["rotation_ref_to_src"] == [
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    assert payload["frames"][1]["rotation_ref_to_src"] is None
    assert payload["frames"][1]["reason"] == "weak outer field"
