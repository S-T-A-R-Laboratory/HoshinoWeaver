"""Terminal report formatters for command-line DAGs."""
from __future__ import annotations

import dataclasses
import json
import os
from statistics import median
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
from loguru import logger

from ..component.norma.bundle import (BAAlignmentPlan, FrameAlignmentStatus)
from ..engine.registry import register_op
from .base import BaseOp
from .bundle_ops import BundleWindowReport


def format_bundle_window_report(
    plan: BAAlignmentPlan,
    report: BundleWindowReport,
) -> str:
    """Return a concise, self-contained BA window summary."""
    if not isinstance(plan, BAAlignmentPlan):
        raise TypeError("alignment_plan must be a BAAlignmentPlan")
    if not isinstance(report, BundleWindowReport):
        raise TypeError("window_report must be a BundleWindowReport")

    emitted = {
        frame.center_index: frame for frame in report.frames
        if frame.status == "ready"
    }
    excluded = [
        frame for frame in plan.frames
        if frame.status == FrameAlignmentStatus.EXCLUDED
    ]
    solved = [
        frame for frame in plan.frames
        if frame.status == FrameAlignmentStatus.SOLVED
    ]
    explicit_insufficient = [
        frame for frame in report.frames
        if frame.status == "insufficient_contributors"
    ]
    reported_indices = {frame.center_index for frame in report.frames}
    legacy_insufficient = [
        frame for frame in solved
        if frame.index not in reported_indices
    ]
    contributor_counts = [
        len(frame.contributor_indices) for frame in emitted.values()
    ]

    lines = [
        f"Bundle alignment: {len(plan.frames)} input, {len(solved)} solved, "
        f"{len(excluded)} excluded",
        "Shared camera: "
        f"{type(plan.shared_camera).__name__}, "
        f"focal={plan.shared_camera.intrinsics.focal_length_mm:.6g}mm, "
        f"distortion={plan.shared_camera.distortion.to_cv2().tolist()}",
        f"Bundle edges: {plan.accepted_edge_count} accepted, "
        f"{plan.rejected_edge_count} rejected, "
        f"condition={plan.observability_condition}",
        f"Camera solve: {plan.camera_solve_mode}",
        f"Window output: {len(emitted)} emitted, "
        f"{len(explicit_insufficient) + len(legacy_insufficient)} "
        "insufficient contributors",
    ]
    if plan.camera_fallback_reason:
        lines.append(
            f"Camera fallback reason: {plan.camera_fallback_reason}")
    if contributor_counts:
        lines.append(
            "Contributors: "
            f"min={min(contributor_counts)}, "
            f"median={median(contributor_counts):g}, "
            f"max={max(contributor_counts)}")
    if excluded:
        lines.append("Excluded frames:")
        lines.extend(
            f"  {frame.index}: {frame.reason or 'excluded by bundle adjustment'}"
            for frame in excluded
        )
    if explicit_insufficient or legacy_insufficient:
        lines.append("Skipped window centers:")
        lines.extend(
            f"  {frame.center_index}: {frame.reason or 'insufficient contributors'}"
            for frame in explicit_insufficient
        )
        lines.extend(
            f"  {frame.index}: insufficient contributors"
            for frame in legacy_insufficient
        )
    return "\n".join(lines)


@register_op()
class BundleWindowReportDisplayOp(BaseOp):
    """Log a window summary and expose the same text for future frontends."""

    CONFIGS: dict[str, Any] = {
        "alignment_plan": {"type": "object", "required": True},
        "window_report": {"type": "object", "required": True},
    }
    OUTPUTS: dict[str, Any] = {
        "summary": {"type": "str"},
    }

    async def _async_execute(self, configs: dict[str, Any]) -> None:
        summary = format_bundle_window_report(
            configs["alignment_plan"], configs["window_report"])
        logger.info("\n{}", summary)
        await self._broadcast_outputs({"summary": summary})


@dataclasses.dataclass(frozen=True)
class BundleFrameRotationEntry:
    """Per-frame pose diagnostic derived from a ``BAAlignmentPlan``.

    ``rotation_ref_to_src`` maps reference-frame rays to this frame's rays
    (see ``FrameAlignment``); it is relative to ``BAAlignmentPlan.reference_frame_index``,
    not an absolute sky pointing.
    """

    index: int
    status: str
    pose_source: str
    rotation_ref_to_src: Optional[NDArray[np.float64]]
    residual_p90_rad: Optional[float]
    incident_edge_count: int
    reason: Optional[str]


@dataclasses.dataclass(frozen=True)
class BundleCameraSummary:
    """Shared camera calibration and solve diagnostics for one BA plan."""

    model_type: str
    focal_length_mm: float
    sensor_width_mm: float
    sensor_height_mm: float
    image_width_px: int
    image_height_px: int
    principal_point_px: tuple[float, float]
    distortion_cv2: NDArray[np.float64]
    reference_frame_index: int
    accepted_edge_count: int
    rejected_edge_count: int
    active_camera_parameter_count: int
    observability_condition: Optional[float]
    camera_solve_mode: str
    camera_fallback_reason: Optional[str]


@dataclasses.dataclass(frozen=True)
class BundleFrameRotationExport:
    """Structured diagnostic export: overall camera + every frame's pose."""

    camera: BundleCameraSummary
    frames: tuple[BundleFrameRotationEntry, ...]


def build_bundle_frame_rotation_export(
        plan: BAAlignmentPlan) -> BundleFrameRotationExport:
    """Derive a structured per-frame rotation + camera export from a BA plan."""
    if not isinstance(plan, BAAlignmentPlan):
        raise TypeError("alignment_plan must be a BAAlignmentPlan")

    camera = plan.shared_camera
    intrinsics = camera.intrinsics
    distortion_cv2 = camera.distortion.to_cv2()
    distortion_cv2.setflags(write=False)

    camera_summary = BundleCameraSummary(
        model_type=type(camera).__name__,
        focal_length_mm=intrinsics.focal_length_mm,
        sensor_width_mm=intrinsics.sensor_width_mm,
        sensor_height_mm=intrinsics.sensor_height_mm,
        image_width_px=intrinsics.image_width_px,
        image_height_px=intrinsics.image_height_px,
        principal_point_px=intrinsics.principal_point_px,
        distortion_cv2=distortion_cv2,
        reference_frame_index=plan.reference_frame_index,
        accepted_edge_count=plan.accepted_edge_count,
        rejected_edge_count=plan.rejected_edge_count,
        active_camera_parameter_count=plan.active_camera_parameter_count,
        observability_condition=plan.observability_condition,
        camera_solve_mode=plan.camera_solve_mode,
        camera_fallback_reason=plan.camera_fallback_reason,
    )

    def _readonly(rotation: Optional[NDArray[np.float64]]):
        if rotation is None:
            return None
        rotation = np.asarray(rotation, dtype=np.float64)
        if rotation.flags.writeable:
            rotation = rotation.copy()
            rotation.setflags(write=False)
        return rotation

    frames = tuple(
        BundleFrameRotationEntry(
            index=frame.index,
            status=frame.status.value,
            pose_source=frame.pose_source,
            rotation_ref_to_src=_readonly(frame.rotation_ref_to_src),
            residual_p90_rad=frame.residual_p90_rad,
            incident_edge_count=frame.incident_edge_count,
            reason=frame.reason,
        ) for frame in plan.frames)

    return BundleFrameRotationExport(camera=camera_summary, frames=frames)


@register_op()
class BundleFrameRotationExportOp(BaseOp):
    """Development/debug diagnostic: structured per-frame rotation + camera.

    Exposes the same geometry ``BundleWindowReportDisplayOp`` summarizes as
    text, but as a structured object suitable for further inspection or
    tooling (e.g. plotting frame pointing drift across a sequence).
    """

    CONFIGS: dict[str, Any] = {
        "alignment_plan": {"type": "object", "required": True},
    }
    OUTPUTS: dict[str, Any] = {
        "export": {"type": "object"},
    }

    async def _async_execute(self, configs: dict[str, Any]) -> None:
        export = build_bundle_frame_rotation_export(configs["alignment_plan"])
        await self._broadcast_outputs({"export": export})


def _json_safe(obj: Any) -> Any:
    """``json.dump`` default hook: flatten numpy values dacite-style tools skip.

    ``dataclasses.asdict`` already unpacks nested dataclasses/tuples/dicts;
    this only needs to cover the leaf numpy types that survive that pass.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


@register_op()
class BundleFrameRotationExportSaveOp(BaseOp):
    """Serialize a ``BundleFrameRotationExport`` to a JSON file for offline use.

    Development/debug sink: writes the same structured diagnostic
    ``BundleFrameRotationExportOp`` produces to disk, so a sequence's
    per-frame rotations and shared camera state can be inspected outside
    the running pipeline (plotting, notebooks, comparing runs).
    """

    CONFIGS: dict[str, Any] = {
        "export": {"type": "object", "required": True},
        "output_path": {"type": "str", "required": True},
    }
    OUTPUTS: dict[str, Any] = {
        "result": {"type": "str"},
    }

    async def _async_execute(self, configs: dict[str, Any]) -> None:
        export = configs["export"]
        if not isinstance(export, BundleFrameRotationExport):
            raise TypeError("export must be a BundleFrameRotationExport")
        output_path = configs["output_path"]
        payload = dataclasses.asdict(export)
        await self._run_cpu(self._write_json, output_path, payload)
        logger.info("Wrote frame rotation export to {}", output_path)
        await self._broadcast_outputs({"result": output_path})

    @staticmethod
    def _write_json(output_path: str, payload: dict[str, Any]) -> None:
        directory = os.path.dirname(output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=_json_safe)
