"""Sequence-wide bundle adjustment and BA-plan consumers."""
import asyncio
import dataclasses
import enum
from collections import Counter
from typing import Any, Optional

import cv2
import numpy as np

from .._custom_op import median_reduce_chunk, sigma_clip_fused_chunk
from ..component.data_container import FloatImage
from ..component.merger import MaxMerger, MeanMerger
from ..component.norma.bundle import (BAAlignmentPlan, BundleAdjustmentError,
                                      BundleFrame, FrameAlignmentStatus,
                                      build_bundle_plan, estimate_edge_count)
from ..component.norma.bundle_window import (BundleWindowSpec,
                                             build_bundle_window_schedule,
                                             build_identity_window_schedule,
                                             build_identity_window_spec)
from ..component.norma.frame_align import (CameraInitializationPolicy,
                                           build_camera_candidate)
from ..component.norma.geometry_view import StarDetectionCache
from ..component.norma.intrinsics_from_exif import lens_type_from_exif
from ..component.queue import StreamExhausted
from ..engine.registry import register_op
from .base import BaseOp, FilterBaseOp


@dataclasses.dataclass(frozen=True)
class BundleRemapFrameReport:
    index: int
    status: FrameAlignmentStatus
    emitted: bool
    reason: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class BundleRemapReport:
    frames: tuple[BundleRemapFrameReport, ...]


@dataclasses.dataclass(frozen=True)
class BundleWindowFrameReport:
    center_index: int
    contributor_indices: tuple[int, ...]
    status: str = "ready"
    reason: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class BundleWindowReport:
    frames: tuple[BundleWindowFrameReport, ...]


class WindowFrameStatus(str, enum.Enum):
    READY = "ready"
    EXCLUDED = "excluded"
    INSUFFICIENT_CONTRIBUTORS = "insufficient_contributors"


@dataclasses.dataclass(frozen=True)
class WindowFrame:
    """One position-preserving window result consumed by a filter gate."""

    center_index: int
    image: Any
    exif: Any
    contributor_indices: tuple[int, ...]
    status: WindowFrameStatus
    reason: Optional[str] = None


def _append_validity_channel(
    image: np.ndarray,
    source_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, int | float]:
    """Append one hard-validity channel for remap coverage and source ROI."""
    dtype = image.dtype
    if np.issubdtype(dtype, np.integer):
        validity_scale = int(np.iinfo(dtype).max)
    elif np.issubdtype(dtype, np.floating):
        validity_scale = 1.0
    else:
        raise TypeError(f"unsupported image dtype for window mean: {dtype}")
    if source_mask is None:
        validity = np.full(image.shape[:2], validity_scale, dtype=dtype)
    else:
        if source_mask.shape != image.shape[:2]:
            raise ValueError(
                "source mask shape must match the image spatial shape")
        validity = source_mask.astype(dtype, copy=False) * validity_scale
    if image.ndim == 2:
        return np.stack((image, validity), axis=-1), validity_scale
    if image.ndim != 3 or image.shape[2] <= 0:
        raise ValueError("window images must have shape (H, W) or (H, W, C)")
    return np.concatenate((image, validity[..., None]), axis=-1), validity_scale


def _mean_bundle_window(
    camera: Any,
    spec: BundleWindowSpec,
    images: dict[int, np.ndarray],
    output_size: tuple[int, int],
    map_scale: float,
    participation_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Remap one window and merge only fully supported samples."""
    center = images[spec.center_index]
    merger = MeanMerger()

    for sample, valid in _iter_bundle_window_samples(
            camera, spec, images, output_size, map_scale, participation_mask):
        merger.merge(sample, spatial_mask=valid)

    statistics = merger.result
    if statistics is None:
        raise RuntimeError("bundle window has no valid contributors")
    if np.issubdtype(center.dtype, np.floating):
        denominator = np.where(statistics.n > 0, statistics.n, 1)
        return (statistics.sum_mu / denominator).astype(center.dtype)
    merged = merger.merged_image
    assert merged is not None
    return merged.data.astype(center.dtype)


def _iter_bundle_window_samples(
    camera: Any,
    spec: BundleWindowSpec,
    images: dict[int, np.ndarray],
    output_size: tuple[int, int],
    map_scale: float,
    participation_mask: Optional[np.ndarray] = None,
):
    """Yield each center-coordinate sample and its full-coverage mask."""
    for source in spec.sources:
        image = images[source.index]
        if camera is None or source.index == spec.center_index:
            valid = (np.ones(image.shape[:2], dtype=bool)
                     if participation_mask is None else participation_mask)
            yield image, valid.astype(np.uint8)
            continue
        augmented, validity_scale = _append_validity_channel(
            image, participation_mask)
        remapped = camera.project_image_from_camera(
            camera,
            augmented,
            output_size,
            rotation_dst_to_src=source.rotation_center_to_source,
            map_scale=map_scale,
        )
        sample = remapped[..., :-1]
        if image.ndim == 2:
            sample = sample[..., 0]
        remapped_validity = remapped[..., -1]
        if np.issubdtype(remapped_validity.dtype, np.integer):
            valid = remapped_validity == validity_scale
        else:
            valid = remapped_validity >= float(validity_scale) - 1e-6
        if participation_mask is not None:
            valid &= participation_mask
        yield sample, valid.astype(np.uint8)


def _collect_bundle_window_stack(
    camera: Any,
    spec: BundleWindowSpec,
    images: dict[int, np.ndarray],
    output_size: tuple[int, int],
    map_scale: float,
    participation_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize one reducer-local stack for non-streaming algorithms."""
    center = images[spec.center_index]
    stack = np.empty((len(spec.sources), *center.shape), dtype=center.dtype)
    masks = np.empty((len(spec.sources), *center.shape[:2]), dtype=np.uint8)
    for position, (sample, valid) in enumerate(_iter_bundle_window_samples(
            camera, spec, images, output_size, map_scale,
            participation_mask)):
        stack[position] = sample
        masks[position] = valid
    return stack, masks


def _max_bundle_window(camera: Any, spec: BundleWindowSpec,
                       images: dict[int, np.ndarray],
                       output_size: tuple[int, int],
                       map_scale: float,
                       participation_mask: Optional[np.ndarray] = None,
                       ) -> np.ndarray:
    merger = MaxMerger()
    for sample, valid in _iter_bundle_window_samples(
            camera, spec, images, output_size, map_scale, participation_mask):
        merger.merge(sample, spatial_mask=valid)
    result = merger.merged_image
    if result is None:
        raise RuntimeError("bundle window has no valid contributors")
    return result


def _masked_median(stack: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Exact per-pixel median with a frame-level spatial validity mask."""
    if np.all(masks):
        return median_reduce_chunk(stack)
    expanded = masks if stack.ndim == 3 else masks[..., None]
    fill = (np.iinfo(stack.dtype).max
            if np.issubdtype(stack.dtype, np.integer) else np.inf)
    np.copyto(stack, fill, where=~expanded.astype(bool))
    stack.sort(axis=0)
    counts = masks.sum(axis=0, dtype=np.intp)
    lower_indices = (counts - 1) // 2
    upper_indices = counts // 2
    if stack.ndim == 4:
        lower_indices = np.broadcast_to(lower_indices[..., None], stack.shape[1:])
        upper_indices = np.broadcast_to(upper_indices[..., None], stack.shape[1:])
    lower = np.take_along_axis(stack, lower_indices[None, ...], axis=0)[0]
    upper = np.take_along_axis(stack, upper_indices[None, ...], axis=0)[0]
    if np.issubdtype(stack.dtype, np.integer):
        result = (lower.astype(np.float64) + upper.astype(np.float64)) / 2.0
        return result.astype(stack.dtype)
    return ((lower + upper) * 0.5).astype(stack.dtype)


def _median_bundle_window(camera: Any, spec: BundleWindowSpec,
                          images: dict[int, np.ndarray],
                          output_size: tuple[int, int],
                          map_scale: float,
                          participation_mask: Optional[np.ndarray] = None,
                          ) -> np.ndarray:
    stack, masks = _collect_bundle_window_stack(
        camera, spec, images, output_size, map_scale, participation_mask)
    return _masked_median(stack, masks)


def _sigma_clip_bundle_window(
    camera: Any,
    spec: BundleWindowSpec,
    images: dict[int, np.ndarray],
    output_size: tuple[int, int],
    map_scale: float,
    rej_high: float,
    rej_low: float,
    max_iter: int,
    participation_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    stack, masks = _collect_bundle_window_stack(
        camera, spec, images, output_size, map_scale, participation_mask)
    if stack.dtype not in (np.dtype("uint8"), np.dtype("uint16")):
        raise ValueError("BundleWindowSigmaClipStackOp supports uint8/uint16")
    channels = stack.shape[-1] if stack.ndim == 4 else 1
    flat_stack = np.ascontiguousarray(stack.reshape(len(stack), -1))
    expanded_masks = (masks if channels == 1 else np.broadcast_to(
        masks[..., None], stack.shape))
    flat_masks = np.ascontiguousarray(expanded_masks.reshape(len(stack), -1),
                                      dtype=np.uint8)
    accepted_sum, _, accepted_n = sigma_clip_fused_chunk(
        flat_stack,
        rej_high=rej_high,
        rej_low=rej_low,
        max_iter=max_iter,
        mask=flat_masks,
        channels=channels,
    )
    mean = np.divide(
        accepted_sum,
        accepted_n,
        out=np.zeros_like(accepted_sum),
        where=accepted_n > 0,
    ).reshape(stack.shape[1:])
    return np.rint(mean).clip(0, np.iinfo(stack.dtype).max).astype(stack.dtype)


def _prepare_detection_mask(mask: Any,
                            image_shape: tuple[int, ...]) -> np.ndarray:
    """Convert a static workflow mask to a binary detector-sized mask."""
    array = mask.data if isinstance(mask, FloatImage) else np.asarray(mask)
    if array.ndim == 3:
        if array.shape[2] not in (1, 3, 4):
            raise ValueError("BA detection mask must have 1, 3, or 4 channels")
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError("BA detection mask must be a 2D image")

    if np.issubdtype(array.dtype, np.bool_):
        binary = array
    elif np.issubdtype(array.dtype, np.integer):
        binary = array > (np.iinfo(array.dtype).max * 0.5)
    else:
        binary = array > 0.5

    target_shape = image_shape[:2]
    if binary.shape != target_shape:
        binary = cv2.resize(
            binary.astype(np.uint8),
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    return np.ascontiguousarray(binary, dtype=bool)


@register_op()
class BundleAdjustmentOp(BaseOp):
    """Collect a same-camera sequence and emit its geometry-only BA plan.

    This op intentionally has no sequence output. A downstream plan consumer
    must use a second load of the original files after this object is available;
    this avoids a plan/data join deadlock in the streaming graph.
    """

    EXECUTOR = "cpu"
    INPUTS: dict[str, Any] = {
        "data": {"type": "sequence"},
        "exifs": {"type": "sequence", "required": False},
    }
    CONFIGS: dict[str, Any] = {
        "mask": {"type": "image", "default": None},
        "reference_frame_index": {"type": "int", "default": None},
        "method": {"type": "str", "default": "distortion"},
        "camera_setup_mode": {"type": "str", "default": None},
        "lens_type": {"type": "str", "default": None},
        "distortion": {"type": "list", "default": None},
        "focal_length_mm": {"type": "float", "default": None},
        "crop_factor": {"type": "float", "default": 1.0},
        "fallback_focal_equiv_mm": {"type": "float", "default": 20.0},
        "optimize_focal": {"type": "bool", "default": None},
        "optimize_distortion": {"type": "bool", "default": None},
        "optimize_principal_point": {"type": "bool", "default": None},
        "allow_large_principal_point_offset": {"type": "bool", "default": False},
        "pair_offsets": {"type": "list", "default": [1, 2, 4]},
        "max_pairs_per_edge": {"type": "int", "default": 128},
        "random_seed": {"type": "int", "default": 0},
        "camera_solve_frames": {"type": "int", "default": 32},
        "edge_topology": {"type": "str", "default": "dense"},
        "max_pair_offset": {"type": "int", "default": None},
        "max_nfev": {"type": "int", "default": 2000},
    }
    OUTPUTS: dict[str, Any] = {
        "alignment_plan": {"type": "object"},
    }
    REPORTS_PROGRESS = True

    @classmethod
    def estimate_resources(cls, configs, frame_bytes, n_frames,
                           dtype_bytes=None):
        _ = (configs, dtype_bytes, n_frames)
        # Only compact star observations survive collection; pixel memory is
        # bounded by the frame currently being detected.
        return (frame_bytes * 2, 0)

    async def _async_execute(self, configs: dict[str, Any]) -> None:
        if configs.get("method", "distortion") != "distortion":
            raise ValueError(
                "BundleAdjustmentOp supports camera-model distortion mode only")
        pair_offsets = tuple(sorted({
            int(value) for value in
            configs.get("pair_offsets") or (1, 2, 4)
            if int(value) > 0
        }))
        edge_topology = configs.get("edge_topology") or "dense"
        max_pair_offset = configs.get("max_pair_offset")
        if self.length is not None:
            edge_count = estimate_edge_count(
                self.length, pair_offsets, edge_topology, max_pair_offset)
            self.tracker.create_bar(
                self.name, self.length + edge_count + 1,
                desc=self.display_name)
        exifs_active = self.inputs["exifs"].active
        camera_setup_mode = configs.get("camera_setup_mode")
        if camera_setup_mode not in (None, "auto", "manual"):
            raise ValueError(
                f"Unsupported camera_setup_mode {camera_setup_mode!r}; "
                "expected 'auto' or 'manual'")
        focal_length = configs.get("focal_length_mm")
        if camera_setup_mode == "auto":
            focal_length = None
        elif camera_setup_mode == "manual" and focal_length in (None, ""):
            raise ValueError(
                "focal_length_mm is required when "
                "camera_setup_mode='manual'")
        focal_equiv = (
            float(focal_length) * float(configs.get("crop_factor") or 1.0)
            if focal_length is not None else None)
        fallback = float(configs.get("fallback_focal_equiv_mm", 20.0))
        configured_reference = configs.get("reference_frame_index")
        configured_mask = configs.get("mask")
        detection_mask = None
        detection_mask_shape = None
        observations = []
        for index in self._input_range():
            data = self._async_convert_inputs()
            try:
                image = await data["data"]
            except StreamExhausted:
                break
            exif_obj = None
            if exifs_active:
                try:
                    exif_obj = await data["exifs"]
                except StreamExhausted as exc:
                    raise ValueError("data/exifs sequences have different lengths") from exc
            array = image.data if isinstance(image, FloatImage) else image
            tags = exif_obj.exif if exif_obj is not None else None
            if (configured_mask is not None
                    and detection_mask_shape != array.shape[:2]):
                detection_mask = _prepare_detection_mask(
                    configured_mask, array.shape)
                detection_mask_shape = array.shape[:2]
            detection = await self._run_cpu(
                StarDetectionCache.from_image, array, detection_mask)
            stars = await self._run_cpu(lambda: detection.pywt_stars)
            observations.append((index, stars, array.shape, tags))
            self.tracker.update(self.name)
        if not observations:
            raise BundleAdjustmentError("cannot bundle-adjust an empty sequence")
        reference_index = (
            observations[len(observations) // 2][0]
            if configured_reference is None else int(configured_reference))
        reference_observation = next(
            (item for item in observations if item[0] == reference_index), None)
        if reference_observation is None:
            raise BundleAdjustmentError(
                "reference_frame_index must identify one collected frame")
        _, _, reference_array_shape, reference_tags = reference_observation
        lens_type = (configs.get("lens_type")
                     or lens_type_from_exif(reference_tags))
        policy = CameraInitializationPolicy(
            lens_type=lens_type,
            fallback_focal_equiv_mm=fallback,
            optimize_focal=configs.get("optimize_focal"),
            optimize_distortion=configs.get("optimize_distortion"),
            optimize_principal_point=configs.get("optimize_principal_point"),
            allow_large_principal_point_offset=bool(
                configs.get("allow_large_principal_point_offset", False)),
        )
        camera_tags = (
            None if camera_setup_mode == "manual" else reference_tags)
        shared_candidate = build_camera_candidate(
            camera_tags, reference_array_shape, "distortion",
            configs.get("distortion"), focal_equiv, policy)
        reference_shape = reference_array_shape[:2]
        if any(shape[:2] != reference_shape
               for _, _, shape, _ in observations):
            raise BundleAdjustmentError(
                "sequence BA requires a single image geometry")
        frames = [BundleFrame(index=index, stars=stars,
                              candidate=shared_candidate)
                  for index, stars, _, _ in observations]
        edge_completed = None
        if self.length is not None:
            loop = asyncio.get_running_loop()

            def report_edge_completed() -> None:
                loop.call_soon_threadsafe(
                    self.tracker.update, self.name)

            edge_completed = report_edge_completed
        try:
            configured_max_pairs = configs.get("max_pairs_per_edge", 128)
            max_pairs_per_edge = (
                None if configured_max_pairs is None
                or int(configured_max_pairs) <= 0
                else int(configured_max_pairs))
            plan = await self._run_cpu(
                build_bundle_plan, frames,
                reference_frame_index=reference_index,
                pair_offsets=pair_offsets,
                max_pairs_per_edge=max_pairs_per_edge,
                random_seed=configs.get("random_seed", 0),
                camera_solve_frames=configs.get("camera_solve_frames"),
                edge_completed=edge_completed,
                edge_topology=edge_topology,
                max_pair_offset=max_pair_offset,
                max_nfev=configs.get("max_nfev", 2000))
        except BundleAdjustmentError:
            raise
        except Exception as exc:
            raise BundleAdjustmentError(f"sequence BA failed: {exc}") from exc
        if self.length is not None:
            # Drain edge callbacks scheduled from the worker thread before
            # completing the one-step optimization phase.
            await asyncio.sleep(0)
            self.tracker.update(self.name)
        await self._broadcast_outputs({"alignment_plan": plan})
        if self.length is not None:
            self.tracker.close_bar(self.name)


@register_op()
class BundleReferenceRemapOp(FilterBaseOp):
    """Apply a geometry-only BA plan to a second image stream in input order."""

    EXECUTOR = "cpu"
    INPUTS: dict[str, Any] = {
        "data": {"type": "sequence"},
        "exifs": {"type": "sequence", "required": False},
    }
    CONFIGS: dict[str, Any] = {
        "alignment_plan": {"type": "object", "required": True},
        "remap_map_scale": {"type": "float", "default": 0.5},
    }
    OUTPUTS: dict[str, Any] = {
        "result": {"type": "sequence"},
        "aligned_exifs": {"type": "sequence"},
        "remap_report": {"type": "object"},
    }
    REPORTS_PROGRESS = True

    @classmethod
    def estimate_resources(cls, configs, frame_bytes, n_frames,
                           dtype_bytes=None):
        _ = (configs, n_frames, dtype_bytes)
        # One input and one remapped output. Projection-specific workspace is
        # temporary and does not grow with the number or position of frames.
        return (2 * frame_bytes, 0)

    async def _async_filter(self, configs: dict[str, Any]) -> None:
        plan = configs["alignment_plan"]
        if not isinstance(plan, BAAlignmentPlan):
            raise TypeError("alignment_plan must be a BAAlignmentPlan")
        map_scale = float(configs.get("remap_map_scale", 0.5))
        if not 0.0 < map_scale <= 1.0:
            raise ValueError("remap_map_scale must be in (0, 1]")
        if not 0 <= plan.reference_frame_index < len(plan.frames):
            raise ValueError("AlignmentPlan reference index is outside its frames")
        if self.length is not None:
            self.tracker.create_bar(
                self.name, self.length, desc=self.display_name)
        exifs_active = self.inputs["exifs"].active
        reports: list[BundleRemapFrameReport] = []
        seen_count = 0
        intrinsics = plan.shared_camera.intrinsics
        expected_shape = (intrinsics.image_height_px,
                          intrinsics.image_width_px)
        output_size = (intrinsics.image_width_px,
                       intrinsics.image_height_px)

        for index in self._input_range():
            data = self._async_convert_inputs()
            try:
                frame = await data["data"]
            except StreamExhausted:
                break
            exif_obj = None
            if exifs_active:
                try:
                    exif_obj = await data["exifs"]
                except StreamExhausted as exc:
                    raise ValueError("data/exifs sequences have different lengths") from exc
            seen_count += 1
            if index >= len(plan.frames):
                raise ValueError(
                    "second image load is longer than AlignmentPlan: "
                    f"got at least {seen_count}, expected {len(plan.frames)}")
            entry = plan.frame(index)
            if entry.index != index:
                raise ValueError(
                    "AlignmentPlan frames must use contiguous stream indices")
            array = frame.data if isinstance(frame, FloatImage) else frame
            if array.shape[:2] != expected_shape:
                raise ValueError(
                    f"frame {index} geometry differs from AlignmentPlan: "
                    f"got {array.shape[:2]}, expected {expected_shape}")

            if entry.status == FrameAlignmentStatus.EXCLUDED:
                reports.append(BundleRemapFrameReport(
                    index, entry.status, False, entry.reason))
                self.tracker.update(self.name)
                continue
            if entry.rotation_ref_to_src is None:
                raise ValueError(
                    f"solved frame {index} has no reference-relative pose")

            if index == plan.reference_frame_index:
                output = frame
            else:
                aligned = await self._run_cpu(
                    plan.shared_camera.project_image_from_camera,
                    plan.shared_camera,
                    array,
                    output_size,
                    rotation_dst_to_src=entry.rotation_ref_to_src,
                    map_scale=map_scale,
                )
                output = (FloatImage(data=aligned, dtype=frame.dtype)
                          if isinstance(frame, FloatImage) else aligned)
            await self._broadcast_outputs(
                {"result": output, "aligned_exifs": exif_obj})
            reports.append(BundleRemapFrameReport(
                index, entry.status, True, entry.reason))
            self.tracker.update(self.name)
        if seen_count != len(plan.frames):
            raise ValueError(
                "second image load length differs from AlignmentPlan: "
                f"got {seen_count}, expected {len(plan.frames)}")
        await self._broadcast_outputs({
            "remap_report": BundleRemapReport(tuple(reports))})
        if self.length is not None:
            self.tracker.close_bar(self.name)


class _BundleWindowStackOp(BaseOp):
    """Length-preserving window execution for aligned or identity streams."""

    EXECUTOR = "cpu"
    INPUTS: dict[str, Any] = {
        "data": {"type": "sequence"},
        "exifs": {"type": "sequence", "required": False},
    }
    CONFIGS: dict[str, Any] = {
        "alignment_plan": {"type": "object", "default": None},
        "window_size": {"type": "int", "default": 5},
        "min_contributors": {"type": "int", "default": 1},
        "remap_map_scale": {"type": "float", "default": 0.5},
        "mask": {"type": "image", "default": None},
    }
    OUTPUTS: dict[str, Any] = {
        "result": {"type": "sequence"},
        "window_report": {"type": "object"},
    }

    REPORTS_PROGRESS = True

    @staticmethod
    def _reduce_window(camera, spec, images, output_size, map_scale, configs,
                       participation_mask=None):
        raise NotImplementedError

    async def _async_execute(self, configs: dict[str, Any]) -> None:
        plan = configs.get("alignment_plan")
        window_size = int(configs.get("window_size", 5))
        min_contributors = int(configs.get("min_contributors", 1))
        if window_size <= 0 or window_size % 2 == 0:
            raise ValueError("window_size must be a positive odd integer")
        if not 1 <= min_contributors <= window_size:
            raise ValueError("min_contributors must be in [1, window_size]")
        streaming_identity = plan is None and self.length is None
        if plan is None:
            schedule = (None if streaming_identity else
                        build_identity_window_schedule(
                            self.length, window_size,
                            min_contributors=min_contributors))
            camera = None
            expected_shape = None
        else:
            if not isinstance(plan, BAAlignmentPlan):
                raise TypeError("alignment_plan must be a BAAlignmentPlan or None")
            schedule = build_bundle_window_schedule(
                plan, window_size, min_contributors=min_contributors)
            camera = plan.shared_camera
            intrinsics = camera.intrinsics
            expected_shape = (intrinsics.image_height_px,
                              intrinsics.image_width_px)
        map_scale = float(configs.get("remap_map_scale", 0.5))
        if not 0.0 < map_scale <= 1.0:
            raise ValueError("remap_map_scale must be in (0, 1]")
        if self.length is not None:
            self.tracker.create_bar(
                self.name, self.length, desc=self.display_name)

        output_size = ((expected_shape[1], expected_shape[0])
                       if expected_shape is not None else None)
        exifs_active = self.inputs["exifs"].active
        if schedule is None:
            specs = {}
            remaining_uses = None
            frame_count = None
        else:
            specs = {window.center_index: window
                     for window in schedule.windows}
            remaining_uses = Counter(
                source.index
                for window in schedule.windows
                for source in window.sources
            )
            # Every center needs its image/exif once even if it cannot be reduced.
            remaining_uses.update(range(schedule.frame_count))
            frame_count = schedule.frame_count
        frame_cache: dict[int, tuple[Any, np.ndarray, Any]] = {}
        reports: list[BundleWindowFrameReport] = []
        next_center = 0
        seen_count = 0
        data_dtype = None
        float_image_input = None
        participation_mask = None
        configured_mask = configs.get("mask")
        radius = window_size // 2

        async def emit_center(spec: BundleWindowSpec | None) -> None:
            center_frame, _, center_exif = frame_cache[next_center]
            if spec is None:
                plan_frame = (plan.frame(next_center)
                              if plan is not None else None)
                if (plan_frame is not None
                        and plan_frame.status == FrameAlignmentStatus.EXCLUDED):
                    status = WindowFrameStatus.EXCLUDED
                    reason = plan_frame.reason or "excluded by bundle adjustment"
                else:
                    status = WindowFrameStatus.INSUFFICIENT_CONTRIBUTORS
                    reason = "insufficient contributors"
                contributor_indices: tuple[int, ...] = ()
                output = None
            else:
                missing = [source.index for source in spec.sources
                           if source.index not in frame_cache]
                if missing:
                    raise RuntimeError(
                        f"window {next_center} lost buffered frames {missing}")
                images = {
                    source.index: frame_cache[source.index][1]
                    for source in spec.sources
                }
                assert output_size is not None
                result = await self._run_cpu(
                    self._reduce_window, camera, spec, images, output_size,
                    map_scale, configs,
                    participation_mask=participation_mask)
                output = (FloatImage(data=result, dtype=center_frame.dtype)
                          if isinstance(center_frame, FloatImage) else result)
                contributor_indices = tuple(
                    source.index for source in spec.sources)
                status = WindowFrameStatus.READY
                reason = None

            await self._broadcast_outputs({"result": WindowFrame(
                center_index=next_center,
                image=output,
                exif=center_exif,
                contributor_indices=contributor_indices,
                status=status,
                reason=reason,
            )})
            reports.append(BundleWindowFrameReport(
                center_index=next_center,
                contributor_indices=contributor_indices,
                status=status.value,
                reason=reason,
            ))

        for index in self._input_range():
            data = self._async_convert_inputs()
            try:
                frame = await data["data"]
            except StreamExhausted:
                break
            exif_obj = None
            if exifs_active:
                try:
                    exif_obj = await data["exifs"]
                except StreamExhausted as exc:
                    raise ValueError(
                        "data/exifs sequences have different lengths") from exc

            seen_count += 1
            if frame_count is not None and index >= frame_count:
                raise ValueError(
                    "second image load is longer than AlignmentPlan: "
                    f"got at least {seen_count}, expected {frame_count}")
            array = frame.data if isinstance(frame, FloatImage) else frame
            if not isinstance(array, np.ndarray):
                raise TypeError("window input frames must be numpy arrays or FloatImage")
            if expected_shape is None:
                expected_shape = array.shape[:2]
                output_size = (expected_shape[1], expected_shape[0])
                if configured_mask is not None:
                    participation_mask = _prepare_detection_mask(
                        configured_mask, array.shape)
            elif array.shape[:2] != expected_shape:
                raise ValueError(
                    f"frame {index} geometry differs from AlignmentPlan: "
                    f"got {array.shape[:2]}, expected {expected_shape}")
            if array.ndim not in (2, 3):
                raise ValueError("window images must have shape (H, W) or (H, W, C)")
            if data_dtype is None:
                data_dtype = array.dtype
                float_image_input = isinstance(frame, FloatImage)
            elif array.dtype != data_dtype:
                raise ValueError("window input frames must share one array dtype")
            elif isinstance(frame, FloatImage) != float_image_input:
                raise ValueError(
                    "window input frames must use one container representation")

            if configured_mask is not None and participation_mask is None:
                participation_mask = _prepare_detection_mask(
                    configured_mask, array.shape)
            frame_cache[index] = (frame, array, exif_obj)

            if streaming_identity:
                while next_center + radius <= index:
                    spec = build_identity_window_spec(
                        index + 1, next_center, window_size,
                        min_contributors=min_contributors)
                    await emit_center(spec)
                    next_center += 1
                    earliest_needed = max(0, next_center - radius)
                    for cached_index in tuple(frame_cache):
                        if cached_index < earliest_needed:
                            frame_cache.pop(cached_index)
            else:
                assert frame_count is not None
                assert remaining_uses is not None
                while next_center < frame_count:
                    spec = specs.get(next_center)
                    ready_index = (max(source.index for source in spec.sources)
                                   if spec is not None else next_center)
                    if ready_index > index:
                        break
                    await emit_center(spec)
                    if spec is not None:
                        for source in spec.sources:
                            remaining_uses[source.index] -= 1
                            if remaining_uses[source.index] == 0:
                                frame_cache.pop(source.index, None)
                    remaining_uses[next_center] -= 1
                    if remaining_uses[next_center] == 0:
                        frame_cache.pop(next_center, None)
                    next_center += 1
            self.tracker.update(self.name)

        if streaming_identity:
            while next_center < seen_count:
                spec = build_identity_window_spec(
                    seen_count, next_center, window_size,
                    min_contributors=min_contributors)
                await emit_center(spec)
                next_center += 1
        elif seen_count != frame_count:
            raise ValueError(
                "second image load length differs from AlignmentPlan: "
                f"got {seen_count}, expected {frame_count}")
        if next_center != seen_count:
            raise RuntimeError("not all window positions were emitted")
        await self._broadcast_outputs({
            "window_report": BundleWindowReport(tuple(reports)),
        })
        if self.length is not None:
            self.tracker.close_bar(self.name)


@register_op()
class WindowFrameFilterGateOp(FilterBaseOp):
    """Drop invalid window positions while keeping payload sidecars aligned."""

    INPUTS: dict[str, Any] = {
        "data": {"type": "sequence", "required": True},
    }
    OUTPUTS: dict[str, Any] = {
        "result": {"type": "sequence"},
        "aligned_exifs": {"type": "sequence"},
        "center_indices": {"type": "sequence"},
    }
    
    REPORTS_PROGRESS = False

    async def _async_filter(self, configs: dict[str, Any]) -> None:
        _ = configs
        for _index in self._input_range():
            try:
                frame = await self._async_convert_inputs()["data"]
            except StreamExhausted:
                break
            if not isinstance(frame, WindowFrame):
                raise TypeError("WindowFrameFilterGateOp expects WindowFrame items")
            if frame.status == WindowFrameStatus.READY:
                if frame.image is None:
                    raise ValueError(
                        f"ready window {frame.center_index} has no image")
                await self._broadcast_outputs({
                    "result": frame.image,
                    "aligned_exifs": frame.exif,
                    "center_indices": frame.center_index,
                })
            self.tracker.update(self.name)


@register_op()
class WindowFrameMaskedBlendOp(BaseOp):
    """Blend two position-preserving window streams before filtering."""

    INPUTS: dict[str, Any] = {
        "sky": {"type": "sequence", "required": True},
        "ground": {"type": "sequence", "required": False},
    }
    CONFIGS: dict[str, Any] = {
        "mask": {"type": "image", "default": None},
    }
    OUTPUTS: dict[str, Any] = {
        "result": {"type": "sequence"},
    }

    async def _async_execute(self, configs: dict[str, Any]) -> None:
        ground_active = self.inputs["ground"].active
        configured_mask = configs.get("mask")
        runtime_mask = None
        mask_shape = None

        for _index in self._input_range():
            inputs = self._async_convert_inputs()
            try:
                sky = await inputs["sky"]
                ground = await inputs["ground"] if ground_active else None
            except StreamExhausted as exc:
                raise ValueError(
                    "sky and ground window streams have different lengths") from exc
            if not isinstance(sky, WindowFrame):
                raise TypeError("sky input must contain WindowFrame items")
            if ground is not None and not isinstance(ground, WindowFrame):
                raise TypeError("ground input must contain WindowFrame items")
            if ground is not None and ground.center_index != sky.center_index:
                raise ValueError(
                    "sky and ground center indices differ: "
                    f"{sky.center_index} != {ground.center_index}")

            if sky.status != WindowFrameStatus.READY or ground is None:
                output = sky
            elif ground.status != WindowFrameStatus.READY:
                output = WindowFrame(
                    center_index=sky.center_index,
                    image=None,
                    exif=sky.exif,
                    contributor_indices=sky.contributor_indices,
                    status=ground.status,
                    reason=ground.reason or "ground window is unavailable",
                )
            else:
                sky_array = (sky.image.data
                             if isinstance(sky.image, FloatImage) else sky.image)
                ground_array = (ground.image.data
                                if isinstance(ground.image, FloatImage)
                                else ground.image)
                if not isinstance(sky_array, np.ndarray) or not isinstance(
                        ground_array, np.ndarray):
                    raise TypeError("ready window images must be numpy arrays")
                if sky_array.shape != ground_array.shape:
                    raise ValueError("sky and ground window image shapes differ")
                if configured_mask is None:
                    blended = sky_array
                else:
                    if mask_shape != sky_array.shape[:2]:
                        runtime_mask = _prepare_detection_mask(
                            configured_mask, sky_array.shape)
                        mask_shape = sky_array.shape[:2]
                    assert runtime_mask is not None
                    mask = (runtime_mask[..., None]
                            if sky_array.ndim == 3 else runtime_mask)
                    blended = np.where(mask, sky_array, ground_array)
                image = (FloatImage(data=blended, dtype=sky.image.dtype)
                         if isinstance(sky.image, FloatImage) else blended)
                output = dataclasses.replace(sky, image=image)
            await self._broadcast_outputs({"result": output})


@register_op()
class BundleWindowMeanStackOp(_BundleWindowStackOp):
    """Remap each BA window to its center frame and emit a masked mean."""

    @staticmethod
    def _reduce_window(camera, spec, images, output_size, map_scale, configs,
                       participation_mask=None):
        _ = configs
        return _mean_bundle_window(
            camera, spec, images, output_size, map_scale, participation_mask)

    @classmethod
    def estimate_resources(cls, configs, frame_bytes, n_frames,
                           dtype_bytes=None):
        _ = n_frames
        window_size = int(configs.get("window_size", 5))
        source_bytes = max(1, int(dtype_bytes or 2))
        work_multiplier = 3 + 16 / source_bytes
        return (int((window_size + work_multiplier) * frame_bytes), 0)


@register_op()
class BundleWindowMaxStackOp(_BundleWindowStackOp):
    """Remap each BA window to its center frame and emit a masked maximum."""

    @staticmethod
    def _reduce_window(camera, spec, images, output_size, map_scale, configs,
                       participation_mask=None):
        _ = configs
        return _max_bundle_window(
            camera, spec, images, output_size, map_scale, participation_mask)

    @classmethod
    def estimate_resources(cls, configs, frame_bytes, n_frames,
                           dtype_bytes=None):
        _ = (n_frames, dtype_bytes)
        window_size = int(configs.get("window_size", 5))
        return ((window_size + 4) * frame_bytes, 0)


@register_op()
class BundleWindowMedianStackOp(_BundleWindowStackOp):
    """Remap and materialize one window for an exact masked median."""

    @staticmethod
    def _reduce_window(camera, spec, images, output_size, map_scale, configs,
                       participation_mask=None):
        _ = configs
        return _median_bundle_window(
            camera, spec, images, output_size, map_scale, participation_mask)

    @classmethod
    def estimate_resources(cls, configs, frame_bytes, n_frames,
                           dtype_bytes=None):
        _ = n_frames
        window_size = int(configs.get("window_size", 5))
        source_bytes = max(1, int(dtype_bytes or 2))
        multiplier = 2 * window_size + window_size / source_bytes + 3
        return (int(multiplier * frame_bytes), 0)


@register_op()
class BundleWindowSigmaClipStackOp(_BundleWindowStackOp):
    """Remap and materialize one window for iterative sigma clipping."""

    CONFIGS = {
        **_BundleWindowStackOp.CONFIGS,
        "rej_high": {"type": "float", "default": 3.0},
        "rej_low": {"type": "float", "default": 3.0},
        "max_iter": {"type": "int", "default": 5},
    }

    @staticmethod
    def _reduce_window(camera, spec, images, output_size, map_scale, configs,
                       participation_mask=None):
        rej_high = float(configs.get("rej_high", 3.0))
        rej_low = float(configs.get("rej_low", 3.0))
        max_iter = int(configs.get("max_iter", 5))
        if rej_high <= 0.0 or rej_low <= 0.0:
            raise ValueError("sigma clipping thresholds must be positive")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        return _sigma_clip_bundle_window(
            camera, spec, images, output_size, map_scale,
            rej_high, rej_low, max_iter, participation_mask)

    @classmethod
    def estimate_resources(cls, configs, frame_bytes, n_frames,
                           dtype_bytes=None):
        _ = n_frames
        window_size = int(configs.get("window_size", 5))
        source_bytes = max(1, int(dtype_bytes or 2))
        multiplier = (2 * window_size + window_size / source_bytes
                      + 24 / source_bytes + 3)
        return (int(multiplier * frame_bytes), 0)
