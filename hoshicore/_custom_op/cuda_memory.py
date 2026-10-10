"""CUDA memory estimates and runtime admission for custom-op backends."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Callable, Iterator, TypeVar

from hoshicore._custom_op._dispatch import CudaProbeError
from hoshicore._custom_op._dispatch import CustomOpCudaRuntimeUnavailableError
from hoshicore._custom_op._dispatch import CustomOpResourceExhaustedError
from hoshicore._custom_op._dispatch import cuda_memory_info
from hoshicore._custom_op._dispatch import load_compiled_module

CUDA_ADMISSION_FIXED_HEADROOM_BYTES = 256 * 1024 * 1024
CUDA_ADMISSION_HEADROOM_FRACTION = 0.05
STAR_DETECT_MAX_FOREGROUND_FRACTION = 0.25
STAR_SHRINK_THREADS_PER_BLOCK = 256
MATCHING_COSINE_TILE_SIZE = 16

_T = TypeVar("_T")


@dataclass(frozen=True)
class CudaMemoryEstimate:
    logical_op: str
    peak_device_bytes: int
    peak_pinned_bytes: int = 0
    confidence: str = "bounded"
    reason: str = ""


@dataclass(frozen=True)
class CudaChunkMemoryModel:
    logical_op: str
    host_bytes_per_row: int
    device_bytes_per_row: int
    fixed_device_bytes: int = 0
    confidence: str = "bounded"
    reason: str = ""

    def estimate(self, rows: int) -> CudaMemoryEstimate:
        if rows <= 0:
            raise ValueError("CUDA chunk memory estimate requires positive rows")
        return CudaMemoryEstimate(
            logical_op=self.logical_op,
            peak_device_bytes=(
                self.device_bytes_per_row * rows + self.fixed_device_bytes
            ),
            confidence=self.confidence,
            reason=self.reason,
        )


@dataclass(frozen=True)
class CudaAdmissionDecision:
    logical_op: str
    granted: bool
    checked: bool
    reason_code: str
    estimated_peak_bytes: int
    device: int | None = None
    free_bytes: int | None = None
    total_bytes: int | None = None
    headroom_bytes: int = 0
    reserved_bytes: int = 0
    cache_evicted: bool = False


_reservation_lock = threading.Lock()
_reserved_bytes_by_device: dict[int, int] = {}


def _dwt_len(value: int) -> int:
    return (value + 15) // 2


def _idwt_len(value: int) -> int:
    return 2 * value - 14


def _next_power_of_two(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


def estimate_camera_model_remap(
    *,
    source_height: int,
    source_width: int,
    channels: int,
    dtype_bytes: int,
    out_height: int,
    out_width: int,
) -> CudaMemoryEstimate:
    if min(
        source_height,
        source_width,
        channels,
        dtype_bytes,
        out_height,
        out_width,
    ) <= 0:
        raise ValueError(
            "camera-model remap CUDA memory estimate requires positive dimensions"
        )

    source_bytes = source_height * source_width * channels * dtype_bytes
    output_bytes = out_height * out_width * channels * dtype_bytes
    return CudaMemoryEstimate(
        logical_op="camera_model_remap",
        peak_device_bytes=source_bytes + output_bytes,
        peak_pinned_bytes=max(source_bytes, output_bytes),
        confidence="exact",
        reason=(
            "source and destination device buffers plus one shared pinned "
            "input/output staging buffer"
        ),
    )


def estimate_median_reduce_chunk(
    *, n_frames: int, plane_size: int, dtype_bytes: int
) -> CudaMemoryEstimate:
    if min(n_frames, plane_size, dtype_bytes) <= 0:
        raise ValueError("median CUDA memory estimate requires positive dimensions")
    input_bytes = n_frames * plane_size * dtype_bytes
    output_bytes = plane_size * dtype_bytes
    return CudaMemoryEstimate(
        logical_op="median_reduce_chunk",
        peak_device_bytes=input_bytes + output_bytes,
        peak_pinned_bytes=input_bytes,
        confidence="exact",
        reason="input and output device buffers; one reused pinned staging buffer",
    )


def estimate_matching_cosine_bidirectional_nearest(
    *,
    n1: int,
    n2: int,
    feature_dim: int,
) -> CudaMemoryEstimate:
    if min(n1, n2, feature_dim) <= 0:
        raise ValueError(
            "matching cosine nearest CUDA memory estimate requires positive dimensions"
        )

    features_bytes = (n1 + n2) * feature_dim * 8
    norms_bytes = (n1 + n2) * 8
    tile_size = MATCHING_COSINE_TILE_SIZE
    row_tiles = (n1 + tile_size - 1) // tile_size
    col_tiles = (n2 + tile_size - 1) // tile_size
    partial_candidates = n1 * col_tiles + n2 * row_tiles
    partial_candidates_bytes = partial_candidates * (8 + 8 + 4)
    nearest_outputs_bytes = (n1 + n2) * (8 + 8)
    status_bytes = 4
    return CudaMemoryEstimate(
        logical_op="matching_cosine_bidirectional_nearest",
        peak_device_bytes=(
            features_bytes
            + norms_bytes
            + partial_candidates_bytes
            + nearest_outputs_bytes
            + status_bytes
        ),
        peak_pinned_bytes=(
            features_bytes + nearest_outputs_bytes + status_bytes
        ),
        confidence="exact",
        reason=(
            "two feature matrices, norms, tiled row/column partial nearest "
            "candidates, bidirectional nearest outputs, and ambiguity status"
        ),
    )


# Pinned double-buffer slot of the staged host transfers (cuda_host_staging.cuh).
CUDA_STAGING_SLOT_BYTES = 8 * 1024 * 1024

ASTERISM_MAX_GRID_CELLS = 1 << 22
_ASTERISM_EXTENT_BYTES = 2 * 64 * 7 * 8
_ASTERISM_SCAN_TILE = 4096


def asterism_device_bytes(n1: int, n2: int, cells: int) -> int:
    """Device bytes of the native asterism op for a grid of ``cells`` cells:
    token values, block extents, two cell-offset grids with slot-ordered
    indices and coordinates, scan tile sums, both nearest outputs and the
    tie flag."""
    tiles = -(-cells // _ASTERISM_SCAN_TILE)
    return ((n1 + n2) * (3 * 8 + 4 + 3 * 8 + 4) + _ASTERISM_EXTENT_BYTES
            + 2 * (cells + 1) * 4 + tiles * 4 + 4)


def estimate_asterism_mutual_nearest(
    *,
    n1: int,
    n2: int,
) -> CudaMemoryEstimate:
    if min(n1, n2) <= 0:
        raise ValueError(
            "asterism mutual nearest CUDA memory estimate requires positive sizes")
    return CudaMemoryEstimate(
        logical_op="asterism_mutual_nearest",
        peak_device_bytes=asterism_device_bytes(n1, n2, ASTERISM_MAX_GRID_CELLS),
        confidence="bounded",
        reason=(
            "token buffers are exact; the device-built cell grid is bounded by "
            "the largest grid the kernel accepts"
        ),
    )


POINT_FEATURE_BINS = 120


def estimate_extract_point_features(
    *,
    n_points: int,
    k: int,
) -> CudaMemoryEstimate:
    if min(n_points, k) <= 0:
        raise ValueError(
            "extract_point_features CUDA memory estimate requires positive sizes")
    max_cells = 8 * n_points + 64
    points = n_points * (3 * 8 + 8 + 8)
    grid = n_points * 3 * 8 + (max_cells + 1) * 8 + n_points * 8
    return CudaMemoryEstimate(
        logical_op="extract_point_features",
        peak_device_bytes=(points + grid + n_points * k * 3 * 8
                           + n_points * POINT_FEATURE_BINS * 8 + 4),
        peak_pinned_bytes=2 * CUDA_STAGING_SLOT_BYTES,
        confidence="bounded",
        reason=(
            "points, norms, per-neighbour parameters and descriptors are exact; "
            "the direction grid uses its cell-count limit"
        ),
    )


def _star_shrink_image_sizes(
    *,
    height: int,
    width: int,
    channels: int,
    dtype_bytes: int,
) -> tuple[int, int, int, int]:
    if min(height, width, channels, dtype_bytes) <= 0:
        raise ValueError(
            "star-shrink CUDA memory estimate requires positive dimensions"
        )
    pixels = height * width
    total = pixels * channels
    image_bytes = total * dtype_bytes
    return pixels, total, image_bytes, pixels


def _validate_dog_kernel_sizes(
    small_kernel_size: int,
    large_kernel_size: int,
) -> None:
    if min(small_kernel_size, large_kernel_size) <= 0:
        raise ValueError("star-shrink DoG kernel sizes must be positive")
    if small_kernel_size % 2 == 0 or large_kernel_size % 2 == 0:
        raise ValueError("star-shrink DoG kernel sizes must be odd")


def _star_shrink_process_device_bytes(
    *,
    pixels: int,
    total: int,
    image_bytes: int,
    mask_bytes: int,
) -> int:
    plane_float_bytes = pixels * 4
    total_float_bytes = total * 4
    return (
        2 * image_bytes
        + mask_bytes
        + 4 * plane_float_bytes
        + 3 * total_float_bytes
    )


def _star_mask_dog_device_bytes(
    *,
    pixels: int,
    image_bytes: int,
    mask_bytes: int,
    small_kernel_size: int,
    large_kernel_size: int,
) -> int:
    _validate_dog_kernel_sizes(small_kernel_size, large_kernel_size)
    blocks = (pixels + STAR_SHRINK_THREADS_PER_BLOCK - 1) // (
        STAR_SHRINK_THREADS_PER_BLOCK
    )
    plane_float_bytes = pixels * 4
    kernel_bytes = (small_kernel_size + large_kernel_size) * 4
    reduction_bytes = blocks * 8
    return (
        image_bytes
        + 5 * plane_float_bytes
        + kernel_bytes
        + 2 * mask_bytes
        + 2 * reduction_bytes
    )


def estimate_star_shrink_process(
    *,
    height: int,
    width: int,
    channels: int,
    dtype_bytes: int,
) -> CudaMemoryEstimate:
    pixels, total, image_bytes, mask_bytes = _star_shrink_image_sizes(
        height=height,
        width=width,
        channels=channels,
        dtype_bytes=dtype_bytes,
    )
    return CudaMemoryEstimate(
        logical_op="star_shrink_process",
        peak_device_bytes=_star_shrink_process_device_bytes(
            pixels=pixels,
            total=total,
            image_bytes=image_bytes,
            mask_bytes=mask_bytes,
        ),
        confidence="exact",
        reason="all image, mask, Lab, morphology, and blur workspace buffers",
    )


def estimate_star_mask_dog(
    *,
    height: int,
    width: int,
    channels: int,
    dtype_bytes: int,
    small_kernel_size: int,
    large_kernel_size: int,
) -> CudaMemoryEstimate:
    pixels, _, image_bytes, mask_bytes = _star_shrink_image_sizes(
        height=height,
        width=width,
        channels=channels,
        dtype_bytes=dtype_bytes,
    )
    return CudaMemoryEstimate(
        logical_op="star_mask_dog",
        peak_device_bytes=_star_mask_dog_device_bytes(
            pixels=pixels,
            image_bytes=image_bytes,
            mask_bytes=mask_bytes,
            small_kernel_size=small_kernel_size,
            large_kernel_size=large_kernel_size,
        ),
        confidence="exact",
        reason="image, DoG planes, Gaussian kernels, masks, and reduction buffers",
    )


def estimate_star_shrink_dog_process(
    *,
    height: int,
    width: int,
    channels: int,
    dtype_bytes: int,
    small_kernel_size: int,
    large_kernel_size: int,
) -> CudaMemoryEstimate:
    pixels, total, image_bytes, mask_bytes = _star_shrink_image_sizes(
        height=height,
        width=width,
        channels=channels,
        dtype_bytes=dtype_bytes,
    )
    process_bytes = _star_shrink_process_device_bytes(
        pixels=pixels,
        total=total,
        image_bytes=image_bytes,
        mask_bytes=mask_bytes,
    )
    dog_bytes = _star_mask_dog_device_bytes(
        pixels=pixels,
        image_bytes=image_bytes,
        mask_bytes=mask_bytes,
        small_kernel_size=small_kernel_size,
        large_kernel_size=large_kernel_size,
    )
    return CudaMemoryEstimate(
        logical_op="star_shrink_dog_process",
        peak_device_bytes=process_bytes + dog_bytes - image_bytes - mask_bytes,
        confidence="exact",
        reason=(
            "fused DoG and shrink workspaces with one shared image and mask buffer"
        ),
    )


def _wavelet_device_peak_bytes(height: int, width: int, level: int) -> int:
    current_h = height
    current_w = width
    current_bytes = height * width * 8
    details: list[tuple[int, int]] = []
    detail_bytes = 0
    peak = current_bytes

    # The reconstruction zeroes the finest detail band, so level 0 allocates
    # neither its high-pass row buffer nor detail planes nor the high column.
    for index in range(level):
        out_h = _dwt_len(current_h)
        out_w = _dwt_len(current_w)
        row_bytes = current_h * out_w * 8
        level_bytes = out_h * out_w * 8
        kept_bands = 0 if index == 0 else 3
        peak = max(
            peak,
            current_bytes
            + detail_bytes
            + (1 if index == 0 else 2) * row_bytes
            + (1 + kept_bands) * level_bytes,
        )
        detail_bytes += kept_bands * level_bytes
        details.append((out_h, out_w, kept_bands))
        current_h = out_h
        current_w = out_w
        current_bytes = level_bytes

    for detail_h, detail_w, kept_bands in reversed(details):
        out_h = _idwt_len(detail_h)
        out_w = _idwt_len(detail_w)
        col_bytes = out_h * detail_w * 8
        output_bytes = out_h * out_w * 8
        peak = max(
            peak,
            current_bytes
            + detail_bytes
            + (2 if kept_bands else 1) * col_bytes
            + output_bytes,
        )
        detail_bytes -= kept_bands * detail_h * detail_w * 8
        current_bytes = output_bytes

    return peak


def estimate_wavelet_dec_rec(
    *,
    height: int,
    width: int,
    level: int,
) -> CudaMemoryEstimate:
    if min(height, width, level) <= 0:
        raise ValueError(
            "wavelet reconstruction CUDA memory estimate requires positive inputs"
        )
    return CudaMemoryEstimate(
        logical_op="wavelet_dec_rec_cuda_core",
        peak_device_bytes=_wavelet_device_peak_bytes(height, width, level),
        confidence="exact",
        reason="level-aware DWT detail retention and IDWT reconstruction workspace",
    )


def cuda_chunk_memory_model(
    logical_op: str,
    *,
    n_frames: int,
    row_bytes: int,
    dtype_bytes: int,
    include_mask: bool = True,
    include_weights: bool = True,
) -> CudaChunkMemoryModel:
    if min(n_frames, row_bytes, dtype_bytes) <= 0:
        raise ValueError("CUDA chunk memory model requires positive inputs")
    if row_bytes % dtype_bytes != 0:
        raise ValueError("row_bytes must be divisible by dtype_bytes")

    items_per_row = row_bytes // dtype_bytes
    float64_row = items_per_row * 8
    if logical_op == "sigma_clip_fused_chunk":
        mask_bytes = n_frames * items_per_row if include_mask else 0
        return CudaChunkMemoryModel(
            logical_op=logical_op,
            host_bytes_per_row=(
                3 * n_frames * row_bytes
                + n_frames * items_per_row
                + 3 * float64_row
            ),
            device_bytes_per_row=(
                n_frames * row_bytes + mask_bytes + 3 * float64_row
            ),
            confidence="exact",
            reason=(
                "stack, optional mask, and three float64 output planes; "
                "planner host cost also includes source and stacked chunks"
            ),
        )
    if logical_op == "huber_weighted_chunk":
        weights_bytes = n_frames * 8 if include_weights else 0
        return CudaChunkMemoryModel(
            logical_op=logical_op,
            host_bytes_per_row=(
                2 * n_frames * row_bytes + 4 * float64_row
            ),
            device_bytes_per_row=(
                n_frames * row_bytes + 4 * float64_row
            ),
            fixed_device_bytes=weights_bytes,
            confidence="exact",
            reason=(
                "stack, reference mean/std, two float64 outputs, and "
                "optional per-frame weights"
            ),
        )
    raise KeyError(f"no CUDA chunk memory model registered for {logical_op}")


def chunk_host_cost_per_row(
    logical_op: str,
    backend: str,
    *,
    n_frames: int,
    row_bytes: int,
    dtype_bytes: int,
) -> int:
    if min(n_frames, row_bytes, dtype_bytes) <= 0:
        raise ValueError("chunk memory cost requires positive inputs")
    if row_bytes % dtype_bytes != 0:
        raise ValueError("row_bytes must be divisible by dtype_bytes")
    items_per_row = row_bytes // dtype_bytes
    float64_row = items_per_row * 8

    if logical_op == "sigma_clip_fused_chunk":
        if backend == "numpy":
            return (
                2 * n_frames * row_bytes
                + 2 * n_frames * float64_row
                + 2 * n_frames * items_per_row
                + 3 * float64_row
                + 6 * float64_row
            )
        return cuda_chunk_memory_model(
            logical_op,
            n_frames=n_frames,
            row_bytes=row_bytes,
            dtype_bytes=dtype_bytes,
        ).host_bytes_per_row
    if logical_op == "huber_weighted_chunk":
        return cuda_chunk_memory_model(
            logical_op,
            n_frames=n_frames,
            row_bytes=row_bytes,
            dtype_bytes=dtype_bytes,
        ).host_bytes_per_row
    raise KeyError(f"no chunk host memory model registered for {logical_op}")


def _wavelet_reconstructed_bytes(height: int, width: int, level: int) -> int:
    current_h, current_w = height, width
    shapes: list[tuple[int, int]] = []
    for _ in range(level):
        current_h, current_w = _dwt_len(current_h), _dwt_len(current_w)
        shapes.append((current_h, current_w))
    for detail_h, detail_w in reversed(shapes):
        current_h, current_w = _idwt_len(detail_h), _idwt_len(detail_w)
    return current_h * current_w * 8


STAR_DETECT_SELECT_BINS = 2048


def estimate_star_detect_fused_pixel_components(
    *,
    height: int,
    width: int,
    small_height: int,
    small_width: int,
    level: int,
    gaussian_ksize: int,
    foreground_count: int | None = None,
) -> CudaMemoryEstimate:
    if min(height, width, small_height, small_width, level, gaussian_ksize) <= 0:
        raise ValueError("star detection CUDA memory estimate requires positive dimensions")

    pixels = height * width
    double_plane = pixels * 8
    small_double_plane = small_height * small_width * 8
    byte_plane = pixels
    int_plane = pixels * 4
    counter = 4
    identity_resize = (small_height, small_width) == (height, width)

    # Phases follow the kernel's allocation order; each term is the set of
    # project-controlled buffers alive at that point.
    blur_peak = 2 * double_plane + gaussian_ksize * 8 + byte_plane + counter
    resize_down_peak = (
        0 if identity_resize
        else double_plane + small_double_plane + byte_plane + counter)
    wavelet_peak = byte_plane + counter + _wavelet_device_peak_bytes(
        small_height, small_width, level)
    reconstructed = _wavelet_reconstructed_bytes(small_height, small_width, level)
    resize_up_peak = (
        0 if reconstructed == double_plane
        else double_plane + reconstructed + byte_plane + counter)
    # Selection candidates are bounded by the masked pixel count.
    select_peak = (
        2 * double_plane + byte_plane + counter + STAR_DETECT_SELECT_BINS * 4)
    morphology_peak = double_plane + 3 * byte_plane + counter

    max_foreground = max(
        1, int(pixels * STAR_DETECT_MAX_FOREGROUND_FRACTION))
    foreground = max_foreground if foreground_count is None else foreground_count
    foreground = max(0, min(foreground, max_foreground))
    hash_capacity = _next_power_of_two(max(16, foreground * 2))
    hash_bytes = hash_capacity * (2 * 4 + 3 * 8)
    labels_peak = (
        double_plane + byte_plane + foreground * 4 + 2 * int_plane + 2 * counter)
    cc_peak = (
        double_plane + foreground * 4 + 2 * int_plane + hash_bytes + 3 * counter)
    output_peak = hash_bytes + foreground * 3 * 8 + 3 * counter
    pinned_peak = (
        2 * 8 + 4 + 2 * CUDA_STAGING_SLOT_BYTES + STAR_DETECT_SELECT_BINS * 4)

    return CudaMemoryEstimate(
        logical_op="star_detect_fused_pixel_components",
        peak_device_bytes=max(
            blur_peak,
            resize_down_peak,
            wavelet_peak,
            resize_up_peak,
            select_peak,
            morphology_peak,
            labels_peak,
            cc_peak,
            output_peak,
        ),
        peak_pinned_bytes=pinned_peak,
        confidence="bounded",
        reason=(
            "phase-aware project-controlled buffers; selection candidates and "
            "foreground_count (capped at 25%) use their upper bounds, and the "
            "small Thrust reduction scratch is not ledger-tracked"
        ),
    )


_CUDA_STATIC_MEMORY_ESTIMATORS: dict[
    str, Callable[..., CudaMemoryEstimate]
] = {
    "asterism_mutual_nearest": estimate_asterism_mutual_nearest,
    "camera_model_remap": estimate_camera_model_remap,
    "extract_point_features": estimate_extract_point_features,
    "matching_cosine_bidirectional_nearest": (
        estimate_matching_cosine_bidirectional_nearest
    ),
    "median_reduce_chunk": estimate_median_reduce_chunk,
    "star_mask_dog": estimate_star_mask_dog,
    "star_shrink_dog_process": estimate_star_shrink_dog_process,
    "star_shrink_process": estimate_star_shrink_process,
    "wavelet_dec_rec": estimate_wavelet_dec_rec,
    "wavelet_dec_rec_cuda_core": estimate_wavelet_dec_rec,
}
_CUDA_PHASE_MEMORY_ESTIMATORS: dict[
    str, Callable[..., CudaMemoryEstimate]
] = {
    "star_detect_fused_pixel_components": (
        estimate_star_detect_fused_pixel_components
    ),
}
_CUDA_CHUNK_MEMORY_MODELS = frozenset(
    {"huber_weighted_chunk", "sigma_clip_fused_chunk"}
)


def cuda_memory_model_kind(logical_op: str) -> str:
    if logical_op in _CUDA_CHUNK_MEMORY_MODELS:
        return "cuda_chunk"
    if logical_op in _CUDA_PHASE_MEMORY_ESTIMATORS:
        return "phase_estimator"
    if logical_op in _CUDA_STATIC_MEMORY_ESTIMATORS:
        return "static_estimator"
    raise KeyError(f"no CUDA memory model registered for {logical_op}")


def cuda_memory_estimate(
    logical_op: str,
    **kwargs: int,
) -> CudaMemoryEstimate:
    estimator = _CUDA_STATIC_MEMORY_ESTIMATORS.get(logical_op)
    if estimator is None:
        estimator = _CUDA_PHASE_MEMORY_ESTIMATORS.get(logical_op)
    if estimator is None:
        raise KeyError(f"no CUDA non-chunk memory estimator registered for {logical_op}")
    estimate = estimator(**kwargs)
    if estimate.logical_op != logical_op:
        estimate = replace(estimate, logical_op=logical_op)
    return estimate


def _headroom_bytes(total_bytes: int) -> int:
    return max(
        CUDA_ADMISSION_FIXED_HEADROOM_BYTES,
        int(total_bytes * CUDA_ADMISSION_HEADROOM_FRACTION),
    )


def cuda_usable_memory_bytes(
    free_bytes: int,
    total_bytes: int,
    reserved_bytes: int = 0,
) -> int:
    headroom = _headroom_bytes(total_bytes)
    return max(0, min(total_bytes, free_bytes) - headroom - reserved_bytes)


def _clear_current_thread_cuda_cache() -> bool:
    module, _ = load_compiled_module()
    if module is None or not hasattr(module, "clear_cuda_host_io_cache"):
        return False
    return bool(module.clear_cuda_host_io_cache())


def _probe_admission(
    estimate: CudaMemoryEstimate,
    *,
    cache_evicted: bool,
) -> CudaAdmissionDecision:
    try:
        info = cuda_memory_info()
    except CudaProbeError as exc:
        if exc.category == "resource":
            raise CustomOpResourceExhaustedError(
                f"CUDA memory probe exhausted resources: {exc}"
            ) from exc
        raise
    if not info.get("available"):
        if info.get("status") == "explicitly_unavailable":
            reason_code = str(
                info.get("reason_code") or "cuda_runtime_unavailable"
            )
            raise CustomOpCudaRuntimeUnavailableError(
                str(info.get("reason") or "CUDA runtime is unavailable"),
                reason_code=reason_code,
            )
        return CudaAdmissionDecision(
            logical_op=estimate.logical_op,
            granted=True,
            checked=False,
            reason_code="cuda_memory_probe_unavailable",
            estimated_peak_bytes=estimate.peak_device_bytes,
            cache_evicted=cache_evicted,
        )

    device = int(info["device"])
    free_bytes = int(info["free_bytes"])
    total_bytes = int(info["total_bytes"])
    headroom = _headroom_bytes(total_bytes)
    reserved = _reserved_bytes_by_device.get(device, 0)
    # Reservations may later also reduce free_bytes.  Counting both is a
    # deliberate conservative v0 policy: a false rejection is safer than two
    # workers concurrently committing more than the currently free memory.
    granted = estimate.peak_device_bytes <= cuda_usable_memory_bytes(
        free_bytes, total_bytes, reserved)
    return CudaAdmissionDecision(
        logical_op=estimate.logical_op,
        granted=granted,
        checked=True,
        reason_code=("admitted" if granted else "insufficient_vram_estimate"),
        estimated_peak_bytes=estimate.peak_device_bytes,
        device=device,
        free_bytes=free_bytes,
        total_bytes=total_bytes,
        headroom_bytes=headroom,
        reserved_bytes=reserved,
        cache_evicted=cache_evicted,
    )


@contextmanager
def cuda_memory_admission(
    estimate: CudaMemoryEstimate,
    *,
    evict_cache_once: bool = True,
) -> Iterator[CudaAdmissionDecision]:
    with _reservation_lock:
        decision = _probe_admission(estimate, cache_evicted=False)
        if decision.checked and not decision.granted and evict_cache_once:
            if _clear_current_thread_cuda_cache():
                decision = _probe_admission(estimate, cache_evicted=True)
        if decision.granted and decision.checked and decision.device is not None:
            _reserved_bytes_by_device[decision.device] = (
                _reserved_bytes_by_device.get(decision.device, 0)
                + estimate.peak_device_bytes
            )

    try:
        yield decision
    finally:
        if decision.granted and decision.checked and decision.device is not None:
            with _reservation_lock:
                remaining = (
                    _reserved_bytes_by_device.get(decision.device, 0)
                    - estimate.peak_device_bytes
                )
                if remaining > 0:
                    _reserved_bytes_by_device[decision.device] = remaining
                else:
                    _reserved_bytes_by_device.pop(decision.device, None)


def run_admitted_cuda(
    estimate: CudaMemoryEstimate,
    kernel: Callable[..., _T],
    *args: object,
) -> _T:
    """Run a CUDA host-I/O kernel under VRAM admission control.

    Raises CustomOpResourceExhaustedError when the estimated peak exceeds
    usable VRAM, so callers re-resolve onto their declared fallback backend.
    """
    with cuda_memory_admission(estimate) as admission:
        if not admission.granted:
            raise CustomOpResourceExhaustedError(
                f"{estimate.logical_op} skipped CUDA because estimated peak "
                f"{admission.estimated_peak_bytes} bytes exceeds usable VRAM "
                f"(free={admission.free_bytes}, "
                f"reserved={admission.reserved_bytes}, "
                f"headroom={admission.headroom_bytes})"
            )
        return kernel(*args)


def _reset_cuda_reservations_for_tests() -> None:
    with _reservation_lock:
        _reserved_bytes_by_device.clear()
