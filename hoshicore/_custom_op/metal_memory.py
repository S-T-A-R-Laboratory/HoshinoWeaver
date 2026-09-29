"""Metal unified-memory estimates and runtime admission."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator, TypeVar

from hoshicore._custom_op._dispatch import CustomOpMetalRuntimeUnavailableError
from hoshicore._custom_op._dispatch import CustomOpResourceExhaustedError
from hoshicore._custom_op._dispatch import MetalProbeError
from hoshicore._custom_op._dispatch import load_metal_module
from hoshicore._custom_op._dispatch import metal_device_info


METAL_ADMISSION_FIXED_HEADROOM_BYTES = 256 * 1024 * 1024
METAL_ADMISSION_HEADROOM_FRACTION = 0.05

_T = TypeVar("_T")


@dataclass(frozen=True)
class MetalMemoryEstimate:
    logical_op: str
    peak_device_bytes: int
    confidence: str = "bounded"
    reason: str = ""


@dataclass(frozen=True)
class MetalAdmissionDecision:
    logical_op: str
    granted: bool
    checked: bool
    reason_code: str
    estimated_peak_bytes: int
    recommended_max_working_set_bytes: int | None = None
    current_allocated_bytes: int | None = None
    headroom_bytes: int = 0
    reserved_bytes: int = 0
    cache_evicted: bool = False


_reservation_lock = threading.Lock()
_reserved_bytes = 0


def estimate_star_shrink_process(
    *,
    height: int,
    width: int,
    channels: int,
    dtype_bytes: int,
) -> MetalMemoryEstimate:
    if min(height, width, channels, dtype_bytes) <= 0:
        raise ValueError(
            "star-shrink Metal memory estimate requires positive dimensions"
        )
    pixels = height * width
    total = pixels * channels
    image_bytes = total * dtype_bytes
    mask_bytes = pixels
    plane_float_bytes = pixels * 4
    total_float_bytes = total * 4
    return MetalMemoryEstimate(
        logical_op="star_shrink_process",
        peak_device_bytes=(
            2 * image_bytes
            + mask_bytes
            + 4 * plane_float_bytes
            + 3 * total_float_bytes
        ),
        confidence="exact",
        reason=(
            "shared Metal image, mask, Lab, morphology, and blur workspace buffers"
        ),
    )


def estimate_star_mask_dog(
    *,
    height: int,
    width: int,
    channels: int,
    dtype_bytes: int,
    small_kernel_size: int,
    large_kernel_size: int,
) -> MetalMemoryEstimate:
    if min(height, width, channels, dtype_bytes) <= 0:
        raise ValueError(
            "star-mask-DoG Metal memory estimate requires positive dimensions"
        )
    if min(small_kernel_size, large_kernel_size) <= 0:
        raise ValueError(
            "star-mask-DoG Metal memory estimate requires positive kernel sizes"
        )
    pixels = height * width
    image_bytes = pixels * channels * dtype_bytes
    plane_float_bytes = pixels * 4
    weight_bytes = (small_kernel_size + large_kernel_size) * 4
    return MetalMemoryEstimate(
        logical_op="star_mask_dog",
        # gray, tmp, blur_small, blur_large, dog; mask and scratch are uint8.
        peak_device_bytes=(
            image_bytes + 5 * plane_float_bytes + weight_bytes + 2 * pixels
        ),
        confidence="exact",
        reason="shared Metal image, blur, DoG, Gaussian weight, and mask buffers",
    )


def estimate_median_filter_2d(*, height: int, width: int) -> MetalMemoryEstimate:
    if min(height, width) <= 0:
        raise ValueError("Metal median filter requires positive dimensions")
    return MetalMemoryEstimate(
        logical_op="median_filter_2d",
        peak_device_bytes=2 * height * width * 2,
        confidence="exact",
        reason="input and output uint16 planes in the Metal host-I/O workspace",
    )


def estimate_star_shrink_dog_process(
    *,
    height: int,
    width: int,
    channels: int,
    dtype_bytes: int,
    small_kernel_size: int,
    large_kernel_size: int,
) -> MetalMemoryEstimate:
    if min(height, width, channels, dtype_bytes) <= 0:
        raise ValueError(
            "fused DoG-shrink Metal memory estimate requires positive dimensions"
        )
    if min(small_kernel_size, large_kernel_size) <= 0:
        raise ValueError(
            "fused DoG-shrink Metal memory estimate requires positive kernel sizes"
        )
    pixels = height * width
    total = pixels * channels
    image_bytes = total * dtype_bytes
    plane_float_bytes = pixels * 4
    total_float_bytes = total * 4
    weight_bytes = (small_kernel_size + large_kernel_size) * 4
    return MetalMemoryEstimate(
        logical_op="star_shrink_dog_process",
        # image + output; 5 plane floats (gray, tmp, blur_small, blur_large, dog)
        # with the shrink stage aliasing the first four; mask + scratch; shrunk,
        # box_tmp, blurred.
        peak_device_bytes=(
            2 * image_bytes
            + 5 * plane_float_bytes
            + weight_bytes
            + 2 * pixels
            + 3 * total_float_bytes
        ),
        confidence="exact",
        reason="shared Metal detection and shrink workspace buffers, fused in one pass",
    )


_METAL_STATIC_MEMORY_ESTIMATORS = {
    "median_filter_2d": estimate_median_filter_2d,
    "star_shrink_process": estimate_star_shrink_process,
    "star_mask_dog": estimate_star_mask_dog,
    "star_shrink_dog_process": estimate_star_shrink_dog_process,
}


def metal_memory_model_kind(logical_op: str) -> str:
    if logical_op in _METAL_STATIC_MEMORY_ESTIMATORS:
        return "static_estimator"
    raise KeyError(f"no Metal memory model registered for {logical_op}")


def metal_memory_estimate(
    logical_op: str,
    **kwargs: int,
) -> MetalMemoryEstimate:
    estimator = _METAL_STATIC_MEMORY_ESTIMATORS.get(logical_op)
    if estimator is None:
        raise KeyError(f"no Metal memory estimator registered for {logical_op}")
    return estimator(**kwargs)


def _headroom_bytes(recommended_max_working_set_bytes: int) -> int:
    return max(
        METAL_ADMISSION_FIXED_HEADROOM_BYTES,
        int(
            recommended_max_working_set_bytes
            * METAL_ADMISSION_HEADROOM_FRACTION
        ),
    )


def metal_usable_memory_bytes(
    recommended_max_working_set_bytes: int,
    current_allocated_bytes: int,
    reserved_bytes: int = 0,
) -> int:
    headroom = _headroom_bytes(recommended_max_working_set_bytes)
    return max(
        0,
        recommended_max_working_set_bytes
        - current_allocated_bytes
        - headroom
        - reserved_bytes,
    )


def _clear_current_thread_metal_cache() -> bool:
    module, _ = load_metal_module()
    if module is None or not hasattr(module, "clear_metal_host_io_cache"):
        return False
    return bool(module.clear_metal_host_io_cache())


def _probe_admission(
    estimate: MetalMemoryEstimate,
    *,
    cache_evicted: bool,
) -> MetalAdmissionDecision:
    try:
        info = metal_device_info()
    except MetalProbeError as exc:
        if exc.category == "resource":
            raise CustomOpResourceExhaustedError(
                f"Metal device probe exhausted resources: {exc}"
            ) from exc
        raise
    if not info.get("available"):
        if info.get("status") == "explicitly_unavailable":
            reason_code = str(
                info.get("reason_code") or "metal_runtime_unavailable"
            )
            raise CustomOpMetalRuntimeUnavailableError(
                str(info.get("reason") or "Metal runtime is unavailable"),
                reason_code=reason_code,
            )
        return MetalAdmissionDecision(
            logical_op=estimate.logical_op,
            granted=True,
            checked=False,
            reason_code="metal_memory_probe_unavailable",
            estimated_peak_bytes=estimate.peak_device_bytes,
            cache_evicted=cache_evicted,
        )

    recommended = int(info["recommended_max_working_set_bytes"])
    current = int(info["current_allocated_bytes"])
    headroom = _headroom_bytes(recommended)
    granted = estimate.peak_device_bytes <= metal_usable_memory_bytes(
        recommended,
        current,
        _reserved_bytes,
    )
    return MetalAdmissionDecision(
        logical_op=estimate.logical_op,
        granted=granted,
        checked=True,
        reason_code=("admitted" if granted else "insufficient_working_set_estimate"),
        estimated_peak_bytes=estimate.peak_device_bytes,
        recommended_max_working_set_bytes=recommended,
        current_allocated_bytes=current,
        headroom_bytes=headroom,
        reserved_bytes=_reserved_bytes,
        cache_evicted=cache_evicted,
    )


@contextmanager
def metal_memory_admission(
    estimate: MetalMemoryEstimate,
    *,
    evict_cache_once: bool = True,
) -> Iterator[MetalAdmissionDecision]:
    global _reserved_bytes
    with _reservation_lock:
        decision = _probe_admission(estimate, cache_evicted=False)
        if decision.checked and not decision.granted and evict_cache_once:
            if _clear_current_thread_metal_cache():
                decision = _probe_admission(estimate, cache_evicted=True)
        if decision.granted and decision.checked:
            _reserved_bytes += estimate.peak_device_bytes

    try:
        yield decision
    finally:
        if decision.granted and decision.checked:
            with _reservation_lock:
                _reserved_bytes = max(
                    0,
                    _reserved_bytes - estimate.peak_device_bytes,
                )


def run_admitted_metal(
    estimate: MetalMemoryEstimate,
    kernel: Callable[..., _T],
    *args: object,
) -> _T:
    """Run a Metal host-I/O kernel under unified-memory admission control."""
    with metal_memory_admission(estimate) as admission:
        if not admission.granted:
            raise CustomOpResourceExhaustedError(
                f"{estimate.logical_op} skipped Metal because estimated peak "
                f"{admission.estimated_peak_bytes} bytes exceeds the usable "
                "working set "
                f"(recommended={admission.recommended_max_working_set_bytes}, "
                f"current={admission.current_allocated_bytes}, "
                f"reserved={admission.reserved_bytes}, "
                f"headroom={admission.headroom_bytes})"
            )
        return kernel(*args)


def _reset_metal_reservations_for_tests() -> None:
    global _reserved_bytes
    with _reservation_lock:
        _reserved_bytes = 0
