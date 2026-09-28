"""Star detection pixel-stage custom-op runtime backends."""

from __future__ import annotations

from functools import lru_cache
from functools import partial
from typing import Callable

import cv2
import numpy as np
from numpy.typing import NDArray

from hoshicore._custom_op._dispatch import CustomOpUnavailableError
from hoshicore._custom_op._dispatch import apply_compiled_threads as _apply_compiled_threads
from hoshicore._custom_op._dispatch import debug_log
from hoshicore._custom_op._dispatch import fallback_preference as _fallback_preference
from hoshicore._custom_op._dispatch import load_compiled_module as _load_compiled_module_result
from hoshicore._custom_op.backend_registry import BackendSelection
from hoshicore._custom_op.backend_registry import resolve_after_cuda_failure
from hoshicore._custom_op.backend_registry import select_backend as _select_backend
from hoshicore._custom_op.cuda_memory import cuda_memory_estimate
from hoshicore._custom_op.cuda_memory import run_admitted_cuda as _run_admitted_cuda
from hoshicore._custom_op.ops.filter import median_filter_2d_numpy as _median_filter_2d_numpy
from hoshicore._custom_op.ops.wavelet import _wavelet_level


_debug_log = partial(debug_log, "detection")


class StarDetectCapacityError(RuntimeError):
    """Raised when the selected connected-component algorithm exceeds its limits."""


def _validate_median_star_mask_inputs(
    image: np.ndarray,
    mask: np.ndarray | None,
) -> tuple[np.ndarray, NDArray[np.uint8] | None]:
    image_arr = np.asarray(image)
    if image_arr.ndim != 2 or image_arr.shape[0] <= 0 or image_arr.shape[1] <= 0:
        raise ValueError("median_star_mask: image must be a non-empty 2D array")
    if image_arr.dtype not in (np.uint8, np.uint16, np.float32, np.float64):
        raise TypeError(
            "median_star_mask: expected uint8, uint16, float32, or float64 image")
    if image_arr.dtype.kind == "f":
        if not np.all(np.isfinite(image_arr)):
            raise ValueError("median_star_mask: floating-point image must be finite")
        if image_arr.size and (
            float(image_arr.min()) < 0.0 or float(image_arr.max()) > 1.0
        ):
            raise ValueError(
                "median_star_mask: floating-point image must be normalized to [0, 1]")
    image_arr = np.ascontiguousarray(image_arr)

    if mask is None:
        return image_arr, None
    mask_arr = np.asarray(mask)
    if mask_arr.shape != image_arr.shape:
        raise ValueError("median_star_mask: mask shape must match image")
    mask_u8 = np.ascontiguousarray(mask_arr > 0, dtype=np.uint8)
    if not np.any(mask_u8):
        raise ValueError("median_star_mask: mask selects no pixels")
    return image_arr, mask_u8


def _validate_median_star_mask_params(
    median_ksize: int,
    threshold_ratio: float,
    open_ksize: int,
    dilate_ksize: int,
) -> tuple[int, float, int, int]:
    values = {
        "median_ksize": median_ksize,
        "open_ksize": open_ksize,
        "dilate_ksize": dilate_ksize,
    }
    for name, value in values.items():
        if not isinstance(value, (int, np.integer)):
            raise TypeError(f"median_star_mask: {name} must be an int")
        if name == "median_ksize":
            valid = value > 0 and value % 2 == 1
        else:
            valid = value == 0 or (value > 0 and value % 2 == 1)
        if not valid:
            requirement = (
                "a positive odd integer"
                if name == "median_ksize"
                else "zero or a positive odd integer"
            )
            raise ValueError(f"median_star_mask: {name} must be {requirement}")
    ratio = float(threshold_ratio)
    if not np.isfinite(ratio):
        raise ValueError("median_star_mask: threshold_ratio must be finite")
    return int(median_ksize), ratio, int(open_ksize), int(dilate_ksize)


def _median_working_u16(image: np.ndarray) -> NDArray[np.uint16]:
    if image.dtype == np.uint16:
        return image
    if image.dtype == np.uint8:
        return image.astype(np.uint16) * np.uint16(257)
    image_f32 = image.astype(np.float32, copy=False)
    return np.rint(image_f32 * np.float32(65535.0)).astype(np.uint16)


def median_star_mask_numpy(
    image: np.ndarray,
    median_ksize: int = 13,
    threshold_ratio: float = 1.0,
    open_ksize: int = 3,
    dilate_ksize: int = 0,
    mask: np.ndarray | None = None,
) -> tuple[NDArray[np.uint8], NDArray[np.float32], float]:
    image_arr, mask_u8 = _validate_median_star_mask_inputs(image, mask)
    median_ksize, threshold_ratio, open_ksize, dilate_ksize = (
        _validate_median_star_mask_params(
            median_ksize, threshold_ratio, open_ksize, dilate_ksize)
    )
    working = _median_working_u16(image_arr)
    background = _median_filter_2d_numpy(working, median_ksize)
    response = (
        working.astype(np.float32) - background.astype(np.float32)
    ) / np.float32(65535.0)
    valid = np.ones(response.shape, dtype=bool) if mask_u8 is None else mask_u8 > 0
    threshold = float(np.std(response[valid]) * threshold_ratio)
    star_mask = np.logical_and(response > threshold, valid).astype(np.uint8)
    if open_ksize > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_CROSS, (open_ksize, open_ksize))
        star_mask = cv2.morphologyEx(star_mask, cv2.MORPH_OPEN, kernel)
    if dilate_ksize > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_CROSS, (dilate_ksize, dilate_ksize))
        star_mask = cv2.morphologyEx(star_mask, cv2.MORPH_DILATE, kernel)
    return star_mask, response, threshold


def median_star_mask_cpu_compiled(
    image: np.ndarray,
    median_ksize: int = 13,
    threshold_ratio: float = 1.0,
    open_ksize: int = 3,
    dilate_ksize: int = 0,
    mask: np.ndarray | None = None,
) -> tuple[NDArray[np.uint8], NDArray[np.float32], float]:
    image_arr, mask_u8 = _validate_median_star_mask_inputs(image, mask)
    median_ksize, threshold_ratio, open_ksize, dilate_ksize = (
        _validate_median_star_mask_params(
            median_ksize, threshold_ratio, open_ksize, dilate_ksize)
    )
    module, _ = _load_compiled_module_result()
    if module is None or not hasattr(module, "median_star_mask_cpu"):
        raise RuntimeError("compiled median_star_mask CPU backend is unavailable")
    _apply_compiled_threads("median_star_mask", image_arr)
    star_mask, response, threshold = module.median_star_mask_cpu(
        image_arr,
        median_ksize,
        threshold_ratio,
        open_ksize,
        dilate_ksize,
        mask_u8,
    )
    return star_mask, response, float(threshold)


@lru_cache(maxsize=3)
def _select_median_star_mask_backend(preference: str) -> BackendSelection:
    return _select_backend(
        "median_star_mask",
        preference,
        load_module=_load_compiled_module_result,
    )


def median_star_mask(
    image: np.ndarray,
    median_ksize: int = 13,
    threshold_ratio: float = 1.0,
    open_ksize: int = 3,
    dilate_ksize: int = 0,
    mask: np.ndarray | None = None,
) -> tuple[NDArray[np.uint8], NDArray[np.float32], float]:
    preference = _fallback_preference()
    selection = _select_median_star_mask_backend(preference)
    backend = (
        median_star_mask_cpu_compiled
        if selection.native
        else median_star_mask_numpy
    )
    return backend(
        image,
        median_ksize=median_ksize,
        threshold_ratio=threshold_ratio,
        open_ksize=open_ksize,
        dilate_ksize=dilate_ksize,
        mask=mask,
    )


def _is_native_star_detect_capacity_error(exc: RuntimeError) -> bool:
    if isinstance(exc, StarDetectCapacityError):
        return True
    module, _ = _load_compiled_module_result()
    capacity_type = (
        getattr(module, "StarDetectCapacityError", None)
        if module is not None
        else None
    )
    return capacity_type is not None and isinstance(exc, capacity_type)


def _validate_image_mask(
    image: np.ndarray,
    mask: np.ndarray,
) -> tuple[NDArray[np.float64], NDArray[np.uint8]]:
    image_arr = np.asarray(image, dtype=np.float64)
    mask_arr = np.asarray(mask)
    if image_arr.ndim != 2:
        raise ValueError("star_detect_threshold_morph: image must be 2D")
    if mask_arr.ndim != 2:
        raise ValueError("star_detect_threshold_morph: mask must be 2D")
    if image_arr.shape != mask_arr.shape:
        raise ValueError(
            "star_detect_threshold_morph: image and mask shapes must match")
    if image_arr.shape[0] <= 0 or image_arr.shape[1] <= 0:
        raise ValueError(
            "star_detect_threshold_morph: image height and width must be positive")
    if not image_arr.flags.c_contiguous:
        image_arr = np.ascontiguousarray(image_arr)
    mask_u8 = (mask_arr > 0).astype(np.uint8, copy=False)
    if not mask_u8.flags.c_contiguous:
        mask_u8 = np.ascontiguousarray(mask_u8)
    return image_arr, mask_u8


def star_detect_threshold_morph_numpy(
    image: np.ndarray,
    mask: np.ndarray,
) -> NDArray[np.uint8]:
    """Return the NumPy/OpenCV pixel-stage baseline used by tests and benchmarks."""
    image_arr, mask_u8 = _validate_image_mask(image, mask)
    mask_bool = mask_u8 > 0
    threshold = np.percentile(image_arr[mask_bool], 99.5)
    bw = ((image_arr > threshold) * mask_bool).astype(np.uint8) * 255
    return cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


class GraySource:
    """An image standing for the detection gray ``convert(raw)``.

    ``raw`` is the source image and ``convert`` the host conversion (Norma's
    ``to_gray_f64``). The CUDA backend uploads a uint8/uint16 (H, W) or
    (H, W, 3) BGR ``raw`` and converts it on the device once that is verified
    to reproduce ``convert`` exactly; other backends use ``host_gray()``.
    """

    __slots__ = ("raw", "_convert", "_host_gray", "_gray")

    def __init__(
        self,
        raw: np.ndarray,
        convert: Callable[[np.ndarray], NDArray[np.float64]],
        host_gray: Callable[[], NDArray[np.float64]] | None = None,
    ) -> None:
        self.raw = raw
        self._convert = convert
        self._host_gray = host_gray
        self._gray: NDArray[np.float64] | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.raw.shape[:2])

    def host_gray(self) -> NDArray[np.float64]:
        if self._gray is None:
            self._gray = (self._host_gray() if self._host_gray is not None
                          else self._convert(self.raw))
        return self._gray


def _validate_fused_pixel_component_mask(
    mask: np.ndarray | None,
    shape: tuple[int, int],
) -> NDArray[np.uint8] | None:
    if mask is None:
        return None
    mask_arr = np.asarray(mask)
    if mask_arr.ndim != 2 or mask_arr.shape != tuple(shape):
        raise ValueError(
            "star_detect_fused_pixel_components: mask shape must match image"
        )
    mask_u8 = (mask_arr > 0).astype(np.uint8, copy=False)
    if not mask_u8.flags.c_contiguous:
        mask_u8 = np.ascontiguousarray(mask_u8)
    return mask_u8


def _validate_fused_pixel_component_inputs(
    image: np.ndarray,
    mask: np.ndarray | None,
) -> tuple[NDArray[np.float64], NDArray[np.uint8] | None]:
    image_arr = np.asarray(image, dtype=np.float64)
    if image_arr.ndim != 2:
        raise ValueError("star_detect_fused_pixel_components: image must be 2D")
    if image_arr.shape[0] <= 0 or image_arr.shape[1] <= 0:
        raise ValueError(
            "star_detect_fused_pixel_components: image height and width must be positive"
        )
    if not image_arr.flags.c_contiguous:
        image_arr = np.ascontiguousarray(image_arr)
    return image_arr, _validate_fused_pixel_component_mask(mask, image_arr.shape)


# Widths covering every leftover-pixel count of the conversion's SIMD blocks.
_GRAY_PROBE_WIDTHS = (*range(1, 18), 64)


def _opencv_gray_config() -> tuple[bool, ...]:
    """OpenCV switches that select its float BGR->GRAY implementation."""
    config = [cv2.useOptimized(), cv2.ipp.useIPP()]
    if hasattr(cv2.ipp, "useIPP_NotExact"):
        config.append(cv2.ipp.useIPP_NotExact())
    return tuple(config)


@lru_cache(maxsize=8)
def _device_gray_matches(
    dtype: str,
    channels: int,
    convert: Callable[[np.ndarray], NDArray[np.float64]],
    opencv_config: tuple[bool, ...],
) -> bool:
    """Whether the device conversion reproduces ``convert`` bit for bit.

    It mirrors OpenCV's IPP float BGR->GRAY arithmetic, which other OpenCV
    builds, CPUs or runtime switches may not share, so the check runs once per
    format and ``opencv_config``. CUDA errors propagate uncached.
    """
    del opencv_config  # cache key only
    module, _ = _load_compiled_module_result()
    if module is None or not hasattr(module, "star_detect_source_gray_cuda"):
        return False
    rng = np.random.default_rng(0)
    info = np.iinfo(np.dtype(dtype))
    for width in _GRAY_PROBE_WIDTHS:
        shape = (3, width, 3) if channels == 3 else (3, width)
        probe = rng.integers(0, int(info.max) + 1, size=shape, dtype=np.dtype(dtype))
        if not np.array_equal(module.star_detect_source_gray_cuda(probe), convert(probe)):
            _debug_log(f"device gray differs from the host conversion for {dtype} x{channels}")
            return False
    return True


def _device_gray_input(source: GraySource) -> np.ndarray | None:
    raw = source.raw
    if raw.dtype not in (np.uint8, np.uint16):
        return None
    if not (raw.ndim == 2 or (raw.ndim == 3 and raw.shape[2] == 3)):
        return None
    if min(raw.shape[:2]) <= 0:
        return None
    channels = 3 if raw.ndim == 3 else 1
    if not _device_gray_matches(raw.dtype.str, channels, source._convert,
                                _opencv_gray_config()):
        return None
    return np.ascontiguousarray(raw)


def _fused_pixel_component_kernel_params(
    shape: tuple[int, int],
    resize_factor: float,
    gaussian_ksize: int,
    sigma: float,
) -> tuple[int, int, int, NDArray[np.float64]]:
    if resize_factor <= 0:
        raise ValueError(
            "star_detect_fused_pixel_components: resize_factor must be positive")
    if gaussian_ksize <= 0 or gaussian_ksize % 2 == 0:
        raise ValueError(
            "star_detect_fused_pixel_components: gaussian_ksize must be positive and odd"
        )
    height, width = shape
    small_height = max(1, int(round(height * resize_factor)))
    small_width = max(1, int(round(width * resize_factor)))
    level = _wavelet_level(float(resize_factor))
    gaussian_kernel = cv2.getGaussianKernel(
        int(gaussian_ksize), float(sigma), ktype=cv2.CV_64F
    ).reshape(-1)
    if not gaussian_kernel.flags.c_contiguous:
        gaussian_kernel = np.ascontiguousarray(gaussian_kernel)
    return small_height, small_width, level, gaussian_kernel


def _star_detect_fused_pixel_components_native_validated(
    kernel_name: str,
    image_arr: NDArray[np.float64],
    mask_u8: NDArray[np.uint8] | None,
    resize_factor: float,
    gaussian_ksize: int = 9,
    sigma: float = 2.0,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.uint8],
]:
    module, _ = _load_compiled_module_result()
    if module is None or not hasattr(module, kernel_name):
        raise RuntimeError("compiled custom op backend is unavailable")
    small_height, small_width, level, gaussian_kernel = (
        _fused_pixel_component_kernel_params(
            image_arr.shape, resize_factor, gaussian_ksize, sigma)
    )
    kernel = getattr(module, kernel_name)
    kernel_args = (
        image_arr,
        mask_u8,
        small_height,
        small_width,
        level,
        gaussian_kernel,
    )
    if kernel_name == "star_detect_fused_pixel_components_cpu":
        _apply_compiled_threads("star_detect_fused_pixel_components", image_arr)
        return kernel(*kernel_args)

    estimate = cuda_memory_estimate(
        "star_detect_fused_pixel_components",
        height=image_arr.shape[0],
        width=image_arr.shape[1],
        small_height=small_height,
        small_width=small_width,
        level=level,
        gaussian_ksize=gaussian_kernel.size,
    )
    return _run_admitted_cuda(estimate, kernel, *kernel_args)


def _star_detect_fused_pixel_components_source_validated(
    raw: np.ndarray,
    mask_u8: NDArray[np.uint8] | None,
    resize_factor: float,
    gaussian_ksize: int = 9,
    sigma: float = 2.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.uint8]] | None:
    """CUDA detection of a device-converted gray; None when it is constant."""
    module, _ = _load_compiled_module_result()
    kernel_name = "star_detect_fused_pixel_components_cuda_from_source"
    if module is None or not hasattr(module, kernel_name):
        raise RuntimeError("compiled custom op backend is unavailable")
    small_height, small_width, level, gaussian_kernel = (
        _fused_pixel_component_kernel_params(
            raw.shape[:2], resize_factor, gaussian_ksize, sigma)
    )
    # The source is staged in the blur-row buffer, so the peak is unchanged.
    estimate = cuda_memory_estimate(
        "star_detect_fused_pixel_components",
        height=raw.shape[0],
        width=raw.shape[1],
        small_height=small_height,
        small_width=small_width,
        level=level,
        gaussian_ksize=gaussian_kernel.size,
    )
    return _run_admitted_cuda(
        estimate, getattr(module, kernel_name), raw, mask_u8,
        small_height, small_width, level, gaussian_kernel)


def _star_detect_fused_pixel_components_compiled_validated(
    image_arr: NDArray[np.float64],
    mask_u8: NDArray[np.uint8] | None,
    resize_factor: float,
    gaussian_ksize: int = 9,
    sigma: float = 2.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.uint8]]:
    return _star_detect_fused_pixel_components_native_validated(
        "star_detect_fused_pixel_components_cuda",
        image_arr,
        mask_u8,
        resize_factor,
        gaussian_ksize,
        sigma,
    )


def _star_detect_fused_pixel_components_cpu_validated(
    image_arr: NDArray[np.float64],
    mask_u8: NDArray[np.uint8] | None,
    resize_factor: float,
    gaussian_ksize: int = 9,
    sigma: float = 2.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.uint8]]:
    return _star_detect_fused_pixel_components_native_validated(
        "star_detect_fused_pixel_components_cpu",
        image_arr,
        mask_u8,
        resize_factor,
        gaussian_ksize,
        sigma,
    )


def star_detect_fused_pixel_components_compiled(
    image: np.ndarray,
    mask: np.ndarray | None,
    resize_factor: float,
    gaussian_ksize: int = 9,
    sigma: float = 2.0,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.uint8],
]:
    image_arr, mask_u8 = _validate_fused_pixel_component_inputs(image, mask)
    return _star_detect_fused_pixel_components_compiled_validated(
        image_arr,
        mask_u8,
        resize_factor,
        gaussian_ksize,
        sigma,
    )


def star_detect_fused_pixel_components_compiled_source(
    source: GraySource,
    mask: np.ndarray | None,
    resize_factor: float,
    gaussian_ksize: int = 9,
    sigma: float = 2.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.uint8]] | None:
    """CUDA detection of a :class:`GraySource`, converted on the device when
    verified and on the host otherwise; None when its gray is constant."""
    mask_u8 = _validate_fused_pixel_component_mask(mask, source.shape)
    raw = _device_gray_input(source)
    if raw is not None:
        return _star_detect_fused_pixel_components_source_validated(
            raw, mask_u8, resize_factor, gaussian_ksize, sigma)
    gray = source.host_gray()
    if np.ptp(gray) == 0:
        return None
    return _star_detect_fused_pixel_components_compiled_validated(
        _validate_fused_pixel_component_inputs(gray, None)[0], mask_u8, resize_factor,
        gaussian_ksize, sigma)


def star_detect_fused_pixel_components_compiled_cpu(
    image: np.ndarray,
    mask: np.ndarray | None,
    resize_factor: float,
    gaussian_ksize: int = 9,
    sigma: float = 2.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.uint8]]:
    image_arr, mask_u8 = _validate_fused_pixel_component_inputs(image, mask)
    return _star_detect_fused_pixel_components_cpu_validated(
        image_arr,
        mask_u8,
        resize_factor,
        gaussian_ksize,
        sigma,
    )


@lru_cache(maxsize=3)
def _select_star_detect_fused_pixel_components_backend(
    preference: str,
) -> BackendSelection:
    return _select_backend(
        "star_detect_fused_pixel_components",
        preference,
        load_module=_load_compiled_module_result,
    )


def _star_detect_fused_pixel_components_backend(
    selection: BackendSelection,
) -> tuple[str, Callable[..., tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.uint8]]]]:
    if not selection.native or selection.candidate is None:
        reason = selection.reason or "compiled pixel-component backend unavailable"
        raise CustomOpUnavailableError(
            "star_detect_fused_pixel_components requires a compiled backend; "
            f"use the contour detector for production fallback. {reason}"
        )
    if selection.candidate.kernel_name == "star_detect_fused_pixel_components_cuda":
        return "cuda", _star_detect_fused_pixel_components_compiled_validated
    if selection.candidate.kernel_name == "star_detect_fused_pixel_components_cpu":
        return "cpu", _star_detect_fused_pixel_components_cpu_validated
    raise RuntimeError(
        "unknown star_detect_fused_pixel_components backend candidate: "
        f"{selection.candidate}"
    )


def star_detect_fused_pixel_components(
    image: np.ndarray | GraySource,
    mask: np.ndarray | None,
    resize_factor: float,
    gaussian_ksize: int = 9,
    sigma: float = 2.0,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.uint8],
] | None:
    """Fused native pixel stage of star detection.

    ``image`` is a float64 gray, or a :class:`GraySource` whose gray the CUDA
    backend may convert on the device; for a source, returns None when its
    gray is constant.
    """
    source = image if isinstance(image, GraySource) else None
    if source is None:
        image_arr, mask_u8 = _validate_fused_pixel_component_inputs(image, mask)
    else:
        mask_u8 = _validate_fused_pixel_component_mask(mask, source.shape)
    preference = _fallback_preference()
    selection = _select_star_detect_fused_pixel_components_backend(preference)
    backend_name, backend = _star_detect_fused_pixel_components_backend(selection)

    def host_args():
        gray = image_arr if source is None else source.host_gray()
        if source is not None:
            if np.ptp(gray) == 0:
                return None
            gray = _validate_fused_pixel_component_inputs(gray, None)[0]
        return gray, mask_u8, resize_factor, int(gaussian_ksize), float(sigma)

    if backend_name != "cuda":
        kernel_args = host_args()
        return None if kernel_args is None else backend(*kernel_args)
    try:
        # The device-gray check runs CUDA too, so its failures take the same
        # fallback path as the detection kernel.
        device_raw = _device_gray_input(source) if source is not None else None
        if device_raw is not None:
            return _star_detect_fused_pixel_components_source_validated(
                device_raw, mask_u8, resize_factor, int(gaussian_ksize), float(sigma))
        kernel_args = host_args()
        if kernel_args is None:
            return None
        return backend(*kernel_args)
    except RuntimeError as exc:
        if _is_native_star_detect_capacity_error(exc):
            fallback_selection = _select_backend(
                "star_detect_fused_pixel_components",
                preference,
                load_module=_load_compiled_module_result,
                exclude_backends={"cuda_host_io"},
            )
            if not fallback_selection.native:
                raise StarDetectCapacityError(str(exc)) from exc
            _debug_log(f"CUDA component capacity exceeded, falling back to CPU: {exc}")
        else:
            fallback_selection = resolve_after_cuda_failure(
                "star_detect_fused_pixel_components",
                exc,
                load_module=_load_compiled_module_result,
                log=_debug_log,
            )

    fallback_name, fallback_backend = _star_detect_fused_pixel_components_backend(
        fallback_selection
    )
    if fallback_name == "cuda":
        raise RuntimeError("CUDA backend remained selected after runtime exclusion")
    kernel_args = host_args()
    if kernel_args is None:
        return None
    return fallback_backend(*kernel_args)
