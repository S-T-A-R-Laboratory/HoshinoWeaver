"""Spatial filter custom-op runtime backends."""

from __future__ import annotations

from functools import lru_cache
from functools import partial
from typing import Callable

import cv2
import numpy as np

from hoshicore._custom_op._dispatch import apply_compiled_threads as _apply_compiled_threads
from hoshicore._custom_op._dispatch import debug_log
from hoshicore._custom_op._dispatch import fallback_preference as _fallback_preference
from hoshicore._custom_op._dispatch import load_compiled_module as _load_compiled_module_result
from hoshicore._custom_op.backend_registry import native_backend_available as _native_backend_available


_debug_log = partial(debug_log, "filter")

_SUPPORTED_DTYPES = (np.uint8, np.uint16)
_SUPPORTED_CHANNELS = {1, 3, 4}
_MAX_MEDIAN_FILTER_KSIZE = 65535


def _validate_ksize(ksize: int) -> int:
    if not isinstance(ksize, int):
        raise TypeError("median_filter_2d: ksize must be an int")
    if ksize <= 0 or ksize % 2 == 0:
        raise ValueError("median_filter_2d: ksize must be a positive odd integer")
    if ksize > _MAX_MEDIAN_FILTER_KSIZE:
        raise ValueError("median_filter_2d: ksize is too large")
    return ksize


def _validate_image(image: np.ndarray) -> np.ndarray:
    image_arr = np.asarray(image)
    if image_arr.ndim not in {2, 3}:
        raise ValueError(
            "median_filter_2d: image must have shape (H, W) or (H, W, C)")
    if image_arr.shape[0] <= 0 or image_arr.shape[1] <= 0:
        raise ValueError("median_filter_2d: image height and width must be positive")
    if image_arr.ndim == 3 and image_arr.shape[2] not in _SUPPORTED_CHANNELS:
        raise ValueError("median_filter_2d: channel count must be 1, 3, or 4")
    if image_arr.dtype.type not in _SUPPORTED_DTYPES:
        raise ValueError("median_filter_2d: unsupported dtype; expected uint8/uint16")
    if not image_arr.flags.c_contiguous:
        image_arr = np.ascontiguousarray(image_arr)
    return image_arr


def median_filter_2d_numpy(image: np.ndarray, ksize: int) -> np.ndarray:
    image_arr = _validate_image(image)
    ksize = _validate_ksize(ksize)

    # OpenCV is the fastest exact fallback for supported cases.
    if image_arr.dtype == np.uint8 or ksize <= 5:
        result = cv2.medianBlur(image_arr, ksize)
        if image_arr.ndim == 3 and image_arr.shape[2] == 1 and result.ndim == 2:
            result = result[:, :, None]
        return np.ascontiguousarray(result)

    from scipy.ndimage import median_filter as scipy_median_filter

    size = (ksize, ksize, 1) if image_arr.ndim == 3 else ksize
    result = scipy_median_filter(image_arr, size=size, mode="nearest")
    if result.dtype != image_arr.dtype:
        result = result.astype(image_arr.dtype)
    return np.ascontiguousarray(result)


def median_filter_2d_compiled(image: np.ndarray, ksize: int) -> np.ndarray:
    module, _ = _load_compiled_module_result()
    if module is None or not hasattr(module, "median_filter_2d"):
        raise RuntimeError("compiled custom op backend is unavailable")
    image_arr = _validate_image(image)
    ksize = _validate_ksize(ksize)
    _apply_compiled_threads("median_filter_2d", image_arr)
    return module.median_filter_2d(image_arr, ksize)


@lru_cache(maxsize=2)
def _select_median_filter_backend(
    preference: str,
) -> tuple[str, Callable[[np.ndarray, int], np.ndarray]]:
    available, compiled_error = _native_backend_available(
        "median_filter_2d",
        preference,
        load_module=_load_compiled_module_result,
    )
    if available:
        return "compiled", median_filter_2d_compiled

    if compiled_error:
        _debug_log(f"compiled backend unavailable, reason: {compiled_error}")

    return "numpy", median_filter_2d_numpy


def median_filter_2d(image: np.ndarray, ksize: int) -> np.ndarray:
    backend_name, backend = _select_median_filter_backend(_fallback_preference())
    if backend_name == "compiled":
        image_arr = _validate_image(image)
        _apply_compiled_threads("median_filter_2d", image_arr)
        return backend(image_arr, ksize)
    return backend(image, ksize)
