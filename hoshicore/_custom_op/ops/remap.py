"""Camera-model remap custom-op runtime backends."""

from __future__ import annotations

from functools import lru_cache
from functools import partial
from typing import Callable

import cv2
import numpy as np

from hoshicore._custom_op._dispatch import debug_enabled as _debug_enabled
from hoshicore._custom_op._dispatch import debug_log
from hoshicore._custom_op._dispatch import fallback_preference as _fallback_preference
from hoshicore._custom_op._dispatch import load_compiled_module as _load_compiled_module_result
from hoshicore._custom_op.backend_registry import native_backend_available as _native_backend_available


_debug_log = partial(debug_log, "remap")


def _validate_rotation(rotation_dst_to_src: np.ndarray) -> np.ndarray:
    rotation_arr = np.asarray(rotation_dst_to_src, dtype=np.float32)
    if rotation_arr.shape != (3, 3):
        raise ValueError(
            "camera_model_remap: rotation_dst_to_src must have shape (3, 3)")
    if not rotation_arr.flags.c_contiguous:
        rotation_arr = np.ascontiguousarray(rotation_arr)
    return rotation_arr


def _validate_image(image: np.ndarray) -> np.ndarray:
    image_arr = np.asarray(image)
    if image_arr.ndim not in {2, 3}:
        raise ValueError("camera_model_remap: image must have shape (H, W) or (H, W, C)")
    if not image_arr.flags.c_contiguous:
        image_arr = np.ascontiguousarray(image_arr)
    return image_arr


def _is_cuda_runtime_unavailable_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return (
        "no cuda-capable device is detected" in message
        or "cuda driver version is insufficient" in message
        or "cuda initialization error" in message
        or "cudaunknown" in message
    )


def camera_model_remap_numpy(
    *,
    image: np.ndarray,
    out_height: int,
    out_width: int,
    fx_src: float,
    fy_src: float,
    cx_src: float,
    cy_src: float,
    fx_dst: float,
    fy_dst: float,
    cx_dst: float,
    cy_dst: float,
    rotation_dst_to_src: np.ndarray,
) -> np.ndarray:
    image_arr = _validate_image(image)
    if out_height <= 0 or out_width <= 0:
        raise ValueError("camera_model_remap: output height and width must be positive")
    rotation_arr = _validate_rotation(rotation_dst_to_src)
    xs = np.arange(out_width, dtype=np.float32)
    ys = np.arange(out_height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    x = (grid_x - np.float32(cx_dst)) / np.float32(fx_dst)
    y = (grid_y - np.float32(cy_dst)) / np.float32(fy_dst)
    proj_x = rotation_arr[0, 0] * x + rotation_arr[0, 1] * y + rotation_arr[0, 2]
    proj_y = rotation_arr[1, 0] * x + rotation_arr[1, 1] * y + rotation_arr[1, 2]
    proj_z = rotation_arr[2, 0] * x + rotation_arr[2, 1] * y + rotation_arr[2, 2]

    map_x = np.full((out_height, out_width), np.nan, dtype=np.float32)
    map_y = np.full((out_height, out_width), np.nan, dtype=np.float32)
    valid = proj_z > 0.0
    if np.any(valid):
        inv_z = (1.0 / proj_z[valid]).astype(np.float32, copy=False)
        map_x[valid] = np.float32(fx_src) * proj_x[valid] * inv_z + np.float32(cx_src)
        map_y[valid] = np.float32(fy_src) * proj_y[valid] * inv_z + np.float32(cy_src)
    return cv2.remap(
        image_arr,
        map_x.astype(np.float32, copy=False),
        map_y.astype(np.float32, copy=False),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0 if image_arr.ndim == 2 else (0, 0, 0),
    )


def camera_model_remap_compiled(
    *,
    image: np.ndarray,
    out_height: int,
    out_width: int,
    fx_src: float,
    fy_src: float,
    cx_src: float,
    cy_src: float,
    fx_dst: float,
    fy_dst: float,
    cx_dst: float,
    cy_dst: float,
    rotation_dst_to_src: np.ndarray,
) -> np.ndarray:
    module, _ = _load_compiled_module_result()
    if module is None or not hasattr(module, "camera_model_remap"):
        raise RuntimeError("compiled custom op backend is unavailable")
    image_arr = _validate_image(image)
    rotation_arr = _validate_rotation(rotation_dst_to_src)
    return module.camera_model_remap(
        image_arr,
        int(out_height),
        int(out_width),
        float(fx_src),
        float(fy_src),
        float(cx_src),
        float(cy_src),
        float(fx_dst),
        float(fy_dst),
        float(cx_dst),
        float(cy_dst),
        rotation_arr,
    )


@lru_cache(maxsize=2)
def _select_camera_model_remap_backend(
    preference: str,
) -> tuple[str, Callable[..., np.ndarray]]:
    available, compiled_error = _native_backend_available(
        "camera_model_remap",
        preference,
        load_module=_load_compiled_module_result,
    )
    if available:
        return "compiled", camera_model_remap_compiled

    if compiled_error:
        _debug_log(f"compiled backend unavailable, reason: {compiled_error}")

    return "numpy", camera_model_remap_numpy


def camera_model_remap(
    *,
    image: np.ndarray,
    out_height: int,
    out_width: int,
    fx_src: float,
    fy_src: float,
    cx_src: float,
    cy_src: float,
    fx_dst: float,
    fy_dst: float,
    cx_dst: float,
    cy_dst: float,
    rotation_dst_to_src: np.ndarray,
) -> np.ndarray:
    backend_name, backend = _select_camera_model_remap_backend(
        _fallback_preference())
    kwargs = {
        "image": image,
        "out_height": out_height,
        "out_width": out_width,
        "fx_src": fx_src,
        "fy_src": fy_src,
        "cx_src": cx_src,
        "cy_src": cy_src,
        "fx_dst": fx_dst,
        "fy_dst": fy_dst,
        "cx_dst": cx_dst,
        "cy_dst": cy_dst,
        "rotation_dst_to_src": rotation_dst_to_src,
    }
    if backend_name != "compiled":
        return backend(**kwargs)

    try:
        return backend(**kwargs)
    except RuntimeError as exc:
        if not _is_cuda_runtime_unavailable_error(exc):
            raise
        _debug_log(
            f"compiled CUDA backend unavailable at runtime, falling back to numpy: {exc}"
        )
        return camera_model_remap_numpy(**kwargs)
