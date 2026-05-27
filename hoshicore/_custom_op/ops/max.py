"""Max custom-op runtime backends."""

from __future__ import annotations

from functools import lru_cache
from functools import partial
from typing import Any, Callable

import numpy as np

from hoshicore._custom_op import thread_tuning as _thread_tuning
from hoshicore._custom_op._dispatch import apply_compiled_threads as _apply_compiled_threads
from hoshicore._custom_op._dispatch import compiled_build_info as _compiled_build_info
from hoshicore._custom_op._dispatch import debug_enabled as _debug_enabled
from hoshicore._custom_op._dispatch import debug_log
from hoshicore._custom_op._dispatch import fallback_preference as _fallback_preference
from hoshicore._custom_op._dispatch import load_compiled_module as _load_compiled_module_result


_debug_log = partial(debug_log, "max")


def _validate_pair(base: np.ndarray, fresh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    base_arr = np.asarray(base)
    fresh_arr = np.asarray(fresh)
    if base_arr.ndim != fresh_arr.ndim:
        raise ValueError("max_combine: ndim mismatch")
    if base_arr.shape != fresh_arr.shape:
        raise ValueError("max_combine: shape mismatch")
    if base_arr.dtype != fresh_arr.dtype:
        raise ValueError("max_combine: dtype mismatch")
    if not base_arr.flags.c_contiguous:
        raise ValueError("max_combine: base must be C-contiguous")
    if not base_arr.flags.writeable:
        raise ValueError("max_combine: base must be writeable")
    if not fresh_arr.flags.c_contiguous:
        fresh_arr = np.ascontiguousarray(fresh_arr)
    return base_arr, fresh_arr


def _validate_threshold_inputs(
    frame: np.ndarray,
    mean_img: np.ndarray,
    std_img: np.ndarray,
    result: np.ndarray,
    *,
    op_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame_arr = np.asarray(frame)
    mean_arr = np.asarray(mean_img)
    std_arr = np.asarray(std_img)
    result_arr = np.asarray(result)
    if frame_arr.shape != mean_arr.shape or frame_arr.shape != std_arr.shape or frame_arr.shape != result_arr.shape:
        raise ValueError(f"{op_name}: shape mismatch")
    if frame_arr.dtype != mean_arr.dtype or frame_arr.dtype != std_arr.dtype or frame_arr.dtype != result_arr.dtype:
        raise ValueError(f"{op_name}: dtype mismatch")
    if not result_arr.flags.c_contiguous:
        raise ValueError(f"{op_name}: result must be C-contiguous")
    if not result_arr.flags.writeable:
        raise ValueError(f"{op_name}: result must be writeable")
    if not frame_arr.flags.c_contiguous:
        frame_arr = np.ascontiguousarray(frame_arr)
    if not mean_arr.flags.c_contiguous:
        mean_arr = np.ascontiguousarray(mean_arr)
    if not std_arr.flags.c_contiguous:
        std_arr = np.ascontiguousarray(std_arr)
    return frame_arr, mean_arr, std_arr, result_arr


def _validate_scalar_weight(weight: Any, *, op_name: str) -> float | None:
    if weight is None:
        return None
    if isinstance(weight, np.ndarray):
        if weight.ndim == 0:
            return float(weight.item())
        return None
    if np.isscalar(weight):
        return float(weight)
    raise TypeError(f"{op_name}: unsupported weight type")


def max_combine_numpy(base: np.ndarray, fresh: np.ndarray) -> np.ndarray:
    base_arr, fresh_arr = _validate_pair(base, fresh)
    np.maximum(base_arr, fresh_arr, out=base_arr)
    return base_arr


def max_combine_compiled(base: np.ndarray, fresh: np.ndarray) -> np.ndarray:
    module, _ = _load_compiled_module_result()
    if module is None:
        raise RuntimeError("compiled custom op backend is unavailable")
    base_arr, fresh_arr = _validate_pair(base, fresh)
    _apply_compiled_threads("max_combine", base_arr)
    return module.max_combine(base_arr, fresh_arr)


def threshold_max_merge_numpy(
    frame: np.ndarray,
    mean_img: np.ndarray,
    std_img: np.ndarray,
    result: np.ndarray,
    n_sigma: float,
    weight: Any = None,
) -> np.ndarray:
    frame_arr, mean_arr, std_arr, result_arr = _validate_threshold_inputs(
        frame,
        mean_img,
        std_img,
        result,
        op_name="threshold_max_merge",
    )
    scalar_weight = _validate_scalar_weight(weight, op_name="threshold_max_merge")
    threshold = mean_arr + float(n_sigma) * std_arr
    mask = frame_arr > threshold
    if weight is None:
        candidate = np.where(mask, frame_arr, mean_arr)
    elif scalar_weight is not None:
        candidate = np.where(mask, frame_arr * scalar_weight, mean_arr)
    else:
        candidate = np.where(mask, frame_arr * np.asarray(weight), mean_arr)
    np.maximum(result_arr, candidate, out=result_arr)
    return result_arr


def threshold_max_merge_compiled(
    frame: np.ndarray,
    mean_img: np.ndarray,
    std_img: np.ndarray,
    result: np.ndarray,
    n_sigma: float,
    weight: Any = None,
) -> np.ndarray:
    module, _ = _load_compiled_module_result()
    if module is None:
        raise RuntimeError("compiled custom op backend is unavailable")
    scalar_weight = _validate_scalar_weight(weight, op_name="threshold_max_merge")
    if weight is not None and scalar_weight is None:
        raise ValueError("threshold_max_merge: compiled backend only supports scalar weight")
    frame_arr, mean_arr, std_arr, result_arr = _validate_threshold_inputs(
        frame,
        mean_img,
        std_img,
        result,
        op_name="threshold_max_merge",
    )
    _apply_compiled_threads("threshold_max_merge", frame_arr)
    return module.threshold_max_merge(
        result_arr,
        frame_arr,
        mean_arr,
        std_arr,
        float(n_sigma),
        scalar_weight,
    )


@lru_cache(maxsize=2)
def _select_max_backend(preference: str) -> tuple[str, Callable[[np.ndarray, np.ndarray], np.ndarray]]:
    module, compiled_error = _load_compiled_module_result()
    if module is not None:
        return "compiled", max_combine_compiled

    if compiled_error:
        _debug_log(f"compiled backend unavailable, reason: {compiled_error}")

    return "numpy", max_combine_numpy


@lru_cache(maxsize=2)
def _select_threshold_max_backend(
    preference: str,
) -> tuple[str, Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, Any], np.ndarray]]:
    module, compiled_error = _load_compiled_module_result()
    if module is not None and hasattr(module, "threshold_max_merge"):
        return "compiled", threshold_max_merge_compiled

    if compiled_error:
        _debug_log(f"compiled backend unavailable, reason: {compiled_error}")

    return "numpy", threshold_max_merge_numpy


def custom_ops_available() -> bool:
    module, _ = _load_compiled_module_result()
    return module is not None


def build_info() -> dict[str, Any]:
    module, compiled_error = _load_compiled_module_result()
    if module is None:
        backend_name, _ = _select_max_backend(_fallback_preference())
        payload = {
            "available": False,
            "module": None,
            "backend": backend_name,
            "thread_policy": _thread_tuning.thread_policy_value(),
        }
        if _debug_enabled() and compiled_error is not None:
            payload["compiled_error"] = compiled_error
        return payload

    payload = {
        "available": True,
        "module": module.__name__,
        "backend": "compiled",
        "thread_policy": _thread_tuning.thread_policy_value(),
    }
    if hasattr(module, "build_info"):
        payload.update(module.build_info())
    return payload


def max_combine(base: np.ndarray, fresh: np.ndarray) -> np.ndarray:
    backend_name, backend = _select_max_backend(_fallback_preference())
    if backend_name == "compiled":
        base_arr, _ = _validate_pair(base, fresh)
        _apply_compiled_threads("max_combine", base_arr)
    return backend(base, fresh)


def threshold_max_merge(
    frame: np.ndarray,
    mean_img: np.ndarray,
    std_img: np.ndarray,
    result: np.ndarray,
    n_sigma: float,
    weight: Any = None,
) -> np.ndarray:
    scalar_weight = _validate_scalar_weight(weight, op_name="threshold_max_merge")
    if weight is not None and scalar_weight is None:
        return threshold_max_merge_numpy(
            frame,
            mean_img,
            std_img,
            result,
            n_sigma,
            weight,
        )
    backend_name, backend = _select_threshold_max_backend(_fallback_preference())
    if backend_name == "compiled":
        frame_arr, mean_arr, std_arr, result_arr = _validate_threshold_inputs(
            frame,
            mean_img,
            std_img,
            result,
            op_name="threshold_max_merge",
        )
        _apply_compiled_threads("threshold_max_merge", frame_arr)
        return backend(frame_arr, mean_arr, std_arr, result_arr, n_sigma, scalar_weight)
    return backend(frame, mean_img, std_img, result, n_sigma, scalar_weight)

