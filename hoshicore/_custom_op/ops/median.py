"""Median custom-op runtime backends."""

from __future__ import annotations

from functools import lru_cache
from functools import partial
from typing import Any, Callable

import numpy as np

from hoshicore._custom_op._dispatch import apply_compiled_threads as _apply_compiled_threads
from hoshicore._custom_op._dispatch import compiled_build_info as _compiled_build_info
from hoshicore._custom_op._dispatch import debug_enabled as _debug_enabled
from hoshicore._custom_op._dispatch import debug_log
from hoshicore._custom_op._dispatch import fallback_preference as _fallback_preference
from hoshicore._custom_op._dispatch import load_compiled_module as _load_compiled_module_result
from hoshicore._custom_op.backend_registry import native_backend_available as _native_backend_available


_debug_log = partial(debug_log, "median")


_SUPPORTED_DTYPES = (np.uint8, np.uint16, np.float32, np.float64)


def _validate_stack(stack: np.ndarray) -> np.ndarray:
    stack_arr = np.asarray(stack)
    if stack_arr.ndim not in {3, 4}:
        raise ValueError(
            "median_reduce_chunk: stack must have shape (N, H, W) or (N, H, W, C)"
        )
    if stack_arr.shape[0] <= 0:
        raise ValueError("median_reduce_chunk: frame axis must be non-empty")
    if stack_arr.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            "median_reduce_chunk: unsupported dtype; "
            "expected uint8/uint16/float32/float64"
        )
    if not stack_arr.flags.c_contiguous:
        stack_arr = np.ascontiguousarray(stack_arr)
    return stack_arr


def median_reduce_chunk_numpy(stack: np.ndarray) -> np.ndarray:
    stack_arr = _validate_stack(stack)
    result = np.median(stack_arr, axis=0)
    # np.median always returns float64; cast back to match compiled backend behavior
    if result.dtype != stack_arr.dtype:
        result = result.astype(stack_arr.dtype)
    return result


def median_reduce_chunk_compiled(stack: np.ndarray) -> np.ndarray:
    module, _ = _load_compiled_module_result()
    if module is None or not hasattr(module, "median_reduce_chunk"):
        raise RuntimeError("compiled custom op backend is unavailable")
    stack_arr = _validate_stack(stack)
    _apply_compiled_threads("median_reduce_chunk", stack_arr)
    return module.median_reduce_chunk(stack_arr)


@lru_cache(maxsize=2)
def _select_median_backend(
    preference: str,
) -> tuple[str, Callable[[np.ndarray], np.ndarray]]:
    available, compiled_error = _native_backend_available(
        "median_reduce_chunk",
        preference,
        load_module=_load_compiled_module_result,
    )
    if available:
        return "compiled", median_reduce_chunk_compiled

    if compiled_error:
        _debug_log(f"compiled backend unavailable, reason: {compiled_error}")

    return "numpy", median_reduce_chunk_numpy


def median_reduce_chunk(stack: np.ndarray) -> np.ndarray:
    backend_name, backend = _select_median_backend(_fallback_preference())
    if backend_name == "compiled":
        stack_arr = _validate_stack(stack)
        _apply_compiled_threads("median_reduce_chunk", stack_arr)
        return backend(stack_arr)
    return backend(stack)
