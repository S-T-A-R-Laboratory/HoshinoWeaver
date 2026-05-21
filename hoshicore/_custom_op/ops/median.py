"""Median custom-op runtime backends."""

from __future__ import annotations

import importlib
import os
import sys
from functools import lru_cache
from typing import Any, Callable

import numpy as np

from hoshicore._custom_op import thread_tuning as _thread_tuning


def _debug_enabled() -> bool:
    return os.environ.get("HNW_CUSTOM_OPS_DEBUG", "0") not in {"", "0", "false", "False"}


def _debug_log(message: str) -> None:
    if _debug_enabled():
        print(f"[hoshicore._custom_op.median] {message}", file=sys.stderr)


def _fallback_preference() -> str:
    raw = os.environ.get("HNW_CUSTOM_OPS_FALLBACK", "auto").strip().lower()
    if raw in {"auto", "numpy"}:
        return raw
    return "auto"


@lru_cache(maxsize=1)
def _compiled_build_info() -> dict[str, Any]:
    module, _ = _load_compiled_module_result()
    if module is None or not hasattr(module, "build_info"):
        return {}
    payload = module.build_info()
    return payload if isinstance(payload, dict) else {}


_LAST_APPLIED_COMPILED_THREADS: int | None = None


@lru_cache(maxsize=1)
def _load_compiled_module_result() -> tuple[Any | None, str | None]:
    try:
        return importlib.import_module("hoshicore._custom_op._C"), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


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


def _apply_compiled_threads(op_name: str, sample: np.ndarray) -> None:
    global _LAST_APPLIED_COMPILED_THREADS
    module, _ = _load_compiled_module_result()
    if module is None:
        return
    build = _compiled_build_info()
    if not build.get("openmp"):
        return
    if not hasattr(module, "set_openmp_threads"):
        return
    threads = _thread_tuning.resolve_runtime_threads(
        op_name=op_name,
        shape=sample.shape,
        dtype=sample.dtype,
        build_info=build,
    )
    if threads == _LAST_APPLIED_COMPILED_THREADS:
        return
    if module.set_openmp_threads(int(threads)):
        _LAST_APPLIED_COMPILED_THREADS = int(threads)


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
    module, compiled_error = _load_compiled_module_result()
    if module is not None and hasattr(module, "median_reduce_chunk"):
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
