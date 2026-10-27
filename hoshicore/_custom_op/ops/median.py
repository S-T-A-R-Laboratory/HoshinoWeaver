"""Median custom-op runtime backends."""

from __future__ import annotations

from functools import partial
from typing import Callable, Sequence

import numpy as np

from hoshicore._custom_op._dispatch import apply_compiled_threads as _apply_compiled_threads
from hoshicore._custom_op._dispatch import debug_log
from hoshicore._custom_op._dispatch import fallback_preference as _fallback_preference
from hoshicore._custom_op._dispatch import load_compiled_module as _load_compiled_module_result
from hoshicore._custom_op._dispatch import load_metal_module as _load_metal_module_result
from hoshicore._custom_op.backend_registry import BackendSelection
from hoshicore._custom_op.backend_registry import resolve_backend as _resolve_backend
from hoshicore._custom_op.backend_registry import run_with_accelerator_fallback
from hoshicore._custom_op.cuda_memory import estimate_median_reduce_chunk
from hoshicore._custom_op.cuda_memory import run_admitted_cuda as _run_admitted_cuda
from hoshicore._custom_op.metal_memory import estimate_median_reduce_chunk as _estimate_metal_median
from hoshicore._custom_op.metal_memory import run_admitted_metal as _run_admitted_metal


_debug_log = partial(debug_log, "median")


_SUPPORTED_DTYPES = (np.uint8, np.uint16, np.float32, np.float64)
_METAL_MAX_STACK_ELEMENTS = (1 << 32) - 1


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


def median_reduce_chunk_numpy(
    stack: np.ndarray, skip_zero_rgb: bool = False
) -> np.ndarray:
    stack_arr = _validate_stack(stack)
    if skip_zero_rgb and stack_arr.ndim == 4 and stack_arr.shape[-1] >= 3:
        # Process one row at a time to bound the sort and validity workspace.
        result = np.zeros(stack_arr.shape[1:], dtype=stack_arr.dtype)
        floating = np.issubdtype(stack_arr.dtype, np.floating)
        sentinel = np.inf if floating else np.iinfo(stack_arr.dtype).max
        for row in range(stack_arr.shape[1]):
            samples = stack_arr[:, row].copy()
            valid = np.any(samples[..., :3] != 0, axis=-1)
            count = valid.sum(axis=0)
            samples[~valid] = sentinel
            # Invalid alpha NaNs were replaced above; valid NaNs propagate.
            has_nan = np.any(np.isnan(samples), axis=0) if floating else None
            samples.sort(axis=0)
            high_index = (count // 2)[None, :, None]
            low_index = (np.maximum(count - 1, 0) // 2)[None, :, None]
            high = np.take_along_axis(samples, high_index, axis=0)[0]
            low = np.take_along_axis(samples, low_index, axis=0)[0]
            # np.mean preserves float32 arithmetic and widens integer addition.
            values = np.mean(high[None], axis=0)
            even = (count > 0) & (count % 2 == 0)
            values[even] = np.mean(np.stack((low[even], high[even])), axis=0)
            if floating:
                values[has_nan] = np.nan
            values[count == 0] = 0
            result[row] = values
        return result
    result = np.median(stack_arr, axis=0)
    # Integer medians are promoted; restore the input dtype.
    if result.dtype != stack_arr.dtype:
        result = result.astype(stack_arr.dtype)
    return result


def median_reduce_chunk_compiled(
    stack: np.ndarray, skip_zero_rgb: bool = False
) -> np.ndarray:
    module, _ = _load_compiled_module_result()
    if module is None or not hasattr(module, "median_reduce_chunk"):
        raise RuntimeError("compiled custom op backend is unavailable")
    stack_arr = _validate_stack(stack)
    _apply_compiled_threads("median_reduce_chunk", stack_arr)
    if skip_zero_rgb:
        return module.median_reduce_chunk(stack_arr, skip_zero_rgb=True)
    return module.median_reduce_chunk(stack_arr)


def median_reduce_chunk_cuda(
    stack: np.ndarray, skip_zero_rgb: bool = False
) -> np.ndarray:
    stack_arr = _validate_stack(stack)
    if stack_arr.dtype not in (np.uint8, np.uint16) or stack_arr.shape[0] > 128:
        raise ValueError("median_reduce_chunk_cuda: expected uint8/uint16 and at most 128 frames")
    module, _ = _load_compiled_module_result()
    if module is None or not hasattr(module, "median_reduce_chunk_cuda"):
        raise RuntimeError("CUDA median backend is unavailable")
    estimate = estimate_median_reduce_chunk(
        n_frames=stack_arr.shape[0],
        plane_size=stack_arr.size // stack_arr.shape[0],
        dtype_bytes=stack_arr.dtype.itemsize,
    )
    return _run_admitted_cuda(
        estimate, module.median_reduce_chunk_cuda, stack_arr, skip_zero_rgb
    )


def median_reduce_chunk_metal(
    stack: np.ndarray, skip_zero_rgb: bool = False
) -> np.ndarray:
    stack_arr = _validate_stack(stack)
    if stack_arr.dtype not in (np.uint8, np.uint16) or stack_arr.shape[0] > 128:
        raise ValueError("median_reduce_chunk_metal: expected uint8/uint16 and at most 128 frames")
    module, error = _load_metal_module_result()
    if module is None or not hasattr(module, "median_reduce_chunk_metal"):
        raise RuntimeError(error or "Metal median backend is unavailable")
    estimate = _estimate_metal_median(
        n_frames=stack_arr.shape[0],
        plane_size=stack_arr.size // stack_arr.shape[0],
        dtype_bytes=stack_arr.dtype.itemsize,
    )
    return _run_admitted_metal(
        estimate, module.median_reduce_chunk_metal, stack_arr, skip_zero_rgb
    )


def _validate_frames(frames: Sequence[np.ndarray]) -> tuple[np.ndarray, ...]:
    if not frames:
        raise ValueError("median_reduce_frames: frame sequence must be non-empty")
    first = np.asarray(frames[0])
    if first.ndim not in (2, 3) or first.dtype not in _SUPPORTED_DTYPES:
        raise ValueError("median_reduce_frames: expected 2D/3D frames with supported dtype")
    result = []
    for frame in frames:
        array = np.asarray(frame)
        if array.shape != first.shape:
            raise ValueError("median_reduce_frames: frame shapes must match")
        result.append(np.ascontiguousarray(array, dtype=first.dtype))
    return tuple(result)


def median_reduce_frames_cuda(
    frames: Sequence[np.ndarray], skip_zero_rgb: bool = False
) -> np.ndarray:
    prepared = _validate_frames(frames)
    first = prepared[0]
    if first.dtype not in (np.uint8, np.uint16) or len(prepared) > 128:
        raise ValueError("median_reduce_frames_cuda: expected uint8/uint16 and at most 128 frames")
    module, _ = _load_compiled_module_result()
    if module is None or not hasattr(module, "median_reduce_chunk_cuda_frames"):
        raise RuntimeError("CUDA median frames backend is unavailable")
    estimate = estimate_median_reduce_chunk(
        n_frames=len(prepared), plane_size=first.size, dtype_bytes=first.dtype.itemsize
    )
    return _run_admitted_cuda(
        estimate, module.median_reduce_chunk_cuda_frames, prepared, skip_zero_rgb
    )


def median_reduce_frames_metal(
    frames: Sequence[np.ndarray], skip_zero_rgb: bool = False
) -> np.ndarray:
    prepared = _validate_frames(frames)
    first = prepared[0]
    if first.dtype not in (np.uint8, np.uint16) or len(prepared) > 128:
        raise ValueError("median_reduce_frames_metal: expected uint8/uint16 and at most 128 frames")
    module, error = _load_metal_module_result()
    if module is None or not hasattr(module, "median_reduce_chunk_metal_frames"):
        raise RuntimeError(error or "Metal median frames backend is unavailable")
    estimate = _estimate_metal_median(
        n_frames=len(prepared), plane_size=first.size, dtype_bytes=first.dtype.itemsize
    )
    return _run_admitted_metal(
        estimate, module.median_reduce_chunk_metal_frames, prepared, skip_zero_rgb
    )


def _resolve_median_selection(
    preference: str, n_frames: int, plane_size: int, dtype: np.dtype
) -> BackendSelection:
    use_gpu = dtype in (np.uint8, np.uint16) and 8 <= n_frames <= 128 and plane_size >= 65536
    if not use_gpu:
        excluded = ("cuda_host_io", "metal_host_io")
    elif n_frames * plane_size > _METAL_MAX_STACK_ELEMENTS:
        excluded = ("metal_host_io",)
    else:
        excluded = ()
    selection = _resolve_backend(
        "median_reduce_chunk",
        preference,
        load_module=_load_compiled_module_result,
        exclude_backends=excluded,
    )
    if selection.reason:
        _debug_log(f"compiled backend unavailable, reason: {selection.reason}")
    return selection


def _select_median_backend(
    preference: str,
    stack: np.ndarray,
) -> BackendSelection:
    return _resolve_median_selection(
        preference, stack.shape[0], stack.size // stack.shape[0], stack.dtype
    )


def _median_backend(
    selection: BackendSelection,
) -> tuple[str, Callable[[np.ndarray], np.ndarray]]:
    if not selection.native or selection.candidate is None:
        return "numpy", median_reduce_chunk_numpy
    if selection.candidate.kernel_name == "median_reduce_chunk_cuda":
        return "cuda", median_reduce_chunk_cuda
    if selection.candidate.kernel_name == "median_reduce_chunk_metal":
        return "metal", median_reduce_chunk_metal
    if selection.candidate.kernel_name == "median_reduce_chunk":
        return "cpu", median_reduce_chunk_compiled
    raise RuntimeError(f"unknown median backend candidate: {selection.candidate}")


def median_reduce_chunk(stack: np.ndarray, skip_zero_rgb: bool = False) -> np.ndarray:
    """Reduce frames, optionally excluding pixels whose first three channels are zero."""
    stack_arr = _validate_stack(stack)
    selection = _select_median_backend(_fallback_preference(), stack_arr)
    return run_with_accelerator_fallback(
        "median_reduce_chunk",
        selection,
        _median_backend,
        lambda backend: backend(stack_arr, skip_zero_rgb=skip_zero_rgb),
        load_module=_load_compiled_module_result,
        log=_debug_log,
    )


def median_reduce_frames(
    frames: Sequence[np.ndarray], skip_zero_rgb: bool = False
) -> np.ndarray:
    """Frame-list equivalent of median_reduce_chunk; grayscale ignores skip_zero_rgb."""
    prepared = _validate_frames(frames)
    selection = _resolve_median_selection(
        _fallback_preference(), len(prepared), prepared[0].size, prepared[0].dtype
    )

    def map_backend(
        selected: BackendSelection,
    ) -> tuple[str, Callable[[Sequence[np.ndarray]], np.ndarray]]:
        if not selected.native or selected.candidate is None:
            return "numpy", lambda items: median_reduce_chunk_numpy(
                np.stack(items), skip_zero_rgb=skip_zero_rgb
            )
        if selected.candidate.kernel_name == "median_reduce_chunk_cuda":
            return "cuda", partial(median_reduce_frames_cuda, skip_zero_rgb=skip_zero_rgb)
        if selected.candidate.kernel_name == "median_reduce_chunk_metal":
            return "metal", partial(median_reduce_frames_metal, skip_zero_rgb=skip_zero_rgb)
        if selected.candidate.kernel_name == "median_reduce_chunk":
            return "cpu", lambda items: median_reduce_chunk_compiled(
                np.stack(items), skip_zero_rgb=skip_zero_rgb
            )
        raise RuntimeError(f"unknown median backend candidate: {selected.candidate}")

    return run_with_accelerator_fallback(
        "median_reduce_chunk",
        selection,
        map_backend,
        lambda backend: backend(prepared),
        load_module=_load_compiled_module_result,
        log=_debug_log,
    )
