"""Sigma-clip iterative chunk custom-op runtime backends."""

from __future__ import annotations

from functools import partial

import numpy as np

from hoshicore._custom_op._dispatch import debug_log
from hoshicore._custom_op._dispatch import load_compiled_module as _load_compiled_module_result
from hoshicore._custom_op.backend_registry import native_backend_available as _native_backend_available


_debug_log = partial(debug_log, "sigma_clip")


_SUPPORTED_DTYPES = (np.uint8, np.uint16)


def _compiled_backend_available(logical_op: str) -> tuple[bool, str | None]:
    return _native_backend_available(
        logical_op,
        "auto",
        load_module=_load_compiled_module_result,
    )


def _validate_inputs(
    stack: np.ndarray,
    total_sum: np.ndarray,
    total_sq: np.ndarray,
    total_n: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    if stack.ndim != 2:
        raise ValueError(
            "sigma_clip_iterative_chunk: stack must be 2D (n_frames, plane_size)")
    if stack.shape[0] <= 0:
        raise ValueError(
            "sigma_clip_iterative_chunk: n_frames must be > 0")
    if stack.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            "sigma_clip_iterative_chunk: unsupported stack dtype; "
            "expected uint8/uint16")
    plane_size = stack.shape[1]
    if total_sum.size != plane_size or total_sq.size != plane_size or total_n.size != plane_size:
        raise ValueError(
            "sigma_clip_iterative_chunk: total stats size must match plane_size")

    if not stack.flags.c_contiguous:
        stack = np.ascontiguousarray(stack)
    total_sum = np.ascontiguousarray(total_sum, dtype=np.float64)
    total_sq = np.ascontiguousarray(total_sq, dtype=np.float64)
    total_n = np.ascontiguousarray(total_n, dtype=np.float64)

    if mask is not None:
        mask = np.ascontiguousarray(mask, dtype=np.uint8)
        if mask.ndim != 2 or mask.shape[0] != stack.shape[0] or mask.shape[1] != plane_size:
            raise ValueError(
                "sigma_clip_iterative_chunk: mask must have shape (n_frames, plane_size)")

    return stack, total_sum, total_sq, total_n, mask


def sigma_clip_iterative_chunk_numpy(
    stack: np.ndarray,
    total_sum: np.ndarray,
    total_sq: np.ndarray,
    total_n: np.ndarray,
    rej_high: float = 3.0,
    rej_low: float = 3.0,
    max_iter: int = 5,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numpy fallback: per-pixel iterative sigma clip on a chunk."""
    stack, total_sum, total_sq, total_n, mask = _validate_inputs(
        stack, total_sum, total_sq, total_n, mask)

    n_frames, plane_size = stack.shape
    stack_f64 = stack.astype(np.float64)

    cur_sum = total_sum.copy()
    cur_sq = total_sq.copy()
    cur_n = total_n.copy()
    converged = np.zeros(plane_size, dtype=np.bool_)

    for _ in range(max_iter):
        # Compute thresholds
        safe_n = np.where(cur_n > 1, cur_n, 2.0)
        mu = cur_sum / safe_n
        var = (cur_sq - cur_sum * cur_sum / safe_n) / (safe_n - 1.0)
        std = np.sqrt(np.maximum(var, 0.0))
        high = np.floor(mu + std * rej_high)
        low = np.ceil(mu - std * rej_low)

        # Scan all frames
        rej_sum = np.zeros(plane_size)
        rej_sq = np.zeros(plane_size)
        rej_n = np.zeros(plane_size)
        for f in range(n_frames):
            vals = stack_f64[f]
            valid = ~converged
            if mask is not None:
                valid = valid & (mask[f].astype(np.bool_))
            rejected = valid & ((vals < low) | (vals > high))
            rej_sum += vals * rejected
            rej_sq += (vals * vals) * rejected
            rej_n += rejected

        # Update accepted
        new_n = total_n - rej_n
        new_sum = total_sum - rej_sum
        new_sq = total_sq - rej_sq

        # Convergence: n, sum, and sq all unchanged
        unchanged = (
            (new_n == cur_n) & (new_sum == cur_sum) & (new_sq == cur_sq)
        ) | converged
        converged |= unchanged & (~converged)

        # Handle all-rejected pixels: restore total stats
        all_rejected = (new_n <= 0) & (~converged)
        cur_sum[all_rejected] = total_sum[all_rejected]
        cur_sq[all_rejected] = total_sq[all_rejected]
        cur_n[all_rejected] = total_n[all_rejected]
        converged |= all_rejected

        # Update non-converged
        active = ~converged
        cur_sum[active] = new_sum[active]
        cur_sq[active] = new_sq[active]
        cur_n[active] = new_n[active]

        if converged.all():
            break

    return cur_sum, cur_sq, cur_n


def sigma_clip_iterative_chunk_compiled(
    stack: np.ndarray,
    total_sum: np.ndarray,
    total_sq: np.ndarray,
    total_n: np.ndarray,
    rej_high: float = 3.0,
    rej_low: float = 3.0,
    max_iter: int = 5,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compiled backend: delegates to C++ kernel."""
    module, _ = _load_compiled_module_result()
    if module is None or not hasattr(module, "sigma_clip_iterative_chunk"):
        raise RuntimeError("compiled custom op backend is unavailable")
    stack, total_sum, total_sq, total_n, mask = _validate_inputs(
        stack, total_sum, total_sq, total_n, mask)
    return module.sigma_clip_iterative_chunk(
        stack, total_sum, total_sq, total_n,
        rej_high, rej_low, max_iter, mask)


def sigma_clip_iterative_chunk(
    stack: np.ndarray,
    total_sum: np.ndarray,
    total_sq: np.ndarray,
    total_n: np.ndarray,
    rej_high: float = 3.0,
    rej_low: float = 3.0,
    max_iter: int = 5,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Iterative sigma clip on a 2D chunk stack.

    Args:
        stack: (n_frames, plane_size) uint8/uint16, C-contiguous
        total_sum: (plane_size,) float64 — total FGP sum_mu for this chunk
        total_sq: (plane_size,) float64 — total FGP square_sum
        total_n: (plane_size,) float64 — total FGP n
        rej_high: rejection threshold (sigma units, upper)
        rej_low: rejection threshold (sigma units, lower)
        max_iter: maximum iterations
        mask: optional (n_frames, plane_size) uint8, 1=valid 0=excluded

    Returns:
        (accepted_sum, accepted_sq, accepted_n) as float64 arrays
    """
    available, compiled_error = _compiled_backend_available("sigma_clip_iterative_chunk")
    if available:
        return sigma_clip_iterative_chunk_compiled(
            stack, total_sum, total_sq, total_n,
            rej_high, rej_low, max_iter, mask)
    if compiled_error:
        _debug_log(f"compiled backend unavailable, reason: {compiled_error}")
    return sigma_clip_iterative_chunk_numpy(
        stack, total_sum, total_sq, total_n,
        rej_high, rej_low, max_iter, mask)


# --- Fused variant: mean + iterative clip in one call ---


def sigma_clip_fused_chunk_numpy(
    stack: np.ndarray,
    rej_high: float = 3.0,
    rej_low: float = 3.0,
    max_iter: int = 5,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numpy fallback: compute mean then iterative clip."""
    if stack.ndim != 2:
        raise ValueError(
            "sigma_clip_fused_chunk: stack must be 2D (n_frames, plane_size)")
    if stack.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            "sigma_clip_fused_chunk: unsupported stack dtype; expected uint8/uint16")
    if not stack.flags.c_contiguous:
        stack = np.ascontiguousarray(stack)

    n_frames, plane_size = stack.shape
    stack_f64 = stack.astype(np.float64)

    # Compute masked totals
    if mask is not None:
        mask = np.ascontiguousarray(mask, dtype=np.uint8)
        if mask.ndim != 2 or mask.shape[0] != n_frames or mask.shape[1] != plane_size:
            raise ValueError(
                "sigma_clip_fused_chunk: mask must have shape (n_frames, plane_size)")
        mask_f64 = mask.astype(np.float64)
        total_sum = (stack_f64 * mask_f64).sum(axis=0)
        total_sq = (stack_f64 ** 2 * mask_f64).sum(axis=0)
        total_n = mask_f64.sum(axis=0)
    else:
        total_sum = stack_f64.sum(axis=0)
        total_sq = (stack_f64 ** 2).sum(axis=0)
        total_n = np.full(plane_size, float(n_frames))

    return sigma_clip_iterative_chunk_numpy(
        stack, total_sum, total_sq, total_n, rej_high, rej_low, max_iter, mask)


def sigma_clip_fused_chunk_compiled(
    stack: np.ndarray,
    rej_high: float = 3.0,
    rej_low: float = 3.0,
    max_iter: int = 5,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compiled backend: delegates to C++ fused kernel."""
    module, _ = _load_compiled_module_result()
    if module is None or not hasattr(module, "sigma_clip_fused_chunk"):
        raise RuntimeError("compiled custom op backend is unavailable")
    if stack.ndim != 2:
        raise ValueError(
            "sigma_clip_fused_chunk: stack must be 2D (n_frames, plane_size)")
    if stack.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            "sigma_clip_fused_chunk: unsupported stack dtype; expected uint8/uint16")
    if not stack.flags.c_contiguous:
        stack = np.ascontiguousarray(stack)
    if mask is not None:
        mask = np.ascontiguousarray(mask, dtype=np.uint8)
        if mask.ndim != 2 or mask.shape[0] != stack.shape[0] or mask.shape[1] != stack.shape[1]:
            raise ValueError(
                "sigma_clip_fused_chunk: mask must have shape (n_frames, plane_size)")
    return module.sigma_clip_fused_chunk(stack, rej_high, rej_low, max_iter, mask)


def sigma_clip_fused_chunk(
    stack: np.ndarray,
    rej_high: float = 3.0,
    rej_low: float = 3.0,
    max_iter: int = 5,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fused mean + iterative sigma clip on a 2D chunk stack.

    Args:
        stack: (n_frames, plane_size) uint8/uint16, C-contiguous
        rej_high: rejection threshold (sigma units, upper)
        rej_low: rejection threshold (sigma units, lower)
        max_iter: maximum iterations
        mask: optional (n_frames, plane_size) uint8, 1=valid 0=excluded

    Returns:
        (accepted_sum, accepted_sq, accepted_n) as float64 arrays
    """
    available, compiled_error = _compiled_backend_available("sigma_clip_fused_chunk")
    if available:
        return sigma_clip_fused_chunk_compiled(
            stack, rej_high, rej_low, max_iter, mask)
    if compiled_error:
        _debug_log(f"compiled backend unavailable, reason: {compiled_error}")
    return sigma_clip_fused_chunk_numpy(
        stack, rej_high, rej_low, max_iter, mask)
