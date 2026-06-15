"""FastGaussianParam custom-op runtime backends."""

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


_debug_log = partial(debug_log, "fgp")


def _compiled_backend_available(logical_op: str, preference: str) -> tuple[bool, str | None]:
    return _native_backend_available(
        logical_op,
        preference,
        load_module=_load_compiled_module_result,
    )


def _validate_target(base: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sum_mu = np.asarray(base.sum_mu)
    square_sum = np.asarray(base.square_sum)
    count = np.asarray(base.n)
    if sum_mu.shape != square_sum.shape or sum_mu.shape != count.shape:
        raise ValueError("fgp_accumulate: base buffers shape mismatch")
    if not sum_mu.flags.c_contiguous or not square_sum.flags.c_contiguous or not count.flags.c_contiguous:
        raise ValueError("fgp_accumulate: base buffers must be C-contiguous")
    if not sum_mu.flags.writeable or not square_sum.flags.writeable or not count.flags.writeable:
        raise ValueError("fgp_accumulate: base buffers must be writeable")
    return sum_mu, square_sum, count


def _validate_peer(other: Any, shape: tuple[int, ...], *, op_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sum_mu = np.asarray(other.sum_mu)
    square_sum = np.asarray(other.square_sum)
    count = np.asarray(other.n)
    if sum_mu.shape != shape or square_sum.shape != shape or count.shape != shape:
        raise ValueError(f"{op_name}: accumulator shape mismatch")
    if not sum_mu.flags.c_contiguous:
        sum_mu = np.ascontiguousarray(sum_mu)
    if not square_sum.flags.c_contiguous:
        square_sum = np.ascontiguousarray(square_sum)
    if not count.flags.c_contiguous:
        count = np.ascontiguousarray(count)
    return sum_mu, square_sum, count


def _validate_buffers(
    sum_mu: np.ndarray,
    square_sum: np.ndarray,
    count: np.ndarray,
    *,
    op_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sum_arr = np.asarray(sum_mu)
    square_arr = np.asarray(square_sum)
    count_arr = np.asarray(count)
    if sum_arr.shape != square_arr.shape or sum_arr.shape != count_arr.shape:
        raise ValueError(f"{op_name}: accumulator shape mismatch")
    if not sum_arr.flags.c_contiguous or not square_arr.flags.c_contiguous or not count_arr.flags.c_contiguous:
        raise ValueError(f"{op_name}: accumulators must be C-contiguous")
    if not sum_arr.flags.writeable or not square_arr.flags.writeable or not count_arr.flags.writeable:
        raise ValueError(f"{op_name}: accumulators must be writeable")
    return sum_arr, square_arr, count_arr


def _validate_fresh(sum_mu: np.ndarray, fresh: np.ndarray, *, op_name: str = "fgp_accumulate") -> np.ndarray:
    fresh_arr = np.asarray(fresh)
    if fresh_arr.shape != sum_mu.shape:
        raise ValueError(f"{op_name}: shape mismatch")
    if not fresh_arr.flags.c_contiguous:
        fresh_arr = np.ascontiguousarray(fresh_arr)
    return fresh_arr


def _validate_integer_weight(weight: Any) -> int | None:
    if weight is None:
        return None
    if isinstance(weight, (int, np.integer)):
        value = int(weight)
        if value <= 0:
            raise ValueError("fgp_accumulate: weight must be positive")
        return value
    return None


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


def _python_fallback(base: Any, fresh: np.ndarray, weight: Any) -> Any:
    from hoshicore.component.data_container import FastGaussianParam

    patch = FastGaussianParam(fresh, source_dtype=fresh.dtype)
    if weight is not None:
        patch = patch * weight
    return base + patch


def _maybe_prepare_target(base: Any, weight: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sum_mu, square_sum, count = _validate_target(base)
    fresh_weight = _validate_integer_weight(weight)
    delta = 1 if fresh_weight is None else fresh_weight
    if getattr(base, "max_n", None) is not None:
        # 与原有 FastGaussianParam 路径保持一致：在真正写入前先处理计数和精度扩容。
        next_max_n = int(base.max_n) + int(delta)
        if next_max_n > base._safe_add_count():
            base.upscale()
            sum_mu, square_sum, count = _validate_target(base)
        from hoshicore.component.data_container import DTYPE_MAX_VALUE, DTYPE_UPSCALE_MAP

        if count.dtype in DTYPE_MAX_VALUE and next_max_n > DTYPE_MAX_VALUE[count.dtype]:
            if count.dtype in DTYPE_UPSCALE_MAP:
                base.n = count.astype(DTYPE_UPSCALE_MAP[count.dtype])
                sum_mu, square_sum, count = _validate_target(base)
        base.max_n = next_max_n
    return sum_mu, square_sum, count


def _maybe_prepare_target_add(base: Any, other: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sum_mu, square_sum, count = _validate_target(base)
    other_sum_mu, _, _ = _validate_peer(other, sum_mu.shape, op_name="fgp_add")
    if getattr(base, "ddof", None) != getattr(other, "ddof", None):
        raise ValueError("fgp_add: ddof mismatch")
    if getattr(base, "max_n", None) is not None:
        delta = int(getattr(other, "max_n", np.max(other.n)))
        next_max_n = int(base.max_n) + delta
        if next_max_n > base._safe_add_count():
            base.upscale()
            sum_mu, square_sum, count = _validate_target(base)
        from hoshicore.component.data_container import DTYPE_MAX_VALUE, DTYPE_UPSCALE_MAP

        if count.dtype in DTYPE_MAX_VALUE and next_max_n > DTYPE_MAX_VALUE[count.dtype]:
            if count.dtype in DTYPE_UPSCALE_MAP:
                base.n = count.astype(DTYPE_UPSCALE_MAP[count.dtype])
                sum_mu, square_sum, count = _validate_target(base)
        base.max_n = next_max_n
    _validate_peer(other, sum_mu.shape, op_name="fgp_add")
    return sum_mu, square_sum, count


def _validate_spatial_mask(fresh: np.ndarray, mask: np.ndarray, *, op_name: str) -> np.ndarray:
    mask_arr = np.asarray(mask, dtype=np.uint8)
    if fresh.ndim == mask_arr.ndim + 1:
        if fresh.shape[:-1] != mask_arr.shape:
            raise ValueError(f"{op_name}: mask shape mismatch")
    elif fresh.ndim == mask_arr.ndim:
        if fresh.shape != mask_arr.shape:
            raise ValueError(f"{op_name}: mask shape mismatch")
    else:
        raise ValueError(f"{op_name}: mask ndim mismatch")
    if not mask_arr.flags.c_contiguous:
        mask_arr = np.ascontiguousarray(mask_arr)
    return mask_arr


def _validate_rejection_images(
    fresh: np.ndarray,
    rej_high_img: np.ndarray,
    rej_low_img: np.ndarray,
    *,
    op_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    rej_high_arr = np.asarray(rej_high_img)
    rej_low_arr = np.asarray(rej_low_img)
    if rej_high_arr.shape != fresh.shape or rej_low_arr.shape != fresh.shape:
        raise ValueError(f"{op_name}: rejection image shape mismatch")
    if rej_high_arr.dtype != fresh.dtype or rej_low_arr.dtype != fresh.dtype:
        raise ValueError(f"{op_name}: rejection image dtype mismatch")
    if not rej_high_arr.flags.c_contiguous:
        rej_high_arr = np.ascontiguousarray(rej_high_arr)
    if not rej_low_arr.flags.c_contiguous:
        rej_low_arr = np.ascontiguousarray(rej_low_arr)
    return rej_high_arr, rej_low_arr


def _broadcast_mask(mask: np.ndarray, fresh: np.ndarray) -> np.ndarray:
    if mask.ndim == fresh.ndim:
        return mask
    return mask[..., None]


def _validate_huber_target(base: Any) -> tuple[np.ndarray, np.ndarray]:
    weighted_sum = np.asarray(base.weighted_sum)
    weight_total = np.asarray(base.weight_total)
    if weighted_sum.shape != weight_total.shape:
        raise ValueError("huber_weighted_accumulate: accumulator shape mismatch")
    if not weighted_sum.flags.c_contiguous or not weight_total.flags.c_contiguous:
        raise ValueError("huber_weighted_accumulate: accumulators must be C-contiguous")
    if not weighted_sum.flags.writeable or not weight_total.flags.writeable:
        raise ValueError("huber_weighted_accumulate: accumulators must be writeable")
    if weighted_sum.dtype != np.float64 or weight_total.dtype != np.float64:
        raise ValueError("huber_weighted_accumulate: accumulators must be float64")
    return weighted_sum, weight_total


def _validate_ref_stats(
    fresh: np.ndarray,
    ref_mean: np.ndarray,
    ref_std: np.ndarray,
    *,
    op_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    ref_mean_arr = np.asarray(ref_mean, dtype=np.float32)
    ref_std_arr = np.asarray(ref_std, dtype=np.float32)
    if ref_mean_arr.shape != fresh.shape or ref_std_arr.shape != fresh.shape:
        raise ValueError(f"{op_name}: reference stats shape mismatch")
    if not ref_mean_arr.flags.c_contiguous:
        ref_mean_arr = np.ascontiguousarray(ref_mean_arr)
    if not ref_std_arr.flags.c_contiguous:
        ref_std_arr = np.ascontiguousarray(ref_std_arr)
    return ref_mean_arr, ref_std_arr


def fgp_accumulate_numpy(base: Any, fresh: np.ndarray, weight: Any = None,
                         skip_zero_rgb: bool = False) -> Any:
    sum_mu, square_sum, count = _maybe_prepare_target(base, weight)
    fresh_arr = _validate_fresh(sum_mu, fresh)
    int_weight = _validate_integer_weight(weight)

    if skip_zero_rgb and fresh_arr.ndim >= 3 and fresh_arr.shape[-1] >= 3:
        zero_mask = np.all(fresh_arr[..., :3] == 0, axis=-1, keepdims=True)
        active = ~np.broadcast_to(zero_mask, fresh_arr.shape)
    else:
        active = None

    if int_weight is None:
        if active is not None:
            sum_mu += np.where(active, fresh_arr, 0)
            square_sum += np.where(active, np.square(fresh_arr, dtype=square_sum.dtype), 0)
            count += np.where(active, 1, 0).astype(count.dtype)
        else:
            sum_mu += fresh_arr
            square_sum += np.square(fresh_arr, dtype=square_sum.dtype)
            count += 1
        return base

    if active is not None:
        sum_mu += np.where(active, np.multiply(fresh_arr, int_weight, dtype=sum_mu.dtype), 0)
        square_sum += np.where(active, np.multiply(
            np.square(fresh_arr, dtype=square_sum.dtype), int_weight, dtype=square_sum.dtype), 0)
        count += np.where(active, int_weight, 0).astype(count.dtype)
    else:
        sum_mu += np.multiply(fresh_arr, int_weight, dtype=sum_mu.dtype)
        square_sum += np.multiply(
            np.square(fresh_arr, dtype=square_sum.dtype),
            int_weight,
            dtype=square_sum.dtype,
        )
        count += int_weight
    return base


def fgp_accumulate_compiled(base: Any, fresh: np.ndarray, weight: Any = None,
                            skip_zero_rgb: bool = False) -> Any:
    module, _ = _load_compiled_module_result()
    if module is None:
        raise RuntimeError("compiled custom op backend is unavailable")
    sum_mu, square_sum, count = _maybe_prepare_target(base, weight)
    fresh_arr = _validate_fresh(sum_mu, fresh)
    int_weight = _validate_integer_weight(weight)
    _apply_compiled_threads("fgp_accumulate", fresh_arr)
    module.fgp_accumulate(sum_mu, square_sum, count, fresh_arr, int_weight,
                          skip_zero_rgb)
    return base


@lru_cache(maxsize=2)
def _select_fgp_backend(preference: str) -> tuple[str, Callable[[Any, np.ndarray, Any], Any]]:
    available, compiled_error = _compiled_backend_available("fgp_accumulate", preference)
    if available:
        return "compiled", fgp_accumulate_compiled

    if compiled_error:
        _debug_log(f"compiled backend unavailable, reason: {compiled_error}")

    return "numpy", fgp_accumulate_numpy


def fgp_accumulate(base: Any, fresh: np.ndarray, weight: Any = None,
                   skip_zero_rgb: bool = False) -> Any:
    int_weight = _validate_integer_weight(weight)
    if weight is not None and int_weight is None:
        return _python_fallback(base, np.asarray(fresh), weight)
    backend_name, backend = _select_fgp_backend(_fallback_preference())
    if backend_name == "compiled":
        sum_mu, _, _ = _validate_target(base)
        fresh_arr = _validate_fresh(sum_mu, fresh)
        _apply_compiled_threads("fgp_accumulate", fresh_arr)
    return backend(base, fresh, int_weight, skip_zero_rgb=skip_zero_rgb)


def fgp_add_numpy(base: Any, other: Any) -> Any:
    sum_mu, square_sum, count = _maybe_prepare_target_add(base, other)
    other_sum_mu, other_square_sum, other_n = _validate_peer(
        other,
        sum_mu.shape,
        op_name="fgp_add",
    )
    np.add(sum_mu, other_sum_mu, out=sum_mu, casting="unsafe")
    np.add(square_sum, other_square_sum, out=square_sum, casting="unsafe")
    np.add(count, other_n, out=count, casting="unsafe")
    return base


def fgp_add_compiled(base: Any, other: Any) -> Any:
    module, _ = _load_compiled_module_result()
    if module is None:
        raise RuntimeError("compiled custom op backend is unavailable")
    sum_mu, square_sum, count = _maybe_prepare_target_add(base, other)
    other_sum_mu, other_square_sum, other_n = _validate_peer(
        other,
        sum_mu.shape,
        op_name="fgp_add",
    )
    _apply_compiled_threads("fgp_add", sum_mu)
    module.fgp_add(
        sum_mu,
        square_sum,
        count,
        other_sum_mu,
        other_square_sum,
        other_n,
    )
    return base


@lru_cache(maxsize=2)
def _select_fgp_add_backend(preference: str) -> tuple[str, Callable[[Any, Any], Any]]:
    available, compiled_error = _compiled_backend_available("fgp_add", preference)
    if available:
        return "compiled", fgp_add_compiled

    if compiled_error:
        _debug_log(f"compiled backend unavailable, reason: {compiled_error}")

    return "numpy", fgp_add_numpy


def fgp_add(base: Any, other: Any) -> Any:
    backend_name, backend = _select_fgp_add_backend(_fallback_preference())
    if backend_name == "compiled":
        sum_mu, _, _ = _validate_target(base)
        _validate_peer(other, sum_mu.shape, op_name="fgp_add")
        _apply_compiled_threads("fgp_add", sum_mu)
    return backend(base, other)


def huber_weighted_accumulate_numpy(
    base: Any,
    fresh: np.ndarray,
    ref_mean: np.ndarray,
    ref_std: np.ndarray,
    huber_c: float,
    weight: Any = None,
) -> Any:
    weighted_sum, weight_total = _validate_huber_target(base)
    fresh_arr = _validate_fresh(
        weighted_sum,
        fresh,
        op_name="huber_weighted_accumulate",
    )
    ref_mean_arr, ref_std_arr = _validate_ref_stats(
        fresh_arr,
        ref_mean,
        ref_std,
        op_name="huber_weighted_accumulate",
    )
    scalar_weight = _validate_scalar_weight(
        weight,
        op_name="huber_weighted_accumulate",
    )
    residual = (
        fresh_arr.astype(np.float32) - ref_mean_arr
    ) / (ref_std_arr + np.float32(1e-10))
    abs_residual = np.abs(residual)
    huber_weight = np.where(
        abs_residual <= huber_c,
        np.ones_like(abs_residual, dtype=np.float32),
        (huber_c / (abs_residual + np.float32(1e-10))).astype(np.float32),
    )
    if weight is not None:
        if scalar_weight is not None:
            huber_weight = huber_weight * scalar_weight
        else:
            huber_weight = huber_weight * np.asarray(weight)
    weighted_sum += np.multiply(fresh_arr, huber_weight, dtype=weighted_sum.dtype)
    weight_total += huber_weight.astype(weight_total.dtype, copy=False)
    return base


def huber_weighted_accumulate_compiled(
    base: Any,
    fresh: np.ndarray,
    ref_mean: np.ndarray,
    ref_std: np.ndarray,
    huber_c: float,
    weight: Any = None,
) -> Any:
    module, _ = _load_compiled_module_result()
    if module is None:
        raise RuntimeError("compiled custom op backend is unavailable")
    scalar_weight = _validate_scalar_weight(
        weight,
        op_name="huber_weighted_accumulate",
    )
    if weight is not None and scalar_weight is None:
        raise ValueError("huber_weighted_accumulate: compiled backend only supports scalar weight")
    weighted_sum, weight_total = _validate_huber_target(base)
    fresh_arr = _validate_fresh(
        weighted_sum,
        fresh,
        op_name="huber_weighted_accumulate",
    )
    ref_mean_arr, ref_std_arr = _validate_ref_stats(
        fresh_arr,
        ref_mean,
        ref_std,
        op_name="huber_weighted_accumulate",
    )
    _apply_compiled_threads("huber_weighted_accumulate", fresh_arr)
    module.huber_weighted_accumulate(
        weighted_sum,
        weight_total,
        fresh_arr,
        ref_mean_arr,
        ref_std_arr,
        float(huber_c),
        scalar_weight,
    )
    return base


@lru_cache(maxsize=2)
def _select_huber_backend(
    preference: str,
) -> tuple[str, Callable[[Any, np.ndarray, np.ndarray, np.ndarray, float, Any], Any]]:
    available, compiled_error = _compiled_backend_available(
        "huber_weighted_accumulate",
        preference,
    )
    if available:
        return "compiled", huber_weighted_accumulate_compiled

    if compiled_error:
        _debug_log(f"compiled backend unavailable, reason: {compiled_error}")

    return "numpy", huber_weighted_accumulate_numpy


def huber_weighted_accumulate(
    base: Any,
    fresh: np.ndarray,
    ref_mean: np.ndarray,
    ref_std: np.ndarray,
    huber_c: float,
    weight: Any = None,
) -> Any:
    scalar_weight = _validate_scalar_weight(
        weight,
        op_name="huber_weighted_accumulate",
    )
    if weight is not None and scalar_weight is None:
        return huber_weighted_accumulate_numpy(
            base,
            fresh,
            ref_mean,
            ref_std,
            huber_c,
            weight,
        )
    backend_name, backend = _select_huber_backend(_fallback_preference())
    if backend_name == "compiled":
        weighted_sum, _ = _validate_huber_target(base)
        fresh_arr = _validate_fresh(
            weighted_sum,
            fresh,
            op_name="huber_weighted_accumulate",
        )
        _validate_ref_stats(
            fresh_arr,
            ref_mean,
            ref_std,
            op_name="huber_weighted_accumulate",
        )
        _apply_compiled_threads("huber_weighted_accumulate", fresh_arr)
    return backend(base, fresh, ref_mean, ref_std, huber_c, scalar_weight)


def fgp_masked_mean_merge_numpy(
    fresh: np.ndarray,
    mask: np.ndarray,
    sum_mu: np.ndarray,
    square_sum: np.ndarray,
    n: np.ndarray,
    skip_zero_rgb: bool = False,
) -> None:
    sum_arr, square_arr, count_arr = _validate_buffers(
        sum_mu, square_sum, n, op_name="fgp_masked_mean_merge")
    fresh_arr = _validate_fresh(sum_arr, fresh, op_name="fgp_masked_mean_merge")
    mask_arr = _validate_spatial_mask(fresh_arr, mask, op_name="fgp_masked_mean_merge")
    active = _broadcast_mask(mask_arr, fresh_arr)
    if skip_zero_rgb and fresh_arr.ndim >= 3 and fresh_arr.shape[-1] >= 3:
        zero_mask = np.all(fresh_arr[..., :3] == 0, axis=-1, keepdims=True)
        active = active & ~np.broadcast_to(zero_mask, fresh_arr.shape)
    sum_arr += np.multiply(fresh_arr, active, dtype=sum_arr.dtype)
    square_arr += np.multiply(
        np.square(fresh_arr, dtype=square_arr.dtype),
        active,
        dtype=square_arr.dtype,
    )
    count_arr += active.astype(count_arr.dtype, copy=False)


def fgp_masked_mean_merge_compiled(
    fresh: np.ndarray,
    mask: np.ndarray,
    sum_mu: np.ndarray,
    square_sum: np.ndarray,
    n: np.ndarray,
    skip_zero_rgb: bool = False,
) -> None:
    module, _ = _load_compiled_module_result()
    if module is None:
        raise RuntimeError("compiled custom op backend is unavailable")
    sum_arr, square_arr, count_arr = _validate_buffers(
        sum_mu, square_sum, n, op_name="fgp_masked_mean_merge")
    fresh_arr = _validate_fresh(sum_arr, fresh, op_name="fgp_masked_mean_merge")
    mask_arr = _validate_spatial_mask(fresh_arr, mask, op_name="fgp_masked_mean_merge")
    _apply_compiled_threads("fgp_masked_mean_merge", fresh_arr)
    module.fgp_masked_mean_merge(sum_arr, square_arr, count_arr, fresh_arr, mask_arr,
                                 skip_zero_rgb)


def fgp_masked_mean_merge(
    fresh: np.ndarray,
    mask: np.ndarray,
    sum_mu: np.ndarray,
    square_sum: np.ndarray,
    n: np.ndarray,
    skip_zero_rgb: bool = False,
) -> None:
    available, compiled_error = _compiled_backend_available(
        "fgp_masked_mean_merge",
        _fallback_preference(),
    )
    if available:
        fgp_masked_mean_merge_compiled(fresh, mask, sum_mu, square_sum, n,
                                       skip_zero_rgb)
        return
    if compiled_error:
        _debug_log(f"compiled backend unavailable, reason: {compiled_error}")
    fgp_masked_mean_merge_numpy(fresh, mask, sum_mu, square_sum, n,
                                skip_zero_rgb)


def sigma_clip_fused_merge_numpy(
    fresh: np.ndarray,
    rej_high_img: np.ndarray,
    rej_low_img: np.ndarray,
    sum_mu: np.ndarray,
    square_sum: np.ndarray,
    n: np.ndarray,
    skip_zero_rgb: bool = False,
) -> None:
    sum_arr, square_arr, count_arr = _validate_buffers(
        sum_mu, square_sum, n, op_name="sigma_clip_fused_merge")
    fresh_arr = _validate_fresh(sum_arr, fresh, op_name="sigma_clip_fused_merge")
    rej_high_arr, rej_low_arr = _validate_rejection_images(
        fresh_arr, rej_high_img, rej_low_img, op_name="sigma_clip_fused_merge")
    rejected = (fresh_arr < rej_low_arr) | (fresh_arr > rej_high_arr)
    if skip_zero_rgb and fresh_arr.ndim >= 3 and fresh_arr.shape[-1] >= 3:
        zero_mask = np.all(fresh_arr[..., :3] == 0, axis=-1, keepdims=True)
        rejected = rejected & ~np.broadcast_to(zero_mask, fresh_arr.shape)
    sum_arr += np.multiply(fresh_arr, rejected, dtype=sum_arr.dtype)
    square_arr += np.multiply(
        np.square(fresh_arr, dtype=square_arr.dtype),
        rejected,
        dtype=square_arr.dtype,
    )
    count_arr += rejected.astype(count_arr.dtype, copy=False)


def sigma_clip_fused_merge_compiled(
    fresh: np.ndarray,
    rej_high_img: np.ndarray,
    rej_low_img: np.ndarray,
    sum_mu: np.ndarray,
    square_sum: np.ndarray,
    n: np.ndarray,
    skip_zero_rgb: bool = False,
) -> None:
    module, _ = _load_compiled_module_result()
    if module is None:
        raise RuntimeError("compiled custom op backend is unavailable")
    sum_arr, square_arr, count_arr = _validate_buffers(
        sum_mu, square_sum, n, op_name="sigma_clip_fused_merge")
    fresh_arr = _validate_fresh(sum_arr, fresh, op_name="sigma_clip_fused_merge")
    rej_high_arr, rej_low_arr = _validate_rejection_images(
        fresh_arr, rej_high_img, rej_low_img, op_name="sigma_clip_fused_merge")
    _apply_compiled_threads("sigma_clip_fused_merge", fresh_arr)
    module.sigma_clip_fused_merge(
        sum_arr,
        square_arr,
        count_arr,
        fresh_arr,
        rej_high_arr,
        rej_low_arr,
        skip_zero_rgb,
    )


def sigma_clip_fused_merge(
    fresh: np.ndarray,
    rej_high_img: np.ndarray,
    rej_low_img: np.ndarray,
    sum_mu: np.ndarray,
    square_sum: np.ndarray,
    n: np.ndarray,
    skip_zero_rgb: bool = False,
) -> None:
    available, compiled_error = _compiled_backend_available(
        "sigma_clip_fused_merge",
        _fallback_preference(),
    )
    if available:
        sigma_clip_fused_merge_compiled(
            fresh, rej_high_img, rej_low_img, sum_mu, square_sum, n,
            skip_zero_rgb)
        return
    if compiled_error:
        _debug_log(f"compiled backend unavailable, reason: {compiled_error}")
    sigma_clip_fused_merge_numpy(
        fresh, rej_high_img, rej_low_img, sum_mu, square_sum, n,
        skip_zero_rgb)


def sigma_clip_fused_masked_merge_numpy(
    fresh: np.ndarray,
    mask: np.ndarray,
    rej_high_img: np.ndarray,
    rej_low_img: np.ndarray,
    sum_mu: np.ndarray,
    square_sum: np.ndarray,
    n: np.ndarray,
    skip_zero_rgb: bool = False,
) -> None:
    sum_arr, square_arr, count_arr = _validate_buffers(
        sum_mu, square_sum, n, op_name="sigma_clip_fused_masked_merge")
    fresh_arr = _validate_fresh(sum_arr, fresh, op_name="sigma_clip_fused_masked_merge")
    mask_arr = _validate_spatial_mask(
        fresh_arr, mask, op_name="sigma_clip_fused_masked_merge")
    rej_high_arr, rej_low_arr = _validate_rejection_images(
        fresh_arr,
        rej_high_img,
        rej_low_img,
        op_name="sigma_clip_fused_masked_merge",
    )
    active = _broadcast_mask(mask_arr, fresh_arr)
    if skip_zero_rgb and fresh_arr.ndim >= 3 and fresh_arr.shape[-1] >= 3:
        zero_mask = np.all(fresh_arr[..., :3] == 0, axis=-1, keepdims=True)
        active = active & ~np.broadcast_to(zero_mask, fresh_arr.shape)
    rejected = active & ((fresh_arr < rej_low_arr) | (fresh_arr > rej_high_arr))
    sum_arr += np.multiply(fresh_arr, rejected, dtype=sum_arr.dtype)
    square_arr += np.multiply(
        np.square(fresh_arr, dtype=square_arr.dtype),
        rejected,
        dtype=square_arr.dtype,
    )
    count_arr += rejected.astype(count_arr.dtype, copy=False)


def sigma_clip_fused_masked_merge_compiled(
    fresh: np.ndarray,
    mask: np.ndarray,
    rej_high_img: np.ndarray,
    rej_low_img: np.ndarray,
    sum_mu: np.ndarray,
    square_sum: np.ndarray,
    n: np.ndarray,
    skip_zero_rgb: bool = False,
) -> None:
    module, _ = _load_compiled_module_result()
    if module is None:
        raise RuntimeError("compiled custom op backend is unavailable")
    sum_arr, square_arr, count_arr = _validate_buffers(
        sum_mu, square_sum, n, op_name="sigma_clip_fused_masked_merge")
    fresh_arr = _validate_fresh(sum_arr, fresh, op_name="sigma_clip_fused_masked_merge")
    mask_arr = _validate_spatial_mask(
        fresh_arr, mask, op_name="sigma_clip_fused_masked_merge")
    rej_high_arr, rej_low_arr = _validate_rejection_images(
        fresh_arr,
        rej_high_img,
        rej_low_img,
        op_name="sigma_clip_fused_masked_merge",
    )
    _apply_compiled_threads("sigma_clip_fused_masked_merge", fresh_arr)
    module.sigma_clip_fused_masked_merge(
        sum_arr,
        square_arr,
        count_arr,
        fresh_arr,
        rej_high_arr,
        rej_low_arr,
        mask_arr,
        skip_zero_rgb,
    )


def sigma_clip_fused_masked_merge(
    fresh: np.ndarray,
    mask: np.ndarray,
    rej_high_img: np.ndarray,
    rej_low_img: np.ndarray,
    sum_mu: np.ndarray,
    square_sum: np.ndarray,
    n: np.ndarray,
    skip_zero_rgb: bool = False,
) -> None:
    available, compiled_error = _compiled_backend_available(
        "sigma_clip_fused_masked_merge",
        _fallback_preference(),
    )
    if available:
        sigma_clip_fused_masked_merge_compiled(
            fresh, mask, rej_high_img, rej_low_img, sum_mu, square_sum, n,
            skip_zero_rgb)
        return
    if compiled_error:
        _debug_log(f"compiled backend unavailable, reason: {compiled_error}")
    sigma_clip_fused_masked_merge_numpy(
        fresh, mask, rej_high_img, rej_low_img, sum_mu, square_sum, n,
        skip_zero_rgb)
