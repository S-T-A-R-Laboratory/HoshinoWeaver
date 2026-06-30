"""Alignment matching custom-op runtime backends."""

from __future__ import annotations

from functools import lru_cache
from functools import partial
from typing import Callable

import numpy as np
import numpy.linalg as la
from numpy.typing import NDArray
from scipy.spatial import distance as spd

from hoshicore._custom_op._dispatch import apply_compiled_threads as _apply_compiled_threads
from hoshicore._custom_op._dispatch import debug_log
from hoshicore._custom_op._dispatch import fallback_preference as _fallback_preference
from hoshicore._custom_op._dispatch import load_compiled_module as _load_compiled_module_result
from hoshicore._custom_op.backend_registry import native_backend_available as _native_backend_available


_debug_log = partial(debug_log, "alignment")

MIN_FILTERED_UNIQUE_PAIRS = 4
LOW_PAIR_COUNT_THRESHOLD = 10
MIN_FILTER_KEEP_RATIO = 0.5


def _make_cross_matrix(v: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def _as_float64_c(name: str, value: np.ndarray, ndim: int, trailing: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != ndim:
        raise ValueError(f"{name}: expected {ndim} dimensions")
    if trailing is not None and arr.shape[-1] != trailing:
        raise ValueError(f"{name}: expected trailing dimension {trailing}")
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    return arr


def _validate_match_inputs(
    features1: NDArray[np.float64],
    features2: NDArray[np.float64],
    pts1: NDArray[np.float64],
    pts2: NDArray[np.float64],
) -> None:
    if features1.shape[1] != features2.shape[1]:
        raise ValueError("find_initial_match: feature dimensions must match")
    if pts1.shape[0] != features1.shape[0] or pts2.shape[0] != features2.shape[0]:
        raise ValueError("find_initial_match: point counts must match feature counts")


def _should_fallback_to_unfiltered(
    before_filter_pair_idx: NDArray[np.int32],
    after_filter_pair_idx: NDArray[np.int32],
) -> bool:
    before_count = len(before_filter_pair_idx)
    after_count = len(after_filter_pair_idx)
    unique_after_count = len(np.unique(after_filter_pair_idx, axis=0))
    kept_ratio = after_count / before_count if before_count else 0.0
    return (
        unique_after_count < MIN_FILTERED_UNIQUE_PAIRS
        or (
            before_count < LOW_PAIR_COUNT_THRESHOLD
            and kept_ratio < MIN_FILTER_KEEP_RATIO
        )
    )


def extract_point_features_numpy(
    vec: NDArray[np.float64],
    vol: NDArray[np.float64],
    k: int = 15,
) -> NDArray[np.float64]:
    vec = _as_float64_c("extract_point_features: vec", vec, 2, 3)
    vol = _as_float64_c("extract_point_features: vol", vol, 1)
    if len(vol) != len(vec):
        raise ValueError("extract_point_features: vol length must match vec")

    pts_num = len(vec)
    dist_mat = 1 - spd.cdist(vec, vec, "cosine")
    vec_dist_ind = np.argsort(-dist_mat)
    dist_mat = np.clip(dist_mat, -1, 1)

    dist_mat = np.arccos(dist_mat[np.array(range(pts_num))[:, np.newaxis],
                                  vec_dist_ind[:, :2 * k]])
    vol = vol[vec_dist_ind[:, :2 * k]]
    vol_ind = np.argsort(-vol * dist_mat)

    theta_feature = np.zeros((pts_num, k))
    rho_feature = np.zeros((pts_num, k))
    vol_feature = np.zeros((pts_num, k))

    for i in range(pts_num):
        v0 = vec[i]
        vs = vec[vec_dist_ind[i, vol_ind[i, :k]]]
        angles = np.inner(vs, _make_cross_matrix(v0))
        angles = angles / la.norm(angles, axis=1)[:, np.newaxis]
        cr = np.inner(angles, _make_cross_matrix(angles[0]))
        s = la.norm(cr, axis=1) * np.sign(np.inner(cr, v0))
        c = np.inner(angles, angles[0])
        theta_feature[i] = np.arctan2(s, c)
        rho_feature[i] = dist_mat[i, vol_ind[i, :k]]
        vol_feature[i] = vol[i, vol_ind[i, :k]]

    fx = np.arange(-np.pi, np.pi, 3 * np.pi / 180)
    features = np.zeros((pts_num, len(fx)))
    for i in range(k):
        sigma = 2.5 * np.exp(-rho_feature[:, i] * 100) + .04
        tmp = np.exp(-np.subtract.outer(theta_feature[:, i], fx)**2 / 2 /
                     sigma[:, np.newaxis]**2)
        tmp = tmp * (vol_feature[:, i] * rho_feature[:, i]**2 /
                     sigma)[:, np.newaxis]
        features += tmp

    features = features / np.sqrt(np.sum(features**2, axis=1)).reshape(
        (pts_num, 1))
    return np.ascontiguousarray(features)


def extract_point_features_compiled(
    vec: NDArray[np.float64],
    vol: NDArray[np.float64],
    k: int = 15,
) -> NDArray[np.float64]:
    module, _ = _load_compiled_module_result()
    if module is None or not hasattr(module, "extract_point_features"):
        raise RuntimeError("compiled custom op backend is unavailable")
    vec_arr = _as_float64_c("extract_point_features: vec", vec, 2, 3)
    vol_arr = _as_float64_c("extract_point_features: vol", vol, 1)
    _apply_compiled_threads("extract_point_features", vec_arr)
    return module.extract_point_features(vec_arr, vol_arr, int(k))


@lru_cache(maxsize=2)
def _select_extract_point_features_backend(
    preference: str,
) -> tuple[str, Callable[[NDArray[np.float64], NDArray[np.float64], int], NDArray[np.float64]]]:
    available, compiled_error = _native_backend_available(
        "extract_point_features",
        preference,
        load_module=_load_compiled_module_result,
    )
    if available:
        return "compiled", extract_point_features_compiled

    if compiled_error:
        _debug_log(f"compiled backend unavailable, reason: {compiled_error}")

    return "numpy", extract_point_features_numpy


def extract_point_features(
    vec: NDArray[np.float64],
    vol: NDArray[np.float64],
    k: int = 15,
) -> NDArray[np.float64]:
    _, backend = _select_extract_point_features_backend(_fallback_preference())
    return backend(vec, vol, k)


def find_initial_match_numpy(
    features1: NDArray[np.float64],
    features2: NDArray[np.float64],
    pts1: NDArray[np.float64],
    pts2: NDArray[np.float64],
    vectors1: NDArray[np.float64] | None = None,
    vectors2: NDArray[np.float64] | None = None,
    alpha: float = 0.00,
    apply_threshold_filter: bool = True,
    theta_th: float = np.pi / 6,
    dist_multiplier: float = 0.3,
) -> NDArray[np.int32]:
    features1 = _as_float64_c("find_initial_match: features1", features1, 2)
    features2 = _as_float64_c("find_initial_match: features2", features2, 2)
    pts1 = _as_float64_c("find_initial_match: pts1", pts1, 2, 2)
    pts2 = _as_float64_c("find_initial_match: pts2", pts2, 2, 2)
    _validate_match_inputs(features1, features2, pts1, pts2)

    measure_dist_mat = spd.cdist(features1, features2, "cosine")
    if alpha > 0:
        pts_stack = np.vstack((pts1, pts2))
        pts_mean = np.mean(pts_stack, axis=0)
        pts_min = np.min(pts_stack, axis=0)
        pts_max = np.max(pts_stack, axis=0)
        pts_dist_mat = spd.cdist((pts1 - pts_mean) / (pts_max - pts_min),
                                 (pts2 - pts_mean) / (pts_max - pts_min),
                                 "euclidean")
        dist_mat = measure_dist_mat * (1 - alpha) + pts_dist_mat * alpha
    else:
        dist_mat = measure_dist_mat

    num1, num2 = dist_mat.shape

    idx12 = np.argsort(dist_mat, axis=1)
    idx21 = np.argsort(dist_mat, axis=0)
    ind = idx21[0, idx12[:, 0]] == range(num1)

    d_th = min(np.percentile(dist_mat[range(num1), idx12[:, 0]], 30),
               np.percentile(dist_mat[idx21[0, :], range(num2)], 30))
    ind = np.logical_and(ind, dist_mat[range(num1), idx12[:, 0]] < d_th)

    pair_idx = np.stack((np.where(ind)[0], idx12[ind, 0]), axis=-1).astype(
        np.int32, copy=False)

    if apply_threshold_filter:
        if vectors1 is None or vectors2 is None:
            raise ValueError(
                "vectors1 and vectors2 required when apply_threshold_filter=True"
            )
        vectors1 = _as_float64_c("find_initial_match: vectors1", vectors1, 2, 3)
        vectors2 = _as_float64_c("find_initial_match: vectors2", vectors2, 2, 3)
        unfiltered_pair_idx = pair_idx.copy()
        before_filter_count = len(pair_idx)
        if before_filter_count == 0:
            return pair_idx
        theta = np.arccos(
            np.clip(
                np.sum(vectors1[pair_idx[:, 0]] * vectors2[pair_idx[:, 1]],
                       axis=1), -1, 1))
        theta_th = min(np.percentile(theta, 75), theta_th)

        pts_dist = la.norm(pts1[pair_idx[:, 0]] - pts2[pair_idx[:, 1]], axis=1)
        dist_th = max(np.max(pts1), np.max(pts2)) * dist_multiplier
        pair_idx = pair_idx[np.logical_and(theta < theta_th,
                                           pts_dist < dist_th)]
        if _should_fallback_to_unfiltered(unfiltered_pair_idx, pair_idx):
            pair_idx = unfiltered_pair_idx
    return np.ascontiguousarray(pair_idx, dtype=np.int32)


def find_initial_match_compiled(
    features1: NDArray[np.float64],
    features2: NDArray[np.float64],
    pts1: NDArray[np.float64],
    pts2: NDArray[np.float64],
    vectors1: NDArray[np.float64] | None = None,
    vectors2: NDArray[np.float64] | None = None,
    alpha: float = 0.00,
    apply_threshold_filter: bool = True,
    theta_th: float = np.pi / 6,
    dist_multiplier: float = 0.3,
) -> NDArray[np.int32]:
    if alpha != 0.0:
        # Keep alpha's less common coordinate-weighted path on the exact NumPy
        # implementation so direct compiled benchmarks cannot accidentally
        # compare a different normalization edge case.
        return find_initial_match_numpy(
            features1,
            features2,
            pts1,
            pts2,
            vectors1,
            vectors2,
            alpha,
            apply_threshold_filter,
            theta_th,
            dist_multiplier,
        )
    module, _ = _load_compiled_module_result()
    if module is None or not hasattr(module, "find_initial_match"):
        raise RuntimeError("compiled custom op backend is unavailable")
    features1_arr = _as_float64_c("find_initial_match: features1", features1, 2)
    features2_arr = _as_float64_c("find_initial_match: features2", features2, 2)
    pts1_arr = _as_float64_c("find_initial_match: pts1", pts1, 2, 2)
    pts2_arr = _as_float64_c("find_initial_match: pts2", pts2, 2, 2)
    vectors1_arr = None if vectors1 is None else _as_float64_c(
        "find_initial_match: vectors1", vectors1, 2, 3)
    vectors2_arr = None if vectors2 is None else _as_float64_c(
        "find_initial_match: vectors2", vectors2, 2, 3)
    _apply_compiled_threads("find_initial_match", features1_arr)
    return module.find_initial_match(
        features1_arr,
        features2_arr,
        pts1_arr,
        pts2_arr,
        vectors1_arr,
        vectors2_arr,
        float(alpha),
        bool(apply_threshold_filter),
        float(theta_th),
        float(dist_multiplier),
    )


@lru_cache(maxsize=2)
def _select_find_initial_match_backend(
    preference: str,
) -> tuple[str, Callable[..., NDArray[np.int32]]]:
    available, compiled_error = _native_backend_available(
        "find_initial_match",
        preference,
        load_module=_load_compiled_module_result,
    )
    if available:
        return "compiled", find_initial_match_compiled

    if compiled_error:
        _debug_log(f"compiled backend unavailable, reason: {compiled_error}")

    return "numpy", find_initial_match_numpy


def find_initial_match(
    features1: NDArray[np.float64],
    features2: NDArray[np.float64],
    pts1: NDArray[np.float64],
    pts2: NDArray[np.float64],
    vectors1: NDArray[np.float64] | None = None,
    vectors2: NDArray[np.float64] | None = None,
    alpha: float = 0.00,
    apply_threshold_filter: bool = True,
    theta_th: float = np.pi / 6,
    dist_multiplier: float = 0.3,
) -> NDArray[np.int32]:
    _, backend = _select_find_initial_match_backend(_fallback_preference())
    return backend(
        features1,
        features2,
        pts1,
        pts2,
        vectors1,
        vectors2,
        alpha,
        apply_threshold_filter,
        theta_th,
        dist_multiplier,
    )
