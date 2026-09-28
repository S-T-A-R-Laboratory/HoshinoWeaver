"""Alignment matching custom-op runtime backends."""

from __future__ import annotations

from functools import partial
from typing import Callable

import numpy as np
import numpy.linalg as la
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from scipy.spatial import distance as spd

from hoshicore._custom_op._dispatch import apply_compiled_threads as _apply_compiled_threads
from hoshicore._custom_op._dispatch import debug_log
from hoshicore._custom_op._dispatch import fallback_preference as _fallback_preference
from hoshicore._custom_op._dispatch import load_compiled_module as _load_compiled_module_result
from hoshicore._custom_op.backend_registry import BackendSelection
from hoshicore._custom_op.backend_registry import run_with_accelerator_fallback
from hoshicore._custom_op.backend_registry import resolve_backend as _resolve_backend
from hoshicore._custom_op.cuda_memory import cuda_memory_estimate
from hoshicore._custom_op.cuda_memory import run_admitted_cuda as _run_admitted_cuda


_debug_log = partial(debug_log, "alignment")


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


def extract_point_features_numpy(
    vec: NDArray[np.float64],
    vol: NDArray[np.float64],
    k: int = 15,
) -> NDArray[np.float64]:
    """Canonical angular-histogram descriptor, mirroring Norma's original
    matching.extract_point_features algorithm (kept here directly, rather
    than imported back from matching.py, to avoid a runtime import cycle
    through the custom-op dispatch)."""
    vec = _as_float64_c("extract_point_features: vec", vec, 2, 3)
    vol = _as_float64_c("extract_point_features: vol", vol, 1)
    if len(vol) != len(vec):
        raise ValueError("extract_point_features: vol length must match vec")

    pts_num = len(vec)
    neighbor_count = min(2 * k, pts_num)
    if neighbor_count < k:
        raise ValueError(
            f"extract_point_features requires at least k={k} points, got {pts_num}")

    # Unit-vector chord distance is monotonic with angular distance, so the
    # tree returns the same nearest-neighbor set without building an N x N
    # cosine-distance matrix.
    _, vec_dist_ind = cKDTree(vec).query(vec, k=neighbor_count)
    if neighbor_count == 1:
        vec_dist_ind = vec_dist_ind[:, np.newaxis]

    neighbor_vec = vec[vec_dist_ind]
    cos_dist = np.sum(vec[:, np.newaxis, :] * neighbor_vec, axis=2)
    dist_mat = np.arccos(np.clip(cos_dist, -1, 1))
    neighbor_vol = vol[vec_dist_ind]
    vol_ind = np.argsort(-neighbor_vol * dist_mat)

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
        vol_feature[i] = neighbor_vol[i, vol_ind[i, :k]]

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


def _validate_point_feature_inputs(
    vec: NDArray[np.float64],
    vol: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    return (_as_float64_c("extract_point_features: vec", vec, 2, 3),
            _as_float64_c("extract_point_features: vol", vol, 1))


def _checked_feature_layout(
    result: NDArray[np.float64] | None,
    vec_arr: NDArray[np.float64],
    vol_arr: NDArray[np.float64],
    k: int,
) -> NDArray[np.float64] | None:
    # Reject stale or incompatible local extensions rather than silently
    # mixing descriptor layouts.
    if result is not None and (result.ndim != 2 or result.shape != (len(vec_arr), 120)):
        _debug_log(
            "compiled extract_point_features has an incompatible layout; "
            "falling back to the canonical angular-histogram implementation")
        return extract_point_features_numpy(vec_arr, vol_arr, k=k)
    return result


def extract_point_features_compiled(
    vec: NDArray[np.float64],
    vol: NDArray[np.float64],
    k: int = 15,
) -> NDArray[np.float64]:
    module, _ = _load_compiled_module_result()
    if module is None or not hasattr(module, "extract_point_features"):
        raise RuntimeError("compiled custom op backend is unavailable")
    vec_arr, vol_arr = _validate_point_feature_inputs(vec, vol)
    _apply_compiled_threads("extract_point_features", vec_arr)
    result = module.extract_point_features(vec_arr, vol_arr, int(k))
    return _checked_feature_layout(result, vec_arr, vol_arr, k)


def extract_point_features_cuda(
    vec: NDArray[np.float64],
    vol: NDArray[np.float64],
    k: int = 15,
) -> NDArray[np.float64] | None:
    """CUDA descriptors; neighbour selection matches the CPU backend exactly
    and values differ only by device acos/atan2/exp rounding. Returns None
    when 2k exceeds the device neighbour pool or a near tie in the vol*rho
    ordering could select differently from the CPU backend."""
    module, _ = _load_compiled_module_result()
    if module is None or not hasattr(module, "extract_point_features_cuda"):
        raise RuntimeError("compiled custom op backend is unavailable")
    vec_arr, vol_arr = _validate_point_feature_inputs(vec, vol)
    if len(vec_arr) == 0:
        return module.extract_point_features_cuda(vec_arr, vol_arr, int(k))
    estimate = cuda_memory_estimate(
        "extract_point_features", n_points=len(vec_arr), k=int(k))
    result = _run_admitted_cuda(
        estimate, module.extract_point_features_cuda, vec_arr, vol_arr, int(k))
    return _checked_feature_layout(result, vec_arr, vol_arr, k)


def _extract_point_features_backend(
    selection: BackendSelection,
) -> tuple[str, Callable[..., NDArray[np.float64] | None]]:
    if not selection.native or selection.candidate is None:
        return "numpy", extract_point_features_numpy
    if selection.candidate.kernel_name == "extract_point_features_cuda":
        return "cuda", extract_point_features_cuda
    if selection.candidate.kernel_name == "extract_point_features":
        return "cpu", extract_point_features_compiled
    raise RuntimeError(
        f"unknown extract_point_features backend candidate: {selection.candidate}")


def extract_point_features(
    vec: NDArray[np.float64],
    vol: NDArray[np.float64],
    k: int = 15,
) -> NDArray[np.float64]:
    vec_arr, vol_arr = _validate_point_feature_inputs(vec, vol)
    selection = _resolve_backend(
        "extract_point_features",
        _fallback_preference(),
        load_module=_load_compiled_module_result,
    )
    if selection.reason:
        _debug_log(f"compiled backend unavailable, reason: {selection.reason}")

    result = run_with_accelerator_fallback(
        "extract_point_features",
        selection,
        _extract_point_features_backend,
        lambda entry: entry(vec_arr, vol_arr, k),
        load_module=_load_compiled_module_result,
        log=_debug_log,
    )
    if result is None:
        _debug_log("CUDA extract_point_features declined the input; using OpenMP")
        return extract_point_features_compiled(vec_arr, vol_arr, k)
    return result


MatchingNearestResult = tuple[
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.float64],
]


def _validate_matching_features(
    features1: NDArray[np.float64],
    features2: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    features1_arr = _as_float64_c(
        "matching_cosine_bidirectional_nearest: features1", features1, 2)
    features2_arr = _as_float64_c(
        "matching_cosine_bidirectional_nearest: features2", features2, 2)
    if features1_arr.shape[1] != features2_arr.shape[1]:
        raise ValueError(
            "matching_cosine_bidirectional_nearest: feature dimensions must match"
        )
    if min(
        features1_arr.shape[0],
        features2_arr.shape[0],
        features1_arr.shape[1],
    ) <= 0:
        raise ValueError(
            "matching_cosine_bidirectional_nearest: feature dimensions must be positive"
        )
    return features1_arr, features2_arr


def matching_cosine_bidirectional_nearest_numpy(
    features1: NDArray[np.float64],
    features2: NDArray[np.float64],
) -> MatchingNearestResult:
    features1_arr, features2_arr = _validate_matching_features(
        features1, features2)
    distance_matrix = spd.cdist(features1_arr, features2_arr, "cosine")
    row_order = np.argsort(distance_matrix, axis=1)
    col_order = np.argsort(distance_matrix, axis=0)
    row_indices = np.ascontiguousarray(row_order[:, 0], dtype=np.int64)
    col_indices = np.ascontiguousarray(col_order[0, :], dtype=np.int64)
    row_distances = np.ascontiguousarray(
        distance_matrix[np.arange(len(features1_arr)), row_indices],
        dtype=np.float64,
    )
    col_distances = np.ascontiguousarray(
        distance_matrix[col_indices, np.arange(len(features2_arr))],
        dtype=np.float64,
    )
    return row_indices, row_distances, col_indices, col_distances


def _matching_cosine_bidirectional_nearest_compiled_kernel(
    kernel_name: str,
    features1: NDArray[np.float64],
    features2: NDArray[np.float64],
) -> MatchingNearestResult | None:
    module, _ = _load_compiled_module_result()
    if module is None or not hasattr(module, kernel_name):
        raise RuntimeError("compiled custom op backend is unavailable")
    features1_arr, features2_arr = _validate_matching_features(
        features1, features2)
    kernel = getattr(module, kernel_name)
    if kernel_name == "matching_cosine_bidirectional_nearest_cpu":
        _apply_compiled_threads(
            "matching_cosine_bidirectional_nearest", features1_arr)
        return kernel(features1_arr, features2_arr)

    estimate = cuda_memory_estimate(
        "matching_cosine_bidirectional_nearest",
        n1=features1_arr.shape[0],
        n2=features2_arr.shape[0],
        feature_dim=features1_arr.shape[1],
    )
    return _run_admitted_cuda(estimate, kernel, features1_arr, features2_arr)


def matching_cosine_bidirectional_nearest_cpu_compiled(
    features1: NDArray[np.float64],
    features2: NDArray[np.float64],
) -> MatchingNearestResult | None:
    return _matching_cosine_bidirectional_nearest_compiled_kernel(
        "matching_cosine_bidirectional_nearest_cpu", features1, features2)


def matching_cosine_bidirectional_nearest_cuda(
    features1: NDArray[np.float64],
    features2: NDArray[np.float64],
) -> MatchingNearestResult | None:
    return _matching_cosine_bidirectional_nearest_compiled_kernel(
        "matching_cosine_bidirectional_nearest_cuda", features1, features2)


def _matching_cosine_bidirectional_nearest_backend(
    selection: BackendSelection,
) -> tuple[
    str,
    Callable[
        [NDArray[np.float64], NDArray[np.float64]],
        MatchingNearestResult | None,
    ],
]:
    if not selection.native or selection.candidate is None:
        return "numpy", matching_cosine_bidirectional_nearest_numpy
    if selection.candidate.kernel_name == "matching_cosine_bidirectional_nearest_cuda":
        return "cuda", matching_cosine_bidirectional_nearest_cuda
    if selection.candidate.kernel_name == "matching_cosine_bidirectional_nearest_cpu":
        return "cpu", matching_cosine_bidirectional_nearest_cpu_compiled
    raise RuntimeError(
        "unknown matching cosine bidirectional nearest backend candidate: "
        f"{selection.candidate}"
    )


def matching_cosine_bidirectional_nearest(
    features1: NDArray[np.float64],
    features2: NDArray[np.float64],
) -> MatchingNearestResult:
    features1_arr, features2_arr = _validate_matching_features(
        features1, features2)
    selection = _resolve_backend(
        "matching_cosine_bidirectional_nearest",
        _fallback_preference(),
        load_module=_load_compiled_module_result,
    )
    if selection.reason:
        _debug_log(f"matching backend unavailable, reason: {selection.reason}")

    result = run_with_accelerator_fallback(
        "matching_cosine_bidirectional_nearest",
        selection,
        _matching_cosine_bidirectional_nearest_backend,
        lambda entry: entry(features1_arr, features2_arr),
        load_module=_load_compiled_module_result,
        log=_debug_log,
    )

    if result is None:
        _debug_log(
            "matching native backend found ambiguous cosine ordering; "
            "recomputing with SciPy/NumPy"
        )
        return matching_cosine_bidirectional_nearest_numpy(
            features1_arr, features2_arr)
    return result
