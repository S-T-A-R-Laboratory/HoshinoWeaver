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


AsterismPairs = tuple[NDArray[np.int64], NDArray[np.int64]]


def _validate_asterism_values(
    values1: NDArray[np.float64],
    values2: NDArray[np.float64],
    threshold: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    values1_arr = _as_float64_c("asterism_mutual_nearest: values1", values1, 2, 3)
    values2_arr = _as_float64_c("asterism_mutual_nearest: values2", values2, 2, 3)
    threshold_value = float(threshold)
    if not np.isfinite(threshold_value) or threshold_value <= 0:
        raise ValueError("asterism_mutual_nearest: threshold must be positive")
    return values1_arr, values2_arr, threshold_value


def _empty_asterism_pairs() -> AsterismPairs:
    return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)


def asterism_mutual_nearest_numpy(
    values1: NDArray[np.float64],
    values2: NDArray[np.float64],
    threshold: float,
) -> AsterismPairs:
    """Mutual nearest token pairs within ``threshold`` via SciPy trees.

    This is the reference semantics: ``(i, j)`` is returned when ``j`` is the
    nearest token of set 2 to token ``i`` of set 1 and vice versa, and their
    Euclidean distance is at most ``threshold``. Pairs are in ascending ``i``.
    """
    values1_arr, values2_arr, threshold_value = _validate_asterism_values(
        values1, values2, threshold)
    if len(values1_arr) == 0 or len(values2_arr) == 0:
        return _empty_asterism_pairs()
    distance12, nearest12 = cKDTree(values2_arr).query(values1_arr, k=1)
    distance21, nearest21 = cKDTree(values1_arr).query(values2_arr, k=1)
    indices1 = np.arange(len(values1_arr), dtype=np.int64)
    mutual = (
        np.isfinite(distance12)
        & (distance12 <= threshold_value)
        & (nearest21[nearest12] == indices1)
        & (distance21[nearest12] <= threshold_value)
    )
    return (np.ascontiguousarray(indices1[mutual]),
            np.ascontiguousarray(nearest12[mutual], dtype=np.int64))


def _asterism_mutual_nearest_compiled_kernel(
    kernel_name: str,
    values1: NDArray[np.float64],
    values2: NDArray[np.float64],
    threshold: float,
) -> AsterismPairs | None:
    module, _ = _load_compiled_module_result()
    if module is None or not hasattr(module, kernel_name):
        raise RuntimeError("compiled custom op backend is unavailable")
    values1_arr, values2_arr, threshold_value = _validate_asterism_values(
        values1, values2, threshold)
    if len(values1_arr) == 0 or len(values2_arr) == 0:
        return _empty_asterism_pairs()
    kernel = getattr(module, kernel_name)
    if kernel_name == "asterism_mutual_nearest_cpu":
        _apply_compiled_threads("asterism_mutual_nearest", values1_arr)
        return kernel(values1_arr, values2_arr, threshold_value)

    estimate = cuda_memory_estimate(
        "asterism_mutual_nearest",
        n1=values1_arr.shape[0],
        n2=values2_arr.shape[0],
    )
    return _run_admitted_cuda(
        estimate, kernel, values1_arr, values2_arr, threshold_value)


def asterism_mutual_nearest_cpu_compiled(
    values1: NDArray[np.float64],
    values2: NDArray[np.float64],
    threshold: float,
) -> AsterismPairs | None:
    return _asterism_mutual_nearest_compiled_kernel(
        "asterism_mutual_nearest_cpu", values1, values2, threshold)


def asterism_mutual_nearest_cuda(
    values1: NDArray[np.float64],
    values2: NDArray[np.float64],
    threshold: float,
) -> AsterismPairs | None:
    return _asterism_mutual_nearest_compiled_kernel(
        "asterism_mutual_nearest_cuda", values1, values2, threshold)


def _asterism_mutual_nearest_backend(
    selection: BackendSelection,
) -> tuple[str, Callable[..., AsterismPairs | None]]:
    if not selection.native or selection.candidate is None:
        return "numpy", asterism_mutual_nearest_numpy
    if selection.candidate.kernel_name == "asterism_mutual_nearest_cuda":
        return "cuda", asterism_mutual_nearest_cuda
    if selection.candidate.kernel_name == "asterism_mutual_nearest_cpu":
        return "cpu", asterism_mutual_nearest_cpu_compiled
    raise RuntimeError(
        f"unknown asterism mutual nearest backend candidate: {selection.candidate}")


def asterism_mutual_nearest(
    values1: NDArray[np.float64],
    values2: NDArray[np.float64],
    threshold: float,
) -> AsterismPairs:
    """Mutual nearest asterism-token pairs within ``threshold``.

    Native backends search a threshold-sized cell grid in exact float64 and
    return the same pairs as :func:`asterism_mutual_nearest_numpy`. Inputs
    they cannot reproduce exactly (non-finite values, an oversized extent, or
    a tie at a nearest distance, where SciPy's pick is unspecified) are
    recomputed with SciPy.
    """
    values1_arr, values2_arr, threshold_value = _validate_asterism_values(
        values1, values2, threshold)
    if len(values1_arr) == 0 or len(values2_arr) == 0:
        return _empty_asterism_pairs()
    selection = _resolve_backend(
        "asterism_mutual_nearest",
        _fallback_preference(),
        load_module=_load_compiled_module_result,
    )
    if selection.reason:
        _debug_log(f"asterism backend unavailable, reason: {selection.reason}")

    result = run_with_accelerator_fallback(
        "asterism_mutual_nearest",
        selection,
        _asterism_mutual_nearest_backend,
        lambda entry: entry(values1_arr, values2_arr, threshold_value),
        load_module=_load_compiled_module_result,
        log=_debug_log,
    )
    if result is None:
        _debug_log("asterism native backend declined the input; using SciPy")
        return asterism_mutual_nearest_numpy(values1_arr, values2_arr, threshold_value)
    return result


AsterismTokenArrays = tuple[NDArray[np.float64], NDArray[np.int32]]
AsterismVotes = tuple[NDArray[np.int32], int, NDArray[np.int32]]


def _validate_asterism_token_inputs(
    vectors: NDArray[np.float64],
    neighbor_count: int,
) -> NDArray[np.float64]:
    vectors_arr = np.asarray(vectors, dtype=np.float64)
    if vectors_arr.ndim != 2 or vectors_arr.shape[1:] != (3,):
        raise ValueError("asterism tokens require an (N, 3) vector array")
    if not np.all(np.isfinite(vectors_arr)):
        raise ValueError("asterism tokens require finite vectors")
    if neighbor_count < 2:
        raise ValueError("asterism neighbor_count must be at least 2")
    if len(vectors_arr) < 3:
        raise ValueError("asterism tokens require at least 3 stars")
    if np.any(np.linalg.norm(vectors_arr, axis=1) <= 1e-12):
        raise ValueError("asterism tokens require non-zero vectors")
    return np.ascontiguousarray(vectors_arr)


def asterism_tokens_numpy(
    vectors: NDArray[np.float64],
    neighbor_count: int = 8,
) -> AsterismTokenArrays:
    """Unordered local spherical-triangle tokens for every star.

    Each star and two of its ``neighbor_count`` nearest neighbours give the
    token ``(short_anchor_edge / long_anchor_edge, neighbor_edge /
    long_anchor_edge, log(long_anchor_edge))`` from unit-sphere chords, which
    avoid inverse trigonometry while remaining rotation invariant. Returns the
    ``(M, 3)`` token values and the anchor star of each token.
    """
    vectors = _validate_asterism_token_inputs(vectors, neighbor_count)
    norms = np.linalg.norm(vectors, axis=1)
    unit = vectors / norms[:, np.newaxis]
    k = min(int(neighbor_count), len(unit) - 1)
    _, neighbor_indices = cKDTree(unit).query(unit, k=k + 1)
    neighbor_indices = np.asarray(neighbor_indices[:, 1:], dtype=np.int32)

    neighbor_vectors = unit[neighbor_indices]
    anchor_dot = np.sum(unit[:, np.newaxis, :] * neighbor_vectors, axis=2)
    anchor_edges = np.sqrt(
        np.maximum(2.0 - 2.0 * np.clip(anchor_dot, -1.0, 1.0), 0.0))
    left, right = np.triu_indices(k, k=1)
    first_edge = anchor_edges[:, left]
    second_edge = anchor_edges[:, right]
    short_edge = np.minimum(first_edge, second_edge)
    long_edge = np.maximum(first_edge, second_edge)

    first_neighbor = neighbor_vectors[:, left, :]
    second_neighbor = neighbor_vectors[:, right, :]
    neighbor_dot = np.sum(first_neighbor * second_neighbor, axis=2)
    neighbor_edge = np.sqrt(
        np.maximum(2.0 - 2.0 * np.clip(neighbor_dot, -1.0, 1.0), 0.0))

    valid = long_edge > 1e-12
    values = np.stack((
        short_edge[valid] / long_edge[valid],
        neighbor_edge[valid] / long_edge[valid],
        np.log(long_edge[valid]),
    ), axis=1)
    anchor_grid = np.broadcast_to(
        np.arange(len(unit), dtype=np.int32)[:, np.newaxis], long_edge.shape)
    return (np.ascontiguousarray(values, dtype=np.float64),
            np.ascontiguousarray(anchor_grid[valid], dtype=np.int32))


def asterism_tokens_cpu_compiled(
    vectors: NDArray[np.float64],
    neighbor_count: int = 8,
) -> AsterismTokenArrays | None:
    module, _ = _load_compiled_module_result()
    if module is None or not hasattr(module, "asterism_tokens_cpu"):
        raise RuntimeError("compiled custom op backend is unavailable")
    vectors = _validate_asterism_token_inputs(vectors, neighbor_count)
    _apply_compiled_threads("asterism_tokens", vectors)
    tokens = module.asterism_tokens_cpu(vectors, int(neighbor_count))
    if tokens is None:
        return None
    values, long_edge, anchors = tokens
    # The log stays with NumPy, whose SIMD log may round differently from libm.
    values[:, 2] = np.log(long_edge, out=long_edge)
    return values, anchors


def asterism_tokens(
    vectors: NDArray[np.float64],
    neighbor_count: int = 8,
) -> AsterismTokenArrays:
    """Local spherical-triangle tokens; see :func:`asterism_tokens_numpy`.

    The OpenMP backend reproduces the reference bitwise. When two of a star's
    nearest distances tie, cKDTree's neighbour order is unspecified, so the
    input is recomputed with the reference.
    """
    selection = _resolve_backend(
        "asterism_tokens",
        _fallback_preference(),
        load_module=_load_compiled_module_result,
    )
    if selection.native:
        result = asterism_tokens_cpu_compiled(vectors, neighbor_count)
        if result is not None:
            return result
        _debug_log("asterism token backend declined the input; using SciPy")
    elif selection.reason:
        _debug_log(f"asterism token backend unavailable, reason: {selection.reason}")
    return asterism_tokens_numpy(vectors, neighbor_count)


def _validate_asterism_vote_inputs(
    token_pairs1: NDArray[np.int64],
    token_pairs2: NDArray[np.int64],
    anchor_indices1: NDArray[np.int32],
    anchor_indices2: NDArray[np.int32],
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int32], NDArray[np.int32]]:
    pairs1 = np.ascontiguousarray(token_pairs1, dtype=np.int64)
    pairs2 = np.ascontiguousarray(token_pairs2, dtype=np.int64)
    anchors1 = np.ascontiguousarray(anchor_indices1, dtype=np.int32)
    anchors2 = np.ascontiguousarray(anchor_indices2, dtype=np.int32)
    if (pairs1.ndim != 1 or pairs1.shape != pairs2.shape
            or anchors1.ndim != 1 or anchors2.ndim != 1):
        raise ValueError(
            "asterism_anchor_votes: token pairs and anchors must be matching 1-D arrays")
    return pairs1, pairs2, anchors1, anchors2


def asterism_anchor_votes_numpy(
    token_pairs1: NDArray[np.int64],
    token_pairs2: NDArray[np.int64],
    anchor_indices1: NDArray[np.int32],
    anchor_indices2: NDArray[np.int32],
    num1: int,
    num2: int,
    min_votes: int,
    min_vote_margin: int,
) -> AsterismVotes:
    """Star pairs whose anchors vote for each other through matched tokens.

    Every token pair votes for its anchor-star pair. A pair is accepted when
    each star's best partner is the other, with at least ``min_votes`` votes
    and a lead of ``min_vote_margin`` over its runner-up; ties for the best
    partner go to the lower index. Returns ``(pair_idx, voted_pairs,
    accepted_votes)``.
    """
    pairs1, pairs2, anchors1, anchors2 = _validate_asterism_vote_inputs(
        token_pairs1, token_pairs2, anchor_indices1, anchor_indices2)
    anchor1 = anchors1[pairs1].astype(np.int64)
    anchor2 = anchors2[pairs2].astype(np.int64)
    pair_codes = anchor1 * num2 + anchor2
    unique_codes, votes = np.unique(pair_codes, return_counts=True)
    pair_anchor1 = unique_codes // num2
    pair_anchor2 = unique_codes % num2

    best2_for_1 = np.full(num1, -1, dtype=np.int64)
    best_votes1 = np.zeros(num1, dtype=np.int32)
    second_votes1 = np.zeros(num1, dtype=np.int32)
    best1_for_2 = np.full(num2, -1, dtype=np.int64)
    best_votes2 = np.zeros(num2, dtype=np.int32)
    second_votes2 = np.zeros(num2, dtype=np.int32)

    for first, second, vote_count in zip(pair_anchor1, pair_anchor2, votes):
        if vote_count > best_votes1[first]:
            second_votes1[first] = best_votes1[first]
            best_votes1[first] = vote_count
            best2_for_1[first] = second
        elif vote_count > second_votes1[first]:
            second_votes1[first] = vote_count

        if vote_count > best_votes2[second]:
            second_votes2[second] = best_votes2[second]
            best_votes2[second] = vote_count
            best1_for_2[second] = first
        elif vote_count > second_votes2[second]:
            second_votes2[second] = vote_count

    first_indices = np.flatnonzero(
        (best2_for_1 >= 0)
        & (best_votes1 >= min_votes)
        & ((best_votes1 - second_votes1) >= min_vote_margin))
    second_indices = best2_for_1[first_indices]
    accepted = (
        (best1_for_2[second_indices] == first_indices)
        & (best_votes2[second_indices] >= min_votes)
        & ((best_votes2[second_indices] - second_votes2[second_indices])
           >= min_vote_margin)
    )
    pair_idx = np.column_stack(
        (first_indices[accepted], second_indices[accepted])).astype(
            np.int32, copy=False)
    return pair_idx, len(unique_codes), best_votes1[first_indices[accepted]]


def asterism_anchor_votes_cpu_compiled(
    token_pairs1: NDArray[np.int64],
    token_pairs2: NDArray[np.int64],
    anchor_indices1: NDArray[np.int32],
    anchor_indices2: NDArray[np.int32],
    num1: int,
    num2: int,
    min_votes: int,
    min_vote_margin: int,
) -> AsterismVotes:
    module, _ = _load_compiled_module_result()
    if module is None or not hasattr(module, "asterism_anchor_votes_cpu"):
        raise RuntimeError("compiled custom op backend is unavailable")
    pairs1, pairs2, anchors1, anchors2 = _validate_asterism_vote_inputs(
        token_pairs1, token_pairs2, anchor_indices1, anchor_indices2)
    pair_idx, voted_pairs, accepted_votes = module.asterism_anchor_votes_cpu(
        pairs1, pairs2, anchors1, anchors2, int(num1), int(num2),
        int(min_votes), int(min_vote_margin))
    return pair_idx, int(voted_pairs), accepted_votes


def asterism_anchor_votes(
    token_pairs1: NDArray[np.int64],
    token_pairs2: NDArray[np.int64],
    anchor_indices1: NDArray[np.int32],
    anchor_indices2: NDArray[np.int32],
    num1: int,
    num2: int,
    min_votes: int,
    min_vote_margin: int,
) -> AsterismVotes:
    """Anchor-level voting; see :func:`asterism_anchor_votes_numpy`."""
    selection = _resolve_backend(
        "asterism_anchor_votes",
        _fallback_preference(),
        load_module=_load_compiled_module_result,
    )
    backend = (asterism_anchor_votes_cpu_compiled if selection.native
               else asterism_anchor_votes_numpy)
    return backend(token_pairs1, token_pairs2, anchor_indices1, anchor_indices2,
                   num1, num2, min_votes, min_vote_margin)
