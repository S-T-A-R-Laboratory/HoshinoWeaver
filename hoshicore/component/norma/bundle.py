"""Sequence bundle adjustment for a same-camera star-field capture.

The public result deliberately contains geometry only. Applying that geometry
to image streams belongs to downstream ops such as ``BundleReferenceRemapOp``.

The solve has four stages:

1. Match nearby frame pairs selected by ``pair_offsets``.
2. Keep the graph component connected to the reference frame and propagate
   pairwise rotations to obtain an initial pose for every connected frame.
3. Jointly optimize one shared camera model and one rotation per frame.
4. Reject high-residual edges, refit when necessary, then describe frames that
   were disconnected from the final graph.
"""
import dataclasses
import enum
import time
from collections import defaultdict, deque
from typing import Callable, Optional, Sequence

import cv2
import numpy as np
from loguru import logger
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.optimize._numdiff import approx_derivative
from scipy.spatial.transform import Rotation

from .detection import DetectedStars
from .frame_align import (DEFAULT_BOOTSTRAP_SCALES, AlignmentCameraCandidate,
                          solve_star_alignment)
from .optimization import CameraOptimizationPolicy
from .types import (BaseCameraModel, CameraModel, Distortion, FisheyeCameraModel,
                    FisheyeDistortion)


class BundleAdjustmentError(RuntimeError):
    """The requested sequence does not have a trustworthy global solution."""


class FrameAlignmentStatus(str, enum.Enum):
    SOLVED = "solved"
    EXCLUDED = "excluded"


@dataclasses.dataclass(frozen=True)
class BundleFrame:
    """Compact star observation retained while collecting a sequence."""

    index: int
    stars: DetectedStars
    candidate: AlignmentCameraCandidate


@dataclasses.dataclass(frozen=True)
class FrameAlignment:
    index: int
    status: FrameAlignmentStatus
    rotation_ref_to_src: Optional[NDArray[np.float64]]
    pose_source: str
    residual_p90_rad: Optional[float] = None
    incident_edge_count: int = 0
    reason: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class BAAlignmentPlan:
    """Immutable geometry hand-off from BA collection to stream integration."""

    reference_frame_index: int
    shared_camera: BaseCameraModel
    frames: tuple[FrameAlignment, ...]
    accepted_edge_count: int
    rejected_edge_count: int
    active_camera_parameter_count: int
    observability_condition: Optional[float]
    camera_solve_mode: str = "requested"
    camera_fallback_reason: Optional[str] = None

    def frame(self, index: int) -> FrameAlignment:
        if index < 0 or index >= len(self.frames):
            raise IndexError(f"frame index {index} is outside this plan")
        return self.frames[index]


@dataclasses.dataclass
class _BundleEdge:
    """Pairwise observations and an initial rotation from first to second."""

    first_index: int
    second_index: int
    first_pts: NDArray[np.float64]
    second_pts: NDArray[np.float64]
    initial_rotation: NDArray[np.float64]
    error: Optional[str] = None
    selected_scale: float = 1.0


_SCALE_PROBE_EDGE_COUNT = 3
_ROTATION_ONLY_POLICY = CameraOptimizationPolicy(False, False, False, 0)
_FOCAL_FALLBACK_POLICY = CameraOptimizationPolicy(True, False, False, 0)
_FOCAL_SCALE_DELTA_LIMIT = 0.3
_DISTORTION_ABS_LIMIT = 1.0
_DEFAULT_MAX_PAIRS_PER_EDGE = 128
_SPATIAL_RADIAL_BINS = 3
_SPATIAL_SECTORS = 8
_SPATIAL_BIN_RESERVE = 2


def _log_solve_summary(stage: str, fit, *, n_vars: int, n_edges: int,
                       elapsed_seconds: float) -> None:
    """Record least_squares convergence cost for scaling analysis across runs.

    ``nfev`` grows super-linearly with sequence length in practice (chained
    initial-rotation propagation and multiscale long-range edges disagree
    more as the sequence gets longer), so these numbers are the input to any
    future max_nfev sizing decision, not just a debugging aid.
    """
    logger.debug(
        f"{stage}: n_vars={n_vars} n_edges={n_edges} "
        f"nfev={getattr(fit, 'nfev', None)} njev={getattr(fit, 'njev', None)} "
        f"status={getattr(fit, 'status', None)} success={fit.success} "
        f"cost={getattr(fit, 'cost', float('nan')):.6e} "
        f"elapsed={elapsed_seconds:.3f}s")


def _camera_parameter_count(policy: CameraOptimizationPolicy) -> int:
    return ((1 if policy.optimize_focal else 0)
            + (policy.n_dist if policy.optimize_distortion else 0)
            + (2 if policy.optimize_principal_point else 0))


def _pack_camera_parameters(camera: BaseCameraModel,
                            policy: CameraOptimizationPolicy) -> NDArray[np.float64]:
    """Pack shared camera variables before the per-frame rotation variables.

    Focal length and principal point use relative offsets from the initial
    camera; distortion coefficients use their current model values directly.
    """
    parts: list[NDArray[np.float64]] = []
    if policy.optimize_focal:
        parts.append(np.zeros(1, dtype=np.float64))
    if policy.optimize_distortion:
        parts.append(np.asarray(camera.distortion.to_opt_params(policy.n_dist),
                                dtype=np.float64))
    if policy.optimize_principal_point:
        parts.append(np.zeros(2, dtype=np.float64))
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)


def _camera_parameter_bounds(
    policy: CameraOptimizationPolicy,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return physical bounds in the same order as packed camera parameters."""
    lower: list[float] = []
    upper: list[float] = []
    if policy.optimize_focal:
        lower.append(-_FOCAL_SCALE_DELTA_LIMIT)
        upper.append(_FOCAL_SCALE_DELTA_LIMIT)
    if policy.optimize_distortion:
        lower.extend([-_DISTORTION_ABS_LIMIT] * policy.n_dist)
        upper.extend([_DISTORTION_ABS_LIMIT] * policy.n_dist)
    if policy.optimize_principal_point:
        limit = float(policy.principal_point_offset_limit)
        if not 0.0 < limit <= 0.5:
            raise ValueError(
                "principal_point_offset_limit must be in (0, 0.5]")
        lower.extend([-limit] * 2)
        upper.extend([limit] * 2)
    return np.asarray(lower, dtype=np.float64), np.asarray(
        upper, dtype=np.float64)


def _camera_from_parameters(camera: BaseCameraModel,
                            policy: CameraOptimizationPolicy,
                            values: NDArray[np.float64]) -> BaseCameraModel:
    offset = 0
    intrinsics = camera.intrinsics
    if policy.optimize_focal:
        intrinsics = intrinsics.with_focal_length(
            intrinsics.focal_length_mm * (1.0 + float(values[offset])))
        offset += 1
    distortion = camera.distortion
    if policy.optimize_distortion:
        solved = np.asarray(values[offset:offset + policy.n_dist], dtype=np.float64)
        offset += policy.n_dist
        if isinstance(camera, FisheyeCameraModel):
            full = camera.distortion.to_cv2().copy()
            full[:len(solved)] = solved
            distortion = FisheyeDistortion.from_array(full)
        else:
            full = camera.distortion.to_cv2().copy()
            full[:len(solved)] = solved
            distortion = Distortion.from_cv2(full)
    if policy.optimize_principal_point:
        cx, cy = camera.intrinsics.principal_point_px
        intrinsics = intrinsics.with_principal_point(
            cx + float(values[offset]) * intrinsics.image_width_px,
            cy + float(values[offset + 1]) * intrinsics.image_height_px)
    return camera.with_intrinsics(intrinsics).with_distortion(distortion)


def _rotation_from_rvec(rvec: NDArray[np.float64]) -> NDArray[np.float64]:
    return cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))[0]


def _rvec_from_rotation(rotation: NDArray[np.float64]) -> NDArray[np.float64]:
    return cv2.Rodrigues(np.asarray(rotation, dtype=np.float64))[0].reshape(3)


def _rotation_derivative(rvec: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return ``dR/drvec`` indexed as ``[row, column, rvec_component]``.

    ``cv2.Rodrigues`` reports the derivative as a ``(3, 9)`` block whose
    transpose is the row-major flattened rotation differentiated by each
    rotation-vector component.
    """
    _, jacobian = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return jacobian.T.reshape(3, 3, 3)


def _skew_rows(vectors: NDArray[np.float64]) -> NDArray[np.float64]:
    """Stack one skew-symmetric matrix per input row."""
    stacked = np.zeros((len(vectors), 3, 3), dtype=np.float64)
    stacked[:, 0, 1] = -vectors[:, 2]
    stacked[:, 0, 2] = vectors[:, 1]
    stacked[:, 1, 0] = vectors[:, 2]
    stacked[:, 1, 2] = -vectors[:, 0]
    stacked[:, 2, 0] = -vectors[:, 1]
    stacked[:, 2, 1] = vectors[:, 0]
    return stacked


def _edge_rotation_jacobians(
    first_rays: NDArray[np.float64],
    second_rays: NDArray[np.float64],
    first_rvec: Optional[NDArray[np.float64]],
    second_rvec: Optional[NDArray[np.float64]],
    first_rotation: NDArray[np.float64],
    second_rotation: NDArray[np.float64],
) -> tuple[Optional[NDArray[np.float64]], Optional[NDArray[np.float64]]]:
    """Differentiate one edge's cross-product residual by its two rotations.

    The residual for a matched pair is ``(R_second R_first^T f) x s``, so the
    outer derivative is ``-skew(s)`` and the inner derivatives follow from the
    Rodrigues derivative of each rotation. Passing ``None`` for a rotation
    vector marks the gauge-fixed reference frame, which owns no variables.
    """
    outer = -_skew_rows(second_rays)
    first_jacobian = None
    second_jacobian = None
    if second_rvec is not None:
        # ``predicted`` is linear in the second rotation once the first frame
        # has mapped its rays into the shared reference system.
        reference_rays = first_rays @ first_rotation
        derivative = _rotation_derivative(second_rvec)
        inner = np.einsum('abl,pb->pal', derivative, reference_rays)
        second_jacobian = np.einsum('pab,pbl->pal', outer, inner)
    if first_rvec is not None:
        derivative = _rotation_derivative(first_rvec)
        inner = np.einsum('bal,pb->pal', derivative, first_rays)
        inner = np.einsum('ab,pbl->pal', second_rotation, inner)
        first_jacobian = np.einsum('pab,pbl->pal', outer, inner)
    return first_jacobian, second_jacobian


def _readonly_rotation(rotation: Optional[NDArray[np.float64]]) -> Optional[NDArray[np.float64]]:
    if rotation is None:
        return None
    result = np.asarray(rotation, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


def _connected_component(edges: Sequence[_BundleEdge], root: int) -> set[int]:
    neighbours: dict[int, set[int]] = defaultdict(set)
    for edge in edges:
        neighbours[edge.first_index].add(edge.second_index)
        neighbours[edge.second_index].add(edge.first_index)
    seen = {root}
    pending = deque([root])
    while pending:
        current = pending.popleft()
        for candidate in neighbours[current]:
            if candidate not in seen:
                seen.add(candidate)
                pending.append(candidate)
    return seen


def _initial_rotations(
    edges: Sequence[_BundleEdge], fixed_rotations: dict[int, NDArray[np.float64]],
    component: set[int],
) -> dict[int, NDArray[np.float64]]:
    """Propagate pairwise rotations over a BFS forest seeded at every fixed frame.

    The returned rotation for frame ``i`` maps reference-frame rays to frame
    ``i`` rays. All graph edges, not only this implicit forest, are used later
    by the global optimizer. Seeding the BFS from every fixed frame (rather
    than a single root) lets a hierarchical solve fix both boundary poses of a
    segment instead of only one reference frame.
    """
    graph: dict[int, list[tuple[int, NDArray[np.float64]]]] = defaultdict(list)
    for edge in edges:
        graph[edge.first_index].append((edge.second_index, edge.initial_rotation))
        graph[edge.second_index].append((edge.first_index, edge.initial_rotation.T))
    rotations = dict(fixed_rotations)
    pending = deque(fixed_rotations.keys())
    while pending:
        first = pending.popleft()
        for second, first_to_second in graph[first]:
            if second in rotations or second not in component:
                continue
            rotations[second] = first_to_second @ rotations[first]
            pending.append(second)
    if rotations.keys() != component:
        raise BundleAdjustmentError("reference component has incomplete pose initialization")
    return rotations


def _edge_residuals(edge: _BundleEdge, rotations: dict[int, NDArray[np.float64]],
                    camera: BaseCameraModel) -> NDArray[np.float64]:
    first = camera.unproject(edge.first_pts)
    second = camera.unproject(edge.second_pts)
    predicted = (rotations[edge.second_index] @ rotations[edge.first_index].T
                 @ first.T).T
    dots = np.sum(predicted * second, axis=1)
    return np.arccos(np.clip(dots, -1.0, 1.0))


def _coverage_precheck(edges: Sequence[_BundleEdge], policy: CameraOptimizationPolicy,
                       camera: BaseCameraModel) -> None:
    if not policy.optimize_distortion:
        return
    all_points = np.concatenate([edge.first_pts for edge in edges])
    h, w = camera.intrinsics.image_height_px, camera.intrinsics.image_width_px
    center = np.array([w / 2.0, h / 2.0])
    radial = np.linalg.norm(all_points - center, axis=1)
    outer = np.count_nonzero(radial >= 0.66 * np.hypot(w / 2.0, h / 2.0))
    if len(all_points) < 20 or outer < max(4, int(0.08 * len(all_points))):
        raise BundleAdjustmentError(
            "shared distortion is not observable: insufficient outer-field coverage")


def _camera_observability(jacobian: NDArray[np.float64], camera_width: int) -> Optional[float]:
    """Check shared-camera observability after removing pose directions."""
    if camera_width == 0:
        return None
    camera_j = np.asarray(jacobian[:, :camera_width], dtype=np.float64)
    pose_j = np.asarray(jacobian[:, camera_width:], dtype=np.float64)
    if pose_j.size:
        q_mat, _ = np.linalg.qr(pose_j, mode="reduced")
        camera_j = camera_j - q_mat @ (q_mat.T @ camera_j)
    singular = np.linalg.svd(camera_j, compute_uv=False)
    if len(singular) < camera_width or singular[-1] <= 1e-10:
        raise BundleAdjustmentError("shared camera parameters are rank deficient")
    condition = float(singular[0] / singular[-1])
    if not np.isfinite(condition) or condition > 1e8:
        raise BundleAdjustmentError(
            f"shared camera parameters are ill-conditioned ({condition:.3e})")
    return condition


def _make_edge(first: BundleFrame, second: BundleFrame,
               random_seed: int | None,
               bootstrap_scales: Sequence[float] = DEFAULT_BOOTSTRAP_SCALES,
               ) -> _BundleEdge:
    """Match one edge and refine only its initial relative rotation."""
    try:
        ref_candidate = dataclasses.replace(
            first.candidate, optimization_policy=_ROTATION_ONLY_POLICY)
        src_candidate = dataclasses.replace(
            second.candidate, optimization_policy=_ROTATION_ONLY_POLICY)
        alignment, match = solve_star_alignment(
            first.stars,
            second.stars,
            ref_candidate,
            src_candidate,
            bootstrap_scales=tuple(bootstrap_scales),
            same_camera=True,
            use_asterism_bootstrap=True,
            random_seed=random_seed,
            residual_space="cross",
        )
        if len(match.pair_idx) < 6:
            raise ValueError(f"only {len(match.pair_idx)} matched stars")
        base_focal = first.candidate.camera.intrinsics.focal_length_mm
        selected_scale = alignment.ref_camera.intrinsics.focal_length_mm / base_focal
        return _BundleEdge(
            first.index, second.index, match.ref_pts, match.src_pts,
            np.asarray(alignment.rotation_ref_to_src, dtype=np.float64),
            selected_scale=float(selected_scale))
    except Exception as exc:
        return _BundleEdge(first.index, second.index,
                           np.empty((0, 2), dtype=np.float64),
                           np.empty((0, 2), dtype=np.float64),
                           np.eye(3), str(exc))


def _spatial_pair_bins(points: NDArray[np.float64], width: int,
                       height: int) -> NDArray[np.int32]:
    """Assign reference points to 3 radial by 8 angular image bins."""
    center = np.array([width / 2.0, height / 2.0])
    scale = np.array([width / 2.0, height / 2.0])
    normalized = (points - center) / scale
    radius = np.linalg.norm(normalized, axis=1) / np.sqrt(2.0)
    radial = np.minimum(
        (radius * _SPATIAL_RADIAL_BINS).astype(np.int32),
        _SPATIAL_RADIAL_BINS - 1)
    angle = (np.arctan2(normalized[:, 1], normalized[:, 0])
             + 2.0 * np.pi) % (2.0 * np.pi)
    sector = np.minimum(
        (angle * _SPATIAL_SECTORS / (2.0 * np.pi)).astype(np.int32),
        _SPATIAL_SECTORS - 1)
    return radial * _SPATIAL_SECTORS + sector


def _sample_edge_pairs(edge: _BundleEdge, max_pairs: int | None,
                       width: int, height: int,
                       random_seed: int | None) -> _BundleEdge:
    """Keep sparse-bin coverage, then sample the remainder without reweighting.

    At most two points are reserved from every non-empty spatial bin. Remaining
    slots are drawn uniformly from all other inliers, so the sample mostly
    preserves the full match distribution instead of equalizing every bin.
    """
    if max_pairs is None or max_pairs <= 0 or len(edge.first_pts) <= max_pairs:
        return edge
    rng = np.random.default_rng(random_seed)
    bins = _spatial_pair_bins(edge.first_pts, width, height)
    bin_order = np.unique(bins)
    rng.shuffle(bin_order)
    groups: dict[int, NDArray[np.intp]] = {}
    for bin_index in bin_order:
        indices = np.flatnonzero(bins == bin_index)
        rng.shuffle(indices)
        groups[int(bin_index)] = indices

    reserved: list[int] = []
    for reserve_round in range(_SPATIAL_BIN_RESERVE):
        for bin_index in bin_order:
            group = groups[int(bin_index)]
            if reserve_round < len(group):
                reserved.append(int(group[reserve_round]))

    selected_mask = np.zeros(len(edge.first_pts), dtype=bool)
    selected_mask[reserved] = True
    remaining = np.flatnonzero(~selected_mask)
    rng.shuffle(remaining)
    order = np.concatenate((
        np.asarray(reserved, dtype=np.intp), remaining.astype(np.intp)))
    chosen = order[:max_pairs]
    return dataclasses.replace(
        edge, first_pts=edge.first_pts[chosen],
        second_pts=edge.second_pts[chosen])


def _ordered_scales(votes: dict[float, int]) -> tuple[float, ...]:
    """Try the most reliable sequence scale first; prefer 1.0 on ties."""
    original_order = {scale: index for index, scale in enumerate(votes)}
    return tuple(sorted(
        votes,
        key=lambda scale: (-votes[scale], abs(scale - 1.0),
                           original_order[scale]),
    ))


def _vote_for_scale(votes: dict[float, int], selected: float) -> None:
    nearest = min(votes, key=lambda scale: abs(scale - selected))
    votes[nearest] += 1


def _sequence_scales(frames: Sequence[BundleFrame]) -> tuple[float, ...]:
    """Keep reliable perspective EXIF/provided sequences at scale 1.0."""
    reliable_perspective = all(
        not isinstance(frame.candidate.camera, FisheyeCameraModel)
        and frame.candidate.init_source in ("exif", "provided")
        for frame in frames)
    return ((1.0,) if reliable_perspective else
            tuple(float(scale) for scale in DEFAULT_BOOTSTRAP_SCALES))


def _dense_edge_pairs(
    frame_count: int, offsets: Sequence[int],
) -> list[tuple[int, int]]:
    """Every ``(pos, pos + offset)`` in range, for each offset, position-major."""
    pairs = []
    for pos in range(frame_count):
        for offset in offsets:
            if pos + offset < frame_count:
                pairs.append((pos, pos + offset))
    return pairs


def _multiscale_edge_pairs(
    frame_count: int,
    max_offset: int | None = None,
    min_degree: int = 4,
) -> list[tuple[int, int]]:
    """Sparse multi-scale topology: ~2N edges, interleaved dyadic starts.

    Offset 1 forms a full chain. Each larger offset ``2**m`` contributes a
    disjoint set of edges starting at ``2**(m - 1) - 1`` and spaced by
    ``2**(m + 1)`` (i.e. non-overlapping pairs at that scale), so every scale
    adds roughly ``frame_count / offset`` edges instead of one per position.
    This keeps the total near ``2 * frame_count`` while covering long-range
    connections that a purely dense ``{1, 2, 4}`` topology would need far more
    edges to reach. Sequence lengths that are not a clean power of two starve
    a few nodes near the tail of long-range coverage; a greedy top-up with
    offsets 2, then 4, then 1 (in that order of preference, cheapest first)
    brings every node besides the two sequence endpoints back up to
    ``min_degree``. The endpoints are structurally short by one edge since
    they only extend in a single direction.
    """
    if frame_count < 2:
        return []
    pairs: set[tuple[int, int]] = set()
    for i in range(frame_count - 1):
        pairs.add((i, i + 1))
    scale = 2
    while True:
        offset = 2 ** scale
        if offset >= frame_count or (max_offset is not None and offset > max_offset):
            break
        start = 2 ** (scale - 1) - 1
        for i in range(start, frame_count - offset, offset):
            pairs.add((i, i + offset))
        scale += 1

    degree = [0] * frame_count
    for a, b in pairs:
        degree[a] += 1
        degree[b] += 1
    for fill_offset, require_both in ((2, True), (4, False), (1, False)):
        for i in range(0, frame_count - fill_offset):
            j = i + fill_offset
            if (i, j) in pairs:
                continue
            below = (degree[i] < min_degree and degree[j] < min_degree) if require_both \
                else (degree[i] < min_degree or degree[j] < min_degree)
            if below:
                pairs.add((i, j))
                degree[i] += 1
                degree[j] += 1
    return sorted(pairs)


def _build_edges_from_pairs(
    frames: Sequence[BundleFrame],
    pairs: Sequence[tuple[int, int]],
    random_seed: int | None,
    edge_completed: Callable[[], None] | None = None,
    max_pairs_per_edge: int | None = _DEFAULT_MAX_PAIRS_PER_EDGE,
) -> tuple[list[_BundleEdge], float]:
    """Build edges for explicit ``(pos, pos)`` position pairs into ``frames``.

    A successful single-scale solve already passed the matcher's rotation
    quality checks. It is accepted without a vote because the other scales
    were not compared. Failed preferred-scale solves evaluate every remaining
    scale and vote for the best successful fallback.
    """
    votes = {scale: 0 for scale in _sequence_scales(frames)}
    successful_probes = 0
    edges = []
    for pos, second_pos in pairs:
        first = frames[pos]
        second = frames[second_pos]
        ordered_scales = _ordered_scales(votes)
        if successful_probes < _SCALE_PROBE_EDGE_COUNT:
            edge = _make_edge(first, second, random_seed, ordered_scales)
            if edge.error is None:
                _vote_for_scale(votes, edge.selected_scale)
                successful_probes += 1
        else:
            edge = _make_edge(first, second, random_seed,
                              ordered_scales[:1])
            if edge.error is not None and len(ordered_scales) > 1:
                edge = _make_edge(first, second, random_seed,
                                  ordered_scales[1:])
                if edge.error is None:
                    _vote_for_scale(votes, edge.selected_scale)
        if edge.error is None:
            camera = first.candidate.camera
            sample_seed = (None if random_seed is None else
                           (int(random_seed) + first.index * 1009
                            + second.index * 9176) % (2 ** 32))
            edge = _sample_edge_pairs(
                edge, max_pairs_per_edge,
                camera.intrinsics.image_width_px,
                camera.intrinsics.image_height_px,
                sample_seed)
        edges.append(edge)
        if edge_completed is not None:
            edge_completed()
    return edges, _ordered_scales(votes)[0]


_EDGE_TOPOLOGIES = ("dense", "multiscale")


def estimate_edge_count(
    frame_count: int,
    offsets: Sequence[int],
    edge_topology: str = "dense",
    max_pair_offset: int | None = None,
) -> int:
    """Count edges a solve would build, without matching or allocating them."""
    return len(_select_edge_pairs(
        frame_count, offsets, edge_topology, max_pair_offset))


def _select_edge_pairs(
    frame_count: int,
    offsets: Sequence[int],
    topology: str,
    max_pair_offset: int | None,
) -> list[tuple[int, int]]:
    if topology == "dense":
        return _dense_edge_pairs(frame_count, offsets)
    if topology == "multiscale":
        return _multiscale_edge_pairs(frame_count, max_pair_offset)
    raise BundleAdjustmentError(f"unknown edge topology: {topology!r}")


def _build_edges(
    frames: Sequence[BundleFrame],
    offsets: Sequence[int],
    random_seed: int | None,
    edge_completed: Callable[[], None] | None = None,
    max_pairs_per_edge: int | None = _DEFAULT_MAX_PAIRS_PER_EDGE,
    edge_topology: str = "dense",
    max_pair_offset: int | None = None,
) -> tuple[list[_BundleEdge], float]:
    """Build edges over either the dense or sparse multi-scale topology.

    ``edge_topology="dense"`` (the default, unchanged behavior) builds every
    ``(pos, pos + offset)`` pair for each offset. ``edge_topology="multiscale"``
    instead builds the sparse interleaved-start topology from
    ``_multiscale_edge_pairs``, which keeps the total near ``2 * len(frames)``
    edges regardless of how many dyadic scales it spans, optionally capped at
    ``max_pair_offset``.
    """
    pairs = _select_edge_pairs(
        len(frames), offsets, edge_topology, max_pair_offset)
    return _build_edges_from_pairs(
        frames, pairs, random_seed, edge_completed, max_pairs_per_edge)


def _solve_bundle_parameters(
    edges: Sequence[_BundleEdge],
    reference_frame_index: int,
    component: set[int],
    camera: BaseCameraModel,
    policy: CameraOptimizationPolicy,
    max_nfev: int,
) -> tuple[BaseCameraModel, dict[int, NDArray[np.float64]],
           list[_BundleEdge], Optional[float]]:
    """Jointly solve one bounded camera model and the connected frame poses."""
    _coverage_precheck(edges, policy, camera)
    initial = _initial_rotations(
        edges, {reference_frame_index: np.eye(3, dtype=np.float64)}, component)
    variable_indices = [
        index for index in sorted(component) if index != reference_frame_index]
    camera0 = _pack_camera_parameters(camera, policy)
    camera_lower, camera_upper = _camera_parameter_bounds(policy)
    if (np.any(camera0 < camera_lower)
            or np.any(camera0 > camera_upper)):
        raise BundleAdjustmentError(
            "initial shared camera parameters are outside optimization bounds")

    def pack_with_poses(
        camera_values: NDArray[np.float64],
        pose_values: dict[int, NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        return np.concatenate((
            camera_values,
            *[_rvec_from_rotation(pose_values[index])
              for index in variable_indices],
        ))

    def parameter_bounds() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        pose_width = 3 * len(variable_indices)
        return (
            np.concatenate((camera_lower, np.full(pose_width, -np.inf))),
            np.concatenate((camera_upper, np.full(pose_width, np.inf))),
        )

    # Optimizer layout: [shared camera parameters, 3D rotation vectors...].
    # The reference rotation is fixed to identity to remove gauge freedom.
    x0 = pack_with_poses(camera0, initial)

    def unpack(values):
        solved_camera = _camera_from_parameters(
            camera, policy, values[:len(camera0)])
        rotations = {reference_frame_index: np.eye(3, dtype=np.float64)}
        offset = len(camera0)
        for index in variable_indices:
            rotations[index] = _rotation_from_rvec(
                values[offset:offset + 3])
            offset += 3
        return solved_camera, rotations

    def residual(values, selected_edges):
        solved_camera, rotations = unpack(values)
        values_out = []
        for edge in selected_edges:
            first = solved_camera.unproject(edge.first_pts)
            second = solved_camera.unproject(edge.second_pts)
            # Convert the first-frame ray through the shared reference system
            # into the second frame, then compare the two unit directions.
            predicted = (
                rotations[edge.second_index]
                @ rotations[edge.first_index].T
                @ first.T
            ).T
            values_out.append(np.cross(predicted, second).reshape(-1))
        return np.concatenate(values_out)

    def validate_fit(fit, stage: str) -> None:
        if not fit.success or not np.all(np.isfinite(fit.x)):
            raise BundleAdjustmentError(
                f"{stage} bundle optimization failed: {fit.message}")
        if np.any(np.asarray(fit.active_mask[:len(camera0)]) != 0):
            raise BundleAdjustmentError(
                f"{stage} shared camera parameters reached optimization bounds")

    def jacobian(values, selected_edges):
        """Differentiate the residual analytically in the rotation variables.

        Finite differencing this residual costs one full residual evaluation
        per variable, so a long sequence spends most of its time rebuilding
        unchanged edge geometry. The rotation columns are cheap to derive in
        closed form; the few shared-camera columns keep using finite
        differences because they pass through the lens model.
        """
        solved_camera, rotations = unpack(values)
        camera_width = len(camera0)
        pose_column = {index: camera_width + 3 * position
                       for position, index in enumerate(variable_indices)}
        total_rows = sum(len(edge.first_pts) * 3 for edge in selected_edges)
        matrix = np.zeros((total_rows, len(values)), dtype=np.float64)

        if camera_width:
            def camera_residual(camera_values):
                merged = np.concatenate((camera_values, values[camera_width:]))
                return residual(merged, selected_edges)

            matrix[:, :camera_width] = approx_derivative(
                camera_residual, values[:camera_width], method="2-point")

        row_offset = 0
        for edge in selected_edges:
            rows = len(edge.first_pts) * 3
            first_rays = solved_camera.unproject(edge.first_pts)
            second_rays = solved_camera.unproject(edge.second_pts)
            first_rvec = (
                None if edge.first_index == reference_frame_index else
                values[pose_column[edge.first_index]:
                       pose_column[edge.first_index] + 3])
            second_rvec = (
                None if edge.second_index == reference_frame_index else
                values[pose_column[edge.second_index]:
                       pose_column[edge.second_index] + 3])
            first_block, second_block = _edge_rotation_jacobians(
                first_rays, second_rays, first_rvec, second_rvec,
                rotations[edge.first_index], rotations[edge.second_index])
            for index, block in ((edge.first_index, first_block),
                                 (edge.second_index, second_block)):
                if block is None:
                    continue
                column = pose_column[index]
                matrix[row_offset:row_offset + rows,
                       column:column + 3] = block.reshape(rows, 3)
            row_offset += rows
        return matrix

    # First pass uses every successfully matched edge in the component. Huber
    # loss limits individual bad pairs before edge-level rejection is possible.
    initial_residual = residual(x0, edges)
    scale = max(float(np.median(np.abs(initial_residual))) * 2.0, 1e-6)
    started = time.perf_counter()
    fit = least_squares(
        residual, x0, jac=jacobian, args=(edges,), method="trf", loss="huber",
        f_scale=scale, max_nfev=max_nfev, bounds=parameter_bounds())
    _log_solve_summary("_solve_bundle_parameters[global]", fit,
                       n_vars=len(x0), n_edges=len(edges),
                       elapsed_seconds=time.perf_counter() - started)
    validate_fit(fit, "global")
    solved_camera, rotations = unpack(fit.x)

    # Reject complete edges only when their robust edge statistic is an obvious
    # outlier. This prevents one structured correspondence field from steering
    # the shared camera model.
    edge_p90 = np.array([
        np.percentile(_edge_residuals(edge, rotations, solved_camera), 90)
        for edge in edges
    ])
    median = float(np.median(edge_p90))
    mad = float(np.median(np.abs(edge_p90 - median))) * 1.4826
    cutoff = max(median + 3.0 * mad, np.deg2rad(0.15))
    retained = [edge for edge, p90 in zip(edges, edge_p90)
                if p90 <= cutoff]
    if len(retained) < len(edges):
        # Removing one edge may detach frames. Refit all surviving poses and
        # the same bounded shared camera parameters on the remaining component.
        component = _connected_component(retained, reference_frame_index)
        retained = [
            edge for edge in retained
            if edge.first_index in component and edge.second_index in component]
        if len(component) < 2:
            raise BundleAdjustmentError(
                "outlier rejection disconnected the reference frame")
        variable_indices = [
            index for index in sorted(component)
            if index != reference_frame_index]
        initial = {index: rotations[index] for index in component}
        x0 = pack_with_poses(fit.x[:len(camera0)], initial)
        started = time.perf_counter()
        fit = least_squares(
            residual, x0, jac=jacobian, args=(retained,), method="trf",
            loss="huber", f_scale=scale, max_nfev=max_nfev,
            bounds=parameter_bounds())
        _log_solve_summary("_solve_bundle_parameters[refined]", fit,
                           n_vars=len(x0), n_edges=len(retained),
                           elapsed_seconds=time.perf_counter() - started)
        validate_fit(fit, "refined")
        solved_camera, rotations = unpack(fit.x)

    _coverage_precheck(retained, policy, solved_camera)
    condition = _camera_observability(fit.jac, len(camera0))
    return solved_camera, rotations, retained, condition


def _sample_camera_frames(
    ordered: Sequence[BundleFrame],
    sample_count: int,
    reference_frame_index: int,
) -> list[BundleFrame]:
    """Pick an evenly spaced subset of frames for the camera calibration pass.

    Sequence endpoints are always included so that long-range rotation drift is
    observable; the reference frame is included so the sampled graph keeps the
    same gauge. Interior frames are drawn at uniform positions, which spreads
    the shared camera constraints across the whole sequence.
    """
    if sample_count is None or len(ordered) <= sample_count:
        return list(ordered)
    positions = {
        frame.index: position for position, frame in enumerate(ordered)}
    target = max(3, int(sample_count))
    picked = np.unique(np.linspace(
        0, len(ordered) - 1, target, dtype=np.int64))
    picked = set(int(position) for position in picked)
    picked.add(positions[reference_frame_index])
    picked.add(0)
    picked.add(len(ordered) - 1)
    return [ordered[position] for position in sorted(picked)]


def _remap_frames(
    frames: Sequence[BundleFrame],
) -> tuple[list[BundleFrame], dict[int, int]]:
    """Return a copy with contiguous 0..K-1 indices and the original mapping."""
    remapped: list[BundleFrame] = []
    original_index: dict[int, int] = {}
    for local_index, frame in enumerate(sorted(
            frames, key=lambda item: item.index)):
        original_index[local_index] = frame.index
        remapped.append(dataclasses.replace(frame, index=local_index))
    return remapped, original_index


def _replace_camera(
    frames: Sequence[BundleFrame],
    camera: BaseCameraModel,
) -> list[BundleFrame]:
    """Point every frame's candidate at one shared camera model."""
    updated: list[BundleFrame] = []
    for frame in frames:
        candidate = dataclasses.replace(
            frame.candidate, camera=camera, init_source="staged")
        updated.append(dataclasses.replace(frame, candidate=candidate))
    return updated


def _solve_single_stage(
    frames: Sequence[BundleFrame],
    reference_frame_index: int,
    offsets: Sequence[int],
    random_seed: int | None,
    max_nfev: int,
    max_pairs_per_edge: int | None,
    edge_completed: Callable[[], None] | None,
    edge_topology: str = "dense",
    max_pair_offset: int | None = None,
) -> BAAlignmentPlan:
    """Jointly solve one shared camera and every frame pose (the original path).

    The whole sequence is bundled together: edge matching, scale voting, and a
    single bounded camera+rotation solve. This is the reference implementation
    and also the path for sequences short enough that the joint solve is cheap.
    """
    by_index = {frame.index: frame for frame in frames}
    reference = by_index[reference_frame_index]
    camera = reference.candidate.camera
    policy = reference.candidate.optimization_policy
    if any(type(frame.candidate.camera) is not type(camera) for frame in frames):
        raise BundleAdjustmentError("sequence BA requires one camera projection family")
    if any(frame.candidate.camera.intrinsics.image_width_px != camera.intrinsics.image_width_px
           or frame.candidate.camera.intrinsics.image_height_px != camera.intrinsics.image_height_px
           for frame in frames):
        raise BundleAdjustmentError("sequence BA requires a single image geometry")

    ordered = sorted(frames, key=lambda item: item.index)
    edges, sequence_scale = _build_edges(
        ordered, offsets, random_seed, edge_completed,
        max_pairs_per_edge=max_pairs_per_edge,
        edge_topology=edge_topology, max_pair_offset=max_pair_offset)
    # Pair-scale voting estimates the projection scale needed to establish
    # reliable rotations. Use the same consensus as the shared-camera BA
    # starting point instead of discarding it after edge construction.
    camera = camera.with_focal_length(
        camera.intrinsics.focal_length_mm * sequence_scale)
    accepted = [edge for edge in edges if edge.error is None]

    # Only the reference-connected component has a defined reference-relative
    # pose. Failed or disconnected frames are handled after the global solve.
    component = _connected_component(accepted, reference_frame_index)
    if len(component) < 2:
        raise BundleAdjustmentError("reference frame has no reliable sequence connection")
    accepted = [edge for edge in accepted if edge.first_index in component and edge.second_index in component]
    solve_mode = "requested"
    fallback_reason = None
    solved_policy = policy
    try:
        solved_camera, rotations, retained, condition = (
            _solve_bundle_parameters(
                accepted, reference_frame_index, component, camera, policy,
                max_nfev))
    except BundleAdjustmentError as exc:
        can_fallback = (
            policy.optimize_distortion or policy.optimize_principal_point)
        if not can_fallback:
            raise
        fallback_reason = str(exc)
        solve_mode = "focal_fallback"
        solved_policy = _FOCAL_FALLBACK_POLICY
        solved_camera, rotations, retained, condition = (
            _solve_bundle_parameters(
                accepted, reference_frame_index, component, camera,
                solved_policy, max_nfev))
    return _assemble_plan(
        by_index, reference_frame_index, edges, retained, solved_camera,
        rotations, solved_policy, condition, solve_mode, fallback_reason)


def _assemble_plan(
    by_index: dict[int, BundleFrame],
    reference_frame_index: int,
    edges: Sequence[_BundleEdge],
    retained: Sequence[_BundleEdge],
    solved_camera: BaseCameraModel,
    rotations: dict[int, NDArray[np.float64]],
    solved_policy: CameraOptimizationPolicy,
    condition: Optional[float],
    solve_mode: str,
    fallback_reason: Optional[str],
) -> BAAlignmentPlan:
    """Wrap solver output into the immutable plan handed to stream integration."""
    edge_by_frame: dict[int, list[float]] = defaultdict(list)
    for edge in retained:
        p90 = float(np.percentile(_edge_residuals(edge, rotations, solved_camera), 90))
        edge_by_frame[edge.first_index].append(p90)
        edge_by_frame[edge.second_index].append(p90)
    frame_results = []
    for index in sorted(by_index):
        solved = index in rotations
        pose = rotations.get(index)
        residual_p90 = (float(np.median(edge_by_frame[index]))
                        if edge_by_frame[index] else None)
        frame_results.append(FrameAlignment(
            index=index,
            status=(FrameAlignmentStatus.SOLVED if solved
                    else FrameAlignmentStatus.EXCLUDED),
            rotation_ref_to_src=_readonly_rotation(pose),
            pose_source="bundle" if solved else "none",
            residual_p90_rad=residual_p90,
            incident_edge_count=len(edge_by_frame[index]),
            reason=(None if solved else
                    "frame is outside the reference-connected solve graph")))
    retained_keys = {(edge.first_index, edge.second_index) for edge in retained}
    accepted_edge_count = len(retained_keys)
    rejected_edge_count = len(edges) - accepted_edge_count
    return BAAlignmentPlan(
        reference_frame_index=reference_frame_index, shared_camera=solved_camera,
        frames=tuple(frame_results), accepted_edge_count=accepted_edge_count,
        rejected_edge_count=rejected_edge_count,
        active_camera_parameter_count=_camera_parameter_count(solved_policy),
        observability_condition=condition,
        camera_solve_mode=solve_mode,
        camera_fallback_reason=fallback_reason)


def solve_anchor_camera_and_rotations(
    frames: Sequence[BundleFrame],
    reference_frame_index: int,
    *,
    sample_count: int,
    pair_offsets: Sequence[int] = (1, 2, 4),
    random_seed: int | None = 0,
    max_nfev: int = 300,
    max_pairs_per_edge: int | None = _DEFAULT_MAX_PAIRS_PER_EDGE,
    edge_topology: str = "dense",
    max_pair_offset: int | None = None,
) -> tuple[BaseCameraModel, dict[int, NDArray[np.float64]]]:
    """Stage A: calibrate the shared lens model on an evenly spaced subset,
    keeping the subset's solved poses as anchors for the segmented Stage B
    rotation solve.

    The subset always contains the sequence endpoints and the reference
    frame, so the returned camera observes the whole rotation range. The
    hierarchical rotation solve fixes each segment's boundary poses to these
    anchors, so the anchor subset must use the same edge topology as the
    full sequence: a long-range edge at this small anchor-subset scale is a
    direct observation between two anchors, not merely a low-hop path
    threaded through intermediate ones, and that independence is what keeps
    drift in check over long sequences.
    """
    ordered = sorted(frames, key=lambda item: item.index)
    sampled = _sample_camera_frames(ordered, sample_count, reference_frame_index)
    remapped, original_index = _remap_frames(sampled)
    local_reference = next(
        local for local, original in original_index.items()
        if original == reference_frame_index)
    plan = build_bundle_plan(
        remapped, local_reference, pair_offsets=pair_offsets,
        random_seed=random_seed, max_nfev=max_nfev,
        max_pairs_per_edge=max_pairs_per_edge,
        edge_topology=edge_topology, max_pair_offset=max_pair_offset)
    anchor_rotations = {
        original_index[local]: entry.rotation_ref_to_src
        for local, entry in enumerate(plan.frames)
        if entry.status == FrameAlignmentStatus.SOLVED
        and entry.rotation_ref_to_src is not None
    }
    if reference_frame_index not in anchor_rotations:
        raise BundleAdjustmentError(
            "anchor subset solve excluded the reference frame")
    return plan.shared_camera, anchor_rotations


def solve_rotations_fixed_camera(
    frames: Sequence[BundleFrame],
    reference_frame_index: int,
    camera: BaseCameraModel,
    *,
    anchor_rotations: dict[int, NDArray[np.float64]] | None = None,
    pair_offsets: Sequence[int] = (1, 2, 4),
    random_seed: int | None = 0,
    max_nfev: int = 300,
    max_pairs_per_edge: int | None = _DEFAULT_MAX_PAIRS_PER_EDGE,
    edge_completed: Callable[[], None] | None = None,
    edge_topology: str = "dense",
    max_pair_offset: int | None = None,
) -> tuple[dict[int, NDArray[np.float64]], Sequence[_BundleEdge],
           Sequence[_BundleEdge]]:
    """Stage B: solve only the per-frame rotations with the camera held fixed.

    The shared camera is an input rather than an optimize-able variable, so the
    residual still uses the cross-product of the two projected rays but the
    camera columns are constant. This is meaningfully cheaper than the joint
    solve for long sequences because the rotation block is much smaller and the
    noisy shared-camera columns no longer couple all edges together.

    ``anchor_rotations``, when given (at least two entries, already solved
    against this same fixed ``camera``), splits the sequence into independent
    segments between consecutive anchors instead of solving every pose in one
    least-squares problem. Each segment's boundary poses are gauge-fixed to
    the anchor solve's result, so segments do not depend on each other and
    stay at the variable-count scale validated on real sequences (tens to
    ~100 frames) regardless of total sequence length. Without anchors (or
    with fewer than two), this falls back to the original single-solve path
    gauge-fixed at ``reference_frame_index``.
    """
    ordered = sorted(frames, key=lambda item: item.index)
    if any(type(frame.candidate.camera) is not type(camera) for frame in frames):
        raise BundleAdjustmentError("sequence BA requires one camera projection family")
    edges, _ = _build_edges(
        ordered, tuple(sorted({int(v) for v in pair_offsets if int(v) > 0})),
        random_seed, edge_completed, max_pairs_per_edge=max_pairs_per_edge,
        edge_topology=edge_topology, max_pair_offset=max_pair_offset)
    accepted = [edge for edge in edges if edge.error is None]
    component = _connected_component(accepted, reference_frame_index)
    if len(component) < 2:
        raise BundleAdjustmentError("reference frame has no reliable sequence connection")
    accepted = [edge for edge in accepted if edge.first_index in component and edge.second_index in component]

    if anchor_rotations is not None and len(anchor_rotations) >= 2:
        rotations, retained = _solve_rotations_hierarchical(
            accepted, component, anchor_rotations, camera, max_nfev)
        return rotations, retained, edges

    rotations, retained, _ = _solve_rotations_only(
        accepted, {reference_frame_index: np.eye(3, dtype=np.float64)},
        component, camera, max_nfev)
    return rotations, retained, edges


def _solve_rotations_hierarchical(
    edges: Sequence[_BundleEdge],
    component: set[int],
    anchor_rotations: dict[int, NDArray[np.float64]],
    camera: BaseCameraModel,
    max_nfev: int,
) -> tuple[dict[int, NDArray[np.float64]], list[_BundleEdge]]:
    """Solve each segment between consecutive anchors as an independent
    two-point boundary value problem, instead of one full-sequence solve.

    ``anchor_rotations`` (already gauge-consistent, solved against this same
    fixed camera by ``solve_anchor_camera_and_rotations``) fixes both ends of
    every segment, so each segment's variable count only depends on segment
    length, not on the total sequence length. This keeps every segment at
    the scale already validated on real sequences instead of paying the
    super-linear cost growth of solving the whole sequence at once.

    Frames that are not covered by any segment (outside the anchor span, or
    disconnected within their segment) are simply absent from the returned
    dict, matching how the caller already handles unreachable frames from
    the single-solve path.
    """
    anchor_positions = sorted(index for index in anchor_rotations if index in component)
    if len(anchor_positions) < 2:
        raise BundleAdjustmentError(
            "anchor rotations do not cover the reference-connected component")
    rotations: dict[int, NDArray[np.float64]] = {}
    retained: list[_BundleEdge] = []
    for lo, hi in zip(anchor_positions[:-1], anchor_positions[1:]):
        segment_edges = [
            edge for edge in edges
            if lo <= edge.first_index <= hi and lo <= edge.second_index <= hi]
        segment_component = _connected_component(segment_edges, lo)
        if hi not in segment_component:
            raise BundleAdjustmentError(
                f"anchors {lo} and {hi} are not connected within their segment")
        segment_edges = [
            edge for edge in segment_edges
            if edge.first_index in segment_component
            and edge.second_index in segment_component]
        segment_fixed = {lo: anchor_rotations[lo], hi: anchor_rotations[hi]}
        segment_rotations, segment_retained, _ = _solve_rotations_only(
            segment_edges, segment_fixed, segment_component, camera, max_nfev)
        for index, rotation in segment_rotations.items():
            rotations.setdefault(index, rotation)
        retained.extend(segment_retained)
    return rotations, retained


def _solve_rotations_only(
    edges: Sequence[_BundleEdge],
    fixed_rotations: dict[int, NDArray[np.float64]],
    component: set[int],
    camera: BaseCameraModel,
    max_nfev: int,
) -> tuple[dict[int, NDArray[np.float64]], list[_BundleEdge], Optional[float]]:
    """Full rotation-only backend: factor the camera out of the variables.

    Mirrors ``_solve_bundle_parameters`` but always with an empty camera block,
    so the optimization vector is exactly the per-frame rotation vectors. The
    camera is passed in as a constant model used by the residual closure.
    ``fixed_rotations`` gauge-fixes one or more frames to known absolute
    rotations; a single fixed frame removes the usual rotation-group gauge
    freedom, and two fixed frames turn the remaining component into a
    two-point boundary value problem (used by the hierarchical segment
    solver to fix both ends of a sub-sequence).
    """
    from scipy import sparse
    from scipy.optimize import least_squares

    initial = _initial_rotations(edges, fixed_rotations, component)
    variable_indices = [
        index for index in sorted(component) if index not in fixed_rotations]
    pose_column = {index: 3 * position
                   for position, index in enumerate(variable_indices)}

    def pack(pose_values: dict[int, NDArray[np.float64]]) -> NDArray[np.float64]:
        return np.concatenate(
            [_rvec_from_rotation(pose_values[index])
             for index in variable_indices])

    def unpack(values: NDArray[np.float64]) -> dict[int, NDArray[np.float64]]:
        rotations = dict(fixed_rotations)
        for index in variable_indices:
            rotations[index] = _rotation_from_rvec(
                values[pose_column[index]:pose_column[index] + 3])
        return rotations

    def residual(values: NDArray[np.float64]) -> NDArray[np.float64]:
        rotations = unpack(values)
        values_out = []
        for edge in edges:
            first = camera.unproject(edge.first_pts)
            second = camera.unproject(edge.second_pts)
            predicted = (
                rotations[edge.second_index]
                @ rotations[edge.first_index].T
                @ first.T).T
            values_out.append(np.cross(predicted, second).reshape(-1))
        return np.concatenate(values_out)

    def jacobian(values: NDArray[np.float64]) -> "sparse.csr_matrix":
        """Assemble the block-sparse Jacobian directly as COO triplets.

        Every residual row only has nonzero entries in the (up to two) pose
        columns of its edge's two frames, so a dense ``(rows, cols)`` array
        wastes memory that scales with ``edge_count * frame_count`` instead
        of the true nonzero count. Long sequences (hundreds of frames, each
        edge capped near ``max_pairs_per_edge``) make that dense allocation
        large enough to exceed typical memory limits.
        """
        rotations = unpack(values)
        rows = sum(len(edge.first_pts) * 3 for edge in edges)
        row_chunks: list[NDArray[np.intp]] = []
        col_chunks: list[NDArray[np.intp]] = []
        data_chunks: list[NDArray[np.float64]] = []
        row_offset = 0
        for edge in edges:
            edge_rows = len(edge.first_pts) * 3
            first_rays = camera.unproject(edge.first_pts)
            second_rays = camera.unproject(edge.second_pts)
            first_rvec = (
                None if edge.first_index in fixed_rotations else
                values[pose_column[edge.first_index]:
                       pose_column[edge.first_index] + 3])
            second_rvec = (
                None if edge.second_index in fixed_rotations else
                values[pose_column[edge.second_index]:
                       pose_column[edge.second_index] + 3])
            first_block, second_block = _edge_rotation_jacobians(
                first_rays, second_rays, first_rvec, second_rvec,
                rotations[edge.first_index], rotations[edge.second_index])
            for index, block in ((edge.first_index, first_block),
                                 (edge.second_index, second_block)):
                if block is None:
                    continue
                column = pose_column[index]
                block = block.reshape(edge_rows, 3)
                local_rows = np.repeat(np.arange(edge_rows), 3)
                local_cols = np.tile(np.arange(column, column + 3), edge_rows)
                row_chunks.append(row_offset + local_rows)
                col_chunks.append(local_cols)
                data_chunks.append(block.reshape(-1))
            row_offset += edge_rows
        if row_chunks:
            all_rows = np.concatenate(row_chunks)
            all_cols = np.concatenate(col_chunks)
            all_data = np.concatenate(data_chunks)
        else:
            all_rows = np.empty(0, dtype=np.intp)
            all_cols = np.empty(0, dtype=np.intp)
            all_data = np.empty(0, dtype=np.float64)
        return sparse.coo_matrix(
            (all_data, (all_rows, all_cols)),
            shape=(rows, len(values))).tocsr()

    x0 = pack(initial)
    initial_residual = residual(x0)
    scale = max(float(np.median(np.abs(initial_residual))) * 2.0, 1e-6)
    started = time.perf_counter()
    fit = least_squares(
        residual, x0, jac=jacobian, method="trf", loss="huber",
        f_scale=scale, max_nfev=max_nfev, tr_solver="lsmr", x_scale="jac",
        bounds=(-np.inf, np.inf))
    _log_solve_summary("_solve_rotations_only", fit,
                       n_vars=len(x0), n_edges=len(edges),
                       elapsed_seconds=time.perf_counter() - started)
    if not fit.success or not np.all(np.isfinite(fit.x)):
        raise BundleAdjustmentError(
            f"rotation-only bundle optimization failed: {fit.message}")
    rotations = unpack(fit.x)
    return rotations, list(edges), None


def build_bundle_plan(
    frames: Sequence[BundleFrame],
    reference_frame_index: int,
    *,
    pair_offsets: Sequence[int] = (1, 2, 4),
    random_seed: int | None = 0,
    max_nfev: int = 2000,
    max_pairs_per_edge: int | None = _DEFAULT_MAX_PAIRS_PER_EDGE,
    camera_solve_frames: int | None = None,
    edge_completed: Callable[[], None] | None = None,
    edge_topology: str = "multiscale",
    max_pair_offset: int | None = None,
) -> BAAlignmentPlan:
    """Build a robust same-camera geometry plan from collected frames.

    Frames contain detected stars rather than source pixels. Integration remaps
    the original sequence separately after the plan is final.

    ``camera_solve_frames`` switches to the two-stage solver when the sequence
    is longer than that value. A subset (endpoints and the reference frame
    included) calibrates the shared camera first, then the rotation-only pass
    recovers every pose with that camera held fixed. Shorter sequences keep the
    single-stage joint solve, which is both exact and already cheap there.

    ``edge_topology="dense"`` builds every ``(pos, pos + offset)`` pair for 
    each entry in ``pair_offsets``. ``edge_topology="multiscale"`` (default)
    instead uses the sparse interleaved-start topology (``pair_offsets`` is
    then ignored; edges span every dyadic offset up to ``max_pair_offset``,
    or the whole sequence if unset) which keeps the total edge count near
    ``2 * len(frames)`` regardless of how many scales it spans.
    """
    if not frames:
        raise BundleAdjustmentError("cannot bundle-adjust an empty sequence")
    if edge_topology not in _EDGE_TOPOLOGIES:
        raise BundleAdjustmentError(f"unknown edge topology: {edge_topology!r}")
    by_index = {frame.index: frame for frame in frames}
    if len(by_index) != len(frames) or reference_frame_index not in by_index:
        raise BundleAdjustmentError("reference_frame_index must identify one collected frame")

    if camera_solve_frames is None or len(frames) <= camera_solve_frames:
        return _solve_single_stage(
            frames, reference_frame_index,
            tuple(sorted({int(v) for v in pair_offsets if int(v) > 0})),
            random_seed, max_nfev, max_pairs_per_edge, edge_completed,
            edge_topology=edge_topology, max_pair_offset=max_pair_offset)

    # Two-stage: calibrate the camera on a subset, then solve poses fixed-camera.
    # The subset solve's own poses become anchors for the segmented Stage B
    # solve below, so the full sequence is never solved as one least-squares
    # problem regardless of its length.
    offsets = tuple(sorted({int(v) for v in pair_offsets if int(v) > 0}))
    camera, anchor_rotations = solve_anchor_camera_and_rotations(
        frames, reference_frame_index, sample_count=camera_solve_frames,
        pair_offsets=offsets, random_seed=random_seed, max_nfev=max_nfev,
        max_pairs_per_edge=max_pairs_per_edge,
        edge_topology=edge_topology, max_pair_offset=max_pair_offset)
    injections = _replace_camera(frames, camera)
    rotations, retained, all_edges = solve_rotations_fixed_camera(
        injections, reference_frame_index, camera,
        anchor_rotations=anchor_rotations,
        pair_offsets=offsets, random_seed=random_seed, max_nfev=max_nfev,
        max_pairs_per_edge=max_pairs_per_edge,
        edge_completed=edge_completed,
        edge_topology=edge_topology, max_pair_offset=max_pair_offset)
    reference = next(
        frame for frame in injections if frame.index == reference_frame_index)
    return _assemble_plan(
        by_index, reference_frame_index, all_edges, retained, camera,
        rotations, reference.candidate.optimization_policy, None,
        "staged", None)

