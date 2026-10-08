"""Star point feature extraction and matching."""
import dataclasses

import cv2
import numpy as np
import numpy.linalg as la
from loguru import logger
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from scipy.spatial import distance as spd

from hoshicore._custom_op import (
    extract_point_features as custom_extract_point_features,
)
from hoshicore._custom_op import asterism_anchor_votes
from hoshicore._custom_op import asterism_mutual_nearest
from hoshicore._custom_op import asterism_tokens
from hoshicore._custom_op import matching_cosine_bidirectional_nearest


@dataclasses.dataclass
class MatchResult:
    pair_idx: NDArray[np.int32]
    ref_pts: NDArray[np.float64]
    src_pts: NDArray[np.float64]
    rotation: NDArray[np.float64]
    homography: NDArray[np.float64] | None = None
    initial_pair_count: int | None = None


@dataclasses.dataclass(frozen=True)
class RotationDiagnostics:
    inlier_count: int
    median_angle_error_rad: float
    p90_angle_error_rad: float
    max_angle_error_rad: float
    coverage_ratio: float
    radial_bin_count: int
    active_sector_count: int
    outer_inlier_count: int


@dataclasses.dataclass
class RotationCandidate:
    rotation: NDArray[np.float64]
    pair_idx: NDArray[np.int32]
    diagnostics: RotationDiagnostics
    angle_errors_rad: NDArray[np.float64]
    accepted: bool
    rejection_reason: str | None
    iteration: int
    sample_size: int
    sampling_mode: str


@dataclasses.dataclass(frozen=True)
class RotationValidationConfig:
    max_angle_median_rad: float
    max_angle_p90_rad: float
    min_coverage_ratio: float


@dataclasses.dataclass(frozen=True)
class CoverageRepairContext:
    enabled: bool = True
    num_sectors: int = 8
    radial_inner_ratio: float = 0.33
    radial_mid_ratio: float = 0.66
    active_sector_min_candidates: int = 6
    active_outer_min_candidates: int = 12
    sector_min_keep: int = 1
    outer_min_keep_abs: int = 4
    outer_min_keep_ratio: float = 0.08
    quality_relax_ratio: float = 1.35


@dataclasses.dataclass(frozen=True)
class AsterismMatchingConfig:
    """Local spherical-triangle token and voting parameters."""

    neighbor_count: int = 8
    edge_ratio_tolerance: float = 0.025
    log_scale_tolerance: float = 0.04
    token_distance_threshold: float = 1.0
    min_votes: int = 5
    min_vote_margin: int = 1


@dataclasses.dataclass(frozen=True)
class AsterismTokens:
    values: NDArray[np.float64]
    anchor_indices: NDArray[np.int32]


FISHEYE_ROTATION_VALIDATION = RotationValidationConfig(
    max_angle_median_rad=np.deg2rad(0.35),
    max_angle_p90_rad=np.deg2rad(0.9),
    min_coverage_ratio=0.02,
)

MIN_ROTATION_INLIERS = 6
MIN_ROTATION_SAMPLE_SIZE = 3
ROTATION_RANSAC_ANGLE_THRESHOLD_RAD = np.deg2rad(0.75)
MAX_ROTATION_TRIALS = 200
MIN_FILTERED_UNIQUE_PAIRS = 4
LOW_PAIR_COUNT_THRESHOLD = 10
MIN_FILTER_KEEP_RATIO = 0.5

DEFAULT_COVERAGE_REPAIR = CoverageRepairContext()


def _convex_hull_area(pts: NDArray[np.float64]) -> float:
    """Convex hull area for a point cloud."""
    if len(pts) < 3:
        return 0.0
    hull = cv2.convexHull(np.asarray(pts, dtype=np.float32))
    return float(abs(cv2.contourArea(hull)))


def _make_canvas_extent(
        pts1: NDArray[np.float64],
        pts2: NDArray[np.float64]) -> tuple[NDArray[np.float64], float, float]:
    """Derive a conservative image extent from all available points."""
    all_pts = np.vstack((pts1, pts2))
    min_xy = np.min(all_pts, axis=0)
    max_xy = np.max(all_pts, axis=0)
    span = np.maximum(max_xy - min_xy, 1.0)
    corners = np.array([
        min_xy,
        [max_xy[0], min_xy[1]],
        max_xy,
        [min_xy[0], max_xy[1]],
    ],
                       dtype=np.float64)
    canvas_area = float(span[0] * span[1])
    canvas_diag = float(la.norm(span))
    return corners, canvas_area, canvas_diag


def _format_pair_distribution(
    pts: NDArray[np.float64],
    pair_idx: NDArray[np.int32],
    num_sectors: int = 8,
) -> str:
    """Summarize matched reference-point distribution by radial bins and sectors."""
    if len(pair_idx) == 0:
        return "count=0"

    ref_pts = pts[pair_idx[:, 0]]
    min_xy = np.min(pts, axis=0)
    max_xy = np.max(pts, axis=0)
    center = 0.5 * (min_xy + max_xy)
    canvas_radius = float(la.norm(0.5 * (max_xy - min_xy)))
    if canvas_radius <= 1e-12:
        canvas_radius = 1.0

    rel = ref_pts - center
    radius_norm = la.norm(rel, axis=1) / canvas_radius
    angle_deg = (np.degrees(np.arctan2(rel[:, 1], rel[:, 0])) + 360.0) % 360.0

    radial_bins = np.zeros(len(pair_idx), dtype=np.int32)
    radial_bins[radius_norm >= 0.33] = 1
    radial_bins[radius_norm >= 0.66] = 2
    radial_counts = np.bincount(radial_bins, minlength=3)

    sector_width = 360.0 / num_sectors
    sector_ids = np.floor(angle_deg / sector_width).astype(np.int32)
    sector_ids = np.clip(sector_ids, 0, num_sectors - 1)
    sector_counts = np.bincount(sector_ids, minlength=num_sectors)
    active_sectors = int(np.count_nonzero(sector_counts))

    return (
        f"count={len(pair_idx)}  "
        f"radial={int(radial_counts[0])}/{int(radial_counts[1])}/{int(radial_counts[2])}  "
        f"sectors={active_sectors}/{num_sectors} "
        f"min/p50/max={int(np.min(sector_counts))}/"
        f"{float(np.percentile(sector_counts, 50)):.0f}/"
        f"{int(np.max(sector_counts))}")


def _log_pair_distribution(
    stage: str,
    pts: NDArray[np.float64],
    pair_idx: NDArray[np.int32],
) -> None:
    logger.debug(
        "Pair distribution [{}]: {}",
        stage,
        _format_pair_distribution(pts, pair_idx),
    )


def _pair_geometry_bins(
    pts: NDArray[np.float64],
    pair_idx: NDArray[np.int32],
    coverage_ctx: CoverageRepairContext,
) -> tuple[NDArray[np.float64], NDArray[np.int32], NDArray[np.int32]]:
    """Return normalized radius, radial-bin ids, and sector ids for ref points."""
    if len(pair_idx) == 0:
        empty_f = np.zeros(0, dtype=np.float64)
        empty_i = np.zeros(0, dtype=np.int32)
        return empty_f, empty_i, empty_i

    ref_pts = pts[pair_idx[:, 0]]
    min_xy = np.min(pts, axis=0)
    max_xy = np.max(pts, axis=0)
    center = 0.5 * (min_xy + max_xy)
    canvas_radius = float(la.norm(0.5 * (max_xy - min_xy)))
    if canvas_radius <= 1e-12:
        canvas_radius = 1.0

    rel = ref_pts - center
    radius_norm = la.norm(rel, axis=1) / canvas_radius
    radial_bin = np.zeros(len(pair_idx), dtype=np.int32)
    radial_bin[radius_norm >= coverage_ctx.radial_inner_ratio] = 1
    radial_bin[radius_norm >= coverage_ctx.radial_mid_ratio] = 2

    sector_deg = 360.0 / max(coverage_ctx.num_sectors, 1)
    angle_deg = (np.degrees(np.arctan2(rel[:, 1], rel[:, 0])) + 360.0) % 360.0
    sector_bin = np.floor(angle_deg / sector_deg).astype(np.int32)
    sector_bin = np.clip(sector_bin, 0, max(coverage_ctx.num_sectors - 1, 0))
    return radius_norm, radial_bin, sector_bin


def _repair_pair_coverage(
    pts: NDArray[np.float64],
    mutual_pair_idx: NDArray[np.int32],
    mutual_pair_dist: NDArray[np.float64],
    selected_pair_idx: NDArray[np.int32],
    distance_threshold: float,
    coverage_ctx: CoverageRepairContext,
) -> NDArray[np.int32]:
    """Soft coverage repair after percentile filtering.

    Adds a small number of near-threshold candidates from underrepresented bins
    without forcing coverage in bins that never had meaningful candidates.
    """
    if (not coverage_ctx.enabled or len(mutual_pair_idx) == 0
            or len(selected_pair_idx) == 0):
        return selected_pair_idx

    _, radial_bin, sector_bin = _pair_geometry_bins(pts, mutual_pair_idx,
                                                    coverage_ctx)
    selected_mask = mutual_pair_dist <= distance_threshold
    quality_cap = distance_threshold * coverage_ctx.quality_relax_ratio

    def _add_best(mask: NDArray[np.bool_], target_count: int) -> int:
        if target_count <= 0:
            return 0
        candidate_idx = np.where(mask & (~selected_mask)
                                 & (mutual_pair_dist <= quality_cap))[0]
        if candidate_idx.size == 0:
            return 0
        order = candidate_idx[np.argsort(mutual_pair_dist[candidate_idx])]
        keep = order[:target_count]
        selected_mask[keep] = True
        return int(len(keep))

    sector_additions = 0
    active_sector_count = 0
    for sector_id in range(coverage_ctx.num_sectors):
        sector_mask = sector_bin == sector_id
        available = int(np.count_nonzero(sector_mask))
        if available < coverage_ctx.active_sector_min_candidates:
            continue
        active_sector_count += 1
        selected = int(np.count_nonzero(selected_mask & sector_mask))
        deficit = min(coverage_ctx.sector_min_keep, available) - selected
        if deficit > 0:
            sector_additions += _add_best(sector_mask, deficit)

    outer_mask = radial_bin == 2
    outer_available = int(np.count_nonzero(outer_mask))
    outer_selected_before = int(np.count_nonzero(selected_mask & outer_mask))
    outer_additions = 0
    if outer_available >= coverage_ctx.active_outer_min_candidates:
        target_outer = min(
            outer_available,
            max(
                coverage_ctx.outer_min_keep_abs,
                int(
                    np.ceil(
                        len(selected_pair_idx) *
                        coverage_ctx.outer_min_keep_ratio)),
            ),
        )
        outer_deficit = target_outer - int(
            np.count_nonzero(selected_mask & outer_mask))
        if outer_deficit > 0:
            outer_additions = _add_best(outer_mask, outer_deficit)

    repaired_pair_idx = mutual_pair_idx[selected_mask]
    if sector_additions > 0 or outer_additions > 0:
        logger.debug(
            "Coverage repair added pairs: baseline={} repaired={} added={} "
            "active_sectors={} outer={}->{} available={} quality_cap={:.6f}",
            len(selected_pair_idx),
            len(repaired_pair_idx),
            len(repaired_pair_idx) - len(selected_pair_idx),
            active_sector_count,
            outer_selected_before,
            int(np.count_nonzero(selected_mask & outer_mask)),
            outer_available,
            float(quality_cap),
        )
    return repaired_pair_idx


def _format_rotation_residual_profile(
    pts: NDArray[np.float64],
    pair_idx: NDArray[np.int32],
    angle_errors_rad: NDArray[np.float64],
    threshold_rad: float = ROTATION_RANSAC_ANGLE_THRESHOLD_RAD,
) -> str:
    """Summarize angular residuals by radial zone and threshold pass rate."""
    if len(pair_idx) == 0 or len(angle_errors_rad) == 0:
        return "count=0"

    coverage_ctx = DEFAULT_COVERAGE_REPAIR
    _, radial_bin, _ = _pair_geometry_bins(pts, pair_idx, coverage_ctx)
    zone_names = ("inner", "mid", "outer")
    threshold_list_rad = (
        threshold_rad,
        np.deg2rad(1.0),
        np.deg2rad(1.5),
    )

    zone_parts = []
    for zone_id, zone_name in enumerate(zone_names):
        zone_mask = radial_bin == zone_id
        zone_count = int(np.count_nonzero(zone_mask))
        if zone_count == 0:
            zone_parts.append(f"{zone_name}=count0")
            continue
        zone_err = angle_errors_rad[zone_mask]
        pass_counts = [
            int(np.count_nonzero(zone_err < th_rad))
            for th_rad in threshold_list_rad
        ]
        zone_parts.append(f"{zone_name}=count{zone_count}"
                          f"/med{np.rad2deg(np.median(zone_err)):.3f}"
                          f"/p90{np.rad2deg(np.percentile(zone_err, 90)):.3f}"
                          f"/max{np.rad2deg(np.max(zone_err)):.3f}"
                          f"/pass0.75={pass_counts[0]}"
                          f"/pass1.00={pass_counts[1]}"
                          f"/pass1.50={pass_counts[2]}")
    return "  ".join(zone_parts)


def _log_rotation_candidate_profile(
    stage: str,
    pts: NDArray[np.float64],
    pair_idx: NDArray[np.int32],
    angle_errors_rad: NDArray[np.float64],
) -> None:
    logger.debug(
        "Rotation residual profile [{}]: {}",
        stage,
        _format_rotation_residual_profile(
            pts,
            pair_idx,
            angle_errors_rad,
            threshold_rad=ROTATION_RANSAC_ANGLE_THRESHOLD_RAD,
        ),
    )


def _log_rotation_candidate(rotation_candidate: RotationCandidate,
                            header_text: str, unique_pair_count: int,
                            early_stop: bool) -> None:
    logger.debug(
        "{}: iteration={} inliers={} "
        "inlier_ratio={:.3f} median_angle={:.4f}deg "
        "p90_angle={:.4f}deg max_angle={:.4f}deg coverage_ratio={:.4f} "
        "radial_bins={} sectors={} outer_inliers={} "
        "sample_size={} sampling_mode={} early_stop={}", header_text,
        rotation_candidate.iteration,
        rotation_candidate.diagnostics.inlier_count,
        rotation_candidate.diagnostics.inlier_count / unique_pair_count,
        np.rad2deg(rotation_candidate.diagnostics.median_angle_error_rad),
        np.rad2deg(rotation_candidate.diagnostics.p90_angle_error_rad),
        np.rad2deg(rotation_candidate.diagnostics.max_angle_error_rad),
        rotation_candidate.diagnostics.coverage_ratio,
        rotation_candidate.diagnostics.radial_bin_count,
        rotation_candidate.diagnostics.active_sector_count,
        rotation_candidate.diagnostics.outer_inlier_count,
        rotation_candidate.sample_size, rotation_candidate.sampling_mode,
        early_stop)


def _sample_pair_subset(
    pair_idx: NDArray[np.int32],
    sample_size: int,
    rng: np.random.Generator,
) -> tuple[NDArray[np.int32], str]:
    """Sample candidate pairs directly from init_pair_idx."""
    if len(pair_idx) <= sample_size:
        return pair_idx, "all_pairs"
    selected = rng.choice(len(pair_idx), size=sample_size, replace=False)
    return pair_idx[selected], "random_no_replacement"


def _estimate_rotation_svd(
    ref_vectors: NDArray[np.float64],
    src_vectors: NDArray[np.float64],
) -> NDArray[np.float64]:
    # Solve rotation_ref_to_src such that src ~= R @ ref.
    h_mat = ref_vectors.T @ src_vectors
    u_mat, _, vt_mat = np.linalg.svd(h_mat)
    rotation = vt_mat.T @ u_mat.T
    if np.linalg.det(rotation) < 0:
        vt_mat[-1, :] *= -1
        rotation = vt_mat.T @ u_mat.T
    return rotation


def _rotation_errors_for_pairs(
    vectors1: NDArray[np.float64],
    vectors2: NDArray[np.float64],
    pair_idx: NDArray[np.int32],
    rotation: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Angular residuals for pair_idx under rotation_ref_to_src."""
    ref_vec = vectors1[pair_idx[:, 0]]
    src_vec = vectors2[pair_idx[:, 1]]
    rotated_ref = (rotation @ ref_vec.T).T
    dots = np.sum(rotated_ref * src_vec, axis=1)
    return np.arccos(np.clip(dots, -1.0, 1.0))


def refine_rotation_candidate(
    vectors1: NDArray[np.float64],
    vectors2: NDArray[np.float64],
    seed_pair_idx: NDArray[np.int32],
    all_pair_idx: NDArray[np.int32],
    rounds: int = 2,
) -> tuple[NDArray[np.float64], NDArray[np.int32], NDArray[np.float64]]:
    """Local optimization for one RANSAC seed.

    Repeatedly fit R from current pairs and reselect inliers from all pairs.
    Returns the final rotation, final inlier pairs, and residuals for those
    final inliers under the final rotation.
    """
    pair_idx = seed_pair_idx
    rotation = np.eye(3, dtype=np.float64)
    for round_idx in range(rounds):
        rotation = _estimate_rotation_svd(vectors1[pair_idx[:, 0]],
                                          vectors2[pair_idx[:, 1]])
        all_errors = _rotation_errors_for_pairs(vectors1, vectors2,
                                                all_pair_idx, rotation)
        pair_idx = all_pair_idx[all_errors <
                                ROTATION_RANSAC_ANGLE_THRESHOLD_RAD]
        if len(pair_idx) < MIN_ROTATION_INLIERS:
            raise ValueError(
                f"rotation refinement round {round_idx + 1} inlier count below minimum: {len(pair_idx)}"
            )
    # final rotation is based on the final inlier set.
    rotation = _estimate_rotation_svd(vectors1[pair_idx[:, 0]],
                                      vectors2[pair_idx[:, 1]])
    final_errors = _rotation_errors_for_pairs(vectors1, vectors2, pair_idx,
                                              rotation)
    return rotation, pair_idx, final_errors


def evaluate_rotation(
    pts1: NDArray[np.float64],
    pts2: NDArray[np.float64],
    pair_idx: NDArray[np.int32],
    vectors1: NDArray[np.float64],
    vectors2: NDArray[np.float64],
    rotation: NDArray[np.float64],
) -> RotationDiagnostics:
    angle_errors = _rotation_errors_for_pairs(vectors1, vectors2, pair_idx,
                                              rotation)

    ref_pts = pts1[pair_idx[:, 0]]
    src_pts = pts2[pair_idx[:, 1]]
    _, canvas_area, _ = _make_canvas_extent(pts1, pts2)
    coverage_ratio = min(_convex_hull_area(ref_pts),
                         _convex_hull_area(src_pts))
    coverage_ratio = coverage_ratio / max(canvas_area, 1.0)
    _, radial_bin, sector_bin = _pair_geometry_bins(pts1, pair_idx,
                                                    DEFAULT_COVERAGE_REPAIR)

    return RotationDiagnostics(
        inlier_count=int(len(pair_idx)),
        median_angle_error_rad=float(np.median(angle_errors)),
        p90_angle_error_rad=float(np.percentile(angle_errors, 90)),
        max_angle_error_rad=float(np.max(angle_errors)),
        coverage_ratio=float(coverage_ratio),
        radial_bin_count=int(np.unique(radial_bin).size)
        if len(radial_bin) else 0,
        active_sector_count=int(np.unique(sector_bin).size)
        if len(sector_bin) else 0,
        outer_inlier_count=int(np.count_nonzero(radial_bin == 2)),
    )


def validate_rotation(
    pts1: NDArray[np.float64],
    pts2: NDArray[np.float64],
    pair_idx: NDArray[np.int32],
    vectors1: NDArray[np.float64],
    vectors2: NDArray[np.float64],
    rotation: NDArray[np.float64],
    config: RotationValidationConfig = FISHEYE_ROTATION_VALIDATION,
) -> RotationDiagnostics:
    diagnostics = evaluate_rotation(pts1, pts2, pair_idx, vectors1, vectors2,
                                    rotation)
    reject_reasons = []
    if not np.all(np.isfinite(rotation)):
        reject_reasons.append("rotation contains non-finite values")
    if diagnostics.median_angle_error_rad > config.max_angle_median_rad:
        reject_reasons.append(
            f"median_angle={np.rad2deg(diagnostics.median_angle_error_rad):.4f}deg > "
            f"{np.rad2deg(config.max_angle_median_rad):.4f}deg")
    if diagnostics.p90_angle_error_rad > config.max_angle_p90_rad:
        reject_reasons.append(
            f"p90_angle={np.rad2deg(diagnostics.p90_angle_error_rad):.4f}deg > "
            f"{np.rad2deg(config.max_angle_p90_rad):.4f}deg")
    if diagnostics.coverage_ratio < config.min_coverage_ratio:
        reject_reasons.append(
            f"coverage_ratio={diagnostics.coverage_ratio:.4f} < {config.min_coverage_ratio:.4f}"
        )
    if reject_reasons:
        raise ValueError("Rotation rejected: " + "; ".join(reject_reasons))
    return diagnostics


def build_rotation_candidate(
    pts1: NDArray[np.float64],
    pts2: NDArray[np.float64],
    vectors1: NDArray[np.float64],
    vectors2: NDArray[np.float64],
    all_pair_idx: NDArray[np.int32],
    sampled_pair_idx: NDArray[np.int32],
    iteration: int,
    sample_size: int,
    sampling_mode: str,
    config: RotationValidationConfig = FISHEYE_ROTATION_VALIDATION,
) -> RotationCandidate:
    if len(sampled_pair_idx) < 3:
        raise ValueError(
            f"sampled pair count below rotation minimum: {len(sampled_pair_idx)}"
        )

    final_rotation, final_inlier_pair_idx, final_angle_errors = (
        refine_rotation_candidate(
            vectors1,
            vectors2,
            sampled_pair_idx,
            all_pair_idx,
            rounds=2,
        ))

    diagnostics = evaluate_rotation(pts1, pts2, final_inlier_pair_idx,
                                    vectors1, vectors2, final_rotation)
    accepted = True
    rejection_reason = None
    try:
        validate_rotation(pts1, pts2, final_inlier_pair_idx, vectors1,
                          vectors2, final_rotation, config)
    except ValueError as exc:
        accepted = False
        rejection_reason = str(exc)

    return RotationCandidate(
        rotation=final_rotation,
        pair_idx=final_inlier_pair_idx,
        diagnostics=diagnostics,
        angle_errors_rad=final_angle_errors,
        accepted=accepted,
        rejection_reason=rejection_reason,
        iteration=iteration,
        sample_size=sample_size,
        sampling_mode=sampling_mode,
    )


def _rotation_candidate_rank(
        candidate: RotationCandidate) -> tuple[float, ...]:
    diag = candidate.diagnostics
    return (
        1.0 if candidate.accepted else 0.0,
        float(diag.radial_bin_count),
        float(diag.active_sector_count),
        float(diag.outer_inlier_count),
        diag.coverage_ratio,
        float(diag.inlier_count),
        -diag.median_angle_error_rad,
        -diag.p90_angle_error_rad,
    )


def _is_high_quality_rotation_candidate(
    candidate: RotationCandidate,
    unique_pair_count: int,
    config: RotationValidationConfig,
) -> bool:
    """Early-stop only for candidates that are both accurate and well covered."""
    if not candidate.accepted:
        return False
    diag = candidate.diagnostics
    inlier_ratio = diag.inlier_count / max(unique_pair_count, 1)
    return (diag.radial_bin_count >= 2 and diag.active_sector_count >= 4
            and diag.coverage_ratio >= config.min_coverage_ratio * 2.0
            and diag.p90_angle_error_rad <= config.max_angle_p90_rad * 0.75
            and inlier_ratio >= 0.5)


def _should_fallback_to_unfiltered(
    before_filter_pair_idx: NDArray[np.int32],
    after_filter_pair_idx: NDArray[np.int32],
) -> tuple[bool, str | None]:
    """Decide whether filtered pairs should fallback to the unfiltered set."""
    before_count = len(before_filter_pair_idx)
    after_count = len(after_filter_pair_idx)
    unique_after_count = len(np.unique(after_filter_pair_idx, axis=0))
    kept_ratio = after_count / before_count if before_count else 0.0

    if unique_after_count < MIN_FILTERED_UNIQUE_PAIRS:
        return (
            True,
            f"unique_pairs={unique_after_count} < {MIN_FILTERED_UNIQUE_PAIRS}")
    if (before_count < LOW_PAIR_COUNT_THRESHOLD
            and kept_ratio < MIN_FILTER_KEEP_RATIO):
        return (
            True,
            f"kept_ratio={kept_ratio:.3f} < {MIN_FILTER_KEEP_RATIO:.3f} with "
            f"before_pairs={before_count} < {LOW_PAIR_COUNT_THRESHOLD}")
    return False, None


def adaptive_k(star_count: int) -> int:
    """Choose neighbor count for feature extraction based on available stars.

    With fewer stars, using a large k makes each descriptor depend on a large
    fraction of all points, causing instability across frames. This reduces k
    for sparse fields while preserving full discriminative power when stars are
    plentiful.
    """
    if star_count < 30:
        return 6
    if star_count < 45:
        return 8
    if star_count < 70:
        return 12
    return 15


def extract_point_features(vec: NDArray[np.float64],
                           vol: NDArray[np.float64],
                           k: int = 15) -> NDArray[np.float64]:
    """Extract geometric star descriptors through the custom-op dispatch."""
    return custom_extract_point_features(vec, vol, k=k)


def extract_asterism_tokens(
    vectors: NDArray[np.float64],
    neighbor_count: int = 8,
) -> AsterismTokens:
    """Build unordered local spherical-triangle tokens for every star.

    Each token distinguishes the center star from its two neighbors and stores
    ``short_anchor_edge / long_anchor_edge``, ``neighbor_edge /
    long_anchor_edge`` and ``log(long_anchor_edge)``. Unit-sphere chord lengths
    avoid inverse trigonometry while remaining rotation invariant.
    """
    values, anchor_indices = asterism_tokens(vectors, neighbor_count)
    return AsterismTokens(values=values, anchor_indices=anchor_indices)


def find_asterism_initial_match(
    vectors1: NDArray[np.float64],
    vectors2: NDArray[np.float64],
    config: AsterismMatchingConfig = AsterismMatchingConfig(),
) -> NDArray[np.int32]:
    """Match stars by mutual local-triangle tokens and anchor-level voting."""
    if (config.edge_ratio_tolerance <= 0
            or config.log_scale_tolerance <= 0
            or config.token_distance_threshold <= 0):
        raise ValueError("asterism tolerances must be positive")
    if config.min_votes < 1 or config.min_vote_margin < 0:
        raise ValueError("asterism vote thresholds are invalid")

    tokens1 = extract_asterism_tokens(vectors1, config.neighbor_count)
    tokens2 = extract_asterism_tokens(vectors2, config.neighbor_count)
    scale = np.array([
        config.edge_ratio_tolerance,
        config.edge_ratio_tolerance,
        config.log_scale_tolerance,
    ], dtype=np.float64)
    values1 = tokens1.values / scale
    values2 = tokens2.values / scale
    token_pairs1, token_pairs2 = asterism_mutual_nearest(
        values1, values2, config.token_distance_threshold)
    if len(token_pairs1) == 0:
        logger.debug("Asterism match: no mutually compatible tokens")
        return np.empty((0, 2), dtype=np.int32)

    pair_idx, voted_pairs, accepted_votes = asterism_anchor_votes(
        token_pairs1,
        token_pairs2,
        tokens1.anchor_indices,
        tokens2.anchor_indices,
        len(vectors1),
        len(vectors2),
        config.min_votes,
        config.min_vote_margin,
    )
    logger.debug(
        "Asterism match: tokens={}/{} mutual_tokens={} voted_pairs={} "
        "accepted_pairs={} votes_min/median/max={}/{:.1f}/{}",
        len(tokens1.values),
        len(tokens2.values),
        len(token_pairs1),
        voted_pairs,
        len(pair_idx),
        int(np.min(accepted_votes)) if len(accepted_votes) else 0,
        float(np.median(accepted_votes)) if len(accepted_votes) else 0.0,
        int(np.max(accepted_votes)) if len(accepted_votes) else 0,
    )
    return pair_idx


def find_guided_mutual_match(
    pts1: NDArray[np.float64],
    pts2: NDArray[np.float64],
    predicted_pts2: NDArray[np.float64],
    predicted_pts1: NDArray[np.float64],
    max_distance_px: float,
) -> NDArray[np.int32]:
    """Match two point sets using bidirectional projected-position guidance.

    ``predicted_pts2[i]`` is the predicted location of ``pts1[i]`` in the
    second image.  ``predicted_pts1[j]`` is the inverse prediction for
    ``pts2[j]``.  Non-finite predictions are ignored.  A pair is retained only
    when both nearest-neighbor searches agree and both native-image distances
    are within ``max_distance_px``.
    """
    pts1 = np.asarray(pts1, dtype=np.float64)
    pts2 = np.asarray(pts2, dtype=np.float64)
    predicted_pts2 = np.asarray(predicted_pts2, dtype=np.float64)
    predicted_pts1 = np.asarray(predicted_pts1, dtype=np.float64)

    if (pts1.ndim != 2 or pts1.shape[1:] != (2, ) or pts2.ndim != 2
            or pts2.shape[1:] != (2, )):
        raise ValueError("guided mutual matching expects (N, 2) point arrays")
    if predicted_pts2.shape != pts1.shape:
        raise ValueError("predicted_pts2 must have the same shape as pts1")
    if predicted_pts1.shape != pts2.shape:
        raise ValueError("predicted_pts1 must have the same shape as pts2")
    if not np.isfinite(max_distance_px) or max_distance_px <= 0:
        raise ValueError("max_distance_px must be finite and positive")
    if len(pts1) == 0 or len(pts2) == 0:
        return np.empty((0, 2), dtype=np.int32)
    if not np.all(np.isfinite(pts1)) or not np.all(np.isfinite(pts2)):
        raise ValueError("guided mutual matching requires finite input points")

    valid12 = np.all(np.isfinite(predicted_pts2), axis=1)
    valid21 = np.all(np.isfinite(predicted_pts1), axis=1)

    nearest12 = np.full(len(pts1), -1, dtype=np.int64)
    distance12 = np.full(len(pts1), np.inf, dtype=np.float64)
    if np.any(valid12):
        distance12[valid12], nearest12[valid12] = cKDTree(pts2).query(
            predicted_pts2[valid12], k=1)

    nearest21 = np.full(len(pts2), -1, dtype=np.int64)
    distance21 = np.full(len(pts2), np.inf, dtype=np.float64)
    if np.any(valid21):
        distance21[valid21], nearest21[valid21] = cKDTree(pts1).query(
            predicted_pts1[valid21], k=1)

    ref_indices = np.flatnonzero(
        valid12 & (nearest12 >= 0) & (distance12 <= max_distance_px))
    if len(ref_indices) == 0:
        return np.empty((0, 2), dtype=np.int32)

    src_indices = nearest12[ref_indices]
    mutual = (
        valid21[src_indices]
        & (distance21[src_indices] <= max_distance_px)
        & (nearest21[src_indices] == ref_indices)
    )
    return np.column_stack((ref_indices[mutual], src_indices[mutual])).astype(
        np.int32, copy=False)


def find_initial_match(features1: NDArray[np.float64],
                       features2: NDArray[np.float64],
                       pts1: NDArray[np.float64],
                       pts2: NDArray[np.float64],
                       vectors1: NDArray[np.float64] = None,
                       vectors2: NDArray[np.float64] = None,
                       alpha: float = 0.00,
                       apply_threshold_filter: bool = True,
                       coverage_repair_context:
                       CoverageRepairContext = DEFAULT_COVERAGE_REPAIR,
                       theta_th: float = np.pi / 6,
                       dist_multiplier: float = 0.3) -> NDArray[np.int32]:
    """Find initial matches between two star images using feature similarity.

    Args:
        features1, features2: (n, d) feature matrices.
        pts1, pts2: (n, 2) pixel coordinates.
        vectors1, vectors2: (n, 3) unit vectors (needed if apply_threshold_filter=True).
        alpha: weight of Euclidean distance in matching.
        apply_threshold_filter: whether to apply angular/distance threshold.
        theta_th: angular distance threshold.
        dist_multiplier: distance multiplier for pixel threshold.

    Returns:
        (m, 2) array of matched index pairs.
    """
    if alpha > 0:
        measure_dist_mat = spd.cdist(features1, features2, "cosine")
        pts_stack = np.vstack((pts1, pts2))
        pts_mean = np.mean(pts_stack, axis=0)
        pts_min = np.min(pts_stack, axis=0)
        pts_max = np.max(pts_stack, axis=0)
        pts_dist_mat = spd.cdist((pts1 - pts_mean) / (pts_max - pts_min),
                                 (pts2 - pts_mean) / (pts_max - pts_min),
                                 "euclidean")
        dist_mat = measure_dist_mat * (1 - alpha) + pts_dist_mat * alpha
        row_order = np.argsort(dist_mat, axis=1)
        col_order = np.argsort(dist_mat, axis=0)
        row_indices = row_order[:, 0]
        col_indices = col_order[0, :]
        row_distances = dist_mat[range(len(features1)), row_indices]
        col_distances = dist_mat[col_indices, range(len(features2))]
    else:
        (
            row_indices,
            row_distances,
            col_indices,
            col_distances,
        ) = matching_cosine_bidirectional_nearest(features1, features2)

    num1 = len(features1)

    ind = col_indices[row_indices] == range(num1)
    mutual_pair_count = int(np.count_nonzero(ind))
    mutual_pair_idx = np.stack((np.where(ind)[0], row_indices[ind]), axis=-1)

    d_th = min(np.percentile(row_distances, 30),
               np.percentile(col_distances, 30))
    ind = np.logical_and(ind, row_distances < d_th)

    pair_idx = np.stack((np.where(ind)[0], row_indices[ind]), axis=-1)
    percentile_pair_count = len(pair_idx)

    mutual_pair_dist = row_distances[mutual_pair_idx[:, 0]]
    pair_idx = _repair_pair_coverage(
        pts1,
        mutual_pair_idx,
        mutual_pair_dist,
        pair_idx,
        d_th,
        coverage_repair_context,
    )
    logger.debug(
        "Initial match summary: mutual={} percentile30={} repaired={} "
        "distance_threshold={:.6f} repaired_distribution=({})",
        mutual_pair_count,
        percentile_pair_count,
        len(pair_idx),
        float(d_th),
        _format_pair_distribution(pts1, pair_idx),
    )

    if apply_threshold_filter:
        if vectors1 is None or vectors2 is None:
            raise ValueError(
                "vectors1 and vectors2 required when apply_threshold_filter=True"
            )
        unfiltered_pair_idx = pair_idx.copy()
        before_filter_count = len(pair_idx)
        if before_filter_count == 0:
            logger.debug(
                "Threshold filter skipped: before_pairs=0, fallback_to_unfiltered={}",
                False)
            return pair_idx
        logger.debug("Applying threshold filter.")
        theta = np.arccos(
            np.clip(
                np.sum(vectors1[pair_idx[:, 0]] * vectors2[pair_idx[:, 1]],
                       axis=1), -1, 1))
        theta_th = min(np.percentile(theta, 75), theta_th)

        pts_dist = la.norm(pts1[pair_idx[:, 0]] - pts2[pair_idx[:, 1]], axis=1)
        dist_th = max(np.max(pts1), np.max(pts2)) * dist_multiplier
        pair_idx = pair_idx[np.logical_and(theta < theta_th, pts_dist
                                           < dist_th)]
        fallback_to_unfiltered, fallback_reason = _should_fallback_to_unfiltered(
            unfiltered_pair_idx, pair_idx)
        if fallback_to_unfiltered:
            logger.debug(
                "Threshold filter fallback: fallback_to_unfiltered={}, "
                "reason={}", True, fallback_reason)
            pair_idx = unfiltered_pair_idx
        logger.debug(
            "Threshold filter stats: before_pairs={}, after_pairs={}, "
            "kept_ratio={:.3f}, theta_threshold={:.6f}, pixel_threshold={:.3f}, "
            "fallback_to_unfiltered={}", before_filter_count, len(pair_idx),
            len(pair_idx) /
            before_filter_count if before_filter_count else 0.0,
            float(theta_th), float(dist_th), fallback_to_unfiltered)
        _log_pair_distribution("threshold_filter", pts1, pair_idx)
    return pair_idx


def fine_tune_rotation(
    pts1: NDArray[np.float64],
    pts2: NDArray[np.float64],
    vectors1: NDArray[np.float64],
    vectors2: NDArray[np.float64],
    init_pair_idx: NDArray[np.int32],
    config: RotationValidationConfig = FISHEYE_ROTATION_VALIDATION,
    random_seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Refine matching using RANSAC on unit-vector rotation consistency."""
    unique_pair_idx = np.unique(init_pair_idx, axis=0).astype(np.int32,
                                                              copy=False)
    unique_pair_count = len(unique_pair_idx)
    if unique_pair_count < MIN_ROTATION_INLIERS:
        raise ValueError(
            f"rotation RANSAC requires at least {MIN_ROTATION_INLIERS} unique point pairs, "
            f"got {unique_pair_count}")

    sample_size = min(MIN_ROTATION_SAMPLE_SIZE, unique_pair_count)
    max_iterations = 1 if unique_pair_count <= sample_size else MAX_ROTATION_TRIALS

    logger.debug(
        "Rotation fine-tune setup: init_pairs={}, unique_pairs={}, sample_size={}, "
        "max_iterations={}, angle_threshold_deg={:.3f}", len(init_pair_idx),
        unique_pair_count, sample_size, max_iterations,
        np.rad2deg(ROTATION_RANSAC_ANGLE_THRESHOLD_RAD))

    rng = np.random.default_rng(random_seed)
    best_candidate: RotationCandidate | None = None
    refine_failure_count = 0
    last_refine_error: str | None = None

    for iteration in range(1, max_iterations + 1):
        sampled_pairs, sampling_mode = _sample_pair_subset(
            unique_pair_idx, sample_size, rng)

        try:
            candidate = build_rotation_candidate(
                pts1,
                pts2,
                vectors1,
                vectors2,
                unique_pair_idx,
                sampled_pairs,
                iteration=iteration,
                sample_size=len(sampled_pairs),
                sampling_mode=sampling_mode,
                config=config)
        except ValueError as exc:
            refine_failure_count += 1
            last_refine_error = str(exc)
            continue

        if (best_candidate is None or _rotation_candidate_rank(candidate)
                > _rotation_candidate_rank(best_candidate)):
            best_candidate = candidate

        if _is_high_quality_rotation_candidate(candidate, unique_pair_count,
                                               config):
            _log_rotation_candidate(candidate, "Rotation RANSAC selected: ",
                                    unique_pair_count, True)
            return candidate.rotation, candidate.pair_idx

    if best_candidate is not None:
        if best_candidate.accepted:
            _log_rotation_candidate(
                best_candidate, "Rotation RANSAC selected after full search: ",
                unique_pair_count, False)
            return best_candidate.rotation, best_candidate.pair_idx

        _log_rotation_candidate_profile(
            "rotation_ransac_best_candidate_inlier_residuals",
            pts1,
            best_candidate.pair_idx,
            best_candidate.angle_errors_rad,
        )
        _log_pair_distribution(
            "rotation_ransac_best_candidate_inliers",
            pts1,
            best_candidate.pair_idx,
        )
        raise ValueError(
            "Optimal fisheye rotation alignment cannot be achieved. "
            f"best_candidate_iteration={best_candidate.iteration}, "
            f"inliers={best_candidate.diagnostics.inlier_count}, "
            f"coverage_ratio={best_candidate.diagnostics.coverage_ratio:.4f}, "
            f"rejection_reason={best_candidate.rejection_reason}")

    raise ValueError(
        "Optimal fisheye rotation alignment cannot be achieved. "
        f"refine_failures={refine_failure_count}, "
        f"last_refine_error={last_refine_error}")
