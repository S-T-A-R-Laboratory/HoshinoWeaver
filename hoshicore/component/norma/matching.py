"""Star point feature extraction and matching."""
import dataclasses

import cv2
import numpy as np
import numpy.linalg as la
from loguru import logger
from numpy.typing import NDArray

from hoshicore._custom_op.ops.alignment import (
    extract_point_features as _extract_point_features_custom,
)
from hoshicore._custom_op.ops.alignment import (
    find_initial_match as _find_initial_match_custom,
)


@dataclasses.dataclass
class MatchResult:
    pair_idx: NDArray[np.int32]
    ref_pts: NDArray[np.float64]
    src_pts: NDArray[np.float64]
    init_homography: NDArray[np.float64]


@dataclasses.dataclass(frozen=True)
class HomographyDiagnostics:
    inlier_count: int
    median_reproj_error: float
    p90_reproj_error: float
    coverage_ratio: float
    area_ratio: float
    projective_magnitude: float
    is_flipped: bool


@dataclasses.dataclass
class HomographyCandidate:
    homography: NDArray[np.float64]
    pair_idx: NDArray[np.int32]
    diagnostics: HomographyDiagnostics
    accepted: bool
    rejection_reason: str | None
    iteration: int
    sample_size: int
    sampling_mode: str


@dataclasses.dataclass(frozen=True)
class HomographyValidationConfig:
    # 中位重投影误差
    max_reproj_median_px: float
    # P90 重投影误差
    max_reproj_p90_px: float
    # 内点覆盖率
    min_coverage_ratio: float
    # 面积比率（变形程度）
    min_area_ratio: float
    max_area_ratio: float
    # 投影分量幅度（透视变化程度）
    max_projective_magnitude: float


# Consecutive-frame near-neighbor matching (satellite removal, star trail stacking).
# Strict: no perspective, near-unit area ratio.
SATELLITE_VALIDATION = HomographyValidationConfig(
    max_reproj_median_px=1.0,
    max_reproj_p90_px=2.0,
    min_coverage_ratio=0.01,
    min_area_ratio=0.8,
    max_area_ratio=1.2,
    max_projective_magnitude=0.1,
)

# General frame alignment: large perspective, low overlap expected.
ALIGN_VALIDATION = HomographyValidationConfig(
    max_reproj_median_px=1.5,
    max_reproj_p90_px=3.0,
    min_coverage_ratio=0.005,
    min_area_ratio=0.4,
    max_area_ratio=2.5,
    max_projective_magnitude=1.0,
)

MIN_HOMOGRAPHY_INLIERS = 4
HOMOGRAPHY_RANSAC_REPROJ_THRESHOLD = 5.0
FULL_SAMPLE_PAIR_LIMIT = 6
MAX_SAMPLE_SIZE = 12
MAX_HOMOGRAPHY_TRIALS = 10
MIN_FILTERED_UNIQUE_PAIRS = 4
LOW_PAIR_COUNT_THRESHOLD = 10
MIN_FILTER_KEEP_RATIO = 0.5


def _perspective_transform(pts: NDArray[np.float64],
                           H: NDArray[np.float64]) -> NDArray[np.float64]:
    """Apply homography to Nx2 points."""
    transformed = cv2.perspectiveTransform(
        np.asarray([[p] for p in pts], dtype=np.float32),
        np.asarray(H, dtype=np.float64))
    return transformed[:, 0, :].astype(np.float64)


def _polygon_signed_area(pts: NDArray[np.float64]) -> float:
    """Shoelace signed area for polygon points in order."""
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _convex_hull_area(pts: NDArray[np.float64]) -> float:
    """Convex hull area for a point cloud."""
    if len(pts) < 3:
        return 0.0
    hull = cv2.convexHull(np.asarray(pts, dtype=np.float32))
    return float(abs(cv2.contourArea(hull)))


def _make_canvas_extent(
    pts1: NDArray[np.float64], pts2: NDArray[np.float64]
) -> tuple[NDArray[np.float64], float, float]:
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
    ], dtype=np.float64)
    canvas_area = float(span[0] * span[1])
    canvas_diag = float(la.norm(span))
    return corners, canvas_area, canvas_diag


def evaluate_homography(
    pts1: NDArray[np.float64],
    pts2: NDArray[np.float64],
    pair_idx: NDArray[np.int32],
    H: NDArray[np.float64],
) -> HomographyDiagnostics:
    """Compute quality metrics for a candidate homography."""
    ref_pts = pts1[pair_idx[:, 0]]
    src_pts = pts2[pair_idx[:, 1]]
    reproj_pts = _perspective_transform(ref_pts, H)
    reproj_errors = la.norm(reproj_pts - src_pts, axis=1)

    corners, canvas_area, canvas_diag = _make_canvas_extent(pts1, pts2)
    warped_corners = _perspective_transform(corners, H)
    signed_area = _polygon_signed_area(warped_corners)
    area_ratio = abs(signed_area) / max(canvas_area, 1.0)
    coverage_ratio = min(_convex_hull_area(ref_pts), _convex_hull_area(src_pts))
    coverage_ratio = coverage_ratio / max(canvas_area, 1.0)
    projective_magnitude = float(la.norm(H[2, :2]) * max(canvas_diag, 1.0))

    return HomographyDiagnostics(
        inlier_count=int(len(pair_idx)),
        median_reproj_error=float(np.median(reproj_errors)),
        p90_reproj_error=float(np.percentile(reproj_errors, 90)),
        coverage_ratio=float(coverage_ratio),
        area_ratio=float(area_ratio),
        projective_magnitude=projective_magnitude,
        is_flipped=bool(signed_area <= 0),
    )


def validate_homography(
    pts1: NDArray[np.float64],
    pts2: NDArray[np.float64],
    pair_idx: NDArray[np.int32],
    H: NDArray[np.float64],
    config: HomographyValidationConfig = SATELLITE_VALIDATION,
) -> HomographyDiagnostics:
    """Validate a candidate homography and raise on obvious bad solutions."""
    diagnostics = evaluate_homography(pts1, pts2, pair_idx, H)
    reject_reasons = []

    if not np.all(np.isfinite(H)):
        reject_reasons.append("homography contains non-finite values")
    if diagnostics.median_reproj_error > config.max_reproj_median_px:
        reject_reasons.append(
            f"median_reproj={diagnostics.median_reproj_error:.3f}px > {config.max_reproj_median_px:.3f}px")
    if diagnostics.p90_reproj_error > config.max_reproj_p90_px:
        reject_reasons.append(
            f"p90_reproj={diagnostics.p90_reproj_error:.3f}px > {config.max_reproj_p90_px:.3f}px")
    if diagnostics.coverage_ratio < config.min_coverage_ratio:
        reject_reasons.append(
            f"coverage_ratio={diagnostics.coverage_ratio:.4f} < {config.min_coverage_ratio:.4f}")
    if diagnostics.is_flipped:
        reject_reasons.append("warped canvas is flipped")
    if not config.min_area_ratio <= diagnostics.area_ratio <= config.max_area_ratio:
        reject_reasons.append(
            f"area_ratio={diagnostics.area_ratio:.3f} not in [{config.min_area_ratio:.3f}, {config.max_area_ratio:.3f}]")
    if diagnostics.projective_magnitude > config.max_projective_magnitude:
        reject_reasons.append(
            f"projective_magnitude={diagnostics.projective_magnitude:.4f} > {config.max_projective_magnitude:.4f}")

    if reject_reasons:
        raise ValueError("Homography rejected: " + "; ".join(reject_reasons))
    return diagnostics


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


def _build_homography_candidate(
    pts1: NDArray[np.float64],
    pts2: NDArray[np.float64],
    all_pair_idx: NDArray[np.int32],
    sampled_pair_idx: NDArray[np.int32],
    iteration: int,
    sample_size: int,
    sampling_mode: str,
    config: HomographyValidationConfig = SATELLITE_VALIDATION,
) -> HomographyCandidate:
    """Fit and evaluate one candidate homography."""
    if len(sampled_pair_idx) < MIN_HOMOGRAPHY_INLIERS:
        raise ValueError(
            f"sampled pair count below homography minimum: {len(sampled_pair_idx)}")

    tf = cv2.findHomography(
        pts1[sampled_pair_idx[:, 0]],
        pts2[sampled_pair_idx[:, 1]],
        method=cv2.RANSAC,
        ransacReprojThreshold=HOMOGRAPHY_RANSAC_REPROJ_THRESHOLD)
    if tf[0] is None:
        raise ValueError("sampled findHomography returned None")

    projected_ref = _perspective_transform(pts1[all_pair_idx[:, 0]], tf[0])
    reproj_errors = la.norm(projected_ref - pts2[all_pair_idx[:, 1]], axis=1)
    inlier_pair_idx = all_pair_idx[
        reproj_errors < HOMOGRAPHY_RANSAC_REPROJ_THRESHOLD]
    if len(inlier_pair_idx) < MIN_HOMOGRAPHY_INLIERS:
        raise ValueError(
            f"inlier count below homography minimum: {len(inlier_pair_idx)}")

    refined_tf = cv2.findHomography(
        pts1[inlier_pair_idx[:, 0]],
        pts2[inlier_pair_idx[:, 1]],
        method=cv2.RANSAC,
        ransacReprojThreshold=HOMOGRAPHY_RANSAC_REPROJ_THRESHOLD)
    if refined_tf[0] is None:
        raise ValueError("refined findHomography returned None")

    refined_pair_idx = inlier_pair_idx
    if refined_tf[1] is not None:
        refined_mask = refined_tf[1].ravel().astype(bool)
        refined_inlier_pair_idx = inlier_pair_idx[refined_mask]
        if len(refined_inlier_pair_idx) >= MIN_HOMOGRAPHY_INLIERS:
            refined_pair_idx = refined_inlier_pair_idx

    diagnostics = evaluate_homography(
        pts1, pts2, refined_pair_idx, refined_tf[0])
    accepted = True
    rejection_reason = None
    try:
        validate_homography(pts1, pts2, refined_pair_idx, refined_tf[0], config)
    except ValueError as exc:
        accepted = False
        rejection_reason = str(exc)

    return HomographyCandidate(
        homography=refined_tf[0],
        pair_idx=refined_pair_idx,
        diagnostics=diagnostics,
        accepted=accepted,
        rejection_reason=rejection_reason,
        iteration=iteration,
        sample_size=sample_size,
        sampling_mode=sampling_mode,
    )


def _candidate_rank(candidate: HomographyCandidate) -> tuple[float, ...]:
    """Ranking for diagnostics-only best-candidate tracking."""
    return (
        1.0 if candidate.accepted else 0.0,
        float(candidate.diagnostics.inlier_count),
        -candidate.diagnostics.median_reproj_error,
        candidate.diagnostics.coverage_ratio,
        -candidate.diagnostics.p90_reproj_error,
        -candidate.diagnostics.projective_magnitude,
    )


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
        return (True,
                f"unique_pairs={unique_after_count} < {MIN_FILTERED_UNIQUE_PAIRS}")
    if (before_count < LOW_PAIR_COUNT_THRESHOLD and
            kept_ratio < MIN_FILTER_KEEP_RATIO):
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
    """Extract geometric features for each star point based on neighbor relationships.

    Args:
        vec: (n, 3) unit vectors of star points.
        vol: (n,) volume (area * intensity) of each star point.
        k: number of neighbors to use.

    Returns:
        (n, 120) feature matrix.
    """
    return _extract_point_features_custom(vec, vol, k)


def find_initial_match(features1: NDArray[np.float64],
                       features2: NDArray[np.float64],
                       pts1: NDArray[np.float64],
                       pts2: NDArray[np.float64],
                       vectors1: NDArray[np.float64] = None,
                       vectors2: NDArray[np.float64] = None,
                       alpha: float = 0.00,
                       apply_threshold_filter: bool = True,
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
    return _find_initial_match_custom(
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


def fine_tune_transform(
        pts1: NDArray[np.float64], pts2: NDArray[np.float64],
        init_pair_idx: NDArray[np.int32],
        config: HomographyValidationConfig = SATELLITE_VALIDATION,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Refine matching using RANSAC homography.

    Returns:
        (homography_matrix, refined_pair_idx)
    """
    unique_pair_idx = np.unique(init_pair_idx, axis=0).astype(np.int32, copy=False)
    unique_pair_count = len(unique_pair_idx)
    if unique_pair_count < MIN_HOMOGRAPHY_INLIERS:
        raise ValueError(
            f"findHomography requires at least {MIN_HOMOGRAPHY_INLIERS} unique point pairs, "
            f"got {unique_pair_count}")

    if unique_pair_count <= FULL_SAMPLE_PAIR_LIMIT:
        sample_size = unique_pair_count
    else:
        sample_size = min(MAX_SAMPLE_SIZE, unique_pair_count)
    max_iterations = 1 if unique_pair_count <= sample_size else MAX_HOMOGRAPHY_TRIALS

    logger.debug(
        "Fine-tune setup: init_pairs={}, unique_pairs={}, sample_size={}, "
        "max_iterations={}",
        len(init_pair_idx), unique_pair_count, sample_size,
        max_iterations)

    rng = np.random.default_rng()
    best_candidate: HomographyCandidate | None = None

    for iteration in range(1, max_iterations + 1):
        sampled_pairs, sampling_mode = _sample_pair_subset(
            unique_pair_idx, sample_size, rng)
        logger.debug(
            "Fine-tune iteration {}: unique_pairs={}, sampled_pairs={}, "
            "sampling_mode={}",
            iteration, unique_pair_count, len(sampled_pairs), sampling_mode)

        try:
            candidate = _build_homography_candidate(
                pts1, pts2, unique_pair_idx, sampled_pairs,
                iteration=iteration,
                sample_size=len(sampled_pairs),
                sampling_mode=sampling_mode,
                config=config)
        except ValueError as exc:
            logger.warning(
                "Fine-tune iteration {} rejected before validation: {}",
                iteration, exc)
            continue

        if best_candidate is None or _candidate_rank(candidate) > _candidate_rank(best_candidate):
            best_candidate = candidate

        logger.debug(
            "Homography diagnostics: iteration={}, inliers={}, inlier_ratio={:.3f}, "
            "median_reproj={:.3f}px, p90_reproj={:.3f}px, coverage_ratio={:.4f}, "
            "area_ratio={:.3f}, projective_magnitude={:.4f}, flipped={}, "
            "accepted={}, rejection_reason={}",
            iteration,
            candidate.diagnostics.inlier_count,
            candidate.diagnostics.inlier_count / unique_pair_count,
            candidate.diagnostics.median_reproj_error,
            candidate.diagnostics.p90_reproj_error,
            candidate.diagnostics.coverage_ratio,
            candidate.diagnostics.area_ratio,
            candidate.diagnostics.projective_magnitude,
            candidate.diagnostics.is_flipped,
            candidate.accepted,
            candidate.rejection_reason)

        if candidate.accepted:
            logger.debug(
                "Fine-tune early stop: accepted_iteration={}, early_stop_triggered={}, "
                "sample_size={}, sampling_mode={}",
                iteration, True, candidate.sample_size, candidate.sampling_mode)
            return candidate.homography, candidate.pair_idx

    if best_candidate is not None:
        raise ValueError(
            "Optimal alignment cannot be achieved. "
            f"best_candidate_iteration={best_candidate.iteration}, "
            f"inliers={best_candidate.diagnostics.inlier_count}, "
            f"median_reproj={best_candidate.diagnostics.median_reproj_error:.3f}px, "
            f"coverage_ratio={best_candidate.diagnostics.coverage_ratio:.4f}, "
            f"rejection_reason={best_candidate.rejection_reason}")

    raise ValueError("Optimal alignment cannot be achieved.")
