"""单帧对齐：封装两条路径的纯函数 API。
"""
import dataclasses
from functools import partial
from time import perf_counter
from typing import Callable, Optional

import cv2
import numpy as np
from loguru import logger

from .alignment import (AlignmentResult, match_star_pairs,
                         match_star_pairs_asterism, optimize_alignment,
                         run_guided_refine_stage)
from .detection import DetectedStars
from .matching import MatchResult, RotationDiagnostics, evaluate_rotation
from .optimization import (CameraOptimizationPolicy,
                           DEFAULT_PRINCIPAL_POINT_OFFSET_LIMIT,
                           LARGE_PRINCIPAL_POINT_OFFSET_LIMIT)
from .geometry_view import GeometryView, StarDetectionCache
from .intrinsics_from_exif import (intrinsics_from_exif,
                                   intrinsics_from_focal_equiv,
                                   intrinsics_from_fisheye_estimate)
from .types import (BaseCameraModel, CameraModel, Distortion,
                    FisheyeCameraModel, FisheyeDistortion)


class AlignmentError(Exception):
    """对齐失败异常。"""
    pass


DEFAULT_BOOTSTRAP_SCALES = (0.7, 1.0, 1.3)
MATCHING_PATH_MEDIAN = "median"
MATCHING_PATH_HISTOGRAM = "histogram"
MATCHING_PATH_ASTERISM = "asterism"
DEFAULT_MATCHING_PATH = MATCHING_PATH_ASTERISM
MATCHING_PATHS = (
    MATCHING_PATH_MEDIAN,
    MATCHING_PATH_HISTOGRAM,
    MATCHING_PATH_ASTERISM,
)
BOOTSTRAP_MATCHING_PATHS = (
    MATCHING_PATH_HISTOGRAM,
    MATCHING_PATH_ASTERISM,
)


def _has_identical_star_geometry(ref_geo: GeometryView,
                                 src_geo: GeometryView) -> bool:
    """Conservatively recognize the same detected frame from geometry alone."""
    if (ref_geo.positions.shape != src_geo.positions.shape
            or not np.array_equal(ref_geo.positions, src_geo.positions)):
        return False
    if (ref_geo.volumes.shape != src_geo.volumes.shape
            or not np.array_equal(ref_geo.volumes, src_geo.volumes)):
        return False
    return (ref_geo.features.shape == src_geo.features.shape
            and np.array_equal(ref_geo.features, src_geo.features))


@dataclasses.dataclass(frozen=True)
class CameraInitializationPolicy:
    lens_type: Optional[str] = None
    fallback_focal_equiv_mm: float = 20.0
    optimize_focal: Optional[bool] = None
    optimize_distortion: Optional[bool] = None
    optimize_principal_point: Optional[bool] = None
    allow_large_principal_point_offset: bool = False


@dataclasses.dataclass(frozen=True)
class AlignmentCameraCandidate:
    camera: BaseCameraModel
    optimization_policy: CameraOptimizationPolicy
    init_source: str
    scale: float = 1.0


@dataclasses.dataclass(frozen=True)
class StagedSolveResult:
    """Bootstrap result plus an optional independently detected refine stage."""

    bootstrap_ref: GeometryView
    bootstrap_src: GeometryView
    refine_ref: Optional[GeometryView]
    refine_src: Optional[GeometryView]
    bootstrap_match: MatchResult
    final_match: MatchResult
    final_alignment: AlignmentResult
    refine_status: str
    refine_error: Optional[str]
    timings: dict[str, float]


def build_camera(
    exif_tags: Optional[dict[str, str]],
    img_shape: tuple,
    method: str,
    init_distortion: Optional[list] = None,
    lens_type: Optional[str] = None,
    focal_equiv_mm: Optional[float] = None,
    fallback_focal_equiv_mm: float = 20.0,
) -> BaseCameraModel:
    """尝试构建 CameraModel 或 FisheyeCameraModel。

    内参解析优先级（高→低）：
      1. EXIF（FocalLength + FocalPlaneResolution）
      2. 手动参数 focal_equiv_mm（35mm 等效焦距）
      3. 兜底：鱼眼假设 FOV=180° ，直线镜头35mm 等效焦距 Fallback

    lens_type="fisheye" 时返回 FisheyeCameraModel（零系数=等距起点）。
    """
    if method not in ("distortion", "homography"):
        raise ValueError(f"Unsupported alignment method {method!r}; expected "
                         "'distortion' or 'homography'")

    h, w = img_shape[:2]

    # 1. 优先从 EXIF 解析
    intrinsics = intrinsics_from_exif(exif_tags, w, h) if exif_tags else None

    # 2. EXIF 失败时尝试手动焦距
    if intrinsics is None and focal_equiv_mm is not None:
        intrinsics = intrinsics_from_focal_equiv(focal_equiv_mm, w, h)
        logger.debug(
            f"build_camera: using manual focal_equiv={focal_equiv_mm:.1f}mm")

    # 3. 兜底
    if lens_type == "fisheye":
        if method == "homography":
            logger.warning(
                "Applying homography method for a fisheye lens. The result may "
                "be inaccurate. Consider using 'camera_model' method instead.")
        else:
            if intrinsics is None:
                intrinsics = intrinsics_from_fisheye_estimate(w, h)
                logger.warning(
                    f"build_camera: fisheye but no EXIF or focal_length_mm; "
                    f"estimating K from 180° FOV "
                    f"(focal_equiv≈{intrinsics.focal_length_mm:.1f}mm). "
                    f"Provide focal_length_mm for better accuracy.")
            dist = FisheyeDistortion.from_array(
                init_distortion) if init_distortion else FisheyeDistortion()
            return FisheyeCameraModel(intrinsics=intrinsics, distortion=dist)

    if intrinsics is None:
        intrinsics = intrinsics_from_focal_equiv(fallback_focal_equiv_mm, w, h)
        if intrinsics is not None:
            logger.debug(
                f"build_camera: using perspective fallback focal_equiv={fallback_focal_equiv_mm:.1f}mm"
            )

    dist = Distortion.from_cv2(init_distortion) if (
        init_distortion and method != "homography") else Distortion()
    return CameraModel(intrinsics=intrinsics, distortion=dist)


def _camera_init_source(
    exif_tags: Optional[dict[str, str]],
    img_shape: tuple,
    lens_type: Optional[str],
    focal_equiv_mm: Optional[float],
    fallback_focal_equiv_mm: Optional[float],
) -> str:
    h, w = img_shape[:2]
    if exif_tags and intrinsics_from_exif(exif_tags, w, h) is not None:
        return "exif"
    if focal_equiv_mm is not None:
        return "manual"
    if lens_type == "fisheye":
        return "fisheye_estimate"
    if fallback_focal_equiv_mm is not None:
        return "fallback_focal"
    return "none"


def _policy_for_camera(
    init_source: str,
    init_policy: CameraInitializationPolicy,
) -> CameraOptimizationPolicy:
    is_ideal = init_policy.lens_type == "ideal"
    default_optimize_focal = True
    # A synthetic fallback focal is deliberately conservative: allow focal
    # recovery, but do not let distortion absorb a poor focal initial value.
    default_optimize_distortion = (not is_ideal
                                   and init_source != "fallback_focal")
    optimize_focal = (default_optimize_focal if init_policy.optimize_focal
                      is None else init_policy.optimize_focal)
    optimize_distortion = (default_optimize_distortion
                           if init_policy.optimize_distortion is None else
                           init_policy.optimize_distortion)
    optimize_principal_point = (
        True if init_policy.optimize_principal_point is None
        else init_policy.optimize_principal_point)
    return CameraOptimizationPolicy(
        optimize_focal=optimize_focal,
        optimize_distortion=optimize_distortion,
        optimize_principal_point=optimize_principal_point,
        # The fourth Kannala--Brandt term is weakly identifiable for the
        # typical star-field coverage. Keep it fixed by default and optimize
        # k1..k3; perspective cameras retain their four-parameter policy.
        n_dist=3 if init_policy.lens_type == "fisheye" else 4,
        principal_point_offset_limit=(
            LARGE_PRINCIPAL_POINT_OFFSET_LIMIT
            if init_policy.allow_large_principal_point_offset
            else DEFAULT_PRINCIPAL_POINT_OFFSET_LIMIT),
    )


def build_camera_candidate(
        exif_tags: Optional[dict[str, str]],
        img_shape: tuple,
        method: str,
        init_distortion: Optional[list] = None,
        focal_equiv_mm: Optional[float] = None,
        init_policy: CameraInitializationPolicy = CameraInitializationPolicy(),
):
    camera = build_camera(
        exif_tags,
        img_shape,
        method,
        init_distortion,
        init_policy.lens_type,
        focal_equiv_mm,
        init_policy.fallback_focal_equiv_mm,
    )
    source = _camera_init_source(
        exif_tags,
        img_shape,
        init_policy.lens_type,
        focal_equiv_mm,
        init_policy.fallback_focal_equiv_mm,
    )
    if method == "homography":
        policy = CameraOptimizationPolicy(
            optimize_focal=False,
            optimize_distortion=False,
            optimize_principal_point=False,
            n_dist=0,
        )
    else:
        policy = _policy_for_camera(source, init_policy)
    return AlignmentCameraCandidate(camera=camera,
                                    optimization_policy=policy,
                                    init_source=source)


def _check_star_count(ref_geo: GeometryView,
                      src_geo: GeometryView,
                      min_stars: int = 20) -> None:
    """检查星点数量是否满足对齐要求。"""
    if len(ref_geo.positions) < min_stars or len(
            src_geo.positions) < min_stars:
        raise AlignmentError(
            f"Insufficient stars: ref={len(ref_geo.positions)}, "
            f"src={len(src_geo.positions)} (need >={min_stars})")


def _log_rotation_diagnostics(prefix: str, rotation: np.ndarray) -> None:
    rvec, _ = cv2.Rodrigues(rotation)
    rvec = rvec[:, 0]
    angle_deg = float(np.linalg.norm(rvec)) * 180.0 / np.pi
    logger.debug("{}: angle_deg={:.6f} rvec=({:.6f},{:.6f},{:.6f})", prefix,
                 angle_deg, rvec[0], rvec[1], rvec[2])


def _compute_radial_zone_stats(
    pts_ref: np.ndarray,
    residuals_px: np.ndarray,
    reference_shape: tuple[int, ...],
) -> list[dict[str, float | int | str]]:
    h, w = reference_shape[:2]
    cx = w / 2.0
    cy = h / 2.0
    radii = np.sqrt((pts_ref[:, 0] - cx)**2 + (pts_ref[:, 1] - cy)**2)
    max_radius = float(np.sqrt(cx * cx + cy * cy))
    if max_radius <= 0:
        max_radius = 1.0
    r_norm = radii / max_radius

    bins = [
        ("inner", 0.0, 0.33),
        ("mid", 0.33, 0.66),
        ("outer", 0.66, np.inf),
    ]
    stats: list[dict[str, float | int | str]] = []
    for name, lo, hi in bins:
        mask = (r_norm >= lo) & (r_norm < hi)
        if not np.any(mask):
            stats.append({"zone": name, "count": 0})
            continue
        err = residuals_px[mask]
        stats.append({
            "zone": name,
            "count": int(mask.sum()),
            "median": float(np.median(err)),
            "p90": float(np.percentile(err, 90)),
            "max": float(np.max(err)),
        })
    return stats


def _log_residual_diagnostics(
    prefix: str,
    pts_ref: np.ndarray,
    pts_src: np.ndarray,
    pts_src_pred: np.ndarray,
    reference_shape: tuple[int, ...],
) -> None:
    if len(pts_ref) == 0:
        logger.debug(f"{prefix}: no matched points")
        return

    residual_vec = pts_src_pred - pts_src
    residuals_px = np.linalg.norm(residual_vec, axis=1)
    zones = _compute_radial_zone_stats(pts_ref, residuals_px, reference_shape)
    zone_summary = ", ".join(f"{stat['zone']}={stat['count']}:"
                             f"{stat.get('median', float('nan')):.3f}/"
                             f"{stat.get('p90', float('nan')):.3f}px"
                             for stat in zones)
    logger.debug(
        "{}: count={} median={:.3f}px p90={:.3f}px max={:.3f}px zones[{}]",
        prefix, len(residuals_px), np.median(residuals_px),
        np.percentile(residuals_px, 90), np.max(residuals_px), zone_summary)


def _count_match_radial_bins(
    ref_geo: GeometryView,
    pair_idx: np.ndarray,
) -> tuple[int, int, int]:
    if len(pair_idx) == 0:
        return 0, 0, 0
    pts_ref = ref_geo.positions[pair_idx[:, 0]]
    image_h, image_w = ref_geo.camera.intrinsics.image_height_px, ref_geo.camera.intrinsics.image_width_px
    cx = image_w / 2.0
    cy = image_h / 2.0
    radii = np.sqrt((pts_ref[:, 0] - cx)**2 + (pts_ref[:, 1] - cy)**2)
    max_radius = float(np.sqrt(cx * cx + cy * cy))
    if max_radius <= 0:
        max_radius = 1.0
    r_norm = radii / max_radius
    inner = int(np.count_nonzero(r_norm < 0.33))
    mid = int(np.count_nonzero((r_norm >= 0.33) & (r_norm < 0.66)))
    outer = int(np.count_nonzero(r_norm >= 0.66))
    return inner, mid, outer


def _count_match_active_sectors(
    ref_geo: GeometryView,
    pair_idx: np.ndarray,
    num_sectors: int = 8,
) -> int:
    if len(pair_idx) == 0:
        return 0
    pts_ref = ref_geo.positions[pair_idx[:, 0]]
    image_h = ref_geo.camera.intrinsics.image_height_px
    image_w = ref_geo.camera.intrinsics.image_width_px
    cx = image_w / 2.0
    cy = image_h / 2.0
    ang = (np.degrees(np.arctan2(pts_ref[:, 1] - cy, pts_ref[:, 0] - cx)) +
           360.0) % 360.0
    sector_width = 360.0 / num_sectors
    sector_ids = np.floor(ang / sector_width).astype(np.int32)
    sector_ids = np.clip(sector_ids, 0, num_sectors - 1)
    return int(np.unique(sector_ids).size)


def _candidate_family(
    candidate: AlignmentCameraCandidate,
    scales: tuple[float, ...],
) -> list[AlignmentCameraCandidate]:
    if not scales:
        scales = (1.0, )
    return [
        dataclasses.replace(
            candidate,
            camera=candidate.camera.with_focal_length(
                candidate.camera.intrinsics.focal_length_mm * scale),
            scale=scale,
        ) for scale in scales
    ]


def _candidate_scales(
    candidate: AlignmentCameraCandidate,
    bootstrap_scales: tuple[float, ...],
) -> tuple[float, ...]:
    """Choose focal search scales from projection type and init source."""
    if isinstance(candidate.camera, FisheyeCameraModel):
        return bootstrap_scales
    if candidate.init_source in ("exif", "provided"):
        return (1.0, )
    return bootstrap_scales


def _bootstrap_candidate_score(
    diagnostics: RotationDiagnostics,
    match_count: int,
    outer_count: int,
    active_sectors: int,
    ref_scale: float,
    src_scale: float,
) -> tuple[float, dict[str, float]]:
    """Score a focal candidate using bounded quality and coverage utilities."""
    p90_scale = np.deg2rad(0.25)
    median_scale = np.deg2rad(0.10)
    p90_score = float(np.exp(-(diagnostics.p90_angle_error_rad / p90_scale)**2))
    median_score = float(
        np.exp(-(diagnostics.median_angle_error_rad / median_scale)**2))
    coverage_score = float(1.0 - np.exp(-diagnostics.coverage_ratio / 0.10))
    # Counts saturate deliberately: after a few hundred well-distributed
    # inliers, more central points add little camera-model information.
    match_score = float(1.0 - np.exp(-max(match_count, 0) / 256.0))
    outer_score = float(1.0 - np.exp(-max(outer_count, 0) / 32.0))
    sector_score = float(np.clip(active_sectors / 8.0, 0.0, 1.0))
    focal_score = float(
        np.exp(-(abs(ref_scale - 1.0) + abs(src_scale - 1.0)) / 0.30))
    components = {
        "p90": p90_score,
        "median": median_score,
        "coverage": coverage_score,
        "matches": match_score,
        "outer": outer_score,
        "sectors": sector_score,
        "focal": focal_score,
    }
    # Every candidate has already passed the rotation acceptance gates.  The
    # ranking therefore balances residual quality with how well the selected
    # points constrain the image, instead of allowing P90 alone to dominate.
    score = (
        0.30 * p90_score
        + 0.15 * median_score
        + 0.20 * coverage_score
        + 0.10 * match_score
        + 0.10 * outer_score
        + 0.10 * sector_score
        + 0.05 * focal_score
    )
    return score, components


def _bootstrap_candidate_pairs(
    ref_family: list[AlignmentCameraCandidate],
    src_family: list[AlignmentCameraCandidate],
    same_camera: bool,
) -> list[tuple[AlignmentCameraCandidate, AlignmentCameraCandidate]]:
    if not same_camera:
        return [
            (cand_ref, cand_src)
            for cand_ref in ref_family
            for cand_src in src_family
        ]
    src_by_scale = {candidate.scale: candidate for candidate in src_family}
    return [
        (candidate, src_by_scale[candidate.scale])
        for candidate in ref_family
        if candidate.scale in src_by_scale
    ]


def _select_initial_alignment_candidate(
    ref_geo: GeometryView,
    src_geo: GeometryView,
    ref_candidate: AlignmentCameraCandidate,
    src_candidate: AlignmentCameraCandidate,
    bootstrap_scales: tuple[float, ...],
    same_camera: bool = False,
    match_function: Callable[[GeometryView, GeometryView], MatchResult] =
    match_star_pairs,
):
    ref_scales = _candidate_scales(ref_candidate, bootstrap_scales)
    src_scales = _candidate_scales(src_candidate, bootstrap_scales)
    ref_family = _candidate_family(ref_candidate, ref_scales)
    src_family = _candidate_family(src_candidate, src_scales)
    best_payload = None
    best_score = None

    logger.debug(
        "Camera candidate search: ref_source={} src_source={} "
        "ref_scales={} src_scales={}",
        ref_candidate.init_source,
        src_candidate.init_source,
        ref_scales,
        src_scales,
    )

    candidate_pairs = _bootstrap_candidate_pairs(
        ref_family, src_family, same_camera)
    logger.debug(
        "Camera candidate combinations: same_camera={} count={}",
        same_camera,
        len(candidate_pairs),
    )

    for cand_ref, cand_src in candidate_pairs:
        cand_ref_geo = ref_geo.with_camera(cand_ref.camera)
        cand_src_geo = src_geo.with_camera(cand_src.camera)

        try:
            cand_match = match_function(cand_ref_geo, cand_src_geo)
        except Exception as exc:
            logger.debug(
                "Camera candidate failed: ref_scale={:.3f} src_scale={:.3f} "
                "ref_projection={} src_projection={} failed={}",
                cand_ref.scale,
                cand_src.scale,
                "fisheye"
                if isinstance(cand_ref.camera,
                              FisheyeCameraModel) else "perspective",
                "fisheye"
                if isinstance(cand_src.camera, FisheyeCameraModel) else
                "perspective",
                exc,
            )
            continue

        inner, mid, outer = _count_match_radial_bins(
            cand_ref_geo, cand_match.pair_idx)
        sectors = _count_match_active_sectors(cand_ref_geo,
                                              cand_match.pair_idx)
        diagnostics = evaluate_rotation(
            cand_ref_geo.positions,
            cand_src_geo.positions,
            cand_match.pair_idx,
            cand_ref_geo.unit_vectors,
            cand_src_geo.unit_vectors,
            cand_match.rotation,
        )
        score, score_parts = _bootstrap_candidate_score(
            diagnostics,
            len(cand_match.pair_idx),
            outer,
            sectors,
            cand_ref.scale,
            cand_src.scale,
        )
        logger.debug(
            "Camera candidate: ref_scale={:.3f} src_scale={:.3f} "
            "ref_focal={:.6f}mm src_focal={:.6f}mm matches={} radial={}/{}/{} sectors={}/8 "
            "median_angle={:.4f}deg p90_angle={:.4f}deg coverage_ratio={:.4f} "
            "score={:.6f} score_parts={}",
            cand_ref.scale,
            cand_src.scale,
            cand_ref.camera.intrinsics.focal_length_mm,
            cand_src.camera.intrinsics.focal_length_mm,
            len(cand_match.pair_idx),
            inner,
            mid,
            outer,
            sectors,
            np.rad2deg(diagnostics.median_angle_error_rad),
            np.rad2deg(diagnostics.p90_angle_error_rad),
            diagnostics.coverage_ratio,
            score,
            score_parts,
        )

        if best_score is None or score > best_score:
            best_score = score
            best_payload = (
                cand_ref_geo,
                cand_src_geo,
                cand_ref,
                cand_src,
                cand_match,
                inner,
                mid,
                outer,
                sectors,
            )

    if best_payload is None:
        raise AlignmentError(
            "Camera candidate search failed for all focal scales")

    (best_ref_geo, best_src_geo, best_ref_candidate, best_src_candidate,
     best_match, inner, mid, outer, sectors) = best_payload
    logger.debug(
        "Camera candidate selected: ref_source={} src_source={} "
        "ref_scale={:.3f} src_scale={:.3f} "
        "ref_focal={:.6f}mm src_focal={:.6f}mm matches={} radial={}/{}/{} sectors={}/8 "
        "ref_policy={} src_policy={}",
        best_ref_candidate.init_source,
        best_src_candidate.init_source,
        best_ref_candidate.scale,
        best_src_candidate.scale,
        best_ref_candidate.camera.intrinsics.focal_length_mm,
        best_src_candidate.camera.intrinsics.focal_length_mm,
        len(best_match.pair_idx),
        inner,
        mid,
        outer,
        sectors,
        best_ref_candidate.optimization_policy,
        best_src_candidate.optimization_policy,
    )
    return best_ref_geo, best_src_geo, best_ref_candidate, best_src_candidate, best_match


def _match_stars(
    ref_geo: GeometryView,
    src_geo: GeometryView,
    ref_candidate: AlignmentCameraCandidate,
    src_candidate: AlignmentCameraCandidate,
    bootstrap_scales: tuple[float, ...],
    same_camera: bool,
    use_asterism_bootstrap: bool,
    random_seed: int | None,
):
    """Shared star-only matching and camera-candidate selection."""
    _check_star_count(ref_geo, src_geo)
    match_function = (match_star_pairs_asterism
                      if use_asterism_bootstrap else match_star_pairs)
    if random_seed is not None:
        match_function = partial(match_function, random_seed=random_seed)
    (ref_geo, src_geo, ref_candidate, src_candidate,
     match) = _select_initial_alignment_candidate(
         ref_geo,
         src_geo,
         ref_candidate,
         src_candidate,
         bootstrap_scales,
         same_camera=same_camera,
         match_function=match_function,
     )
    return ref_geo, src_geo, ref_candidate, src_candidate, match


def solve_star_alignment(
    ref_stars: DetectedStars,
    src_stars: DetectedStars,
    ref_candidate: AlignmentCameraCandidate,
    src_candidate: AlignmentCameraCandidate,
    *,
    bootstrap_scales: tuple[float, ...] = DEFAULT_BOOTSTRAP_SCALES,
    same_camera: bool = False,
    use_asterism_bootstrap: bool = True,
    random_seed: int | None = None,
    residual_space: str = "cross",
) -> tuple[AlignmentResult, MatchResult]:
    """Align pre-detected stars without retaining source image pixels."""
    ref_geo = GeometryView(ref_stars, ref_candidate.camera)
    src_geo = GeometryView(src_stars, src_candidate.camera)
    *_, selected_ref, selected_src, match = _match_stars(
        ref_geo,
        src_geo,
        ref_candidate,
        src_candidate,
        bootstrap_scales,
        same_camera,
        use_asterism_bootstrap,
        random_seed,
    )
    alignment = optimize_alignment(
        match,
        selected_ref.camera,
        selected_src.camera,
        same_camera=same_camera,
        ref_policy=selected_ref.optimization_policy,
        src_policy=selected_src.optimization_policy,
        residual_space=residual_space,
    )
    return alignment, match


def solve_staged_alignment(
    ref_bootstrap_stars: DetectedStars,
    src_bootstrap_stars: DetectedStars,
    ref_candidate: AlignmentCameraCandidate,
    src_candidate: AlignmentCameraCandidate,
    bootstrap_scales: tuple[float, ...] = DEFAULT_BOOTSTRAP_SCALES,
    same_camera: bool = False,
    guided_refine: bool = False,
    guided_refine_radius_px: float = 8.0,
    use_asterism_bootstrap: bool = False,
    ref_refine_stars: Optional[DetectedStars] = None,
    src_refine_stars: Optional[DetectedStars] = None,
    random_seed: int | None = None,
    residual_space: str = "cross",
) -> StagedSolveResult:
    """Solve from explicit bootstrap and optional refine star sets.

    Detection belongs to the caller. The two stages keep separate GeometryView
    instances and therefore separate point/feature index spaces.
    """
    timings: dict[str, float] = {}
    bootstrap_ref = GeometryView(ref_bootstrap_stars, ref_candidate.camera)
    bootstrap_src = GeometryView(src_bootstrap_stars, src_candidate.camera)
    _check_star_count(bootstrap_ref, bootstrap_src)
    started = perf_counter()
    (bootstrap_ref, bootstrap_src, ref_candidate, src_candidate,
     bootstrap_match) = _match_stars(
         bootstrap_ref,
         bootstrap_src,
         ref_candidate,
         src_candidate,
         bootstrap_scales,
         same_camera,
         use_asterism_bootstrap,
         random_seed,
    )
    timings["bootstrap_and_matching"] = perf_counter() - started

    started = perf_counter()
    bootstrap_alignment = optimize_alignment(
        bootstrap_match,
        ref_candidate.camera,
        src_candidate.camera,
        same_camera=same_camera,
        ref_policy=ref_candidate.optimization_policy,
        src_policy=src_candidate.optimization_policy,
        residual_space=residual_space,
    )
    timings["first_optimization"] = perf_counter() - started

    final_alignment = bootstrap_alignment
    final_match = bootstrap_match
    refine_ref: Optional[GeometryView] = None
    refine_src: Optional[GeometryView] = None
    refine_status = "disabled"
    refine_error: Optional[str] = None

    if guided_refine:
        if ref_refine_stars is None or src_refine_stars is None:
            raise ValueError(
                "guided refinement requires explicit refine star sets")
        try:
            refine_ref = GeometryView(
                ref_refine_stars, bootstrap_alignment.ref_camera)
            refine_src = GeometryView(
                src_refine_stars, bootstrap_alignment.src_camera)
            _check_star_count(refine_ref, refine_src)

            started = perf_counter()
            final_alignment, final_match, refine_status = run_guided_refine_stage(
                refine_ref,
                refine_src,
                bootstrap_alignment,
                bootstrap_match=bootstrap_match,
                same_camera=same_camera,
                max_distance_px=guided_refine_radius_px,
                ref_policy=ref_candidate.optimization_policy,
                src_policy=src_candidate.optimization_policy,
                residual_space=residual_space,
            )
            timings["guided_rematch"] = perf_counter() - started
        except Exception as exc:
            refine_status = "failed_fallback"
            refine_error = str(exc)
            logger.warning(
                "Guided refinement failed; keeping bootstrap result: {}",
                exc,
            )

    return StagedSolveResult(
        bootstrap_ref=bootstrap_ref,
        bootstrap_src=bootstrap_src,
        refine_ref=refine_ref,
        refine_src=refine_src,
        bootstrap_match=bootstrap_match,
        final_match=final_match,
        final_alignment=final_alignment,
        refine_status=refine_status,
        refine_error=refine_error,
        timings=timings,
    )


def align_frame_homography(
        frame: np.ndarray,
        ref_geo: GeometryView,
        reference: np.ndarray,
        fallback_focal_equiv_mm: float = 20.0,
        src_camera: BaseCameraModel | None = None) -> np.ndarray:
    """Fixed-camera fast path: unit-ray match → rotation-derived H → warpPerspective。

    Args:
        frame: 待对齐帧。
        ref_geo: 参考帧的 GeometryView（预计算，可复用）。
        reference: 参考帧原始数组（用于获取输出尺寸）。

    Returns:
        对齐后的图像数组。

    Raises:
        AlignmentError: 星点不足或匹配失败。
    """
    if src_camera is None:
        src_camera = build_camera(
            None, frame.shape, "homography",
            fallback_focal_equiv_mm=fallback_focal_equiv_mm)
    source_detection = StarDetectionCache.from_image(frame)
    src_geo = GeometryView(source_detection.median_stars, src_camera)
    _check_star_count(ref_geo, src_geo)
    if _has_identical_star_geometry(ref_geo, src_geo):
        logger.debug(
            "align_frame_homography: identical star geometry; bypassing matching and warp"
        )
        return frame.copy()

    try:
        match = match_star_pairs(ref_geo, src_geo)
    except Exception as e:
        raise AlignmentError(f"Star matching failed: {e}") from e

    h, w = reference.shape[:2]
    if match.homography is None:
        raise AlignmentError(
            "Rotation-derived homography is unavailable for this camera pair; "
            "use camera-model remap path instead.")
    H = np.linalg.inv(match.homography)
    pts_src_pred = cv2.perspectiveTransform(
        match.ref_pts[:, None, :].astype(np.float32),
        match.homography.astype(np.float64),
    )[:, 0, :].astype(np.float64)
    _log_residual_diagnostics(
        "align_frame_homography: residual",
        match.ref_pts,
        match.src_pts,
        pts_src_pred,
        reference.shape,
    )
    return cv2.warpPerspective(frame, H, (w, h))


def align_frame_camera_model(
    frame: np.ndarray,
    ref_geo: GeometryView,
    reference: np.ndarray,
    ref_candidate: AlignmentCameraCandidate,
    src_candidate: AlignmentCameraCandidate,
    same_camera: bool = True,
    bootstrap_scales: tuple[float, ...] = DEFAULT_BOOTSTRAP_SCALES,
    remap_map_scale: float = 0.5,
    guided_refine: bool = False,
    guided_refine_radius_px: float = 8.0,
    matching_path: str = DEFAULT_MATCHING_PATH,
    ref_refine_geo: Optional[GeometryView] = None,
) -> np.ndarray:
    """Camera-model alignment with explicit ref-to-src remap construction."""
    ref_camera = ref_candidate.camera
    src_camera = src_candidate.camera
    ref_projection = "fisheye" if isinstance(
        ref_camera, FisheyeCameraModel) else "perspective"
    src_projection = "fisheye" if isinstance(
        src_camera, FisheyeCameraModel) else "perspective"
    if same_camera and ref_projection != src_projection:
        logger.warning(
            "align_frame_camera_model: mixed projection cameras cannot use same_camera=True; forcing same_camera=False"
        )
        same_camera = False
    if matching_path not in MATCHING_PATHS:
        raise ValueError(
            f"Unknown matching_path {matching_path!r}; expected one of "
            f"{MATCHING_PATHS}")

    source_detection = StarDetectionCache.from_image(frame)
    source_stars = (source_detection.pywt_stars
                    if matching_path in BOOTSTRAP_MATCHING_PATHS
                    else source_detection.median_stars)
    src_geo = GeometryView(source_stars, src_camera)
    _check_star_count(ref_geo, src_geo)
    if same_camera and _has_identical_star_geometry(ref_geo, src_geo):
        logger.debug(
            "align_frame_camera_model: identical star geometry; bypassing bootstrap, optimization, and remap"
        )
        return frame.copy()

    if matching_path in BOOTSTRAP_MATCHING_PATHS:
        try:
            refine_ref_stars = (ref_refine_geo.stars
                                if guided_refine and ref_refine_geo is not None
                                else None)
            refine_src_stars = (source_detection.median_stars
                                if guided_refine else None)
            staged = solve_staged_alignment(
                ref_geo.stars,
                src_geo.stars,
                ref_candidate,
                src_candidate,
                bootstrap_scales=bootstrap_scales,
                same_camera=same_camera,
                guided_refine=guided_refine,
                guided_refine_radius_px=guided_refine_radius_px,
                use_asterism_bootstrap=(
                    matching_path
                    == MATCHING_PATH_ASTERISM),
                ref_refine_stars=refine_ref_stars,
                src_refine_stars=refine_src_stars,
            )
            result = staged.final_alignment
            match = staged.final_match
            logger.debug(
                "align_frame_camera_model: matching_path={} bootstrap_pairs={} "
                "final_pairs={} refine_status={}",
                matching_path,
                len(staged.bootstrap_match.pair_idx),
                len(match.pair_idx),
                staged.refine_status,
            )
        except Exception as exc:
            raise AlignmentError(
                f"{matching_path} alignment failed: {exc}") from exc
    else:
        ref_geo, src_geo, ref_candidate, src_candidate, match = (
            _select_initial_alignment_candidate(
                ref_geo,
                src_geo,
                ref_candidate,
                src_candidate,
                bootstrap_scales,
                same_camera=same_camera,
            ))
        ref_camera = ref_candidate.camera
        src_camera = src_candidate.camera

        try:
            result = optimize_alignment(
                match,
                ref_camera,
                src_camera,
                same_camera=same_camera,
                ref_policy=ref_candidate.optimization_policy,
                src_policy=src_candidate.optimization_policy,
            )
        except Exception as e:
            raise AlignmentError(f"Optimization failed: {e}") from e

        if guided_refine:
            initial_pair_count = len(match.pair_idx)
            try:
                result, match, guided_stage_status = run_guided_refine_stage(
                    ref_geo,
                    src_geo,
                    result,
                    bootstrap_match=match,
                    same_camera=same_camera,
                    max_distance_px=guided_refine_radius_px,
                    ref_policy=ref_candidate.optimization_policy,
                    src_policy=src_candidate.optimization_policy,
                )
                logger.debug(
                    "align_frame_camera_model: guided refinement status={} "
                    "initial_pairs={} guided_pairs={} radius_px={:.3f}",
                    guided_stage_status,
                    initial_pair_count,
                    len(match.pair_idx),
                    guided_refine_radius_px,
                )
            except Exception as exc:
                logger.warning(
                    "align_frame_camera_model: guided refinement failed; "
                    "keeping first-stage result: {}",
                    exc,
                )

    rotation_ref_to_src = result.rotation_ref_to_src
    cam1 = result.ref_camera
    cam2 = result.src_camera

    _log_rotation_diagnostics("align_frame_camera_model: rotation_ref_to_src",
                              rotation_ref_to_src)
    logger.debug(
        "align_frame_camera_model: backend=remap ref_projection={} "
        "src_projection={} ref_focal={:.6f}mm src_focal={:.6f}mm",
        ref_projection, src_projection, cam1.intrinsics.focal_length_mm,
        cam2.intrinsics.focal_length_mm)

    pts_ref = match.ref_pts
    pts_src = match.src_pts
    ref_vecs_match = cam1.unproject(pts_ref)
    src_vecs_match = (rotation_ref_to_src @ ref_vecs_match.T).T
    pts_src_pred = cam2.project(src_vecs_match)
    valid_pred = np.all(np.isfinite(pts_src_pred), axis=1)
    if np.any(valid_pred):
        _log_residual_diagnostics(
            "align_frame_camera_model: residual",
            pts_ref[valid_pred],
            pts_src[valid_pred],
            pts_src_pred[valid_pred],
            reference.shape,
        )
    else:
        logger.debug(
            "align_frame_camera_model: residual: no finite projected points")

    h, w = reference.shape[:2]
    return cam1.project_image_from_camera(
        cam2,
        frame,
        (w, h),
        rotation_dst_to_src=rotation_ref_to_src,
        map_scale=remap_map_scale,
    )
