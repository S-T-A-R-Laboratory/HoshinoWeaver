"""High-level alignment pipeline.

Provides two paths:
1. Star-point matching alignment (heavy): detect → match → optimize
2. Known-pointing projection (light): direct remap given known transforms
"""
import dataclasses
from typing import Optional

import cv2
import numpy as np
from loguru import logger
from numpy.typing import NDArray

from .cache import GeometryView
from .matching import (MatchResult, adaptive_k, extract_point_features,
                       find_initial_match, fine_tune_transform)
from .optimization import (AlignmentParams, OptimizationContext,
                           run_optimization)
from .types import CameraModel, Distortion


@dataclasses.dataclass(frozen=True)
class AlignmentResult:
    rotation: NDArray[np.float64]
    camera1_refined: CameraModel
    camera2_refined: CameraModel
    pair_idx: Optional[NDArray[np.int32]] = None

    def compose(self, other: "AlignmentResult") -> "AlignmentResult":
        """Chain: self is A→B, other is B→C, returns A→C."""
        return AlignmentResult(
            rotation=other.rotation @ self.rotation,
            camera1_refined=self.camera1_refined,
            camera2_refined=other.camera2_refined,
        )


def match_star_pairs(
    ref_vectors: NDArray[np.float64],
    src_vectors: NDArray[np.float64],
    ref_volumes: NDArray[np.float64],
    src_volumes: NDArray[np.float64],
    ref_pts: NDArray[np.float64],
    src_pts: NDArray[np.float64],
    k: int | None = None,
    apply_threshold_filter: bool = True,
    theta_th: float = np.pi / 6,
) -> MatchResult:
    """Initial matching + RANSAC refinement. Returns MatchResult."""
    if k is None:
        k = adaptive_k(min(len(ref_vectors), len(src_vectors)))
    ref_features = extract_point_features(ref_vectors, ref_volumes, k=k)
    src_features = extract_point_features(src_vectors, src_volumes, k=k)

    pair_idx = find_initial_match(
        ref_features, src_features, ref_pts, src_pts,
        vectors1=ref_vectors, vectors2=src_vectors,
        apply_threshold_filter=apply_threshold_filter,
        theta_th=theta_th)

    tf, pair_idx = fine_tune_transform(ref_pts, src_pts, pair_idx)

    return MatchResult(
        pair_idx=pair_idx,
        ref_pts=ref_pts[pair_idx[:, 0]],
        src_pts=src_pts[pair_idx[:, 1]],
        init_homography=tf,
    )


def match_star_pairs_from_geo(
    ref_geo: GeometryView,
    src_geo: GeometryView,
    apply_threshold_filter: bool = True,
    theta_th: float = np.pi / 6,
) -> MatchResult:
    """match_star_pairs variant that accepts GeometryView directly.

    Reuses cached features from GeometryView instead of recomputing them.
    """
    pair_idx = find_initial_match(
        ref_geo.features, src_geo.features,
        ref_geo.positions, src_geo.positions,
        vectors1=ref_geo.unit_vectors, vectors2=src_geo.unit_vectors,
        apply_threshold_filter=apply_threshold_filter,
        theta_th=theta_th)

    tf, pair_idx = fine_tune_transform(
        ref_geo.positions, src_geo.positions, pair_idx)

    return MatchResult(
        pair_idx=pair_idx,
        ref_pts=ref_geo.positions[pair_idx[:, 0]],
        src_pts=src_geo.positions[pair_idx[:, 1]],
        init_homography=tf,
    )


def optimize_alignment(
    match: MatchResult,
    camera1: CameraModel,
    camera2: CameraModel,
    same_camera: bool = False,
    n_dist: int = 4,
) -> AlignmentResult:
    """Optimize rotation and camera parameters from matched points.

    Args:
        match: MatchResult from match_star_pairs.
        camera1, camera2: initial camera models for ref and src images.
        same_camera: whether both images are from the same camera.
        strategy: "joint" or "staged".
        n_dist: number of distortion parameters to optimize (2 or 4).

    Returns:
        AlignmentResult with optimized rotation and refined cameras.
    """
    ref_pts = match.ref_pts
    src_pts = match.src_pts

    R_init = (np.linalg.inv(camera2.K) @ match.init_homography @ camera1.K)
    rvec, _ = cv2.Rodrigues(R_init)
    rvec = rvec[:, 0]

    dist1_init = camera1.distortion.to_cv2()[:n_dist] if not camera1.distortion.is_zero else np.zeros(n_dist)
    dist2_init = camera2.distortion.to_cv2()[:n_dist] if not camera2.distortion.is_zero else np.zeros(n_dist)

    p0 = AlignmentParams(
        rvec=rvec,
        focal_scale_1=0.0,
        distortion_1=dist1_init,
        focal_scale_2=0.0 if not same_camera else None,
        distortion_2=dist2_init if not same_camera else None,
    )
    x0 = p0.pack(same_camera)

    intr1 = camera1.intrinsics
    intr2 = camera2.intrinsics

    ctx = OptimizationContext(
        ref_pts=ref_pts,
        src_pts=src_pts,
        base_focal_1=intr1.focal_length_mm,
        base_focal_2=intr2.focal_length_mm,
        sensor_w_mm=intr1.sensor_width_mm,
        sensor_h_mm=intr1.sensor_height_mm,
        img_w_1=intr1.image_width_px,
        img_h_1=intr1.image_height_px,
        img_w_2=intr2.image_width_px,
        img_h_2=intr2.image_height_px,
        same_camera=same_camera,
        n_dist=n_dist,
        params0=x0.copy(),
    )

    result = _joint_optimization(x0, ctx, camera1, camera2)

    return result


def _joint_optimization(x0, ctx, camera1, camera2) -> AlignmentResult:
    """Single-stage joint optimization."""
    res = run_optimization(x0, ctx, max_nfev=300)
    logger.debug(f"Joint optimization cost: {res.cost:.6f}")
    return _build_result(res.x, ctx, camera1, camera2)


def _build_result(params_flat, ctx, camera1, camera2) -> AlignmentResult:
    """Extract optimized cameras and rotation from parameter vector."""
    p = AlignmentParams.unpack(params_flat, ctx.same_camera, ctx.n_dist)
    R, _ = cv2.Rodrigues(p.rvec.reshape(3, 1))

    focal_1 = ctx.base_focal_1 * (1 + p.focal_scale_1)
    dist_arr_1 = np.zeros(5)
    dist_arr_1[:ctx.n_dist] = p.distortion_1
    cam1_refined = camera1.with_focal_length(focal_1).with_distortion(
        Distortion.from_cv2(dist_arr_1))

    if ctx.same_camera:
        cam2_refined = camera2.with_focal_length(focal_1).with_distortion(
            Distortion.from_cv2(dist_arr_1))
    else:
        focal_2 = ctx.base_focal_2 * (1 + (p.focal_scale_2 or 0.0))
        dist_arr_2 = np.zeros(5)
        if p.distortion_2 is not None:
            dist_arr_2[:ctx.n_dist] = p.distortion_2
        cam2_refined = camera2.with_focal_length(focal_2).with_distortion(
            Distortion.from_cv2(dist_arr_2))

    return AlignmentResult(rotation=R, camera1_refined=cam1_refined,
                           camera2_refined=cam2_refined)


def warp_image(src_image: NDArray[np.uint8],
               src_camera: CameraModel,
               dst_camera: CameraModel,
               rotation: NDArray[np.float64],
               output_size: tuple[int, int]) -> NDArray[np.uint8]:
    """Warp src_image into dst_camera frame via undistort → perspective → redistort.

    Args:
        src_image: source image.
        src_camera: source camera model (with distortion).
        dst_camera: destination camera model (with distortion).
        rotation: 3x3 rotation matrix from src to dst.
        output_size: (width, height) of output image.
    """
    src_has_dist = not src_camera.distortion.is_zero
    dst_has_dist = not dst_camera.distortion.is_zero

    if not src_has_dist and not dst_has_dist:
        tf = dst_camera.K @ rotation @ np.linalg.inv(src_camera.K)
        return cv2.warpPerspective(src_image, tf, output_size)

    if src_has_dist:
        undist_src = cv2.undistort(src_image, src_camera.K, src_camera.dist_coeffs)
    else:
        undist_src = src_image

    tf = dst_camera.K @ rotation @ np.linalg.inv(src_camera.K)
    warped = cv2.warpPerspective(undist_src, tf, output_size)

    if dst_has_dist:
        from .projection import distort_image
        warped = distort_image(warped, dst_camera.K, dst_camera.dist_coeffs,
                               output_size)

    return warped


def warp_image_by_remap(src_image: NDArray[np.uint8],
                        src_camera: CameraModel,
                        dst_camera: CameraModel,
                        output_size: tuple[int, int],
                        roi: Optional[tuple] = None) -> NDArray[np.uint8]:
    """Per-pixel remap: unproject dst pixels → project into src → remap.

    More accurate for large angular differences or non-planar projections.
    """
    return dst_camera.project_image_from_camera(
        src_camera, src_image, output_size, roi=roi)
