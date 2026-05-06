"""Optimization primitives for star-point alignment.

Pure functions for computing reprojection error used inside scipy.optimize.least_squares.
No CameraModel objects are constructed per iteration — only arrays.
"""
import dataclasses
from typing import Optional

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from .projection import make_intrinsic_matrix, unproject_pixels


@dataclasses.dataclass
class AlignmentParams:
    rvec: NDArray[np.float64]
    focal_scale_1: float
    distortion_1: NDArray[np.float64]
    focal_scale_2: Optional[float] = None
    distortion_2: Optional[NDArray[np.float64]] = None

    def pack(self, same_camera: bool) -> NDArray[np.float64]:
        parts = [self.rvec, [self.focal_scale_1], self.distortion_1]
        if not same_camera:
            parts.append([self.focal_scale_2 if self.focal_scale_2 is not None else 0.0])
            parts.append(self.distortion_2 if self.distortion_2 is not None
                         else np.zeros_like(self.distortion_1))
        return np.concatenate(parts)

    @classmethod
    def unpack(cls, arr: NDArray[np.float64], same_camera: bool,
               n_dist: int = 4) -> "AlignmentParams":
        rvec = arr[:3]
        focal_scale_1 = arr[3]
        dist_1 = arr[4:4 + n_dist]
        if same_camera:
            return cls(rvec=rvec, focal_scale_1=focal_scale_1,
                       distortion_1=dist_1)
        offset = 4 + n_dist
        focal_scale_2 = arr[offset]
        dist_2 = arr[offset + 1:offset + 1 + n_dist]
        return cls(rvec=rvec, focal_scale_1=focal_scale_1,
                   distortion_1=dist_1, focal_scale_2=focal_scale_2,
                   distortion_2=dist_2)


@dataclasses.dataclass
class OptimizationContext:
    ref_pts: NDArray[np.float64]
    src_pts: NDArray[np.float64]
    base_focal_1: float
    base_focal_2: float
    sensor_w_mm: float
    sensor_h_mm: float
    img_w_1: int
    img_h_1: int
    img_w_2: int
    img_h_2: int
    same_camera: bool
    n_dist: int = 4
    params0: Optional[NDArray[np.float64]] = None
    pts_weight: Optional[NDArray[np.float64]] = None
    reg_weight: Optional[NDArray[np.float64]] = None
    robust_loss: Optional[str] = "huber"
    adaptive_threshold: bool = True
    adaptive_method: str = "median"
    adaptive_multiplier: float = 2.0


def compute_adaptive_threshold(error: NDArray[np.float64],
                               method: str = "median",
                               multiplier: float = 2.0) -> float:
    abs_error = np.abs(error)
    if method == "median":
        base = np.median(abs_error)
    elif method == "percentile75":
        base = np.percentile(abs_error, 75)
    elif method == "percentile90":
        base = np.percentile(abs_error, 90)
    elif method == "mad":
        median = np.median(abs_error)
        mad = np.median(np.abs(abs_error - median))
        base = mad * 1.4826
    else:
        base = np.median(abs_error)
    threshold = base * multiplier
    return float(np.clip(threshold, 1e-6, 0.1))


def huber_loss(error: NDArray[np.float64], threshold: float) -> NDArray[np.float64]:
    abs_error = np.abs(error)
    return np.where(abs_error < threshold,
                    0.5 * error**2,
                    threshold * (abs_error - 0.5 * threshold))


def cauchy_loss(error: NDArray[np.float64], scale: float) -> NDArray[np.float64]:
    return 0.5 * scale**2 * np.log(1 + (error / scale)**2)


def reproject_error(params_flat: NDArray[np.float64],
                    ctx: OptimizationContext) -> NDArray[np.float64]:
    """Pure reprojection error function for least_squares.

    Unpacks params → builds K matrices → unprojects both point sets →
    applies rotation → computes angular distance on the unit sphere.
    """
    p = AlignmentParams.unpack(params_flat, ctx.same_camera, ctx.n_dist)

    focal_1 = ctx.base_focal_1 * (1 + p.focal_scale_1)
    K1 = make_intrinsic_matrix(focal_1, ctx.sensor_w_mm, ctx.sensor_h_mm,
                               ctx.img_w_1, ctx.img_h_1)
    dist_1 = _expand_dist_coeffs(p.distortion_1)

    if ctx.same_camera:
        focal_2 = focal_1
        K2 = K1
        dist_2 = dist_1
    else:
        focal_2 = ctx.base_focal_2 * (1 + (p.focal_scale_2 or 0.0))
        K2 = make_intrinsic_matrix(focal_2, ctx.sensor_w_mm, ctx.sensor_h_mm,
                                   ctx.img_w_2, ctx.img_h_2)
        dist_2 = _expand_dist_coeffs(p.distortion_2) if p.distortion_2 is not None else None

    pts1_3d = unproject_pixels(ctx.ref_pts, K1, dist_1)
    R, _ = cv2.Rodrigues(p.rvec.reshape(3, 1))
    pts1_rotated = (R @ pts1_3d.T).T

    pts2_3d = unproject_pixels(ctx.src_pts, K2, dist_2)

    dot = np.sum(pts1_rotated * pts2_3d, axis=1)
    pts_loss = np.arccos(np.clip(dot, -1.0, 1.0))

    if ctx.robust_loss is not None:
        if ctx.adaptive_threshold:
            th = compute_adaptive_threshold(pts_loss, ctx.adaptive_method,
                                            ctx.adaptive_multiplier)
        else:
            th = 0.01
        if ctx.robust_loss == "huber":
            pts_loss = huber_loss(pts_loss, th)
        elif ctx.robust_loss == "cauchy":
            pts_loss = cauchy_loss(pts_loss, th)

    if ctx.pts_weight is not None:
        pts_loss = pts_loss * ctx.pts_weight

    if ctx.reg_weight is not None and ctx.params0 is not None:
        reg = ctx.reg_weight * (params_flat - ctx.params0)
    else:
        reg = np.zeros(1)

    return np.concatenate((pts_loss, reg))


def run_optimization(x0: NDArray[np.float64],
                     ctx: OptimizationContext,
                     max_nfev: int = 300):
    """Run least_squares optimization with the given context."""
    if ctx.params0 is None:
        ctx.params0 = x0.copy()
    res = least_squares(reproject_error, x0, args=(ctx,), method='lm',
                        max_nfev=max_nfev)
    return res


def _expand_dist_coeffs(arr: Optional[NDArray[np.float64]]) -> Optional[NDArray[np.float64]]:
    """Expand distortion array to OpenCV's 5-element format [k1, k2, p1, p2, k3]."""
    if arr is None:
        return None
    if len(arr) == 4:
        return np.array([arr[0], arr[1], arr[2], arr[3], 0.0])
    if len(arr) == 2:
        return np.array([arr[0], arr[1], 0.0, 0.0, 0.0])
    return arr
