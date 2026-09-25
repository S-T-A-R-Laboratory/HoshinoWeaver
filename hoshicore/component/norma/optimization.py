"""Optimization primitives for star-point alignment.

Main entry points:
  - ``pack_flexible_initial_params`` builds the least-squares parameter vector.
  - ``run_flexible_optimization`` runs scipy's optimizer.
  - ``unpack_flexible_params`` converts the solved vector back to per-camera
    parameters.

The optimizer works on arrays only.  Camera objects are converted by
``alignment.py`` into per-camera ``CameraOptimizationState`` values before the
solve starts.  During each residual evaluation the flow is:

    ref pixels -> ref camera unproject -> unit rays
    src pixels -> src camera unproject -> unit rays
    rotate ref rays -> selected residual against src rays

``residual_space="angular"`` produces one scalar angle per pair. ``"cross"``
keeps the three-component ray cross product used by sequence BA. If ``"pixel"``
is requested, the rotated ref rays are projected through the source camera and
compared to source pixels. Projection dispatch is per camera, so
perspective/fisheye mixed pairs use the same solver path.

Robust loss is delegated to ``scipy.optimize.least_squares``.  When
``robust_scale`` is not provided, it is estimated once from the initial residual
vector and then kept fixed for the solve.
"""
import dataclasses
from typing import Optional

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from .projection import (
    make_intrinsic_matrix,
    project_fisheye_vectors,
    project_vectors,
    unproject_fisheye_pixels,
    unproject_pixels,
)


_FOCAL_SCALE_DELTA_LIMIT = 0.3
_DISTORTION_ABS_LIMIT = 1.0
DEFAULT_PRINCIPAL_POINT_OFFSET_LIMIT = 0.05
LARGE_PRINCIPAL_POINT_OFFSET_LIMIT = 0.5


@dataclasses.dataclass(frozen=True)
class CameraOptimizationPolicy:
    """Controls which camera parameters may move during one alignment solve.

    Regularization is intentionally not stored here.  It is represented by the
    context-level ``reg_weight`` vector so every penalty is visible in the same
    parameter space as the optimizer.
    """

    optimize_focal: bool = True
    optimize_distortion: bool = True
    optimize_principal_point: bool = True
    n_dist: int = 4
    principal_point_offset_limit: float = DEFAULT_PRINCIPAL_POINT_OFFSET_LIMIT


@dataclasses.dataclass(frozen=True)
class CameraOptimizationState:
    """Static per-camera optimization state used by the flexible solver."""

    projection_type: str
    base_focal: float
    sensor_w_mm: float
    sensor_h_mm: float
    img_w: int
    img_h: int
    base_cx: float
    base_cy: float
    # Full initialized coefficient vector.  The policy controls how many
    # leading coefficients are optimized; trailing coefficients remain fixed
    # but must stay available to the projection model.
    base_distortion: NDArray[np.float64]
    policy: CameraOptimizationPolicy


@dataclasses.dataclass(frozen=True)
class CameraSolvedParams:
    focal_scale: float
    distortion: NDArray[np.float64]
    principal_point_offset_x_px: float = 0.0
    principal_point_offset_y_px: float = 0.0


@dataclasses.dataclass
class FlexibleOptimizationContext:
    """All immutable inputs for one joint camera/rotation optimization."""

    ref_pts: NDArray[np.float64]
    src_pts: NDArray[np.float64]
    ref_state: CameraOptimizationState
    src_state: CameraOptimizationState
    same_camera: bool = False
    params0: Optional[NDArray[np.float64]] = None
    pts_weight: Optional[NDArray[np.float64]] = None
    reg_weight: Optional[NDArray[np.float64]] = None
    robust_loss: Optional[str] = "huber"
    robust_scale: Optional[float] = None
    robust_scale_method: str = "median"
    robust_scale_multiplier: float = 2.0
    residual_space: str = "cross"  # "angular" | "cross" | "pixel"


def _camera_param_width(state: CameraOptimizationState) -> int:
    width = 0
    if state.policy.optimize_focal:
        width += 1
    if state.policy.optimize_distortion:
        width += state.policy.n_dist
    if state.policy.optimize_principal_point:
        width += 2
    return width


def iter_optimized_camera_param_slices(
    ctx: FlexibleOptimizationContext,
) -> list[tuple[str, CameraOptimizationState, slice]]:
    """Return parameter-vector slices for each optimized camera block."""
    slices: list[tuple[str, CameraOptimizationState, slice]] = []
    offset = 3
    slices.append((
        "ref",
        ctx.ref_state,
        slice(offset, offset + _camera_param_width(ctx.ref_state)),
    ))
    offset += _camera_param_width(ctx.ref_state)
    if not ctx.same_camera:
        slices.append((
            "src",
            ctx.src_state,
            slice(offset, offset + _camera_param_width(ctx.src_state)),
        ))
    return slices


def make_flexible_parameter_bounds(
    ctx: FlexibleOptimizationContext,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Bound camera variables while leaving the relative rotation unbounded."""
    lower = np.full(3 + sum(
        param_slice.stop - param_slice.start
        for _, _, param_slice in iter_optimized_camera_param_slices(ctx)
    ), -np.inf, dtype=np.float64)
    upper = np.full_like(lower, np.inf)
    for _, state, param_slice in iter_optimized_camera_param_slices(ctx):
        offset = param_slice.start
        if state.policy.optimize_focal:
            lower[offset] = -_FOCAL_SCALE_DELTA_LIMIT
            upper[offset] = _FOCAL_SCALE_DELTA_LIMIT
            offset += 1
        if state.policy.optimize_distortion:
            width = state.policy.n_dist
            lower[offset:offset + width] = -_DISTORTION_ABS_LIMIT
            upper[offset:offset + width] = _DISTORTION_ABS_LIMIT
            offset += width
        if state.policy.optimize_principal_point:
            limit = float(state.policy.principal_point_offset_limit)
            if not 0.0 < limit <= LARGE_PRINCIPAL_POINT_OFFSET_LIMIT:
                raise ValueError(
                    "principal_point_offset_limit must be in (0, 0.5]")
            lower[offset:offset + 2] = -limit
            upper[offset:offset + 2] = limit
    return lower, upper


def make_flexible_regularization_weights(
    ctx: FlexibleOptimizationContext,
    x0: NDArray[np.float64],
    fisheye_focal_weight: float = 1.0,
) -> NDArray[np.float64]:
    """Build the full regularization vector in optimizer parameter order.

    The current default keeps the previous fisheye focal-scale prior, but it is
    now expressed as part of the unified ``reg_weight`` vector.  Rotation,
    distortion, and principal-point slots are left unregularized unless callers
    fill them explicitly later.
    """
    weights = np.zeros_like(x0, dtype=np.float64)
    for _, state, param_slice in iter_optimized_camera_param_slices(ctx):
        if state.projection_type != "fisheye" or not state.policy.optimize_focal:
            continue
        weights[param_slice.start] = fisheye_focal_weight
    return weights


def _initial_camera_param_values(
        state: CameraOptimizationState) -> NDArray[np.float64]:
    parts: list[NDArray[np.float64]] = []
    if state.policy.optimize_focal:
        parts.append(np.zeros(1, dtype=np.float64))
    if state.policy.optimize_distortion:
        parts.append(
            np.asarray(state.base_distortion[:state.policy.n_dist],
                       dtype=np.float64))
    if state.policy.optimize_principal_point:
        parts.append(np.zeros(2, dtype=np.float64))
    if not parts:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(parts)


def pack_flexible_initial_params(
    rvec: NDArray[np.float64],
    ctx: FlexibleOptimizationContext,
) -> NDArray[np.float64]:
    parts = [np.asarray(rvec, dtype=np.float64)]
    parts.append(_initial_camera_param_values(ctx.ref_state))
    if not ctx.same_camera:
        parts.append(_initial_camera_param_values(ctx.src_state))
    return np.concatenate(parts)


def _unpack_camera_params(
    arr: NDArray[np.float64],
    offset: int,
    state: CameraOptimizationState,
) -> tuple[CameraSolvedParams, int]:
    focal_scale = 0.0
    if state.policy.optimize_focal:
        focal_scale = float(arr[offset])
        offset += 1

    distortion = np.asarray(state.base_distortion, dtype=np.float64).copy()
    if state.policy.optimize_distortion:
        optimized_distortion = np.asarray(
            arr[offset:offset + state.policy.n_dist], dtype=np.float64)
        distortion[:state.policy.n_dist] = optimized_distortion
        offset += state.policy.n_dist

    pp_x = 0.0
    pp_y = 0.0
    if state.policy.optimize_principal_point:
        # The optimizer uses image-relative offsets so principal-point columns
        # have comparable scale across resolutions.  The solved result remains
        # expressed in pixels for camera construction and diagnostics.
        pp_x = float(arr[offset]) * state.img_w
        pp_y = float(arr[offset + 1]) * state.img_h
        offset += 2

    return CameraSolvedParams(
        focal_scale=focal_scale,
        distortion=distortion,
        principal_point_offset_x_px=pp_x,
        principal_point_offset_y_px=pp_y,
    ), offset


def unpack_flexible_params(
    params_flat: NDArray[np.float64],
    ctx: FlexibleOptimizationContext,
) -> tuple[NDArray[np.float64], CameraSolvedParams, CameraSolvedParams]:
    rvec = np.asarray(params_flat[:3], dtype=np.float64)
    offset = 3
    ref_params, offset = _unpack_camera_params(params_flat, offset,
                                               ctx.ref_state)
    if ctx.same_camera:
        src_params = ref_params
    else:
        src_params, offset = _unpack_camera_params(params_flat, offset,
                                                   ctx.src_state)
    return rvec, ref_params, src_params


def _camera_matrix_from_state(
    state: CameraOptimizationState,
    params: CameraSolvedParams,
) -> NDArray[np.float64]:
    focal = state.base_focal * (1.0 + params.focal_scale)
    return make_intrinsic_matrix(
        focal,
        state.sensor_w_mm,
        state.sensor_h_mm,
        state.img_w,
        state.img_h,
        cx_px=state.base_cx + params.principal_point_offset_x_px,
        cy_px=state.base_cy + params.principal_point_offset_y_px,
    )


def _unproject_by_state(
    pts: NDArray[np.float64],
    state: CameraOptimizationState,
    params: CameraSolvedParams,
) -> NDArray[np.float64]:
    K = _camera_matrix_from_state(state, params)
    if state.projection_type == "fisheye":
        return unproject_fisheye_pixels(pts, K, params.distortion[:4])
    return unproject_pixels(pts, K, _expand_dist_coeffs(params.distortion))


def _project_by_state(
    vecs: NDArray[np.float64],
    state: CameraOptimizationState,
    params: CameraSolvedParams,
) -> NDArray[np.float64]:
    K = _camera_matrix_from_state(state, params)
    if state.projection_type == "fisheye":
        return project_fisheye_vectors(vecs, K, params.distortion[:4])
    return project_vectors(vecs, K, _expand_dist_coeffs(params.distortion))


def _flexible_regularization(
    params_flat: NDArray[np.float64],
    ctx: FlexibleOptimizationContext,
) -> NDArray[np.float64]:
    if ctx.reg_weight is not None and ctx.params0 is not None:
        return ctx.reg_weight * (params_flat - ctx.params0)
    return np.zeros(0, dtype=np.float64)


def _compute_flexible_residual_components(
    params_flat: NDArray[np.float64],
    ctx: FlexibleOptimizationContext,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64],
           float]:
    rvec, ref_params, src_params = unpack_flexible_params(params_flat, ctx)
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))

    ref_vecs = _unproject_by_state(ctx.ref_pts, ctx.ref_state, ref_params)
    src_vecs = _unproject_by_state(ctx.src_pts, ctx.src_state, src_params)
    ref_rotated = (R @ ref_vecs.T).T

    dot = np.sum(ref_rotated * src_vecs, axis=1)
    raw_angle = np.arccos(np.clip(dot, -1.0, 1.0))

    if ctx.residual_space == "pixel":
        src_pred = _project_by_state(ref_rotated, ctx.src_state, src_params)
        pixel_residual = src_pred - ctx.src_pts
        finite = np.all(np.isfinite(pixel_residual), axis=1)
        if not np.all(finite):
            pixel_residual = pixel_residual.copy()
            pixel_residual[~finite] = 1e6
        data_residual = pixel_residual.reshape(-1)
    elif ctx.residual_space == "cross":
        # Keep both tangent-plane error directions.  This matches the residual
        # representation used by the sequence BA solver; for small angular
        # errors, ||cross(a, b)|| ~= angle(a, b).
        data_residual = np.cross(ref_rotated, src_vecs).reshape(-1)
    elif ctx.residual_space == "angular":
        data_residual = raw_angle.copy()
    else:
        raise ValueError(
            f"Unsupported residual_space: {ctx.residual_space!r}")

    if ctx.pts_weight is not None:
        if ctx.residual_space == "angular":
            data_residual = data_residual * ctx.pts_weight
        else:
            component_count = 2 if ctx.residual_space == "pixel" else 3
            data_residual = data_residual * np.repeat(
                ctx.pts_weight, component_count)

    reg_residual = _flexible_regularization(params_flat, ctx)

    K1 = _camera_matrix_from_state(ctx.ref_state, ref_params)
    K2 = _camera_matrix_from_state(ctx.src_state, src_params)
    pixel_scale = float((K1[0, 0] + K1[1, 1] + K2[0, 0] + K2[1, 1]) * 0.25)
    return raw_angle, data_residual, reg_residual, pixel_scale


def flexible_reproject_error(
    params_flat: NDArray[np.float64],
    ctx: FlexibleOptimizationContext,
) -> NDArray[np.float64]:
    _, data_residual, reg_residual, _ = _compute_flexible_residual_components(
        params_flat, ctx)
    return np.concatenate((data_residual, reg_residual))


def compute_flexible_residual_diagnostics(
    params_flat: NDArray[np.float64],
    ctx: FlexibleOptimizationContext,
) -> dict[str, float]:
    raw_angle, data_residual, reg_residual, pixel_scale = (
        _compute_flexible_residual_components(params_flat, ctx))
    raw_stats = _describe_array(raw_angle)
    data_stats = _describe_array(data_residual)
    reg_stats = _describe_array(reg_residual)
    data_cost = 0.5 * float(np.dot(data_residual, data_residual))
    reg_cost = 0.5 * float(np.dot(reg_residual, reg_residual))
    return {
        "pixel_scale_px_per_rad": pixel_scale,
        "raw_angle_median_rad": raw_stats["median"],
        "raw_angle_p90_rad": raw_stats["p90"],
        "raw_angle_max_rad": raw_stats["max"],
        "raw_angle_rms_rad": raw_stats["rms"],
        "raw_angle_median_px": raw_stats["median"] * pixel_scale,
        "raw_angle_p90_px": raw_stats["p90"] * pixel_scale,
        "raw_angle_max_px": raw_stats["max"] * pixel_scale,
        "raw_angle_rms_px": raw_stats["rms"] * pixel_scale,
        "data_l2": data_stats["l2"],
        "reg_l2": reg_stats["l2"],
        "data_cost": data_cost,
        "reg_cost": reg_cost,
        "total_cost": data_cost + reg_cost,
    }


def run_flexible_optimization(
    x0: NDArray[np.float64],
    ctx: FlexibleOptimizationContext,
    max_nfev: int = 300,
):
    """Run least_squares for per-camera projection/policy optimization."""
    if ctx.params0 is None:
        ctx.params0 = x0.copy()
    loss = "linear" if ctx.robust_loss is None else ctx.robust_loss
    f_scale = estimate_robust_scale_from_initial_residual(x0, ctx)
    bounds = make_flexible_parameter_bounds(ctx)
    if np.any(x0 < bounds[0]) or np.any(x0 > bounds[1]):
        raise ValueError(
            "initial camera parameters are outside optimization bounds")
    return least_squares(flexible_reproject_error,
                         x0,
                         args=(ctx, ),
                         method="trf",
                         loss=loss,
                         f_scale=f_scale,
                         max_nfev=max_nfev,
                         bounds=bounds)


def estimate_robust_scale(error: NDArray[np.float64],
                          method: str = "median",
                          multiplier: float = 2.0) -> float:
    """Estimate a fixed scipy ``f_scale`` from an initial residual vector."""
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


def estimate_robust_scale_from_initial_residual(
    x0: NDArray[np.float64],
    ctx: FlexibleOptimizationContext,
) -> float:
    """Return scipy ``f_scale`` for the current solve.

    The scale is estimated from data residuals only.  Regularization residuals
    remain part of the least-squares residual vector, but they should not affect
    the robust scale estimate.
    """
    if ctx.robust_loss is None:
        return 1.0
    if ctx.robust_scale is not None:
        return float(ctx.robust_scale)
    raw_angle, data_residual, _, _ = _compute_flexible_residual_components(
        x0, ctx)
    scale_residual = (
        raw_angle if ctx.residual_space == "cross" else data_residual)
    return estimate_robust_scale(
        scale_residual,
        method=ctx.robust_scale_method,
        multiplier=ctx.robust_scale_multiplier,
    )


def _describe_array(arr: NDArray[np.float64]) -> dict[str, float]:
    if arr.size == 0:
        return {
            "median": 0.0,
            "p90": 0.0,
            "max": 0.0,
            "rms": 0.0,
            "l2": 0.0,
            "count": 0.0,
        }
    return {
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
        "rms": float(np.sqrt(np.mean(arr * arr))),
        "l2": float(np.linalg.norm(arr)),
        "count": float(arr.size),
    }


def _expand_dist_coeffs(
        arr: Optional[NDArray[np.float64]]) -> Optional[NDArray[np.float64]]:
    """Expand distortion array to OpenCV's 5-element format [k1, k2, p1, p2, k3]."""
    if arr is None:
        return None
    if len(arr) == 4:
        return np.array([arr[0], arr[1], arr[2], arr[3], 0.0])
    if len(arr) == 2:
        return np.array([arr[0], arr[1], 0.0, 0.0, 0.0])
    return arr
