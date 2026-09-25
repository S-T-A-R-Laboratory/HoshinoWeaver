"""Tests for norma alignment optimization residuals."""
import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import least_squares

from hoshicore.component.norma.alignment import (
    AlignmentOptimizationError,
    _camera_optimization_state,
    _validate_flexible_optimization,
)
from hoshicore.component.norma.optimization import (
    CameraOptimizationPolicy,
    CameraOptimizationState,
    FlexibleOptimizationContext,
    flexible_reproject_error,
    make_flexible_parameter_bounds,
    make_flexible_regularization_weights,
    pack_flexible_initial_params,
    unpack_flexible_params,
)
from hoshicore.component.norma.types import (
    FisheyeCameraModel,
    FisheyeDistortion,
    Intrinsics,
)


def _perspective_state() -> CameraOptimizationState:
    return CameraOptimizationState(
        projection_type="perspective",
        base_focal=20.0,
        sensor_w_mm=36.0,
        sensor_h_mm=24.0,
        img_w=1200,
        img_h=800,
        base_cx=600.0,
        base_cy=400.0,
        base_distortion=np.zeros(5, dtype=np.float64),
        policy=CameraOptimizationPolicy(
            optimize_focal=False,
            optimize_distortion=False,
            optimize_principal_point=False,
        ),
    )


def test_two_image_bounds_match_bundle_camera_limits():
    state = dataclasses.replace(
        _perspective_state(),
        policy=CameraOptimizationPolicy(True, True, False, 4),
    )
    ctx = FlexibleOptimizationContext(
        ref_pts=np.empty((0, 2)), src_pts=np.empty((0, 2)),
        ref_state=state, src_state=state, same_camera=True)

    lower, upper = make_flexible_parameter_bounds(ctx)

    np.testing.assert_allclose(lower[3:], [-0.3, -1.0, -1.0, -1.0, -1.0])
    np.testing.assert_allclose(upper[3:], [0.3, 1.0, 1.0, 1.0, 1.0])
    assert np.all(np.isneginf(lower[:3]))
    assert np.all(np.isposinf(upper[:3]))


def test_two_image_principal_point_uses_bounded_image_relative_offsets():
    state = dataclasses.replace(
        _perspective_state(),
        policy=CameraOptimizationPolicy(False, False, True, 0),
    )
    ctx = FlexibleOptimizationContext(
        ref_pts=np.empty((0, 2)), src_pts=np.empty((0, 2)),
        ref_state=state, src_state=state, same_camera=True)
    lower, upper = make_flexible_parameter_bounds(ctx)

    np.testing.assert_allclose(lower[3:], [-0.05, -0.05])
    np.testing.assert_allclose(upper[3:], [0.05, 0.05])
    packed = pack_flexible_initial_params(np.zeros(3), ctx)
    packed[3:] = [0.1, -0.2]
    _, solved, _ = unpack_flexible_params(packed, ctx)
    assert solved.principal_point_offset_x_px == pytest.approx(120.0)
    assert solved.principal_point_offset_y_px == pytest.approx(-160.0)


def test_two_image_principal_point_large_offset_uses_explicit_policy():
    state = dataclasses.replace(
        _perspective_state(),
        policy=CameraOptimizationPolicy(
            False, False, True, 0, principal_point_offset_limit=0.5),
    )
    ctx = FlexibleOptimizationContext(
        ref_pts=np.empty((0, 2)), src_pts=np.empty((0, 2)),
        ref_state=state, src_state=state, same_camera=True)

    lower, upper = make_flexible_parameter_bounds(ctx)

    np.testing.assert_allclose(lower[3:], [-0.5, -0.5])
    np.testing.assert_allclose(upper[3:], [0.5, 0.5])


def test_fixed_camera_keeps_existing_fisheye_distortion_in_residual_state():
    camera = FisheyeCameraModel(
        Intrinsics(15.0, 36.0, 24.0, 1200, 800),
        FisheyeDistortion(0.1, -0.02, 0.003, -0.0004),
    )
    state = _camera_optimization_state(
        camera, CameraOptimizationPolicy(False, False, False, 0))

    np.testing.assert_allclose(
        state.base_distortion, [0.1, -0.02, 0.003, -0.0004])


def test_two_image_validation_rejects_unsuccessful_fit():
    state = _perspective_state()
    ctx = FlexibleOptimizationContext(
        ref_pts=np.empty((0, 2)), src_pts=np.empty((0, 2)),
        ref_state=state, src_state=state, same_camera=True)
    fit = SimpleNamespace(
        success=False, x=np.zeros(3), message="maximum evaluations reached",
        active_mask=np.zeros(3), jac=np.eye(3))

    with pytest.raises(AlignmentOptimizationError,
                       match="maximum evaluations reached"):
        _validate_flexible_optimization(fit, ctx)


def test_two_image_validation_rejects_camera_parameter_at_bound():
    state = dataclasses.replace(
        _perspective_state(),
        policy=CameraOptimizationPolicy(True, False, False, 0),
    )
    ctx = FlexibleOptimizationContext(
        ref_pts=np.empty((0, 2)), src_pts=np.empty((0, 2)),
        ref_state=state, src_state=state, same_camera=True)
    fit = SimpleNamespace(
        success=True, x=np.zeros(4), message="ok",
        active_mask=np.array([0, 0, 0, 1]), jac=np.eye(4))

    with pytest.raises(AlignmentOptimizationError,
                       match="reached optimization bounds"):
        _validate_flexible_optimization(fit, ctx)


def test_two_image_validation_rejects_unobservable_camera_parameter():
    state = dataclasses.replace(
        _perspective_state(),
        policy=CameraOptimizationPolicy(True, False, False, 0),
    )
    ctx = FlexibleOptimizationContext(
        ref_pts=np.empty((0, 2)), src_pts=np.empty((0, 2)),
        ref_state=state, src_state=state, same_camera=True)
    jacobian = np.array([
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 1.0],
    ])
    fit = SimpleNamespace(
        success=True, x=np.zeros(4), message="ok",
        active_mask=np.zeros(4), jac=jacobian)

    with pytest.raises(AlignmentOptimizationError, match="rank deficient"):
        _validate_flexible_optimization(fit, ctx)


def _scipy_constant_residual_cost(
    errors: np.ndarray,
    loss: str,
    f_scale: float,
) -> float:
    res = least_squares(
        lambda _: errors,
        np.zeros(1, dtype=np.float64),
        method="trf",
        loss=loss,
        f_scale=f_scale,
        max_nfev=1,
    )
    return float(res.cost)


def test_scipy_huber_loss_matches_expected_cost():
    errors = np.array([-0.004, -0.001, 0.0, 0.001, 0.004], dtype=np.float64)
    threshold = 0.002
    abs_error = np.abs(errors)
    expected_loss = np.where(
        abs_error < threshold,
        0.5 * errors**2,
        threshold * (abs_error - 0.5 * threshold),
    )

    cost = _scipy_constant_residual_cost(errors, "huber", threshold)

    np.testing.assert_allclose(cost, np.sum(expected_loss))


def test_scipy_cauchy_loss_matches_expected_cost():
    errors = np.array([-0.006, -0.002, 0.0, 0.002, 0.006], dtype=np.float64)
    scale = 0.003
    expected_loss = 0.5 * scale**2 * np.log(1 + (errors / scale)**2)

    cost = _scipy_constant_residual_cost(errors, "cauchy", scale)

    np.testing.assert_allclose(cost, np.sum(expected_loss))


def test_same_camera_fisheye_pixel_residual_uses_shared_distortion():
    pts = np.array([
        [2600.0, 1700.0],
        [3100.0, 2100.0],
        [1800.0, 900.0],
    ], dtype=np.float64)
    policy = CameraOptimizationPolicy(
        optimize_focal=True,
        optimize_distortion=True,
        optimize_principal_point=False,
        n_dist=3,
    )
    state = CameraOptimizationState(
        projection_type="fisheye",
        base_focal=16.0,
        sensor_w_mm=36.0,
        sensor_h_mm=24.0,
        img_w=5472,
        img_h=3648,
        base_cx=2736.0,
        base_cy=1824.0,
        base_distortion=np.array([0.02, -0.01, 0.004, -0.002], dtype=np.float64),
        policy=policy,
    )
    x = np.array([
        0.0, 0.0, 0.0,       # rvec
        0.0,                 # focal scale
        0.02, -0.01, 0.004, # optimized k1..k3; initialized k4 stays fixed
    ], dtype=np.float64)
    ctx = FlexibleOptimizationContext(
        ref_pts=pts,
        src_pts=pts.copy(),
        ref_state=state,
        src_state=state,
        same_camera=True,
        robust_loss=None,
        residual_space="pixel",
    )

    residual = flexible_reproject_error(x, ctx)

    np.testing.assert_allclose(residual, 0.0, atol=1e-9)


def test_cross_residual_preserves_three_components_per_pair():
    ref_pts = np.array([[600.0, 400.0], [720.0, 460.0]], dtype=np.float64)
    src_pts = ref_pts + np.array([[2.0, -1.0], [-3.0, 2.0]])
    state = _perspective_state()
    ctx = FlexibleOptimizationContext(
        ref_pts=ref_pts,
        src_pts=src_pts,
        ref_state=state,
        src_state=state,
        same_camera=True,
        robust_loss=None,
        residual_space="cross",
    )

    residual = flexible_reproject_error(np.zeros(3, dtype=np.float64), ctx)

    assert residual.shape == (6,)
    cross_vectors = residual.reshape(-1, 3)
    assert np.all(np.linalg.norm(cross_vectors, axis=1) > 0)


def test_cross_residual_repeats_pair_weights_for_each_component():
    ref_pts = np.array([[600.0, 400.0], [720.0, 460.0]], dtype=np.float64)
    src_pts = ref_pts + np.array([[2.0, -1.0], [-3.0, 2.0]])
    state = _perspective_state()
    base = FlexibleOptimizationContext(
        ref_pts=ref_pts,
        src_pts=src_pts,
        ref_state=state,
        src_state=state,
        same_camera=True,
        robust_loss=None,
        residual_space="cross",
    )
    weighted = dataclasses.replace(
        base, pts_weight=np.array([2.0, 3.0], dtype=np.float64))

    unweighted_residual = flexible_reproject_error(
        np.zeros(3, dtype=np.float64), base)
    weighted_residual = flexible_reproject_error(
        np.zeros(3, dtype=np.float64), weighted)

    np.testing.assert_allclose(
        weighted_residual,
        unweighted_residual * np.repeat([2.0, 3.0], 3),
    )


def test_unknown_residual_space_is_rejected():
    state = _perspective_state()
    ctx = FlexibleOptimizationContext(
        ref_pts=np.array([[600.0, 400.0]], dtype=np.float64),
        src_pts=np.array([[600.0, 400.0]], dtype=np.float64),
        ref_state=state,
        src_state=state,
        same_camera=True,
        robust_loss=None,
        residual_space="unknown",
    )

    with np.testing.assert_raises_regex(ValueError,
                                        "Unsupported residual_space"):
        flexible_reproject_error(np.zeros(3, dtype=np.float64), ctx)


def test_fisheye_focal_prior_is_context_weight():
    policy = CameraOptimizationPolicy(optimize_focal=True, optimize_distortion=False)
    state = CameraOptimizationState(
        projection_type="fisheye",
        base_focal=16.0,
        sensor_w_mm=36.0,
        sensor_h_mm=24.0,
        img_w=5472,
        img_h=3648,
        base_cx=2736.0,
        base_cy=1824.0,
        base_distortion=np.zeros(4, dtype=np.float64),
        policy=policy,
    )
    ctx = FlexibleOptimizationContext(
        ref_pts=np.zeros((1, 2), dtype=np.float64),
        src_pts=np.zeros((1, 2), dtype=np.float64),
        ref_state=state,
        src_state=state,
        same_camera=True,
    )
    x0 = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    weights = make_flexible_regularization_weights(ctx, x0)

    np.testing.assert_allclose(weights, np.array([0.0, 0.0, 0.0, 1.0]))
