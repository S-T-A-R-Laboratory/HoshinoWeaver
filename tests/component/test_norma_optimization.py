"""Tests for norma alignment optimization residuals."""
import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import least_squares

from hoshicore.component.norma.alignment import (
    AlignmentOptimizationError,
    _camera_optimization_state,
    _relaxed_camera_policies,
    _validate_flexible_optimization,
    optimize_alignment,
)
from hoshicore.component.norma import alignment as alignment_module
from hoshicore.component.norma.matching import MatchResult
from hoshicore.component.norma.optimization import (
    FOCAL_ONLY_FROZEN_TERMS,
    FOCAL_ONLY_POLICY,
    ROTATION_ONLY_FROZEN_TERMS,
    ROTATION_ONLY_POLICY,
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
    CameraModel,
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


# ---------------------------------------------------------------------------
# Relaxed camera-policy ladder: a bound hit re-solves with fewer camera
# degrees of freedom instead of dropping the frame.
# ---------------------------------------------------------------------------

FULL_POLICY = CameraOptimizationPolicy(True, True, True, 4)


def _rung_label(policy: CameraOptimizationPolicy) -> str:
    if not policy.optimize_focal:
        return "rotation_only"
    if not policy.optimize_distortion:
        return "focal_only"
    return "requested"


def _install_fake_solver(monkeypatch, *, bounded_rungs, failing_rungs=(),
                         residual_p90_rad):
    """Script the optimizer and the residual diagnostics per ladder rung."""
    real_residual_diagnostics = alignment_module.compute_flexible_residual_diagnostics

    def fake_run(x0, ctx, max_nfev=300):
        width = len(x0)
        label = _rung_label(ctx.ref_state.policy)
        if label in failing_rungs:
            return SimpleNamespace(
                success=False, x=np.zeros(width), message="fake failure",
                active_mask=np.zeros(width, dtype=int), jac=np.eye(width))
        mask = np.zeros(width, dtype=int)
        if label in bounded_rungs:
            mask[-1] = 1
        return SimpleNamespace(success=True, x=np.zeros(width), message="ok",
                               active_mask=mask, jac=np.eye(width))

    def fake_diagnostics(params_flat, ctx):
        diagnostics = real_residual_diagnostics(params_flat, ctx)
        p90 = residual_p90_rad[_rung_label(ctx.ref_state.policy)]
        diagnostics["raw_angle_p90_rad"] = p90
        diagnostics["raw_angle_p90_px"] = (
            p90 * diagnostics["pixel_scale_px_per_rad"])
        return diagnostics

    monkeypatch.setattr(alignment_module, "run_flexible_optimization",
                        fake_run)
    monkeypatch.setattr(alignment_module,
                        "compute_flexible_residual_diagnostics",
                        fake_diagnostics)


def _matched_pair(count: int = 4) -> MatchResult:
    points = np.column_stack([np.linspace(100.0, 500.0, count),
                              np.linspace(120.0, 420.0, count)])
    return MatchResult(pair_idx=np.arange(count),
                       ref_pts=points.copy(), src_pts=points.copy(),
                       rotation=np.eye(3))


def _camera() -> CameraModel:
    return CameraModel(Intrinsics(20.0, 36.0, 24.0, 1200, 800))


def test_ladder_keeps_requested_policy_when_it_validates(monkeypatch):
    _install_fake_solver(
        monkeypatch, bounded_rungs=set(),
        residual_p90_rad={"requested": 1e-3, "focal_only": 1e-3,
                          "rotation_only": 1e-3})

    result = optimize_alignment(_matched_pair(), _camera(), _camera(),
                                same_camera=True, ref_policy=FULL_POLICY)

    assert result.camera_policy == "requested"


def test_ladder_falls_back_to_focal_only_on_bound_hit(monkeypatch):
    _install_fake_solver(
        monkeypatch, bounded_rungs={"requested"},
        residual_p90_rad={"requested": 1e-3, "focal_only": 1.05e-3,
                          "rotation_only": 1e-3})

    result = optimize_alignment(_matched_pair(), _camera(), _camera(),
                                same_camera=True, ref_policy=FULL_POLICY)

    assert result.camera_policy == "focal_only"


def test_ladder_skips_rung_whose_residual_misses_the_floor(monkeypatch):
    _install_fake_solver(
        monkeypatch, bounded_rungs={"requested", "focal_only"},
        residual_p90_rad={"requested": 1e-2, "focal_only": 5e-3,
                          "rotation_only": 1e-3})

    result = optimize_alignment(_matched_pair(), _camera(), _camera(),
                                same_camera=True, ref_policy=FULL_POLICY)

    # focal_only is 5x the rotation-only residual: the simpler rung wins
    assert result.camera_policy == "rotation_only"


def test_ladder_refuses_rung_above_the_pixel_bar(monkeypatch):
    # 2.7e-3 rad is 1.8 px P90 at this camera (f_px = 667): inside the pixel
    # bar, so rotation-only is usable, but a rung 1.2x worse is 2.16 px and must
    # not be accepted just because it is close to a mediocre floor.
    _install_fake_solver(
        monkeypatch, bounded_rungs={"requested", "focal_only"},
        residual_p90_rad={"requested": 2.7e-3, "focal_only": 3.24e-3,
                          "rotation_only": 2.7e-3})

    result = optimize_alignment(_matched_pair(), _camera(), _camera(),
                                same_camera=True, ref_policy=FULL_POLICY)

    assert result.camera_policy == "rotation_only"


def test_ladder_still_rejects_when_rotation_only_is_not_usable(monkeypatch):
    _install_fake_solver(
        monkeypatch, bounded_rungs={"requested"},
        residual_p90_rad={"requested": 5e-2, "focal_only": 5e-2,
                          "rotation_only": 5e-2})

    # 5e-2 rad is ~33 px at this camera, far beyond the 2 px quality bar, so the
    # frame is still rejected: relaxing the camera model must not accept a bad
    # fit.
    with pytest.raises(AlignmentOptimizationError,
                       match="reached optimization bounds"):
        optimize_alignment(_matched_pair(), _camera(), _camera(),
                           same_camera=True, ref_policy=FULL_POLICY)


def test_ladder_propagates_original_error_when_rotation_only_fails(monkeypatch):
    _install_fake_solver(
        monkeypatch, bounded_rungs={"requested"},
        failing_rungs={"rotation_only"},
        residual_p90_rad={"requested": 1e-3, "focal_only": 1e-3,
                          "rotation_only": 1e-3})

    with pytest.raises(AlignmentOptimizationError,
                       match="reached optimization bounds"):
        optimize_alignment(_matched_pair(), _camera(), _camera(),
                           same_camera=True, ref_policy=FULL_POLICY)


@pytest.mark.parametrize(
    ("frozen_terms", "expected_policy"),
    [
        (FOCAL_ONLY_FROZEN_TERMS, FOCAL_ONLY_POLICY),
        (ROTATION_ONLY_FROZEN_TERMS, ROTATION_ONLY_POLICY),
    ],
)
def test_relaxed_rung_flags_match_the_shared_policies(frozen_terms,
                                                      expected_policy):
    """The frozen-term tuples and the shared policy objects must not drift."""
    ref_policy, src_policy = _relaxed_camera_policies(
        FULL_POLICY, FULL_POLICY, True, frozen_terms)

    assert (ref_policy.optimize_focal,
            ref_policy.optimize_distortion,
            ref_policy.optimize_principal_point) == (
        expected_policy.optimize_focal,
        expected_policy.optimize_distortion,
        expected_policy.optimize_principal_point)
    assert src_policy == ref_policy


def test_relaxed_rung_keeps_per_camera_settings():
    """Freezing flags must not clobber per-camera settings such as n_dist."""
    fisheye_policy = CameraOptimizationPolicy(
        True, True, True, 3,
        principal_point_offset_limit=0.5)
    perspective_policy = CameraOptimizationPolicy(True, True, True, 4)

    ref_policy, src_policy = _relaxed_camera_policies(
        fisheye_policy, perspective_policy, False, FOCAL_ONLY_FROZEN_TERMS)

    assert ref_policy.n_dist == 3
    assert ref_policy.principal_point_offset_limit == pytest.approx(0.5)
    assert src_policy.n_dist == 4
    assert not ref_policy.optimize_distortion
    assert not src_policy.optimize_distortion
