"""Tests for fisheye camera model: projection roundtrip and pipeline integration."""
import cv2
import numpy as np
import pytest

from hoshicore.component.norma import (
    FisheyeCameraModel, FisheyeDistortion, Intrinsics,
    CameraModel, CoordSystem, Distortion, Pointing, View,
)
from hoshicore.component.norma.projection import (
    unproject_fisheye_pixels, project_fisheye_vectors,
)
from hoshicore.component.norma.frame_align import (
    AlignmentCameraCandidate,
    CameraInitializationPolicy,
    build_camera_candidate,
    _candidate_scales,
    build_camera,
)
from hoshicore.component.norma.alignment import MatchResult, optimize_alignment
from hoshicore.component.norma.optimization import CameraOptimizationPolicy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fisheye_cam():
    """8 mm equidistant fisheye on Sony APS-C sensor (6000×4000)."""
    intr = Intrinsics(
        focal_length_mm=8.0,
        sensor_width_mm=23.5,
        sensor_height_mm=15.6,
        image_width_px=6000,
        image_height_px=4000,
    )
    return FisheyeCameraModel(intrinsics=intr, distortion=FisheyeDistortion())


@pytest.fixture
def fisheye_exif():
    return {
        "Exif.Photo.FocalLength": "8/1",
        "Exif.Photo.FocalPlaneXResolution": "255/1",
        "Exif.Photo.FocalPlaneYResolution": "255/1",
        "Exif.Photo.FocalPlaneResolutionUnit": "3",  # cm
    }


# ---------------------------------------------------------------------------
# Projection pure-function tests
# ---------------------------------------------------------------------------

class TestFisheyeRoundtrip:
    """unproject → project should be identity to sub-pixel accuracy."""

    def _valid_pixels(self, cam: FisheyeCameraModel):
        """Pixels clearly inside the fisheye FOV circle."""
        cx, cy = cam.intrinsics.image_width_px / 2, cam.intrinsics.image_height_px / 2
        # FOV boundary at ~90°: r ≈ fx * pi/2
        fx = cam.K[0, 0]
        r_boundary = fx * (np.pi / 2)
        # Use pixels at 0%, 30%, 60% of boundary radius in various directions
        offsets = [0, 0.3, 0.6]
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]
        pts = [[cx, cy]]  # center
        for frac in offsets[1:]:
            for ang in angles:
                r = frac * r_boundary
                pts.append([cx + r * np.cos(ang), cy + r * np.sin(ang)])
        return np.array(pts, dtype=np.float64)

    def test_roundtrip_equidistant(self, fisheye_cam):
        pts = self._valid_pixels(fisheye_cam)
        vecs = fisheye_cam.unproject(pts)
        pts_back = fisheye_cam.project(vecs)
        err = np.abs(pts_back - pts).max()
        assert err < 0.5, f"Roundtrip error too large: {err:.2f} px"

    def test_unit_vectors_normalized(self, fisheye_cam):
        pts = self._valid_pixels(fisheye_cam)
        vecs = fisheye_cam.unproject(pts)
        norms = np.linalg.norm(vecs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_roundtrip_with_distortion(self):
        """Non-zero k1..k4 should also roundtrip."""
        intr = Intrinsics(
            focal_length_mm=8.0,
            sensor_width_mm=23.5,
            sensor_height_mm=15.6,
            image_width_px=6000,
            image_height_px=4000,
        )
        dist = FisheyeDistortion(k1=0.02, k2=-0.005, k3=0.001, k4=0.0)
        cam = FisheyeCameraModel(intrinsics=intr, distortion=dist)

        cx, cy = 3000.0, 2000.0
        fx = cam.K[0, 0]
        r_safe = fx * np.pi / 4  # stay well inside FOV for distorted model
        pts = np.array([
            [cx, cy],
            [cx + r_safe * 0.4, cy],
            [cx - r_safe * 0.3, cy + r_safe * 0.3],
        ])
        vecs = cam.unproject(pts)
        pts_back = cam.project(vecs)
        err = np.abs(pts_back - pts).max()
        assert err < 0.5, f"Distorted roundtrip error: {err:.2f} px"

    def test_center_pixel_projects_to_optical_axis(self, fisheye_cam):
        cx, cy = 3000.0, 2000.0
        vecs = fisheye_cam.unproject(np.array([[cx, cy]]))
        np.testing.assert_allclose(vecs[0], [0.0, 0.0, 1.0], atol=1e-6)

    def test_pure_functions_match_camera_model(self, fisheye_cam):
        """project_fisheye_vectors / unproject_fisheye_pixels match FisheyeCameraModel."""
        pts = np.array([[3000.0, 2000.0], [2000.0, 1500.0], [4000.0, 2500.0]])
        K = fisheye_cam.K
        d4 = fisheye_cam.dist_k4

        vecs_model = fisheye_cam.unproject(pts)
        vecs_fn = unproject_fisheye_pixels(pts, K, d4)
        np.testing.assert_allclose(vecs_model, vecs_fn, atol=1e-10)

        pts_model = fisheye_cam.project(vecs_model)
        pts_fn = project_fisheye_vectors(vecs_model, K, d4)
        np.testing.assert_allclose(pts_model, pts_fn, atol=1e-10)

    def test_roundtrip_handles_vectors_beyond_ninety_degrees(self, fisheye_cam):
        pts = np.array([
            [40.0, 2000.0],
            [5960.0, 2000.0],
            [3000.0, 40.0],
            [3000.0, 3960.0],
            [120.0, 120.0],
            [5880.0, 3880.0],
        ], dtype=np.float64)
        vecs = fisheye_cam.unproject(pts)
        assert np.any(vecs[:, 2] < 0.0)

        pts_back = fisheye_cam.project(vecs)
        np.testing.assert_allclose(pts_back, pts, atol=0.5)


class TestCameraFieldOfView:

    def test_perspective_fov_matches_pinhole_formula(self):
        intr = Intrinsics(20.0, 36.0, 24.0, 6000, 4000)
        camera = CameraModel(intrinsics=intr)

        fov = camera.fov

        assert fov.horizontal_deg == pytest.approx(
            np.rad2deg(2.0 * np.arctan(36.0 / (2.0 * 20.0))))
        assert fov.vertical_deg == pytest.approx(
            np.rad2deg(2.0 * np.arctan(24.0 / (2.0 * 20.0))))
        assert fov.diagonal_deg == pytest.approx(
            np.rad2deg(2.0 * np.arctan(np.hypot(36.0, 24.0) /
                                       (2.0 * 20.0))))

    def test_equidistant_fisheye_fov_uses_angular_radius(self):
        intr = Intrinsics(15.0, 36.0, 24.0, 6000, 4000)
        camera = FisheyeCameraModel(intrinsics=intr)

        fov = camera.fov

        assert fov.horizontal_deg == pytest.approx(np.rad2deg(36.0 / 15.0))
        assert fov.vertical_deg == pytest.approx(np.rad2deg(24.0 / 15.0))
        assert fov.diagonal_deg == pytest.approx(
            np.rad2deg(np.hypot(36.0, 24.0) / 15.0))

    def test_fisheye_distortion_changes_estimated_fov(self):
        intr = Intrinsics(15.0, 36.0, 24.0, 6000, 4000)
        ideal = FisheyeCameraModel(intrinsics=intr)
        distorted = FisheyeCameraModel(
            intrinsics=intr,
            distortion=FisheyeDistortion(k1=0.05),
        )

        assert distorted.fov.horizontal_deg < ideal.fov.horizontal_deg


# ---------------------------------------------------------------------------
# Type and builder tests
# ---------------------------------------------------------------------------

class TestFisheyeDistortion:
    def test_zero_is_equidistant(self):
        d = FisheyeDistortion()
        assert d.is_zero

    def test_to_cv2_shape(self):
        d = FisheyeDistortion(k1=0.1, k2=0.02, k3=-0.005, k4=0.001)
        arr = d.to_cv2()
        assert arr.shape == (4,)
        assert arr[0] == 0.1

    def test_to_opt_params(self):
        d = FisheyeDistortion(k1=0.1, k2=0.02, k3=-0.005, k4=0.001)
        assert len(d.to_opt_params(4)) == 4
        assert len(d.to_opt_params(2)) == 2

    def test_from_array_roundtrip(self):
        d = FisheyeDistortion(k1=0.1, k2=0.02, k3=-0.005, k4=0.001)
        d2 = FisheyeDistortion.from_array(d.to_cv2())
        assert d == d2


class TestFisheyeCameraModel:
    def test_with_focal_length(self, fisheye_cam):
        cam2 = fisheye_cam.with_focal_length(10.0)
        assert cam2.intrinsics.focal_length_mm == 10.0
        assert fisheye_cam.intrinsics.focal_length_mm == 8.0  # original unchanged

    def test_with_distortion(self, fisheye_cam):
        d = FisheyeDistortion(k1=0.05)
        cam2 = fisheye_cam.with_distortion(d)
        assert cam2.distortion.k1 == 0.05
        assert fisheye_cam.distortion.is_zero  # original unchanged

    def test_inherits_project_image_from_camera(self, fisheye_cam):
        """FisheyeCameraModel should have project_image_from_camera via BaseCameraModel."""
        assert hasattr(fisheye_cam, "project_image_from_camera")
        # Dummy image roundtrip (just verifies it runs without error)
        img = np.zeros((40, 60, 3), dtype=np.uint8)
        intr_small = Intrinsics(8.0, 23.5, 15.6, 60, 40)
        small_cam = FisheyeCameraModel(intrinsics=intr_small)
        result = small_cam.project_image_from_camera(small_cam, img, (60, 40))
        assert result.shape == img.shape

    def test_image_projection_uses_explicit_rotation(self):
        intr = Intrinsics(8.0, 23.5, 15.6, 60, 40)
        camera = FisheyeCameraModel(intrinsics=intr)
        image = np.arange(40 * 60, dtype=np.uint16).reshape(40, 60)
        angle_rad = np.deg2rad(0.5)
        rotation_ref_to_src, _ = cv2.Rodrigues(
            np.array([0.0, angle_rad, 0.0], dtype=np.float64)
        )

        ys, xs = np.meshgrid(np.arange(40), np.arange(60), indexing="ij")
        dst_pts = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)
        ref_vecs = camera.unproject(dst_pts)
        src_pts = camera.project((rotation_ref_to_src @ ref_vecs.T).T)
        expected = cv2.remap(
            image,
            src_pts[:, 0].reshape(40, 60).astype(np.float32),
            src_pts[:, 1].reshape(40, 60).astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            hint=cv2.ALGO_HINT_ACCURATE,
        )
        actual = camera.project_image_from_camera(
            camera,
            image,
            (60, 40),
            rotation_dst_to_src=rotation_ref_to_src,
        )
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1)


# ---------------------------------------------------------------------------
# build_camera integration
# ---------------------------------------------------------------------------

class TestTryBuildCamera:
    def test_fisheye_type_returns_fisheye_model(self, fisheye_exif):
        cam = build_camera(fisheye_exif, (4000, 6000), "distortion", lens_type="fisheye")
        assert isinstance(cam, FisheyeCameraModel)
        assert cam.distortion.is_zero

    def test_no_lens_type_returns_perspective(self, fisheye_exif):
        cam = build_camera(fisheye_exif, (4000, 6000), "distortion")
        assert isinstance(cam, CameraModel)
        assert not isinstance(cam, FisheyeCameraModel)

    def test_homography_method_still_builds_zero_distortion_camera(self, fisheye_exif):
        cam = build_camera(fisheye_exif, (4000, 6000), "homography", lens_type="fisheye")
        assert isinstance(cam, CameraModel)
        assert not isinstance(cam, FisheyeCameraModel)
        assert cam.distortion.is_zero

    def test_no_exif_fisheye_uses_fov_estimate(self):
        """Without EXIF, fisheye path falls back to 180° FOV estimate rather than None."""
        cam = build_camera(None, (4000, 6000), "distortion", lens_type="fisheye")
        assert isinstance(cam, FisheyeCameraModel)
        # K should have been estimated from image size (180° equidistant assumption)
        fx = cam.K[0, 0]
        assert fx > 0

    def test_fisheye_default_policy_optimizes_three_distortion_terms(self):
        cand = build_camera_candidate(
            None,
            (4000, 6000),
            "distortion",
            init_policy=CameraInitializationPolicy(lens_type="fisheye"),
        )
        assert cand.optimization_policy.n_dist == 3
        assert cand.optimization_policy.optimize_principal_point
        assert cand.optimization_policy.principal_point_offset_limit == pytest.approx(
            0.05)

    def test_large_principal_point_option_uses_half_axis_bound(self):
        cand = build_camera_candidate(
            None,
            (4000, 6000),
            "distortion",
            init_policy=CameraInitializationPolicy(
                lens_type="fisheye",
                allow_large_principal_point_offset=True,
            ),
        )

        assert cand.optimization_policy.optimize_principal_point
        assert cand.optimization_policy.principal_point_offset_limit == pytest.approx(
            0.5)

    def test_no_exif_perspective_uses_20mm_fallback(self):
        cam = build_camera(None, (4000, 6000), "distortion")
        assert isinstance(cam, CameraModel)
        assert cam.intrinsics.focal_length_mm == pytest.approx(20.0)
        assert cam.intrinsics.sensor_width_mm == pytest.approx(36.0)
        assert cam.intrinsics.sensor_height_mm == pytest.approx(24.0)

    def test_auto_method_is_not_an_implicit_policy(self):
        with pytest.raises(ValueError, match="Unsupported alignment method"):
            build_camera(None, (4000, 6000), "auto")

    def test_fallback_candidate_does_not_optimize_distortion_by_default(self):
        cand = build_camera_candidate(
            None,
            (4000, 6000),
            "distortion",
        )
        assert cand is not None
        assert cand.init_source == "fallback_focal"
        assert cand.optimization_policy.optimize_focal
        assert not cand.optimization_policy.optimize_distortion

    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("exif", (1.0, )),
            ("provided", (1.0, )),
            ("manual", (0.7, 1.0, 1.3)),
            ("fallback_focal", (0.7, 1.0, 1.3)),
        ],
    )
    def test_perspective_bootstrap_depends_on_init_source(self, source,
                                                          expected):
        camera = CameraModel(Intrinsics(20.0, 36.0, 24.0, 6000, 4000))
        candidate = AlignmentCameraCandidate(camera,
                                             CameraOptimizationPolicy(), source)
        assert _candidate_scales(candidate, (0.7, 1.0, 1.3)) == expected

    def test_fisheye_always_bootstraps(self, fisheye_cam):
        candidate = AlignmentCameraCandidate(fisheye_cam,
                                             CameraOptimizationPolicy(),
                                             "exif")
        assert _candidate_scales(candidate,
                                 (0.7, 1.0, 1.3)) == (0.7, 1.0, 1.3)


def test_camera_and_pointing_are_built_separately_from_view():
    view = View(
        focal_length=20.0,
        sensor_width_mm=36.0,
        sensor_height_mm=24.0,
        img_width=6000,
        img_height=4000,
        az_deg=120.0,
        alt_deg=45.0,
        world_roll_deg=3.0,
    )

    camera = CameraModel.from_view(view)
    pointing = Pointing.from_view(view)

    assert not hasattr(camera, "pointing")
    assert camera.intrinsics.focal_length_mm == pytest.approx(20.0)
    assert pointing is not None
    assert pointing.system is CoordSystem.ALTAZ
    assert pointing.R.shape == (3, 3)

    def test_ideal_candidate_keeps_distortion_out_of_policy(self):
        cand = build_camera_candidate(
            None,
            (4000, 6000),
            "distortion",
            focal_equiv_mm=24.0,
            init_policy=CameraInitializationPolicy(lens_type="ideal"),
        )
        assert cand is not None
        assert isinstance(cand.camera, CameraModel)
        assert cand.camera.distortion.is_zero
        assert not cand.optimization_policy.optimize_distortion

    def test_mixed_projection_forces_independent_cameras(self):
        intr = Intrinsics(20.0, 36.0, 24.0, 6000, 4000)
        ref = FisheyeCameraModel(intrinsics=intr)
        src = CameraModel(intrinsics=intr)
        pts = np.array([
            [3000.0, 2000.0],
            [3100.0, 2050.0],
            [2900.0, 1950.0],
        ], dtype=np.float64)
        match = MatchResult(
            pair_idx=np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int32),
            ref_pts=pts,
            src_pts=pts.copy(),
            rotation=np.eye(3, dtype=np.float64),
            homography=np.eye(3, dtype=np.float64),
        )
        result = optimize_alignment(match, ref, src, same_camera=True)
        assert isinstance(result.ref_camera, FisheyeCameraModel)
        assert isinstance(result.src_camera, CameraModel)
        assert result.rotation is result.rotation_ref_to_src
        assert result.camera1_refined is result.ref_camera
        assert result.camera2_refined is result.src_camera


# ---------------------------------------------------------------------------
# Existing Distortion.to_opt_params regression
# ---------------------------------------------------------------------------

def test_distortion_to_opt_params():
    d = Distortion(k1=0.1, k2=0.02, p1=0.001, p2=-0.002, k3=0.0)
    arr = d.to_opt_params(4)
    assert list(arr) == pytest.approx([0.1, 0.02, 0.001, -0.002])
    arr2 = d.to_opt_params(2)
    assert list(arr2) == pytest.approx([0.1, 0.02])
