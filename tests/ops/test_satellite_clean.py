"""Tests for SatelliteCleanOp: sliding window alignment + median."""
import numpy as np
import pytest

from hoshicore.ops.satellite_clean_op import SatelliteCleanOp, _FrameSlot
from hoshicore.component.norma.detection import DetectedStars
from hoshicore.component.norma.geometry_view import GeometryView
from hoshicore.component.norma.types import CameraModel, Intrinsics


def _dummy_geo(shape):
    h, w = shape[:2]
    camera = CameraModel(
        intrinsics=Intrinsics(
            focal_length_mm=20.0,
            sensor_width_mm=36.0,
            sensor_height_mm=24.0,
            image_width_px=w,
            image_height_px=h,
        ))
    return GeometryView(
        DetectedStars(np.empty((0, 2)), np.empty(0)), camera)


class TestCameraInitialization:

    def test_exif_intrinsics_take_priority_over_manual_focal(self):
        exif = {
            "Exif.Photo.FocalLength": "50/1",
            "Exif.Photo.FocalPlaneXResolution": "500/3",
            "Exif.Photo.FocalPlaneYResolution": "500/3",
            "Exif.Photo.FocalPlaneResolutionUnit": "4",
        }

        camera = SatelliteCleanOp._build_frame_camera(
            exif,
            (4000, 6000, 3),
            focal_equiv_mm=35.0,
            fallback_focal_equiv_mm=20.0,
        )

        assert camera.intrinsics.focal_length_mm == pytest.approx(50.0)
        assert camera.intrinsics.sensor_width_mm == pytest.approx(36.0)
        assert camera.intrinsics.sensor_height_mm == pytest.approx(24.0)

    def test_exif_mode_ignores_stale_manual_focal(self):
        camera = SatelliteCleanOp._build_frame_camera(
            {"Exif.Photo.FocalLength": "50/1"},
            (4000, 6000, 3),
            focal_equiv_mm=35.0,
            fallback_focal_equiv_mm=20.0,
        )

        assert camera.intrinsics.focal_length_mm == pytest.approx(20.0)

    def test_manual_mode_ignores_complete_exif(self):
        exif = {
            "Exif.Photo.FocalLength": "50/1",
            "Exif.Photo.FocalPlaneXResolution": "500/3",
            "Exif.Photo.FocalPlaneYResolution": "500/3",
            "Exif.Photo.FocalPlaneResolutionUnit": "4",
        }
        camera = SatelliteCleanOp._build_frame_camera(
            exif,
            (4000, 6000, 3),
            focal_equiv_mm=35.0,
            fallback_focal_equiv_mm=20.0,
            camera_mode="manual",
        )

        assert camera.intrinsics.focal_length_mm == pytest.approx(35.0)

    def test_manual_mode_requires_focal_length(self):
        with pytest.raises(ValueError, match="focal_length_mm is required"):
            SatelliteCleanOp._build_frame_camera(
                None,
                (4000, 6000, 3),
                focal_equiv_mm=None,
                fallback_focal_equiv_mm=20.0,
                camera_mode="manual",
            )

    def test_configured_fallback_is_used_without_exif_or_manual_focal(self):
        camera = SatelliteCleanOp._build_frame_camera(
            None,
            (4000, 6000, 3),
            focal_equiv_mm=None,
            fallback_focal_equiv_mm=28.0,
        )

        assert camera.intrinsics.focal_length_mm == pytest.approx(28.0)


class TestChainRotation:
    """Test _chain_rotation correctness."""

    def _rot_z(self, angle_deg):
        angle = np.deg2rad(angle_deg)
        return np.array([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

    def _make_buffer(self, angles):
        """Create a buffer of FrameSlots with known adjacent rotations."""
        from collections import deque
        buffer = deque()
        for i, angle in enumerate(angles):
            slot = _FrameSlot(
                original=np.zeros((100, 100, 3), dtype=np.uint8),
                geo=None,
            )
            if i < len(angles) - 1:
                slot.R_to_next = self._rot_z(angles[i + 1] - angle)
            buffer.append(slot)
        return buffer

    def test_identity(self):
        buffer = self._make_buffer([0, 5, 10])
        R = SatelliteCleanOp._chain_rotation(buffer, 1, 1)
        np.testing.assert_allclose(R, np.eye(3))

    def test_forward_chain(self):
        buffer = self._make_buffer([0, 5, 10])
        R = SatelliteCleanOp._chain_rotation(buffer, 0, 2)
        np.testing.assert_allclose(R, self._rot_z(10), atol=1e-10)

    def test_reverse_chain(self):
        buffer = self._make_buffer([0, 5, 10])
        R = SatelliteCleanOp._chain_rotation(buffer, 2, 0)
        np.testing.assert_allclose(R, self._rot_z(-10), atol=1e-10)

    def test_forward_reverse_inverse(self):
        buffer = self._make_buffer([0, 3, 7, 12])
        R_fwd = SatelliteCleanOp._chain_rotation(buffer, 0, 3)
        R_rev = SatelliteCleanOp._chain_rotation(buffer, 3, 0)
        product = R_fwd @ R_rev
        np.testing.assert_allclose(product, np.eye(3), atol=1e-10)

    def test_none_rotation_returns_none(self):
        from collections import deque
        buffer = deque()
        for _ in range(3):
            buffer.append(_FrameSlot(
                original=np.zeros((10, 10), dtype=np.uint8),
                geo=None, R_to_next=None))
        result = SatelliteCleanOp._chain_rotation(buffer, 0, 2)
        assert result is None


class TestHomographyFromRotation:

    def test_uses_endpoint_intrinsics_and_rotation(self):
        from_intrinsics = Intrinsics(20.0, 36.0, 24.0, 600, 400)
        to_intrinsics = Intrinsics(35.0, 36.0, 24.0, 900, 600)
        from_geo = GeometryView(
            DetectedStars(np.empty((0, 2)), np.empty(0)),
            CameraModel(intrinsics=from_intrinsics),
        )
        to_geo = GeometryView(
            DetectedStars(np.empty((0, 2)), np.empty(0)),
            CameraModel(intrinsics=to_intrinsics),
        )
        angle = np.deg2rad(7.0)
        rotation = np.array([
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ], dtype=np.float64)

        actual = SatelliteCleanOp._homography_from_rotation(
            from_geo, to_geo, rotation)
        expected = to_geo.camera.K @ rotation @ np.linalg.inv(from_geo.camera.K)
        expected /= expected[2, 2]

        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


class TestProcessCenter:
    """Test _process_center with synthetic translated frames."""

    def _make_shifted_buffer(self, n_frames=5, shape=(100, 150, 3),
                             shift_x=5, satellite_frame=None,
                             satellite_region=None):
        """Create buffer with pure-translation shifted frames."""
        from collections import deque
        rng = np.random.default_rng(42)
        base = rng.normal(100, 10, shape).clip(0, 255).astype(np.uint8)

        buffer = deque()
        for i in range(n_frames):
            # Shift image by i*shift_x pixels horizontally
            M = np.float32([[1, 0, i * shift_x], [0, 1, 0]])
            import cv2
            shifted = cv2.warpAffine(
                base, M, (shape[1], shape[0]),
                borderMode=cv2.BORDER_REPLICATE)

            if satellite_frame == i and satellite_region is not None:
                y1, y2, x1, x2 = satellite_region
                shifted[y1:y2, x1:x2] = 255

            slot = _FrameSlot(original=shifted, geo=_dummy_geo(shape))
            if i > 0:
                buffer[-1].R_to_next = np.eye(3, dtype=np.float64)
            buffer.append(slot)

        return buffer

    def test_satellite_removal(self):
        """Satellite line in center frame should be removed by median."""
        buffer = self._make_shifted_buffer(
            n_frames=5, shift_x=5,
            satellite_frame=2,
            satellite_region=(40, 45, 30, 120))

        center_pos = 2
        result = SatelliteCleanOp._process_center(buffer, center_pos)

        # The satellite region in the result should NOT be 255
        satellite_pixels = result[40:45, 30:120]
        assert satellite_pixels.max() < 200, (
            f"Satellite not removed: max={satellite_pixels.max()}")

    def test_no_satellite_preserves_signal(self):
        """Without satellite, output should approximate the median (≈ original)."""
        buffer = self._make_shifted_buffer(n_frames=5, shift_x=5)
        center_pos = 2
        result = SatelliteCleanOp._process_center(buffer, center_pos)

        center_original = buffer[center_pos].original
        # Median of aligned frames ≈ original (with slight noise reduction)
        # Check that the result is close to the original
        diff = np.abs(
            result.astype(np.float32) - center_original.astype(np.float32))
        assert diff.mean() < 10, f"Mean diff too large: {diff.mean()}"

    def test_single_frame_passthrough(self):
        """Single frame (no neighbors) should pass through unchanged."""
        from collections import deque
        arr = np.random.default_rng(0).integers(
            0, 255, (50, 60, 3), dtype=np.uint8)
        buffer = deque([_FrameSlot(original=arr, geo=None)])
        # actual_W = 0 → process_center with center_pos=0, no neighbors
        result = SatelliteCleanOp._process_center(buffer, 0)
        np.testing.assert_array_equal(result, arr)

    def test_output_dtype_matches_input(self):
        """Output dtype should match input dtype."""
        buffer = self._make_shifted_buffer(n_frames=5, shift_x=3)
        result = SatelliteCleanOp._process_center(buffer, 2)
        assert result.dtype == buffer[2].original.dtype


class TestFrameCount:
    """Verify that output frame count equals input frame count."""

    def test_output_length_inference(self):
        op = SatelliteCleanOp("test")
        lengths = op._infer_output_length({"data": 10})
        assert lengths == 10

    def test_output_length_none(self):
        op = SatelliteCleanOp("test")
        lengths = op._infer_output_length({"data": None})
        assert lengths is None
