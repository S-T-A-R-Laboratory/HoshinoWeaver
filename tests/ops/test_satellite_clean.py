"""Tests for SatelliteCleanOp: sliding window alignment + median."""
import numpy as np
import pytest

from hoshicore.ops.satellite_clean_op import SatelliteCleanOp, _FrameSlot


class TestChainHomography:
    """Test _chain_homography correctness."""

    def _make_buffer(self, shifts):
        """Create a buffer of FrameSlots with known translation homographies."""
        from collections import deque
        buffer = deque()
        for i, (dx, dy) in enumerate(shifts):
            slot = _FrameSlot(
                original=np.zeros((100, 100, 3), dtype=np.uint8),
                geo=None,
            )
            if i < len(shifts) - 1:
                next_dx, next_dy = shifts[i + 1]
                # H maps frame[i] pixels to frame[i+1] pixels (translation)
                H = np.eye(3, dtype=np.float64)
                H[0, 2] = next_dx - dx
                H[1, 2] = next_dy - dy
                slot.H_to_next = H
            buffer.append(slot)
        return buffer

    def test_identity(self):
        buffer = self._make_buffer([(0, 0), (5, 0), (10, 0)])
        H = SatelliteCleanOp._chain_homography(buffer, 1, 1)
        np.testing.assert_allclose(H, np.eye(3))

    def test_forward_chain(self):
        buffer = self._make_buffer([(0, 0), (5, 0), (10, 0)])
        H = SatelliteCleanOp._chain_homography(buffer, 0, 2)
        # Should map frame[0] to frame[2]: translation of (10, 0)
        expected = np.eye(3, dtype=np.float64)
        expected[0, 2] = 10.0
        np.testing.assert_allclose(H, expected, atol=1e-10)

    def test_reverse_chain(self):
        buffer = self._make_buffer([(0, 0), (5, 0), (10, 0)])
        H = SatelliteCleanOp._chain_homography(buffer, 2, 0)
        expected = np.eye(3, dtype=np.float64)
        expected[0, 2] = -10.0
        np.testing.assert_allclose(H, expected, atol=1e-10)

    def test_forward_reverse_inverse(self):
        buffer = self._make_buffer([(0, 0), (3, 2), (7, 5), (12, 1)])
        H_fwd = SatelliteCleanOp._chain_homography(buffer, 0, 3)
        H_rev = SatelliteCleanOp._chain_homography(buffer, 3, 0)
        product = H_fwd @ H_rev
        np.testing.assert_allclose(product, np.eye(3), atol=1e-10)

    def test_none_homography_returns_none(self):
        from collections import deque
        buffer = deque()
        for _ in range(3):
            buffer.append(_FrameSlot(
                original=np.zeros((10, 10), dtype=np.uint8),
                geo=None, H_to_next=None))
        result = SatelliteCleanOp._chain_homography(buffer, 0, 2)
        assert result is None


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

            slot = _FrameSlot(original=shifted, geo=None)
            if i > 0:
                # H maps frame[i-1] to frame[i]: translate by (shift_x, 0)
                H = np.eye(3, dtype=np.float64)
                H[0, 2] = shift_x
                buffer[-1].H_to_next = H
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
