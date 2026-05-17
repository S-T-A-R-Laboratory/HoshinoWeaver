import cv2
import numpy as np
import pytest
from scipy.stats import norm as scipy_norm

from hoshicore.component.noise_equalization import (
    compute_adaptive_n_sigma,
    equalize_noise,
    threshold_max_merge,
)


class TestComputeAdaptiveNSigma:
    def test_known_values(self):
        assert compute_adaptive_n_sigma(50) == pytest.approx(
            max(3.0, float(scipy_norm.ppf(1.0 - 0.01 / 50))), abs=0.01)
        assert compute_adaptive_n_sigma(100) == pytest.approx(
            max(3.0, float(scipy_norm.ppf(1.0 - 0.01 / 100))), abs=0.01)

    def test_floor_at_3(self):
        assert compute_adaptive_n_sigma(1) >= 3.0

    def test_monotonically_increasing(self):
        prev = compute_adaptive_n_sigma(10)
        for n in [50, 100, 200, 500, 1000]:
            cur = compute_adaptive_n_sigma(n)
            assert cur >= prev
            prev = cur


class TestThresholdMaxMerge:
    def _make_test_data(self, h=32, w=32, channels=3, bg=100.0, star_val=250.0):
        mean_img = np.full((h, w, channels), bg, dtype=np.float64)
        std_img = np.full((h, w, channels), 5.0, dtype=np.float64)
        frame_bg = np.full((h, w, channels), bg + 2, dtype=np.float64)  # slightly above mean
        # Add a bright "star trail" line
        frame_star = frame_bg.copy()
        frame_star[h // 2, :, :] = star_val
        return mean_img, std_img, frame_bg, frame_star

    def test_background_preserved(self):
        mean_img, std_img, frame_bg, _ = self._make_test_data()
        result = mean_img.copy()
        threshold_max_merge(frame_bg, mean_img, std_img, result, n_sigma=3.0)
        # Background frame should not change result significantly (frame_bg is within n_sigma)
        np.testing.assert_allclose(result, mean_img, atol=1)

    def test_star_signal_preserved(self):
        mean_img, std_img, _, frame_star = self._make_test_data()
        result = mean_img.copy()
        threshold_max_merge(frame_star, mean_img, std_img, result, n_sigma=3.0)
        # Star trail row should have bright values
        h = mean_img.shape[0]
        assert np.all(result[h // 2, :, :] > mean_img[h // 2, :, :] + 50)
        # Non-star rows should remain at mean
        np.testing.assert_allclose(result[0, :, :], mean_img[0, :, :], atol=1)

    def test_weight_scales_signal(self):
        mean_img, std_img, _, frame_star = self._make_test_data(star_val=250.0)
        h = mean_img.shape[0]

        result_full = mean_img.copy()
        threshold_max_merge(frame_star, mean_img, std_img, result_full,
                            n_sigma=3.0, weight=1.0)
        star_brightness_full = result_full[h // 2, 0, 0]

        result_half = mean_img.copy()
        threshold_max_merge(frame_star, mean_img, std_img, result_half,
                            n_sigma=3.0, weight=0.5)
        star_brightness_half = result_half[h // 2, 0, 0]

        assert star_brightness_half < star_brightness_full
        assert star_brightness_half == pytest.approx(250.0 * 0.5, abs=1)

    def test_morph_open_removes_isolated_noise(self):
        mean_img = np.full((32, 32, 1), 100.0, dtype=np.float64)
        std_img = np.full((32, 32, 1), 5.0, dtype=np.float64)
        frame = mean_img.copy()
        # Single pixel spike (isolated noise)
        frame[10, 10, 0] = 300.0
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        result = mean_img.copy()
        threshold_max_merge(frame, mean_img, std_img, result,
                            n_sigma=3.0, morph_kernel=kernel)
        # Isolated pixel should be removed by opening
        np.testing.assert_allclose(result[10, 10, 0], mean_img[10, 10, 0], atol=1)

    def test_morph_open_keeps_wide_structure(self):
        mean_img = np.full((32, 32, 1), 100.0, dtype=np.float64)
        std_img = np.full((32, 32, 1), 5.0, dtype=np.float64)
        frame = mean_img.copy()
        # Wide 3x3 block (survives RECT 3x3 opening)
        frame[14:17, 14:17, 0] = 300.0
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        result = mean_img.copy()
        threshold_max_merge(frame, mean_img, std_img, result,
                            n_sigma=3.0, morph_kernel=kernel)
        assert np.any(result[14:17, 14:17, 0] > 200)

    def test_2d_grayscale(self):
        mean_img = np.full((16, 16), 100.0, dtype=np.float64)
        std_img = np.full((16, 16), 5.0, dtype=np.float64)
        frame = mean_img.copy()
        frame[8, :] = 250.0
        result = mean_img.copy()
        threshold_max_merge(frame, mean_img, std_img, result, n_sigma=3.0)
        assert np.all(result[8, :] > 200)


class TestEqualizeNoise:
    def test_reduces_spatial_variance(self):
        rng = np.random.default_rng(42)
        n_frames = 100
        h, w = 100, 100

        y, x = np.ogrid[:h, :w]
        cy, cx = h // 2, w // 2
        r_sq = ((y - cy)**2 + (x - cx)**2) / (min(h, w) / 2)**2
        sigma_map = 10.0 * (1 + r_sq)

        bg_value = 30000.0
        frames_2d = np.array([
            bg_value + rng.standard_normal((h, w)) * sigma_map
            for _ in range(n_frames)
        ])

        frames = np.stack([frames_2d] * 3, axis=-1)

        mean_img = np.mean(frames, axis=0)
        std_img = np.std(frames, axis=0, ddof=1)
        max_img = np.max(frames, axis=0)
        n_img = np.full((h, w, 3), n_frames, dtype=np.uint32)

        corrected = equalize_noise(max_img, mean_img, std_img, n_img)

        residual_before = max_img - mean_img
        residual_after = corrected - mean_img

        spatial_std_before = np.std(residual_before)
        spatial_std_after = np.std(residual_after)

        assert spatial_std_after < spatial_std_before

    def test_no_background_returns_unchanged(self):
        h, w = 16, 16
        max_img = np.ones((h, w, 3), dtype=np.float64) * 100
        mean_img = np.ones((h, w, 3), dtype=np.float64) * 90
        std_img = np.ones((h, w, 3), dtype=np.float64) * 5
        n_img = np.zeros((h, w, 3), dtype=np.uint32)  # no valid pixels
        result = equalize_noise(max_img, mean_img, std_img, n_img)
        np.testing.assert_array_equal(result, max_img)
