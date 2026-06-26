import cv2
import numpy as np
import pytest
from hoshicore.component.star_shrink import morph_shrink_luma


class TestMorphShrinkLuma:
    def test_color_preserved_in_center_of_patch(self):
        """5×5 彩色色块中心像素经 luma 腐蚀后 hue 不变（a/b 通道不变）。"""
        img = np.zeros((11, 11, 3), dtype=np.uint8)
        img[3:8, 3:8] = (0, 0, 200)  # BGR: bright red patch
        result = morph_shrink_luma(img, ksize=3, shape="RECT", times=1)

        orig_f = img.astype(np.float32) / 255.0
        res_f = result.astype(np.float32) / 255.0
        orig_lab = cv2.cvtColor(orig_f, cv2.COLOR_BGR2LAB)
        res_lab = cv2.cvtColor(res_f, cv2.COLOR_BGR2LAB)

        # 中心 (5,5) 被 patch 包围，腐蚀后仍为红色
        np.testing.assert_allclose(orig_lab[5, 5, 1:], res_lab[5, 5, 1:], atol=1.0)
        assert res_lab[5, 5, 0] > 0

    def test_isolated_pixel_luma_eroded_to_zero(self):
        """孤立灰白像素（a=b=0）被 3×3 腐蚀后 L→0，BGR 全部为 0。"""
        img = np.zeros((7, 7, 3), dtype=np.uint8)
        img[3, 3] = (200, 200, 200)  # 灰白像素：LAB ab≈0，反变换安全
        result = morph_shrink_luma(img, ksize=3, shape="RECT", times=1)
        assert result[3, 3, 0] == 0
        assert result[3, 3, 1] == 0
        assert result[3, 3, 2] == 0

    def test_grayscale_path(self):
        """灰度图走普通腐蚀分支，shape 和 dtype 不变。"""
        img = np.zeros((9, 9), dtype=np.uint8)
        img[4, 4] = 200
        result = morph_shrink_luma(img, ksize=3, shape="RECT", times=1)
        assert result.shape == img.shape
        assert result.dtype == img.dtype
        assert result[4, 4] < img[4, 4]

    def test_dtype_preserved_uint16(self):
        img = np.full((5, 5, 3), 32000, dtype=np.uint16)
        result = morph_shrink_luma(img, ksize=3, shape="RECT", times=1)
        assert result.dtype == np.uint16

    def test_uniform_image_unchanged(self):
        """均匀图像腐蚀后不变。"""
        img = np.full((9, 9, 3), 100, dtype=np.uint8)
        result = morph_shrink_luma(img, ksize=3, shape="RECT", times=1)
        np.testing.assert_array_equal(result, img)

    def test_ratio1_matches_opencv_iterations(self):
        """times=N, ratio=1.0 must exactly match old implementation (OpenCV iterations=N)."""
        import cv2
        from hoshicore.component.star_shrink import get_morph_kernel
        rng = np.random.default_rng(42)
        img = rng.integers(1000, 60000, (60, 60, 3)).astype(np.uint16)
        ksize, times = 5, 2
        kernel = get_morph_kernel("CIRCLE", ksize)
        # Replicate old implementation directly
        img_f = img.astype(np.float32) / 65535.0
        lab = cv2.cvtColor(img_f, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.morphologyEx(lab[:, :, 0], cv2.MORPH_ERODE, kernel, iterations=times)
        expected_f = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), 0.0, 1.0)
        expected = np.round(expected_f * 65535).astype(np.uint16)
        result = morph_shrink_luma(img, ksize=ksize, times=times, ratio=1.0)
        np.testing.assert_array_equal(result, expected)

    def test_default_ratio_equals_inv_times(self):
        """ratio=None default must equal ratio=1/times for times in {1, 2, 3}."""
        rng = np.random.default_rng(7)
        img = rng.integers(1000, 60000, (60, 60, 3)).astype(np.uint16)
        for times in [1, 2, 3]:
            r_auto = morph_shrink_luma(img, ksize=5, times=times, ratio=None)
            r_explicit = morph_shrink_luma(img, ksize=5, times=times, ratio=1.0 / times)
            np.testing.assert_array_equal(r_auto, r_explicit,
                                          err_msg=f"Failed for times={times}")

    def test_lower_ratio_weaker_than_ratio1(self):
        """ratio=0.5 must produce less erosion than ratio=1.0 (higher total brightness)."""
        rng = np.random.default_rng(13)
        img = rng.integers(1000, 60000, (60, 60, 3)).astype(np.uint16)
        strong = morph_shrink_luma(img, ksize=7, times=2, ratio=1.0)
        gradual = morph_shrink_luma(img, ksize=7, times=2, ratio=0.5)
        assert int(gradual.sum()) > int(strong.sum())

    def test_grayscale_with_ratio(self):
        """2D grayscale input must work with ratio<1 without error."""
        rng = np.random.default_rng(99)
        img = rng.integers(1000, 60000, (60, 60)).astype(np.uint16)
        result = morph_shrink_luma(img, ksize=5, times=2, ratio=0.5)
        assert result.shape == img.shape
        assert result.dtype == img.dtype
        assert int(result.sum()) <= int(img.sum())


from hoshicore.component.star_shrink import peak_recovery


class TestPeakRecovery:
    def test_bright_star_recovered_more_than_dim(self):
        """
        两颗星：亮星 above_bg_n=1.0，暗星 above_bg_n≈0.1。
        recovery_scale=0.5 时，亮星 weight=1.0，暗星 weight≈0.2。
        亮星恢复量远大于暗星。
        """
        img = np.zeros((15, 15, 3), dtype=np.uint16)
        img[3, 3] = (60000, 60000, 60000)   # 亮星
        img[11, 11] = (6000, 6000, 6000)    # 暗星（约 1/10）
        eroded = np.zeros_like(img)

        result = peak_recovery(img, eroded, bg_ksize=3, strength=1.0, scale=0.5)
        assert result[3, 3, 0] > result[11, 11, 0]
        assert result[3, 3, 0] > 50000  # 亮星几乎完全恢复

    def test_background_pixel_not_recovered(self):
        """均匀背景图：above_bg ≈ 0，recovery weight ≈ 0，结果等于 eroded。"""
        img = np.full((9, 9, 3), 1000, dtype=np.uint16)
        eroded = np.full((9, 9, 3), 500, dtype=np.uint16)
        result = peak_recovery(img, eroded, bg_ksize=3, strength=1.0, scale=0.1)
        np.testing.assert_array_equal(result, eroded)

    def test_flat_image_returns_eroded(self):
        """完全平坦图像（peak_bg ≈ 0）直接返回 eroded。"""
        img = np.full((7, 7, 3), 5000, dtype=np.uint16)
        eroded = np.full((7, 7, 3), 4000, dtype=np.uint16)
        result = peak_recovery(img, eroded, bg_ksize=3, strength=1.0, scale=0.2)
        np.testing.assert_array_equal(result, eroded)

    def test_output_dtype_preserved(self):
        img = np.zeros((7, 7, 3), dtype=np.uint16)
        img[3, 3] = (60000, 60000, 60000)
        eroded = np.zeros_like(img)
        result = peak_recovery(img, eroded, bg_ksize=3, strength=0.8, scale=0.3)
        assert result.dtype == np.uint16

    def test_strength_zero_returns_eroded(self):
        """strength=0 等同于不恢复。"""
        img = np.zeros((7, 7, 3), dtype=np.uint8)
        img[3, 3] = (200, 200, 200)
        eroded = np.zeros_like(img)
        result = peak_recovery(img, eroded, bg_ksize=3, strength=0.0, scale=0.1)
        np.testing.assert_array_equal(result, eroded)


from hoshicore.component.star_shrink import deringing


class TestDeringing:
    def test_output_never_below_shrink_img(self):
        """max(shrunk, blurred) 保证输出 >= shrunk。"""
        rng = np.random.default_rng(42)
        img = (rng.random((50, 50, 3)) * 255).astype(np.uint8)
        shrunk = (img * 0.5).astype(np.uint8)
        result = deringing(img, shrunk, algo="gaussian", ksize=7)
        assert np.all(result >= shrunk)

    def test_float32_precision_not_quantized(self):
        """旧代码将 float32 降位到 uint8 处理，有量化精度损失（≤256 unique 值）。
        新代码应保持 float32 精度（远超 256 unique 值）。"""
        y, x = np.mgrid[0:100, 0:100]
        vals = ((y * 0.006 + x * 0.001 + 0.1).clip(0, 1)).astype(np.float32)
        img = np.stack([vals, vals, vals], axis=-1)
        shrunk = (img * 0.3).astype(np.float32)
        result = deringing(img, shrunk, algo="gaussian", ksize=7)
        assert result.dtype == np.float32
        unique_vals = len(np.unique(result[:, :, 0]))
        assert unique_vals > 256

    def test_uint16_precision_not_quantized_to_uint8_steps(self):
        """旧代码降位到 uint8 后反变换，uint16 输出至多 256 种不同值。
        新代码 float32 全程处理，应输出远超 256 种不同值。"""
        y, x = np.mgrid[0:100, 0:100]
        vals = (y * 600 + x * 10 + 5000).clip(0, 65535).astype(np.uint16)
        img = np.stack([vals, vals, vals], axis=-1)
        shrunk = (vals * 0.3).astype(np.uint16)
        shrunk = np.stack([shrunk, shrunk, shrunk], axis=-1)
        result = deringing(img, shrunk, algo="gaussian", ksize=7)
        unique_vals = len(np.unique(result[:, :, 0]))
        assert unique_vals > 256

    def test_mean_algo_works(self):
        img = np.full((9, 9, 3), 200, dtype=np.uint8)
        shrunk = np.full((9, 9, 3), 100, dtype=np.uint8)
        result = deringing(img, shrunk, algo="mean", ksize=3)
        assert result.dtype == np.uint8
        assert np.all(result >= shrunk)


from hoshicore.ops.star_ops import SHRINK_MODE_PRESETS, _star_shrink_pipeline


class TestStarShrinkPipeline:
    def _make_star_img(self) -> np.ndarray:
        """BGR uint8 with one bright star patch on dark background."""
        img = np.full((64, 64, 3), 30, dtype=np.uint8)
        img[28:37, 28:37] = 220
        return img

    def _base_configs(self) -> dict:
        return {
            'detect_method': 'threshold', 'detect_ksize': 5,
            'detect_threshold': 1.0, 'detect_open': 0, 'detect_dilate': 0,
            'dog_sigma_small': 1.5, 'dog_sigma_large': 12.0,
        }

    def test_presets_have_required_keys(self):
        required = {
            'shrink_ksize', 'shrink_times', 'shrink_ratio',
            'deringing_ksize',
        }
        for mode_name, preset in SHRINK_MODE_PRESETS.items():
            missing = required - preset.keys()
            assert not missing, f"Mode '{mode_name}' missing keys: {missing}"

    def test_all_presets_run_without_error(self):
        img = self._make_star_img()
        for name in SHRINK_MODE_PRESETS:
            result = _star_shrink_pipeline(img.copy(), {**self._base_configs(), 'mode': name})
            assert result.shape == img.shape, f"Shape mismatch for mode '{name}'"
            assert result.dtype == img.dtype, f"dtype mismatch for mode '{name}'"

    def test_removal_more_aggressive_than_light(self):
        """removal must reduce the star region more than light does."""
        img = self._make_star_img()
        removal = _star_shrink_pipeline(img, {**self._base_configs(), 'mode': 'removal'})
        light = _star_shrink_pipeline(img, {**self._base_configs(), 'mode': 'light'})
        star = np.s_[28:37, 28:37, :]
        assert float(light[star].mean()) > float(removal[star].mean())

    def test_unknown_mode_raises(self):
        img = self._make_star_img()
        with pytest.raises(ValueError, match="Unknown mode"):
            _star_shrink_pipeline(img, {**self._base_configs(), 'mode': 'nonexistent'})

    def test_output_shape_and_dtype_preserved(self):
        img = self._make_star_img()
        result = _star_shrink_pipeline(img, {**self._base_configs(), 'mode': 'moderate'})
        assert result.shape == img.shape
        assert result.dtype == img.dtype

    def test_custom_mode_ratio_zero_means_auto(self):
        """shrink_ratio=0.0 sentinel in custom mode must equal ratio=1/times."""
        img = self._make_star_img()
        base = {
            **self._base_configs(),
            'mode': 'custom',
            'shrink_ksize': 3, 'shrink_times': 2, 'shrink_shape': 'CIRCLE',
            'deringing_ksize': 11, 'blend_method': 'hard',
        }
        result_auto = _star_shrink_pipeline(img.copy(), {**base, 'shrink_ratio': 0.0})
        result_explicit = _star_shrink_pipeline(img.copy(), {**base, 'shrink_ratio': 0.5})
        np.testing.assert_array_equal(result_auto, result_explicit)
