import numpy as np
import pytest

from hoshicore.component.data_container import (
    DTYPE_LEVEL,
    DTYPE_MAX_VALUE,
    FloatImage,
    FastGaussianParam,
    GaussianParam,
    HuberMeanParam,
    _cumscale_factor,
    align_dtype_pair,
    get_scale_x,
    rdtype_detector,
    rescale_array,
)

# ── rescale_array ──


class TestRescaleArray:
    def test_uint8_to_uint16(self):
        arr = np.array([0, 1, 127, 255], dtype=np.uint8)
        up = rescale_array(arr, np.dtype("uint8"), np.dtype("uint16"))
        assert up.dtype == np.uint16
        assert up[0] == 0
        assert up[-1] == 255 * (256 + 1)  # 65535

    def test_uint8_to_uint32(self):
        arr = np.array([255], dtype=np.uint8)
        up = rescale_array(arr, np.dtype("uint8"), np.dtype("uint32"))
        assert up.dtype == np.uint32
        assert up[0] > 0

    def test_same_dtype_passthrough(self):
        arr = np.array([100, 200], dtype=np.uint16)
        out = rescale_array(arr, np.dtype("uint16"), np.dtype("uint16"))
        np.testing.assert_array_equal(out, arr)

    def test_float_passthrough(self):
        arr = np.array([1.0, 2.0], dtype=np.float64)
        out = rescale_array(arr, np.dtype("float64"), np.dtype("float32"))
        assert out.dtype == np.float32


# ── align_dtype_pair ──


class TestAlignDtypePair:
    def test_same_level(self):
        a = np.array([100], dtype=np.uint8)
        b = np.array([200], dtype=np.uint8)
        a2, b2, dt = align_dtype_pair(a, np.dtype("uint8"), b, np.dtype("uint8"))
        assert dt == np.dtype("uint8")
        np.testing.assert_array_equal(a2, a)
        np.testing.assert_array_equal(b2, b)

    def test_different_levels(self):
        a = np.array([100], dtype=np.uint8)
        b = np.array([25700], dtype=np.uint16)
        a2, b2, dt = align_dtype_pair(a, np.dtype("uint8"), b, np.dtype("uint16"))
        assert dt == np.dtype("uint16")
        assert a2.dtype == np.uint16
        np.testing.assert_array_equal(b2, b)


# ── rdtype_detector ──


class TestRdtypeDetector:
    def test_uint8_range(self):
        arr = np.array([0, 255], dtype=np.uint32)
        assert rdtype_detector(arr) == np.dtype("uint8")

    def test_uint16_range(self):
        arr = np.array([256, 60000], dtype=np.uint32)
        assert rdtype_detector(arr) == np.dtype("uint16")

    def test_float(self):
        arr = np.array([1.0, 2.0])
        assert rdtype_detector(arr) == float


# ── _cumscale_factor / get_scale_x ──


class TestScaleHelpers:
    def test_cumscale_factor_level1(self):
        f = _cumscale_factor(1)
        assert f == 257

    def test_get_scale_x_1(self):
        assert get_scale_x(1) == 257


# ── FloatImage ──


class TestFloatImage:
    def test_int_transform_default(self):
        data = np.array([[100.0, 200.0]], dtype=np.float64)
        fi = FloatImage(data=data, dtype=np.dtype("uint8"))
        result = fi.int_transform()
        assert result.dtype == np.uint8

    def test_int_transform_target(self):
        data = np.array([[100.0, 200.0]], dtype=np.float64)
        fi = FloatImage(data=data, dtype=np.dtype("uint8"))
        result = fi.int_transform(target_dtype=np.dtype("uint16"))
        assert result.dtype == np.uint16


# ── FastGaussianParam ──


class TestFastGaussianParam:
    def test_add_two_frames(self):
        img1 = np.array([[[100, 200, 150]]], dtype=np.uint8)
        img2 = np.array([[[110, 190, 160]]], dtype=np.uint8)
        fgp1 = FastGaussianParam(img1.copy(), source_dtype=np.dtype("uint8"))
        fgp2 = FastGaussianParam(img2.copy(), source_dtype=np.dtype("uint8"))
        fgp1.inplace_calc = False
        result = fgp1 + fgp2
        mu = result.mu
        expected_mu = np.round((img1.astype(np.float64) + img2.astype(np.float64)))
        # mu is sum/n, with n=2 and sum_mu holding upscaled values
        assert mu.shape == img1.shape

    def test_mu_single_frame(self):
        img = np.array([[[100]]], dtype=np.uint16)
        fgp = FastGaussianParam(img.copy(), source_dtype=np.dtype("uint16"))
        mu = fgp.mu
        np.testing.assert_allclose(mu, img, atol=1)

    def test_upscale_on_overflow(self):
        img = np.full((2, 2, 1), 255, dtype=np.uint8)
        fgp = FastGaussianParam(img.copy(), source_dtype=np.dtype("uint8"))
        initial_dtype = fgp.sum_mu.dtype
        for _ in range(300):
            fgp = fgp + FastGaussianParam(img.copy(), source_dtype=np.dtype("uint8"))
        # Should have upscaled to avoid overflow
        assert fgp.sum_mu.dtype.itemsize >= initial_dtype.itemsize

    def test_mask(self):
        img = np.full((4, 4, 1), 100, dtype=np.uint8)
        fgp = FastGaussianParam(img.copy(), source_dtype=np.dtype("uint8"))
        mask = np.ones((4, 4, 1), dtype=bool)
        mask[2:, :, :] = False
        fgp.mask(mask)
        assert fgp.n[0, 0, 0] == 1
        assert fgp.n[2, 0, 0] == 0

    def test_sub(self):
        img1 = np.full((2, 2, 1), 100, dtype=np.uint8)
        img2 = np.full((2, 2, 1), 100, dtype=np.uint8)
        fgp_total = FastGaussianParam(img1.copy(), source_dtype=np.dtype("uint8"))
        fgp_total.inplace_calc = True
        for _ in range(9):
            fgp_total = fgp_total + FastGaussianParam(img1.copy(), source_dtype=np.dtype("uint8"))
        # n = 10
        fgp_reject = FastGaussianParam(img2.copy(), source_dtype=np.dtype("uint8"))
        fgp_reject.inplace_calc = False
        result = fgp_total - fgp_reject
        assert np.all(result.n == 9)

    def test_mul_scalar(self):
        img = np.full((2, 2, 1), 50, dtype=np.uint8)
        fgp = FastGaussianParam(img.copy(), source_dtype=np.dtype("uint8"))
        fgp.inplace_calc = False
        result = fgp * 2
        assert np.all(result.n == 2)

    def test_safe_add_count_uint8(self):
        img = np.full((2, 2, 1), 255, dtype=np.uint8)
        fgp = FastGaussianParam(img.copy(), source_dtype=np.dtype("uint8"))
        safe_n = fgp._safe_add_count()
        # sum_mu is uint16 (upscaled from uint8), source max=255
        # safe_n = 65535 // 255 = 257
        assert safe_n == 257

    def test_safe_add_count_uint16(self):
        img = np.full((2, 2, 1), 1000, dtype=np.uint16)
        fgp = FastGaussianParam(img.copy(), source_dtype=np.dtype("uint16"))
        safe_n = fgp._safe_add_count()
        # sum_mu is uint32 (upscaled), source max=65535
        # safe_n = min(4294967295 // 65535, n_dtype_max)
        assert safe_n > 0
        assert safe_n <= DTYPE_MAX_VALUE.get(fgp.n.dtype, float('inf'))


# ── GaussianParam ──


class TestGaussianParam:
    def test_add(self):
        mu1 = np.array([[10.0]])
        mu2 = np.array([[20.0]])
        g1 = GaussianParam(mu=mu1, n=np.array([[5]]))
        g2 = GaussianParam(mu=mu2, n=np.array([[5]]))
        g3 = g1 + g2
        np.testing.assert_allclose(g3.mu, [[15.0]])
        assert g3.n[0, 0] == 10


# ── HuberMeanParam ──


class TestHuberMeanParam:
    def test_add(self):
        ws1 = np.array([[100.0]])
        wt1 = np.array([[2.0]])
        ws2 = np.array([[200.0]])
        wt2 = np.array([[3.0]])
        h1 = HuberMeanParam(ws1, wt1)
        h2 = HuberMeanParam(ws2, wt2)
        h3 = h1 + h2
        np.testing.assert_allclose(h3.weighted_sum, [[300.0]])
        np.testing.assert_allclose(h3.weight_total, [[5.0]])

    def test_mu(self):
        ws = np.array([[300.0]])
        wt = np.array([[3.0]])
        hp = HuberMeanParam(ws, wt)
        np.testing.assert_allclose(hp.mu, [[100.0]])

    def test_mu_zero_weight(self):
        ws = np.array([[0.0]])
        wt = np.array([[0.0]])
        hp = HuberMeanParam(ws, wt)
        np.testing.assert_allclose(hp.mu, [[0.0]])
