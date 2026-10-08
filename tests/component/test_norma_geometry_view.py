import numpy as np
import pytest

import hoshicore.component.norma.geometry_view as geometry_module
from hoshicore._custom_op.ops.detection import GraySource
from hoshicore._custom_op.ops import gray as gray_ops
from hoshicore.component.norma.detection import (DetectedStars,
                                                 _gray_u16_for_detection)
from hoshicore.component.norma.geometry_view import (GeometryView,
                                                      StarDetectionCache,
                                                      to_gray_f64,
                                                      to_median_gray_u16)
from hoshicore.component.norma.types import CameraModel, Intrinsics


def test_to_gray_f64_uses_project_bgr_channel_order():
    image = np.array([[[255, 0, 0], [0, 0, 255]]], dtype=np.uint8)

    gray = to_gray_f64(image)

    np.testing.assert_allclose(gray[0], [0.114, 0.299], atol=1e-6)


def test_detection_cache_keeps_pywt_and_median_results_lazy_and_separate(
        monkeypatch):
    calls = []

    def stars(x):
        return DetectedStars(np.array([[x, 1.0]]), np.ones(1))

    monkeypatch.setattr(
        geometry_module, "detect_star_points",
        lambda gray, mask=None, mode="auto": calls.append("pywt") or stars(1.0))
    monkeypatch.setattr(
        geometry_module, "detect_star_points_median",
        lambda gray, mask=None, threshold_ratio=1.0:
        calls.append("median") or stars(2.0))

    cache = StarDetectionCache(np.zeros((8, 12), dtype=np.float64))
    pywt = cache.pywt_stars
    assert cache.pywt_stars is pywt
    assert calls == ["pywt"]
    median = cache.median_stars
    assert cache.median_stars is median
    assert calls == ["pywt", "median"]

    camera = CameraModel(Intrinsics(20.0, 36.0, 24.0, 12, 8))
    pywt_view = GeometryView(pywt, camera)
    median_view = GeometryView(median, camera)
    assert pywt_view.stars is pywt
    assert median_view.stars is median
    assert not hasattr(pywt_view, "image_gray")


@pytest.mark.parametrize("mode", ["auto", "native_relaxed", "contour"])
def test_detection_cache_from_image_converts_gray_only_on_demand(monkeypatch, mode):
    def stars(x):
        return DetectedStars(np.array([[x, 1.0]]), np.ones(1))

    sources = []
    median_grays = []
    conversions = []
    modes = []
    real_to_gray = geometry_module.to_gray_f64

    def counting_to_gray(image):
        conversions.append(image.shape)
        return real_to_gray(image)

    monkeypatch.setattr(geometry_module, "to_gray_f64", counting_to_gray)
    monkeypatch.setattr(
        geometry_module, "detect_star_points",
        lambda source, mask=None, mode="auto":
        sources.append(source) or modes.append(mode) or stars(1.0))
    monkeypatch.setattr(
        geometry_module, "detect_star_points_median",
        lambda gray, mask=None, threshold_ratio=1.0:
        median_grays.append(gray) or stars(2.0))

    image = np.full((8, 12, 3), 1000, dtype=np.uint16)
    cache = StarDetectionCache.from_image(image, star_detection_mode=mode)
    cache.pywt_stars
    # The pywt detector receives the image; its CUDA backend converts on device.
    assert isinstance(sources[0], GraySource) and sources[0].raw is image
    assert modes == [mode]
    cache.median_stars
    # The median detector takes its uint16 gray straight from the image.
    assert median_grays[0].dtype == np.uint16
    assert conversions == []
    assert sources[0].host_gray() is cache.gray
    assert conversions == [(8, 12, 3)]


def test_detection_cache_requires_exactly_one_input():
    with pytest.raises(ValueError):
        StarDetectionCache()
    with pytest.raises(ValueError):
        StarDetectionCache(np.zeros((4, 4)), image=np.zeros((4, 4), dtype=np.uint16))


def test_median_gray_u16_matches_float64_quantization():
    rng = np.random.default_rng(3)
    for width in (1, 7, 8, 13, 64):
        image = rng.integers(0, 65536, size=(9, width, 3), dtype=np.uint16)
        image[0, 0] = 0
        image[-1, -1] = 65535
        expected = _gray_u16_for_detection(to_gray_f64(image))
        got = to_median_gray_u16(image)
        assert got.dtype == np.uint16
        np.testing.assert_array_equal(got, expected)
    mono = rng.integers(0, 65536, size=(5, 11), dtype=np.uint16)
    np.testing.assert_array_equal(to_median_gray_u16(mono),
                                  _gray_u16_for_detection(to_gray_f64(mono)))


def test_median_gray_u16_leaves_other_inputs_to_float64_path(monkeypatch):
    assert to_median_gray_u16(np.zeros((4, 4, 3), dtype=np.uint8)) is None
    assert to_median_gray_u16(np.zeros((4, 4, 3), dtype=np.float32)) is None
    monkeypatch.setattr(gray_ops.cv2, "cvtColor",
                        lambda *args: np.full((4, 4), 65535.5, dtype=np.float32))
    assert to_median_gray_u16(np.zeros((4, 4, 3), dtype=np.uint16)) is None


def test_median_stars_from_image_match_gray_cache():
    rng = np.random.default_rng(4)
    height, width = 240, 320
    yy, xx = np.mgrid[:height, :width]
    field = rng.normal(2000.0, 30.0, size=(height, width))
    for y, x, amplitude in zip(rng.uniform(8, height - 8, 60),
                               rng.uniform(8, width - 8, 60),
                               rng.uniform(3000, 40000, 60)):
        field += amplitude * np.exp(-((yy - y)**2 + (xx - x)**2) / 4.0)
    image = np.clip(field[..., None] * [0.9, 1.0, 1.1], 0, 65535).astype(np.uint16)

    expected = StarDetectionCache(to_gray_f64(image)).median_stars
    got = StarDetectionCache.from_image(image).median_stars
    assert len(expected.positions) > 20
    for name in ("positions", "volumes", "intensities"):
        np.testing.assert_array_equal(getattr(got, name), getattr(expected, name))
