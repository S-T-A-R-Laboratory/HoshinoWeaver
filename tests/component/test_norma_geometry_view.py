import numpy as np
import pytest

import hoshicore.component.norma.geometry_view as geometry_module
from hoshicore._custom_op.ops.detection import GraySource
from hoshicore.component.norma.detection import DetectedStars
from hoshicore.component.norma.geometry_view import (GeometryView,
                                                      StarDetectionCache,
                                                      to_gray_f64)
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
        lambda gray, mask=None: calls.append("pywt") or stars(1.0))
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


def test_detection_cache_from_image_converts_gray_only_on_demand(monkeypatch):
    sources = []
    conversions = []
    real_to_gray = geometry_module.to_gray_f64

    def counting_to_gray(image):
        conversions.append(image.shape)
        return real_to_gray(image)

    def stars():
        return DetectedStars(np.array([[1.0, 1.0]]), np.ones(1))

    monkeypatch.setattr(geometry_module, "to_gray_f64", counting_to_gray)
    monkeypatch.setattr(geometry_module, "detect_star_points",
                        lambda source, mask=None: sources.append(source) or stars())
    monkeypatch.setattr(geometry_module, "detect_star_points_median",
                        lambda gray, mask=None, threshold_ratio=1.0: stars())
    image = np.full((8, 12, 3), 1000, dtype=np.uint16)
    cache = StarDetectionCache.from_image(image)
    cache.pywt_stars
    assert isinstance(sources[0], GraySource) and sources[0].raw is image
    assert conversions == []
    cache.median_stars
    assert conversions == [(8, 12, 3)]
    assert sources[0].host_gray() is cache.gray
    assert conversions == [(8, 12, 3)]


def test_detection_cache_requires_exactly_one_input():
    with pytest.raises(ValueError):
        StarDetectionCache()
    with pytest.raises(ValueError):
        StarDetectionCache(np.zeros((4, 4)), image=np.zeros((4, 4), dtype=np.uint16))
