import numpy as np

import hoshicore.component.norma.geometry_view as geometry_module
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
