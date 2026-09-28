import numpy as np
import pytest

from bench.cpu.detection_profile import make_starfield, profile_frame, summarize_profiles
from hoshicore.component.norma.geometry_view import StarDetectionCache


@pytest.fixture
def cpu_backend(monkeypatch):
    pytest.importorskip("hoshicore._custom_op._C")
    monkeypatch.setenv("HNW_CUSTOM_OPS_FALLBACK", "cpu")
    monkeypatch.setenv("HNW_CUSTOM_OPS_THREADS", "2")


def test_profile_preserves_production_stars_and_accounts_for_time(cpu_backend):
    image, mask = make_starfield(160, 224, seed=17)
    expected = StarDetectionCache.from_image(image, mask).median_stars
    sample, actual = profile_frame(image, mask)
    assert len(actual.positions) > 0
    for name in ("positions", "volumes", "intensities"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    assert sample["total_sec"] > 0
    stages = [value for name, value in sample.items() if name != "total_sec"]
    assert all(value >= 0 for value in stages)
    assert sum(stages) == pytest.approx(sample["total_sec"])
    assert {"gray_sec", "pixel_sec", "find_contours_sec", "geometry_intensity_filter_sec"} == (
        sample.keys() - {"total_sec"})


def test_profile_summary_keeps_samples_and_uses_median():
    rows = [{"total_sec": x, "gray_sec": x / 2} for x in (1.0, 9.0, 2.0)]
    result = summarize_profiles(rows)
    assert result["total"]["samples_sec"] == [1.0, 9.0, 2.0]
    assert result["total"]["median_sec"] == 2.0
    assert result["gray"]["median_sec"] == 1.0


def test_profile_starfield_is_repeatable_and_has_a_sky_mask():
    first, mask = make_starfield(128, 192, seed=23)
    second, second_mask = make_starfield(128, 192, seed=23)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(mask, second_mask)
    assert first.dtype == np.uint16 and first.shape == (128, 192, 3)
    assert 0 < np.count_nonzero(mask) < mask.size
