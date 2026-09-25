import pytest

from hoshicore.component.norma.intrinsics_from_exif import (
    intrinsics_from_exif,
)


def test_35mm_equivalent_focal_is_sufficient_without_sensor_metadata():
    intrinsics = intrinsics_from_exif(
        {"Exif.Photo.FocalLengthIn35mmFilm": "28/1"},
        6000,
        4000,
    )

    assert intrinsics is not None
    assert intrinsics.focal_length_mm == pytest.approx(28.0)
    assert intrinsics.sensor_width_mm == pytest.approx(36.0)
    assert intrinsics.sensor_height_mm == pytest.approx(24.0)


def test_35mm_equivalent_focal_takes_priority_over_physical_focal_metadata():
    intrinsics = intrinsics_from_exif(
        {
            "Exif.Photo.FocalLengthIn35mmFilm": "35/1",
            "Exif.Photo.FocalLength": "50/1",
            "Exif.Photo.FocalPlaneXResolution": "500/3",
            "Exif.Photo.FocalPlaneYResolution": "500/3",
            "Exif.Photo.FocalPlaneResolutionUnit": "4",
        },
        6000,
        4000,
    )

    assert intrinsics is not None
    assert intrinsics.focal_length_mm == pytest.approx(35.0)
    assert intrinsics.sensor_width_mm == pytest.approx(36.0)
