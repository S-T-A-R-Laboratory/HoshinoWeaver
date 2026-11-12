"""EXIF → 相机内参推导的契约测试。

覆盖两条推导路径（像素密度优先、35mm 等效回退）、归一化约定（长边 = 36mm，
任何朝向/画幅比都必须 fx == fy）、以及两条路径的交叉校验。
"""
import pytest

from hoshicore.component.norma.frame_align import build_camera_candidate
from hoshicore.component.norma.intrinsics_from_exif import (
    exif_focal_sources,
    intrinsics_from_exif,
    intrinsics_from_fisheye_estimate,
    intrinsics_from_focal_equiv,
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


def test_pixel_density_derivation_takes_priority_over_35mm_equivalent():
    """像素密度路径对裁切免疫，因此优先于整幅画幅的等效焦距标签。"""
    exif = {
        "Exif.Photo.FocalLengthIn35mmFilm": "35/1",
        "Exif.Photo.FocalLength": "50/1",
        "Exif.Photo.FocalPlaneXResolution": "500/3",
        "Exif.Photo.FocalPlaneYResolution": "500/3",
        "Exif.Photo.FocalPlaneResolutionUnit": "4",
    }
    sources = exif_focal_sources(exif, 6000, 4000)

    assert sources.pixel_density_focal_px == pytest.approx(50.0 * 500.0 / 3.0)
    assert sources.focal_equiv_focal_px == pytest.approx(35.0 * 6000 / 36.0)
    assert sources.ratio == pytest.approx(1.4286, abs=1e-3)

    intrinsics = intrinsics_from_exif(exif, 6000, 4000)

    assert intrinsics is not None
    # FocalLength + 像素密度（而非 35mm 等效标签）给出精确内参
    assert intrinsics.focal_length_mm == pytest.approx(50.0)
    assert intrinsics.sensor_width_mm == pytest.approx(36.0)
    assert intrinsics.sensor_height_mm == pytest.approx(24.0)


def test_35mm_equivalent_is_used_when_pixel_density_is_unavailable():
    intrinsics = intrinsics_from_exif(
        {"Exif.Photo.FocalLength": "17/1",
         "Exif.Photo.FocalLengthIn35mmFilm": "17/1"},
        6048,
        4024,
    )

    assert intrinsics is not None
    # 仅有 FocalLength、没有 FocalPlane* → 必须回退到等效焦距路径
    assert intrinsics.focal_length_mm == pytest.approx(17.0)
    assert intrinsics.sensor_width_mm == pytest.approx(36.0)


def test_no_usable_focal_tags_returns_none():
    assert intrinsics_from_exif({"Exif.Image.Model": "NIKON Z 6"}, 6048,
                                4024) is None
    assert intrinsics_from_exif({"Exif.Photo.FocalLength": "0/1"}, 6048,
                                4024) is None


@pytest.mark.parametrize(
    ("width", "height", "long_side"),
    [
        (6048, 4024, 6048),  # 3:2 landscape
        (4024, 6048, 6048),  # 3:2 portrait
        (4000, 3000, 4000),  # 4:3 landscape
        (3000, 4000, 4000),  # 4:3 portrait
        (3000, 3000, 3000),  # square
    ],
)
def test_equivalent_focal_normalises_on_the_long_side(width, height, long_side):
    """任何朝向/画幅比下都必须是各向同性，且长边锚定 36mm。"""
    intrinsics = intrinsics_from_focal_equiv(17.0, width, height)

    expected_focal_px = 17.0 * long_side / 36.0
    assert intrinsics.K[0, 0] == pytest.approx(expected_focal_px)
    assert intrinsics.K[1, 1] == pytest.approx(expected_focal_px)
    # sensor 两轴与图像像素边等比例，长边恒为 36mm
    assert intrinsics.sensor_width_mm == pytest.approx(36.0 * width / long_side)
    assert intrinsics.sensor_height_mm == pytest.approx(36.0 * height /
                                                        long_side)
    assert (intrinsics.sensor_width_mm / intrinsics.sensor_height_mm
            == pytest.approx(width / height, rel=1e-9))


def test_portrait_equivalent_focal_is_not_2_25x_anisotropic():
    """回归：3:2 竖幅曾被当作 fx=W/36、fy=H/24，得到 2.25 倍各向异性。"""
    intrinsics = intrinsics_from_exif(
        {"Exif.Photo.FocalLengthIn35mmFilm": "17/1"}, 4024, 6048)

    assert intrinsics is not None
    assert intrinsics.K[0, 0] / intrinsics.K[1, 1] == pytest.approx(1.0)
    assert intrinsics.K[0, 0] == pytest.approx(17.0 * 6048 / 36.0)
    assert intrinsics.sensor_width_mm < intrinsics.sensor_height_mm


def test_exif_pixel_density_camera_is_isotropic_for_portrait_capture():
    """像素密度相同（方形像素）时，竖幅也必须得到各向同性 K。"""
    exif = {
        "Exif.Photo.FocalLength": "17/1",
        "Exif.Photo.FocalPlaneXResolution": "1682/1",
        "Exif.Photo.FocalPlaneYResolution": "1682/1",
        "Exif.Photo.FocalPlaneResolutionUnit": "3",  # cm
    }
    intrinsics = intrinsics_from_exif(exif, 4024, 6048)

    assert intrinsics is not None
    assert intrinsics.K[0, 0] == pytest.approx(intrinsics.K[1, 1])
    assert intrinsics.K[0, 0] == pytest.approx(17.0 * 1682.0 / 10.0)


@pytest.mark.parametrize(("width", "height"), [(6000, 4000), (4000, 6000)])
def test_fisheye_estimate_is_isotropic_and_uses_short_side(width, height):
    """180° 假设：短边对应 θ_max = π/2，两轴同值。"""
    intrinsics = intrinsics_from_fisheye_estimate(width, height)

    expected_focal_px = min(width, height) / 3.141592653589793
    assert intrinsics.K[0, 0] == pytest.approx(expected_focal_px)
    assert intrinsics.K[1, 1] == pytest.approx(expected_focal_px)


def test_focal_metadata_ratio_is_none_without_both_sources():
    sources = exif_focal_sources(
        {"Exif.Photo.FocalLengthIn35mmFilm": "17/1"}, 6048, 4024)

    assert sources.pixel_density_focal_px is None
    assert sources.ratio is None


def test_camera_candidate_reports_focal_metadata_ratio():
    exif = {
        "Exif.Photo.FocalLengthIn35mmFilm": "17/1",
        "Exif.Photo.FocalLength": "17/1",
        "Exif.Photo.FocalPlaneXResolution": "1682/1",
        "Exif.Photo.FocalPlaneYResolution": "1682/1",
        "Exif.Photo.FocalPlaneResolutionUnit": "3",
    }
    candidate = build_camera_candidate(exif, (6048, 4024, 3), "distortion")

    assert candidate.focal_metadata_ratio == pytest.approx(1.0, abs=0.05)
    # 裁切后的图像会让 35mm 等价标签偏小，比值随之升高并保持可读
    cropped = build_camera_candidate(exif, (3024, 2012, 3), "distortion")
    assert cropped.focal_metadata_ratio == pytest.approx(2.0, abs=0.05)


def test_invalid_focal_equiv_raises_readable_error():
    with pytest.raises(ValueError) as excinfo:
        intrinsics_from_focal_equiv(0.0, 6000, 4000)

    assert "got 0.0, 6000" in str(excinfo.value)


def _six_d_mark_ii_tags():
    # The actual rational values in IMG_6239.CR2--IMG_6270.CR2. No dataset is
    # needed: the faulty density alone reproduces the camera initialization.
    return {
        "Exif.Image.Make": "Canon",
        "Exif.Image.Model": "Canon EOS 6D Mark II",
        "Exif.Photo.FocalLength": "14/1",
        "Exif.Photo.FocalPlaneXResolution": "6240000/935",
        "Exif.Photo.FocalPlaneYResolution": "4160000/619",
        "Exif.Photo.FocalPlaneResolutionUnit": "2",
        "Exif.Photo.PixelXDimension": "6240",
        "Exif.Photo.PixelYDimension": "4160",
    }


def test_known_raw_density_uses_native_pixel_pitch_instead_of_preview_geometry():
    tags = _six_d_mark_ii_tags()
    candidate = build_camera_candidate(tags, (4180, 6264, 3), "distortion")
    intrinsics = candidate.camera.intrinsics
    expected = 14.0 * 6240 / 35.9
    assert intrinsics.K[0, 0] == pytest.approx(expected)
    assert intrinsics.K[1, 1] == pytest.approx(expected)
    # LibRaw retains a few border pixels beyond EXIF's 6240x4160 output crop.
    assert intrinsics.sensor_width_mm == pytest.approx(6264 * 35.9 / 6240)
    assert candidate.init_source == "exif"
    assert candidate.focal_metadata_ratio is None
    assert tags["Exif.Photo.FocalPlaneXResolution"] == "6240000/935"


@pytest.mark.parametrize(("make", "model", "width", "sensor_width"), [
    ("Canon", "Canon EOS 5D Mark IV", 6720, 36.0),
    ("Canon", "EOS 6D Mark II", 6240, 35.9),
    ("Canon", "Canon EOS 20D", 3504, 22.5),
    ("FUJIFILM", "XF10", 6000, 23.5),
    (" fujifilm ", " FUJIFILM  xf10 ", 6000, 23.5),
])
def test_camera_density_exceptions_match_normalized_make_and_complete_model(
        make, model, width, sensor_width):
    tags = _six_d_mark_ii_tags()
    tags.update({
        "Exif.Image.Make": make, "Exif.Image.Model": model,
        "Exif.Photo.FocalPlaneXResolution": "1000",
        "Exif.Photo.FocalPlaneYResolution": "1000",
        "Exif.Photo.FocalPlaneResolutionUnit": "4",
    })
    intrinsics = intrinsics_from_exif(tags, width, round(width * 2 / 3))
    assert intrinsics.K[0, 0] == pytest.approx(14.0 * width / sensor_width)
    assert intrinsics.K[1, 1] == pytest.approx(intrinsics.K[0, 0])


@pytest.mark.parametrize(("make", "model"), [
    ("Canon", "Canon EOS 6D"),
    ("Canon", "Canon EOS 6D Mark III"),
    ("Canon", "Canon EOS 20Da"),
    ("OTHER", "Canon EOS 6D Mark II"),
    ("", "Canon EOS 6D Mark II"),
    ("FUJIFILM", "XF10-24mm"),
])
def test_unlisted_or_incomplete_identity_keeps_original_density(make, model):
    tags = _six_d_mark_ii_tags()
    tags.update({"Exif.Image.Make": make, "Exif.Image.Model": model})
    intrinsics = intrinsics_from_exif(tags, 6264, 4180)
    assert intrinsics.K[0, 0] == pytest.approx(14.0 * (6240000 / 935) / 25.4)


def test_known_camera_preserves_density_already_corrected_by_adobe():
    tags = _six_d_mark_ii_tags()
    tags.update({
        "Exif.Photo.FocalPlaneXResolution": "56907257/32768",
        "Exif.Photo.FocalPlaneYResolution": "56907257/32768",
        "Exif.Photo.FocalPlaneResolutionUnit": "3",
    })
    intrinsics = intrinsics_from_exif(tags, 6264, 4180)
    assert intrinsics.K[0, 0] == pytest.approx(14.0 * (56907257 / 32768) / 10)


def test_twenty_d_with_reasonable_original_density_is_not_overridden():
    tags = _six_d_mark_ii_tags()
    tags.update({
        "Exif.Image.Model": "Canon EOS 20D",
        "Exif.Photo.FocalPlaneXResolution": "3959.322034",
        "Exif.Photo.FocalPlaneYResolution": "3959.322034",
        "Exif.Photo.PixelXDimension": "3504",
        "Exif.Photo.PixelYDimension": "2336",
    })
    intrinsics = intrinsics_from_exif(tags, 3504, 2336)
    assert intrinsics.K[0, 0] == pytest.approx(14.0 * 3959.322034 / 25.4)


def test_known_camera_preserves_consistent_reduced_export_density():
    tags = _six_d_mark_ii_tags()
    tags.update({
        "Exif.Photo.PixelXDimension": "3120",
        "Exif.Photo.PixelYDimension": "2080",
        "Exif.Photo.FocalPlaneXResolution": str(3120 / 35.9),
        "Exif.Photo.FocalPlaneYResolution": str(3120 / 35.9),
        "Exif.Photo.FocalPlaneResolutionUnit": "4",
    })
    intrinsics = intrinsics_from_exif(tags, 3120, 2080)
    assert intrinsics.K[0, 0] == pytest.approx(14.0 * 3120 / 35.9)


@pytest.mark.parametrize(("width", "height"), [
    (6240, 4160), (4160, 6240), (3120, 2080),
])
def test_corrected_focal_sources_and_candidate_agree_for_portrait_and_crop(
        width, height):
    tags = _six_d_mark_ii_tags()
    tags["Exif.Photo.FocalLengthIn35mmFilm"] = "14/1"
    sources = exif_focal_sources(tags, width, height)
    candidate = build_camera_candidate(tags, (height, width, 3), "distortion")
    assert sources.pixel_density_focal_px == pytest.approx(14.0 * 6240 / 35.9)
    assert candidate.camera.intrinsics.K[0, 0] == pytest.approx(sources.pixel_density_focal_px)
    assert candidate.camera.intrinsics.K[1, 1] == pytest.approx(sources.pixel_density_focal_px)
    assert candidate.focal_metadata_ratio == sources.ratio
    assert sources.ratio == pytest.approx(6240 / max(width, height), rel=0.005)


def test_known_model_without_valid_focal_or_density_keeps_existing_fallbacks():
    tags = _six_d_mark_ii_tags()
    tags["Exif.Photo.FocalLength"] = "0/1"
    assert intrinsics_from_exif(tags, 6240, 4160) is None
    tags["Exif.Photo.FocalLengthIn35mmFilm"] = "14/1"
    assert intrinsics_from_exif(tags, 6240, 4160).K[0, 0] == pytest.approx(2426.666667)
    tags["Exif.Photo.FocalLength"] = "14/1"
    del tags["Exif.Photo.FocalPlaneXResolution"]
    assert intrinsics_from_exif(tags, 6240, 4160).K[0, 0] == pytest.approx(2426.666667)
