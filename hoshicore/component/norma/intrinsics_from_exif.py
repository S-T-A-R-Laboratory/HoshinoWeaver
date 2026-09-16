"""从 EXIF 标签字典推算相机内参 (Intrinsics)。

"""
from typing import Optional

from loguru import logger

from .types import Intrinsics

# 35mm 全画幅参考传感器尺寸（mm）
_FF_W_MM = 36.0
_FF_H_MM = 24.0

_RESOLUTION_UNIT_FACTORS = {
    "2": 25.4,  # inch → mm
    "3": 10.0,  # cm → mm
    "4": 1.0,  # mm
    "5": 0.001,  # μm → mm
}


def lens_type_from_exif(exif_tags: Optional[dict[str, str]]) -> Optional[str]:
    """Infer a supported projection family from descriptive EXIF fields.

    EXIF has no standard projection-model tag.  Therefore this deliberately
    only returns ``"fisheye"`` for explicit fish-eye wording and otherwise
    returns ``None`` so callers can apply their configured/default policy.
    """
    if not exif_tags:
        return None
    fields = (
        exif_tags.get("Exif.Photo.LensModel"),
        exif_tags.get("Exif.Image.Model"),
        exif_tags.get("Exif.Image.Make"),
    )
    text = " ".join(str(value) for value in fields if value).lower()
    if any(token in text.lower() for token in ("fisheye", "fish-eye", "fish eye", "鱼眼")):
        return "fisheye"
    return None


def _parse_rational(value: Optional[str]) -> Optional[float]:
    """解析 EXIF 有理数字符串 (如 "50/1", "4.5") 为 float。"""
    if value is None:
        return None
    value = value.strip()
    if "/" in value:
        parts = value.split("/")
        try:
            return float(parts[0]) / float(parts[1])
        except (ValueError, ZeroDivisionError):
            return None
    try:
        return float(value)
    except ValueError:
        return None


def intrinsics_from_fisheye_estimate(img_width: int,
                                     img_height: int) -> Intrinsics:
    """为鱼眼镜头构建 180° FOV 估算内参（无 EXIF 且无手动焦距时的兜底）。

    假设 FOV = 180°，短边对应 θ_max = π/2：
      r_edge = fx · (π/2) = min(w, h) / 2
      → fx = min(w, h) / π

    折算为 35mm 等效焦距后复用 intrinsics_from_focal_equiv。
    """
    # fx_px = min(w,h)/π → equiv_mm = fx_px × 36 / w
    equiv_mm = min(img_width,
                   img_height) / 3.14159265358979 * _FF_W_MM / img_width
    return Intrinsics(
        focal_length_mm=equiv_mm,
        sensor_width_mm=_FF_W_MM,
        sensor_height_mm=_FF_H_MM,
        image_width_px=img_width,
        image_height_px=img_height,
    )


def intrinsics_from_focal_equiv(focal_equiv_mm: float, img_width: int,
                                img_height: int):
    """从 35mm 等效焦距构建 Intrinsics。

    fx = focal_equiv · img_width  / 36
    fy = focal_equiv · img_height / 24

    这等价于假设传感器为 36mm×24mm 全画幅，以等效焦距为实际焦距。
    对非 3:2 传感器 fy 有微小近似误差，optimizer 的 focal_scale 参数可吸收。

    Args:
        focal_equiv_mm: 35mm 等效焦距（mm）。等于 真实焦距 × 裁切系数。
        img_width: 图像宽度（像素）。
        img_height: 图像高度（像素）。

    Returns:
        Intrinsics，或在参数非法时返回 None。
    """
    if focal_equiv_mm <= 0 or img_width <= 0 or img_height <= 0:
        logger.error(
            f"intrinsics_from_focal_equiv: invalid args "
            f"focal_equiv={focal_equiv_mm}, size={img_width}×{img_height}")
        raise ValueError(
            "Invalid focal_equiv_mm or image size: expected positive values, ",
            f"got {focal_equiv_mm}, {img_width}×{img_height}")
    return Intrinsics(
        focal_length_mm=focal_equiv_mm,
        sensor_width_mm=_FF_W_MM,
        sensor_height_mm=_FF_H_MM,
        image_width_px=img_width,
        image_height_px=img_height,
    )


def intrinsics_from_exif(exif_tags: dict[str, str], img_width: int,
                         img_height: int) -> Optional[Intrinsics]:
    """尝试从 EXIF 标签字典构建 Intrinsics。缺少必要标签时返回 None。

    推算路径：
        FocalLength (mm) + FocalPlaneX/YResolution + ResolutionUnit
        → sensor_width_mm = img_width / (FocalPlaneXResolution * unit_to_mm)
        → sensor_height_mm = img_height / (FocalPlaneYResolution * unit_to_mm)

    Args:
        exif_tags: EXIF 标签原始字典 (key → value 均为字符串)。
        img_width: 图像宽度（像素）。
        img_height: 图像高度（像素）。

    Returns:
        Intrinsics 或 None。
    """
    focal_equiv_mm = _parse_rational(
        exif_tags.get("Exif.Photo.FocalLengthIn35mmFilm"))
    if focal_equiv_mm is not None and focal_equiv_mm > 0:
        logger.debug(
            "intrinsics_from_exif: using FocalLengthIn35mmFilm="
            f"{focal_equiv_mm:.1f}mm")
        return intrinsics_from_focal_equiv(
            focal_equiv_mm, img_width, img_height)

    focal_mm = _parse_rational(exif_tags.get("Exif.Photo.FocalLength"))
    if focal_mm is None or focal_mm <= 0:
        logger.debug("intrinsics_from_exif: FocalLength missing or invalid")
        return None

    fp_x = _parse_rational(exif_tags.get("Exif.Photo.FocalPlaneXResolution"))
    fp_y = _parse_rational(exif_tags.get("Exif.Photo.FocalPlaneYResolution"))
    if fp_x is None or fp_y is None or fp_x <= 0 or fp_y <= 0:
        logger.debug(
            "intrinsics_from_exif: FocalPlaneResolution missing or invalid")
        return None

    unit_str = exif_tags.get("Exif.Photo.FocalPlaneResolutionUnit", "2")
    unit_factor = _RESOLUTION_UNIT_FACTORS.get(unit_str.strip())
    if unit_factor is None:
        logger.debug(
            f"intrinsics_from_exif: unknown ResolutionUnit={unit_str}")
        return None

    sensor_width_mm = img_width / fp_x * unit_factor
    sensor_height_mm = img_height / fp_y * unit_factor

    if sensor_width_mm <= 0 or sensor_height_mm <= 0:
        return None

    return Intrinsics(
        focal_length_mm=focal_mm,
        sensor_width_mm=sensor_width_mm,
        sensor_height_mm=sensor_height_mm,
        image_width_px=img_width,
        image_height_px=img_height,
    )
