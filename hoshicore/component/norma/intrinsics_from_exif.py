"""从 EXIF 标签字典推算相机内参 (Intrinsics)。

本模块不依赖 ExifData 类型——仅接收 dict[str, str]，保持 norma 包的独立性。
"""
from typing import Optional

from loguru import logger

from .types import Intrinsics


_RESOLUTION_UNIT_FACTORS = {
    "2": 25.4,    # inch → mm
    "3": 10.0,    # cm → mm
    "4": 1.0,     # mm
    "5": 0.001,   # μm → mm
}


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


def intrinsics_from_exif(
    exif_tags: dict[str, str], img_width: int, img_height: int
) -> Optional[Intrinsics]:
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
    focal_mm = _parse_rational(exif_tags.get("Exif.Photo.FocalLength"))
    if focal_mm is None or focal_mm <= 0:
        logger.debug("intrinsics_from_exif: FocalLength missing or invalid")
        return None

    fp_x = _parse_rational(exif_tags.get("Exif.Photo.FocalPlaneXResolution"))
    fp_y = _parse_rational(exif_tags.get("Exif.Photo.FocalPlaneYResolution"))
    if fp_x is None or fp_y is None or fp_x <= 0 or fp_y <= 0:
        logger.debug("intrinsics_from_exif: FocalPlaneResolution missing or invalid")
        return None

    unit_str = exif_tags.get("Exif.Photo.FocalPlaneResolutionUnit", "2")
    unit_factor = _RESOLUTION_UNIT_FACTORS.get(unit_str.strip())
    if unit_factor is None:
        logger.debug(f"intrinsics_from_exif: unknown ResolutionUnit={unit_str}")
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
