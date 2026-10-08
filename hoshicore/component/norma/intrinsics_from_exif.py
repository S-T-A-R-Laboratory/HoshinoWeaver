"""从 EXIF 标签字典推算相机内参 (Intrinsics)。

"""
import dataclasses
import math
from typing import Optional

from loguru import logger

from .types import Intrinsics

# 35mm 参考画幅的长边（mm）：等效焦距按"长边视场"归一化。
_REFERENCE_LONG_SIDE_MM = 36.0

# 两条 EXIF 路径的像素焦距相对偏差超过该比例即视为元数据不一致。
FOCAL_METADATA_TOLERANCE = 0.05

# 像素密度路径换算出的 fx/fy 相对偏差超过该比例即视为标签异常。
_PIXEL_PITCH_TOLERANCE = 0.02

_RESOLUTION_UNIT_FACTORS = {
    "2": 25.4,  # inch → mm
    "3": 10.0,  # cm → mm
    "4": 1.0,  # mm
    "5": 0.001,  # μm → mm
}


@dataclasses.dataclass(frozen=True)
class ExifFocalSources:
    """两条独立 EXIF 焦距推导得到的像素焦距（各自可能缺失）。"""

    pixel_density_focal_px: Optional[float]
    focal_equiv_focal_px: Optional[float]

    @property
    def ratio(self) -> Optional[float]:
        """像素密度 / 35mm 等效焦距之比；两条都可用时才非 None。

        `1.0` 表示两条路径一致。显著偏离 1 表示图像在拍摄后被裁切（等效焦距
        标签描述整幅画幅）或被重采样（像素密度标签描述原始采样）。
        """
        if (self.pixel_density_focal_px is None
                or self.focal_equiv_focal_px is None):
            return None
        return self.pixel_density_focal_px / self.focal_equiv_focal_px


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


def _positive_finite(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        return None
    return value


def _intrinsics_with_pixel_focal(focal_px: float, img_width: int,
                                 img_height: int,
                                 focal_label_mm: float) -> Intrinsics:
    """构造 K 各向同性的 Intrinsics：sensor 两轴与图像像素边严格等比例。"""
    return Intrinsics(
        focal_length_mm=focal_label_mm,
        sensor_width_mm=focal_label_mm * img_width / focal_px,
        sensor_height_mm=focal_label_mm * img_height / focal_px,
        image_width_px=img_width,
        image_height_px=img_height,
    )


def _pixel_density_intrinsics(exif_tags: dict[str, str], img_width: int,
                              img_height: int
                              ) -> tuple[Optional[Intrinsics], Optional[float]]:
    """`FocalLength` + `FocalPlaneX/YResolution` → (Intrinsics, 像素焦距)。

    X 轴像素间距（focal_mm · FocalPlaneXResolution / unit）即像素焦距；X/Y
    间距不一致时 K 会各向异性——可能是真实的非方形像素，也可能是标签异常，
    由调用方 `intrinsics_from_exif` 判定并告警。
    """
    focal_mm = _positive_finite(
        _parse_rational(exif_tags.get("Exif.Photo.FocalLength")))
    if focal_mm is None:
        return None, None

    density_x = _positive_finite(
        _parse_rational(exif_tags.get("Exif.Photo.FocalPlaneXResolution")))
    density_y = _positive_finite(
        _parse_rational(exif_tags.get("Exif.Photo.FocalPlaneYResolution")))
    if density_x is None or density_y is None:
        logger.debug(
            "intrinsics_from_exif: FocalPlaneResolution missing or invalid")
        return None, None

    unit_str = exif_tags.get("Exif.Photo.FocalPlaneResolutionUnit", "2")
    unit_mm = _RESOLUTION_UNIT_FACTORS.get(str(unit_str).strip())
    if unit_mm is None:
        logger.debug(f"intrinsics_from_exif: unknown ResolutionUnit={unit_str}")
        return None, None

    intrinsics = Intrinsics(
        focal_length_mm=focal_mm,
        sensor_width_mm=img_width * unit_mm / density_x,
        sensor_height_mm=img_height * unit_mm / density_y,
        image_width_px=img_width,
        image_height_px=img_height,
    )
    return intrinsics, focal_mm * density_x / unit_mm


def _focal_equiv_intrinsics(exif_tags: dict[str, str], img_width: int,
                            img_height: int
                            ) -> tuple[Optional[Intrinsics], Optional[float]]:
    """`FocalLengthIn35mmFilm` → (Intrinsics, 像素焦距)。"""
    focal_equiv_mm = _positive_finite(
        _parse_rational(exif_tags.get("Exif.Photo.FocalLengthIn35mmFilm")))
    if focal_equiv_mm is None:
        return None, None
    focal_px = (focal_equiv_mm * max(img_width, img_height)
                / _REFERENCE_LONG_SIDE_MM)
    intrinsics = _intrinsics_with_pixel_focal(focal_px, img_width, img_height,
                                              focal_equiv_mm)
    return intrinsics, focal_px


def exif_focal_sources(exif_tags: dict[str, str], img_width: int,
                       img_height: int) -> ExifFocalSources:
    """返回两条 EXIF 焦距路径的像素焦距，用于交叉校验与诊断。"""
    _, density_px = _pixel_density_intrinsics(exif_tags, img_width, img_height)
    _, equiv_px = _focal_equiv_intrinsics(exif_tags, img_width, img_height)
    return ExifFocalSources(pixel_density_focal_px=density_px,
                            focal_equiv_focal_px=equiv_px)


def intrinsics_from_fisheye_estimate(img_width: int,
                                     img_height: int) -> Intrinsics:
    """为鱼眼镜头构建 180° FOV 估算内参（无 EXIF 且无手动焦距时的兜底）。

    假设 FOV = 180°，短边对应 θ_max = π/2：
      r_edge = fx · (π/2) = min(w, h) / 2
      → fx = fy = min(w, h) / π

    两轴必须同值：存储的 focal_length_mm 仍按长边 = 36mm 的同一约定折算，以便与
    `intrinsics_from_focal_equiv` 的语义一致。
    """
    focal_px = min(img_width, img_height) / math.pi
    focal_label_mm = (focal_px * _REFERENCE_LONG_SIDE_MM
                      / max(img_width, img_height))
    return _intrinsics_with_pixel_focal(focal_px, img_width, img_height,
                                        focal_label_mm)


def intrinsics_from_focal_equiv(focal_equiv_mm: float, img_width: int,
                                img_height: int) -> Intrinsics:
    """从 35mm 等效焦距构建 Intrinsics（长边 = 36mm 归一化）。

    fx = fy = focal_equiv · max(w, h) / 36

    等价焦距是按视场定义的，因此归一化必须落在图像的**长边**上。sensor 两轴按图像像素边
    等比例分配，因此本函数对朝向与画幅比都是各向同性的。

    注意：本函数假设图像覆盖整个等价画幅。后期裁切过的图像会被系统性低估（偏小裁切比例倍）。

    Args:
        focal_equiv_mm: 35mm 等效焦距（mm）。等于 真实焦距 × 裁切系数。
        img_width: 图像宽度（像素）。
        img_height: 图像高度（像素）。

    Returns:
        Intrinsics。参数非法时抛出 ValueError。
    """
    if focal_equiv_mm <= 0 or img_width <= 0 or img_height <= 0:
        logger.error(
            f"intrinsics_from_focal_equiv: invalid args "
            f"focal_equiv={focal_equiv_mm}, size={img_width}×{img_height}")
        raise ValueError(
            "Invalid focal_equiv_mm or image size: expected positive values, "
            f"got {focal_equiv_mm}, {img_width}×{img_height}")
    focal_px = (focal_equiv_mm * max(img_width, img_height)
                / _REFERENCE_LONG_SIDE_MM)
    return _intrinsics_with_pixel_focal(focal_px, img_width, img_height,
                                        focal_equiv_mm)


def intrinsics_from_exif(exif_tags: dict[str, str], img_width: int,
                         img_height: int) -> Optional[Intrinsics]:
    """尝试从 EXIF 标签字典构建 Intrinsics。缺少可用标签时返回 None。

    推算路径：
        1. FocalLength + FocalPlaneX/YResolution + ResolutionUnit
           → 像素密度路径，与裁切无关
        2. FocalLengthIn35mmFilm
           → 长边 = 36mm 归一化的等效焦距路径，各向同性。

    两条路径同时可用时以像素密度路径为准；两者偏差超过
    `FOCAL_METADATA_TOLERANCE` 时记录告警（图像疑似在拍摄后被裁切或重采样）。

    Args:
        exif_tags: EXIF 标签原始字典 (key → value 均为字符串)。
        img_width: 图像宽度（像素）。
        img_height: 图像高度（像素）。

    Returns:
        Intrinsics 或 None。
    """
    density_intrinsics, density_px = _pixel_density_intrinsics(
        exif_tags, img_width, img_height)
    equiv_intrinsics, equiv_px = _focal_equiv_intrinsics(
        exif_tags, img_width, img_height)

    if density_px is not None and equiv_px is not None:
        ratio = density_px / equiv_px
        if abs(ratio - 1.0) > FOCAL_METADATA_TOLERANCE:
            logger.warning(
                "intrinsics_from_exif: EXIF focal sources disagree by "
                f"{ratio:.3f}x (pixel density {density_px:.1f}px vs "
                f"35mm-equivalent {equiv_px:.1f}px); the image was probably "
                "cropped or resampled after capture; using the pixel-density "
                "derivation")

    if density_intrinsics is not None:
        anisotropy = (density_intrinsics.K[0, 0] / density_intrinsics.K[1, 1])
        if abs(anisotropy - 1.0) > _PIXEL_PITCH_TOLERANCE:
            logger.warning(
                "intrinsics_from_exif: FocalPlane X/Y pixel density imply "
                f"fx/fy={anisotropy:.4f}; the camera model is anisotropic "
                "(non-square pixels or inconsistent EXIF)")
        logger.debug(
            "intrinsics_from_exif: using FocalLength+FocalPlaneResolution "
            f"(focal_px={density_px:.1f})")
        return density_intrinsics

    if equiv_intrinsics is not None:
        logger.debug(
            "intrinsics_from_exif: using FocalLengthIn35mmFilm "
            f"(focal_px={equiv_px:.1f})")
        return equiv_intrinsics

    logger.debug("intrinsics_from_exif: no usable focal EXIF tags")
    return None
