"""
EXIF 元数据读写
"""
from dataclasses import dataclass

import pyexiv2
from loguru import logger
from numpy.typing import NDArray
from .utils import SUPPORT_COLOR_SPACE, SOFTWARE_NAME, VERSION


# 常用属性的简称和全称映射，方便编辑属性
class CommonExifTags:
    ImageWidth = "Exif.Image.ImageWidth"
    ImageLength = "Exif.Image.ImageLength"
    BitsPerSample = "Exif.Image.BitsPerSample"
    Make = "Exif.Image.Make"
    Model = "Exif.Image.Model"
    XResolution = "Exif.Image.XResolution"
    YResolution = "Exif.Image.YResolution"
    Software = "Exif.Image.Software"
    InterColorProfile = "Exif.Image.InterColorProfile"
    ExposureTime = "Exif.Photo.ExposureTime"
    FNumber = "Exif.Photo.FNumber"
    ISOSpeedRatings = "Exif.Photo.ISOSpeedRatings"
    ExifVersion = "Exif.Photo.ExifVersion"
    DateTimeOriginal = "Exif.Photo.DateTimeOriginal"
    DateTimeDigitized = "Exif.Photo.DateTimeDigitized"
    OffsetTime = "Exif.Photo.OffsetTime"
    FocalLength = "Exif.Photo.FocalLength"
    FocalPlaneXResolution = "Exif.Photo.FocalPlaneXResolution"
    FocalPlaneYResolution = "Exif.Photo.FocalPlaneYResolution"
    FocalPlaneResolutionUnit = "Exif.Photo.FocalPlaneResolutionUnit"
    LensSpecification = "Exif.Photo.LensSpecification"
    LensModel = "Exif.Photo.LensModel"


@dataclass(slots=True)
class ExifData:
    exif: dict[str, str]
    colorprofile: bytes

    def __repr__(self) -> str:
        ret = "ExifData(\n"
        colorspace_str = get_color_profile(self.colorprofile)
        for (k, v) in self.exif.items():
            # 长度大于20的exif简化显示
            if len(v) > 20:
                ret += f"  {k}: {v[:10]}...{v[-10:]}\n"
            else:
                ret += f"  {k}: {v}\n"
        ret += f"  colorprofile: {colorspace_str}\n)"
        return ret

    def set_exif(self, key: str, value: str):
        """设置EXIF属性。
        
        Args:
            key (str): EXIF属性的名称。
            value (str): 要设置的属性值。
        """
        if not key in self.exif:
            logger.warning(f"Trying to set a new EXIF tag {key} that does "
                           "not exist in the original image.")
        self.exif[key] = value
    
    def get_exif(self, key: str):
        """获取EXIF属性值。
        
        Args:
            key (str): EXIF属性的名称。
        
        Returns:
            str: 对应属性的值，如果属性不存在则返回None。
        """
        return self.exif.get(key, None)


def read_exif_data(fname: str):
    """Load EXIF and icc_profile information of the given image file.

    Args:
        fname (str): /path/to/the/image.file

    Returns:
        Optional[ExifData]: ExifData that stores EXIF information.
        When exception occurs, None will be returned instead.
    """
    try:
        with open(fname, mode='rb') as f:
            with pyexiv2.ImageData(f.read()) as image_data:
                # 基础信息
                exifdata = image_data.read_exif()
                colorprofile = image_data.read_icc()
                return ExifData(exif=exifdata, colorprofile=colorprofile)
    except (ImportError, OSError) as e:
        logger.error(
            f"Failed to load EXIF data and colorprofile because: {e}.")
        return None


def encode_exif_data(buf: NDArray, exifdata: ExifData):
    """Write EXIF and icc_profile information to the given image file.

    Args:
        fname (str): /path/to/the/image.file
        exifdata (dict): a dict that stores EXIF information.
        colorprofile (bytes): a byte string that stores the color profile.
    """
    exifdata.set_exif(CommonExifTags.Software, f"{SOFTWARE_NAME} V{VERSION}")
    colorprofile = exifdata.colorprofile
    exif = exifdata.exif
    with pyexiv2.ImageData(buf.tobytes()) as image_data:
        if colorprofile is not None and colorprofile != b"":
            image_data.modify_icc(colorprofile)
        if exif is not None and len(exif) > 0:
            image_data.modify_exif(exif)
        return image_data.get_bytes()


def get_color_profile(color_bstring):
    color_profile = color_bstring.decode("latin-1", errors="ignore")
    if not color_profile: return None
    for color_space in SUPPORT_COLOR_SPACE:
        if color_space in color_profile:
            return color_space
    return NotImplementedError(
        "Unsupported color space. For now only these color spaces are supported: %s"
        % SUPPORT_COLOR_SPACE)
