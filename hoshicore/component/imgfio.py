"""
imgfio contains functions and classes about image file i/o.

imgfio包含了与图像IO相关的函数和类。
"""
from __future__ import annotations

from typing import Optional, Union

import cv2
import numpy as np
import PIL.Image
import rawpy
import tifffile
from easydict import EasyDict
from loguru import logger

from .utils import (COMMON_SUFFIX, NOT_RECOM_SUFFIX, SAME_SUFFIX_MAPPING,
                    SUPPORT_COLOR_SPACE, get_scale_x, is_support_format,
                    time_cost_warpper)

def load_img(file_path: str) -> Optional[np.ndarray]:
    """ Using OpenCV API to load a single image from the given path.
    
    If necessary, the image will be converted to the given dtype.

    Args:
        file_path (str): /path/to/the/image.suffix

    Returns:
        np.ndarray: normally a `numpy.ndarray` object will be returned. 
        But the image fails to be loaded, an error will be logged, and `None` will be returned under such condition.
    """
    try:
        # suffix check and warning raising
        suffix = file_path.split(".")[-1].lower()
        assert is_support_format(
            file_path), f"Unsupported img suffix:{suffix}."
        if suffix in NOT_RECOM_SUFFIX:
            logger.warning("Got an Image with not recommended suffix. \
                We do not guarantee the stability of EXIF extraction and the output image quality."
                           )
        if (suffix in COMMON_SUFFIX) or (suffix in NOT_RECOM_SUFFIX):
            # TODO: not sure if uint32/float is available.
            img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint16),
                               cv2.IMREAD_UNCHANGED)
            if img is None:
                # some images can not be decoded using option dtype=np.uint16.
                # this is a temp fix.
                #logger.info(
                #    "Uint16 decoding failed. Fallback to uint8 loading...")
                img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8),
                                   cv2.IMREAD_UNCHANGED)
        else:
            # load images with rawpy
            with rawpy.imread(file_path) as raw:
                img = raw.postprocess(
                    output_bps=16,
                    output_color=rawpy.rawpy.ColorSpace(4))  # type: ignore
            # switch RGB to BGR
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    except Exception as e:
        logger.error(f"Failed to read {file_path} Because {e}!")
        return None


def get_color_profile(color_bstring):
    color_profile = color_bstring.decode("latin-1", errors="ignore")
    if not color_profile: return None
    for color_space in SUPPORT_COLOR_SPACE:
        if color_space in color_profile:
            return color_space
    return NotImplementedError(
        "Unsupported color space. For now only these color spaces are supported: %s"
        % SUPPORT_COLOR_SPACE)



def load_info(fname: str) -> EasyDict:
    """Load EXIF and icc_profile information of the given image file.

    Args:
        fname (str): /path/to/the/image.file

    Returns:
        Optional[EasyDict]: a Easydict that stores EXIF information.
        When exception occurs, an easyDict with no EXIF data and empty colorprofile will be returned instead.
    """
    info = EasyDict(exif=EasyDict(), colorprofile=b"")
    with open(fname, mode='rb') as f:
        try:
            import pyexiv2
            with pyexiv2.ImageData(f.read()) as image_data:
                # 基础信息
                exifdata = image_data.read_exif()
                colorprofile = image_data.read_icc()
                info = EasyDict(
                    exif=EasyDict(exifdata),
                    colorprofile=colorprofile,
                )
        except (ImportError, OSError) as e:
            logger.warning(
                "Failed to load pyexiv2. EXIF data and colorprofile can not be loaded from files."
            )
    return info


@time_cost_warpper
def save_img(filename: str,
             img: np.ndarray,
             png_compressing: int = 0,
             jpg_quality: int = 90,
             exif: Union[dict, EasyDict, None] = None,
             colorprofile: bytes = b""):
    """保存单个图像到指定路径下，并添加exif信息和色彩配置文件。
    
    该函数会将图像转换为字节流，随后使用pyexiv2将exif和icc_profile信息写入文件。
    如果pyexiv2不可用，则直接将图像写入文件。

    Args:
        filename (str): The tgt filename.
        img (np.ndarray): The image to be saved.
        png_compressing (int): PNG compressing arguments, ranges from 0 (no compressing) to 9. Defaults to 0.
        jpg_quality (int): JPG quality parameter, ranges from 0 to 100. Defaults to 90.
        exif (Union[dict, EasyDict, None]): exif info in dict or EasyDict format.
        colorprofile (bytes): icc_profile in bytes format. Defaults to b"".

    Raises:
        NameError: 要求输出不支持的文件格式时出错。
    """
    logger.info(f"Saving image to {filename} ...")
    suffix = filename.upper().split(".")[-1]

    # 将图像通过OpenCV进行编码
    if suffix == "PNG":
        ext = ".png"
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), png_compressing]
    elif suffix in ["JPG", "JPEG"]:
        # 导出 jpg 时，位深度强制校验为8
        assert img.dtype == np.uint8, "Invalid: JPEG only supports 8-bit image!"
        ext = ".jpg"
        params = [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality]
    elif suffix in ["TIF", "TIFF"]:
        # 使用 tiff 时，默认无损不压缩
        ext = ".tif"
        params = [int(cv2.IMWRITE_TIFF_COMPRESSION), 1]
    else:
        raise NameError(f"Unsupported suffix \"{suffix}\".")
    status, buf = cv2.imencode(ext, img, params)
    assert status, "imencode failed."
    with open(filename, mode='wb') as f:
        f.write(buf.tobytes())


def get_img_attrs(fname: str) -> dict:
    """
    在不加载完整图像的情况下，使用Pillow 与 tifffile 获取图像基本信息。
    
    获取的信息包含：
    * 后缀名
    * 图像尺寸
    * 位深度

    Args:
        fname (str): 文件名。

    Returns:
        dict: 图像基本信息
    """
    img_obj = PIL.Image.open(fname)
    suffix = fname.split(".")[-1].lower()
    if suffix in SAME_SUFFIX_MAPPING:
        suffix = SAME_SUFFIX_MAPPING[suffix]
    size = (getattr(img_obj, "width", None), getattr(img_obj, "height", None))
    bits = getattr(img_obj, "bits", None)
    if suffix in ["tif", "tiff"]:
        bits = tifffile.TiffFile(fname).pages[0].dtype.itemsize * 8
    return dict(fname=fname,
                suffix=suffix,
                size=size,
                size_str=f"{size[0]}x{size[1]}",
                bits=bits)


def analyze_attr(attr_list: list[dict], attr_name: str) -> dict:
    """分析输入符合给定属性的情况。

    Args:
        attr_list (list): _description_

    Returns:
        dict: _description_
    """
    attrs = [attr_dict[attr_name] for attr_dict in attr_list]
    sorted_attr_count = sorted([(attr, attrs.count(attr))
                                for attr in set(attrs)],
                               key=lambda x: x[-1],
                               reverse=True)
    other_attr = [x[0] for x in sorted_attr_count[1:]]
    if other_attr:
        other_fname_list = [
            attr_dict["fname"] for attr_dict in attr_list
            if attr_dict[attr_name] in other_attr
        ]
    else:
        other_fname_list = None
    assert len(sorted_attr_count) > 0
    return dict(attr_name=attr_name,
                mode_attr=sorted_attr_count[0][0],
                mode_num=sorted_attr_count[0][1],
                other_dist=sorted_attr_count[1:],
                other_fname_list=other_fname_list)
