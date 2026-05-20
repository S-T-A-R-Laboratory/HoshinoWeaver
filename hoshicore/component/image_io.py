"""
image_io contains functions and classes about image file i/o.

image_io包含了与图像IO相关的函数和类。
"""
from __future__ import annotations

from typing import Optional, Union

import cv2
import numpy as np
import PIL.Image
import rawpy
import tifffile
from loguru import logger
from turbojpeg import TurboJPEG, TJPF_BGR, TJPF_GRAY

from .exif import ExifData, encode_exif_data
from .utils import (COMMON_SUFFIX, NOT_RECOM_SUFFIX, RAW_SUFFIX,
                    SAME_SUFFIX_MAPPING, is_support_format, time_cost_warpper)

_tj = TurboJPEG()


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
        if suffix in ("jpg", "jpeg"):
            with open(file_path, 'rb') as f:
                buf = f.read()
            img = _tj.decode(buf, pixel_format=TJPF_BGR)
            if img is None:
                # TODO: 暂未适配灰度
                img = _tj.decode(buf, pixel_format=TJPF_GRAY)
        elif suffix in ("tif", "tiff"):
            img = tifffile.imread(file_path)
            if img.ndim == 3:
                if img.shape[2] == 3:
                    img = img[:, :, ::-1].copy()
                elif img.shape[2] == 4:
                    # 暂时丢弃A维度
                    img = img[:, :, [2, 1, 0]].copy()
        elif (suffix in COMMON_SUFFIX) or (suffix in NOT_RECOM_SUFFIX):
            img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint16),
                               cv2.IMREAD_UNCHANGED)
            if img is None:
                # TODO: 需要确认是否还有其他数据可能fallback到uint8。
                logger.warning(
                    "Uint16 decoding failed. Fallback to uint8 loading...")
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


@time_cost_warpper
def save_img(filename: str,
             img: np.ndarray,
             png_compressing: int = 0,
             jpg_quality: int = 90,
             exif: Optional[ExifData] = None):
    """保存单个图像到指定路径下，并添加exif信息和色彩配置文件。
    
    该函数会将图像转换为字节流，随后使用pyexiv2将exif和icc_profile信息写入文件。
    如果pyexiv2不可用，则直接将图像写入文件。

    Args:
        filename (str): The tgt filename.
        img (np.ndarray): The image to be saved.
        png_compressing (int): PNG compressing arguments, ranges from 0 (no compressing) to 9. Defaults to 0.
        jpg_quality (int): JPG quality parameter, ranges from 0 to 100. Defaults to 90.
        exif (Union[ExifData, None]): optional exif info in ExifData format.
        colorprofile (bytes): icc_profile in bytes format. Defaults to b"".

    Raises:
        NameError: 要求输出不支持的文件格式时出错。
    """
    logger.info(f"Saving image to {filename} ...")
    suffix = filename.upper().split(".")[-1]

    
    if suffix in ["JPG", "JPEG"]:
        # JPEG 走 turbojpeg
        assert img.dtype == np.uint8, "Invalid: JPEG only supports 8-bit image!"
        image_bytes = _tj.encode(img, quality=jpg_quality,
                                 pixel_format=TJPF_BGR)
        if exif is not None:
            image_bytes = encode_exif_data(
                np.frombuffer(image_bytes, dtype=np.uint8), exif)
    else:
        # 将图像通过OpenCV进行编码
        if suffix == "PNG":
            ext = ".png"
            params = [int(cv2.IMWRITE_PNG_COMPRESSION), png_compressing]
        elif suffix in ["TIF", "TIFF"]:
            ext = ".tif"
            params = [int(cv2.IMWRITE_TIFF_COMPRESSION), 1]
        else:
            raise NameError(f"Unsupported suffix \"{suffix}\".")
        status, buf = cv2.imencode(ext, img, params)
        assert status, "imencode failed."    
        if exif is not None:
            image_bytes = encode_exif_data(buf, exif)
        else:
            image_bytes = buf.tobytes()

    with open(filename, mode='wb') as f:
        f.write(image_bytes)


def peek_shape(file_path: str) -> tuple[tuple[int, ...], int]:
    """只读文件头，返回 (shape, dtype_bytes)。不做完整解码。

    比 get_img_attrs 更精确（含 channels）且覆盖 RAW 格式。

    Returns:
        (shape, dtype_bytes): shape 为 (H, W) 或 (H, W, C)，
        dtype_bytes 为每像素每通道字节数。
    """
    import os
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"peek_shape: file not found: {file_path}")

    suffix = file_path.rsplit(".", 1)[-1].lower()
    if suffix in SAME_SUFFIX_MAPPING:
        suffix = SAME_SUFFIX_MAPPING[suffix]

    if suffix in ("tif", "tiff"):
        with tifffile.TiffFile(file_path) as tf:
            page = tf.pages[0]
            return tuple(page.shape), page.dtype.itemsize

    if suffix in RAW_SUFFIX:
        with rawpy.imread(file_path) as raw:
            h = raw.sizes.height
            w = raw.sizes.width
            return (h, w, 3), 2

    if suffix not in COMMON_SUFFIX and suffix not in NOT_RECOM_SUFFIX:
        raise ValueError(f"peek_shape: unsupported format: {suffix}")

    # jpg, png, bmp, etc.
    img = PIL.Image.open(file_path)
    w, h = img.size
    bands = len(img.getbands())
    mode = img.mode
    img.close()
    dtype_bytes = 2 if mode in ("I;16", "I;16B") else 1
    if bands == 1:
        return (h, w), dtype_bytes
    return (h, w, bands), dtype_bytes


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



@time_cost_warpper
def scan_all_exif(fname_list: list[str]) -> list:
    """
    快速检查输入，并给出一系列可能会导致叠加任务无法正常进行的风险提示。
    
    目前有后缀名检查suffix，图像尺寸检查size_str，位数检查bits。位数检查有一定局限性，tiff不支持（pillow的底层问题，对tiff支持弱）

    由于部分数值不一定能够读取到，不推荐作为强制卡控。

    返回一个dict的list。每个dict包含5个字段：
    
    1. 检查的属性名 attr_name (str)
    2. 该属性最主要的模式 mode_attr (str)
    3. 主要模式的数量 mode_num (int)
    4. 其他模式的及数量分布 other_dist (Optional[list[tuple[str,int]]])
    5. 非主要模式的文件名列表 other_fname_list (Optional[list])

    Args:
        fname_list (list[str]): 文件名列表

    Returns:
        list[dict]: 返回风险提示列表。
    """
    attr_list = list(map(get_img_attrs, fname_list))
    # 后缀名检查
    suffix_dict = analyze_attr(attr_list, "suffix")
    # 尺寸检查
    size_dict = analyze_attr(attr_list, "size")
    # 位数检查
    bits_dict = analyze_attr(attr_list, "bits")
    return [suffix_dict, size_dict, bits_dict]
