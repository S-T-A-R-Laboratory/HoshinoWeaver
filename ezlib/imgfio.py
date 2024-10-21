"""
imgfio contains functions and classes about image file i/o.

imgfio包含了与图像IO相关的函数和类。
"""
from __future__ import annotations
import queue
import threading
from typing import Optional, Union, Sequence

import cv2
import numpy as np
import PIL.Image
import rawpy
from easydict import EasyDict
from loguru import logger

from .utils import (COMMON_SUFFIX, NOT_RECOM_SUFFIX, SAME_SUFFIX_MAPPING,
                    SUPPORT_COLOR_SPACE, get_scale_x, is_support_format,
                    time_cost_warpper)


class ImgSeriesLoader(object):
    """用于多线程读取图像序列的类。

    Args:
        object (_type_): _description_
    """

    def __init__(self,
                 fname_list: Sequence,
                 dtype: Optional[np.dtype] = None,
                 resize: Optional[list[int]] = None,
                 max_poolsize: int = 2,
                 **kwargs):
        """_summary_

        Args:
            fname_list (Sequence): _description_
            dtype (Optional[np.dtype]): 如果读出图像需要强制类型转换，在此项配置。
            resize (Optional[list[int]]): 如果读出图像需要尺寸变换，在此项配置。
            max_poolsize (int): 线程最大缓冲图像大小。 Defaults to 2.
        """
        self.fname_list = fname_list
        self.dtype = dtype
        self.resize = resize
        self.buffer = queue.Queue(maxsize=max_poolsize)
        self.stopped = True
        self.prog = 0
        self.tot_num = len(fname_list)
        self.thread = threading.Thread(target=self.loop, args=(), daemon=True)

    def start(self):
        self.stopped = False
        self.prog = 0
        while not self.buffer.empty():
            self.buffer.get()
        self.thread.start()

    def stop(self):
        self.stopped = True

    def pop(self) -> Optional[np.ndarray]:
        if self.stopped and self.buffer.empty():
            return None
        return self.buffer.get()

    def loop(self):
        try:
            for imgname in self.fname_list:
                if self.stopped:
                    break
                self.buffer.put(
                    load_img(imgname, dtype=self.dtype, resize=self.resize))
                self.prog += 1
        except Exception as e:
            logger.error(
                f"Fatal error:{e.__repr__()}. {self.__class__.__name__} will be terminated."
            )
        finally:
            logger.debug(
                f"{self.__class__.__name__} has successfully stopped.")
            self.stop()


def get_color_profile(color_bstring):
    color_profile = color_bstring.decode("latin-1", errors="ignore")
    if not color_profile: return None
    for color_space in SUPPORT_COLOR_SPACE:
        if color_space in color_profile:
            return color_space
    return NotImplementedError(
        "Unsupported color space. For now only these color spaces are supported: %s"
        % SUPPORT_COLOR_SPACE)


def load_img(fname: str,
             dtype: Union[type, None, np.dtype] = None,
             resize=None) -> Optional[np.ndarray]:
    """ Using OpenCV API to load a single image from the given path.
    
    If necessary, the image will be converted to the given dtype.

    Args:
        fname (str): /path/to/the/image.suffix

    Returns:
        np.ndarray: normally a `numpy.ndarray` object will be returned. 
        But the image fails to be loaded, an error will be logged, and `None` will be returned under such condition.
    """
    try:
        # suffix check and warning raising
        suffix = fname.split(".")[-1].lower()
        assert is_support_format(fname), f"Unsupported img suffix:{suffix}."
        if suffix in NOT_RECOM_SUFFIX:
            logger.warning("Got an Image with not recommended suffix. \
                We do not guarantee the stability of EXIF extraction and the output image quality."
                           )
        if (suffix in COMMON_SUFFIX) or (suffix in NOT_RECOM_SUFFIX):
            # TODO: not sure if uint32/float is available.
            img = cv2.imdecode(np.fromfile(fname, dtype=np.uint16),
                               cv2.IMREAD_UNCHANGED)
            if img is None:
                # some images can not be decoded using option dtype=np.uint16.
                # this is a temp fix.
                logger.info(
                    "Uint16 decoding failed. Fallback to uint8 loading...")
                img = cv2.imdecode(np.fromfile(fname, dtype=np.uint8),
                                   cv2.IMREAD_UNCHANGED)
        else:
            # load images with rawpy
            with rawpy.imread(fname) as raw:
                img = raw.postprocess(
                    output_bps=16,
                    output_color=rawpy.rawpy.ColorSpace(4))  # type: ignore
        if dtype:
            # TODO: 目前属于临时解决方案。以后需要更通用的写法。
            # 这部分有一些问题。例如，仅有星轨模式会使用整数权重（并在此处上升dtype范围）
            if dtype == np.dtype("uint8") and img.dtype == np.dtype("uint16"):
                img = np.array(img // get_scale_x(1), dtype=np.uint8)
            else:
                img = np.array(img, dtype=dtype)
        if resize is not None:
            # resize is in shape order (i.e, [h,w])
            # so when using OpenCV resize, it should be reversed.
            assert len(
                resize
            ) == 2, f"invalid resize arg! expect length=2, got {resize}."
            [h, w] = resize
            img = cv2.resize(img, [w, h])
        logger.debug(
            f"Successfully read img with shape={img.shape}; dtype={img.dtype}."
        )
        return img
    except Exception as e:
        logger.error(f"Failed to read {fname} Because {e}!")
        return None


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
    
    主要工作流程如下：
    使用openCV，将单个图像转换为字节流，岁后使用不包含exif和icc_profile信息。

    Args:
        filename (str): The tgt filename.
        img (np.ndarray): The image to be saved.
        png_compressing (int): PNG compressing arguments, ranges from 0 (no compressing) to 9. Defaults to 0.
        jpg_quality (int): JPG quality parameter, ranges from 0 to 100. Defaults to 90.

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
    try:
        import pyexiv2
        with pyexiv2.ImageData(buf.tobytes()) as image_data:
            if colorprofile is not None and colorprofile != b"":
                image_data.modify_icc(colorprofile)
            if exif is not None and len(exif) > 0:
                image_data.modify_exif(exif)
            # 写入文件
            with open(filename, mode='wb') as f:
                f.write(image_data.get_bytes())
    except (ImportError, OSError) as e:
        logger.warning(
            "Failed to load pyexiv2. EXIF data and colorprofile can not be written to files."
        )
        # 降级写入文件
        with open(filename, mode='wb') as f:
            f.write(buf.tobytes())


def get_img_attrs_by_pil(fname: str) -> dict:
    """使用Pillow接口获取图像基本信息。

    Pillow支持的EXIF信息相对有限。

    Args:
        fname (str): 文件名。

    Returns:
        dict: _description_
    """
    img_obj = PIL.Image.open(fname)
    suffix = fname.split(".")[-1].lower()
    if suffix in SAME_SUFFIX_MAPPING:
        suffix = SAME_SUFFIX_MAPPING[suffix]
    size = (getattr(img_obj, "width", None), getattr(img_obj, "height", None))
    return dict(fname=fname,
                suffix=suffix,
                size=size,
                size_str=f"{size[0]}x{size[1]}",
                bits=getattr(img_obj, "bits", None))


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
