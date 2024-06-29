"""
imgfio contains functions and classes about image file i/o.

imgfio包含了与图像IO相关的函数和类。
"""

import threading
import cv2
import numpy as np
import pyexiv2
import rawpy
import queue
from easydict import EasyDict
from .utils import COMMON_SUFFIX, NOT_RECOM_SUFFIX, SUPPORT_COLOR_SPACE, is_support_format, time_cost_warpper
from typing import Optional
from loguru import logger

BITS2DTYPE = {
    '8': np.dtype('uint8'),
    '16': np.dtype('uint16'),
    '32': np.dtype('uint32')
}


class ImgSeriesLoader(object):
    """用于多线程读取图像序列的类。

    Args:
        object (_type_): _description_
    """

    def __init__(self,
                 fname_list,
                 dtype=None,
                 resize=None,
                 max_poolsize=8,
                 **kwargs):
        """_summary_

        Args:
            fname_list (_type_): _description_
            dtype (_type_): 如果读出图像需要强制类型转换，在此项配置。
            resize (_type_): 如果读出图像需要尺寸变换，在此项配置。
            max_poolsize (int, optional): _description_. Defaults to 8.
        """
        self.fname_list = fname_list
        self.dtype = dtype
        self.resize = resize
        self.buffer = queue.Queue(maxsize=max_poolsize)
        self.stopped = True
        self.prog = 0
        self.tot_num = len(fname_list)
        self.thread = threading.Thread(target=self.loop, args=())

    def start(self):
        self.stopped = False
        self.prog = 0
        while not self.buffer.empty():
            self.buffer.get()
        self.thread.start()

    def stop(self):
        self.stopped = True

    def pop(self) -> Optional[np.ndarray]:
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
            raise e
        finally:
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
             dtype: Optional[type] = None,
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
            img = cv2.imdecode(np.fromfile(fname, dtype=np.uint16),
                               cv2.IMREAD_UNCHANGED)
        else:
            # load images with rawpy
            with rawpy.imread(fname) as raw:
                img = raw.postprocess(
                    output_bps=16,
                    output_color=rawpy.rawpy.ColorSpace(4))  # type: ignore
        if dtype:
            img = np.array(img, dtype=dtype)
        if resize:
            # TODO: 添加插值相关
            img = cv2.resize(img, resize)
        logger.debug(
            f"Successfully read img with shape={img.shape}; dtype={img.dtype}."
        )
        return img
    except Exception as e:
        logger.warning(f"Failed to read {fname} Because {e}!")
        return None


def load_info(fname: str) -> EasyDict:
    """Load EXIF and icc_profile information of the given image file.

    Args:
        fname (str): /path/to/the/image.file

    Returns:
        EasyDict: a Easydict that stores EXIF information.
    """
    info = None
    with open(fname, mode='rb') as f:
        with pyexiv2.ImageData(f.read()) as image_data:
            # 基础信息
            exifdata = image_data.read_exif()
            colorprofile = image_data.read_icc()
            info = EasyDict(
                exif=EasyDict(exifdata),
                colorprofile=colorprofile,
            )
    return info


@time_cost_warpper
def save_img(filename: str,
             img: np.ndarray,
             png_compressing: int = 0,
             jpg_quality: int = 90,
             exif: bytes = b"",
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
    # TODO: 为无exif/无colorprofile的场景增加兜底逻辑
    # TODO: 增加colorprofile转换的情况
    logger.info(f"Saving image to {filename} ...")
    suffix = filename.upper().split(".")[-1]

    # 将图像通过OpenCV进行编码
    if suffix == "PNG":
        ext = ".png"
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), png_compressing]
    elif suffix in ["JPG", "JPEG"]:
        # 导出 jpg 时，位深度强制转换为8
        if img.dtype == np.uint16:
            img = np.array(img // 255, dtype=np.uint8)
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

    with pyexiv2.ImageData(buf.tobytes()) as image_data:
        image_data.modify_icc(colorprofile)

        # TODO: 增加exif的写入
        #for key, value in exif_data.items():
        #    image_data[key] = pyexiv2.ExifTag(key, value)

        # 写入文件
        with open(filename, mode='wb') as f:
            f.write(image_data.get_bytes())