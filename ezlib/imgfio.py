"""
imgfio包含了与图像IO相关的函数和类。

"""

# In this implementation, metadata extraction is done by Pillow, and image loading is via OpenCV-python.

import threading
import cv2
import numpy as np
import rawpy
import queue
from easydict import EasyDict
from PIL import Image
from .utils import MetaInfo
from typing import Optional

support_color_space = ["Adobe RGB", "ProPhoto RGB", "sRGB"]
common_suffix = ["tiff", "tif", "jpg", "png", "jpeg"]
not_recom_suffix = ["bmp", "gif", "fits"]
raw_suffix = ["cr2", "cr3", "arw", "nef", "dng"]
support_bits = [8, 16]


class ImgSeriesLoader(object):
    """用于多线程读取图像序列的类。

    Args:
        object (_type_): _description_
    """

    def __init__(self, fname_list, dtype=None, resize=None, max_poolsize=8):
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
        self.thread = threading.Thread(target=self.loop, args=())

    def start(self):
        self.stopped = False
        self.prog = 0
        while not self.buffer.empty():
            self.buffer.get()
        self.thread.start()

    def stop(self):
        self.stopped = True

    def pop(self):
        return self.buffer.get()

    def loop(self):
        try:
            for imgname in self.fname_list:
                if self.stopped:
                    break
                self.buffer.put(load_img(imgname, dtype=self.dtype, resize=self.resize))
                self.prog += 1
        except AssertionError as e:
            raise e
        finally:
            self.stop()


def get_color_profile(color_bstring):
    color_profile = color_bstring.decode("latin-1", errors="ignore")
    if not color_profile: return None
    for color_space in support_color_space:
        if color_space in color_profile:
            return color_space
    return NotImplementedError(
        "Unsupported color space. For now only these color spaces are supported: %s"
        % support_color_space)


def load_img(fname, dtype=None, resize=None):
    """_summary_

    Args:
        fname (_type_): _description_

    Returns:
        _type_: _description_
    """
    img = None

    # suffix check and warning raising
    suffix = fname.split(".")[-1].lower()
    assert ((suffix in common_suffix) or (suffix in not_recom_suffix)
            or (suffix in raw_suffix)), f"Unsupported img suffix:{suffix}."

    if suffix in not_recom_suffix:
        print(
            Warning("Got an Image with not recommended suffix. \
            We do not guarantee the stability of EXIF extraction and the output image quality."
                    ))

    if (suffix in common_suffix) or (suffix in not_recom_suffix):
        img = cv2.imdecode(np.fromfile(fname, dtype=np.uint16),
                           cv2.IMREAD_UNCHANGED)

    else:
        # load images with rawpy
        with rawpy.imread(fname) as raw:
            img = raw.postprocess(output_bps=16,
                                  output_color=rawpy.rawpy.ColorSpace(4))

    if dtype:
        img = np.array(img, dtype=dtype)
    if resize:
        # TODO: 添加插值相关
        img = cv2.resize(img, resize)
    return img


def load_exif(fname: str) -> EasyDict:
    """Load EXIF information of the given image file.
    (TODO: colorprofile is still what I'm concerning about...)

    Args:
        fname (str): /path/to/the/image.file

    Returns:
        EasyDict: a Easydict that stores EXIF information.
    """
    suffix = fname.lower()
    # load metadata and img for non-raw images
    img = load_img(fname)
    # cannot work for 16bit TIFF
    #with open(fname, mode='rb') as img_file:
    #    img = Image.open(img_file)
    img_size = img.shape[:2]
    dtype=img.dtype
    # FIXME: It just cannot work!
    # Color Management: Get ICC Profile
    if suffix in ["jpg", "png", "jpeg"]:
        # for images with compressed metadata(?)
        exifdata = MetaInfo(img.getexif())
        colorprofile = get_color_profile(getattr(exifdata, 'icc_profile', b''))
    elif suffix in ["fits", "tiff", "tif"]:
        exifdata = MetaInfo(
            {key: img.tag[key]
             for key in img.tag_v2})
        potiental_key = [x for x in exifdata.tags if "color" in x.lower()]
        if len(potiental_key) > 0:
            colorprofile = get_color_profile(
                getattr(exifdata, potiental_key[0]))
    return EasyDict(img_size=img_size,
                    dtype=dtype)


def save_img(filename: str,
                    img: np.ndarray,
                    png_compressing: Optional[int] = 1,
                    jpg_quality: Optional[int] = 90):
    """保存单个图像到指定路径下。

    Args:
        filename (str): _description_
        img (np.ndarray): _description_
        png_compressing (Optional[int], optional): PNG压缩参数，1-10. Defaults to 1.
        jpg_quality (Optional[int], optional): JPG质量参数（0-100）. Defaults to 90.

    Raises:
        NameError: _description_
        Exception: _description_
    """
    suffix = filename.upper().split(".")[-1]
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
    if status:
        with open(filename, mode='wb') as f:
            f.write(buf)
    else:
        raise Exception("imencode failed.")