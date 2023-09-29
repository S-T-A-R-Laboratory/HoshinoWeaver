"""
imgloader is designed to support common kinds of images.
RAW images are not supported so far.
For color space management, now we can only do this in a dirty way. Hope that I can fix it sometime in the future.
"""

# In this implementation, metadata extraction is done by Pillow, and image loading is via OpenCV-python.

import threading
import time
import multiprocessing
import cv2
import numpy as np
import rawpy
import queue
from easydict import EasyDict
from PIL import Image

from utils import MetaInfo

support_color_space = ["Adobe RGB", "ProPhoto RGB", "sRGB"]
common_suffix = ["tiff", "tif", "jpg", "png", "jpeg"]
not_recom_suffix = ["bmp", "gif", "fits"]
raw_suffix = ["cr2", "cr3", "arw", "nef", "dng"]
support_bits = [8, 16]


class BatchImgLoader(object):
    """_summary_

    Args:
        object (_type_): _description_
    """

    def __init__(self, imgname_list, max_poolsize=8):
        self.imgname_list = imgname_list
        self.buffer = queue.Queue(maxsize=max_poolsize)
        self.stopped = True
        self.prog = 0
        self.thread = threading.Thread(target=self.loop, args=())

    def start(self):
        self.stopped = False
        self.prog = 0
        self.buffer = []
        self.thread.start()
        return self

    def is_popable(self, num):
        if num <= len(self.buffer):
            return True
        return False

    def stop(self):
        self.stopped = True

    def pop(self, num=-1):
        """pop [num] images from the buffer. When num=-1, pop all.

        Args:
            num (int, optional): num of images. Defaults to -1.
        """
        if num == -1 or num > self.max_poolsize: num = self.max_poolsize
        while (not self.is_popable(num)):
            if self.stopped:
                break
            time.sleep(self.wait_interval)
        self.lock.acquire()
        ret = self.buffer[:num]
        self.buffer = self.buffer[num:]
        self.lock.release()
        return ret

    def loop(self):
        try:
            for i, (imgname, imgweight) in enumerate(
                    zip(self.imgname_list, self.weight_list)):
                while (len(self.buffer) >= self.max_poolsize
                       and not self.stopped):
                    time.sleep(self.wait_interval)
                if self.stopped:
                    break
                self.lock.acquire()
                self.buffer.append(
                    load_single_img(imgname).img * imgweight)
                self.lock.release()
                self.prog += 1
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


def load_single_img(img_filename):
    """_summary_

    Args:
        img_filename (_type_): _description_

    Returns:
        _type_: _description_
    """
    img = None
    colorprofile = None
    exifdata = None

    # suffix check and warning raising
    suffix = img_filename.split(".")[-1].lower()
    assert ((suffix in common_suffix) or (suffix in not_recom_suffix) or
            (suffix in raw_suffix)), f"Unsupported img suffix:{suffix}."
    
    if suffix in not_recom_suffix:
        print(
            Warning("Got an Image with not recommended suffix. \
            We do not guarantee the stability of EXIF extraction and the output image quality."
                    ))

    if (suffix in common_suffix) or (suffix in not_recom_suffix):
        # load metadata and img for non-raw images
        with open(img_filename, mode='rb') as img_file:
            meta_img = Image.open(img_file)
        img = cv2.imdecode(np.fromfile(img_filename, dtype=np.uint16),
                           cv2.IMREAD_UNCHANGED)
        # Color Management: Get ICC Profile
        if suffix in ["jpg", "png", "jpeg"]:
            # for images with compressed metadata(?)
            exifdata = MetaInfo(meta_img.getexif())
            colorprofile = get_color_profile(
                getattr(exifdata, 'icc_profile', b''))
        elif suffix in ["fits", "tiff", "tif"]:
            exifdata = MetaInfo({key: meta_img.tag[key] for key in meta_img.tag_v2})
            potiental_key = [x for x in exifdata.tags if "color" in x.lower()]
            if len(potiental_key) > 0:
                colorprofile = get_color_profile(
                    getattr(exifdata, potiental_key[0]))
    else:
        # load images with rawpy
        with rawpy.imread(img_filename) as raw:
            img = raw.postprocess(output_bps=16,
                                  output_color=rawpy.rawpy.ColorSpace(4))
        # by default selecting ProRGB
        colorprofile = "ProPhoto RGB"        

    # extract array from img. Force converting to int16.
    # (support int32 in the future?)
    return EasyDict(img=img, color_space=colorprofile, exif=exifdata)