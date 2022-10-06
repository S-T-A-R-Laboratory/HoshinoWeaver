# imgloader is designed to support common kinds of images.
# RAW images are not supported so far.
# For color space management, now we can only do this in a dirty way. Hope that I can fix it sometime in the future.

# In this implementation, metadata extraction is done by Pillow, and image loading is via OpenCV-python.

import numpy as np
import cv2
import rawpy
from PIL import Image
from .utils import MetaCls, Munch

support_color_space = ["Adobe RGB", "ProPhoto RGB", "sRGB"]
common_suffix = ["tiff", "tif", "jpg", "png", "jpeg"]
not_recom_suffix = ["bmp", "gif", "fits"]
raw_suffix = ["cr2", "cr3", "arw", "nef", "dng"]
support_bits = [8, 16]


def get_color_profile(color_bstring):
    color_profile = color_bstring.decode("latin-1", errors="ignore")
    if not color_profile: return None
    for color_space in support_color_space:
        if color_space in color_profile:
            return color_space
    return NotImplementedError(
        "Unsupported color space. For now only these color spaces are supported: %s"
        % support_color_space)


def load_single_img(img_filename, with_exif=True):
    """_summary_

    Args:
        img_filename (_type_): _description_
        with_exif (bool, optional): _description_. Defaults to True.

    Raises:
        NotImplementedError: _description_
    """
    img = None
    colorprofile = None
    exifdata = None

    # suffix check and warning raising
    suffix = img_filename.split(".")[-1].lower()
    if not ((suffix in common_suffix) or (suffix in not_recom_suffix) or
            (suffix in raw_suffix)):
        raise NotImplementedError(
            "For now only images with these suffix are supported: %s" %
            common_suffix + raw_suffix)
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
    else:
        # load images with rawpy
        with rawpy.imread(img_filename) as raw:
            img = raw.postprocess(output_bps=16,
                                  output_color=rawpy.rawpy.ColorSpace(4))
            # by default selecting ProRGB
            colorprofile = "ProPhoto RGB"

    # load exif
    if with_exif:
        # Color Management: Get ICC Profile
        if suffix in ["jpg", "png", "jpeg"]:
            # for images with compressed metadata(?)
            exifdata = MetaCls(meta_img.getexif())
            colorprofile = get_color_profile(
                getattr(exifdata, 'icc_profile', b''))
        elif suffix in ["fits", "tiff", "tif"]:
            exifdata = MetaCls(
                {key: meta_img.tag[key]
                 for key in meta_img.tag_v2})
            potiental_key = [x for x in exifdata.tags if "color" in x.lower()]
            if len(potiental_key) > 0:
                colorprofile = get_color_profile(
                    getattr(exifdata, potiental_key[0]))

    # extract array from img. Force converting to int16.
    # (support int32 in the future?)
    return Munch(img=img, color_space=colorprofile, exif=exifdata)
