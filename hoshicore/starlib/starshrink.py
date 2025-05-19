import cv2
import numpy as np

from ..ezlib.utils import DTYPE_NUM2TYPE, DTYPE_REVERSE_MAP, get_scale_x, rdtype_detector
from .stardetect import detect_starmask_by_threshold
from typing import Union, Optional

OPENCV_SHAPE_MAPPING = {"RECT": cv2.MORPH_RECT, "CROSS": cv2.MORPH_CLOSE}


def get_morph_kernel(shape_str: str, ksize: int) -> np.ndarray:
    """generate the specified kernel with ksize.

    Args:
        shape_str (str): shape type str. Should be selected from OPENCV_SHAPE_MAPPING.
        ksize (int): kernel size

    Raises:
        NotImplementedError: Raised when shape is not implemented.

    Returns:
        np.ndarray: generated kernel.
    """
    if shape_str in OPENCV_SHAPE_MAPPING:
        return cv2.getStructuringElement(OPENCV_SHAPE_MAPPING[shape_str],
                                         (ksize, ksize))
    elif shape_str == "CIRCLE":
        return generate_circle_op(ksize)
    raise NotImplementedError(
        f"Unknown shape_str {shape_str}: Only {OPENCV_SHAPE_MAPPING.keys()} and \"CIRCLE\" are supported."
    )


def generate_circle_op(ksize: int) -> np.ndarray:
    bg = np.zeros((ksize, ksize), dtype=np.uint8)
    center = ksize // 2
    radius = ksize // 2
    bg = cv2.circle(bg, (center, center),
                    radius=radius,
                    color=[1],
                    thickness=-1)
    return bg


def star_shrink_by_morphology(img: np.ndarray,
                              star_mask: Optional[np.ndarray] = None,
                              star_detect_params: Optional[dict] = None,
                              ksize: int = 5,
                              ratio: float = 1.0,
                              shape: str = "RECT",
                              mode: int = cv2.MORPH_ERODE,
                              times: int = 1,
                              deringing: bool = False,
                              deringing_algo: str = "median",
                              int_weight: bool = False):
    """基于形态学的缩星算法。该算法简单且快速，适用于星点比较稀疏的场景。
    
    该算法的主要流程如下：
    1. （在未指定其他星点检测算法的情况下）使用形态学方法检测出星点蒙版。
    2. 对星点蒙版内的区域执行给定的形态学算子。通常来说，是使用腐蚀或者最小值。
    3. 如果需要的话，缓解星点区域周围的振铃（黑圈）现象。

    Args:
        img (np.ndarray): 输入图像，支持常见图像dtype。
        star_mask (Optional[np.ndarray]): 星点掩模图像，当不指定时将使用阈值法自动提取星点。Defaults to None.
        star_detect_params (Optional[dict]): 如果需要缩星方法内使用阈值法自动提取星点，并需要指定参数，则配置在该项。Defaults to None.
        ksize (int, optional): _description_. Defaults to 5.
        ratio (float, optional): _description_. Defaults to 1.0.
        shape (str, optional): _description_. Defaults to "RECT".
        mode (int, optional): 缩星所使用的算法. Defaults to cv2.MORPH_ERODE.
        times (int, optional): _description_. Defaults to 1.
        deringing (bool, optional): 是否缓解黑圈. Defaults to False.
        int_weight (bool, optional): 是否在按ratio混合图像时使用整形数据代替浮点以加速运算. Defaults to False.

    Returns:
        _type_: _description_
    """
    assert 0 < ratio <= 1, "Invalid ratio!"
    # TODO: 未适配FLOAT输入。
    # TODO: 未适配整数权重逻辑.
    if int_weight:
        raise NotImplementedError("Int weight version is not ready yet!")
    if star_mask is None:
        if star_detect_params is None:
            star_detect_params = {}
        star_mask = detect_starmask_by_threshold(img, **star_detect_params)
    cv_kernel = get_morph_kernel(shape, ksize)
    deringing_img = img.copy()
    raw_dtype = img.dtype
    shrink_img = img.copy()
    for _ in range(times):
        processed_img = cv2.morphologyEx(shrink_img, mode, cv_kernel)
        if ratio == 1:
            shrink_img = processed_img
        else:
            shrink_img = processed_img * ratio + shrink_img * (1 - ratio)
    shrink_img = np.array(shrink_img, dtype=raw_dtype)
    if deringing:
        # TODO: 目前按照uint8统一处理。未来可能需要解决精度损失，可用数据范围问题。
        # TODO: 需要处理可能的IndexError。
        scaler_time = DTYPE_NUM2TYPE.index(rdtype_detector(deringing_img))
        downscale = get_scale_x(scaler_time)
        deringing_img = np.array(deringing_img // downscale, dtype=np.uint8)
        # TODO: 5 is a magic num.
        #deringing_img = cv2.medianBlur(deringing_img, ksize=ksize * 5)
        if deringing_algo == "median":
            deringing_img = cv2.medianBlur(deringing_img, ksize=ksize * 5)
        elif deringing_algo == "mean":
            deringing_img = cv2.blur(deringing_img,
                                     ksize=(ksize * 5, ksize * 5))
        else:
            raise NotImplementedError()
        if downscale != 1:
            deringing_img = np.array(deringing_img,
                                     dtype=raw_dtype) * downscale
        shrink_img = np.max([shrink_img, deringing_img], axis=0)
    shrink_img = shrink_img * star_mask[..., None] + img * (
        1 - star_mask[..., None])
    return np.array(shrink_img, dtype=raw_dtype)
