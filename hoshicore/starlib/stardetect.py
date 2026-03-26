from typing import Union, Optional

import cv2
import numpy as np


def detect_starmask_by_threshold(img: np.ndarray,
                             ksize: int = 13,
                             med_algo: str = "median",
                             threshold_ratio: Union[int, float] = 5,
                             filter_noise: bool = False,
                             enhance_range: bool = False,
                             remove_large_area: bool = False) -> np.ndarray:
    """基于阈值的星点提取方法。可用于单张图像，是从单张图像估算星点的常用方法。

    Args:
        img (np.ndarray): _description_
        ksize (int, optional): _description_. Defaults to 13.
        threshold_ratio (int, optional): _description_. Defaults to 5.

    Returns:
        np.ndarray (np.uint8): 星点蒙版图像。取值为0，1。
    """
    # TODO: this is an ugly fix. Should be optimized in the future.
    # DataReArr
    if np.max(img) > 65535:
        img = np.array(img // 65537, dtype=np.uint8)
    elif np.max(img) > 255:
        img = np.array(img // 257, dtype=np.uint8)
    else:
        img = np.array(img, dtype=np.uint8)

    if img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if med_algo == "median":
        med_img = cv2.medianBlur(img, ksize=ksize)
    elif med_algo == "mean":
        med_img = cv2.blur(img, ksize=(ksize,ksize))
    else:
        raise NotImplementedError(f"Unknown med algo: {med_algo}.")
    # 一种粗略的估计法：除以2后加128，处理低于阈值的负数情况。
    # 在放缩回方差时，该值需要*2。
    diff_img = (img // 2 + 128) - med_img // 2
    rej_diff = np.array(np.std(diff_img) * 2 * threshold_ratio, dtype=np.uint8)
    # perserve overflow
    med_img[med_img > 255 - rej_diff] = 255 - rej_diff
    rej_upper = med_img + rej_diff
    star_mask = np.array(img > rej_upper, dtype=np.uint8)
    # post-processing
    # TODO: 排除掉大面积/非圆形的干扰项
    cv_op = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    # 过滤散粒噪声和暗弱星点
    # TODO: 支持强度选项
    if filter_noise:
        star_mask = cv2.morphologyEx(star_mask, cv2.MORPH_OPEN, cv_op)
    # 扩大星点范围，降低星点边缘影响
    # TODO：支持强度选项
    if enhance_range:
        star_mask = cv2.morphologyEx(star_mask, cv2.MORPH_DILATE, cv_op)
    if remove_large_area:
        raise NotImplementedError("To be done")
    return star_mask

def starmask2star_coords(img: np.ndarray,max_num:Optional[int] = None, order: Optional[str] = None) -> np.ndarray:
    """从星点图像生成星点位置坐标序列。

    Args:
        img (np.ndarray): _description_
        max_num (Optional[int], optional): 最大返回的星点数量. Defaults to None.

    Returns:
        np.ndarray: _description_
    """
    contours1, counter2 = cv2.findContours(img,cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
    