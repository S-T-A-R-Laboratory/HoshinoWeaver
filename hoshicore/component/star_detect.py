from typing import Optional, Union

import cv2
import numpy as np


def detect_starmask_by_threshold(img: np.ndarray,
                                 ksize: int = 13,
                                 med_algo: str = "median",
                                 threshold_ratio: Union[int, float] = 5,
                                 open_ksize: int = 3,
                                 dilate_ksize: int = 0) -> np.ndarray:
    """基于阈值的星点提取方法。可用于单张图像，是从单张图像估算星点的常用方法。

    算法：局部背景估计（中值/均值模糊）→ 全局 σ 阈值 → 形态学后处理。

    Args:
        img: 输入图像，支持任意整型/浮点 dtype，支持灰度或 BGR。
        ksize: 中值/均值滤波核大小，用于估计局部背景。
        med_algo: 背景估计算法，"median" 或 "mean"。
        threshold_ratio: 阈值倍率，mask = (img > bg + ratio * σ)。
        open_ksize: 开运算核大小，>0 时过滤散粒噪声和暗弱星点。0 为禁用。
        dilate_ksize: 膨胀核大小，>0 时扩展蒙版覆盖星点边缘。0 为禁用。

    Returns:
        np.ndarray (uint8): 星点蒙版，取值 0/1。
    """
    if img.dtype.kind == 'f':
        gray = img.astype(np.float32)
    else:
        gray = img.astype(np.float32) / np.iinfo(img.dtype).max

    if gray.ndim == 3 and gray.shape[-1] == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    if med_algo == "median":
        if ksize <= 5:
            bg = cv2.medianBlur(gray, ksize=ksize)
        else:
            gray_u8 = np.clip(gray * 255, 0, 255).astype(np.uint8)
            bg = cv2.medianBlur(gray_u8, ksize=ksize).astype(np.float32) / 255.0
    elif med_algo == "mean":
        bg = cv2.blur(gray, ksize=(ksize, ksize))
    else:
        raise NotImplementedError(f"Unknown med algo: {med_algo}.")

    diff = gray - bg
    threshold = np.std(diff) * threshold_ratio
    star_mask = (diff > threshold).astype(np.uint8)

    if open_ksize > 0:
        k = cv2.getStructuringElement(cv2.MORPH_CROSS, (open_ksize, open_ksize))
        star_mask = cv2.morphologyEx(star_mask, cv2.MORPH_OPEN, k)
    if dilate_ksize > 0:
        k = cv2.getStructuringElement(cv2.MORPH_CROSS, (dilate_ksize, dilate_ksize))
        star_mask = cv2.morphologyEx(star_mask, cv2.MORPH_DILATE, k)
    return star_mask


def detect_starmask_by_dog(img: np.ndarray,
                           sigma_small: float = 1.5,
                           sigma_large: float = 12.0,
                           threshold_ratio: Union[int, float] = 3,
                           open_ksize: int = 3,
                           dilate_ksize: int = 0) -> np.ndarray:
    """基于 DoG（Difference of Gaussians）的星点提取方法。

    算法：双高斯差分带通滤波 → 全局 σ 阈值 → 形态学后处理。
    相比中值/均值背景估计，DoG 对点源响应更强，对大尺度结构（星云、渐变）抑制更好，
    且 GaussianBlur 在大核时比 medianBlur 更快。

    Args:
        img: 输入图像，支持任意整型/浮点 dtype，支持灰度或 BGR。
        sigma_small: 小尺度高斯 σ，保留星点信号并抑制读出噪声。
        sigma_large: 大尺度高斯 σ，模糊掉星点以估计背景。
        threshold_ratio: 阈值倍率，mask = (DoG > ratio * σ(DoG))。
        open_ksize: 开运算核大小，>0 时过滤散粒噪声。0 为禁用。
        dilate_ksize: 膨胀核大小，>0 时扩展蒙版覆盖星点边缘。0 为禁用。

    Returns:
        np.ndarray (uint8): 星点蒙版，取值 0/1。
    """
    if img.dtype.kind == 'f':
        gray = img.astype(np.float32)
    else:
        gray = img.astype(np.float32) / np.iinfo(img.dtype).max

    if gray.ndim == 3 and gray.shape[-1] == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    blur_small = cv2.GaussianBlur(gray, (0, 0), sigma_small)
    blur_large = cv2.GaussianBlur(gray, (0, 0), sigma_large)
    dog = blur_small - blur_large

    threshold = np.std(dog) * threshold_ratio
    star_mask = (dog > threshold).astype(np.uint8)

    if open_ksize > 0:
        k = cv2.getStructuringElement(cv2.MORPH_CROSS, (open_ksize, open_ksize))
        star_mask = cv2.morphologyEx(star_mask, cv2.MORPH_OPEN, k)
    if dilate_ksize > 0:
        k = cv2.getStructuringElement(cv2.MORPH_CROSS, (dilate_ksize, dilate_ksize))
        star_mask = cv2.morphologyEx(star_mask, cv2.MORPH_DILATE, k)
    return star_mask


def starmask_to_star_coords(
    mask: np.ndarray,
    max_num: Optional[int] = None,
) -> np.ndarray:
    """从星点蒙版提取星点位置和大小，按面积降序排列。

    Args:
        mask (np.ndarray): uint8 二值蒙版（0/1）。
        max_num (Optional[int]): 最多返回的星点数量，None 表示全部返回。

    Returns:
        np.ndarray: shape (N, 3) float32，每行 (cx, cy, area)，按 area 降序。
                    无星点时返回 shape (0, 3)。
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.empty((0, 3), dtype=np.float32)

    stars = np.empty((len(contours), 3), dtype=np.float32)
    for i, cnt in enumerate(contours):
        m = cv2.moments(cnt)
        area = m["m00"]
        if area > 0:
            stars[i] = (m["m10"] / area, m["m01"] / area, area)
        else:
            x, y, w, h = cv2.boundingRect(cnt)
            stars[i] = (x + w * 0.5, y + h * 0.5, 1.0)

    order = np.argsort(-stars[:, 2])
    stars = stars[order]

    if max_num is not None:
        stars = stars[:max_num]
    return stars
