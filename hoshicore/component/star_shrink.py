from typing import Optional

import cv2
import numpy as np

from .data_container import rescale_array

OPENCV_SHAPE_MAPPING = {"RECT": cv2.MORPH_RECT, "CROSS": cv2.MORPH_CROSS}


def get_morph_kernel(shape_str: str, ksize: int) -> np.ndarray:
    if shape_str in OPENCV_SHAPE_MAPPING:
        return cv2.getStructuringElement(OPENCV_SHAPE_MAPPING[shape_str],
                                         (ksize, ksize))
    elif shape_str == "CIRCLE":
        return _generate_circle_kernel(ksize)
    raise NotImplementedError(
        f"Unknown shape_str {shape_str}: "
        f"Only {list(OPENCV_SHAPE_MAPPING.keys())} and \"CIRCLE\" are supported."
    )


def _generate_circle_kernel(ksize: int) -> np.ndarray:
    bg = np.zeros((ksize, ksize), dtype=np.uint8)
    center = ksize // 2
    cv2.circle(bg, (center, center), radius=center, color=1, thickness=-1)
    return bg


def morph_shrink(img: np.ndarray,
                 ksize: int = 5,
                 ratio: float = 1.0,
                 shape: str = "RECT",
                 mode: int = cv2.MORPH_ERODE,
                 times: int = 1) -> np.ndarray:
    """对图像执行形态学缩星，返回全图处理结果（不含蒙版裁剪）。

    Args:
        img: 输入图像，2D 或 3D。
        ksize: 形态学核大小。
        ratio: 缩星强度 (0, 1]，1 为完全替换，<1 为每步与上一步结果混合。
        shape: 核形状，"RECT" / "CROSS" / "CIRCLE"。
        mode: OpenCV 形态学操作类型，默认 cv2.MORPH_ERODE。
        times: 迭代次数。

    Returns:
        np.ndarray: 形态学处理后的图像，dtype 与输入一致。
    """
    assert 0 < ratio <= 1, f"ratio must be in (0, 1], got {ratio}"
    cv_kernel = get_morph_kernel(shape, ksize)

    if ratio == 1:
        return cv2.morphologyEx(img, mode, cv_kernel, iterations=times)

    result = img.copy()
    raw_dtype = img.dtype
    for _ in range(times):
        processed = cv2.morphologyEx(result, mode, cv_kernel)
        result = np.round(
            processed * ratio + result * (1 - ratio)
        ).astype(raw_dtype)
    return result


def deringing(img: np.ndarray,
              shrink_img: np.ndarray,
              algo: str = "median",
              ksize: int = 25) -> np.ndarray:
    """缓解缩星后的振铃（黑圈）现象。

    对原图做大核模糊得到平滑背景估计，取 max(shrink, blurred) 填补凹陷。

    Args:
        img: 原始图像（缩星前），用于估计背景。
        shrink_img: 缩星后的图像。
        algo: 模糊算法，"median" 或 "mean"。
        ksize: 模糊核大小（奇数）。

    Returns:
        np.ndarray: 振铃修复后的图像，dtype 与 shrink_img 一致。
    """
    dk = ksize if ksize % 2 == 1 else ksize + 1
    raw_dtype = shrink_img.dtype

    blurred = rescale_array(img, img.dtype, np.dtype("uint8"))
    if algo == "median":
        blurred = cv2.medianBlur(blurred, ksize=dk)
    elif algo == "mean":
        blurred = cv2.blur(blurred, ksize=(dk, dk))
    else:
        raise NotImplementedError(f"Unknown deringing algo: {algo}")
    blurred = rescale_array(blurred, np.dtype("uint8"), raw_dtype)

    return np.maximum(shrink_img, blurred)


def apply_mask(img: np.ndarray,
               processed: np.ndarray,
               star_mask: np.ndarray) -> np.ndarray:
    """用蒙版合成：蒙版区域取 processed，其余取 img。

    Args:
        img: 原始图像。
        processed: 处理后的图像（缩星 / 振铃修复后）。
        star_mask: uint8 二值蒙版 (0/1)，2D。

    Returns:
        np.ndarray: 合成结果，dtype 与 img 一致。
    """
    assert img.shape[:2] == star_mask.shape, \
        f"star_mask shape {star_mask.shape} != img spatial shape {img.shape[:2]}"
    mask_nd = star_mask[..., None] if img.ndim == 3 else star_mask
    return np.where(mask_nd, processed, img).astype(img.dtype)


def _guided_filter(guide: np.ndarray,
                   src: np.ndarray,
                   radius: int,
                   eps: float) -> np.ndarray:
    """单通道引导滤波（O(N) box-filter 实现）。

    参考 He et al. "Guided Image Filtering", ECCV 2010.

    Args:
        guide: 引导图，float32，2D。
        src: 输入图，float32，2D。
        radius: 滤波半径，box filter 窗口 = (2*radius+1)。
        eps: 正则化项，控制边缘保持强度（越小越锐利）。

    Returns:
        np.ndarray: 滤波结果，float32，2D。
    """
    ksize = (2 * radius + 1, 2 * radius + 1)

    mean_g = cv2.blur(guide, ksize)
    mean_s = cv2.blur(src, ksize)
    corr_gs = cv2.blur(guide * src, ksize)
    corr_gg = cv2.blur(guide * guide, ksize)

    var_g = corr_gg - mean_g * mean_g
    cov_gs = corr_gs - mean_g * mean_s

    a = cov_gs / (var_g + eps)
    b = mean_s - a * mean_g

    mean_a = cv2.blur(a, ksize)
    mean_b = cv2.blur(b, ksize)

    return mean_a * guide + mean_b


def apply_mask_guided(img: np.ndarray,
                      processed: np.ndarray,
                      star_mask: np.ndarray,
                      radius: int = 8,
                      eps: float = 0.01) -> np.ndarray:
    """用引导滤波生成软蒙版，边缘保持地混合缩星结果与原图。

    相比 apply_mask 的硬切边界，引导滤波让过渡区域跟随图像边缘自然衰减，
    避免缩星后的明显边界和振铃。

    Args:
        img: 原始图像。
        processed: 处理后的图像（缩星后）。
        star_mask: uint8 二值蒙版 (0/1)，2D。
        radius: 引导滤波半径，控制过渡区域宽度。
        eps: 正则化项，越小边缘越锐利，越大越平滑。
             参考值基于 [0,1] 归一化图像：0.001（锐利）~ 0.1（平滑）。

    Returns:
        np.ndarray: 合成结果，dtype 与 img 一致。
    """
    assert img.shape[:2] == star_mask.shape, \
        f"star_mask shape {star_mask.shape} != img spatial shape {img.shape[:2]}"

    if img.dtype.kind == 'f':
        guide = img.astype(np.float32)
    else:
        guide = img.astype(np.float32) / np.iinfo(img.dtype).max

    if guide.ndim == 3:
        guide_gray = cv2.cvtColor(guide, cv2.COLOR_BGR2GRAY)
    else:
        guide_gray = guide

    mask_f = star_mask.astype(np.float32)
    soft_mask = _guided_filter(guide_gray, mask_f, radius, eps)
    soft_mask = np.clip(soft_mask, 0, 1)

    if img.ndim == 3:
        soft_mask = soft_mask[..., None]

    result = processed.astype(np.float64) * soft_mask + \
             img.astype(np.float64) * (1 - soft_mask)
    return np.round(result).astype(img.dtype)
