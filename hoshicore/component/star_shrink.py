from typing import Optional

import cv2
import numpy as np
from scipy.ndimage import median_filter

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


def morph_shrink_luma(img: np.ndarray,
                      ksize: int = 5,
                      shape: str = "CIRCLE",
                      times: int = 1,
                      ratio: Optional[float] = None) -> np.ndarray:
    """LAB 空间仅腐蚀 L 通道，ab 通道原封不动，避免色偏。

    Args:
        img: 输入图像，2D 或 3D（BGR），uint8/uint16/float32。
        ksize: 腐蚀核大小。
        shape: 核形状，"RECT" / "CROSS" / "CIRCLE"。
        times: 腐蚀迭代次数。
        ratio: 每步腐蚀结果的混合权重 [0, 1]。None = 自动（1/times）。
               1.0 = 每步完全替换（等价于旧 iterations=times 行为）。

    Returns:
        np.ndarray: 腐蚀结果，dtype 与输入一致。
    """
    if ratio is None:
        ratio = 1.0 / times
    cv_kernel = get_morph_kernel(shape, ksize)
    raw_dtype = img.dtype

    if img.dtype.kind == 'f':
        img_f = img.astype(np.float32)
        max_val = 1.0
    else:
        max_val = float(np.iinfo(img.dtype).max)
        img_f = img.astype(np.float32) / max_val

    if img.ndim == 2:
        current = img_f.copy()
        for _ in range(times):
            eroded = cv2.morphologyEx(current, cv2.MORPH_ERODE, cv_kernel)
            current = eroded * ratio + current * (1.0 - ratio)
        result_f = current
    else:
        lab = cv2.cvtColor(img_f, cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0].copy()
        for _ in range(times):
            eroded = cv2.morphologyEx(L, cv2.MORPH_ERODE, cv_kernel)
            L = eroded * ratio + L * (1.0 - ratio)
        lab[:, :, 0] = L
        result_f = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), 0.0, 1.0)

    if img.dtype.kind == 'f':
        return result_f.astype(raw_dtype)
    return np.round(result_f * max_val).astype(raw_dtype)


def peak_recovery(img_original: np.ndarray,
                  img_eroded: np.ndarray,
                  bg_ksize: int = 25,
                  strength: float = 0.85,
                  scale: float = 0.20) -> np.ndarray:
    """亮度自适应峰值恢复：按像素高于背景的归一化程度决定恢复权重。

    公式（float32 [0,1] 空间）：
        bg_L       = mean_blur(L_original, bg_ksize)
        above_bg   = clip(L_original - bg_L, 0, inf)
        above_bg_n = above_bg / max(above_bg)     # 归一化，最亮星=1
        w = clip(above_bg_n / scale, 0, 1) * strength
        result = img_eroded + w * (img_original - img_eroded)

    Args:
        img_original: 腐蚀前原始图像。
        img_eroded: 腐蚀后图像（由 morph_shrink_luma 生成）。
        bg_ksize: 背景估算核大小（均值模糊），需大于最大星点直径。
        strength: 最大恢复比例 [0,1]，0=不恢复，1=完全恢复峰值。
        scale: above_bg_n 归一化阈值，达到此值时 weight 饱和到 strength。
               值越小，受保护的星越多。

    Returns:
        np.ndarray: 恢复后图像，dtype 与 img_original 一致。
    """
    raw_dtype = img_original.dtype

    def _to_float(arr: np.ndarray) -> np.ndarray:
        if arr.dtype.kind == 'f':
            return arr.astype(np.float32)
        return arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)

    orig_f = _to_float(img_original)
    eroded_f = _to_float(img_eroded)

    gray = cv2.cvtColor(orig_f, cv2.COLOR_BGR2GRAY) if orig_f.ndim == 3 else orig_f

    dk = bg_ksize if bg_ksize % 2 == 1 else bg_ksize + 1
    bg = cv2.blur(gray, ksize=(dk, dk))

    above_bg = np.maximum(gray - bg, 0.0)
    peak_bg = float(np.max(above_bg))
    if peak_bg < 1e-6:
        return img_eroded

    above_bg_n = above_bg / peak_bg
    w = np.clip(above_bg_n / scale, 0.0, 1.0) * strength
    if orig_f.ndim == 3:
        w = w[:, :, None]

    result_f = eroded_f + w * (orig_f - eroded_f)

    if raw_dtype.kind == 'f':
        return result_f.astype(raw_dtype)
    max_val = float(np.iinfo(raw_dtype).max)
    return np.round(np.clip(result_f * max_val, 0.0, max_val)).astype(raw_dtype)


def deringing(img: np.ndarray,
              shrink_img: np.ndarray,
              algo: str = "gaussian",
              ksize: int = 11) -> np.ndarray:
    """缓解缩星后的振铃（黑圈）现象。

    对原图做大核模糊得到平滑背景估计，取 max(shrink, blurred) 填补凹陷。
    内部使用 float32 计算，避免降位到 uint8 导致的量化失真。

    Args:
        img: 原始图像（缩星前），用于估计背景。
        shrink_img: 缩星后的图像。
        algo: 模糊算法，"gaussian"（默认，快速）/ "mean" / "median"（慢，仅小图）。
        ksize: 模糊核大小（奇数）。"gaussian" 使用 sigmaX = ksize / 3。

    Returns:
        np.ndarray: 振铃修复后的图像，dtype 与 shrink_img 一致。
    """
    dk = ksize if ksize % 2 == 1 else ksize + 1
    raw_dtype = shrink_img.dtype

    def _to_float(arr: np.ndarray) -> np.ndarray:
        if arr.dtype.kind == 'f':
            return arr.astype(np.float32)
        return arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)

    img_f = _to_float(img)
    shrink_f = _to_float(shrink_img)

    if algo == "gaussian":
        blurred = cv2.GaussianBlur(img_f, (dk, dk), sigmaX=dk / 3.0)
    elif algo == "mean":
        blurred = cv2.blur(img_f, ksize=(dk, dk))
    elif algo == "median":
        blurred = median_filter(img_f, size=dk, mode='reflect').astype(np.float32)
    else:
        raise NotImplementedError(f"Unknown deringing algo: {algo}")

    result_f = np.maximum(shrink_f, blurred)

    if raw_dtype.kind == 'f':
        return result_f.astype(raw_dtype)
    max_val = float(np.iinfo(raw_dtype).max)
    return np.round(np.clip(result_f * max_val, 0.0, max_val)).astype(raw_dtype)


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

    max_val = np.iinfo(img.dtype).max if img.dtype.kind != 'f' else 1.0
    if processed.dtype.kind == 'f':
        guide = processed.astype(np.float32)
    else:
        guide = processed.astype(np.float32) / max_val

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
