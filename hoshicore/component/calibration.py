"""
图像校准纯函数：减法校准、除法校准、缩放、裁切等基础操作。

所有函数接收裸 np.ndarray 和 dtype 信息，返回处理后的数组。
Op 层负责 FloatImage 拆包/重包装和 DAG 调度。
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .data_container import DTYPE_MAX_VALUE, align_dtype_pair

# 整数减法的最小安全有符号 dtype 映射
_SIGNED_DTYPE = {
    np.dtype("uint8"):  np.dtype("int16"),
    np.dtype("uint16"): np.dtype("int32"),
    np.dtype("uint32"): np.dtype("int64"),
}

# resize 插值方法映射
INTERPOLATION_METHODS: dict[str, int] = {
    "nearest":  cv2.INTER_NEAREST,
    "linear":   cv2.INTER_LINEAR,
    "area":     cv2.INTER_AREA,
    "cubic":    cv2.INTER_CUBIC,
    "lanczos":  cv2.INTER_LANCZOS4,
}


def calibration_subtract(
    frame: np.ndarray,
    reference: np.ndarray,
    frame_dtype: np.dtype,
    ref_dtype: np.dtype,
) -> tuple[np.ndarray, np.dtype]:
    """通用校准减法：frame - reference，结果 clamp 到 [0, max_val]。

    适用于暗场减法（dark subtraction）和偏置减法（bias subtraction）。
    两个输入先通过 align_dtype_pair 对齐到同一 dtype 级别，
    然后提升到最小安全有符号类型做减法，避免 float64 的内存和性能开销。

    Args:
        frame: 光场帧数组。
        reference: 校准参考帧（暗场 / 偏置帧）。
        frame_dtype: frame 的语义 dtype。
        ref_dtype: reference 的语义 dtype。

    Returns:
        (result_array, output_dtype)
    """
    frame_aligned, ref_aligned, output_dtype = align_dtype_pair(
        frame, frame_dtype, reference, ref_dtype)

    # 选择最小安全计算 dtype
    calc_dtype = _SIGNED_DTYPE.get(output_dtype)
    if calc_dtype is not None:
        # 整数路径：提升到有符号类型，做减法，clip 后转回
        result = frame_aligned.astype(calc_dtype) - ref_aligned.astype(calc_dtype)
        np.clip(result, 0, DTYPE_MAX_VALUE[output_dtype], out=result)
        return result.astype(output_dtype), output_dtype
    else:
        # 浮点路径：直接减，clip 到 0
        result = frame_aligned - ref_aligned
        np.maximum(result, 0, out=result)
        return result, output_dtype


def calibration_divide(
    frame: np.ndarray,
    reference: np.ndarray,
    frame_dtype: np.dtype,
    ref_dtype: np.dtype,
) -> tuple[np.ndarray, np.dtype]:
    """通用校准除法：frame / reference * mean(reference)，归一化平场校正。

    适用于平场校正（flat field correction）。
    结果保持与原始帧相同的亮度水平。

    Args:
        frame: 光场帧数组。
        reference: 校准参考帧（主平场）。
        frame_dtype: frame 的语义 dtype。
        ref_dtype: reference 的语义 dtype。

    Returns:
        (result_array, output_dtype)
    """
    frame_aligned, ref_aligned, output_dtype = align_dtype_pair(
        frame, frame_dtype, reference, ref_dtype)

    ref_f = ref_aligned.astype(np.float64)
    ref_mean = np.mean(ref_f)

    # 避免除零
    ref_safe = np.where(ref_f > 0, ref_f, 1.0)

    result = frame_aligned.astype(np.float64) / ref_safe * ref_mean

    if output_dtype in DTYPE_MAX_VALUE:
        np.clip(result, 0, DTYPE_MAX_VALUE[output_dtype], out=result)

    return result.astype(output_dtype), output_dtype


def resize_image(
    img: np.ndarray,
    scale: Optional[float] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    interpolation: str = "auto",
) -> np.ndarray:
    """按缩放比例或目标尺寸缩放图像。

    优先级: scale > (width, height) > passthrough。

    Args:
        img: 输入图像 (H, W) 或 (H, W, C)。
        scale: 缩放比例，如 0.5 表示缩小一半。
        width: 目标宽度（仅指定一个维度时自动计算另一个）。
        height: 目标高度。
        interpolation: 插值方法。
            "auto" — 缩小用 area，放大用 cubic。
            可选: "nearest", "linear", "area", "cubic", "lanczos"。

    Returns:
        缩放后的图像，无缩放参数时返回原图。
    """
    h, w = img.shape[:2]

    if scale is not None:
        new_w, new_h = int(w * scale), int(h * scale)
    elif width is not None or height is not None:
        if width is not None and height is not None:
            new_w, new_h = width, height
        elif width is not None:
            new_w = width
            new_h = int(h * (width / w))
        else:
            new_h = height
            new_w = int(w * (height / h))
    else:
        return img

    if new_w <= 0 or new_h <= 0:
        return img

    if interpolation == "auto":
        interp = cv2.INTER_AREA if (new_w < w or new_h < h) else cv2.INTER_CUBIC
    else:
        interp = INTERPOLATION_METHODS.get(interpolation)
        if interp is None:
            raise ValueError(
                f"Unknown interpolation '{interpolation}'. "
                f"Available: {sorted(INTERPOLATION_METHODS.keys())}")

    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def crop_roi(
    img: np.ndarray,
    x: int,
    y: int,
    roi_width: int,
    roi_height: int,
) -> np.ndarray:
    """从图像中裁切感兴趣区域 (ROI)。

    坐标系：(x, y) 为 ROI 左上角，x 为列方向，y 为行方向。
    超出图像边界的部分自动裁剪到有效范围。

    Args:
        img: 输入图像 (H, W) 或 (H, W, C)。
        x: ROI 左上角 x 坐标（列）。
        y: ROI 左上角 y 坐标（行）。
        roi_width: ROI 宽度。
        roi_height: ROI 高度。

    Returns:
        裁切后的图像。
    """
    h, w = img.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x + roi_width)
    y2 = min(h, y + roi_height)

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Invalid ROI: ({x}, {y}, {roi_width}, {roi_height}) "
            f"for image of size ({w}, {h})")

    return img[y1:y2, x1:x2].copy()


def natural_sort_key(s: str):
    """自然排序 key：数字部分按数值比较，如 img_2 < img_10。"""
    import re
    return [int(c) if c.isdigit() else c.lower()
            for c in re.split(r'(\d+)', str(s))]
