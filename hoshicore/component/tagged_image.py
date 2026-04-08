"""
dtype 级差与 rescale 工具函数。

节点间传递裸 np.ndarray，约定"值域填满容器 dtype 的范围"。
当需要在保存或跨 dtype 对齐时做 rescale，使用本模块的纯函数。

设计原则：
    - 不引入包装类型，节点间直接传递 ndarray
    - Merger / Op 内部做运算，保存时按需调用 rescale_array
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# ── dtype 级差表 ──
# 级差 = DTYPE_LEVEL[dtype]，两个 dtype 之间每差一级对应 256^(n+1) 的放缩
# 例如 uint8(level=0) → uint16(level=1) 的 scale = 256^1 + 1 = 257

DTYPE_LEVEL: dict[np.dtype, int] = {
    np.dtype("uint8"): 0,
    np.dtype("uint16"): 1,
    np.dtype("uint32"): 2,
    np.dtype("uint64"): 3,
}

_SCALE_BASE = 256


def _cumscale_factor(level: int) -> int:
    """级差 level 对应的累积放缩 factor。

    例如 level=2（uint8 → uint32）时，factor = 256^1 + 1 (uint8→uint16) × 256^1 + 1 (uint16→uint32) = 66049
    """
    factor = 1
    for i in range(abs(level)):
        factor *= _SCALE_BASE + 1
    return factor

def rescale_array(
    data: np.ndarray,
    from_dtype: np.dtype,
    to_dtype: np.dtype,
) -> np.ndarray:
    """在两个 dtype 级别之间双向放缩数据。

    根据 from_dtype 与 to_dtype 的级差自动选择方向：
        - to 级别更高：向上放缩（× scale），如 uint8 [0,255] → uint16 [0,65535]
        - to 级别更低：向下缩放（÷ scale），如 uint16 [0,65535] → uint8 [0,255]
        - 同级或无法判定：仅做 dtype cast

    Args:
        data: 像素数组，值域填满 from_dtype 的范围。
        from_dtype: data 当前的语义 dtype 级别。
        to_dtype: 目标 dtype 级别。

    Returns:
        放缩后的 np.ndarray（dtype 为 to_dtype）。
    """
    from_level = DTYPE_LEVEL.get(np.dtype(from_dtype))
    to_level = DTYPE_LEVEL.get(np.dtype(to_dtype))

    if from_level is None or to_level is None:
        return data.astype(to_dtype)

    diff = to_level - from_level
    if diff == 0:
        return data.astype(to_dtype)
    sf = _cumscale_factor(diff)
    if diff > 0:
        # 向上放缩
        return data.astype(to_dtype) * sf
    else:
        # 向下缩放
        return (data // sf).astype(to_dtype)


def align_dtype_pair(
    arr_a: np.ndarray,
    dtype_a: np.dtype,
    arr_b: np.ndarray,
    dtype_b: np.dtype,
) -> tuple[np.ndarray, np.ndarray, np.dtype]:
    """对齐两个数组的 dtype 级别，将较低级别的数组放缩到较高级别。

    用于多来源数据的 Op（如 MaxNoiseEqualizationOp），确保两个
    来自不同上游 stacker 的数据在同一数值范围内进行运算。

    Args:
        arr_a: 第一个数组。
        dtype_a: arr_a 的语义 dtype（即它"填满"哪个 dtype 的范围）。
        arr_b: 第二个数组。
        dtype_b: arr_b 的语义 dtype。

    Returns:
        (aligned_a, aligned_b, common_dtype) 元组。
        common_dtype 为对齐后的公共 dtype（取较高级别者）。
    """
    level_a = DTYPE_LEVEL.get(np.dtype(dtype_a))
    level_b = DTYPE_LEVEL.get(np.dtype(dtype_b))

    # 无法判定级别时（float 或未知 dtype），不做放缩
    if level_a is None or level_b is None:
        return arr_a, arr_b, dtype_a

    if level_a == level_b:
        return arr_a, arr_b, dtype_a

    if level_a < level_b:
        # a 级别低，放缩 a 到 b 的级别
        return rescale_array(arr_a, dtype_a, dtype_b), arr_b, dtype_b
    else:
        # b 级别低，放缩 b 到 a 的级别
        return arr_a, rescale_array(arr_b, dtype_b, dtype_a), dtype_a


@dataclass(slots=True)
class FloatImage(object):
    """轻量级的Float矩阵包装器，包含原始数据范围dtype。

    Args:
        data (np.ndarray): _description_
        dtype (np.dtype): _description_
    """
    data: np.ndarray
    dtype: np.dtype

    def int_transform(self,
                      target_dtype: Optional[np.dtype] = None) -> np.ndarray:
        if target_dtype is None:
            target_dtype = self.dtype

        return rescale_array(self.data, self.dtype, target_dtype)
