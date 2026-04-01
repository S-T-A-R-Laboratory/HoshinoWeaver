"""
TaggedImage：图像 + dtype 元信息的复合结构体。

在 DAG 节点间流转，使每个节点能自包含地理解数据的真实标尺，
解决多节点图中 "谁负责 rescale" 的职责归属问题。

设计原则：
    - 只承载元信息，不重载 numpy 运算符
    - Merger / Op 内部解包 .data 做运算，完成后重新打包
    - 与 FastGaussianParam 平行共存，不嵌套
    - scale_factor 由 source_dtype 与 data.dtype 的级差自动推算，
      Op 只需在 DataLoader 时设一次 source_dtype，后续全自动
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# ── dtype 级差表 ──
# 级差 = DTYPE_LEVEL[dtype]，两个 dtype 之间每差一级对应 256^1 的放缩
# 例如 uint8(level=0) → uint16(level=1) 的 scale = 256^1 + 1 = 257

DTYPE_LEVEL: dict[np.dtype, int] = {
    np.dtype("uint8"):  0,
    np.dtype("uint16"): 1,
    np.dtype("uint32"): 2,
    np.dtype("uint64"): 3,
}

_SCALE_BASE = 256


def _compute_scale(source_dtype: np.dtype, current_dtype: np.dtype) -> int:
    """根据 source 和 current 的 dtype 级差计算 scale_factor。

    仅在两端都是整型 (在 DTYPE_LEVEL 中) 时计算。
    如果 source == current，scale = 1（未放缩）。
    如果 current 级别更高，scale = 256^级差 + 1。
    如果 current 级别更低或为 float，scale = 1（不做自动推算）。
    """
    src_level = DTYPE_LEVEL.get(source_dtype)
    cur_level = DTYPE_LEVEL.get(current_dtype)
    if src_level is None or cur_level is None:
        return 1
    diff = cur_level - src_level
    if diff <= 0:
        return 1
    return _SCALE_BASE ** diff + 1


@dataclass(slots=True)
class TaggedImage:
    """图像数据 + dtype 放缩元信息。

    Attributes:
        data:         实际像素数组（可能处于 upscaled dtype）。
        source_dtype: 原始输入图像的 dtype（如 uint8）。

    scale_factor 自动从 source_dtype 与 data.dtype 的级差推算：
        - uint8 data + uint8 source  → scale = 1
        - uint16 data + uint8 source → scale = 257  (int_weight 放缩)
        - uint16 data + uint16 source → scale = 1   (原生 16bit 输入)
    """
    data: np.ndarray
    source_dtype: np.dtype

    # ── 自动推算的 scale_factor ──

    @property
    def scale_factor(self) -> int:
        """根据 source_dtype 与当前 data.dtype 的级差自动计算。"""
        return _compute_scale(self.source_dtype, self.data.dtype)

    # ── numpy 友好属性 ──

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    # ── rescale 工具方法 ──

    def rescale_to(self, target_dtype: Optional[np.dtype] = None) -> np.ndarray:
        """将数据还原到目标 dtype。

        Args:
            target_dtype: 目标 dtype。None 时还原到 source_dtype。

        Returns:
            rescale 后的 np.ndarray（不修改自身）。
        """
        target = target_dtype if target_dtype is not None else self.source_dtype
        sf = self.scale_factor
        if sf <= 1:
            return self.data.astype(target)
        return (self.data // sf).astype(target)

    def rescale_to_source(self) -> np.ndarray:
        """还原到原始 dtype 的便捷方法。"""
        return self.rescale_to(self.source_dtype)

    @property
    def is_scaled(self) -> bool:
        """数据当前是否处于放缩状态。"""
        return self.scale_factor > 1
