""" merger管理所有合并器类型。该类型定义不同堆栈模式时的后处理和合并逻辑，并暂存叠加结果。
"""
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Optional, Union, Any, cast

import numpy as np
from numpy.typing import NDArray

from .tagged_image import DTYPE_LEVEL, _SCALE_BASE, FloatImage
from .utils import (DTYPE_MAX_VALUE, DTYPE_UPSCALE_MAP,
                     FastGaussianParam, HuberMeanParam)


class BaseMerger(metaclass=ABCMeta):
    """合并器基类。

    Args:
        int_weight: 是否启用整型权重放缩。
            当 True 时，merge() 将自动：
            1) 把图像 upscale 到更高 dtype（如 uint8→uint16）
            2) 把 float 权重映射到对应整型范围（如 [0,1]→[0,257]）
            3) 在整型域完成加权乘法，避免 float64 中间数组
    """

    def __init__(self, int_weight: bool = False, **kwargs) -> None:
        self.result = None
        self.shape_check = True
        self.int_weight = int_weight
        # 由第一帧自动设置（记录原始 dtype 用于 int_weight 放缩）
        self._source_dtype: Optional[np.dtype] = None

    def merge(self,
              new_img: np.ndarray,
              weight: Optional[Union[float, NDArray]] = None):
        """合并新图像到堆叠结果。

        Args:
            new_img: np.ndarray 图像。
            weight:  浮点权重 (0-1 范围)。Merger 根据 int_weight 开关
                     自动决定是否转为整型放缩权重。
        """
        raw = new_img
        if self._source_dtype is None:
            self._source_dtype = raw.dtype

        # ── int_weight 放缩：提升 dtype + 转换权重 ──
        if self.int_weight and weight is not None and self._source_dtype is not None:
            raw, weight = self._apply_int_weight(raw, weight)

        # 预处理 + 加权（子类各自决定如何施加权重）
        processed = self._pre_process(raw, weight)

        if self.result is None:
            self.result = processed
        else:
            if self.shape_check:
                assert self.result.shape == processed.shape, (
                    f"{self.__class__.__name__} failed to merge new image. "
                    f"It should have the same shape as merged image "
                    f"{self.result.shape}, but {processed.shape} got.")
            self.result = self._merge(self.result, processed)

    def _apply_int_weight(
        self, raw: np.ndarray,
        weight: Union[float,
                      NDArray]) -> tuple[np.ndarray, Union[int, NDArray]]:
        """将 float 权重映射到整型域，同时 upscale 图像 dtype。

        规则：
            source_dtype 在 DTYPE_UPSCALE_MAP 中时，
            图像 upscale 一级（如 uint8→uint16），
            权重从 [0,1] 映射到 [0, 256^1+1] 的整型范围。
        """
        src = self._source_dtype
        if src in DTYPE_UPSCALE_MAP and DTYPE_UPSCALE_MAP[src] != float:
            upscaled_dtype = DTYPE_UPSCALE_MAP[src]
            src_level = DTYPE_LEVEL.get(src, 0)
            up_level = DTYPE_LEVEL.get(upscaled_dtype, src_level)
            diff = up_level - src_level
            if diff > 0:
                scale = _SCALE_BASE**diff + 1
                raw = raw.astype(upscaled_dtype)
                if isinstance(weight, np.ndarray):
                    weight = np.array(weight * scale, dtype=upscaled_dtype)
                else:
                    weight = int(round(weight * scale))
        return raw, weight

    def clear(self):
        self.result = None

    @abstractmethod
    def _merge(self, base_img, new_img):
        raise NotImplementedError

    def _pre_process(self, img: NDArray, weight=None) -> Any:
        """预处理 + 加权。子类可覆写以实现特定加权逻辑。

        默认实现：直接对 ndarray 乘以权重（适用于 Max/Min）。
        """
        if weight is not None:
            return img * weight
        return img

    @property
    def merged_image(self) -> Union[np.ndarray, Any, None]:
        """返回合并结果（裸 ndarray）。"""
        return self.result


class MaxMerger(BaseMerger):

    def _merge(self, base_img, new_img):
        return np.maximum(base_img, new_img)


class MinMerger(BaseMerger):

    def _merge(self, base_img, new_img):
        return np.minimum(base_img, new_img)


class MeanMerger(BaseMerger):

    def _merge(self, base_img, new_img: FastGaussianParam):
        return base_img + new_img

    def _pre_process(self, img: NDArray, weight=None) -> FastGaussianParam:
        fgp = FastGaussianParam(img, source_dtype=img.dtype)
        if weight is not None:
            fgp = fgp * weight
        return fgp

    @property
    def merged_image(self) -> Union[FloatImage, None]:
        """从 FastGaussianParam 提取均值数组。"""
        if self.result is None:
            return None
        return FloatImage(self.result.mu, dtype=self._source_dtype)


class SigmaClippingMerger(MeanMerger):
    """带有N*Sigma拒绝平均值叠加Merger。

    该进程叠加的是被拒绝的叠加结果。取值和输出时需要转换。

    Args:
        BaseMergerSubprocess (_type_): _description_
    """

    def __init__(self, ref_img: FastGaussianParam, rej_high: float,
                 rej_low: float, **kwargs) -> None:
        # TODO: 迭代加速（对已收敛的区域取mask）？
        self.ref_img = ref_img
        ref_mu = ref_img.mu
        ref_std = np.sqrt(ref_img.var)
        rej_dtype = ref_img.source_dtype
        self.rej_high_img = np.array(
            np.floor(ref_mu + ref_std * rej_high).clip(
                min=0, max=DTYPE_MAX_VALUE[rej_dtype]),
            dtype=rej_dtype)
        self.rej_low_img = np.array(np.ceil(ref_mu - ref_std * rej_low).clip(
            min=0, max=DTYPE_MAX_VALUE[rej_dtype]),
                                    dtype=rej_dtype)
        super().__init__()

    def _pre_process(self, img: np.ndarray, weight=None) -> FastGaussianParam:
        new_img = FastGaussianParam(img, source_dtype=img.dtype)
        new_img.mask((img > self.rej_high_img) | (img < self.rej_low_img))
        if weight is not None:
            new_img = new_img * weight
        return new_img


class HuberWeightedMerger(BaseMerger):
    """Huber 加权均值合并器（Phase 2 专用）。

    接收外部提供的全局 mean/std（来自 Phase 1 的 MeanMerger），
    对每帧计算 Huber 权重后累加到 HuberMeanParam。

    用法与 SigmaClippingMerger 对称：
        # Phase 1
        mean_merger = MeanMerger(int_weight=...)
        for frame in frames: mean_merger.merge(frame, weight)
        fgp = mean_merger.result  # FastGaussianParam

        # Phase 2
        huber_merger = HuberWeightedMerger(ref_stats=fgp, huber_c=1.345)
        for frame in frames: huber_merger.merge(frame, weight)
        result = huber_merger.merged_image  # FloatImage

    Args:
        ref_stats: Phase 1 的 FastGaussianParam（提供 mean/std）。
        huber_c: Huber 常数。默认 1.345（正态分布下 95% 渐近效率）。
    """

    def __init__(self, ref_stats: FastGaussianParam,
                 huber_c: float = 1.345, **kwargs) -> None:
        super().__init__(**kwargs)
        self.ref_stats = ref_stats
        self.huber_c = huber_c
        self._ref_mean = ref_stats.mu.astype(np.float32)
        self._ref_std = np.sqrt(
            np.maximum(ref_stats.var, 0)).astype(np.float32)

    def _pre_process(self, img: np.ndarray, weight=None) -> HuberMeanParam:
        """计算 Huber 权重，构造单帧的 HuberMeanParam。"""
        r = (img.astype(np.float32) - self._ref_mean) / (self._ref_std + 1e-10)
        abs_r = np.abs(r)
        huber_w = np.where(
            abs_r <= self.huber_c,
            np.ones_like(abs_r, dtype=np.float32),
            (self.huber_c / (abs_r + 1e-10)).astype(np.float32),
        )
        if weight is not None:
            huber_w = huber_w * weight

        w_sum = (img * huber_w).astype(np.float64)
        w_total = huber_w.astype(np.float64)
        return HuberMeanParam(
            weighted_sum=w_sum,
            weight_total=w_total,
            source_dtype=img.dtype,
        )

    def _merge(self, base_img: HuberMeanParam,
               new_img: HuberMeanParam) -> HuberMeanParam:
        return base_img + new_img

    @property
    def merged_image(self) -> Optional[FloatImage]:
        if self.result is None:
            return None
        return FloatImage(self.result.mu, dtype=self._source_dtype)
