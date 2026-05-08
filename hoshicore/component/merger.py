""" merger管理所有合并器类型。该类型定义不同堆栈模式时的后处理和合并逻辑，并暂存叠加结果。
"""
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Optional, Union, Any, cast

import numpy as np
from numpy.typing import NDArray

from .data_container import (DTYPE_LEVEL, DTYPE_MAX_VALUE, DTYPE_UPSCALE_MAP,
                             _SCALE_BASE, FloatImage, FastGaussianParam,
                             HuberMeanParam)
from .numba_kernels import (fgp_masked_mean_merge, fgp_mean_merge,
                            fgp_weighted_mean_merge, sigma_clip_fused_merge,
                            sigma_clip_fused_masked_merge)


def _accum_dtypes(
    source_dtype: np.dtype,
    int_weight: bool = False,
) -> tuple[np.dtype, np.dtype, np.dtype]:
    """根据源图像 dtype 确定 FGP 累加器的 dtype。

    Returns:
        (sum_mu_dtype, square_sum_dtype, n_dtype)
    """
    dt = np.dtype(source_dtype)
    if int_weight and dt in DTYPE_UPSCALE_MAP:
        dt = np.dtype(DTYPE_UPSCALE_MAP[dt])

    sum_dt = np.dtype(DTYPE_UPSCALE_MAP[dt]
                      ) if dt in DTYPE_UPSCALE_MAP else np.dtype("float64")
    sq_dt = np.dtype(DTYPE_UPSCALE_MAP[sum_dt]
                     ) if sum_dt in DTYPE_UPSCALE_MAP else np.dtype("float64")

    n_dt = np.dtype("uint32") if int_weight else np.dtype("uint16")
    return sum_dt, sq_dt, n_dt


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

    def __init__(self, int_weight: bool = False, **kwargs) -> None:
        super().__init__(int_weight=int_weight, **kwargs)
        self._sum_mu: Optional[np.ndarray] = None
        self._square_sum: Optional[np.ndarray] = None
        self._n: Optional[np.ndarray] = None

    def merge(self,
              new_img: np.ndarray,
              weight: Optional[Union[float, NDArray]] = None,
              spatial_mask: Optional[np.ndarray] = None):
        raw = new_img
        if self._source_dtype is None:
            self._source_dtype = raw.dtype

        if self.int_weight and weight is not None and self._source_dtype is not None:
            raw, weight = self._apply_int_weight(raw, weight)

        if self._sum_mu is None:
            sum_dt, sq_dt, n_dt = _accum_dtypes(self._source_dtype,
                                                self.int_weight)
            self._sum_mu = np.zeros(raw.shape, dtype=sum_dt)
            self._square_sum = np.zeros(raw.shape, dtype=sq_dt)
            self._n = np.zeros(raw.shape, dtype=n_dt)

        if spatial_mask is not None:
            if weight is not None:
                raise NotImplementedError(
                    "spatial_mask + weight combination is not yet supported")
            fgp_masked_mean_merge(raw, spatial_mask, self._sum_mu,
                                  self._square_sum, self._n)
        elif weight is not None and isinstance(weight, (int, np.integer)):
            fgp_weighted_mean_merge(raw, weight, self._sum_mu,
                                    self._square_sum, self._n)
        else:
            if weight is not None:
                raw = (raw * weight).astype(raw.dtype)
            fgp_mean_merge(raw, self._sum_mu, self._square_sum, self._n)

    @property
    def result(self):
        if self._sum_mu is None:
            return None
        return FastGaussianParam(
            sum_mu=self._sum_mu,
            square_sum=self._square_sum,
            n=self._n,
            ddof=1,
            source_dtype=self._source_dtype,
            inplace_calc=True,
        )

    @result.setter
    def result(self, value):
        if value is None:
            self._sum_mu = None
            self._square_sum = None
            self._n = None
        elif isinstance(value, FastGaussianParam):
            self._sum_mu = value.sum_mu
            self._square_sum = value.square_sum
            self._n = value.n

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
        r = self.result
        if r is None:
            return None
        return FloatImage(r.mu, dtype=self._source_dtype)

    def clear(self):
        super().clear()
        self._sum_mu = None
        self._square_sum = None
        self._n = None


class SigmaClippingMerger(MeanMerger):
    """带有N*Sigma拒绝平均值叠加Merger。

    使用 Numba fused kernel 将 clip 判断 + rejected FGP 累加
    融合为单次遍历，消除 ~500MB 临时数组和 ~10 次全图遍历。

    累加的是被拒绝像素的统计量。最终结果 = total_fgp - rejected_fgp。
    """

    def __init__(self, ref_img: FastGaussianParam, rej_high: float,
                 rej_low: float, **kwargs) -> None:
        super().__init__()
        # n=0 的位置（被 mask 排除）mu/var 均无意义，归零以避免 NaN 传播
        valid = ref_img.n > 0
        ref_mu = np.where(valid, ref_img.mu, 0)
        ref_std = np.where(valid, np.sqrt(np.maximum(ref_img.var, 0)), 0)
        rej_dtype = ref_img.source_dtype
        self.rej_high_img = np.array(
            np.floor(ref_mu + ref_std * rej_high).clip(
                min=0, max=DTYPE_MAX_VALUE[rej_dtype]),
            dtype=rej_dtype)
        self.rej_low_img = np.array(np.ceil(ref_mu - ref_std * rej_low).clip(
            min=0, max=DTYPE_MAX_VALUE[rej_dtype]),
                                    dtype=rej_dtype)
        self._sum_mu: Optional[np.ndarray] = None
        self._square_sum: Optional[np.ndarray] = None
        self._n: Optional[np.ndarray] = None

    def merge(self,
              new_img: np.ndarray,
              weight: Optional[Union[float, NDArray]] = None,
              spatial_mask: Optional[np.ndarray] = None):
        raw = new_img
        if self._source_dtype is None:
            self._source_dtype = raw.dtype

        img = raw

        if self._sum_mu is None:
            sum_dt, sq_dt, _ = _accum_dtypes(self._source_dtype, False)
            n_dt = np.dtype("uint16")
            self._sum_mu = np.zeros(img.shape, dtype=sum_dt)
            self._square_sum = np.zeros(img.shape, dtype=sq_dt)
            self._n = np.zeros(img.shape, dtype=n_dt)

        if spatial_mask is not None:
            sigma_clip_fused_masked_merge(img, spatial_mask, self.rej_high_img,
                                          self.rej_low_img, self._sum_mu,
                                          self._square_sum, self._n)
        else:
            sigma_clip_fused_merge(img, self.rej_high_img, self.rej_low_img,
                                   self._sum_mu, self._square_sum, self._n)


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

    def __init__(self,
                 ref_stats: FastGaussianParam,
                 huber_c: float = 1.345,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.huber_c = huber_c
        self._ref_mean = ref_stats.mu.astype(np.float32)
        self._ref_std = np.sqrt(np.maximum(ref_stats.var,
                                           0)).astype(np.float32)

    def _merge(self, base_img: HuberMeanParam,
               new_img: HuberMeanParam) -> HuberMeanParam:
        return base_img + new_img

    @property
    def merged_image(self) -> Union[FloatImage, None]:
        if self.result is None:
            return None
        return FloatImage(self.result.mu,
                          dtype=self._source_dtype or np.dtype("float64"))

    def _pre_process(self, img: np.ndarray, weight=None) -> HuberMeanParam:
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
