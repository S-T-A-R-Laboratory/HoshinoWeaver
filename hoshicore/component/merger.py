""" merger管理所有合并器类型。该类型定义不同堆栈模式时的后处理和合并逻辑，并暂存叠加结果。
"""
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Optional, Union, Any, cast

import numpy as np
from numpy.typing import NDArray

from .utils import DTYPE_MAX_VALUE, FastGaussianParam


class BaseMerger(metaclass=ABCMeta):

    def __init__(self, **kwargs) -> None:
        self.result = None
        self.shape_check = True

    def merge(self, new_img, weight: Optional[Union[float, NDArray]] = None):
        """ `merge` should be called when combining new image to the stack.

        If `shape_check` is true, it will first do shape-checking to make sure that they can be merged.
        This requires image (or other data) have `shape` attributes.

        Args:
            new_img (Any): the new image.
            weight (float or np.ndarray): weight to apply to the new image. Default is 1.
        """
        # 预处理（如转换为 FastGaussianParam）
        new_img = self._pre_process(new_img)
        # Apply weight to the new image
        if weight is not None:
            new_img = new_img * weight

        if self.result is None:
            self.result = new_img
        else:
            if self.shape_check:
                assert self.result.shape == new_img.shape, (
                    f"{self.__class__.__name__} failed to merge new image. It should have the same shape as "
                    f"merged image {self.result.shape}, but {new_img.shape} got."
                )
            self.result = self._merge(self.result, new_img)

    def clear(self):
        self.result = None

    @abstractmethod
    def _merge(self, base_img, new_img):
        raise NotImplementedError

    def _pre_process(self, img: NDArray)->Any:
        # no post-processing by default.
        return img

    def upscale(self):
        raise NotImplementedError(
            "this merger does not support `upscale` method.")

    @property
    def merged_image(self):
        return self.result


class MaxMerger(BaseMerger):

    def _merge(self, base_img, new_img):
        return np.max([base_img, new_img], axis=0)


class MinMerger(BaseMerger):

    def _merge(self, base_img, new_img):
        return np.min([base_img, new_img], axis=0)


class MeanMerger(BaseMerger):

    def _merge(self, base_img, new_img: FastGaussianParam):
        return base_img + new_img

    def _pre_process(self, img: NDArray):
        return FastGaussianParam(img)

    def upscale(self):
        if self.result is None:
            super().upscale()
        else:
            self.result = cast(FastGaussianParam, self.result)
            self.result.upscale()


class SigmaClippingMerger(MeanMerger):
    """带有N*Sigma拒绝平均值叠加Merger。

    该进程叠加的是被拒绝的叠加结果。取值和输出时需要转换。

    Args:
        BaseMergerSubprocess (_type_): _description_
    """

    def __init__(self, ref_img: FastGaussianParam, rej_high: float,
                 rej_low: float, **kwargs) -> None:
        # TODO: 迭代加速（对已收敛的区域取mask）？
        ref_mu = ref_img.mu
        ref_std = np.sqrt(ref_img.var)
        rej_dtype = ref_img.sum_mu.dtype
        self.rej_high_img = np.array(
            np.floor(ref_mu + ref_std * rej_high).clip(
                min=0, max=DTYPE_MAX_VALUE[rej_dtype]),
            dtype=rej_dtype)
        self.rej_low_img = np.array(np.ceil(ref_mu - ref_std * rej_low).clip(
            min=0, max=DTYPE_MAX_VALUE[rej_dtype]),
                                    dtype=rej_dtype)
        super().__init__()

    def _pre_process(self, img: np.ndarray) -> FastGaussianParam:
        new_img = FastGaussianParam(img)
        new_img.mask((img > self.rej_high_img) | (img < self.rej_low_img))
        return new_img