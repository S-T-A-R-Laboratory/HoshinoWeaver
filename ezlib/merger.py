""" merger管理所有合并器类型。该类型定义不同堆栈模式时的后处理和合并逻辑，并暂存叠加结果。
"""
from abc import ABCMeta, abstractmethod
from typing import Optional, Union

import numpy as np

from .utils import FastGaussianParam, DTYPE_MAX_VALUE


class BaseMerger(metaclass=ABCMeta):

    def __init__(self, **kwargs) -> None:
        self.merged_image = None

    def merge(self, new_img):
        if self.merged_image is None:
            self.merged_image = new_img
        else:
            self.merged_image = self._merge(self.merged_image, new_img)

    @abstractmethod
    def _merge(self, base_img, new_img):
        raise NotImplementedError

    @abstractmethod
    def post_process(self, img: np.ndarray, index: Optional[int] = None):
        raise NotImplementedError

    def upscale(self):
        raise NotImplementedError(
            "this merger does not support `upscale` method.")


class MaxMerger(BaseMerger):

    def __init__(self,
                 weight_list: Union[list, np.ndarray, None] = None,
                 **kwargs) -> None:
        self.weight_list = weight_list
        super().__init__(**kwargs)

    def _merge(self, base_img, new_img):
        return np.max([base_img, new_img], axis=0)

    def post_process(self,
                     img: np.ndarray,
                     index: Optional[int] = None) -> np.ndarray:
        if index is not None and self.weight_list is not None:
            assert 0 <= index < len(
                self.weight_list
            ), f"Invalid index {index} encountered. Expect 0<=index<={len(self.weight_list)}."
            return img * self.weight_list[index]
        else:
            return img


class MinMerger(BaseMerger):

    def __init__(self,
                 weight_list: Union[list, np.ndarray, None] = None,
                 **kwargs) -> None:
        self.weight_list = weight_list
        super().__init__(**kwargs)

    def _merge(self, base_img, new_img):
        return np.min([base_img, new_img], axis=0)

    def post_process(self,
                     img: np.ndarray,
                     index: Optional[int] = None) -> np.ndarray:
        if index is not None and self.weight_list is not None:
            assert 0 <= index < len(
                self.weight_list
            ), f"Invalid index {index} encountered. Expect 0<=index<={len(self.weight_list)}."
            return img * self.weight_list[index]
        else:
            return img


class MeanMerger(BaseMerger):

    def _merge(self, base_img, new_img: FastGaussianParam):
        return base_img + new_img

    def post_process(self, img: np.ndarray, index: Optional[int] = None):
        return FastGaussianParam(img)

    def upscale(self):
        if self.merged_image is None:
            super().upscale()
        else:
            self.merged_image.upscale()


class SigmaClippingMerger(MeanMerger):
    """带有N*Sigma拒绝平均值叠加Merger。

    该进程叠加的是被拒绝的叠加结果。取值和输出时需要转换。

    Args:
        BaseMergerSubprocess (_type_): _description_
    """

    def __init__(self, ref_img: FastGaussianParam, rej_high: float,
                 rej_low: float, **kwargs) -> None:
        # TODO
        # 迭代加速（对已收敛的区域取mask）？
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

    def post_process(self,
                     img: np.ndarray,
                     index: Optional[int] = None) -> FastGaussianParam:
        new_img = FastGaussianParam(img)
        new_img.mask((img > self.rej_high_img) | (img < self.rej_low_img))
        return new_img


class CacheMerger(BaseMerger):
    """用于创建缓存的Merger。保存所有原始数据

    Args:
        BaseMerger (_type_): _description_
    """

    def _merge(self, base_img, new_img: np.ndarray):
        return np.concatenate([base_img, new_img], axis=0)

    def post_process(self, img: np.ndarray, index: Optional[int] = None):
        # convert to [1, h, w, c]
        return img[None, ...]
