""" merger管理所有合并器类型。该类型定义不同堆栈模式时的后处理和合并逻辑，并暂存叠加结果。
"""
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray

from .utils import DTYPE_MAX_VALUE, FastGaussianParam


class BaseMerger(metaclass=ABCMeta):

    def __init__(self, **kwargs) -> None:
        self.result = None
        self.shape_check = True

    def merge(self, new_img, weight: Union[float, NDArray] = 1.0):
        """ `merge` should be called when combining new image to the stack.

        If `shape_check` is true, it will first do shape-checking to make sure that they can be merged.
        This requires image (or other data) have `shape` attributes.

        Args:
            new_img (Any): the new image.
            weight (float or np.ndarray): weight to apply to the new image. Default is 1.
        """
        # Apply weight to the new image
        if weight != 1:
            new_img = new_img * weight

        if self.result is None:
            self.result = new_img
        else:
            if self.shape_check:
                assert self.result.shape == new_img.shape, (
                    f"{self.__class__.__name__} failed to merge new image. It should have the same shape as "
                    +
                    f"merged image {self.result.shape}, but {new_img.shape} got."
                )
            self.result = self._merge(self.result, new_img)

    def clear(self):
        self.result = None

    @abstractmethod
    def _merge(self, base_img, new_img):
        raise NotImplementedError

    def post_process(self, img: np.ndarray):
        # no post-processing by default.
        return img

    def upscale(self):
        raise NotImplementedError(
            "this merger does not support `upscale` method.")

    def merge_array(self, array: np.ndarray, **kwargs) -> np.ndarray:
        """ `merge_array` should be called when handling input array in shape (n, h, w, c).
        It merges n images to the result (h, w, c) in one step.

        This requires all data in memory.

        Args:
            array (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """
        raise NotImplementedError

    @property
    def merged_image(self):
        return self.result


class MaxMerger(BaseMerger):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _merge(self, base_img, new_img):
        return np.max([base_img, new_img], axis=0)

    def post_process(self, img: np.ndarray) -> np.ndarray:
        # No post-processing needed since weight is applied during merge
        return img

    def merge_array(self, array: np.ndarray, weights: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        if weights is not None:
            # Apply weights to each image in the array
            weighted_array = array * weights[:, None, None, None]
            return np.max(weighted_array, axis=0)
        else:
            return np.max(array, axis=0)


class MinMerger(BaseMerger):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _merge(self, base_img, new_img):
        return np.min([base_img, new_img], axis=0)

    def post_process(self, img: np.ndarray) -> np.ndarray:
        # No post-processing needed since weight is applied during merge
        return img

    def merge_array(self, array: np.ndarray, weights: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        if weights is not None:
            # Apply weights to each image in the array
            weighted_array = array * weights[:, None, None, None]
            return np.min(weighted_array, axis=0)
        else:
            return np.min(array, axis=0)


class MeanMerger(BaseMerger):

    def _merge(self, base_img, new_img: FastGaussianParam):
        return base_img + new_img

    def post_process(self, img: np.ndarray):
        return FastGaussianParam(img)

    def upscale(self):
        if self.result is None:
            super().upscale()
        else:
            self.result.upscale()

    def merge_array(self, array: np.ndarray, weights: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        if weights is not None:
            # Apply weights to each image in the array
            weighted_array = array * weights[:, None, None, None]
            return np.mean(weighted_array, axis=0)
        else:
            return np.mean(array, axis=0)


class SigmaClippingMerger(MeanMerger):
    """带有N*Sigma拒绝平均值叠加Merger。

    该进程叠加的是被拒绝的叠加结果。取值和输出时需要转换。

    Args:
        BaseMergerSubprocess (_type_): _description_
    """

    def __init__(self, ref_img: FastGaussianParam, rej_high: float,
                 rej_low: float, **kwargs) -> None:
        # TODO:
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
                     img: np.ndarray) -> FastGaussianParam:
        new_img = FastGaussianParam(img)
        new_img.mask((img > self.rej_high_img) | (img < self.rej_low_img))
        return new_img

    def merge_array(self, array: np.ndarray, weights: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        # not tested.
        if weights is not None:
            # Apply weights to each image in the array
            weighted_array = array * weights[:, None, None, None]
        else:
            weighted_array = array

        array_mask = (weighted_array > self.rej_high_img[None, ...]) | (
            weighted_array < self.rej_low_img[None, ...])
        array_num = np.sum(np.array(array_mask, dtype=np.uint16), axis=0)
        return np.sum(weighted_array * array_mask, axis=0) / array_num


class DataMerger(BaseMerger):
    """用于创建缓存的Merger。保存所有原始数据。
    返回包含顺序id的tuple，不支持直接其他Merger进一步合并，
    因此需要和OrderedDataMerger搭配使用。

    Args:
        BaseMerger (_type_): _description_
    """

    def __init__(self, **kwargs) -> None:
        self.proc_id = kwargs["proc_id"]
        self.result = None
        self.shape_check = False

    def _merge(self, base_img, new_img: np.ndarray):
        return np.concatenate([base_img, new_img], axis=0)

    def post_process(self, img: np.ndarray):
        # convert to [1, h, w, c]
        return img[None, ...]

    @property
    def merged_image(self) -> tuple:
        return (self.proc_id, self.result)


class OrderedDataMerger(BaseMerger):
    """用于汇总缓存的Merger。
    接受tuple而非其他可直接加和的Merger，因此需要和 DataMerger 搭配使用。

    Args:
        BaseMerger (_type_): _description_
    """

    def __init__(self, **kwargs) -> None:
        self.result = []
        self.shape_check = False

    def _merge(self, base_img: list, new_img: tuple):
        base_img.append(new_img)
        return base_img

    @property
    def merged_image(self) -> np.ndarray:
        return np.concatenate(
            [array for (_, array) in sorted(self.result, key=lambda x: x[0])],
            axis=0)
