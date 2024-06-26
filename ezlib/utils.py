import multiprocessing as mp
import time
from loguru import logger
from math import floor, sqrt
from typing import Callable, Union

import numpy as np
from PIL.ExifTags import TAGS

DTYPE_UPSCALE_MAP = {
    np.dtype('uint8'): np.dtype('uint16'),
    np.dtype('uint16'): np.dtype('uint32'),
    np.dtype('uint32'): np.dtype('uint64'),
    np.dtype('uint64'): float
}

SUPPORT_COLOR_SPACE = ["Adobe RGB", "ProPhoto RGB", "sRGB"]
COMMON_SUFFIX = ["tiff", "tif", "jpg", "png", "jpeg"]
NOT_RECOM_SUFFIX = ["bmp", "gif", "fits"]
RAW_SUFFIX = ["cr2", "cr3", "arw", "nef", "dng"]
SUPPORT_BITS = [8, 16]


def is_support_format(fname) -> bool:
    # suffix check and warning raising
    suffix = fname.split(".")[-1].lower()
    return ((suffix in COMMON_SUFFIX) or (suffix in NOT_RECOM_SUFFIX)
            or (suffix in RAW_SUFFIX))


def get_resize(tgt_wh: int, raw_wh: Union[list, tuple]):
    w, h = raw_wh
    tgt_wh_list = [tgt_wh, -1] if w > h else [-1, tgt_wh]
    idn = 0 if tgt_wh_list[0] <= 0 else 1
    idx = 1 - idn
    tgt_wh_list[idn] = int(raw_wh[idn] * tgt_wh_list[idx] / raw_wh[idx])
    return tgt_wh_list


def time_cost_warpper(func: Callable) -> Callable:
    """A decorator that supports to record time cost of the given function.

    Args:
        func (Callable): _description_

    Returns:
        Callable: _description_
    """

    def do_func(*args, **kwargs):
        t0 = time.time()
        res = func(*args, **kwargs)
        logger.info(f"{func.__name__} Time Cost: {(time.time()-t0):.2f}s.")
        return res

    return do_func


def get_mp_num(tot_num: int) -> tuple[int, float]:
    """
    设置处理器使用数目，在不超出处理器数目限制的情况下，尽可能使每个处理器叠加sqrt(N)张图像
    推导：n 图像分 m 组叠加，时间开销近似为 [n/m]+m ；min([n/m]+m)-> m取得sqrt(N)
    """
    mp_num = min(floor(sqrt(tot_num)), mp.cpu_count() - 1)
    sub_length = tot_num / mp_num
    return mp_num, sub_length


class GaussianParam(object):
    """

    Args:
        object (_type_): _description_
    """

    def __init__(self,
                 mu,
                 var: Union[np.ndarray, int, float] = 0,
                 n: int = 1,
                 ddof: int = 1):
        self.mu = mu
        self.var = var
        self.n = n
        self.ddof = ddof

    def __add__(self, g2):
        g1 = self
        assert isinstance(g2, GaussianParam), "unacceptable object"
        assert g1.ddof == g2.ddof, "unmatched var calculation!"
        ddof = g1.ddof
        new_mu = (g1.mu * g1.n + g2.mu * g2.n) / (g1.n + g2.n)
        new_var = ((g1.n - ddof) * g1.var + g1.n * np.square(g1.mu) +
                   (g2.n - ddof) * g2.var + g2.n * np.square(g2.mu) -
                   (g1.n + g2.n) * np.square(new_mu)) / (g1.n + g2.n - ddof)
        return GaussianParam(mu=new_mu, var=new_var, n=g1.n + g2.n, ddof=ddof)

    def __sub__(self, g2):
        g1 = self
        assert isinstance(g2, GaussianParam), "unacceptable object"
        assert g1.ddof == g2.ddof, "unmatched var calculation!"
        assert g1.n > g2.n, "generate n<0 fistribution!"
        ddof = g1.ddof
        new_mu = (g1.mu * g1.n - g2.mu * g2.n) / (g1.n - g2.n)
        new_var = ((g1.n - ddof) * g1.var + g1.n * np.square(g1.mu) -
                   (g2.n - ddof) * g2.var - g2.n * np.square(g2.mu) -
                   (g1.n - g2.n) * np.square(new_mu)) / (g1.n - g2.n - ddof)
        return GaussianParam(mu=new_mu, var=new_var, n=g1.n - g2.n, ddof=ddof)


class FastGaussianParam(object):
    """
    GaussianParam, but faster. 通过INT量化+优化数据储存提速。缺点是只支持INT型。
    Args:
        object (_type_): _description_
    TODO: 验证量化带来的精度损失是否可接受
    TODO: 优化接口，和普通版本统一；进一步支持float类型
    （理论可以通过后置除法+提高数据范围提高精度）
    """

    def __init__(self,
                 sum_mu,
                 square_num: Union[np.ndarray, int, float, None] = None,
                 n: int = 1,
                 ddof: int = 1):
        self.sum_mu = sum_mu
        if square_num is not None:
            self.square_sum = square_num
        else:
            sq_dtype = DTYPE_UPSCALE_MAP[self.sum_mu.dtype] if self.sum_mu.dtype in DTYPE_UPSCALE_MAP else float
            print(f"use {sq_dtype} as var-dtype.")
            self.square_sum = np.square(sum_mu, dtype = sq_dtype)
        self.n = n
        self.ddof = ddof

    @property
    def mu(self):
        return np.round(self.sum_mu / self.n)

    @property
    def var(self):
        # TODO: This is not validated.
        #D(X) = ∑((X-E(X))^2)/(n-ddof)
        #
        #D(X) = (∑X^2 + nE(X)^2 - 2∑XE(X))/(n-ddof)
        #    = ∑X^2 - nE(X)^2 /(n-ddof)

        return self.square_sum - self.n * np.square(self.sum_mu) / (self.n - self.ddof)
        #return self.square_sum - np.square(self.sum_mu) / (self.n - self.ddof)

    def __add__(self, g2):
        g1 = self
        assert isinstance(g2, FastGaussianParam), "unacceptable object"
        assert g1.ddof == g2.ddof, "unmatched var calculation!"
        return FastGaussianParam(sum_mu=g1.sum_mu + g2.sum_mu,
                                 square_num=g1.square_sum + g2.square_sum,
                                 n=g1.n + g2.n,
                                 ddof=g1.ddof)

    def __sub__(self, g2):
        g1 = self
        assert isinstance(g2, FastGaussianParam), "unacceptable object"
        assert g1.ddof == g2.ddof, "unmatched var calculation!"
        assert g1.n > g2.n, "generate n<0 fistribution!"
        return FastGaussianParam(sum_mu=g1.sum_mu - g2.sum_mu,
                                 square_num=g1.square_sum - g2.square_sum,
                                 n=g1.n - g2.n,
                                 ddof=g1.ddof)


def test_GaussianParam():
    tot_num = 40
    series = np.array(np.random.rand(tot_num)*32767,dtype=np.uint32)
    print(series)
    base = FastGaussianParam(series[0])
    for i in range(1, tot_num):
        print("n =", i)
        add = FastGaussianParam(series[i])
        base = base + add
        print(base.mu, np.mean(series[:i + 1]))
        assert np.abs(base.mu -
                      np.mean(series[:i + 1])) < 1e-8, (base.mu,
                                                        np.mean(series[:i +
                                                                       1]))
        assert np.abs(base.var -
                      np.var(series[:i + 1], ddof=1)) < 1e-8, (base.var,
                                                               np.var(
                                                                   series[:i +
                                                                          1],
                                                                   ddof=1),
                                                               series)
        assert base.n == i + 1
        assert base.ddof == 1