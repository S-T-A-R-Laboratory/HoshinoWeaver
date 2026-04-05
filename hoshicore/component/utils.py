from __future__ import annotations

import multiprocessing as mp
import sys
import time
from functools import wraps
from math import floor, log, sqrt
from typing import Callable, Optional, Union

import numpy as np
import psutil
import tifffile
from easydict import EasyDict
from loguru import logger
from PIL.ExifTags import TAGS

DTYPE_UPSCALE_MAP = {
    np.dtype('uint8'): np.dtype('uint16'),
    np.dtype('uint16'): np.dtype('uint32'),
    np.dtype('uint32'): np.dtype('uint64'),
    np.dtype('uint64'): float
}

BITS2DTYPE = {
    8: np.dtype('uint8'),
    16: np.dtype('uint16'),
    32: np.dtype('uint32')
}

DTYPE_REVERSE_MAP = {
    np.dtype('uint8'): 1,
    np.dtype('uint16'): 2,
    np.dtype('uint32'): 4,
    np.dtype('uint64'): 8,
    float: 8
}

# TODO: All these things will be fixed into one obj in the future
DTYPE_NUM2TYPE = [
    np.dtype('uint8'),
    np.dtype('uint16'),
    np.dtype('uint32'),
    np.dtype('uint64'),
]

DTYPE_MAX_VALUE = {
    np.dtype('uint8'): np.iinfo(np.uint8).max,
    np.dtype('uint16'): np.iinfo(np.uint16).max,
    np.dtype('uint32'): np.iinfo(np.uint32).max,
    np.dtype('uint64'): np.iinfo(np.uint64).max,
}

ERROR_NAME_MAPPING = {
    "MemoryError": "内存不足",
    "AssertionError": "图像尺寸不统一",
    "KeyboardInterrupt": "人为终止"
}

SAME_SUFFIX_MAPPING = {"tiff": "tif", "jpeg": "jpg"}

SUPPORT_COLOR_SPACE = ["Adobe RGB", "ProPhoto RGB", "sRGB"]
COMMON_SUFFIX = ["tiff", "tif", "jpg", "png", "jpeg"]
NOT_RECOM_SUFFIX = ["bmp", "gif", "fits"]
RAW_SUFFIX = ["cr2", "cr3", "arw", "nef", "dng", "rw2", "raf"]
SUPPORT_BITS = [8, 16]
MAGIC_NUM = 3

VERSION = "0.5.0"

ORG_NAME = f"STARLab"
SOFTWARE_NAME = f"HoshinoWeaver"


def rdtype_detector(data: np.ndarray) -> Union[np.dtype, type]:
    """For data that real scale does not match dtype, this function returns real scale.

    Args:
        data (np.ndarray): _description_
    """
    if data.dtype == float:
        return float
    if np.max(data) <= DTYPE_MAX_VALUE[np.dtype("uint8")]:
        return np.dtype("uint8")
    if np.max(data) <= DTYPE_MAX_VALUE[np.dtype("uint16")]:
        return np.dtype("uint16")
    if np.max(data) <= DTYPE_MAX_VALUE[np.dtype("uint32")]:
        return np.dtype("uint16")
    if np.max(data) <= DTYPE_MAX_VALUE[np.dtype("uint64")]:
        return np.dtype("uint64")
    raise NotImplementedError("Unrecognized data type.")


def dtype_scaler(raw_type: np.dtype, times: int) -> np.dtype:
    """A simple implementation of dtype_scaler, get up-scaled data-type with given times.
    TODO: update in the future.
    """
    if times >= 0:
        while times > 0 and raw_type != float:
            raw_type = DTYPE_UPSCALE_MAP[raw_type]
            times -= 1
        return raw_type
    else:
        # downscale. For now only uint16->uint8 is used.
        # this will be updated in the future.
        if times == -1 and raw_type == np.dtype("uint16"):
            return np.dtype("uint8")
        else:
            raise NotImplementedError(
                f"not supported dtype scaling time {times}!")


def error_raiser(error, result_queue):
    """A simple error raiser. For subprocessor callback function.

    Args:
        error (Exception): exception

    Raises:
        error: the error that accepts.
    """
    result_queue.put(EasyDict(img=None, err_msg=[
        error,
    ]))


def is_support_format(fname: str) -> bool:
    # suffix check and warning raising
    suffix = fname.split(".")[-1].lower()
    return ((suffix in COMMON_SUFFIX) or (suffix in NOT_RECOM_SUFFIX)
            or (suffix in RAW_SUFFIX))


def get_resize(opt: Optional[str], raw_wh: Union[list, tuple]):
    """
    accept raw_wh in any order. [h, w] is recommended to avoid misuse.
    
    but if opt is given as "1920x1080", it will return in [h, w] order.

    Args:
        opt (Optional[str]): _description_
        raw_wh (Union[list, tuple]): _description_

    Returns:
        _type_: _description_
    """
    if not opt: return None
    # 如果直接以类似"1280x720"的方式指定，则直接返回值
    if "x" in opt and len(opt.split("x")) == 2:
        return list(map(int, opt.split("x")))[::-1]
    tgt_wh = None
    try:
        tgt_wh = int(opt)
    except ValueError as e:
        logger.error(
            f"Got invalid resize option {opt}. Except format like \"1280x720\""
            + " or an int like \"720\" that specify the length.")
        return None
    tgt_wh_list = [tgt_wh, -1] if raw_wh[0] > raw_wh[1] else [-1, tgt_wh]
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

    @wraps(func)
    def do_func(*args, **kwargs):
        t0 = time.time()
        res = func(*args, **kwargs)
        cls_name = ""
        if hasattr(args[0], func.__name__):
            cls_name = args[0].__class__.__name__ + "."
        logger.info(
            f"{cls_name}{func.__name__} time cost: {(time.time()-t0):.2f}s.")
        return res

    return do_func


def get_mp_num(tot_num: int,
               prefer_num: Optional[int] = None) -> tuple[int, float]:
    """
    设置处理器使用数目，在不超出处理器数目限制的情况下，尽可能使每个处理器叠加sqrt(N)张图像
    推导：n 图像分 m 组叠加，时间开销近似为 [n/m]+m ；min([n/m]+m)-> m取得sqrt(N)
    """
    # TODO: 根据内存和图像规格增设限制
    cpu_num = mp.cpu_count()
    if prefer_num:
        mp_num = prefer_num
        if prefer_num > cpu_num:
            logger.warning(
                f"Preferred multiprocessing num ({prefer_num}) is larger " +
                f"than cpu num ({cpu_num})!")
    else:
        psutil.virtual_memory().available
        cpu_num = cpu_num // 4 + (1 if cpu_num <= 8 else 0)
        mp_num = min(floor(sqrt(tot_num)), cpu_num)
    sub_length = tot_num / mp_num
    return mp_num, sub_length


class GaussianParam(object):
    """
    维护np.ndarray的流方差与流均值的原始实现。

    Args:
            mu (np.ndarray): 均值。当以单张图像作为输入时，只用填写该值。
            var (Optional[np.ndarray], optional): 方差。 Defaults to None.
            n (Optional[np.ndarray], optional): 每个位置参与的叠加数。可以通过指定为1以默认全图叠加。 Defaults to None.
            ddof (int, optional): DDOF. Defaults to 1.
            dtype_var (np.dtype, optional): var使用的数据类型. Defaults to np.dtype("float32").
            dtype_n (np.dtype, optional): n使用的数据类型. Defaults to np.dtype("uint16").
        """

    def __init__(self,
                 mu: np.ndarray,
                 var: Optional[np.ndarray] = None,
                 n: Optional[np.ndarray] = None,
                 ddof: int = 1,
                 dtype_var: np.dtype = np.dtype("float32"),
                 dtype_n: np.dtype = np.dtype("int16")):
        self.mu = mu
        self.var = var if var is not None else np.zeros_like(mu,
                                                             dtype=dtype_var)
        self.n = n if n is not None else np.ones_like(self.mu, dtype=dtype_n)
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
    GaussianParam, but faster. 
    通过INT量化+优化数据储存提速，仅在输出时换算为浮点数。
    Streaming mean and variance.
    Args:
        object (_type_): _description_
    TODO: 优化接口，和普通版本统一；进一步支持float类型
    （理论可以通过后置除法+提高数据范围提高精度）
    """

    def __init__(self,
                 sum_mu: np.ndarray,
                 square_num: Optional[np.ndarray] = None,
                 n: Optional[np.ndarray] = None,
                 ddof: int = 1,
                 dtype_n: np.dtype = np.dtype("uint32"),
                 source_dtype: Optional[np.dtype] = None):
        self.sum_mu = sum_mu
        self.source_dtype = source_dtype if source_dtype is not None else sum_mu.dtype
        if square_num is not None:
            self.square_sum = square_num
        else:
            # var默认根据sum_mu构造而成
            sq_dtype = self.get_upscale_dtype_as(self.sum_mu)
            self.square_sum = np.square(sum_mu, dtype=sq_dtype)
        self.n = n if n is not None else np.ones_like(self.sum_mu,
                                                      dtype=dtype_n)
        # 单张图像初始化时（source_dtype = sum_mu.dtype），默认提升一次范围
        if self.sum_mu.dtype == self.source_dtype:
            self.upscale()
        self.ddof = ddof

    @property
    def mu(self) -> np.ndarray:
        return np.round(self.sum_mu / self.n)

    @property
    def var(self) -> np.ndarray:
        #D(X) = ∑((X-E(X))^2)/(n-ddof)
        #     = (∑X^2 - nE(X)^2) /(n-ddof)
        #     = (∑X^2 - (∑X)^2/n) /(n-ddof)
        sum_mu = np.array(self.sum_mu, dtype=self.square_sum.dtype)
        return (self.square_sum - np.square(sum_mu) / self.n) / (self.n -
                                                                 self.ddof)

    def upscale(self):
        upscaled_sum_mu_dtype = self.get_upscale_dtype_as(self.sum_mu)
        upscaled_sum_sq_dtype = self.get_upscale_dtype_as(self.square_sum)
        self.sum_mu = np.array(self.sum_mu, dtype=upscaled_sum_mu_dtype)
        self.square_sum = np.array(self.square_sum,
                                   dtype=upscaled_sum_sq_dtype)

    def get_upscale_dtype_as(self, ref_array: np.ndarray):
        """必要时候提升数据范围
        """
        return DTYPE_UPSCALE_MAP[
            ref_array.dtype] if ref_array.dtype in DTYPE_UPSCALE_MAP else float

    def _safe_add_count(self) -> Union[int, float]:
        """计算当前 dtype 下可安全叠加的最大图像数量。
        
        实践上几乎仅受 sum_mu 限制。
        """
        if self.source_dtype not in DTYPE_MAX_VALUE:
            return float('inf')
        # sum_mu 的限制
        source_max = DTYPE_MAX_VALUE[self.source_dtype]
        sum_limit = DTYPE_MAX_VALUE.get(self.sum_mu.dtype,
                                        float('inf')) // source_max
        # n 的限制
        n_limit = DTYPE_MAX_VALUE.get(self.n.dtype, float('inf'))
        return min(sum_limit, n_limit)

    def apply_zero_var(self, full_img):
        """修复n为0的情况。应用修复。
        
        TODO: 需要长期观测该逻辑。
        """
        zero_pos = (self.n == 0)
        logger.info(f"Zero-mask {np.where(zero_pos)[0].size} pixels.")
        self.n[zero_pos] = full_img.n[zero_pos]
        self.sum_mu[zero_pos] = full_img.sum_mu[zero_pos]
        self.square_sum[zero_pos] = full_img.square_sum[zero_pos]

    def __add__(self, g2):
        g1 = self
        assert isinstance(g2, FastGaussianParam), "unacceptable object"
        assert g1.ddof == g2.ddof, "unmatched var calculation!"

        # 计算累加后的 n
        new_n = g1.n + g2.n
        max_n = new_n.max()

        # 检查是否超过安全叠加数量
        if max_n > g1._safe_add_count():
            g1.upscale()

        # 检查 n 是否溢出（极少发生）
        if g1.n.dtype in DTYPE_MAX_VALUE and max_n > DTYPE_MAX_VALUE[
                g1.n.dtype]:
            if g1.n.dtype in DTYPE_UPSCALE_MAP:
                new_n_dtype = DTYPE_UPSCALE_MAP[g1.n.dtype]
                g1.n = g1.n.astype(new_n_dtype)
                new_n = new_n.astype(new_n_dtype)

        return FastGaussianParam(sum_mu=g1.sum_mu + g2.sum_mu,
                                 square_num=g1.square_sum + g2.square_sum,
                                 n=new_n,
                                 ddof=g1.ddof,
                                 source_dtype=g1.source_dtype)

    def __sub__(self, g2):
        g1 = self
        assert isinstance(g2, FastGaussianParam), "unacceptable object"
        assert g1.ddof == g2.ddof, "unmatched var calculation!"
        assert (g1.n - g2.n).any() >= 0, "generate n<0 fistribution!"
        return FastGaussianParam(sum_mu=g1.sum_mu - g2.sum_mu,
                                 square_num=g1.square_sum - g2.square_sum,
                                 n=g1.n - g2.n,
                                 ddof=g1.ddof,
                                 source_dtype=g1.source_dtype)

    def __mul__(self, weight: Union[float, int, np.ndarray]):
        """重要性加权：每帧以权重 w 贡献到流式统计中。

        语义：该帧的"重要性"为 w，等价于该帧被计入 w 次。
            sum_mu   *= w   (加权求和)
            square_sum *= w  (加权平方和，非 w²)
            n        *= w   (等效帧数)

        由此可得：
            μ_w = Σ(w·x) / Σw
            var_w = (Σ(w·x²) - (Σ(w·x))²/Σw) / (Σw - ddof)
        """
        if isinstance(weight, (int, float, np.ndarray)):
            return FastGaussianParam(
                sum_mu=self.sum_mu * weight,
                square_num=self.square_sum * weight,
                n=self.n * weight,
                ddof=self.ddof,
                source_dtype=self.source_dtype)
        raise NotImplementedError(
            f"Unsupported weight type {type(weight)} for "
            f"multiplication with {self.__class__.__name__}.")

    def __rmul__(self, weight: Union[float, int, np.ndarray]):
        return self.__mul__(weight)

    def mask(self, mask_pos: np.ndarray):
        assert mask_pos.dtype == np.dtype("bool"), "Invalid mask!"
        self.sum_mu *= mask_pos
        self.square_sum *= mask_pos
        self.n = self.n * mask_pos.astype(self.n.dtype)

    @property
    def shape(self):
        return self.sum_mu.shape


def get_scale_x(time: int, base: int = 256):
    return base**time + 1


def init_logger(logger, debug_mode: bool, log_path: Optional[str]):
    """用于初始化Loguru的logger

    Args:
        logger (Logger): _description_
        debug_mode (bool): _description_
        log_path (Optional[str]): _description_
    """
    logger.remove()
    if debug_mode:
        logger.add(sys.stdout, level="DEBUG")
    else:
        logger.add(sys.stdout, level="INFO")
    # 向文件中写入时，默认级别为DEBUG。
    if log_path:
        logger.add(log_path, level="DEBUG")