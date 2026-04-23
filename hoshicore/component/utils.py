from __future__ import annotations

import asyncio
import multiprocessing as mp
import sys
import time
from functools import wraps
from math import floor, sqrt
from typing import Callable, Optional, Union

import numpy as np
import psutil
from easydict import EasyDict
from loguru import logger

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


def error_raiser(error, result_queue):
    """A simple error raiser. For subprocessor callback function."""
    result_queue.put(EasyDict(img=None, err_msg=[
        error,
    ]))


def is_support_format(fname: str) -> bool:
    suffix = fname.split(".")[-1].lower()
    return ((suffix in COMMON_SUFFIX) or (suffix in NOT_RECOM_SUFFIX)
            or (suffix in RAW_SUFFIX))


def get_resize(opt: Optional[str], raw_wh: Union[list, tuple]):
    """
    accept raw_wh in any order. [h, w] is recommended to avoid misuse.

    but if opt is given as "1920x1080", it will return in [h, w] order.
    """
    if not opt: return None
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
    Supports both sync and async functions.
    """

    def _log_cost(t0: float, args):
        cls_name = ""
        if args and hasattr(args[0], func.__name__):
            cls_name = args[0].__class__.__name__ + "."
        logger.info(
            f"{cls_name}{func.__name__} time cost: {(time.time()-t0):.2f}s.")

    if asyncio.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            t0 = time.time()
            res = await func(*args, **kwargs)
            _log_cost(t0, args)
            return res

        return async_wrapper

    @wraps(func)
    def do_func(*args, **kwargs):
        t0 = time.time()
        res = func(*args, **kwargs)
        _log_cost(t0, args)
        return res

    return do_func


def get_mp_num(tot_num: int,
               prefer_num: Optional[int] = None) -> tuple[int, float]:
    """
    设置处理器使用数目，在不超出处理器数目限制的情况下，尽可能使每个处理器叠加sqrt(N)张图像
    """
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


def init_logger(logger, debug_mode: bool, log_path: Optional[str]):
    """用于初始化Loguru的logger"""
    logger.remove()
    if debug_mode:
        logger.add(sys.stdout, level="DEBUG")
    else:
        logger.add(sys.stdout, level="INFO")
    if log_path:
        logger.add(log_path, level="DEBUG")
