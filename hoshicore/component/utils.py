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

VERSION = "1.0.0"
RELEASE_NAME = "Lyra"
ORG_NAME = f"STARLab"
SOFTWARE_NAME = f"HoshinoWeaver"


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


def init_logger(logger, debug_mode: bool, trace_mode:bool, log_path: Optional[str]):
    """用于初始化Loguru的logger"""
    logger.remove()
    if trace_mode:
        logger.add(sys.stderr, level="TRACE")
    if debug_mode:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")
    if log_path:
        logger.add(log_path, level="TRACE")
    return logger