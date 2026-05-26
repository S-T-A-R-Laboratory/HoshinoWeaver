from __future__ import annotations

import asyncio
import datetime
import inspect
import multiprocessing as mp
import os
import platform
import sys
import time
from functools import wraps
from math import floor, sqrt
from pathlib import Path
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
ASTRO_SUFFIX = ["fits", "fts"]
SUPPORT_BITS = [8, 16]
MAGIC_NUM = 3

VERSION = "1.0.0-beta.0"
RELEASE_NAME = "Vega"
ORG_NAME = f"STARLab"
SOFTWARE_NAME = f"HoshinoWeaver"

if getattr(sys, 'frozen', False):
    _EXE_ROOT = Path(sys._MEIPASS)
else:
    _EXE_ROOT = Path(__file__).resolve().parent.parent.parent


def is_support_format(fname: str) -> bool:
    suffix = fname.split(".")[-1].lower()
    return ((suffix in COMMON_SUFFIX) or (suffix in NOT_RECOM_SUFFIX)
            or (suffix in RAW_SUFFIX)  or (suffix in ASTRO_SUFFIX))


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

    if inspect.iscoroutinefunction(func):

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


def _make_log_filename(task: str) -> str:
    """构造日志文件路径：logs/hnw_版本_平台_任务_时间戳.log"""
    os_name = platform.system().lower()  # windows / darwin / linux
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"hnw_{VERSION}_{os_name}_{task}_{ts}.log"
    log_dir = str(_EXE_ROOT / "logs")
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, fname)


def init_logger(logger,
                debug_mode: bool,
                trace_mode: bool,
                log_path: Optional[str],
                task: str = "unknown"):
    """用于初始化Loguru的logger。

    始终向 logs/ 目录写入 TRACE 级别日志文件，文件名含版本、平台、任务和时间戳。
    log_path 若显式传入则使用该路径，否则自动生成。
    """
    logger.remove()
    if sys.stderr is not None:
        if trace_mode:
            logger.add(sys.stderr, level="TRACE")
        elif debug_mode:
            logger.add(sys.stderr, level="DEBUG")
        else:
            logger.add(sys.stderr, level="INFO")
    file_path = log_path if log_path else _make_log_filename(task)
    logger.add(file_path, level="TRACE", encoding="utf-8")
    return logger
