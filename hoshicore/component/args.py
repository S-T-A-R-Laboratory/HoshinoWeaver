from dataclasses import dataclass
from fractions import Fraction
from typing import Optional, Union

import numpy as np
from easydict import EasyDict
from loguru import logger
from numpy.typing import NDArray

from .imgfio import ImgSeriesLoader
from .merger import BaseMerger


@dataclass
class StackConfigArg(object):
    """叠加配置参数类。"""
    img_loader: ImgSeriesLoader
    merger_type: BaseMerger
    resize: Optional[list[int]] = None
    max_poolsize: int = 1
    debug: bool = False
