from typing import Any, Optional

import cv2
import numpy as np
from loguru import logger

from ..component.frame_buffer import DiskFrameBuffer
from ..component.imgfio import load_img
from ..component.merger import (MaxMerger, MeanMerger, MinMerger,
                                SigmaClippingMerger)
from ..component.noise_equalization import equalize_noise
from ..component.tagged_image import FloatImage, align_dtype_pair
from ..component.utils import DTYPE_MAX_VALUE, FastGaussianParam
from ..engine.registry import register_op
from .base import BaseOp


@register_op()
class LoadSingleImageOp(BaseOp):
    """
    加载单张图片
    """
    CONFIGS: dict[str, dict[str, Any]] = {
        "path": {
            "type": "str",
            "required": True
        },
    }
    OUTPUTS = {
        "result": {
            "type": "image"
        },
    }

    async def _async_execute(self, configs: dict[str, Any]) -> None:
        path = configs['path']
        result = load_img(path)
        await self._broadcast_outputs({"result": result})


@register_op()
class LoadMaskImageOp(BaseOp):
    """
    加载单张图片作为掩模
    """
    CONFIGS: dict[str, dict[str, Any]] = {
        "path": {
            "type": "str",
            "required": False,
            "default": None
        },
    }
    OUTPUTS = {
        "result": {
            "type": "image"
        },
    }

    async def _async_execute(self, configs: dict[str, Any]) -> None:
        mask_name: Optional[str] = configs['path']
        mask = load_img(mask_name) if mask_name is not None else None
        if mask is None:
            await self._broadcast_outputs({"result": None})
            return

        # mask 预处理：归一化，灰度化处理-三通道mask 转换
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) if len(
            mask.shape) == 3 else mask
        mask = np.repeat(mask[..., None], 3, axis=-1)

        if mask.dtype in DTYPE_MAX_VALUE:
            max_mask_value = DTYPE_MAX_VALUE[mask.dtype]
            normalized_mask = mask / max_mask_value
        else:
            # assume float
            normalized_mask = mask / np.max(mask)

        await self._broadcast_outputs({"result": normalized_mask})
