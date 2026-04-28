from typing import Any, Awaitable, Mapping

import numpy as np
from loguru import logger

from ..component.data_container import FloatImage
from ..component.stardetect import (detect_starmask_by_dog,
                                    detect_starmask_by_threshold)
from ..component.starshrink import (apply_mask, apply_mask_guided,
                                    deringing, morph_shrink)
from ..engine.registry import register_op
from .base import ParallelBaseOp

_DETECT_METHODS = {
    "threshold": detect_starmask_by_threshold,
    "dog": detect_starmask_by_dog,
}


def _star_shrink_pipeline(img: np.ndarray, configs: dict[str, Any]) -> np.ndarray:
    """缩星核心流程：检测 → 缩星 → 振铃修复 → 混合。"""
    method = configs['detect_method']
    detect_fn = _DETECT_METHODS.get(method)
    if detect_fn is None:
        raise ValueError(
            f"Unknown detect_method: {method}, "
            f"available: {list(_DETECT_METHODS.keys())}")

    detect_kwargs: dict[str, Any] = {
        "threshold_ratio": configs['detect_threshold'],
        "open_ksize": configs['detect_open'],
        "dilate_ksize": configs['detect_dilate'],
    }
    if method == "threshold":
        detect_kwargs["ksize"] = configs['detect_ksize']
    elif method == "dog":
        detect_kwargs["sigma_small"] = configs['dog_sigma_small']
        detect_kwargs["sigma_large"] = configs['dog_sigma_large']

    star_mask = detect_fn(img, **detect_kwargs)

    shrunk = morph_shrink(
        img,
        ksize=configs['shrink_ksize'],
        ratio=configs['shrink_ratio'],
        shape=configs['shrink_shape'],
        times=configs['shrink_times'],
    )

    if configs['deringing']:
        shrunk = deringing(
            img, shrunk,
            algo=configs['deringing_algo'],
            ksize=configs['deringing_ksize'],
        )

    blend = configs['blend_method']
    if blend == "hard":
        return apply_mask(img, shrunk, star_mask)
    elif blend == "guided":
        return apply_mask_guided(
            img, shrunk, star_mask,
            radius=configs['guided_radius'],
            eps=configs['guided_eps'],
        )
    raise ValueError(f"Unknown blend_method: {blend}, available: hard, guided")


@register_op()
class StarShrinkOp(ParallelBaseOp):
    """缩星算子。

    兼容两种接入方式：
    - 序列输入（INPUTS data）：逐帧处理，用于管线中间
    - 单图输入（stacker image 输出）：借助 set_length(1) 退化为单帧处理
    """

    INPUTS: dict[str, Any] = {
        "data": {"type": "sequence", "required": True},
    }
    CONFIGS: dict[str, Any] = {
        "detect_method":    {"type": "str",   "default": "dog"},
        "detect_ksize":     {"type": "int",   "default": 13},
        "detect_threshold": {"type": "float", "default": 5.0},
        "detect_open":      {"type": "int",   "default": 3},
        "detect_dilate":    {"type": "int",   "default": 0},
        "dog_sigma_small":  {"type": "float", "default": 1.5},
        "dog_sigma_large":  {"type": "float", "default": 12.0},
        "shrink_ksize":     {"type": "int",   "default": 5},
        "shrink_ratio":     {"type": "float", "default": 1.0},
        "shrink_shape":     {"type": "str",   "default": "RECT"},
        "shrink_times":     {"type": "int",   "default": 1},
        "deringing":        {"type": "bool",  "default": False},
        "deringing_algo":   {"type": "str",   "default": "median"},
        "deringing_ksize":  {"type": "int",   "default": 25},
        "blend_method":     {"type": "str",   "default": "guided"},
        "guided_radius":    {"type": "int",   "default": 8},
        "guided_eps":       {"type": "float", "default": 0.01},
    }
    OUTPUTS: dict[str, Any] = {
        "result": {"type": "sequence"},
    }

    async def _async_execute_single(self, data: Mapping[str, Awaitable[Any]],
                                    configs: dict[str, Any]) -> dict[str, Any]:
        raw = await data['data']

        if isinstance(raw, FloatImage):
            img = raw.data
            if img.dtype.kind == 'f':
                work_img = np.round(img).astype(raw.dtype)
            else:
                work_img = img
            result = await self._run_cpu(_star_shrink_pipeline, work_img, configs)
            out = FloatImage(data=result.astype(img.dtype), dtype=raw.dtype)
        else:
            result = await self._run_cpu(_star_shrink_pipeline, raw, configs)
            out = result

        return {"result": out}
