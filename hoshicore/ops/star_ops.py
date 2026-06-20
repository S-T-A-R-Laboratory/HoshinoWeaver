from typing import Any, Awaitable, Mapping

import numpy as np
from loguru import logger

from ..component.data_container import FloatImage
from ..component.star_detect import (detect_starmask_by_dog,
                                    detect_starmask_by_threshold)
from ..component.star_shrink import (apply_mask, apply_mask_guided,
                                    deringing, morph_shrink,
                                    morph_shrink_luma, peak_recovery)
from ..engine.registry import register_op
from .base import ParallelBaseOp

_DETECT_METHODS = {
    "threshold": detect_starmask_by_threshold,
    "dog": detect_starmask_by_dog,
}

SHRINK_MODE_PRESETS: dict[str, dict] = {
    "trail": {
        "shrink_ksize": 7, "shrink_times": 2, "shrink_shape": "CIRCLE",
        "bg_ksize": 31, "recovery_scale": 0.50, "peak_recover_strength": 0.65,
        "deringing_ksize": 27, "blend_method": "hard",
        "guided_radius": 10, "guided_eps": 0.01,
    },
    "standard": {
        "shrink_ksize": 5, "shrink_times": 1, "shrink_shape": "CIRCLE",
        "bg_ksize": 25, "recovery_scale": 0.20, "peak_recover_strength": 0.85,
        "deringing_ksize": 23, "blend_method": "hard",
        "guided_radius": 8, "guided_eps": 0.01,
    },
    "post_align": {
        "shrink_ksize": 3, "shrink_times": 1, "shrink_shape": "CIRCLE",
        "bg_ksize": 21, "recovery_scale": 0.05, "peak_recover_strength": 0.95,
        "deringing_ksize": 15, "blend_method": "hard",
        "guided_radius": 6, "guided_eps": 0.01,
    },
}


def _star_shrink_pipeline(img: np.ndarray, configs: dict[str, Any]) -> np.ndarray:
    """缩星核心流程：检测 → luma 腐蚀 → 峰值恢复 → 振铃修复 → 混合。"""
    mode = configs.get('mode', 'standard')
    if mode == 'custom':
        p = configs
    elif mode in SHRINK_MODE_PRESETS:
        p = SHRINK_MODE_PRESETS[mode]
    else:
        raise ValueError(
            f"Unknown mode: {mode!r}, "
            f"available: {list(SHRINK_MODE_PRESETS.keys())} + 'custom'"
        )

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

    shrunk = morph_shrink_luma(
        img,
        ksize=p['shrink_ksize'],
        shape=p['shrink_shape'],
        times=p['shrink_times'],
    )

    #shrunk = peak_recovery(
    #    img, shrunk,
    #    bg_ksize=p['bg_ksize'],
    #    strength=p['peak_recover_strength'],
    #    scale=p['recovery_scale'],
    #)

    shrunk = deringing(img, shrunk, algo="gaussian", ksize=p['deringing_ksize'])

    blend = p['blend_method']
    if blend == "hard":
        return apply_mask(img, shrunk, star_mask)
    elif blend == "guided":
        return apply_mask_guided(
            img, shrunk, star_mask,
            radius=p['guided_radius'],
            eps=p['guided_eps'],
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
        # 主接口
        "mode":               {"type": "str",   "default": "post_align"},
        # 检测参数（始终独立配置，不受 mode 影响）
        "detect_method":      {"type": "str",   "default": "threshold"},
        "detect_ksize":       {"type": "int",   "default": 13},
        "detect_threshold":   {"type": "float", "default": 1.0},
        "detect_open":        {"type": "int",   "default": 3},
        "detect_dilate":      {"type": "int",   "default": 0},
        "dog_sigma_small":    {"type": "float", "default": 1.5},
        "dog_sigma_large":    {"type": "float", "default": 12.0},
        # 高级参数（仅 mode="custom" 时生效）
        "shrink_ksize":           {"type": "int",   "default": 5},
        "shrink_times":           {"type": "int",   "default": 1},
        "shrink_shape":           {"type": "str",   "default": "CIRCLE"},
        "bg_ksize":               {"type": "int",   "default": 25},
        "recovery_scale":         {"type": "float", "default": 0.20},
        "peak_recover_strength":  {"type": "float", "default": 0.85},
        "deringing_ksize":        {"type": "int",   "default": 23},
        "blend_method":           {"type": "str",   "default": "hard"},
        "guided_radius":          {"type": "int",   "default": 8},
        "guided_eps":             {"type": "float", "default": 0.01},
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
