from typing import Any, Awaitable, Mapping

import numpy as np
from loguru import logger

from ..component.data_container import FloatImage
from ..component.star_detect import (detect_starmask_by_dog,
                                     detect_starmask_by_threshold)
from ..component.star_shrink import apply_mask, deringing, morph_shrink_luma
from ..engine.registry import register_op
from .base import ParallelBaseOp

_DETECT_METHODS = {
    "threshold": detect_starmask_by_threshold,
    "dog": detect_starmask_by_dog,
}

SHRINK_MODE_PRESETS: dict[str, dict] = {
    "light": {
        "shrink_ksize": 3,
        "shrink_times": 1,
        "shrink_ratio": 0.5,
        "deringing_ksize": 51
    },
    "moderate": {
        "shrink_ksize": 3,
        "shrink_times": 1,
        "shrink_ratio": 1.0,
        "deringing_ksize": 51
    },
    "strong": {
        "shrink_ksize": 3,
        "shrink_times": 2,
        "shrink_ratio": 0.75,
        "deringing_ksize": 51
    },
    "aggressive": {
        "shrink_ksize": 3,
        "shrink_times": 3,
        "shrink_ratio": 0.66,
        "deringing_ksize": 51
    },
    "removal": {
        "shrink_ksize": 7,
        "shrink_times": 2,
        "shrink_ratio": 1.0,
        "deringing_ksize": 51
    },
}


def _star_shrink_pipeline(img: np.ndarray, configs: dict[str,
                                                         Any]) -> np.ndarray:
    """缩星核心流程：检测 → luma 腐蚀（迭代混合）→ 振铃修复 → 硬蒙版合成。"""
    mode = configs.get('mode', 'moderate')
    if mode == 'custom':
        p = configs
    elif mode in SHRINK_MODE_PRESETS:
        p = SHRINK_MODE_PRESETS[mode]
    else:
        raise ValueError(
            f"Unknown mode: {mode!r}, "
            f"available: {list(SHRINK_MODE_PRESETS.keys())} + 'custom'")

    method = configs['detect_method']
    detect_fn = _DETECT_METHODS.get(method)
    if detect_fn is None:
        raise ValueError(f"Unknown detect_method: {method}, "
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

    raw_ratio = p.get('shrink_ratio', 0.0)
    shrunk = morph_shrink_luma(
        img,
        ksize=p['shrink_ksize'],
        shape=p.get('shrink_shape', 'CIRCLE'),
        times=p['shrink_times'],
        ratio=None if raw_ratio == 0.0 else raw_ratio,
    )

    shrunk = deringing(img, shrunk, algo="mean", ksize=p['deringing_ksize'])

    return apply_mask(img, shrunk, star_mask)


@register_op()
class StarShrinkOp(ParallelBaseOp):
    """缩星算子。

    兼容两种接入方式：
    - 序列输入（INPUTS data）：逐帧处理，用于管线中间
    - 单图输入（stacker image 输出）：借助 set_length(1) 退化为单帧处理
    """

    INPUTS: dict[str, Any] = {
        "data": {
            "type": "sequence",
            "required": True
        },
    }
    CONFIGS: dict[str, Any] = {
        "mode": {
            "type": "str",
            "default": "moderate"
        },
        # 检测参数（始终独立配置，不受 mode 影响）
        "detect_method": {
            "type": "str",
            "default": "threshold"
        },
        "detect_ksize": {
            "type": "int",
            "default": 13
        },
        "detect_threshold": {
            "type": "float",
            "default": 1.0
        },
        "detect_open": {
            "type": "int",
            "default": 3
        },
        "detect_dilate": {
            "type": "int",
            "default": 0
        },
        "dog_sigma_small": {
            "type": "float",
            "default": 1.5
        },
        "dog_sigma_large": {
            "type": "float",
            "default": 12.0
        },
        # 高级参数（仅 mode="custom" 时生效）
        "shrink_ksize": {
            "type": "int",
            "default": 5
        },
        "shrink_times": {
            "type": "int",
            "default": 1
        },
        "shrink_ratio": {
            "type": "float",
            "default": 0.0
        },
        "shrink_shape": {
            "type": "str",
            "default": "CIRCLE"
        },
        "deringing_ksize": {
            "type": "int",
            "default": 11
        },
        "blend_method": {
            "type": "str",
            "default": "hard"
        },
    }
    OUTPUTS: dict[str, Any] = {
        "result": {
            "type": "sequence"
        },
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
            result = await self._run_cpu(_star_shrink_pipeline, work_img,
                                         configs)
            out = FloatImage(data=result.astype(img.dtype), dtype=raw.dtype)
        else:
            result = await self._run_cpu(_star_shrink_pipeline, raw, configs)
            out = result

        return {"result": out}
