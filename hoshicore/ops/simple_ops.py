import os
from typing import Any, Awaitable, Mapping, Optional

import cv2
import numpy as np
from loguru import logger

from ..component.calibration import (calibration_divide, calibration_subtract,
                                     crop_roi, natural_sort_key, resize_image)
from ..component.data_container import DTYPE_MAX_VALUE, FloatImage
from ..component.image_io import load_img
from ..component.queue import StreamExhausted
from ..engine.registry import register_op
from .base import BaseOp, ParallelBaseOp


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
        result = None
        if path is not None:
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


# ── 图像缩放 ──


@register_op()
class ImageResizeOp(ParallelBaseOp):
    """图像缩放：按比例或目标尺寸缩放序列帧。

    优先级: scale > (width, height) > passthrough（全部为 None 则不缩放）。
    interpolation 为 "auto" 时缩小用 area，放大用 cubic。
    """

    INPUTS: dict[str, Any] = {
        "data": {
            "type": "sequence"
        },
    }
    CONFIGS: dict[str, Any] = {
        "scale": {
            "type": "float",
            "default": None
        },
        "width": {
            "type": "int",
            "default": None
        },
        "height": {
            "type": "int",
            "default": None
        },
        "interpolation": {
            "type": "str",
            "default": "auto"
        },
    }
    OUTPUTS: dict[str, Any] = {
        "result": {
            "type": "sequence"
        },
    }

    async def _async_execute_single(self, data: Mapping[str, Awaitable[Any]],
                                    configs: dict[str, Any]) -> dict[str, Any]:
        frame = await data['data']
        scale = configs.get('scale')
        width = configs.get('width')
        height = configs.get('height')
        interpolation = configs.get('interpolation', 'auto')

        if isinstance(frame, FloatImage):
            resized = resize_image(frame.data, scale, width, height,
                                   interpolation)
            result = FloatImage(data=resized, dtype=frame.dtype)
        elif isinstance(frame, np.ndarray):
            result = resize_image(frame, scale, width, height, interpolation)
        else:
            result = frame

        return {"result": result}


# ── 图像裁切 ──


@register_op()
class ImageCropOp(ParallelBaseOp):
    """图像 ROI 裁切：从序列帧中裁取指定区域。"""

    INPUTS: dict[str, Any] = {
        "data": {
            "type": "sequence"
        },
    }
    CONFIGS: dict[str, Any] = {
        "x": {
            "type": "int",
            "required": True
        },
        "y": {
            "type": "int",
            "required": True
        },
        "roi_width": {
            "type": "int",
            "required": True
        },
        "roi_height": {
            "type": "int",
            "required": True
        },
    }
    OUTPUTS: dict[str, Any] = {
        "result": {
            "type": "sequence"
        },
    }

    async def _async_execute_single(self, data: Mapping[str, Awaitable[Any]],
                                    configs: dict[str, Any]) -> dict[str, Any]:
        frame = await data['data']
        x = configs['x']
        y = configs['y']
        w = configs['roi_width']
        h = configs['roi_height']

        if isinstance(frame, FloatImage):
            cropped = crop_roi(frame.data, x, y, w, h)
            result = FloatImage(data=cropped, dtype=frame.dtype)
        elif isinstance(frame, np.ndarray):
            result = crop_roi(frame, x, y, w, h)
        else:
            result = frame

        return {"result": result}


# ── 校准减法 ──


@register_op()
class CalibrationSubtractOp(ParallelBaseOp):
    """通用校准减法：逐帧减去参考帧（暗场 / 偏置帧）。

    reference 为 None 时直接 passthrough（用户未提供校准帧）。
    """

    EXECUTOR = "cpu"
    INPUTS: dict[str, Any] = {
        "data": {
            "type": "sequence"
        },
    }
    CONFIGS: dict[str, Any] = {
        "reference": {
            "type": "image",
            "default": None
        },
    }
    OUTPUTS: dict[str, Any] = {
        "result": {
            "type": "sequence"
        },
    }

    async def _async_execute_single(self, data: Mapping[str, Awaitable[Any]],
                                    configs: dict[str, Any]) -> dict[str, Any]:
        frame = await data['data']
        ref = configs.get('reference')

        if ref is None:
            return {"result": frame}

        # 拆包 FloatImage
        if isinstance(frame, FloatImage):
            frame_arr, frame_dtype = frame.data, frame.dtype
        else:
            frame_arr, frame_dtype = frame, frame.dtype

        if isinstance(ref, FloatImage):
            ref_arr, ref_dtype = ref.data, ref.dtype
        else:
            ref_arr, ref_dtype = ref, ref.dtype

        result_arr, out_dtype = calibration_subtract(frame_arr, ref_arr,
                                                     frame_dtype, ref_dtype)

        # 重包装
        if isinstance(frame, FloatImage):
            result = FloatImage(data=result_arr, dtype=out_dtype)
        else:
            result = result_arr

        return {"result": result}


# ── 校准除法 ──


@register_op()
class CalibrationDivideOp(ParallelBaseOp):
    """通用校准除法：逐帧除以参考帧（平场校正）。

    reference 为 None 时直接 passthrough。
    """

    EXECUTOR = "cpu"
    INPUTS: dict[str, Any] = {
        "data": {
            "type": "sequence"
        },
    }
    CONFIGS: dict[str, Any] = {
        "reference": {
            "type": "image",
            "default": None
        },
    }
    OUTPUTS: dict[str, Any] = {
        "result": {
            "type": "sequence"
        },
    }

    async def _async_execute_single(self, data: Mapping[str, Awaitable[Any]],
                                    configs: dict[str, Any]) -> dict[str, Any]:
        frame = await data['data']
        ref = configs.get('reference')

        if ref is None:
            return {"result": frame}

        if isinstance(frame, FloatImage):
            frame_arr, frame_dtype = frame.data, frame.dtype
        else:
            frame_arr, frame_dtype = frame, frame.dtype

        if isinstance(ref, FloatImage):
            ref_arr, ref_dtype = ref.data, ref.dtype
        else:
            ref_arr, ref_dtype = ref, ref.dtype

        result_arr, out_dtype = calibration_divide(frame_arr, ref_arr,
                                                   frame_dtype, ref_dtype)

        if isinstance(frame, FloatImage):
            result = FloatImage(data=result_arr, dtype=out_dtype)
        else:
            result = result_arr

        return {"result": result}


# ── 序列排序 ──


@register_op()
class SequenceSortOp(BaseOp):
    """序列排序：收集所有输入项，按指定规则排序后重新流式输出。

    支持三种排序模式:
    - "natural": 自然排序（数字部分按数值比较，如 img_2 < img_10）
    - "name": 字符串排序
    - "mtime": 按文件修改时间排序
    """

    INPUTS: dict[str, Any] = {
        "data": {
            "type": "sequence"
        },
    }
    CONFIGS: dict[str, Any] = {
        "sort_key": {
            "type": "str",
            "default": "natural"
        },
        "reverse": {
            "type": "bool",
            "default": False
        },
    }
    OUTPUTS: dict[str, Any] = {
        "result": {
            "type": "sequence"
        },
    }

    def _infer_output_length(self, input_lengths):
        return input_lengths.get('data')

    async def _async_execute(self, configs: dict[str, Any]) -> None:
        sort_key_name: str = configs['sort_key']
        reverse: bool = configs['reverse']
        tot_num = self.length
        assert tot_num is not None, \
            "SequenceSortOp requires sequence length information."

        # 收集阶段
        items = []
        for _ in range(tot_num):
            item = await self.inputs['data'].get()
            items.append(item)

        # 排序阶段
        key_funcs = {
            "natural": natural_sort_key,
            "name": str,
            "mtime": lambda f: os.path.getmtime(str(f)),
        }
        key_fn = key_funcs.get(sort_key_name)
        if key_fn is None:
            raise ValueError(f"Unsupported sort_key: {sort_key_name}. "
                             f"Available: {sorted(key_funcs.keys())}")

        sorted_items = sorted(items, key=key_fn, reverse=reverse)
        logger.info(
            f"{self.name}: sorted {len(sorted_items)} items by '{sort_key_name}'"
        )

        # 输出阶段
        for item in sorted_items:
            await self._broadcast_outputs({"result": item})


# ── None 输出 ──


@register_op()
class NoneOutputOp(BaseOp):
    """输出 None 的极简 Op，用于 Meta YAML route="none" 场景。

    下游 Passthrough 类 Op（如 CalibrationSubtractOp）收到 None 后直接跳过处理。
    """

    OUTPUTS: dict[str, Any] = {
        "result": {
            "type": "image"
        },
    }

    async def _async_execute(self, configs: dict[str, Any]) -> None:
        await self._broadcast_outputs({"result": None})


# ── Mask 操作 ──


@register_op()
class MaskInvertOp(BaseOp):
    """反转 mask：output = 1 - input。input 为 None 时输出 np.zeros((1,1))（全0哨兵，下游负责 resize）。"""

    CONFIGS: dict[str, Any] = {
        "mask": {
            "type": "image",
            "required": False,
            "default": None
        },
    }
    OUTPUTS: dict[str, Any] = {
        "result": {
            "type": "image"
        },
    }

    async def _async_execute(self, configs: dict[str, Any]) -> None:
        mask = configs['mask']
        if mask is None:
            result = np.zeros((1, 1), dtype=np.float32)
        else:
            result = 1.0 - mask
        await self._broadcast_outputs({"result": result})


# ── 图像算术 ──


@register_op()
class ImageAddOp(BaseOp):
    """两路图像逐像素相加。用于互补 mask 分离叠加后的合成。

    当 image_b 为 None 时直接输出 image_a（支持 prune 断路场景）。
    """

    CONFIGS: dict[str, Any] = {
        "image_a": {
            "type": "image",
            "required": True
        },
        "image_b": {
            "type": "image",
            "default": None
        },
    }
    OUTPUTS: dict[str, Any] = {
        "result": {
            "type": "image"
        },
    }

    async def _async_execute(self, configs: dict[str, Any]) -> None:
        a = configs['image_a']
        b = configs['image_b']
        if b is None:
            await self._broadcast_outputs({"result": a})
            return
        a_arr = a.data if isinstance(a, FloatImage) else a
        b_arr = b.data if isinstance(b, FloatImage) else b
        result_arr = np.add(a_arr, b_arr)
        if isinstance(a, FloatImage):
            result = FloatImage(data=result_arr, dtype=a.dtype)
        else:
            result = result_arr
        await self._broadcast_outputs({"result": result})


@register_op()
class ApplyMaskOp(BaseOp):
    """图像乘以 mask。mask 为 None 时直接 passthrough。"""

    INPUTS: dict[str, Any] = {"image": {"type": "sequence", "required": True}}

    CONFIGS: dict[str, Any] = {
        "mask": {
            "type": "image",
            "required": False,
            "default": None
        }
    }
    OUTPUTS: dict[str, Any] = {"result": {"type": "sequence"}}

    async def _async_execute(self, configs: dict[str, Any]) -> None:
        tot_num = self.length
        mask = configs['mask']
        runtime_mask = None
        if mask is not None:
            if mask.ndim==2:
                mask = np.repeat(mask[..., None], 3, axis=-1)
            runtime_mask = mask > 0.5
        for i in self._input_range():
            try:
                img = await self._async_convert_inputs()['image']
            except StreamExhausted:
                if tot_num is not None:
                    logger.warning(
                        f"{self.name}: upstream ended at {i}/{tot_num}")
                break

            if runtime_mask is None:
                result = img
            else:
                img_arr = img.data if isinstance(img, FloatImage) else img
                if runtime_mask.shape != img_arr.shape:
                    h, w, _ = img_arr.shape
                    runtime_mask = cv2.resize(
                        mask.astype(np.float32), dsize=(w, h),
                        interpolation=cv2.INTER_NEAREST) > 0.5
                result_arr = np.multiply(img_arr, runtime_mask)
                if isinstance(img, FloatImage):
                    result = FloatImage(data=result_arr, dtype=img.dtype)
                else:
                    result = result_arr
            await self._broadcast_outputs({"result": result})
