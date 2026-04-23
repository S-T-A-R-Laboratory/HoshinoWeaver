"""
图像保存算子：单张保存（ImageSaveOp）和批量序列保存（BatchImageSaveOp）。
"""
import asyncio
import os
from typing import Any, Awaitable, Mapping, Union

import numpy as np
from loguru import logger

from hoshicore.component.data_container import FloatImage, rescale_array

from ..component.imgfio import save_img
from ..engine.registry import register_op
from .base import BaseOp, ParallelBaseOp


@register_op()
class ImageSaveOp(BaseOp):
    """图像保存算子：将图像写入磁盘，可选写入 EXIF 和色彩配置。

    所有输入均为单次值（非序列），通过 configs 接收：
    - image: 待保存的 np.ndarray 图像
    - output_filename: 目标文件路径
    - exif: (可选) EXIF 信息字典
    - colorprofile: (可选) ICC 色彩配置字节串

    输出 return_code: 0 表示成功，1 表示失败。
    """

    CONFIGS: dict[str, Any] = {
        "image": {
            "type": "image",
            "description": "待保存的图像",
        },
        "output_filename": {
            "type": "str",
            "description": "输出文件路径",
        },
        "output_dtype": {
            "type": "str",
            "description": "输出图像的目标 dtype（如 'uint8', 'uint16'）。"
            "None 时直接使用图像当前 dtype 保存。",
            "default": None,
        },
        "exif": {
            "type": "object",
            "description": "EXIF 及色彩配置信息",
            "default": None,
        },
    }
    OUTPUTS: dict[str, Any] = {
        "return_code": {
            "type": "int",
            "description": "返回码 (0=成功, 1=失败)",
        },
    }

    def __init__(self, name: str):
        super().__init__(name)

    async def _async_execute(self, configs: dict[str, Any]) -> None:
        image = configs['image']
        output_dtype_str = configs.get('output_dtype')
        output_filename = configs['output_filename']
        exif = configs.get('exif')

        target_dtype = None
        # JPEG 强制要求 uint8
        if output_filename.lower().endswith(".jpg"):
            target_dtype = np.dtype('uint8')
        elif output_dtype_str:
            # 按需 dtype 转换
            target_dtype = np.dtype(output_dtype_str)
        if target_dtype is not None:
            logger.debug(f"Image dtype cast: {image.dtype} → {target_dtype}")

        if isinstance(image, FloatImage):
            image = image.int_transform(target_dtype)
        elif isinstance(image, np.ndarray):
            if target_dtype is not None and image.dtype != target_dtype:
                image = rescale_array(image, image.dtype, target_dtype)

        try:
            await asyncio.to_thread(save_img,
                                    output_filename,
                                    image,
                                    exif=exif)
            return_code = 0
            logger.info(f"Image saved successfully to {output_filename}")
        except Exception as e:
            logger.error(f"Failed to save image to {output_filename}: {e}")
            return_code = 1

        # 广播返回码
        await self._broadcast_outputs({"return_code": return_code})


def _format_output_path(template: str, index: int, total: int) -> str:
    """根据模板生成带序号的输出路径。

    支持两种模式：
    - 模板中包含 {index} 占位符：直接格式化
    - 模板无占位符：在扩展名前插入 _NNNNN 序号

    序号位数根据 total 自动确定（至少 4 位）。
    """
    if "{index" in template:
        return template.format(index=index)

    base, ext = os.path.splitext(template)
    width = max(4, len(str(total)))
    return f"{base}_{index:0{width}d}{ext}"


@register_op()
class BatchImageSaveOp(ParallelBaseOp):
    """批量序列保存算子：逐帧保存序列图像到磁盘。

    用于生成延时视频帧序列（如累积星轨的逐帧输出）。

    文件命名：
    - output_dir + output_template 组合生成路径
    - output_template 支持 {index} 占位符（如 "frame_{index:05d}.png"）
    - 无占位符时自动在扩展名前插入序号（如 "frame.png" → "frame_0001.png"）
    """

    INPUTS: dict[str, Any] = {
        "data": {"type": "sequence"},
    }
    CONFIGS: dict[str, Any] = {
        "output_dir": {
            "type": "str",
            "description": "输出目录路径",
        },
        "output_template": {
            "type": "str",
            "default": "frame.png",
            "description": "文件名模板，支持 {index} 占位符",
        },
        "output_dtype": {
            "type": "str",
            "default": None,
            "description": "输出 dtype（如 'uint8'）。None 时保持原始 dtype",
        },
        "start_index": {
            "type": "int",
            "default": 0,
            "description": "起始序号",
        },
    }
    OUTPUTS: dict[str, Any] = {
        "result": {"type": "sequence"},
    }

    _frame_counter: int = 0

    async def _async_execute(self, configs: dict[str, Any]) -> None:
        output_dir = configs['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        self._frame_counter = configs.get('start_index', 0)
        await super()._async_execute(configs)

    async def _async_execute_single(self, data: Mapping[str, Awaitable[Any]],
                                    configs: dict[str, Any]) -> dict[str, Any]:
        frame = await data['data']
        output_dir = configs['output_dir']
        template = configs.get('output_template', 'frame.png')
        output_dtype_str = configs.get('output_dtype')

        idx = self._frame_counter
        self._frame_counter += 1
        total = self.length or 0

        filename = _format_output_path(template, idx, total)
        filepath = os.path.join(output_dir, filename)

        # dtype 转换
        target_dtype = None
        if filepath.lower().endswith(".jpg"):
            target_dtype = np.dtype('uint8')
        elif output_dtype_str:
            target_dtype = np.dtype(output_dtype_str)

        save_frame = frame
        if isinstance(frame, FloatImage):
            save_frame = frame.int_transform(target_dtype)
        elif isinstance(frame, np.ndarray):
            if target_dtype is not None and frame.dtype != target_dtype:
                save_frame = rescale_array(frame, frame.dtype, target_dtype)

        try:
            await asyncio.to_thread(save_img, filepath, save_frame)
        except Exception as e:
            logger.error(f"Failed to save frame {idx} to {filepath}: {e}")

        return {"result": frame}
