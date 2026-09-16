"""
图像保存算子：单张保存（ImageSaveOp）和批量序列保存（BatchImageSaveOp）。
"""
import asyncio
import os
from typing import Any, Awaitable, Mapping, Union

import numpy as np
from loguru import logger

from hoshicore.component.data_container import FloatImage, rescale_array

from ..component.image_io import save_img
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
        "jpg_quality": {
            "type": "int",
            "description": "JPG 图像质量（1-100）",
            "default": 85,
        },
        "png_compressing": {
            "type": "int",
            "description": "PNG 压缩级别（1-9）",
            "default": 7,
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
        jpg_quality = configs['jpg_quality']
        png_compressing = configs['png_compressing']

        target_dtype = None
        # JPEG 强制要求 uint8
        if output_filename.lower().endswith((".jpg", ".jpeg")):
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
                                    png_compressing=png_compressing,
                                    jpg_quality=jpg_quality,
                                    exif=exif)
            logger.info(f"Image saved successfully to {output_filename}")
        except Exception as e:
            err_msg = f"Failed to save image to {output_filename}: {e}"
            logger.error(err_msg)
            raise RuntimeError(err_msg) from e

        # 广播返回码0
        await self._broadcast_outputs({"return_code": 0})


def _format_output_path(
    template: str,
    index: int,
    total: int,
    *,
    frame_index: int | None = None,
) -> str:
    """根据模板生成带序号的输出路径。

    支持两种模式：
    - 模板中包含 {index} 占位符：直接格式化
    - 模板无占位符：在扩展名前插入 _NNNNN 序号

    序号位数根据 total 自动确定（至少 4 位）。
    """
    if "{frame_index" in template and frame_index is None:
        raise ValueError(
            "output_template uses {frame_index}, but frame_indices is not wired")
    if "{index" in template or "{frame_index" in template:
        return template.format(index=index, frame_index=frame_index)

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
        "frame_indices": {"type": "sequence", "required": False},
        "exifs": {"type": "sequence", "required": False},
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
        "jpg_quality": {
            "type": "int",
            "default": 85,
        },
        "png_compressing": {
            "type": "int",
            "default": 7,
        },
    }
    OUTPUTS: dict[str, Any] = {
        "result": {"type": "sequence", "item_type": "str",
                   "description": "已保存的文件路径"},
    }
    CONCURRENCY: int = 4
    
    _frame_counter: int = 0

    async def _async_execute(self, configs: dict[str, Any]) -> None:
        output_dir = configs['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        self._frame_counter = configs.get('start_index', 0)
        await super()._async_execute(configs)

    async def _async_execute_single(self, data: Mapping[str, Awaitable[Any]],
                                    configs: dict[str, Any]) -> dict[str, Any]:
        frame = await data['data']
        frame_index = (await data['frame_indices']
                       if self.inputs['frame_indices'].active else None)
        exif = (await data['exifs'] if self.inputs['exifs'].active else None)
        output_dir = configs['output_dir']
        template = configs.get('output_template', 'frame.png')
        output_dtype_str = configs.get('output_dtype')

        idx = self._frame_counter
        self._frame_counter += 1
        total = self.length or 0

        filename = _format_output_path(
            template, idx, total, frame_index=frame_index)
        filepath = os.path.join(output_dir, filename)

        # dtype 转换
        target_dtype = None
        if filepath.lower().endswith((".jpg", ".jpeg")):
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
            await asyncio.to_thread(
                save_img,
                filepath,
                save_frame,
                png_compressing=configs.get('png_compressing', 7),
                jpg_quality=configs.get('jpg_quality', 85),
                exif=exif,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to save frame {idx} to {filepath}: {exc}") from exc

        return {"result": filepath}
