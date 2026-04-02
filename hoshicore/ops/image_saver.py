"""
ImageSaveOp：将叠加结果图像保存到磁盘。

保存逻辑复制自 component/imgfio.py::save_img，
适配为异步 DAG Op。
"""
import asyncio
from typing import Any, Union

import cv2
import numpy as np
from loguru import logger

from .base import BaseOp
from ..component.tagged_image import TaggedImage
from ..component.imgfio import save_img


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
            "None 时自动还原到 TaggedImage 的 source_dtype。",
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

        # TaggedImage 自动 rescale
        if isinstance(image, TaggedImage):
            target_dtype = (
                np.dtype(output_dtype_str)
                if output_dtype_str else None  # None → 还原到 source_dtype
            )
            logger.debug(f"TaggedImage rescale: source={image.source_dtype}, "
                         f"current={image.dtype}, scale={image.scale_factor}, "
                         f"target={target_dtype or image.source_dtype}")
            image = image.rescale_to(target_dtype)

        output_filename = configs['output_filename']
        info = configs.get('exif')

        # 从 info 中提取 exif / colorprofile（兼容 EasyDict 和普通 dict）
        if info is not None:
            exif_data = info.get('exif') if hasattr(info, 'get') else None
            colorprofile = (info.get('colorprofile', b"") if hasattr(
                info, 'get') else b"")
        else:
            exif_data = None
            colorprofile = b""

        try:
            await asyncio.to_thread(
                save_img,
                output_filename,
                image,
                exif=exif_data,
                colorprofile=colorprofile,
            )
            return_code = 0
            logger.info(f"Image saved successfully to {output_filename}")
        except Exception as e:
            logger.error(f"Failed to save image to {output_filename}: {e}")
            return_code = 1

        # 广播返回码
        for queue in self.outputs['return_code']:
            await queue.put(return_code)
