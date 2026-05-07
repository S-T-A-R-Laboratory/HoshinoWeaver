"""
对齐算子：星点对齐等帧间配准操作。

StarAlignmentOp 支持两条对齐路径：
  1. 2D 单应性（homography）：FlatCameraModel + warpPerspective（无需 EXIF）
  2. 相机模型优化（camera_model）：Intrinsics + optimize_alignment + remap

method="auto" 时，根据 EXIF 是否能构建完整 Intrinsics 自动选择。
对齐失败的帧被丢弃，输出为变长序列（sentinel 驱动）。
"""
from typing import Any, Optional

import numpy as np
from loguru import logger

from ..component.norma.frame_align import (AlignmentError,
                                           align_frame_camera_model,
                                           align_frame_homography,
                                           make_geometry, try_build_camera)
from ..component.norma.types import CameraModel
from ..component.norma.cache import GeometryView
from ..component.queue import StreamExhausted
from ..engine.registry import register_op
from ..component.data_container import FloatImage
from .base import BaseOp


@register_op()
class StarAlignmentOp(BaseOp):
    """星点对齐：将序列帧对齐到参考帧。

    支持两条路径：
    - homography: 2D 单应性（FlatCameraModel），无需相机信息
    - camera_model: 联合优化旋转+焦距+畸变，需要 EXIF 提供内参

    method="auto" 时根据 EXIF 可用性自动选择。
    对齐失败的帧被丢弃（不输出），因此输出为变长序列。
    """

    EXECUTOR = "cpu"
    VARIABLE_OUTPUT = True
    INPUTS: dict[str, Any] = {
        "data": {"type": "sequence"},
        "exifs": {"type": "sequence", "required": False},
    }
    CONFIGS: dict[str, Any] = {
        "reference":   {"type": "image", "default": None},
        "method":      {"type": "str",   "default": "auto"},
        "same_camera": {"type": "bool",  "default": True},
        "distortion":  {"type": "list",  "default": None},
    }
    OUTPUTS: dict[str, Any] = {
        "result": {"type": "sequence"},
        "aligned_exifs": {"type": "sequence"},
    }

    def _infer_output_length(self, input_lengths):
        return None

    async def _async_execute(self, configs: dict[str, Any]) -> None:
        reference = configs.get('reference')
        method = configs.get('method', 'auto')
        same_camera = configs.get('same_camera', True)
        init_distortion = configs.get('distortion')

        exifs_active = self.inputs['exifs'].active

        ref_geo: Optional[GeometryView] = None
        ref_arr: Optional[np.ndarray] = None
        ref_camera: Optional[CameraModel] = None
        aligned_count = 0
        skipped_count = 0

        if reference is not None:
            ref_arr = reference.data if isinstance(reference, FloatImage) else reference
            ref_geo = await self._run_cpu(make_geometry, ref_arr)

        for i in self._input_range():
            data = self._async_convert_inputs()
            try:
                frame = await data['data']
            except StreamExhausted:
                break

            # 消费 EXIF 并拆包为 dict（Op 层负责 ExifData → dict 转换）
            exif_tags = None
            exif_obj = None
            if exifs_active:
                try:
                    exif_obj = await data['exifs']
                    exif_tags = exif_obj.exif if exif_obj is not None else None
                except StreamExhausted:
                    pass

            frame_arr = frame.data if isinstance(frame, FloatImage) else frame

            # 首帧：设定参考 + 确定路径
            if ref_geo is None:
                ref_arr = frame_arr
                ref_geo = await self._run_cpu(make_geometry, ref_arr)
                ref_camera = try_build_camera(
                    exif_tags, ref_arr.shape, method, init_distortion)
                if ref_camera:
                    logger.info(
                        f"{self.name}: camera model path enabled "
                        f"(focal={ref_camera.intrinsics.focal_length_mm:.1f}mm)")
                else:
                    logger.info(f"{self.name}: using homography path")
                await self._broadcast_outputs(
                    {"result": frame, "aligned_exifs": exif_obj})
                aligned_count += 1
                continue

            # 后续帧：对齐
            try:
                src_camera = try_build_camera(
                    exif_tags, frame_arr.shape, method, init_distortion)

                if ref_camera and src_camera and method != "homography":
                    aligned_arr = await self._run_cpu(
                        align_frame_camera_model,
                        frame_arr, ref_geo, ref_arr,
                        ref_camera, src_camera, same_camera)
                else:
                    aligned_arr = await self._run_cpu(
                        align_frame_homography, frame_arr, ref_geo, ref_arr)

                aligned = (FloatImage(data=aligned_arr, dtype=frame.dtype)
                           if isinstance(frame, FloatImage) else aligned_arr)
                await self._broadcast_outputs(
                    {"result": aligned, "aligned_exifs": exif_obj})
                aligned_count += 1
            except AlignmentError as e:
                skipped_count += 1
                logger.warning(
                    f"{self.name}: frame {i} alignment failed ({e}), skipping")

        logger.info(
            f"{self.name}: aligned {aligned_count} frames, "
            f"skipped {skipped_count}")
