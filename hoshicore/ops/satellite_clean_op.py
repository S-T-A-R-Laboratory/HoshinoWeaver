"""卫星线去除算子：滑动窗口对齐中位数法。"""
import dataclasses
from collections import deque
from typing import Any, Optional

import cv2
import numpy as np
from loguru import logger

from ..component.data_container import FloatImage
from ..component.norma.cache import GeometryView
from ..component.norma.frame_align import make_geometry
from ..component.norma.alignment import match_star_pairs_from_geo
from ..component.queue import StreamExhausted
from .._custom_op.ops.median import median_reduce_chunk
from ..engine.registry import register_op
from .base import BaseOp


@dataclasses.dataclass
class _FrameSlot:
    original: np.ndarray
    geo: Optional[GeometryView]
    H_to_next: Optional[np.ndarray] = None


@register_op()
class SatelliteCleanOp(BaseOp):
    """滑动窗口卫星线去除。

    将前后 W 帧对齐到当前帧坐标系，输出所有帧的逐像素中位数。
    中位数天然排斥单帧异常（卫星线），保留多帧一致信号（星点）。
    """

    EXECUTOR = "cpu"
    INPUTS: dict[str, Any] = {
        "data": {"type": "sequence", "required": True},
    }
    CONFIGS: dict[str, Any] = {
        "window_size": {"type": "int", "default": 3},
        "mask": {"type": "image", "default": None},
    }
    OUTPUTS: dict[str, Any] = {
        "result": {"type": "sequence"},
    }

    @classmethod
    def estimate_resources(cls, configs, frame_bytes, n_frames):
        # deque 持有 W 帧 + _process_center 中 W 帧对齐副本用于 median
        # TODO: 对齐的资源开销未计入
        w = configs.get("window_size", 3)
        return (w * 2 * frame_bytes, 0)

    def _infer_output_length(self, input_lengths):
        return input_lengths.get('data')

    async def _async_execute(self, configs: dict[str, Any]) -> None:
        W: int = configs['window_size']
        mask: Optional[np.ndarray] = configs['mask']
        if mask is not None:
            if mask.ndim == 3:
                mask = mask.mean(axis=2)
            mask = (mask > 0.5).astype(np.uint8)
        tot_num = self.length

        assert W >= 1 and W % 2 == 1, "window_size must be an odd integer >= 1"
        half_W = (W - 1) // 2
        
        if tot_num is not None:
            self.tracker.create_bar(self.name, tot_num)

        buffer: deque[_FrameSlot] = deque()
        output_count = 0

        try:
            for i in self._input_range():
                upper = self._async_convert_inputs()
                try:
                    frame = await upper['data']
                except StreamExhausted:
                    break

                frame_arr = frame.data if isinstance(frame, FloatImage) else frame
                try:
                    geo = await self._run_cpu(make_geometry, frame_arr, mask)
                except Exception as e:
                    logger.warning(
                        f"{self.name}: star extraction failed, frame will not be aligned ({e})")
                    geo = None

                slot = _FrameSlot(original=frame_arr, geo=geo)
                if buffer:
                    H = await self._run_cpu(
                        self._compute_homography, buffer[-1].geo, geo)
                    buffer[-1].H_to_next = H
                    if H is None:
                        logger.debug(f"Fail to compute homography for frame {i}.")

                # only pop when next frame is ready and buffer is full 
                # this ensures the residual frames in buffer to be enough, and can still be processed after input is exhausted
                if len(buffer) >= W:
                    buffer.popleft()
                buffer.append(slot)

                if len(buffer) == W:
                    cleaned = await self._run_cpu(
                        self._process_center, buffer, half_W, mask)
                    out = self._wrap_output(cleaned, frame)
                    await self._broadcast_outputs({"result": out})
                    output_count += 1
                    if tot_num is not None:
                        self.tracker.update(self.name)

            # Flush remaining frames in buffer
            res_center_pos = (len(buffer) - 1) // 2
            while res_center_pos < len(buffer):
                cleaned = await self._run_cpu(
                    self._process_center, buffer, res_center_pos, mask)
                out = self._wrap_output(cleaned, frame)
                await self._broadcast_outputs({"result": out})
                res_center_pos += 1
                output_count += 1
                if tot_num is not None:
                    self.tracker.update(self.name)

            logger.info(
                f"{self.name}: processed {output_count} frames with window={W}")

        finally:
            if tot_num is not None:
                self.tracker.close_bar(self.name)

    @staticmethod
    def _wrap_output(arr: np.ndarray, ref_frame) -> Any:
        if isinstance(ref_frame, FloatImage):
            return FloatImage(data=arr, dtype=ref_frame.dtype)
        return arr

    @staticmethod
    def _process_center(
        buffer: deque, center_pos: int, mask: Optional[np.ndarray]
    ) -> np.ndarray:
        center = buffer[center_pos]
        h, w = center.original.shape[:2]

        aligned_all = [center.original]
        original_all = [center.original]
        for pos in range(len(buffer)):
            if pos == center_pos:
                continue
            H = SatelliteCleanOp._chain_homography(buffer, pos, center_pos)
            if H is None:
                continue
            aligned = cv2.warpPerspective(
                buffer[pos].original, H, (w, h),
                borderMode=cv2.BORDER_REPLICATE)
            aligned_all.append(aligned)
            original_all.append(buffer[pos].original)

        if len(aligned_all) == 1:
            return center.original

        if mask is None:
            sky_stack = np.stack(aligned_all, axis=0)
            return median_reduce_chunk(sky_stack)

        sky_stack = np.stack(aligned_all, axis=0)
        ground_stack = np.stack(original_all, axis=0)
        sky_median = median_reduce_chunk(sky_stack)
        ground_median = median_reduce_chunk(ground_stack)

        if sky_median.ndim == 3:
            mask_3d = mask[:, :, np.newaxis]
        else:
            mask_3d = mask
        result = np.where(mask_3d, sky_median, ground_median)
        return result

    @staticmethod
    def _chain_homography(
        buffer: deque, from_pos: int, to_pos: int
    ) -> Optional[np.ndarray]:
        if from_pos == to_pos:
            return np.eye(3, dtype=np.float64)

        if from_pos < to_pos:
            H = np.eye(3, dtype=np.float64)
            for k in range(from_pos, to_pos):
                H_k = buffer[k].H_to_next
                if H_k is None:
                    return None
                H = H_k @ H
            return H
        else:
            H_forward = SatelliteCleanOp._chain_homography(
                buffer, to_pos, from_pos)
            if H_forward is None:
                return None
            return np.linalg.inv(H_forward)

    @staticmethod
    def _compute_homography(
        prev_geo: Optional[GeometryView], curr_geo: Optional[GeometryView]
    ) -> Optional[np.ndarray]:
        if prev_geo is None or curr_geo is None:
            return None
        try:
            match = match_star_pairs_from_geo(prev_geo, curr_geo)
            return match.init_homography
        except Exception as e:
            logger.warning(
                f"Satellite clean: homography failed ({e}), frame link broken")
            return None
