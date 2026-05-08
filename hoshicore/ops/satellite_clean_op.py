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
from ..component.norma.alignment import match_star_pairs
from ..component.queue import StreamExhausted
from ..engine.registry import register_op
from .base import BaseOp


@dataclasses.dataclass
class _FrameSlot:
    original: np.ndarray
    geo: GeometryView
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
        "window_size": {"type": "int", "default": 2},
    }
    OUTPUTS: dict[str, Any] = {
        "result": {"type": "sequence"},
    }

    def _infer_output_length(self, input_lengths):
        return input_lengths.get('data')

    async def _async_execute(self, configs: dict[str, Any]) -> None:
        W: int = configs['window_size']
        tot_num = self.length

        if tot_num is not None:
            self.tracker.create_bar(self.name, tot_num)

        buffer: deque[_FrameSlot] = deque()
        output_count = 0

        try:
            for _ in self._input_range():
                upper = self._async_convert_inputs()
                try:
                    frame = await upper['data']
                except StreamExhausted:
                    break

                frame_arr = frame.data if isinstance(frame, FloatImage) else frame
                geo = await self._run_cpu(make_geometry, frame_arr)

                slot = _FrameSlot(original=frame_arr, geo=geo)
                if buffer:
                    H = await self._run_cpu(
                        self._compute_homography, buffer[-1].geo, geo)
                    buffer[-1].H_to_next = H

                buffer.append(slot)

                if len(buffer) == 2 * W + 1:
                    cleaned = await self._run_cpu(
                        self._process_center, buffer, W)
                    out = self._wrap_output(cleaned, frame)
                    await self._broadcast_outputs({"result": out})
                    buffer.popleft()
                    output_count += 1
                    if tot_num is not None:
                        self.tracker.update(self.name)

            # Flush remaining frames in buffer
            while buffer:
                actual_W = (len(buffer) - 1) // 2
                if actual_W >= 1:
                    center_pos = actual_W
                    cleaned = await self._run_cpu(
                        self._process_center, buffer, center_pos)
                    out = self._wrap_output(cleaned, frame)
                else:
                    out = buffer[0].original
                    if isinstance(frame, FloatImage):
                        out = FloatImage(data=out, dtype=frame.dtype)
                await self._broadcast_outputs({"result": out})
                buffer.popleft()
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
    def _process_center(buffer: deque, center_pos: int) -> np.ndarray:
        center = buffer[center_pos]
        h, w = center.original.shape[:2]

        aligned_all = [center.original]
        for offset in range(-center_pos, len(buffer) - center_pos):
            if offset == 0:
                continue
            pos = center_pos + offset
            if pos < 0 or pos >= len(buffer):
                continue
            H = SatelliteCleanOp._chain_homography(buffer, pos, center_pos)
            if H is None:
                continue
            aligned = cv2.warpPerspective(
                buffer[pos].original, H, (w, h),
                borderMode=cv2.BORDER_REPLICATE)
            aligned_all.append(aligned)

        if len(aligned_all) == 1:
            return center.original

        result = np.median(
            np.stack(aligned_all, axis=0).astype(np.float32), axis=0)
        return result.astype(center.original.dtype)

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
        prev_geo: GeometryView, curr_geo: GeometryView
    ) -> np.ndarray:
        try:
            match = match_star_pairs(
                prev_geo.unit_vectors, curr_geo.unit_vectors,
                prev_geo.volumes, curr_geo.volumes,
                prev_geo.positions, curr_geo.positions,
            )
            return match.init_homography
        except Exception as e:
            logger.warning(
                f"Satellite clean: homography failed ({e}), using identity")
            return np.eye(3, dtype=np.float64)
