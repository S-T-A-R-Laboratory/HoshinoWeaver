"""
对齐算子：星点对齐等帧间配准操作。

StarAlignmentOp 实现基于 norma 的 2D 单应性路径：
  detect_star_points → 近似单位向量 → match_star_pairs → warpPerspective

对齐失败的帧被丢弃，输出为变长序列（sentinel 驱动）。
"""
from typing import Any

import cv2
import numpy as np
from loguru import logger

from ..component.norma.alignment import match_star_pairs
from ..component.norma.cache import GeometryView, StarDetectionCache
from ..component.queue import StreamExhausted
from ..engine.registry import register_op
from ..component.data_container import FloatImage
from .base import BaseOp


def _to_gray_f64(arr: np.ndarray) -> np.ndarray:
    """Convert image array to grayscale float64 in [0, 1] range.

    Handles any input dtype: integer arrays are divided by their dtype max,
    float arrays with values > 1 are divided by their actual max.
    """
    if arr.ndim == 3:
        gray = cv2.cvtColor(arr.astype(np.float32), cv2.COLOR_RGB2GRAY).astype(np.float64)
    else:
        gray = arr.astype(np.float64)

    if np.issubdtype(arr.dtype, np.integer):
        gray /= np.iinfo(arr.dtype).max
    else:
        max_val = gray.max()
        if max_val > 1.0:
            gray /= max_val

    return gray


def _make_geometry(arr: np.ndarray) -> GeometryView:
    """Build a flat-projection GeometryView from a raw image array."""
    gray = _to_gray_f64(arr)
    cache = StarDetectionCache(gray)
    return GeometryView.from_flat_projection(cache)


class AlignmentError(Exception):
    """对齐失败异常。"""
    pass


@register_op()
class StarAlignmentOp(BaseOp):
    """星点对齐：将序列帧对齐到参考帧。

    参考帧可通过 reference config 传入（来自上游或用户指定）。
    若 reference 未连接（为 None），则自动使用第一帧作为参考帧。

    对齐失败的帧被丢弃（不输出），因此输出为变长序列。
    """

    EXECUTOR = "cpu"
    VARIABLE_OUTPUT = True
    INPUTS: dict[str, Any] = {
        "data": {"type": "sequence"},
    }
    CONFIGS: dict[str, Any] = {
        "reference": {"type": "image", "default": None},
        "method":    {"type": "str",   "default": "auto"},
    }
    OUTPUTS: dict[str, Any] = {
        "result": {"type": "sequence"},
    }

    def _infer_output_length(self, input_lengths):
        return None  # sentinel 驱动

    async def _async_execute(self, configs: dict[str, Any]) -> None:
        reference = configs.get('reference')
        ref_geo: GeometryView | None = None
        ref_arr: np.ndarray | None = None
        aligned_count = 0
        skipped_count = 0

        if reference is not None:
            ref_arr = reference.data if isinstance(reference, FloatImage) else reference
            ref_geo = await self._run_cpu(_make_geometry, ref_arr)

        for i in self._input_range():
            data = self._async_convert_inputs()
            try:
                frame = await data['data']
            except StreamExhausted:
                break

            frame_arr = frame.data if isinstance(frame, FloatImage) else frame

            # 未指定参考帧时，使用第一帧
            if ref_geo is None:
                ref_arr = frame_arr
                ref_geo = await self._run_cpu(_make_geometry, ref_arr)
                await self._broadcast_outputs({"result": frame})
                aligned_count += 1
                continue

            try:
                aligned_arr = await self._run_cpu(
                    self._align, frame_arr, ref_geo, ref_arr)
                aligned = FloatImage(data=aligned_arr, dtype=frame.dtype) if isinstance(frame, FloatImage) else aligned_arr
                await self._broadcast_outputs({"result": aligned})
                aligned_count += 1
            except (AlignmentError, NotImplementedError) as e:
                skipped_count += 1
                logger.warning(
                    f"{self.name}: frame {i} alignment failed ({e}), skipping")

        logger.info(
            f"{self.name}: aligned {aligned_count} frames, "
            f"skipped {skipped_count}")

    def _align(self, frame: np.ndarray, ref_geo: GeometryView,
               reference: np.ndarray) -> np.ndarray:
        """将 frame 对齐到 reference（2D 单应性路径）。

        ref_geo 为预计算的参考帧 GeometryView，避免每帧重复检测星点。
        """
        src_geo = _make_geometry(frame)

        if len(ref_geo.positions) < 20 or len(src_geo.positions) < 20:
            raise AlignmentError(
                f"Insufficient stars: ref={len(ref_geo.positions)}, "
                f"src={len(src_geo.positions)} (need ≥20)")

        try:
            match = match_star_pairs(
                ref_geo.unit_vectors, src_geo.unit_vectors,
                ref_geo.volumes, src_geo.volumes,
                ref_geo.positions, src_geo.positions,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise AlignmentError(f"Star matching failed: {e}") from e

        h, w = reference.shape[:2]
        H = np.linalg.inv(match.init_homography)
        return cv2.warpPerspective(frame, H, (w, h))
