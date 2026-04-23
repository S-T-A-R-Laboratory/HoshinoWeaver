"""
对齐算子：星点对齐等帧间配准操作。

StarAlignmentOp 为接口预留，核心对齐实现后续接入。
对齐失败的帧被丢弃，输出为变长序列（sentinel 驱动）。
"""
from typing import Any

import numpy as np
from loguru import logger

from ..component.queue import StreamExhausted
from ..engine.registry import register_op
from ..component.data_container import FloatImage
from .base import BaseOp


class AlignmentError(Exception):
    """对齐失败异常。"""
    pass


@register_op()
class StarAlignmentOp(BaseOp):
    """星点对齐：将序列帧对齐到参考帧。

    参考帧可通过 reference config 传入（来自上游或用户指定）。
    若 reference 未连接（为 None），则自动使用第一帧作为参考帧。

    对齐失败的帧被丢弃（不输出），因此输出为变长序列。

    当前为接口预留，_align 方法 raise NotImplementedError。
    接入实际对齐实现时，可通过子类覆盖 _align 方法。
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
        method: str = configs['method']
        reference = configs.get('reference')
        aligned_count = 0
        skipped_count = 0

        # 提取参考帧的裸数组（供 _align 使用）
        if reference is not None:
            ref_arr = reference.data if isinstance(
                reference, FloatImage) else reference

        for i in self._input_range():
            data = self._async_convert_inputs()
            try:
                frame = await data['data']
            except StreamExhausted:
                break

            # 未指定参考帧时，使用第一帧
            if reference is None:
                reference = frame
                ref_arr = frame.data if isinstance(
                    frame, FloatImage) else frame
                await self._broadcast_outputs({"result": frame})
                aligned_count += 1
                continue

            try:
                if isinstance(frame, FloatImage):
                    aligned_arr = await self._run_cpu(
                        self._align, frame.data, ref_arr, method)
                    aligned = FloatImage(data=aligned_arr, dtype=frame.dtype)
                else:
                    aligned = await self._run_cpu(
                        self._align, frame, ref_arr, method)

                await self._broadcast_outputs({"result": aligned})
                aligned_count += 1
            except (AlignmentError, NotImplementedError) as e:
                skipped_count += 1
                logger.warning(
                    f"{self.name}: frame {i} alignment failed ({e}), skipping")

        logger.info(
            f"{self.name}: aligned {aligned_count} frames, "
            f"skipped {skipped_count}")

    def _align(self, frame: np.ndarray, reference: np.ndarray,
               method: str) -> np.ndarray:
        """将 frame 对齐到 reference。

        子类覆盖此方法以接入实际对齐算法。

        Args:
            frame: 待对齐帧。
            reference: 参考帧。
            method: 对齐方法名称。

        Returns:
            对齐后的帧。

        Raises:
            AlignmentError: 对齐失败（特征点不足等）。
            NotImplementedError: 对齐模块未安装。
        """
        raise NotImplementedError(
            "Star alignment module not available. "
            "Please provide an alignment implementation by subclassing "
            "StarAlignmentOp and overriding _align().")
