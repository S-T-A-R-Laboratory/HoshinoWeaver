from typing import Any, Optional

import cv2
import numpy as np
from loguru import logger

from ..component.frame_buffer import DiskFrameBuffer
from ..component.merger import (MaxMerger, MeanMerger, MinMerger,
                                SigmaClippingMerger)
from ..component.noise_equalization import equalize_noise
from ..component.tagged_image import FloatImage, align_dtype_pair
from ..component.utils import FastGaussianParam
from ..engine.registry import register_op
from ..component.queue import StreamExhausted
from .base import BaseOp


@register_op()
class TrailStackerOp(BaseOp):
    """
    叠加星轨
    """
    EXECUTOR = "cpu"
    INPUTS: dict[str, dict[str, Any]] = {
        "data": {
            "type": "sequence",
            "required": True
        },
        "weight": {
            "type": "sequence",
            "required": False
        },
    }
    CONFIGS: dict[str, dict[str, Any]] = {
        "int_weight": {
            "type": "bool",
            "default": False
        }
    }
    OUTPUTS = {
        "result": {
            "type": "image"
        },
    }
    MERGER = MaxMerger
    MAX_SIZE: int = 1

    async def _async_execute(self, configs: dict[str, Any]) -> None:
        int_weight: bool = configs['int_weight']
        merger = self.MERGER(int_weight=int_weight)
        tot_num = self.length
        assert tot_num is not None, "TrailStackerOp requires sequence length information."

        has_weight = self.inputs['weight'].active

        stacked_num = 0
        failed_num = 0
        err_msg_collector = []

        self.tracker.create_bar(self.name, tot_num)

        try:
            for i in range(tot_num):
                cur_filename = f"the {i+1}-th frame"

                try:
                    upper_stream_data = self._async_convert_inputs()
                    cur_img = await upper_stream_data['data']
                    weight = (await upper_stream_data['weight']
                              ) if has_weight else None
                except StreamExhausted:
                    logger.warning(
                        f"{self.name}: upstream ended at {i}/{tot_num}")
                    break

                # Empty result handling
                if cur_img is None:
                    warning_msg = f"{self.name} failed to load {cur_filename}."
                    err_msg_collector.append(warning_msg)
                    logger.warning(warning_msg)
                    logger.warning(f"Skip {cur_filename}.")
                    failed_num += 1
                    self.tracker.update(self.name)
                    continue

                try:
                    await self._run_cpu(merger.merge, cur_img, weight)
                except AssertionError as e:
                    err_msg_collector.append(
                        f"Shape of {cur_filename} does not match.")
                    raise e
                stacked_num += 1
                self.tracker.update(self.name)

            if stacked_num == 0:
                raise ValueError(
                    f"{self.name}: No valid frames loaded from {tot_num} inputs."
                )

            logger.info(
                f"{self.name} successfully stacked {stacked_num} " +
                f"images from {tot_num} images. ({failed_num} fail(s)).")

            # 输出结果
            outputs: dict[str, Any] = {"result": merger.merged_image}
            if "statistics" in self.OUTPUTS:
                merger.result.inplace_calc = False  # 输出前关闭 inplace_calc，避免下游误用导致数据被修改
                outputs["statistics"] = merger.result
            await self._broadcast_outputs(outputs)

        except Exception as e:
            logger.error(f"{self.name} failed: {e}")
            raise
        finally:
            self.tracker.close_bar(self.name)


@register_op()
class MinStackerOp(TrailStackerOp):
    MERGER = MinMerger


@register_op()
class MeanStackerOp(TrailStackerOp):
    MERGER = MeanMerger
    OUTPUTS = {
        "result": {
            "type": "image"
        },
        "statistics": {
            "type": "image"  # FastGaussianParam，不连接时静默忽略
        },
    }


@register_op()
class MaxNoiseEqualizationOp(BaseOp):
    """最大值叠加噪声均匀化算子。

    接收：
        - max_img: 最大值叠加结果（来自 TrailStackerOp）
        - statistics: FastGaussianParam（来自 SigmaClippingStackerOp）

    输出校正后的最大值图像。
    """
    EXECUTOR = "cpu"
    CONFIGS: dict[str, dict[str, Any]] = {
        "max_img": {
            "type": "image",
            "required": True
        },
        "statistics": {
            "type": "image",
            "required": True
        },
        "mask": {
            "type": "image",
            "required": False,
            "default": None
        },
        "minus_only": {
            "type": "bool",
            "required": False,
            "default": False,
        },
        "top_fraction": {
            "type": "float",
            "default": 0.02
        },
        "sigma_reject": {
            "type": "float",
            "default": 3.0
        },
        "highlight_preserve": {
            "type": "float",
            "default": 0.9
        },
    }
    OUTPUTS = {"result": {"type": "image"}}

    async def _async_execute(self, configs: dict[str, Any]) -> None:
        top_fraction: float = configs['top_fraction']
        max_raw = configs['max_img']
        accepted: FastGaussianParam = configs['statistics']
        minus_only: bool = configs['minus_only']
        sigma_reject: float = configs['sigma_reject']
        highlight_preserve: float = configs['highlight_preserve']
        try:

            # ── dtype 对齐 ──
            # max_raw 来自 TrailStackerOp，其 dtype 即语义级别（可能被 int_weight 放缩）
            # accepted.source_dtype 来自 SigmaClippingStackerOp 的 FGP 内部记录
            # 若两者级别不同（如一侧 int_weight=True 另一侧 False），需要放缩到同一范围
            max_aligned, mean_aligned, output_dtype = align_dtype_pair(
                max_raw,
                max_raw.dtype,
                accepted.mu,
                accepted.source_dtype,
            )
            if output_dtype != max_raw.dtype or output_dtype != accepted.source_dtype:
                logger.info(
                    f"{self.name} dtype alignment: max_img {max_raw.dtype} + "
                    f"statistics {accepted.source_dtype} → {output_dtype}")

            max_img = max_aligned.astype(np.float64)
            mean_img = mean_aligned.astype(np.float64)
            std_img = np.sqrt(np.maximum(accepted.var, 0).astype(np.float64))
            n_img = accepted.n
            corrected = equalize_noise(max_img,
                                       mean_img,
                                       std_img,
                                       n_img,
                                       minus_only=minus_only,
                                       top_fraction=top_fraction,
                                       sigma_reject=sigma_reject,
                                       highlight_preserve=highlight_preserve)
            mask: Optional[np.ndarray] = configs['mask']
            if mask is not None:
                # mask 尺寸修正
                mask = cv2.resize(mask,
                                  (corrected.shape[1], corrected.shape[0]),
                                  interpolation=cv2.INTER_CUBIC)
                result = (corrected * mask + mean_img *
                          (1 - mask)).astype(output_dtype)
            else:
                result = np.round(corrected).astype(output_dtype)

            await self._broadcast_outputs({"result": result})

            logger.info(f"{self.name} complete.")

        except Exception as e:
            logger.error(f"{self.name} failed: {e}")
            raise
