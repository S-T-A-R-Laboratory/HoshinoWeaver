"""
拆解后的 Sigma Clipping 子图组件：DiskBufferWriterOp + SigmaClipIteratorOp。

原 SigmaClippingStackerOp 被拆分为三个阶段：

    MeanStackerOp (已有)
        ↓ result (image) + statistics (FastGaussianParam)
    DiskBufferWriterOp
        ↓ buffer_handle (DiskFrameBuffer 实例)
    SigmaClipIteratorOp
        ↓ result (image) + statistics (FastGaussianParam)

DiskBufferWriterOp：
    消费序列输入，逐帧写入 DiskFrameBuffer，完成后将 buffer 实例推送给下游。
    清理策略：
        - 正常完成：buffer 由下游 SigmaClipIteratorOp 在 finally 中清理
        - 自身异常：在 except 中立即清理，防止泄漏
        - 用户中断 / 未捕获异常：DiskFrameBuffer.__del__ 安全网兜底

SigmaClipIteratorOp：
    接收 buffer_handle + mean FGP，执行迭代 sigma clipping。
    清理策略：
        - 在 finally 中无条件清理 buffer，确保不泄漏
"""
from typing import Any, Optional

import numpy as np
from loguru import logger

from ..component.frame_buffer import DiskFrameBuffer
from ..component.merger import MeanMerger, SigmaClippingMerger
from ..component.tagged_image import FloatImage
from ..component.utils import FastGaussianParam
from ..engine.registry import register_op
from .base import BaseOp


@register_op()
class DiskBufferWriterOp(BaseOp):
    """将序列帧写入磁盘缓冲区，供下游多 pass 算法重放。

    消费 data（必选）和 weight（可选）序列，逐帧写入 DiskFrameBuffer。
    完成后通过 buffer_handle 端口将 DiskFrameBuffer 实例推给下游。
    """

    EXECUTOR = "cpu"
    INPUTS: dict[str, dict[str, Any]] = {
        "data": {
            "type": "sequence",
            "required": True,
        },
        "weight": {
            "type": "sequence",
            "required": False,
        },
    }
    OUTPUTS = {
        "buffer_handle": {
            "type": "image",  # DiskFrameBuffer 实例，单次传递
        },
    }

    async def _async_execute(self, configs: dict[str, Any]) -> None:
        tot_num = self.length
        assert tot_num is not None, \
            "DiskBufferWriterOp requires sequence length information."

        has_weight = self.inputs['weight'].active
        frame_buffer = DiskFrameBuffer()
        stacked_num = 0
        failed_num = 0

        self.tracker.create_bar(self.name,
                                tot_num,
                                desc=f"{self.name} [Buffer]")
        try:
            for i in range(tot_num):
                cur_filename = f"the {i + 1}-th frame"
                try:
                    upper = self._async_convert_inputs()
                    cur_img = await upper['data']
                    weight = (await upper['weight']) if has_weight else None
                except StopIteration:
                    logger.warning(
                        f"{self.name}: upstream ended at {i}/{tot_num}")
                    break

                if cur_img is None:
                    logger.warning(
                        f"{self.name} failed to load {cur_filename}, skip.")
                    failed_num += 1
                    self.tracker.update(self.name)
                    continue

                frame_buffer.append(cur_img, weight)
                stacked_num += 1
                self.tracker.update(self.name)

            if stacked_num == 0:
                logger.warning(f"{self.name}: No valid frames buffered!")
                frame_buffer.cleanup()
                return

            logger.info(
                f"{self.name}: buffered {stacked_num}/{tot_num} frames "
                f"({failed_num} fail(s)).")

            # 将 buffer 实例推给下游
            await self._broadcast_outputs({"buffer_handle": frame_buffer})

        except Exception as e:
            # 自身异常：立即清理 buffer 防止磁盘泄漏
            logger.error(f"{self.name} failed: {e}")
            frame_buffer.cleanup()
            raise
        finally:
            self.tracker.close_bar(self.name)


@register_op()
class SigmaClipIteratorOp(BaseOp):
    """迭代式 Sigma Clipping：基于 mean FGP 和磁盘缓冲帧进行多 pass 迭代。

    接收：
        - fgp_total: FastGaussianParam（来自 MeanStackerOp.statistics）
        - buffer_handle: DiskFrameBuffer 实例（来自 DiskBufferWriterOp）
        - rej_high / rej_low / max_iter / early_converge_ratio 配置

    输出：
        - result: sigma clipping 后的均值图像 (FloatImage)
        - statistics: accepted FastGaussianParam
    """

    EXECUTOR = "cpu"
    CONFIGS: dict[str, dict[str, Any]] = {
        "fgp_total": {
            "type": "image",
            "required": True,
        },
        "buffer_handle": {
            "type": "image",
            "required": True,
        },
        "rej_high": {
            "type": "float",
            "default": 3.0,
        },
        "rej_low": {
            "type": "float",
            "default": 3.0,
        },
        "max_iter": {
            "type": "int",
            "default": 5,
        },
        "early_converge_ratio": {
            "type": "float",
            "default": 0.99,
        },
    }
    OUTPUTS = {
        "result": {
            "type": "image",
        },
        "statistics": {
            "type": "image",  # FastGaussianParam
        },
    }

    async def _async_execute(self, configs: dict[str, Any]) -> None:
        fgp_total: FastGaussianParam = configs['fgp_total']
        frame_buffer: DiskFrameBuffer = configs['buffer_handle']
        rej_high: float = configs['rej_high']
        rej_low: float = configs['rej_low']
        max_iter: int = configs['max_iter']
        early_converge_ratio: float = configs['early_converge_ratio']

        try:
            ref_fgp = fgp_total
            last_n = ref_fgp.n.copy()
            accepted = None

            for iteration in range(max_iter):
                clip_merger = SigmaClippingMerger(
                    ref_img=ref_fgp,
                    rej_high=rej_high,
                    rej_low=rej_low,
                )
                self.tracker.create_bar(
                    self.name,
                    len(frame_buffer),
                    desc=f"{self.name} [Clip {iteration + 1}]")

                for idx in range(len(frame_buffer)):
                    raw, weight = frame_buffer[idx]
                    await self._run_cpu(clip_merger.merge, raw, weight)
                    self.tracker.update(self.name)

                self.tracker.close_bar(self.name)

                # accepted = fgp_total - rejected
                accepted = fgp_total - clip_merger.result
                accepted.apply_zero_var(fgp_total)

                # 收敛检查
                cur_n = accepted.n
                converge_ratio = (np.sum(cur_n == last_n) /
                                  np.prod(cur_n.shape))
                if converge_ratio >= early_converge_ratio:
                    logger.info(f"{self.name} converged at iteration "
                                f"{iteration + 1}.")
                    break
                else:
                    logger.info(f"{self.name} converge ratio: "
                                f"{converge_ratio * 100:.2f}%")
                last_n = cur_n.copy()
                ref_fgp = accepted
                logger.info(
                    f"{self.name} iteration {iteration + 1}/{max_iter} done.")
            else:
                logger.info(
                    f"{self.name} reached max iterations ({max_iter}).")

            # 输出
            result = FloatImage(accepted.mu, dtype=accepted.source_dtype)
            await self._broadcast_outputs({
                "result": result,
                "statistics": accepted,
            })

            logger.info(f"{self.name} sigma clipping complete.")

        except Exception as e:
            logger.error(f"{self.name} failed: {e}")
            raise
        finally:
            # 无条件清理 buffer：无论成功、失败还是中断
            frame_buffer.cleanup()
