from asyncio import gather
from typing import Any, Optional

import numpy as np
from loguru import logger

from ..component.frame_buffer import DiskFrameBuffer
from ..component.merger import MaxMerger, MinMerger, MeanMerger, SigmaClippingMerger
from ..component.noise_equalization import equalize_noise
from ..component.tagged_image import FloatImage, align_dtype_pair
from ..component.utils import FastGaussianParam
from .base import BaseOp


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
                except StopIteration:
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
                logger.warning(f"No valid frames are loaded!")
                return

            logger.info(
                f"{self.name} successfully stacked {stacked_num} " +
                f"images from {tot_num} images. ({failed_num} fail(s)).")

            # 输出结果
            put_tasks = []
            for queue in self.outputs['result']:
                put_tasks.append(queue.put(merger.merged_image))
            await gather(*put_tasks)

        except Exception as e:
            logger.error(f"{self.name} failed: {e}")
            raise
        finally:
            self.tracker.close_bar(self.name)


class MinStackerOp(TrailStackerOp):
    MERGER = MinMerger


class MeanStackerOp(TrailStackerOp):
    MERGER = MeanMerger


class SigmaClippingStackerOp(BaseOp):
    """迭代式 Sigma Clipping 均值叠加。

    算法流程：
        Pass 0: 消费输入队列，同时做 MeanMerger 累加得到全帧均值参考 (FGP_TOTAL)，
                帧数据写入磁盘缓冲以供后续 pass 重放。
        Pass 1~K: 以上一迭代的 accepted FGP 为参考，构造 SigmaClippingMerger
                  计算拒绝阈值 (μ ± kσ)，遍历缓冲帧累加 rejected 像素，
                  然后 ref_fgp - rejected = accepted。
                  收敛条件：相邻两次迭代的 per-pixel accepted count 不变。

    对外接口与 TrailStackerOp 一致（消费 sequence 输入，输出单张 image）。
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
            "default": True
        },
        "rej_high": {
            "type": "float",
            "default": 3.0
        },
        "rej_low": {
            "type": "float",
            "default": 3.0
        },
        "max_iter": {
            "type": "int",
            "default": 5
        },
        "early_converge_ratio": {
            "type": "float",
            "default": 0.99
        },
    }
    OUTPUTS = {
        "result": {
            "type": "image"
        },
        "statistics": {
            "type": "image"  # FastGaussianParam，不连接时静默忽略
        },
    }
    MAX_SIZE: int = 1

    async def _async_execute(self, configs: dict[str, Any]) -> None:
        int_weight: bool = configs['int_weight']
        rej_high: float = configs['rej_high']
        rej_low: float = configs['rej_low']
        max_iter: int = configs['max_iter']
        early_converge_ratio: float = configs['early_converge_ratio']
        has_weight = self.inputs['weight'].active
        tot_num = self.length
        assert tot_num is not None, \
            "SigmaClippingStackerOp requires sequence length information."

        # ── Phase 1: 消费队列 + 磁盘缓冲 + Pass 0 (Mean) ──
        mean_merger = MeanMerger(int_weight=int_weight)
        frame_buffer = DiskFrameBuffer()
        stacked_num = 0
        failed_num = 0

        self.tracker.create_bar(self.name, tot_num, desc=f"{self.name} [Mean]")

        try:
            for i in range(tot_num):
                cur_filename = f"the {i + 1}-th frame"
                try:
                    upper_stream_data = self._async_convert_inputs()
                    cur_img = await upper_stream_data['data']
                    weight = (await upper_stream_data['weight']
                              ) if has_weight else None
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
                await self._run_cpu(mean_merger.merge, cur_img, weight)
                stacked_num += 1
                self.tracker.update(self.name)

            if stacked_num == 0:
                logger.warning(f"{self.name}: No valid frames are loaded!")
                frame_buffer.cleanup()
                return

            logger.info(
                f"{self.name} Pass 0 (Mean): stacked {stacked_num}/{tot_num} "
                f"frames. ({failed_num} fail(s)).")

            # Pass 0 结果
            fgp_total = mean_merger.result  # FastGaussianParam

            # ── Phase 2: 迭代 Sigma Clipping ──
            ref_fgp = fgp_total
            last_n = ref_fgp.n.copy()  # 收敛检查
            accepted = None

            for iteration in range(max_iter):
                clip_merger = SigmaClippingMerger(ref_img=ref_fgp,
                                                  rej_high=rej_high,
                                                  rej_low=rej_low)
                self.tracker.reset_bar(
                    self.name,
                    len(frame_buffer),
                    desc=f"{self.name} [Clip {iteration + 1}]")

                for idx in range(len(frame_buffer)):
                    raw, weight = frame_buffer[idx]
                    await self._run_cpu(clip_merger.merge, raw, weight)
                    self.tracker.update(self.name)

                # 构造 accepted FGP: fgp_total - rejected
                # 注意：必须从 fgp_total 减去 rejected，而非从 ref_fgp 减。
                # 因为 clip_merger 遍历的是全部原始帧，其 rejected 是对全部帧的统计。
                # ref_fgp 仅用于计算拒绝阈值（μ±kσ），不参与减法。
                accepted = fgp_total - clip_merger.result
                accepted.apply_zero_var(fgp_total)

                # 收敛检查
                cur_n = accepted.n
                converge_ratio = np.sum(cur_n == last_n) / np.prod(cur_n.shape)
                if last_n is not None and converge_ratio >= early_converge_ratio:
                    logger.info(
                        f"{self.name} converged at iteration {iteration + 1}.")
                    break
                else:
                    logger.info(
                        f"{self.name} converge ratio: {converge_ratio * 100:.2f}%"
                    )
                last_n = cur_n.copy()
                ref_fgp = accepted
                logger.info(
                    f"{self.name} iteration {iteration + 1}/{max_iter} done.")
            else:
                logger.info(
                    f"{self.name} reached max iterations ({max_iter}).")

            # ── Phase 3: 清理 + 输出 ──
            frame_buffer.cleanup()

            result = FloatImage(accepted.mu, dtype=accepted.source_dtype)
            put_tasks = []
            for queue in self.outputs['result']:
                put_tasks.append(queue.put(result))
            for queue in self.outputs['statistics']:
                put_tasks.append(queue.put(accepted))
            await gather(*put_tasks)

            logger.info(f"{self.name} sigma clipping complete.")

        except Exception as e:
            logger.error(f"{self.name} failed: {e}")
            frame_buffer.cleanup()
            raise
        finally:
            self.tracker.close_bar(self.name)


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
        "top_fraction": {
            "type": "float",
            "default": 0.02
        },
    }
    OUTPUTS = {"result": {"type": "image"}}

    async def _async_execute(self, configs: dict[str, Any]) -> None:
        top_fraction: float = configs['top_fraction']
        try:
            max_raw = configs['max_img']
            accepted: FastGaussianParam = configs['statistics']

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
                                       top_fraction=top_fraction)

            result = np.round(corrected).astype(output_dtype)

            put_tasks = [q.put(result) for q in self.outputs['result']]
            await gather(*put_tasks)

            logger.info(f"{self.name} complete.")

        except Exception as e:
            logger.error(f"{self.name} failed: {e}")
            raise
