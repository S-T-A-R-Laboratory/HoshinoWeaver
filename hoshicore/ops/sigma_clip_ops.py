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
    消费序列输入，逐帧缓存供下游多 pass 算法重放。
    支持两种缓冲策略（通过 buffer_mode config 控制）：
        - disk（默认）：将解码后的帧写入 DiskFrameBuffer（临时 .npz），读取快但占磁盘
        - replay：保留原始文件路径到 SourceReplayBuffer，零临时文件但每 pass 重新 decode
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

from ..component.frame_buffer import DiskFrameBuffer, SourceReplayBuffer
from ..component.merger import (MeanMerger, SigmaClippingMerger,
                                HuberWeightedMerger)
from ..component.tagged_image import FloatImage
from ..component.utils import FastGaussianParam
from ..engine.registry import register_op
from ..component.queue import StreamExhausted
from .base import BaseOp


@register_op()
class DiskBufferWriterOp(BaseOp):
    """将序列帧缓存供下游多 pass 算法重放。

    支持两种缓冲策略：
        - disk（默认）：解码后的帧写入 DiskFrameBuffer（临时 .npz 文件）
        - replay：保留原始文件路径到 SourceReplayBuffer，零临时文件

    buffer_mode 配置：
        - "auto"（默认）：有 fnames 输入 → replay，否则 → disk
        - "disk"：强制使用 DiskFrameBuffer
        - "replay"：强制使用 SourceReplayBuffer（需要 fnames 输入）
    """

    EXECUTOR = "cpu"
    IS_DISK_BUFFER = True  # 段检测标记：识别为磁盘缓冲终端
    INPUTS: dict[str, dict[str, Any]] = {
        "data": {
            "type": "sequence",
            "required": True,
        },
        "weight": {
            "type": "sequence",
            "required": False,
        },
        "fnames": {
            "type": "sequence",
            "required": False,
        },
    }
    CONFIGS: dict[str, dict[str, Any]] = {
        "buffer_mode": {
            "type": "str",
            "default": "auto",
        },
    }
    OUTPUTS = {
        "buffer_handle": {
            "type": "image",  # BaseFrameBuffer 实例，单次传递
        },
    }

    async def _async_execute(self, configs: dict[str, Any]) -> None:
        tot_num = self.length
        assert tot_num is not None, \
            "DiskBufferWriterOp requires sequence length information."

        has_weight = self.inputs['weight'].active
        has_fnames = self.inputs['fnames'].active
        buffer_mode = configs.get("buffer_mode", "auto")

        # 确定缓冲策略
        use_replay = (buffer_mode == "replay" or
                      (buffer_mode == "auto" and has_fnames))
        if use_replay and not has_fnames:
            raise ValueError(
                f"{self.name}: replay mode requires 'fnames' input, "
                f"but fnames is not wired.")

        if use_replay:
            frame_buffer = SourceReplayBuffer()
            mode_label = "Replay"
        else:
            frame_buffer = DiskFrameBuffer()
            mode_label = "Buffer"

        stacked_num = 0
        failed_num = 0

        self.tracker.create_bar(self.name,
                                tot_num,
                                desc=f"{self.name} [{mode_label}]")
        try:
            for i in range(tot_num):
                cur_filename = f"the {i + 1}-th frame"
                try:
                    upper = self._async_convert_inputs()
                    cur_img = await upper['data']
                    fname = (await upper['fnames']) if has_fnames else None
                    weight = (await upper['weight']) if has_weight else None
                except StreamExhausted:
                    logger.warning(
                        f"{self.name}: upstream ended at {i}/{tot_num}")
                    break

                if cur_img is None:
                    logger.warning(
                        f"{self.name} failed to load {cur_filename}, skip.")
                    failed_num += 1
                    self.tracker.update(self.name)
                    continue

                if use_replay:
                    frame_buffer.append(fname, weight)
                else:
                    frame_buffer.append(cur_img, weight)
                stacked_num += 1
                self.tracker.update(self.name)

            if stacked_num == 0:
                logger.warning(f"{self.name}: No valid frames buffered!")
                frame_buffer.cleanup()
                return

            logger.info(
                f"{self.name}: buffered {stacked_num}/{tot_num} frames "
                f"({failed_num} fail(s)), mode={mode_label}.")

            # 将 buffer 实例推给下游
            await self._broadcast_outputs({"buffer_handle": frame_buffer})

        except Exception as e:
            # 自身异常：立即清理 buffer 防止泄漏
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
    BUFFER_ITERATOR = True        # 段检测标记：消费 buffer 的迭代式 Reduce
    ITERATOR_TYPE = "sigma_clip"  # 迭代类型标识（多阶段协议用）
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
            fgp_total.inplace_calc = False
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
            accepted.inplace_calc = False  # 输出前关闭 inplace_calc，避免下游误用导致数据被修改
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


@register_op()
class HuberMeanIteratorOp(BaseOp):
    """Huber 加权均值（Phase 2）：基于 mean FGP 和缓冲帧进行单 pass Huber 加权。

    接收：
        - fgp_total: FastGaussianParam（来自 MeanStackerOp.statistics，Phase 1）
        - buffer_handle: BaseFrameBuffer 实例（来自 DiskBufferWriterOp）
        - huber_c: Huber 常数（默认 1.345，正态分布 95% 渐近效率）

    输出：
        - result: Huber 加权均值图像 (FloatImage)

    与 SigmaClipIteratorOp 的结构对称，但只需单 pass（无迭代）。
    """

    EXECUTOR = "cpu"
    BUFFER_ITERATOR = True        # 段检测标记：消费 buffer 的迭代式 Reduce
    ITERATOR_TYPE = "huber_mean"  # 迭代类型标识（多阶段协议用）
    CONFIGS: dict[str, dict[str, Any]] = {
        "fgp_total": {
            "type": "image",
            "required": True,
        },
        "buffer_handle": {
            "type": "image",
            "required": True,
        },
        "huber_c": {
            "type": "float",
            "default": 1.345,
        },
    }
    OUTPUTS = {
        "result": {
            "type": "image",
        },
    }

    async def _async_execute(self, configs: dict[str, Any]) -> None:
        fgp_total: FastGaussianParam = configs['fgp_total']
        frame_buffer: DiskFrameBuffer = configs['buffer_handle']
        huber_c: float = configs['huber_c']

        try:
            huber_merger = HuberWeightedMerger(
                ref_stats=fgp_total,
                huber_c=huber_c,
            )

            n_frames = len(frame_buffer)
            self.tracker.create_bar(
                self.name, n_frames,
                desc=f"{self.name} [Huber]")

            for idx in range(n_frames):
                raw, weight = frame_buffer[idx]
                await self._run_cpu(huber_merger.merge, raw, weight)
                self.tracker.update(self.name)

            self.tracker.close_bar(self.name)

            result = huber_merger.merged_image
            if result is None:
                raise ValueError(
                    f"{self.name}: No valid frames processed.")

            await self._broadcast_outputs({"result": result})
            logger.info(f"{self.name} Huber mean complete.")

        except Exception as e:
            logger.error(f"{self.name} failed: {e}")
            raise
        finally:
            frame_buffer.cleanup()


@register_op()
class MedianReduceOp(BaseOp):
    """中位数堆栈：从磁盘缓冲帧中计算逐像素中位数。

    按空间分块（chunk_rows 行）处理以控制内存峰值。
    对每个块加载所有帧的对应行范围，沿帧轴取 median。

    输入 buffer_handle 来自 DiskBufferWriterOp。

    注意：中位数不可分布式归约，多进程时需要回退到主进程单线程计算。
    """

    EXECUTOR = "cpu"
    BUFFER_ITERATOR = True     # 段检测标记：消费 buffer
    ITERATOR_TYPE = "median"   # 不可分布式，Collector 需特殊处理
    CONFIGS: dict[str, dict[str, Any]] = {
        "buffer_handle": {
            "type": "image",
            "required": True,
        },
        "chunk_rows": {
            "type": "int",
            "default": 32,
        },
    }
    OUTPUTS = {
        "result": {
            "type": "image",
        },
    }

    async def _async_execute(self, configs: dict[str, Any]) -> None:
        frame_buffer: DiskFrameBuffer = configs['buffer_handle']
        chunk_rows: int = configs['chunk_rows']
        n_frames = len(frame_buffer)

        if n_frames == 0:
            raise ValueError(f"{self.name}: buffer is empty, nothing to stack.")

        # 读取第一帧获取尺寸信息
        first_frame, _ = frame_buffer[0]
        h, w = first_frame.shape[:2]
        channels = first_frame.shape[2] if first_frame.ndim == 3 else 1
        source_dtype = first_frame.dtype

        logger.info(
            f"{self.name}: computing median of {n_frames} frames "
            f"({h}x{w}x{channels}, dtype={source_dtype}), "
            f"chunk_rows={chunk_rows}")

        # 按行分块计算中位数
        result_chunks = []
        n_chunks = (h + chunk_rows - 1) // chunk_rows

        self.tracker.create_bar(self.name, n_chunks,
                                desc=f"{self.name} [Median]", unit="chunks")

        try:
            for chunk_idx in range(n_chunks):
                row_start = chunk_idx * chunk_rows
                row_end = min(row_start + chunk_rows, h)
                actual_rows = row_end - row_start

                # 加载所有帧的对应行范围
                if first_frame.ndim == 3:
                    stack = np.empty(
                        (n_frames, actual_rows, w, channels),
                        dtype=np.float32)
                else:
                    stack = np.empty(
                        (n_frames, actual_rows, w), dtype=np.float32)

                for frame_idx in range(n_frames):
                    frame_data, _ = frame_buffer[frame_idx]
                    stack[frame_idx] = frame_data[
                        row_start:row_end].astype(np.float32)

                # 沿帧轴取中位数
                chunk_median = await self._run_cpu(
                    np.median, stack, axis=0)
                result_chunks.append(chunk_median)
                self.tracker.update(self.name)

            # 拼接所有块
            result_array = np.concatenate(result_chunks, axis=0)

            result = FloatImage(data=result_array, dtype=source_dtype)
            await self._broadcast_outputs({"result": result})

            logger.info(f"{self.name}: median stacking complete.")

        except Exception as e:
            logger.error(f"{self.name} failed: {e}")
            raise
        finally:
            self.tracker.close_bar(self.name)
            frame_buffer.cleanup()
