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
    支持三种缓冲策略（通过 buffer_mode config 控制）：
        - disk（默认）：将解码后的帧写入 DiskFrameBuffer（临时 .npz），读取快但占磁盘
        - memory：帧直接保存在 RAM 中（MemoryFrameBuffer），零 I/O 但占内存
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

import cv2
import numpy as np
from loguru import logger

from .._custom_op import median_reduce_chunk as custom_median_reduce_chunk
from .._custom_op.ops.sigma_clip import (
    sigma_clip_iterative_chunk as custom_sigma_clip_iterative_chunk,
    _load_compiled_module_result as _sc_load_compiled,
)
from ..component.data_container import FastGaussianParam, FloatImage
from ..component.frame_buffer import (BaseFrameBuffer, DiskFrameBuffer,
                                      MemoryFrameBuffer, SourceReplayBuffer)
from ..component.merger import (HuberWeightedMerger, MeanMerger,
                                SigmaClippingMerger)
from ..component.noise_equalization import (compute_adaptive_n_sigma,
                                            threshold_max_merge)
from ..component.queue import StreamExhausted
from ..engine.registry import register_op
from .base import BaseOp, ChunkIteratorBaseOp


@register_op()
class DiskBufferWriterOp(BaseOp):
    """将序列帧缓存供下游多 pass 算法重放。

    支持三种缓冲策略（通过 buffer_mode 配置）：
        - "disk"（默认）：解码后的帧写入 DiskFrameBuffer（临时 .npz 文件）
        - "memory"：帧直接保存在 RAM 中（MemoryFrameBuffer），零 I/O 但占内存
        - "replay"：保留原始文件路径到 SourceReplayBuffer（需要 fnames 输入）
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
            "default": "disk",
            "global": True,
        },
        "temp_path": {
            "type": "str",
            "default": None,
            "global": True,
        }
    }
    OUTPUTS = {
        "buffer_handle": {
            "type": "image",  # BaseFrameBuffer 实例，单次传递
        },
    }

    @classmethod
    def estimate_resources(cls, configs, frame_bytes, n_frames,
                           dtype_bytes=None):
        _ = dtype_bytes
        if n_frames is None:
            n_frames = 0
        mode = configs.get("buffer_mode", "disk")
        if mode == "memory":
            return (n_frames * frame_bytes, 0)
        elif mode == "disk":
            return (0, n_frames * frame_bytes)
        return (0, 0)

    async def _async_execute(self, configs: dict[str, Any]) -> None:
        tot_num = self.length

        has_weight = self.inputs['weight'].active
        has_fnames = self.inputs['fnames'].active
        buffer_mode = configs.get("buffer_mode", "disk")
        temp_path = configs.get("temp_path", None)

        # 确定缓冲策略
        if buffer_mode == "memory":
            frame_buffer = MemoryFrameBuffer()
            mode_label = "Memory"
        elif buffer_mode == "replay":
            if not has_fnames:
                raise ValueError(
                    f"{self.name}: replay mode requires 'fnames' input, "
                    f"but fnames is not wired.")
            frame_buffer = SourceReplayBuffer()
            mode_label = "Replay"
        else:
            frame_buffer = DiskFrameBuffer(temp_path=temp_path)
            mode_label = "Disk"

        stacked_num = 0
        failed_num = 0

        if tot_num is not None:
            self.tracker.create_bar(self.name,
                                tot_num,
                                desc=f"{self.name} [{mode_label}]")
        try:
            for i in self._input_range():
                cur_filename = f"the {i + 1}-th frame"
                try:
                    upper = self._async_convert_inputs()
                    cur_img = await upper['data']
                    fname = (await upper['fnames']) if has_fnames else None
                    weight = (await upper['weight']) if has_weight else None
                except StreamExhausted:
                    logger.warning(
                        f"{self.name}: upstream ended at {i}/{tot_num or '?'}")
                    break

                if cur_img is None:
                    logger.warning(
                        f"{self.name} failed to load {cur_filename}, skip.")
                    failed_num += 1
                    self.tracker.update(self.name)
                    continue

                if buffer_mode == "replay":
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
                f"{self.name}: buffered {stacked_num}/{tot_num or '?'} frames "
                f"({failed_num} fail(s)), mode={mode_label}.")

            # 按下游消费者数量设置引用计数
            n_consumers = len(self.outputs.get("buffer_handle", []))
            for _ in range(n_consumers):
                frame_buffer.acquire()
            await self._broadcast_outputs({"buffer_handle": frame_buffer})

        except Exception as e:
            # 自身异常：立即清理 buffer 防止泄漏
            logger.error(f"{self.name} failed: {e}")
            frame_buffer.cleanup()
            raise
        finally:
            self.tracker.close_bar(self.name)


@register_op()
class SigmaClipIteratorOp(ChunkIteratorBaseOp):
    """迭代式 Sigma Clipping：基于 mean FGP 和磁盘缓冲帧进行多 pass 迭代。

    使用 chunk-level multi-pass 模式：将 pass 循环嵌套进 chunk 循环内层，
    使每个 chunk 的所有 pass 复用 OS page cache，IO 从 n_passes × data 降为 ~1 × data。

    接收：
        - fgp_total: FastGaussianParam（来自 MeanStackerOp.statistics）
        - buffer_handle: DiskFrameBuffer 实例（来自 DiskBufferWriterOp）
        - rej_high / rej_low / max_iter / early_converge_ratio 配置

    输出：
        - result: sigma clipping 后的均值图像 (FloatImage)
        - statistics: accepted FastGaussianParam
    """

    EXECUTOR = "cpu"
    ITERATOR_TYPE = "sigma_clip"
    CHUNK_ROWS = 256
    CONFIGS: dict[str, dict[str, Any]] = {
        "fgp_total": {
            "type": "image",
            "required": True,
        },
        "buffer_handle": {
            "type": "image",
            "required": True,
        },
        "chunk_rows": {
            "type": "int",
            "default": 256,
            "global": True,
        },
        "mask": {
            "type": "image",
            "required": False,
            "default": None,
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
    
    @classmethod
    def estimate_resources(cls, configs, frame_bytes, n_frames,
                           dtype_bytes=None):
        _ = dtype_bytes
        # TODO: frame_bytes压缩太多信息，无法准确估计，此处是经验值。
        # 待 preflight 资源预分配完善后再调整。
        return (cls.CHUNK_ROWS * n_frames * 2000, 0)

    @classmethod
    def chunk_cost_per_row(cls, n_frames, row_bytes, dtype_bytes):
        float64_row = row_bytes // dtype_bytes * 8
        # compiled 路径峰值包含 chunk 双缓冲、stack_2d、mask、total_* 与 acc_*。
        stack = 2 * n_frames * row_bytes
        active_stack = n_frames * row_bytes
        mask = n_frames * row_bytes // dtype_bytes
        totals = 3 * float64_row
        acc = 3 * float64_row
        return stack + active_stack + mask + totals + acc

    def _init_chunk_state(self, configs, row_start, row_end, w):
        fgp_total: FastGaussianParam = configs['fgp_total']
        rej_high: float = configs['rej_high']
        rej_low: float = configs['rej_low']

        fgp_chunk = FastGaussianParam(
            sum_mu=fgp_total.sum_mu[row_start:row_end].copy(),
            square_sum=fgp_total.square_sum[row_start:row_end].copy(),
            n=fgp_total.n[row_start:row_end].copy(),
            ddof=fgp_total.ddof,
            source_dtype=fgp_total.source_dtype,
            inplace_calc=False,
        )

        # 静态 mask 切片
        raw_mask = configs.get('mask')
        static_mask_chunk = None
        if raw_mask is not None:
            mask = raw_mask
            if mask.ndim == 3:
                mask = mask[..., 0]
            static_mask_chunk = (mask > 0.5)[row_start:row_end]

        # Check if C++ kernel can be used for this chunk
        first_frame_dtype = fgp_total.source_dtype
        use_cpp = (
            self._cpp_kernel_available()
            and first_frame_dtype in (np.dtype('uint8'), np.dtype('uint16'))
        )

        clip_merger = None
        if not use_cpp:
            clip_merger = SigmaClippingMerger(
                ref_img=fgp_chunk,
                rej_high=rej_high,
                rej_low=rej_low,
            )

        return {
            'fgp_chunk': fgp_chunk,
            'clip_merger': clip_merger,
            'last_n': fgp_chunk.n.copy(),
            'static_mask': static_mask_chunk,
            'accepted': None,
            '_mask_cache': {},
            '_use_cpp': use_cpp,
            '_cpp_done': False,
        }

    @staticmethod
    def _cpp_kernel_available() -> bool:
        module, _ = _sc_load_compiled()
        return module is not None and hasattr(module, "sigma_clip_iterative_chunk")

    def _max_passes(self, configs):
        return configs['max_iter']

    def _run_pass(self, state, chunk_stack):
        """Override: use C++ kernel for all iterations in one call."""
        if state['_use_cpp'] and not state['_cpp_done']:
            self._run_pass_cpp(state, chunk_stack)
            return
        # Fallback: per-frame merge (one pass)
        for frame_idx, (chunk_data, chunk_weight) in enumerate(chunk_stack):
            self._merge_chunk(state, chunk_data, chunk_weight, frame_idx)

    def _run_pass_cpp(self, state, chunk_stack):
        """C++ path: stack all frames, compute mask, call kernel."""
        n_frames = len(chunk_stack)
        first_data = chunk_stack[0][0]
        h, w = first_data.shape[:2]
        channels = first_data.shape[2] if first_data.ndim == 3 else 1
        is_rgb = first_data.ndim == 3 and first_data.shape[2] >= 3
        plane_size = h * w * channels

        # Stack all frames into (n_frames, plane_size)
        stack_2d = np.empty((n_frames, plane_size), dtype=first_data.dtype)
        for f, (chunk_data, _) in enumerate(chunk_stack):
            stack_2d[f] = chunk_data.reshape(-1)

        # Build per-frame mask
        static_mask = state['static_mask']
        chunk_mask = None
        if static_mask is not None or is_rgb:
            chunk_mask = np.ones((n_frames, plane_size), dtype=np.uint8)
            static_flat = None
            if static_mask is not None:
                if channels > 1:
                    static_flat = np.broadcast_to(
                        static_mask[..., np.newaxis],
                        (h, w, channels)).reshape(-1).astype(np.uint8)
                else:
                    static_flat = static_mask.reshape(-1).astype(np.uint8)

            for f, (chunk_data, _) in enumerate(chunk_stack):
                if static_flat is not None:
                    chunk_mask[f] &= static_flat
                if is_rgb:
                    empty = np.all(chunk_data[..., :3] == 0, axis=-1)
                    if empty.any():
                        if channels > 1:
                            empty_flat = np.broadcast_to(
                                empty[..., np.newaxis],
                                (h, w, channels)).reshape(-1)
                        else:
                            empty_flat = empty.reshape(-1)
                        chunk_mask[f] &= (~empty_flat).astype(np.uint8)

            if chunk_mask.all():
                chunk_mask = None

        # Prepare FGP totals
        fgp_chunk = state['fgp_chunk']
        total_sum = fgp_chunk.sum_mu.reshape(-1).astype(np.float64)
        total_sq = fgp_chunk.square_sum.reshape(-1).astype(np.float64)
        total_n = fgp_chunk.n.reshape(-1).astype(np.float64)

        # Call C++ kernel (all iterations internally)
        acc_sum, acc_sq, acc_n = custom_sigma_clip_iterative_chunk(
            stack_2d, total_sum, total_sq, total_n,
            self._configs['rej_high'], self._configs['rej_low'],
            self._configs['max_iter'], mask=chunk_mask)

        # Build accepted FGP from results
        chunk_shape = first_data.shape
        accepted = FastGaussianParam(
            sum_mu=acc_sum.reshape(chunk_shape).astype(fgp_chunk.sum_mu.dtype),
            square_sum=acc_sq.reshape(chunk_shape).astype(fgp_chunk.square_sum.dtype),
            n=acc_n.reshape(chunk_shape).astype(fgp_chunk.n.dtype),
            ddof=fgp_chunk.ddof,
            source_dtype=fgp_chunk.source_dtype,
            inplace_calc=False,
        )
        accepted.apply_zero_var(fgp_chunk)
        state['accepted'] = accepted
        state['_cpp_done'] = True

    def _merge_chunk(self, state, chunk_data, chunk_weight, frame_idx):
        cache = state['_mask_cache']
        if frame_idx not in cache:
            static_mask = state['static_mask']
            spatial_mask = None
            if chunk_data.ndim == 3 and chunk_data.shape[2] >= 3:
                empty_mask = np.all(chunk_data[..., :3] == 0, axis=-1)
                if static_mask is not None:
                    spatial_mask = static_mask & (~empty_mask)
                elif empty_mask.any():
                    spatial_mask = ~empty_mask
            elif static_mask is not None:
                spatial_mask = static_mask
            cache[frame_idx] = spatial_mask
        state['clip_merger'].merge(chunk_data, chunk_weight,
                                   spatial_mask=cache[frame_idx])

    def _check_convergence(self, state, pass_idx):
        if state['_cpp_done']:
            return True

        fgp_chunk = state['fgp_chunk']
        clip_merger = state['clip_merger']

        accepted = fgp_chunk - clip_merger.result
        accepted.apply_zero_var(fgp_chunk)
        state['accepted'] = accepted

        cur_n = accepted.n
        ratio = np.sum(cur_n == state['last_n']) / cur_n.size
        state['last_n'] = cur_n.copy()

        converged = ratio >= self._configs['early_converge_ratio']
        if converged:
            logger.debug(
                f"{self.name} chunk converged at pass {pass_idx + 1} "
                f"(ratio={ratio * 100:.1f}%)")
        return converged

    def _prepare_next_pass(self, state, pass_idx):
        accepted = state['accepted']
        state['clip_merger'] = SigmaClippingMerger(
            ref_img=accepted,
            rej_high=self._configs['rej_high'],
            rej_low=self._configs['rej_low'],
        )

    def _finalize_chunk(self, state):
        if state['accepted'] is None:
            accepted = state['fgp_chunk'] - state['clip_merger'].result
            accepted.apply_zero_var(state['fgp_chunk'])
            state['accepted'] = accepted
        return state['accepted'].mu

    def _wrap_output(self, result, configs):
        fgp_total: FastGaussianParam = configs['fgp_total']

        # 拼接 chunk-level accepted FGP 为完整 statistics
        chunk_states = self._chunk_states
        sum_mu = np.concatenate(
            [s['accepted'].sum_mu for s in chunk_states], axis=0)
        square_sum = np.concatenate(
            [s['accepted'].square_sum for s in chunk_states], axis=0)
        n = np.concatenate(
            [s['accepted'].n for s in chunk_states], axis=0)

        accepted_full = FastGaussianParam(
            sum_mu=sum_mu,
            square_sum=square_sum,
            n=n,
            ddof=fgp_total.ddof,
            source_dtype=fgp_total.source_dtype,
            inplace_calc=False,
        )

        result_img = FloatImage(result, dtype=fgp_total.source_dtype)
        logger.info(f"{self.name} sigma clipping complete.")
        return {"result": result_img, "statistics": accepted_full}

    async def _async_execute(self, configs: dict[str, Any]) -> None:
        # 缓存 configs 供 _check_convergence / _prepare_next_pass 使用
        self._configs = configs
        configs['fgp_total'].inplace_calc = False
        await super()._async_execute(configs)


@register_op()
class SigmaClipFusedChunkOp(ChunkIteratorBaseOp):
    """融合式 Sigma Clipping：直接从 buffer 计算 mean FGP + 迭代 clip。

    省去独立的 MeanStackerOp，在一次 chunk 扫描中完成 FGP 累加和迭代剔除。
    使用 C++ sigma_clip_fused_chunk kernel（available 时）或 numpy fallback。
    """

    EXECUTOR = "cpu"
    ITERATOR_TYPE = "sigma_clip_fused"
    CHUNK_ROWS = 256
    CONFIGS: dict[str, dict[str, Any]] = {
        "buffer_handle": {
            "type": "image",
            "required": True,
        },
        "chunk_rows": {
            "type": "int",
            "default": 256,
            "global": True,
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
        "mask": {
            "type": "image",
            "required": False,
            "default": None,
        },
    }
    OUTPUTS = {
        "result": {
            "type": "image",
        },
        "statistics": {
            "type": "image",
        },
    }

    @classmethod
    def chunk_cost_per_row(cls, n_frames, row_bytes, dtype_bytes):
        float64_row = row_bytes // dtype_bytes * 8
        stack = 2 * n_frames * row_bytes
        active = n_frames * row_bytes
        mask = n_frames * row_bytes // dtype_bytes
        state = 3 * float64_row
        return stack + active + mask + state

    def _init_chunk_state(self, configs, row_start, row_end, w):
        # 静态 mask 切片
        raw_mask = configs.get('mask')
        static_mask_chunk = None
        if raw_mask is not None:
            mask = raw_mask
            if mask.ndim == 3:
                mask = mask[..., 0]
            static_mask_chunk = (mask > 0.5)[row_start:row_end]

        return {
            'row_start': row_start,
            'row_end': row_end,
            'w': w,
            'static_mask': static_mask_chunk,
            'accepted': None,
            '_done': False,
        }

    def _max_passes(self, configs):
        return 1

    def _run_pass(self, state, chunk_stack):
        from .._custom_op.ops.sigma_clip import sigma_clip_fused_chunk

        n_frames = len(chunk_stack)
        first_data = chunk_stack[0][0]
        h, w = first_data.shape[:2]
        channels = first_data.shape[2] if first_data.ndim == 3 else 1
        is_rgb = first_data.ndim == 3 and first_data.shape[2] >= 3
        plane_size = h * w * channels

        # Stack all frames
        stack_2d = np.empty((n_frames, plane_size), dtype=first_data.dtype)
        for f, (chunk_data, _) in enumerate(chunk_stack):
            stack_2d[f] = chunk_data.reshape(-1)

        # Build per-frame mask (static + RGB empty)
        static_mask = state['static_mask']
        chunk_mask = None
        if static_mask is not None or is_rgb:
            chunk_mask = np.ones((n_frames, plane_size), dtype=np.uint8)
            static_flat = None
            if static_mask is not None:
                if channels > 1:
                    static_flat = np.broadcast_to(
                        static_mask[..., np.newaxis],
                        (h, w, channels)).reshape(-1).astype(np.uint8)
                else:
                    static_flat = static_mask.reshape(-1).astype(np.uint8)

            for f, (chunk_data, _) in enumerate(chunk_stack):
                if static_flat is not None:
                    chunk_mask[f] &= static_flat
                if is_rgb:
                    empty = np.all(chunk_data[..., :3] == 0, axis=-1)
                    if empty.any():
                        if channels > 1:
                            empty_flat = np.broadcast_to(
                                empty[..., np.newaxis],
                                (h, w, channels)).reshape(-1)
                        else:
                            empty_flat = empty.reshape(-1)
                        chunk_mask[f] &= (~empty_flat).astype(np.uint8)

            if chunk_mask.all():
                chunk_mask = None

        # Call fused kernel (computes mean + iterative clip)
        acc_sum, acc_sq, acc_n = sigma_clip_fused_chunk(
            stack_2d, self._configs['rej_high'], self._configs['rej_low'],
            self._configs['max_iter'], mask=chunk_mask)

        chunk_shape = first_data.shape
        source_dtype = first_data.dtype
        accepted = FastGaussianParam(
            sum_mu=acc_sum.reshape(chunk_shape),
            square_sum=acc_sq.reshape(chunk_shape),
            n=acc_n.reshape(chunk_shape),
            ddof=1,
            source_dtype=source_dtype,
            inplace_calc=False,
        )
        state['accepted'] = accepted
        state['_done'] = True

    def _merge_chunk(self, state, chunk_data, chunk_weight, frame_idx):
        pass  # Not used — _run_pass handles everything

    def _check_convergence(self, state, pass_idx):
        return state['_done']

    def _finalize_chunk(self, state):
        return state['accepted'].mu

    def _wrap_output(self, result, configs):
        chunk_states = self._chunk_states
        sum_mu = np.concatenate(
            [s['accepted'].sum_mu for s in chunk_states], axis=0)
        square_sum = np.concatenate(
            [s['accepted'].square_sum for s in chunk_states], axis=0)
        n = np.concatenate(
            [s['accepted'].n for s in chunk_states], axis=0)

        # Infer source_dtype from first chunk
        source_dtype = chunk_states[0]['accepted'].source_dtype

        accepted_full = FastGaussianParam(
            sum_mu=sum_mu,
            square_sum=square_sum,
            n=n,
            ddof=1,
            source_dtype=source_dtype,
            inplace_calc=False,
        )

        result_img = FloatImage(result, dtype=source_dtype)
        logger.info(f"{self.name} fused sigma clipping complete.")
        return {"result": result_img, "statistics": accepted_full}

    async def _async_execute(self, configs: dict[str, Any]) -> None:
        self._configs = configs
        await super()._async_execute(configs)


@register_op()
class HuberMeanIteratorOp(ChunkIteratorBaseOp):
    """Huber 加权均值（Phase 2）：基于 mean FGP 和缓冲帧进行单 pass Huber 加权。

    使用 chunk-level 模式减少内存峰值和 page cache 压力。

    接收：
        - fgp_total: FastGaussianParam（来自 MeanStackerOp.statistics，Phase 1）
        - buffer_handle: BaseFrameBuffer 实例（来自 DiskBufferWriterOp）
        - huber_c: Huber 常数（默认 1.345，正态分布 95% 渐近效率）

    输出：
        - result: Huber 加权均值图像 (FloatImage)
    """

    EXECUTOR = "cpu"
    ITERATOR_TYPE = "huber_mean"
    CHUNK_ROWS = 256
    CONFIGS: dict[str, dict[str, Any]] = {
        "fgp_total": {
            "type": "image",
            "required": True,
        },
        "buffer_handle": {
            "type": "image",
            "required": True,
        },
        "chunk_rows": {
            "type": "int",
            "default": 256,
            "global": True,
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

    @classmethod
    def chunk_cost_per_row(cls, n_frames, row_bytes, dtype_bytes):
        float64_row = row_bytes // dtype_bytes * 8
        stack = 2 * n_frames * row_bytes
        merger_state = 4 * float64_row
        return stack + merger_state

    def _init_chunk_state(self, configs, row_start, row_end, w):
        fgp_total: FastGaussianParam = configs['fgp_total']
        huber_c: float = configs['huber_c']

        ref_chunk = FastGaussianParam(
            sum_mu=fgp_total.sum_mu[row_start:row_end].copy(),
            square_sum=fgp_total.square_sum[row_start:row_end].copy(),
            n=fgp_total.n[row_start:row_end].copy(),
            ddof=fgp_total.ddof,
            source_dtype=fgp_total.source_dtype,
            inplace_calc=False,
        )

        merger = HuberWeightedMerger(ref_stats=ref_chunk, huber_c=huber_c)
        return {'merger': merger, 'source_dtype': fgp_total.source_dtype}

    def _merge_chunk(self, state, chunk_data, chunk_weight, frame_idx):
        state['merger'].merge(chunk_data, chunk_weight)

    def _finalize_chunk(self, state):
        result = state['merger'].merged_image
        if result is None:
            raise ValueError("HuberMeanIteratorOp: no frames processed in chunk")
        return result.data

    def _wrap_output(self, result, configs):
        fgp_total: FastGaussianParam = configs['fgp_total']
        result_img = FloatImage(result, dtype=fgp_total.source_dtype)
        logger.info(f"{self.name} Huber mean complete.")
        return {"result": result_img}


@register_op()
class MedianReduceOp(BaseOp):
    """中位数堆栈：从磁盘缓冲帧中计算逐像素中位数。

    按空间分块（chunk_rows 行）处理以控制内存峰值。
    对每个块加载所有帧的对应行范围，沿帧轴取 median。

    输入 buffer_handle 来自 DiskBufferWriterOp。

    注意：中位数不可分布式归约。
    """

    EXECUTOR = "cpu"
    BUFFER_ITERATOR = True     # 段检测标记：消费 buffer
    CHUNK_PLANNED = True
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

    @staticmethod
    def _reduce_chunk(stack: np.ndarray) -> np.ndarray:
        return custom_median_reduce_chunk(stack)

    @classmethod
    def chunk_cost_per_row(cls, n_frames, row_bytes, dtype_bytes):
        _ = dtype_bytes
        return (n_frames + 1) * row_bytes

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
                        dtype=source_dtype)
                else:
                    stack = np.empty(
                        (n_frames, actual_rows, w), dtype=source_dtype)

                for frame_idx in range(n_frames):
                    frame_data, _ = frame_buffer[frame_idx]
                    stack[frame_idx] = frame_data[row_start:row_end]

                # 沿帧轴取中位数
                chunk_median = await self._run_cpu(self._reduce_chunk, stack)
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


@register_op()
class ThresholdMaxIteratorOp(BaseOp):
    """Threshold-Max 归约：从缓冲帧中提取显著亮于背景的像素叠入均值图像。

    背景 = sigma-clipped 均值，亮特征 = 各帧最大值。
    用于替代 MaxNoiseEqualizationOp，提供对局部亮度调整更鲁棒的噪声均匀化。

    接收：
        - fgp_total: FastGaussianParam（sigma-clip 后的统计量）
        - buffer_handle: BaseFrameBuffer 实例（来自 DiskBufferWriterOp）
        - n_sigma: 信号检测阈值（-1 = 按帧数自适应）

    输出：
        - result: 校正后的图像 (FloatImage)
    """

    EXECUTOR = "cpu"
    BUFFER_ITERATOR = True
    ITERATOR_TYPE = "threshold_max"
    CONFIGS: dict[str, dict[str, Any]] = {
        "fgp_total": {
            "type": "image",
            "required": True,
        },
        "buffer_handle": {
            "type": "image",
            "required": True,
        },
        "n_sigma": {
            "type": "float",
            "default": -1,
        },
        "morph_kernel_size": {
            "type": "int",
            "default": 3,
        },
    }
    OUTPUTS = {
        "result": {
            "type": "image",
        },
    }

    async def _async_execute(self, configs: dict[str, Any]) -> None:
        fgp: FastGaussianParam = configs['fgp_total']
        frame_buffer: BaseFrameBuffer = configs['buffer_handle']
        n_sigma_cfg: float = configs['n_sigma']
        kernel_size: int = configs['morph_kernel_size']

        try:
            n_frames = len(frame_buffer)
            if n_sigma_cfg <= 0:
                n_sigma = compute_adaptive_n_sigma(n_frames)
                logger.info(
                    f"{self.name}: auto n_sigma={n_sigma:.2f} "
                    f"for {n_frames} frames")
            else:
                n_sigma = n_sigma_cfg

            mean_img = fgp.mu.astype(np.float64)
            std_img = np.sqrt(np.maximum(fgp.var, 0).astype(np.float64))
            result = mean_img.copy()

            kernel = None
            if kernel_size > 1:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_RECT, (kernel_size, kernel_size))

            self.tracker.create_bar(
                self.name, n_frames,
                desc=f"{self.name} [ThresholdMax]")

            async for raw, weight in frame_buffer.iter_prefetch():
                frame = raw.astype(np.float64)
                await self._run_cpu(
                    threshold_max_merge,
                    frame, mean_img, std_img, result,
                    n_sigma, weight, kernel)
                self.tracker.update(self.name)

            self.tracker.close_bar(self.name)

            out = FloatImage(data=result, dtype=fgp.source_dtype)
            await self._broadcast_outputs({"result": out})
            logger.info(f"{self.name} threshold-max complete.")

        except Exception as e:
            logger.error(f"{self.name} failed: {e}")
            raise
        finally:
            frame_buffer.cleanup()
