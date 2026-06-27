from collections import deque
from typing import Any, Optional

import numpy as np
from loguru import logger

from ..component.frame_buffer import DiskFrameBuffer, MemoryFrameBuffer
from ..component.queue import StreamExhausted
from .._custom_op import max_combine as custom_max_combine
from ..engine.registry import register_op
from .base import BaseOp
from .weight_generator import generate_weight


def _create_buffer(mode: str):
    if mode == "memory":
        return MemoryFrameBuffer()
    return DiskFrameBuffer()


@register_op()
class SlidingWindowMaxOp(BaseOp):
    """滑窗最大值算子：序列 → 序列，每帧输出为窗口内逐像素最大值。

    使用分块前缀/后缀分解实现 O(m·HW) 复杂度，与窗口大小无关。
    block_size = window_size = n，保证滑窗最多跨两个连续 block。

    流水线：每帧到达时立即计算 prefix 并发射输出（非首 block 场景下
    prev_suffix 已就绪），无需等整块收齐。
    """

    INPUTS: dict[str, dict[str, Any]] = {
        "data": {"type": "sequence", "required": True},
    }
    CONFIGS: dict[str, dict[str, Any]] = {
        "window_size": {"type": "int", "default": 30},
        "buffer_mode": {"type": "str", "default": "memory"},
    }
    OUTPUTS: dict[str, dict[str, Any]] = {
        "result": {"type": "sequence"},
    }
    REPORTS_PROGRESS = True

    @classmethod
    def estimate_resources(
        cls,
        configs: dict[str, Any],
        frame_bytes: int,
        n_frames: Optional[int],
        dtype_bytes: Optional[int] = None,
    ) -> tuple[int, int]:
        _ = dtype_bytes
        n = configs.get("window_size", 30)
        mode = configs.get("buffer_mode", "memory")
        # Peak: n raw frames (for suffix scan) + n prev_suffix + 1 prefix rolling
        if mode == "memory":
            return ((2 * n + 1) * frame_bytes, 0)
        else:
            return (frame_bytes, 2 * n * frame_bytes)

    async def _async_execute(self, configs: dict[str, Any]) -> None:
        n: int = configs["window_size"]
        buffer_mode: str = configs.get("buffer_mode", "memory")

        if n < 1:
            raise ValueError(
                f"{self.name}: window_size must be >= 1, got {n}"
            )

        total = self.length
        if total is not None:
            self.tracker.create_bar(self.name, total, desc=self.display_name)

        prev_suffix_buf: Optional[MemoryFrameBuffer | DiskFrameBuffer] = None
        global_offset = 0
        upstream_exhausted = False

        try:
            while not upstream_exhausted:
                is_first_block = (global_offset == 0)

                # --- INGEST + streaming PREFIX + EMIT ---
                raw_buf = _create_buffer(buffer_mode)
                prefix_running: Optional[np.ndarray] = None

                for j in range(n):
                    try:
                        upper = self._async_convert_inputs()
                        frame = await upper["data"]
                    except StreamExhausted:
                        upstream_exhausted = True
                        break

                    raw_buf.append(frame)

                    # Rolling prefix: prefix[j] = max(prefix[j-1], frame)
                    if prefix_running is None:
                        prefix_running = frame.copy()
                    else:
                        prefix_running = np.maximum(prefix_running, frame)

                    # Emit output for this frame
                    if is_first_block:
                        # Warmup: window [0, t], output = prefix[t]
                        await self._broadcast_outputs({"result": prefix_running.copy()})
                    elif j < n - 1:
                        # output = max(prev_suffix[j+1], prefix[j])
                        prev_suffix_frame = prev_suffix_buf[j + 1][0]
                        output_frame = await self._run_cpu(
                            custom_max_combine,
                            prefix_running.copy(),
                            prev_suffix_frame,
                        )
                        await self._broadcast_outputs({"result": output_frame})
                    else:
                        # j == n-1: window is exactly current full block
                        await self._broadcast_outputs({"result": prefix_running.copy()})

                    if total is not None:
                        self.tracker.update(self.name)

                actual_len = len(raw_buf)
                if actual_len == 0:
                    raw_buf.cleanup()
                    break

                # Release prev_suffix (fully consumed during emit)
                if prev_suffix_buf is not None:
                    prev_suffix_buf.cleanup()
                    prev_suffix_buf = None

                # --- SUFFIX: reverse scan of raw frames (for next block) ---
                if not upstream_exhausted:
                    prev_suffix_buf = _create_buffer(buffer_mode)
                    suffix_running = raw_buf[actual_len - 1][0].copy()
                    prev_suffix_buf.append(suffix_running)
                    for j in range(actual_len - 2, -1, -1):
                        suffix_running = np.maximum(raw_buf[j][0], suffix_running)
                        prev_suffix_buf.append(suffix_running)

                    # Reverse the buffer so index 0 = suffix[0]
                    reordered = _create_buffer(buffer_mode)
                    for j in range(actual_len - 1, -1, -1):
                        reordered.append(prev_suffix_buf[j][0])
                    prev_suffix_buf.cleanup()
                    prev_suffix_buf = reordered

                raw_buf.cleanup()
                global_offset += actual_len

        finally:
            if prev_suffix_buf is not None:
                prev_suffix_buf.cleanup()
            if total is not None:
                self.tracker.close_bar(self.name)

        logger.info(
            f"{self.name}: completed {global_offset} frames, window_size={n}"
        )


@register_op()
class EmaDecayMaxOp(BaseOp):
    """指数衰减最大值算子：序列 → 序列，每帧输出为 max(当前帧, γ·前一输出)。

    递推公式：S_t = max(x_t, γ · S_{t-1})
    等效衰减函数：d(k) = γ^k（纯指数衰减）

    用户通过 half_life（半衰期帧数）控制衰减速度：
        γ = 2^{-1/half_life}

    内存开销恒定：仅维护一张状态图，与序列长度无关。
    """

    INPUTS: dict[str, dict[str, Any]] = {
        "data": {"type": "sequence", "required": True},
    }
    CONFIGS: dict[str, dict[str, Any]] = {
        "half_life": {"type": "float", "default": 30.0},
    }
    OUTPUTS: dict[str, dict[str, Any]] = {
        "result": {"type": "sequence"},
    }
    REPORTS_PROGRESS = True

    @classmethod
    def estimate_resources(
        cls,
        configs: dict[str, Any],
        frame_bytes: int,
        n_frames: Optional[int],
        dtype_bytes: Optional[int] = None,
    ) -> tuple[int, int]:
        return (frame_bytes, 0)

    async def _async_execute(self, configs: dict[str, Any]) -> None:
        half_life: float = configs["half_life"]
        if half_life <= 0:
            raise ValueError(
                f"{self.name}: half_life must be > 0, got {half_life}"
            )
        gamma = 2.0 ** (-1.0 / half_life)

        total = self.length
        if total is not None:
            self.tracker.create_bar(self.name, total, desc=self.display_name)

        state: Optional[np.ndarray] = None

        try:
            for _ in self._input_range():
                try:
                    upper = self._async_convert_inputs()
                    frame = await upper["data"]
                except StreamExhausted:
                    break

                if state is None:
                    state = frame.copy()
                else:
                    state = await self._run_cpu(
                        self._ema_step, state, frame, gamma
                    )

                await self._broadcast_outputs({"result": state.copy()})

                if total is not None:
                    self.tracker.update(self.name)
        finally:
            if total is not None:
                self.tracker.close_bar(self.name)

        logger.info(
            f"{self.name}: completed, half_life={half_life:.1f}, gamma={gamma:.6f}"
        )

    @staticmethod
    def _ema_step(
        state: np.ndarray, frame: np.ndarray, gamma: float
    ) -> np.ndarray:
        np.multiply(state, gamma, out=state)
        np.maximum(state, frame, out=state)
        return state


@register_op()
class WeightedSlidingWindowMaxOp(BaseOp):
    """加权滑窗最大值算子：序列 → 序列，每帧输出为窗口内加权逐像素最大值。

    权重曲线为梯形（渐入渐出），由 fade_in / fade_out 比例控制。
    复杂度 O(m·n·HW)：每帧需扫描整个窗口计算加权最大值。
    """

    INPUTS: dict[str, dict[str, Any]] = {
        "data": {"type": "sequence", "required": True},
    }
    CONFIGS: dict[str, dict[str, Any]] = {
        "window_size": {"type": "int", "default": 30},
        "fade_in": {"type": "float", "default": 0.3},
        "fade_out": {"type": "float", "default": 0.3},
        "buffer_mode": {"type": "str", "default": "memory"},
    }
    OUTPUTS: dict[str, dict[str, Any]] = {
        "result": {"type": "sequence"},
    }
    REPORTS_PROGRESS = True

    @classmethod
    def estimate_resources(
        cls,
        configs: dict[str, Any],
        frame_bytes: int,
        n_frames: Optional[int],
        dtype_bytes: Optional[int] = None,
    ) -> tuple[int, int]:
        n = configs.get("window_size", 30)
        mode = configs.get("buffer_mode", "memory")
        if mode == "memory":
            return ((n + 1) * frame_bytes, 0)
        else:
            return (frame_bytes, n * frame_bytes)

    async def _async_execute(self, configs: dict[str, Any]) -> None:
        window_size: int = configs["window_size"]
        fade_in: float = configs["fade_in"]
        fade_out: float = configs["fade_out"]

        if window_size < 1:
            raise ValueError(
                f"{self.name}: window_size must be >= 1, got {window_size}"
            )
        if fade_in + fade_out > 1.0:
            raise ValueError(
                f"{self.name}: fade_in + fade_out must be <= 1.0, "
                f"got {fade_in} + {fade_out} = {fade_in + fade_out}"
            )

        weights = generate_weight(window_size, fin=fade_in, fout=fade_out)

        total = self.length
        if total is not None:
            self.tracker.create_bar(self.name, total, desc=self.display_name)

        buffer: deque[np.ndarray] = deque(maxlen=window_size)

        try:
            for _ in self._input_range():
                try:
                    upper = self._async_convert_inputs()
                    frame = await upper["data"]
                except StreamExhausted:
                    break

                buffer.append(frame)
                output = await self._run_cpu(
                    self._weighted_max, list(buffer), weights
                )
                await self._broadcast_outputs({"result": output})

                if total is not None:
                    self.tracker.update(self.name)
        finally:
            if total is not None:
                self.tracker.close_bar(self.name)

        logger.info(
            f"{self.name}: completed, window_size={window_size}, "
            f"fade_in={fade_in}, fade_out={fade_out}"
        )

    @staticmethod
    def _weighted_max(
        frames: list[np.ndarray], weights: np.ndarray
    ) -> np.ndarray:
        n = len(weights)
        k = len(frames)
        # During warmup (k < n), align to tail of weight vector
        # so newest frame always gets weights[n-1]
        w = weights[n - k:]

        result = (frames[0].astype(np.float32) * w[0])
        for i in range(1, k):
            weighted = frames[i].astype(np.float32) * w[i]
            np.maximum(result, weighted, out=result)

        return result
