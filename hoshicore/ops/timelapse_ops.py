from typing import Any, Optional

import numpy as np
from loguru import logger

from ..component.frame_buffer import DiskFrameBuffer, MemoryFrameBuffer
from ..component.queue import StreamExhausted
from .._custom_op import max_combine as custom_max_combine
from ..engine.registry import register_op
from .base import BaseOp


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
