"""Tests for SlidingWindowMaxOp: sliding window max timelapse."""
import asyncio

import numpy as np
import pytest

from hoshicore.component.queue import RichContextQueue
from hoshicore.ops.timelapse_ops import SlidingWindowMaxOp

pytestmark = pytest.mark.asyncio


# ── Helpers ──

def _naive_sliding_max(frames: list[np.ndarray], window_size: int) -> list[np.ndarray]:
    """Brute-force O(m*n) reference implementation."""
    m = len(frames)
    results = []
    for t in range(m):
        l = max(0, t - window_size + 1)
        window_max = frames[l].copy()
        for k in range(l + 1, t + 1):
            np.maximum(window_max, frames[k], out=window_max)
        results.append(window_max)
    return results


async def _run_op(frames: list[np.ndarray], window_size: int,
                  buffer_mode: str = "memory") -> list[np.ndarray]:
    """Wire and execute SlidingWindowMaxOp, return collected outputs."""
    from hoshicore.component.queue import BaseQueue

    op = SlidingWindowMaxOp("test_sliding_max")

    # Wire input queue
    input_queue = op.inputs["data"]

    # Wire output queue
    output_queue = RichContextQueue(maxsize=1)
    op.outputs["result"].append(output_queue)

    # Wire config queue
    await op.config["window_size"].put(window_size)
    await op.config["buffer_mode"].put(buffer_mode)

    # Feed input (including sentinel so StreamExhausted fires correctly)
    async def feed():
        await input_queue.set_length(len(frames))
        for frame in frames:
            await input_queue.put(frame)
        await input_queue.put(BaseQueue._SENTINEL)

    # Collect output
    collected = []

    async def collect():
        length = await output_queue.get_length()
        for _ in range(length):
            item = await output_queue.get()
            collected.append(item)

    await asyncio.gather(feed(), op.execute(), collect())
    return collected


# ── Tests ──

class TestSlidingWindowMaxBasic:
    async def test_single_frame(self):
        frame = np.array([[1, 2], [3, 4]], dtype=np.uint16)
        results = await _run_op([frame], window_size=5)
        assert len(results) == 1
        np.testing.assert_array_equal(results[0], frame)

    async def test_window_larger_than_sequence(self):
        frames = [
            np.array([[i, i + 1]], dtype=np.uint16) for i in range(5)
        ]
        results = await _run_op(frames, window_size=100)
        expected = _naive_sliding_max(frames, 100)
        assert len(results) == len(frames)
        for r, e in zip(results, expected):
            np.testing.assert_array_equal(r, e)

    async def test_window_1_is_identity(self):
        frames = [
            np.random.randint(0, 1000, (4, 4), dtype=np.uint16)
            for _ in range(8)
        ]
        results = await _run_op(frames, window_size=1)
        assert len(results) == len(frames)
        for r, f in zip(results, frames):
            np.testing.assert_array_equal(r, f)

    async def test_exact_block_boundary(self):
        """m = 2n: exactly 2 full blocks."""
        n = 4
        m = 8
        frames = [
            np.random.randint(0, 65535, (3, 3), dtype=np.uint16)
            for _ in range(m)
        ]
        results = await _run_op(frames, window_size=n)
        expected = _naive_sliding_max(frames, n)
        assert len(results) == m
        for t in range(m):
            np.testing.assert_array_equal(results[t], expected[t],
                                          err_msg=f"Mismatch at t={t}")

    async def test_partial_last_block(self):
        """m is not a multiple of n."""
        n = 3
        m = 10
        frames = [
            np.random.randint(0, 255, (5, 5, 3), dtype=np.uint8)
            for _ in range(m)
        ]
        results = await _run_op(frames, window_size=n)
        expected = _naive_sliding_max(frames, n)
        assert len(results) == m
        for t in range(m):
            np.testing.assert_array_equal(results[t], expected[t],
                                          err_msg=f"Mismatch at t={t}")

    async def test_many_blocks(self):
        """Stress test with many blocks."""
        n = 3
        m = 20
        frames = [
            np.random.randint(0, 65535, (8, 8), dtype=np.uint16)
            for _ in range(m)
        ]
        results = await _run_op(frames, window_size=n)
        expected = _naive_sliding_max(frames, n)
        for t in range(m):
            np.testing.assert_array_equal(results[t], expected[t],
                                          err_msg=f"Mismatch at t={t}")


class TestSlidingWindowMaxDisk:
    async def test_disk_mode_matches_memory(self):
        n = 3
        m = 7
        frames = [
            np.random.randint(0, 255, (4, 4, 3), dtype=np.uint8)
            for _ in range(m)
        ]
        mem_results = await _run_op(frames, window_size=n, buffer_mode="memory")
        disk_results = await _run_op(frames, window_size=n, buffer_mode="disk")
        for t in range(m):
            np.testing.assert_array_equal(mem_results[t], disk_results[t],
                                          err_msg=f"Mismatch at t={t}")


class TestSlidingWindowMaxEdgeCases:
    async def test_invalid_window_size(self):
        frame = np.zeros((2, 2), dtype=np.uint8)
        with pytest.raises(ValueError, match="window_size must be >= 1"):
            await _run_op([frame], window_size=0)

    async def test_monotonically_increasing(self):
        """Each frame strictly larger: output should equal prefix (cumulative max)."""
        frames = [
            np.full((3, 3), i, dtype=np.uint16) for i in range(10)
        ]
        results = await _run_op(frames, window_size=5)
        expected = _naive_sliding_max(frames, 5)
        for t in range(10):
            np.testing.assert_array_equal(results[t], expected[t])

    async def test_monotonically_decreasing(self):
        """Each frame strictly smaller: window max comes from oldest frame in window."""
        frames = [
            np.full((3, 3), 100 - i, dtype=np.uint16) for i in range(10)
        ]
        results = await _run_op(frames, window_size=4)
        expected = _naive_sliding_max(frames, 4)
        for t in range(10):
            np.testing.assert_array_equal(results[t], expected[t])
