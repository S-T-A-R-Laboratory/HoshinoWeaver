"""Tests for timelapse ops: SlidingWindowMaxOp, EmaDecayMaxOp."""
import asyncio

import numpy as np
import pytest

from hoshicore.component.queue import RichContextQueue
from hoshicore.ops.timelapse_ops import EmaDecayMaxOp, SlidingWindowMaxOp

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


# ── EmaDecayMaxOp helpers ──

async def _run_ema_op(frames: list[np.ndarray], half_life: float) -> list[np.ndarray]:
    """Wire and execute EmaDecayMaxOp, return collected outputs."""
    from hoshicore.component.queue import BaseQueue

    op = EmaDecayMaxOp("test_ema_decay")

    input_queue = op.inputs["data"]
    output_queue = RichContextQueue(maxsize=1)
    op.outputs["result"].append(output_queue)

    await op.config["half_life"].put(half_life)

    async def feed():
        await input_queue.set_length(len(frames))
        for frame in frames:
            await input_queue.put(frame)
        await input_queue.put(BaseQueue._SENTINEL)

    collected = []

    async def collect():
        length = await output_queue.get_length()
        for _ in range(length):
            item = await output_queue.get()
            collected.append(item)

    await asyncio.gather(feed(), op.execute(), collect())
    return collected


def _naive_ema_max(frames: list[np.ndarray], half_life: float) -> list[np.ndarray]:
    """Brute-force reference: S_t = max(x_t, gamma * S_{t-1})."""
    gamma = 2.0 ** (-1.0 / half_life)
    results = []
    state = None
    for frame in frames:
        if state is None:
            state = frame.astype(np.float64).copy()
        else:
            state = np.maximum(frame.astype(np.float64), gamma * state)
        results.append(state.copy())
    return results


# ── EmaDecayMaxOp Tests ──

class TestEmaDecayMaxBasic:
    async def test_single_frame(self):
        frame = np.array([[100, 200], [50, 150]], dtype=np.float32)
        results = await _run_ema_op([frame], half_life=10.0)
        assert len(results) == 1
        np.testing.assert_array_equal(results[0], frame)

    async def test_recursion_correctness(self):
        """Verify EMA recursion against naive implementation."""
        frames = [
            np.array([[100.0]], dtype=np.float32),
            np.array([[50.0]], dtype=np.float32),
            np.array([[200.0]], dtype=np.float32),
            np.array([[10.0]], dtype=np.float32),
            np.array([[10.0]], dtype=np.float32),
        ]
        half_life = 2.0
        results = await _run_ema_op(frames, half_life=half_life)
        expected = _naive_ema_max(frames, half_life)
        for t in range(len(frames)):
            np.testing.assert_allclose(
                results[t], expected[t], rtol=1e-5,
                err_msg=f"Mismatch at t={t}")

    async def test_half_life_property(self):
        """After half_life frames of zero input, state should halve."""
        half_life = 5.0
        initial_val = 1000.0
        n_frames = int(half_life) + 1
        frames = [np.full((4, 4), initial_val, dtype=np.float32)]
        frames += [np.zeros((4, 4), dtype=np.float32) for _ in range(n_frames - 1)]

        results = await _run_ema_op(frames, half_life=half_life)
        # After half_life frames of zero, value ≈ initial / 2
        actual = results[int(half_life)][0, 0]
        expected = initial_val / 2.0
        np.testing.assert_allclose(actual, expected, rtol=1e-4)

    async def test_decay_is_monotonic(self):
        """With zero input after initial pulse, output should monotonically decrease."""
        frames = [np.full((3, 3), 500.0, dtype=np.float32)]
        frames += [np.zeros((3, 3), dtype=np.float32) for _ in range(20)]
        results = await _run_ema_op(frames, half_life=4.0)
        values = [r[0, 0] for r in results]
        for i in range(1, len(values)):
            assert values[i] <= values[i - 1], f"Non-monotonic at {i}"

    async def test_new_bright_pixel_overrides(self):
        """A bright new frame should immediately appear in output."""
        frames = [
            np.full((2, 2), 100.0, dtype=np.float32),
            np.full((2, 2), 50.0, dtype=np.float32),
            np.full((2, 2), 999.0, dtype=np.float32),
        ]
        results = await _run_ema_op(frames, half_life=10.0)
        np.testing.assert_allclose(results[2][0, 0], 999.0, rtol=1e-6)

    async def test_multichannel(self):
        """Works with HWC (3-channel color) images."""
        rng = np.random.default_rng(42)
        frames = [rng.integers(0, 255, (8, 8, 3)).astype(np.float32)
                  for _ in range(10)]
        results = await _run_ema_op(frames, half_life=3.0)
        expected = _naive_ema_max(frames, 3.0)
        for t in range(10):
            np.testing.assert_allclose(
                results[t], expected[t], rtol=1e-5,
                err_msg=f"Mismatch at t={t}")


class TestEmaDecayMaxEdgeCases:
    async def test_invalid_half_life(self):
        frame = np.zeros((2, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="half_life must be > 0"):
            await _run_ema_op([frame], half_life=0.0)

    async def test_very_large_half_life(self):
        """Large half_life ≈ cumulative max (gamma → 1)."""
        frames = [np.full((2, 2), float(i), dtype=np.float32)
                  for i in range(10)]
        results = await _run_ema_op(frames, half_life=1e6)
        # With gamma ≈ 1, output[t] ≈ max(frames[0:t+1])
        for t in range(10):
            expected_val = float(t)  # monotonically increasing
            np.testing.assert_allclose(results[t][0, 0], expected_val, atol=1e-3)

    async def test_very_small_half_life(self):
        """Small half_life ≈ identity (gamma → 0, state decays instantly)."""
        frames = [np.full((2, 2), float(100 - i * 10), dtype=np.float32)
                  for i in range(5)]
        results = await _run_ema_op(frames, half_life=0.01)
        # With gamma ≈ 0, output[t] ≈ frame[t] (no memory)
        for t in range(5):
            np.testing.assert_allclose(
                results[t][0, 0], frames[t][0, 0], rtol=0.01)
