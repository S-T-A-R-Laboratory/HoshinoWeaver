"""Chunk reducers must release completed workspaces while preserving outputs."""
import asyncio
import weakref
from unittest.mock import Mock

import numpy as np
import pytest

from hoshicore._custom_op import _dispatch
from hoshicore._custom_op.ops.median import median_reduce_chunk_numpy
from hoshicore.component.data_container import FastGaussianParam
from hoshicore.component.frame_buffer import MemoryFrameBuffer
from hoshicore.ops.sigma_clip_ops import (
    HuberMeanIteratorOp, MedianReduceOp, SigmaClipFusedChunkOp, SigmaClipIteratorOp,
)


REDUCERS = [SigmaClipIteratorOp, SigmaClipFusedChunkOp,
            HuberMeanIteratorOp, MedianReduceOp]


@pytest.mark.parametrize('backend', ['numpy', 'cpu'])
@pytest.mark.parametrize('op_type', REDUCERS)
@pytest.mark.parametrize('fail', [False, True])
def test_completed_chunk_work_is_released(monkeypatch, backend, op_type, fail):
    monkeypatch.setattr(_dispatch, '_backend_preference_override', backend)
    frames = [np.full((7, 9, 3), value, np.uint16) for value in (98, 100, 102)]
    stack = np.stack(frames).astype(np.float64)
    reference = FastGaussianParam(
        stack.sum(axis=0), np.square(stack).sum(axis=0),
        np.full(frames[0].shape, len(frames), np.uint16),
        source_dtype=np.dtype('uint16'), inplace_calc=False)
    buffer = MemoryFrameBuffer()
    for frame in frames:
        buffer.append(frame)
    buffer.acquire()
    configs = dict(buffer_handle=buffer, fgp_total=reference,
                   mask=np.ones(frames[0].shape[:2], bool), chunk_rows=1,
                   rej_high=10., rej_low=10., max_iter=2,
                   early_converge_ratio=1., huber_c=1.345)
    op = op_type('reduce')
    op.tracker = Mock()
    outputs = []
    work_refs = []
    accepted_refs = []
    init = op._init_chunk_state
    finalize = op._finalize_chunk
    chunks_started = 0

    def observe_init(*args):
        nonlocal chunks_started
        # Before the next block starts, no completed block's working arrays or
        # mergers may remain alive. Output statistics are intentionally retained.
        assert all(ref() is None for ref in work_refs)
        if fail and chunks_started == 2:
            raise RuntimeError('stop after completed chunks')
        chunks_started += 1
        state = init(*args)
        for key in ('fgp_chunk', 'clip_merger', 'last_n', 'static_mask',
                    'merger', 'ref_mean', 'ref_std'):
            if state.get(key) is not None:
                work_refs.append(weakref.ref(state[key]))
        return state

    def observe_finalize(state):
        result = finalize(state)
        if state.get('accepted') is not None:
            accepted_refs.append(weakref.ref(state['accepted']))
        return result

    async def run_cpu(function, *args):
        return function(*args)

    async def broadcast(payload):
        outputs.append(payload)

    monkeypatch.setattr(op, '_init_chunk_state', observe_init)
    monkeypatch.setattr(op, '_finalize_chunk', observe_finalize)
    monkeypatch.setattr(op, '_run_cpu', run_cpu)
    monkeypatch.setattr(op, '_broadcast_outputs', broadcast)
    if fail:
        with pytest.raises(RuntimeError, match='stop after completed chunks'):
            asyncio.run(op._async_execute(configs))
    else:
        asyncio.run(op._async_execute(configs))
        np.testing.assert_array_equal(outputs[0]['result'].data, 100)
        if accepted_refs:
            stats = outputs[0]['statistics']
            np.testing.assert_array_equal(stats.sum_mu, reference.sum_mu)
            np.testing.assert_array_equal(stats.square_sum, reference.square_sum)
            np.testing.assert_array_equal(stats.n, reference.n)
    assert chunks_started == (2 if fail else 7)
    assert all(ref() is None for ref in work_refs + accepted_refs)
    assert op._chunk_states == []
    assert len(buffer) == 0


@pytest.mark.parametrize('op_type', [SigmaClipIteratorOp, SigmaClipFusedChunkOp])
@pytest.mark.parametrize('channels', [None, 1, 3])
def test_static_mask_backing_storage_is_chunk_sized(op_type, channels):
    shape = (13, 17) if channels is None else (13, 17, channels)
    mask = np.linspace(0, 1, np.prod(shape)).reshape(shape)
    reference = FastGaussianParam(np.ones((13, 17), np.uint16))
    op = op_type('mask')
    state = op._init_chunk_state(
        dict(mask=mask, fgp_total=reference, rej_high=3., rej_low=3.), 4, 5, 17)
    chunk = state['static_mask']
    expected = mask if channels is None else mask[..., 0]
    np.testing.assert_array_equal(chunk, expected[4:5] > .5)
    root = chunk
    while isinstance(root.base, np.ndarray):
        root = root.base
    assert root.nbytes == 17


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16, np.float32, np.float64])
@pytest.mark.parametrize('n_frames', [1, 3, 4])
def test_median_result_does_not_retain_frame_stack(dtype, n_frames):
    stack = np.arange(n_frames * 2 * 3, dtype=dtype).reshape(n_frames, 2, 3)
    result = median_reduce_chunk_numpy(stack)
    assert not np.shares_memory(result, stack)
    root = result
    while isinstance(root.base, np.ndarray):
        root = root.base
    assert root.nbytes <= result.nbytes


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16, np.float32, np.float64])
@pytest.mark.parametrize('rows', [1, 3])
def test_median_skip_zero_workspace_fits_planner(dtype, rows):
    import tracemalloc

    stack = np.ones((8, rows, 1024, 3), dtype=dtype)
    stack[:4, :, ::2] = 0
    # The estimate also reserves current and prefetched frame lists, which
    # are not allocated by this direct wrapper call.
    row_bytes = stack[0, 0].nbytes
    budget = MedianReduceOp.chunk_cost_per_row(8, row_bytes, stack.dtype.itemsize) * rows
    tracemalloc.start()
    try:
        result = median_reduce_chunk_numpy(stack, skip_zero_rgb=True)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    np.testing.assert_array_equal(result, 1)
    assert peak + 3 * stack.nbytes <= budget
