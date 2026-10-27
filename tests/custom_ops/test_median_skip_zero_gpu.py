"""GPU median must select ranks from the shared RGB-valid frame subset."""
from unittest import mock

import numpy as np
import pytest

from hoshicore._custom_op import _dispatch, cuda_memory, metal_memory
from hoshicore._custom_op.ops import median


@pytest.fixture(params=['cuda', 'metal'])
def gpu(request):
    name = request.param
    loader = (median._load_compiled_module_result if name == 'cuda'
              else median._load_metal_module_result)
    module, reason = loader()
    if module is None or not hasattr(module, f'median_reduce_chunk_{name}'):
        pytest.skip(reason or f'{name} median unavailable')
    info = module.cuda_memory_info() if name == 'cuda' else module.metal_device_info()
    if not info.get('available'):
        pytest.skip(info.get('reason') or f'{name} device unavailable')
    return name, module


def _reference(stack):
    if stack.ndim == 3 or stack.shape[-1] < 3:
        return np.median(stack, axis=0).astype(stack.dtype)
    output = np.zeros(stack.shape[1:], stack.dtype)
    for row in range(stack.shape[1]):
        for col in range(stack.shape[2]):
            samples = stack[:, row, col]
            valid = np.any(samples[:, :3] != 0, axis=1)
            if valid.any():
                output[row, col] = np.median(samples[valid], axis=0)
    return output


def _assert_workspace(name, module, stack):
    estimator = cuda_memory if name == 'cuda' else metal_memory
    estimate = estimator.estimate_median_reduce_chunk(
        n_frames=stack.shape[0], plane_size=stack[0].size, dtype_bytes=stack.dtype.itemsize)
    if name == 'cuda':
        info = module.cuda_host_io_cache_info()
        assert info['last_device_peak_bytes'] == estimate.peak_device_bytes
        assert info['last_pinned_peak_bytes'] == estimate.peak_pinned_bytes
    else:
        assert module.metal_host_io_cache_info()['last_logical_peak_bytes'] == estimate.peak_device_bytes


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
@pytest.mark.parametrize('channels', [None, 1, 3, 4])
@pytest.mark.parametrize('n_frames', [1, 2, 3, 8, 16, 17, 32, 64, 127, 128])
def test_gpu_skip_zero_native_and_wrappers(gpu, dtype, channels, n_frames):
    name, module = gpu
    rng = np.random.default_rng(405 + n_frames)
    shape = (n_frames, 2, 16) + (() if channels is None else (channels,))
    stack = rng.integers(0, np.iinfo(dtype).max + 1, shape, dtype=dtype)[:, :, ::2]
    # Non-contiguous input; each column has a different valid frame count.
    for col in range(8):
        if channels is None:
            stack[:min(col, n_frames), :, col] = 0
        else:
            stack[:min(col, n_frames), :, col, :3] = 0
    # Include tiny valid subsets even in 128-frame stacks.
    for col, keep in [(2, 1), (3, 2), (4, 3), (5, 4)]:
        if channels is None:
            stack[:max(n_frames - keep, 0), 0, col] = 0
        else:
            stack[:max(n_frames - keep, 0), 0, col, :3] = 0
    stack[-1, 0, 2] = np.iinfo(dtype).max
    stack[:, 0, 0] = 0
    stack[:, 0, 1] = np.iinfo(dtype).max  # Valid maxima tie with CUDA sentinels.
    if channels is not None and channels >= 3:
        stack[:, 1, 0, 0] = 0  # Keep single-channel zeros.
    if channels == 4:
        stack[..., 3] = np.arange(n_frames, dtype=dtype)[:, None, None] + 1
    before = stack.copy()
    expected = _reference(stack)
    for entry, data in [
        (getattr(module, f'median_reduce_chunk_{name}'), stack),
        (getattr(module, f'median_reduce_chunk_{name}_frames'), tuple(stack)),
        (getattr(median, f'median_reduce_chunk_{name}'), stack),
        (getattr(median, f'median_reduce_frames_{name}'), tuple(stack)),
    ]:
        actual = entry(data, skip_zero_rgb=True)
        assert actual.dtype == dtype
        np.testing.assert_array_equal(actual, expected)
        _assert_workspace(name, module, stack)
        np.testing.assert_array_equal(entry(data), np.median(stack, axis=0).astype(dtype))
    np.testing.assert_array_equal(stack, before)


@pytest.mark.parametrize('entry_name', ['chunk', 'frames'])
def test_auto_rgb_skip_zero_really_executes_gpu(gpu, monkeypatch, entry_name):
    name, module = gpu
    monkeypatch.setattr(_dispatch, '_backend_preference_override', 'auto')
    stack = np.zeros((8, 128, 192, 4), np.uint16)
    stack[..., 3] = 1
    stack[-1, ..., :3] = [1000, 0, 2000]
    assert median._select_median_backend('auto', stack).backend == f'{name}_host_io'
    backend_name = f'median_reduce_{entry_name}_{name}'
    with mock.patch.object(median, backend_name, wraps=getattr(median, backend_name)) as called:
        with mock.patch.object(median, 'median_reduce_chunk_compiled',
                               side_effect=AssertionError('unexpected CPU fallback')):
            result = getattr(median, f'median_reduce_{entry_name}')(
                stack if entry_name == 'chunk' else tuple(stack), skip_zero_rgb=True)
    called.assert_called_once()
    np.testing.assert_array_equal(result, stack[-1])



def test_gpu_frames_cast_before_rgb_validity(gpu):
    name, _ = gpu
    frames = [np.array([[[0, 0, 0, 255]]], np.uint8),
              np.array([[[256, 0, 0, 255]]], np.uint16),
              np.array([[[10, 0, 20, 8]]], np.uint16)]
    result = getattr(median, f'median_reduce_frames_{name}')(frames, skip_zero_rgb=True)
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result, frames[-1])
