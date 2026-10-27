"""RGB padding exclusion is independent of alignment and frame-buffer metadata."""
import numpy as np
import pytest

from hoshicore._custom_op import _dispatch
from hoshicore._custom_op.ops import fgp, median
from hoshicore.component.data_container import HuberMeanParam


@pytest.fixture(params=['numpy', 'cpu', 'auto'])
def backend(request, monkeypatch):
    monkeypatch.setattr(_dispatch, '_backend_preference_override', request.param)


def _huber_inputs(dtype, channels):
    stack = np.zeros((4, 1, 3, channels), dtype=dtype)
    for index, value in enumerate((90, 100, 110), start=1):
        stack[index, 0, 0, :3] = value
        stack[index, 0, 1, :3] = [0, value, value]
    if channels == 4:
        stack[..., 3] = 100
    mean = np.full(stack.shape[1:], 100, dtype=np.float32)
    std = np.full_like(mean, 10)
    return stack, mean, std


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16, np.float32, np.float64])
@pytest.mark.parametrize('channels', [3, 4])
@pytest.mark.parametrize('array_weight', [False, True])
def test_huber_accumulation_excludes_padding_from_both_sums(
        backend, dtype, channels, array_weight):
    stack, mean, std = _huber_inputs(dtype, channels)
    result = HuberMeanParam(np.zeros_like(mean, np.float64),
                            np.zeros_like(mean, np.float64), source_dtype=stack.dtype)
    for frame in stack:
        weight = np.ones_like(mean) if array_weight else 1.0
        fgp.huber_weighted_accumulate(result, frame, mean, std, 1.345,
                                      weight, skip_zero_rgb=True)
    np.testing.assert_allclose(result.weighted_sum[0, 0, :3], 300)
    np.testing.assert_allclose(result.weight_total[0, 0, :3], 3)
    # A zero in just one color channel is retained, including its weight.
    assert result.weighted_sum[0, 1, 0] == 0
    assert result.weight_total[0, 1, 0] > 0
    np.testing.assert_allclose(result.weight_total[0, 1, 1:3], 3)
    np.testing.assert_array_equal(result.weighted_sum[0, 2], 0)
    np.testing.assert_array_equal(result.weight_total[0, 2], 0)


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
@pytest.mark.parametrize('channels', [3, 4])
@pytest.mark.parametrize('weighted', [False, True])
def test_huber_chunk_excludes_padding_with_scalar_frame_weights(
        backend, dtype, channels, weighted):
    stack, mean, std = _huber_inputs(dtype, channels)
    weights = np.array([1., 1., 2., .5]) if weighted else None
    kwargs = dict(stack=stack.reshape(4, -1), ref_mean=mean.ravel(),
                  ref_std=std.ravel(), huber_c=1.345, weights=weights,
                  skip_zero_rgb=True, channels=channels)
    actual_sum, actual_weight = fgp.huber_weighted_chunk(**kwargs)
    expected_sum = 345 if weighted else 300
    expected_weight = 3.5 if weighted else 3
    np.testing.assert_allclose(actual_sum[:3], expected_sum)
    np.testing.assert_allclose(actual_weight[:3], expected_weight)
    assert actual_weight[channels] > 0
    np.testing.assert_array_equal(actual_sum[2 * channels:], 0)
    np.testing.assert_array_equal(actual_weight[2 * channels:], 0)


@pytest.mark.parametrize('channels', [1, 2, 4])
def test_huber_chunk_validates_rgb_channel_layout(channels):
    with pytest.raises(ValueError, match='channels'):
        fgp.huber_weighted_chunk_numpy(
            np.zeros((2, 9), np.uint16), np.zeros(9), np.ones(9), 1.345,
            skip_zero_rgb=True, channels=channels)


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16, np.float32, np.float64])
@pytest.mark.parametrize('channels', [1, 2, 3, 4])
def test_median_zero_rgb_semantics(backend, dtype, channels):
    rng = np.random.default_rng(42)
    stack = rng.integers(1, 100, (5, 2, 12, channels)).astype(dtype)[:, :, ::2]
    for pixel in range(6):
        stack[:min(pixel, 5), :, pixel, :3] = 0
    if channels >= 3:
        stack[-1, :, 0, 0] = 0  # A single zero channel remains valid.
    if channels == 4:
        stack[..., 3] = 200  # Alpha cannot rescue invalid RGB.
    before = stack.copy()
    expected = np.zeros(stack.shape[1:], dtype=dtype)
    for row in range(2):
        for pixel in range(6):
            samples = stack[:, row, pixel]
            if channels >= 3:
                samples = samples[np.any(samples[:, :3] != 0, axis=1)]
            if len(samples):
                expected[row, pixel] = np.median(samples, axis=0)
    for entry, data in [(median.median_reduce_chunk, stack),
                        (median.median_reduce_frames, list(stack))]:
        actual = entry(data, skip_zero_rgb=True)
        assert actual.dtype == dtype
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(entry(data), np.median(stack, axis=0).astype(dtype))
    np.testing.assert_array_equal(stack, before)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_median_valid_nan_propagates(backend, dtype):
    stack = np.array([[[[0, 0, 0, np.nan]]], [[[0, 4, 6, 8]]],
                      [[[np.nan, 6, 8, 10]]]], dtype=dtype)
    for entry, data in [(median.median_reduce_chunk, stack),
                        (median.median_reduce_frames, list(stack))]:
        np.testing.assert_array_equal(entry(data, skip_zero_rgb=True),
                                      np.array([[[np.nan, 5, 7, 9]]], dtype=dtype))


def test_median_without_native(monkeypatch):
    monkeypatch.setattr(median, '_load_compiled_module_result', lambda: (None, 'missing'))
    stack = np.array([[[[0, 0, 0]]], [[[10, 0, 20]]]], np.uint16)
    for entry, data in [(median.median_reduce_chunk, stack),
                        (median.median_reduce_frames, list(stack))]:
        np.testing.assert_array_equal(entry(data, skip_zero_rgb=True), stack[1])


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_median_single_large_valid_sample(backend, dtype):
    stack = np.zeros((2, 1, 1, 3), dtype=dtype)
    stack[1] = np.finfo(dtype).max
    np.testing.assert_array_equal(
        median.median_reduce_chunk(stack, skip_zero_rgb=True), stack[1])


@pytest.mark.parametrize('shape', [(3, 2, 3), (3, 2, 3, 1)])
def test_median_grayscale_is_unchanged(backend, shape):
    stack = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    stack[0] = 0
    np.testing.assert_array_equal(
        median.median_reduce_chunk(stack, skip_zero_rgb=True),
        np.median(stack, axis=0).astype(stack.dtype))


@pytest.mark.parametrize('entry_name', ['chunk', 'frames'])
@pytest.mark.parametrize('gpu_name', ['cuda', 'metal'])
@pytest.mark.parametrize('fallback', ['numpy', 'cpu'])
def test_median_resource_fallback_keeps_skip_zero(monkeypatch, entry_name, gpu_name, fallback):
    from hoshicore._custom_op import backend_registry

    candidate = next(c for c in backend_registry.registered_backend_candidates(
        'median_reduce_chunk') if c.backend == f'{gpu_name}_host_io')
    # Inject an accelerator selection to exercise argument forwarding on retry.
    monkeypatch.setattr(median, '_resolve_median_selection',
                        lambda *args: backend_registry.BackendSelection(candidate, object()))
    if fallback == 'numpy':
        monkeypatch.setattr(median, '_load_compiled_module_result', lambda: (None, 'missing'))
    else:
        selected_cpu = median._resolve_backend(
            'median_reduce_chunk', 'cpu', load_module=median._load_compiled_module_result)
        monkeypatch.setattr(backend_registry, 'resolve_after_accelerator_failure',
                            lambda *args, **kwargs: selected_cpu)
    monkeypatch.setattr(backend_registry, 'load_metal_module', lambda: (None, 'missing'))
    calls = []
    def exhausted(data, skip_zero_rgb=False):
        calls.append(skip_zero_rgb)
        raise _dispatch.CustomOpResourceExhaustedError('test resource exhaustion')
    monkeypatch.setattr(median, f'median_reduce_{entry_name}_{gpu_name}', exhausted)
    stack = np.zeros((8, 128, 192, 3), np.uint16)
    stack[-1] = [10, 0, 20]
    data = stack if entry_name == 'chunk' else list(stack)
    result = getattr(median, f'median_reduce_{entry_name}')(data, skip_zero_rgb=True)
    assert calls == [True]
    np.testing.assert_array_equal(result, stack[-1])
