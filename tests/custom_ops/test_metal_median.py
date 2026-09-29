"""Exact uint16 Metal median and its host-I/O workspace contract."""

import numpy as np
import pytest

from hoshicore._custom_op import metal_memory
from hoshicore._custom_op._dispatch import load_metal_module
from hoshicore._custom_op.ops import filter as filter_ops


def _metal_or_skip():
    module, reason = load_metal_module()
    if module is None:
        pytest.skip(reason or "Metal extension unavailable")
    if not module.metal_device_info().get("available"):
        pytest.skip("Metal runtime unavailable")
    return module


def test_metal_median_estimate_counts_two_uint16_planes():
    estimate = metal_memory.estimate_median_filter_2d(height=17, width=19)
    assert estimate.peak_device_bytes == 17 * 19 * 2 * 2
    assert estimate.confidence == "exact"


@pytest.mark.parametrize("shape", [(1, 1), (2, 3), (13, 13), (31, 39), (96, 127)])
def test_metal_median_matches_cpu_at_edges_and_interior(shape):
    module = _metal_or_skip()
    rng = np.random.default_rng(20260929)
    source = rng.integers(0, 65536, shape, dtype=np.uint16)
    source[0, 0] = 65535
    source[-1, -1] = 0
    expected = filter_ops.median_filter_2d_compiled(source, 13)
    result = filter_ops.median_filter_2d_compiled_metal(source, 13)
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.uint16 and result.shape == shape
    assert module.metal_host_io_cache_info()["last_logical_peak_bytes"] == source.size * 4


def test_metal_median_handles_constant_and_repeated_values():
    _metal_or_skip()
    source = np.full((35, 41), 12000, dtype=np.uint16)
    source[0, :] = 65535
    source[20, 7] = 0
    np.testing.assert_array_equal(
        filter_ops.median_filter_2d_compiled_metal(source, 13),
        filter_ops.median_filter_2d_compiled(source, 13),
    )


def test_metal_median_rejects_other_layouts_and_ksizes():
    module = _metal_or_skip()
    source = np.ones((17, 19), dtype=np.uint16)
    with pytest.raises(ValueError):
        module.median_filter_2d_metal(source, 11)
    with pytest.raises(ValueError):
        module.median_filter_2d_metal(source.astype(np.uint8), 13)
    with pytest.raises(ValueError):
        module.median_filter_2d_metal(source[..., None], 13)


def test_metal_median_wrapper_accepts_strided_input():
    _metal_or_skip()
    source = np.random.default_rng(91).integers(0, 65536, (33, 54), dtype=np.uint16)[:, ::2]
    np.testing.assert_array_equal(
        filter_ops.median_filter_2d_compiled_metal(source, 13),
        filter_ops.median_filter_2d_compiled(source, 13),
    )
