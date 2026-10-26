"""Exact Metal frame-stack median and fallback behavior."""

from unittest import mock

import numpy as np
import pytest

from hoshicore._custom_op import metal_memory
import hoshicore._custom_op.backend_registry as backend_registry
from hoshicore._custom_op._dispatch import CustomOpResourceExhaustedError
from hoshicore._custom_op._dispatch import load_metal_module
from hoshicore._custom_op.backend_registry import BackendSelection
from hoshicore._custom_op.backend_registry import registered_backend_candidates
from hoshicore._custom_op.ops import median as median_ops


def _metal_or_skip():
    module, reason = load_metal_module()
    if module is None:
        pytest.skip(reason or "Metal extension unavailable")
    if not module.metal_device_info().get("available"):
        pytest.skip("Metal runtime unavailable")
    return module


def test_metal_median_stack_estimate_matches_two_buffers():
    estimate = metal_memory.estimate_median_reduce_chunk(
        n_frames=16, plane_size=32 * 1536 * 3, dtype_bytes=2
    )
    assert estimate.logical_op == "median_reduce_chunk"
    assert estimate.peak_device_bytes == 17 * 32 * 1536 * 3 * 2
    assert estimate.confidence == "exact"


@pytest.mark.parametrize("dtype", (np.uint8, np.uint16))
@pytest.mark.parametrize("n_frames", (1, 2, 3, 8, 15, 16, 17, 31, 32, 64, 127, 128))
def test_metal_median_stack_matches_numpy_exactly(dtype, n_frames):
    module = _metal_or_skip()
    rng = np.random.default_rng(281 + n_frames)
    stack = rng.integers(
        0, np.iinfo(dtype).max + 1, size=(n_frames, 7, 13, 3), dtype=dtype
    )
    stack[:, 0, 0] = np.iinfo(dtype).max
    stack[:, 0, 1] = 0
    expected = np.median(stack, axis=0).astype(dtype)
    np.testing.assert_array_equal(module.median_reduce_chunk_metal(stack), expected)
    np.testing.assert_array_equal(
        module.median_reduce_chunk_metal_frames(tuple(stack)), expected
    )
    estimate = metal_memory.estimate_median_reduce_chunk(
        n_frames=n_frames, plane_size=stack[0].size, dtype_bytes=stack.dtype.itemsize
    )
    assert module.metal_host_io_cache_info()["last_logical_peak_bytes"] == estimate.peak_device_bytes


def test_metal_median_stack_resource_error_falls_back_to_cpu(monkeypatch):
    frames = [np.full((7, 11), frame, dtype=np.uint16) for frame in range(16)]
    metal = next(
        candidate for candidate in registered_backend_candidates("median_reduce_chunk")
        if candidate.backend == "metal_host_io"
    )
    cpu = next(
        candidate for candidate in registered_backend_candidates("median_reduce_chunk")
        if candidate.backend == "openmp_cpu"
    )
    selection = BackendSelection(metal, object())
    monkeypatch.setattr(median_ops, "_resolve_median_selection", lambda *args: selection)
    with mock.patch.object(backend_registry, "resolve_after_accelerator_failure",
                           return_value=BackendSelection(cpu, object())):
        with mock.patch.object(
            median_ops, "median_reduce_frames_metal",
            side_effect=CustomOpResourceExhaustedError("Metal capacity"),
        ):
            got = median_ops.median_reduce_frames(frames)
    np.testing.assert_array_equal(got, np.full((7, 11), 7, dtype=np.uint16))
