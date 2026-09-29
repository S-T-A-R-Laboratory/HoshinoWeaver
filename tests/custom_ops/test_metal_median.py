"""Exact uint16 Metal median and its host-I/O workspace contract."""

import numpy as np
import pytest
from unittest import mock

from hoshicore._custom_op import metal_memory
import hoshicore._custom_op.backend_registry as backend_registry
from hoshicore._custom_op._dispatch import load_metal_module
from hoshicore._custom_op._dispatch import CustomOpResourceExhaustedError
from hoshicore._custom_op.backend_registry import BackendCandidate, BackendSelection
from hoshicore._custom_op.ops import detection as detection_ops
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
    fused = metal_memory.estimate_median_star_mask(height=17, width=19)
    assert fused.logical_op == "median_star_mask"
    assert fused.peak_device_bytes == estimate.peak_device_bytes


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


def test_cpu_finisher_with_precomputed_background_matches_fused():
    from hoshicore._custom_op._dispatch import load_compiled_module

    module, reason = load_compiled_module()
    if module is None:
        pytest.skip(reason or "CPU custom ops unavailable")
    rng = np.random.default_rng(93)
    image = rng.integers(0, 65536, (37, 45), dtype=np.uint16)
    mask = (rng.random(image.shape) > 0.15).astype(np.uint8)
    previous_threads = module.get_openmp_max_threads()
    module.set_openmp_threads(1)
    try:
        background = module.median_filter_2d(image, 13)
        expected = module.median_star_mask_cpu(image, 13, 1.2, 3, 3, mask)
        actual = module.median_star_mask_with_background_cpu(image, background, 13, 1.2, 3, 3, mask)
    finally:
        module.set_openmp_threads(previous_threads)
    for got, wanted in zip(actual, expected):
        np.testing.assert_array_equal(got, wanted)


def test_cpu_finisher_rejects_incompatible_background():
    from hoshicore._custom_op._dispatch import load_compiled_module

    module, reason = load_compiled_module()
    if module is None:
        pytest.skip(reason or "CPU custom ops unavailable")
    image = np.ones((17, 19), dtype=np.uint16)
    for background in (np.ones((17, 18), dtype=np.uint16),
                       np.ones((17, 19), dtype=np.float32)):
        with pytest.raises(ValueError):
            module.median_star_mask_with_background_cpu(image, background, 13, 1.0, 3, 0, None)


def test_metal_composed_mask_matches_cpu_when_available(monkeypatch):
    _metal_or_skip()
    monkeypatch.setenv("HNW_CUSTOM_OPS_FALLBACK", "auto")
    image = np.random.default_rng(94).integers(0, 65536, (47, 53), dtype=np.uint16)
    mask = np.ones(image.shape, dtype=np.uint8)
    mask[::5, ::7] = 0
    expected = detection_ops.median_star_mask_cpu_compiled(image, 13, 1.1, 3, 3, mask)
    actual = detection_ops.median_star_mask_compiled_metal(image, 13, 1.1, 3, 3, mask)
    for got, wanted in zip(actual[:2], expected[:2]):
        np.testing.assert_array_equal(got, wanted)
    np.testing.assert_allclose(actual[2], expected[2], rtol=1e-14, atol=1e-15)


def _selection(backend: str, kernel: str) -> BackendSelection:
    return BackendSelection(BackendCandidate("median_star_mask", backend, kernel), object())


def test_metal_median_resource_error_falls_back_to_cpu(monkeypatch):
    image = np.random.default_rng(95).integers(0, 65536, (31, 37), dtype=np.uint16)
    expected = detection_ops.median_star_mask_cpu_compiled(image)
    metal = _selection("metal_host_io", "median_filter_2d_metal")
    cpu = _selection("openmp_cpu", "median_star_mask_cpu")
    monkeypatch.setattr(detection_ops, "_select_median_star_mask_backend", lambda _: metal)
    monkeypatch.setattr(detection_ops, "median_star_mask_compiled_metal",
                        mock.Mock(side_effect=CustomOpResourceExhaustedError("Metal capacity")),
                        raising=False)
    with mock.patch.object(backend_registry, "resolve_after_accelerator_failure", return_value=cpu) as resolve:
        got = detection_ops.median_star_mask(image)
    resolve.assert_called_once()
    for actual, wanted in zip(got, expected):
        np.testing.assert_array_equal(actual, wanted)
