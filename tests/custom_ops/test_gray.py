import cv2
import numpy as np
import pytest

from hoshicore._custom_op.ops import gray as gray_ops


def _original_gray(image):
    gray = (cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2GRAY).astype(np.float64)
            if image.ndim == 3 else image.astype(np.float64))
    if np.issubdtype(image.dtype, np.integer):
        gray /= np.iinfo(image.dtype).max
    elif gray.max() > 1.0:
        gray /= gray.max()
    return gray


@pytest.fixture
def native():
    module, error = gray_ops._load_compiled_module_result()
    if module is None:
        pytest.skip(error or "compiled custom ops unavailable")
    return module


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("ipp", [False, True])
def test_compiled_gray_matches_current_opencv(native, dtype, channels, ipp):
    previous = cv2.ipp.useIPP()
    cv2.ipp.setUseIPP(ipp)
    try:
        rng = np.random.default_rng(31)
        for width in (1, 4, 7, 8, 13, 64):
            shape = (9, width * 2) if channels == 1 else (9, width * 2, channels)
            image = rng.integers(0, np.iinfo(dtype).max + 1, shape, dtype=dtype)[:, ::2]
            expected = _original_gray(image)
            np.testing.assert_array_equal(gray_ops.detection_gray_f64_compiled(image), expected)
            if dtype == np.uint16:
                expected_u16 = np.rint(expected.astype(np.float32) * np.float32(65535)).astype(np.uint16)
                np.testing.assert_array_equal(
                    gray_ops.detection_gray_u16_compiled(image), expected_u16)
    finally:
        cv2.ipp.setUseIPP(previous)


def test_native_quantization_ties_and_range(native):
    values = np.array([[0, 0.5, 1.5, 2.5, 65534.5, 65535]], dtype=np.float32)
    np.testing.assert_array_equal(
        native.detection_gray_cast_cpu(values, "quantize"), [[0, 0, 2, 2, 65534, 65535]])
    for value in (-1, 65535.5, np.nan, np.inf):
        values[0, 1] = value
        assert native.detection_gray_cast_cpu(values, "quantize") is None


@pytest.mark.parametrize("preference", ["cpu", "numpy"])
def test_public_gray_dispatch_matches_reference(monkeypatch, preference):
    monkeypatch.setenv("HNW_CUSTOM_OPS_FALLBACK", preference)
    rng = np.random.default_rng(32)
    image = rng.integers(0, 65536, (257, 300, 3), dtype=np.uint16)
    np.testing.assert_array_equal(gray_ops.detection_gray_f64(image), _original_gray(image))
    np.testing.assert_array_equal(
        gray_ops.detection_gray_u16(image), gray_ops.detection_gray_u16_numpy(image))


def test_gray_without_extension_uses_reference(monkeypatch):
    monkeypatch.setenv("HNW_CUSTOM_OPS_FALLBACK", "auto")
    monkeypatch.setattr(gray_ops, "_load_compiled_module_result", lambda: (None, "missing"))
    image = np.arange(120, dtype=np.uint16).reshape(5, 8, 3)
    np.testing.assert_array_equal(gray_ops.detection_gray_f64(image), _original_gray(image))
    np.testing.assert_array_equal(
        gray_ops.detection_gray_u16(image), gray_ops.detection_gray_u16_numpy(image))


def test_native_gray_errors_propagate(native, monkeypatch):
    monkeypatch.setenv("HNW_CUSTOM_OPS_FALLBACK", "cpu")

    def fail(*args):
        raise RuntimeError("native conversion failed")

    monkeypatch.setattr(native, "detection_gray_cast_cpu", fail)
    with pytest.raises(RuntimeError, match="native conversion failed"):
        gray_ops.detection_gray_f64(np.ones((3, 4, 3), dtype=np.uint16))


def test_native_cast_rejects_invalid_types_and_scale(native):
    with pytest.raises(ValueError):
        native.detection_gray_cast_cpu(np.ones((3, 4), dtype=np.float64), "prepare")
    with pytest.raises(ValueError):
        native.detection_gray_cast_cpu(np.ones((3, 4), dtype=np.uint16), "normalize", 0.0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32])
def test_other_dtypes_keep_original_path(dtype):
    image = np.arange(120, dtype=dtype).reshape(5, 8, 3)
    np.testing.assert_array_equal(gray_ops.detection_gray_f64(image), _original_gray(image))
    assert gray_ops.detection_gray_u16(image) is None


def test_gray_is_thread_count_independent(native, monkeypatch):
    image = np.random.default_rng(33).integers(0, 65536, (257, 300, 3), dtype=np.uint16)
    for backend in (gray_ops.detection_gray_f64_compiled, gray_ops.detection_gray_u16_compiled):
        monkeypatch.setenv("HNW_CUSTOM_OPS_THREADS", "1")
        expected = backend(image)
        monkeypatch.setenv("HNW_CUSTOM_OPS_THREADS", "3")
        np.testing.assert_array_equal(backend(image), expected)
