import unittest
from unittest import mock

import cv2
import numpy as np

from hoshicore._custom_op.backend_registry import registered_backend_candidates
from hoshicore._custom_op.ops import wavelet as wavelet_ops
import hoshicore.component.norma.detection as detection


class TestWaveletDecRecCustomOp(unittest.TestCase):
    def tearDown(self) -> None:
        wavelet_ops._load_compiled_module_result.cache_clear()
        wavelet_ops._select_wavelet_dec_rec_backend.cache_clear()

    def test_wavelet_dec_rec_core_compiled_matches_pywavelets(self) -> None:
        rng = np.random.default_rng(0)
        cases = [
            ((32, 33), 2),
            ((25, 25), 4),
            ((64, 64), 4),
            ((100, 101), 3),
            ((101, 100), 6),
            ((128, 192), 4),
        ]

        for shape, level in cases:
            with self.subTest(shape=shape, level=level):
                image = np.ascontiguousarray(rng.normal(size=shape))
                expected = wavelet_ops.wavelet_dec_rec_core_numpy(image, level)
                got = wavelet_ops.wavelet_dec_rec_core_compiled(image, level)

                self.assertEqual(got.shape, expected.shape)
                np.testing.assert_allclose(
                    got, expected, rtol=1e-10, atol=1e-12)

    def test_wavelet_dec_rec_wrapper_matches_pywavelets(self) -> None:
        rng = np.random.default_rng(1)
        image = rng.normal(size=(101, 100))

        for resize_factor in (1.0, 0.5, 0.25):
            with self.subTest(resize_factor=resize_factor):
                expected = wavelet_ops.wavelet_dec_rec_core_numpy(
                    cv2.resize(image, None, fx=resize_factor, fy=resize_factor),
                    wavelet_ops._wavelet_level(resize_factor),
                )
                expected = cv2.resize(expected, (image.shape[1], image.shape[0]))
                got = wavelet_ops.wavelet_dec_rec(image, resize_factor)

                self.assertEqual(got.shape, image.shape)
                np.testing.assert_allclose(
                    got, expected, rtol=1e-10, atol=1e-12)

    def test_wavelet_dec_rec_wrapper_can_use_compiled_backend(self) -> None:
        rng = np.random.default_rng(11)
        image = rng.normal(size=(24, 26))
        expected = wavelet_ops.wavelet_dec_rec(image, 1.0)

        with mock.patch.dict(
                "os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "auto"},
                clear=False):
            with mock.patch.object(
                    wavelet_ops, "MIN_COMPILED_WAVELET_PIXELS", 0):
                with mock.patch.object(
                        wavelet_ops,
                        "wavelet_dec_rec_core_compiled",
                        wraps=wavelet_ops.wavelet_dec_rec_core_compiled,
                ) as compiled:
                    with mock.patch.object(
                            wavelet_ops, "_apply_compiled_threads") as apply_threads:
                        wavelet_ops._select_wavelet_dec_rec_backend.cache_clear()
                        got = wavelet_ops.wavelet_dec_rec(image, 1.0)

        compiled.assert_called_once()
        apply_threads.assert_called_once()
        np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)

    def test_wavelet_dec_rec_can_force_numpy_fallback(self) -> None:
        rng = np.random.default_rng(2)
        image = rng.normal(size=(32, 34))

        with mock.patch.dict(
                "os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"},
                clear=False):
            with mock.patch.object(
                    wavelet_ops,
                    "_load_compiled_module_result",
                    return_value=(None, "mock error")):
                with mock.patch.object(
                        wavelet_ops,
                        "MIN_COMPILED_WAVELET_PIXELS",
                        0):
                    wavelet_ops._select_wavelet_dec_rec_backend.cache_clear()
                    got = wavelet_ops.wavelet_dec_rec(image, 0.5)

        expected = wavelet_ops.wavelet_dec_rec_core_numpy(
            cv2.resize(image, None, fx=0.5, fy=0.5),
            wavelet_ops._wavelet_level(0.5),
        )
        expected = cv2.resize(expected, (image.shape[1], image.shape[0]))
        np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)

    def test_wavelet_dec_rec_missing_compiled_backend_falls_back(self) -> None:
        rng = np.random.default_rng(3)
        image = rng.normal(size=(30, 32))
        expected = wavelet_ops.wavelet_dec_rec(image, 1.0)

        with mock.patch.dict(
                "os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "auto"},
                clear=False):
            with mock.patch.object(
                    wavelet_ops, "_load_compiled_module_result",
                    return_value=(None, "mock error")) as load_module:
                with mock.patch.object(
                        wavelet_ops, "MIN_COMPILED_WAVELET_PIXELS", 0):
                    wavelet_ops._select_wavelet_dec_rec_backend.cache_clear()
                    got = wavelet_ops.wavelet_dec_rec(image, 1.0)

        load_module.assert_called_once()
        np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)

    def test_wavelet_dec_rec_backend_registered(self) -> None:
        candidates = registered_backend_candidates("wavelet_dec_rec")
        self.assertTrue(
            any(candidate.kernel_name == "wavelet_dec_rec_cpu"
                and candidate.backend == "openmp_cpu"
                for candidate in candidates))

    def test_wavelet_dec_rec_public_facade_exported(self) -> None:
        from hoshicore._custom_op import wavelet_dec_rec
        from hoshicore._custom_op.api import wavelet_dec_rec as api_wavelet_dec_rec

        self.assertIs(wavelet_dec_rec, wavelet_ops.wavelet_dec_rec)
        self.assertIs(api_wavelet_dec_rec, wavelet_ops.wavelet_dec_rec)

    def test_detection_routes_wavelet_through_custom_op(self) -> None:
        image = np.ones((8, 10), dtype=np.float64)
        expected = np.full_like(image, 3.0)

        with mock.patch.object(
                detection, "wavelet_dec_rec", return_value=expected) as patched:
            got = detection._wavelet_dec_rec(image, resize_factor=0.5)

        patched.assert_called_once_with(image, resize_factor=0.5)
        self.assertIs(got, expected)
