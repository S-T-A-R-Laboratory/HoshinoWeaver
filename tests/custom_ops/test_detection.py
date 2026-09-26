import unittest
from unittest import mock

import cv2
import numpy as np

from hoshicore._custom_op import build_info
from hoshicore._custom_op.backend_registry import registered_backend_candidates
from hoshicore._custom_op.backend_registry import BackendCandidate
from hoshicore._custom_op.backend_registry import BackendSelection
from hoshicore._custom_op._dispatch import CustomOpResourceExhaustedError
from hoshicore._custom_op._dispatch import CustomOpUnavailableError
from hoshicore._custom_op._dispatch import is_cuda_runtime_unavailable_error
from hoshicore._custom_op.ops import detection as detection_ops
import hoshicore.component.norma.detection as star_detection
import hoshicore._custom_op.backend_registry as backend_registry
import hoshicore._custom_op.cuda_memory as cuda_memory


def _is_compiled_backend_unavailable(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return (
        is_cuda_runtime_unavailable_error(exc)
        or "compiled custom op backend is unavailable" in message
        or "compiled cuda custom op backend is unavailable" in message
    )


def _threshold_morph_reference(
    image: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    mask_bool = mask > 0
    bw = ((image > np.percentile(image[mask_bool], 99.5)) * mask_bool).astype(
        np.uint8) * 255
    return cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


class TestStarDetectCustomOps(unittest.TestCase):
    def tearDown(self) -> None:
        detection_ops._load_compiled_module_result.cache_clear()
        detection_ops._select_median_star_mask_backend.cache_clear()
        detection_ops._select_star_detect_fused_pixel_components_backend.cache_clear()

    def test_median_star_mask_backend_registered(self) -> None:
        candidates = registered_backend_candidates("median_star_mask")
        self.assertTrue(
            any(candidate.kernel_name == "median_star_mask_cpu"
                and candidate.backend == "openmp_cpu"
                for candidate in candidates))

    def test_median_star_mask_cpu_matches_numpy(self) -> None:
        rng = np.random.default_rng(19)
        for dtype in (np.uint8, np.uint16, np.float32, np.float64):
            with self.subTest(dtype=np.dtype(dtype).name):
                if np.issubdtype(dtype, np.integer):
                    image = rng.integers(
                        0,
                        np.iinfo(dtype).max + 1,
                        size=(37, 41),
                        dtype=dtype,
                    )
                else:
                    image = rng.random((37, 41)).astype(dtype)
                mask = rng.random(image.shape) > 0.2
                expected = detection_ops.median_star_mask_numpy(
                    image,
                    median_ksize=13,
                    threshold_ratio=1.2,
                    open_ksize=3,
                    dilate_ksize=3,
                    mask=mask,
                )
                got = detection_ops.median_star_mask_cpu_compiled(
                    image,
                    median_ksize=13,
                    threshold_ratio=1.2,
                    open_ksize=3,
                    dilate_ksize=3,
                    mask=mask,
                )

                np.testing.assert_array_equal(got[0], expected[0])
                np.testing.assert_array_equal(got[1], expected[1])
                self.assertAlmostEqual(got[2], expected[2], places=7)

    def test_median_star_mask_large_mask_decisions_match_numpy(self) -> None:
        rng = np.random.default_rng(41)
        image = rng.random((257, 263), dtype=np.float32)
        image[::17, ::19] = 1.0

        expected = detection_ops.median_star_mask_numpy(image)
        got = detection_ops.median_star_mask_cpu_compiled(image)

        np.testing.assert_array_equal(got[0], expected[0])
        np.testing.assert_array_equal(got[1], expected[1])
        np.testing.assert_allclose(got[2], expected[2], rtol=5e-6, atol=1e-10)

    def test_median_star_mask_can_force_numpy_fallback(self) -> None:
        image = np.arange(31 * 33, dtype=np.uint16).reshape(31, 33)
        expected = detection_ops.median_star_mask_numpy(image)

        with mock.patch.dict(
                "os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"},
                clear=False):
            detection_ops._select_median_star_mask_backend.cache_clear()
            with mock.patch.object(
                    detection_ops,
                    "median_star_mask_cpu_compiled",
                    side_effect=AssertionError("compiled backend must not run")):
                got = detection_ops.median_star_mask(image)

        np.testing.assert_array_equal(got[0], expected[0])
        np.testing.assert_array_equal(got[1], expected[1])
        self.assertEqual(got[2], expected[2])

    def test_norma_large_median_routes_through_fused_mask(self) -> None:
        image = np.linspace(0.0, 1.0, num=31 * 33, dtype=np.float64).reshape(
            31, 33)
        expected_mask = np.zeros(image.shape, dtype=np.uint8)
        expected_response = np.zeros(image.shape, dtype=np.float32)

        with mock.patch.object(
                star_detection,
                "median_star_mask",
                return_value=(expected_mask, expected_response, 0.125)) as fused:
            got = star_detection.detect_starmask_by_threshold_with_response(
                image,
                ksize=13,
                threshold_ratio=1.5,
                open_ksize=3,
                dilate_ksize=5,
            )

        fused.assert_called_once()
        args, kwargs = fused.call_args
        np.testing.assert_array_equal(
            args[0],
            np.rint(image.astype(np.float32) * np.float32(65535.0)).astype(
                np.uint16
            ),
        )
        self.assertEqual(
            kwargs,
            {
                "median_ksize": 13,
                "threshold_ratio": 1.5,
                "open_ksize": 3,
                "dilate_ksize": 5,
                "mask": None,
            },
        )
        self.assertIs(got[0], expected_mask)
        self.assertIs(got[1], expected_response)

    def test_star_detect_threshold_morph_numpy_matches_reference(self) -> None:
        rng = np.random.default_rng(0)
        image = rng.normal(size=(37, 41)).astype(np.float64)
        mask = rng.random(size=image.shape) > 0.2

        expected = _threshold_morph_reference(image, mask)
        got = detection_ops.star_detect_threshold_morph_numpy(image, mask)

        np.testing.assert_array_equal(got, expected)

    def test_star_detect_fused_pixel_components_backend_registered(self) -> None:
        candidates = registered_backend_candidates(
            "star_detect_fused_pixel_components")
        self.assertTrue(
            any(candidate.kernel_name == "star_detect_fused_pixel_components_cuda"
                and candidate.backend == "cuda_host_io"
                and candidate.build_flag == "cuda"
                for candidate in candidates))
        self.assertTrue(
            any(candidate.kernel_name == "star_detect_fused_pixel_components_cpu"
                and candidate.backend == "openmp_cpu"
                for candidate in candidates))

    def test_star_detect_fused_pixel_components_rejects_numpy_fallback(
            self) -> None:
        image = np.zeros((64, 64), dtype=np.float64)
        mask = np.ones(image.shape, dtype=np.uint8)

        with mock.patch.dict(
                "os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"},
                clear=False):
            detection_ops._select_star_detect_fused_pixel_components_backend.cache_clear()
            with self.assertRaisesRegex(CustomOpUnavailableError, "requires a compiled backend"):
                detection_ops.star_detect_fused_pixel_components(
                    image, mask, 1.0)

    def test_star_detect_fused_pixel_components_wrapper_can_use_compiled_backend(
            self) -> None:
        image = np.arange(64, dtype=np.float32).reshape(8, 8)
        mask = np.zeros(image.shape, dtype=bool)
        mask[1:-1, 2:-2] = True
        expected = (
            np.array([[2.0, 3.0]], dtype=np.float64),
            np.array([5.0], dtype=np.float64),
            np.ones(image.shape, dtype=np.uint8),
        )

        compiled = mock.Mock(return_value=expected)
        candidate = BackendCandidate(
            "star_detect_fused_pixel_components",
            "openmp_cpu",
            "star_detect_fused_pixel_components_cpu",
        )
        selection = BackendSelection(candidate, object())
        with mock.patch.dict(
                "os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "auto"},
                clear=False):
            with mock.patch.object(
                    detection_ops,
                    "_select_star_detect_fused_pixel_components_backend",
                    return_value=selection):
                with mock.patch.object(
                        detection_ops,
                        "_star_detect_fused_pixel_components_cpu_validated",
                        compiled):
                    got = detection_ops.star_detect_fused_pixel_components(
                        image, mask, 0.5, gaussian_ksize=7, sigma=1.5)

        compiled.assert_called_once()
        args = compiled.call_args.args
        self.assertEqual(args[0].dtype, np.float64)
        np.testing.assert_allclose(args[0], image.astype(np.float64))
        self.assertEqual(args[1].dtype, np.uint8)
        np.testing.assert_array_equal(args[1], mask.astype(np.uint8))
        self.assertEqual(args[2], 0.5)
        self.assertEqual(args[3], 7)
        self.assertEqual(args[4], 1.5)
        self.assertIs(got, expected)

    def test_star_detect_public_wrapper_validates_mask_once(self) -> None:
        image = np.arange(64, dtype=np.float32).reshape(8, 8)
        mask = image > 4
        expected = (
            np.empty((0, 2), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            np.zeros(image.shape, dtype=np.uint8),
        )
        native = mock.Mock(return_value=expected)
        module = mock.Mock(star_detect_fused_pixel_components_cuda=native)
        candidate = BackendCandidate(
            "star_detect_fused_pixel_components",
            "cuda_host_io",
            "star_detect_fused_pixel_components_cuda",
        )
        selection = BackendSelection(candidate, module)
        admission = mock.MagicMock()
        admission.__enter__.return_value = mock.Mock(
            granted=True,
            estimated_peak_bytes=1024,
        )

        with mock.patch.object(
            detection_ops,
            "_select_star_detect_fused_pixel_components_backend",
            return_value=selection,
        ):
            with mock.patch.object(
                detection_ops,
                "_load_compiled_module_result",
                return_value=(module, None),
            ):
                with mock.patch.object(
                    cuda_memory,
                    "cuda_memory_admission",
                    return_value=admission,
                ):
                    with mock.patch.object(
                        detection_ops,
                        "_validate_fused_pixel_component_inputs",
                        wraps=detection_ops._validate_fused_pixel_component_inputs,
                    ) as validate:
                        got = detection_ops.star_detect_fused_pixel_components(
                            image,
                            mask,
                            1.0,
                        )

        validate.assert_called_once_with(image, mask)
        native.assert_called_once()
        self.assertIs(got, expected)

    def test_star_detect_cuda_runtime_unavailable_falls_back_to_cpu(self) -> None:
        image = np.arange(64, dtype=np.float64).reshape(8, 8)
        expected = (
            np.array([[2.0, 3.0]], dtype=np.float64),
            np.array([5.0], dtype=np.float64),
            np.ones(image.shape, dtype=np.uint8),
        )
        cuda_candidate = BackendCandidate(
            "star_detect_fused_pixel_components",
            "cuda_host_io",
            "star_detect_fused_pixel_components_cuda",
        )
        cpu_candidate = BackendCandidate(
            "star_detect_fused_pixel_components",
            "openmp_cpu",
            "star_detect_fused_pixel_components_cpu",
        )
        cuda_selection = BackendSelection(cuda_candidate, object())
        cpu_selection = BackendSelection(cpu_candidate, object())

        with mock.patch.object(
                detection_ops,
                "_select_star_detect_fused_pixel_components_backend",
                return_value=cuda_selection):
            with mock.patch.object(
                    detection_ops,
                    "_star_detect_fused_pixel_components_compiled_validated",
                    side_effect=RuntimeError("no CUDA-capable device is detected")):
                with mock.patch.object(
                        backend_registry,
                        "resolve_after_runtime_unavailable",
                        return_value=cpu_selection):
                    with mock.patch.object(
                            detection_ops,
                            "_star_detect_fused_pixel_components_cpu_validated",
                            return_value=expected) as cpu_backend:
                        got = detection_ops.star_detect_fused_pixel_components(
                            image, None, 1.0)

        cpu_backend.assert_called_once()
        self.assertIs(got, expected)

    def test_star_detect_cpu_backend_error_propagates(self) -> None:
        image = np.arange(64, dtype=np.float64).reshape(8, 8)
        candidate = BackendCandidate(
            "star_detect_fused_pixel_components",
            "openmp_cpu",
            "star_detect_fused_pixel_components_cpu",
        )
        selection = BackendSelection(candidate, object())

        with mock.patch.object(
                detection_ops,
                "_select_star_detect_fused_pixel_components_backend",
                return_value=selection):
            with mock.patch.object(
                    detection_ops,
                    "_star_detect_fused_pixel_components_cpu_validated",
                    side_effect=RuntimeError("CPU detector failed")):
                with self.assertRaisesRegex(RuntimeError, "CPU detector failed"):
                    detection_ops.star_detect_fused_pixel_components(
                        image, None, 1.0)

    def test_star_detect_fused_pixel_components_maps_capacity_error(self) -> None:
        image = np.arange(64, dtype=np.float64).reshape(8, 8)
        candidate = BackendCandidate(
            "star_detect_fused_pixel_components",
            "cuda_host_io",
            "star_detect_fused_pixel_components_cuda",
        )
        selection = BackendSelection(candidate, object())
        backend = mock.Mock(
            side_effect=detection_ops.StarDetectCapacityError(
                "GPU CC did not converge"))
        unavailable = BackendSelection(None, object(), "no CPU backend")

        with mock.patch.object(
                detection_ops,
                "_select_star_detect_fused_pixel_components_backend",
                return_value=selection):
            with mock.patch.object(
                    detection_ops,
                    "_star_detect_fused_pixel_components_compiled_validated",
                    backend):
                with mock.patch.object(
                        detection_ops, "_select_backend", return_value=unavailable):
                    with self.assertRaisesRegex(
                            detection_ops.StarDetectCapacityError,
                            "did not converge"):
                        detection_ops.star_detect_fused_pixel_components(
                            image, None, 1.0)

    def test_star_detect_fused_pixel_components_maps_resource_error(self) -> None:
        image = np.arange(64, dtype=np.float64).reshape(8, 8)
        candidate = BackendCandidate(
            "star_detect_fused_pixel_components",
            "cuda_host_io",
            "star_detect_fused_pixel_components_cuda",
        )
        selection = BackendSelection(candidate, object())
        backend = mock.Mock(
            side_effect=CustomOpResourceExhaustedError("cudaMalloc input"))
        unavailable = BackendSelection(None, object(), "no CPU backend")

        with mock.patch.object(
                detection_ops,
                "_select_star_detect_fused_pixel_components_backend",
                return_value=selection):
            with mock.patch.object(
                    detection_ops,
                    "_star_detect_fused_pixel_components_compiled_validated",
                    backend):
                with mock.patch.object(
                        backend_registry,
                        "resolve_after_resource_exhausted",
                        return_value=unavailable):
                    with self.assertRaisesRegex(CustomOpUnavailableError, "no CPU backend"):
                        detection_ops.star_detect_fused_pixel_components(
                            image, None, 1.0)

    def test_star_detect_fused_pixel_components_cpu_matches_opencv_on_synthetic(
            self) -> None:
        image = np.zeros((384, 384), dtype=np.float64)
        for x, y, value in [(80, 90, 7.0), (140, 260, 9.0),
                            (220, 130, 8.0), (300, 300, 10.0)]:
            cv2.circle(image, (x, y), 5, value, -1)
            cv2.circle(image, (x, y), 9, value * 0.35, 1)
        image += np.linspace(0.0, 0.05, image.shape[1], dtype=np.float64)[None, :]

        candidates = detection_ops.star_detect_fused_pixel_components_compiled_cpu(
            image, None, 1.0, gaussian_ksize=9, sigma=2.0)
        measured = star_detection._measure_native_hybrid_contour_candidates(*candidates)
        got = star_detection._filter_star_candidates(*measured)
        expected = star_detection._detect_star_points_opencv(
            image,
            min_star_points=0,
            resize_length=10000,
            gaussian_ksize=9,
            sigma=2,
        )
        self.assertGreaterEqual(len(got.positions), 3)
        self.assertEqual(len(got.positions), len(expected.positions))
        distances = np.linalg.norm(
            expected.positions[:, None, :] - got.positions[None, :, :], axis=2)
        self.assertTrue(np.all(np.min(distances, axis=1) < 1.0))
        self.assertTrue(np.all(np.min(distances, axis=0) < 1.0))
        nearest = np.argmin(distances, axis=1)
        np.testing.assert_allclose(
            got.volumes[nearest], expected.volumes, rtol=1e-12, atol=1e-12)

    def test_star_detect_fused_pixel_components_cpu_respects_external_mask(
            self) -> None:
        image = np.zeros((192, 256), dtype=np.float64)
        cv2.circle(image, (64, 80), 6, 8.0, -1)
        cv2.circle(image, (196, 110), 6, 10.0, -1)
        mask = np.zeros(image.shape, dtype=np.uint8)
        mask[:, :128] = 255

        component_positions, component_intensities, binary_mask = (
            detection_ops.star_detect_fused_pixel_components_compiled_cpu(
                image, mask, 1.0, gaussian_ksize=9, sigma=2.0))

        self.assertGreater(len(component_positions), 0)
        self.assertEqual(len(component_positions), len(component_intensities))
        self.assertTrue(np.all(component_positions[:, 0] < 128))
        self.assertTrue(np.all(np.isfinite(component_intensities)))
        self.assertTrue(np.all(component_intensities > 0))
        self.assertEqual(np.count_nonzero(binary_mask[:, 128:]), 0)

    def test_star_detect_fused_pixel_components_compiled_matches_opencv_on_synthetic(
            self) -> None:
        if not build_info().get("cuda"):
            self.skipTest("CUDA fused pixel-component backend is not built")

        image = np.zeros((384, 384), dtype=np.float64)
        stars = [
            (80, 90, 7.0),
            (140, 260, 9.0),
            (220, 130, 8.0),
            (300, 300, 10.0),
        ]
        for x, y, value in stars:
            cv2.circle(image, (x, y), 5, value, -1)
            cv2.circle(image, (x, y), 9, value * 0.35, 1)
        image += np.linspace(0.0, 0.05, image.shape[1], dtype=np.float64)[None, :]

        try:
            candidates = detection_ops.star_detect_fused_pixel_components_compiled(
                image,
                None,
                1.0,
                gaussian_ksize=9,
                sigma=2.0,
            )
        except RuntimeError as exc:
            if _is_compiled_backend_unavailable(exc):
                self.skipTest(
                    f"CUDA fused pixel-component runtime unavailable: {exc}")
            raise

        measured = star_detection._measure_native_hybrid_contour_candidates(*candidates)
        got = star_detection._filter_star_candidates(*measured)
        expected = star_detection._detect_star_points_opencv(
            image,
            min_star_points=0,
            resize_length=10000,
            gaussian_ksize=9,
            sigma=2,
        )
        self.assertGreaterEqual(len(got.positions), 3)
        self.assertGreaterEqual(len(expected.positions), 3)

        distances = np.linalg.norm(
            expected.positions[:, None, :] - got.positions[None, :, :],
            axis=2,
        )
        nearest = np.min(distances, axis=1)
        self.assertLess(float(np.percentile(nearest, 95)), 1.5)

    def test_detect_star_points_uses_native_hybrid_backend_by_default(self) -> None:
        image = np.zeros((16, 16), dtype=np.float64)
        expected = star_detection.DetectedStars(
            positions=np.array([[4.0, 5.0]]),
            volumes=np.array([8.0]),
        )

        with mock.patch.object(
                star_detection,
                "_detect_star_points_native_hybrid",
                return_value=expected) as native_hybrid:
            with mock.patch.object(
                    star_detection,
                    "_detect_star_points_contour") as contour:
                got = star_detection.detect_star_points(
                    image, min_star_points=0)

        native_hybrid.assert_called_once()
        contour.assert_not_called()
        self.assertIs(got, expected)

    def test_detect_star_points_falls_back_when_cuda_unavailable(self) -> None:
        image = np.zeros((16, 16), dtype=np.float64)
        expected = star_detection.DetectedStars(
            positions=np.empty((0, 2)),
            volumes=np.empty((0,)),
        )
        with mock.patch.object(
                star_detection,
                "_detect_star_points_native_hybrid",
                side_effect=CustomOpUnavailableError("no CUDA")):
            with mock.patch.object(
                    star_detection,
                    "_detect_star_points_contour",
                    return_value=expected) as contour:
                got = star_detection.detect_star_points(
                    image, min_star_points=0)

        contour.assert_called_once()
        self.assertIs(got, expected)

    def test_detect_star_points_respects_forced_numpy_fallback(self) -> None:
        image = np.arange(16 * 16, dtype=np.float64).reshape(16, 16)
        expected = star_detection.DetectedStars(
            positions=np.empty((0, 2)),
            volumes=np.empty((0,)),
        )
        with mock.patch.dict(
                "os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"},
                clear=False):
            detection_ops._select_star_detect_fused_pixel_components_backend.cache_clear()
            with mock.patch.object(
                    star_detection,
                    "_detect_star_points_contour",
                    return_value=expected) as contour:
                got = star_detection.detect_star_points(
                    image, min_star_points=0)

        contour.assert_called_once()
        self.assertIs(got, expected)

    def test_detect_star_points_falls_back_on_geometry_guard(self) -> None:
        image = np.zeros((16, 16), dtype=np.float64)
        expected = star_detection.DetectedStars(
            positions=np.empty((0, 2)),
            volumes=np.empty((0,)),
        )
        with mock.patch.object(
                star_detection,
                "_detect_star_points_native_hybrid",
                side_effect=star_detection._NativeHybridGeometryMismatch(
                    "ambiguous mapping")):
            with mock.patch.object(
                    star_detection,
                    "_detect_star_points_contour",
                    return_value=expected) as contour:
                got = star_detection.detect_star_points(
                    image, min_star_points=0)

        contour.assert_called_once()
        self.assertIs(got, expected)

    def test_detect_star_points_falls_back_on_cuda_capacity_guard(self) -> None:
        image = np.zeros((16, 16), dtype=np.float64)
        expected = star_detection.DetectedStars(
            positions=np.empty((0, 2)),
            volumes=np.empty((0,)),
        )
        with mock.patch.object(
                star_detection,
                "_detect_star_points_native_hybrid",
                side_effect=detection_ops.StarDetectCapacityError(
                    "GPU CC did not converge")):
            with mock.patch.object(
                    star_detection,
                    "_detect_star_points_contour",
                    return_value=expected) as contour:
                got = star_detection.detect_star_points(
                    image, min_star_points=0)

        contour.assert_called_once()
        self.assertIs(got, expected)

    def test_long_connected_structure_uses_cpu_without_contour_fallback(
            self) -> None:
        if not build_info().get("cuda"):
            self.skipTest("CUDA fused pixel-component backend is not built")
        module, error = detection_ops._load_compiled_module_result()
        if module is None:
            self.skipTest(error or "compiled custom ops unavailable")
        memory_info = module.cuda_memory_info()
        if not memory_info.get("available"):
            self.skipTest(memory_info.get("reason", "CUDA runtime unavailable"))

        height, width = 512, 3072
        image = np.broadcast_to(
            np.linspace(0.0, 0.01, width, dtype=np.float64)[None, :],
            (height, width),
        ).copy()
        image[250:257, 100:2972] = 10.0
        cpu_impl = detection_ops._star_detect_fused_pixel_components_cpu_validated
        with mock.patch.dict(
                "os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "auto"},
                clear=False):
            detection_ops._select_star_detect_fused_pixel_components_backend.cache_clear()
            with mock.patch.object(
                    detection_ops,
                    "_star_detect_fused_pixel_components_cpu_validated",
                    wraps=cpu_impl,
            ) as cpu_backend:
                with mock.patch.object(
                        star_detection,
                        "_detect_star_points_contour",
                ) as contour:
                    got = star_detection.detect_star_points(
                        image,
                        resize_length=10000,
                        min_star_points=0,
                    )

        cpu_backend.assert_called_once()
        contour.assert_not_called()
        self.assertEqual(got.positions.shape, (0, 2))
        self.assertEqual(got.volumes.shape, (0,))
        self.assertFalse(
            module.cuda_host_io_cache_info()["measurement_active"])

        stack = np.arange(4 * 64, dtype=np.uint16).reshape(4, 64)
        ref_mean = np.mean(stack, axis=0, dtype=np.float64)
        ref_std = np.std(stack, axis=0, dtype=np.float64)
        module.huber_weighted_chunk_cuda(
            stack, ref_mean, ref_std, 1.5, None)
        after_recovery = module.cuda_host_io_cache_info()
        self.assertFalse(after_recovery["measurement_active"])
        self.assertIn(
            "huber_weighted_chunk_cuda", after_recovery["last_operation"])

    def test_detect_star_points_falls_back_on_cuda_oom(self) -> None:
        image = np.zeros((16, 16), dtype=np.float64)
        expected = star_detection.DetectedStars(
            positions=np.empty((0, 2)),
            volumes=np.empty((0,)),
        )
        with mock.patch.object(
                star_detection,
                "_detect_star_points_native_hybrid",
                side_effect=CustomOpResourceExhaustedError(
                    "cudaMalloc image: out of memory")):
            with mock.patch.object(
                    star_detection,
                    "_detect_star_points_contour",
                    return_value=expected) as contour:
                got = star_detection.detect_star_points(
                    image, min_star_points=0)

        contour.assert_called_once()
        self.assertIs(got, expected)

    def test_detect_star_points_gpu_kernel_error_propagates(self) -> None:
        image = np.arange(16 * 16, dtype=np.float64).reshape(16, 16)

        with mock.patch.object(
                star_detection,
                "star_detect_fused_pixel_components",
                side_effect=RuntimeError("kernel bug")):
            with mock.patch.object(
                    star_detection, "_detect_star_points_contour") as contour:
                with self.assertRaisesRegex(RuntimeError, "kernel bug"):
                    star_detection.detect_star_points(
                        image, min_star_points=0)
        contour.assert_not_called()

    def test_detect_star_points_constant_image_returns_empty(self) -> None:
        image = np.zeros((16, 16), dtype=np.float64)

        with mock.patch.object(
                star_detection,
                "star_detect_fused_pixel_components") as backend:
            got = star_detection.detect_star_points(
                image, min_star_points=0)

        backend.assert_not_called()
        self.assertEqual(got.positions.shape, (0, 2))
        self.assertEqual(got.volumes.shape, (0,))

    def test_native_hybrid_candidate_filter_uses_norma_thresholds(self) -> None:
        positions = np.arange(24, dtype=np.float64).reshape(12, 2)
        areas = np.arange(1, 13, dtype=np.float64)
        intensities = np.arange(12, 0, -1, dtype=np.float64)
        eccentricities = np.zeros(12, dtype=np.float64)

        got = star_detection._filter_star_candidates(
            positions, areas, intensities, eccentricities)

        area_threshold = np.percentile(
            areas, star_detection.STAR_FILTER_PERCENTILE)
        intensity_threshold = np.percentile(
            intensities, star_detection.STAR_FILTER_PERCENTILE)
        valid = (
            (areas > star_detection.MIN_STAR_AREA)
            & (areas > area_threshold)
            & (intensities > intensity_threshold)
        )
        np.testing.assert_array_equal(got.positions, positions[valid])
        np.testing.assert_array_equal(
            got.volumes, areas[valid] * intensities[valid])

    def test_native_hybrid_contours_map_component_intensities_by_position(self) -> None:
        binary_mask = np.zeros((64, 64), dtype=np.uint8)
        cv2.circle(binary_mask, (15, 16), 4, 255, -1)
        cv2.circle(binary_mask, (46, 45), 5, 255, -1)
        component_positions = np.array(
            [[46.0, 45.0], [15.0, 16.0]], dtype=np.float64)
        component_intensities = np.array([7.0, 3.0], dtype=np.float64)

        positions, _, intensities, _ = (
            star_detection._measure_native_hybrid_contour_candidates(
                component_positions,
                component_intensities,
                binary_mask,
            ))

        nearest = np.argmin(
            np.linalg.norm(
                positions[:, None, :] - component_positions[None, :, :],
                axis=2,
            ),
            axis=1,
        )
        np.testing.assert_array_equal(
            intensities, component_intensities[nearest])

    def test_native_hybrid_contour_mapping_keeps_small_mismatch(self) -> None:
        binary_mask = np.zeros((64, 64), dtype=np.uint8)
        cv2.circle(binary_mask, (15, 16), 4, 255, -1)
        cv2.circle(binary_mask, (46, 45), 5, 255, -1)

        positions, _, intensities, _ = (
            star_detection._measure_native_hybrid_contour_candidates(
                np.array([[15.0, 16.0]], dtype=np.float64),
                np.array([3.0], dtype=np.float64),
                binary_mask,
            ))

        self.assertEqual(len(positions), 2)
        np.testing.assert_array_equal(intensities, [3.0, 3.0])

    def test_native_hybrid_contour_mapping_allows_small_duplicate_match(self) -> None:
        binary_mask = np.zeros((64, 64), dtype=np.uint8)
        cv2.circle(binary_mask, (32, 32), 10, 255, 3)

        positions, _, intensities, _ = (
            star_detection._measure_native_hybrid_contour_candidates(
                np.array([[32.0, 32.0], [0.0, 0.0]], dtype=np.float64),
                np.array([3.0, 1.0], dtype=np.float64),
                binary_mask,
            ))

        self.assertEqual(len(positions), 2)
        np.testing.assert_array_equal(intensities, [3.0, 3.0])

    def test_native_hybrid_contour_mapping_keeps_single_distant_component(self) -> None:
        binary_mask = np.zeros((64, 64), dtype=np.uint8)
        cv2.circle(binary_mask, (15, 16), 4, 255, -1)

        positions, areas, intensities, eccentricities = (
            star_detection._measure_native_hybrid_contour_candidates(
                np.array([[20.0, 20.0]], dtype=np.float64),
                np.array([3.0], dtype=np.float64),
                binary_mask,
            ))

        self.assertEqual(positions.shape, (1, 2))
        self.assertEqual(areas.shape, (1,))
        np.testing.assert_array_equal(intensities, [3.0])
        self.assertEqual(eccentricities.shape, (1,))

    def test_native_hybrid_contour_mapping_rejects_excessive_mismatch(self) -> None:
        binary_mask = np.zeros((64, 64), dtype=np.uint8)
        for position in (
            (8, 8), (8, 32), (8, 56), (32, 8),
            (32, 32), (32, 56), (56, 8), (56, 32),
        ):
            cv2.circle(binary_mask, position, 4, 255, -1)

        with self.assertRaisesRegex(
                star_detection._NativeHybridGeometryMismatch,
                "exceeds tolerance"):
            star_detection._measure_native_hybrid_contour_candidates(
                np.array([[8.0, 8.0]], dtype=np.float64),
                np.array([3.0], dtype=np.float64),
                binary_mask,
            )

    def test_relaxed_mapping_accepts_excessive_geometry_mismatch(self) -> None:
        binary_mask = np.zeros((160, 160), dtype=np.uint8)
        for x in range(20, 150, 20):
            cv2.circle(binary_mask, (x, 50), 4, 255, -1)
        positions, _, intensities, _ = (
            star_detection._measure_native_hybrid_contour_candidates(
                np.array([[20., 50.]]), np.array([3.]), binary_mask,
                mode="native_relaxed"))
        self.assertEqual(len(positions), 7)
        np.testing.assert_array_equal(intensities, np.full(7, 3.))

    def test_ineligible_geometry_mismatch_does_not_reject_frame(self) -> None:
        binary_mask = np.zeros((180, 180), dtype=np.uint8)
        for y in range(20, 170, 20):
            cv2.ellipse(binary_mask, (80, y), (12, 3), 0, 0, 360, 255, -1)
        components = np.array([[82., float(y)] for y in range(20, 170, 20)])
        positions, _, intensities, eccentricities = (
            star_detection._measure_native_hybrid_contour_candidates(
                components, np.full(8, 3.), binary_mask))
        self.assertEqual(len(positions), 8)
        self.assertTrue(np.all(eccentricities >= .8))
        np.testing.assert_array_equal(intensities, np.full(8, 3.))

    def test_contour_mode_skips_native(self) -> None:
        with mock.patch.object(star_detection, "_detect_star_points_contour") as contour:
            with mock.patch.object(star_detection, "_detect_star_points_native_hybrid") as native:
                got = star_detection.detect_star_points(np.zeros((16, 16)), mode="contour")
        native.assert_not_called()
        self.assertIs(got, contour.return_value)

    def test_relaxed_mode_keeps_empty_component_guard(self) -> None:
        binary_mask = np.zeros((32, 32), dtype=np.uint8)
        cv2.circle(binary_mask, (16, 16), 4, 255, -1)
        with self.assertRaises(star_detection._NativeHybridGeometryMismatch):
            star_detection._measure_native_hybrid_contour_candidates(
                np.empty((0, 2)), np.empty(0), binary_mask,
                mode="native_relaxed")

    def test_invalid_detection_mode_is_reported(self) -> None:
        with self.assertRaisesRegex(ValueError, "star_detection_mode"):
            star_detection.detect_star_points(np.zeros((16, 16)), mode="invalid")

    def test_detection_cache_passes_explicit_mode(self) -> None:
        from hoshicore.component.norma.geometry_view import StarDetectionCache
        with mock.patch("hoshicore.component.norma.geometry_view.detect_star_points") as detect:
            cache = StarDetectionCache.from_image(
                np.zeros((16, 16)), star_detection_mode="native_relaxed")
            self.assertIs(cache.pywt_stars, detect.return_value)
        self.assertEqual(detect.call_args.kwargs["mode"], "native_relaxed")

    def test_star_detect_cuda_external_empty_mask_is_not_relaxed(self) -> None:
        if not build_info().get("cuda"):
            self.skipTest("CUDA fused pixel-component backend is not built")
        image = np.arange(128 * 128, dtype=np.float64).reshape(128, 128)
        mask = np.zeros(image.shape, dtype=np.uint8)

        try:
            detection_ops.star_detect_fused_pixel_components_compiled(
                image, mask, 1.0)
        except RuntimeError as exc:
            if _is_compiled_backend_unavailable(exc):
                self.skipTest(
                    f"CUDA fused pixel-component runtime unavailable: {exc}")
            self.assertRegex(str(exc), "mask selects no pixels")
        else:
            self.fail("an empty external mask must reject all pixels")

    def test_star_detect_public_facade_exports_fused_pixel_components(self) -> None:
        from hoshicore._custom_op import star_detect_fused_pixel_components
        from hoshicore._custom_op.api import (
            star_detect_fused_pixel_components as api_star_detect_fused_pixel_components,
        )

        self.assertIs(
            star_detect_fused_pixel_components,
            detection_ops.star_detect_fused_pixel_components,
        )
        self.assertIs(
            api_star_detect_fused_pixel_components,
            detection_ops.star_detect_fused_pixel_components,
        )
