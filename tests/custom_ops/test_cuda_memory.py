import unittest
from concurrent.futures import ThreadPoolExecutor
import threading
from unittest import mock

import cv2
import numpy as np

from hoshicore._custom_op import build_info
from hoshicore._custom_op import cuda_memory
from hoshicore._custom_op._dispatch import CustomOpResourceExhaustedError
from hoshicore._custom_op._dispatch import CustomOpCudaRuntimeUnavailableError
from hoshicore._custom_op._dispatch import CudaProbeError
from hoshicore._custom_op._dispatch import is_cuda_runtime_unavailable_error
from hoshicore._custom_op.ops import alignment as alignment_ops
from hoshicore._custom_op.ops import detection as detection_ops
from hoshicore._custom_op.ops import remap as remap_ops
from hoshicore._custom_op.ops import star_shrink as star_shrink_ops
from hoshicore._custom_op.ops import wavelet as wavelet_ops


MiB = 1024 * 1024
GiB = 1024 * MiB


def _available_memory_info(
    *,
    free_bytes: int = 4 * 1024 * MiB,
    total_bytes: int = 8 * 1024 * MiB,
    device: int = 0,
) -> dict[str, object]:
    return {
        "available": True,
        "status": "available",
        "reason_code": "cuda_available",
        "category": "available",
        "device": device,
        "free_bytes": free_bytes,
        "total_bytes": total_bytes,
    }


def _estimate(peak_device_bytes: int = 128 * MiB) -> cuda_memory.CudaMemoryEstimate:
    return cuda_memory.CudaMemoryEstimate(
        logical_op="test_cuda_op",
        peak_device_bytes=peak_device_bytes,
        confidence="exact",
        reason="test fixture",
    )


class TestCudaMemoryEstimate(unittest.TestCase):
    def tearDown(self) -> None:
        cuda_memory._reset_cuda_reservations_for_tests()

    def test_detection_estimate_scales_with_image_size(self) -> None:
        small = cuda_memory.estimate_star_detect_fused_pixel_components(
            height=512,
            width=768,
            small_height=256,
            small_width=384,
            level=1,
            gaussian_ksize=9,
        )
        large = cuda_memory.estimate_star_detect_fused_pixel_components(
            height=1024,
            width=1536,
            small_height=512,
            small_width=768,
            level=1,
            gaussian_ksize=9,
        )

        self.assertEqual(
            small.logical_op, "star_detect_fused_pixel_components")
        self.assertGreater(small.peak_device_bytes, 0)
        self.assertGreater(large.peak_device_bytes, small.peak_device_bytes)
        self.assertGreater(small.peak_pinned_bytes, 0)
        self.assertEqual(small.confidence, "bounded")

    def test_camera_model_remap_estimate_is_exact(self) -> None:
        for source_shape, output_shape in (
            ((12, 20), (8, 16)),
            ((8, 16), (12, 20)),
            ((12, 20), (12, 20)),
        ):
            with self.subTest(source=source_shape, output=output_shape):
                source_bytes = source_shape[0] * source_shape[1] * 3 * 2
                output_bytes = output_shape[0] * output_shape[1] * 3 * 2
                estimate = cuda_memory.estimate_camera_model_remap(
                    source_height=source_shape[0],
                    source_width=source_shape[1],
                    channels=3,
                    dtype_bytes=2,
                    out_height=output_shape[0],
                    out_width=output_shape[1],
                )

                self.assertEqual(estimate.logical_op, "camera_model_remap")
                self.assertEqual(
                    estimate.peak_device_bytes, source_bytes + output_bytes
                )
                self.assertEqual(
                    estimate.peak_pinned_bytes, max(source_bytes, output_bytes)
                )
                self.assertEqual(estimate.confidence, "exact")

    def test_camera_model_remap_estimate_rejects_invalid_dimensions(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive dimensions"):
            cuda_memory.estimate_camera_model_remap(
                source_height=0,
                source_width=20,
                channels=1,
                dtype_bytes=2,
                out_height=8,
                out_width=16,
            )

    def test_matching_bidirectional_nearest_estimate_is_exact(self) -> None:
        n1 = 17
        n2 = 19
        feature_dim = 11
        estimate = (
            cuda_memory.estimate_matching_cosine_bidirectional_nearest(
                n1=n1,
                n2=n2,
                feature_dim=feature_dim,
            )
        )
        features_bytes = (n1 + n2) * feature_dim * 8
        outputs_bytes = (n1 + n2) * 16
        tile_size = cuda_memory.MATCHING_COSINE_TILE_SIZE
        row_tiles = (n1 + tile_size - 1) // tile_size
        col_tiles = (n2 + tile_size - 1) // tile_size
        partial_candidates_bytes = (
            n1 * col_tiles + n2 * row_tiles
        ) * (8 + 8 + 4)
        expected_device = (
            features_bytes
            + (n1 + n2) * 8
            + partial_candidates_bytes
            + outputs_bytes
            + 4
        )

        self.assertEqual(
            estimate.logical_op, "matching_cosine_bidirectional_nearest")
        self.assertEqual(estimate.peak_device_bytes, expected_device)
        self.assertEqual(
            estimate.peak_pinned_bytes,
            features_bytes + outputs_bytes + 4,
        )
        self.assertEqual(estimate.confidence, "exact")

    def test_matching_bidirectional_nearest_tile_size_matches_native(self) -> None:
        module, error = alignment_ops._load_compiled_module_result()
        if module is None or not module.build_info().get("cuda"):
            self.skipTest(error or "CUDA matching backend is not built")
        self.assertEqual(
            module.MATCHING_COSINE_TILE_SIZE,
            cuda_memory.MATCHING_COSINE_TILE_SIZE,
        )

    def test_matching_bidirectional_nearest_estimate_rejects_invalid_shape(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "positive dimensions"):
            cuda_memory.estimate_matching_cosine_bidirectional_nearest(
                n1=0,
                n2=19,
                feature_dim=11,
            )

    def test_star_shrink_estimates_match_workspace_formulas(self) -> None:
        height = 5
        width = 7
        channels = 3
        dtype_bytes = 2
        small_kernel_size = 7
        large_kernel_size = 19
        pixels = height * width
        total = pixels * channels
        image_bytes = total * dtype_bytes
        mask_bytes = pixels
        plane_float_bytes = pixels * 4
        total_float_bytes = total * 4
        blocks = 1

        process_expected = (
            2 * image_bytes
            + mask_bytes
            + 4 * plane_float_bytes
            + 3 * total_float_bytes
        )
        dog_expected = (
            image_bytes
            + 5 * plane_float_bytes
            + (small_kernel_size + large_kernel_size) * 4
            + 2 * mask_bytes
            + 2 * blocks * 8
        )
        common = {
            "height": height,
            "width": width,
            "channels": channels,
            "dtype_bytes": dtype_bytes,
        }
        process = cuda_memory.estimate_star_shrink_process(**common)
        dog = cuda_memory.estimate_star_mask_dog(
            **common,
            small_kernel_size=small_kernel_size,
            large_kernel_size=large_kernel_size,
        )
        fused = cuda_memory.estimate_star_shrink_dog_process(
            **common,
            small_kernel_size=small_kernel_size,
            large_kernel_size=large_kernel_size,
        )

        self.assertEqual(process.peak_device_bytes, process_expected)
        self.assertEqual(dog.peak_device_bytes, dog_expected)
        self.assertEqual(
            fused.peak_device_bytes,
            process_expected + dog_expected - image_bytes - mask_bytes,
        )
        self.assertEqual(process.peak_pinned_bytes, 0)
        self.assertEqual(dog.peak_pinned_bytes, 0)
        self.assertEqual(fused.peak_pinned_bytes, 0)
        self.assertEqual(process.confidence, "exact")
        self.assertEqual(dog.confidence, "exact")
        self.assertEqual(fused.confidence, "exact")

    def test_star_shrink_estimates_reject_invalid_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive dimensions"):
            cuda_memory.estimate_star_shrink_process(
                height=0,
                width=8,
                channels=3,
                dtype_bytes=2,
            )
        with self.assertRaisesRegex(ValueError, "must be odd"):
            cuda_memory.estimate_star_mask_dog(
                height=8,
                width=8,
                channels=1,
                dtype_bytes=1,
                small_kernel_size=6,
                large_kernel_size=19,
            )

    def test_wavelet_estimate_is_level_aware_and_exact(self) -> None:
        shallow = cuda_memory.estimate_wavelet_dec_rec(
            height=8,
            width=8,
            level=2,
        )
        deep = cuda_memory.estimate_wavelet_dec_rec(
            height=8,
            width=8,
            level=4,
        )

        self.assertEqual(shallow.logical_op, "wavelet_dec_rec_cuda_core")
        self.assertGreater(shallow.peak_device_bytes, 0)
        self.assertGreater(deep.peak_device_bytes, 0)
        self.assertNotEqual(deep.peak_device_bytes, shallow.peak_device_bytes)
        self.assertEqual(shallow.peak_pinned_bytes, 0)
        self.assertEqual(shallow.confidence, "exact")

    def test_wavelet_estimate_rejects_invalid_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive inputs"):
            cuda_memory.estimate_wavelet_dec_rec(
                height=64,
                width=63,
                level=0,
            )

    def test_detection_estimate_rejects_invalid_dimensions(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive dimensions"):
            cuda_memory.estimate_star_detect_fused_pixel_components(
                height=0,
                width=64,
                small_height=32,
                small_width=32,
                level=1,
                gaussian_ksize=9,
            )

    def test_chunk_model_separates_host_and_device_costs(self) -> None:
        n_frames = 5
        row_bytes = 120
        dtype_bytes = 2
        items_per_row = row_bytes // dtype_bytes
        float64_row = items_per_row * 8

        sigma = cuda_memory.cuda_chunk_memory_model(
            "sigma_clip_fused_chunk",
            n_frames=n_frames,
            row_bytes=row_bytes,
            dtype_bytes=dtype_bytes,
        )
        self.assertEqual(
            sigma.host_bytes_per_row,
            3 * n_frames * row_bytes
            + n_frames * items_per_row
            + 3 * float64_row,
        )
        self.assertEqual(
            sigma.device_bytes_per_row,
            n_frames * row_bytes
            + n_frames * items_per_row
            + 3 * float64_row,
        )

        huber = cuda_memory.cuda_chunk_memory_model(
            "huber_weighted_chunk",
            n_frames=n_frames,
            row_bytes=row_bytes,
            dtype_bytes=dtype_bytes,
        )
        self.assertEqual(
            huber.host_bytes_per_row,
            2 * n_frames * row_bytes + 4 * float64_row,
        )
        self.assertEqual(
            huber.device_bytes_per_row,
            n_frames * row_bytes + 4 * float64_row,
        )
        self.assertEqual(huber.fixed_device_bytes, n_frames * 8)

    def test_chunk_model_estimate_matches_direct_array_shape(self) -> None:
        model = cuda_memory.cuda_chunk_memory_model(
            "sigma_clip_fused_chunk",
            n_frames=4,
            row_bytes=64 * 2,
            dtype_bytes=2,
            include_mask=False,
        )
        estimate = model.estimate(32)

        self.assertEqual(
            estimate.peak_device_bytes,
            4 * 32 * 64 * 2 + 3 * 32 * 64 * 8,
        )

    def test_usable_memory_subtracts_headroom_from_free_bytes(self) -> None:
        total_bytes = 24 * GiB
        free_bytes = 5 * GiB
        headroom = cuda_memory._headroom_bytes(total_bytes)

        self.assertEqual(
            cuda_memory.cuda_usable_memory_bytes(free_bytes, total_bytes),
            free_bytes - headroom,
        )

    def test_usable_memory_clamps_free_bytes_to_total(self) -> None:
        """Probe quirks may report free above total; the cap must still hold."""
        total_bytes = 8 * GiB
        headroom = cuda_memory._headroom_bytes(total_bytes)

        self.assertEqual(
            cuda_memory.cuda_usable_memory_bytes(total_bytes + 3 * GiB, total_bytes),
            total_bytes - headroom,
        )

    def test_usable_memory_deducts_reservations(self) -> None:
        total_bytes = 24 * GiB
        free_bytes = 5 * GiB
        reserved_bytes = 2 * GiB
        headroom = cuda_memory._headroom_bytes(total_bytes)

        self.assertEqual(
            cuda_memory.cuda_usable_memory_bytes(
                free_bytes, total_bytes, reserved_bytes
            ),
            free_bytes - headroom - reserved_bytes,
        )

    def test_usable_memory_is_never_negative(self) -> None:
        total_bytes = 24 * GiB
        headroom = cuda_memory._headroom_bytes(total_bytes)

        self.assertEqual(
            cuda_memory.cuda_usable_memory_bytes(headroom, total_bytes), 0
        )
        self.assertEqual(
            cuda_memory.cuda_usable_memory_bytes(headroom // 2, total_bytes), 0
        )
        self.assertEqual(
            cuda_memory.cuda_usable_memory_bytes(
                headroom + GiB, total_bytes, 4 * GiB
            ),
            0,
        )

    def test_admission_reserves_and_releases_current_device(self) -> None:
        with mock.patch.object(
                cuda_memory,
                "cuda_memory_info",
                return_value=_available_memory_info()):
            with cuda_memory.cuda_memory_admission(_estimate()) as first:
                self.assertTrue(first.granted)
                self.assertEqual(first.reserved_bytes, 0)
                with cuda_memory.cuda_memory_admission(_estimate()) as second:
                    self.assertTrue(second.granted)
                    self.assertEqual(second.reserved_bytes, 128 * MiB)

            with cuda_memory.cuda_memory_admission(_estimate()) as after:
                self.assertTrue(after.granted)
                self.assertEqual(after.reserved_bytes, 0)

    def test_admission_reservations_are_separated_by_device(self) -> None:
        probe = mock.Mock(side_effect=[
            _available_memory_info(device=0),
            _available_memory_info(device=1),
        ])
        with mock.patch.object(cuda_memory, "cuda_memory_info", probe):
            with cuda_memory.cuda_memory_admission(_estimate()) as first:
                self.assertTrue(first.granted)
                with cuda_memory.cuda_memory_admission(_estimate()) as second:
                    self.assertTrue(second.granted)
                    self.assertEqual(second.device, 1)
                    self.assertEqual(second.reserved_bytes, 0)

    def test_admission_reservation_is_visible_to_another_worker(self) -> None:
        first_admitted = threading.Event()
        release_first = threading.Event()

        def hold_first_reservation() -> int:
            with cuda_memory.cuda_memory_admission(_estimate()) as decision:
                first_admitted.set()
                self.assertTrue(release_first.wait(timeout=10))
                return decision.reserved_bytes

        def inspect_second_reservation() -> int:
            self.assertTrue(first_admitted.wait(timeout=10))
            with cuda_memory.cuda_memory_admission(_estimate()) as decision:
                release_first.set()
                return decision.reserved_bytes

        with mock.patch.object(
                cuda_memory,
                "cuda_memory_info",
                return_value=_available_memory_info()):
            with ThreadPoolExecutor(max_workers=2) as executor:
                first = executor.submit(hold_first_reservation)
                second = executor.submit(inspect_second_reservation)
                self.assertEqual(first.result(timeout=10), 0)
                self.assertEqual(second.result(timeout=10), 128 * MiB)

    def test_admission_deducts_reservation_from_constrained_free_memory(self) -> None:
        first_admitted = threading.Event()
        release_first = threading.Event()
        estimate = _estimate(3 * GiB)
        info = _available_memory_info(
            free_bytes=5 * GiB,
            total_bytes=24 * GiB,
        )

        def hold_first_reservation() -> cuda_memory.CudaAdmissionDecision:
            with cuda_memory.cuda_memory_admission(estimate) as decision:
                first_admitted.set()
                self.assertTrue(release_first.wait(timeout=10))
                return decision

        def inspect_second_reservation() -> cuda_memory.CudaAdmissionDecision:
            self.assertTrue(first_admitted.wait(timeout=10))
            with cuda_memory.cuda_memory_admission(estimate) as decision:
                release_first.set()
                return decision

        with mock.patch.object(
                cuda_memory, "cuda_memory_info", return_value=info):
            with mock.patch.object(
                    cuda_memory,
                    "_clear_current_thread_cuda_cache",
                    return_value=False):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    first = executor.submit(hold_first_reservation)
                    second = executor.submit(inspect_second_reservation)
                    second_decision = second.result(timeout=10)
                    first_decision = first.result(timeout=10)

        self.assertTrue(first_decision.granted)
        self.assertFalse(second_decision.granted)
        self.assertEqual(second_decision.reserved_bytes, 3 * GiB)
        self.assertEqual(
            second_decision.reason_code, "insufficient_vram_estimate")

    def test_admission_rejects_when_estimate_exceeds_usable_vram(self) -> None:
        info = _available_memory_info(
            free_bytes=300 * MiB,
            total_bytes=2 * 1024 * MiB,
        )
        with mock.patch.object(
                cuda_memory, "cuda_memory_info", return_value=info):
            with mock.patch.object(
                    cuda_memory,
                    "_clear_current_thread_cuda_cache",
                    return_value=False):
                with cuda_memory.cuda_memory_admission(_estimate()) as decision:
                    self.assertFalse(decision.granted)
                    self.assertTrue(decision.checked)
                    self.assertEqual(
                        decision.reason_code, "insufficient_vram_estimate")

    def test_admission_retries_after_cache_eviction(self) -> None:
        probe = mock.Mock(side_effect=[
            _available_memory_info(
                free_bytes=300 * MiB,
                total_bytes=2 * 1024 * MiB,
            ),
            _available_memory_info(
                free_bytes=1024 * MiB,
                total_bytes=2 * 1024 * MiB,
            ),
        ])
        with mock.patch.object(cuda_memory, "cuda_memory_info", probe):
            with mock.patch.object(
                    cuda_memory,
                    "_clear_current_thread_cuda_cache",
                    return_value=True) as clear_cache:
                with cuda_memory.cuda_memory_admission(_estimate()) as decision:
                    self.assertTrue(decision.granted)
                    self.assertTrue(decision.cache_evicted)

        clear_cache.assert_called_once_with()
        self.assertEqual(probe.call_count, 2)

    def test_cache_eviction_retry_preserves_existing_reservation(self) -> None:
        probe = mock.Mock(
            side_effect=[
                _available_memory_info(
                    free_bytes=1024 * MiB,
                    total_bytes=2 * 1024 * MiB,
                ),
                _available_memory_info(
                    free_bytes=700 * MiB,
                    total_bytes=2 * 1024 * MiB,
                ),
                _available_memory_info(
                    free_bytes=1024 * MiB,
                    total_bytes=2 * 1024 * MiB,
                ),
            ]
        )
        with mock.patch.object(cuda_memory, "cuda_memory_info", probe):
            with mock.patch.object(
                cuda_memory,
                "_clear_current_thread_cuda_cache",
                return_value=True,
            ) as clear_cache:
                with cuda_memory.cuda_memory_admission(_estimate()) as outer:
                    self.assertTrue(outer.granted)
                    with cuda_memory.cuda_memory_admission(
                        _estimate(512 * MiB)
                    ) as inner:
                        self.assertTrue(inner.granted)
                        self.assertTrue(inner.cache_evicted)
                        self.assertEqual(inner.reserved_bytes, 128 * MiB)

        clear_cache.assert_called_once_with()
        self.assertEqual(probe.call_count, 3)

    def test_explicitly_unavailable_probe_uses_typed_fallback_semantics(self) -> None:
        info = {
            "available": False,
            "status": "explicitly_unavailable",
            "reason_code": "cuda_compute_capability_unsupported",
            "category": "compatibility",
            "reason": "CUDA compute capability 5.2 is unsupported",
        }
        with mock.patch.object(
                cuda_memory, "cuda_memory_info", return_value=info):
            with self.assertRaisesRegex(
                    CustomOpCudaRuntimeUnavailableError,
                    "compute capability 5.2") as caught:
                with cuda_memory.cuda_memory_admission(_estimate()):
                    pass

        self.assertEqual(
            caught.exception.reason_code,
            "cuda_compute_capability_unsupported",
        )
        self.assertTrue(is_cuda_runtime_unavailable_error(caught.exception))

    def test_unavailable_build_probe_still_defers_to_backend_semantics(self) -> None:
        info = {
            "available": False,
            "status": "unavailable",
            "reason_code": "probe_unavailable",
            "category": "build",
            "reason": "compiled backend does not expose CUDA memory info",
        }
        with mock.patch.object(
                cuda_memory, "cuda_memory_info", return_value=info):
            with cuda_memory.cuda_memory_admission(_estimate()) as decision:
                self.assertTrue(decision.granted)
                self.assertFalse(decision.checked)
                self.assertEqual(
                    decision.reason_code, "cuda_memory_probe_unavailable")

    def test_resource_probe_error_uses_typed_resource_semantics(self) -> None:
        probe_error = CudaProbeError(
            "out of memory",
            reason_code="cuda_runtime_error",
            error_code=2,
            category="resource",
        )
        with mock.patch.object(
                cuda_memory,
                "cuda_memory_info",
                side_effect=probe_error):
            with self.assertRaisesRegex(
                    CustomOpResourceExhaustedError,
                    "memory probe exhausted resources"):
                with cuda_memory.cuda_memory_admission(_estimate()):
                    pass

    def test_reservation_is_released_when_backend_raises(self) -> None:
        with mock.patch.object(
                cuda_memory,
                "cuda_memory_info",
                return_value=_available_memory_info()):
            with self.assertRaisesRegex(RuntimeError, "kernel failure"):
                with cuda_memory.cuda_memory_admission(_estimate()):
                    raise RuntimeError("kernel failure")
            with cuda_memory.cuda_memory_admission(_estimate()) as after:
                self.assertEqual(after.reserved_bytes, 0)

    def test_run_admitted_cuda_executes_kernel_when_granted(self) -> None:
        kernel = mock.Mock(return_value="kernel-result")
        with mock.patch.object(
                cuda_memory,
                "cuda_memory_info",
                return_value=_available_memory_info()):
            result = cuda_memory.run_admitted_cuda(_estimate(), kernel, 1, 2)

        self.assertEqual(result, "kernel-result")
        kernel.assert_called_once_with(1, 2)

    def test_run_admitted_cuda_raises_typed_error_without_calling_kernel(
        self,
    ) -> None:
        kernel = mock.Mock()
        info = _available_memory_info(
            free_bytes=300 * MiB,
            total_bytes=2 * 1024 * MiB,
        )
        with mock.patch.object(
                cuda_memory, "cuda_memory_info", return_value=info):
            with mock.patch.object(
                    cuda_memory,
                    "_clear_current_thread_cuda_cache",
                    return_value=False):
                with self.assertRaisesRegex(
                        CustomOpResourceExhaustedError,
                        "test_cuda_op skipped CUDA"):
                    cuda_memory.run_admitted_cuda(_estimate(), kernel)

        kernel.assert_not_called()

    def test_detection_compiled_path_stops_before_kernel_when_denied(self) -> None:
        image = np.arange(64, dtype=np.float64).reshape(8, 8)
        native = mock.Mock()
        module = mock.Mock(
            star_detect_fused_pixel_components_cuda=native,
        )
        info = _available_memory_info(
            free_bytes=256 * MiB,
            total_bytes=2 * 1024 * MiB,
        )
        with mock.patch.object(
                detection_ops,
                "_load_compiled_module_result",
                return_value=(module, None)):
            with mock.patch.object(
                    cuda_memory, "cuda_memory_info", return_value=info):
                with mock.patch.object(
                        cuda_memory,
                        "_clear_current_thread_cuda_cache",
                        return_value=False):
                    with self.assertRaisesRegex(
                            CustomOpResourceExhaustedError,
                            "estimated peak"):
                        detection_ops.star_detect_fused_pixel_components_compiled(
                            image, None, 1.0)

        native.assert_not_called()

    def test_matching_estimate_matches_workspace_high_water(self) -> None:
        if not build_info().get("cuda"):
            self.skipTest("CUDA matching backend is not built")
        module, error = alignment_ops._load_compiled_module_result()
        if module is None:
            self.skipTest(error or "compiled custom ops unavailable")
        memory_info = module.cuda_memory_info()
        if not memory_info.get("available"):
            self.skipTest(memory_info.get("reason", "CUDA runtime unavailable"))

        rng = np.random.default_rng(23)
        features1 = rng.normal(size=(137, 31))
        features2 = rng.normal(size=(149, 31))
        estimate = cuda_memory.estimate_matching_cosine_bidirectional_nearest(
            n1=features1.shape[0],
            n2=features2.shape[0],
            feature_dim=features1.shape[1],
        )
        self.assertTrue(module.clear_cuda_host_io_cache())
        result = module.matching_cosine_bidirectional_nearest_cuda(
            features1, features2
        )
        self.assertIsNotNone(result)
        cache_info = module.cuda_host_io_cache_info()

        self.assertEqual(
            cache_info["last_device_peak_bytes"], estimate.peak_device_bytes
        )
        self.assertEqual(
            cache_info["last_pinned_peak_bytes"], estimate.peak_pinned_bytes
        )

    def test_camera_model_remap_estimate_matches_workspace_high_water(self) -> None:
        if not build_info().get("cuda"):
            self.skipTest("CUDA remap backend is not built")
        module, error = remap_ops._load_compiled_module_result()
        if module is None:
            self.skipTest(error or "compiled custom ops unavailable")
        memory_info = module.cuda_memory_info()
        if not memory_info.get("available"):
            self.skipTest(memory_info.get("reason", "CUDA runtime unavailable"))

        for source_shape, output_shape in (
            ((32, 40), (24, 30)),
            ((24, 30), (32, 40)),
        ):
            with self.subTest(source=source_shape, output=output_shape):
                image = np.arange(
                    source_shape[0] * source_shape[1] * 3, dtype=np.uint16
                ).reshape(*source_shape, 3)
                out_height, out_width = output_shape
                estimate = cuda_memory.estimate_camera_model_remap(
                    source_height=image.shape[0],
                    source_width=image.shape[1],
                    channels=image.shape[2],
                    dtype_bytes=image.dtype.itemsize,
                    out_height=out_height,
                    out_width=out_width,
                )
                self.assertTrue(module.clear_cuda_host_io_cache())
                module.camera_model_remap(
                    image,
                    out_height,
                    out_width,
                    28.0,
                    28.0,
                    (source_shape[1] - 1) * 0.5,
                    (source_shape[0] - 1) * 0.5,
                    24.0,
                    24.0,
                    (out_width - 1) * 0.5,
                    (out_height - 1) * 0.5,
                    np.eye(3, dtype=np.float64),
                )
                cache_info = module.cuda_host_io_cache_info()

                self.assertEqual(
                    cache_info["last_operation"],
                    "camera_model_remap cudaGetDevice",
                )
                self.assertEqual(
                    cache_info["last_device_peak_bytes"],
                    estimate.peak_device_bytes,
                )
                self.assertEqual(
                    cache_info["last_pinned_peak_bytes"],
                    estimate.peak_pinned_bytes,
                )

    def test_star_shrink_process_estimate_matches_workspace_high_water(self) -> None:
        if not build_info().get("cuda"):
            self.skipTest("CUDA star-shrink backend is not built")
        module, error = star_shrink_ops._load_compiled_module_result()
        if module is None:
            self.skipTest(error or "compiled custom ops unavailable")
        memory_info = module.cuda_memory_info()
        if not memory_info.get("available"):
            self.skipTest(memory_info.get("reason", "CUDA runtime unavailable"))

        image = np.arange(24 * 27 * 3, dtype=np.uint16).reshape(24, 27, 3)
        mask = np.ones(image.shape[:2], dtype=np.uint8)
        estimate = cuda_memory.estimate_star_shrink_process(
            height=image.shape[0],
            width=image.shape[1],
            channels=image.shape[2],
            dtype_bytes=image.dtype.itemsize,
        )
        self.assertTrue(module.clear_cuda_host_io_cache())
        star_shrink_ops.star_shrink_process_compiled_cuda(
            image, mask, 3, "CIRCLE", 1, 1.0, 5
        )
        cache_info = module.cuda_host_io_cache_info()

        self.assertEqual(
            cache_info["last_operation"],
            "star_shrink_process_cuda cudaGetDevice",
        )
        self.assertEqual(
            cache_info["last_device_peak_bytes"],
            estimate.peak_device_bytes,
        )
        self.assertEqual(cache_info["last_pinned_peak_bytes"], 0)

    def test_star_mask_dog_estimate_matches_workspace_high_water(self) -> None:
        if not build_info().get("cuda"):
            self.skipTest("CUDA star-shrink backend is not built")
        module, error = star_shrink_ops._load_compiled_module_result()
        if module is None:
            self.skipTest(error or "compiled custom ops unavailable")
        memory_info = module.cuda_memory_info()
        if not memory_info.get("available"):
            self.skipTest(memory_info.get("reason", "CUDA runtime unavailable"))

        image = np.arange(24 * 27 * 3, dtype=np.uint16).reshape(24, 27, 3)
        estimate = cuda_memory.estimate_star_mask_dog(
            height=image.shape[0],
            width=image.shape[1],
            channels=image.shape[2],
            dtype_bytes=image.dtype.itemsize,
            small_kernel_size=7,
            large_kernel_size=19,
        )
        self.assertTrue(module.clear_cuda_host_io_cache())
        star_shrink_ops.star_mask_dog_compiled_cuda(
            image,
            sigma_small=1.0,
            sigma_large=3.0,
            threshold_ratio=1.0,
            open_ksize=0,
            dilate_ksize=0,
        )
        cache_info = module.cuda_host_io_cache_info()

        self.assertEqual(
            cache_info["last_operation"],
            "star_mask_dog_cuda cudaGetDevice",
        )
        self.assertEqual(
            cache_info["last_device_peak_bytes"],
            estimate.peak_device_bytes,
        )
        self.assertEqual(cache_info["last_pinned_peak_bytes"], 0)

    def test_star_shrink_dog_estimate_matches_workspace_high_water(self) -> None:
        if not build_info().get("cuda"):
            self.skipTest("CUDA star-shrink backend is not built")
        module, error = star_shrink_ops._load_compiled_module_result()
        if module is None:
            self.skipTest(error or "compiled custom ops unavailable")
        memory_info = module.cuda_memory_info()
        if not memory_info.get("available"):
            self.skipTest(memory_info.get("reason", "CUDA runtime unavailable"))

        image = np.arange(24 * 27 * 3, dtype=np.uint16).reshape(24, 27, 3)
        estimate = cuda_memory.estimate_star_shrink_dog_process(
            height=image.shape[0],
            width=image.shape[1],
            channels=image.shape[2],
            dtype_bytes=image.dtype.itemsize,
            small_kernel_size=7,
            large_kernel_size=19,
        )
        self.assertTrue(module.clear_cuda_host_io_cache())
        star_shrink_ops.star_shrink_dog_process_compiled_cuda(
            image,
            sigma_small=1.0,
            sigma_large=3.0,
            threshold_ratio=1.0,
            open_ksize=0,
            dilate_ksize=0,
            shrink_ksize=3,
            shrink_shape="CIRCLE",
            shrink_times=1,
            shrink_ratio=1.0,
            deringing_ksize=5,
        )
        cache_info = module.cuda_host_io_cache_info()

        self.assertEqual(
            cache_info["last_operation"],
            "star_shrink_dog_cuda cudaGetDevice",
        )
        self.assertEqual(
            cache_info["last_device_peak_bytes"],
            estimate.peak_device_bytes,
        )
        self.assertEqual(cache_info["last_pinned_peak_bytes"], 0)

    def test_wavelet_estimate_matches_workspace_high_water_across_levels(
        self,
    ) -> None:
        if not build_info().get("cuda"):
            self.skipTest("CUDA wavelet backend is not built")
        module, error = wavelet_ops._load_compiled_module_result()
        if module is None:
            self.skipTest(error or "compiled custom ops unavailable")
        memory_info = module.cuda_memory_info()
        if not memory_info.get("available"):
            self.skipTest(memory_info.get("reason", "CUDA runtime unavailable"))

        rng = np.random.default_rng(20260717)
        for shape, level in (((32, 33), 2), ((64, 63), 4), ((101, 100), 6)):
            with self.subTest(shape=shape, level=level):
                image = np.ascontiguousarray(rng.normal(size=shape))
                estimate = cuda_memory.estimate_wavelet_dec_rec(
                    height=shape[0],
                    width=shape[1],
                    level=level,
                )
                self.assertTrue(module.clear_cuda_host_io_cache())
                wavelet_ops.wavelet_dec_rec_core_cuda(image, level)
                cache_info = module.cuda_host_io_cache_info()

                self.assertEqual(
                    cache_info["last_operation"],
                    "wavelet_dec_rec_cuda_core cudaGetDevice",
                )
                self.assertEqual(
                    cache_info["last_device_peak_bytes"],
                    estimate.peak_device_bytes,
                )
                self.assertEqual(cache_info["last_pinned_peak_bytes"], 0)

    def test_detection_estimate_bounds_workspace_high_water(self) -> None:
        if not build_info().get("cuda"):
            self.skipTest("CUDA detection backend is not built")
        module, error = detection_ops._load_compiled_module_result()
        if module is None:
            self.skipTest(error or "compiled custom ops unavailable")
        memory_info = module.cuda_memory_info()
        if not memory_info.get("available"):
            self.skipTest(memory_info.get("reason", "CUDA runtime unavailable"))

        image = np.zeros((384, 384), dtype=np.float64)
        for x, y, value in (
            (80, 90, 7.0),
            (140, 260, 9.0),
            (220, 130, 8.0),
            (300, 300, 10.0),
        ):
            cv2.circle(image, (x, y), 5, value, -1)
        image += np.linspace(
            0.0, 0.05, image.shape[1], dtype=np.float64)[None, :]

        small_height, small_width, level, kernel = (
            detection_ops._fused_pixel_component_kernel_params(
                image.shape, 1.0, 9, 2.0)
        )
        estimate = cuda_memory.estimate_star_detect_fused_pixel_components(
            height=image.shape[0],
            width=image.shape[1],
            small_height=small_height,
            small_width=small_width,
            level=level,
            gaussian_ksize=kernel.size,
        )
        self.assertTrue(module.clear_cuda_host_io_cache())
        try:
            detection_ops.star_detect_fused_pixel_components_compiled(
                image, None, 1.0, gaussian_ksize=9, sigma=2.0)
        except RuntimeError as exc:
            if is_cuda_runtime_unavailable_error(exc):
                self.skipTest(f"CUDA runtime unavailable: {exc}")
            raise
        cache_info = module.cuda_host_io_cache_info()

        self.assertEqual(
            cache_info["last_operation"],
            "star_detect_fused_pixel_components cudaGetDevice",
        )
        self.assertGreater(cache_info["last_device_peak_bytes"], 0)
        self.assertLessEqual(
            cache_info["last_device_peak_bytes"],
            estimate.peak_device_bytes,
        )
        self.assertGreater(cache_info["last_pinned_peak_bytes"], 0)
        self.assertLessEqual(
            cache_info["last_pinned_peak_bytes"],
            estimate.peak_pinned_bytes,
        )
