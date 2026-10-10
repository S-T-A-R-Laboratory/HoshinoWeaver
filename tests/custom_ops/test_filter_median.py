import asyncio
from unittest import mock

import cv2
import numpy as np

from hoshicore._custom_op import (
    median_filter_2d,
    median_reduce_chunk,
)
from hoshicore._custom_op._dispatch import cuda_memory_info
from hoshicore._custom_op._dispatch import CustomOpResourceExhaustedError
from hoshicore._custom_op.backend_registry import BackendSelection
from hoshicore._custom_op.backend_registry import registered_backend_candidates
import hoshicore._custom_op.ops.filter as filter_ops
import hoshicore._custom_op.ops.detection as detection_ops
import hoshicore._custom_op.ops.median as median_ops
import hoshicore.component.norma.detection as norma_detection
from hoshicore.component.frame_buffer import MemoryFrameBuffer
import hoshicore.component.star_detect as star_detect
import hoshicore.ops.sigma_clip_ops as sigma_clip_ops
from hoshicore.ops.sigma_clip_ops import MedianReduceOp


from tests.custom_ops._base import CustomOpsTestCase


def _naive_median_filter_2d(image: np.ndarray, ksize: int) -> np.ndarray:
    radius = ksize // 2
    if image.ndim == 2:
        padded = np.pad(image, ((radius, radius), (radius, radius)), mode="edge")
        out = np.empty_like(image)
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                window = padded[y : y + ksize, x : x + ksize]
                out[y, x] = np.median(window).astype(image.dtype)
        return out

    padded = np.pad(
        image,
        ((radius, radius), (radius, radius), (0, 0)),
        mode="edge",
    )
    out = np.empty_like(image)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for c in range(image.shape[2]):
                window = padded[y : y + ksize, x : x + ksize, c]
                out[y, x, c] = np.median(window).astype(image.dtype)
    return out


class TestFilterMedianCustomOps(CustomOpsTestCase):
    def test_median_reduce_chunk_cuda_matches_numpy_exactly(self) -> None:
        if not cuda_memory_info().get("available"):
            self.skipTest("CUDA device unavailable")
        rng = np.random.default_rng(174)
        for dtype in (np.uint8, np.uint16):
            for n_frames in (1, 2, 3, 8, 15, 16, 17, 31, 32, 64, 127, 128):
                stack = rng.integers(
                    0, np.iinfo(dtype).max + 1, size=(n_frames, 19, 23, 3), dtype=dtype
                )
                stack[:, 0, 0] = np.iinfo(dtype).max
                stack[:, 0, 1] = 0
                with self.subTest(dtype=dtype, n_frames=n_frames):
                    got = median_ops.median_reduce_chunk_cuda(stack)
                    expected = np.median(stack, axis=0).astype(dtype)
                    np.testing.assert_array_equal(got, expected)

    def test_median_reduce_chunk_cuda_resource_error_falls_back_to_cpu(self) -> None:
        stack = np.arange(16 * 256 * 256, dtype=np.uint16).reshape(16, 256, 256)
        cuda_candidate = registered_backend_candidates("median_reduce_chunk")[0]
        selection = BackendSelection(cuda_candidate, object())
        with mock.patch.object(median_ops, "_select_median_backend", return_value=selection):
            with mock.patch.object(
                median_ops,
                "median_reduce_chunk_cuda",
                side_effect=CustomOpResourceExhaustedError("test CUDA OOM"),
            ):
                got = median_reduce_chunk(stack)
        expected = median_ops.median_reduce_chunk_compiled(stack)
        np.testing.assert_array_equal(got, expected)

    def test_median_reduce_chunk_auto_selects_cuda_for_large_integer_stack(self) -> None:
        if not cuda_memory_info().get("available"):
            self.skipTest("CUDA device unavailable")
        for n_frames in (16, 128):
            stack = np.arange(n_frames * 256 * 256, dtype=np.uint16).reshape(
                n_frames, 256, 256
            )
            with self.subTest(n_frames=n_frames):
                with mock.patch.dict("os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "auto"}):
                    selection = median_ops._select_median_backend("auto", stack)
                    self.assertEqual(selection.backend, "cuda_host_io")
                    got = median_reduce_chunk(stack)
                np.testing.assert_array_equal(
                    got, np.median(stack, axis=0).astype(np.uint16)
                )

    def test_median_reduce_frames_matches_stack_with_cpu_fallback(self) -> None:
        rng = np.random.default_rng(276)
        frames = [rng.integers(0, 65536, size=(7, 11, 3), dtype=np.uint16)
                  for _ in range(5)]
        expected = np.median(np.stack(frames), axis=0).astype(np.uint16)
        for preference in ("cpu", "numpy"):
            with mock.patch.dict("os.environ", {"HNW_CUSTOM_OPS_FALLBACK": preference}):
                got = median_ops.median_reduce_frames(frames)
            np.testing.assert_array_equal(got, expected)

    def test_median_reduce_frames_casts_to_first_frame_dtype(self) -> None:
        frames = [
            np.array([[1, 5]], dtype=np.uint8),
            np.array([[2, 4]], dtype=np.uint16),
            np.array([[3, 3]], dtype=np.uint8),
        ]
        with mock.patch.dict("os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "cpu"}):
            got = median_ops.median_reduce_frames(frames)
        np.testing.assert_array_equal(got, np.array([[2, 4]], dtype=np.uint8))

    def test_median_reduce_frames_cuda_resource_error_falls_back_to_cpu(self) -> None:
        frames = [np.full((7, 11), frame, dtype=np.uint16) for frame in range(16)]
        cuda_candidate = registered_backend_candidates("median_reduce_chunk")[0]
        selection = BackendSelection(cuda_candidate, object())
        with mock.patch.object(median_ops, "_resolve_median_selection", return_value=selection):
            with mock.patch.object(
                median_ops,
                "median_reduce_frames_cuda",
                side_effect=CustomOpResourceExhaustedError("test CUDA OOM"),
            ):
                got = median_ops.median_reduce_frames(frames)
        np.testing.assert_array_equal(got, np.full((7, 11), 7, dtype=np.uint16))

    def test_median_reduce_frames_cuda_matches_stacked_result(self) -> None:
        if not cuda_memory_info().get("available"):
            self.skipTest("CUDA device unavailable")
        rng = np.random.default_rng(277)
        for dtype in (np.uint8, np.uint16):
            for n_frames in (1, 15, 16, 17, 32, 64, 127, 128):
                frames = [
                    rng.integers(0, np.iinfo(dtype).max + 1, size=(19, 23, 3), dtype=dtype)
                    for _ in range(n_frames)
                ]
                with self.subTest(dtype=dtype, n_frames=n_frames):
                    got = median_ops.median_reduce_frames_cuda(frames)
                    expected = np.median(np.stack(frames), axis=0).astype(dtype)
                    np.testing.assert_array_equal(got, expected)

    def test_median_reduce_chunk_matches_numpy(self) -> None:
        stack = np.array(
            [
                [[[1.0], [5.0]], [[3.0], [4.0]]],
                [[[2.0], [4.0]], [[7.0], [1.0]]],
                [[[6.0], [8.0]], [[5.0], [9.0]]],
                [[[0.0], [7.0]], [[2.0], [6.0]]],
            ],
            dtype=np.float32,
        )

        got = median_reduce_chunk(stack)
        expected = np.median(stack, axis=0)

        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)

    def test_median_reduce_chunk_integer_types(self) -> None:
        """Test uint8/uint16 with both odd and even frame counts."""
        for dtype in (np.uint8, np.uint16):
            # Odd frame count: exact median, no averaging
            stack_odd = np.array(
                [[[10, 50], [30, 40]], [[20, 40], [70, 10]], [[60, 80], [50, 90]]],
                dtype=dtype,
            )
            got = median_reduce_chunk(stack_odd)
            expected = np.median(stack_odd, axis=0).astype(dtype)
            np.testing.assert_array_equal(
                got, expected, err_msg=f"{dtype.__name__} odd-N"
            )

            # Even frame count: average of two middle values (truncated)
            stack_even = np.array(
                [
                    [[10, 50], [30, 40]],
                    [[20, 40], [70, 10]],
                    [[60, 80], [50, 90]],
                    [[0, 70], [20, 60]],
                ],
                dtype=dtype,
            )
            got = median_reduce_chunk(stack_even)
            expected = np.median(stack_even, axis=0).astype(dtype)
            np.testing.assert_array_equal(
                got, expected, err_msg=f"{dtype.__name__} even-N"
            )

    def test_median_reduce_chunk_can_force_numpy_fallback(self) -> None:
        stack = np.array(
            [
                [[[1.0], [5.0]], [[3.0], [4.0]]],
                [[[2.0], [4.0]], [[7.0], [1.0]]],
                [[[6.0], [8.0]], [[5.0], [9.0]]],
                [[[0.0], [7.0]], [[2.0], [6.0]]],
            ],
            dtype=np.float32,
        )

        with mock.patch.dict(
            "os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False
        ):
            with mock.patch.object(
                median_ops,
                "_load_compiled_module_result",
                return_value=(None, "mock error"),
            ):
                got = median_reduce_chunk(stack)

        expected = np.median(stack, axis=0)
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)

    def test_median_filter_2d_uint8_matches_opencv(self) -> None:
        rng = np.random.default_rng(123)
        for shape in ((17, 19), (13, 15, 3)):
            image = rng.integers(0, 255, size=shape, dtype=np.uint8)
            for ksize in (3, 7, 13):
                got = median_filter_2d(image, ksize)
                expected = cv2.medianBlur(image, ksize)
                np.testing.assert_array_equal(got, expected)

    def test_median_filter_2d_uint16_large_kernel_matches_naive(self) -> None:
        rng = np.random.default_rng(456)
        image_2d = rng.integers(0, 65535, size=(6, 7), dtype=np.uint16)
        image_3d = rng.integers(0, 65535, size=(5, 6, 3), dtype=np.uint16)

        for image, ksize in ((image_2d, 7), (image_3d, 7)):
            got = median_filter_2d(image, ksize)
            expected = _naive_median_filter_2d(image, ksize)
            np.testing.assert_array_equal(got, expected)

    def test_median_filter_2d_compiled_matches_numpy_large_kernel(self) -> None:
        rng = np.random.default_rng(654)
        for shape in ((6, 7), (5, 6, 1), (5, 6, 4)):
            image = rng.integers(0, 65535, size=shape, dtype=np.uint16)
            got = filter_ops.median_filter_2d_compiled(image, 7)
            expected = filter_ops.median_filter_2d_numpy(image, 7)
            np.testing.assert_array_equal(got, expected)

    def test_median_filter_2d_uint16_small_kernel_matches_opencv(self) -> None:
        rng = np.random.default_rng(789)
        for shape in ((11, 13), (9, 10, 1), (9, 10, 3)):
            image = rng.integers(0, 65535, size=shape, dtype=np.uint16)
            for ksize in (3, 5):
                got = median_filter_2d(image, ksize)
                expected = cv2.medianBlur(image, ksize)
                if image.ndim == 3 and image.shape[2] == 1:
                    expected = expected[:, :, None]
                np.testing.assert_array_equal(got, expected)

    def test_median_filter_2d_rejects_invalid_ksize(self) -> None:
        image = np.arange(9, dtype=np.uint16).reshape(3, 3)
        with self.assertRaises(ValueError):
            median_filter_2d(image, 4)
        with self.assertRaises(ValueError):
            median_filter_2d(image, 65537)

    def test_median_filter_2d_can_force_numpy_fallback(self) -> None:
        image = np.array(
            [[100, 2, 300], [4, 5000, 6], [700, 8, 900]],
            dtype=np.uint16,
        )

        with mock.patch.dict(
            "os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False
        ):
            with mock.patch.object(
                filter_ops,
                "_load_compiled_module_result",
                return_value=(None, "mock error"),
            ):
                filter_ops._select_median_filter_backend.cache_clear()
                got = filter_ops.median_filter_2d(image, 7)

        expected = _naive_median_filter_2d(image, 7)
        np.testing.assert_array_equal(got, expected)

    def test_star_detect_uint16_large_median_uses_custom_filter(self) -> None:
        image = np.zeros((9, 11), dtype=np.uint16)
        image[4, 5] = 50000

        with mock.patch.object(
            norma_detection,
            "median_star_mask",
            return_value=(
                np.zeros_like(image, dtype=np.uint8),
                np.zeros_like(image, dtype=np.float32),
                0.0,
            ),
        ) as patched_filter:
            mask = star_detect.detect_starmask_by_threshold(
                image,
                ksize=7,
                threshold_ratio=1,
                open_ksize=0,
                dilate_ksize=0,
            )

        patched_filter.assert_called_once()
        args, kwargs = patched_filter.call_args
        np.testing.assert_array_equal(args[0], image)
        self.assertEqual(kwargs["median_ksize"], 7)
        self.assertEqual(mask.shape, image.shape)
        self.assertEqual(mask.dtype, np.uint8)

    def test_star_detect_uint16_large_median_forced_fallback_keeps_precision(
        self,
    ) -> None:
        image = np.zeros((9, 11), dtype=np.uint16)
        image[4, 5] = 50000
        original_median_blur = cv2.medianBlur

        with mock.patch.dict(
            "os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False
        ):
            filter_ops._select_median_filter_backend.cache_clear()
            with mock.patch.object(
                star_detect.cv2,
                "medianBlur",
                wraps=original_median_blur,
            ) as patched_median_blur:
                mask = star_detect.detect_starmask_by_threshold(
                    image,
                    ksize=7,
                    threshold_ratio=1,
                    open_ksize=0,
                    dilate_ksize=0,
                )

        patched_median_blur.assert_not_called()
        self.assertEqual(mask.shape, image.shape)
        self.assertEqual(mask.dtype, np.uint8)

    def test_star_detect_normalized_float_uses_uint16_median_working_data(
        self,
    ) -> None:
        image = np.array([[0.0, 0.25, 1.0]], dtype=np.float32)
        filtered_bg = np.zeros(image.shape, dtype=np.uint16)

        with mock.patch.object(
            detection_ops,
            "_median_filter_2d_numpy",
            return_value=filtered_bg,
        ) as patched_filter:
            detection_ops.median_star_mask_numpy(
                image,
                median_ksize=7,
                threshold_ratio=1,
                open_ksize=0,
            )

        working_image = patched_filter.call_args.args[0]
        self.assertEqual(working_image.dtype, np.uint16)
        np.testing.assert_array_equal(
            working_image,
            np.rint(image * 65535).astype(np.uint16),
        )

    def test_star_detect_response_subtraction_does_not_underflow(self) -> None:
        image = np.zeros((3, 3), dtype=np.uint16)
        filtered_bg = np.full_like(image, 1000)

        with mock.patch.object(
            detection_ops,
            "_median_filter_2d_numpy",
            return_value=filtered_bg,
        ):
            mask, response, _ = detection_ops.median_star_mask_numpy(
                image,
                median_ksize=7,
                threshold_ratio=1,
                open_ksize=0,
            )

        self.assertEqual(response.dtype, np.float32)
        self.assertTrue(np.all(response < 0))
        self.assertFalse(np.any(mask))

    def test_star_detect_rejects_unnormalized_float_input(self) -> None:
        image = np.array([[0.0, 2.0]], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "normalized to"):
            star_detect.detect_starmask_by_threshold(image, ksize=7)

    def test_median_reduce_op_routes_chunk_through_custom_op(self) -> None:
        frame_buffer = MemoryFrameBuffer()
        frames = [
            np.array([[1, 5], [3, 4]], dtype=np.uint16),
            np.array([[2, 4], [7, 1]], dtype=np.uint16),
            np.array([[6, 8], [5, 9]], dtype=np.uint16),
            np.array([[0, 7], [2, 6]], dtype=np.uint16),
        ]
        for frame in frames:
            frame_buffer.append(frame)
        frame_buffer.acquire()

        async def iter_chunk_prefetch(row_ranges):
            for row_start, row_end in row_ranges:
                yield [
                    frame_buffer.get_rows(i, row_start, row_end)
                    for i in range(len(frame_buffer))
                ]

        frame_buffer.iter_chunk_prefetch = iter_chunk_prefetch

        op = MedianReduceOp("median_reduce")
        outputs = {}

        async def run_case() -> None:
            with mock.patch.object(
                sigma_clip_ops,
                "custom_median_reduce_frames",
                side_effect=lambda chunks: median_ops.median_reduce_chunk_numpy(np.stack(chunks)),
            ) as patched_custom:

                async def run_cpu(fn, *args, **kwargs):
                    return fn(*args, **kwargs)

                async def capture(payload):
                    outputs.update(payload)

                op._run_cpu = run_cpu
                op._broadcast_outputs = capture
                await op._async_execute(
                    {
                        "buffer_handle": frame_buffer,
                        "chunk_rows": 1,
                    }
                )
                self.assertEqual(patched_custom.call_count, 2)

        asyncio.run(run_case())
        expected = np.median(np.stack(frames, axis=0), axis=0).astype(np.uint16)
        self.assertIn("result", outputs)
        np.testing.assert_array_equal(outputs["result"].data, expected)
