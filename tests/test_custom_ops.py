import asyncio
import unittest
from unittest import mock

import cv2
import numpy as np

from hoshicore._custom_op import (
    build_info,
    camera_model_remap,
    custom_ops_available,
    equalize_noise_correct,
    fgp_add,
    fgp_accumulate,
    fgp_masked_mean_merge,
    huber_weighted_accumulate,
    max_combine,
    median_reduce_chunk,
    sigma_clip_fused_masked_merge,
    sigma_clip_fused_merge,
    threshold_max_merge as custom_threshold_max_merge,
)
import hoshicore._custom_op.ops.fgp as fgp_ops
import hoshicore._custom_op.ops.max as max_ops
import hoshicore._custom_op.ops.median as median_ops
import hoshicore._custom_op.ops.noise as noise_ops
import hoshicore._custom_op.ops.remap as remap_ops
from hoshicore.component.data_container import FastGaussianParam, HuberMeanParam
from hoshicore.component.frame_buffer import DiskFrameBuffer
import hoshicore.component.noise_equalization as noise_equalization
import hoshicore.component.norma.frame_align as frame_align
import hoshicore.component.norma.types as norma_types
from hoshicore.component.merger import HuberWeightedMerger
from hoshicore.component.merger import MaxMerger
from hoshicore.component.merger import MeanMerger
from hoshicore.component.norma.types import CameraModel, Intrinsics
import hoshicore.ops.sigma_clip_ops as sigma_clip_ops
from hoshicore.ops.sigma_clip_ops import MedianReduceOp
from hoshicore.ops.sigma_clip_ops import ThresholdMaxIteratorOp
from hoshicore.ops.trailstacker import MeanStackerOp


class TestCustomOpsFallback(unittest.TestCase):
    def tearDown(self) -> None:
        fgp_ops._load_compiled_module_result.cache_clear()
        fgp_ops._compiled_build_info.cache_clear()
        fgp_ops._select_fgp_backend.cache_clear()
        fgp_ops._select_fgp_add_backend.cache_clear()
        fgp_ops._select_huber_backend.cache_clear()
        fgp_ops._LAST_APPLIED_COMPILED_THREADS = None
        max_ops._load_compiled_module_result.cache_clear()
        max_ops._compiled_build_info.cache_clear()
        max_ops._select_max_backend.cache_clear()
        max_ops._select_threshold_max_backend.cache_clear()
        max_ops._LAST_APPLIED_COMPILED_THREADS = None
        noise_ops._load_compiled_module_result.cache_clear()
        noise_ops._compiled_build_info.cache_clear()
        noise_ops._select_equalize_noise_backend.cache_clear()
        noise_ops._LAST_APPLIED_COMPILED_THREADS = None
        median_ops._load_compiled_module_result.cache_clear()
        median_ops._compiled_build_info.cache_clear()
        median_ops._select_median_backend.cache_clear()
        median_ops._LAST_APPLIED_COMPILED_THREADS = None
        remap_ops._load_compiled_module_result.cache_clear()
        remap_ops._select_camera_model_remap_backend.cache_clear()

    def test_max_combine_matches_numpy(self) -> None:
        base = np.array([[1, 5], [3, 4]], dtype=np.uint16)
        fresh = np.array([[2, 4], [7, 1]], dtype=np.uint16)

        got = max_combine(base, fresh)
        expected = np.maximum(np.array([[1, 5], [3, 4]], dtype=np.uint16), fresh)

        self.assertIs(got, base)
        np.testing.assert_array_equal(base, expected)
        np.testing.assert_array_equal(got, expected)

    def test_max_merger_works_with_custom_op_fallback(self) -> None:
        merger = MaxMerger(int_weight=False)
        merger.merge(np.array([[1, 9], [2, 3]], dtype=np.uint16), None)
        merger.merge(np.array([[5, 1], [2, 8]], dtype=np.uint16), None)

        np.testing.assert_array_equal(
            merger.merged_image,
            np.array([[5, 9], [2, 8]], dtype=np.uint16),
        )

    def test_threshold_max_merge_matches_numpy(self) -> None:
        frame = np.array([[6.0, 12.0], [4.0, 15.0]], dtype=np.float64)
        mean_img = np.array([[5.0, 8.0], [4.5, 10.0]], dtype=np.float64)
        std_img = np.array([[0.5, 1.0], [0.5, 2.0]], dtype=np.float64)
        base = np.array([[5.0, 9.0], [4.5, 11.0]], dtype=np.float64)
        expected = np.array(base, copy=True)

        got = custom_threshold_max_merge(frame, mean_img, std_img, base, 2.0, 0.5)
        max_ops.threshold_max_merge_numpy(frame, mean_img, std_img, expected, 2.0, 0.5)

        self.assertIs(got, base)
        np.testing.assert_allclose(base, expected, rtol=1e-7, atol=1e-7)

    def test_threshold_max_merge_can_force_numpy_fallback(self) -> None:
        frame = np.array([[6.0, 12.0], [4.0, 15.0]], dtype=np.float64)
        mean_img = np.array([[5.0, 8.0], [4.5, 10.0]], dtype=np.float64)
        std_img = np.array([[0.5, 1.0], [0.5, 2.0]], dtype=np.float64)
        base = np.array([[5.0, 9.0], [4.5, 11.0]], dtype=np.float64)
        expected = np.array(base, copy=True)

        with mock.patch.dict("os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False):
            with mock.patch.object(max_ops, "_load_compiled_module_result", return_value=(None, "mock error")):
                max_ops._select_threshold_max_backend.cache_clear()
                got = custom_threshold_max_merge(frame, mean_img, std_img, base, 2.0, 0.5)

        max_ops.threshold_max_merge_numpy(frame, mean_img, std_img, expected, 2.0, 0.5)
        self.assertIs(got, base)
        np.testing.assert_allclose(base, expected, rtol=1e-7, atol=1e-7)

    def test_threshold_max_merge_keeps_morphology_on_numpy_path(self) -> None:
        frame = np.array([[6.0, 12.0], [4.0, 15.0]], dtype=np.float64)
        mean_img = np.array([[5.0, 8.0], [4.5, 10.0]], dtype=np.float64)
        std_img = np.array([[0.5, 1.0], [0.5, 2.0]], dtype=np.float64)
        result = np.array(mean_img, copy=True)
        expected = np.array(mean_img, copy=True)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        with mock.patch.object(noise_equalization, "custom_threshold_max_merge") as patched_custom:
            noise_equalization.threshold_max_merge(
                frame,
                mean_img,
                std_img,
                result,
                2.0,
                0.5,
                kernel,
            )

        mask = frame > (mean_img + 2.0 * std_img)
        mask = cv2.morphologyEx(mask.view(np.uint8), cv2.MORPH_OPEN, kernel).view(bool)
        signal = frame * 0.5
        np.maximum(expected, np.where(mask, signal, mean_img), out=expected)

        patched_custom.assert_not_called()
        np.testing.assert_allclose(result, expected, rtol=1e-7, atol=1e-7)

    def test_equalize_noise_correct_matches_numpy(self) -> None:
        max_img = np.array([[20.0, 180.0], [90.0, 250.0]], dtype=np.float64)
        filled_std_img = np.array([[8.0, 20.0], [12.0, 25.0]], dtype=np.float64)

        got = equalize_noise_correct(max_img, filled_std_img, 10.0, 1.25, 255.0, 0.9)
        expected = noise_ops.equalize_noise_correct_numpy(
            max_img,
            filled_std_img,
            10.0,
            1.25,
            255.0,
            0.9,
        )

        np.testing.assert_allclose(got, expected, rtol=1e-7, atol=1e-7)

    def test_equalize_noise_correct_can_force_numpy_fallback(self) -> None:
        max_img = np.array([[20.0, 180.0], [90.0, 250.0]], dtype=np.float64)
        filled_std_img = np.array([[8.0, 20.0], [12.0, 25.0]], dtype=np.float64)

        with mock.patch.dict("os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False):
            with mock.patch.object(noise_ops, "_load_compiled_module_result", return_value=(None, "mock error")):
                noise_ops._select_equalize_noise_backend.cache_clear()
                got = equalize_noise_correct(max_img, filled_std_img, 10.0, 1.25, 255.0, 0.9)

        expected = noise_ops.equalize_noise_correct_numpy(
            max_img,
            filled_std_img,
            10.0,
            1.25,
            255.0,
            0.9,
        )
        np.testing.assert_allclose(got, expected, rtol=1e-7, atol=1e-7)

    def test_equalize_noise_routes_pixel_correction_through_custom_op(self) -> None:
        max_img = np.array(
            [[[12.0, 15.0, 18.0], [30.0, 36.0, 42.0]],
             [[24.0, 20.0, 16.0], [48.0, 45.0, 51.0]]],
            dtype=np.float64,
        )
        mean_img = np.array(
            [[[10.0, 11.0, 13.0], [28.0, 30.0, 35.0]],
             [[21.0, 18.0, 14.0], [40.0, 39.0, 43.0]]],
            dtype=np.float64,
        )
        std_img = np.array(
            [[[2.0, 3.0, 4.0], [5.0, 7.0, 6.0]],
             [[3.0, 2.0, 1.5], [8.0, 7.0, 9.0]]],
            dtype=np.float64,
        )
        n_img = np.array(
            [[[10, 10, 10], [9, 9, 9]],
             [[10, 10, 10], [8, 8, 8]]],
            dtype=np.uint16,
        )

        max_value = float(np.max(max_img))
        threshold = np.quantile(n_img, 1.0 - 0.25)
        bg_mask = n_img >= threshold
        residual = (max_img - mean_img)[bg_mask]
        sigma_bg = std_img[bg_mask]
        valid = sigma_bg > 0
        r_valid = residual[valid]
        s_valid = sigma_bg[valid]
        c_n_eff = float(np.median(r_valid / s_valid))
        sigma_ref = np.median(s_valid)
        squeeze_std = std_img.reshape((-1, 3))
        mean_std = np.mean(squeeze_std, axis=0)
        std_std = np.std(squeeze_std, axis=0)
        mask = (std_img > (mean_std + 3.0 * std_std)[None, None, ...])
        filled_std_img = noise_equalization.fill_local_mean(std_img, mask, kernel_size=21)
        expected = noise_ops.equalize_noise_correct_numpy(
            max_img,
            filled_std_img,
            sigma_ref,
            c_n_eff,
            max_value,
            0.9,
        )

        with mock.patch.object(
            noise_equalization,
            "custom_equalize_noise_correct",
            wraps=noise_ops.equalize_noise_correct_numpy,
        ) as patched_custom:
            got = noise_equalization.equalize_noise(
                max_img,
                mean_img,
                std_img,
                n_img,
                top_fraction=0.25,
                sigma_reject=3.0,
                highlight_preserve=0.9,
            )

        patched_custom.assert_called_once()
        np.testing.assert_allclose(got, expected, rtol=1e-7, atol=1e-7)

    def test_camera_model_remap_matches_numpy(self) -> None:
        image = np.linspace(0.0, 1.0, num=4 * 5 * 3, dtype=np.float32).reshape(4, 5, 3)
        pitch = np.deg2rad(1.5)
        rotation = np.array([
            [np.cos(pitch), 0.0, np.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-np.sin(pitch), 0.0, np.cos(pitch)],
        ], dtype=np.float32)

        got = camera_model_remap(
            image=image,
            out_height=3,
            out_width=4,
            fx_src=9.0,
            fy_src=8.5,
            cx_src=2.0,
            cy_src=1.5,
            fx_dst=8.0,
            fy_dst=7.5,
            cx_dst=1.5,
            cy_dst=1.0,
            rotation_dst_to_src=rotation,
        )
        expected = remap_ops.camera_model_remap_numpy(
            image=image,
            out_height=3,
            out_width=4,
            fx_src=9.0,
            fy_src=8.5,
            cx_src=2.0,
            cy_src=1.5,
            fx_dst=8.0,
            fy_dst=7.5,
            cx_dst=1.5,
            cy_dst=1.0,
            rotation_dst_to_src=rotation,
        )

        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)

    def test_camera_model_remap_can_force_numpy_fallback(self) -> None:
        image = np.arange(18, dtype=np.uint8).reshape(3, 3, 2)
        rotation = np.eye(3, dtype=np.float32)
        with mock.patch.dict("os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False):
            with mock.patch.object(remap_ops, "_load_compiled_module_result", return_value=(None, "mock error")):
                remap_ops._select_camera_model_remap_backend.cache_clear()
                got = camera_model_remap(
                    image=image,
                    out_height=2,
                    out_width=2,
                    fx_src=6.0,
                    fy_src=6.0,
                    cx_src=1.0,
                    cy_src=1.0,
                    fx_dst=5.0,
                    fy_dst=5.0,
                    cx_dst=1.0,
                    cy_dst=1.0,
                    rotation_dst_to_src=rotation,
                )

        expected = remap_ops.camera_model_remap_numpy(
            image=image,
            out_height=2,
            out_width=2,
            fx_src=6.0,
            fy_src=6.0,
            cx_src=1.0,
            cy_src=1.0,
            fx_dst=5.0,
            fy_dst=5.0,
            cx_dst=1.0,
            cy_dst=1.0,
            rotation_dst_to_src=rotation,
        )
        np.testing.assert_array_equal(got, expected)

    def test_camera_model_remap_falls_back_when_cuda_runtime_is_unavailable(self) -> None:
        image = np.arange(18, dtype=np.uint8).reshape(3, 3, 2)
        rotation = np.eye(3, dtype=np.float32)
        expected = remap_ops.camera_model_remap_numpy(
            image=image,
            out_height=2,
            out_width=2,
            fx_src=6.0,
            fy_src=6.0,
            cx_src=1.0,
            cy_src=1.0,
            fx_dst=5.0,
            fy_dst=5.0,
            cx_dst=1.0,
            cy_dst=1.0,
            rotation_dst_to_src=rotation,
        )

        with mock.patch.object(
                remap_ops,
                "_select_camera_model_remap_backend",
                return_value=("compiled", mock.Mock(side_effect=RuntimeError(
                    "camera_model_remap cudaMalloc(image): no CUDA-capable device is detected")))):
            got = camera_model_remap(
                image=image,
                out_height=2,
                out_width=2,
                fx_src=6.0,
                fy_src=6.0,
                cx_src=1.0,
                cy_src=1.0,
                fx_dst=5.0,
                fy_dst=5.0,
                cx_dst=1.0,
                cy_dst=1.0,
                rotation_dst_to_src=rotation,
            )

        np.testing.assert_array_equal(got, expected)

    def test_project_image_from_camera_routes_zero_distortion_through_custom_fused(self) -> None:
        img = np.arange(16, dtype=np.uint8).reshape(4, 4)
        intrinsics = Intrinsics(
            focal_length_mm=8.0,
            sensor_width_mm=8.0,
            sensor_height_mm=8.0,
            image_width_px=4,
            image_height_px=4,
        )
        src_camera = CameraModel(intrinsics=intrinsics)
        dst_camera = CameraModel(intrinsics=intrinsics)

        expected = remap_ops.camera_model_remap_numpy(
            image=img,
            out_height=4,
            out_width=4,
            fx_src=float(src_camera.K[0, 0]),
            fy_src=float(src_camera.K[1, 1]),
            cx_src=float(src_camera.K[0, 2]),
            cy_src=float(src_camera.K[1, 2]),
            fx_dst=float(dst_camera.K[0, 0]),
            fy_dst=float(dst_camera.K[1, 1]),
            cx_dst=float(dst_camera.K[0, 2]),
            cy_dst=float(dst_camera.K[1, 2]),
            rotation_dst_to_src=np.eye(3, dtype=np.float32),
        )

        with mock.patch.object(
                norma_types,
                "custom_camera_model_remap",
                wraps=norma_types.custom_camera_model_remap) as patched_custom:
            got = dst_camera.project_image_from_camera(src_camera, img, (4, 4))

        patched_custom.assert_called_once()
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

        with mock.patch.dict("os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False):
            with mock.patch.object(median_ops, "_load_compiled_module_result", return_value=(None, "mock error")):
                median_ops._select_median_backend.cache_clear()
                got = median_reduce_chunk(stack)

        expected = np.median(stack, axis=0)
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)

    def test_median_reduce_op_routes_chunk_through_custom_op(self) -> None:
        frame_buffer = DiskFrameBuffer()
        frames = [
            np.array([[1, 5], [3, 4]], dtype=np.uint16),
            np.array([[2, 4], [7, 1]], dtype=np.uint16),
            np.array([[6, 8], [5, 9]], dtype=np.uint16),
            np.array([[0, 7], [2, 6]], dtype=np.uint16),
        ]
        for frame in frames:
            frame_buffer.append(frame)
        frame_buffer.acquire()

        op = MedianReduceOp("median_reduce")
        outputs = {}

        async def run_case() -> None:
            with mock.patch.object(
                sigma_clip_ops,
                "custom_median_reduce_chunk",
                wraps=median_ops.median_reduce_chunk_numpy,
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
        expected = np.median(np.stack(frames, axis=0).astype(np.float32), axis=0)
        self.assertIn("result", outputs)
        np.testing.assert_allclose(outputs["result"].data, expected, rtol=1e-6, atol=1e-6)

    def test_fgp_accumulate_matches_python_unweighted(self) -> None:
        base = FastGaussianParam(
            np.array([[1, 5], [3, 4]], dtype=np.uint16),
            source_dtype=np.dtype("uint16"),
        )
        expected = FastGaussianParam(
            np.array([[1, 5], [3, 4]], dtype=np.uint16),
            source_dtype=np.dtype("uint16"),
        )
        fresh = np.array([[2, 4], [7, 1]], dtype=np.uint16)

        got = fgp_accumulate(base, fresh)
        expected = expected + FastGaussianParam(fresh, source_dtype=fresh.dtype)

        self.assertIs(got, base)
        np.testing.assert_array_equal(base.sum_mu, expected.sum_mu)
        np.testing.assert_array_equal(base.square_sum, expected.square_sum)
        np.testing.assert_array_equal(base.n, expected.n)

    def test_fgp_accumulate_matches_python_weighted_int(self) -> None:
        weight = 3
        base = FastGaussianParam(
            np.array([[1, 5], [3, 4]], dtype=np.uint16),
            source_dtype=np.dtype("uint16"),
        )
        expected = FastGaussianParam(
            np.array([[1, 5], [3, 4]], dtype=np.uint16),
            source_dtype=np.dtype("uint16"),
        )
        fresh = np.array([[2, 4], [7, 1]], dtype=np.uint16)

        got = fgp_accumulate(base, fresh, weight)
        patch = FastGaussianParam(fresh, source_dtype=fresh.dtype)
        patch = patch * weight
        expected = expected + patch

        self.assertIs(got, base)
        np.testing.assert_array_equal(base.sum_mu, expected.sum_mu)
        np.testing.assert_array_equal(base.square_sum, expected.square_sum)
        np.testing.assert_array_equal(base.n, expected.n)

    def test_fgp_accumulate_can_force_numpy_fallback(self) -> None:
        base = FastGaussianParam(
            np.array([[1, 5], [3, 4]], dtype=np.uint16),
            source_dtype=np.dtype("uint16"),
        )
        fresh = np.array([[2, 4], [7, 1]], dtype=np.uint16)

        with mock.patch.dict("os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False):
            with mock.patch.object(fgp_ops, "_load_compiled_module_result", return_value=(None, "mock error")):
                fgp_ops._select_fgp_backend.cache_clear()
                got = fgp_accumulate(base, fresh)

        expected = FastGaussianParam(
            np.array([[1, 5], [3, 4]], dtype=np.uint16),
            source_dtype=np.dtype("uint16"),
        )
        expected = expected + FastGaussianParam(fresh, source_dtype=fresh.dtype)

        self.assertIs(got, base)
        np.testing.assert_array_equal(base.sum_mu, expected.sum_mu)
        np.testing.assert_array_equal(base.square_sum, expected.square_sum)
        np.testing.assert_array_equal(base.n, expected.n)

    def test_fgp_add_matches_python(self) -> None:
        base = FastGaussianParam(
            np.array([[1, 5], [3, 4]], dtype=np.uint16),
            source_dtype=np.dtype("uint16"),
        )
        other = FastGaussianParam(
            np.array([[2, 4], [7, 1]], dtype=np.uint16),
            source_dtype=np.dtype("uint16"),
        )
        expected = FastGaussianParam(
            np.array([[1, 5], [3, 4]], dtype=np.uint16),
            source_dtype=np.dtype("uint16"),
        )
        expected = expected + other

        got = fgp_add(base, other)

        self.assertIs(got, base)
        np.testing.assert_array_equal(base.sum_mu, expected.sum_mu)
        np.testing.assert_array_equal(base.square_sum, expected.square_sum)
        np.testing.assert_array_equal(base.n, expected.n)

    def test_fgp_add_can_force_numpy_fallback(self) -> None:
        base = FastGaussianParam(
            np.array([[1, 5], [3, 4]], dtype=np.uint16),
            source_dtype=np.dtype("uint16"),
        )
        other = FastGaussianParam(
            np.array([[2, 4], [7, 1]], dtype=np.uint16),
            source_dtype=np.dtype("uint16"),
        )

        with mock.patch.dict("os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False):
            with mock.patch.object(fgp_ops, "_load_compiled_module_result", return_value=(None, "mock error")):
                fgp_ops._select_fgp_add_backend.cache_clear()
                got = fgp_add(base, other)

        expected = FastGaussianParam(
            np.array([[1, 5], [3, 4]], dtype=np.uint16),
            source_dtype=np.dtype("uint16"),
        )
        expected = expected + other

        self.assertIs(got, base)
        np.testing.assert_array_equal(base.sum_mu, expected.sum_mu)
        np.testing.assert_array_equal(base.square_sum, expected.square_sum)
        np.testing.assert_array_equal(base.n, expected.n)

    def test_huber_weighted_accumulate_matches_numpy_scalar_weight(self) -> None:
        img = np.array([[[4], [9]], [[6], [3]]], dtype=np.uint16)
        ref_mean = np.array([[[5], [7]], [[4], [2]]], dtype=np.float32)
        ref_std = np.array([[[2], [1]], [[3], [2]]], dtype=np.float32)
        base = HuberMeanParam(
            weighted_sum=np.zeros_like(img, dtype=np.float64),
            weight_total=np.zeros_like(img, dtype=np.float64),
            source_dtype=img.dtype,
        )

        got = huber_weighted_accumulate(base, img, ref_mean, ref_std, 1.345, weight=0.5)

        residual = (img.astype(np.float32) - ref_mean) / (ref_std + 1e-10)
        abs_residual = np.abs(residual)
        huber_weight = np.where(
            abs_residual <= 1.345,
            np.ones_like(abs_residual, dtype=np.float32),
            (1.345 / (abs_residual + 1e-10)).astype(np.float32),
        ) * 0.5
        expected_sum = (img * huber_weight).astype(np.float64)
        expected_total = huber_weight.astype(np.float64)

        self.assertIs(got, base)
        np.testing.assert_allclose(base.weighted_sum, expected_sum, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(base.weight_total, expected_total, rtol=1e-6, atol=1e-6)

    def test_huber_weighted_accumulate_keeps_array_weight_on_numpy_path(self) -> None:
        img = np.array([[[4], [9]], [[6], [3]]], dtype=np.uint16)
        ref_mean = np.array([[[5], [7]], [[4], [2]]], dtype=np.float32)
        ref_std = np.array([[[2], [1]], [[3], [2]]], dtype=np.float32)
        frame_weight = np.array([[[1.0], [0.25]], [[0.5], [0.75]]], dtype=np.float32)
        base = HuberMeanParam(
            weighted_sum=np.zeros_like(img, dtype=np.float64),
            weight_total=np.zeros_like(img, dtype=np.float64),
            source_dtype=img.dtype,
        )

        got = huber_weighted_accumulate(base, img, ref_mean, ref_std, 1.345, weight=frame_weight)

        residual = (img.astype(np.float32) - ref_mean) / (ref_std + 1e-10)
        abs_residual = np.abs(residual)
        huber_weight = np.where(
            abs_residual <= 1.345,
            np.ones_like(abs_residual, dtype=np.float32),
            (1.345 / (abs_residual + 1e-10)).astype(np.float32),
        ) * frame_weight
        expected_sum = (img * huber_weight).astype(np.float64)
        expected_total = huber_weight.astype(np.float64)

        self.assertIs(got, base)
        np.testing.assert_allclose(base.weighted_sum, expected_sum, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(base.weight_total, expected_total, rtol=1e-6, atol=1e-6)

    def test_mean_merger_works_with_fgp_accumulate(self) -> None:
        merger = MeanMerger(int_weight=False)
        merger.merge(np.array([[1, 9], [2, 3]], dtype=np.uint16), None)
        merger.merge(np.array([[5, 1], [2, 8]], dtype=np.uint16), None)

        expected = FastGaussianParam(
            np.array([[1, 9], [2, 3]], dtype=np.uint16),
            source_dtype=np.dtype("uint16"),
        )
        expected = expected + FastGaussianParam(
            np.array([[5, 1], [2, 8]], dtype=np.uint16),
            source_dtype=np.dtype("uint16"),
        )
        np.testing.assert_array_equal(merger.result.sum_mu, expected.sum_mu)
        np.testing.assert_array_equal(merger.result.square_sum, expected.square_sum)
        np.testing.assert_array_equal(merger.result.n, expected.n)

    def test_huber_weighted_merger_routes_through_custom_accumulate(self) -> None:
        ref_stats = FastGaussianParam(
            np.array([[5, 7], [4, 2]], dtype=np.uint16),
            source_dtype=np.dtype("uint16"),
        )
        merger = HuberWeightedMerger(ref_stats=ref_stats, huber_c=1.345)
        frame_a = np.array([[4, 9], [6, 3]], dtype=np.uint16)
        frame_b = np.array([[5, 6], [7, 1]], dtype=np.uint16)

        with mock.patch(
            "hoshicore.component.merger.custom_huber_weighted_accumulate",
            wraps=fgp_ops.huber_weighted_accumulate_numpy,
        ) as patched_huber:
            merger.merge(frame_a, 0.5)
            merger.merge(frame_b, 0.25)

        self.assertEqual(patched_huber.call_count, 2)

        ref_mean = ref_stats.mu.astype(np.float32)
        ref_std = np.sqrt(np.maximum(ref_stats.var, 0)).astype(np.float32)
        expected_sum = np.zeros_like(frame_a, dtype=np.float64)
        expected_total = np.zeros_like(frame_a, dtype=np.float64)
        for frame, weight in ((frame_a, 0.5), (frame_b, 0.25)):
            residual = (frame.astype(np.float32) - ref_mean) / (ref_std + 1e-10)
            abs_residual = np.abs(residual)
            huber_weight = np.where(
                abs_residual <= 1.345,
                np.ones_like(abs_residual, dtype=np.float32),
                (1.345 / (abs_residual + 1e-10)).astype(np.float32),
            ) * weight
            expected_sum += (frame * huber_weight).astype(np.float64)
            expected_total += huber_weight.astype(np.float64)

        np.testing.assert_allclose(merger.result.weighted_sum, expected_sum, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(merger.result.weight_total, expected_total, rtol=1e-6, atol=1e-6)

    def test_threshold_max_iterator_routes_noop_morphology_through_custom_kernel(self) -> None:
        class FakeFrameBuffer:
            def __init__(self, items):
                self._items = items
                self.cleaned = False

            def __len__(self):
                return len(self._items)

            def __getitem__(self, idx):
                return self._items[idx]

            async def iter_prefetch(self, start=0, stop=None):
                for item in self._items[start:stop]:
                    yield item

            def cleanup(self):
                self.cleaned = True

        fgp_total = FastGaussianParam(
            np.array([[5, 8], [4, 10]], dtype=np.uint16),
            source_dtype=np.dtype("uint16"),
        )
        frame_buffer = FakeFrameBuffer(
            [
                (np.array([[6, 12], [4, 15]], dtype=np.uint16), 0.5),
                (np.array([[4, 9], [9, 12]], dtype=np.uint16), 1.0),
            ]
        )
        op = ThresholdMaxIteratorOp("threshold_max")
        outputs = {}

        async def run_case() -> None:
            with mock.patch.object(
                noise_equalization,
                "custom_threshold_max_merge",
                wraps=max_ops.threshold_max_merge_numpy,
            ) as patched_custom:
                async def run_cpu(fn, *args, **kwargs):
                    return fn(*args, **kwargs)

                async def capture(payload):
                    outputs.update(payload)

                op._run_cpu = run_cpu
                op._broadcast_outputs = capture
                await op._async_execute(
                    {
                        "fgp_total": fgp_total,
                        "buffer_handle": frame_buffer,
                        "n_sigma": 2.0,
                        "morph_kernel_size": 1,
                    }
                )
                self.assertEqual(patched_custom.call_count, len(frame_buffer))

        asyncio.run(run_case())
        self.assertTrue(frame_buffer.cleaned)
        self.assertIn("result", outputs)

    def test_custom_ops_available_returns_bool(self) -> None:
        self.assertIsInstance(custom_ops_available(), bool)

    def test_build_info_returns_minimal_metadata(self) -> None:
        info = build_info()
        self.assertIsInstance(info, dict)
        self.assertIn("available", info)

    def test_build_info_reports_fallback_backend(self) -> None:
        with mock.patch.object(max_ops, "_load_compiled_module_result", return_value=(None, "mock error")):
            max_ops._select_max_backend.cache_clear()
            info = max_ops.build_info()

        self.assertFalse(info["available"])
        self.assertEqual(info["backend"], "numpy")

    def test_max_combine_can_force_numpy_fallback(self) -> None:
        base = np.array([[1, 5], [3, 4]], dtype=np.uint16)
        fresh = np.array([[2, 4], [7, 1]], dtype=np.uint16)

        with mock.patch.dict("os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False):
            with mock.patch.object(max_ops, "_load_compiled_module_result", return_value=(None, "mock error")):
                max_ops._select_max_backend.cache_clear()
                got = max_ops.max_combine(base, fresh)

        expected = np.maximum(np.array([[1, 5], [3, 4]], dtype=np.uint16), fresh)
        self.assertIs(got, base)
        np.testing.assert_array_equal(base, expected)
        np.testing.assert_array_equal(got, expected)

    def test_build_info_includes_thread_policy(self) -> None:
        info = build_info()
        self.assertIn("thread_policy", info)

    def test_fgp_masked_mean_merge_can_force_numpy_fallback(self) -> None:
        img = np.array(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
            dtype=np.uint16,
        )
        mask = np.array([[True, False], [True, True]])
        sum_mu = np.zeros_like(img, dtype=np.uint32)
        square_sum = np.zeros_like(img, dtype=np.uint64)
        count = np.zeros_like(img, dtype=np.uint16)

        with mock.patch.dict("os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False):
            with mock.patch.object(fgp_ops, "_load_compiled_module_result", return_value=(None, "mock error")):
                fgp_masked_mean_merge(img, mask, sum_mu, square_sum, count)

        expected_mask = np.broadcast_to(mask[..., None], img.shape).astype(np.uint16)
        np.testing.assert_array_equal(sum_mu, img.astype(np.uint32) * expected_mask)
        np.testing.assert_array_equal(
            square_sum,
            np.square(img, dtype=np.uint64) * expected_mask.astype(np.uint64),
        )
        np.testing.assert_array_equal(count, expected_mask)

    def test_sigma_clip_fused_merge_can_force_numpy_fallback(self) -> None:
        img = np.array([[[1], [8]], [[5], [2]]], dtype=np.uint16)
        rej_high = np.array([[[3], [6]], [[7], [4]]], dtype=np.uint16)
        rej_low = np.array([[[0], [2]], [[4], [3]]], dtype=np.uint16)
        sum_mu = np.zeros_like(img, dtype=np.uint32)
        square_sum = np.zeros_like(img, dtype=np.uint64)
        count = np.zeros_like(img, dtype=np.uint16)

        with mock.patch.dict("os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False):
            with mock.patch.object(fgp_ops, "_load_compiled_module_result", return_value=(None, "mock error")):
                sigma_clip_fused_merge(img, rej_high, rej_low, sum_mu, square_sum, count)

        rejected = ((img < rej_low) | (img > rej_high)).astype(np.uint16)
        np.testing.assert_array_equal(sum_mu, img.astype(np.uint32) * rejected)
        np.testing.assert_array_equal(
            square_sum,
            np.square(img, dtype=np.uint64) * rejected.astype(np.uint64),
        )
        np.testing.assert_array_equal(count, rejected)

    def test_sigma_clip_fused_masked_merge_can_force_numpy_fallback(self) -> None:
        img = np.array(
            [[[1, 2], [8, 9]], [[5, 6], [2, 1]]],
            dtype=np.uint16,
        )
        mask = np.array([[True, False], [True, True]])
        rej_high = np.array(
            [[[3, 3], [6, 6]], [[7, 7], [4, 4]]],
            dtype=np.uint16,
        )
        rej_low = np.array(
            [[[0, 0], [2, 2]], [[4, 4], [3, 3]]],
            dtype=np.uint16,
        )
        sum_mu = np.zeros_like(img, dtype=np.uint32)
        square_sum = np.zeros_like(img, dtype=np.uint64)
        count = np.zeros_like(img, dtype=np.uint16)

        with mock.patch.dict("os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False):
            with mock.patch.object(fgp_ops, "_load_compiled_module_result", return_value=(None, "mock error")):
                sigma_clip_fused_masked_merge(
                    img,
                    mask,
                    rej_high,
                    rej_low,
                    sum_mu,
                    square_sum,
                    count,
                )

        rejected = (mask[..., None] & ((img < rej_low) | (img > rej_high))).astype(np.uint16)
        np.testing.assert_array_equal(sum_mu, img.astype(np.uint32) * rejected)
        np.testing.assert_array_equal(
            square_sum,
            np.square(img, dtype=np.uint64) * rejected.astype(np.uint64),
        )
        np.testing.assert_array_equal(count, rejected)


if __name__ == "__main__":
    unittest.main()
