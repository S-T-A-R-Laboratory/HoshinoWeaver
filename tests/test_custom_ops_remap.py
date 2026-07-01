import unittest
from unittest import mock

import numpy as np

from hoshicore._custom_op import build_info, camera_model_remap
import hoshicore._custom_op.ops.remap as remap_ops
import hoshicore.component.norma.types as norma_types
from hoshicore.component.norma.types import CameraModel, Distortion, Intrinsics


class TestCameraModelRemapCustomOp(unittest.TestCase):
    def tearDown(self) -> None:
        remap_ops._load_compiled_module_result.cache_clear()
        remap_ops._select_camera_model_remap_backend.cache_clear()

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

    def test_camera_model_remap_distortion_matches_numpy(self) -> None:
        image = np.linspace(0.0, 1.0, num=7 * 8 * 3, dtype=np.float32).reshape(7, 8, 3)
        yaw = np.deg2rad(0.8)
        rotation = np.array([
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        src_dist = np.array([0.015, -0.002, 0.001, -0.0005, 0.0001],
                            dtype=np.float32)
        dst_dist = np.array([-0.012, 0.0015, -0.0007, 0.0004, -0.00008],
                            dtype=np.float32)

        got = camera_model_remap(
            image=image,
            out_height=6,
            out_width=7,
            fx_src=12.0,
            fy_src=11.5,
            cx_src=3.5,
            cy_src=3.0,
            fx_dst=10.5,
            fy_dst=10.0,
            cx_dst=3.0,
            cy_dst=2.5,
            rotation_dst_to_src=rotation,
            src_dist_coeffs=src_dist,
            dst_dist_coeffs=dst_dist,
        )
        expected = remap_ops.camera_model_remap_numpy(
            image=image,
            out_height=6,
            out_width=7,
            fx_src=12.0,
            fy_src=11.5,
            cx_src=3.5,
            cy_src=3.0,
            fx_dst=10.5,
            fy_dst=10.0,
            cx_dst=3.0,
            cy_dst=2.5,
            rotation_dst_to_src=rotation,
            src_dist_coeffs=src_dist,
            dst_dist_coeffs=dst_dist,
        )

        np.testing.assert_allclose(got, expected, rtol=2e-4, atol=2e-4)

    def test_camera_model_remap_distortion_compiled_uint16_matches_numpy(self) -> None:
        if not build_info().get("cuda"):
            self.skipTest("CUDA remap backend is not built")

        image = (np.arange(7 * 8 * 2, dtype=np.uint16).reshape(7, 8, 2) * 17)
        rotation = np.array([
            [0.9999, -0.0100, 0.0020],
            [0.0100, 0.9999, -0.0015],
            [-0.0020, 0.0015, 1.0000],
        ], dtype=np.float32)
        src_dist = np.array([0.012, -0.0015, 0.0008, -0.0004, 0.0001],
                            dtype=np.float32)
        dst_dist = np.array([-0.010, 0.0012, -0.0006, 0.0003, -0.00008],
                            dtype=np.float32)

        kwargs = {
            "image": image,
            "out_height": 6,
            "out_width": 7,
            "fx_src": 13.0,
            "fy_src": 12.5,
            "cx_src": 3.5,
            "cy_src": 3.0,
            "fx_dst": 11.5,
            "fy_dst": 11.0,
            "cx_dst": 3.0,
            "cy_dst": 2.5,
            "rotation_dst_to_src": rotation,
            "src_dist_coeffs": src_dist,
            "dst_dist_coeffs": dst_dist,
        }
        expected = remap_ops.camera_model_remap_numpy(**kwargs)
        try:
            got = remap_ops.camera_model_remap_compiled(**kwargs)
        except RuntimeError as exc:
            if remap_ops._is_cuda_runtime_unavailable_error(exc):
                self.skipTest(f"CUDA runtime unavailable: {exc}")
            raise

        np.testing.assert_allclose(got, expected, rtol=0, atol=1)

    def test_camera_model_remap_compiled_zero_distortion_matches_numpy(self) -> None:
        if not build_info().get("cuda"):
            self.skipTest("CUDA remap backend is not built")

        image = np.linspace(0.0, 1.0, num=6 * 7 * 3, dtype=np.float32).reshape(6, 7, 3)
        rotation = np.array([
            [0.9999, -0.0120, 0.0020],
            [0.0120, 0.9999, -0.0010],
            [-0.0020, 0.0010, 1.0000],
        ], dtype=np.float32)
        kwargs = {
            "image": image,
            "out_height": 5,
            "out_width": 6,
            "fx_src": 11.0,
            "fy_src": 10.5,
            "cx_src": 3.0,
            "cy_src": 2.5,
            "fx_dst": 9.5,
            "fy_dst": 9.0,
            "cx_dst": 2.5,
            "cy_dst": 2.0,
            "rotation_dst_to_src": rotation,
        }
        expected = remap_ops.camera_model_remap_numpy(**kwargs)
        try:
            got = remap_ops.camera_model_remap_compiled(**kwargs)
        except RuntimeError as exc:
            if remap_ops._is_cuda_runtime_unavailable_error(exc):
                self.skipTest(f"CUDA runtime unavailable: {exc}")
            raise

        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)

    def test_camera_model_remap_distortion_compiled_float32_rgb_matches_numpy(self) -> None:
        if not build_info().get("cuda"):
            self.skipTest("CUDA remap backend is not built")

        image = np.linspace(0.0, 1.0, num=9 * 10 * 3, dtype=np.float32).reshape(9, 10, 3)
        rotation = np.array([
            [0.9998, -0.0180, 0.0030],
            [0.0180, 0.9998, -0.0020],
            [-0.0030, 0.0020, 1.0000],
        ], dtype=np.float32)
        src_dist = np.array([0.010, -0.0012, 0.0006, -0.0003, 0.00008],
                            dtype=np.float32)
        dst_dist = np.array([-0.011, 0.0014, -0.0007, 0.0004, -0.00007],
                            dtype=np.float32)
        kwargs = {
            "image": image,
            "out_height": 8,
            "out_width": 9,
            "fx_src": 15.0,
            "fy_src": 14.5,
            "cx_src": 4.5,
            "cy_src": 4.0,
            "fx_dst": 13.5,
            "fy_dst": 13.0,
            "cx_dst": 4.0,
            "cy_dst": 3.5,
            "rotation_dst_to_src": rotation,
            "src_dist_coeffs": src_dist,
            "dst_dist_coeffs": dst_dist,
        }
        expected = remap_ops.camera_model_remap_numpy(**kwargs)
        try:
            got = remap_ops.camera_model_remap_compiled(**kwargs)
        except RuntimeError as exc:
            if remap_ops._is_cuda_runtime_unavailable_error(exc):
                self.skipTest(f"CUDA runtime unavailable: {exc}")
            raise

        np.testing.assert_allclose(got, expected, rtol=3e-4, atol=3e-4)

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

    def test_camera_model_remap_distortion_can_force_numpy_fallback(self) -> None:
        image = np.arange(5 * 6 * 3, dtype=np.uint8).reshape(5, 6, 3)
        rotation = np.eye(3, dtype=np.float32)
        src_dist = np.array([0.012, -0.001, 0.0005, -0.0003, 0.0],
                            dtype=np.float32)
        dst_dist = np.array([-0.01, 0.001, -0.0004, 0.0002, 0.0],
                            dtype=np.float32)
        kwargs = {
            "image": image,
            "out_height": 5,
            "out_width": 6,
            "fx_src": 8.0,
            "fy_src": 8.0,
            "cx_src": 3.0,
            "cy_src": 2.5,
            "fx_dst": 8.0,
            "fy_dst": 8.0,
            "cx_dst": 3.0,
            "cy_dst": 2.5,
            "rotation_dst_to_src": rotation,
            "src_dist_coeffs": src_dist,
            "dst_dist_coeffs": dst_dist,
        }

        with mock.patch.dict("os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False):
            with mock.patch.object(remap_ops, "_load_compiled_module_result", return_value=(None, "mock error")):
                remap_ops._select_camera_model_remap_backend.cache_clear()
                got = camera_model_remap(**kwargs)

        expected = remap_ops.camera_model_remap_numpy(**kwargs)
        np.testing.assert_array_equal(got, expected)

    def test_camera_model_remap_float64_uses_numpy_fallback(self) -> None:
        image = np.arange(18, dtype=np.float64).reshape(3, 3, 2)
        rotation = np.eye(3, dtype=np.float32)
        kwargs = {
            "image": image,
            "out_height": 2,
            "out_width": 2,
            "fx_src": 6.0,
            "fy_src": 6.0,
            "cx_src": 1.0,
            "cy_src": 1.0,
            "fx_dst": 5.0,
            "fy_dst": 5.0,
            "cx_dst": 1.0,
            "cy_dst": 1.0,
            "rotation_dst_to_src": rotation,
        }
        compiled = mock.Mock(side_effect=AssertionError("compiled should not run"))

        with mock.patch.object(
                remap_ops,
                "_select_camera_model_remap_backend",
                return_value=("compiled", compiled)):
            got = camera_model_remap(**kwargs)

        expected = remap_ops.camera_model_remap_numpy(**kwargs)
        compiled.assert_not_called()
        np.testing.assert_array_equal(got, expected)

    def test_camera_model_remap_rejects_nonfinite_distortion(self) -> None:
        image = np.arange(18, dtype=np.uint8).reshape(3, 3, 2)
        rotation = np.eye(3, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "finite"):
            camera_model_remap(
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
                src_dist_coeffs=np.array([np.nan, 0.0, 0.0, 0.0, 0.0],
                                         dtype=np.float32),
            )

    def test_camera_model_remap_rejects_invalid_camera_params(self) -> None:
        image = np.arange(18, dtype=np.uint8).reshape(3, 3, 2)
        rotation = np.eye(3, dtype=np.float32)
        kwargs = {
            "image": image,
            "out_height": 2,
            "out_width": 2,
            "fx_src": 6.0,
            "fy_src": 6.0,
            "cx_src": 1.0,
            "cy_src": 1.0,
            "fx_dst": 5.0,
            "fy_dst": 5.0,
            "cx_dst": 1.0,
            "cy_dst": 1.0,
            "rotation_dst_to_src": rotation,
        }
        with self.assertRaisesRegex(ValueError, "fx_src"):
            camera_model_remap(**{**kwargs, "fx_src": 0.0})
        bad_rotation = rotation.copy()
        bad_rotation[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "rotation_dst_to_src"):
            camera_model_remap(**{**kwargs, "rotation_dst_to_src": bad_rotation})

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

    def test_camera_model_remap_falls_back_for_unsupported_cuda_device(self) -> None:
        image = np.arange(18, dtype=np.uint8).reshape(3, 3, 2)
        rotation = np.eye(3, dtype=np.float32)
        kwargs = {
            "image": image,
            "out_height": 2,
            "out_width": 2,
            "fx_src": 6.0,
            "fy_src": 6.0,
            "cx_src": 1.0,
            "cy_src": 1.0,
            "fx_dst": 5.0,
            "fy_dst": 5.0,
            "cx_dst": 1.0,
            "cy_dst": 1.0,
            "rotation_dst_to_src": rotation,
        }
        expected = remap_ops.camera_model_remap_numpy(**kwargs)

        with mock.patch.object(
                remap_ops,
                "_select_camera_model_remap_backend",
                return_value=("compiled", mock.Mock(side_effect=RuntimeError(
                    "camera_model_remap kernel launch: no kernel image is available for execution on the device")))):
            got = camera_model_remap(**kwargs)

        np.testing.assert_array_equal(got, expected)

    def test_camera_model_remap_falls_back_for_cuda_allocation_failure(self) -> None:
        image = np.arange(18, dtype=np.uint8).reshape(3, 3, 2)
        rotation = np.eye(3, dtype=np.float32)
        kwargs = {
            "image": image,
            "out_height": 2,
            "out_width": 2,
            "fx_src": 6.0,
            "fy_src": 6.0,
            "cx_src": 1.0,
            "cy_src": 1.0,
            "fx_dst": 5.0,
            "fy_dst": 5.0,
            "cx_dst": 1.0,
            "cy_dst": 1.0,
            "rotation_dst_to_src": rotation,
        }
        expected = remap_ops.camera_model_remap_numpy(**kwargs)

        with mock.patch.object(
                remap_ops,
                "_select_camera_model_remap_backend",
                return_value=("compiled", mock.Mock(side_effect=RuntimeError(
                    "camera_model_remap cudaMallocHost(image): out of memory")))):
            got = camera_model_remap(**kwargs)

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
        np.testing.assert_allclose(got, expected, rtol=0, atol=1)

    def test_project_image_from_camera_routes_distortion_through_custom_fused(self) -> None:
        img = np.arange(5 * 6 * 3, dtype=np.uint8).reshape(5, 6, 3)
        intrinsics = Intrinsics(
            focal_length_mm=8.0,
            sensor_width_mm=8.0,
            sensor_height_mm=8.0,
            image_width_px=6,
            image_height_px=5,
        )
        src_camera = CameraModel(
            intrinsics=intrinsics,
            distortion=Distortion(k1=0.012, k2=-0.001, p1=0.0005, p2=-0.0003),
        )
        dst_camera = CameraModel(
            intrinsics=intrinsics,
            distortion=Distortion(k1=-0.01, k2=0.001, p1=-0.0004, p2=0.0002),
        )

        expected = remap_ops.camera_model_remap_numpy(
            image=img,
            out_height=5,
            out_width=6,
            fx_src=float(src_camera.K[0, 0]),
            fy_src=float(src_camera.K[1, 1]),
            cx_src=float(src_camera.K[0, 2]),
            cy_src=float(src_camera.K[1, 2]),
            fx_dst=float(dst_camera.K[0, 0]),
            fy_dst=float(dst_camera.K[1, 1]),
            cx_dst=float(dst_camera.K[0, 2]),
            cy_dst=float(dst_camera.K[1, 2]),
            rotation_dst_to_src=np.eye(3, dtype=np.float32),
            src_dist_coeffs=src_camera.dist_coeffs,
            dst_dist_coeffs=dst_camera.dist_coeffs,
        )

        with mock.patch.object(
                norma_types,
                "custom_camera_model_remap",
                wraps=norma_types.custom_camera_model_remap) as patched_custom:
            got = dst_camera.project_image_from_camera(src_camera, img, (6, 5))

        patched_custom.assert_called_once()
        np.testing.assert_allclose(got, expected, rtol=0, atol=1)

    def test_project_image_from_camera_distortion_matches_generic_path(self) -> None:
        img = np.arange(5 * 6 * 3, dtype=np.uint8).reshape(5, 6, 3)
        intrinsics = Intrinsics(
            focal_length_mm=8.0,
            sensor_width_mm=8.0,
            sensor_height_mm=8.0,
            image_width_px=6,
            image_height_px=5,
        )
        src_camera = CameraModel(
            intrinsics=intrinsics,
            distortion=Distortion(k1=0.012, k2=-0.001, p1=0.0005, p2=-0.0003),
        )
        dst_camera = CameraModel(
            intrinsics=intrinsics,
            distortion=Distortion(k1=-0.01, k2=0.001, p1=-0.0004, p2=0.0002),
        )

        got = dst_camera.project_image_from_camera(src_camera, img, (6, 5))
        expected = dst_camera.project_image_from_camera(
            src_camera,
            img,
            (6, 5),
            roi=(0, 0, 6, 5),
        )

        np.testing.assert_allclose(got, expected, rtol=0, atol=1)


