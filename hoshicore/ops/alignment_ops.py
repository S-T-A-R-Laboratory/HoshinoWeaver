"""
对齐算子：星点对齐等帧间配准操作。

StarAlignmentOp 支持两条显式对齐路径：
  1. 2D 单应性（homography）：zero-distortion CameraModel + warpPerspective（无需 EXIF）
  2. 相机模型优化（camera_model）：Intrinsics + optimize_alignment + remap

相机参数按 EXIF、手动焦距和 fallback 顺序构造。
对齐失败的帧被丢弃，输出为变长序列（sentinel 驱动）。
"""
from typing import Any

import numpy as np
from loguru import logger

from ..component.norma.frame_align import (AlignmentError,
                                           CameraInitializationPolicy,
                                           DEFAULT_MATCHING_PATH,
                                           MATCHING_PATH_MEDIAN,
                                           align_frame_camera_model,
                                           align_frame_homography,
                                           build_camera_candidate)
from ..component.norma.geometry_view import GeometryView, StarDetectionCache
from ..component.norma.intrinsics_from_exif import lens_type_from_exif
from ..component.queue import StreamExhausted
from ..engine.registry import register_op
from ..component.data_container import FloatImage
from .base import FilterBaseOp


@register_op()
class StarAlignmentOp(FilterBaseOp):
    """星点对齐：将序列帧对齐到参考帧。

    支持两条路径，显式选择：
    - homography: 仅优化旋转+焦距，无需相机信息
    - distortion: 联合优化旋转+焦距+畸变，需要 EXIF 提供内参

    对齐失败的帧被丢弃（不输出），输出为变长序列。
    """

    EXECUTOR = "cpu"
    INPUTS: dict[str, Any] = {
        "data": {"type": "sequence"},
        "exifs": {"type": "sequence", "required": False},
    }
    CONFIGS: dict[str, Any] = {
        "reference":        {"type": "image",  "required": True},
        "reference_exif":   {"type": "exif",   "default": None},
        "method":           {"type": "str",    "default": "distortion"},
        "camera_setup_mode": {"type": "str",   "default": None},
        "same_camera":      {"type": "bool",   "default": True},
        "matching_path":    {"type": "str",    "default": DEFAULT_MATCHING_PATH},
        "distortion":       {"type": "list",   "default": None},
        "lens_type":        {"type": "str",    "default": None},
        "ref_lens_type":    {"type": "str",    "default": None},
        "src_lens_type":    {"type": "str",    "default": None},
        "focal_length_mm":  {"type": "float",  "default": None},
        "crop_factor":      {"type": "float",  "default": 1.0},
        "fallback_focal_equiv_mm": {"type": "float", "default": 20.0},
        "bootstrap_scales": {"type": "list", "default": None},
        "remap_map_scale": {"type": "float", "default": 0.5},
        "guided_refine": {"type": "bool", "default": False},
        "optimize_focal": {"type": "bool", "default": None},
        "optimize_distortion": {"type": "bool", "default": None},
        "optimize_principal_point": {"type": "bool", "default": None},
        "allow_large_principal_point_offset": {"type": "bool", "default": False},
    }
    OUTPUTS: dict[str, Any] = {
        "result": {"type": "sequence"},
        "aligned_exifs": {"type": "sequence"},
    }

    @classmethod
    def estimate_resources(cls, configs, frame_bytes, n_frames,
                           dtype_bytes=None):
        _ = dtype_bytes
        # 持有 1 张参考帧 + 1 张当前帧的对齐输出
        # TODO: 对齐本身资源未计算
        return (2 * frame_bytes, 0)

    async def _async_filter(self, configs: dict[str, Any]) -> None:
        method = configs.get('method', 'distortion')
        camera_setup_mode = configs.get('camera_setup_mode')
        same_camera = configs.get('same_camera', True)
        matching_path = configs.get('matching_path', DEFAULT_MATCHING_PATH)
        init_distortion = configs.get('distortion')
        shared_lens_type = configs.get('lens_type')
        configured_ref_lens = configs.get('ref_lens_type')
        configured_src_lens = configs.get('src_lens_type')
        focal_length_mm = configs.get('focal_length_mm')
        crop_factor = configs.get('crop_factor') or 1.0

        # camera_setup_mode is the user-facing controller.  Keep
        # same_camera and the individual lens fields as a backwards-compatible
        # low-level API for callers which do not provide the mode.
        if camera_setup_mode is not None:
            if camera_setup_mode == "auto":
                same_camera = True
                shared_lens_type = None
                configured_ref_lens = None
                configured_src_lens = None
                focal_length_mm = None
                crop_factor = 1.0
            elif camera_setup_mode == "manual":
                same_camera = True
                configured_ref_lens = None
                configured_src_lens = None
            elif camera_setup_mode == "separate":
                same_camera = False
                shared_lens_type = None
            else:
                raise ValueError(
                    f"Unsupported camera_setup_mode {camera_setup_mode!r}; "
                    "expected 'auto', 'manual', or 'separate'")

        focal_equiv_mm = focal_length_mm * crop_factor if focal_length_mm else None
        fallback_focal_equiv_mm: float = configs.get('fallback_focal_equiv_mm', 20.0)
        bootstrap_scales = configs.get('bootstrap_scales')
        if bootstrap_scales is None:
            bootstrap_scales_tuple = (0.7, 1.0, 1.3)
        else:
            bootstrap_scales_tuple = tuple(float(x) for x in bootstrap_scales)
        remap_map_scale = float(configs.get('remap_map_scale', 0.5))
        guided_refine = bool(configs.get('guided_refine', False))
        policy_kwargs = dict(
            fallback_focal_equiv_mm=float(fallback_focal_equiv_mm),
            optimize_focal=configs.get('optimize_focal'),
            optimize_distortion=configs.get('optimize_distortion'),
            optimize_principal_point=configs.get('optimize_principal_point'),
            allow_large_principal_point_offset=bool(
                configs.get('allow_large_principal_point_offset', False)),
        )

        exifs_active = self.inputs['exifs'].active

        # Initialize the mandatory reference before consuming the frame
        # stream. This removes the previous first-frame/fallback mode.
        reference = configs['reference']
        ref_arr = reference.data if isinstance(reference, FloatImage) else reference

        ref_exif_obj = configs.get('reference_exif')
        ref_exif_tags = (ref_exif_obj.exif
                         if ref_exif_obj is not None else None)
        ref_lens_type = (configured_ref_lens if configured_ref_lens is not None
                         else shared_lens_type)
        if ref_lens_type is None:
            ref_lens_type = lens_type_from_exif(ref_exif_tags)
        ref_policy = CameraInitializationPolicy(
            lens_type=ref_lens_type, **policy_kwargs)

        use_manual_intrinsics = camera_setup_mode == "manual"
        ref_candidate = build_camera_candidate(
            None if use_manual_intrinsics else ref_exif_tags,
            ref_arr.shape, method, init_distortion,
            focal_equiv_mm, ref_policy)
        ref_camera = ref_candidate.camera
        ref_detection = await self._run_cpu(
            StarDetectionCache.from_image, ref_arr)
        if method == "homography" or matching_path == MATCHING_PATH_MEDIAN:
            ref_stars = await self._run_cpu(
                lambda: ref_detection.median_stars)
        else:
            ref_stars = await self._run_cpu(
                lambda: ref_detection.pywt_stars)
        ref_geo = GeometryView(ref_stars, ref_camera)
        ref_refine_geo = None
        if (method != "homography" and guided_refine
                and matching_path != MATCHING_PATH_MEDIAN):
            ref_refine_stars = await self._run_cpu(
                lambda: ref_detection.median_stars)
            ref_refine_geo = GeometryView(ref_refine_stars, ref_camera)
        path_name = ("fixed-camera homography fast path"
                     if method == "homography" else "camera model path")
        logger.info(
            f"{self.name}: {path_name} enabled for mandatory reference "
            f"(focal={ref_camera.intrinsics.focal_length_mm:.1f}mm, "
            f"lens_type={ref_lens_type or 'perspective'}, "
            f"source={ref_candidate.init_source})")
        aligned_count = 0
        skipped_count = 0

        for i in self._input_range():
            data = self._async_convert_inputs()
            try:
                frame = await data['data']
            except StreamExhausted:
                break

            # 消费 EXIF 并拆包为 dict
            exif_tags = None
            exif_obj = None
            if exifs_active:
                try:
                    exif_obj = await data['exifs']
                    exif_tags = exif_obj.exif if exif_obj is not None else None
                except StreamExhausted:
                    pass

            frame_arr = frame.data if isinstance(frame, FloatImage) else frame

            # 对齐
            try:
                src_lens_type = (configured_src_lens
                                 if configured_src_lens is not None else
                                 shared_lens_type)
                if src_lens_type is None:
                    src_lens_type = lens_type_from_exif(exif_tags)
                src_policy = CameraInitializationPolicy(
                    lens_type=src_lens_type, **policy_kwargs)
                src_candidate = build_camera_candidate(
                    None if use_manual_intrinsics else exif_tags,
                    frame_arr.shape, method, init_distortion,
                    focal_equiv_mm, src_policy)
                src_camera = src_candidate.camera

                if ref_camera and src_camera and method != "homography":
                    aligned_arr = await self._run_cpu(
                        align_frame_camera_model,
                        frame_arr, ref_geo, ref_arr,
                        ref_candidate, src_candidate, same_camera,
                         bootstrap_scales_tuple, remap_map_scale,
                         guided_refine=guided_refine,
                         matching_path=matching_path,
                         ref_refine_geo=ref_refine_geo)
                else:
                    aligned_arr = await self._run_cpu(
                        align_frame_homography,
                        frame_arr,
                        ref_geo,
                        ref_arr,
                        float(fallback_focal_equiv_mm),
                        src_camera,
                    )

                aligned = (FloatImage(data=aligned_arr, dtype=frame.dtype)
                           if isinstance(frame, FloatImage) else aligned_arr)
                await self._broadcast_outputs(
                    {"result": aligned, "aligned_exifs": exif_obj})
                aligned_count += 1
            except AlignmentError as e:
                skipped_count += 1
                logger.warning(
                    f"{self.name}: frame {i} alignment failed ({e}), skipping")
            except Exception as e:
                import traceback
                traceback_str = traceback.format_exc()
                logger.error(traceback_str)
                raise e

            self.tracker.update(self.name)

        logger.info(
            f"{self.name}: aligned {aligned_count} frames, "
            f"skipped {skipped_count}")
