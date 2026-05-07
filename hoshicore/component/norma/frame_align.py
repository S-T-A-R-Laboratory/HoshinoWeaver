"""单帧对齐：封装两条路径的纯函数 API。

所有函数接收 ndarray / GeometryView / CameraModel，返回 ndarray。
不依赖 DAG 框架或 ExifData 类型。
"""
from typing import Optional

import cv2
import numpy as np
from loguru import logger

from .alignment import match_star_pairs, optimize_alignment, warp_image_by_remap
from .cache import GeometryView, StarDetectionCache
from .intrinsics_from_exif import intrinsics_from_exif
from .types import CameraModel, Distortion, Intrinsics


class AlignmentError(Exception):
    """对齐失败异常。"""
    pass


def to_gray_f64(arr: np.ndarray) -> np.ndarray:
    """将图像数组转换为 [0, 1] 范围的 float64 灰度图。"""
    if arr.ndim == 3:
        gray = cv2.cvtColor(arr.astype(np.float32), cv2.COLOR_RGB2GRAY).astype(np.float64)
    else:
        gray = arr.astype(np.float64)

    if np.issubdtype(arr.dtype, np.integer):
        gray /= np.iinfo(arr.dtype).max
    else:
        max_val = gray.max()
        if max_val > 1.0:
            gray /= max_val

    return gray


def make_geometry(arr: np.ndarray) -> GeometryView:
    """从原始图像数组构建 FlatCameraModel GeometryView。"""
    gray = to_gray_f64(arr)
    cache = StarDetectionCache(gray)
    return GeometryView.from_flat_projection(cache)


def try_build_camera(
    exif_tags: Optional[dict[str, str]],
    img_shape: tuple,
    method: str,
    init_distortion: Optional[list] = None,
) -> Optional[CameraModel]:
    """尝试从 EXIF 标签字典构建 CameraModel。

    返回 None 时调用方应降级为 homography 路径。
    """
    if method == "homography":
        return None
    if exif_tags is None:
        return None

    h, w = img_shape[:2]
    intrinsics = intrinsics_from_exif(exif_tags, w, h)
    if intrinsics is None:
        return None

    dist = Distortion.from_cv2(init_distortion) if init_distortion else Distortion()
    return CameraModel(intrinsics=intrinsics, distortion=dist)


def _check_star_count(ref_geo: GeometryView, src_geo: GeometryView,
                      min_stars: int = 20) -> None:
    """检查星点数量是否满足对齐要求。"""
    if len(ref_geo.positions) < min_stars or len(src_geo.positions) < min_stars:
        raise AlignmentError(
            f"Insufficient stars: ref={len(ref_geo.positions)}, "
            f"src={len(src_geo.positions)} (need >={min_stars})")


def _match_stars(ref_geo: GeometryView, src_geo: GeometryView):
    """执行星点匹配，失败时抛出 AlignmentError。"""
    try:
        return match_star_pairs(
            ref_geo.unit_vectors, src_geo.unit_vectors,
            ref_geo.volumes, src_geo.volumes,
            ref_geo.positions, src_geo.positions,
        )
    except Exception as e:
        raise AlignmentError(f"Star matching failed: {e}") from e


def align_frame_homography(
    frame: np.ndarray, ref_geo: GeometryView, reference: np.ndarray
) -> np.ndarray:
    """2D 单应性路径：match → RANSAC homography → warpPerspective。

    Args:
        frame: 待对齐帧。
        ref_geo: 参考帧的 GeometryView（预计算，可复用）。
        reference: 参考帧原始数组（用于获取输出尺寸）。

    Returns:
        对齐后的图像数组。

    Raises:
        AlignmentError: 星点不足或匹配失败。
    """
    src_geo = make_geometry(frame)
    _check_star_count(ref_geo, src_geo)
    match = _match_stars(ref_geo, src_geo)

    h, w = reference.shape[:2]
    H = np.linalg.inv(match.init_homography)
    return cv2.warpPerspective(frame, H, (w, h))


def align_frame_camera_model(
    frame: np.ndarray, ref_geo: GeometryView, reference: np.ndarray,
    ref_camera: CameraModel, src_camera: CameraModel,
    same_camera: bool = True,
) -> np.ndarray:
    """相机模型路径：match → optimize_alignment → warp_image_by_remap。

    Args:
        frame: 待对齐帧。
        ref_geo: 参考帧的 GeometryView。
        reference: 参考帧原始数组。
        ref_camera: 参考帧相机模型。
        src_camera: 当前帧相机模型。
        same_camera: 是否共享内参（同机身序列）。

    Returns:
        对齐后的图像数组。

    Raises:
        AlignmentError: 星点不足、匹配失败或优化失败。
    """
    src_geo = make_geometry(frame)
    _check_star_count(ref_geo, src_geo)
    match = _match_stars(ref_geo, src_geo)

    try:
        result = optimize_alignment(
            match, ref_camera, src_camera, same_camera=same_camera)
    except Exception as e:
        raise AlignmentError(f"Optimization failed: {e}") from e

    h, w = reference.shape[:2]
    return warp_image_by_remap(
        frame, result.camera2_refined, result.camera1_refined, (w, h))
