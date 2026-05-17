"""Pure functions for camera projection and unprojection."""
from typing import Optional

import cv2
import numpy as np
from numpy.typing import NDArray


def make_intrinsic_matrix(focal_mm: float, sensor_w_mm: float,
                          sensor_h_mm: float, img_w_px: int,
                          img_h_px: int) -> NDArray[np.float64]:
    fx = focal_mm * img_w_px / sensor_w_mm
    fy = focal_mm * img_h_px / sensor_h_mm
    cx = img_w_px / 2.0
    cy = img_h_px / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def undistort_points(pts: NDArray[np.float64], K: NDArray[np.float64],
                     dist_coeffs: Optional[NDArray[np.float64]]
                     ) -> NDArray[np.float64]:
    """Remove distortion from pixel coordinates.
    pts: (n, 2), K: 3x3, dist_coeffs: 5-element or None.
    Returns (n, 2) undistorted pixel coordinates.
    """
    if dist_coeffs is not None and np.any(dist_coeffs != 0):
        upts = cv2.undistortPoints(pts[:, None, :].astype(np.float64), K,
                                   dist_coeffs, P=K)
        return upts[:, 0, :]
    return pts


def unproject_pixels(pts: NDArray[np.float64], K: NDArray[np.float64],
                     dist_coeffs: Optional[NDArray[np.float64]] = None,
                     R: Optional[NDArray[np.float64]] = None
                     ) -> NDArray[np.float64]:
    """Pixel coordinates to unit direction vectors in world frame.
    pts: (n, 2), K: 3x3, dist_coeffs: 5-element or None, R: 3x3 or None.
    Returns (n, 3) unit vectors.
    """
    upts = undistort_points(pts, K, dist_coeffs)
    xyz_h = np.concatenate([upts, np.ones((upts.shape[0], 1))], axis=1)
    vec = (np.linalg.inv(K) @ xyz_h.T).T
    vec = vec / np.linalg.norm(vec, axis=1)[:, None]
    if R is not None:
        vec = (R.T @ vec.T).T
    return vec


def project_vectors(v: NDArray[np.float64], K: NDArray[np.float64],
                    dist_coeffs: Optional[NDArray[np.float64]] = None,
                    R: Optional[NDArray[np.float64]] = None
                    ) -> NDArray[np.float64]:
    """Unit direction vectors in world frame to pixel coordinates.
    v: (n, 3), K: 3x3, dist_coeffs: 5-element or None, R: 3x3 or None.
    Returns (n, 2). NaN for vectors behind the camera.
    """
    assert v.shape[1] == 3 and len(v.shape) == 2
    n = v.shape[0]

    rotated = (R @ v.T).T if R is not None else v
    valid = np.where(rotated[:, 2] > 0)[0]
    result = np.full((n, 2), np.nan, dtype=np.float64)

    if len(valid) == 0:
        return result

    rv = rotated[valid]

    if dist_coeffs is not None and np.any(dist_coeffs != 0):
        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.zeros((3, 1), dtype=np.float64)
        image_points, _ = cv2.projectPoints(rv[None, ...], rvec, tvec, K,
                                            dist_coeffs)
        result[valid, :] = image_points[:, 0, :]
    else:
        normalized = (rv * (1 / rv[:, 2][:, None])).T
        image_points = (K @ normalized).T[:, :2]
        result[valid, :] = image_points

    return result


def distort_image(img_undist: NDArray[np.uint8], K: NDArray[np.float64],
                  dist_coeffs: NDArray[np.float64],
                  output_size: tuple[int, int]) -> NDArray[np.uint8]:
    """Apply distortion to an undistorted image via remap."""
    target_width, target_height = output_size

    ys, xs = np.meshgrid(np.arange(target_height),
                         np.arange(target_width),
                         indexing='ij')
    pixels_dist = np.stack([xs, ys], axis=-1).reshape(-1, 2).astype(np.float64)

    pixels_norm_undist = cv2.undistortPoints(pixels_dist.reshape(-1, 1, 2), K,
                                            dist_coeffs,
                                            P=None).reshape(-1, 2)

    pixels_h = np.hstack(
        [pixels_norm_undist,
         np.ones((len(pixels_norm_undist), 1))])
    pixels_undist = (K @ pixels_h.T).T[:, :2]

    map_xy = pixels_undist.reshape(target_height, target_width,
                                   2).astype(np.float32)

    return cv2.remap(img_undist,
                     map_xy[..., 0],
                     map_xy[..., 1],
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)
