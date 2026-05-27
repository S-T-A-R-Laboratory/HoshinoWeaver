"""Star point feature extraction and matching."""
import dataclasses

import cv2
import numpy as np
import numpy.linalg as la
from loguru import logger
from numpy.typing import NDArray
from scipy.spatial import distance as spd

from .geometry import make_cross_matrix


@dataclasses.dataclass
class MatchResult:
    pair_idx: NDArray[np.int32]
    ref_pts: NDArray[np.float64]
    src_pts: NDArray[np.float64]
    init_homography: NDArray[np.float64]


def extract_point_features(vec: NDArray[np.float64],
                           vol: NDArray[np.float64],
                           k: int = 15) -> NDArray[np.float64]:
    """Extract geometric features for each star point based on neighbor relationships.

    Args:
        vec: (n, 3) unit vectors of star points.
        vol: (n,) volume (area * intensity) of each star point.
        k: number of neighbors to use.

    Returns:
        (n, 120) feature matrix.
    """
    pts_num = len(vec)
    dist_mat = 1 - spd.cdist(vec, vec, "cosine")
    vec_dist_ind = np.argsort(-dist_mat)
    dist_mat = np.clip(dist_mat, -1, 1)

    dist_mat = np.arccos(dist_mat[np.array(range(pts_num))[:, np.newaxis],
                                  vec_dist_ind[:, :2 * k]])
    vol = vol[vec_dist_ind[:, :2 * k]]
    vol_ind = np.argsort(-vol * dist_mat)

    theta_feature = np.zeros((pts_num, k))
    rho_feature = np.zeros((pts_num, k))
    vol_feature = np.zeros((pts_num, k))

    for i in range(pts_num):
        v0 = vec[i]
        vs = vec[vec_dist_ind[i, vol_ind[i, :k]]]
        angles = np.inner(vs, make_cross_matrix(v0))
        angles = angles / la.norm(angles, axis=1)[:, np.newaxis]
        cr = np.inner(angles, make_cross_matrix(angles[0]))
        s = la.norm(cr, axis=1) * np.sign(np.inner(cr, v0))
        c = np.inner(angles, angles[0])
        theta_feature[i] = np.arctan2(s, c)
        rho_feature[i] = dist_mat[i, vol_ind[i, :k]]
        vol_feature[i] = vol[i, vol_ind[i, :k]]

    fx = np.arange(-np.pi, np.pi, 3 * np.pi / 180)
    features = np.zeros((pts_num, len(fx)))
    for i in range(k):
        sigma = 2.5 * np.exp(-rho_feature[:, i] * 100) + .04
        tmp = np.exp(-np.subtract.outer(theta_feature[:, i], fx)**2 / 2 /
                     sigma[:, np.newaxis]**2)
        tmp = tmp * (vol_feature[:, i] * rho_feature[:, i]**2 /
                     sigma)[:, np.newaxis]
        features += tmp

    features = features / np.sqrt(np.sum(features**2, axis=1)).reshape(
        (pts_num, 1))
    return features


def find_initial_match(features1: NDArray[np.float64],
                       features2: NDArray[np.float64],
                       pts1: NDArray[np.float64],
                       pts2: NDArray[np.float64],
                       vectors1: NDArray[np.float64] = None,
                       vectors2: NDArray[np.float64] = None,
                       alpha: float = 0.00,
                       apply_threshold_filter: bool = True,
                       theta_th: float = np.pi / 6,
                       dist_multiplier: float = 0.3) -> NDArray[np.int32]:
    """Find initial matches between two star images using feature similarity.

    Args:
        features1, features2: (n, d) feature matrices.
        pts1, pts2: (n, 2) pixel coordinates.
        vectors1, vectors2: (n, 3) unit vectors (needed if apply_threshold_filter=True).
        alpha: weight of Euclidean distance in matching.
        apply_threshold_filter: whether to apply angular/distance threshold.
        theta_th: angular distance threshold.
        dist_multiplier: distance multiplier for pixel threshold.

    Returns:
        (m, 2) array of matched index pairs.
    """
    measure_dist_mat = spd.cdist(features1, features2, "cosine")
    if alpha > 0:
        pts_stack = np.vstack((pts1, pts2))
        pts_mean = np.mean(pts_stack, axis=0)
        pts_min = np.min(pts_stack, axis=0)
        pts_max = np.max(pts_stack, axis=0)
        pts_dist_mat = spd.cdist((pts1 - pts_mean) / (pts_max - pts_min),
                                 (pts2 - pts_mean) / (pts_max - pts_min),
                                 "euclidean")
        dist_mat = measure_dist_mat * (1 - alpha) + pts_dist_mat * alpha
    else:
        dist_mat = measure_dist_mat

    num1, num2 = dist_mat.shape

    idx12 = np.argsort(dist_mat, axis=1)
    idx21 = np.argsort(dist_mat, axis=0)
    ind = idx21[0, idx12[:, 0]] == range(num1)

    d_th = min(np.percentile(dist_mat[range(num1), idx12[:, 0]], 30),
               np.percentile(dist_mat[idx21[0, :], range(num2)], 30))
    ind = np.logical_and(ind, dist_mat[range(num1), idx12[:, 0]] < d_th)

    pair_idx = np.stack((np.where(ind)[0], idx12[ind, 0]), axis=-1)
    logger.debug(f"Found {len(pair_idx)} initial pairs.")

    if apply_threshold_filter:
        if vectors1 is None or vectors2 is None:
            raise ValueError(
                "vectors1 and vectors2 required when apply_threshold_filter=True"
            )
        logger.debug("Applying threshold filter.")
        theta = np.arccos(
            np.clip(
                np.sum(vectors1[pair_idx[:, 0]] * vectors2[pair_idx[:, 1]],
                       axis=1), -1, 1))
        theta_th = min(np.percentile(theta, 75), theta_th)

        pts_dist = la.norm(pts1[pair_idx[:, 0]] - pts2[pair_idx[:, 1]], axis=1)
        dist_th = max(np.max(pts1), np.max(pts2)) * dist_multiplier
        pair_idx = pair_idx[np.logical_and(theta < theta_th,
                                           pts_dist < dist_th)]
        logger.debug(
            f"Found {len(pair_idx)} initial pairs after threshold filter.")
    return pair_idx


def fine_tune_transform(
        pts1: NDArray[np.float64], pts2: NDArray[np.float64],
        init_pair_idx: NDArray[np.int32]
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Refine matching using RANSAC homography.

    Returns:
        (homography_matrix, refined_pair_idx)
    """
    ind = []
    k = 1
    while len(ind) < 0.6 * min(len(pts1), len(pts2)) and k <= 10:
        if k >= 10:
            raise ValueError("Optimal alignment cannot be achieved.")
        rand_pts = np.random.rand(20, 2) * (
            np.amax(pts1, axis=0) - np.amin(pts1, axis=0)
        ) * np.array([1, 0.8]) + np.amin(pts1, axis=0)
        dist_mat = spd.cdist(rand_pts, pts1[init_pair_idx[:, 0]])
        tmp_ind = np.unique(np.argmin(dist_mat, axis=1))
        if len(tmp_ind) < 4:
            logger.warning(
                f"findHomography skipped (iteration {k}): only {len(tmp_ind)} unique point pairs after dedup")
            k += 1
            continue
        tf = cv2.findHomography(pts1[init_pair_idx[tmp_ind, 0]],
                                pts2[init_pair_idx[tmp_ind, 1]],
                                method=cv2.RANSAC,
                                ransacReprojThreshold=5)
        if tf[0] is None:
            logger.warning(f"findHomography returned None (iteration {k}), skipping")
            k += 1
            continue
        try:
            pts12 = cv2.perspectiveTransform(
                np.array([[p] for p in pts1], dtype="float32"), tf[0])[:, 0, :]
        except Exception as e:
            logger.warning(f"RANSAC homography failed (iteration {k}): {e}")
            k += 1
            continue
        dist_mat = spd.cdist(pts12, pts2)
        num1, num2 = dist_mat.shape

        idx12 = np.argsort(dist_mat, axis=1)
        tmp_ind = np.argwhere(
            np.array([dist_mat[i, idx12[i, 0]] for i in range(num1)]) < 5)
        if len(tmp_ind) > len(ind):
            ind = tmp_ind
        logger.debug(
            f"len(ind) = {len(ind)}, len(feature) = {min(len(pts1), len(pts2))}"
        )
        k += 1

    pair_idx = np.hstack((ind, idx12[ind, 0]))

    if len(pair_idx) < 4:
        raise ValueError(
            f"findHomography requires at least 4 point pairs, got {len(pair_idx)}")
    tf = cv2.findHomography(pts1[pair_idx[:, 0]],
                            pts2[pair_idx[:, 1]],
                            method=cv2.RANSAC,
                            ransacReprojThreshold=5)
    if tf[0] is None:
        raise ValueError("Final findHomography returned None")
    return tf[0], pair_idx
