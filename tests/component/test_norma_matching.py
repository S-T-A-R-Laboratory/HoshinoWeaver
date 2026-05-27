"""Tests for norma matching diagnostics and refinement."""
import numpy as np
import pytest

from hoshicore.component.norma.matching import (find_initial_match,
                                                fine_tune_transform,
                                                validate_homography)


def _make_points():
    pts1 = np.array([
        [10.0, 10.0],
        [90.0, 10.0],
        [90.0, 90.0],
        [10.0, 90.0],
        [50.0, 30.0],
        [30.0, 60.0],
    ], dtype=np.float64)
    pair_idx = np.column_stack((
        np.arange(len(pts1), dtype=np.int32),
        np.arange(len(pts1), dtype=np.int32),
    ))
    return pts1, pair_idx


def _make_features_with_controlled_diag_distance(diag_distances: list[float]) -> tuple[np.ndarray, np.ndarray]:
    n = len(diag_distances)
    features1 = np.zeros((n, 2 * n), dtype=np.float64)
    features2 = np.zeros((n, 2 * n), dtype=np.float64)
    for i, dist in enumerate(diag_distances):
        cos_theta = 1.0 - dist
        sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta**2))
        features1[i, i] = 1.0
        features2[i, i] = cos_theta
        features2[i, n + i] = sin_theta
    return features1, features2


def test_validate_homography_accepts_small_translation():
    pts1, pair_idx = _make_points()
    H = np.array([
        [1.0, 0.0, 2.5],
        [0.0, 1.0, -1.5],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    pts2 = pts1 + np.array([2.5, -1.5], dtype=np.float64)

    diagnostics = validate_homography(pts1, pts2, pair_idx, H)

    assert diagnostics.inlier_count == len(pair_idx)
    assert diagnostics.median_reproj_error < 1e-6
    assert diagnostics.p90_reproj_error < 1e-6
    assert diagnostics.coverage_ratio > 0.5
    assert diagnostics.area_ratio == pytest.approx(1.0, abs=1e-6)
    assert diagnostics.is_flipped is False


def test_validate_homography_rejects_large_reprojection_error():
    pts1, pair_idx = _make_points()
    H = np.eye(3, dtype=np.float64)
    pts2 = pts1 + np.array([4.0, 3.0], dtype=np.float64)

    with pytest.raises(ValueError, match="median_reproj|p90_reproj"):
        validate_homography(pts1, pts2, pair_idx, H)


def test_validate_homography_rejects_flipped_canvas():
    pts1, pair_idx = _make_points()
    H = np.array([
        [-1.0, 0.0, 100.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    with pytest.raises(ValueError, match="flipped"):
        validate_homography(pts1, pts1, pair_idx, H)


def test_fine_tune_transform_accepts_four_unique_pairs_with_duplicates():
    pts1 = np.array([
        [20.0, 20.0],
        [80.0, 20.0],
        [80.0, 80.0],
        [20.0, 80.0],
        [50.0, 50.0],
    ], dtype=np.float64)
    translation = np.array([1.5, -2.0], dtype=np.float64)
    pts2 = pts1 + translation
    init_pair_idx = np.array([
        [0, 0],
        [1, 1],
        [2, 2],
        [3, 3],
        [0, 0],
        [1, 1],
    ], dtype=np.int32)

    H, pair_idx = fine_tune_transform(pts1, pts2, init_pair_idx)

    assert len(np.unique(pair_idx, axis=0)) >= 4
    reproj = pair_idx[:, 0]
    np.testing.assert_allclose(
        pts1[reproj] + translation,
        pts2[pair_idx[:, 1]],
        atol=1e-4)
    np.testing.assert_allclose(H[:2, 2], translation, atol=1e-4)


def test_fine_tune_transform_rejects_when_unique_pairs_below_four():
    pts1 = np.array([
        [20.0, 20.0],
        [80.0, 20.0],
        [80.0, 80.0],
    ], dtype=np.float64)
    pts2 = pts1.copy()
    init_pair_idx = np.array([
        [0, 0],
        [1, 1],
        [0, 0],
        [2, 2],
        [1, 1],
    ], dtype=np.int32)

    with pytest.raises(ValueError, match="at least 4 unique"):
        fine_tune_transform(pts1, pts2, init_pair_idx)


def test_find_initial_match_fallbacks_when_filtered_unique_pairs_below_four():
    n = 20
    features1, features2 = _make_features_with_controlled_diag_distance(
        [0.001] * 6 + [0.2] * (n - 6))
    pts = np.stack((np.arange(n, dtype=np.float64) * 10.0,
                    np.zeros(n, dtype=np.float64)), axis=1)
    vectors1 = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float64), (n, 1))
    vectors2 = vectors1.copy()
    vectors2[3:] = np.array([-1.0, 0.0, 0.0], dtype=np.float64)

    pair_idx = find_initial_match(
        features1, features2, pts, pts,
        vectors1=vectors1, vectors2=vectors2,
        apply_threshold_filter=True)

    assert len(pair_idx) == 6
    np.testing.assert_array_equal(
        pair_idx,
        np.column_stack((np.arange(6, dtype=np.int32), np.arange(6, dtype=np.int32))))


def test_find_initial_match_fallbacks_when_keep_ratio_too_low_on_small_set():
    n = 30
    features1, features2 = _make_features_with_controlled_diag_distance(
        [0.001] * 9 + [0.2] * (n - 9))
    pts = np.stack((np.arange(n, dtype=np.float64) * 10.0,
                    np.zeros(n, dtype=np.float64)), axis=1)
    vectors1 = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float64), (n, 1))
    vectors2 = vectors1.copy()
    vectors2[4:9] = np.array([-1.0, 0.0, 0.0], dtype=np.float64)

    pair_idx = find_initial_match(
        features1, features2, pts, pts,
        vectors1=vectors1, vectors2=vectors2,
        apply_threshold_filter=True)

    assert len(pair_idx) == 9
    np.testing.assert_array_equal(
        pair_idx,
        np.column_stack((np.arange(9, dtype=np.int32), np.arange(9, dtype=np.int32))))
