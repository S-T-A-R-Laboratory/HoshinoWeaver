"""Deterministic ROI geometry and fusion without external weights or sources."""
import numpy as np
import pytest

from hoshicore.component.sky_roi import plan_tiles, refine_rois


def test_no_selection_preserves_global_map_and_runs_no_windows():
    parent = np.zeros((180, 320), np.float32)
    calls, events = [], []
    result, info = refine_rois(np.zeros((180, 320, 3), np.uint8), parent,
                              lambda patch: calls.append(patch),
                              report=lambda *event: events.append(event))
    np.testing.assert_array_equal(result, parent)
    assert not calls and info['scanned_tiles'] == 0
    assert events == [('roi_planned', 0, 0, None), ('fusing', 0, 0, None)]


@pytest.mark.parametrize('shape', [(1, 1), (100, 190), (430, 910)])
def test_uncertain_selection_is_covered_and_hann_fusion_is_constant(shape):
    parent = np.full(shape, .5, np.float32)
    calls = []
    def predict(patch):
        calls.append(patch.shape[:2])
        return np.full(patch.shape[:2], .8, np.float32)
    result, info = refine_rois(np.zeros((*shape, 3), np.uint8), parent, predict)
    np.testing.assert_allclose(result, .71, atol=2e-7)
    assert info['selection_fraction'] == 1 and info['covered_fraction'] == 1
    assert info['scanned_tiles'] == info['total_tiles'] == len(calls)
    assert all(max(size) <= 512 for size in calls)
    assert info['working_shape'] == list(shape)


def test_capped_working_image_does_not_resize_native_result_grid():
    shape = (400, 901)
    calls = []
    result, info = refine_rois(np.zeros((*shape, 3), np.uint8), np.full(shape, .5, np.float32),
                              lambda patch: calls.append(patch.shape) or np.full(patch.shape[:2], .8, np.float32),
                              long_edge=512)
    assert result.shape == shape and result.dtype == np.float32
    assert info['working_shape'] == [227, 512]
    assert all(max(size[:2]) <= 512 for size in calls)
    np.testing.assert_allclose(result, .71, atol=2e-7)


def test_selected_windows_only_and_confident_parent_conflict_guard():
    shape = (200, 1024)
    parent = np.full(shape, .99, np.float32)
    parent[:, 512:] = .01
    result, info = refine_rois(np.zeros((*shape, 3), np.uint8), parent,
                              lambda patch: np.zeros(patch.shape[:2], np.float32))
    assert 0 < info['scanned_tiles'] < info['total_tiles']
    assert info['scanned_tiles'] == sum(box['selected'] for box in info['boxes'])
    assert (result[:, :512] >= .5).all()  # even opposite local predictions cannot erase confident sky
    np.testing.assert_array_equal(result[:, :200], parent[:, :200])


def test_window_geometry_covers_edges_and_keeps_context_inside_image():
    shape = (430, 910)
    boxes = plan_tiles(shape, np.ones(shape, np.bool_))
    covered = np.zeros(shape, np.bool_)
    for box in boxes:
        x, y, right, bottom = box['core']
        px, py, pr, pb = box['patch']
        assert 0 <= px <= x < right <= pr <= shape[1]
        assert 0 <= py <= y < bottom <= pb <= shape[0]
        covered[y:bottom, x:right] = True
    assert covered.all()
    assert boxes[-1]['core'][2:] == [shape[1], shape[0]]


@pytest.mark.parametrize('value', [np.nan, -.1, 1.1])
def test_invalid_local_probabilities_fail(value):
    with pytest.raises(ValueError, match='Invalid local probability'):
        refine_rois(np.zeros((13, 17, 3), np.uint8), np.full((13, 17), .5, np.float32),
                    lambda patch: np.full(patch.shape[:2], value, np.float32))
