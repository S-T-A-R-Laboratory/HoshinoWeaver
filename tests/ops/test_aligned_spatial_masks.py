"""Independent mask remap clips alignment outputs without changing frame payloads."""
import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from hoshicore._custom_op import _dispatch
from hoshicore.component.data_container import FloatImage
from hoshicore.component.frame_buffer import DiskFrameBuffer, MemoryFrameBuffer
from hoshicore.component.norma.bundle import (
    BAAlignmentPlan, FrameAlignment, FrameAlignmentStatus,
)
from hoshicore.component.norma.types import CameraModel, Distortion, Intrinsics
from hoshicore.component.norma.frame_align import (
    prepare_spatial_mask, remap_image_with_mask,
)
from hoshicore.ops.alignment_ops import StarAlignmentOp
from hoshicore.ops.bundle_ops import BundleReferenceRemapOp
from hoshicore.ops.sigma_clip_ops import (
    HuberMeanIteratorOp, SigmaClipFusedChunkOp, MedianReduceOp,
)
from hoshicore.ops.trailstacker import MeanStackerOp


@pytest.fixture(params=['numpy', 'cpu', 'auto'])
def backend(request, monkeypatch):
    monkeypatch.setattr(_dispatch, '_backend_preference_override', request.param)


def _drive(monkeypatch, op, frames, configs, *, method='_async_execute', **streams):
    op.length = len(frames)
    op.tracker = Mock()
    streams = {'data': frames, **streams}
    for name in op.inputs:
        op.inputs[name] = SimpleNamespace(active=name in streams)
    position = 0
    outputs = []

    async def ready(value):
        return value

    def inputs():
        nonlocal position
        values = {key: ready(values[position]) for key, values in streams.items()}
        position += 1
        return values

    async def run_cpu(function, *args, **kwargs):
        return function(*args, **kwargs)

    async def broadcast(value):
        outputs.append(value)

    monkeypatch.setattr(op, '_async_convert_inputs', inputs)
    monkeypatch.setattr(op, '_run_cpu', run_cpu)
    monkeypatch.setattr(op, '_broadcast_outputs', broadcast)
    asyncio.run(getattr(op, method)(configs))
    return outputs


def _samples():
    mask = np.zeros((4, 12), dtype=bool)
    mask[:, :6] = True
    first = np.full((4, 12, 3), 1000, np.uint16) * mask[..., None]
    transform = np.array([[1., 0., -.5], [0., 1., 0.], [0., 0., 1.]])

    def warp(image):
        return cv2.warpPerspective(image, transform, (12, 4), flags=cv2.INTER_LINEAR)

    remapped = warp(first)
    assert remapped[1, 5, 0] == 500
    second = remap_image_with_mask(first, warp, mask, mask)
    assert second[1, 5, 0] == 0
    return [first, second], mask


def test_prepare_mask_and_coverage_reject_partial_support():
    mask = prepare_spatial_mask(np.array([[[255], [0]]], np.uint8), (2, 4, 3))
    np.testing.assert_array_equal(mask, [[True, True, False, False]] * 2)
    target = np.ones((2, 4), bool)
    target[0, 0] = False
    coverage = np.array([[1., 1., .9999, 0.], [1., .5, 0., np.nan]], np.float32)
    expected = np.array([[False, True, False, False], [True, False, False, False]])
    for shape in ((2, 4), (2, 4, 1), (2, 4, 3)):
        image = np.ones(shape, np.uint16)
        calls = []

        def remap(array):
            calls.append(array.dtype)
            return coverage if array.dtype == np.float32 else array.copy()

        result = remap_image_with_mask(image, remap, mask, target)
        assert calls == [np.dtype('uint16'), np.dtype('float32')]
        assert result is not image
        np.testing.assert_array_equal(image, 1)
        np.testing.assert_array_equal(result[~expected], 0)
        np.testing.assert_array_equal(result[expected], 1)
    image = np.ones((2, 4, 3), np.uint8)
    calls = []

    def remap(array):
        calls.append(array)
        return array.copy()

    result = remap_image_with_mask(image, remap)
    assert len(calls) == 1 and calls[0] is image
    np.testing.assert_array_equal(result, image)


def test_rgb_mean_ignores_clipped_edge(monkeypatch, backend):
    frames, mask = _samples()
    outputs = _drive(monkeypatch, MeanStackerOp('mean'), frames,
                     {'int_weight': False, 'mask': mask})
    np.testing.assert_array_equal(outputs[0]['result'].data, frames[0])
    assert outputs[0]['statistics'].n[1, 5, 0] == 1
    assert outputs[0]['statistics'].n[1, 4, 0] == 2


@pytest.mark.parametrize('buffer_type', [DiskFrameBuffer, MemoryFrameBuffer])
@pytest.mark.parametrize('op_type', [SigmaClipFusedChunkOp, HuberMeanIteratorOp, MedianReduceOp])
def test_rgb_reducers_consume_plain_zero_clipped_arrays(
        monkeypatch, tmp_path, backend, buffer_type, op_type):
    frames, mask = _samples()
    reference = _drive(monkeypatch, MeanStackerOp('mean'), frames,
                       {'int_weight': False, 'mask': mask})[0]['statistics']
    buffer = (DiskFrameBuffer(tmp_path) if buffer_type is DiskFrameBuffer
              else MemoryFrameBuffer())
    for frame in frames:
        buffer.append(frame)
    buffer.acquire()
    configs = dict(buffer_handle=buffer, fgp_total=reference, mask=mask,
                   chunk_rows=2, rej_high=10., rej_low=10., max_iter=3,
                   huber_c=1.345)
    outputs = _drive(monkeypatch, op_type('reduce'), [], configs)
    np.testing.assert_array_equal(outputs[0]['result'].data, frames[0])
    assert len(buffer) == 0
    assert not list(tmp_path.glob('*.npy'))


def test_bundle_clips_independent_mask_and_preserves_stream_types(monkeypatch, backend):
    camera = CameraModel(Intrinsics(20., 36., 24., 12, 8), Distortion())
    angle = .075
    rotation = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
                         [-np.sin(angle), 0., np.cos(angle)]])
    entries = tuple(FrameAlignment(
        index=index, status=status, rotation_ref_to_src=pose,
        pose_source='bundle' if pose is not None else 'none')
        for index, status, pose in [
            (0, FrameAlignmentStatus.SOLVED, np.eye(3)),
            (1, FrameAlignmentStatus.EXCLUDED, None),
            (2, FrameAlignmentStatus.SOLVED, rotation)])
    plan = BAAlignmentPlan(
        reference_frame_index=0, shared_camera=camera, frames=entries,
        accepted_edge_count=1, rejected_edge_count=0,
        active_camera_parameter_count=0, observability_condition=None)
    mask = np.zeros((8, 12), bool)
    mask[:, :6] = True
    image = np.full((8, 12, 3), 1000, np.uint16) * mask[..., None]
    original = CameraModel.project_image_from_camera
    calls = []

    def project(self, source, array, *args, **kwargs):
        calls.append((array.shape, array.dtype))
        return original(self, source, array, *args, **kwargs)

    monkeypatch.setattr(CameraModel, 'project_image_from_camera', project)
    outputs = _drive(monkeypatch, BundleReferenceRemapOp('remap'),
                     [image] * 3, {'alignment_plan': plan, 'mask': mask},
                     method='_async_filter', exifs=['ref', 'excluded', 'source'])
    assert calls == [((8, 12, 3), np.dtype('uint16')),
                     ((8, 12), np.dtype('float32'))]
    assert [item['aligned_exifs'] for item in outputs[:-1]] == ['ref', 'source']
    aligned = outputs[1]['result']
    assert isinstance(aligned, np.ndarray)
    assert aligned.dtype == image.dtype
    np.testing.assert_array_equal(aligned[:, 5:], 0)
    np.testing.assert_array_equal(image[:, :6], 1000)


@pytest.mark.parametrize('identity', [False, True])
def test_homography_clips_mask_without_changing_return_type(monkeypatch, identity):
    import hoshicore.component.norma.frame_align as module

    frames, mask = _samples()
    image = frames[0]
    target = mask.copy()
    target[0, 0] = False
    monkeypatch.setattr(module.StarDetectionCache, 'from_image',
                        lambda *args, **kwargs: SimpleNamespace(median_stars=object()))
    monkeypatch.setattr(module, 'GeometryView', lambda *args: object())
    monkeypatch.setattr(module, '_check_star_count', lambda *args: None)
    monkeypatch.setattr(module, '_has_identical_star_geometry', lambda *args: identity)
    match = SimpleNamespace(
        homography=np.array([[1., 0., .5], [0., 1., 0.], [0., 0., 1.]]),
        ref_pts=np.array([[1., 1.]]), src_pts=np.array([[1.5, 1.]]))
    monkeypatch.setattr(module, 'match_star_pairs', lambda *args: match)
    monkeypatch.setattr(module, '_log_residual_diagnostics', lambda *args: None)
    result = module.align_frame_homography(
        image, object(), image, src_camera=object(),
        source_mask=mask, target_mask=target)
    assert isinstance(result, np.ndarray)
    assert result.dtype == image.dtype
    np.testing.assert_array_equal(result[0, 0], 0)
    assert result[1, 5, 0] == (1000 if identity else 0)
    assert image[0, 0, 0] == 1000


@pytest.mark.parametrize('wrapped', [False, True])
def test_pair_camera_alignment_preserves_original_payload(monkeypatch, backend, wrapped):
    import hoshicore.component.norma.frame_align as module
    import hoshicore.ops.alignment_ops as op_module

    camera = CameraModel(Intrinsics(20., 36., 24., 12, 8), Distortion())
    candidate = SimpleNamespace(camera=camera, init_source='manual')
    angle = .075
    rotation = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
                         [-np.sin(angle), 0., np.cos(angle)]])
    stars = object()
    monkeypatch.setattr(op_module, 'build_camera_candidate', lambda *args: candidate)
    monkeypatch.setattr(module.StarDetectionCache, 'from_image',
                        lambda *args, **kwargs: SimpleNamespace(pywt_stars=stars))
    for owner in (module, op_module):
        monkeypatch.setattr(owner, 'GeometryView',
                            lambda stars, camera: SimpleNamespace(stars=stars))
    monkeypatch.setattr(module, '_check_star_count', lambda *args: None)
    monkeypatch.setattr(module, '_has_identical_star_geometry', lambda *args: False)
    match = SimpleNamespace(ref_pts=np.array([[3., 3.]]),
                            src_pts=np.array([[3.5, 3.]]), pair_idx=np.array([[0, 0]]))
    solution = SimpleNamespace(rotation_ref_to_src=rotation,
                              ref_camera=camera, src_camera=camera)
    monkeypatch.setattr(module, 'solve_staged_alignment', lambda *args, **kwargs:
                        SimpleNamespace(final_alignment=solution, final_match=match,
                                        bootstrap_match=match, refine_status='disabled'))
    mask = np.zeros((8, 12), bool)
    mask[:, :6] = True
    array = np.full((8, 12, 3), 1000, np.uint16) * mask[..., None]
    frame = FloatImage(array.astype(np.float32), array.dtype) if wrapped else array
    outputs = _drive(monkeypatch, StarAlignmentOp('pair'), [frame],
                     {'reference': array, 'mask': mask, 'method': 'distortion'},
                     method='_async_filter')
    result = outputs[0]['result']
    assert isinstance(result, FloatImage if wrapped else np.ndarray)
    assert not hasattr(result, 'spatial_mask')
    assert result.dtype == np.dtype('uint16')
    data = result.data if wrapped else result
    assert data.dtype == (np.dtype('float32') if wrapped else array.dtype)
    np.testing.assert_array_equal(data[:, 5:], 0)
