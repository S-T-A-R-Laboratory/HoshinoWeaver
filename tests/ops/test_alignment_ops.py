import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import hoshicore.ops.bundle_ops as bundle_ops
from hoshicore.component.norma.bundle import (BAAlignmentPlan, FrameAlignment,
                                               FrameAlignmentStatus)
from hoshicore.component.norma.bundle_window import (
    build_bundle_window_schedule,
)
from hoshicore.component.norma.types import CameraModel, Distortion, Intrinsics
from hoshicore.component.queue import StreamExhausted
from hoshicore.ops.bundle_ops import (BundleAdjustmentOp,
                                      BundleReferenceRemapOp,
                                      BundleRemapReport,
                                      BundleWindowMaxStackOp,
                                      BundleWindowMeanStackOp,
                                      BundleWindowMedianStackOp,
                                      BundleWindowSigmaClipStackOp,
                                      BundleWindowReport, WindowFrame,
                                      WindowFrameFilterGateOp,
                                      WindowFrameMaskedBlendOp,
                                      WindowFrameStatus)


class _Camera:
    def __init__(self, calls):
        self.intrinsics = SimpleNamespace(
            image_height_px=8, image_width_px=12)
        self.calls = calls

    def project_image_from_camera(
        self,
        camera,
        image,
        output_size,
        *,
        rotation_dst_to_src,
        map_scale,
    ):
        self.calls.append((image, output_size, rotation_dst_to_src, map_scale))
        assert camera is self
        return image + 10


class _WindowCamera:
    def __init__(self):
        self.intrinsics = SimpleNamespace(
            image_height_px=8, image_width_px=12)
        self.calls = []

    def project_image_from_camera(
        self,
        camera,
        image,
        output_size,
        *,
        rotation_dst_to_src,
        map_scale,
    ):
        assert camera is self
        self.calls.append((rotation_dst_to_src, map_scale, image.shape))
        return image.copy()


class _PartialCoverageCamera(_WindowCamera):
    def project_image_from_camera(self, *args, **kwargs):
        result = super().project_image_from_camera(*args, **kwargs)
        result[:, 0] //= 2
        return result


def _plan(camera, frames, reference=0):
    return BAAlignmentPlan(
        reference_frame_index=reference,
        shared_camera=camera,
        frames=tuple(frames),
        accepted_edge_count=1,
        rejected_edge_count=0,
        active_camera_parameter_count=0,
        observability_condition=None,
    )


def _frame(index, status=FrameAlignmentStatus.SOLVED, rotation=None,
           reason=None):
    if rotation is None and status == FrameAlignmentStatus.SOLVED:
        rotation = np.eye(3)
    return FrameAlignment(
        index=index,
        status=status,
        rotation_ref_to_src=rotation,
        pose_source="bundle" if status == FrameAlignmentStatus.SOLVED else "none",
        reason=reason,
    )


@pytest.mark.parametrize(
    (
        "configured_reference",
        "expected_reference",
        "camera_setup_mode",
        "expected_camera_tags",
        "expected_focal",
    ),
    [
        (1, 1, None, {"frame": 1}, 14.0),
        (None, 2, "auto", {"frame": 2}, None),
        (None, 2, "manual", None, 14.0),
    ],
)
def test_bundle_adjustment_uses_reference_camera_for_every_frame(
        monkeypatch,
        configured_reference,
        expected_reference,
        camera_setup_mode,
        expected_camera_tags,
        expected_focal,
):
    op = BundleAdjustmentOp("bundle")
    op.length = 4
    op.display_name = "Bundle adjustment"
    op.tracker = Mock()
    op.inputs["exifs"] = SimpleNamespace(active=True)
    images = [np.zeros((8, 12), dtype=np.uint16) for _ in range(4)]
    exifs = [SimpleNamespace(exif={"frame": index}) for index in range(4)]
    position = 0

    async def ready(value):
        return value

    def inputs():
        nonlocal position
        current = position
        position += 1
        return {"data": ready(images[current]), "exifs": ready(exifs[current])}

    async def run_cpu(function, *args, **kwargs):
        return function(*args, **kwargs)

    shared_candidate = object()
    candidate_calls = []
    detection_masks = []

    def build_candidate(tags, shape, method, distortion, focal, policy):
        candidate_calls.append((tags, shape, method, distortion, focal, policy))
        return shared_candidate

    captured = {}

    def build_plan(frames, **kwargs):
        captured["frames"] = frames
        captured["kwargs"] = kwargs
        for _ in range(5):
            kwargs["edge_completed"]()
        return "plan"

    async def broadcast(result):
        captured["output"] = result

    monkeypatch.setattr(op, "_async_convert_inputs", inputs)
    monkeypatch.setattr(op, "_run_cpu", run_cpu)
    monkeypatch.setattr(op, "_broadcast_outputs", broadcast)
    monkeypatch.setattr(
        bundle_ops.StarDetectionCache, "from_image",
        staticmethod(lambda image, mask: (
            detection_masks.append(mask)
            or SimpleNamespace(pywt_stars=object()))))
    monkeypatch.setattr(bundle_ops, "build_camera_candidate", build_candidate)
    monkeypatch.setattr(bundle_ops, "build_bundle_plan", build_plan)

    configs = {
        "method": "distortion",
        "camera_setup_mode": camera_setup_mode,
        "focal_length_mm": 14.0,
        "crop_factor": 1.0,
        "fallback_focal_equiv_mm": 20.0,
        "pair_offsets": [1, 2],
        "random_seed": 7,
        "mask": np.ones((4, 6, 3), dtype=np.float32),
    }
    if configured_reference is not None:
        configs["reference_frame_index"] = configured_reference
    asyncio.run(op._async_execute(configs))

    assert len(candidate_calls) == 1
    assert candidate_calls[0][0] == expected_camera_tags
    assert candidate_calls[0][4] == expected_focal
    assert all(frame.candidate is shared_candidate for frame in captured["frames"])
    assert len(detection_masks) == 4
    assert all(mask.shape == (8, 12) and mask.dtype == np.bool_
               and np.all(mask) for mask in detection_masks)
    edge_completed = captured["kwargs"].pop("edge_completed")
    assert callable(edge_completed)
    assert captured["kwargs"] == {
        "reference_frame_index": expected_reference,
        "pair_offsets": (1, 2),
        "max_pairs_per_edge": 128,
        "random_seed": 7,
        "camera_solve_frames": None,
        "edge_topology": "dense",
        "max_pair_offset": None,
        "max_nfev": 2000,
    }
    assert captured["output"] == {"alignment_plan": "plan"}
    op.tracker.create_bar.assert_called_once_with(
        "bundle", 10, desc="Bundle adjustment")
    op.tracker.reset_bar.assert_not_called()
    assert op.tracker.update.call_count == 10
    op.tracker.close_bar.assert_called_once_with("bundle")


def test_bundle_reference_remap_streams_in_order_and_skips_excluded(monkeypatch):
    op = BundleReferenceRemapOp("remap")
    op.length = 4
    op.display_name = "Bundle remap"
    op.tracker = Mock()
    op.inputs["exifs"] = SimpleNamespace(active=True)
    images = [np.full((8, 12), index, dtype=np.uint16)
              for index in range(4)]
    exifs = [object() for _ in images]
    position = 0

    async def ready(value):
        return value

    def inputs():
        nonlocal position
        current = position
        position += 1
        return {
            "data": ready(images[current]),
            "exifs": ready(exifs[current]),
        }

    async def run_cpu(function, *args, **kwargs):
        return function(*args, **kwargs)

    broadcasts = []

    async def broadcast(result):
        broadcasts.append(result)

    remap_calls = []
    camera = _Camera(remap_calls)
    plan = _plan(camera, (
        _frame(0),
        _frame(1, FrameAlignmentStatus.EXCLUDED, reason="disconnected"),
        _frame(2),
        _frame(3),
    ), reference=2)
    monkeypatch.setattr(op, "_async_convert_inputs", inputs)
    monkeypatch.setattr(op, "_run_cpu", run_cpu)
    monkeypatch.setattr(op, "_broadcast_outputs", broadcast)

    asyncio.run(op._async_filter({
        "alignment_plan": plan,
        "remap_map_scale": 0.25,
    }))

    frame_outputs = broadcasts[:-1]
    assert [int(result["result"][0, 0]) for result in frame_outputs] == [10, 2, 13]
    assert [result["aligned_exifs"] for result in frame_outputs] == [
        exifs[0], exifs[2], exifs[3]]
    assert len(remap_calls) == 2
    assert [int(call[0][0, 0]) for call in remap_calls] == [0, 3]
    assert all(call[1] == (12, 8) and call[3] == 0.25
               for call in remap_calls)

    report = broadcasts[-1]["remap_report"]
    assert isinstance(report, BundleRemapReport)
    assert [item.index for item in report.frames] == [0, 1, 2, 3]
    assert [item.emitted for item in report.frames] == [True, False, True, True]
    assert report.frames[1].reason == "disconnected"
    op.tracker.create_bar.assert_called_once_with(
        "remap", 4, desc="Bundle remap")
    assert op.tracker.update.call_count == 4
    op.tracker.close_bar.assert_called_once_with("remap")


def test_bundle_reference_remap_rejects_solved_frame_without_pose(monkeypatch):
    op = BundleReferenceRemapOp("remap")
    op.length = 1
    op.inputs["exifs"] = SimpleNamespace(active=False)
    image = np.zeros((8, 12), dtype=np.uint16)

    async def ready(value):
        return value

    monkeypatch.setattr(
        op, "_async_convert_inputs", lambda: {"data": ready(image)})
    camera = _Camera([])
    missing_pose = FrameAlignment(
        index=0,
        status=FrameAlignmentStatus.SOLVED,
        rotation_ref_to_src=None,
        pose_source="bundle",
    )

    with np.testing.assert_raises_regex(
            ValueError, "solved frame 0 has no reference-relative pose"):
        asyncio.run(op._async_filter({
            "alignment_plan": _plan(camera, (missing_pose,)),
        }))


def test_bundle_reference_remap_validates_second_load_geometry(monkeypatch):
    op = BundleReferenceRemapOp("remap")
    op.length = 1
    op.inputs["exifs"] = SimpleNamespace(active=False)

    async def ready(value):
        return value

    monkeypatch.setattr(
        op, "_async_convert_inputs",
        lambda: {"data": ready(np.zeros((7, 12), dtype=np.uint16))})
    camera = _Camera([])

    with np.testing.assert_raises_regex(
            ValueError, "geometry differs from AlignmentPlan"):
        asyncio.run(op._async_filter({
            "alignment_plan": _plan(camera, (_frame(0),)),
        }))


def test_bundle_reference_remap_resource_estimate_is_sequence_constant():
    assert BundleReferenceRemapOp.estimate_resources({}, 1024, 3) == (2048, 0)
    assert BundleReferenceRemapOp.estimate_resources({}, 1024, 300) == (2048, 0)


def test_bundle_window_mean_streams_shrinking_centered_windows(monkeypatch):
    op = BundleWindowMeanStackOp("window_mean")
    op.length = 3
    op.display_name = "Bundle window mean"
    op.tracker = Mock()
    op.inputs["exifs"] = SimpleNamespace(active=True)
    images = [np.full((8, 12), value, dtype=np.uint16)
              for value in (10, 20, 30)]
    exifs = [object() for _ in images]
    position = 0

    async def ready(value):
        return value

    def inputs():
        nonlocal position
        current = position
        position += 1
        return {
            "data": ready(images[current]),
            "exifs": ready(exifs[current]),
        }

    async def run_cpu(function, *args, **kwargs):
        return function(*args, **kwargs)

    broadcasts = []

    async def broadcast(result):
        broadcasts.append(result)

    camera = _WindowCamera()
    plan = _plan(camera, (_frame(0), _frame(1), _frame(2)))
    monkeypatch.setattr(op, "_async_convert_inputs", inputs)
    monkeypatch.setattr(op, "_run_cpu", run_cpu)
    monkeypatch.setattr(op, "_broadcast_outputs", broadcast)

    asyncio.run(op._async_execute({
        "alignment_plan": plan,
        "window_size": 3,
        "min_contributors": 2,
        "remap_map_scale": 0.25,
    }))

    outputs = broadcasts[:-1]
    frames = [item["result"] for item in outputs]
    assert [item.center_index for item in frames] == [0, 1, 2]
    assert [int(item.image[0, 0]) for item in frames] == [15, 20, 25]
    assert [item.exif for item in frames] == exifs
    assert all(item.status == WindowFrameStatus.READY for item in frames)
    assert len(camera.calls) == 4
    assert all(call[1] == 0.25 for call in camera.calls)
    assert all(call[2] == (8, 12, 2) for call in camera.calls)

    report = broadcasts[-1]["window_report"]
    assert isinstance(report, BundleWindowReport)
    assert [item.contributor_indices for item in report.frames] == [
        (0, 1), (0, 1, 2), (1, 2)]
    op.tracker.create_bar.assert_called_once_with(
        "window_mean", 3, desc="Bundle window mean")
    assert op.tracker.update.call_count == 3
    op.tracker.close_bar.assert_called_once_with("window_mean")


def test_bundle_window_keeps_excluded_center_as_invalid_position(monkeypatch):
    op = BundleWindowMeanStackOp("window_mean")
    op.length = 3
    op.tracker = Mock()
    op.inputs["exifs"] = SimpleNamespace(active=False)
    images = [np.full((8, 12), value, dtype=np.uint16)
              for value in (10, 20, 30)]
    position = 0

    async def ready(value):
        return value

    def inputs():
        nonlocal position
        image = images[position]
        position += 1
        return {"data": ready(image)}

    async def run_cpu(function, *args, **kwargs):
        return function(*args, **kwargs)

    broadcasts = []
    camera = _WindowCamera()
    plan = _plan(camera, (
        _frame(0),
        _frame(1, FrameAlignmentStatus.EXCLUDED, reason="disconnected"),
        _frame(2),
    ))
    monkeypatch.setattr(op, "_async_convert_inputs", inputs)
    monkeypatch.setattr(op, "_run_cpu", run_cpu)
    monkeypatch.setattr(
        op, "_broadcast_outputs", lambda result: _append_async(broadcasts, result))

    asyncio.run(op._async_execute({
        "alignment_plan": plan,
        "window_size": 3,
        "min_contributors": 1,
        "remap_map_scale": 0.5,
    }))

    frames = [item["result"] for item in broadcasts[:-1]]
    assert len(frames) == 3
    assert frames[1].center_index == 1
    assert frames[1].image is None
    assert frames[1].status == WindowFrameStatus.EXCLUDED
    assert frames[1].reason == "disconnected"


async def _append_async(items, value):
    items.append(value)


def test_bundle_window_identity_path_does_not_remap(monkeypatch):
    op = BundleWindowMeanStackOp("window_mean")
    op.length = 3
    op.tracker = Mock()
    op.inputs["exifs"] = SimpleNamespace(active=False)
    images = [np.full((8, 12), value, dtype=np.uint16)
              for value in (10, 20, 30)]
    position = 0

    async def ready(value):
        return value

    def inputs():
        nonlocal position
        image = images[position]
        position += 1
        return {"data": ready(image)}

    async def run_cpu(function, *args, **kwargs):
        return function(*args, **kwargs)

    broadcasts = []
    monkeypatch.setattr(op, "_async_convert_inputs", inputs)
    monkeypatch.setattr(op, "_run_cpu", run_cpu)
    monkeypatch.setattr(
        op, "_broadcast_outputs", lambda result: _append_async(broadcasts, result))

    asyncio.run(op._async_execute({
        "alignment_plan": None,
        "window_size": 3,
        "min_contributors": 1,
        "remap_map_scale": 0.5,
    }))

    frames = [item["result"] for item in broadcasts[:-1]]
    assert [int(frame.image[0, 0]) for frame in frames] == [15, 20, 25]
    assert all(frame.status == WindowFrameStatus.READY for frame in frames)


def test_bundle_window_identity_streams_unknown_length_and_flushes_edges(
        monkeypatch):
    op = BundleWindowMeanStackOp("window_mean")
    op.length = None
    op.tracker = Mock()
    op.inputs["exifs"] = SimpleNamespace(active=False)
    images = [np.full((8, 12), value, dtype=np.uint16)
              for value in (10, 20, 30, 40)]
    position = 0

    async def ready(value):
        return value

    async def exhausted():
        raise StreamExhausted

    def inputs():
        nonlocal position
        if position == len(images):
            return {"data": exhausted()}
        image = images[position]
        position += 1
        return {"data": ready(image)}

    async def run_cpu(function, *args, **kwargs):
        return function(*args, **kwargs)

    broadcasts = []
    monkeypatch.setattr(op, "_async_convert_inputs", inputs)
    monkeypatch.setattr(op, "_run_cpu", run_cpu)
    monkeypatch.setattr(
        op, "_broadcast_outputs", lambda result: _append_async(broadcasts, result))

    asyncio.run(op._async_execute({
        "alignment_plan": None,
        "window_size": 3,
        "min_contributors": 1,
        "remap_map_scale": 0.5,
    }))

    frames = [item["result"] for item in broadcasts[:-1]]
    assert [frame.center_index for frame in frames] == [0, 1, 2, 3]
    assert [int(frame.image[0, 0]) for frame in frames] == [15, 20, 30, 35]
    assert all(frame.status == WindowFrameStatus.READY for frame in frames)
    report = broadcasts[-1]["window_report"]
    assert len(report.frames) == 4
    op.tracker.create_bar.assert_not_called()
    assert op.tracker.update.call_count == 4
    op.tracker.close_bar.assert_not_called()


def test_window_frame_filter_gate_keeps_sidecars_aligned(monkeypatch):
    op = WindowFrameFilterGateOp("gate")
    op.length = 3
    op.tracker = Mock()
    frames = [
        WindowFrame(0, np.array([10]), "a", (0,), WindowFrameStatus.READY),
        WindowFrame(1, None, "b", (), WindowFrameStatus.EXCLUDED, "bad"),
        WindowFrame(2, np.array([30]), "c", (2,), WindowFrameStatus.READY),
    ]
    position = 0

    async def ready(value):
        return value

    def inputs():
        nonlocal position
        frame = frames[position]
        position += 1
        return {"data": ready(frame)}

    broadcasts = []
    monkeypatch.setattr(op, "_async_convert_inputs", inputs)
    monkeypatch.setattr(
        op, "_broadcast_outputs", lambda result: _append_async(broadcasts, result))

    asyncio.run(op._async_filter({}))

    assert [item["center_indices"] for item in broadcasts] == [0, 2]
    assert [item["aligned_exifs"] for item in broadcasts] == ["a", "c"]
    assert [int(item["result"][0]) for item in broadcasts] == [10, 30]


def test_window_mean_propagates_hard_mask_through_remap():
    camera = _WindowCamera()
    plan = _plan(camera, (_frame(0), _frame(1)))
    spec = build_bundle_window_schedule(
        plan, 3, min_contributors=2).windows[0]
    images = {
        0: np.full((8, 12), 10, dtype=np.uint16),
        1: np.full((8, 12), 20, dtype=np.uint16),
    }
    mask = np.zeros((8, 12), dtype=bool)
    mask[:, :6] = True

    result = bundle_ops._mean_bundle_window(
        camera, spec, images, (12, 8), 0.5, mask)

    np.testing.assert_array_equal(result[:, :6], 15)
    np.testing.assert_array_equal(result[:, 6:], 0)
    assert camera.calls[0][2] == (8, 12, 2)


def test_window_frame_masked_blend_uses_center_index_and_hard_mask(
        monkeypatch):
    op = WindowFrameMaskedBlendOp("blend")
    op.length = 1
    op.inputs["ground"] = SimpleNamespace(active=True)
    sky = WindowFrame(
        0, np.full((2, 4), 10, np.uint16), "sky", (0,),
        WindowFrameStatus.READY)
    ground = WindowFrame(
        0, np.full((2, 4), 20, np.uint16), None, (0,),
        WindowFrameStatus.READY)

    async def ready(value):
        return value

    monkeypatch.setattr(op, "_async_convert_inputs", lambda: {
        "sky": ready(sky), "ground": ready(ground)})
    broadcasts = []
    monkeypatch.setattr(
        op, "_broadcast_outputs", lambda result: _append_async(broadcasts, result))

    asyncio.run(op._async_execute({
        "mask": np.array([[255, 255, 0, 0], [255, 255, 0, 0]], np.uint8),
    }))

    result = broadcasts[0]["result"]
    assert result.center_index == 0
    np.testing.assert_array_equal(
        result.image,
        np.array([[10, 10, 20, 20], [10, 10, 20, 20]], np.uint16))


def test_bundle_window_mean_resource_estimate_scales_with_window_only():
    small = BundleWindowMeanStackOp.estimate_resources(
        {"window_size": 3}, 1024, 10, dtype_bytes=2)
    long_sequence = BundleWindowMeanStackOp.estimate_resources(
        {"window_size": 3}, 1024, 1000, dtype_bytes=2)
    wider_window = BundleWindowMeanStackOp.estimate_resources(
        {"window_size": 5}, 1024, 10, dtype_bytes=2)
    assert small == long_sequence
    assert wider_window[0] > small[0]


def test_bundle_window_mean_coverage_channel_uses_real_camera_remap():
    camera = CameraModel(
        Intrinsics(20.0, 36.0, 24.0, 12, 8), Distortion())
    plan = _plan(camera, (_frame(0), _frame(1)))
    spec = build_bundle_window_schedule(
        plan, 3, min_contributors=2).windows[0]
    images = {
        0: np.full((8, 12, 3), 10, dtype=np.uint16),
        1: np.full((8, 12, 3), 20, dtype=np.uint16),
    }

    result = bundle_ops._mean_bundle_window(
        camera, spec, images, (12, 8), 0.5)

    np.testing.assert_array_equal(
        result, np.full((8, 12, 3), 15, np.uint16))


def test_bundle_window_mean_discards_partial_border_samples():
    camera = _PartialCoverageCamera()
    plan = _plan(camera, (_frame(0), _frame(1)))
    spec = build_bundle_window_schedule(
        plan, 3, min_contributors=2).windows[0]
    images = {
        0: np.full((8, 12), 10, dtype=np.uint16),
        1: np.full((8, 12), 20, dtype=np.uint16),
    }

    result = bundle_ops._mean_bundle_window(
        camera, spec, images, (12, 8), 0.5)

    np.testing.assert_array_equal(result[:, 0], 10)
    np.testing.assert_array_equal(result[:, 1:], 15)


def test_bundle_window_mean_preserves_float_values():
    camera = _WindowCamera()
    plan = _plan(camera, (_frame(0), _frame(1)))
    spec = build_bundle_window_schedule(
        plan, 3, min_contributors=2).windows[0]
    images = {
        0: np.full((8, 12), 0.25, dtype=np.float32),
        1: np.full((8, 12), 0.75, dtype=np.float32),
    }

    result = bundle_ops._mean_bundle_window(
        camera, spec, images, (12, 8), 0.5)

    np.testing.assert_allclose(result, 0.5)


@pytest.mark.parametrize(("op_class", "expected"), [
    (BundleWindowMaxStackOp, 30),
    (BundleWindowMedianStackOp, 20),
    (BundleWindowSigmaClipStackOp, 20),
])
def test_bundle_window_additional_reducers(op_class, expected):
    camera = _WindowCamera()
    plan = _plan(camera, (_frame(0), _frame(1), _frame(2)))
    spec = build_bundle_window_schedule(
        plan, 3, min_contributors=2).windows[1]
    images = {
        index: np.full((8, 12), value, dtype=np.uint16)
        for index, value in enumerate((10, 20, 30))
    }
    configs = {"rej_high": 3.0, "rej_low": 3.0, "max_iter": 5}

    result = op_class._reduce_window(
        camera, spec, images, (12, 8), 0.5, configs)

    np.testing.assert_array_equal(result, expected)


def test_bundle_window_sigma_clip_rejects_outlier():
    camera = _WindowCamera()
    frames = tuple(_frame(index) for index in range(5))
    spec = build_bundle_window_schedule(
        _plan(camera, frames), 5, min_contributors=2).windows[2]
    images = {
        index: np.full((8, 12), value, dtype=np.uint16)
        for index, value in enumerate((10, 10, 10, 10, 250))
    }

    result = BundleWindowSigmaClipStackOp._reduce_window(
        camera, spec, images, (12, 8), 0.5,
        {"rej_high": 1.0, "rej_low": 1.0, "max_iter": 5})

    np.testing.assert_array_equal(result, 10)


def test_bundle_window_reducer_resource_estimates_include_window_stack():
    mean = BundleWindowMeanStackOp.estimate_resources(
        {"window_size": 5}, 1024, 100, dtype_bytes=2)[0]
    maximum = BundleWindowMaxStackOp.estimate_resources(
        {"window_size": 5}, 1024, 100, dtype_bytes=2)[0]
    median = BundleWindowMedianStackOp.estimate_resources(
        {"window_size": 5}, 1024, 100, dtype_bytes=2)[0]
    sigma = BundleWindowSigmaClipStackOp.estimate_resources(
        {"window_size": 5}, 1024, 100, dtype_bytes=2)[0]
    assert maximum < median
    assert mean < sigma
    assert median < sigma


@pytest.mark.parametrize(("op_class", "interior"), [
    (BundleWindowMaxStackOp, 200),
    (BundleWindowMedianStackOp, 105),
])
def test_bundle_window_reducers_respect_partial_coverage_mask(
        op_class, interior):
    camera = _PartialCoverageCamera()
    plan = _plan(camera, (_frame(0), _frame(1)))
    spec = build_bundle_window_schedule(
        plan, 3, min_contributors=2).windows[0]
    images = {
        0: np.full((8, 12), 10, dtype=np.uint16),
        1: np.full((8, 12), 200, dtype=np.uint16),
    }

    result = op_class._reduce_window(
        camera, spec, images, (12, 8), 0.5, {})

    np.testing.assert_array_equal(result[:, 0], 10)
    np.testing.assert_array_equal(result[:, 1:], interior)
