import asyncio

import numpy as np
import pytest
from loguru import logger

from hoshicore.component.data_container import FloatImage
from hoshicore.component.queue import RichContextQueue
from hoshicore.ops.simple_ops import ApplyMaskOp, CalibrationSubtractImageOp


pytestmark = pytest.mark.asyncio


async def _run_single_image_subtract(frame, reference):
    op = CalibrationSubtractImageOp("subtract_image")
    sink = RichContextQueue(maxsize=1)
    op.outputs["result"].append(sink)
    await op.config["data"].put(frame)
    await op.config["reference"].put(reference)
    task = asyncio.create_task(op.execute())
    result = await asyncio.wait_for(sink.get(), timeout=0.1)
    await asyncio.wait_for(task, timeout=0.1)
    return result


async def test_calibration_subtract_image_passthrough_when_reference_is_none():
    frame = np.array([[10, 20], [30, 40]], dtype=np.uint16)

    result = await _run_single_image_subtract(frame, None)

    assert result is frame


async def test_calibration_subtract_image_subtracts_reference():
    frame = np.array([[10, 20], [30, 40]], dtype=np.uint16)
    reference = np.array([[1, 2], [3, 4]], dtype=np.uint16)

    result = await _run_single_image_subtract(frame, reference)

    np.testing.assert_array_equal(result, frame - reference)


async def _run_apply_mask(frames, mask):
    op = ApplyMaskOp("apply_mask")
    sink = RichContextQueue(maxsize=1)
    op.outputs["result"].append(sink)
    await op.config["mask"].put(mask)

    async def feed():
        await op.inputs["image"].set_length(len(frames))
        for frame in frames:
            await op.inputs["image"].put(frame)

    async def collect():
        assert await sink.get_length() == len(frames)
        return [await sink.get() for _ in frames]

    _, _, results = await asyncio.wait_for(
        asyncio.gather(feed(), op.execute(), collect()), timeout=2)
    return results


@pytest.mark.parametrize("mask_channels", [None, 1, 3])
@pytest.mark.parametrize("wrapped", [False, True])
async def test_apply_mask_auto_resize_warns_once(mask_channels, wrapped):
    # A one-pixel width mismatch must not fail the whole sequence.
    mask = np.array([[0, 0.5, 1], [1, 0, 1]], dtype=np.float32)
    if mask_channels is not None:
        mask = np.repeat(mask[..., None], mask_channels, axis=-1)
    data = np.full((2, 4, 3), 7, dtype=np.float32)
    frame = FloatImage(data=data, dtype=np.dtype("uint16")) if wrapped else data
    warnings = []
    handler = logger.add(lambda msg: warnings.append(str(msg)), level="WARNING")
    try:
        results = await _run_apply_mask([frame, frame], mask)
    finally:
        logger.remove(handler)

    expected = np.repeat(
        np.array([[0, 0, 0, 7], [7, 7, 0, 7]], dtype=np.float32)[..., None],
        3, axis=-1)
    for result in results:
        if wrapped:
            assert isinstance(result, FloatImage)
            assert result.dtype == frame.dtype
            result = result.data
        np.testing.assert_array_equal(result, expected)
        assert result.dtype == data.dtype
    assert len(warnings) == 1
    assert "(2, 3)" in warnings[0] and "(2, 4)" in warnings[0]


async def test_apply_mask_resolution_changes_use_original_mask():
    mask = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.float32)
    frames = [np.ones((2, 2)), np.ones((2, 3)), np.ones((2, 3, 3))]

    results = await _run_apply_mask(frames, mask)

    np.testing.assert_array_equal(results[0], [[0, 1], [1, 0]])
    np.testing.assert_array_equal(results[1], mask)
    np.testing.assert_array_equal(results[2], np.repeat(mask[..., None], 3, axis=-1))


async def test_apply_mask_passthrough_and_matching_size_do_not_warn():
    frame = np.ones((2, 3, 3), dtype=np.uint16)
    warnings = []
    handler = logger.add(lambda msg: warnings.append(str(msg)), level="WARNING")
    try:
        passthrough = await _run_apply_mask([frame], None)
        results = await _run_apply_mask([frame], np.ones((2, 3)))
    finally:
        logger.remove(handler)

    assert passthrough[0] is frame
    np.testing.assert_array_equal(results[0], frame)
    assert warnings == []
