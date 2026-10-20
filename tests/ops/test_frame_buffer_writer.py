"""Only actual input frames may be retained for later algorithm passes."""
import asyncio
from pathlib import Path

import numpy as np
import pytest
import yaml

from hoshicore.component.data_container import FloatImage
from hoshicore.component.queue import RichContextQueue
from hoshicore.engine.build import validate_and_build_order
from hoshicore.engine.flatten import flatten_sub_dags
from hoshicore.engine.wiring import run_dag
from hoshicore.ops.sigma_clip_ops import DiskBufferWriterOp


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['memory', 'disk'])
async def test_writer_retains_processed_pixels_and_weights(mode, tmp_path):
    original = np.arange(6 * 5 * 3, dtype=np.uint16).reshape(6, 5, 3)
    frames = [np.ascontiguousarray(original[1:5, ::-1] * 2 + i) for i in range(3)]
    weights = [None, 0.5, np.full(frames[0].shape, 0.75, np.float32)]
    op = DiskBufferWriterOp('buffer')
    sink = RichContextQueue(maxsize=1)
    op.outputs['buffer_handle'].append(sink)

    async def feed():
        await op.inputs['data'].set_length(len(frames))
        await op.inputs['weight'].set_length(len(frames))
        await op.config['buffer_mode'].put(mode)
        await op.config['temp_path'].put(str(tmp_path))
        for frame, weight in zip(frames, weights):
            await asyncio.gather(op.inputs['data'].put(frame), op.inputs['weight'].put(weight))

    await asyncio.wait_for(asyncio.gather(feed(), op.execute()), 5)
    assert await sink.get_length() == 1
    buffer = await sink.get()
    try:
        assert len(buffer) == len(frames)
        for index, (frame, weight) in enumerate(zip(frames, weights)):
            actual, actual_weight = buffer.get_rows(index, 0, len(frame))
            np.testing.assert_array_equal(actual, frame)
            if isinstance(weight, np.ndarray):
                np.testing.assert_array_equal(actual_weight, weight)
            else:
                assert actual_weight == weight
    finally:
        buffer.cleanup()
    assert not list(tmp_path.glob('*.npy'))


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['replay', 'unknown', None])
async def test_writer_rejects_unsupported_mode_before_consuming_frames(mode):
    op = DiskBufferWriterOp('buffer')
    with pytest.raises(ValueError, match='use memory or disk'):
        await asyncio.wait_for(op._async_execute({'buffer_mode': mode}), 1)


@pytest.mark.asyncio
@pytest.mark.parametrize('workflow', [
    'sigma_clip', 'sigma_clip_fused', 'median_stack_core',
    'huber_mean', 'mix_core', 'threshold_max_startrail',
])
async def test_stacking_subgraphs_work_without_source_paths(workflow, tmp_path):
    # Exercise reusable subgraphs through their normal parent-graph contract.
    inputs = {'data': 'inputs.data'}
    if workflow in ('mix_core', 'threshold_max_startrail'):
        inputs['weight'] = 'inputs.weight'
    spec = {
        'inputs': {'data': {'type': 'sequence'}, 'weight': {'type': 'sequence'}},
        'configs': {'int_weight': {'type': 'bool', 'default': True}},
        'nodes': {'stack': {
            'op': str(ROOT / f'hoshicore/dag/base/{workflow}.yaml'),
            'inputs': inputs, 'configs': {'int_weight': 'configs.int_weight'},
            'outputs': {'result': {'type': 'image'}},
        }},
        'outputs': {'result': 'stack.result'},
    }
    dag = validate_and_build_order(flatten_sub_dags(spec))
    frame = (np.arange(8 * 8 * 3, dtype=np.uint16).reshape(8, 8, 3) + 100)
    frames = [frame.copy() for _ in range(4)]
    outputs = []
    for mode in ('memory', 'disk'):
        result = await asyncio.wait_for(run_dag(
            dag, {'data': frames, 'weight': [1.0] * len(frames)},
            {'buffer_mode': mode, 'temp_path': str(tmp_path)}, progress=False), 10)
        image = result['result']
        pixels = image.data if isinstance(image, FloatImage) else image
        assert pixels.shape == frame.shape
        assert np.all(np.isfinite(pixels))
        outputs.append(pixels.copy())
    np.testing.assert_allclose(outputs[0], outputs[1])


def test_global_buffer_options_are_memory_and_disk():
    settings = yaml.safe_load((ROOT / 'hoshicore/dag/global_settings.ui.yaml').read_text(
        encoding='utf-8'))
    assert {entry['value'] for entry in settings['buffer_mode']['options']} == {'memory', 'disk'}
