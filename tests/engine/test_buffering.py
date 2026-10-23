"""Bounded fan-out must keep supplying a delayed branch until it can emit."""
import asyncio

import numpy as np
import pytest

from hoshicore.engine.buffering import plan_input_buffers
from hoshicore.engine.build import _load_yaml, validate_and_build_order
from hoshicore.engine.flatten import flatten_sub_dags
from hoshicore.engine.meta import meta_resolve
from hoshicore.engine.wiring import run_dag
from hoshicore.engine.registry import REGISTERED_OP
from hoshicore.ops.base import BaseOp


class CopyOp(BaseOp):
    INPUTS = {'data': {'type': 'sequence'}}
    OUTPUTS = {'result': {'type': 'sequence'}}

    async def _async_execute(self, configs):
        for _ in self._input_range():
            await self._broadcast_outputs({'result': await self.inputs['data'].get()})


def window_graph(sky, ground, *, intermediate=False, independent=False):
    configs = {'sky_w': sky, 'ground_w': ground, 'kind': 'img_array',
               'frames': {'data': [np.full((2, 3, 3), i, np.uint8) for i in range(30)]}}
    loader = {'op': 'ImgDataLoaderOp', 'inputs': {'src': 'inputs.src'},
              'configs': {'loader_type': 'configs.kind', 'configs': 'configs.frames'},
              'outputs': {'result': {'type': 'sequence'}}}
    nodes = {'loader': loader}
    if independent:
        nodes['other_loader'] = loader.copy()
    source = 'loader.result'
    if intermediate:
        nodes['copy'] = {'op': 'CopyOp', 'inputs': {'data': source},
                         'outputs': {'result': {'type': 'sequence'}}}
        source = 'copy.result'
    for name, input_source in [('sky', source), ('ground',
            'other_loader.result' if independent else 'loader.result')]:
        nodes[name] = {'op': 'BundleWindowMeanStackOp', 'inputs': {'data': input_source},
                       'configs': {'window_size': f'configs.{name}_w'},
                       'outputs': {'result': {'type': 'sequence'}}}
    nodes['blend'] = {'op': 'WindowFrameMaskedBlendOp',
                      'inputs': {'sky': 'sky.result', 'ground': 'ground.result'},
                      'outputs': {'result': {'type': 'sequence'}}}
    dag = validate_and_build_order({
        'inputs': {'src': {'type': 'sequence'}},
        'configs': {key: {'type': 'object'} for key in configs},
        'nodes': nodes, 'outputs': {'result': 'blend.result'}})
    return dag, configs


@pytest.mark.asyncio
@pytest.mark.parametrize('sky,ground', [(1, 15), (15, 1), (3, 5), (15, 15)])
@pytest.mark.parametrize('intermediate', [False, True])
async def test_unequal_windows_complete_and_preserve_positions(sky, ground, intermediate):
    dag, configs = window_graph(sky, ground, intermediate=intermediate)
    registry = {**REGISTERED_OP, 'CopyOp': CopyOp}
    capacities = plan_input_buffers(dag, registry, configs)
    assert capacities['blend', 'sky'] == ground // 2 + 1
    assert capacities['blend', 'ground'] == sky // 2 + 1
    result = await asyncio.wait_for(run_dag(
        dag, {'src': list(range(30))}, configs, registry, progress=False), 10)
    frames = result['result']
    assert [f.center_index for f in frames] == list(range(30))
    for i, frame in enumerate(frames):
        start, stop = max(0, i - sky // 2), min(30, i + sky // 2 + 1)
        np.testing.assert_allclose(frame.image, (start + stop - 1) / 2, atol=0.5)


def test_independent_feeders_do_not_need_compensation():
    dag, configs = window_graph(1, 15, independent=True)
    capacities = plan_input_buffers(dag, REGISTERED_OP, configs)
    assert set(capacities.values()) == {1}


def test_serial_windows_accumulate_and_unjoined_fanout_stays_small():
    dag, configs = window_graph(15, 1)
    spec = {'inputs': {'src': {'type': 'sequence'}},
            'configs': {key: {'type': 'object'} for key in configs},
            'nodes': dict(dag.nodes), 'outputs': {'result': 'blend.result'}}
    spec['nodes']['sky_second'] = {
        'op': 'BundleWindowMeanStackOp', 'inputs': {'data': 'sky.result'},
        'configs': {'window_size': 'configs.sky_w'},
        'outputs': {'result': {'type': 'sequence'}}}
    spec['nodes']['blend'] = {**spec['nodes']['blend'],
                              'inputs': {'sky': 'sky_second.result', 'ground': 'ground.result'}}
    chained = validate_and_build_order(spec)
    assert plan_input_buffers(chained, REGISTERED_OP, configs)['blend', 'ground'] == 15
    spec['nodes'].pop('blend')
    spec['outputs'] = {'sky': 'sky_second.result', 'ground': 'ground.result'}
    assert set(plan_input_buffers(validate_and_build_order(spec), REGISTERED_OP,
                                  configs).values()) == {1}


def test_preflight_includes_planned_queue_memory():
    from hoshicore.engine.preflight import _estimate_queue_overhead
    dag, configs = window_graph(1, 15)
    baseline = _estimate_queue_overhead(dag, REGISTERED_OP, 100)
    assert _estimate_queue_overhead(dag, REGISTERED_OP, 100, configs) == baseline + 700


def test_static_parameter_required_and_existing_capacity_preserved():
    dag, configs = window_graph(1, 15)
    from hoshicore.ops.bundle_ops import WindowFrameMaskedBlendOp

    class LargerJoin(WindowFrameMaskedBlendOp):
        MAX_SIZE = 20

    registry = {**REGISTERED_OP, 'WindowFrameMaskedBlendOp': LargerJoin}
    assert plan_input_buffers(dag, registry, configs)['blend', 'sky'] == 20
    configs.pop('sky_w')
    with pytest.raises(ValueError, match='requires static config'):
        plan_input_buffers(dag, registry, configs)


@pytest.mark.asyncio
async def test_satellite_exif_short_branch_completes(monkeypatch, tmp_path):
    from hoshicore.component import dataloader
    from hoshicore.component.exif import ExifData
    from hoshicore.ops import image_saver
    from hoshicore.ops.satellite_clean_op import SatelliteCleanOp

    monkeypatch.setattr(dataloader, 'load_img', lambda _: np.ones((8, 8, 3), np.uint8))
    monkeypatch.setattr(dataloader, 'read_exif_data', lambda _: ExifData({}, b''))
    monkeypatch.setattr(SatelliteCleanOp, '_detect_geometry', lambda *args: None)
    monkeypatch.setattr(SatelliteCleanOp, '_compute_rotation', lambda *args: None)
    monkeypatch.setattr(SatelliteCleanOp, '_process_center',
                        lambda self, buffer, pos, mask: buffer[pos].original)
    saved = []
    monkeypatch.setattr(image_saver, 'save_img', lambda *args, **kwargs: saved.append(args))
    configs = {'enable_satellite_clean': True, 'sat_window_size': 3,
               'output_dir': str(tmp_path)}
    spec = meta_resolve(_load_yaml('hoshicore/dag/timelapse_startrail.meta.yaml'),
                        {'mode': 'sliding_max'}, configs)
    dag = validate_and_build_order(flatten_sub_dags(spec))
    await asyncio.wait_for(run_dag(dag, {'fnames': [f'{i}.jpg' for i in range(8)]},
                                  configs, progress=False), 10)
    assert len(saved) == 8
