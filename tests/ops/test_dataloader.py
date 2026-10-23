"""Loader and DAG flow tests, with the external renderer replaced by a mock."""
import asyncio
from pathlib import Path
import threading
from unittest.mock import Mock

import numpy as np
import pytest

from hoshicore.component import dataloader
from hoshicore.services import photoshop as photoshop_service
from hoshicore.component.exif import ExifData
from hoshicore.component.queue import BaseQueue, RichContextQueue, StreamExhausted
from hoshicore.engine.build import _load_yaml, validate_and_build_order
from hoshicore.engine.executor import DAGExecutionError
from hoshicore.engine.flatten import flatten_sub_dags
from hoshicore.engine.meta import meta_resolve
from hoshicore.engine.wiring import run_dag
from hoshicore.ops.dataloader import ImgDataLoaderOp
from hoshicore.services import create_service_registry


ROOT = Path(__file__).resolve().parents[2]


async def run_loader(paths, *, length, options=None, metadata=True):
    op = ImgDataLoaderOp('loader')
    sinks = {name: RichContextQueue(maxsize=1)
             for name in (('result', 'exifs') if metadata else ('result',))}
    for name, sink in sinks.items():
        op.outputs[name].append(sink)
    configs = {name: entry['default'] for name, entry in op.CONFIGS.items()}
    configs.update(options or {})
    for name, value in configs.items():
        if name in op.config:
            await op.config[name].put(value)

    async def feed():
        await op.inputs['src'].set_length(length)
        for path in paths:
            await op.inputs['src'].put(path)
        if length is None:
            await op.inputs['src'].put(BaseQueue._SENTINEL)

    async def drain(sink):
        assert await sink.get_length() == length
        frames = []
        try:
            while True:
                frames.append(await sink.get())
        except StreamExhausted:
            return frames

    async with create_service_registry().scope(configs) as services:
        op._services = services
        results = await asyncio.wait_for(asyncio.gather(
            feed(), op.execute(), *(drain(sink) for sink in sinks.values())), 5)
    return dict(zip(sinks, results[2:]))


@pytest.mark.asyncio
@pytest.mark.parametrize('length', [8, None, 0])
async def test_native_results_and_metadata_stay_ordered_with_bounded_queues(monkeypatch, length):
    paths = [f'{i}.jpg' for i in range(8)] if length != 0 else []
    barrier = threading.Barrier(4, timeout=3)

    def read(path):
        barrier.wait()
        return path

    monkeypatch.setattr(dataloader, 'load_img', read)
    monkeypatch.setattr(dataloader, 'read_exif_data', lambda p: ExifData({'source': p}, b'icc'))
    results = await run_loader(paths, length=length, options={'raw_backend': 'photoshop'})
    assert results['result'] == paths
    assert [e.exif['source'] for e in results['exifs']] == paths


@pytest.mark.asyncio
async def test_array_loader_uses_its_persistent_configured_instance():
    frames = [np.full((2, 3, 3), i, np.uint8) for i in range(3)]
    results = await run_loader([2, 0, 1], length=3, options={
        'loader_type': 'img_array', 'configs': {'data': frames}})
    assert all(frame is frames[index] for frame, index in zip(results['result'], [2, 0, 1]))
    assert results['exifs'] == [None, None, None]


@pytest.mark.asyncio
async def test_render_failure_is_not_fallen_back_and_cleanup_preserves_root_error(monkeypatch):
    renderer = Mock(render=Mock(side_effect=RuntimeError('ACR opening failed')),
                    close=Mock(side_effect=RuntimeError('exit also failed')))
    monkeypatch.setattr(photoshop_service, 'PhotoshopRawRenderer', Mock(return_value=renderer))
    native = Mock(side_effect=AssertionError('must not fall back to rawpy'))
    monkeypatch.setattr(dataloader, 'load_img', native)
    spec = {
        'inputs': {'src': {'type': 'sequence'}},
        'configs': {},
        'nodes': {'loader': {'op': 'ImgDataLoaderOp', 'inputs': {'src': 'inputs.src'},
                             'outputs': {'result': {'type': 'sequence'}}}},
        'outputs': {'result': 'loader.result'},
    }
    dag = validate_and_build_order(spec)
    with pytest.raises(DAGExecutionError) as error:
        await asyncio.wait_for(run_dag(dag, {'src': ['bad.CR2']},
                                      {'raw_backend': 'photoshop'}, progress=False), 5)
    assert error.value.root_node == 'loader'
    assert str(error.value.root_cause) == 'ACR opening failed'
    renderer.close.assert_called_once()
    native.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize('name,routes', [
    ('stack', {'stacker': 'mean'}),
    ('startrail', {'mode': 'fifo'}),
    ('timelapse_startrail', {'mode': 'sliding_max'}),
])
async def test_builtin_workflow_uses_rendered_metadata_through_to_saving(monkeypatch, tmp_path,
                                                                      name, routes):
    from hoshicore.ops import image_saver

    renderer = Mock(render=Mock(side_effect=lambda path: (
        np.full((4, 5, 3), 1000, np.uint16),
        ExifData({'Exif.Photo.ExposureTime': '1/8'}, b'rendered-profile'))))
    constructor = Mock(return_value=renderer)
    monkeypatch.setattr(photoshop_service, 'PhotoshopRawRenderer', constructor)
    monkeypatch.setattr(dataloader, 'read_exif_data',
                        Mock(side_effect=AssertionError('must use rendered metadata')))
    saved = []

    def save(path, image, **kwargs):
        saved.append((path, kwargs['exif']))
        return 0

    monkeypatch.setattr(image_saver, 'save_img', save)
    configs = {'raw_backend': 'photoshop', 'photoshop_path': 'mock',
               'output_filename': str(tmp_path / 'out.tif'),
               'output_dir': str(tmp_path), 'window_size': 1}
    spec = meta_resolve(_load_yaml(str(ROOT / f'hoshicore/dag/{name}.meta.yaml')),
                        routes, configs)
    dag = validate_and_build_order(flatten_sub_dags(spec))
    await asyncio.wait_for(run_dag(dag, {'fnames': ['a.CR2', 'b.CR2']},
                                  configs, progress=False), 5)
    assert saved and all(exif.colorprofile == b'rendered-profile' for _, exif in saved)
    if name != 'timelapse_startrail':
        assert saved[0][1].exif['Exif.Photo.ExposureTime'] == '1/4'
    constructor.assert_called_once()
    assert renderer.render.call_count == 2
    renderer.close.assert_called_once()


@pytest.mark.asyncio
async def test_selected_photoshop_without_raw_needs_no_executable_or_metadata(monkeypatch):
    constructor = Mock(side_effect=AssertionError('no RAW means no Photoshop'))
    monkeypatch.setattr(photoshop_service, 'PhotoshopRawRenderer', constructor)
    monkeypatch.setattr(dataloader, 'load_img', lambda p: p)
    monkeypatch.setattr(dataloader, 'read_exif_data', Mock(side_effect=AssertionError('unused EXIF')))
    paths = ['a.jpg', 'b.tif', 'c.png']
    assert (await run_loader(paths, length=3, options={'raw_backend': 'photoshop',
                                                    'photoshop_path': None}, metadata=False)
            )['result'] == paths
    constructor.assert_not_called()


@pytest.mark.parametrize('name', ['stack', 'startrail', 'timelapse_startrail'])
def test_gui_exposes_raw_backend_as_config_and_keeps_path_hidden(name):
    pytest.importorskip('PySide6')
    from ui.panel_builder import PanelSchema

    path = ROOT / f'hoshicore/dag/{name}.meta.yaml'
    schema = PanelSchema.from_yaml(path, path.with_name(f'{name}.ui.yaml'))
    configs = {entry.key: entry for entry in schema.configs}
    assert configs['raw_backend'].default == 'native'
    assert configs['raw_backend'].widget == 'select'
    assert configs['photoshop_path'].hidden
    assert 'raw_backend' not in schema.routes


@pytest.mark.asyncio
@pytest.mark.parametrize('name,routes', [('stack', {'stacker': 'mean'}),
                                        ('startrail', {'mode': 'fifo'})])
async def test_disabled_exif_does_not_read_metadata_or_start_ps(monkeypatch, name, routes):
    from hoshicore.ops import image_saver

    monkeypatch.setattr(dataloader, 'load_img', lambda path: np.zeros((2, 3, 3), np.uint8))
    monkeypatch.setattr(dataloader, 'read_exif_data',
                        Mock(side_effect=AssertionError('EXIF disabled')))
    constructor = Mock(side_effect=AssertionError('no RAW'))
    monkeypatch.setattr(photoshop_service, 'PhotoshopRawRenderer', constructor)
    saver = Mock(return_value=0)
    monkeypatch.setattr(image_saver, 'save_img', saver)
    configs = {'enable_exif': False, 'raw_backend': 'photoshop', 'output_filename': 'out.tif'}
    spec = meta_resolve(_load_yaml(str(ROOT / f'hoshicore/dag/{name}.meta.yaml')), routes, configs)
    dag = validate_and_build_order(flatten_sub_dags(spec))
    await asyncio.wait_for(run_dag(dag, {'fnames': ['input.jpg']}, configs, progress=False), 5)
    saver.assert_called_once()
    assert saver.call_args.kwargs['exif'] is None
    constructor.assert_not_called()
