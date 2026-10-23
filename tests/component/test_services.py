"""Service lifetimes and extension points; no Photoshop, ORT or models required."""
import asyncio
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from hoshicore.component import dataloader
from hoshicore.services import photoshop as photoshop_service
from hoshicore.engine.build import validate_and_build_order
from hoshicore.engine.wiring import run_dag, run_from_yaml
from hoshicore.ops.base import BaseOp
from hoshicore.services import (SerialService, ServiceKey, ServiceRegistry,
                                create_service_registry)


class EchoService(SerialService):
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.closed = False

    async def echo(self, value):
        return await self._run(lambda: value)

    def _close(self):
        self.closed = True


ECHO = ServiceKey('extension.echo', EchoService)


@pytest.mark.asyncio
async def test_extension_registration_is_lazy_typed_and_uses_a_settings_snapshot():
    registry = ServiceRegistry()
    factory = Mock(side_effect=EchoService)
    registry.register(ECHO, factory)
    settings = {'model': ['original']}
    async with registry.scope(settings) as scope:
        factory.assert_not_called()
        settings['model'][0] = 'changed'
        service = scope.require(ECHO)
        assert scope.require(ECHO) is service
        assert await service.echo(42) == 42
        assert service.settings == {'model': ['original']}
        with pytest.raises(LookupError, match='not registered'):
            scope.require(ServiceKey('missing', EchoService))
        with pytest.raises(TypeError, match='type mismatch'):
            scope.require(ServiceKey(ECHO.name, SerialService))
    assert service.closed
    with pytest.raises(RuntimeError, match='scope is closed'):
        scope.require(ECHO)
    with pytest.raises(RuntimeError, match='released'):
        await service.echo(1)
    with pytest.raises(ValueError, match='already registered'):
        registry.register(ECHO, factory)


@pytest.mark.asyncio
async def test_overlapping_scopes_share_until_last_release_and_keep_options_isolated():
    registry = ServiceRegistry()
    registry.register(ECHO, EchoService, settings=lambda c: {'model': c['model']})
    first = registry.scope({'model': 'a', 'unrelated': 1})
    second = registry.scope({'model': 'a', 'unrelated': 2})
    other = registry.scope({'model': 'b'})
    a = first.require(ECHO)
    assert second.require(ECHO) is a
    b = other.require(ECHO)
    assert b is not a
    await first.aclose()
    assert not a.closed and await second.require(ECHO).echo('live') == 'live'
    await second.aclose()
    await other.aclose()
    assert a.closed and b.closed


@pytest.mark.asyncio
async def test_next_scope_waits_for_previous_resource_exit():
    closing, release = threading.Event(), threading.Event()

    class SlowClose(EchoService):
        def _close(self):
            closing.set()
            assert release.wait(3)
            super()._close()

    key = ServiceKey('slow', SlowClose)
    registry = ServiceRegistry()
    registry.register(key, SlowClose)
    first = registry.scope()
    original = first.require(key)
    close = asyncio.create_task(first.aclose())
    assert await asyncio.to_thread(closing.wait, 3)
    second = registry.scope()
    replacement = second.require(key)
    work = asyncio.create_task(replacement.echo('new'))
    try:
        await asyncio.sleep(0.02)
        assert not work.done() and not original.closed
    finally:
        release.set()
    await close
    assert await work == 'new' and original.closed
    await second.aclose()


@pytest.mark.asyncio
async def test_same_photoshop_executable_with_conflicting_options_is_not_a_second_owner():
    registry = create_service_registry()
    async with registry.scope({'photoshop_path': 'mock', 'photoshop_hide': True}) as first:
        first.photoshop
        async with registry.scope({'photoshop_path': str(Path('mock').resolve()),
                                   'photoshop_hide': False}) as second:
            with pytest.raises(ValueError, match='Conflicting settings'):
                second.photoshop


@pytest.mark.asyncio
async def test_multiple_loaders_share_serialization_and_survive_loader_close(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    calls = []

    def render(path):
        calls.append(path)
        if len(calls) == 1:
            entered.set()
            assert release.wait(3)
        return path, None

    renderer = Mock(render=Mock(side_effect=render))
    constructor = Mock(return_value=renderer)
    monkeypatch.setattr(photoshop_service, 'PhotoshopRawRenderer', constructor)
    async with create_service_registry().scope({'photoshop_path': 'mock'}) as services:
        sources = [dataloader.ImgFileListLoader(None, None, {'raw_backend': 'photoshop'},
                                               services=services) for _ in range(2)]
        first = asyncio.create_task(sources[0].aload('a.CR2'))
        assert await asyncio.to_thread(entered.wait, 3)
        second = asyncio.create_task(sources[1].aload('b.CR2'))
        try:
            await asyncio.sleep(0.02)
            assert calls == ['a.CR2']
        finally:
            release.set()
        await first
        await sources[0].aclose()
        renderer.close.assert_not_called()
        assert await second == ('b.CR2', None)
        assert await sources[1].aload('c.CR2') == ('c.CR2', None)
    constructor.assert_called_once()
    renderer.close.assert_called_once()


@pytest.mark.asyncio
async def test_onnx_session_is_lazy_serial_shared_and_released(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    calls = []

    def run(outputs, inputs):
        calls.append((outputs, inputs))
        if len(calls) == 1:
            entered.set()
            assert release.wait(3)
        return [inputs['image']]

    constructor = Mock(return_value=SimpleNamespace(run=run))
    monkeypatch.setitem(sys.modules, 'onnxruntime', SimpleNamespace(InferenceSession=constructor))
    registry = create_service_registry()
    settings = {'segmentation_model_path': 'mock.onnx'}
    second_scope = registry.scope(settings)
    async with registry.scope(settings) as scope:
        service = scope.segmentation
        assert second_scope.segmentation is service
        constructor.assert_not_called()
        first = asyncio.create_task(service.infer({'image': 1}))
        assert await asyncio.to_thread(entered.wait, 3)
        second = asyncio.create_task(service.infer({'image': 2}, ['mask']))
        try:
            await asyncio.sleep(0.02)
            assert len(calls) == 1
        finally:
            release.set()
        assert await asyncio.gather(first, second) == [[1], [2]]
        constructor.assert_called_once_with(os.path.normcase(str(Path('mock.onnx').resolve())),
                                            providers=['CPUExecutionProvider'])
        assert calls == [(None, {'image': 1}), (['mask'], {'image': 2})]
    assert service._session is not None
    await second_scope.aclose()
    assert service._session is None


@pytest.mark.asyncio
async def test_unused_services_do_not_copy_large_loader_payloads_or_import_ort(monkeypatch):
    class LoaderPayload:
        def __deepcopy__(self, memo):
            raise AssertionError('Unrelated frame payload must not be copied')

    monkeypatch.setitem(sys.modules, 'onnxruntime', None)
    async with create_service_registry().scope({'loader_configs': LoaderPayload()}):
        pass


@pytest.mark.asyncio
async def test_yaml_subgraphs_inject_same_scope_and_custom_typed_service(tmp_path):
    scopes, services = [], []

    class EchoOp(BaseOp):
        OUTPUTS = {'result': {'type': 'data'}}

        async def _async_execute(self, configs):
            scopes.append(self.services)
            service = self.services.require(ECHO)
            services.append(service)
            await self._broadcast_outputs({'result': await service.echo('ok')})

    child = tmp_path / 'child.yaml'
    child.write_text(yaml.safe_dump({
        'nodes': {'echo': {'op': 'EchoOp', 'outputs': {'result': {'type': 'data'}}}},
        'outputs': {'result': 'echo.result'}}), encoding='utf-8')
    main = tmp_path / 'main.yaml'
    main.write_text(yaml.safe_dump({
        'configs': {'model': {'type': 'str', 'default': 'my-model'}},
        'nodes': {name: {'op': str(child), 'outputs': {'result': {'type': 'data'}}}
                  for name in ('first', 'second')},
        'outputs': {'first': 'first.result', 'second': 'second.result'}}), encoding='utf-8')
    registry = ServiceRegistry()
    factory = Mock(side_effect=EchoService)
    registry.register(ECHO, factory, settings=lambda c: {'model': c.get('model')})
    result = await run_from_yaml(str(main), {}, {}, progress=False,
                                  op_registry={'EchoOp': EchoOp}, service_registry=registry)
    assert result == {'first': 'ok', 'second': 'ok'}
    assert len(scopes) == 2 and scopes[0] is scopes[1]
    assert services[0] is services[1] and services[0].closed
    factory.assert_called_once_with({'model': 'my-model'})


def loader_dag():
    return validate_and_build_order({
        'inputs': {'src': {'type': 'sequence'}}, 'configs': {},
        'nodes': {'loader': {'op': 'ImgDataLoaderOp', 'inputs': {'src': 'inputs.src'},
                             'outputs': {'result': {'type': 'sequence'}}}},
        'outputs': {'result': 'loader.result'}})


@pytest.mark.asyncio
async def test_graph_cancellation_drains_native_work_before_service_cleanup(monkeypatch):
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    def render(path):
        entered.set()
        assert release.wait(3)
        finished.set()
        return path, None

    def close():
        assert finished.is_set()

    renderer = Mock(render=Mock(side_effect=render), close=Mock(side_effect=close))
    monkeypatch.setattr(photoshop_service, 'PhotoshopRawRenderer', Mock(return_value=renderer))
    graph = asyncio.create_task(run_dag(loader_dag(), {'src': ['a.CR2', 'b.CR2']},
                                       {'raw_backend': 'photoshop'}, progress=False,
                                       service_registry=create_service_registry()))
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        graph.cancel()
        await asyncio.sleep(0)
        graph.cancel()
        await asyncio.sleep(0.02)
        assert not graph.done()
        renderer.close.assert_not_called()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await graph
    renderer.render.assert_called_once_with('a.CR2')
    renderer.close.assert_called_once()
