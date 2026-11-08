"""Provider selection stays lazy and never overrides an explicit choice."""
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hoshicore.component.onnx_backend import OnnxBackend
from hoshicore.services.onnx import onnx_settings
from hoshicore.services.segmentation import segmentation_settings


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    model = tmp_path / 'model.onnx'
    model.touch()
    sessions = []

    def construct(path, *, providers, sess_options):
        name = providers[0] if isinstance(providers[0], str) else providers[0][0]
        actual = [name] if name == 'CPUExecutionProvider' else [name, 'CPUExecutionProvider']
        session = SimpleNamespace(get_providers=lambda: actual, disable_fallback=Mock(),
                                  run=Mock(return_value=['result']))
        sessions.append(session)
        return session

    ort = SimpleNamespace(
        InferenceSession=Mock(side_effect=construct), SessionOptions=SimpleNamespace,
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL='sequential'),
        get_available_providers=Mock(return_value=['CPUExecutionProvider', 'DmlExecutionProvider']))
    monkeypatch.setitem(sys.modules, 'onnxruntime', ort)
    logger = Mock()
    monkeypatch.setattr('hoshicore.component.onnx_backend.logger', logger)
    return model, ort, sessions, logger


@pytest.mark.parametrize('available,expected', [
    (['CPUExecutionProvider'], 'CPUExecutionProvider'),
    (['CPUExecutionProvider', 'DmlExecutionProvider'], 'DmlExecutionProvider'),
    (['CPUExecutionProvider', 'DmlExecutionProvider', 'CUDAExecutionProvider'], 'CUDAExecutionProvider'),
    (['CPUExecutionProvider', 'CoreMLExecutionProvider'], 'CoreMLExecutionProvider'),
    (['CPUExecutionProvider', 'ROCMExecutionProvider'], 'ROCMExecutionProvider'),
])
def test_auto_selects_accelerator_lazily(runtime, available, expected):
    model, ort, sessions, logger = runtime
    ort.get_available_providers.return_value = available
    backend = OnnxBackend({'model_path': str(model)})
    ort.get_available_providers.assert_not_called()
    assert backend.infer({'x': 1}) == ['result']
    assert backend.infer({'x': 2}) == ['result']
    ort.InferenceSession.assert_called_once()
    assert ort.InferenceSession.call_args.kwargs['providers'] == [expected]
    sessions[0].disable_fallback.assert_called_once()
    if expected == 'DmlExecutionProvider':
        options = ort.InferenceSession.call_args.kwargs['sess_options']
        assert options.enable_mem_pattern is False
        assert options.execution_mode == 'sequential'
    assert logger.info.call_args.args[1] == 'auto'
    assert expected in logger.info.call_args.args[3]
    backend.close()


@pytest.mark.parametrize('providers', [['CPUExecutionProvider'],
    [('DmlExecutionProvider', {'device_id': 1}), 'CPUExecutionProvider']])
def test_explicit_provider_options_and_cpu_override(runtime, providers):
    model, ort, _, logger = runtime
    backend = OnnxBackend({'model_path': str(model), 'providers': providers})
    backend.infer({})
    assert ort.InferenceSession.call_args.kwargs['providers'] == providers
    assert logger.info.call_args.args[1] == 'explicit'


@pytest.mark.parametrize('silent_cpu', [False, True])
def test_auto_initialization_failure_is_logged_and_retries(runtime, silent_cpu):
    model, ort, _, logger = runtime
    construct = ort.InferenceSession.side_effect

    def failing(path, **kwargs):
        if kwargs['providers'][0] == 'DmlExecutionProvider':
            if silent_cpu:
                return construct(path, providers=['CPUExecutionProvider'], sess_options=kwargs['sess_options'])
            raise RuntimeError('device unavailable')
        return construct(path, **kwargs)

    ort.InferenceSession.side_effect = failing
    backend = OnnxBackend({'model_path': str(model)})
    assert backend.infer({}) == ['result']
    assert [call.kwargs['providers'] for call in ort.InferenceSession.call_args_list] == [
        ['DmlExecutionProvider'], ['CPUExecutionProvider']]
    logger.warning.assert_called_once()
    assert logger.info.call_args.args[3] == ['CPUExecutionProvider']


def test_explicit_initialization_failure_never_retries(runtime):
    model, ort, _, logger = runtime
    ort.InferenceSession.side_effect = RuntimeError('device unavailable')
    backend = OnnxBackend({'model_path': str(model), 'providers': ['DmlExecutionProvider']})
    with pytest.raises(RuntimeError, match='device unavailable'):
        backend.infer({})
    ort.InferenceSession.assert_called_once()
    logger.warning.assert_not_called()


@pytest.mark.parametrize('providers', [[], 'DmlExecutionProvider', ['CUDAExecutionProvider']])
def test_invalid_or_unavailable_explicit_selection_is_rejected(runtime, providers):
    model, ort, _, _ = runtime
    backend = OnnxBackend({'model_path': str(model), 'providers': providers})
    with pytest.raises(ValueError):
        backend.infer({})
    ort.InferenceSession.assert_not_called()


def test_inference_errors_do_not_trigger_provider_switch(runtime):
    model, ort, sessions, _ = runtime
    backend = OnnxBackend({'model_path': str(model)})
    backend.infer({})
    sessions[0].run.side_effect = RuntimeError('inference failed')
    with pytest.raises(RuntimeError, match='inference failed'):
        backend.infer({})
    ort.InferenceSession.assert_called_once()


@pytest.mark.parametrize('selector,key', [(onnx_settings, 'onnx_providers'),
                                         (segmentation_settings, 'segmentation_providers')])
def test_service_settings_preserve_explicit_choice_without_importing_ort(monkeypatch, selector, key):
    monkeypatch.setitem(sys.modules, 'onnxruntime', None)
    assert selector({})['providers'] is None
    assert selector({key: ['CPUExecutionProvider']})['providers'] == ['CPUExecutionProvider']
