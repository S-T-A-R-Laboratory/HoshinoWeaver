"""SkySeg inference contract without a model download or optional runtime."""
import json
import asyncio
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from hoshicore.component.sky_segmentation import SkySegPredictor, sky_probability
from hoshicore.component.onnx_backend import ModelDescription, TensorDescription
from hoshicore.services import create_service_registry


class FakeSession:
    def __init__(self):
        self.tensors = []
        self.metadata = dict(classes=['non_sky', 'sky'], color_order='RGB',
                             mean=[123.675, 116.28, 103.53],
                             std=[58.395, 57.12, 57.375], size_divisor=32)

    def get_inputs(self):
        return [SimpleNamespace(name='image', type='tensor(float16)', shape=[1, 3, 'h', 'w'])]

    def get_outputs(self):
        return [SimpleNamespace(name='logits', type='tensor(float)', shape=[1, 2, 'h', 'w'])]

    def get_modelmeta(self):
        return SimpleNamespace(custom_metadata_map={'skyseg': json.dumps(self.metadata)})

    def get_providers(self):
        return ['CPUExecutionProvider']

    def disable_fallback(self):
        pass

    def run(self, names, inputs):
        assert names == ['logits'] and set(inputs) == {'image'}
        tensor = inputs['image']
        self.tensors.append(tensor.copy())
        # Deliberately different per scale; average probabilities, not logits.
        height, width = tensor.shape[2:]
        values = np.zeros((1, 2, height, width), np.float32)
        values[0, 1] = height / 32
        return [values]


class FakeBackend:
    """Adapter tests use the backend contract, with no Session in the adapter."""

    def __init__(self, session):
        self.session = session

    def describe(self):
        return ModelDescription(
            (TensorDescription('image', 'tensor(float16)', (1, 3, 'h', 'w')),),
            (TensorDescription('logits', 'tensor(float)', (1, 2, 'h', 'w')),),
            {'skyseg': json.dumps(self.session.metadata)}, ('CPUExecutionProvider',))

    def infer(self, inputs, output_names=None):
        return self.session.run(output_names, inputs)


def test_two_stage_resize_normalization_color_order_and_probability_pyramid():
    session = FakeSession()
    predictor = SkySegPredictor(FakeBackend(session))
    image = np.random.default_rng(8).integers(0, 256, (19, 41, 3), dtype=np.uint8)
    prediction = predictor.predict(image, scales=(32, 96), threshold=.8)
    rgb = image[:, :, ::-1]
    # 19x41 -> 15x32 -> 32x32 and 44x96 -> 64x96, both via bilinear.
    assert [x.shape for x in session.tensors] == [(1, 3, 32, 32), (1, 3, 64, 96)]
    intermediate = cv2.resize(rgb, (32, 15), interpolation=cv2.INTER_LINEAR)
    resized = cv2.resize(intermediate, (32, 32), interpolation=cv2.INTER_LINEAR)
    expected = ((resized.astype(np.float32) - np.array(session.metadata['mean'])) /
                np.array(session.metadata['std'])).astype(np.float16)
    np.testing.assert_allclose(session.tensors[0][0].transpose(1, 2, 0), expected, atol=.002)
    assert session.tensors[0].dtype == np.float16
    assert session.tensors[0].flags.c_contiguous
    expected_probability = (1 / (1 + np.exp(-1)) + 1 / (1 + np.exp(-2))) / 2
    np.testing.assert_allclose(prediction.probability, expected_probability, atol=1e-7)
    assert prediction.probability.shape == (19, 41)
    assert prediction.probability.dtype == np.float32
    assert prediction.mask.dtype == np.uint8 and prediction.mask.all()
    repeated = predictor.predict(rgb, scales=(32, 96), color_order='RGB')
    np.testing.assert_array_equal(session.tensors[0], session.tensors[2])
    np.testing.assert_array_equal(prediction.probability, repeated.probability)


def test_resize_logits_before_softmax_and_extreme_logits_are_finite():
    logits = np.array([[[[0, 0]], [[-2, 6]]]], np.float32)
    actual = sky_probability(logits, (1, 3))
    assert actual[0, 1] == pytest.approx(1 / (1 + np.exp(-2)))
    assert not np.isclose(actual[0, 1], (actual[0, 0] + actual[0, 2]) / 2)
    logits[0, 1] = [-1000, 1000]
    assert np.isfinite(sky_probability(logits, (1, 3))).all()
    logits[0, 1, 0, 0] = np.nan
    with pytest.raises(RuntimeError, match='finite'):
        sky_probability(logits, (1, 3))


@pytest.mark.parametrize('options', [dict(scales=()), dict(scales=(True,)),
    dict(scales=(32, 32)), dict(scales=(31,)), dict(threshold=float('nan')),
    dict(threshold=-.1), dict(color_order='xyz')])
def test_reject_invalid_request(options):
    session = FakeSession()
    with pytest.raises(ValueError):
        SkySegPredictor(FakeBackend(session)).predict(np.zeros((4, 5, 3), np.uint8), **options)
    assert not session.tensors


def test_reject_incompatible_models_and_ambiguous_pixel_ranges():
    session = FakeSession()
    session.metadata['classes'] = ['sky', 'non_sky']
    with pytest.raises(ValueError, match='class order'):
        SkySegPredictor(FakeBackend(session))
    session.metadata['classes'] = ['non_sky', 'sky']
    predictor = SkySegPredictor(FakeBackend(session))
    for image in (np.zeros((2, 3, 3), np.uint16), np.zeros((2, 3, 3), np.float32),
                  np.zeros((2, 3), np.uint8), np.zeros((0, 3, 3), np.uint8)):
        with pytest.raises(ValueError, match='uint8'):
            predictor.predict(image)


@pytest.mark.asyncio
async def test_segmentation_reuses_one_backend_and_releases_predictor(monkeypatch, tmp_path):
    model = tmp_path / 'model.onnx'
    model.touch()
    session = FakeSession()
    constructor = Mock(return_value=session)
    monkeypatch.setitem(sys.modules, 'onnxruntime', SimpleNamespace(
        InferenceSession=constructor, SessionOptions=SimpleNamespace,
        get_available_providers=lambda: ['CPUExecutionProvider']))
    async with create_service_registry().scope({'segmentation_model_path': str(model)}) as scope:
        service = scope.segmentation
        constructor.assert_not_called()
        image = np.zeros((13, 17, 3), np.uint8)
        first = await service.predict_sky(image, scales=(32, 64))
        second = await scope.segmentation.predict_sky(image, scales=(32, 64))
        np.testing.assert_array_equal(first.probability, second.probability)
        constructor.assert_called_once()
        assert service._predictor is not None
    assert service._backend._session is None and service._predictor is None


@pytest.mark.asyncio
async def test_requested_provider_cannot_silently_fall_back(monkeypatch, tmp_path):
    model = tmp_path / 'model.onnx'
    model.touch()
    session = FakeSession()
    monkeypatch.setitem(sys.modules, 'onnxruntime', SimpleNamespace(
        InferenceSession=lambda *a, **kw: session, SessionOptions=SimpleNamespace,
        get_available_providers=lambda: ['CUDAExecutionProvider', 'CPUExecutionProvider']))
    async with create_service_registry().scope({
            'segmentation_model_path': str(model),
            'segmentation_providers': ['CUDAExecutionProvider', 'CPUExecutionProvider']}) as scope:
        with pytest.raises(RuntimeError, match='failed to initialize'):
            await scope.segmentation.predict_sky(np.zeros((4, 4, 3), np.uint8), scales=(32,))


@pytest.mark.asyncio
async def test_generic_and_sky_services_own_distinct_sessions_even_for_same_model(monkeypatch, tmp_path):
    model = tmp_path / 'sky.onnx'
    model.touch()
    sessions = [FakeSession(), FakeSession()]
    constructor = Mock(side_effect=sessions)
    monkeypatch.setitem(sys.modules, 'onnxruntime', SimpleNamespace(
        InferenceSession=constructor, SessionOptions=SimpleNamespace,
        get_available_providers=lambda: ['CPUExecutionProvider']))
    async with create_service_registry().scope({
            'onnx_model_path': str(model), 'segmentation_model_path': str(model),
            'onnx_threads': 2, 'segmentation_threads': 3}) as scope:
        generic, sky = scope.onnx, scope.segmentation
        await generic.describe()
        await sky.predict_sky(np.zeros((9, 13, 3), np.uint8), scales=(32,))
        assert generic._backend._session is sessions[0]
        assert sky._backend._session is sessions[1]
        assert not hasattr(sky, 'infer')
        assert not hasattr(generic, 'predict_sky')
        assert [call.kwargs['sess_options'].intra_op_num_threads
                for call in constructor.call_args_list] == [2, 3]
    assert generic._backend._session is None and sky._backend._session is None


@pytest.mark.asyncio
async def test_cancelled_pyramid_drains_all_scales_before_backend_close(monkeypatch, tmp_path):
    model = tmp_path / 'sky.onnx'
    model.touch()
    session = FakeSession()
    entered, release = threading.Event(), threading.Event()
    threads = []
    run = session.run

    def blocking_run(names, inputs):
        threads.append(threading.get_ident())
        if len(threads) == 1:
            entered.set()
            assert release.wait(5)
        return run(names, inputs)

    session.run = blocking_run
    monkeypatch.setitem(sys.modules, 'onnxruntime', SimpleNamespace(
        InferenceSession=lambda *a, **kw: session, SessionOptions=SimpleNamespace,
        get_available_providers=lambda: ['CPUExecutionProvider']))
    async with create_service_registry().scope({'segmentation_model_path': str(model)}) as scope:
        service = scope.segmentation
        close = service._backend.close
        closed_threads = []

        def checked_close():
            assert len(session.tensors) == 2
            closed_threads.append(threading.get_ident())
            close()

        monkeypatch.setattr(service._backend, 'close', checked_close)
        task = asyncio.create_task(service.predict_sky(np.zeros((9, 13, 3), np.uint8), scales=(32, 64)))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done() and not closed_threads
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert len(session.tensors) == 2
    assert set(threads) == set(closed_threads)
    assert service._predictor is None and service._backend._session is None
