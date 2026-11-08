"""Sky segmentation capability owning one backend and one serial worker."""
from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
from loguru import logger

from ..component.onnx_backend import OnnxBackend
from ..component.data_container import FloatImage
from ..component.model_metadata import load_model_metadata
from ..component.sky_segmentation import (SkyPrediction, SkyProgress,
                                          SkySegPredictor, report_progress)
from ..component.utils import _EXE_ROOT
from .registry import SerialService, ServiceKey


class SkySegmentationService(SerialService):
    def __init__(self, settings):
        super().__init__()
        self._backend = OnnxBackend(settings)
        self._model_path = settings['model_path']
        self._predictor = None

    def _predict_sky(self, image, scales, threshold, color_order, progress, strategy,
                     working_long_edge, value_range, linear, stretch_gamma):
        started = perf_counter()
        logger.info('SkySeg v2 inference started: shape={} scales={} threshold={} strategy={} working_long_edge={}',
                    getattr(image, 'shape', None), scales, threshold, strategy, working_long_edge)
        report_progress(progress, SkyProgress('initializing', 0, 0, strategy=strategy))
        try:
            if self._predictor is None:
                self._predictor = SkySegPredictor(self._backend, load_model_metadata(self._model_path))
            result = self._predictor.predict(image, scales, threshold, color_order, progress,
                                             strategy=strategy, working_long_edge=working_long_edge,
                                             value_range=value_range, linear=linear, stretch_gamma=stretch_gamma)
        except Exception as error:
            logger.warning('SkySeg inference failed after {:.3f}s: {}', perf_counter() - started, error)
            raise
        logger.info('SkySeg inference completed: providers={} total_seconds={:.3f}',
                    result.info['providers'], perf_counter() - started)
        return result

    async def predict_sky(self, image: np.ndarray | FloatImage, *, scales=(512, 768, 1024), threshold=.5,
                          color_order='BGR',
                          progress: Callable[[SkyProgress], None] | None = None,
                          strategy='roi', working_long_edge=1536,
                          value_range=None, linear=False, stretch_gamma=2.2) -> SkyPrediction:
        """Predict from HWC RGB/BGR unsigned integers or floats; keep input unchanged.

        Floats default to [0,1], integers to their full range. value_range sets
        black/white levels with clipping; FloatImage supplies its range dtype.
        linear=True applies x**(1/stretch_gamma)
        before RGB8 conversion. This bridge does not perform color management.

        Preparation, the global pyramid, selected ROI windows and fusion run as one worker
        request. The scope alone closes this service and its backend.
        Optional progress(SkyProgress) runs on the service worker; GUI callers
        must forward notifications to the UI thread. Observer errors are logged.
        """
        return await self._run(self._predict_sky, image, tuple(scales), threshold, color_order,
                               progress, strategy, working_long_edge,
                               None if value_range is None else tuple(value_range), linear, stretch_gamma)

    def _close(self):
        self._predictor = None
        self._backend.close()


SEGMENTATION = ServiceKey('segmentation', SkySegmentationService)


def segmentation_settings(configs):
    path = configs.get('segmentation_model_path')
    if not path:
        path = _EXE_ROOT / 'models/skyseg_v2/skyseg_v2_fp16.onnx'
    return {'model_path': os.path.normcase(str(Path(path).expanduser().resolve())),
            'providers': configs.get('segmentation_providers'),
            'threads': configs.get('segmentation_threads', 4)}
