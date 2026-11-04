"""Sky segmentation capability owning one backend and one serial worker."""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from ..component.onnx_backend import OnnxBackend
from .registry import SerialService, ServiceKey

if TYPE_CHECKING:
    import numpy as np
    from ..component.sky_segmentation import SkyPrediction


class SkySegmentationService(SerialService):
    def __init__(self, settings):
        super().__init__()
        self._backend = OnnxBackend(settings)
        self._predictor = None

    def _predict_sky(self, image, scales, threshold, color_order):
        if self._predictor is None:
            from ..component.sky_segmentation import SkySegPredictor
            self._predictor = SkySegPredictor(self._backend)
        return self._predictor.predict(image, scales, threshold, color_order)

    async def predict_sky(self, image: np.ndarray, *, scales=(512, 768, 1024), threshold=.5,
                          color_order='BGR') -> SkyPrediction:
        """Predict from uint8 sRGB pixels; keep input unchanged until return.

        Preparation, the entire pyramid and postprocessing run as one worker
        request. The scope alone closes this service and its backend.
        """
        return await self._run(self._predict_sky, image, tuple(scales), threshold, color_order)

    def _close(self):
        self._predictor = None
        self._backend.close()


SEGMENTATION = ServiceKey('segmentation', SkySegmentationService)


def segmentation_settings(configs):
    path = configs.get('segmentation_model_path')
    if not path:
        from ..component.utils import _EXE_ROOT
        path = _EXE_ROOT / 'models/skyseg_v1/skyseg_v1_fp16.onnx'
    return {'model_path': os.path.normcase(str(Path(path).expanduser().resolve())),
            'providers': configs.get('segmentation_providers', ['CPUExecutionProvider']),
            'threads': configs.get('segmentation_threads', 4)}
