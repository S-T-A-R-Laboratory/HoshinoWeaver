"""Optional ONNX backend; model-specific image preparation belongs to its Op."""
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .registry import SerialService, ServiceKey


class OnnxService(SerialService):
    def __init__(self, settings):
        super().__init__()
        self._settings = settings
        self._session = None

    def _infer(self, inputs, output_names):
        if self._session is None:
            if not self._settings.get('model_path'):
                raise ValueError('segmentation_model_path is required for ONNX inference')
            try:
                import onnxruntime as ort
            except ImportError as error:
                raise RuntimeError('ONNX inference requires onnxruntime') from error
            self._session = ort.InferenceSession(
                self._settings['model_path'], providers=self._settings['providers'])
        return self._session.run(output_names, inputs)

    async def infer(self, inputs: Mapping[str, Any],
                    output_names: Sequence[str] | None = None) -> list[Any]:
        """Named tensors in, requested tensors out; requests execute serially."""
        return await self._run(self._infer, dict(inputs),
                               list(output_names) if output_names is not None else None)

    def _close(self):
        # InferenceSession has no public close(); release it on its worker.
        self._session = None


SEGMENTATION = ServiceKey('segmentation', OnnxService)


def segmentation_settings(configs):
    path = configs.get('segmentation_model_path')
    return {'model_path': os.path.normcase(str(Path(path).expanduser().resolve())) if path else None,
            'providers': configs.get('segmentation_providers', ['CPUExecutionProvider'])}
