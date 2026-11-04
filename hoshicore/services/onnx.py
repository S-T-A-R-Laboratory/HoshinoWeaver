"""Generic named-tensor service, with no model-specific defaults or adapters."""
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..component.onnx_backend import ModelDescription, OnnxBackend
from .registry import SerialService, ServiceKey


class OnnxService(SerialService):
    def __init__(self, settings):
        super().__init__()
        self._backend = OnnxBackend(settings)

    async def infer(self, inputs: Mapping[str, Any],
                    output_names: Sequence[str] | None = None) -> list[Any]:
        """Named tensors in, requested tensors out; requests execute serially."""
        return await self._run(self._backend.infer, dict(inputs),
                               list(output_names) if output_names is not None else None)

    async def describe(self) -> ModelDescription:
        """Read tensor signatures, metadata and providers on the worker."""
        return await self._run(self._backend.describe)

    def _close(self):
        self._backend.close()


ONNX = ServiceKey('onnx', OnnxService)


def onnx_settings(configs):
    path = configs.get('onnx_model_path')
    return {'model_path': os.path.normcase(str(Path(path).expanduser().resolve())) if path else None,
            'providers': configs.get('onnx_providers', ['CPUExecutionProvider']),
            'threads': configs.get('onnx_threads', 4)}
