"""Synchronous ONNX resource backend; its owner supplies execution and lifetime.

All methods run on the owning service's worker. Native sessions never cross
the public interface; importing this module does not import ONNX Runtime.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class TensorDescription:
    name: str
    type: str
    shape: tuple[int | str | None, ...] | None


@dataclass(frozen=True)
class ModelDescription:
    inputs: tuple[TensorDescription, ...]
    outputs: tuple[TensorDescription, ...]
    metadata: Mapping[str, str]
    providers: tuple[str, ...]


class InferenceBackend(Protocol):
    """Small synchronous interface used by model adapters, without raw ORT."""

    def describe(self) -> ModelDescription:
        ...

    def infer(self,
              inputs: Mapping[str, Any],
              output_names: Sequence[str] | None = None) -> list[Any]:
        ...


class OnnxBackend(InferenceBackend):

    def __init__(self, settings):
        self._settings = settings
        self._session = None

    def _get_session(self):
        if self._session is None:
            path = self._settings.get('model_path')
            if not path:
                raise ValueError('ONNX model_path is required')
            if not Path(path).is_file():
                raise FileNotFoundError(f'ONNX model not found: {path}')
            try:
                import onnxruntime as ort
            except ImportError as error:
                raise RuntimeError(
                    'ONNX inference requires onnxruntime') from error
            providers = self._settings.get('providers',
                                           ['CPUExecutionProvider'])
            names = [p if isinstance(p, str) else p[0] for p in providers]
            available = ort.get_available_providers()
            if not names or any(p not in available for p in names):
                raise ValueError(
                    f'ONNX providers {names} unavailable; installed: {available}'
                )
            threads = self._settings.get('threads', 4)
            if isinstance(threads,
                          bool) or not isinstance(threads, int) or threads < 1:
                raise ValueError('ONNX threads must be a positive integer')
            options = ort.SessionOptions()
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
            if 'DmlExecutionProvider' in names:
                options.enable_mem_pattern = False
                options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            session = ort.InferenceSession(path,
                                           sess_options=options,
                                           providers=providers)
            session.disable_fallback()
            if names[0] not in session.get_providers():
                raise RuntimeError(
                    f'ONNX provider {names[0]} failed to initialize')
            self._session = session
        return self._session

    def describe(self) -> ModelDescription:
        session = self._get_session()

        def tensors(items):
            return tuple(
                TensorDescription(
                    item.name, item.type,
                    tuple(item.shape) if item.shape is not None else None)
                for item in items)

        return ModelDescription(
            tensors(session.get_inputs()), tensors(session.get_outputs()),
            dict(session.get_modelmeta().custom_metadata_map),
            tuple(session.get_providers()))

    def infer(self,
              inputs: Mapping[str, Any],
              output_names: Sequence[str] | None = None) -> list[Any]:
        return self._get_session().run(
            list(output_names) if output_names is not None else None,
            dict(inputs))

    def close(self):
        # InferenceSession has no public close(); release on the owning worker.
        self._session = None
