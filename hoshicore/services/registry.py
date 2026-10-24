"""Typed services owned by one root DAG and shared by its nodes."""
from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Generic, TYPE_CHECKING, TypeVar, cast

from loguru import logger

if TYPE_CHECKING:
    from .onnx import OnnxService
    from .photoshop import PhotoshopService

T = TypeVar('T', bound='SerialService')
R = TypeVar('R')


async def _drain_on_cancel(future, cancel_pending=None):
    """A cancelled await must not abandon native work or resource cleanup."""
    try:
        return await asyncio.shield(future)
    except asyncio.CancelledError:
        if cancel_pending is not None:
            cancel_pending()
        while not future.done():
            try:
                await asyncio.shield(future)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not future.cancelled():
            future.exception()
        raise


class SerialService:
    """Blocking resource adapter. Construct cheaply; initialize in _run().

    Each resource has one worker, independent of asyncio's IO thread pool.
    Subclasses expose typed async methods and release native state in _close().
    The owning scope alone retires services; Ops must not close shared instances.
    """

    def __init__(self):
        self._worker = ThreadPoolExecutor(max_workers=1,
                                          thread_name_prefix='hoshi-service')
        self._retired = False

    async def _run(self, function: Callable[..., R], *args) -> R:
        if self._retired:
            raise RuntimeError('Service has been released')
        worker = self._worker.submit(function, *args)
        return await _drain_on_cancel(asyncio.wrap_future(worker), worker.cancel)

    def _close(self) -> None:
        """Release native state on the resource's worker thread."""

    def _retire(self) -> Future:
        self._retired = True
        future = self._worker.submit(self._close)
        self._worker.shutdown(wait=False)
        return future


@dataclass(frozen=True)
class ServiceKey(Generic[T]):
    name: str
    service_type: type[T]


@dataclass(frozen=True)
class _Registration:
    key: ServiceKey
    factory: Callable
    settings: Callable


class ServiceRegistry:
    """Service factories and configuration selectors; holds no live services.

    Register extensions before running a graph. Each scope snapshots creation
    options and owns its instances. Root graphs are run sequentially by callers.
    """

    def __init__(self):
        self._registrations: dict[str, _Registration] = {}

    def register(self, key: ServiceKey[T], factory: Callable[[Mapping[str, Any]], T],
                 *, settings: Callable = dict) -> None:
        if key.name in self._registrations:
            raise ValueError(f'Service already registered: {key.name}')
        self._registrations[key.name] = _Registration(key, factory, settings)

    def scope(self, settings: Mapping[str, Any] | None = None) -> ServiceScope:
        # Snapshot only service creation options, not unrelated loader arrays
        # or other potentially large graph configuration payloads.
        registrations = dict(self._registrations)
        options = {name: deepcopy(registration.settings(settings or {}))
                   for name, registration in registrations.items()}
        return ServiceScope(registrations, options)


class ServiceScope:
    """Owns lazy service instances shared by all nodes of one root DAG.

    Use on the graph's event loop; await node/native work before closing it.
    """

    def __init__(self, registrations, settings):
        self._registrations = registrations
        self._settings = settings
        self._instances: dict[ServiceKey, SerialService] = {}
        self._closing: asyncio.Task | None = None

    def require(self, key: ServiceKey[T]) -> T:
        if self._closing is not None:
            raise RuntimeError('Service scope is closed')
        if key not in self._instances:
            registration = self._registrations.get(key.name)
            if registration is None:
                raise LookupError(f'Service is not registered: {key.name}')
            if registration.key != key:
                raise TypeError(f'Service type mismatch: {key.name}')
            service = registration.factory(self._settings[key.name])
            if not isinstance(service, key.service_type):
                raise TypeError(f'Factory returned the wrong service type: {key.name}')
            self._instances[key] = service
        return cast(T, self._instances[key])

    @property
    def photoshop(self) -> PhotoshopService:
        from .photoshop import PHOTOSHOP
        return self.require(PHOTOSHOP)

    @property
    def segmentation(self) -> OnnxService:
        from .onnx import SEGMENTATION
        return self.require(SEGMENTATION)

    async def aclose(self):
        if self._closing is None:
            retired = [service._retire() for service in self._instances.values()]

            async def finish():
                results = await asyncio.gather(
                    *(asyncio.wrap_future(f) for f in retired),
                    return_exceptions=True)
                for result in results:
                    if isinstance(result, BaseException):
                        raise result

            self._closing = asyncio.create_task(finish())
        await _drain_on_cancel(self._closing)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        try:
            await self.aclose()
        except Exception as error:
            if exc_type is None:
                raise
            logger.warning('Service cleanup failed after a graph error: {}', error)
