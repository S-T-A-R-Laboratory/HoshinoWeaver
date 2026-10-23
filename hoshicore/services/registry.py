"""Typed, process-local services leased for the duration of a root DAG."""
from __future__ import annotations

import asyncio
from collections.abc import Callable, Hashable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from threading import RLock
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
    The registry alone retires services; Ops must not close shared instances.
    """

    def __init__(self):
        self._worker = ThreadPoolExecutor(max_workers=1,
                                          thread_name_prefix='hoshi-service')
        self._predecessor: Future | None = None
        self._retired = False
        self._submission_lock = RLock()

    def _execute(self, function, args):
        if self._predecessor is not None:
            # A new graph can arrive while the previous graph is closing this
            # resource. Its initialization must wait for that close to finish.
            self._predecessor.result()
        return function(*args)

    async def _run(self, function: Callable[..., R], *args) -> R:
        with self._submission_lock:
            if self._retired:
                raise RuntimeError('Service has been released')
            worker = self._worker.submit(self._execute, function, args)
        return await _drain_on_cancel(asyncio.wrap_future(worker), worker.cancel)

    def _close(self) -> None:
        """Release native state on the resource's worker thread."""

    def _retire(self) -> Future:
        with self._submission_lock:
            self._retired = True
            future = self._worker.submit(self._execute, self._close, ())
            self._worker.shutdown(wait=False)
            return future


@dataclass(frozen=True)
class ServiceKey(Generic[T]):
    name: str
    service_type: type[T]


def _freeze(value):
    if isinstance(value, Mapping):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    hash(value)  # Fail explicitly for unsupported mutable configuration values.
    return value


@dataclass
class _Registration:
    key: ServiceKey
    factory: Callable
    settings: Callable
    resource_key: Callable


@dataclass
class _Entry:
    service: SerialService
    signature: Hashable
    users: int = 0


class ServiceRegistry:
    """Application-owned factories and resource leases, safe across loop threads.

    register() factories create cheap Python adapters, never native resources.
    settings selects creation options from a graph's configuration snapshot.
    resource_key identifies the physical resource; conflicting creation options
    for an already leased resource are rejected rather than silently ignored.
    """

    def __init__(self):
        self._registrations: dict[str, _Registration] = {}
        self._entries: dict[tuple, _Entry] = {}
        self._retiring: dict[tuple, Future] = {}
        self._lock = RLock()

    def register(self, key: ServiceKey[T], factory: Callable[[Mapping[str, Any]], T],
                 *, settings: Callable = dict, resource_key: Callable = _freeze) -> None:
        with self._lock:
            if key.name in self._registrations:
                raise ValueError(f'Service already registered: {key.name}')
            self._registrations[key.name] = _Registration(
                key, factory, settings, resource_key)

    def scope(self, settings: Mapping[str, Any] | None = None) -> ServiceScope:
        # Snapshot only service creation options, not unrelated loader arrays
        # or other potentially large graph configuration payloads.
        with self._lock:
            options = {name: deepcopy(registration.settings(settings or {}))
                       for name, registration in self._registrations.items()}
        return ServiceScope(self, options)

    def _acquire(self, key: ServiceKey[T], settings) -> tuple[tuple, T]:
        with self._lock:
            registration = self._registrations.get(key.name)
            if registration is None:
                raise LookupError(f'Service is not registered: {key.name}')
            if registration.key != key:
                raise TypeError(f'Service type mismatch: {key.name}')
            options = deepcopy(settings[key.name])
            identity = (key.name, registration.resource_key(options))
            signature = _freeze(options)
            entry = self._entries.get(identity)
            if entry is None:
                service = registration.factory(options)
                if not isinstance(service, key.service_type):
                    raise TypeError(f'Factory returned the wrong service type: {key.name}')
                service._predecessor = self._retiring.get(identity)
                entry = self._entries[identity] = _Entry(service, signature)
            elif entry.signature != signature:
                raise ValueError(f'Conflicting settings for active service: {key.name}')
            entry.users += 1
            return identity, cast(T, entry.service)

    def _release(self, identity) -> Future | None:
        with self._lock:
            entry = self._entries[identity]
            entry.users -= 1
            if entry.users:
                return None
            del self._entries[identity]
            future = entry.service._retire()
            self._retiring[identity] = future
            return future

    def _forget_retired(self, identity, future):
        with self._lock:
            if self._retiring.get(identity) is future:
                del self._retiring[identity]


class ServiceScope:
    """Lazy graph leases. All flattened subgraphs receive this same scope."""

    def __init__(self, registry: ServiceRegistry, settings):
        self._registry = registry
        self._settings = settings
        self._leases: dict[ServiceKey, tuple[tuple, SerialService]] = {}
        self._closing: asyncio.Task | None = None

    def require(self, key: ServiceKey[T]) -> T:
        if self._closing is not None:
            raise RuntimeError('Service scope is closed')
        if key not in self._leases:
            self._leases[key] = self._registry._acquire(key, self._settings)
        return cast(T, self._leases[key][1])

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
            retired = [(identity, self._registry._release(identity))
                       for identity, _ in self._leases.values()]

            async def finish():
                pending = [(identity, f) for identity, f in retired if f is not None]
                results = await asyncio.gather(
                    *(asyncio.wrap_future(f) for _, f in pending),
                    return_exceptions=True)
                for identity, future in pending:
                    self._registry._forget_retired(identity, future)
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
