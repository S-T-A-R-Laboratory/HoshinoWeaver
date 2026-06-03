"""Runtime backend candidates for custom-op wrappers."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from hoshicore._custom_op._dispatch import load_compiled_module

ModuleLoader = Callable[[], tuple[Any | None, str | None]]


@dataclass(frozen=True)
class BackendCandidate:
    logical_op: str
    backend: str
    kernel_name: str
    placement: str = "host_to_host"
    priority: int = 0
    requires_contiguous: bool = True
    dtypes: tuple[str, ...] = ()
    fallback: str = "numpy"
    build_flag: str | None = None


@dataclass(frozen=True)
class BackendSelection:
    candidate: BackendCandidate | None
    module: Any | None
    reason: str | None = None

    @property
    def native(self) -> bool:
        return self.candidate is not None and self.module is not None

    @property
    def backend(self) -> str:
        if self.candidate is None:
            return "numpy"
        return self.candidate.backend


_CANDIDATES: tuple[BackendCandidate, ...] = (
    BackendCandidate("fgp_accumulate", "openmp_cpu", "fgp_accumulate"),
    BackendCandidate("fgp_add", "openmp_cpu", "fgp_add"),
    BackendCandidate("fgp_masked_mean_merge", "openmp_cpu", "fgp_masked_mean_merge"),
    BackendCandidate("huber_weighted_accumulate", "openmp_cpu", "huber_weighted_accumulate"),
    BackendCandidate("sigma_clip_fused_merge", "openmp_cpu", "sigma_clip_fused_merge"),
    BackendCandidate(
        "sigma_clip_fused_masked_merge",
        "openmp_cpu",
        "sigma_clip_fused_masked_merge",
    ),
    BackendCandidate("max_combine", "openmp_cpu", "max_combine"),
    BackendCandidate("threshold_max_merge", "openmp_cpu", "threshold_max_merge"),
    BackendCandidate("median_reduce_chunk", "openmp_cpu", "median_reduce_chunk"),
    BackendCandidate("equalize_noise_correct", "openmp_cpu", "equalize_noise_correct"),
    BackendCandidate("sigma_clip_iterative_chunk", "openmp_cpu", "sigma_clip_iterative_chunk"),
    BackendCandidate("sigma_clip_fused_chunk", "openmp_cpu", "sigma_clip_fused_chunk"),
    BackendCandidate(
        "camera_model_remap",
        "cuda_host_io",
        "camera_model_remap",
        build_flag="cuda",
    ),
)

_CANDIDATES_BY_OP: dict[str, tuple[BackendCandidate, ...]] = {}
for _candidate in _CANDIDATES:
    _CANDIDATES_BY_OP.setdefault(_candidate.logical_op, ())
    _CANDIDATES_BY_OP[_candidate.logical_op] += (_candidate,)


def registered_backend_candidates(logical_op: str | None = None) -> tuple[BackendCandidate, ...]:
    if logical_op is None:
        return _CANDIDATES
    return _CANDIDATES_BY_OP.get(logical_op, ())


def select_backend(
    logical_op: str,
    preference: str = "auto",
    *,
    load_module: ModuleLoader = load_compiled_module,
    build_info: Mapping[str, Any] | None = None,
) -> BackendSelection:
    if preference == "numpy":
        return BackendSelection(None, None, "numpy backend forced by preference")

    candidates = registered_backend_candidates(logical_op)
    if not candidates:
        return BackendSelection(None, None, f"no backend candidate registered for {logical_op}")

    module, module_error = load_module()
    if module is None:
        return BackendSelection(None, None, module_error or "compiled backend unavailable")

    missing_kernel: str | None = None
    for candidate in sorted(candidates, key=lambda item: item.priority, reverse=True):
        if not _has_static_attr(module, candidate.kernel_name):
            missing_kernel = candidate.kernel_name
            continue
        if candidate.build_flag is not None:
            info = build_info if build_info is not None else _module_build_info(module)
            if info and not info.get(candidate.build_flag):
                return BackendSelection(
                    None,
                    module,
                    f"compiled backend missing build flag: {candidate.build_flag}",
                )
        return BackendSelection(candidate, module, None)

    if missing_kernel is not None:
        return BackendSelection(None, module, f"compiled backend missing kernel: {missing_kernel}")
    return BackendSelection(None, module, f"no available backend candidate for {logical_op}")


def native_backend_available(
    logical_op: str,
    preference: str = "auto",
    *,
    load_module: ModuleLoader = load_compiled_module,
    build_info: Mapping[str, Any] | None = None,
) -> tuple[bool, str | None]:
    selection = select_backend(
        logical_op,
        preference,
        load_module=load_module,
        build_info=build_info,
    )
    return selection.native, selection.reason


def _module_build_info(module: Any) -> Mapping[str, Any]:
    if not _has_static_attr(module, "build_info"):
        return {}
    payload = module.build_info()
    return payload if isinstance(payload, Mapping) else {}


def _has_static_attr(value: Any, name: str) -> bool:
    try:
        inspect.getattr_static(value, name)
    except AttributeError:
        return False
    return True
