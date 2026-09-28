"""Runtime backend candidates for custom-op wrappers."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, replace
from typing import Any, Callable, Collection, Mapping

from hoshicore._custom_op._dispatch import (
    cuda_memory_info,
    is_cuda_resource_exhausted_error,
    is_cuda_runtime_unavailable_error,
    is_metal_resource_exhausted_error,
    is_metal_runtime_unavailable_error,
    load_compiled_module,
    load_metal_module,
    metal_device_info,
)
from hoshicore._custom_op.cuda_memory import cuda_memory_model_kind
from hoshicore._custom_op.metal_memory import metal_memory_model_kind

ModuleLoader = Callable[[], tuple[Any | None, str | None]]
RuntimeProbe = Callable[[], dict[str, Any]]
ModuleLoaders = Mapping[str, ModuleLoader]
CUDA_MEMORY_MODELS = frozenset({"cuda_chunk", "phase_estimator", "static_estimator"})
METAL_MEMORY_MODELS = frozenset({"static_estimator"})
ACCELERATOR_BACKENDS = frozenset({"cuda_host_io", "metal_host_io"})


@dataclass(frozen=True)
class BackendCandidate:
    logical_op: str
    backend: str
    kernel_name: str
    placement: str = "host_to_host"
    priority: int = 0
    requires_contiguous: bool = True
    dtypes: tuple[str, ...] = ()
    fallback: str | None = "numpy"
    module_key: str = "compiled"
    build_flag: str | None = None
    memory_model: str | None = None
    memory_model_reason: str | None = None


@dataclass(frozen=True)
class BackendDecision:
    logical_op: str
    backend: str
    kernel_name: str | None
    placement: str
    fallback: str | None
    native: bool
    reason_code: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_op": self.logical_op,
            "backend": self.backend,
            "kernel_name": self.kernel_name,
            "placement": self.placement,
            "fallback": self.fallback,
            "native": self.native,
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True)
class BackendSelection:
    candidate: BackendCandidate | None
    module: Any | None
    reason: str | None = None
    reason_code: str = "unspecified"
    decision: BackendDecision | None = None

    @property
    def native(self) -> bool:
        return self.candidate is not None and self.module is not None

    @property
    def backend(self) -> str:
        if self.candidate is None:
            return "numpy"
        return self.candidate.backend

    def to_decision(self, logical_op: str) -> BackendDecision:
        candidate = self.candidate
        if candidate is None:
            return BackendDecision(
                logical_op=logical_op,
                backend="numpy",
                kernel_name=None,
                placement="host_to_host",
                fallback=None,
                native=False,
                reason_code=self.reason_code,
            )
        return BackendDecision(
            logical_op=logical_op,
            backend=candidate.backend,
            kernel_name=candidate.kernel_name,
            placement=candidate.placement,
            fallback=candidate.fallback,
            native=self.native,
            reason_code=self.reason_code,
        )


_CANDIDATES: tuple[BackendCandidate, ...] = (
    BackendCandidate("detection_gray", "openmp_cpu", "detection_gray_cast_cpu"),
    BackendCandidate(
        "extract_point_features",
        "cuda_host_io",
        "extract_point_features_cuda",
        priority=10,
        fallback="openmp_cpu",
        build_flag="cuda",
        memory_model="static_estimator",
    ),
    BackendCandidate("extract_point_features", "openmp_cpu", "extract_point_features"),
    BackendCandidate(
        "asterism_mutual_nearest",
        "cuda_host_io",
        "asterism_mutual_nearest_cuda",
        priority=10,
        fallback="openmp_cpu",
        build_flag="cuda",
        memory_model="static_estimator",
    ),
    BackendCandidate("asterism_mutual_nearest", "openmp_cpu", "asterism_mutual_nearest_cpu"),
    BackendCandidate("asterism_tokens", "openmp_cpu", "asterism_tokens_cpu"),
    BackendCandidate("asterism_anchor_votes", "openmp_cpu", "asterism_anchor_votes_cpu"),
    BackendCandidate(
        "matching_cosine_bidirectional_nearest",
        "cuda_host_io",
        "matching_cosine_bidirectional_nearest_cuda",
        priority=10,
        fallback="openmp_cpu",
        build_flag="cuda",
        memory_model="static_estimator",
    ),
    BackendCandidate(
        "matching_cosine_bidirectional_nearest",
        "openmp_cpu",
        "matching_cosine_bidirectional_nearest_cpu",
    ),
    BackendCandidate("calibration_subtract", "openmp_cpu", "calibration_subtract"),
    BackendCandidate("calibration_divide", "openmp_cpu", "calibration_divide"),
    BackendCandidate("fgp_accumulate", "openmp_cpu", "fgp_accumulate"),
    BackendCandidate("fgp_masked_mean_merge", "openmp_cpu", "fgp_masked_mean_merge"),
    # Public primitive and fused-mask benchmark baseline; the production
    # median detector consumes median_star_mask instead.
    BackendCandidate("median_filter_2d", "openmp_cpu", "median_filter_2d"),
    BackendCandidate("median_star_mask", "openmp_cpu", "median_star_mask_cpu"),
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
    BackendCandidate("noise_fill_local_mean", "openmp_cpu", "noise_fill_local_mean"),
    BackendCandidate("noise_equalization_params", "openmp_cpu", "noise_equalization_params"),
    # Provisional: above OpenMP so the Metal -> OpenMP -> NumPy chain is
    # exercised, explicitly below CUDA instead of relying on declaration order.
    # Real Apple hardware timing decides whether it stays the macOS default.
    BackendCandidate(
        "star_shrink_process",
        "metal_host_io",
        "star_shrink_process_metal",
        priority=9,
        fallback="openmp_cpu",
        module_key="metal",
        build_flag="metal",
        memory_model="static_estimator",
    ),
    BackendCandidate(
        "star_shrink_process",
        "cuda_host_io",
        "star_shrink_process_cuda",
        priority=10,
        fallback="openmp_cpu",
        build_flag="cuda",
        memory_model="static_estimator",
    ),
    BackendCandidate("star_shrink_process", "openmp_cpu", "star_shrink_process"),
    BackendCandidate("star_shrink_detect_mask", "openmp_cpu", "star_shrink_detect_mask"),
    BackendCandidate(
        "star_mask_dog",
        "cuda_host_io",
        "star_mask_dog_cuda",
        priority=10,
        fallback="openmp_cpu",
        build_flag="cuda",
        memory_model="static_estimator",
    ),
    BackendCandidate(
        "star_mask_dog",
        "metal_host_io",
        "star_mask_dog_metal",
        priority=9,
        fallback="openmp_cpu",
        module_key="metal",
        build_flag="metal",
        memory_model="static_estimator",
    ),
    BackendCandidate("star_mask_dog", "openmp_cpu", "star_mask_dog_cpu"),
    BackendCandidate(
        "star_shrink_dog_process",
        "metal_host_io",
        "star_shrink_dog_process_metal",
        priority=9,
        # Same as CUDA: the real fallback is the composed path in the wrapper.
        fallback=None,
        module_key="metal",
        build_flag="metal",
        memory_model="static_estimator",
    ),
    BackendCandidate(
        "star_shrink_dog_process",
        "cuda_host_io",
        "star_shrink_dog_process_cuda",
        priority=10,
        # Real fallback is the composed star_mask_dog + star_shrink_process
        # path, which is not expressible as a single registry candidate.
        fallback=None,
        build_flag="cuda",
        memory_model="static_estimator",
    ),
    BackendCandidate("sigma_clip_iterative_chunk", "openmp_cpu", "sigma_clip_iterative_chunk"),
    BackendCandidate(
        "huber_weighted_chunk",
        "cuda_host_io",
        "huber_weighted_chunk_cuda",
        priority=10,
        build_flag="cuda",
        memory_model="cuda_chunk",
    ),
    BackendCandidate(
        "sigma_clip_fused_chunk",
        "cuda_host_io",
        "sigma_clip_fused_chunk_cuda",
        priority=10,
        fallback="openmp_cpu",
        build_flag="cuda",
        memory_model="cuda_chunk",
    ),
    BackendCandidate("sigma_clip_fused_chunk", "openmp_cpu", "sigma_clip_fused_chunk"),
    BackendCandidate(
        "wavelet_dec_rec",
        "cuda_host_io",
        "wavelet_dec_rec_cuda_core",
        priority=10,
        fallback="openmp_cpu",
        build_flag="cuda",
        memory_model="static_estimator",
    ),
    BackendCandidate("wavelet_dec_rec", "openmp_cpu", "wavelet_dec_rec_cpu"),
    BackendCandidate(
        "wavelet_dec_rec_cuda_core",
        "cuda_host_io",
        "wavelet_dec_rec_cuda_core",
        build_flag="cuda",
        memory_model="static_estimator",
    ),
    BackendCandidate(
        "star_detect_fused_pixel_components",
        "cuda_host_io",
        "star_detect_fused_pixel_components_cuda",
        priority=10,
        fallback="openmp_cpu",
        build_flag="cuda",
        memory_model="phase_estimator",
    ),
    BackendCandidate(
        "star_detect_fused_pixel_components",
        "openmp_cpu",
        "star_detect_fused_pixel_components_cpu",
        # No numpy backend exists; production falls back to Norma's contour
        # detector at the component layer.
        fallback=None,
    ),
    BackendCandidate(
        "camera_model_remap",
        "cuda_host_io",
        "camera_model_remap",
        priority=10,
        fallback="openmp_cpu",
        build_flag="cuda",
        memory_model="static_estimator",
    ),
    BackendCandidate("camera_model_remap", "openmp_cpu", "camera_model_remap_cpu"),
)

_CANDIDATES_BY_OP: dict[str, tuple[BackendCandidate, ...]] = {}
for _candidate in _CANDIDATES:
    _CANDIDATES_BY_OP.setdefault(_candidate.logical_op, ())
    _CANDIDATES_BY_OP[_candidate.logical_op] += (_candidate,)


def validate_cuda_memory_policy_declarations(
    candidates: tuple[BackendCandidate, ...] = _CANDIDATES,
    *,
    require_memory_models: bool = False,
) -> None:
    for candidate in candidates:
        if candidate.backend != "cuda_host_io":
            continue
        if (
            candidate.memory_model is not None
            and candidate.memory_model not in CUDA_MEMORY_MODELS
        ):
            raise RuntimeError(
                "CUDA backend candidate declares an unknown memory model: "
                f"{candidate.logical_op}/{candidate.memory_model}"
            )
        if candidate.memory_model is not None:
            try:
                registered_kind = cuda_memory_model_kind(candidate.logical_op)
            except KeyError as exc:
                raise RuntimeError(
                    "CUDA backend candidate declares a memory model without a "
                    f"registered estimator: {candidate.logical_op}/"
                    f"{candidate.memory_model}"
                ) from exc
            if registered_kind != candidate.memory_model:
                raise RuntimeError(
                    "CUDA backend candidate memory model does not match its "
                    f"registered estimator: {candidate.logical_op}/"
                    f"{candidate.memory_model} != {registered_kind}"
                )
        if candidate.memory_model is None and require_memory_models:
            raise RuntimeError(
                "built-in CUDA backend candidate must declare a consumed "
                f"memory model: {candidate.logical_op}/{candidate.kernel_name}"
            )
        if candidate.memory_model is None and not candidate.memory_model_reason:
            raise RuntimeError(
                "CUDA backend candidate must declare a memory model or an "
                f"explicit deferral reason: {candidate.logical_op}/"
                f"{candidate.kernel_name}"
            )


validate_cuda_memory_policy_declarations(require_memory_models=True)


def validate_metal_memory_policy_declarations(
    candidates: tuple[BackendCandidate, ...] = _CANDIDATES,
    *,
    require_memory_models: bool = False,
) -> None:
    for candidate in candidates:
        if candidate.backend != "metal_host_io":
            continue
        if (
            candidate.memory_model is not None
            and candidate.memory_model not in METAL_MEMORY_MODELS
        ):
            raise RuntimeError(
                "Metal backend candidate declares an unknown memory model: "
                f"{candidate.logical_op}/{candidate.memory_model}"
            )
        if candidate.memory_model is not None:
            try:
                registered_kind = metal_memory_model_kind(candidate.logical_op)
            except KeyError as exc:
                raise RuntimeError(
                    "Metal backend candidate declares a memory model without a "
                    f"registered estimator: {candidate.logical_op}/"
                    f"{candidate.memory_model}"
                ) from exc
            if registered_kind != candidate.memory_model:
                raise RuntimeError(
                    "Metal backend candidate memory model does not match its "
                    f"registered estimator: {candidate.logical_op}/"
                    f"{candidate.memory_model} != {registered_kind}"
                )
        if candidate.memory_model is None and require_memory_models:
            raise RuntimeError(
                "built-in Metal backend candidate must declare a consumed "
                f"memory model: {candidate.logical_op}/{candidate.kernel_name}"
            )
        if candidate.memory_model is None and not candidate.memory_model_reason:
            raise RuntimeError(
                "Metal backend candidate must declare a memory model or an "
                f"explicit deferral reason: {candidate.logical_op}/"
                f"{candidate.kernel_name}"
            )


validate_metal_memory_policy_declarations(require_memory_models=True)


def registered_backend_candidates(logical_op: str | None = None) -> tuple[BackendCandidate, ...]:
    if logical_op is None:
        return _CANDIDATES
    return _CANDIDATES_BY_OP.get(logical_op, ())


def _candidate_module_loader(
    candidate: BackendCandidate,
    *,
    load_module: ModuleLoader,
    module_loaders: ModuleLoaders | None,
) -> ModuleLoader:
    if candidate.module_key == "compiled":
        return load_module
    if module_loaders is not None and candidate.module_key in module_loaders:
        return module_loaders[candidate.module_key]
    if candidate.module_key == "metal":
        return load_metal_module
    raise RuntimeError(
        "backend candidate references an unknown native module: "
        f"{candidate.logical_op}/{candidate.module_key}"
    )


def select_backend(
    logical_op: str,
    preference: str = "auto",
    *,
    load_module: ModuleLoader = load_compiled_module,
    module_loaders: ModuleLoaders | None = None,
    build_info: Mapping[str, Any] | None = None,
    exclude_backends: Collection[str] = (),
) -> BackendSelection:
    if preference == "numpy":
        return BackendSelection(
            None,
            None,
            "numpy backend forced by preference",
            "forced_numpy",
        )

    candidates = registered_backend_candidates(logical_op)
    if not candidates:
        return BackendSelection(
            None,
            None,
            f"no backend candidate registered for {logical_op}",
            "logical_op_unregistered",
        )

    cpu_only = preference == "cpu"
    if cpu_only and all(
        candidate.backend in ACCELERATOR_BACKENDS for candidate in candidates
    ):
        return BackendSelection(
            None,
            None,
            "CPU backend preference excludes accelerators and no native CPU backend "
            f"is registered for {logical_op}",
            "forced_cpu",
        )

    missing_kernel: str | None = None
    missing_build_flag: str | None = None
    module_errors: list[tuple[str, str]] = []
    loaded_modules: dict[str, tuple[Any | None, str | None]] = {}
    last_module: Any | None = None
    excluded = set(exclude_backends)
    if cpu_only:
        excluded.update(ACCELERATOR_BACKENDS)
    excluded = frozenset(excluded)
    for candidate in sorted(candidates, key=lambda item: item.priority, reverse=True):
        if candidate.backend in excluded:
            continue
        if candidate.module_key not in loaded_modules:
            loader = _candidate_module_loader(
                candidate,
                load_module=load_module,
                module_loaders=module_loaders,
            )
            loaded_modules[candidate.module_key] = loader()
        module, module_error = loaded_modules[candidate.module_key]
        if module is None:
            if module_error:
                module_errors.append((candidate.module_key, module_error))
            continue
        last_module = module
        if not _has_static_attr(module, candidate.kernel_name):
            missing_kernel = candidate.kernel_name
            continue
        if candidate.build_flag is not None:
            info = (
                build_info
                if candidate.module_key == "compiled" and build_info is not None
                else _module_build_info(module)
            )
            if info and not info.get(candidate.build_flag):
                missing_build_flag = candidate.build_flag
                continue
        reason_code = "forced_cpu" if cpu_only else "selected_native"
        return BackendSelection(candidate, module, None, reason_code)

    if missing_kernel is not None:
        return BackendSelection(
            None,
            last_module,
            f"compiled backend missing kernel: {missing_kernel}",
            "kernel_unavailable",
        )
    if missing_build_flag is not None:
        return BackendSelection(
            None,
            last_module,
            f"compiled backend missing build flag: {missing_build_flag}",
            "build_flag_unavailable",
        )
    if excluded:
        excluded_names = ", ".join(sorted(excluded))
        return BackendSelection(
            None,
            last_module,
            f"no available backend candidate for {logical_op} after excluding: "
            f"{excluded_names}",
            "backends_excluded",
        )
    if module_errors and last_module is None:
        unique_errors = tuple(dict.fromkeys(module_errors))
        reason = (
            unique_errors[0][1]
            if len(unique_errors) == 1
            else "; ".join(
                f"{module_key}: {error}"
                for module_key, error in unique_errors
            )
        )
        return BackendSelection(
            None,
            None,
            reason,
            "compiled_module_unavailable",
        )
    return BackendSelection(
        None,
        last_module,
        f"no available backend candidate for {logical_op}",
        "backend_unavailable",
    )


def resolve_backend(
    logical_op: str,
    preference: str = "auto",
    *,
    load_module: ModuleLoader = load_compiled_module,
    module_loaders: ModuleLoaders | None = None,
    build_info: Mapping[str, Any] | None = None,
    exclude_backends: Collection[str] = (),
    cuda_probe: RuntimeProbe = cuda_memory_info,
    metal_probe: RuntimeProbe = metal_device_info,
) -> BackendSelection:
    selection = select_backend(
        logical_op,
        preference,
        load_module=load_module,
        module_loaders=module_loaders,
        build_info=build_info,
        exclude_backends=exclude_backends,
    )
    decision = selection.to_decision(logical_op)
    if (
        not selection.native
        or selection.candidate is None
        or selection.candidate.backend not in ACCELERATOR_BACKENDS
    ):
        return BackendSelection(
            selection.candidate,
            selection.module,
            selection.reason,
            selection.reason_code,
            decision,
        )

    backend = selection.candidate.backend
    probe = cuda_probe() if backend == "cuda_host_io" else metal_probe()
    if probe.get("available") is True:
        return BackendSelection(
            selection.candidate,
            selection.module,
            selection.reason,
            selection.reason_code,
            decision,
        )
    status = probe.get("status")
    if status != "explicitly_unavailable":
        raise RuntimeError(
            f"{backend} runtime probe failed without an unavailable status"
        )

    excluded = set(exclude_backends)
    excluded.add(backend)
    fallback_selection = select_backend(
        logical_op,
        "auto",
        load_module=load_module,
        module_loaders=module_loaders,
        build_info=build_info,
        exclude_backends=excluded,
    )
    fallback_decision = fallback_selection.to_decision(logical_op)
    unavailable_reason_code = str(
        probe.get("reason_code")
        or f"{backend.removesuffix('_host_io')}_runtime_unavailable"
    )
    fallback_decision = replace(
        fallback_decision,
        reason_code=unavailable_reason_code,
    )
    return BackendSelection(
        fallback_selection.candidate,
        fallback_selection.module,
        probe.get("reason") or fallback_selection.reason,
        unavailable_reason_code,
        fallback_decision,
    )


def _resolve_after_accelerator_backend_failure(
    logical_op: str,
    failed_backend: str,
    exc: RuntimeError,
    reason_code: str,
    *,
    load_module: ModuleLoader = load_compiled_module,
    module_loaders: ModuleLoaders | None = None,
    build_info: Mapping[str, Any] | None = None,
) -> BackendSelection:
    selection = resolve_backend(
        logical_op,
        "auto",
        load_module=load_module,
        module_loaders=module_loaders,
        build_info=build_info,
        exclude_backends={failed_backend},
    )
    decision = selection.to_decision(logical_op)
    decision = replace(decision, reason_code=reason_code)
    return BackendSelection(
        selection.candidate,
        selection.module,
        str(exc),
        reason_code,
        decision,
    )


def resolve_after_runtime_unavailable(
    logical_op: str,
    failed_backend: str,
    exc: RuntimeError,
    *,
    load_module: ModuleLoader = load_compiled_module,
    module_loaders: ModuleLoaders | None = None,
    build_info: Mapping[str, Any] | None = None,
) -> BackendSelection:
    classifiers = {
        "cuda_host_io": is_cuda_runtime_unavailable_error,
        "metal_host_io": is_metal_runtime_unavailable_error,
    }
    classifier = classifiers.get(failed_backend)
    if classifier is None or not classifier(exc):
        raise exc
    return _resolve_after_accelerator_backend_failure(
        logical_op,
        failed_backend,
        exc,
        f"{failed_backend.removesuffix('_host_io')}_runtime_unavailable",
        load_module=load_module,
        module_loaders=module_loaders,
        build_info=build_info,
    )


def resolve_after_resource_exhausted(
    logical_op: str,
    failed_backend: str,
    exc: RuntimeError,
    *,
    load_module: ModuleLoader = load_compiled_module,
    module_loaders: ModuleLoaders | None = None,
    build_info: Mapping[str, Any] | None = None,
) -> BackendSelection:
    classifiers = {
        "cuda_host_io": is_cuda_resource_exhausted_error,
        "metal_host_io": is_metal_resource_exhausted_error,
    }
    classifier = classifiers.get(failed_backend)
    if classifier is None or not classifier(exc):
        raise exc
    return _resolve_after_accelerator_backend_failure(
        logical_op,
        failed_backend,
        exc,
        f"{failed_backend.removesuffix('_host_io')}_resource_exhausted",
        load_module=load_module,
        module_loaders=module_loaders,
        build_info=build_info,
    )


def resolve_after_cuda_failure(
    logical_op: str,
    exc: RuntimeError,
    *,
    load_module: ModuleLoader = load_compiled_module,
    build_info: Mapping[str, Any] | None = None,
    log: Callable[[str], None] | None = None,
) -> BackendSelection:
    """Compatibility wrapper for CUDA host-I/O failure resolution."""
    return resolve_after_accelerator_failure(
        logical_op,
        "cuda_host_io",
        exc,
        load_module=load_module,
        build_info=build_info,
        log=log,
    )


def resolve_after_accelerator_failure(
    logical_op: str,
    failed_backend: str,
    exc: RuntimeError,
    *,
    load_module: ModuleLoader = load_compiled_module,
    module_loaders: ModuleLoaders | None = None,
    build_info: Mapping[str, Any] | None = None,
    log: Callable[[str], None] | None = None,
) -> BackendSelection:
    """Classify a typed accelerator failure and resolve the next backend."""
    resource_classifiers = {
        "cuda_host_io": is_cuda_resource_exhausted_error,
        "metal_host_io": is_metal_resource_exhausted_error,
    }
    classifier = resource_classifiers.get(failed_backend)
    if classifier is None:
        raise exc
    if classifier(exc):
        selection = resolve_after_resource_exhausted(
            logical_op,
            failed_backend,
            exc,
            load_module=load_module,
            module_loaders=module_loaders,
            build_info=build_info,
        )
        if log is not None:
            log(
                f"{failed_backend} exhausted resources, falling back to the "
                f"next backend: {exc}"
            )
        return selection
    selection = resolve_after_runtime_unavailable(
        logical_op,
        failed_backend,
        exc,
        load_module=load_module,
        module_loaders=module_loaders,
        build_info=build_info,
    )
    if log is not None:
        log(
            f"{failed_backend} unavailable at runtime, falling back to the "
            f"next backend: {exc}"
        )
    return selection


def run_with_accelerator_fallback(
    logical_op: str,
    selection: BackendSelection,
    map_backend: Callable[[BackendSelection], tuple[str, Callable[..., Any]]],
    call: Callable[[Callable[..., Any]], Any],
    *,
    load_module: ModuleLoader = load_compiled_module,
    log: Callable[[str], None] | None = None,
) -> Any:
    """Run the selected backend, retrying once on the registry fallback.

    Owns the wrapper-side accelerator contract in one place: non-accelerator
    selections run plainly (errors propagate), typed accelerator failures
    re-resolve and retry the mapped fallback exactly once, and anything else
    propagates. ``call(backend)`` performs the actual invocation so per-op
    argument shapes stay with the op.
    """
    _, backend = map_backend(selection)
    candidate = selection.candidate
    if candidate is None or candidate.backend not in ACCELERATOR_BACKENDS:
        return call(backend)

    try:
        return call(backend)
    except RuntimeError as exc:
        fallback_selection = resolve_after_accelerator_failure(
            logical_op,
            candidate.backend,
            exc,
            load_module=load_module,
            log=log,
        )

    fallback_candidate = fallback_selection.candidate
    if fallback_candidate is not None and fallback_candidate.backend == candidate.backend:
        raise RuntimeError(
            f"{candidate.backend} backend remained selected after runtime exclusion"
        )
    _, fallback_backend = map_backend(fallback_selection)
    return call(fallback_backend)


def native_backend_available(
    logical_op: str,
    preference: str = "auto",
    *,
    load_module: ModuleLoader = load_compiled_module,
    module_loaders: ModuleLoaders | None = None,
    build_info: Mapping[str, Any] | None = None,
    exclude_backends: Collection[str] = (),
) -> tuple[bool, str | None]:
    selection = select_backend(
        logical_op,
        preference,
        load_module=load_module,
        module_loaders=module_loaders,
        build_info=build_info,
        exclude_backends=exclude_backends,
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
