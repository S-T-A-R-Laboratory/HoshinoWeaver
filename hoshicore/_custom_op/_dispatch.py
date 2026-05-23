"""Shared runtime dispatch helpers for custom-op wrappers."""

from __future__ import annotations

import importlib
import os
import sys
from functools import lru_cache
from typing import Any

import numpy as np

from hoshicore._custom_op import thread_tuning


def debug_enabled() -> bool:
    return os.environ.get("HNW_CUSTOM_OPS_DEBUG", "0") not in {"", "0", "false", "False"}


def debug_log(module_name: str, message: str) -> None:
    if debug_enabled():
        print(f"[hoshicore._custom_op.{module_name}] {message}", file=sys.stderr)


def fallback_preference() -> str:
    raw = os.environ.get("HNW_CUSTOM_OPS_FALLBACK", "auto").strip().lower()
    if raw in {"auto", "numpy"}:
        return raw
    return "auto"


@lru_cache(maxsize=1)
def load_compiled_module() -> tuple[Any | None, str | None]:
    try:
        return importlib.import_module("hoshicore._custom_op._C"), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


@lru_cache(maxsize=1)
def compiled_build_info() -> dict[str, Any]:
    module, _ = load_compiled_module()
    if module is None or not hasattr(module, "build_info"):
        return {}
    payload = module.build_info()
    return payload if isinstance(payload, dict) else {}


_LAST_APPLIED_COMPILED_THREADS: int | None = None


def apply_compiled_threads(op_name: str, sample: np.ndarray) -> None:
    global _LAST_APPLIED_COMPILED_THREADS
    module, _ = load_compiled_module()
    if module is None:
        return
    build = compiled_build_info()
    if not build.get("openmp"):
        return
    if not hasattr(module, "set_openmp_threads"):
        return
    threads = thread_tuning.resolve_runtime_threads(
        op_name=op_name,
        shape=sample.shape,
        dtype=sample.dtype,
        build_info=build,
    )
    if threads == _LAST_APPLIED_COMPILED_THREADS:
        return
    if module.set_openmp_threads(int(threads)):
        _LAST_APPLIED_COMPILED_THREADS = int(threads)
