"""Minimal runtime thread policy helpers for compiled custom ops."""

from __future__ import annotations

import os


def available_cpu_count() -> int:
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    return max(1, os.cpu_count() or 1)


def thread_policy_value() -> str:
    raw = os.environ.get("HNW_CUSTOM_OPS_THREADS", "auto").strip().lower()
    if not raw:
        return "auto"
    return raw


def parse_thread_policy(policy: str) -> tuple[str, int | None]:
    if policy == "auto":
        return "auto", None
    value = int(policy)
    if value <= 0:
        raise ValueError("HNW_CUSTOM_OPS_THREADS must be a positive integer or 'auto'")
    return "manual", value


def resolve_runtime_threads(
    *,
    op_name: str | None = None,
    shape: tuple[int, ...] | list[int] | None = None,
    dtype: object | None = None,
    build_info: dict[str, object] | None = None,
) -> int:
    _ = (op_name, shape, dtype, build_info)
    policy_name, manual_value = parse_thread_policy(thread_policy_value())
    if policy_name == "manual":
        return manual_value or 1
    return available_cpu_count()
