"""
资源预检：在 DAG 执行前估算峰值内存/磁盘开销，与系统可用资源比对。

行为模式（由全局配置 auto_fallback 控制）：
    - auto_fallback=true + 无 callback → 静默降级 + info log
    - auto_fallback=false + 无 callback → 只 warn，不阻断
    - 有 preflight_callback → 由回调函数决定行为

回调返回值（PreflightAction）：
    - "apply"  — 应用降级建议并继续执行
    - "ignore" — 忽略建议，按原配置继续执行（仅回调模式下允许）
    - "abort"  — 中止执行
"""
from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

PreflightAction = Literal["apply", "ignore", "abort"]

import psutil
from loguru import logger

from ..component.image_io import peek_shape
from ..ops.base import BaseOp
from .build import ValidatedDag
from .registry import REGISTERED_OP

MEMORY_SAFETY_FACTOR = 0.7
MEMORY_FIXED_OVERHEAD = 200 * 1024 * 1024  # 200MB


class PreflightAbortError(Exception):
    """用户拒绝预检建议时抛出。"""
    pass


@dataclass
class ResourceEstimate:
    peak_memory_bytes: int
    peak_disk_bytes: int


@dataclass
class FallbackProposal:
    config_key: str
    current_value: str
    proposed_value: str
    reason: str


@dataclass
class PreflightReport:
    estimate: ResourceEstimate
    available_memory_bytes: int
    available_disk_bytes: int
    warnings: list[str] = field(default_factory=list)
    proposed_fallbacks: list[FallbackProposal] = field(default_factory=list)
    applied_fallbacks: list[str] = field(default_factory=list)


def preflight_check(
    dag: ValidatedDag,
    effective_configs: dict[str, Any],
    global_inputs: dict[str, Any],
    op_registry: Optional[dict[str, type[BaseOp]]] = None,
) -> PreflightReport:
    """估算资源需求，生成警告和降级建议。纯函数，不修改 configs。"""
    registry = op_registry or REGISTERED_OP

    # 1. Peek frame shape
    fnames = global_inputs.get("fnames")
    if not fnames or not isinstance(fnames, (list, tuple)) or len(fnames) == 0:
        return PreflightReport(
            estimate=ResourceEstimate(0, 0),
            available_memory_bytes=psutil.virtual_memory().available,
            available_disk_bytes=0,
        )

    try:
        shape, dtype_bytes = peek_shape(fnames[0])
    except (FileNotFoundError, ValueError, OSError) as e:
        logger.warning(f"[Preflight] 无法 peek 首帧: {e}")
        return PreflightReport(
            estimate=ResourceEstimate(0, 0),
            available_memory_bytes=psutil.virtual_memory().available,
            available_disk_bytes=0,
        )

    frame_bytes = 1
    for dim in shape:
        frame_bytes *= dim
    frame_bytes *= dtype_bytes
    n_frames = len(fnames)

    # 2. 遍历节点，累加资源估算
    total_mem = 0
    total_disk = 0
    for node_name in dag.exec_order:
        node_spec = dag.nodes[node_name]
        op_name = node_spec["op"]
        op_cls = registry.get(op_name)
        if op_cls is None:
            continue
        node_configs = _resolve_node_configs(node_spec, effective_configs, op_cls)
        mem, disk = op_cls.estimate_resources(node_configs, frame_bytes, n_frames)
        total_mem += mem
        total_disk += disk

    # 3. 获取可用资源
    avail_mem = psutil.virtual_memory().available
    temp_path = effective_configs.get("temp_path") or tempfile.gettempdir()
    avail_disk = shutil.disk_usage(temp_path).free

    estimate = ResourceEstimate(total_mem, total_disk)
    report = PreflightReport(
        estimate=estimate,
        available_memory_bytes=avail_mem,
        available_disk_bytes=avail_disk,
    )

    # 4. 比较 + 生成建议
    mem_budget = int(avail_mem * MEMORY_SAFETY_FACTOR) - MEMORY_FIXED_OVERHEAD
    if total_mem > 0 and (mem_budget <= 0 or total_mem > mem_budget):
        report.warnings.append(
            f"预估峰值内存 {total_mem / 1e9:.2f} GB，"
            f"可用预算 {mem_budget / 1e9:.2f} GB"
            f"（超出 {(total_mem - mem_budget) / 1e9:.2f} GB）")
        if effective_configs.get("buffer_mode") == "memory":
            report.proposed_fallbacks.append(FallbackProposal(
                config_key="buffer_mode",
                current_value="memory",
                proposed_value="disk",
                reason="内存不足"))

    if total_disk > avail_disk:
        report.warnings.append(
            f"预估磁盘缓存 {total_disk / 1e9:.2f} GB，"
            f"可用空间 {avail_disk / 1e9:.2f} GB"
            f"（不足 {(total_disk - avail_disk) / 1e9:.2f} GB）")
        if effective_configs.get("buffer_mode") == "disk":
            has_fnames = "fnames" in global_inputs and global_inputs["fnames"]
            if has_fnames:
                report.proposed_fallbacks.append(FallbackProposal(
                    config_key="buffer_mode",
                    current_value="disk",
                    proposed_value="replay",
                    reason="磁盘空间不足"))

    if not report.warnings:
        logger.info(
            f"[Preflight] 资源充足 — "
            f"内存 {total_mem / 1e9:.2f}/{mem_budget / 1e9:.2f} GB, "
            f"磁盘 {total_disk / 1e9:.2f}/{avail_disk / 1e9:.2f} GB")

    return report


def apply_fallbacks(
    report: PreflightReport, effective_configs: dict[str, Any]
) -> None:
    """将 proposed_fallbacks 应用到 effective_configs。"""
    for fb in report.proposed_fallbacks:
        effective_configs[fb.config_key] = fb.proposed_value
        msg = (f"[Preflight] {fb.config_key}: "
               f"{fb.current_value} → {fb.proposed_value}（{fb.reason}）")
        report.applied_fallbacks.append(msg)


def _resolve_node_configs(
    node_spec: dict[str, Any],
    effective_configs: dict[str, Any],
    op_cls: type[BaseOp],
) -> dict[str, Any]:
    """简化 config 解析：configs.X → effective_configs，其他 → Op default。"""
    resolved: dict[str, Any] = {}
    op_config_defaults = getattr(op_cls, "CONFIGS", {})
    node_configs = node_spec.get("configs", {})

    for key, link in node_configs.items():
        if isinstance(link, str) and link.startswith("configs."):
            config_name = link[len("configs."):]
            resolved[key] = effective_configs.get(config_name)
        elif not isinstance(link, str) or "." not in link:
            # literal value
            resolved[key] = link

    # fill defaults for unresolved keys
    for key, spec in op_config_defaults.items():
        if key not in resolved:
            resolved[key] = spec.get("default") if isinstance(spec, dict) else None

    return resolved
