"""
预检系统：在 DAG 执行前串行运行检测项，每项生成问题列表并通过回调交互处置。

检测项分类：
    - 资源检查：估算峰值内存/磁盘开销，与系统可用资源比对（始终运行，排第一位）
    - 参数合法性检查：验证 temp_path 等配置参数的合法性（可通过 checks 注入）

回调签名（preflight_callback）：
    (result: CheckResult) -> PreflightAction

回调返回值（PreflightAction）：
    - "apply"  — 应用当前检测项的 fix 建议并继续
    - "ignore" — 忽略当前检测项的问题，继续执行
    - "abort"  — 中止执行

行为模式（无 callback 时，由全局配置 auto_fallback 控制）：
    - auto_fallback=true + 有 fix → 静默应用 + info log
    - auto_fallback=false 或无 fix → 只 warn，不阻断
"""
from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Optional

PreflightAction = Literal["apply", "ignore", "abort"]

import psutil
from loguru import logger

from ..component.image_io import peek_shape
from ..ops.base import BaseOp
from .build import ValidatedDag
from .registry import REGISTERED_OP

MEMORY_SAFETY_FACTOR = 0.7
MEMORY_FIXED_OVERHEAD = 200 * 1024 * 1024  # 200MB
RESOURCE_CHECK_NAME = "资源检查"
CONFIG_VALIDITY_CHECK_NAME = "参数合法性"


class PreflightAbortError(Exception):
    """用户拒绝预检建议时抛出。"""
    pass


@dataclass
class ResourceEstimate:
    peak_memory_bytes: int
    peak_disk_bytes: int


@dataclass
class _ResourceBreakdown:
    total_mem: int
    total_disk: int
    non_chunk_mem: int


@dataclass
class PreflightFix:
    """预检问题的修复建议（纯 config key 替换，无副作用）。"""
    config_key: str
    current_value: str
    proposed_value: str
    reason: str


@dataclass
class PreflightIssue:
    severity: Literal["warning", "error"]
    code: str
    message: str
    fix: PreflightFix | None = None


@dataclass
class CheckResult:
    """单个检测项的全部结果，传给 preflight_callback。"""
    check_name: str
    issues: list[PreflightIssue]
    still_problematic_after_fix: bool = False

    @property
    def has_fix(self) -> bool:
        return any(i.fix is not None for i in self.issues)


CheckFn = Callable[[ValidatedDag, dict[str, Any], dict[str, Any]], CheckResult]


@dataclass
class PreflightReport:
    """内部数据，仅供 runtime_plan.py 使用；不传给 preflight_callback。"""
    estimate: ResourceEstimate
    available_memory_bytes: int
    available_disk_bytes: int
    non_chunk_mem: int = 0
    post_fallback_non_chunk_mem: int | None = None


def apply_check_fixes(
    result: CheckResult,
    effective_configs: dict[str, Any],
) -> list[str]:
    """将 result.issues 中所有带 fix 的 issue 应用到 effective_configs。
    返回人类可读的变更描述列表。"""
    applied: list[str] = []
    for issue in result.issues:
        if issue.fix is not None:
            fx = issue.fix
            effective_configs[fx.config_key] = fx.proposed_value
            applied.append(
                f"[Preflight] {fx.config_key}: "
                f"{fx.current_value} → {fx.proposed_value}（{fx.reason}）"
            )
    return applied


# ── 内部资源估算辅助函数 ──────────────────────────────────────────────────────

def _estimate_dag_resources(
    dag: ValidatedDag,
    effective_configs: dict[str, Any],
    registry: dict[str, type[BaseOp]],
    frame_bytes: int,
    n_frames: int,
    dtype_bytes: int,
    *,
    log_details: bool = False,
) -> _ResourceBreakdown:
    """统一估算 DAG 资源，并拆出 planner 可扣减的非 chunk 内存。"""
    total_mem = 0
    total_disk = 0
    non_chunk_mem = 0

    for node_name in dag.exec_order:
        node_spec = dag.nodes[node_name]
        op_name = node_spec["op"]
        op_cls = registry.get(op_name)
        if op_cls is None:
            continue
        node_configs = _resolve_node_configs(node_spec, effective_configs, op_cls)
        mem, disk = op_cls.estimate_resources(
            node_configs, frame_bytes, n_frames, dtype_bytes)
        if log_details and (mem != 0 or disk != 0):
            logger.info(
                f"[Preflight] {op_name} 资源需求: {mem/1e9:.2f} GB, {disk/1e9:.2f} GB"
            )
        total_mem += mem
        total_disk += disk
        if not getattr(op_cls, "CHUNK_PLANNED", False):
            non_chunk_mem += mem

    # queue 是 DAG 流水线固定开销，和 chunk_rows 无关，必须从 chunk_budget 扣除。
    queue_mem = _estimate_queue_overhead(dag, registry, frame_bytes)
    if log_details and queue_mem > 0:
        logger.info(f"[Preflight] 队列开销: {queue_mem / 1e9:.2f} GB")
    total_mem += queue_mem
    non_chunk_mem += queue_mem
    return _ResourceBreakdown(total_mem, total_disk, non_chunk_mem)


def _simulate_post_fallback(
    dag: ValidatedDag,
    effective_configs: dict[str, Any],
    fixes: list[PreflightFix],
    registry: dict[str, type[BaseOp]],
    frame_bytes: int,
    n_frames: int,
    dtype_bytes: int,
) -> _ResourceBreakdown:
    """模拟 fix 应用后的资源估算，不修改原始 configs。"""
    simulated = {**effective_configs}
    for fx in fixes:
        simulated[fx.config_key] = fx.proposed_value

    return _estimate_dag_resources(
        dag, simulated, registry, frame_bytes, n_frames, dtype_bytes)


def _estimate_queue_overhead(
    dag: ValidatedDag,
    registry: dict[str, type[BaseOp]],
    frame_bytes: int,
) -> int:
    """估算 DAG 队列中 in-flight 帧的内存开销。

    广播传递引用（非拷贝），扇出场景同一数据源只计一次。
    每个节点的每个 sequence 类型输出端口贡献 1 帧（maxsize=1）。
    """
    queue_frames = 0

    for node_name in dag.exec_order:
        node_spec = dag.nodes[node_name]
        op_name = node_spec["op"]
        op_cls = registry.get(op_name)
        if op_cls is None:
            continue

        outputs_decl = getattr(op_cls, "OUTPUTS", {})
        for port_spec in outputs_decl.values():
            if isinstance(port_spec, dict) and port_spec.get("type") == "sequence":
                queue_frames += 1

    return queue_frames * frame_bytes


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
            if config_name in effective_configs:
                resolved[key] = effective_configs[config_name]
            elif config_name != key and key in effective_configs:
                resolved[key] = effective_configs[key]
            else:
                resolved[key] = None
        elif not isinstance(link, str) or "." not in link:
            resolved[key] = link

    # fill defaults for unresolved keys
    for key, spec in op_config_defaults.items():
        if key not in resolved:
            if isinstance(spec, dict) and spec.get("global") and key in effective_configs:
                resolved[key] = effective_configs[key]
            else:
                resolved[key] = spec.get("default") if isinstance(spec, dict) else None

    return resolved


# ── 内置检测项 ────────────────────────────────────────────────────────────────

def _resource_check(
    dag: ValidatedDag,
    effective_configs: dict[str, Any],
    global_inputs: dict[str, Any],
    op_registry: Optional[dict[str, type[BaseOp]]] = None,
) -> tuple[CheckResult, PreflightReport]:
    """资源估算检测，始终作为第一个检测项运行。
    返回 (CheckResult, PreflightReport)，后者仅供 runtime_plan.py 使用。"""
    registry = op_registry or REGISTERED_OP

    # 1. Peek frame shape
    fnames = global_inputs.get("fnames")
    if not fnames or not isinstance(fnames, (list, tuple)) or len(fnames) == 0:
        return (
            CheckResult(check_name=RESOURCE_CHECK_NAME, issues=[]),
            PreflightReport(
                estimate=ResourceEstimate(0, 0),
                available_memory_bytes=psutil.virtual_memory().available,
                available_disk_bytes=0,
            ),
        )

    try:
        shape, dtype_bytes = peek_shape(fnames[0])
    except (FileNotFoundError, ValueError, OSError) as e:
        logger.warning(f"[Preflight] 无法 peek 首帧: {e}")
        return (
            CheckResult(check_name=RESOURCE_CHECK_NAME, issues=[]),
            PreflightReport(
                estimate=ResourceEstimate(0, 0),
                available_memory_bytes=psutil.virtual_memory().available,
                available_disk_bytes=0,
            ),
        )

    frame_bytes = 1
    for dim in shape:
        frame_bytes *= dim
    frame_bytes *= dtype_bytes
    n_frames = len(fnames)

    # 2. 遍历节点，累加资源估算
    breakdown = _estimate_dag_resources(
        dag, effective_configs, registry, frame_bytes, n_frames, dtype_bytes,
        log_details=True)
    total_mem = breakdown.total_mem
    total_disk = breakdown.total_disk

    # 3. 获取可用资源
    avail_mem = psutil.virtual_memory().available
    temp_path = effective_configs.get("temp_path") or tempfile.gettempdir()
    # 路径不存在时回退到系统默认路径估算
    try:
        avail_disk = shutil.disk_usage(temp_path).free
    except OSError:
        avail_disk = shutil.disk_usage(tempfile.gettempdir()).free

    planner_report = PreflightReport(
        estimate=ResourceEstimate(total_mem, total_disk),
        available_memory_bytes=avail_mem,
        available_disk_bytes=avail_disk,
        non_chunk_mem=breakdown.non_chunk_mem,
    )

    # 4. 比较 + 生成 issues
    issues: list[PreflightIssue] = []
    mem_budget = int(avail_mem * MEMORY_SAFETY_FACTOR) - MEMORY_FIXED_OVERHEAD

    if total_mem > 0 and (mem_budget <= 0 or total_mem > mem_budget):
        fix = None
        if effective_configs.get("buffer_mode") == "memory":
            fix = PreflightFix(
                config_key="buffer_mode",
                current_value="memory",
                proposed_value="disk",
                reason="内存不足")
        issues.append(PreflightIssue(
            severity="warning",
            code="resource.memory",
            message=(
                f"预估峰值内存 {total_mem / 1e9:.2f} GB，"
                f"可用预算 {mem_budget / 1e9:.2f} GB"
                f"（超出 {(total_mem - mem_budget) / 1e9:.2f} GB）"),
            fix=fix,
        ))

    if total_disk > avail_disk:
        fix = None
        if effective_configs.get("buffer_mode") == "disk":
            has_fnames = "fnames" in global_inputs and global_inputs["fnames"]
            if has_fnames:
                fix = PreflightFix(
                    config_key="buffer_mode",
                    current_value="disk",
                    proposed_value="replay",
                    reason="磁盘空间不足")
        issues.append(PreflightIssue(
            severity="warning",
            code="resource.disk",
            message=(
                f"预估磁盘缓存 {total_disk / 1e9:.2f} GB，"
                f"可用空间 {avail_disk / 1e9:.2f} GB"
                f"（不足 {(total_disk - avail_disk) / 1e9:.2f} GB）"),
            fix=fix,
        ))

    # 5. 预测 fix 后是否仍然超限
    still_problematic = False
    if issues and any(i.fix for i in issues):
        fixes = [i.fix for i in issues if i.fix]
        post = _simulate_post_fallback(
            dag, effective_configs, fixes,
            registry, frame_bytes, n_frames, dtype_bytes)
        planner_report.post_fallback_non_chunk_mem = post.non_chunk_mem

        mem_still_over = (
            post.total_mem > 0 and
            (mem_budget <= 0 or post.total_mem > mem_budget)
        )
        disk_still_over = post.total_disk > avail_disk
        if mem_still_over or disk_still_over:
            still_problematic = True
            parts = []
            if mem_still_over:
                parts.append(
                    f"内存 {post.total_mem / 1e9:.2f}/{mem_budget / 1e9:.2f} GB"
                    f"（可能需要更改参数或缩小运行输入分辨率）")
            if disk_still_over:
                parts.append(
                    f"磁盘 {post.total_disk / 1e9:.2f}/{avail_disk / 1e9:.2f} GB"
                    f"（可减少输入帧数或清理磁盘空间）")
            issues.append(PreflightIssue(
                severity="warning",
                code="resource.still_problematic_after_fix",
                message=f"降级后资源仍然不足（{'; '.join(parts)}），执行可能失败",
                fix=None,
            ))
    elif issues:
        still_problematic = True

    if not issues:
        logger.info(
            f"[Preflight] 资源充足 — "
            f"内存 {total_mem / 1e9:.2f}/{mem_budget / 1e9:.2f} GB, "
            f"磁盘 {total_disk / 1e9:.2f}/{avail_disk / 1e9:.2f} GB")

    return (
        CheckResult(
            check_name=RESOURCE_CHECK_NAME,
            issues=issues,
            still_problematic_after_fix=still_problematic,
        ),
        planner_report,
    )


def config_validity_check(
    dag: ValidatedDag,
    effective_configs: dict[str, Any],
    global_inputs: dict[str, Any],
) -> CheckResult:
    """检查配置参数合法性。

    当前检查项：
    - temp_path 可用性（仅在 buffer_mode 需要磁盘缓存时）：
        - config.cache_path.missing      — 路径不存在，fix: 回退系统默认
        - config.cache_path.not_writable — 路径不可写，fix: 回退系统默认
    """
    _ = dag, global_inputs
    issues: list[PreflightIssue] = []

    buffer_mode = effective_configs.get("buffer_mode", "disk")
    if buffer_mode in ("memory", "replay"):
        return CheckResult(check_name=CONFIG_VALIDITY_CHECK_NAME, issues=[])

    temp_path_str = effective_configs.get("temp_path")
    if not temp_path_str:
        return CheckResult(check_name=CONFIG_VALIDITY_CHECK_NAME, issues=[])
    system_default = tempfile.gettempdir()
    path = Path(temp_path_str)

    if not path.exists():
        issues.append(PreflightIssue(
            severity="error",
            code="config.cache_path.missing",
            message=f"缓存路径不存在: {temp_path_str}",
            fix=PreflightFix(
                config_key="temp_path",
                current_value=temp_path_str,
                proposed_value=system_default,
                reason="路径不存在，回退到系统默认"),
        ))
    elif not os.access(str(path), os.W_OK):
        issues.append(PreflightIssue(
            severity="error",
            code="config.cache_path.not_writable",
            message=f"缓存路径不可写: {temp_path_str}",
            fix=PreflightFix(
                config_key="temp_path",
                current_value=temp_path_str,
                proposed_value=system_default,
                reason="路径不可写，回退到系统默认"),
        ))

    return CheckResult(check_name=CONFIG_VALIDITY_CHECK_NAME, issues=issues)


# ── 统一入口 ──────────────────────────────────────────────────────────────────

DEFAULT_CHECKS: list[CheckFn] = [config_validity_check]


def run_preflight_checks(
    dag: ValidatedDag,
    effective_configs: dict[str, Any],
    global_inputs: dict[str, Any],
    checks: list[CheckFn] | None = None,
    op_registry: Optional[dict[str, type[BaseOp]]] = None,
) -> tuple[PreflightReport, list[CheckResult]]:
    """串行运行所有预检项。

    始终先运行资源检查（第一位），再运行 checks 列表中的检测项。
    checks=None 时使用 DEFAULT_CHECKS。

    返回 (PreflightReport, list[CheckResult])：
    - PreflightReport：内存规划数据，传给 runtime_plan.py
    - list[CheckResult]：每项检测结果，按顺序传给 preflight_callback
    """
    resource_result, planner_report = _resource_check(
        dag, effective_configs, global_inputs, op_registry)
    additional = checks if checks is not None else DEFAULT_CHECKS
    check_results: list[CheckResult] = [resource_result]
    for check_fn in additional:
        check_results.append(check_fn(dag, effective_configs, global_inputs))

    return planner_report, check_results
