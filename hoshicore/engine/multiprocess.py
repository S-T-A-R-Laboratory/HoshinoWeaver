"""
多进程 DAG 执行引擎（数据并行模式）。

将 DAG 中可并行的完整管段（I/O + Map 链 + 可分解 Reduce）整体复制到
N 个 worker 进程，每个 worker 处理 1/N 帧并独立完成解码→处理→局部归约，
主进程仅做轻量路径分发和最终 merge。

架构：
    主进程 (asyncio event loop)
    ├── Feeders（全局输入/配置推送）
    ├── SegmentAdapter（替代段内所有 ops）
    │   ├── dispatch: 分发文件路径到 workers (~100 bytes/帧)
    │   ├── collect:  收集 partial results (仅 N 个)
    │   └── merge:    合并为最终结果
    ├── 非段化 Ops（不可分解 Reduce 等）
    └── 结果收集

使用方式：
    通过 run_from_yaml(..., num_workers=N) 自动选择执行模式。
    run_dag_multiprocess() 是直接入口。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from ..component.progress import DummyTracker, ProgressTracker
from ..ops.base import BaseOp
from .build import ValidatedDag
from .segment_adapter import apply_data_parallelism, _auto_worker_count
from .executor import DAGExecutor
from .wiring import (
    instantiate_and_wire,
    run_dag,
    set_dag_search_paths,
)


async def run_dag_multiprocess(
    dag: ValidatedDag,
    global_inputs: dict[str, Any],
    global_configs: dict[str, Any],
    op_registry: Optional[dict[str, type[BaseOp]]] = None,
    progress: bool = True,
    dag_search_paths: Optional[list[Path]] = None,
    tracker: Optional[DummyTracker] = None,
    cancel_event: Optional[asyncio.Event] = None,
    num_workers: Optional[int] = None,
) -> dict[str, Any]:
    """数据并行执行 DAG。

    1. 标准布线：实例化 Op、连接队列、创建 feeder
    2. 段检测：识别可数据并行的管段
    3. 段替换：用 SegmentAdapter 替代段内所有 ops
    4. 执行：SegmentAdapter 内部管理 worker 进程

    如果未检测到可并行段，自动回退到单进程模式。

    Args:
        dag: 已验证的 DAG。
        global_inputs: 全局输入数据。
        global_configs: 全局配置。
        op_registry: Op 注册表。
        progress: 是否显示进度条。
        dag_search_paths: 子图搜索路径。
        tracker: 外部进度追踪器。
        cancel_event: 外部取消事件（asyncio.Event）。
        num_workers: worker 进程数。None 使用自动检测。

    Returns:
        全局输出 name → value。
    """
    if num_workers is None:
        num_workers = _auto_worker_count()

    if num_workers <= 1:
        logger.info("Data parallelism disabled (num_workers <= 1), "
                     "falling back to single-process.")
        return await run_dag(
            dag, global_inputs, global_configs, op_registry,
            progress=progress, dag_search_paths=dag_search_paths,
            tracker=tracker, cancel_event=cancel_event)

    if dag_search_paths is not None:
        set_dag_search_paths(dag_search_paths)

    # ── 1) 标准布线 ──
    ops, feeders, output_queues = instantiate_and_wire(
        dag, global_inputs, global_configs, op_registry)

    instances = {op.name: op for op in ops}

    # ── 2) 段检测 + 替换 ──
    new_ops = apply_data_parallelism(dag, ops, instances, num_workers)

    if len(new_ops) == len(ops):
        # 无段替换，回退到单进程
        logger.info("No parallel segments detected, single-process fallback.")
        # 关闭旧 feeders（避免 coroutine 泄漏），重新走 run_dag
        for f in feeders:
            if hasattr(f, 'close'):
                f.close()
        return await run_dag(
            dag, global_inputs, global_configs, op_registry,
            progress=progress, dag_search_paths=dag_search_paths,
            tracker=tracker, cancel_event=cancel_event)

    # ── 3) 注入进度追踪器 ──
    if tracker is None and progress:
        tracker = ProgressTracker()
    if tracker is not None:
        for op in new_ops:
            op.tracker = tracker

    # ── 4) 创建执行器 ──
    executor = DAGExecutor(new_ops)
    if cancel_event is not None:
        executor.cancel_event = cancel_event
    for op in new_ops:
        op._cancel_event = executor.cancel_event

    # ── 5) 结果收集 ──
    results: dict[str, Any] = {}

    async def _collect_outputs():
        async def _get_one(name, queue):
            results[name] = await queue.get()
        await asyncio.gather(
            *[_get_one(n, q) for n, q in output_queues.items()])

    logger.info(f"DAG data-parallel execution starting "
                f"({len(new_ops)} ops, {num_workers} workers)...")

    # feeder 协程作为后台 task 运行：executor/collect 完成后自动取消。
    # 避免 feeder 因目标队列已满、无人消费而永久阻塞在 put() 上。
    feeder_tasks = [asyncio.ensure_future(f) for f in feeders]
    try:
        await asyncio.gather(executor.execute(), _collect_outputs())
    except asyncio.CancelledError:
        logger.info("DAG cancelled by external request")
        raise
    except Exception as e:
        logger.error(f"DAG data-parallel execution failed: {e}")
        raise
    finally:
        for t in feeder_tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*feeder_tasks, return_exceptions=True)
        if tracker is not None:
            tracker.close_all()

    logger.info("DAG data-parallel execution completed.")
    return results
