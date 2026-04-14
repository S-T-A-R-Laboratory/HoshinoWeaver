"""
多进程 DAG 执行引擎。

将 DAG 的 Op 分配到多个进程，跨进程边界使用 IPCQueue（SharedMemory + Pipe）
实现高效数据传输。对 Op 完全透明。

架构（模型 B —— 独立 Event Loop 型）：

    主进程 (asyncio loop)                    Worker 进程 (asyncio loop)
    ├── Feeder 协程                           ├── Op C ──RCQ──► Op D
    ├── Op A ──RCQ──► Op B                    │           (进程内)
    │           (进程内)                       └── IPCQueue(consumer) ──►
    └── IPCQueue(producer) ──────────────────►
             SharedMemory + Pipe

使用方式：
    用 run_dag_multiprocess() 替代 run_dag() 即可启用多进程。
    或通过 run_from_yaml(..., num_workers=N) 自动选择执行模式。
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
from pathlib import Path
from typing import Any, Awaitable, Optional

from loguru import logger

from ..component.ipc_queue import IPCQueue
from ..component.progress import (
    DummyTracker,
    ProgressTracker,
    ProxyTracker,
    TrackerEventConsumer,
)
from ..component.queue import BaseQueue
from ..ops.base import BaseOp
from .build import ValidatedDag, _iter_node_src_links, _parse_link
from .executor import DAGExecutor
from .wiring import (
    DEFAULT_DAG_SEARCH_PATHS,
    _feed_config,
    _feed_sequence,
    _resolve_sub_dag_yaml,
    instantiate_and_wire,
    run_dag,
    set_dag_search_paths,
)

# ────────────────────────────────────────────────────────────────
# 进程分组
# ────────────────────────────────────────────────────────────────


def compute_process_groups(
    dag: ValidatedDag,
    instances: dict[str, BaseOp],
    num_workers: int = 1,
) -> dict[str, int]:
    """贪心拓扑序分组：将 Op 分配到 num_workers+1 个进程组。

    组 0 为主进程（运行 feeder 和 EXECUTOR=None 的 Op）。
    组 1..num_workers 为 worker 进程。

    使用反亲和约束确保相同 EXECUTOR 类型的 Op 分散到不同进程：
        score(p) = α × (上游在 p 中的数量) - β × (p 中同 EXECUTOR 节点数)

    Args:
        dag: 已验证的 DAG 描述。
        instances: 节点名 → Op 实例。
        num_workers: worker 进程数。0 表示纯单进程。

    Returns:
        node_name → process_group_id 的映射。
    """
    if num_workers <= 0:
        return {name: 0 for name in dag.exec_order}

    total_groups = num_workers + 1  # group 0 = main
    assign: dict[str, int] = {}

    # 每个组中 EXECUTOR 类型的计数
    group_executor_count: dict[int, dict[Optional[str], int]] = {
        g: {} for g in range(total_groups)
    }
    # 每个组中的节点集合
    group_nodes: dict[int, set[str]] = {g: set() for g in range(total_groups)}

    alpha = 1.0  # 连通性奖励
    beta = 2.0   # 同色反亲和惩罚

    for node_name in dag.exec_order:
        op = instances[node_name]
        executor_type = op.EXECUTOR

        if executor_type is None:
            # I/O 型 Op 固定在主进程
            assign[node_name] = 0
            group_nodes[0].add(node_name)
            group_executor_count[0][None] = \
                group_executor_count[0].get(None, 0) + 1
            continue

        best_group = 1
        best_score = float('-inf')

        for g in range(1, total_groups):
            upstream_in_g = sum(
                1 for dep in dag.node_deps.get(node_name, set())
                if assign.get(dep) == g
            )
            same_executor_in_g = group_executor_count[g].get(executor_type, 0)
            score = alpha * upstream_in_g - beta * same_executor_in_g
            if score > best_score:
                best_score = score
                best_group = g

        assign[node_name] = best_group
        group_nodes[best_group].add(node_name)
        group_executor_count[best_group][executor_type] = \
            group_executor_count[best_group].get(executor_type, 0) + 1

    for g in range(total_groups):
        if group_nodes[g]:
            label = "main" if g == 0 else f"worker-{g}"
            members = sorted(group_nodes[g])
            logger.info(f"Process group [{label}]: {members}")

    return assign


# ────────────────────────────────────────────────────────────────
# IPCQueue 注入
# ────────────────────────────────────────────────────────────────


def inject_ipc_queues(
    dag: ValidatedDag,
    instances: dict[str, BaseOp],
    assign: dict[str, int],
) -> list[IPCQueue]:
    """在跨进程边界的连接处注入 IPCQueue，替换默认的 RichContextQueue。

    遍历所有 node→node 连接，当 provider 和 consumer 在不同进程组时，
    将 consumer 端的 RichContextQueue 替换为 IPCQueue，并同步更新
    provider 端 outputs 列表中的引用。

    全局 input/config → worker 节点的连接也需要 IPCQueue：feeder 在主进程，
    consumer 可能在 worker。此处一并处理。

    Args:
        dag: 已验证的 DAG。
        instances: 节点名 → Op 实例（已由 instantiate_and_wire 完成布线）。
        assign: 节点名 → 进程组 ID。

    Returns:
        所有创建的 IPCQueue 列表（用于后续清理）。
    """
    ipc_queues: list[IPCQueue] = []
    nodes_spec = dag.nodes

    for node_name in dag.exec_order:
        node_spec = nodes_spec[node_name]
        consumer_group = assign[node_name]

        for loc, src in _iter_node_src_links(node_spec):
            parsed = _parse_link(src)
            section, arg_name = loc.split(".", 1)
            op_inst = instances[node_name]

            if parsed[0] == "node":
                provider_node, output_name = parsed[1], parsed[2]
                provider_group = assign.get(provider_node, 0)
                if provider_group == consumer_group:
                    continue  # 同进程

            elif parsed[0] in ("inputs", "configs"):
                # 全局 input/config → worker 节点
                if consumer_group == 0:
                    continue  # 主进程内，不需要 IPCQueue
                # feeder 在主进程 (group 0)，consumer 在 worker
            else:
                continue

            # 创建 IPCQueue
            ipc_q = IPCQueue(maxsize=op_inst.MAX_SIZE)
            ipc_queues.append(ipc_q)

            # 替换 consumer 端队列
            if section == "inputs":
                old_queue = op_inst.inputs[arg_name]
                op_inst.inputs[arg_name] = ipc_q
            else:
                old_queue = op_inst.config[arg_name]
                op_inst.config[arg_name] = ipc_q

            if parsed[0] == "node":
                # 替换 provider 端 outputs 列表中的引用
                provider_outputs = instances[provider_node].outputs[output_name]
                for idx, q in enumerate(provider_outputs):
                    if q is old_queue:
                        provider_outputs[idx] = ipc_q
                        break

                logger.info(
                    f"IPCQueue: {provider_node}.{output_name} → "
                    f"{node_name}.{section}.{arg_name} "
                    f"(group {provider_group} → {consumer_group})")
            else:
                # 全局 feeder → worker：feeder targets 中的引用需要替换
                # 这由调用者在 feeder targets 列表中处理
                logger.info(
                    f"IPCQueue: global.{parsed[1]} → "
                    f"{node_name}.{section}.{arg_name} "
                    f"(main → group {consumer_group})")

    return ipc_queues


# ────────────────────────────────────────────────────────────────
# Worker 进程入口
# ────────────────────────────────────────────────────────────────


def _worker_main(
    dag: ValidatedDag,
    worker_nodes: list[str],
    boundary_queues: dict[str, dict[str, IPCQueue]],
    output_boundary_queues: dict[str, dict[str, list[IPCQueue]]],
    cancel_event: mp.Event,
    tracker_queue: mp.Queue,
    barrier: mp.Barrier,
    op_registry_names: dict[str, str],
    dag_search_paths_str: list[str],
) -> None:
    """Worker 进程入口：重建局部 Op 实例、布线、运行 event loop。

    Args:
        dag: DAG 描述（frozen dataclass，可 pickle）。
        worker_nodes: 本进程负责的节点名列表（拓扑序）。
        boundary_queues: consumer 端边界映射。
            {node_name: {"inputs.arg": ipc_q, "configs.arg": ipc_q}}
        output_boundary_queues: provider 端边界映射。
            {node_name: {"output_name": [ipc_q, ...]}}
        cancel_event: 跨进程取消事件 (mp.Event)。
        tracker_queue: 进度事件队列。
        barrier: 启动同步栅栏。
        op_registry_names: 节点名 → op 类名映射。
        dag_search_paths_str: DAG 搜索路径（字符串列表）。
    """
    if dag_search_paths_str:
        set_dag_search_paths([Path(p) for p in dag_search_paths_str])

    from .registry import REGISTERED_OP as _reg
    from ..ops.sub_dag import create_sub_dag_op

    registry = dict(_reg)
    for node_name in worker_nodes:
        op_name = op_registry_names[node_name]
        if op_name not in registry and op_name.endswith(".yaml"):
            resolved = _resolve_sub_dag_yaml(op_name)
            registry[op_name] = create_sub_dag_op(resolved, op_name=op_name)

    # ── 实例化本组 Op ──
    nodes_spec = dag.nodes
    instances: dict[str, BaseOp] = {}
    for node_name in worker_nodes:
        op_name = op_registry_names[node_name]
        instances[node_name] = registry[op_name](name=node_name)

    worker_set = set(worker_nodes)

    # ── 布线：组内 node→node + 边界 IPCQueue ──
    for node_name in worker_nodes:
        node_spec = nodes_spec[node_name]
        for loc, src in _iter_node_src_links(node_spec):
            parsed = _parse_link(src)
            section, arg_name = loc.split(".", 1)
            boundary_key = f"{section}.{arg_name}"
            node_boundaries = boundary_queues.get(node_name, {})

            if boundary_key in node_boundaries:
                # 边界连接：替换默认队列为 IPCQueue
                ipc_q = node_boundaries[boundary_key]
                op_inst = instances[node_name]
                if section == "inputs":
                    op_inst.inputs[arg_name] = ipc_q
                else:
                    op_inst.config[arg_name] = ipc_q
                continue

            if parsed[0] != "node":
                continue

            provider_node = parsed[1]
            if provider_node not in worker_set:
                continue  # provider 在其他进程，通过边界队列

            output_name = parsed[2]
            op_inst = instances[node_name]
            if section == "inputs":
                target_queue = op_inst.inputs[arg_name]
            else:
                target_queue = op_inst.config[arg_name]
            instances[provider_node].outputs[output_name].append(target_queue)

    # ── provider 端边界输出 ──
    for node_name, out_map in output_boundary_queues.items():
        if node_name not in instances:
            continue
        for output_name, ipc_list in out_map.items():
            for ipc_q in ipc_list:
                instances[node_name].outputs[output_name].append(ipc_q)

    # ── 处理未布线 config 默认值 + 可选 input 标记 ──
    feeders: list = []
    for node_name in worker_nodes:
        op_inst = instances[node_name]
        node_spec = nodes_spec[node_name]
        yaml_cfg_keys: set[str] = set()
        cfg_section = node_spec.get("configs")
        if isinstance(cfg_section, dict):
            yaml_cfg_keys = set(cfg_section.keys())
        yaml_inp_keys: set[str] = set()
        inp_section = node_spec.get("inputs")
        if isinstance(inp_section, dict):
            yaml_inp_keys = set(inp_section.keys())

        for key, spec in op_inst.CONFIGS.items():
            if key not in yaml_cfg_keys and "default" in spec:
                feeders.append(
                    _feed_config(f"{node_name}.{key}", spec["default"],
                                 [op_inst.config[key]]))
        for key, spec in op_inst.INPUTS.items():
            if key not in yaml_inp_keys:
                if not spec.get("required", True):
                    op_inst.inputs[key].active = False

    # ── 注入 tracker 和 cancel_event ──
    proxy_tracker = ProxyTracker(tracker_queue)
    for op in instances.values():
        op.tracker = proxy_tracker
        op._cancel_event = cancel_event

    worker_executor = DAGExecutor(list(instances.values()))
    worker_executor.cancel_event = cancel_event

    async def _run():
        barrier.wait()
        await asyncio.gather(*feeders, worker_executor.execute())

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_run())
    except Exception as e:
        logger.error(f"Worker process failed: {e}")
        cancel_event.set()
        raise
    finally:
        loop.close()


# ────────────────────────────────────────────────────────────────
# 边界队列收集
# ────────────────────────────────────────────────────────────────


def _collect_boundary_queues(
    dag: ValidatedDag,
    instances: dict[str, BaseOp],
    assign: dict[str, int],
    worker_nodes: set[str],
) -> tuple[
    dict[str, dict[str, IPCQueue]],
    dict[str, dict[str, list[IPCQueue]]],
]:
    """收集指定 worker 组的边界 IPCQueue 映射。

    Returns:
        (consumer_boundaries, provider_boundaries)
        - consumer_boundaries: {node: {"inputs.arg": ipc_q, ...}}
        - provider_boundaries: {node: {"output_name": [ipc_q, ...]}}
    """
    consumer_map: dict[str, dict[str, IPCQueue]] = {}
    provider_map: dict[str, dict[str, list[IPCQueue]]] = {}
    nodes_spec = dag.nodes

    for node_name in worker_nodes:
        node_spec = nodes_spec[node_name]
        for loc, src in _iter_node_src_links(node_spec):
            section, arg_name = loc.split(".", 1)
            op_inst = instances[node_name]

            if section == "inputs":
                queue = op_inst.inputs[arg_name]
            else:
                queue = op_inst.config[arg_name]

            if isinstance(queue, IPCQueue):
                consumer_map.setdefault(node_name, {})[loc] = queue

    # provider 端：本组 Op ��出中包含的 IPCQueue（consumer 在其他进程）
    for node_name in worker_nodes:
        op_inst = instances[node_name]
        for output_name, queue_list in op_inst.outputs.items():
            for q in queue_list:
                if isinstance(q, IPCQueue):
                    provider_map.setdefault(
                        node_name, {}).setdefault(output_name, []).append(q)

    return consumer_map, provider_map


# ────────────────────────────────────────────────────────────────
# 多进程执行入口
# ────────────────────────────────────────────────────────────────


async def run_dag_multiprocess(
    dag: ValidatedDag,
    global_inputs: dict[str, Any],
    global_configs: dict[str, Any],
    op_registry: Optional[dict[str, type[BaseOp]]] = None,
    progress: bool = True,
    dag_search_paths: Optional[list[Path]] = None,
    tracker: Optional[DummyTracker] = None,
    cancel_event: Optional[asyncio.Event] = None,
    num_workers: int = 1,
) -> dict[str, Any]:
    """多进程执行 DAG。

    在主进程运行 I/O 型 Op 和 feeder，在 worker 进程运行 CPU 型 Op。
    跨进程边界自动使用 IPCQueue（SharedMemory + Pipe）。

    如果所有 Op 都被分配到主进程（无 CPU 型 Op），自动回退到单进程模式。

    Args:
        dag: 已验证的 DAG。
        global_inputs: 全局输入数据。
        global_configs: 全局配置。
        op_registry: Op 注册表。
        progress: 是否显示进度条。
        dag_search_paths: 子图搜索路径。
        tracker: 外部进度追踪器。
        cancel_event: 外部取消事件（asyncio.Event）。
        num_workers: worker 进程数。

    Returns:
        全局输出 name → value。
    """
    if dag_search_paths is not None:
        set_dag_search_paths(dag_search_paths)

    # ── 1) 完整布线（单进程模式，获得全部 Op 实例和拓扑）──
    ops, feeders, output_queues = instantiate_and_wire(
        dag, global_inputs, global_configs, op_registry)

    instances = {op.name: op for op in ops}

    # ── 2) 计算进程分组 ──
    assign = compute_process_groups(dag, instances, num_workers)

    if all(g == 0 for g in assign.values()):
        logger.info("All ops in main process, single-process fallback.")
        return await run_dag(
            dag, global_inputs, global_configs, op_registry,
            progress=progress, dag_search_paths=dag_search_paths,
            tracker=tracker, cancel_event=cancel_event)

    # ── 3) 注入 IPCQueue ──
    ipc_queues = inject_ipc_queues(dag, instances, assign)

    # ── 4) 分离主进程和 worker 进程的 Op ──
    main_ops = [op for op in ops if assign[op.name] == 0]
    worker_groups: dict[int, list[BaseOp]] = {}
    for op in ops:
        g = assign[op.name]
        if g > 0:
            worker_groups.setdefault(g, []).append(op)

    # ── 5) 进度追踪 ──
    tracker_queue: mp.Queue = mp.Queue()
    if tracker is None and progress:
        tracker = ProgressTracker()
    if tracker is not None:
        for op in main_ops:
            op.tracker = tracker
        consumer = TrackerEventConsumer(tracker, tracker_queue)
    else:
        consumer = None

    # ── 6) 跨进程取消事件 ──
    mp_cancel: mp.Event = mp.Event()
    main_executor = DAGExecutor(main_ops)
    if cancel_event is not None:
        main_executor.cancel_event = cancel_event
    for op in main_ops:
        op._cancel_event = main_executor.cancel_event

    # ── 7) 启动 worker 进程 ──
    barrier = mp.Barrier(len(worker_groups) + 1)
    processes: list[mp.Process] = []

    for group_id, group_ops in worker_groups.items():
        worker_node_names = [op.name for op in group_ops]
        worker_set = set(worker_node_names)
        op_name_map = {
            name: dag.nodes[name]["op"] for name in worker_node_names
        }

        consumer_boundaries, provider_boundaries = _collect_boundary_queues(
            dag, instances, assign, worker_set)

        search_paths_str = [str(p) for p in DEFAULT_DAG_SEARCH_PATHS]

        p = mp.Process(
            target=_worker_main,
            args=(
                dag,
                worker_node_names,
                consumer_boundaries,
                provider_boundaries,
                mp_cancel,
                tracker_queue,
                barrier,
                op_name_map,
                search_paths_str,
            ),
            daemon=True,
        )
        processes.append(p)
        p.start()
        logger.info(f"Worker {group_id} started (pid={p.pid}): "
                     f"{worker_node_names}")

    # ── 8) 主进程执行 ──
    results: dict[str, Any] = {}

    async def _collect_outputs():
        async def _get_one(name, queue):
            results[name] = await queue.get()
        await asyncio.gather(
            *[_get_one(n, q) for n, q in output_queues.items()])

    async def _main_run():
        barrier.wait()  # 等所有 worker 就绪
        tasks: list[Awaitable] = [*feeders, _collect_outputs()]
        if main_ops:
            tasks.append(main_executor.execute())
        if consumer is not None:
            tasks.append(consumer.run())
        await asyncio.gather(*tasks)

    try:
        await _main_run()
    except asyncio.CancelledError:
        logger.info("DAG cancelled by external request")
        mp_cancel.set()
        raise
    except Exception as e:
        logger.error(f"DAG multiprocess execution failed: {e}")
        mp_cancel.set()
        raise
    finally:
        if consumer is not None:
            consumer.stop()
        for p in processes:
            p.join(timeout=10)
            if p.is_alive():
                logger.warning(f"Worker pid={p.pid} did not exit, terminating.")
                p.terminate()
        for ipc_q in ipc_queues:
            ipc_q.cleanup()
        if tracker is not None:
            tracker.close_all()

    logger.info("DAG multiprocess execution completed.")
    return results
