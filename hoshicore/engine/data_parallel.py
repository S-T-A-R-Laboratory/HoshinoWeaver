"""
数据并行执行引擎：将 DAG 中可并行的完整管段（I/O + Map 链 + 可分解 Reduce）
整体复制到 N 个 worker 进程，每个 worker 处理 1/N 帧。

核心思想：
    - Dispatcher 仅分发文件路径（~100 bytes/帧），不传图像数据
    - 每个 worker 独立完成：解码 → 处理 → 局部归约
    - 主进程仅做轻量路径分发和最终 merge

替代旧的 Op 粒度多进程方案（multiprocess.py），消除图像级 IPC 瓶颈。

架构：
    Main Process:
        Feeders → SegmentAdapter.dispatch() ─── .collect() → .merge() → downstream
                       │                            ↑
                       │ IPC: 文件路径 (~100B/帧)    │ IPC: partial result (仅 N 个)
                       ▼                            │
        Workers:  I/O(decode) → Map chain → Reduce → partial
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from ..component.ipc_queue import IPCQueue
from ..component.progress import DummyTracker, ProxyTracker
from ..component.queue import BaseQueue, RichContextQueue, StreamExhausted
from ..ops.base import BaseOp, ParallelBaseOp, FilterBaseOp
from .build import ValidatedDag, _iter_node_src_links, _parse_link
from .registry import REGISTERED_OP


# ────────────────────────────────────────────────────────────────
# 帧分配策略
# ────────────────────────────────────────────────────────────────


class FrameDistribution(Enum):
    ROUND_ROBIN = "round_robin"  # 文件列表：worker_i 处理帧 i, i+N, i+2N, ...
    BLOCK_RANGE = "block_range"  # 视频：worker_i 处理帧 [start_i, start_i + count_i)


def _compute_block_ranges(
    total_frames: int, num_workers: int
) -> list[tuple[int, int]]:
    """计算每个 worker 的 (start, count)。"""
    base = total_frames // num_workers
    remainder = total_frames % num_workers
    ranges = []
    offset = 0
    for i in range(num_workers):
        count = base + (1 if i < remainder else 0)
        ranges.append((offset, count))
        offset += count
    return ranges


# ────────────────────────────────────────────────────────────────
# 段描述数据结构
# ────────────────────────────────────────────────────────────────


@dataclass
class ParallelSegment:
    """一个可数据并行的完整段。"""
    io_ops: list[str]                          # 段头帧级 I/O ops（拓扑序）
    map_ops: list[str]                         # 段中 Map ops（拓扑序）
    reduce_op: Optional[str]                   # 段尾可分解 Reduce op
    reduce_extra_inputs: dict[str, str]        # Reduce 额外序列输入: {input_key: "node.output"}
    frame_distribution: FrameDistribution
    op_class_map: dict[str, str]               # node_name → op_class_name
    all_configs_spec: dict[str, dict]          # node_name → Op.CONFIGS
    # 段的全局输入映射: {global_input_name: [node_name.input_key, ...]}
    global_input_feeds: dict[str, list[str]] = field(default_factory=dict)


# ────────────────────────────────────────────────────────────────
# 段检测算法
# ────────────────────────────────────────────────────────────────


def _is_io_eligible(op: BaseOp) -> bool:
    """判断 Op 是否为可内嵌 worker 的帧级 I/O。"""
    return op.DATA_PARALLEL and not isinstance(op, (ParallelBaseOp, FilterBaseOp))


def _is_map_eligible(op: BaseOp) -> bool:
    """判断 Op 是否为可内嵌 worker 的 Map Op。"""
    return isinstance(op, ParallelBaseOp) and op.DATA_PARALLEL


def _is_decomposable_reduce(op: BaseOp) -> bool:
    """判断 Op 是否为可分解的 Reduce。"""
    return op.DECOMPOSABLE and op.EXECUTOR == "cpu"


def detect_parallel_segments(
    dag: ValidatedDag,
    instances: dict[str, BaseOp],
) -> list[ParallelSegment]:
    """检测可数据并行的完整段。

    算法：
    1. 遍历拓扑序，识别帧级 I/O op 为段头起点
    2. 从段头向下延伸，收集连续的 Map ops
    3. 检查末端是否为可分解 Reduce
    4. 验证段内 ops 没有来自段外的额外序列输入（Reduce 的额外输入除外）

    段边界切断条件：
    - 下一个 Op 不满足 eligible 条件
    - 遇到不可分解 Reduce
    - 下一个 Op 是 SubDagOp / FilterBaseOp / DiskBufferWriterOp
    - 下一个 Op 有来自段外的额外序列输入

    Returns:
        检测到的可并行段列表。
    """
    nodes_spec = dag.nodes
    segments: list[ParallelSegment] = []
    used_nodes: set[str] = set()

    # 构建 provider→consumer 的正向邻接表
    forward: dict[str, list[tuple[str, str, str]]] = {}  # provider → [(consumer, section, arg)]
    for node_name in dag.exec_order:
        node_spec = nodes_spec[node_name]
        for loc, src in _iter_node_src_links(node_spec):
            parsed = _parse_link(src)
            if parsed[0] == "node":
                provider_node, output_name = parsed[1], parsed[2]
                forward.setdefault(provider_node, []).append(
                    (node_name, loc.split(".")[0], loc.split(".", 1)[1]))

    # 从每个 I/O eligible op 尝试向下延伸段
    for node_name in dag.exec_order:
        if node_name in used_nodes:
            continue
        op = instances[node_name]
        if not _is_io_eligible(op):
            continue

        # 尝试构建段
        segment = _try_build_segment(
            node_name, dag, instances, forward, used_nodes)
        if segment is not None:
            for n in segment.io_ops + segment.map_ops:
                used_nodes.add(n)
            if segment.reduce_op:
                used_nodes.add(segment.reduce_op)
            segments.append(segment)

    return segments


def _try_build_segment(
    io_start: str,
    dag: ValidatedDag,
    instances: dict[str, BaseOp],
    forward: dict[str, list[tuple[str, str, str]]],
    used_nodes: set[str],
) -> Optional[ParallelSegment]:
    """从一个 I/O op 起始，尝试构建一条可并行段。

    段结构：io_ops → map_ops → reduce_op (可选)

    Returns:
        ParallelSegment or None（无法形成有效段时）
    """
    nodes_spec = dag.nodes
    io_ops = [io_start]
    map_ops: list[str] = []
    reduce_op: Optional[str] = None
    reduce_extra_inputs: dict[str, str] = {}
    segment_nodes: set[str] = {io_start}

    # 向下延伸：找到唯一序列输出的消费链
    current = io_start
    while True:
        consumers = forward.get(current, [])
        # 筛选序列输出的消费者
        seq_consumers = []
        for consumer_name, section, arg in consumers:
            if consumer_name in used_nodes:
                continue
            consumer_op = instances[consumer_name]
            input_spec = consumer_op.INPUTS.get(arg, {}) if section == "inputs" else {}
            if input_spec.get("type") == "sequence":
                seq_consumers.append((consumer_name, section, arg))

        if len(seq_consumers) == 0:
            # 无序列消费者 → 段终止（纯 I/O 段，价值不大，跳过）
            break
        if len(seq_consumers) > 1:
            # 分支点 → 当前段终止（Phase 1 不处理分支合并）
            break

        next_name, next_section, next_arg = seq_consumers[0]
        next_op = instances[next_name]

        # 检查下一个 Op 的所有序列输入是否都来自段内
        if not _all_seq_inputs_from_segment(
                next_name, dag, instances, segment_nodes, reduce_extra_inputs):
            # 有段外序列输入 → 检查是否为 Reduce 的额外输入
            if _is_decomposable_reduce(next_op):
                # Reduce 可以有段外的额外序列输入（如 weight）
                reduce_op = next_name
                _collect_reduce_extra_inputs(
                    next_name, dag, instances, segment_nodes, reduce_extra_inputs)
                break
            else:
                break

        if _is_map_eligible(next_op):
            map_ops.append(next_name)
            segment_nodes.add(next_name)
            current = next_name
        elif _is_decomposable_reduce(next_op):
            reduce_op = next_name
            _collect_reduce_extra_inputs(
                next_name, dag, instances, segment_nodes, reduce_extra_inputs)
            break
        elif _is_io_eligible(next_op) and next_name not in used_nodes:
            # 连续 I/O ops（如 data_loader → 另一个 I/O op）
            io_ops.append(next_name)
            segment_nodes.add(next_name)
            current = next_name
        else:
            # 不满足条件 → 段终止
            break

    # 验证：至少有一个有意义的 ops 组合
    if not map_ops and not reduce_op:
        # 仅 I/O 没有 map/reduce → 不值得段化
        return None

    # 构建 op_class_map
    op_class_map: dict[str, str] = {}
    all_configs_spec: dict[str, dict] = {}
    all_segment_ops = io_ops + map_ops + ([reduce_op] if reduce_op else [])
    for n in all_segment_ops:
        op_class_map[n] = dag.nodes[n]["op"]
        all_configs_spec[n] = dict(instances[n].CONFIGS)

    # 确定帧分配策略
    frame_dist = FrameDistribution.ROUND_ROBIN  # 默认 round-robin

    # 收集段的全局输入映射
    global_input_feeds = _collect_global_input_feeds(
        all_segment_ops, dag, instances)

    segment = ParallelSegment(
        io_ops=io_ops,
        map_ops=map_ops,
        reduce_op=reduce_op,
        reduce_extra_inputs=reduce_extra_inputs,
        frame_distribution=frame_dist,
        op_class_map=op_class_map,
        all_configs_spec=all_configs_spec,
        global_input_feeds=global_input_feeds,
    )

    logger.info(
        f"Detected parallel segment: "
        f"io={io_ops}, map={map_ops}, reduce={reduce_op}, "
        f"extra_inputs={reduce_extra_inputs}")
    return segment


def _all_seq_inputs_from_segment(
    node_name: str,
    dag: ValidatedDag,
    instances: dict[str, BaseOp],
    segment_nodes: set[str],
    exclude_extra: dict,
) -> bool:
    """检查节点的所有序列输入是否全部来自段内或全局输入。"""
    node_spec = dag.nodes[node_name]
    op = instances[node_name]
    for loc, src in _iter_node_src_links(node_spec):
        section, arg_name = loc.split(".", 1)
        if section != "inputs":
            continue
        input_spec = op.INPUTS.get(arg_name, {})
        if input_spec.get("type") != "sequence":
            continue
        parsed = _parse_link(src)
        if parsed[0] == "node":
            if parsed[1] not in segment_nodes:
                return False
        # global inputs 是段外的，但由 Dispatcher 统一处理
    return True


def _collect_reduce_extra_inputs(
    reduce_name: str,
    dag: ValidatedDag,
    instances: dict[str, BaseOp],
    segment_nodes: set[str],
    extra_inputs: dict[str, str],
) -> None:
    """收集 Reduce op 的段外额外序列输入。"""
    node_spec = dag.nodes[reduce_name]
    op = instances[reduce_name]
    for loc, src in _iter_node_src_links(node_spec):
        section, arg_name = loc.split(".", 1)
        if section != "inputs":
            continue
        input_spec = op.INPUTS.get(arg_name, {})
        if input_spec.get("type") != "sequence":
            continue
        parsed = _parse_link(src)
        if parsed[0] == "node" and parsed[1] not in segment_nodes:
            extra_inputs[arg_name] = f"{parsed[1]}.{parsed[2]}"


def _collect_global_input_feeds(
    segment_ops: list[str],
    dag: ValidatedDag,
    instances: dict[str, BaseOp],
) -> dict[str, list[str]]:
    """收集段内 ops 引用的全局输入。"""
    feeds: dict[str, list[str]] = {}
    segment_set = set(segment_ops)
    for node_name in segment_ops:
        node_spec = dag.nodes[node_name]
        for loc, src in _iter_node_src_links(node_spec):
            parsed = _parse_link(src)
            if parsed[0] == "inputs":
                section, arg_name = loc.split(".", 1)
                feeds.setdefault(parsed[1], []).append(
                    f"{node_name}.{section}.{arg_name}")
    return feeds


# ────────────────────────────────────────────────────────────────
# Worker 进程入口
# ────────────────────────────────────────────────────────────────


def _segment_worker_main(
    segment_info: dict,
    all_configs: dict[str, dict[str, Any]],
    input_ipc: IPCQueue,
    output_ipc: IPCQueue,
    cancel_event: mp.Event,
    tracker_queue: mp.Queue,
    worker_id: int,
    dag_search_paths_str: list[str],
) -> None:
    """Worker 进程入口：执行 I/O → Map 链 → Reduce 的完整流程。

    Args:
        segment_info: 段描述（可 pickle 的 dict）。
        all_configs: {node_name: {config_key: value}}，段内每个 Op 的 configs。
        input_ipc: 从主进程接收帧输入的 IPCQueue。
        output_ipc: 向主进程发送 partial result 的 IPCQueue。
        cancel_event: 跨进程取消事件。
        tracker_queue: 进度事件队列。
        worker_id: Worker 编号。
        dag_search_paths_str: DAG 搜索路径。
    """
    from .wiring import set_dag_search_paths
    if dag_search_paths_str:
        set_dag_search_paths([Path(p) for p in dag_search_paths_str])

    from .registry import REGISTERED_OP as _reg
    registry = dict(_reg)

    # 实例化段内 ops
    io_ops: list[BaseOp] = []
    for n in segment_info["io_ops"]:
        cls_name = segment_info["op_classes"][n]
        op = registry[cls_name](name=f"{n}_w{worker_id}")
        io_ops.append(op)

    map_ops: list[BaseOp] = []
    for n in segment_info["map_ops"]:
        cls_name = segment_info["op_classes"][n]
        op = registry[cls_name](name=f"{n}_w{worker_id}")
        map_ops.append(op)

    reduce_op_inst: Optional[BaseOp] = None
    reduce_name = segment_info.get("reduce_op")
    if reduce_name:
        cls_name = segment_info["op_classes"][reduce_name]
        reduce_op_inst = registry[cls_name](name=f"{reduce_name}_w{worker_id}")

    # 注入 tracker
    proxy_tracker = ProxyTracker(tracker_queue)
    for op in io_ops + map_ops + ([reduce_op_inst] if reduce_op_inst else []):
        op.tracker = proxy_tracker
        op._cancel_event = cancel_event

    async def _loop():
        frame_count = await input_ipc.get_length()

        # 如果有 reduce op，构建本地队列进行流式归约
        local_reduce_input: Optional[RichContextQueue] = None
        local_reduce_output: Optional[RichContextQueue] = None
        reduce_task = None

        if reduce_op_inst is not None:
            # 为 reduce op 构建本地队列
            local_reduce_input = RichContextQueue(maxsize=1)
            local_reduce_output = RichContextQueue(maxsize=1)

            # 布线：段内 map 链输出 → reduce 输入 'data'
            for key in reduce_op_inst.INPUTS:
                input_spec = reduce_op_inst.INPUTS[key]
                if input_spec.get("type") == "sequence":
                    if key not in segment_info.get("reduce_extra_inputs", {}):
                        reduce_op_inst.inputs[key] = local_reduce_input
                    else:
                        # 段外额外输入：也通过 input_ipc 传递（Dispatcher 打包了额外输入）
                        extra_q = RichContextQueue(maxsize=1)
                        reduce_op_inst.inputs[key] = extra_q

            # 布线：reduce 输出 → local_reduce_output
            for key in reduce_op_inst.OUTPUTS:
                reduce_op_inst.outputs[key].append(local_reduce_output)

            # 注入 configs
            for key, val in all_configs.get(reduce_name, {}).items():
                if key in reduce_op_inst.config:
                    await reduce_op_inst.config[key].put(val)

            # 标记未布线的可选输入
            for key, spec in reduce_op_inst.INPUTS.items():
                if not reduce_op_inst.inputs[key].active:
                    continue
                if key in segment_info.get("reduce_extra_inputs", {}):
                    continue

            # 设置 reduce 输入长度
            await local_reduce_input.set_length(frame_count)

            # 启动 reduce op 协程
            reduce_task = asyncio.create_task(reduce_op_inst.execute())

        # 处理帧
        processed_count = 0
        range_iter = range(frame_count) if frame_count else []

        for i in range_iter:
            if cancel_event.is_set():
                break

            try:
                frame_input = await input_ipc.get()
            except StreamExhausted:
                break

            # frame_input: dict[str, Any] — 包含文件路径和可能的额外输入
            current = frame_input

            # I/O ops: 逐个执行解码
            for op in io_ops:
                configs = all_configs.get(op.name.rsplit("_w", 1)[0], {})
                current = await _execute_single_op(op, current, configs)

            # Map ops: 逐个执行处理
            for op in map_ops:
                configs = all_configs.get(op.name.rsplit("_w", 1)[0], {})
                current = await _execute_single_op(op, current, configs)

            # 输出
            if reduce_op_inst is not None:
                # 提取 reduce 额外输入
                reduce_extras = {}
                for extra_key in segment_info.get("reduce_extra_inputs", {}):
                    reduce_extras[extra_key] = frame_input.get(
                        f"__reduce_extra_{extra_key}")

                # 喂入 reduce 的主数据
                # current 的第一个值作为 reduce 的主序列输入
                main_val = next(iter(current.values()))
                await local_reduce_input.put(main_val)

                # 喂入 reduce 的额外序列输入
                for extra_key, val in reduce_extras.items():
                    if extra_key in reduce_op_inst.inputs:
                        await reduce_op_inst.inputs[extra_key].put(val)
            else:
                # 纯 Map 段：直接输出
                await output_ipc.put(current)

            processed_count += 1

        if reduce_op_inst is not None:
            # 等待 reduce 完成
            await local_reduce_input.put(BaseQueue._SENTINEL)
            # 额外输入也发 sentinel
            for extra_key in segment_info.get("reduce_extra_inputs", {}):
                if extra_key in reduce_op_inst.inputs:
                    q = reduce_op_inst.inputs[extra_key]
                    if isinstance(q, RichContextQueue):
                        await q.put(BaseQueue._SENTINEL)

            if reduce_task is not None:
                await reduce_task

            # 从 reduce 输出收集 partial result
            partial = {}
            for key in reduce_op_inst.OUTPUTS:
                try:
                    partial[key] = await local_reduce_output.get()
                except StreamExhausted:
                    pass
            await output_ipc.put(partial)
        else:
            pass  # Map 段的帧已逐帧输出

        # 发送结束信号
        await output_ipc.put(BaseQueue._SENTINEL)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_loop())
    except Exception as e:
        logger.error(f"Segment worker {worker_id} failed: {e}")
        cancel_event.set()
        raise
    finally:
        loop.close()


async def _execute_single_op(
    op: BaseOp,
    inputs: dict[str, Any],
    configs: dict[str, Any],
) -> dict[str, Any]:
    """在 worker 内执行单个 Op 的单帧处理。

    对于 ParallelBaseOp: 调用 _async_execute_single
    对于 ImgDataLoaderOp 等 I/O op: 模拟单帧执行
    """
    if isinstance(op, ParallelBaseOp):
        # 构造 Awaitable 输入（模拟队列接口）
        async def _make_awaitable(v):
            return v
        data = {k: _make_awaitable(v) for k, v in inputs.items()}
        return await op._async_execute_single(data, configs)
    else:
        # I/O op: 使用内部的 dataloader 机制处理单个输入
        # 这里需要特殊处理 ImgDataLoaderOp
        from ..ops.dataloader import ImgDataLoaderOp
        from ..component.dataloader import ImgFileListLoader, ArrayLoader

        if isinstance(op, ImgDataLoaderOp):
            loader_type = configs.get("loader_type", "img_file_list")
            loader_class = op.build_loader_class(loader_type)
            # inputs 中应该有 "src" 键，值为文件路径
            src_val = inputs.get("src")
            if src_val is None:
                src_val = next(iter(inputs.values()))
            # 直接调用 loader 的同步 load 方法
            loader_configs = configs.get("configs", {})
            if loader_type == "img_file_list":
                result = await asyncio.to_thread(
                    ImgFileListLoader.load, None, src_val)
            elif loader_type == "img_array":
                result = loader_configs.get("data", {}).get(src_val)
            else:
                result = src_val
            return {"result": result}
        else:
            # 其他 I/O op: 通用单帧模式
            # 将输入灌入 op 的队列，执行，收集输出
            return inputs  # fallback: 透传


# ────────────────────────────────────────────────────────────────
# SegmentAdapter: 在 DAG 中替代整个段
# ────────────────────────────────────────────────────────────────


class SegmentAdapter(BaseOp):
    """将 I/O + Map 链 + 可选 Reduce 包装为数据并行执行单元。

    在 DAG 执行中，SegmentAdapter 替代整个段内所有 ops，
    对 DAGExecutor 表现为一个普通的 BaseOp。
    """

    def __init__(
        self,
        name: str,
        segment: ParallelSegment,
        num_workers: int,
        original_instances: dict[str, BaseOp],
        dag: ValidatedDag,
    ):
        # 不调用 super().__init__，手动设置属性
        self.name = name
        self.segment = segment
        self._num_workers = num_workers
        self._dag = dag
        self._original_instances = original_instances
        self.tracker = DummyTracker()
        self._cancel_event = None

        # 从段的入口 Op 继承 INPUTS/CONFIGS/OUTPUTS
        # 段的输入 = 段头 I/O ops 的输入（全局序列输入）
        # 段的输出 = 段尾的输出（reduce 结果或 map 链末端帧输出）
        self.INPUTS = {}
        self.OUTPUTS = {}
        self.CONFIGS = {}

        # inputs: 段头 I/O ops 的输入
        first_io = segment.io_ops[0]
        first_io_inst = original_instances[first_io]
        self.inputs = dict(first_io_inst.inputs)
        self.INPUTS = dict(first_io_inst.INPUTS)

        # 加入 reduce 额外输入
        if segment.reduce_op and segment.reduce_extra_inputs:
            reduce_inst = original_instances[segment.reduce_op]
            for extra_key, src in segment.reduce_extra_inputs.items():
                self.inputs[f"__reduce_extra_{extra_key}"] = (
                    reduce_inst.inputs[extra_key])
                self.INPUTS[f"__reduce_extra_{extra_key}"] = (
                    reduce_inst.INPUTS[extra_key])

        # configs: 合并段内所有 ops 的 configs
        self.config = {}
        for node_name in (segment.io_ops + segment.map_ops +
                          ([segment.reduce_op] if segment.reduce_op else [])):
            inst = original_instances[node_name]
            for key, queue in inst.config.items():
                self.config[f"{node_name}.{key}"] = queue
                self.CONFIGS[f"{node_name}.{key}"] = inst.CONFIGS.get(key, {})

        # outputs: 段尾的输出
        if segment.reduce_op:
            tail = original_instances[segment.reduce_op]
        elif segment.map_ops:
            tail = original_instances[segment.map_ops[-1]]
        else:
            tail = original_instances[segment.io_ops[-1]]
        self.outputs = dict(tail.outputs)
        self.OUTPUTS = dict(tail.OUTPUTS)

        # 长度信息
        self.length = None

    async def _async_execute(self, configs: dict[str, Any]) -> None:
        """执行数据并行段。"""
        segment = self.segment
        num_workers = self._num_workers

        # 重组 configs: {node_name: {config_key: value}}
        node_configs: dict[str, dict[str, Any]] = {}
        for compound_key, value in configs.items():
            parts = compound_key.split(".", 1)
            if len(parts) == 2:
                node_name, config_key = parts
                node_configs.setdefault(node_name, {})[config_key] = value

        # 创建 per-worker IPCQueue
        input_ipcs: list[IPCQueue] = []
        output_ipcs: list[IPCQueue] = []
        for _ in range(num_workers):
            input_ipcs.append(IPCQueue(maxsize=2))
            output_ipcs.append(IPCQueue(maxsize=2))

        # 构建 segment_info dict（可 pickle）
        segment_info = {
            "io_ops": segment.io_ops,
            "map_ops": segment.map_ops,
            "reduce_op": segment.reduce_op,
            "reduce_extra_inputs": segment.reduce_extra_inputs,
            "op_classes": segment.op_class_map,
        }

        # 取消事件
        mp_cancel = mp.Event()
        tracker_queue = mp.Queue()
        from .wiring import DEFAULT_DAG_SEARCH_PATHS
        search_paths_str = [str(p) for p in DEFAULT_DAG_SEARCH_PATHS]

        # 启动 worker 进程
        processes: list[mp.Process] = []
        for i in range(num_workers):
            p = mp.Process(
                target=_segment_worker_main,
                args=(
                    segment_info,
                    node_configs,
                    input_ipcs[i],
                    output_ipcs[i],
                    mp_cancel,
                    tracker_queue,
                    i,
                    search_paths_str,
                ),
                daemon=True,
            )
            processes.append(p)
            p.start()
            logger.info(f"Segment worker {i} started (pid={p.pid})")

        # 进度消费
        from ..component.progress import TrackerEventConsumer
        consumer = TrackerEventConsumer(self.tracker, tracker_queue)
        consumer_task = asyncio.create_task(consumer.run())

        try:
            # 并发运行 dispatcher 和 collector
            await asyncio.gather(
                self._dispatch(input_ipcs, mp_cancel),
                self._collect(output_ipcs, mp_cancel),
            )
        except Exception as e:
            logger.error(f"SegmentAdapter failed: {e}")
            mp_cancel.set()
            raise
        finally:
            # 清理
            consumer.stop()
            await consumer_task
            for p in processes:
                p.join(timeout=10)
                if p.is_alive():
                    logger.warning(
                        f"Segment worker pid={p.pid} did not exit, terminating.")
                    p.terminate()
            for ipc in input_ipcs + output_ipcs:
                ipc.cleanup()

    async def _dispatch(
        self,
        input_ipcs: list[IPCQueue],
        cancel_event: mp.Event,
    ) -> None:
        """分发输入到 workers（round-robin）。"""
        num_workers = len(input_ipcs)
        segment = self.segment
        has_reduce_extras = bool(segment.reduce_extra_inputs)

        # 获取主输入队列（段头 I/O 的第一个序列输入）
        main_input_key = None
        for key, spec in self.INPUTS.items():
            if spec.get("type") == "sequence" and not key.startswith("__reduce_extra_"):
                main_input_key = key
                break

        if main_input_key is None:
            logger.warning("SegmentAdapter: no main sequence input found")
            return

        total_frames = self.length
        if total_frames is None:
            logger.warning("SegmentAdapter: unknown frame count, using sentinel mode")
            return

        # 计算每个 worker 的帧数
        worker_counts = [0] * num_workers

        if segment.frame_distribution == FrameDistribution.ROUND_ROBIN:
            for i in range(total_frames):
                worker_idx = i % num_workers
                worker_counts[worker_idx] += 1
        else:
            # BLOCK_RANGE
            ranges = _compute_block_ranges(total_frames, num_workers)
            worker_counts = [count for _, count in ranges]

        # 设置每个 worker 的输入长度
        for i in range(num_workers):
            await input_ipcs[i].set_length(worker_counts[i])

        # 分发帧
        main_queue = self.inputs[main_input_key]
        extra_queues = {}
        for extra_key in segment.reduce_extra_inputs:
            extra_queues[extra_key] = self.inputs[f"__reduce_extra_{extra_key}"]

        for i in range(total_frames):
            if cancel_event.is_set():
                break

            # 读取主输入
            frame_data = {main_input_key: await main_queue.get()}

            # 读取 reduce 额外输入
            for extra_key, extra_queue in extra_queues.items():
                frame_data[f"__reduce_extra_{extra_key}"] = await extra_queue.get()

            # round-robin 分发
            worker_idx = i % num_workers
            await input_ipcs[worker_idx].put(frame_data)

        # 发送 sentinel
        for ipc in input_ipcs:
            await ipc.put(BaseQueue._SENTINEL)

    async def _collect(
        self,
        output_ipcs: list[IPCQueue],
        cancel_event: mp.Event,
    ) -> None:
        """收集 worker 输出。"""
        segment = self.segment

        if segment.reduce_op:
            # 每个 worker 产生一个 partial result → merge
            partials: list[dict[str, Any]] = []
            for i, ipc in enumerate(output_ipcs):
                try:
                    partial = await ipc.get()
                    partials.append(partial)
                except StreamExhausted:
                    logger.warning(f"Worker {i} sent no partial result")

            if not partials:
                raise ValueError("No partial results from workers")

            # merge
            reduce_cls_name = segment.op_class_map[segment.reduce_op]
            reduce_cls = REGISTERED_OP[reduce_cls_name]
            final = reduce_cls.merge_partial(partials)
            await self._broadcast_outputs(final)
        else:
            # 纯 Map 段：有序合并帧输出
            await self._collect_ordered(output_ipcs)

    async def _collect_ordered(
        self,
        output_ipcs: list[IPCQueue],
    ) -> None:
        """有序收集 Map 段的帧输出（round-robin 顺序）。"""
        num_workers = len(output_ipcs)
        total_frames = self.length or 0
        for i in range(total_frames):
            worker_idx = i % num_workers
            try:
                result = await output_ipcs[worker_idx].get()
                await self._broadcast_outputs(result)
            except StreamExhausted:
                break

    def _infer_output_length(self, input_lengths):
        """段的输出长度推断。"""
        if self.segment.reduce_op:
            # Reduce 段：输出单个结果
            return 1
        # Map 段：输出长度 = 输入长度
        for length in input_lengths.values():
            if length is not None:
                return length
        return None


# ────────────────────────────────────────────────────────────────
# DAG 替换函数
# ────────────────────────────────────────────────────────────────


def _auto_worker_count() -> int:
    """自动检测 worker 数量。"""
    cpu_count = os.cpu_count() or 4
    return max(2, cpu_count - 2)


def apply_data_parallelism(
    dag: ValidatedDag,
    ops: list[BaseOp],
    instances: dict[str, BaseOp],
    num_workers: Optional[int] = None,
) -> list[BaseOp]:
    """检测可并行段并替换为 SegmentAdapter。

    Args:
        dag: 已验证的 DAG。
        ops: 按拓扑序排列的 Op 实例列表。
        instances: 节点名 → Op 实例。
        num_workers: worker 进程数。None 使用自动检测。

    Returns:
        替换后的 Op 列表（SegmentAdapter 替代段内 ops）。
    """
    if num_workers is None:
        num_workers = _auto_worker_count()
    if num_workers <= 1:
        logger.info("Data parallelism disabled (num_workers <= 1)")
        return ops

    segments = detect_parallel_segments(dag, instances)
    if not segments:
        logger.info("No parallel segments detected")
        return ops

    # 收集所有被段化的节点
    segment_nodes: set[str] = set()
    adapters: dict[str, SegmentAdapter] = {}  # 段内第一个 op 名 → adapter

    for i, segment in enumerate(segments):
        all_seg_ops = (segment.io_ops + segment.map_ops +
                       ([segment.reduce_op] if segment.reduce_op else []))
        for n in all_seg_ops:
            segment_nodes.add(n)

        adapter_name = f"segment_adapter_{i}"
        adapter = SegmentAdapter(
            name=adapter_name,
            segment=segment,
            num_workers=num_workers,
            original_instances=instances,
            dag=dag,
        )
        # 以段头第一个 op 的位置插入 adapter
        adapters[segment.io_ops[0]] = adapter

        logger.info(
            f"Created SegmentAdapter '{adapter_name}' "
            f"({num_workers} workers) replacing ops: {all_seg_ops}")

    # 重建 ops 列表：段内 ops 替换为 adapter，其余保留
    new_ops: list[BaseOp] = []
    for op in ops:
        if op.name in segment_nodes:
            if op.name in adapters:
                new_ops.append(adapters[op.name])
            # 段内其他 ops 被移除
        else:
            new_ops.append(op)

    logger.info(
        f"Data parallelism applied: {len(segments)} segment(s), "
        f"{len(new_ops)} ops in final DAG")
    return new_ops
