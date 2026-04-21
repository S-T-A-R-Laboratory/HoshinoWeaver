"""
段检测算法：识别 DAG 中可数据并行的管段（I/O + Map + Reduce/DiskBuffer 终端）。

管段结构：
    io_ops → map_ops → terminal(s)

终端类型：
    - DECOMPOSABLE_REDUCE: 可分解 Reduce（MeanStacker, TrailStacker）
    - DISK_BUFFER: 磁盘帧缓冲（DiskBufferWriterOp）

Phase 2 支持分支点处多终端 + 迭代式 Reduce（BUFFER_ITERATOR）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from loguru import logger

from ..ops.base import BaseOp, ParallelBaseOp, FilterBaseOp
from .build import ValidatedDag, _iter_node_src_links, _parse_link


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
# 终端类型 + 段描述数据结构
# ────────────────────────────────────────────────────────────────


class TerminalType(Enum):
    DECOMPOSABLE_REDUCE = "decomposable_reduce"  # MeanStacker, TrailStacker 等
    DISK_BUFFER = "disk_buffer"                  # DiskBufferWriterOp


@dataclass
class SegmentTerminal:
    """段的一个终端分支。"""
    node_name: str
    terminal_type: TerminalType
    extra_inputs: dict[str, str] = field(default_factory=dict)
    # 段外额外序列输入: {input_key: "node.output"}


@dataclass
class ParallelSegment:
    """一个可数据并行的完整段。"""
    io_ops: list[str]                          # 段头帧级 I/O ops（拓扑序）
    map_ops: list[str]                         # 段中 Map ops（拓扑序）
    reduce_op: Optional[str]                   # 段尾可分解 Reduce op (Phase 1 兼容)
    reduce_extra_inputs: dict[str, str]        # Reduce 额外序列输入: {input_key: "node.output"}
    frame_distribution: FrameDistribution
    op_class_map: dict[str, str]               # node_name → op_class_name
    all_configs_spec: dict[str, dict]          # node_name → Op.CONFIGS
    # 段的全局输入映射: {global_input_name: [node_name.input_key, ...]}
    global_input_feeds: dict[str, list[str]] = field(default_factory=dict)
    # Phase 2 多终端
    terminals: list[SegmentTerminal] = field(default_factory=list)
    # Phase 2 迭代式 Reduce ops（消费 DiskBuffer 的 BUFFER_ITERATOR ops）
    iterator_ops: list[str] = field(default_factory=list)


# ────────────────────────────────────────────────────────────────
# Op 类型判定
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


def _is_disk_buffer(op: BaseOp) -> bool:
    """判断 Op 是否为 DiskBufferWriterOp。"""
    return getattr(op, "IS_DISK_BUFFER", False)


def _is_buffer_iterator(op: BaseOp) -> bool:
    """判断 Op 是否为消费 buffer 的迭代式 Reduce。"""
    return getattr(op, "BUFFER_ITERATOR", False)


# ────────────────────────────────────────────────────────────────
# 段检测主算法
# ────────────────────────────────────────────────────────────────


def detect_parallel_segments(
    dag: ValidatedDag,
    instances: dict[str, BaseOp],
) -> list[ParallelSegment]:
    """检测可数据并行的完整段。

    算法：
    1. 遍历拓扑序，识别帧级 I/O op 为段头起点
    2. 从段头向下延伸，收集连续的 Map ops
    3. 检查末端是否为可分解 Reduce / DiskBuffer / 分支点
    4. 分支点处所有分支都可段化（DECOMPOSABLE 或 DISK_BUFFER）则构建多终端段
    5. DiskBuffer 终端的下游 BUFFER_ITERATOR ops 纳入 iterator_ops

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
            if src == "__inactive__":
                continue
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
            for t in segment.terminals:
                used_nodes.add(t.node_name)
            for it_op in segment.iterator_ops:
                used_nodes.add(it_op)
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

    段结构：io_ops → map_ops → terminal(s)
    终端可以是单个 Reduce（Phase 1 兼容）或多终端（Phase 2）。

    Returns:
        ParallelSegment or None（无法形成有效段时）
    """
    nodes_spec = dag.nodes
    io_ops = [io_start]
    map_ops: list[str] = []
    reduce_op: Optional[str] = None
    reduce_extra_inputs: dict[str, str] = {}
    segment_nodes: set[str] = {io_start}
    terminals: list[SegmentTerminal] = []
    iterator_ops: list[str] = []

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
            # 无序列消费者 → 段终止
            break
        elif len(seq_consumers) == 1:
            # ── 单消费者：Phase 1 逻辑 ──
            next_name, next_section, next_arg = seq_consumers[0]
            next_op = instances[next_name]

            # 检查下一个 Op 的所有序列输入是否都来自段内
            if not _all_seq_inputs_from_segment(
                    next_name, dag, instances, segment_nodes, reduce_extra_inputs):
                if _is_decomposable_reduce(next_op):
                    reduce_op = next_name
                    _collect_reduce_extra_inputs(
                        next_name, dag, instances, segment_nodes, reduce_extra_inputs)
                    terminals = [SegmentTerminal(
                        node_name=next_name,
                        terminal_type=TerminalType.DECOMPOSABLE_REDUCE,
                        extra_inputs=dict(reduce_extra_inputs),
                    )]
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
                terminals = [SegmentTerminal(
                    node_name=next_name,
                    terminal_type=TerminalType.DECOMPOSABLE_REDUCE,
                    extra_inputs=dict(reduce_extra_inputs),
                )]
                break
            elif _is_disk_buffer(next_op):
                # 单消费者为 DiskBuffer → 构建单终端段
                terminals = [SegmentTerminal(
                    node_name=next_name,
                    terminal_type=TerminalType.DISK_BUFFER,
                )]
                # 扫描 DiskBuffer 的下游 iterator ops
                _scan_iterator_ops(
                    next_name, forward, instances, iterator_ops)
                break
            elif _is_io_eligible(next_op) and next_name not in used_nodes:
                io_ops.append(next_name)
                segment_nodes.add(next_name)
                current = next_name
            else:
                break
        else:
            # ── 分支点：多消费者 → Phase 2 多终端段 ──
            branch_terminals = []
            all_reduce_extra: dict[str, str] = {}

            for consumer_name, section, arg in seq_consumers:
                consumer_op = instances[consumer_name]
                if _is_decomposable_reduce(consumer_op):
                    extra = {}
                    _collect_reduce_extra_inputs(
                        consumer_name, dag, instances, segment_nodes, extra)
                    branch_terminals.append(SegmentTerminal(
                        node_name=consumer_name,
                        terminal_type=TerminalType.DECOMPOSABLE_REDUCE,
                        extra_inputs=extra,
                    ))
                    all_reduce_extra.update(extra)
                elif _is_disk_buffer(consumer_op):
                    branch_terminals.append(SegmentTerminal(
                        node_name=consumer_name,
                        terminal_type=TerminalType.DISK_BUFFER,
                    ))
                else:
                    # 该分支不可段化 → 整个分支点无法处理
                    branch_terminals = None
                    break

            if branch_terminals is not None and len(branch_terminals) > 0:
                terminals = branch_terminals
                reduce_extra_inputs = all_reduce_extra
                # 为兼容 Phase 1 字段：取第一个 Reduce 终端
                for t in terminals:
                    if t.terminal_type == TerminalType.DECOMPOSABLE_REDUCE:
                        reduce_op = t.node_name
                        break
                # 扫描所有 DiskBuffer 终端的下游 iterator ops
                for t in terminals:
                    if t.terminal_type == TerminalType.DISK_BUFFER:
                        _scan_iterator_ops(
                            t.node_name, forward, instances, iterator_ops)
                break
            else:
                break  # 回退：无法段化

    # 验证：至少有一个有意义的 ops 组合
    if not map_ops and not reduce_op and not terminals:
        return None

    # 构建 op_class_map + configs_spec
    op_class_map: dict[str, str] = {}
    all_configs_spec: dict[str, dict] = {}
    all_segment_ops = io_ops + map_ops
    for t in terminals:
        all_segment_ops.append(t.node_name)
    for it_op in iterator_ops:
        all_segment_ops.append(it_op)

    for n in all_segment_ops:
        op_class_map[n] = dag.nodes[n]["op"]
        all_configs_spec[n] = dict(instances[n].CONFIGS)

    frame_dist = FrameDistribution.ROUND_ROBIN
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
        terminals=terminals,
        iterator_ops=iterator_ops,
    )

    logger.info(
        f"Detected parallel segment: "
        f"io={io_ops}, map={map_ops}, "
        f"terminals={[(t.node_name, t.terminal_type.value) for t in terminals]}, "
        f"iterator_ops={iterator_ops}")
    return segment


# ────────────────────────────────────────────────────────────────
# 段检测辅助函数
# ────────────────────────────────────────────────────────────────


def _scan_iterator_ops(
    buffer_node: str,
    forward: dict[str, list[tuple[str, str, str]]],
    instances: dict[str, BaseOp],
    iterator_ops: list[str],
) -> None:
    """扫描 DiskBuffer 终端的下游消费者，找到 BUFFER_ITERATOR ops。"""
    for consumer_name, section, arg in forward.get(buffer_node, []):
        consumer_op = instances[consumer_name]
        if _is_buffer_iterator(consumer_op):
            if consumer_name not in iterator_ops:
                iterator_ops.append(consumer_name)


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
        if src == "__inactive__":
            continue
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
        if src == "__inactive__":
            continue
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
    for node_name in segment_ops:
        node_spec = dag.nodes.get(node_name)
        if node_spec is None:
            continue
        for loc, src in _iter_node_src_links(node_spec):
            if src == "__inactive__":
                continue
            parsed = _parse_link(src)
            if parsed[0] == "inputs":
                section, arg_name = loc.split(".", 1)
                feeds.setdefault(parsed[1], []).append(
                    f"{node_name}.{section}.{arg_name}")
    return feeds
