"""
SegmentAdapter：将完整段（I/O + Map + 多终端 + 迭代式 Reduce）包装为单个 BaseOp。

在 DAG 执行中替代整个段内所有 ops，对 DAGExecutor 表现为一个普通的 BaseOp。
内部启动 N 个 worker 进程完成数据并行。

Phase 2 扩展：
    - 多终端段（DECOMPOSABLE_REDUCE + DISK_BUFFER）
    - 多阶段 Worker：Phase 1 流式处理 + Phase 2+ 迭代式 Reduce
    - Iterator Op（SigmaClip / Huber）在 Worker 内执行，主进程只做 merge + 收敛判断
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
from typing import Any, Optional

from loguru import logger

from ..component.ipc_queue import IPCQueue
from ..component.progress import DummyTracker
from ..component.queue import BaseQueue, RichContextQueue, StreamExhausted
from ..ops.base import BaseOp
from .build import ValidatedDag
from .registry import REGISTERED_OP
from .segment_detect import (
    ParallelSegment,
    TerminalType,
    FrameDistribution,
    _compute_block_ranges,
    detect_parallel_segments,
)
from .segment_worker import _segment_worker_main

__all__ = [
    "SegmentAdapter",
    "apply_data_parallelism",
    "_auto_worker_count",
]


# ────────────────────────────────────────────────────────────────
# SegmentAdapter: 在 DAG 中替代整个段
# ────────────────────────────────────────────────────────────────


class SegmentAdapter(BaseOp):
    """将 I/O + Map 链 + 多终端 + 迭代式 Reduce 包装为数据并行执行单元。

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
        self.name = name
        self.segment = segment
        self._num_workers = num_workers
        self._dag = dag
        self._original_instances = original_instances
        self.tracker = DummyTracker()
        self._cancel_event = None

        self.INPUTS = {}
        self.OUTPUTS = {}
        self.CONFIGS = {}

        # inputs: 段头 I/O ops 的输入
        first_io = segment.io_ops[0]
        first_io_inst = original_instances[first_io]
        self.inputs = dict(first_io_inst.inputs)
        self.INPUTS = dict(first_io_inst.INPUTS)

        # 加入所有终端的 reduce 额外输入
        for t in segment.terminals:
            if t.terminal_type == TerminalType.DECOMPOSABLE_REDUCE and t.extra_inputs:
                inst = original_instances[t.node_name]
                for extra_key, src in t.extra_inputs.items():
                    input_key = f"__reduce_extra_{extra_key}"
                    if input_key not in self.inputs:
                        self.inputs[input_key] = inst.inputs[extra_key]
                        self.INPUTS[input_key] = inst.INPUTS[extra_key]

        # Phase 1 兼容：单终端无 terminals 字段
        if not segment.terminals and segment.reduce_op and segment.reduce_extra_inputs:
            reduce_inst = original_instances[segment.reduce_op]
            for extra_key, src in segment.reduce_extra_inputs.items():
                input_key = f"__reduce_extra_{extra_key}"
                if input_key not in self.inputs:
                    self.inputs[input_key] = reduce_inst.inputs[extra_key]
                    self.INPUTS[input_key] = reduce_inst.INPUTS[extra_key]

        # configs: 合并段内所有 ops 的 configs
        # 排除"内部配置"——即从段内其他节点输出获取值的配置，
        # 这些值将由 adapter 在 Phase 1 merge 后内部解析。
        self.config = {}
        all_config_ops = (
            segment.io_ops + segment.map_ops +
            [t.node_name for t in segment.terminals] +
            segment.iterator_ops
        )
        # Phase 1 兼容
        if not segment.terminals and segment.reduce_op:
            all_config_ops.append(segment.reduce_op)

        # 构建段内节点集合，用于识别内部配置
        segment_node_set = set(all_config_ops)
        # 记录内部配置的来源映射：{node.key → (src_node, src_output)}
        self._internal_configs: dict[str, tuple[str, str]] = {}

        for node_name in all_config_ops:
            inst = original_instances.get(node_name)
            if inst is None:
                continue
            # 检查该节点的每个 config 是否来自段内其他节点
            node_spec = dag.nodes.get(node_name, {})
            config_specs = node_spec.get("configs", {})

            for key, queue in inst.config.items():
                compound_key = f"{node_name}.{key}"
                # 检查 config 的 src link
                cfg_binding = config_specs.get(key, "")
                if isinstance(cfg_binding, dict):
                    cfg_src = cfg_binding.get("src", "")
                elif isinstance(cfg_binding, str):
                    cfg_src = cfg_binding
                else:
                    cfg_src = ""

                # 检查 src 是否指向段内节点的输出
                is_internal = False
                if isinstance(cfg_src, str) and "." in cfg_src:
                    src_parts = cfg_src.rsplit(".", 1)
                    if len(src_parts) == 2 and src_parts[0] in segment_node_set:
                        is_internal = True
                        self._internal_configs[compound_key] = (
                            src_parts[0], src_parts[1])

                if not is_internal:
                    self.config[compound_key] = queue
                    self.CONFIGS[compound_key] = inst.CONFIGS.get(key, {})

        # outputs: 段尾的输出
        # 多终端时：收集所有 Reduce 终端 + 所有 iterator ops 的输出
        self.outputs = {}
        self.OUTPUTS = {}

        if segment.terminals:
            for t in segment.terminals:
                if t.terminal_type == TerminalType.DECOMPOSABLE_REDUCE:
                    inst = original_instances[t.node_name]
                    for key, queues in inst.outputs.items():
                        self.outputs[f"{t.node_name}.{key}"] = queues
                        self.OUTPUTS[f"{t.node_name}.{key}"] = inst.OUTPUTS.get(key, {})
            for it_name in segment.iterator_ops:
                inst = original_instances[it_name]
                for key, queues in inst.outputs.items():
                    self.outputs[f"{it_name}.{key}"] = queues
                    self.OUTPUTS[f"{it_name}.{key}"] = inst.OUTPUTS.get(key, {})
        elif segment.reduce_op:
            tail = original_instances[segment.reduce_op]
            self.outputs = dict(tail.outputs)
            self.OUTPUTS = dict(tail.OUTPUTS)
        elif segment.map_ops:
            tail = original_instances[segment.map_ops[-1]]
            self.outputs = dict(tail.outputs)
            self.OUTPUTS = dict(tail.OUTPUTS)
        else:
            tail = original_instances[segment.io_ops[-1]]
            self.outputs = dict(tail.outputs)
            self.OUTPUTS = dict(tail.OUTPUTS)

        # 清除内部配置队列：这些队列属于段内节点间的 config 连接，
        # 由 adapter 内部解析，不应出现在 self.outputs 中。
        # 否则 _send_sentinel 会向已满且无人消费的队列写 SENTINEL 而死锁。
        for compound_key, (src_node, src_output) in self._internal_configs.items():
            parts = compound_key.rsplit(".", 1)
            if len(parts) != 2:
                continue
            target_node, config_key = parts
            target_inst = original_instances.get(target_node)
            if target_inst is None:
                continue
            config_queue = target_inst.config.get(config_key)
            if config_queue is None:
                continue
            output_key = f"{src_node}.{src_output}"
            if output_key in self.outputs:
                queues = self.outputs[output_key]
                if config_queue in queues:
                    queues.remove(config_queue)
                    logger.debug(
                        f"Removed internal config queue {compound_key} "
                        f"from output {output_key}")

        self.length = None

    async def _async_execute(self, configs: dict[str, Any]) -> None:
        """执行数据并行段。"""
        segment = self.segment
        num_workers = self._num_workers

        # 重组 configs: {node_name: {config_key: value}}
        node_configs: dict[str, dict[str, Any]] = {}
        for compound_key, value in configs.items():
            parts = compound_key.rsplit(".", 1)
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
            "terminals": [
                {
                    "node_name": t.node_name,
                    "terminal_type": t.terminal_type.value,
                    "extra_inputs": t.extra_inputs,
                }
                for t in segment.terminals
            ],
            "iterator_ops": segment.iterator_ops,
        }

        mp_cancel = mp.Event()
        tracker_queue = mp.Queue()
        from .wiring import DEFAULT_DAG_SEARCH_PATHS
        search_paths_str = [str(p) for p in DEFAULT_DAG_SEARCH_PATHS]

        # 启动 worker 进程
        processes: list[mp.Process] = []
        ready_events: list[mp.Event] = []
        done_events: list[mp.Event] = []
        for i in range(num_workers):
            ready_evt = mp.Event()
            done_evt = mp.Event()
            ready_events.append(ready_evt)
            done_events.append(done_evt)
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
                    ready_evt,
                    done_evt,
                ),
                daemon=True,
            )
            processes.append(p)
            p.start()
            logger.info(f"Segment worker {i} started (pid={p.pid})")

        # 等待所有 worker import 完成
        for i, (evt, p) in enumerate(zip(ready_events, processes)):
            ready = await asyncio.to_thread(evt.wait, 30)
            if not ready or not p.is_alive():
                msg = (f"Segment worker {i} failed to start "
                       f"(alive={p.is_alive()}, ready={ready})")
                logger.error(msg)
                mp_cancel.set()
                raise RuntimeError(msg)
        logger.info(f"All {num_workers} workers ready")

        from ..component.progress import TrackerEventConsumer
        consumer = TrackerEventConsumer(self.tracker, tracker_queue)
        consumer_task = asyncio.create_task(consumer.run())

        self._worker_processes = processes

        try:
            await asyncio.gather(
                self._dispatch(input_ipcs, mp_cancel),
                self._collect(input_ipcs, output_ipcs, mp_cancel, node_configs),
            )
        except Exception as e:
            logger.error(f"SegmentAdapter failed: {e}")
            mp_cancel.set()
            raise
        finally:
            for evt in done_events:
                evt.set()
            consumer.stop()
            await consumer_task
            for p in processes:
                await asyncio.to_thread(p.join, 10)
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

        # 获取主输入队列
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
            logger.warning("SegmentAdapter: unknown frame count")
            return

        # 计算每个 worker 的帧数
        worker_counts = [0] * num_workers
        if segment.frame_distribution == FrameDistribution.ROUND_ROBIN:
            for i in range(total_frames):
                worker_counts[i % num_workers] += 1
        else:
            ranges = _compute_block_ranges(total_frames, num_workers)
            worker_counts = [count for _, count in ranges]

        for i in range(num_workers):
            await input_ipcs[i].set_length(worker_counts[i])

        # 收集所有额外输入队列
        extra_queues = {}
        all_extra_keys: set[str] = set()
        for t in segment.terminals:
            if t.terminal_type == TerminalType.DECOMPOSABLE_REDUCE:
                all_extra_keys.update(t.extra_inputs.keys())
        # Phase 1 兼容
        all_extra_keys.update(segment.reduce_extra_inputs.keys())

        main_queue = self.inputs[main_input_key]
        for extra_key in all_extra_keys:
            input_key = f"__reduce_extra_{extra_key}"
            if input_key in self.inputs:
                extra_queues[extra_key] = self.inputs[input_key]

        for i in range(total_frames):
            if cancel_event.is_set():
                break

            if hasattr(self, '_worker_processes'):
                dead = [j for j, p in enumerate(self._worker_processes)
                        if not p.is_alive()]
                if dead:
                    raise RuntimeError(
                        f"Worker(s) {dead} died during dispatch "
                        f"(frame {i}/{total_frames})")

            frame_data = {main_input_key: await main_queue.get()}
            for extra_key, extra_queue in extra_queues.items():
                frame_data[f"__reduce_extra_{extra_key}"] = await extra_queue.get()

            worker_idx = i % num_workers
            await input_ipcs[worker_idx].put(frame_data)

        # 仅在无多阶段(Phase 2+)时发送 SENTINEL 给 workers
        # 多终端段中有 iterator_ops 时，由 _collect_multi_terminal 发 finish 命令
        if not (self.segment.terminals and self.segment.iterator_ops):
            for ipc in input_ipcs:
                await ipc.put(BaseQueue._SENTINEL)

    async def _collect(
        self,
        input_ipcs: list[IPCQueue],
        output_ipcs: list[IPCQueue],
        cancel_event: mp.Event,
        node_configs: dict[str, dict[str, Any]],
    ) -> None:
        """收集 worker 输出 + 多阶段迭代式 Reduce 编排。"""
        segment = self.segment
        num_workers = len(output_ipcs)
        has_multi_terminal = len(segment.terminals) > 0
        has_iterators = len(segment.iterator_ops) > 0

        if has_multi_terminal:
            await self._collect_multi_terminal(
                input_ipcs, output_ipcs, cancel_event, node_configs)
        elif segment.reduce_op:
            # Phase 1 兼容：单 Reduce
            partials: list[dict[str, Any]] = []
            for i, ipc in enumerate(output_ipcs):
                try:
                    partial = await ipc.get()
                    partials.append(partial)
                except StreamExhausted:
                    logger.warning(f"Worker {i} sent no partial result")

            if not partials:
                raise ValueError("No partial results from workers")

            reduce_cls_name = segment.op_class_map[segment.reduce_op]
            reduce_cls = REGISTERED_OP[reduce_cls_name]
            final = reduce_cls.merge_partial(partials)
            await self._broadcast_outputs(final)
        else:
            await self._collect_ordered(output_ipcs)

    async def _collect_multi_terminal(
        self,
        input_ipcs: list[IPCQueue],
        output_ipcs: list[IPCQueue],
        cancel_event: mp.Event,
        node_configs: dict[str, dict[str, Any]],
    ) -> None:
        """多终端段的 Phase 1 收集 + Phase 2 迭代编排。"""
        import numpy as np
        from ..component.tagged_image import FloatImage
        from ..component.merger import SigmaClippingMerger, HuberWeightedMerger

        segment = self.segment
        num_workers = len(output_ipcs)

        # ── Phase 1: 收集 N 个 worker 的 partial ──
        partials: list[dict[str, Any]] = []
        for i, ipc in enumerate(output_ipcs):
            try:
                partial = await ipc.get()
                partials.append(partial)
            except StreamExhausted:
                logger.warning(f"Worker {i} sent no Phase 1 partial")

        if not partials:
            raise ValueError("No Phase 1 partial results from workers")

        # ── 处理每个终端 ──
        # 收集 Phase 1 merge 结果，用于解析内部配置
        phase1_merged: dict[str, dict[str, Any]] = {}

        for t in segment.terminals:
            if t.terminal_type == TerminalType.DECOMPOSABLE_REDUCE:
                # 收集并 merge Reduce partials
                reduce_partials = [p.get(t.node_name, {}) for p in partials]
                reduce_cls_name = segment.op_class_map[t.node_name]
                reduce_cls = REGISTERED_OP[reduce_cls_name]
                final = reduce_cls.merge_partial(reduce_partials)
                phase1_merged[t.node_name] = final
                # 推送到该 Reduce 终端的下游队列（仅推送非段内消费的输出）
                for key, value in final.items():
                    output_key = f"{t.node_name}.{key}"
                    if output_key in self.outputs:
                        for q in self.outputs[output_key]:
                            await q.put(value)

            elif t.terminal_type == TerminalType.DISK_BUFFER:
                # buffer 描述符暂存在 worker 本地，不推送下游
                pass

        # ── 解析内部配置（段内节点输出 → 迭代 Op 的 config）──
        for compound_key, (src_node, src_output) in self._internal_configs.items():
            parts = compound_key.rsplit(".", 1)
            if len(parts) != 2:
                continue
            target_node, config_key = parts
            if src_node in phase1_merged and src_output in phase1_merged[src_node]:
                node_configs.setdefault(target_node, {})[config_key] = (
                    phase1_merged[src_node][src_output])
                logger.debug(
                    f"Resolved internal config {compound_key} "
                    f"← {src_node}.{src_output}")

        # ── Phase 2: 分布式迭代 Reduce ──
        if not segment.iterator_ops:
            # 无迭代 → 通知 worker 退出
            for ipc in input_ipcs:
                await ipc.put({"action": "finish"})
            return

        for iter_op_name in segment.iterator_ops:
            iter_op_cls_name = segment.op_class_map[iter_op_name]
            iter_op_cls = REGISTERED_OP[iter_op_cls_name]
            iter_type = getattr(iter_op_cls, "ITERATOR_TYPE", "unknown")
            iter_configs = node_configs.get(iter_op_name, {})

            if iter_type == "sigma_clip":
                await self._run_sigma_clip_iterations(
                    input_ipcs, output_ipcs, iter_op_name, iter_configs,
                    segment, partials)

            elif iter_type == "huber_mean":
                await self._run_huber_iteration(
                    input_ipcs, output_ipcs, iter_op_name, iter_configs,
                    segment, partials)

            elif iter_type == "median":
                # 中位数不可分布式 → 回退到主进程
                await self._run_median_fallback(
                    iter_op_name, iter_configs, segment, partials)

            else:
                logger.warning(
                    f"Unknown iterator type '{iter_type}' for {iter_op_name}")

        # 通知 workers 退出
        for ipc in input_ipcs:
            await ipc.put({"action": "finish"})

    async def _run_sigma_clip_iterations(
        self,
        input_ipcs: list[IPCQueue],
        output_ipcs: list[IPCQueue],
        iter_op_name: str,
        iter_configs: dict[str, Any],
        segment: ParallelSegment,
        phase1_partials: list[dict],
    ) -> None:
        """分布式 Sigma Clip 迭代。"""
        import numpy as np
        from ..component.tagged_image import FloatImage
        from ..component.merger import SigmaClippingMerger
        from ..component.utils import FastGaussianParam

        num_workers = len(output_ipcs)

        # 获取全局 FGP（从 Phase 1 的 MeanStacker 终端输出）
        # sigma_clip_iter 的 fgp_total config 来自 mean_stacker.statistics
        fgp_total = iter_configs.get("fgp_total")
        if fgp_total is None:
            raise ValueError(
                f"{iter_op_name}: fgp_total config not available")

        rej_high = iter_configs.get("rej_high", 3.0)
        rej_low = iter_configs.get("rej_low", 3.0)
        max_iter = iter_configs.get("max_iter", 5)
        early_converge_ratio = iter_configs.get("early_converge_ratio", 0.99)

        fgp_total.inplace_calc = False
        ref_fgp = fgp_total
        last_n = ref_fgp.n.copy()
        accepted = None

        for iteration in range(max_iter):
            # broadcast ref_fgp 给所有 worker
            for ipc in input_ipcs:
                await ipc.put({
                    "action": "iterate",
                    "iter_type": "sigma_clip",
                    "ref": ref_fgp,
                    "params": {"rej_high": rej_high, "rej_low": rej_low},
                })

            # 收集 N 个 partial
            clip_partials = []
            for i, ipc in enumerate(output_ipcs):
                try:
                    partial = await ipc.get()
                    clip_partials.append(partial)
                except StreamExhausted:
                    logger.warning(f"Worker {i} sent no clip partial")

            if not clip_partials:
                raise ValueError("No clip partials from workers")

            # merge rejected FGPs
            total_rejected = SigmaClippingMerger.merge_partial(clip_partials)

            # accepted = fgp_total - total_rejected
            accepted = fgp_total - total_rejected
            accepted.apply_zero_var(fgp_total)

            # 收敛检查
            cur_n = accepted.n
            converge_ratio = np.sum(cur_n == last_n) / np.prod(cur_n.shape)
            if converge_ratio >= early_converge_ratio:
                logger.info(
                    f"{iter_op_name} converged at iteration {iteration + 1}")
                break
            else:
                logger.info(
                    f"{iter_op_name} converge ratio: {converge_ratio * 100:.2f}%")
            last_n = cur_n.copy()
            ref_fgp = accepted
            logger.info(
                f"{iter_op_name} iteration {iteration + 1}/{max_iter} done")
        else:
            logger.info(
                f"{iter_op_name} reached max iterations ({max_iter})")

        # 推送最终结果
        result = FloatImage(accepted.mu, dtype=accepted.source_dtype)
        accepted.inplace_calc = False

        for key, value in [("result", result), ("statistics", accepted)]:
            output_key = f"{iter_op_name}.{key}"
            if output_key in self.outputs:
                for q in self.outputs[output_key]:
                    await q.put(value)

    async def _run_huber_iteration(
        self,
        input_ipcs: list[IPCQueue],
        output_ipcs: list[IPCQueue],
        iter_op_name: str,
        iter_configs: dict[str, Any],
        segment: ParallelSegment,
        phase1_partials: list[dict],
    ) -> None:
        """分布式 Huber Mean 单 pass。"""
        from ..component.tagged_image import FloatImage
        from ..component.merger import HuberWeightedMerger

        num_workers = len(output_ipcs)

        fgp_total = iter_configs.get("fgp_total")
        if fgp_total is None:
            raise ValueError(
                f"{iter_op_name}: fgp_total config not available")

        huber_c = iter_configs.get("huber_c", 1.345)

        # broadcast 给所有 worker
        for ipc in input_ipcs:
            await ipc.put({
                "action": "iterate",
                "iter_type": "huber_mean",
                "ref": fgp_total,
                "params": {"huber_c": huber_c},
            })

        # 收集 N 个 partial
        huber_partials = []
        for i, ipc in enumerate(output_ipcs):
            try:
                partial = await ipc.get()
                huber_partials.append(partial)
            except StreamExhausted:
                logger.warning(f"Worker {i} sent no Huber partial")

        if not huber_partials:
            raise ValueError("No Huber partials from workers")

        # merge
        final_param = HuberWeightedMerger.merge_partial(huber_partials)
        result = FloatImage(final_param.mu, dtype=final_param.source_dtype)

        output_key = f"{iter_op_name}.result"
        if output_key in self.outputs:
            for q in self.outputs[output_key]:
                await q.put(result)

    async def _run_median_fallback(
        self,
        iter_op_name: str,
        iter_configs: dict[str, Any],
        segment: ParallelSegment,
        phase1_partials: list[dict],
    ) -> None:
        """Median 不可分布式归约——从 worker 的描述符重建 buffer，在主进程执行。"""
        from ..component.frame_buffer import DiskFrameBuffer, DiskBufferDescriptor
        import numpy as np
        from ..component.tagged_image import FloatImage

        # 从所有 worker 的 partial 中提取 buffer 描述符
        all_descriptors = []
        for p in phase1_partials:
            desc = p.get("__disk_buffer")
            if desc is not None:
                all_descriptors.append(desc)

        if not all_descriptors:
            raise ValueError(
                f"{iter_op_name}: no buffer descriptors from workers")

        # 重建 buffer 并拼接路径
        all_buffers = [DiskFrameBuffer.from_descriptor(d) for d in all_descriptors]
        total_frames = sum(len(b) for b in all_buffers)

        chunk_rows = iter_configs.get("chunk_rows", 32)

        # 读取第一帧获取尺寸
        first_frame, _ = all_buffers[0][0]
        h, w = first_frame.shape[:2]
        source_dtype = first_frame.dtype

        result_chunks = []
        n_chunks = (h + chunk_rows - 1) // chunk_rows

        for chunk_idx in range(n_chunks):
            row_start = chunk_idx * chunk_rows
            row_end = min(row_start + chunk_rows, h)
            actual_rows = row_end - row_start

            if first_frame.ndim == 3:
                channels = first_frame.shape[2]
                stack = np.empty(
                    (total_frames, actual_rows, w, channels), dtype=np.float32)
            else:
                stack = np.empty(
                    (total_frames, actual_rows, w), dtype=np.float32)

            frame_idx = 0
            for buf in all_buffers:
                for bi in range(len(buf)):
                    frame_data, _ = buf[bi]
                    stack[frame_idx] = frame_data[row_start:row_end].astype(np.float32)
                    frame_idx += 1

            chunk_median = np.median(stack, axis=0)
            result_chunks.append(chunk_median)

        result_array = np.concatenate(result_chunks, axis=0)
        result = FloatImage(data=result_array, dtype=source_dtype)

        output_key = f"{iter_op_name}.result"
        if output_key in self.outputs:
            for q in self.outputs[output_key]:
                await q.put(result)

        # 清理所有重建的 buffer
        for buf in all_buffers:
            buf.cleanup()

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
        if self.segment.reduce_op or self.segment.terminals:
            return 1
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
    adapters: dict[str, SegmentAdapter] = {}

    for i, segment in enumerate(segments):
        all_seg_ops = segment.io_ops + segment.map_ops
        for t in segment.terminals:
            all_seg_ops.append(t.node_name)
        for it_op in segment.iterator_ops:
            all_seg_ops.append(it_op)
        # Phase 1 兼容
        if not segment.terminals and segment.reduce_op:
            all_seg_ops.append(segment.reduce_op)

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
        adapters[segment.io_ops[0]] = adapter

        logger.info(
            f"Created SegmentAdapter '{adapter_name}' "
            f"({num_workers} workers) replacing ops: {all_seg_ops}")

    # 重建 ops 列表
    new_ops: list[BaseOp] = []
    for op in ops:
        if op.name in segment_nodes:
            if op.name in adapters:
                new_ops.append(adapters[op.name])
        else:
            new_ops.append(op)

    logger.info(
        f"Data parallelism applied: {len(segments)} segment(s), "
        f"{len(new_ops)} ops in final DAG")
    return new_ops
