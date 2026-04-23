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

import psutil

import numpy as np

from ..component.data_container import FloatImage
from ..component.frame_buffer import (
    DiskFrameBuffer,
    SourceReplayBuffer, SourceReplayDescriptor,
)
from ..component.ipc_queue import IPCQueue, _TAG_SENTINEL
from ..component.progress import DummyTracker, TrackerEventConsumer
from ..component.queue import BaseQueue, StreamExhausted
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
from .wiring import DEFAULT_DAG_SEARCH_PATHS


def _process_rss_mb() -> float:
    """当前进程的 RSS (MB)，跨平台兼容（Windows/macOS/Linux）。"""
    try:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return -1

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

        # configs: 合并段内所有 ops 的 configs
        # 排除"内部配置"——即从段内其他节点输出获取值的配置，
        # 这些值将由 adapter 在 Phase 1 merge 后内部解析。
        self.config = {}
        all_config_ops = (
            segment.io_ops + segment.map_ops +
            [t.node_name for t in segment.terminals] +
            segment.iterator_ops
        )

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
        # 收集所有 Reduce 终端 + 所有 iterator ops 的输出
        self.outputs = {}
        self.OUTPUTS = {}

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
        logger.trace(f"[MEM] SegmentAdapter._async_execute start: RSS={_process_rss_mb():.0f} MB")

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
            "op_classes": segment.op_class_map,
            "terminals": self._build_terminal_infos(segment),
            "iterator_ops": segment.iterator_ops,
        }

        mp_cancel = mp.Event()
        tracker_queue = mp.Queue()
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

        consumer = TrackerEventConsumer(self.tracker, tracker_queue)
        consumer_task = asyncio.create_task(consumer.run())

        self._worker_processes = processes

        try:
            await asyncio.gather(
                self._dispatch(input_ipcs, mp_cancel),
                self._collect(input_ipcs, output_ipcs, mp_cancel, node_configs),
            )
        except Exception as e:
            logger.exception(f"SegmentAdapter failed: {e}")
            mp_cancel.set()
            raise
        finally:
            logger.debug("SegmentAdapter finally: setting done_events")
            # 仅在异常路径强制解除 workers 阻塞：
            # 正常路径 workers 已通过 "finish" 正常退出，无需注入。
            # 异常路径 workers 可能阻塞在 input_ipc.get()，注入 SENTINEL 解除。
            if mp_cancel.is_set():
                for ipc in input_ipcs:
                    try:
                        ipc._conn_a.send((_TAG_SENTINEL, None))
                        ipc._filled_sem.release()
                    except Exception:
                        pass
            for evt in done_events:
                evt.set()
            consumer.stop()
            await consumer_task
            # 关闭 tracker_queue 的后台 feeder 线程，防止进程退出时挂起
            try:
                tracker_queue.close()
                tracker_queue.cancel_join_thread()
            except Exception:
                pass
            for i, p in enumerate(processes):
                logger.debug(f"Joining worker {i} (pid={p.pid})...")
                await asyncio.to_thread(p.join, 10)
                if p.is_alive():
                    logger.warning(
                        f"Segment worker pid={p.pid} did not exit, terminating.")
                    p.terminate()
                else:
                    logger.debug(f"Worker {i} (pid={p.pid}) exited normally")
            for ipc in input_ipcs + output_ipcs:
                ipc.cleanup()

    def _build_terminal_infos(self, segment: ParallelSegment) -> list[dict]:
        """构建可 pickle 的终端信息列表，附加 DiskBuffer 元数据。"""
        result = []
        for t in segment.terminals:
            t_info = {
                "node_name": t.node_name,
                "terminal_type": t.terminal_type.value,
                "extra_inputs": t.extra_inputs,
            }
            if t.terminal_type == TerminalType.DISK_BUFFER:
                inst = self._original_instances.get(t.node_name)
                if inst is not None:
                    fnames_q = inst.inputs.get("fnames")
                    t_info["has_fnames"] = (
                        fnames_q is not None
                        and getattr(fnames_q, "active", False)
                    )
            result.append(t_info)
        return result

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

        # 有 iterator_ops 时由 _collect_multi_terminal 发 finish 命令
        if not self.segment.iterator_ops:
            logger.debug("_dispatch: sending SENTINEL to workers (no Phase 2)")
            for ipc in input_ipcs:
                await ipc.put(BaseQueue._SENTINEL)
        else:
            logger.debug("_dispatch: NOT sending SENTINEL (Phase 2 will send finish)")

        logger.debug(f"_dispatch completed (total_frames={total_frames})")

    async def _collect(
        self,
        input_ipcs: list[IPCQueue],
        output_ipcs: list[IPCQueue],
        cancel_event: mp.Event,
        node_configs: dict[str, dict[str, Any]],
    ) -> None:
        """收集 worker 输出 + 多阶段迭代式 Reduce 编排。"""
        await self._collect_multi_terminal(
            input_ipcs, output_ipcs, cancel_event, node_configs)

    async def _collect_multi_terminal(
        self,
        input_ipcs: list[IPCQueue],
        output_ipcs: list[IPCQueue],
        cancel_event: mp.Event,
        node_configs: dict[str, dict[str, Any]],
    ) -> None:
        """多终端段的 Phase 1 收集 + Phase 2 迭代编排。"""
        segment = self.segment

        # ── Phase 1: 收集 N 个 worker 的 partial ──
        # 新协议：worker 先发送大对象（FGP/FloatImage 走 ShmTransportable），
        # 最后发送 manifest dict 描述结构。
        logger.trace(f"[MEM] Phase1 collect start: RSS={_process_rss_mb():.0f} MB")
        partials: list[dict[str, Any]] = []
        for i, ipc in enumerate(output_ipcs):
            try:
                # 接收所有 item 直到 manifest（最后一个 dict 类型的 item）
                received_large = []
                while True:
                    item = await ipc.get()
                    if isinstance(item, dict):
                        # 这是 manifest
                        manifest = item
                        break
                    else:
                        received_large.append(item)

                # 从 manifest 和 large_order 重组 partial dict
                large_order = manifest.pop("__large_order", [])
                partial = {}
                large_idx = 0
                for t_name, key in large_order:
                    partial.setdefault(t_name, {})[key] = received_large[large_idx]
                    large_idx += 1
                # 合并 manifest 中的小对象和 __disk_buffer
                for mk, mv in manifest.items():
                    if mk.startswith("__keys_"):
                        continue
                    if mk == "__disk_buffer":
                        partial["__disk_buffer"] = mv
                    elif isinstance(mv, dict):
                        partial.setdefault(mk, {}).update(mv)
                    else:
                        partial[mk] = mv

                logger.trace(f"[MEM] Phase1 got worker {i} partial: RSS={_process_rss_mb():.0f} MB "
                            f"(keys={list(partial.keys())})")
                partials.append(partial)
            except StreamExhausted:
                logger.warning(f"Worker {i} sent no Phase 1 partial")

        if not partials:
            raise ValueError("No Phase 1 partial results from workers")

        # ── 处理每个终端 ──
        # 收集 Phase 1 merge 结果，用于解析内部配置
        phase1_merged: dict[str, dict[str, Any]] = {}

        for t in segment.terminals:
            try:
                if t.terminal_type == TerminalType.DECOMPOSABLE_REDUCE:
                    # 收集并 merge Reduce partials
                    reduce_partials = [p.get(t.node_name, {}) for p in partials]
                    reduce_cls_name = segment.op_class_map[t.node_name]
                    reduce_cls = REGISTERED_OP[reduce_cls_name]
                    logger.debug(
                        f"Phase1 merging terminal '{t.node_name}' "
                        f"({reduce_cls_name}), "
                        f"partial keys: {[list(rp.keys()) for rp in reduce_partials]}")
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
            except Exception:
                logger.exception(
                    f"Phase1 merge FAILED for terminal '{t.node_name}' "
                    f"(type={t.terminal_type})")
                raise

        logger.trace(f"[MEM] Phase1 merge done: RSS={_process_rss_mb():.0f} MB")

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

        # 释放 Phase 1 收集的大对象：Reduce partial 已 merge 并推送下游，
        # 仅保留 __disk_buffer 描述符（median fallback 需要）。
        for p in partials:
            for k in list(p.keys()):
                if k != "__disk_buffer":
                    del p[k]
        phase1_merged.clear()

        logger.trace(f"[MEM] Phase2 start (internal configs resolved): RSS={_process_rss_mb():.0f} MB")

        # ── Phase 2: 分布式迭代 Reduce ──
        if not segment.iterator_ops:
            # 无迭代 → 通知 worker 退出
            logger.debug("No iterator_ops, sending finish to all workers")
            for ipc in input_ipcs:
                await ipc.put({"action": "finish"})
            # 排空 workers 发送的最终 SENTINEL，防止 pipe 残留阻塞进程退出
            for ipc in output_ipcs:
                try:
                    await ipc.get()
                except StreamExhausted:
                    pass
            return

        for iter_op_name in segment.iterator_ops:
            iter_op_cls_name = segment.op_class_map[iter_op_name]
            iter_op_cls = REGISTERED_OP[iter_op_cls_name]
            iter_type = getattr(iter_op_cls, "ITERATOR_TYPE", "unknown")
            iter_configs = node_configs.get(iter_op_name, {})

            if iter_type == "sigma_clip":
                await self._run_sigma_clip_iterations(
                    input_ipcs, output_ipcs, iter_op_name, iter_configs)

            elif iter_type == "huber_mean":
                await self._run_huber_iteration(
                    input_ipcs, output_ipcs, iter_op_name, iter_configs)

            elif iter_type == "median":
                await self._run_median_fallback(
                    iter_op_name, iter_configs, partials)

            else:
                logger.warning(
                    f"Unknown iterator type '{iter_type}' for {iter_op_name}")

        # 通知 workers 退出
        logger.debug("All iterators done, sending finish to all workers")
        for wi, ipc in enumerate(input_ipcs):
            logger.debug(f"Sending finish to worker {wi}")
            await ipc.put({"action": "finish"})
            logger.debug(f"Finish sent to worker {wi}")
        # 排空 workers 发送的最终 SENTINEL，防止 pipe 残留阻塞进程退出
        for ipc in output_ipcs:
            try:
                await ipc.get()
            except StreamExhausted:
                pass

    async def _run_sigma_clip_iterations(
        self,
        input_ipcs: list[IPCQueue],
        output_ipcs: list[IPCQueue],
        iter_op_name: str,
        iter_configs: dict[str, Any],
    ) -> None:
        """分布式 Sigma Clip 迭代。"""
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

        # 计算 FGP 大小用于诊断
        _fgp_mb = (fgp_total.sum_mu.nbytes + fgp_total.square_sum.nbytes
                   + fgp_total.n.nbytes) / 1024 / 1024
        logger.trace(f"[MEM] sigma_clip start: RSS={_process_rss_mb():.0f} MB, "
                    f"fgp_total={_fgp_mb:.0f} MB "
                    f"(sum_mu={fgp_total.sum_mu.dtype}, "
                    f"sq={fgp_total.square_sum.dtype}, n={fgp_total.n.dtype})")

        for iteration in range(max_iter):
            logger.trace(f"[MEM] iter {iteration+1} broadcast start: RSS={_process_rss_mb():.0f} MB")
            # broadcast ref_fgp 给所有 worker
            # FGP 作为独立 item 发送（走 ShmTransportable 协议），
            # 避免嵌套在 dict 中导致整个 dict pickle 序列化时的内存放大
            for wi, ipc in enumerate(input_ipcs):
                await ipc.put(ref_fgp)  # FGP → ShmTransportable → SharedMemory
                await ipc.put({
                    "action": "iterate",
                    "iter_type": "sigma_clip",
                    "params": {"rej_high": rej_high, "rej_low": rej_low},
                })
            logger.trace(f"[MEM] iter {iteration+1} broadcast done: RSS={_process_rss_mb():.0f} MB")

            # 收集 N 个 partial（workers 直接发送 FGP，非 dict 包装）
            clip_partials = []
            for i, ipc in enumerate(output_ipcs):
                try:
                    partial = await ipc.get()
                    clip_partials.append(partial)
                    logger.trace(f"[MEM] iter {iteration+1} got worker {i} clip partial: "
                                f"RSS={_process_rss_mb():.0f} MB")
                except StreamExhausted:
                    logger.warning(f"Worker {i} sent no clip partial")

            if not clip_partials:
                raise ValueError("No clip partials from workers")

            # merge rejected FGPs（直接接收 FGP 对象，非 dict）
            total_rejected = clip_partials[0]
            for p in clip_partials[1:]:
                total_rejected = total_rejected + p
            del clip_partials
            logger.trace(f"[MEM] iter {iteration+1} merge done: RSS={_process_rss_mb():.0f} MB")

            # accepted = fgp_total - total_rejected
            accepted = fgp_total - total_rejected
            del total_rejected
            accepted.apply_zero_var(fgp_total)
            logger.trace(f"[MEM] iter {iteration+1} sub+zerovar done: RSS={_process_rss_mb():.0f} MB")

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
    ) -> None:
        """分布式 Huber Mean 单 pass。"""

        fgp_total = iter_configs.get("fgp_total")
        if fgp_total is None:
            raise ValueError(
                f"{iter_op_name}: fgp_total config not available")

        huber_c = iter_configs.get("huber_c", 1.345)

        # broadcast 给所有 worker
        # FGP 作为独立 item 发送，避免 dict pickle 内存放大
        for ipc in input_ipcs:
            await ipc.put(fgp_total)  # FGP → ShmTransportable → SharedMemory
            await ipc.put({
                "action": "iterate",
                "iter_type": "huber_mean",
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

        # merge（直接接收 HuberMeanParam 对象，非 dict）
        final_param = huber_partials[0]
        for p in huber_partials[1:]:
            final_param = final_param + p
        del huber_partials
        result = FloatImage(final_param.mu, dtype=final_param.source_dtype)
        del final_param

        output_key = f"{iter_op_name}.result"
        if output_key in self.outputs:
            for q in self.outputs[output_key]:
                await q.put(result)

    async def _run_median_fallback(
        self,
        iter_op_name: str,
        iter_configs: dict[str, Any],
        phase1_partials: list[dict],
    ) -> None:
        """Median 不可分布式归约——从 worker 的描述符重建 buffer，在主进程执行。"""

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
        all_buffers = []
        for d in all_descriptors:
            if isinstance(d, SourceReplayDescriptor):
                all_buffers.append(SourceReplayBuffer.from_descriptor(d))
            else:
                all_buffers.append(DiskFrameBuffer.from_descriptor(d))
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

    def _infer_output_length(self, input_lengths):
        """段的输出长度推断。"""
        if self.segment.terminals:
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
        all_seg_ops = (segment.io_ops + segment.map_ops +
                       [t.node_name for t in segment.terminals] +
                       segment.iterator_ops)

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
