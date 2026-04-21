"""
Worker 进程入口：在独立进程中执行段内 I/O + Map + Reduce/DiskBuffer 流水线。

支持多终端 + 多阶段协议：
    Phase 1 (流式):  接收帧路径 → 解码 → Map → 喂入所有终端 → 返回 partial
    Phase 2+ (迭代): 接收命令 → 遍历本地 buffer → Merger → 返回 partial
    Phase 结束:       cleanup buffer → SENTINEL → 退出
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
from pathlib import Path
from typing import Any

from loguru import logger

from ..component.ipc_queue import IPCQueue
from ..component.progress import ProxyTracker
from ..component.queue import BaseQueue, RichContextQueue, StreamExhausted
from ..ops.base import BaseOp, ParallelBaseOp


def _segment_worker_main(
    segment_info: dict,
    all_configs: dict[str, dict[str, Any]],
    input_ipc: IPCQueue,
    output_ipc: IPCQueue,
    cancel_event: mp.Event,
    tracker_queue: mp.Queue,
    worker_id: int,
    dag_search_paths_str: list[str],
    ready_event: mp.Event,
    done_event: mp.Event,
) -> None:
    """Worker 进程入口：支持多终端 + 多阶段执行。

    Phase 1 (流式):
        main → worker: set_length(N), 然后 N 个帧输入
        worker 内部: I/O → Map → 所有终端并行 (Reduce merge + DiskBuffer append)
        worker → main: 1 个 partial dict

    Phase 2+ (命令驱动, per 迭代):
        main → worker: {"action": "iterate", "op_name": ..., "ref": ..., "params": ...}
        worker 内部: 遍历 local_buffer → merger.merge() → partial
        worker → main: 1 个 partial dict

    Phase 结束:
        main → worker: {"action": "finish"}
        worker 内部: cleanup local_buffer
        worker → main: SENTINEL
    """
    try:
        from .wiring import set_dag_search_paths
        if dag_search_paths_str:
            set_dag_search_paths([Path(p) for p in dag_search_paths_str])

        from .registry import REGISTERED_OP as _reg
        registry = dict(_reg)
    except Exception as e:
        logger.error(f"Segment worker {worker_id} failed to import: {e}")
        ready_event.set()
        cancel_event.set()
        return

    ready_event.set()
    logger.debug(f"Segment worker {worker_id} ready")

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

    # ── 多终端实例化 ──
    terminals_info: list[dict] = segment_info.get("terminals", [])
    reduce_terminals: list[dict] = []
    disk_buffer_terminals: list[dict] = []

    reduce_op_instances: dict[str, BaseOp] = {}
    for t_info in terminals_info:
        t_name = t_info["node_name"]
        t_type = t_info["terminal_type"]
        cls_name = segment_info["op_classes"][t_name]
        if t_type == "decomposable_reduce":
            op_inst = registry[cls_name](name=f"{t_name}_w{worker_id}")
            reduce_op_instances[t_name] = op_inst
            reduce_terminals.append(t_info)
        elif t_type == "disk_buffer":
            disk_buffer_terminals.append(t_info)

    # Phase 1 兼容：单终端 Reduce（无 terminals 字段时）
    reduce_name_phase1 = segment_info.get("reduce_op")
    if not terminals_info and reduce_name_phase1:
        cls_name = segment_info["op_classes"][reduce_name_phase1]
        op_inst = registry[cls_name](name=f"{reduce_name_phase1}_w{worker_id}")
        reduce_op_instances[reduce_name_phase1] = op_inst
        reduce_terminals = [{"node_name": reduce_name_phase1,
                             "terminal_type": "decomposable_reduce",
                             "extra_inputs": segment_info.get("reduce_extra_inputs", {})}]

    # 注入 tracker
    proxy_tracker = ProxyTracker(tracker_queue)
    all_ops = io_ops + map_ops + list(reduce_op_instances.values())
    for op in all_ops:
        op.tracker = proxy_tracker
        op._cancel_event = cancel_event

    async def _loop():
        frame_count = await input_ipc.get_length()
        has_disk_buffer = len(disk_buffer_terminals) > 0

        # ── 为每个 Reduce 终端构建本地队列 + 启动协程 ──
        reduce_local: dict[str, dict] = {}
        # reduce_local[node_name] = {
        #   "input_q": RichContextQueue, "outputs": {out_key: q}, "task": Task,
        #   "extra_queues": {key: q}
        # }

        for t_info in reduce_terminals:
            t_name = t_info["node_name"]
            op_inst = reduce_op_instances[t_name]
            extra_keys = set(t_info.get("extra_inputs", {}).keys())

            local_q = RichContextQueue(maxsize=1)
            primary_assigned = False
            extra_qs: dict[str, RichContextQueue] = {}

            for key, inp_spec in op_inst.INPUTS.items():
                if inp_spec.get("type") != "sequence":
                    continue
                if key in extra_keys:
                    eq = RichContextQueue(maxsize=1)
                    op_inst.inputs[key] = eq
                    extra_qs[key] = eq
                elif not primary_assigned:
                    op_inst.inputs[key] = local_q
                    primary_assigned = True
                else:
                    op_inst.inputs[key].active = False

            local_outputs: dict[str, RichContextQueue] = {}
            for key in op_inst.OUTPUTS:
                q = RichContextQueue(maxsize=1)
                op_inst.outputs[key].append(q)
                local_outputs[key] = q

            orig_name = t_name
            for key, val in all_configs.get(orig_name, {}).items():
                if key in op_inst.config:
                    await op_inst.config[key].put(val)

            await local_q.set_length(frame_count)
            for eq in extra_qs.values():
                await eq.set_length(frame_count)

            task = asyncio.create_task(op_inst.execute())
            reduce_local[t_name] = {
                "input_q": local_q,
                "outputs": local_outputs,
                "task": task,
                "extra_queues": extra_qs,
            }

        # ── DiskBuffer 本地实例 ──
        local_buffer = None
        use_replay = False
        if has_disk_buffer:
            db_info = disk_buffer_terminals[0]
            db_node = db_info["node_name"]
            db_configs = all_configs.get(db_node, {})
            buffer_mode = db_configs.get("buffer_mode", "auto")
            has_fnames = db_info.get("has_fnames", False)
            use_replay = (buffer_mode == "replay"
                          or (buffer_mode == "auto" and has_fnames))
            if use_replay:
                from ..component.frame_buffer import SourceReplayBuffer
                local_buffer = SourceReplayBuffer()
                logger.debug(f"Worker {worker_id}: using SourceReplayBuffer (mode={buffer_mode})")
            else:
                from ..component.frame_buffer import DiskFrameBuffer
                local_buffer = DiskFrameBuffer()
                logger.debug(f"Worker {worker_id}: using DiskFrameBuffer (mode={buffer_mode})")

        # ── Phase 1: 流式处理 ──
        processed_count = 0
        for i in range(frame_count):
            if cancel_event.is_set():
                break

            try:
                frame_input = await input_ipc.get()
            except StreamExhausted:
                break

            current = frame_input

            # I/O ops
            for op in io_ops:
                configs = all_configs.get(op.name.rsplit("_w", 1)[0], {})
                current = await _execute_single_op(op, current, configs)

            # Map ops
            for op in map_ops:
                configs = all_configs.get(op.name.rsplit("_w", 1)[0], {})
                current = await _execute_single_op(op, current, configs)

            # 提取主图像
            main_val = next(iter(current.values()))

            # 喂入所有 Reduce 终端
            for t_info in reduce_terminals:
                t_name = t_info["node_name"]
                rl = reduce_local[t_name]
                await rl["input_q"].put(main_val)
                # 额外输入
                for extra_key in t_info.get("extra_inputs", {}):
                    val = frame_input.get(f"__reduce_extra_{extra_key}")
                    if extra_key in rl["extra_queues"]:
                        await rl["extra_queues"][extra_key].put(val)

            # 喂入 DiskBuffer 终端
            if local_buffer is not None:
                if use_replay:
                    # Replay 模式：存储源文件路径，Phase 2 时重新解码
                    src_path = None
                    for k, v in frame_input.items():
                        if not k.startswith("__reduce_extra_"):
                            src_path = v
                            break
                    local_buffer.append(src_path)
                else:
                    local_buffer.append(main_val)

            processed_count += 1

        # ── 收集 Phase 1 partial results ──
        phase1_partial: dict[str, Any] = {}

        for t_info in reduce_terminals:
            t_name = t_info["node_name"]
            rl = reduce_local[t_name]
            # 通知 reduce op 输入结束
            await rl["input_q"].put(BaseQueue._SENTINEL)
            for eq in rl["extra_queues"].values():
                await eq.put(BaseQueue._SENTINEL)
            # 收集输出（先消费再 await task）
            partial = {}
            for key, out_q in rl["outputs"].items():
                try:
                    partial[key] = await out_q.get()
                except StreamExhausted:
                    pass
            await rl["task"]
            phase1_partial[t_name] = partial

        if local_buffer is not None and hasattr(local_buffer, 'to_descriptor'):
            phase1_partial["__disk_buffer"] = local_buffer.to_descriptor()

        await output_ipc.put(phase1_partial)
        logger.debug(f"Worker {worker_id}: Phase 1 partial sent, entering Phase 2 command loop")

        # ── Phase 2+: 命令驱动循环（迭代式 Reduce）──
        while True:
            try:
                cmd = await input_ipc.get()
            except StreamExhausted:
                break

            if isinstance(cmd, dict) and cmd.get("action") == "finish":
                break

            if isinstance(cmd, dict) and cmd.get("action") == "iterate":
                iter_type = cmd["iter_type"]
                ref_data = cmd["ref"]
                params = cmd.get("params", {})

                if iter_type == "sigma_clip":
                    from ..component.merger import SigmaClippingMerger
                    clip_merger = SigmaClippingMerger(
                        ref_img=ref_data,
                        rej_high=params.get("rej_high", 3.0),
                        rej_low=params.get("rej_low", 3.0),
                    )
                    for idx in range(len(local_buffer)):
                        raw, weight = local_buffer[idx]
                        clip_merger.merge(raw, weight)
                        del raw, weight
                    await output_ipc.put(clip_merger.result_as_partial())

                elif iter_type == "huber_mean":
                    from ..component.merger import HuberWeightedMerger
                    huber_merger = HuberWeightedMerger(
                        ref_stats=ref_data,
                        huber_c=params.get("huber_c", 1.345),
                    )
                    for idx in range(len(local_buffer)):
                        raw, weight = local_buffer[idx]
                        huber_merger.merge(raw, weight)
                        del raw, weight
                    await output_ipc.put(huber_merger.result_as_partial())

                else:
                    logger.warning(
                        f"Worker {worker_id}: unknown iterate type '{iter_type}'")

        # Phase 结束：清理 buffer
        if local_buffer is not None:
            local_buffer.cleanup()

        logger.debug(f"Worker {worker_id}: sending final SENTINEL")
        await output_ipc.put(BaseQueue._SENTINEL)
        logger.debug(f"Worker {worker_id}: SENTINEL sent, _loop done")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_loop())
        logger.debug(f"Worker {worker_id}: _loop completed successfully")
    except Exception as e:
        logger.error(f"Segment worker {worker_id} failed: {e}")
        cancel_event.set()
        raise
    finally:
        loop.close()
        logger.debug(f"Worker {worker_id}: waiting for done_event")
        done_event.wait(timeout=30)
        logger.debug(f"Worker {worker_id}: done_event received, exiting")


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
        async def _make_awaitable(v):
            return v
        data = {k: _make_awaitable(v) for k, v in inputs.items()}
        return await op._async_execute_single(data, configs)
    else:
        from ..ops.dataloader import ImgDataLoaderOp
        from ..component.dataloader import ImgFileListLoader, ArrayLoader

        if isinstance(op, ImgDataLoaderOp):
            loader_type = configs.get("loader_type", "img_file_list")
            src_val = inputs.get("src")
            if src_val is None:
                src_val = next(iter(inputs.values()))
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
            return inputs  # fallback: 透传
