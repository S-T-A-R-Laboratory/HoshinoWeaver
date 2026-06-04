import asyncio
import itertools
import sys
from typing import Any, Awaitable, Mapping, Optional, Sequence

import numpy as np

from loguru import logger

from ..component.progress import DummyTracker
from ..component.queue import (BaseQueue, CancellationError, CancellationToken,
                               FileCacheQueue, RichContextQueue,
                               StreamExhausted)
from ..component.utils import time_cost_warpper


class BaseOp(object):
    EXECUTOR: Optional[str] = None
    INPUTS: dict[str, Any] = {}
    CONFIGS: dict[str, Any] = {}
    OUTPUTS: dict[str, Any] = {}
    MAX_SIZE: int = 1
    VARIABLE_OUTPUT: bool = False  # True 时标记为变长输出（Filter 类）
    CHUNK_PLANNED: bool = False  # True 时 chunk_rows 由 runtime planner 管理

    @classmethod
    def estimate_resources(
        cls,
        configs: dict[str, Any],
        frame_bytes: int,
        n_frames: Optional[int],
        dtype_bytes: Optional[int] = None,
    ) -> tuple[int, int]:
        """返回 (peak_memory_bytes, peak_disk_bytes) 的估计值。

        预检阶段调用，用于在执行前估算资源需求。
        子类按需 override，默认返回 (0, 0)。
        """
        return (0, 0)

    @classmethod
    def chunk_cost_per_row(
        cls,
        n_frames: int,
        row_bytes: int,
        dtype_bytes: int,
    ) -> int:
        """返回 chunk_rows 每增加一行带来的内存成本。"""
        _ = dtype_bytes
        return 2 * n_frames * row_bytes

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if '_async_execute' in cls.__dict__:
            cls._async_execute = time_cost_warpper(cls._async_execute)

    def __init__(self, name: str):
        self.config: dict[str, BaseQueue] = {
            x: RichContextQueue(maxsize=self.MAX_SIZE)
            for x in self.CONFIGS.keys()
        }
        self.inputs: dict[str, BaseQueue] = {
            x: RichContextQueue(maxsize=self.MAX_SIZE)
            for x in self.INPUTS.keys()
        }
        self.outputs: dict[str, list[BaseQueue]] = {
            x: []
            for x in self.OUTPUTS.keys()
        }
        self.length: Optional[int] = None
        self.name = name
        self.tracker = DummyTracker()
        self._cancel_event: Optional[
            Any] = None  # asyncio.Event 或 mp.Event，由 wiring 注入

    async def pre_execute(self) -> dict[str, Any]:
        """
        该方法在执行器执行之前被调用。
        等待所有配置数据就绪，确认长度信息就绪，并向下游广播输出长度。
        """

        # ── 1. 收集输入长度（可能含 None）──
        input_lengths: dict[str, Optional[int]] = {}
        for name, queue in self.inputs.items():
            if not queue.active:
                continue
            input_lengths[name] = await queue.get_length()

        # ── 2. 分类：哪些序列输入长度已知，哪些未知 ──
        known_seq = {
            name: length
            for name, length in input_lengths.items() if
            self.INPUTS[name].get("type") == "sequence" and length is not None
        }
        none_seq = {
            name: length
            for name, length in input_lengths.items()
            if self.INPUTS[name].get("type") == "sequence" and length is None
        }

        # ── 3. 混合检查（最先执行，阻止 int+None 混合）──
        if known_seq and none_seq:
            raise ValueError(
                f"{self.name}: cannot mix known-length and sentinel-driven "
                f"sequence inputs — known: {list(known_seq.keys())} "
                f"(lengths: {list(known_seq.values())}), "
                f"unknown: {list(none_seq.keys())}. "
                f"Use FilterGate pattern to align sequences before merging.")

        # ── 4. 等长校验（仅对已知长度）──
        if len(set(known_seq.values())) > 1:
            raise ValueError(
                f"Input sequence length mismatch: {dict(known_seq)}")

        # ── 5. self.length = 输入序列长度（用于迭代）──
        if known_seq:
            self.length = next(iter(known_seq.values()))
        elif none_seq:
            self.length = None  # 全部 sentinel 驱动
        else:
            self.length = None  # 无序列输入（如仅 config 驱动的 Op）

        # ── 6. 输出长度广播 ──
        output_length = self._infer_output_length(input_lengths)
        for key, queue_list in self.outputs.items():
            is_seq = self.OUTPUTS[key].get("type") == "sequence"
            length = output_length if is_seq else 1
            for queue in queue_list:
                await queue.set_length(length)

        # ── 7. 等待配置就绪 ──
        return {x: await self.config[x].get() for x in self.config.keys()}

    def _infer_output_length(
            self, input_lengths: dict[str, Optional[int]]) -> Optional[int]:
        """推断输出序列长度。

        返回 int: 已知长度，在 pre_execute 中立即广播。
        返回 None: 长度未知（Filter 类），sentinel 驱动。

        子类可 override 此方法。Filter 类 Op 返回 None 即可。
        """
        for name, length in input_lengths.items():
            if length is not None:
                return length
        return None if input_lengths else 1

    def _input_range(self):
        """返回输入序列迭代范围。

        - self.length is int: range(N)（长度已知，有界迭代）
        - self.length is None: itertools.count()（sentinel 驱动，无限迭代，
          配合 except StreamExhausted: break 使用）
        """
        if self.length is not None:
            return range(self.length)
        return itertools.count()

    async def _set_output_length(self, length: int) -> None:
        """手动设置输出序列长度（用于 Filter 类 Op 延迟广播场景）。"""
        for key, queue_list in self.outputs.items():
            if self.OUTPUTS[key].get("type") != "sequence":
                continue
            for queue in queue_list:
                await queue.set_length(length)

    async def _async_execute(self, configs: dict[str, Any]) -> None:
        raise NotImplementedError("Subclass must implement this method")

    async def execute(self) -> None:
        """执行入口：捕获异常并传播"""
        try:
            configs = await self.pre_execute()
            await self._async_execute(configs)
            # 正常结束：发送结束信号
            await self._send_sentinel()
        except CancellationError:
            # 上游取消：直接传播
            await self._propagate_cancellation_from_upstream()
            raise
        except asyncio.CancelledError:
            # 外部取消（如 UI）→ 转化为内部取消语义
            logger.info(f"{self.name}: cancelled by external request")
            cancel_err = CancellationError("External cancellation")
            await self._propagate_cancellation(cancel_err)
            raise cancel_err
        except Exception as e:
            # 本节点异常：创建取消令牌并传播
            import traceback
            logger.error(
                f"{self.name} failed: {e.__repr__()}\n{traceback.format_exc()}"
            )
            await self._propagate_cancellation(e)
            raise

    def _async_convert_inputs(self):
        # NOTE: queue.get() returns an awaitable; subclasses may `await` each value.
        # 跳过非活跃队列（未布线的可选输入），避免产生永远不会被 await 的 coroutine。
        converted_inputs: dict[str, Awaitable[Any]] = {}
        for key, queue in self.inputs.items():
            if not queue.active:
                continue
            converted_inputs[key] = queue.get()
        return converted_inputs

    async def _send_sentinel(self) -> None:
        """发送正常结束信号"""
        for queue_list in self.outputs.values():
            for queue in queue_list:
                await queue.put(BaseQueue._SENTINEL)

    async def _propagate_cancellation(self, error: Exception) -> None:
        """传播取消令牌（本节点异常）"""
        token = CancellationToken(error, self.name)
        for queue_list in self.outputs.values():
            for queue in queue_list:
                await queue.put(token)

    async def _propagate_cancellation_from_upstream(self) -> None:
        """传播取消令牌（上游异常）。

        从输入队列中提取 CancellationToken 并转发到所有输出队列。
        """
        token = None
        for input_queue in self.inputs.values():
            try:
                if hasattr(input_queue, 'queue'):
                    while not input_queue.queue.empty():
                        item = input_queue.queue.get_nowait()
                        if isinstance(item, CancellationToken):
                            token = item
                            break
                if token is not None:
                    break
            except Exception:
                pass

        if token is None:
            token = CancellationToken(
                CancellationError("Upstream cancellation"), self.name)

        for queue_list in self.outputs.values():
            for queue in queue_list:
                await queue.put(token)

    async def _broadcast_outputs(self, results: dict[str, Any]) -> None:
        """将结果广播到对应的输出队列。

        与 _async_convert_inputs 对称的输出接口：子类在 _async_execute 中构造
        {output_name: value} 字典，通过本方法统一推送。

        - 仅推送 results 中存在的 key；未提供的输出端口不受影响
        - 输出端口无下游连接（空队列列表）时静默跳过
        - 所有 put 并发执行以最小化延迟

        Args:
            results: 输出名称到值的映射，key 必须是 OUTPUTS 中声明的端口名。
        """
        tasks = []
        for key, value in results.items():
            for queue in self.outputs[key]:
                tasks.append(queue.put(value))
        if tasks:
            await asyncio.gather(*tasks)

    async def _run_cpu(self, fn, *args, **kwargs):
        """将 CPU 密集型同步函数卸载到线程池执行，释放事件循环。

        利用 numpy C 扩展释放 GIL 的特性，使 CPU 计算与 I/O 操作可以重叠执行。

        设计为统一入口：后续阶段可无缝替换为 ProcessPoolExecutor 以实现真正多核并行。
        """
        result = await asyncio.to_thread(fn, *args, **kwargs)
        if self._cancel_event is not None and self._cancel_event.is_set():
            raise CancellationError("Cancelled during CPU execution")
        return result

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__class__.__name__


class ParallelBaseOp(BaseOp):
    """
    ParallelBaseOp is a base class for parallel execution of operations.

    ParallelBaseOp 是并行执行的基类，用于并行执行算子。
    并行执行算子不依赖序列前后的数据，因此在本部分做了简化，只需要实现处理单个元素的_async_execute_single方法即可。

    CONCURRENCY = 1: 串行执行（简化实现）
    CONCURRENCY > 1: 并发执行，使用滑动窗口保证输出有序
    WINDOW_SIZE: 滑动窗口大小，默认为 CONCURRENCY * 2
    """
    CONCURRENCY = 1
    WINDOW_SIZE: Optional[int] = None

    async def _async_execute(self, configs: dict[str, Any]) -> None:

        if self.CONCURRENCY == 1:
            await self._execute_serial(configs)
        else:
            await self._execute_concurrent(configs)

    async def _execute_serial(self, configs: dict[str, Any]) -> None:
        """串行执行（支持定长和 sentinel 驱动两种模式）"""
        if self.length is not None:
            self.tracker.create_bar(self.name, self.length)
        for i in self._input_range():
            data = self._async_convert_inputs()
            try:
                result = await self._async_execute_single(data, configs)
            except StreamExhausted:
                # sentinel 从 _async_execute_single 内部 await data[key] 时抛出
                # 清理未 await 的 coroutine，避免 RuntimeWarning
                for awaitable in data.values():
                    if hasattr(awaitable, 'close'):
                        awaitable.close()
                break
            if self.length is not None:
                self.tracker.update(self.name)
            await self._broadcast_outputs(result)
        if self.length is not None:
            self.tracker.close_bar(self.name)

    async def _execute_concurrent(self, configs: dict[str, Any]) -> None:
        """流式并发执行。

        输出缓冲上限 = CONCURRENCY 帧（由 slots semaphore 保证）。

        数据帧分配顺序依赖 CPython asyncio.Queue 的 FIFO 唤醒实现
        （基于 collections.deque）。语言规范未明确保证此行为，
        但所有主流 CPython 版本均如此。
        """
        if self.length is not None:
            self.tracker.create_bar(self.name, self.length)

        _STOP = object()
        pending: dict[int, Any] = {}
        emit_event = asyncio.Event()
        sentinel_event = asyncio.Event()

        # slots: producer acquire, emitter release → pending 上限 = CONCURRENCY
        slots = asyncio.Semaphore(self.CONCURRENCY)

        async def process_item(idx: int):
            data = self._async_convert_inputs()
            try:
                result = await self._async_execute_single(data, configs)
            except (StreamExhausted, CancellationError):
                for awaitable in data.values():
                    if hasattr(awaitable, 'close'):
                        awaitable.close()
                result = _STOP
                sentinel_event.set()
            pending[idx] = result
            emit_event.set()

        async def emit_loop():
            next_emit = 0
            while True:
                while next_emit in pending:
                    result = pending.pop(next_emit)
                    if result is _STOP:
                        slots.release()
                        return
                    await self._broadcast_outputs(result)
                    if self.length is not None:
                        self.tracker.update(self.name)
                    next_emit += 1
                    slots.release()
                    if self.length is not None and next_emit >= self.length:
                        return
                emit_event.clear()
                await emit_event.wait()

        emit_task = asyncio.create_task(emit_loop())
        tasks: list[asyncio.Task] = []

        for idx in self._input_range():
            if sentinel_event.is_set() or emit_task.done():
                break
            await slots.acquire()
            if sentinel_event.is_set() or emit_task.done():
                slots.release()
                break
            task = asyncio.create_task(process_item(idx))
            tasks.append(task)

        # 等待 in-flight tasks 完成（sentinel 回填保证不会永久阻塞）
        for t in tasks:
            if not t.done():
                try:
                    await asyncio.wait_for(t, timeout=5.0)
                except (asyncio.TimeoutError, Exception):
                    t.cancel()

        emit_event.set()
        if not emit_task.done():
            await emit_task

        if self.length is not None:
            self.tracker.close_bar(self.name)

    async def _async_execute_single(self, data: Mapping[str, Awaitable[Any]],
                                    configs: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("Subclass must implement this method")


class ChunkIteratorBaseOp(BaseOp):
    """Chunk-level 迭代基类：将 buffer 重放按空间分块处理。

    主循环骨架：
        for chunk in chunks:
            state = _init_chunk_state(...)
            for pass in passes:
                for frame in frames:
                    _merge_chunk(state, frame[row_start:row_end], weight)
                if _check_convergence(state): break
            result_chunks.append(_finalize_chunk(state))
        result = concatenate(result_chunks)

    核心收益：multi-pass 算法的 IO 从 n_passes × data 降为 ~1 × data
    （chunk 内所有 pass 复用 OS page cache）。
    """
    BUFFER_ITERATOR = True
    CHUNK_PLANNED = True
    CHUNK_ROWS: int = 256
    CHUNK_OVERLAP: int = 0

    def __init__(self, name: str):
        super().__init__(name)
        self._chunk_states: list[Any] = []

    async def _async_execute(self, configs: dict[str, Any]) -> None:
        frame_buffer = configs['buffer_handle']
        chunk_rows = configs.get('chunk_rows', self.CHUNK_ROWS)
        overlap = self.CHUNK_OVERLAP

        try:
            first_frame, _ = frame_buffer[0]
            h, w = first_frame.shape[:2]
            del first_frame  # release mmap immediately — prevents Windows file-lock on cleanup
            n_chunks = (h + chunk_rows - 1) // chunk_rows
            result_chunks: list[np.ndarray] = []
            self._chunk_states = []

            row_ranges = [
                (max(0, cidx * chunk_rows - overlap),
                 min(h, (cidx + 1) * chunk_rows + overlap))
                for cidx in range(n_chunks)
            ]

            self.tracker.create_bar(self.name, n_chunks, unit="chunks")
            chunk_idx = 0
            async for chunk_stack in frame_buffer.iter_chunk_prefetch(row_ranges):
                row_start, row_end = row_ranges[chunk_idx]
                out_start = (chunk_idx * chunk_rows) - row_start
                out_end = out_start + min(chunk_rows, h - chunk_idx * chunk_rows)

                state = self._init_chunk_state(configs, row_start, row_end, w)

                for pass_idx in range(self._max_passes(configs)):
                    await self._run_cpu(self._run_pass, state, chunk_stack)

                    if self._check_convergence(state, pass_idx):
                        break
                    self._prepare_next_pass(state, pass_idx)

                chunk_result = self._finalize_chunk(state)
                result_chunks.append(chunk_result[out_start:out_end])
                self._chunk_states.append(state)
                self.tracker.update(self.name)
                chunk_idx += 1

            result = np.concatenate(result_chunks, axis=0)
            await self._broadcast_outputs(
                self._wrap_output(result, configs))
        finally:
            self.tracker.close_bar(self.name)
            frame_buffer.cleanup()

    # --- 子类钩子 ---

    def _init_chunk_state(self, configs: dict[str, Any],
                          row_start: int, row_end: int, w: int) -> Any:
        raise NotImplementedError

    def _merge_chunk(self, state: Any, chunk_data: np.ndarray,
                     chunk_weight, frame_idx: int) -> None:
        raise NotImplementedError

    def _max_passes(self, configs: dict[str, Any]) -> int:
        return 1

    def _check_convergence(self, state: Any, pass_idx: int) -> bool:
        return False

    def _prepare_next_pass(self, state: Any, pass_idx: int) -> None:
        pass

    def _finalize_chunk(self, state: Any) -> np.ndarray:
        raise NotImplementedError

    def _wrap_output(self, result: np.ndarray,
                     configs: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def _run_pass(self, state: Any, chunk_stack: list) -> None:
        """Merge all frames for one pass. Called inside _run_cpu."""
        for frame_idx, (chunk_data, chunk_weight) in enumerate(chunk_stack):
            self._merge_chunk(state, chunk_data, chunk_weight, frame_idx)


class FilterBaseOp(BaseOp):
    """Filter 类算子基类：输出序列长度不等于输入序列长度。

    子类实现 _async_execute，在循环中选择性地调用 _broadcast_outputs。
    输出自动标记为 sentinel 驱动（_infer_output_length → None）。
    wiring 层通过 VARIABLE_OUTPUT 做静态冲突检测。

    典型用法::

        @register_op()
        class MyFilter(FilterBaseOp):
            INPUTS = {"data": {"type": "sequence", "required": True}}
            OUTPUTS = {"result": {"type": "sequence"}}

            async def _async_execute(self, configs):
                for i in self._input_range():
                    data = self._async_convert_inputs()
                    try:
                        item = await data['data']
                    except StreamExhausted:
                        break
                    if predicate(item):
                        await self._broadcast_outputs({"result": item})
    """
    VARIABLE_OUTPUT = True

    def _infer_output_length(self, input_lengths):
        return None
