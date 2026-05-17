import asyncio
import itertools
import sys
from typing import Any, Awaitable, Mapping, Optional, Sequence

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
        self._cancel_event: Optional[Any] = None  # asyncio.Event 或 mp.Event，由 wiring 注入

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
            name: length for name, length in input_lengths.items()
            if self.INPUTS[name].get("type") == "sequence" and length is not None
        }
        none_seq = {
            name: length for name, length in input_lengths.items()
            if self.INPUTS[name].get("type") == "sequence" and length is None
        }

        # ── 3. 混合检查（最先执行，阻止 int+None 混合）──
        if known_seq and none_seq:
            raise ValueError(
                f"{self.name}: cannot mix known-length and sentinel-driven "
                f"sequence inputs — known: {list(known_seq.keys())} "
                f"(lengths: {list(known_seq.values())}), "
                f"unknown: {list(none_seq.keys())}. "
                f"Use FilterGate pattern to align sequences before merging."
            )

        # ── 4. 等长校验（仅对已知长度）──
        if len(set(known_seq.values())) > 1:
            raise ValueError(f"Input sequence length mismatch: {dict(known_seq)}")

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

    def _infer_output_length(self, input_lengths: dict[str, Optional[int]]) -> Optional[int]:
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
            logger.error(f"{self.name} failed: {e}")
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

    async def _run_parallel_cpu(self, fn, *args, **kwargs):
        """执行可能自行管理并行的 CPU 计算函数。

        当前主要用于 custom-op / C 扩展 / 其他内部并行实现。
        Windows frozen 环境下，为避免非主线程执行底层并行代码时死锁，
        会退回主线程同步执行；非 frozen 环境正常卸载到线程池。
        """
        if getattr(sys, 'frozen', False) and sys.platform == 'win32':
            result = fn(*args, **kwargs)
        else:
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
        """并发执行：滑动窗口保证输出有序。

        支持定长和 sentinel 驱动两种模式：
        - 定长模式：按 window_size 分批，每批固定数量任务。
        - sentinel 模式：每批启动 window_size 个任务，遇到 StreamExhausted
          的任务标记为 None，广播时跳过。批内有任何 sentinel 命中即结束外层循环。

        注意：窗口内多任务并发调用 queue.get()，数据帧分配顺序依赖
        CPython asyncio.Queue 的 FIFO 唤醒实现（基于 collections.deque）。
        语言规范未明确保证此行为，但所有主流 CPython 版本均如此。
        如需严格保证顺序，需要串行预取数据后并发处理，但这会改变
        _async_execute_single 的接口签名（从 Awaitable 变为已解析值）。
        """
        if self.length is not None:
            self.tracker.create_bar(self.name, self.length)
        window_size = self.WINDOW_SIZE or (self.CONCURRENCY * 2)
        semaphore = asyncio.Semaphore(self.CONCURRENCY)
        _STOP = object()  # 内部标记：该槽位遇到 sentinel

        for window_start in self._input_range():
            if window_start % window_size != 0:
                continue  # _input_range 逐步递增，只在窗口边界启动批次
            if self.length is not None:
                window_len = min(window_size, self.length - window_start)
            else:
                window_len = window_size
            results: list = [_STOP] * window_len

            async def process_item(local_idx: int):
                async with semaphore:
                    data = self._async_convert_inputs()
                    try:
                        result = await self._async_execute_single(data, configs)
                    except StreamExhausted:
                        for awaitable in data.values():
                            if hasattr(awaitable, 'close'):
                                awaitable.close()
                        return  # results[local_idx] 保持 _STOP
                    results[local_idx] = result
                    if self.length is not None:
                        self.tracker.update(self.name)

            await asyncio.gather(*[process_item(i) for i in range(window_len)])

            has_stop = False
            for result in results:
                if result is _STOP:
                    has_stop = True
                    continue
                await self._broadcast_outputs(result)
            if has_stop:
                break  # 本窗口内有 sentinel → 序列结束

        if self.length is not None:
            self.tracker.close_bar(self.name)

    async def _async_execute_single(self, data: Mapping[str, Awaitable[Any]],
                                    configs: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("Subclass must implement this method")


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
