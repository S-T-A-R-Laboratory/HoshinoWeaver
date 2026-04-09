from typing import Any, Optional, Sequence, Awaitable, Mapping
import asyncio
from loguru import logger
from ..component.progress import DummyTracker
from ..component.queue import RichContextQueue, FileCacheQueue, CancellationError, CancellationToken


class BaseOp(object):
    EXECUTOR: Optional[str] = None
    INPUTS: dict[str, Any] = {}
    CONFIGS: dict[str, Any] = {}
    OUTPUTS: dict[str, Any] = {}
    MAX_SIZE: int = 1
    PROGRESS_DESC: Optional[str] = None  # 非 None 时 execute() 自动创建进度条
    _SENTINEL = object()

    def __init__(self, name: str):
        self.config: dict[str, RichContextQueue] = {
            x: RichContextQueue(maxsize=self.MAX_SIZE)
            for x in self.CONFIGS.keys()
        }
        self.inputs: dict[str, RichContextQueue] = {
            x: RichContextQueue(maxsize=self.MAX_SIZE)
            for x in self.INPUTS.keys()
        }
        self.outputs: dict[str, list[RichContextQueue]] = {
            x: []
            for x in self.OUTPUTS.keys()
        }
        self.length: Optional[int] = None
        self.name = name
        self.tracker = DummyTracker()

    async def pre_execute(self) -> dict[str, Any]:
        """
        该方法在执行器执行之前被调用。
        通常而言，执行器会等待所有配置数据都准备好，并且确认长度信息已经就绪。
        """

        # 验证所有序列输入等长（跳过未布线的可选输入）
        input_lengths = {}
        for name, queue in self.inputs.items():
            if not queue.active:
                continue
            input_lengths[name] = await queue.get_length()
        seq_lengths = [
            length for name, length in input_lengths.items()
            if self.INPUTS[name].get("type") == "sequence"
        ]
        if seq_lengths and len(set(seq_lengths)) > 1:
            raise ValueError(f"Input sequence length mismatch: {seq_lengths}")

        # 向所有输出的序列类队列广播长度
        self.length = self._infer_output_length(input_lengths)
        for key, queue_list in self.outputs.items():
            if self.OUTPUTS[key].get("type") != "sequence":
                continue
            for queue in queue_list:
                await queue.set_length(self.length)

        # 等待配置就绪
        return {x: await self.config[x].get() for x in self.config.keys()}

    def _infer_output_length(self, input_lengths: dict[str, int]) -> int:
        """输出序列长度"""
        # 默认：如果有序列输入，输出长度等于输入长度
        seq_lengths = [length for name, length in input_lengths.items()]
        if seq_lengths:
            return seq_lengths[0]
        return 1  # 无序列输入，输出单个结果

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
                await queue.put(RichContextQueue._SENTINEL)

    async def _propagate_cancellation(self, error: Exception) -> None:
        """传播取消令牌（本节点异常）"""
        token = CancellationToken(error, self.name)
        for queue_list in self.outputs.values():
            for queue in queue_list:
                await queue.put(token)

    async def _propagate_cancellation_from_upstream(self) -> None:
        """传播取消令牌（上游异常）"""
        for input_queue in self.inputs.values():
            try:
                while not input_queue.queue.empty():
                    item = input_queue.queue.get_nowait()
                    if isinstance(item, CancellationToken):
                        for queue_list in self.outputs.values():
                            for queue in queue_list:
                                await queue.put(item)
                        return
            except:
                pass

    async def _run_cpu(self, fn, *args, **kwargs):
        """将 CPU 密集型同步函数卸载到线程池执行，释放事件循环。

        利用 numpy C 扩展释放 GIL 的特性，使 CPU 计算与 I/O 操作可以重叠执行。

        设计为统一入口：后续阶段可无缝替换为 ProcessPoolExecutor 以实现真正多核并行。
        """
        return await asyncio.to_thread(fn, *args, **kwargs)

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
        """串行执行"""
        if self.length is None:
            raise ValueError("Length is not set")
        self.tracker.create_bar(self.name, self.length)
        for i in range(self.length):
            try:
                data = self._async_convert_inputs()
            except StopIteration:
                logger.warning(f"{self.name}: upstream ended at {i}/{self.length}")
                break
            result = await self._async_execute_single(data, configs)
            self.tracker.update(self.name)
            await self._broadcast_result(result)
        self.tracker.close_bar(self.name)

    async def _execute_concurrent(self, configs: dict[str, Any]) -> None:
        """并发执行：滑动窗口保证输出有序"""
        if self.length is None:
            raise ValueError("Length is not set")
        self.tracker.create_bar(self.name, self.length)
        window_size = self.WINDOW_SIZE or (self.CONCURRENCY * 2)
        semaphore = asyncio.Semaphore(self.CONCURRENCY)

        for window_start in range(0, self.length, window_size):
            window_end = min(window_start + window_size, self.length)
            window_len = window_end - window_start
            results: list[dict[str, Any]] = [{}] * window_len

            async def process_item(local_idx: int):
                async with semaphore:
                    data = self._async_convert_inputs()
                    result = await self._async_execute_single(data, configs)
                    results[local_idx] = result
                    self.tracker.update(self.name)

            await asyncio.gather(*[process_item(i) for i in range(window_len)])

            for result in results:
                await self._broadcast_result(result)
        self.tracker.close_bar(self.name)

    async def _broadcast_result(self, result: dict[str, Any]) -> None:
        """广播结果到所有输出队列"""
        tasks = []
        for key, queue_list in self.outputs.items():
            for queue in queue_list:
                tasks.append(queue.put(result[key]))
        await asyncio.gather(*tasks)

    async def _async_execute_single(self, data: Mapping[str, Awaitable[Any]],
                                    configs: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("Subclass must implement this method")
