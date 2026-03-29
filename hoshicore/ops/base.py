from typing import Any, Optional, Sequence, Awaitable, Mapping
import asyncio
from ..component.queue import RichContextQueue, FileCacheQueue


class BaseOp(object):
    EXECUTOR: Optional[str] = None
    INPUTS: dict[str, Any] = {}
    CONFIGS: dict[str, Any] = {}
    OUTPUTS: dict[str, Any] = {}
    MAX_SIZE: int = 1
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

    async def pre_execute(self) -> dict[str, Any]:
        """
        该方法在执行器执行之前被调用。
        通常而言，执行器会等待所有配置数据都准备好，并且确认长度信息已经就绪。
        """

        # 验证所有序列输入等长
        input_lengths = {}
        for name, queue in self.inputs.items():
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
        configs = await self.pre_execute()
        await self._async_execute(configs)

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
        for _i in range(self.length):
            data = await self._async_convert_inputs()
            result = await self._async_execute_single(data, configs)
            await self._broadcast_result(result)

    async def _execute_concurrent(self, configs: dict[str, Any]) -> None:
        """并发执行：滑动窗口保证输出有序"""
        if self.length is None:
            raise ValueError("Length is not set")
        window_size = self.WINDOW_SIZE or (self.CONCURRENCY * 2)
        semaphore = asyncio.Semaphore(self.CONCURRENCY)

        for window_start in range(0, self.length, window_size):
            window_end = min(window_start + window_size, self.length)
            window_len = window_end - window_start
            results: list[dict[str, Any]] = [{}] * window_len

            async def process_item(local_idx: int):
                async with semaphore:
                    data = await self._async_convert_inputs()
                    result = await self._async_execute_single(data, configs)
                    results[local_idx] = result

            await asyncio.gather(*[process_item(i) for i in range(window_len)])

            for result in results:
                await self._broadcast_result(result)

    async def _broadcast_result(self, result: dict[str, Any]) -> None:
        """广播结果到所有输出队列"""
        for key, queue_list in self.outputs.items():
            for queue in queue_list:
                await queue.put(result[key])

    async def _async_convert_inputs(self):
        # NOTE: queue.get() returns an awaitable; subclasses may `await` each value.
        converted_inputs: dict[str, Awaitable[Any]] = {}
        for key, queue in self.inputs.items():
            converted_inputs[key] = queue.get()
        return converted_inputs

    async def _async_execute_single(self, data: Mapping[str, Awaitable[Any]],
                                    configs: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("Subclass must implement this method")
