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

    def set_length(self, length: int):
        self.length = length

    async def pre_execute(self) -> dict[str, Any]:
        """
        该方法在执行器执行之前被调用。
        通常而言，需要等待所有配置数据都准备好。
        """
        return {
            x: await self.config[x].get()
            for x in self.config.keys()
        }
    
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
    
    """
    CONCURRENCY = 4

    async def _async_execute(self, configs: dict[str, Any]) -> None:
        if self.length is None:
            raise ValueError("Length is not set")
        # TODO: 暂未考虑并发执行
        # 暂未考虑不确定长度的执行
        for _i in range(self.length):
            data = self._async_convert_inputs()
            result = await self._async_execute_single(data, configs)
            # 将result按名称广播到所有outputs
            for key, queue in self.outputs.items():
                for output in queue:
                    await output.put(result[key])

    async def _async_convert_inputs(self):
        # NOTE: queue.get() returns an awaitable; subclasses may `await` each value.
        converted_inputs: dict[str, Awaitable[Any]] = {}
        for key, queue in self.inputs.items():
            converted_inputs[key] = queue.get()
        return converted_inputs

    async def _async_execute_single(
            self, data: Mapping[str, Awaitable[Any]], configs: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("Subclass must implement this method")

    async def _convert_inputs(self):
        converted_inputs: dict[str, Any] = {}
        for key, queue in self.inputs.items():
            converted_inputs[key] = await queue.get()
        return converted_inputs

    def _execute_single(self, data: dict[str, Any], configs: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("Subclass must implement this method")
