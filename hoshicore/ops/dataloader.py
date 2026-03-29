import asyncio
from typing import Any
from loguru import logger

from .base import ParallelBaseOp
from ..component.dataloader import BaseLoader, ImgFileListLoader, ArrayLoader, VideoFileLoader


class DataLoaderOp(ParallelBaseOp):
    """
    通用异步数据加载器，用于异步预取数据，提高数据加载效率。
    
    Args:
        loader (BaseLoader): 实际数据加载源。
        max_poolsize (int): 缓冲区最大存储样本数。
    
    用法：
        dataloader = AsyncDataLoader(loader, max_poolsize=4)
        dataloader.start()
        for data in dataloader:
            ...
        dataloader.stop()
    """
    INPUTS: dict[str, Any] = {
        "src": {
            "type": "sequence",
            "description": "数据源"
        }
    }
    CONFIGS: dict[str, Any] = {
        "loader_type": {
            "type": "str",
            "description": "数据加载器类型"
        },
        "configs": {
            "type": "dict",
            "description": "最大缓冲区大小"
        }
    }
    OUTPUTS: dict[str, Any] = {
        "result": {
            "type": "sequence",
            "description": "数据序列"
        }
    }
    MAX_SIZE: int = 1
    _SENTINEL = object()

    def __init__(self, loader: BaseLoader, max_poolsize: int = 1):
        self.loader = loader
        
        self.data_queue: asyncio.Queue[Any] = asyncio.Queue(
            maxsize=max_poolsize)
        self._length = loader.length
        self._worker_task = None

    async def _worker(self, data, configs):
        self.loader_class = self.build_loader_class(configs['loader_type'])
        self.max_poolsize = configs['max_poolsize']
        self.loader = self.loader_class(src=data['src'], config=configs)
        for i in range(self._length):
            try:
                item = self.loader.__next__()
            except Exception as e:
                logger.error(
                    f"Error loading item {i} in {self.__class__.__name__}: {e.__repr__()}"
                )
                continue
            await self.data_queue.put(item)
        await self.data_queue.put(self._SENTINEL)

    async def _async_execute_single(self, data, configs):
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker(data, configs))
        # TODO: 没有实现结束信号的处理（返回上层未捕捉）
        return await self.data_queue.get()

    def __len__(self):
        return self._length

    def build_loader_class(self, loader_type: str):
        if loader_type == "img_file_list":
            return ImgFileListLoader
        elif loader_type == "video_file":
            return VideoFileLoader
        elif loader_type == "img_array":
            return ArrayLoader
        else:
            raise ValueError(f"Unsupported loader type: {loader_type}")
