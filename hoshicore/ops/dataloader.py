import asyncio
from typing import Any
from loguru import logger

import numpy as np

from .base import BaseOp
from ..component.dataloader import BaseLoader, ImgFileListLoader, ArrayLoader


class ImgDataLoaderOp(BaseOp):
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
            "description": "加载器配置"
        }
    }
    OUTPUTS: dict[str, Any] = {
        "result": {
            "type": "sequence",
            "description": "数据序列"
        }
    }
    MAX_SIZE: int = 1

    def __init__(self, name: str):
        super().__init__(name)

    async def _async_execute(self, configs):
        loader_class = self.build_loader_class(configs['loader_type'])
        loader = loader_class(src=self.inputs['src'],
                              length=self.length,
                              config=configs)
        self.tracker.create_bar(self.name, self.length or 0,
                                desc=f"{self.name} [Load]")
        try:
            index = 0
            async for item in loader:
                await self._broadcast_result(item)
                self.tracker.update(self.name)
                index += 1
        except Exception as e:
            logger.error(
                f"Error loading item {index} in {self.__class__.__name__}: {e.__repr__()}"
            )
            raise e
        finally:
            self.tracker.close_bar(self.name)

    async def _broadcast_result(self, result):
        tasks = []
        for queue in self.outputs['result']:
            tasks.append(queue.put(result))
        await asyncio.gather(*tasks)

    def build_loader_class(self, loader_type: str):
        mapping = {
            "img_file_list": ImgFileListLoader,
            "img_array": ArrayLoader
        }
        if loader_type not in mapping:
            raise ValueError(f"Unsupported loader type: {loader_type}")
        return mapping[loader_type]
