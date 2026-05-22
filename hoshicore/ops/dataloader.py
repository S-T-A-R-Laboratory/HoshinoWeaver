import asyncio
from typing import Any, Awaitable, Mapping

from loguru import logger

from ..component.dataloader import ArrayLoader, BaseLoader, ImgFileListLoader
from ..engine.registry import register_op
from .base import ParallelBaseOp


@register_op()
class ImgDataLoaderOp(ParallelBaseOp):
    """通用数据加载器：并发解码输入文件，流式有序输出。

    利用 ParallelBaseOp 的流式并发模型，多帧 IO+decode 同时进行，
    打破单帧串行的吞吐瓶颈。
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
    CONCURRENCY = 4

    @classmethod
    def estimate_resources(cls, configs, frame_bytes, n_frames):
        return (cls.CONCURRENCY * frame_bytes, 0)

    async def _async_execute_single(self, data: Mapping[str, Awaitable[Any]],
                                    configs: dict[str, Any]) -> dict[str, Any]:
        src_val = await data['src']
        loader_cls = self.build_loader_class(configs['loader_type'])
        result = await asyncio.to_thread(loader_cls.load, None, src_val)
        return {"result": result}

    def build_loader_class(self, loader_type: str) -> BaseLoader:
        mapping = {
            "img_file_list": ImgFileListLoader,
            "img_array": ArrayLoader
        }
        if loader_type not in mapping:
            raise ValueError(f"Unsupported loader type: {loader_type}")
        return mapping[loader_type]