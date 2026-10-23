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
            "description": "数据加载器类型",
            "default": "img_file_list",
        },
        "configs": {
            "type": "dict",
            "description": "加载器配置",
            "default": {},
        },
        "raw_backend": {
            "type": "str",
            "default": "native",
            "global": True,
            "description": "RAW 解码方式：native / photoshop",
        }
    }
    OUTPUTS: dict[str, Any] = {
        "result": {
            "type": "sequence",
            "description": "数据序列"
        },
        "exifs": {
            "type": "sequence",
            "description": "与解码像素对应的 EXIF 和 ICC"
        },
    }
    CONCURRENCY = 4

    @classmethod
    def estimate_resources(cls,
                           configs,
                           frame_bytes,
                           n_frames,
                           dtype_bytes=None):
        _ = dtype_bytes
        return (cls.CONCURRENCY * frame_bytes, 0)

    async def _async_execute(self, configs):
        loader_cls = self.build_loader_class(configs['loader_type'])
        self._loader = loader_cls(self.inputs['src'], self.length, configs,
                                  services=self._services)
        failed = False
        try:
            if self.length != 0:
                await super()._async_execute(configs)
        except BaseException:
            failed = True
            raise
        finally:
            try:
                await self._loader.aclose()
            except Exception as error:
                if not failed:
                    raise
                logger.warning('Loader cleanup failed after an error: {}',
                               error)

    async def _async_execute_single(self, data: Mapping[str, Awaitable[Any]],
                                    configs: dict[str, Any]) -> dict[str, Any]:
        src_val = await data['src']
        result, exif = await self._loader.aload(src_val,
                                                with_exif=bool(
                                                    self.outputs['exifs']))
        return {"result": result, "exifs": exif}

    def build_loader_class(self, loader_type: str) -> type[BaseLoader]:
        mapping = {
            "img_file_list": ImgFileListLoader,
            "img_array": ArrayLoader
        }
        if loader_type not in mapping:
            raise ValueError(f"Unsupported loader type: {loader_type}")
        return mapping[loader_type]
