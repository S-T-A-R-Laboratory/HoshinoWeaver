"""
WeightGeneratorOp：根据序列长度和渐入渐出参数，生成逐帧权重序列。

权重生成逻辑复制自 ezlib/trailstacker.py::generate_weight，
适配为异步 DAG Op。
"""
import asyncio
from typing import Any

import numpy as np
from loguru import logger

from .base import BaseOp

# ---------------------------------------------------------------------------
# 权重生成函数（从 ezlib/trailstacker.py 迁移，内联必要的工具依赖）
# ---------------------------------------------------------------------------

DTYPE_UPSCALE_MAP = {
    np.dtype('uint8'): np.dtype('uint16'),
    np.dtype('uint16'): np.dtype('uint32'),
    np.dtype('uint32'): np.dtype('uint64'),
    np.dtype('uint64'): float,
}


def _get_scale_x(time: int, base: int = 256) -> int:
    return base ** time + 1


def generate_weight(
    length: int,
    fin: float,
    fout: float,
    int_weight: bool = False,
    input_dtype=np.dtype("uint8"),
) -> np.ndarray:
    """为渐入渐出星轨生成每张图像分配的权重。

    Args:
        length: 序列长度。
        fin: 渐入比例 (0-1)。
        fout: 渐出比例 (0-1)。
        int_weight: 是否将权重映射到 uint8/uint16 整型范围以加速运算。
        input_dtype: 输入图像的 dtype，用于决定整型权重的放缩倍数。

    Returns:
        np.ndarray: 权重数组，shape = (length,)。
    """
    assert fin + fout <= 1, f"fin({fin}) + fout({fout}) > 1"
    in_len = int(length * fin)
    out_len = int(length * fout)
    ret_weight = np.ones((length,), dtype=np.float16)

    multi_base = _get_scale_x({
        np.dtype("uint8"): 1,
        np.dtype("uint16"): 2,
    }[input_dtype])
    dtype = DTYPE_UPSCALE_MAP[input_dtype]

    if in_len > 0:
        l = np.arange(1, 100, 99 / in_len) / 100
        ret_weight[:in_len] = l
    if out_len > 0:
        r = np.arange(1, 100, 99 / out_len)[::-1] / 100
        ret_weight[-out_len:] = r

    if int_weight:
        if in_len + out_len > 0:
            return np.array(ret_weight * multi_base, dtype=dtype)
        return np.array(ret_weight, dtype=dtype)
    return ret_weight

class WeightGeneratorOp(BaseOp):
    """权重生成器：根据序列长度和渐入渐出参数生成逐帧权重序列。

    该 Op 接收一个序列输入（仅用于推断长度），根据 configs 中的
    渐入渐出参数一次性计算全部权重，然后逐帧流式输出。
    """

    INPUTS: dict[str, Any] = {
        "sequence": {
            "type": "sequence",
            "description": "输入序列（仅用于推断长度）",
        },
    }
    CONFIGS: dict[str, Any] = {
        "fin": {
            "type": "float",
            "description": "渐入比例",
            "default": 0,
        },
        "fout": {
            "type": "float",
            "description": "渐出比例",
            "default": 0,
        },
        "int_weight": {
            "type": "bool",
            "description": "是否使用整型权重加速",
            "default": False,
        },
        "input_dtype": {
            "type": "str",
            "description": "输入图像的 dtype 名称",
            "default": "uint8",
        },
    }
    OUTPUTS: dict[str, Any] = {
        "result": {
            "type": "sequence",
            "description": "逐帧权重序列",
        },
    }

    def __init__(self, name: str):
        super().__init__(name)

    async def _async_execute(self, configs: dict[str, Any]) -> None:
        length = self.length
        assert length is not None, (
            "WeightGeneratorOp requires sequence length information."
        )

        weights = generate_weight(
            length=length,
            fin=configs['fin'],
            fout=configs['fout'],
            int_weight=configs['int_weight'],
            input_dtype=np.dtype(configs.get('input_dtype', 'uint8')),
        )

        input_queue = self.inputs['sequence']

        async def drain_input():
            """消费输入队列中的数据以避免阻塞上游生产者。"""
            for _ in range(length):
                await input_queue.get()

        async def stream_weights():
            """逐帧将权重推送到输出队列。"""
            for i in range(length):
                await self._broadcast_result(weights[i])

        await asyncio.gather(drain_input(), stream_weights())

    async def _broadcast_result(self, result) -> None:
        tasks = []
        for queue in self.outputs['result']:
            tasks.append(queue.put(result))
        await asyncio.gather(*tasks)
