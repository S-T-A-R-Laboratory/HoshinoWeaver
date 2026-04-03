"""
WeightGeneratorOp：根据序列长度和渐入渐出参数，生成逐帧权重序列。

权重生成逻辑复制自 ezlib/trailstacker.py::generate_weight，
适配为异步 DAG Op。

注意：本 Op 只负责生成 [0, 1] 范围的浮点权重。
int_weight 的整型放缩由下游 Merger 根据图像的 dtype 信息自主完成。
"""
import asyncio
from typing import Any

import numpy as np
from loguru import logger

from .base import BaseOp


# ---------------------------------------------------------------------------
# 权重生成函数
# ---------------------------------------------------------------------------


def generate_weight(
    length: int,
    fin: float,
    fout: float,
) -> np.ndarray:
    """为渐入渐出星轨生成每张图像的浮点权重 [0, 1]。

    Args:
        length: 序列长度。
        fin: 渐入比例 (0-1)。
        fout: 渐出比例 (0-1)。

    Returns:
        np.ndarray: 权重数组，shape = (length,)，值域 [0, 1]，dtype=float32。
    """
    assert fin + fout <= 1, f"fin({fin}) + fout({fout}) > 1"
    in_len = int(length * fin)
    out_len = int(length * fout)
    ret_weight = np.ones((length,), dtype=np.float32)

    if in_len > 0:
        ret_weight[:in_len] = np.arange(1, 100, 99 / in_len) / 100
    if out_len > 0:
        ret_weight[-out_len:] = np.arange(1, 100, 99 / out_len)[::-1] / 100

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
