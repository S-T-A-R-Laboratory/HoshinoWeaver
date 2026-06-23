"""进度上报机制测试：REPORTS_PROGRESS 标志、display_name、串行上报。"""
import asyncio

import pytest

from hoshicore.component.progress import DummyTracker
from hoshicore.component.queue import (RichContextQueue,
                                       StreamExhausted)
from hoshicore.ops.base import BaseOp, ChunkIteratorBaseOp, ParallelBaseOp


def test_baseop_reports_progress_defaults_false():
    op = BaseOp(name="n1")
    assert op.REPORTS_PROGRESS is False


def test_parallel_baseop_reports_progress_defaults_false():
    assert ParallelBaseOp.REPORTS_PROGRESS is False


def test_chunk_iterator_reports_progress_true():
    assert ChunkIteratorBaseOp.REPORTS_PROGRESS is True


def test_display_name_defaults_to_name():
    op = BaseOp(name="core_algo.reduce")
    assert op.display_name == "core_algo.reduce"


def test_display_name_can_be_overridden():
    op = BaseOp(name="parent.child")
    op.display_name = "Human Readable Label"
    assert op.display_name == "Human Readable Label"
    assert op.name == "parent.child"


def test_reports_progress_not_inherited_by_subclass_by_default():
    class MyOp(BaseOp):
        pass
    assert MyOp.REPORTS_PROGRESS is False


def test_reports_progress_can_be_set_on_subclass():
    class MyLimitingOp(BaseOp):
        REPORTS_PROGRESS = True
    assert MyLimitingOp.REPORTS_PROGRESS is True


class SpyTracker(DummyTracker):
    def __init__(self):
        self.created: list[tuple] = []
        self.updated: dict[str, int] = {}
        self.closed: list[str] = []

    def create_bar(self, name, total, desc=None, unit="imgs"):
        self.created.append((name, desc, total))

    def update(self, name, n=1):
        self.updated[name] = self.updated.get(name, 0) + n

    def close_bar(self, name):
        self.closed.append(name)


class _SerialDouble(ParallelBaseOp):
    """串行 Map：把每个输入原样转发。"""
    INPUTS = {"src": {"type": "sequence", "required": True}}
    OUTPUTS = {"result": {"type": "sequence"}}

    async def _async_execute_single(self, data, configs):
        item = await data["src"]
        return {"result": item}


@pytest.mark.asyncio
async def test_serial_map_reports_with_display_name():
    op = _SerialDouble(name="doubler")
    op.display_name = "翻倍器"
    spy = SpyTracker()
    op.tracker = spy
    src = op.inputs["src"]
    sink = RichContextQueue(maxsize=1)
    op.outputs["result"].append(sink)

    async def feeder():
        await src.set_length(3)
        for v in (10, 20, 30):
            await src.put(v)

    async def drain():
        while True:
            try:
                await sink.get()
            except StreamExhausted:
                break

    await asyncio.gather(feeder(), op.execute(), drain())

    assert spy.created and spy.created[0][0] == "doubler"   # key 仍为 name
    assert spy.created[0][1] == "翻倍器"                      # desc 用 display_name
    assert spy.updated.get("doubler") == 3
