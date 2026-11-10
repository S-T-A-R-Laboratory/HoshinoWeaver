import asyncio

import pytest

from hoshicore.component.queue import RichContextQueue, StreamExhausted
from hoshicore.ops.base import ParallelBaseOp


class _SlowConcurrentMap(ParallelBaseOp):
    INPUTS = {"src": {"type": "sequence", "required": True}}
    CONFIGS = {}
    OUTPUTS = {"result": {"type": "sequence"}}
    CONCURRENCY = 4

    async def _async_execute_single(self, data, configs):
        item = await data["src"]
        await asyncio.sleep(0.02)
        return {"result": item}


@pytest.mark.asyncio
async def test_concurrent_map_does_not_cancel_slow_tail(monkeypatch):
    """Normal completion must not apply the former five-second tail timeout."""
    op = _SlowConcurrentMap(name="slow_map")
    sink = RichContextQueue(maxsize=1)
    op.outputs["result"].append(sink)

    original_wait_for = asyncio.wait_for

    async def reject_tail_timeout(awaitable, timeout):
        if timeout == 5.0:
            if hasattr(awaitable, "close"):
                awaitable.close()
            raise asyncio.TimeoutError
        return await original_wait_for(awaitable, timeout)

    # The old implementation used wait_for(..., 5.0) in normal cleanup and
    # consequently cancelled the unfinished tail. The corrected path does not.
    monkeypatch.setattr("hoshicore.ops.base.asyncio.wait_for", reject_tail_timeout)

    async def feed():
        await op.inputs["src"].set_length(9)
        for item in range(9):
            await op.inputs["src"].put(item)

    received = []

    async def drain():
        try:
            while True:
                received.append(await sink.get())
        except StreamExhausted:
            pass

    await asyncio.gather(feed(), op.execute(), drain())
    assert received == list(range(9))


@pytest.mark.asyncio
@pytest.mark.parametrize('failure_at', ['decode', 'emit'])
async def test_concurrent_failure_escapes_when_producer_waits_for_full_window(failure_at):
    class FailingMap(_SlowConcurrentMap):
        async def _async_execute_single(self, data, configs):
            item = await data['src']
            if failure_at == 'decode' and item == 0:
                raise RuntimeError('first decode failed')
            return {'result': item}

        async def _broadcast_outputs(self, result):
            if failure_at == 'emit':
                raise RuntimeError('first emit failed')
            await super()._broadcast_outputs(result)

    op = FailingMap('failure')
    async def feed():
        await op.inputs['src'].set_length(6)
        for item in range(6):
            await op.inputs['src'].put(item)

    tasks = [asyncio.create_task(feed()), asyncio.create_task(op.execute())]
    try:
        with pytest.raises(RuntimeError, match=f'first {failure_at} failed'):
            await asyncio.wait_for(asyncio.gather(*tasks), 1)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
