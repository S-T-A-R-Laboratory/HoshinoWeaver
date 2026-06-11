"""DAG 终止协议测试：错误传播、取消、队列 force_cancel。"""
import asyncio

import pytest

from hoshicore.component.queue import (
    CancellationError,
    CancellationToken,
    RichContextQueue,
    FileCacheQueue,
)
from hoshicore.engine.executor import DAGExecutionError, DAGExecutor
from hoshicore.engine.registry import register_op
from hoshicore.ops.base import BaseOp, ParallelBaseOp

pytestmark = pytest.mark.asyncio


# ────────────────────────────────────────────────────────────────
# Test Ops
# ────────────────────────────────────────────────────────────────

class PassThroughOp(BaseOp):
    """接收 sequence 输入，原样转发到输出。"""
    INPUTS = {"src": {"type": "sequence", "required": True}}
    OUTPUTS = {"out": {"type": "sequence"}}
    CONFIGS = {}

    async def _async_execute(self, configs):
        for _ in self._input_range():
            data = self._async_convert_inputs()
            item = await data["src"]
            await self._broadcast_outputs({"out": item})


class FailingOp(BaseOp):
    """在第 N 帧时抛出 ValueError。"""
    INPUTS = {"src": {"type": "sequence", "required": True}}
    OUTPUTS = {"out": {"type": "sequence"}}
    CONFIGS = {"fail_at": {"type": "int", "default": 0}}

    async def _async_execute(self, configs):
        fail_at = configs.get("fail_at", 0)
        for i in self._input_range():
            data = self._async_convert_inputs()
            item = await data["src"]
            if i == fail_at:
                raise ValueError(f"Intentional failure at frame {i}")
            await self._broadcast_outputs({"out": item})


class SlowOp(BaseOp):
    """无限等待输入（模拟卡在 queue.get 的节点）。"""
    INPUTS = {"src": {"type": "sequence", "required": True}}
    OUTPUTS = {"out": {"type": "sequence"}}
    CONFIGS = {}

    async def _async_execute(self, configs):
        for _ in self._input_range():
            data = self._async_convert_inputs()
            await data["src"]


class ConfigOnlyOp(BaseOp):
    """只需要 config，不需要 sequence input。"""
    INPUTS = {}
    OUTPUTS = {"out": {"type": "sequence"}}
    CONFIGS = {"value": {"type": "int", "default": 42}}

    async def _async_execute(self, configs):
        await self._broadcast_outputs({"out": configs["value"]})


# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────

def wire_linear(ops: list[BaseOp], length: int):
    """将 ops 串联：ops[i].outputs["out"] → ops[i+1].inputs["src"]。"""
    for i in range(len(ops) - 1):
        queue = ops[i + 1].inputs["src"]
        ops[i].outputs["out"].append(queue)


async def feed_sequence(queue: RichContextQueue, items: list):
    """向队列推送序列 + 设置长度。"""
    await queue.set_length(len(items))
    for item in items:
        await queue.put(item)


# ────────────────────────────────────────────────────────────────
# Tests: 队列 force_cancel
# ────────────────────────────────────────────────────────────────

class TestQueueForceCancel:
    async def test_force_cancel_unblocks_get_length(self):
        q = RichContextQueue(maxsize=1)
        token = CancellationToken(ValueError("test"), "node_a")

        async def wait_length():
            return await q.get_length()

        task = asyncio.create_task(wait_length())
        await asyncio.sleep(0.01)
        assert not task.done()

        q.force_cancel(token)
        with pytest.raises(CancellationError):
            await task

    async def test_force_cancel_unblocks_get(self):
        q = RichContextQueue(maxsize=1)
        token = CancellationToken(ValueError("test"), "node_a")

        task = asyncio.create_task(q.get())
        await asyncio.sleep(0.01)
        assert not task.done()

        q.force_cancel(token)
        with pytest.raises(CancellationError):
            await task

    async def test_force_cancel_full_queue_discards_element(self):
        q = RichContextQueue(maxsize=1)
        await q.put("existing_data")
        token = CancellationToken(ValueError("test"), "node_a")

        q.force_cancel(token)
        # 后续 get 应立即抛出
        with pytest.raises(CancellationError):
            await q.get()

    async def test_force_cancel_is_idempotent(self):
        q = RichContextQueue(maxsize=1)
        token1 = CancellationToken(ValueError("first"), "node_a")
        token2 = CancellationToken(ValueError("second"), "node_b")

        q.force_cancel(token1)
        q.force_cancel(token2)
        assert q._cancelled_token is token1

    async def test_put_after_cancel_raises(self):
        q = RichContextQueue(maxsize=1)
        token = CancellationToken(ValueError("test"), "node_a")
        q.force_cancel(token)

        with pytest.raises(CancellationError):
            await q.put("data")

    async def test_file_cache_queue_cleanup_on_cancel(self, tmp_path):
        q = FileCacheQueue(maxsize=1, serializer="pickle", temp_path=tmp_path)
        await q.put({"data": 123})
        # 队列满，里面有一个文件路径
        token = CancellationToken(ValueError("test"), "node_a")
        q.force_cancel(token)
        # 被弹出的文件路径对应的文件应已删除
        remaining = list(tmp_path.glob(f"{q.prefix}_*"))
        assert remaining == []


# ────────────────────────────────────────────────────────────────
# Tests: DAGExecutor 错误传播
# ────────────────────────────────────────────────────────────────

class TestDAGExecutorFailure:
    async def test_single_node_failure_reports_root_cause(self):
        """单节点失败 → DAGExecutionError.root_node 正确。"""
        op = FailingOp(name="failer")
        op.inputs["src"] = RichContextQueue(maxsize=1)
        op.outputs["out"] = []
        op.config["fail_at"] = RichContextQueue(maxsize=1)

        executor = DAGExecutor([op])

        async def setup():
            await op.config["fail_at"].put(0)
            await feed_sequence(op.inputs["src"], ["frame0", "frame1"])

        asyncio.create_task(setup())

        with pytest.raises(DAGExecutionError) as exc_info:
            await executor.execute()

        err = exc_info.value
        assert err.root_node == "failer"
        assert isinstance(err.root_cause, ValueError)
        assert "frame 0" in str(err.root_cause)

    async def test_upstream_failure_cancels_downstream(self):
        """上游失败 → 下游进入 cancelled_nodes。"""
        upstream = FailingOp(name="upstream")
        downstream = PassThroughOp(name="downstream")

        upstream.inputs["src"] = RichContextQueue(maxsize=1)
        upstream.config["fail_at"] = RichContextQueue(maxsize=1)
        downstream.inputs["src"] = RichContextQueue(maxsize=1)
        upstream.outputs["out"] = [downstream.inputs["src"]]
        downstream.outputs["out"] = []

        executor = DAGExecutor([upstream, downstream])

        async def setup():
            await upstream.config["fail_at"].put(0)
            await feed_sequence(upstream.inputs["src"], ["a", "b"])

        asyncio.create_task(setup())

        with pytest.raises(DAGExecutionError) as exc_info:
            await executor.execute()

        err = exc_info.value
        assert err.root_node == "upstream"
        assert "downstream" in err.cancelled_nodes
        assert not any(n == "downstream" for n, _ in err.failed_nodes)

    async def test_topo_order_selects_root_cause(self):
        """两个独立分支同时失败 → 拓扑序更上游的作为 root_node。"""
        op_a = FailingOp(name="branch_a")
        op_b = FailingOp(name="branch_b")

        for op in [op_a, op_b]:
            op.inputs["src"] = RichContextQueue(maxsize=1)
            op.config["fail_at"] = RichContextQueue(maxsize=1)
            op.outputs["out"] = []

        # op_a 在拓扑序中靠前
        executor = DAGExecutor([op_a, op_b])

        async def setup():
            await op_a.config["fail_at"].put(0)
            await op_b.config["fail_at"].put(0)
            await feed_sequence(op_a.inputs["src"], ["x"])
            await feed_sequence(op_b.inputs["src"], ["y"])

        asyncio.create_task(setup())

        with pytest.raises(DAGExecutionError) as exc_info:
            await executor.execute()

        err = exc_info.value
        assert err.root_node == "branch_a"

    async def test_normal_execution_succeeds(self):
        """正常执行通过，无异常。结果正确收集。"""
        op = PassThroughOp(name="pass")
        op.inputs["src"] = RichContextQueue(maxsize=1)
        collector = RichContextQueue(maxsize=1)
        op.outputs["out"] = [collector]

        executor = DAGExecutor([op])
        collected = []

        async def setup():
            await feed_sequence(op.inputs["src"], ["hello"])

        async def consume():
            """模拟 run_dag 中的 _collect_outputs：并发消费输出。"""
            try:
                while True:
                    item = await collector.get()
                    collected.append(item)
            except Exception:
                pass

        asyncio.create_task(setup())
        consumer_task = asyncio.create_task(consume())
        await executor.execute()
        consumer_task.cancel()
        try:
            await consumer_task
        except asyncio.CancelledError:
            pass

        assert collected == ["hello"]


# ────────────────────────────────────────────────────────────────
# Tests: 外部取消
# ────────────────────────────────────────────────────────────────

class TestDAGExecutorExternalCancel:
    async def test_external_cancel_stops_all_nodes(self):
        """外部 cancel_event → 所有节点退出。"""
        op = SlowOp(name="slow")
        op.inputs["src"] = RichContextQueue(maxsize=1)
        op.outputs["out"] = []

        cancel_event = asyncio.Event()
        executor = DAGExecutor([op])
        executor.cancel_event = cancel_event

        # 设置长度让 pre_execute 通过
        await op.inputs["src"].set_length(100)

        async def trigger_cancel():
            await asyncio.sleep(0.05)
            cancel_event.set()
            executor.cancel_all()
            for task in executor._node_tasks.values():
                if not task.done():
                    task.cancel()

        asyncio.create_task(trigger_cancel())
        # executor 应正常结束（节点被取消不算失败）
        await executor.execute()
        assert "slow" in executor.cancelled_nodes


# ────────────────────────────────────────────────────────────────
# Tests: get_length 卡死场景
# ────────────────────────────────────────────────────────────────

class TestPreExecuteCancel:
    async def test_cancel_unblocks_pre_execute_get_length(self):
        """pre_execute 卡在 get_length 时，force_cancel 能唤醒。"""
        op = PassThroughOp(name="waiting")
        op.inputs["src"] = RichContextQueue(maxsize=1)
        op.outputs["out"] = []

        async def run_op():
            await op.execute()

        task = asyncio.create_task(run_op())
        await asyncio.sleep(0.02)
        assert not task.done()

        # force_cancel 输入队列
        token = CancellationToken(ValueError("upstream died"), "failer")
        op.inputs["src"].force_cancel(token)

        with pytest.raises((CancellationError, asyncio.CancelledError)):
            await task
