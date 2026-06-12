"""
DAG 执行器：负责启动和管理 DAG 节点的执行，统一错误传播与取消协议。

职责：
    - 为每个 Op 创建 asyncio.Task 并发运行
    - 任一节点真实异常后：cancel_all() 取消所有队列 + task.cancel() 打断等待
    - 收集失败/取消节点，按拓扑序选择根因
    - 抛出结构化的 DAGExecutionError 供上层展示
"""
import asyncio
from typing import Any

from loguru import logger

from ..component.queue import CancellationError, CancellationToken
from ..ops.base import BaseOp


class DAGExecutionError(Exception):
    """DAG 执行失败，携带结构化诊断信息。"""

    def __init__(
        self,
        root_cause: Exception,
        root_node: str,
        failed_nodes: list[tuple[str, Exception]],
        cancelled_nodes: list[str],
    ):
        self.root_cause = root_cause
        self.root_node = root_node
        self.failed_nodes = failed_nodes
        self.cancelled_nodes = cancelled_nodes
        super().__init__(f"Node '{root_node}' failed: {root_cause!r}")
        self.__cause__ = root_cause


class DAGExecutor:
    """DAG 执行器：统一取消传播与根因追踪。"""

    def __init__(self, nodes: list[BaseOp]):
        self.nodes = nodes
        self.cancel_event = asyncio.Event()
        self.failed_nodes: list[tuple[str, Exception]] = []
        self.cancelled_nodes: list[str] = []
        self._node_tasks: dict[str, asyncio.Task] = {}

    async def execute(self) -> None:
        """执行所有节点，失败时抛出 DAGExecutionError。"""
        for node in self.nodes:
            self._node_tasks[node.name] = asyncio.create_task(
                self._run_node(node), name=node.name)

        results = await asyncio.gather(
            *self._node_tasks.values(), return_exceptions=True)

        # task.cancel() 可能在 _run_node 开始前就取消了 task，
        # 此时 CancelledError 作为 gather 结果返回，不经过 _run_node 的 except 分支。
        for node, result in zip(self.nodes, results):
            if isinstance(result, asyncio.CancelledError):
                if node.name not in self.cancelled_nodes:
                    self.cancelled_nodes.append(node.name)

        if self.failed_nodes:
            root_node, root_cause = self._select_root_cause()
            logger.error(
                f"DAG execution failed: root cause at '{root_node}': "
                f"{root_cause!r}")
            if len(self.failed_nodes) > 1:
                logger.error(
                    f"Additional failures: "
                    f"{[(n, repr(e)) for n, e in self.failed_nodes[1:]]}")
            raise DAGExecutionError(
                root_cause, root_node,
                self.failed_nodes, self.cancelled_nodes)

        logger.info("DAG execution completed successfully")

    async def _run_node(self, node: BaseOp) -> None:
        """运行单个节点，分类处理异常。"""
        try:
            await node.execute()
        except CancellationError:
            self.cancelled_nodes.append(node.name)
            logger.debug(f"{node.name}: cancelled (CancellationError)")
        except asyncio.CancelledError:
            self.cancelled_nodes.append(node.name)
            logger.debug(f"{node.name}: cancelled (task.cancel)")
        except Exception as e:
            self.failed_nodes.append((node.name, e))
            logger.error(f"{node.name}: execution failed - {e!r}")
            self.cancel_all()
            for name, task in self._node_tasks.items():
                if name != node.name and not task.done():
                    task.cancel()

    def cancel_all(self) -> None:
        """取消所有节点的所有输入/输出/配置队列。幂等。

        用于内部失败后和外部取消时。
        """
        if self.cancel_event.is_set():
            return
        self.cancel_event.set()
        token = CancellationToken(
            CancellationError("DAG cancelled"), "__runtime__")
        for node in self.nodes:
            for port_queues in node.outputs.values():
                for queue in port_queues:
                    queue.force_cancel(token)
            for queue in node.inputs.values():
                queue.force_cancel(token)
            for queue in node.config.values():
                queue.force_cancel(token)

    def _select_root_cause(self) -> tuple[str, Exception]:
        """从 failed_nodes 中选择拓扑序最上游的作为根因。"""
        topo_order = {node.name: i for i, node in enumerate(self.nodes)}
        self.failed_nodes.sort(key=lambda x: topo_order.get(x[0], 999))
        return self.failed_nodes[0]
