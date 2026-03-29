"""
DAG执行器：负责启动和管理DAG节点的执行
"""
import asyncio
from typing import Any
from loguru import logger

from ..ops.base import BaseOp
from ..component.queue import CancellationError


class DAGExecutor:
    """DAG执行器：全局取消机制"""

    def __init__(self, nodes: list[BaseOp]):
        self.nodes = nodes
        self.cancel_event = asyncio.Event()
        self.failed_nodes: list[tuple[str, Exception]] = []

    async def execute(self) -> None:
        """执行DAG"""
        tasks = [self._run_node(node) for node in self.nodes]

        try:
            await asyncio.gather(*tasks, return_exceptions=False)
            logger.info("DAG execution completed successfully")
        except Exception as e:
            # 任一节点失败，触发全局取消
            self.cancel_event.set()
            logger.error(f"DAG execution failed: {e}")

            # 等待所有任务结束
            await asyncio.gather(*tasks, return_exceptions=True)

            # 报告失败节点
            if self.failed_nodes:
                logger.error(f"Failed nodes: {[name for name, _ in self.failed_nodes]}")
            raise

    async def _run_node(self, node: BaseOp) -> None:
        """运行单个节点"""
        try:
            await node.execute()
        except CancellationError:
            # 上游取消，正常传播
            logger.info(f"{node.name}: cancelled by upstream")
        except Exception as e:
            self.failed_nodes.append((node.name, e))
            logger.error(f"{node.name}: execution failed - {e}")
            raise
