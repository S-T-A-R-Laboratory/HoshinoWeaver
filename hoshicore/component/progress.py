"""
异步 DAG 进度追踪器。

使用 tqdm 实现多行并发进度条。通过 run_dag() 注入到各 Op 实例，
Op 在循环体内调用 tracker.update() 推进进度。

设计：DummyTracker 作为 BaseOp 的缺省 tracker，所有方法为空操作。
Op 内部无需任何 if 守卫，直接调用 self.tracker.update() 即可。

多进程支持：ProxyTracker 在 worker 进程中将 tracker 操作序列化为
(method_name, *args) 元组发送到主进程的 multiprocessing.Queue，
主进程的 TrackerEventConsumer 消费事件并调用实际 tracker。
"""
from __future__ import annotations

import multiprocessing as mp
import os
from typing import Optional

import psutil
from loguru import logger
from tqdm import tqdm


class DummyTracker:
    """空操作追踪器，作为 BaseOp.tracker 的缺省值。

    所有方法均为 no-op。
    """

    def create_bar(
        self,
        name: str,
        total: int,
        desc: Optional[str] = None,
        unit: str = "imgs",
    ) -> None:
        pass

    def update(self, name: str, n: int = 1) -> None:
        pass

    def set_description(self, name: str, desc: str) -> None:
        pass

    def reset_bar(
        self,
        name: str,
        total: int,
        desc: Optional[str] = None,
    ) -> None:
        pass

    def close_bar(self, name: str) -> None:
        pass

    def close_all(self) -> None:
        pass


class ProgressTracker(DummyTracker):
    """DAG 全局进度追踪器，管理多个 tqdm 进度条。"""

    def __init__(self) -> None:
        self._bars: dict[str, tqdm] = {}
        self._position: int = 0  # 下一个 bar 的行号

    def create_bar(
        self,
        name: str,
        total: int,
        desc: Optional[str] = None,
        unit: str = "imgs",
    ) -> None:
        """为一个 Op 创建进度条。"""
        bar = tqdm(
            total=total,
            desc=desc or name,
            unit=unit,
            position=self._position,
            dynamic_ncols=True,
            leave=True,
        )
        self._bars[name] = bar
        self._position += 1

    def update(self, name: str, n: int = 1) -> None:
        """推进指定 Op 的进度。"""
        bar = self._bars.get(name)
        if bar is not None:
            bar.update(n)

    def set_description(self, name: str, desc: str) -> None:
        """更新进度条描述文字（用于阶段切换）。"""
        bar = self._bars.get(name)
        if bar is not None:
            bar.set_description(desc)

    def reset_bar(
        self,
        name: str,
        total: int,
        desc: Optional[str] = None,
    ) -> None:
        """重置进度条（用于 SigmaClipping 的阶段切换）。"""
        bar = self._bars.get(name)
        if bar is not None:
            bar.reset(total=total)
            if desc:
                bar.set_description(desc)

    def close_bar(self, name: str) -> None:
        """关闭单个进度条。"""
        bar = self._bars.pop(name, None)
        if bar is not None:
            bar.close()

    def close_all(self) -> None:
        """关闭所有进度条。"""
        for bar in self._bars.values():
            bar.close()
        self._bars.clear()
        self._position = 0


class ProxyTracker(DummyTracker):
    """子进程端的 tracker 代理，将操作序列化后发到主进程。

    所有方法将 (method_name, *args) 元组放入 multiprocessing.Queue，
    由主进程的 TrackerEventConsumer 消费并调用实际 ProgressTracker。
    """

    _STOP = ("__stop__",)

    def __init__(self, event_queue: mp.Queue) -> None:
        self._q = event_queue

    def create_bar(
        self,
        name: str,
        total: int,
        desc: Optional[str] = None,
        unit: str = "imgs",
    ) -> None:
        self._q.put(("create_bar", name, total, desc, unit))

    def update(self, name: str, n: int = 1) -> None:
        self._q.put(("update", name, n))

    def set_description(self, name: str, desc: str) -> None:
        self._q.put(("set_description", name, desc))

    def reset_bar(
        self,
        name: str,
        total: int,
        desc: Optional[str] = None,
    ) -> None:
        self._q.put(("reset_bar", name, total, desc))

    def close_bar(self, name: str) -> None:
        self._q.put(("close_bar", name))

    def close_all(self) -> None:
        self._q.put(("close_all",))

    def report_mem(self, tag: str) -> None:
        """上报当前进程 RSS（MB）到主进程日志。"""
        rss = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        self._q.put(("__mem__", os.getpid(), tag, rss))


class TrackerEventConsumer:
    """主进程端的 tracker 事件消费者。

    从 multiprocessing.Queue 读取 ProxyTracker 发送的事件，
    分发给实际的 ProgressTracker 执行。

    用法::

        tracker = ProgressTracker()
        event_queue = mp.Queue()
        consumer = TrackerEventConsumer(tracker, event_queue)

        # 在主进程的 asyncio loop 中启动
        task = asyncio.create_task(consumer.run())

        # 停止：发送 stop 信号
        consumer.stop()
        await task
    """

    def __init__(self, tracker: ProgressTracker, event_queue: mp.Queue) -> None:
        self._tracker = tracker
        self._q = event_queue

    async def run(self) -> None:
        """持续消费事件直到收到 stop 信号。"""
        import asyncio

        loop = asyncio.get_running_loop()
        while True:
            msg = await loop.run_in_executor(None, self._q.get)
            method_name = msg[0]
            if method_name == "__stop__":
                break
            if method_name == "__mem__":
                _, pid, tag, rss = msg
                logger.debug(f"[MEM][worker pid={pid}] {tag}: RSS={rss:.0f} MB")
                continue
            args = msg[1:]
            fn = getattr(self._tracker, method_name, None)
            if fn is not None:
                fn(*args)

    def stop(self) -> None:
        """发送 stop 信号，使 run() 退出。"""
        self._q.put(ProxyTracker._STOP)
