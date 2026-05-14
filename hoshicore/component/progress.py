"""
异步 DAG 进度追踪器。

使用 tqdm 实现多行并发进度条。通过 run_dag() 注入到各 Op 实例，
Op 在循环体内调用 tracker.update() 推进进度。

设计：DummyTracker 作为 BaseOp 的缺省 tracker，所有方法为空操作。
Op 内部无需任何 if 守卫，直接调用 self.tracker.update() 即可。
"""
from __future__ import annotations

from typing import Optional

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


