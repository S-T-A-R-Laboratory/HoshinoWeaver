from __future__ import annotations

"""
帧缓冲：支持按索引随机读取的帧存储，用于 SigmaClipping 等多 pass 算法。

提供三种实现：
    - DiskFrameBuffer:    将解码后的帧写入临时 .npz 文件，读取快但占磁盘空间。
    - MemoryFrameBuffer:  将帧直接保存在 RAM 中，零 I/O 但占内存。
    - SourceReplayBuffer: 保留原始文件路径，每次访问重新解码，零临时文件但每 pass 有 decode 开销。
"""

import asyncio
import base64
import os
import tempfile
from pathlib import Path
from typing import AsyncIterator, Optional, Union

import numpy as np
from loguru import logger

from .image_io import load_img


class BaseFrameBuffer:
    """帧缓冲基类：定义多 pass 算法消费帧数据的统一协议。

    支持引用计数：多个消费者可通过 acquire() 共享同一个 buffer，
    每个消费者完成后调用 cleanup() 释放引用，最后一个释放时
    触发 _do_cleanup() 执行真正的资源回收。

    子类须实现 append / __getitem__ / _do_cleanup。
    """

    def __init__(self):
        self._ref_count = 0

    def acquire(self):
        """增加引用计数。每个下游消费者对应一次 acquire。"""
        self._ref_count += 1

    def append(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple[np.ndarray, Optional[Union[float, np.ndarray]]]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def cleanup(self) -> None:
        """释放一个引用。引用归零时执行 _do_cleanup()。"""
        self._ref_count -= 1
        if self._ref_count <= 0:
            self._do_cleanup()

    def _do_cleanup(self) -> None:
        raise NotImplementedError

    async def iter_prefetch(
        self,
        start: int = 0,
        stop: Optional[int] = None,
    ) -> AsyncIterator[tuple[np.ndarray, Optional[Union[float, np.ndarray]]]]:
        """异步预读迭代器：将下一帧的 IO 与当前帧的计算重叠。

        在事件循环中以 asyncio.to_thread 提前发起下一帧的读取，
        使 IO 等待与调用方的 CPU 计算（await _run_cpu(...)）并行执行。

        对 MemoryFrameBuffer 无额外开销（to_thread 调用极快）；
        对 DiskFrameBuffer / SourceReplayBuffer 可有效隐藏 IO 延迟。

        用法::

            async for raw, weight in frame_buffer.iter_prefetch():
                await self._run_cpu(merger.merge, raw, weight)

        Args:
            start: 起始索引（含），默认 0。
            stop:  终止索引（不含），默认 len(self)。
        """
        if stop is None:
            stop = len(self)
        if start >= stop:
            return

        def _load(idx: int):
            return self[idx]

        prefetch_task: asyncio.Task = asyncio.create_task(
            asyncio.to_thread(_load, start)
        )
        for idx in range(start, stop):
            next_idx = idx + 1
            if next_idx < stop:
                next_task: asyncio.Task = asyncio.create_task(
                    asyncio.to_thread(_load, next_idx)
                )
            else:
                next_task = None

            yield await prefetch_task

            if next_task is not None:
                prefetch_task = next_task


class DiskFrameBuffer(BaseFrameBuffer):
    """磁盘帧缓冲：将帧写入 .npy 文件，读取时通过 mmap 零拷贝访问。

    权重分流：ndarray 权重存为独立 .npy 文件；标量/None 权重直接内存持有。

    用法：
        buffer = DiskFrameBuffer()
        buffer.append(frame1, weight1)
        buffer.append(frame2)

        frame, weight = buffer[0]   # mmap 读取
        buffer.cleanup()             # 删除所有临时文件
    """

    def __init__(self, temp_path: Optional[Union[str, Path]] = None):
        super().__init__()
        self.temp_path = Path(temp_path) if temp_path else Path(
            tempfile.gettempdir())
        os.makedirs(self.temp_path, exist_ok=True)
        self.prefix = base64.urlsafe_b64encode(os.urandom(6)).decode("ascii")
        self._count = 0
        self._frame_paths: list[Path] = []
        self._weight_paths: list[Optional[Path]] = []
        self._scalar_weights: list[Optional[float]] = []
        self._cleaned = False

    def append(self, frame: np.ndarray,
               weight: Optional[Union[float, np.ndarray]] = None) -> None:
        """保存一帧（及可选权重）到磁盘。

        Args:
            frame: 图像数据 (np.ndarray)。
            weight: 可选权重，标量或与 frame 同形状的 ndarray。
        """
        frame_path = self.temp_path / f"{self.prefix}_{self._count:05d}.npy"
        np.save(frame_path, frame)
        self._frame_paths.append(frame_path)

        if isinstance(weight, np.ndarray):
            weight_path = self.temp_path / f"{self.prefix}_{self._count:05d}_w.npy"
            np.save(weight_path, weight)
            self._weight_paths.append(weight_path)
            self._scalar_weights.append(None)
        else:
            self._weight_paths.append(None)
            self._scalar_weights.append(float(weight) if weight is not None else None)

        self._count += 1

    def __getitem__(self, idx: int) -> tuple[np.ndarray, Optional[Union[float, np.ndarray]]]:
        """按索引 mmap 读取帧和权重。

        Args:
            idx: 帧索引。

        Returns:
            (frame, weight) 元组。frame 为 mmap 只读视图。
        """
        if idx < 0 or idx >= self._count:
            raise IndexError(
                f"DiskFrameBuffer index {idx} out of range [0, {self._count})"
            )
        frame = np.load(self._frame_paths[idx], mmap_mode='r')
        if self._weight_paths[idx] is not None:
            weight = np.load(self._weight_paths[idx], mmap_mode='r')
        else:
            weight = self._scalar_weights[idx]
        return frame, weight

    def __len__(self) -> int:
        return self._count

    def _do_cleanup(self) -> None:
        """删除所有临时缓冲文件。"""
        if self._cleaned:
            return
        for p in self._frame_paths:
            try:
                p.unlink()
            except OSError:
                pass
        for p in self._weight_paths:
            if p is not None:
                try:
                    p.unlink()
                except OSError:
                    pass
        self._frame_paths.clear()
        self._weight_paths.clear()
        self._scalar_weights.clear()
        self._count = 0
        self._cleaned = True
        logger.debug(
            f"DiskFrameBuffer cleaned up (prefix={self.prefix}).")

    def __del__(self):
        """安全网：防止异常中断（如用户 Ctrl-C）导致临时文件泄漏。"""
        if not self._cleaned and self._frame_paths:
            self._do_cleanup()


class MemoryFrameBuffer(BaseFrameBuffer):
    """内存帧缓冲：将帧直接保存在 RAM 中，按索引随机读取。

    零 I/O 开销，适用于帧数少或帧尺寸小的场景。
    """

    def __init__(self):
        super().__init__()
        self._frames: list[tuple[np.ndarray, Optional[Union[float, np.ndarray]]]] = []
        self._cleaned = False

    def append(self, frame: np.ndarray,
               weight: Optional[Union[float, np.ndarray]] = None) -> None:
        self._frames.append((frame, weight))

    def __getitem__(self, idx: int) -> tuple[np.ndarray, Optional[Union[float, np.ndarray]]]:
        if idx < 0 or idx >= len(self._frames):
            raise IndexError(
                f"MemoryFrameBuffer index {idx} out of range "
                f"[0, {len(self._frames)})")
        return self._frames[idx]

    def __len__(self) -> int:
        return len(self._frames)

    def _do_cleanup(self) -> None:
        if self._cleaned:
            return
        self._frames.clear()
        self._cleaned = True
        logger.debug("MemoryFrameBuffer cleaned up.")


class SourceReplayBuffer(BaseFrameBuffer):
    """源文件重放缓冲：保留原始文件路径，每次访问重新解码。

    省磁盘空间（零临时文件），代价是每 pass 都有一次完整 decode 开销。
    适用于硬盘空间受限、图片文件本身已在磁盘上的场景。

    用法：
        buffer = SourceReplayBuffer()
        buffer.append("/path/to/img1.tif", weight=1.0)
        buffer.append("/path/to/img2.tif")

        frame, weight = buffer[0]   # 从原始文件重新解码
        buffer.cleanup()
    """

    def __init__(self):
        super().__init__()
        self._entries: list[tuple[str, Optional[Union[float, np.ndarray]]]] = []
        self._cleaned = False

    def append(self, source_path: str,
               weight: Optional[Union[float, np.ndarray]] = None) -> None:
        """记录一个帧的原始文件路径和可选权重。

        Args:
            source_path: 原始图片文件路径。
            weight: 可选权重，标量或 ndarray。
        """
        self._entries.append((source_path, weight))

    def __getitem__(self, idx: int) -> tuple[np.ndarray, Optional[Union[float, np.ndarray]]]:
        """按索引从原始文件重新解码帧。

        Args:
            idx: 帧索引。

        Returns:
            (frame, weight) 元组。weight 为 None 表示该帧无权重。

        Raises:
            IndexError: 索引越界。
            IOError: 原始文件解码失败。
        """
        if idx < 0 or idx >= len(self._entries):
            raise IndexError(
                f"SourceReplayBuffer index {idx} out of range "
                f"[0, {len(self._entries)})")
        path, weight = self._entries[idx]
        frame = load_img(path)
        if frame is None:
            raise IOError(f"Failed to decode source file: {path}")
        return frame, weight

    def __len__(self) -> int:
        return len(self._entries)

    def _do_cleanup(self) -> None:
        """清理内部状态（无临时文件需要删除）。"""
        if self._cleaned:
            return
        self._entries.clear()
        self._cleaned = True
        logger.debug("SourceReplayBuffer cleaned up.")
