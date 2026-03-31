"""
异步文件缓存队列实现
支持将数据写入文件缓存，并在需要时读取。
"""

import base64
import json
import os
import pickle
import tempfile
from asyncio import Queue, Lock, to_thread, Event
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

SER_EXT_MAP = {"pickle": ".pkl", "json": ".json", "numpy": ".npy"}


class CancellationError(Exception):
    """上游节点取消异常"""
    pass


class CancellationToken:
    """取消令牌：携带异常信息"""
    def __init__(self, error: Exception, source_node: str):
        self.error = error
        self.source_node = source_node


class RichContextQueue(object):
    _SENTINEL = object()  # 正常结束信号

    def __init__(self, maxsize: int, **kwargs):
        self.queue = Queue(maxsize=maxsize)
        self.maxsize = maxsize
        self._put_lock = Lock()
        self.length: Optional[int] = None
        self._length_event = Event()  # 长度就绪事件

    async def put(self, item: Any) -> None:
        """将对象放入队列"""
        async with self._put_lock:
            await self.queue.put(item)

    async def get(self) -> Any:
        """从队列获取对象，自动处理信号"""
        item = await self.queue.get()

        # 检查取消令牌
        if isinstance(item, CancellationToken):
            raise CancellationError(
                f"Upstream node '{item.source_node}' failed: {item.error}"
            ) from item.error

        # 检查结束信号
        if item is RichContextQueue._SENTINEL:
            raise StopIteration("Stream ended normally")

        return item
    
    async def set_length(self, length: int):
        """由生产者设置序列长度"""
        async with self._put_lock:
            if self.length is not None and self.length != length:
                raise ValueError(f"Length mismatch: {self.length} vs {length}")
            self.length = length
            self._length_event.set()
    
    async def get_length(self) -> int:
        """消费者等待并获取序列长度"""
        await self._length_event.wait()
        assert self.length is not None, "Null length error (usually not expected to happen)"
        return self.length


class FileCacheQueue(RichContextQueue):
    """使用中间文件缓存的Queue。

    Args:
        object (_type_): _description_
    """

    def __init__(self,
                 maxsize: int,
                 serializer: str,
                 temp_path: Optional[Union[str, Path]] = None):
        super().__init__(maxsize)
        # 短base64作为文件前缀，避免文件名冲突
        self.prefix = base64.urlsafe_b64encode(os.urandom(6)).decode("ascii")
        self.serializer = serializer
        self._queue_counter = 0
        self.temp_path = Path(temp_path) if temp_path else Path(
            tempfile.gettempdir())
        os.makedirs(self.temp_path, exist_ok=True)

    def _save_to_file(self, item: Any, file_path: Path) -> None:
        if self.serializer == "pickle":
            with open(file_path, "wb") as f:
                pickle.dump(item, f)
        elif self.serializer == "json":
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(item, f)
        elif self.serializer == "numpy":
            np.save(file_path, item)
        else:
            raise ValueError(f"不支持的序列化器: {self.serializer}")

    def _load_from_file(self, file_path: Path) -> Any:
        if self.serializer == "pickle":
            with open(file_path, "rb") as f:
                return pickle.load(f)
        elif self.serializer == "json":
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        elif self.serializer == "numpy":
            return np.load(file_path, allow_pickle=True)
        else:
            raise ValueError(f"不支持的序列化器: {self.serializer}")

    async def put(self, item: Any) -> None:
        """将对象放入队列，使用文件缓存"""
        async with self._put_lock:
            filename = self._get_next_filename()
            file_path = self.temp_path / filename

            # 序列化并写入文件
            await to_thread(self._save_to_file, item, file_path)
            # 将文件路径放入队列
            await self.queue.put(str(file_path))

    async def get(self) -> Any:
        """从队列获取对象，读取文件缓存"""
        file_path = await self.queue.get()
        item: Any = None
        # 读取并反序列化对象
        try:
            item = await to_thread(self._load_from_file, file_path)
        except Exception as e:
            raise ValueError(f"读取文件缓存失败: {e}") from e
        finally:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except OSError:
                pass
        return item

    def _get_next_filename(self) -> str:
        """生成下一个缓存文件名"""
        self._queue_counter += 1
        ext = SER_EXT_MAP.get(self.serializer, ".dat")
        return f"{self.prefix}_{self._queue_counter:05d}{ext}"

    def clear(self) -> None:
        """清空队列，删除所有缓存文件"""
        while not self.queue.empty():
            file_path = self.queue.get_nowait()
            try:
                os.remove(file_path)
            except OSError:
                pass

        # 在清空队列后，再次检查删除所有缓存文件
        for file_path in self.temp_path.glob(f"{self.prefix}_*"):
            if file_path.is_file():
                try:
                    file_path.unlink()
                except OSError:
                    continue
        self._queue_counter = 0
