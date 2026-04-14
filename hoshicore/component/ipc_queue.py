"""
跨进程异步队列：通过 SharedMemory + Pipe 实现高效的进程间数据传输。

设计目标：
    - 继承 BaseQueue，接口与 RichContextQueue 完全一致
    - np.ndarray 通过 SharedMemory 零拷贝传输（跨平台：Windows/Linux/macOS）
    - 实现 ShmTransportable 协议的对象（如 FloatImage）走 shm + pickle 混合
    - 其他小对象走 pickle via Pipe
    - sentinel / cancellation 信号通过控制帧传递

使用场景：
    当两个 Op 被分配到不同进程时，wiring 层自动在它们之间插入 IPCQueue
    替代 RichContextQueue。Op 代码无需任何修改。
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import pickle
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np

from .queue import BaseQueue, CancellationError, CancellationToken, StreamExhausted

# ────────────────────────────────────────────────────────────────
# 数据描述符
# ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SharedArrayRef:
    """共享内存数组引用（可 pickle，通过 Pipe 传递）。

    生产者创建 SharedMemory 并写入数据后，将此描述符发给消费者。
    消费者通过 shm_name attach 到同一块共享内存读取数据。
    """
    shm_name: str
    shape: tuple[int, ...]
    dtype: str


# ────────────────────────────────────────────────────────────────
# ShmTransportable 协议
# ────────────────────────────────────────────────────────────────


@runtime_checkable
class ShmTransportable(Protocol):
    """声明对象支持 SharedMemory 高效传输。

    轻量包装类（如 FloatImage）实现此协议后，IPCQueue 会：
    1. 调用 __shm_pack__ 提取主数组和辅助元数据
    2. 主数组走 SharedMemory，元数据走 pickle
    3. 消费端通过 __shm_unpack__ 重建对象
    """

    def __shm_pack__(self) -> tuple[np.ndarray, bytes]:
        """返回 (主数组, 辅助元数据的 pickle 字节)"""
        ...

    @classmethod
    def __shm_unpack__(cls, array: np.ndarray, meta: bytes) -> Any:
        """从 shm 数组和元数据重建对象"""
        ...


# ShmTransportable 类注册表（class qualname → class）
_SHM_REGISTRY: dict[str, type] = {}


def register_shm_transportable(cls: type) -> type:
    """注册一个 ShmTransportable 类到全局注册表。

    用法::

        @register_shm_transportable
        @dataclass(slots=True)
        class FloatImage:
            ...
    """
    _SHM_REGISTRY[cls.__qualname__] = cls
    return cls


# ────────────────────────────────────────────────────────────────
# IPCQueue
# ────────────────────────────────────────────────────────────────

# Pipe 消息 tag 常量
_TAG_ARRAY = "array"
_TAG_SHM_OBJ = "shm_obj"
_TAG_PICKLE = "pickle"
_TAG_SENTINEL = "sentinel"
_TAG_CANCEL = "cancel"


class IPCQueue(BaseQueue):
    """跨进程异步队列，继承 BaseQueue。

    内部使用两层通道：
        - Pipe: 传输控制帧和小对象（pickle 编码）
        - SharedMemory: 传输大型 numpy 数组（零拷贝）

    背压通过双信号量实现（与 asyncio.Queue(maxsize=N) 语义一致）：
        - _empty_sem: 可用 slot 数（producer 在 put 前 acquire）
        - _filled_sem: 就绪 slot 数（consumer 在 get 前 acquire）

    Args:
        maxsize: 队列最大容量，控制背压。默认 1（与 RichContextQueue 一致）。
        shm_threshold: numpy 数组字节数超过此阈值时走 SharedMemory。
                       小数组走 pickle 更快（避免 shm 创建/销毁开销）。
    """

    def __init__(self, maxsize: int = 1, shm_threshold: int = 1024):
        # 双向 pipe：支持 sentinel 回填（get 端写回 pipe）
        self._conn_a, self._conn_b = mp.Pipe(duplex=True)
        # 约定：_conn_a 用于 put（send），_conn_b 用于 get（recv）
        # 回填时 get 端通过 _conn_b.send() 写入

        self._empty_sem = mp.Semaphore(maxsize)
        self._filled_sem = mp.Semaphore(0)

        # 长度信号
        self._length_val = mp.Value('l', -1)  # -1=未设置, -2=None(sentinel驱动)
        self._length_event = mp.Event()

        self.active: bool = True
        self._shm_threshold = shm_threshold

        # 异常清理：跟踪已创建但未被消费者释放的 shm
        self._pending_shm_names: list[str] = []

    # ── put ──

    async def put(self, item: Any) -> None:
        """将对象放入队列（跨进程安全）。"""
        await asyncio.to_thread(self._empty_sem.acquire)

        try:
            if item is BaseQueue._SENTINEL:
                self._conn_a.send((_TAG_SENTINEL, None))
            elif isinstance(item, CancellationToken):
                self._conn_a.send(
                    (_TAG_CANCEL, str(item.error), item.source_node))
            elif isinstance(item, np.ndarray) and item.nbytes > self._shm_threshold:
                ref = self._write_shm(item)
                self._conn_a.send((_TAG_ARRAY, ref))
            elif isinstance(item, ShmTransportable):
                arr, meta = item.__shm_pack__()
                if arr.nbytes > self._shm_threshold:
                    ref = self._write_shm(arr)
                    self._conn_a.send(
                        (_TAG_SHM_OBJ, type(item).__qualname__, ref, meta))
                else:
                    self._conn_a.send((_TAG_PICKLE, pickle.dumps(item)))
            else:
                self._conn_a.send((_TAG_PICKLE, pickle.dumps(item)))
        except Exception:
            self._empty_sem.release()
            raise

        self._filled_sem.release()

    # ── get ──

    async def get(self) -> Any:
        """从队列获取对象（跨进程安全），自动处理 sentinel/cancellation。"""
        await asyncio.to_thread(self._filled_sem.acquire)
        msg = await asyncio.to_thread(self._conn_b.recv)
        tag = msg[0]

        if tag == _TAG_SENTINEL:
            # 回填语义：再发一次 sentinel，让并发消费者也能收到
            self._conn_b.send((_TAG_SENTINEL, None))
            self._filled_sem.release()
            raise StreamExhausted("Stream ended normally")

        if tag == _TAG_CANCEL:
            _, error_str, source_node = msg
            self._conn_b.send(msg)
            self._filled_sem.release()
            raise CancellationError(
                f"Upstream node '{source_node}' failed: {error_str}")

        # 数据帧：释放 empty slot
        self._empty_sem.release()

        if tag == _TAG_ARRAY:
            ref: SharedArrayRef = msg[1]
            return self._read_shm(ref)

        if tag == _TAG_SHM_OBJ:
            _, cls_name, ref, meta = msg
            arr = self._read_shm(ref)
            cls = _SHM_REGISTRY.get(cls_name)
            if cls is None:
                raise ValueError(
                    f"ShmTransportable class '{cls_name}' not registered. "
                    f"Available: {sorted(_SHM_REGISTRY.keys())}")
            return cls.__shm_unpack__(arr, meta)

        if tag == _TAG_PICKLE:
            return pickle.loads(msg[1])

        raise ValueError(f"Unknown IPC message tag: {tag}")

    # ── length 信号 ──

    async def set_length(self, length: Optional[int]) -> None:
        """由生产者设置序列长度。None 表示长度未知（sentinel 驱动）。"""
        val = length if length is not None else -2
        with self._length_val.get_lock():
            current = self._length_val.value
            if current != -1 and val != -1 and current != val:
                raise ValueError(f"Length mismatch: {current} vs {val}")
            self._length_val.value = val
        self._length_event.set()

    async def get_length(self) -> Optional[int]:
        """消费者等待并获取序列长度。返回 None 表示长度未知。"""
        await asyncio.to_thread(self._length_event.wait)
        val = self._length_val.value
        return val if val >= 0 else None

    # ── SharedMemory 读写 ──

    def _write_shm(self, arr: np.ndarray) -> SharedArrayRef:
        """将 numpy 数组写入新的 SharedMemory 块。"""
        shm = SharedMemory(create=True, size=max(arr.nbytes, 1))
        view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        view[:] = arr  # 一次 memcpy
        ref = SharedArrayRef(shm.name, arr.shape, str(arr.dtype))
        self._pending_shm_names.append(shm.name)
        shm.close()  # 关闭本进程 fd，shm 块仍然存在
        return ref

    def _read_shm(self, ref: SharedArrayRef) -> np.ndarray:
        """从 SharedMemory 块读取 numpy 数组并释放。"""
        shm = SharedMemory(name=ref.shm_name, create=False)
        arr = np.ndarray(
            ref.shape, dtype=np.dtype(ref.dtype), buffer=shm.buf).copy()
        shm.close()
        shm.unlink()  # 消费者负责释放
        try:
            self._pending_shm_names.remove(ref.shm_name)
        except ValueError:
            pass  # 可能已被 cleanup 清理
        return arr

    # ── 清理 ──

    def cleanup(self) -> None:
        """清理所有未释放的 SharedMemory 块和 Pipe 连接。"""
        for name in self._pending_shm_names:
            try:
                shm = SharedMemory(name=name, create=False)
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass  # 已被消费者释放
        self._pending_shm_names.clear()

        for conn in (self._conn_a, self._conn_b):
            try:
                conn.close()
            except Exception:
                pass

    def __del__(self):
        """安全网：防止未清理的 SharedMemory 泄漏。"""
        if self._pending_shm_names:
            self.cleanup()
