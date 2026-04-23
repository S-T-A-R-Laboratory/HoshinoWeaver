"""
跨进程异步队列：通过 SharedMemory + Pipe 实现高效的进程间数据传输。

设计目标：
    - 继承 BaseQueue，接口与 RichContextQueue 完全一致
    - np.ndarray 通过 SharedMemory 零拷贝传输（跨平台：Windows/Linux/macOS）
    - 继承 ShmTransportable 的对象（如 FloatImage）走 shm + pickle 混合
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
from collections import deque
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from abc import ABC, abstractmethod
from typing import Any, Optional

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


class ShmTransportable(ABC):
    """声明对象支持 SharedMemory 高效传输的抽象基类。

    子类实现三个方法后，IPCQueue 自动走高效路径：
    1. shm_nbytes() → 预计算所需 shm 字节数
    2. shm_pack_into(buf) → 将数据直写入 shm buffer，返回元数据
    3. shm_unpack_from(buf, meta) → 从 shm buffer + 元数据重建对象

    子类定义时自动注册到 _SHM_REGISTRY（通过 __init_subclass__）。
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not getattr(cls, '__abstractmethods__', None):
            _SHM_REGISTRY[cls.__qualname__] = cls

    @abstractmethod
    def shm_nbytes(self) -> int:
        """返回 shm 传输所需的总字节数。"""
        ...

    @abstractmethod
    def shm_pack_into(self, buf) -> bytes:
        """将数据直写入 shm buffer，返回辅助元数据（pickle 字节）。"""
        ...

    @classmethod
    @abstractmethod
    def shm_unpack_from(cls, buf, meta: bytes) -> Any:
        """从 shm buffer 和元数据重建对象。"""
        ...


# ShmTransportable 类注册表（class qualname → class）
_SHM_REGISTRY: dict[str, type] = {}


# ────────────────────────────────────────────────────────────────
# IPCQueue
# ────────────────────────────────────────────────────────────────

# Pipe 消息 tag 常量
_TAG_ARRAY = "array"
_TAG_SHM_OBJ = "shm_obj"
_TAG_PICKLE = "pickle"
_TAG_SHM_PICKLE = "shm_pickle"   # pickle 字节存入 SharedMemory（大对象防 pipe 阻塞）
_TAG_SENTINEL = "sentinel"
_TAG_CANCEL = "cancel"

# pickle 超过此阈值时走 SharedMemory，避免 send() 阻塞 event loop 造成死锁。
_PIPE_SAFE_THRESHOLD = 32 * 1024  # 32 KB，留一半余量给 pipe 控制帧


def _safe_close_shm(shm: SharedMemory, *, unlink: bool = False) -> None:
    """关闭 SharedMemory handle，可选 unlink。

    生产者正常流程中仅 close handle（消费者负责 unlink）。
    仅在异常清理路径（cleanup()）中传入 unlink=True 以防止泄漏。

    Args:
        shm: SharedMemory 对象。
        unlink: True 时额外执行 shm.unlink() 删除共享内存块。
    """
    try:
        shm.close()
    except OSError:
        pass
    if unlink:
        try:
            shm.unlink()
        except (FileNotFoundError, OSError):
            pass


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
        self._maxsize = maxsize

        # 长度信号
        self._length_val = mp.Value('l', -1)  # -1=未设置, -2=None(sentinel驱动)
        self._length_event = mp.Event()

        self.active: bool = True
        self._shm_threshold = shm_threshold

        # 异常清理：跟踪已创建但未被消费者释放的 shm。
        # 注意：必须保持 SharedMemory 对象（而非仅名字）存活，
        # 因为 Windows Named File Mapping 在最后一个 handle 关闭时立即销毁。
        # 生产者必须持有 handle 直到消费者成功 attach。
        #
        # _pending_shm: 当前 _pack_item 正在构建的 slot 的 handle（临时）
        # _slot_shm:    已发送但尚未确认消费的 slot handle 列表（按发送顺序）
        # _put_count:   生产者已执行的 put 次数。前 maxsize 次 put 的
        #               _empty_sem.acquire() 来自初始信号量计数（而非消费者释放），
        #               不能关闭旧 slot 的 shm handle（消费者尚未读取）。
        self._pending_shm: dict[str, SharedMemory] = {}
        self._slot_shm: deque[list[SharedMemory]] = deque()
        self._put_count: int = 0

    # ── put ──

    async def put(self, item: Any) -> None:
        """将对象放入队列（跨进程安全）。"""
        await asyncio.to_thread(self._empty_sem.acquire)

        # 仅当此次 acquire 对应消费者释放（而非初始信号量计数）时，
        # 才安全关闭最旧 slot 的 SharedMemory handle。
        # 前 maxsize 次 acquire 来自初始计数，消费者尚未读取任何数据。
        # 第 maxsize+1 次 acquire 才是消费者第 1 次释放的信号量，
        # 此时消费者已经读完了第 1 个 slot 的数据，可以安全关闭。
        # 因此阈值必须是 > maxsize（严格大于），不能是 >=。
        if self._put_count > self._maxsize and self._slot_shm:
            for shm in self._slot_shm.popleft():
                _safe_close_shm(shm)

        self._put_count += 1

        before = set(self._pending_shm.keys())
        try:
            msg = self._pack_item(item)
        except Exception:
            self._empty_sem.release()
            # 清理本次 pack 中途创建的 shm
            for k in list(self._pending_shm.keys()):
                if k not in before:
                    self._pending_shm.pop(k).close()
            raise

        # 将本 slot 新创建的 handle 移入 _slot_shm，_pending_shm 保持干净
        slot_handles = [self._pending_shm.pop(k)
                        for k in list(self._pending_shm.keys())
                        if k not in before]

        # 先释放 _filled_sem，再执行 socket send。
        #
        # mp.Pipe(duplex=True) 底层使用 socketpair，macOS 默认发送缓冲区仅 8KB。
        # 当消息 > 8KB 时，_conn_a.send() 会阻塞，等待消费者 recv() 腾出空间。
        # 但消费者在 get() 中先 acquire _filled_sem，再调用 recv()——
        # 如果 _filled_sem.release() 在 send() 之后，消费者永远无法 recv()，
        # 形成死锁：send 等 recv 腾空间，recv 等 _filled_sem。
        #
        # 先 release 使消费者可以并发 recv()，与 send() 配合完成传输。
        # recv() 是阻塞的 socket read，即使在 release 和 send 之间有短暂窗口，
        # recv 也只会阻塞到 send 开始写入数据——不会出错或读到错误数据。
        # pipe 保证 FIFO 和消息边界完整性（Connection 协议使用长度前缀）。
        self._filled_sem.release()
        self._slot_shm.append(slot_handles)
        await asyncio.to_thread(self._conn_a.send, msg)

    def _pack_item(self, item: Any) -> tuple:
        """将对象打包为 pipe 消息元组（同步，不做 I/O）。"""
        if item is BaseQueue._SENTINEL:
            return (_TAG_SENTINEL, None)

        if isinstance(item, CancellationToken):
            return (_TAG_CANCEL, str(item.error), item.source_node)

        if isinstance(item, np.ndarray) and item.nbytes > self._shm_threshold:
            ref = self._write_shm(item)
            return (_TAG_ARRAY, ref)

        if isinstance(item, ShmTransportable):
            nbytes = item.shm_nbytes()
            if nbytes > self._shm_threshold:
                shm = SharedMemory(create=True, size=max(nbytes, 1))
                self._pending_shm[shm.name] = shm
                meta = item.shm_pack_into(shm.buf)
                ref = SharedArrayRef(shm.name, (nbytes,), 'uint8')
                return (_TAG_SHM_OBJ, type(item).__qualname__, ref, meta)
            # 小 ShmTransportable 走 pickle
            data = pickle.dumps(item)
            return self._pickle_or_shm(data)

        data = pickle.dumps(item)
        return self._pickle_or_shm(data)

    def _pickle_or_shm(self, data: bytes) -> tuple:
        """小 pickle 走 pipe，大 pickle 走 SharedMemory 防死锁。"""
        if len(data) <= _PIPE_SAFE_THRESHOLD:
            return (_TAG_PICKLE, data)
        # 大 pickle：写入 SharedMemory，pipe 只传引用
        shm = SharedMemory(create=True, size=len(data))
        shm.buf[:len(data)] = data
        self._pending_shm[shm.name] = shm  # 保持 handle 开放
        return (_TAG_SHM_PICKLE, shm.name, len(data))

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

        # 数据帧：先读取 SharedMemory 内容，再释放 empty slot。
        # 顺序至关重要：_empty_sem.release() 通知生产者可以关闭旧 slot 的
        # SharedMemory handle。在 Windows 上，关闭最后一个 handle 会立即
        # 销毁 Named File Mapping。如果 release 在 read 之前，生产者可能
        # 在消费者 attach 之前就关闭了 handle → FileNotFoundError。
        try:
            if tag == _TAG_ARRAY:
                ref: SharedArrayRef = msg[1]
                return self._read_shm(ref)

            if tag == _TAG_SHM_OBJ:
                _, cls_name, ref, meta = msg
                cls = _SHM_REGISTRY.get(cls_name)
                if cls is None:
                    raise ValueError(
                        f"ShmTransportable class '{cls_name}' not registered. "
                        f"Available: {sorted(_SHM_REGISTRY.keys())}")
                shm = SharedMemory(name=ref.shm_name, create=False)
                result = cls.shm_unpack_from(shm.buf, meta)
                shm.close()
                shm.unlink()
                return result

            if tag == _TAG_PICKLE:
                return pickle.loads(msg[1])

            if tag == _TAG_SHM_PICKLE:
                shm_name, data_len = msg[1], msg[2]
                shm = SharedMemory(name=shm_name, create=False)
                data = bytes(shm.buf[:data_len])
                shm.close()
                shm.unlink()
                return pickle.loads(data)

            raise ValueError(f"Unknown IPC message tag: {tag}")
        finally:
            self._empty_sem.release()

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
        self._pending_shm[shm.name] = shm  # 保持 handle 开放，Windows 需要
        return ref

    def _read_shm(self, ref: SharedArrayRef) -> np.ndarray:
        """从 SharedMemory 块读取 numpy 数组并释放。"""
        shm = SharedMemory(name=ref.shm_name, create=False)
        arr = np.ndarray(
            ref.shape, dtype=np.dtype(ref.dtype), buffer=shm.buf).copy()
        shm.close()
        shm.unlink()  # 消费者负责释放（Linux/macOS 有效；Windows 为 no-op）
        return arr

    # ── 清理 ──

    def cleanup(self) -> None:
        """清理所有未释放的 SharedMemory 块和 Pipe 连接。"""
        # _pending_shm：pack 中途失败残留（正常情况为空）
        for shm in list(self._pending_shm.values()):
            _safe_close_shm(shm, unlink=True)
        self._pending_shm.clear()

        # _slot_shm：已发送但尚未被下一次 put() 关闭的 handle
        while self._slot_shm:
            for shm in self._slot_shm.popleft():
                _safe_close_shm(shm, unlink=True)

        for conn in (self._conn_a, self._conn_b):
            try:
                conn.close()
            except Exception:
                pass

    def __del__(self):
        """安全网：防止未清理的 SharedMemory 泄漏。"""
        if self._pending_shm or self._slot_shm:
            self.cleanup()
