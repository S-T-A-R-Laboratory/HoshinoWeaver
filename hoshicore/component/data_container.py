"""
dtype 基础设施 + 管线数据容器。

本模块包含：
    1. dtype 级差表与放缩函数（uint 系列的统一抽象）
    2. 数据容器类：FloatImage, GaussianParam, FastGaussianParam, HuberMeanParam

设计原则：
    - _UINT_DTYPES 为唯一真值源，所有 dtype 常量表由其派生
    - 节点间传递裸 np.ndarray，约定"值域填满容器 dtype 的范围"
    - 需要跨 dtype 对齐或保存时使用 rescale_array
    - ShmTransportable 子类支持跨进程 SharedMemory 高效传输
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from loguru import logger

from .ipc_queue import ShmTransportable

# ────────────────────────────────────────────────────────────────
# dtype 基础设施（唯一真值源：_UINT_DTYPES）
# ────────────────────────────────────────────────────────────────

_UINT_DTYPES: tuple[np.dtype, ...] = (
    np.dtype("uint8"),
    np.dtype("uint16"),
    np.dtype("uint32"),
    np.dtype("uint64"),
)

_SCALE_BASE = 256

# dtype → 级序号
DTYPE_LEVEL: dict[np.dtype, int] = {d: i for i, d in enumerate(_UINT_DTYPES)}

# dtype → 最大整型值
DTYPE_MAX_VALUE: dict[np.dtype, int] = {
    d: int(np.iinfo(d).max)
    for d in _UINT_DTYPES
}

# dtype → 上一级 dtype（uint64 → float）
DTYPE_UPSCALE_MAP: dict[np.dtype, Union[np.dtype, type]] = {
    _UINT_DTYPES[i]: _UINT_DTYPES[i + 1]
    for i in range(len(_UINT_DTYPES) - 1)
}
DTYPE_UPSCALE_MAP[_UINT_DTYPES[-1]] = float

# ────────────────────────────────────────────────────────────────
# 放缩函数
# ────────────────────────────────────────────────────────────────


def _cumscale_factor(level: int, exp_base: int = 0) -> int:
    """级差 level 对应的累积放缩 factor。

    每跨一级 scale = _SCALE_BASE ** (i+1) + 1 = 257，n 级累积 = 257^n。
    例如 level=2（uint8 → uint32）时 factor = 257² = 66049。
    """
    factor = 1
    _cumscale_base = _SCALE_BASE**(exp_base+1)
    for i in range(abs(level)):
        factor *= _cumscale_base**(i + 1) + 1
    return factor


def get_scale_x(time: int, base: int = 256) -> int:
    """单步放缩因子：base^time + 1。

    注意与 _cumscale_factor 的区别：
        _cumscale_factor(n) = (base+1)^n  （多步累积）
        get_scale_x(n)      = base^n + 1  （单步公式）
    两者仅在 n=1 时相等（均为 257）。
    """
    return base**time + 1


def rescale_array(
    data: np.ndarray,
    from_dtype: np.dtype,
    to_dtype: np.dtype,
) -> np.ndarray:
    """在两个 dtype 级别之间双向放缩数据。

    根据 from_dtype 与 to_dtype 的级差自动选择方向：
        - to 级别更高：向上放缩（× scale），如 uint8 [0,255] → uint16 [0,65535]
        - to 级别更低：向下缩放（÷ scale），如 uint16 [0,65535] → uint8 [0,255]
        - 同级或无法判定：仅做 dtype cast
    """
    from_level = DTYPE_LEVEL.get(np.dtype(from_dtype))
    to_level = DTYPE_LEVEL.get(np.dtype(to_dtype))

    if from_level is None or to_level is None:
        return data.astype(to_dtype)

    diff = to_level - from_level
    if diff == 0:
        return data.astype(to_dtype)
    sf = _cumscale_factor(diff, exp_base=min(from_level, to_level))
    if diff > 0:
        return data.astype(to_dtype) * sf
    else:
        return (data // sf).astype(to_dtype)


def align_dtype_pair(
    arr_a: np.ndarray,
    dtype_a: np.dtype,
    arr_b: np.ndarray,
    dtype_b: np.dtype,
) -> tuple[np.ndarray, np.ndarray, np.dtype]:
    """对齐两个数组的 dtype 级别，将较低级别的数组放缩到较高级别。"""
    level_a = DTYPE_LEVEL.get(np.dtype(dtype_a))
    level_b = DTYPE_LEVEL.get(np.dtype(dtype_b))

    if level_a is None or level_b is None:
        return arr_a, arr_b, dtype_a

    if level_a == level_b:
        return arr_a, arr_b, dtype_a

    if level_a < level_b:
        return rescale_array(arr_a, dtype_a, dtype_b), arr_b, dtype_b
    else:
        return arr_a, rescale_array(arr_b, dtype_b, dtype_a), dtype_a


def rdtype_detector(data: np.ndarray) -> Union[np.dtype, type]:
    """根据数据最大值推断实际 dtype 级别。"""
    if data.dtype == float:
        return float
    if np.max(data) <= DTYPE_MAX_VALUE[np.dtype("uint8")]:
        return np.dtype("uint8")
    if np.max(data) <= DTYPE_MAX_VALUE[np.dtype("uint16")]:
        return np.dtype("uint16")
    if np.max(data) <= DTYPE_MAX_VALUE[np.dtype("uint32")]:
        return np.dtype("uint32")
    if np.max(data) <= DTYPE_MAX_VALUE[np.dtype("uint64")]:
        return np.dtype("uint64")
    raise NotImplementedError("Unrecognized data type.")


# ────────────────────────────────────────────────────────────────
# 数据容器
# ────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class FloatImage(ShmTransportable):
    """轻量级的Float矩阵包装器，包含原始数据范围dtype。"""
    data: np.ndarray
    dtype: np.dtype

    def int_transform(self,
                      target_dtype: Optional[np.dtype] = None) -> np.ndarray:
        if target_dtype is None:
            target_dtype = self.dtype
        return rescale_array(self.data, self.dtype, target_dtype)

    def shm_nbytes(self) -> int:
        return self.data.nbytes

    def shm_pack_into(self, buf) -> bytes:
        raw = self.data.ravel().view(np.uint8)
        np.ndarray(len(raw), np.uint8, buffer=buf)[:] = raw
        return pickle.dumps({
            "dtype": str(self.dtype),
            "arr_shape": self.data.shape,
            "arr_dtype": str(self.data.dtype),
        })

    @classmethod
    def shm_unpack_from(cls, buf, meta: bytes, shm_handle=None) -> FloatImage:
        m = pickle.loads(meta)
        raw = np.ndarray(np.prod(m["arr_shape"]) *
                         np.dtype(m["arr_dtype"]).itemsize,
                         np.uint8,
                         buffer=buf)
        data = raw.view(np.dtype(m["arr_dtype"])).reshape(
            m["arr_shape"]).copy()
        if shm_handle is not None:
            shm_handle.close()
            shm_handle.unlink()
        return cls(data=data, dtype=np.dtype(m["dtype"]))


class GaussianParam(object):
    """维护np.ndarray的流方差与流均值的原始实现。"""

    def __init__(self,
                 mu: np.ndarray,
                 var: Optional[np.ndarray] = None,
                 n: Optional[np.ndarray] = None,
                 ddof: int = 1,
                 dtype_var: np.dtype = np.dtype("float32"),
                 dtype_n: np.dtype = np.dtype("int16")):
        self.mu = mu
        self.var = var if var is not None else np.zeros_like(mu,
                                                             dtype=dtype_var)
        self.n = n if n is not None else np.ones_like(self.mu, dtype=dtype_n)
        self.ddof = ddof

    def __add__(self, g2):
        g1 = self
        assert isinstance(g2, GaussianParam), "unacceptable object"
        assert g1.ddof == g2.ddof, "unmatched var calculation!"
        ddof = g1.ddof
        new_mu = (g1.mu * g1.n + g2.mu * g2.n) / (g1.n + g2.n)
        new_var = ((g1.n - ddof) * g1.var + g1.n * np.square(g1.mu) +
                   (g2.n - ddof) * g2.var + g2.n * np.square(g2.mu) -
                   (g1.n + g2.n) * np.square(new_mu)) / (g1.n + g2.n - ddof)
        return GaussianParam(mu=new_mu, var=new_var, n=g1.n + g2.n, ddof=ddof)

    def __sub__(self, g2):
        g1 = self
        assert isinstance(g2, GaussianParam), "unacceptable object"
        assert g1.ddof == g2.ddof, "unmatched var calculation!"
        assert g1.n > g2.n, "generate n<0 fistribution!"
        ddof = g1.ddof
        new_mu = (g1.mu * g1.n - g2.mu * g2.n) / (g1.n - g2.n)
        new_var = ((g1.n - ddof) * g1.var + g1.n * np.square(g1.mu) -
                   (g2.n - ddof) * g2.var - g2.n * np.square(g2.mu) -
                   (g1.n - g2.n) * np.square(new_mu)) / (g1.n - g2.n - ddof)
        return GaussianParam(mu=new_mu, var=new_var, n=g1.n - g2.n, ddof=ddof)


class FastGaussianParam(ShmTransportable):
    """GaussianParam 的高速版本。

    通过 INT 量化 + 优化数据储存提速，仅在输出时换算为浮点数。
    Streaming mean and variance.

    支持跨进程 SharedMemory 高效传输，
    避免 pickle 序列化 3 个大型 ndarray 时的内存放大问题。
    """

    def __init__(self,
                 sum_mu: np.ndarray,
                 square_sum: Optional[np.ndarray] = None,
                 n: Optional[np.ndarray] = None,
                 ddof: int = 1,
                 dtype_n: np.dtype = np.dtype("uint16"),
                 source_dtype: Optional[np.dtype] = None,
                 inplace_calc: bool = True):
        self.sum_mu = sum_mu
        self.source_dtype = source_dtype if source_dtype is not None else sum_mu.dtype
        if square_sum is not None:
            self.square_sum = square_sum
        else:
            sq_dtype = self.get_upscale_dtype_as(self.sum_mu)
            self.square_sum = np.square(sum_mu, dtype=sq_dtype)
        self.n = n if n is not None else np.ones_like(self.sum_mu,
                                                      dtype=dtype_n)
        self.max_n = int(np.max(self.n))
        if self.sum_mu.dtype == self.source_dtype:
            self.upscale()
        self.ddof = ddof
        self.inplace_calc = inplace_calc

    @property
    def mu(self) -> np.ndarray:
        return np.round(self.sum_mu / self.n)

    @property
    def var(self) -> np.ndarray:
        sum_mu = np.array(self.sum_mu, dtype=self.square_sum.dtype)
        return (self.square_sum - np.square(sum_mu) / self.n) / (self.n -
                                                                 self.ddof)

    def upscale(self):
        upscaled_sum_mu_dtype = self.get_upscale_dtype_as(self.sum_mu)
        upscaled_sum_sq_dtype = self.get_upscale_dtype_as(self.square_sum)
        self.sum_mu = np.array(self.sum_mu, dtype=upscaled_sum_mu_dtype)
        self.square_sum = np.array(self.square_sum,
                                   dtype=upscaled_sum_sq_dtype)

    def get_upscale_dtype_as(self, ref_array: np.ndarray):
        return DTYPE_UPSCALE_MAP[
            ref_array.dtype] if ref_array.dtype in DTYPE_UPSCALE_MAP else float

    def _safe_add_count(self) -> Union[int, float]:
        if self.source_dtype not in DTYPE_MAX_VALUE:
            return float('inf')
        source_max = DTYPE_MAX_VALUE[self.source_dtype]
        sum_limit = DTYPE_MAX_VALUE.get(self.sum_mu.dtype,
                                        float('inf')) // source_max
        n_limit = DTYPE_MAX_VALUE.get(self.n.dtype, float('inf'))
        return min(sum_limit, n_limit)

    def apply_zero_var(self, full_img):
        zero_pos = (self.n == 0)
        logger.debug(f"Zero-mask {np.where(zero_pos)[0].size} pixels.")
        self.n[zero_pos] = full_img.n[zero_pos]
        self.sum_mu[zero_pos] = full_img.sum_mu[zero_pos]
        self.square_sum[zero_pos] = full_img.square_sum[zero_pos]

    def __add__(self, g2):
        g1 = self
        assert isinstance(g2, FastGaussianParam), "unacceptable object"
        assert g1.ddof == g2.ddof, "unmatched var calculation!"

        self.max_n = self.max_n + g2.max_n

        if self.max_n > g1._safe_add_count():
            g1.upscale()

        if g1.n.dtype in DTYPE_MAX_VALUE and self.max_n > DTYPE_MAX_VALUE[
                g1.n.dtype]:
            if g1.n.dtype in DTYPE_UPSCALE_MAP:
                new_n_dtype = DTYPE_UPSCALE_MAP[g1.n.dtype]
                g1.n = g1.n.astype(new_n_dtype)

        if self.inplace_calc:
            self.sum_mu += g2.sum_mu
            self.square_sum += g2.square_sum
            self.n += g2.n
            return self

        return FastGaussianParam(sum_mu=g1.sum_mu + g2.sum_mu,
                                 square_sum=g1.square_sum + g2.square_sum,
                                 n=g1.n + g2.n,
                                 ddof=g1.ddof,
                                 source_dtype=g1.source_dtype)

    def __sub__(self, g2):
        g1 = self
        assert isinstance(g2, FastGaussianParam), "unacceptable object"
        assert g1.ddof == g2.ddof, "unmatched var calculation!"
        assert (g1.n - g2.n).any() >= 0, "generate n<0 fistribution!"

        if self.inplace_calc:
            self.sum_mu -= g2.sum_mu
            self.square_sum -= g2.square_sum
            self.n -= g2.n
            self.max_n = int(self.n.max())
            return self

        return FastGaussianParam(sum_mu=g1.sum_mu - g2.sum_mu,
                                 square_sum=g1.square_sum - g2.square_sum,
                                 n=g1.n - g2.n,
                                 ddof=g1.ddof,
                                 source_dtype=g1.source_dtype)

    def __mul__(self, weight: Union[float, int, np.ndarray]):
        if isinstance(weight, (int, float, np.ndarray)):
            if self.inplace_calc:
                self.sum_mu *= weight
                self.square_sum *= weight
                self.n = self.n * weight
                self.max_n = int(self.max_n * weight)
                return self
            return FastGaussianParam(sum_mu=self.sum_mu * weight,
                                     square_sum=self.square_sum * weight,
                                     n=self.n * weight,
                                     ddof=self.ddof,
                                     source_dtype=self.source_dtype)
        raise NotImplementedError(
            f"Unsupported weight type {type(weight)} for "
            f"multiplication with {self.__class__.__name__}.")

    def __rmul__(self, weight: Union[float, int, np.ndarray]):
        return self.__mul__(weight)

    def mask(self, mask_pos: np.ndarray):
        assert mask_pos.dtype == np.dtype("bool"), "Invalid mask!"
        self.sum_mu *= mask_pos
        self.square_sum *= mask_pos
        self.n = self.n * mask_pos.astype(self.n.dtype)

    @property
    def shape(self):
        return self.sum_mu.shape

    # ── ShmTransportable ──

    def _shm_meta(self) -> dict:
        return {
            "sum_mu_shape": self.sum_mu.shape,
            "sum_mu_dtype": str(self.sum_mu.dtype),
            "sum_mu_nbytes": self.sum_mu.nbytes,
            "sq_shape": self.square_sum.shape,
            "sq_dtype": str(self.square_sum.dtype),
            "sq_nbytes": self.square_sum.nbytes,
            "n_shape": self.n.shape,
            "n_dtype": str(self.n.dtype),
            "n_nbytes": self.n.nbytes,
            "ddof": self.ddof,
            "source_dtype": str(self.source_dtype),
            "inplace_calc": self.inplace_calc,
            "max_n": self.max_n,
        }

    def shm_nbytes(self) -> int:
        return self.sum_mu.nbytes + self.square_sum.nbytes + self.n.nbytes

    def shm_pack_into(self, buf) -> bytes:
        off = 0
        for arr in (self.sum_mu, self.square_sum, self.n):
            raw = arr.ravel().view(np.uint8)
            np.ndarray(len(raw), np.uint8, buffer=buf, offset=off)[:] = raw
            off += arr.nbytes
        return pickle.dumps(self._shm_meta())

    @classmethod
    def _shm_rebuild(cls, m: dict, sum_mu, square_sum, n) -> FastGaussianParam:
        obj = cls.__new__(cls)
        obj.sum_mu = sum_mu
        obj.square_sum = square_sum
        obj.n = n
        obj.ddof = m["ddof"]
        obj.source_dtype = np.dtype(m["source_dtype"])
        obj.inplace_calc = m["inplace_calc"]
        obj.max_n = m["max_n"]
        return obj

    @classmethod
    def shm_unpack_from(cls, buf, meta: bytes,
                        shm_handle=None) -> FastGaussianParam:
        m = pickle.loads(meta)
        off = 0
        sum_mu = np.ndarray(m["sum_mu_shape"], np.dtype(m["sum_mu_dtype"]),
                            buffer=buf, offset=off)
        off += m["sum_mu_nbytes"]
        square_sum = np.ndarray(m["sq_shape"], np.dtype(m["sq_dtype"]),
                                buffer=buf, offset=off)
        off += m["sq_nbytes"]
        n = np.ndarray(m["n_shape"], np.dtype(m["n_dtype"]),
                       buffer=buf, offset=off)
        obj = cls._shm_rebuild(m, sum_mu, square_sum, n)
        obj._shm_owner = shm_handle  # 持有 handle，防止 shm 被提前销毁
        return obj

    def _force_copy_from_shm(self) -> None:
        """broadcast 场景：将 shm view 数组 copy 为独立内存。"""
        logger.trace("Force copy FastGaussianParam arrays from shared memory.")
        self.sum_mu = self.sum_mu.copy()
        self.square_sum = self.square_sum.copy()
        self.n = self.n.copy()

    def __del__(self):
        shm = getattr(self, '_shm_owner', None)
        if shm is not None:
            self._shm_owner = None
            from .ipc_queue import _safe_close_shm
            _safe_close_shm(shm, unlink=True)


class HuberMeanParam(ShmTransportable):
    """Huber 加权均值的流式累加器。

    存储加权和 weighted_sum = Σ(w_i · x_i) 和权重总和 weight_total = Σ(w_i)，
    最终结果 μ = weighted_sum / weight_total。

    两者均为逐像素数组，支持 __add__ 用于分布式归约（merge_partial）。
    """

    def __init__(
        self,
        weighted_sum: np.ndarray,
        weight_total: np.ndarray,
        source_dtype: Optional[np.dtype] = None,
    ):
        self.weighted_sum = weighted_sum
        self.weight_total = weight_total
        self.source_dtype = source_dtype

    def add(self,
            img: np.ndarray,
            huber_weight: np.ndarray,
            frame_weight: Optional[Union[float, np.ndarray]] = None):
        w = huber_weight
        if frame_weight is not None:
            w = w * frame_weight
        self.weighted_sum += (img * w).astype(self.weighted_sum.dtype)
        self.weight_total += w.astype(self.weight_total.dtype)

    def __add__(self, other: HuberMeanParam) -> HuberMeanParam:
        return HuberMeanParam(
            weighted_sum=self.weighted_sum + other.weighted_sum,
            weight_total=self.weight_total + other.weight_total,
            source_dtype=self.source_dtype,
        )

    @property
    def mu(self) -> np.ndarray:
        safe_total = np.where(self.weight_total > 0, self.weight_total, 1)
        return np.round(self.weighted_sum / safe_total)

    @property
    def shape(self):
        return self.weighted_sum.shape

    # ── ShmTransportable ──

    def _shm_meta(self) -> dict:
        return {
            "ws_shape": self.weighted_sum.shape,
            "ws_dtype": str(self.weighted_sum.dtype),
            "ws_nbytes": self.weighted_sum.nbytes,
            "wt_shape": self.weight_total.shape,
            "wt_dtype": str(self.weight_total.dtype),
            "wt_nbytes": self.weight_total.nbytes,
            "source_dtype":
            str(self.source_dtype) if self.source_dtype else None,
        }

    def shm_nbytes(self) -> int:
        return self.weighted_sum.nbytes + self.weight_total.nbytes

    def shm_pack_into(self, buf) -> bytes:
        off = 0
        for arr in (self.weighted_sum, self.weight_total):
            raw = arr.ravel().view(np.uint8)
            np.ndarray(len(raw), np.uint8, buffer=buf, offset=off)[:] = raw
            off += arr.nbytes
        return pickle.dumps(self._shm_meta())

    @classmethod
    def _shm_rebuild(cls, m: dict, weighted_sum,
                     weight_total) -> HuberMeanParam:
        source_dtype = np.dtype(
            m["source_dtype"]) if m["source_dtype"] else None
        return cls(weighted_sum=weighted_sum,
                   weight_total=weight_total,
                   source_dtype=source_dtype)

    @classmethod
    def shm_unpack_from(cls, buf, meta: bytes,
                        shm_handle=None) -> HuberMeanParam:
        m = pickle.loads(meta)
        off = 0
        weighted_sum = np.ndarray(m["ws_shape"], np.dtype(m["ws_dtype"]),
                                  buffer=buf, offset=off)
        off += m["ws_nbytes"]
        weight_total = np.ndarray(m["wt_shape"], np.dtype(m["wt_dtype"]),
                                  buffer=buf, offset=off)
        obj = cls._shm_rebuild(m, weighted_sum, weight_total)
        obj._shm_owner = shm_handle
        return obj

    def _force_copy_from_shm(self) -> None:
        """broadcast 场景：将 shm view 数组 copy 为独立内存。"""
        self.weighted_sum = self.weighted_sum.copy()
        self.weight_total = self.weight_total.copy()

    def __del__(self):
        shm = getattr(self, '_shm_owner', None)
        if shm is not None:
            self._shm_owner = None
            from .ipc_queue import _safe_close_shm
            _safe_close_shm(shm, unlink=True)
