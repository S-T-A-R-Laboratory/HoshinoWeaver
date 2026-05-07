"""
Numba JIT 加速 kernel：融合逐帧遍历操作，消除临时数组分配。

所有 kernel 遵循约定：
    - 输入图像必须是 3D (H, W, C)，调用方负责 expand_dims / squeeze
    - 累加器数组由调用方预分配并传入，kernel 就地修改
    - 使用 prange 并行化最外层行循环
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def fgp_mean_merge(
    img,
    sum_mu,
    square_sum,
    n,
):
    """将一帧图像累加到 FGP 累加器（无权重）。

    等价于 FastGaussianParam(img) 构造 + inplace __add__，
    但融合为单次遍历，避免 np.square + upscale + 3 个 inplace +=。
    """
    H = img.shape[0]
    W = img.shape[1]
    C = img.shape[2]
    for i in prange(H):
        for j in range(W):
            for c in range(C):
                v = sum_mu.dtype.type(img[i, j, c])
                sum_mu[i, j, c] += v
                square_sum[i, j, c] += square_sum.dtype.type(v) * square_sum.dtype.type(v)
                n[i, j, c] += 1


@njit(parallel=True, cache=True)
def fgp_weighted_mean_merge(
    img,
    weight,
    sum_mu,
    square_sum,
    n,
):
    """将一帧图像以整型权重累加到 FGP 累加器。

    调用方须已完成 int_weight upscale（img 已转为高一级 dtype，
    weight 已转为对应整型标量）。
    """
    H = img.shape[0]
    W = img.shape[1]
    C = img.shape[2]
    w = sum_mu.dtype.type(weight)
    for i in prange(H):
        for j in range(W):
            for c in range(C):
                v = sum_mu.dtype.type(img[i, j, c])
                sum_mu[i, j, c] += v * w
                sq_v = square_sum.dtype.type(v) * square_sum.dtype.type(v)
                square_sum[i, j, c] += sq_v * square_sum.dtype.type(w)
                n[i, j, c] += w


@njit(parallel=True, cache=True)
def fgp_masked_mean_merge(
    img,
    mask,
    sum_mu,
    square_sum,
    n,
):
    """只在 mask[i,j]=True 的位置累加到 FGP 累加器。

    mask: 2D bool array (H, W)，所有通道共享同一 mask。
    """
    H = img.shape[0]
    W = img.shape[1]
    C = img.shape[2]
    for i in prange(H):
        for j in range(W):
            if not mask[i, j]:
                continue
            for c in range(C):
                v = sum_mu.dtype.type(img[i, j, c])
                sum_mu[i, j, c] += v
                square_sum[i, j, c] += square_sum.dtype.type(v) * square_sum.dtype.type(v)
                n[i, j, c] += 1


@njit(parallel=True, cache=True)
def sigma_clip_fused_merge(
    img,
    rej_high_img,
    rej_low_img,
    sum_mu,
    square_sum,
    n,
):
    """Sigma clip 融合 kernel：单次遍历完成 clip 判断 + rejected FGP 累加。

    SigmaClippingMerger 累加的是**被拒绝**帧的像素统计量。
    像素在 [rej_low, rej_high] 范围内时被接受（不累加）；
    超出范围时被拒绝（累加到 rejected FGP）。

    等价于原始流程：
        FGP(img) → mask(rejected) → rejected_fgp += masked_fgp
    但消除了 ~500MB 临时数组和 ~10 次全图遍历。
    """
    H = img.shape[0]
    W = img.shape[1]
    C = img.shape[2]
    for i in prange(H):
        for j in range(W):
            for c in range(C):
                v = img[i, j, c]
                if v < rej_low_img[i, j, c] or v > rej_high_img[i, j, c]:
                    sv = sum_mu.dtype.type(v)
                    sum_mu[i, j, c] += sv
                    square_sum[i, j, c] += square_sum.dtype.type(sv) * square_sum.dtype.type(sv)
                    n[i, j, c] += 1
