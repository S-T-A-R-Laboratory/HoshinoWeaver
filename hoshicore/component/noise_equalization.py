"""噪声均匀化模块

用于消除最大值叠加中因镜头校正导致的空间不均匀噪声伪影。
详见 docs/noise-equalization.md
"""
import numpy as np
from numpy.typing import NDArray
from loguru import logger


def equalize_noise(max_img: NDArray,
                   mean_img: NDArray,
                   std_img: NDArray,
                   n_img: NDArray,
                   min_frames: int = 10) -> NDArray:
    """对最大值叠加图像应用噪声均匀化校正。

    核心公式: M_corrected = M - (σ̂ - σ_ref) · ĉ_n^eff

    Args:
        max_img: 最大值叠加结果 M(i,j)
        mean_img: Sigma-clipped 均值 μ̂(i,j)
        std_img: Sigma-clipped 标准差 σ̂(i,j)
        n_img: 每像素接受的帧数（背景掩码）
        min_frames: 背景像素识别的最小帧数阈值

    Returns:
        校正后的最大值图像
    """
    # 背景掩码：有足够稳定帧的像素
    bg_mask = n_img >= min_frames
    # log背景估算数量
    logger.info(
        f"NoiseEqualization: {np.sum(bg_mask) / np.sum(np.ones_like(n_img))} "
        f"background pixels with >= {min_frames} frames")

    if not np.any(bg_mask):
        raise ValueError(
            f"No background pixels found with >= {min_frames} frames")

    # Step 3: 估计经验偏移系数 ĉ_n^eff
    residual = (max_img - mean_img)[bg_mask]
    sigma_bg = std_img[bg_mask]
    valid = sigma_bg > 0

    logger.info(
        f"NoiseEqualization residual: {np.median(residual):.4f}, sigma_bg: "
        f"{np.median(sigma_bg):.4f}, valid pixels: {np.sum(valid)}/{len(residual)}"
    )

    if not np.any(valid):
        raise ValueError("No valid background pixels with σ > 0")

    c_n_eff = np.median(residual[valid] / sigma_bg[valid])

    # Step 4: 选取参考噪声水平 σ_ref
    sigma_ref = np.median(sigma_bg[valid])

    # Step 5: 校正
    corrected = max_img - (std_img - sigma_ref) * c_n_eff

    return corrected
