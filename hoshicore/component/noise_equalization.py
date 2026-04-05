"""噪声均匀化模块

用于消除最大值叠加中因镜头校正导致的空间不均匀噪声伪影。
详见 docs/noise-equalization.md
"""
import numpy as np
from numpy.typing import NDArray
from loguru import logger


def _ransac_ratio(x: NDArray, y: NDArray,
                  n_iter: int = 100,
                  inlier_thresh: float = 2.0) -> float:
    """用 RANSAC 鲁棒估计 y/x 的比值（即过原点的斜率）。

    每次随机抽一个样本点，计算斜率候选值，统计满足
    |y_i - slope * x_i| < inlier_thresh * MAD 的内点数，
    返回内点最多时对应的斜率（用内点重新做最小二乘）。

    Args:
        x: 分母向量（sigma_bg）
        y: 分子向量（residual）
        n_iter: RANSAC 迭代次数
        inlier_thresh: 内点判定阈值（以 MAD 为单位）

    Returns:
        鲁棒估计的斜率 ĉ_n^eff
    """
    rng = np.random.default_rng(42)
    best_slope = np.median(y / x)  # fallback
    best_inlier_count = 0

    residuals_all = y / x
    mad = np.median(np.abs(residuals_all - np.median(residuals_all)))
    if mad == 0:
        return best_slope

    for _ in range(n_iter):
        idx = rng.integers(0, len(x))
        slope_candidate = y[idx] / x[idx]
        inliers = np.abs(y - slope_candidate * x) < inlier_thresh * mad * x
        if inliers.sum() > best_inlier_count:
            best_inlier_count = inliers.sum()
            # 内点最小二乘：min ||y - s*x||^2 → s = (x·y)/(x·x)
            best_slope = np.dot(x[inliers], y[inliers]) / np.dot(x[inliers], x[inliers])

    logger.debug(
        f"RANSAC c_n_eff: slope={best_slope:.4f}, "
        f"inliers={best_inlier_count}/{len(x)} ({best_inlier_count/len(x)*100:.1f}%)"
    )
    return float(best_slope)


def equalize_noise(max_img: NDArray,
                   mean_img: NDArray,
                   std_img: NDArray,
                   n_img: NDArray,
                   top_fraction: float = 0.02) -> NDArray:
    """对最大值叠加图像应用噪声均匀化校正。

    核心公式: M_corrected = M - (σ̂ - σ_ref) · ĉ_n^eff

    Args:
        max_img: 最大值叠加结果 M(i,j)
        mean_img: Sigma-clipped 均值 μ̂(i,j)
        std_img: Sigma-clipped 标准差 σ̂(i,j)
        n_img: 每像素接受的帧数（背景掩码）
        top_fraction: 背景像素识别阈值：n_img 中前 top_fraction 分位数
                      （例如 0.02 表示取帧数最多的前 2% 作为阈值）

    Returns:
        校正后的最大值图像
    """
    
    max_value = np.max(max_img)
    
    # 背景掩码：帧数位于前 top_fraction 分位的像素
    threshold = np.quantile(n_img, 1.0 - top_fraction)
    bg_mask = n_img >= threshold
    logger.info(
        f"NoiseEqualization: top_fraction={top_fraction*100:.1f}%, "
        f"threshold={threshold:.1f} frames, "
        f"background pixels={np.sum(bg_mask)} ({np.mean(bg_mask)*100:.2f}%)"
    )

    if not np.any(bg_mask):
        raise ValueError(
            f"No background pixels found with top_fraction={top_fraction}")

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

    r_valid = residual[valid]
    s_valid = sigma_bg[valid]

    # 两种估计方式，保留 median 以便对比
    c_n_eff_median = float(np.median(r_valid / s_valid))
    c_n_eff_ransac = _ransac_ratio(s_valid, r_valid)

    logger.info(
        f"c_n_eff  median={c_n_eff_median:.4f}  RANSAC={c_n_eff_ransac:.4f}  "
        f"diff={abs(c_n_eff_ransac - c_n_eff_median):.4f}"
    )
    c_n_eff = c_n_eff_ransac

    # Step 4: 选取参考噪声水平 σ_ref
    sigma_ref = np.median(s_valid)

    # Step 5: 校正
    corrected = max_img - (std_img - sigma_ref) * c_n_eff
    corrected = np.clip(corrected, a_min=0, a_max=max_value)

    return corrected
