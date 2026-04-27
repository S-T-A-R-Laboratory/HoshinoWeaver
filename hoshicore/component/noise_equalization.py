"""噪声均匀化模块

用于消除最大值叠加中因镜头校正导致的空间不均匀噪声伪影。
详见 docs/noise-equalization.md
"""
import cv2
import numpy as np
from loguru import logger
from numpy.typing import NDArray
from scipy.stats import norm as _norm


def _ransac_ratio(x: NDArray,
                  y: NDArray,
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
            best_slope = np.dot(x[inliers], y[inliers]) / np.dot(
                x[inliers], x[inliers])

    logger.debug(
        f"RANSAC c_n_eff: slope={best_slope:.4f}, "
        f"inliers={best_inlier_count}/{len(x)} ({best_inlier_count/len(x)*100:.1f}%)"
    )
    return float(best_slope)


def fill_local_mean(img, mask: NDArray[np.bool], kernel_size=21):
    img_f = img
    valid = (~mask).astype(np.float32)
    kernel = np.ones((kernel_size, kernel_size))

    sum_valid = cv2.filter2D(img_f * valid,
                             -1,
                             kernel,
                             borderType=cv2.BORDER_REFLECT)
    count_valid = cv2.filter2D(valid,
                               -1,
                               kernel,
                               borderType=cv2.BORDER_REFLECT)

    mean_local = sum_valid / np.maximum(count_valid, 1e-8)
    out = img_f.copy()
    out[mask] = mean_local[mask]
    return out


def equalize_noise(max_img: NDArray,
                   mean_img: NDArray,
                   std_img: NDArray,
                   n_img: NDArray,
                   estimate_method: str = "median",
                   minus_only: bool = False,
                   top_fraction: float = 0.02,
                   sigma_reject: float = 3.0,
                   highlight_preserve: float = 0.9) -> NDArray:
    """对最大值叠加图像应用噪声均匀化校正。

    核心公式: M_corrected = M - (σ̂ - σ_ref) · ĉ_n^eff

    Args:
        max_img: 最大值叠加结果 M(i,j)
        mean_img: Sigma-clipped 均值 μ̂(i,j)
        std_img: Sigma-clipped 标准差 σ̂(i,j)
        n_img: 每像素接受的帧数（背景掩码）
        top_fraction: 背景像素识别阈值：n_img 中前 top_fraction 分位数
                      （例如 0.02 表示取帧数最多的前 2% 作为阈值）
        sigma_reject: 标准差的标准差拒绝倍率
        highlight_preserve: 高光保护比率

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
        f"background pixels={np.sum(bg_mask)} ({np.mean(bg_mask)*100:.2f}%)")

    if not np.any(bg_mask):
        logger.warning(
            f"Skip equalize_noise processing because "
            f"no background pixels found with top_fraction={top_fraction}.")
        return max_img

    # Step 3: 估计经验偏移系数 ĉ_n^eff
    residual = (max_img - mean_img)[bg_mask]
    sigma_bg = std_img[bg_mask]
    valid = sigma_bg > 0

    logger.info(
        f"NoiseEqualization residual: {np.median(residual):.4f}, sigma_bg: "
        f"{np.median(sigma_bg):.4f}, valid pixels: {np.sum(valid)}/{len(residual)}"
    )

    if not np.any(valid):
        logger.warning("Skip equalize_noise processing because "
                       "no valid background pixels with σ > 0. "
                       "Maybe all images have same values?")
        return max_img

    r_valid = residual[valid]
    s_valid = sigma_bg[valid]

    if estimate_method == "median":
        c_n_eff = float(np.median(r_valid / s_valid))
    elif estimate_method == "ransac":
        c_n_eff = _ransac_ratio(s_valid, r_valid)
    else:
        raise ValueError(f"unsupport estimate method")

    # Step 4: 选取参考噪声水平 σ_ref （如果不重建噪声，则置0）
    sigma_ref = 0 if minus_only else np.median(s_valid)

    # step4.5: 标准差排异(by channel)
    squeeze_std = std_img.reshape((-1, 3))
    mean_std = np.mean(squeeze_std, axis=0)
    std_std = np.std(squeeze_std, axis=0)
    mask = (std_img > (mean_std + sigma_reject * std_std)[None, None, ...])
    filled_std_img = fill_local_mean(std_img, mask, kernel_size=21)

    # step4.x: 高光保护：亮度高于指定范围的，方差线性下降，直至255惩罚到0。
    hp = highlight_preserve
    fix_strength = (max_value * hp - max_img).clip(max=0) / (max_value *
                                                             (1 - hp)) + 1
    fixed_std_img = fix_strength * filled_std_img

    # Step 5: 校正
    corrected = max_img - (fixed_std_img - sigma_ref) * c_n_eff
    corrected = np.clip(corrected, a_min=0, a_max=max_value)

    return corrected


def compute_adaptive_n_sigma(n_frames: int,
                             target_fpr: float = 0.01) -> float:
    """根据帧数计算自适应 sigma 阈值。

    选取 n_sigma 使得：在 n_frames 帧中，单个背景像素至少一帧
    误超阈值的概率不超过 target_fpr。

    Args:
        n_frames: 总帧数。
        target_fpr: 目标每像素误检率（默认 0.01）。

    Returns:
        自适应 n_sigma，下界 3.0。
    """
    return max(3.0, float(_norm.ppf(1.0 - target_fpr / n_frames)))


def threshold_max_merge(
    frame: NDArray,
    mean_img: NDArray,
    std_img: NDArray,
    result: NDArray,
    n_sigma: float,
    weight: float | None = None,
    morph_kernel: NDArray | None = None,
) -> None:
    """单帧 threshold-max 归约（就地更新 result）。

    保留 frame 中显著高于背景（mean + n_sigma * std）的像素，
    用其（可选加权后的）值与 result 取最大值。
    背景区域始终保持 mean_img 的值。

    Args:
        frame: 当前帧图像 (H, W, C) float64。
        mean_img: sigma-clipped 均值图像。
        std_img: sigma-clipped 标准差图像。
        result: 累积结果图像（就地更新）。
        n_sigma: 阈值倍率。
        weight: 可选渐入渐出权重（标量）。
        morph_kernel: 形态学开运算核，用于清除孤立噪点。None 则跳过。
    """
    mask = frame > (mean_img + n_sigma * std_img)

    if morph_kernel is not None:
        if mask.ndim == 3:
            for c in range(mask.shape[2]):
                mask[:, :, c] = cv2.morphologyEx(
                    mask[:, :, c].view(np.uint8),
                    cv2.MORPH_OPEN, morph_kernel).view(bool)
        else:
            mask = cv2.morphologyEx(
                mask.view(np.uint8),
                cv2.MORPH_OPEN, morph_kernel).view(bool)

    if weight is not None:
        signal = frame * weight
    else:
        signal = frame

    np.maximum(result, np.where(mask, signal, mean_img), out=result)
