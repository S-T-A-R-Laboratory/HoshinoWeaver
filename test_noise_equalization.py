"""简单的噪声均匀化测试

生成合成数据验证算法原理
"""
import numpy as np
from hoshicore.component.noise_equalization import equalize_noise


def test_synthetic_radial_noise():
    """测试径向变化的噪声方差"""
    # 参数
    n_frames = 50
    height, width = 100, 100
    bg_value = 1000.0
    sigma_center = 10.0

    # 生成径向方差模式: σ(r) = σ0 * (1 + 0.5 * r²)
    y, x = np.ogrid[:height, :width]
    cy, cx = height // 2, width // 2
    r_sq = ((y - cy) ** 2 + (x - cx) ** 2) / (min(height, width) / 2) ** 2
    sigma_map = sigma_center * (1 + 0.5 * r_sq)

    # 生成帧：恒定背景 + 空间变化噪声
    frames = []
    for _ in range(n_frames):
        noise = np.random.randn(height, width) * sigma_map
        frame = bg_value + noise
        frames.append(frame)

    frames = np.array(frames)

    # 计算统计量
    mean_img = np.mean(frames, axis=0)
    std_img = np.std(frames, axis=0, ddof=1)
    max_img = np.max(frames, axis=0)
    n_img = np.full((height, width), n_frames, dtype=np.uint32)

    # 调试：检查中间值
    bg_mask = n_img >= 10
    residual = (max_img - mean_img)[bg_mask]
    sigma_bg = std_img[bg_mask]
    valid = sigma_bg > 0
    c_n_eff = np.median(residual[valid] / sigma_bg[valid])
    sigma_ref = np.median(sigma_bg[valid])

    print(f"帧数: {n_frames}")
    print(f"估计的 c_n_eff: {c_n_eff:.4f}")
    print(f"参考噪声水平 σ_ref: {sigma_ref:.4f}")
    print(f"σ 范围: [{std_img.min():.2f}, {std_img.max():.2f}]")

    # 应用校正
    corrected = equalize_noise(max_img, mean_img, std_img, n_img, min_frames=10)

    # 验证：未校正的残差应该与 σ 成正比
    residual_before = max_img - mean_img
    residual_after = corrected - mean_img

    # 检查原始残差的空间模式
    print(f"残差范围（校正前）: [{residual_before.min():.2f}, {residual_before.max():.2f}]")
    print(f"残差范围（校正后）: [{residual_after.min():.2f}, {residual_after.max():.2f}]")

    # 关键指标：原始残差的空间方差（未归一化）
    # 校正的目标是让所有像素的残差都接近同一水平
    spatial_std_raw_before = np.std(residual_before)
    spatial_std_raw_after = np.std(residual_after)

    print(f"\n原始残差的空间std（校正前）: {spatial_std_raw_before:.4f}")
    print(f"原始残差的空间std（校正后）: {spatial_std_raw_after:.4f}")
    print(f"改善比例: {spatial_std_raw_before / spatial_std_raw_after:.2f}x")

    # 归一化残差（用于检查理论）
    normalized_before = residual_before / std_img
    normalized_after = residual_after / std_img

    print(f"\n归一化残差的空间std（校正前）: {np.std(normalized_before):.4f}")
    print(f"归一化残差的空间std（校正后）: {np.std(normalized_after):.4f}")

    # 断言：原始残差的空间方差应显著降低
    assert spatial_std_raw_after < spatial_std_raw_before * 0.8, \
        "校正未能降低原始残差的空间方差"

    print("\n[PASS] 测试通过")


if __name__ == "__main__":
    test_synthetic_radial_noise()
