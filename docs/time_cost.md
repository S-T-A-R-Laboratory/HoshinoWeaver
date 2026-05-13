
# Time Cost

## Baseline

1000 images, 24MP, uint16

| Pipeline / Stage      | Description                                       | Mean Time |
| --------------------- | ------------------------------------------------- | --------- |
| detect_prepare_stream | 灰度/归一化/模糊/mask 等检测前处理                | 25.08 s   |
| detect_wavelet_stream | 小波重建阶段                                      | 44.11 s   |
| detect_extract_stream | threshold/morphology/contours/ellipse/post-filter | 87.01 s   |
| detect_stream         | 完整检测总耗时                                    | 112.71 s  |
| match_stream          | 检测 + 几何 + 匹配                                | 157.90 s  |
| homography_pipeline   | Homography 对齐完整流程总耗时                     | 164.86 s  |
| optimization_stream   | camera_model 参数优化阶段                         | 148.74 s  |
| remap_stream          | remap / 投影重采样阶段                            | 243.32 s  |
| camera_model_pipeline | camera_model 完整对齐流程总耗时                   | 398.46 s  |

---

## 加速分析 (CPU only)

### remap_stream — 243.32s (占 camera_model_pipeline 61%)

每帧 ~243ms。瓶颈: 24M 次 `unproject` (含迭代式 `cv2.undistortPoints`) + 24M 次 `project` (含 `cv2.projectPoints`) + cv2.remap。

| 手段 | 预估加速 | 复杂度 | 备注 |
|------|---------|--------|------|
| BA 固定内参 → unproject 结果缓存 | ~2× | 中 | 砍掉 N-1 次 unproject |
| 降采样 map (1/4 res 计算 → bicubic resize 回全分辨率) | map 计算部分 8-16× | 低 | map 是空间平滑场，误差 <0.1px |
| 二者组合 | 从 243s → **~30-40s** | — | |

### optimization_stream — 148.74s

每帧 ~149ms。scipy `least_squares` (LM, max_nfev=300)，11 参数 (rvec3 + focal1 + dist4 + focal2_or_shared + dist4_or_shared)。

| 手段 | 预估加速 | 复杂度 | 备注 |
|------|---------|--------|------|
| 解析 Jacobian (代替数值差分) | 2-3× | 中 | 需推导 reproject_error 关于 11 参数的导数 |
| warm start (前帧结果做当帧初值) | 迭代数 -30~50% | 低 | 时间相邻帧旋转连续 |
| 引入 BA 后: 逐帧初始阶段降为仅估 rvec (3 参数) | 3-5× per frame | 中 | focal/dist 交由 BA 全局联合优化，逐帧阶段只需粗估旋转初值 |
| 跳帧策略: 每 K 帧优化，中间帧插值 R | K× | 低 | 损失精度，适合预览 |

### detect_extract_stream — 87.01s

每帧 ~87ms。代码 `detection.py` 中:
- `findContours` 在全分辨率二值图上
- 逐轮廓 `fitEllipse` (Python for)
- **L129-139: 逐轮廓 drawContours + cv2.mean 循环** — 最大热点

| 手段 | 预估加速 | 复杂度 | 备注 |
|------|---------|--------|------|
| `cv2.connectedComponentsWithStats` 替代 contours+ellipse | 2-3× | 低 | 一次调用得到质心/面积/bbox |
| 消除 L129-139 循环: label map + regionprops 批量计算 | 显著 | 低 | 当前每帧数百次 drawContours |
| 在降采样图上完成全流程 (当前 wavelet 在 1/4 但 extract 回到全图) | 2-4× | 需验证 | 质心坐标缩放回原图 |

### detect_wavelet_stream — 44.11s

每帧 ~44ms。resize 1/4 → pywt.wavedec2 (db8, level 4) → 置零低频 → waverec2 → resize 回。

| 手段 | 预估加速 | 复杂度 | 备注 |
|------|---------|--------|------|
| DoG 替代小波 (已有 `_bandpass_dog` 实现) | 2-3× | 低 | 两次 GaussianBlur 即可，需验证检测效果 |
| 减少小波层数 (4→3) | ~30% | 低 | |
| 更小 resize_factor (0.125) | ~4× | 需测试 | 可能丢失弱星 |

### match_stream (匹配部分) — ~45s (157.9 - 112.71)

每帧 ~45ms。主要耗时:
- `extract_point_features`: **Python for 循环** (matching.py L49) 遍历每个星点
- `cdist` 特征距离矩阵 (~500×500)
- `fine_tune_transform`: while 循环 + 多次 findHomography

| 手段 | 预估加速 | 复杂度 | 备注 |
|------|---------|--------|------|
| `extract_point_features` L49 循环向量化 | 2-5× on feature extraction | 中 | 几何特征计算可 fully vectorize |
| 参考帧 features 缓存 | 已实现 (GeometryView.features cached_property) | — | |
| `fine_tune_transform` 确定性化 (消除随机 retry) | 减少最坏情况 | 低 | |

### detect_prepare_stream — 25.08s

每帧 ~25ms。灰度+GaussianBlur(9×9)+归一化+mask。已较快，优化优先级低。

---

## 优先级排序

| 优先级 | 优化项 | 预估节省 | 复杂度 | 依赖 |
|--------|--------|---------|--------|------|
| P0 | remap 降采样插值 | ~200s | 低 | 无 |
| P1 | detect_extract connectedComponents 重写 | ~40-50s | 低 | 无 |
| P2 | BA + unproject 缓存 + optim 简化 | ~160s | 中 | 架构改动 |
| P3 | DoG 替代 wavelet | ~30s | 低 | 验证检测效果 |
| P4 | extract_point_features 向量化 | ~20-30s | 中 | 无 |
| P5 | optimization warm start | ~50-70s | 低 | 无 |

全部实施后 camera_model_pipeline 预估: **398s → ~100-130s** (约 3-4× 加速)。
P0+P1+P3 无需架构改动，合计节省 ~270s。
