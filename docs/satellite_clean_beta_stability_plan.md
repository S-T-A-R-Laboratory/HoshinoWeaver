# Satellite Clean / Matching 稳定性改造计划（beta 期）

本文用于记录 `SatelliteCleanOp` 在少星点场景下的失稳现象、原因判断，以及适合 beta 阶段落地的低风险改造方案。目标不是引入新的几何模型，而是在**不改变对外接口、不引入 breaking change** 的前提下，提高城市/地景占比高场景下的成功率，并降低“求解成功但结果错误”的风险。

## 背景与结论

当前卫星线去除链路为：

```text
startrail.meta.yaml
  -> satellite_clean.yaml
  -> SatelliteCleanOp
  -> make_geometry()
  -> detect_star_points()
  -> match_star_pairs()
  -> find_initial_match()
  -> fine_tune_transform()
  -> 相邻帧 homography 串接
  -> 滑窗对齐 median
```

从 `logs/hnw_1.0.0-beta.0_win_gui.log` 可见：

- 本次序列共 24 帧，开启 `enable_satellite_clean=true`，窗口为 `3`。
- 最终有效星点数通常在 `34 ~ 43` 之间，平均约 `39.8`。
- `find_initial_match()` 过滤前平均 `9.09` 对、过滤后平均 `6.39` 对，平均保留率约 `69%`。
- 相邻帧 homography 估计中，出现了 `11` 次失败。

结论：

- 问题不是单一阈值，而是当前匹配流程整体偏向“星点充足”场景。
- 少星点素材会同时触发以下薄弱点：星点筛选过严、初始配对数偏少、种子采样不自适应、收敛门槛过高、结果验收不足。
- beta 阶段建议优先做**渐进式放宽 + 强验收**，而不是直接切换到新的旋转/BA 模型。

## 现象拆解

### 1. 星点检测阶段丢弃比例较高

在 `detect_star_points()` 中，原始 contour 数量经常在 `133 ~ 148`，但 `Final star points` 仅剩 `34 ~ 43`。这意味着后续匹配是在相对稀疏的点集上完成的。

这本身不一定是 bug，但对城市场景而言意味着：

- 参与匹配的点数已经很接近“描述子/配准算法的下限”。
- 任意一步再损失 2~4 个点，就可能从“勉强可解”跌到“不可解”。

### 2. `apply_threshold_filter` 有影响，但不是唯一主因

`find_initial_match()` 中：

- 先做互选最近邻 + 固定 `30` 分位距离阈值；
- 再做 `apply_threshold_filter`：
  - 球面夹角阈值 `theta_th`
  - 像素距离阈值 `dist_th`

日志统计显示，过滤前后平均变化为：

- 平均 `9.09 -> 6.39`
- 平均减少 `2.7` 对
- 平均保留率 `69%`

判断：

- 这一步的影响是**中等偏大**。
- 像素距离阈值 `dist_th = max_coord * 0.3` 对大图通常并不严格，主要裁剪来源更可能是 `theta_th`。
- 对星点充足场景，它有价值；对少星点场景，它可能把边缘可解样本直接裁到不可解。
- 固定 `30` 分位本身也偏保守：它适合“候选很多、优先保精度”的场景，但对仅有 `~40` 个最终星点的样本，往往会在进入 `apply_threshold_filter` 前就把候选压得过低。

### 3. `fine_tune_transform()` 当前假设不适合少星点

当前问题点主要有四个：

1. 收敛条件使用 `0.6 * min(len(pts1), len(pts2))`
2. 固定使用 `20` 个随机种子点
3. 采样不是围绕 `init_pair_idx` 的“唯一匹配对规模”自适应设计
4. `findHomography()` 只看“是否算出来”，缺少结果可靠性验收

这导致少星点时容易出现：

- `unique_pairs < 4`，根本不满足单应求解下限；
- `len(ind)` 已经不差，但因为达不到 `0.6 * min(feature_count)` 被判失败；
- homography 虽然返回了，但几何上明显不可信。

## 改造目标

beta 期目标限定为：

1. 不改 DAG 接口与 YAML 配置结构。
2. 不引入新的相机/旋转模型依赖。
3. 尽量不改变星点充足场景的行为。
4. 对少星点场景启用“渐进式放宽”。
5. 对所有场景增加“结果可信度验收”。

## 非目标

以下内容暂不在本轮范围内：

- 切换到旋转模型求解替代自由 homography
- 引入 bundle adjustment / 光束法平差
- 改变 `SatelliteCleanOp` 的输入输出接口
- 为 GUI 暴露大量新超参

## 具体改动清单

### A. `find_initial_match()`：把阈值过滤改成渐进式

目标：保留当前多星点场景的保守性，同时避免少星点场景被一次过滤直接打死。

建议改动：

1. 保留“互选最近邻”主逻辑不变。
2. 将固定 `30` 分位距离阈值改为**自适应分位 / 保底候选数**策略，建议：
   - 星点充足时仍可使用较保守分位（如 `30`）。
   - 星点偏少时放宽到 `40 ~ 60` 分位。
   - 无论分位结果如何，都应保证至少保留 `min_keep_pairs`（建议 `8 ~ 10`）个距离最优的 mutual pairs。
3. `apply_threshold_filter=True` 时，先执行当前过滤逻辑。
4. 增加回退条件：
   - 若过滤后 `unique_pairs < 4`，直接回退到未过滤结果。
   - 若过滤后保留率过低（例如 `< 0.5`）且过滤前配对总数本就不多（例如 `< 10`），回退到未过滤结果。
5. 为日志补充：
   - `distance_percentile`
   - `before_pairs`
   - `after_pairs`
   - `kept_ratio`
   - `fallback_to_unfiltered`

建议原则：

- 多星点时继续使用较保守阈值与过滤后结果。
- 少星点时，过滤应是“优先选项”，不是硬门槛。

### B. `fine_tune_transform()`：改成“低门槛通过 + 强验收兜底”的收敛策略

目标：避免把“少星点但已足够稳定”的结果错误判失败，同时把主要风险控制放到后置几何验收上。

建议改动：

1. 将成功判据从：

```text
len(ind) >= 0.6 * min(len(pts1), len(pts2))
```

调整为“先满足最低可解条件，再交给后续验收”的策略。

```text
inlier_count >= 4
```

说明：

- 这里的 `4` 不是“最终可信”的充分条件，只是“允许进入验收”的最低门槛。
- 是否最终接受，应由 F1/F2/F3/F4 共同决定，而不是再用 `init_pair_count` 的比例硬判。

建议做法：

1. `inlier_count >= 4` 时允许生成候选 `H`
2. 对候选 `H` 执行：
   - F1: 重投影误差验收
   - F2: inlier 空间覆盖验收
   - F3: 单应矩阵形变合理性验收
   - F4: 与前后 link 的时间连续性验收
3. 只有全部通过时，才接受该 link

这样做的原因：

- 对相邻连续帧这种强先验场景，`4` 个高质量、低重投影误差、空间分布不退化的 inlier，往往已足够支持一个可接受的局部单应估计。
- 真正需要严格把关的不是 `inlier_ratio`，而是“这个解在几何上是否可信”。
- 因此更适合把主要门槛放在 F1/F3/F4，而不是前置在 `init_pair_count` 比例上。

补充建议：

- `inlier_ratio` 仍建议保留为日志与诊断指标。
- 当 `init_pair_count` 很高但 `inlier_ratio` 很低时，可以作为 warning 或降权信号，但不建议作为主要拒绝条件。

### C. `fine_tune_transform()`：修正采样空间，并改为自适应、无放回采样

目标：避免在错误的采样空间中重复命中少量候选点；减少重复点采样、避免 `20` 点硬编码对稀疏场景的系统性失败。

建议改动：

1. 明确采样对象应为 `init_pair_idx` 本身，而不是“先在图像坐标空间随机撒点，再映射到最近 pair”。
2. 基于 `init_pair_idx` 先做唯一化统计。
3. 计算：

```text
unique_pair_count = 去重后的匹配对数量
```

4. 规则建议：
   - `unique_pair_count < 4`：直接按失败处理。
   - `4 <= unique_pair_count <= 6`：采样全部或大部分唯一点对。
   - `unique_pair_count > 6`：无放回采样，`sample_size = min(12, unique_pair_count)`。
5. 采样来源直接基于 `init_pair_idx`，不要通过“随机平面点 -> 最近邻映射”间接采样。
6. 保留 early stop，但改成**验收驱动**而不是“点数比例驱动”：
   - 不再使用 `0.6 * min(len(pts1), len(pts2))` 作为停止条件。
   - 每轮若得到 `inlier_count >= 4` 的候选 `H`，立即执行 F1/F2/F3（F4 暂不在本函数内实现）。
   - 一旦候选通过验收，则直接 early stop。
   - 若未通过验收，则继续下一轮，直到达到 `max_trials`。
   - 同时保留 `best_candidate` 仅用于日志和失败分析，不自动放行未通过验收的候选。
7. 若需要保留“空间覆盖”这一原始设计意图，应将其作为 **pair 子集选择约束**，而不是主采样空间。可选方案包括：
   - 先按图像网格分桶，再从每桶选择质量较高的 pair
   - 在 `init_pair_idx` 上做 farthest-point sampling
   - 先按匹配距离排序，再做带覆盖约束的优先采样
8. 日志建议补充：
   - `unique_pair_count`
   - `sample_size`
   - `sampling_mode`
   - `coverage_score`（若实现空间覆盖约束）
   - `early_stop_triggered`
   - `accepted_iteration`

说明：

- 原始做法可能试图获得“更好的空间分布”，这个出发点本身合理；但它把“空间覆盖”实现成了“图像平面随机采样”，在少星点场景中容易因最近邻映射而重复命中相同 pair。
- 对相邻连续帧，初始匹配对通常已是较强先验，直接在候选对上采样比当前方式更稳定，也更容易保证采样数与可用 pair 数一致。
- `unique_pair_count < 4` 时不建议任何 fallback 到 affine 或 translation，本轮按失败处理更符合 beta 风险控制。

### D. `k=15` 改为自适应邻域规模

目标：降低少星点时局部描述子的脆弱性，同时保留多星点时的区分能力。

建议改动入口：

- `match_star_pairs(..., k=15, ...)`
- 或更早在 `extract_point_features()` 的调用处按星点数量决定 `k`

建议规则：

```text
star_count = min(len(ref_vectors), len(src_vectors))

if star_count < 30:  k = 6
elif star_count < 45: k = 8
elif star_count < 70: k = 12
else:                k = 15
```

原则：

- 少星点时优先保证局部关系稳定；
- 星点足够时再使用更强的判别性。

### E. `detect_star_points()`：仅在少星点时渐进放松末端筛选

目标：减少 `star points detected -> Final star points` 之间的过度损失，但避免显著增加假星点。

不建议大改前面的波段滤波与 contour 检测，beta 期更适合只放松**最终筛选阶段**。

建议流程：

1. 先按当前规则得到 `valid_stars`。
2. 若 `final_star_count >= target_min_final_stars`，直接使用当前结果。
3. 若不足，再逐步放松：
   - 放松 `eccentricities < 0.8` 到 `0.88`
   - 将 area/intensity 的 `20` 分位筛选下调到 `10`
   - 若仍不足，再允许返回一个“较宽松版本”

建议目标值：

- `target_min_final_stars = 50`

注意：

- 放宽只在少星点路径启用。
- 每次放宽后都要打印日志，便于后续对误匹配率做回归检查。

### F. 增加 homography 结果可信度校验

目标：解决“求解成功但其实不正确”的情况。

建议在 `fine_tune_transform()` 得到候选 `H` 后，新增统一验收步骤。

#### F1. 重投影误差

对最终 inliers 计算：

- `median_reproj_error`
- `p90_reproj_error`

建议阈值（相邻连续帧）：

- `median <= 1.0 px`
- `p90 <= 2.0 px`

如果超出，则判失败。

#### F2. inlier 空间覆盖

检查 inlier 点在图像中的空间分布，避免全部挤在局部区域。

可用指标：

- inlier 凸包面积 / 图像面积
- 或覆盖网格的格子数

建议原则：

- 若只覆盖极小区域，则降低置信度或直接判失败。

#### F3. 形变合理性约束

对“连续相邻帧”这一使用场景，单应矩阵不应出现明显非物理形变。

建议验收项：

- 不允许镜像翻转
- 角点变换后的面积比例应接近 `1`
- projective 项 `H[2,0]`、`H[2,1]` 不应异常大

建议作为经验阈值起步：

- `area_ratio` 在 `[0.9, 1.1]` 或稍宽一些的区间
- `abs(H[2,0])`、`abs(H[2,1])` 小于一个按图像尺度归一化后的阈值

这里不追求一次定死数学标准，beta 期可以先记录日志并温和拒绝明显异常值。

#### F4. 时间连续性

因为这里处理的是相邻帧，变换量应平滑。

建议在 `SatelliteCleanOp` 中维护最近一次 accepted `H` 的简要统计量，例如：

- 平移量
- 旋转近似量
- 面积比例

若当前 `H` 相对上一条 accepted `H` 突然剧烈跳变，则：

- 记 warning
- 直接判坏或降级为 link broken

这一步尤其有助于过滤“RANSAC 偶然命中假解”的情况。

## 推荐落地顺序

为降低 beta 风险，建议按以下顺序拆分提交：

### 第 1 步：只补日志与验收，不改匹配策略

包含：

- `before/after pair count`
- `unique_pair_count`
- `kept_ratio`
- reproj error
- area ratio
- projective magnitude
- 是否触发 fallback / reject

目的：

- 先把问题看清楚，避免盲调。

### 第 2 步：修改 `fine_tune_transform()` 的采样与成功判据

包含：

- `unique_pair_count < 4` 直接失败
- 采样对象切换为 `init_pair_idx`
- 无放回自适应采样
- 成功判据改为 `inlier_count >= 4` + F1/F2/F3 强验收
- 增加“验收通过即 early stop”的停止策略

说明：

- F4 时间连续性更适合后续在 `SatelliteCleanOp` 或更高层 link 管理处补充，不建议塞进 `matching.py` 的本轮改动中。
- `best_candidate` 可以记录，但不应替代正式验收通过的候选。

这是最核心、收益最高、风险也相对可控的一步。

### 第 3 步：引入 `apply_threshold_filter` 的渐进回退

包含：

- 过滤后不足 4 对则回退
- 稀疏场景下保留率过低则回退

这一步能显著改善边缘样本。

### 第 4 步：按星点数自适应 `k`

包含：

- 动态 `k`
- 补相关日志

### 第 5 步：少星点路径放松最终星点筛选

这一步风险略高，因为可能引入更多假点。建议排在后面，并结合日志做回归。

## 测试与验证建议

### 1. 日志对照

至少记录并比较以下指标：

- homography failure 次数
- `before/after threshold filter` 的 pair 数
- `unique_pair_count`
- `inlier_count`
- `inlier_ratio`
- `sample_size`
- `sampling_mode`
- reproj error
- reject 原因分布

### 2. 样本分组

建议至少准备三类样本：

1. 星点丰富的纯天空序列
2. 城市/地景占比较高、星点偏少序列
3. 边缘困难样本（薄云、轻微抖动、局部遮挡）

目标：

- 第 1 类行为尽量不退化
- 第 2 类成功率明显提升
- 第 3 类即便失败，也要尽量“安全失败”，避免错对齐

### 3. 结果判读

对 `SatelliteCleanOp`，失败未必比错误更糟；更重要的是：

- 少错杀可接受
- 少误接收更重要

换言之，beta 期策略应偏向：

```text
宁可 link broken，也不要明显错配后进入 median
```

## 建议新增或调整的内部超参

为避免 GUI 复杂化，建议先作为内部常量或私有配置，不暴露给用户：

- `min_unique_pairs_for_homography`
- `sparse_match_filter_fallback_ratio`
- `target_min_final_stars`
- `min_keep_pairs`
- `reproj_median_threshold_px`
- `reproj_p90_threshold_px`
- `homography_area_ratio_range`
- `projective_term_threshold`

待 beta 验证稳定后，再决定是否有必要外显到 YAML 或 GUI。

## 最终建议

当前最值得优先实施的是：

1. `fine_tune_transform()` 的采样策略重写
2. 成功判据改为面向初始匹配对
3. homography 结果可信度验收
4. `apply_threshold_filter` 的渐进回退

`detect_star_points()` 的放宽建议放在后手，因为它收益可能存在，但引入假点的风险也更高。

整体思路应当是：

```text
少星点时，放宽“搜寻阶段”；
所有场景下，收紧“验收阶段”。
```

这比单纯全局放宽阈值更符合当前 beta 阶段的风险控制要求。
