# 前端术语与文案准则

本文档规定用户界面中的术语选择、描述文案撰写原则，以及前后端术语映射关系。
所有 `.ui.yaml` 文件的 `label` / `description` 字段应遵循本文档。

---

## 1. 命名原则

### 原则 A — 用效果命名，不用机制命名；成熟算法保留学名

- 内部工程概念（route, bypass, prune, flatten）不向用户暴露，改用描述效果的词汇
- 成熟的、已成为领域共识的算法名称保留原名：Sigma 裁剪、Huber 均值、中位数等
- 算法名之后可附加简短的效果副标题辅助理解

| 示例 | 做法 |
|------|------|
| Sigma Clip | 保留"Sigma 裁剪"，description 补充适用场景 |
| Huber Mean | 保留"Huber 均值"，description 补充与均值/中位数的对比 |
| bypass（穿透模式） | 用户面呈现为"跳过此步骤，数据直接传递到下一步" |
| route（路由） | 用户面呈现为"处理方式"或"算法选择" |

### 原则 B — description 回答"我该怎么选"而非"这是什么"

description 的核心功能是帮助用户做决策。应包含：
1. **适用场景**（何时该选/何时该调）
2. **与其他选项的对比**（优劣取舍）
3. **推荐用法**（默认值是否够用）

| 差 | 好 |
|----|-----|
| "迭代剔除异常像素后求均值" | "推荐用于有卫星/飞机经过的场景。比均值更干净，比中位数更锐利" |
| "基于 Huber 损失的稳健均值估计" | "自动降低异常像素权重。比 Sigma 裁剪更平滑，适合轻微光污染场景" |
| "高信噪比平均叠加" | "最快最锐利，但不排除异常帧。适合已手动筛选过的干净素材" |

### 原则 C — 给出推荐值和边界感

每个可调参数的 description 应当包含：
- **推荐范围**（如"推荐 2.0–3.5"）
- **调节方向的直觉**（如"值越小排除越激进"）
- 或明确"通常保持默认即可"

```yaml
# 好的示例
rej_high:
  label: "高亮度排异阈值 (σ)"
  description: "高于均值多少个标准差视为异常并排除。值越小排除越多。推荐 2.5–3.5"
```

### 原则 D — 开关参数说明关闭后果

任何 `enable_*` 开关的 description 必须说明关闭时的行为效果：

```yaml
# 好的示例
enable_ground:
  label: "叠加地面图像"
  description: "同时叠加地面区域。关闭后仅输出天空部分，地面区域为黑色"

enable_satellite_clean:
  label: "卫星线去除"
  description: "使用滑动窗口中位数法去除卫星/飞机轨迹。关闭后保留所有轨迹"
```

---

## 2. 前后端术语映射

| 后端术语 | 面向用户的术语 | 呈现方式 | 说明 |
|---------|--------------|---------|------|
| `route` / `route_key` | 处理方式 / 算法选择 | 选项卡 (tabs) 或下拉 (select) | 根据上下文选择合适名词 |
| `route_configs` | （选中算法的）参数 | 选项卡下方的折叠参数组 | 组标题格式："算法名 · 参数" |
| `enable` + `bypass` | 启用/跳过 | 开关 | description 说明"跳过后数据直通下一步" |
| `enable` + `prune` | 启用/关闭 | 开关 | description 说明"关闭后此功能不执行" |
| `configs` | 参数设置 | 表单控件 | 不向用户暴露 configs 一词 |
| `inputs` | 输入文件 | 文件选择器 | — |
| `outputs` | 输出设置 | 输出面板 | — |
| `SubDAG` | （不暴露） | — | 纯内部概念 |
| `flatten` / `wiring` | （不暴露） | — | 纯内部概念 |
| `node` | 处理步骤 | — | 仅在进度显示中可能出现 |
| `sequence` | 图像序列 | — | 在进度/状态显示中使用 |

---

## 3. 条件可见性准则

当 UI 元素的相关性取决于其他参数的值时，应使用条件渲染（`visible_when`），避免呈现无效/不相关的控件。

**原则**：用户在任意时刻看到的控件应当全部是"当前有效的"。不呈现"改了也没用"的参数。

### 语法

在 `.ui.yaml` 的 `configs` 或 `routes` 条目中添加 `visible_when` 字段：

```yaml
visible_when: { key: "<config_key 或 route_key>", eq: <期望值> }
```

- `key`：引用同一 ui.yaml 中的另一个 config 键名或 route 键名
- `eq`：期望值（bool / string）。当 key 当前值 == eq 时，控件可见
- `neq`（可选）：不等于。当 key 当前值 != neq 时，控件可见

### 行为

- 隐藏时控件值保留，`collect_configs` 仍传递用户设定值
- route_configs 的「选中才渲染」是内置行为，无需额外声明
- `visible_when` 用于跨 config/route 之间的联动

### 适用场景

| 场景 | 示例 |
|------|------|
| 开关关闭时隐藏相关参数 | `sat_window_size` 依赖 `enable_satellite_clean` |
| 开关关闭时隐藏整个路由选项卡 | `ground_stacker` 依赖 `enable_ground` |
| 路由选项为特定值时才显示参数 | `bias_master_path` 依赖 `bias_stacker == "master"` |

### 示例

```yaml
# 开关联动：关闭去卫星线时，隐藏窗口大小参数
configs:
  sat_window_size:
    visible_when: { key: "enable_satellite_clean", eq: true }

# 路由联动：仅选择"已有主帧"时显示路径选择器
configs:
  bias_master_path:
    visible_when: { key: "bias_stacker", eq: "master" }

# 开关联动整个路由：关闭叠加地面时隐藏地面算法选项卡
routes:
  ground_stacker:
    visible_when: { key: "enable_ground", eq: true }
```

---

## 4. 算法选项描述模板

每个算法选项的 description 应覆盖三要素：

```
{核心特点}。{适用场景}。{与相邻选项的对比取舍}
```

示例：

| 算法 | description |
|------|------------|
| 均值 | "简单平均，最快最锐利。适合已手动筛选过的干净素材，不排除异常帧" |
| Sigma 裁剪 | "多轮迭代排除异常像素后求平均。推荐用于有卫星/飞机/云层的场景" |
| Huber 均值 | "自动降低异常像素权重，无需设定排除阈值。比 Sigma 裁剪更平滑，适合轻微异常" |
| 中位数 | "逐像素取中位数，天然排除异常值。无需额外参数，但速度较慢、锐度略低" |
| 最大值 | "逐像素取最大值，保留所有星轨。不做降噪处理" |
| 最大值混合 | "最大值叠加 + 噪声均衡，保留轨迹的同时平衡帧间亮度差异" |

---

## 5. 预设系统（规划中）

为降低新手使用门槛，计划支持场景预设：

- 用户选择预设后自动填入推荐参数
- 高级参数默认折叠，可手动展开微调
- 预设不锁定参数，仅作为初始值建议

具体 ui.yaml 语法和 panel_builder 实现待设计确认。

---

## 6. 文案风格

- 使用简洁中文，避免长句
- 术语首次出现时可在括号中给出英文原名（如"Sigma 裁剪 (Sigma Clipping)"），后续使用中文
- 数值参数描述中使用半角数字和符号
- 不使用 emoji
- label 控制在 8 个汉字以内（特殊情况可放宽到 12）
- description 控制在 40 字以内（tooltip 渲染空间有限）

---

## 7. 参数级前后端术语对照表

本表记录所有 `.meta.yaml` 中的后端键名与 `.ui.yaml` 中面向用户的 label 的对应关系。
新增管线或参数时应查阅此表以保持一致。

### 7.1 共享参数（跨管线复用）

| 后端键名 | 前端 label | 可见性 | 备注 |
|---------|-----------|--------|------|
| `int_weight` | — | 隐藏 | 内部优化参数 |
| `enable_exif` | 写入 EXIF | 开关 | 关闭后输出不含 EXIF |
| `output_dtype` | — | 隐藏（输出面板内） | |
| `output_filename` | — | 隐藏（输出面板内） | |
| `exif_reduce_type` | — | 隐藏 | 合并策略 |
| `loader_type` | — | 隐藏 | |
| `loader_configs` | — | 隐藏 | |
| `jpg_quality` | — | 隐藏（输出面板内） | |
| `png_compressing` | — | 隐藏（输出面板内） | |

### 7.2 叠加算法路由选项

| 后端 option 值 | 前端 label | 适用管线 |
|---------------|-----------|---------|
| `mean` | 均值 | stack, sky_ground, calibration |
| `sigma_clip` | Sigma 裁剪 | stack, sky_ground, calibration |
| `median` | 中位数 | stack, sky_ground, calibration |
| `huber_mean` | Huber 均值 | stack, sky_ground, calibration |
| `max` | 最大值 | sky_ground |
| `max_mix` | 最大值混合 | sky_ground |

### 7.3 路由专属参数（route_configs）

| 后端键名 | 所属算法 | 前端 label | 备注 |
|---------|---------|-----------|------|
| `rej_low` | sigma_clip, max_mix | 暗像素排异阈值 (σ) | |
| `rej_high` | sigma_clip, max_mix | 亮像素排异阈值 (σ) | |
| `max_iter` | sigma_clip, max_mix | 最大迭代次数 | |
| `temp_path` | sigma_clip, max_mix | — | 隐藏，内部缓存路径 |
| `huber_c` | huber_mean | Huber 常数 c | |
| `minus_only` | max_mix | 还原均值亮度 | 开关 |
| `buffer_mode` | mix (星轨) | — | 隐藏 |

### 7.4 星轨管线专属

| 后端键名 | 前端 label | 可见性 | 备注 |
|---------|-----------|--------|------|
| `mode: fifo` | 最大值叠加 | 路由选项 | |
| `mode: mix` | 噪声均匀化 | 路由选项 | |
| `fin` | 渐入/渐出 | range_slider 左端 | 与 `fout` 组成双端滑条 |
| `fout` | — | 隐藏（绑定至 fin） | |
| `enable_star_shrink` | 缩星 | 开关 | enable+bypass |
| `enable_satellite_clean` | 去卫星线 | 开关 | enable+bypass |
| `sat_window_size` | 去卫星窗口大小 | slider | |
| `mask`（顶层） | 天空区域遮罩 | file_picker | 限定星点检测区域 |
| `mask`（mix route_config） | 天空区域遮罩 | file_picker | 天空/地面融合遮罩 |

### 7.5 星点对齐叠加管线专属

| 后端键名 | 前端 label | 可见性 | 备注 |
|---------|-----------|--------|------|
| `sky_stacker` | 天空叠加算法 | tabs 路由 | |
| `ground_stacker` | 地面叠加算法 | tabs 路由 | |
| `align_method` | 对齐方式 | select | auto / homography / distortion |
| `align_base` | 对齐基准帧 | file_picker | |
| `same_camera` | 同一相机拍摄 | 开关 | |
| `distortion` | — | 隐藏 | 内部畸变参数 |
| `mask` | 天空遮罩 | file_picker | 白=天空，黑=地面 |
| `enable_ground` | 叠加地面 | 开关 | enable 控制 ground_stacker 节点 |

### 7.6 校准管线专属

| 后端键名 | 前端 label | 可见性 | 备注 |
|---------|-----------|--------|------|
| `light_fnames` | 亮场文件 (Light) | inputs | |
| `bias_fnames` | 偏置场文件 (Bias) | inputs | |
| `dark_fnames` | 暗场文件 (Dark) | inputs | |
| `flat_fnames` | 平场文件 (Flat) | inputs | |
| `bias_stacker` | 偏置场来源 | tabs 路由 | none/master/mean/median |
| `dark_stacker` | 暗场来源 | tabs 路由 | |
| `flat_stacker` | 平场来源 | tabs 路由 | |
| `main_stacker` | 叠加算法 | tabs 路由 | |
| `bias_master_path` | Bias 主帧路径 | file_picker | 仅 route=master 时有效 |
| `dark_master_path` | Dark 主帧路径 | file_picker | 仅 route=master 时有效 |
| `flat_master_path` | Flat 主帧路径 | file_picker | 仅 route=master 时有效 |

### 7.7 校准路由选项

| 后端 option 值 | 前端 label | 语义 |
|---------------|-----------|------|
| `none` | 不使用 | 跳过此校准步骤 |
| `master` | 已有主帧 | 加载预合成文件 |
| `mean` | 均值合成 | 从多帧求平均 |
| `median` | 中位数合成 | 从多帧取中位数 |

### 7.8 同名键的语境差异

| 键名 | 管线 | 前端 label | 语义差异 |
|------|------|-----------|---------|
| `mask` | startrail（顶层） | 天空区域遮罩 | 限定星点检测/缩星区域 |
| `mask` | startrail（mix route_config） | 天空区域遮罩 | 天空星轨+地面均值融合 |
| `mask` | sky_ground_stack | 天空遮罩 | 分割天空/地面区域用于独立叠加 |
| `rej_low/rej_high` | sigma_clip | 暗/亮像素排异阈值 | 排除主叠加中的异常像素 |
| `rej_low/rej_high` | max_mix | 暗/亮像素排异阈值 | 估算背景噪声统计量时排异 |
