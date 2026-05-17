# UI Overlay 规范（`.ui.yaml`）

> 引擎侧 DAG/Meta YAML 格式已迁移至 [dag_node_definition.md](dag_node_definition.md)。
> 本文档仅描述前端 UI 渲染层。

---

## 1. 设计原则

- 引擎**只读** `.meta.yaml`，不 import `.ui.yaml`
- 前端通过 `PanelSchema.from_yaml(meta_path, ui_path)` 合并两份文件生成面板
- `.ui.yaml` 不存在时按 type 做 fallback 渲染
- `.ui.yaml` **不可新增** `.meta.yaml` 中不存在的参数 key——它只做 overlay，不扩展语义（`outputs` 除外：输出面板完全由 ui.yaml 声明）
- 内部组件 SubDAG（纯 `.yaml` 后缀）不需要 `.ui.yaml`，参数由引用方的 meta/ui 声明
- 支持 `!include` + deep-merge 机制复用 fragment 文件，减少重复声明

---

## 2. `!include` 与 Deep Merge

### 2.1 语法

```yaml
# 整块引用：当前节点 = fragment 内容
some_key:
  "!include": path/to/fragment.yaml

# 引用 + override：fragment 为 base，同级其余 key 做 deep merge
some_key:
  "!include": path/to/fragment.yaml
  nested_key:
    field_to_override: new_value
  key_to_delete: null              # null = 从 base 中删除

# list 元素中的引用（如 outputs）
outputs:
  - "!include": fragments/image_output.ui.yaml
    dtype_options: ["uint8", "uint16", "uint32"]   # override
```

### 2.2 Deep merge 规则

1. 遍历 override dict 的每个 key：
   - 若 value 为 `null` → 从 base 中删除该 key（任意深度）
   - 若 value 为 dict 且 base 同名 key 也是 dict → **递归合并**
   - 否则 → value 直接替换 base 中对应 key
2. base 中未被提及的 key 原样保留
3. Fragment 路径相对于当前 yaml 文件所在目录解析
4. Fragment 内部可嵌套 `!include`（递归解析）

### 2.3 实现

加载入口为 `ui.yaml_loader.load_ui_yaml(path)`，使用 `yaml.safe_load` + 自定义 tree-walk 解析 `"!include"` key（非 YAML custom tag），无需 SafeLoader 子类。

### 2.4 Fragment 目录

复用 fragment 文件存放于 `hoshicore/dag/fragments/`：

| Fragment 文件 | 内容 |
|--------------|------|
| `stacker_route_configs.ui.yaml` | sigma_clip + huber_mean 的完整 UI 定义 |
| `image_output.ui.yaml` | 通用图像输出面板声明 |

---

## 3. `.ui.yaml` 完整结构

```yaml
# ─── 输入面板 ───
inputs:
  <input_name>:
    label: "显示名称"
    description: "帮助文本"                # tooltip
    widget: <widget_type>
    accept: ".tif,.fits,.png"             # file_picker 专属

# ─── 路由面板 ───
routes:
  <route_key>:
    label: "显示名称"
    description: "帮助文本"
    widget: tabs | select
    options:
      <option_key>:
        label: "显示名称"
        description: "简短说明"
        icon: "icon-key"                  # 可选

# ─── 全局配置面板 ───
configs:
  <config_name>:
    label: "显示名称"
    description: "帮助文本"
    widget: <widget_type>
    hidden: true                          # 不渲染
    # widget 专属字段 ↓
    min: <number>
    max: <number>
    step: <number>
    options:                              # select 专属
      - { value: "auto", label: "自动" }  # dict 格式（推荐）
      - "scalar_value"                    # 标量格式（兼容）
    bind: <other_key>                     # range_slider 专属
    accept: ".tif,.png"                   # file_picker 专属
    transform: <transform_spec>           # 值变换（见 §4.2）

# ─── 路由专属配置面板 ───
# 注意：ui.yaml 的 route_configs 按 option_key 分组（非 route_key），
# 因为 UI 在路由选项卡内渲染参数，route_key 上下文由选项卡自身提供。
# 同名 option 跨不同 route 共享同一份 UI overlay。
route_configs:
  <option_key>:
    <config_name>:
      label: "显示名称"
      description: "帮助文本"
      widget: <widget_type>
      hidden: true
      # widget 专属字段同上

# ─── 输出面板 ───
outputs:
  - filename_key: <config_key>           # 必需，绑定到 configs 中的文件名 key
    label: "输出图像"
    type: image                          # image | sequence | video
    dtype_key: <config_key>              # 绑定到 configs 中的 dtype key
    formats: ["JPG", "PNG", "TIFF"]      # 允许的格式（须在 IMAGE_FORMAT_PRESETS 中注册）
    dtype_options: ["uint8", "uint16"]   # 允许的位深选项
    format_params:                       # 格式专属参数 → configs key 映射
      jpg_quality: jpg_quality
      png_compressing: png_compressing

# ─── 布局控制（可选） ───
layout:
  groups:
    - name: "分组名称"
      collapsed: false                    # 默认是否折叠
      keys: [config_key, ...]
  order: [route_key, config_key, ...]     # 面板元素排列顺序
```

---

## 4. Widget 类型

### 4.1 Widget 表

| widget | 适用 type | 专属字段 | 说明 |
|--------|-----------|---------|------|
| `switch` | bool | — | 开关 |
| `slider` | int, float | `min`, `max`, `step` | 单滑条 |
| `range_slider` | — | `min`, `max`, `step`, `bind`, `transform` | 双端滑条 |
| `input` | int, float, str | `min`, `max`（数值可选）| 输入框 |
| `select` | str | `options: [...]` | 下拉选择 |
| `tabs` | — | — | 选项卡（仅用于 route） |
| `file_picker` | str, sequence | `accept` | 文件选择器 |
| `dir_picker` | str | — | 目录选择器 |

### 4.2 `range_slider` — 双端滑条

将两个独立参数绑定到同一控件。声明在其中一个参数上，`bind` 指向另一个。

```yaml
configs:
  fin:
    label: "渐入渐出比例"
    widget: range_slider
    bind: fout               # 另一端参数名
    min: 0
    max: 1.0
    step: 0.05
    transform: {right: complement}   # 右端值变换
  fout:
    hidden: true             # 被 bind 吸收
```

前端规则：
- 遇到 `range_slider` + `bind: X` → 渲染双端滑条
- 声明方 = 左端，`bind` 方 = 右端
- 提交时拆回两个独立参数
- `bind_default` 由 `PanelSchema` 自动从被 bind 参数的 meta.yaml default 注入，无需手动声明

### 4.3 `transform` — 值变换

将 widget 的 UI 值映射为后端实际值。定义在 `ui/transforms.py`。

| transform 名 | 公式 | 用途 |
|---|---|---|
| `identity` | x | 无操作 |
| `negate` | -x | 符号取反 |
| `complement` | 1 - x | 补数（如比例互补） |
| `abs` | \|x\| | 绝对值（不可逆） |

**用法**：
- 普通 widget：`transform: "negate"` （字符串）
- range_slider：`transform: {left: "negate", right: "complement"}` （dict，分别作用于左/右端值）

### 4.4 `select` — options 格式

支持两种写法：

```yaml
# 标量列表（value = label）
options: ["uint8", "uint16"]

# dict 列表（value ≠ label 时使用）
options:
  - { value: "auto",       label: "自动" }
  - { value: "homography", label: "透视变换" }
  - { value: "distortion", label: "畸变优化" }
```

---

## 5. `outputs` 面板

`outputs` 块完全由 `.ui.yaml` 声明（不来源于 `.meta.yaml`）。它描述如何渲染文件输出面板。

### 5.1 字段说明

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `filename_key` | str | ✓ | 对应 `configs` 中控制输出文件名的 key |
| `label` | str | — | 面板标题（默认 "输出"）|
| `type` | str | — | 输出类型（默认 "image"）|
| `dtype_key` | str | — | 对应 `configs` 中控制 dtype 的 key |
| `formats` | list[str] | — | 允许的格式列表，须在 `IMAGE_FORMAT_PRESETS` 中注册 |
| `dtype_options` | list[str] | — | 允许的 dtype 列表 |
| `format_params` | dict | — | 格式参数映射：`preset_param → config_key` |

### 5.2 格式预设 (`IMAGE_FORMAT_PRESETS`)

定义在 `ui/output_presets.py`，编码各格式的物理约束：

| 格式 | 允许 dtype | 额外参数 |
|------|-----------|---------|
| JPG | uint8 | `jpg_quality` (1–100) |
| PNG | uint8, uint16 | `png_compressing` (1–9) |
| TIFF | uint8, uint16, uint32 | — |

最终可选范围 = `formats ∩ IMAGE_FORMAT_PRESETS`，dtype 可选范围 = `dtype_options ∩ 格式 allowed_dtypes`。

---

## 6. Fallback 规则

`.ui.yaml` 不存在或某 key 缺失时，前端按 type 推断：

| type | 默认 widget | 默认 label |
|------|-------------|-----------|
| bool | switch | config key 原样 |
| int | input | config key |
| float | input | config key |
| str | input | config key |
| dict | hidden | — |
| list | hidden | — |
| image | hidden | — |
| sequence (input) | file_picker | config key |

route 的默认 widget 为 `tabs`。

---

## 7. 前端合并算法

实现位于 `ui/panel_builder.py` 的 `PanelSchema.from_yaml()` 方法：

```python
@classmethod
def from_yaml(cls, meta_path: str, ui_path: str | None = None) -> PanelSchema:
    meta = yaml.safe_load(open(meta_path))
    ui = load_ui_yaml(ui_path) if ui_path else {}  # 解析 !include

    schema = cls(meta_yaml_path=meta_path)
    schema._parse_routes(meta, ui)       # routes
    schema._parse_configs(meta, ui)      # configs
    schema._parse_route_configs(meta, ui) # route_configs
    schema._parse_outputs(ui)            # outputs（纯 ui.yaml）
    schema._resolve_bind_defaults()      # range_slider bind_default 注入
    schema.layout = ui.get("layout", {})
    return schema
```

### 7.1 Routes 合并

```python
# meta.routes[route_key].options → option 列表
# ui.routes[route_key] → label, widget, options[opt].label/description
# default = meta 声明的 default（或第一个 option）
```

### 7.2 Configs 合并

```python
# 对每个 meta.configs[key]:
#   type, default ← meta
#   widget, label, description, hidden, min, max, step, options, bind, accept, transform ← ui
#   widget fallback = _infer_widget(type)
```

### 7.3 Route Configs 合并

```python
# meta.route_configs[route_key][option_key][param] → type, default
# ui.route_configs[option_key][param] → widget overlay（注意：ui 按 option_key 索引，非 route_key）
```

### 7.4 Outputs 解析

```python
# 完全来自 ui.outputs（list of dict）
# 每个 entry 必须含 filename_key
# format_params 的 key 须在 IMAGE_FORMAT_PRESETS 中声明
# formats 列表中的值须在 IMAGE_FORMAT_PRESETS 中注册
```

---

## 8. 完整示例

### 8.1 `stack.ui.yaml`（使用 !include）

```yaml
inputs:
  fnames:
    label: "图像文件"
    widget: file_picker
    accept: ".tif,.tiff,.fits,.png,.jpg,.cr2,.nef,.arw"

routes:
  stacker:
    label: "叠加算法"
    widget: tabs
    options:
      mean:
        label: "均值"
        description: "高信噪比平均叠加"
      sigma_clip:
        label: "Sigma 裁剪"
        description: "迭代剔除异常像素后求均值"
      median:
        label: "中位数"
        description: "逐像素取中位数，天然抗异常值"
      huber_mean:
        label: "Huber 均值"
        description: "基于 Huber 损失的稳健均值估计"

configs:
  int_weight:
    label: "整数权重"
    description: "启用使用整数累加，改善精度，速度更快"
    widget: switch
  output_dtype:
    hidden: true
  output_filename:
    hidden: true
  exif_reduce_type:
    hidden: true
  loader_type:
    hidden: true
  loader_configs:
    hidden: true
  jpg_quality:
    hidden: true
  png_compressing:
    hidden: true

route_configs:
  "!include": fragments/stacker_route_configs.ui.yaml

layout:
  groups:
    - name: "基础设置"
      keys: [int_weight]
  order: [stacker, int_weight]

outputs:
  - "!include": fragments/image_output.ui.yaml
```

### 8.2 `calibration_stack.ui.yaml`（include + override）

```yaml
route_configs:
  "!include": fragments/stacker_route_configs.ui.yaml
  sigma_clip:
    max_iter:
      max: 20                  # override: 扩大最大迭代次数
    early_converge_ratio: null  # 删除此参数的 UI 渲染

outputs:
  - "!include": fragments/image_output.ui.yaml
```

### 8.3 `startrail.ui.yaml`（内联 route_configs + outputs override）

```yaml
# route_configs: 内联声明（mix 模式有独特的 mask/buffer_mode/minus_only）
route_configs:
  mix:
    rej_low:
      label: "低亮度排异 σ 阈值"
      widget: slider
      min: 0
      max: 5.0
      step: 0.1
    # ...

# outputs: include + dtype override
outputs:
  - "!include": fragments/image_output.ui.yaml
    dtype_options: ["uint8", "uint16", "uint32"]
```

### 8.4 Fragment: `fragments/stacker_route_configs.ui.yaml`

```yaml
sigma_clip:
  rej_low:
    label: "低亮度排异 σ 阈值"
    description: "低于均值多少个标准差视为异常值"
    widget: slider
    min: 0
    max: 5.0
    step: 0.1
  rej_high:
    label: "高亮度排异 σ 阈值"
    description: "高于均值多少个标准差视为异常值"
    widget: slider
    min: 0
    max: 5.0
    step: 0.1
  max_iter:
    label: "最大迭代次数"
    widget: slider
    min: 1
    max: 10
    step: 1
  early_converge_ratio:
    label: "早停比例"
    description: "连续两轮保留像素比例超过此值时提前终止"
    widget: slider
    min: 0.9
    max: 1.0
    step: 0.01
huber_mean:
  huber_c:
    label: "Huber 常数 c"
    description: "越小越稳健（更多像素被降权），越大越接近普通均值"
    widget: slider
    min: 0.1
    max: 5.0
    step: 0.05
```

### 8.5 Fragment: `fragments/image_output.ui.yaml`

```yaml
filename_key: output_filename
label: "输出图像"
type: image
dtype_key: output_dtype
formats: ["JPG", "PNG", "TIFF"]
dtype_options: ["uint8", "uint16", "uint32"]
format_params:
  jpg_quality: jpg_quality
  png_compressing: png_compressing
```
