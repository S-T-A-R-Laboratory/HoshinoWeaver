# Meta YAML v2 格式规范

> 状态：**草案**

## 1. 文件约定

每个用户可见算法由两个文件描述，存放于 `hoshicore/dag/`：

| 文件 | 角色 | 消费方 | 必须 |
|------|------|--------|------|
| `<name>.meta.yaml` | DAG 逻辑拓扑 + 路由定义 + 参数声明（type/default） | 引擎 | 是 |
| `<name>.ui.yaml` | 渲染 overlay（label/widget/min/max/group…） | 前端 | 否 |

内部组件 SubDAG（如 `sigma_clip.yaml`、`median_stack_core.yaml`）保持纯 `.yaml` 后缀，
不需要 `.ui.yaml`——它们的参数由引用方的 meta/ui 声明。

**设计原则**：
- 引擎**只读** `.meta.yaml`，不 import `.ui.yaml`。
- 前端通过 `deep_merge(meta, ui)` 生成面板。`.ui.yaml` 不存在时前端按 type 做 fallback 渲染。
- `.ui.yaml` **不可新增** `.meta.yaml` 中不存在的参数 key；它只做 overlay，不扩展语义。

---

## 2. `.meta.yaml` 完整结构

```yaml
description: "算法的一句话描述"
version: "2"

# ─── 全局输入 ───
inputs:
  <input_name>:
    type: sequence | image | object
    required: true | false            # 默认 true

# ─── 路由定义 ───
routes:
  <route_key>:
    options:
      <option_key>:
        <OpName | sub.yaml>
    default: <option_key>

# ─── 全局配置（所有模式共享） ───
configs:
  <config_name>:
    type: bool | int | float | str | dict
    default: <value>

# ─── 路由专属配置 ───
route_configs:
  <option_key>:
    <config_name>:
      type: bool | int | float | str
      default: <value>

# ─── 节点 ───
nodes:
  <node_name>:
    # 普通节点：
    op: <OpName>
    # 路由节点：
    route_key: <route_key>       # op 由 meta_resolve 填入

    inputs:
      <slot>: <link>
    configs:
      <slot>: <link>

    enabled: <configs.xxx>       # 可选，引用 bool 型 config
    bypass: <input_key>          # 可选，指定 bypass 时转发的 input

    route_inputs:                # 仅特定 option 需要的输入
      <option_key>: { <slot>: <link> }
    route_configs:               # 仅特定 option 需要的配置布线
      <option_key>: { <slot>: <link> }

    outputs:
      <output_name>: { type: <type> }

# ─── 全局输出 ───
outputs:
  <name>: <node.output>
```

### 2.1 `route_configs` 语义与 merge 规则

顶层 `route_configs` 按 **route option key** 分组声明参数。

`meta_resolve()` 处理流程：

```
1. 确定每个 route_key 的选中 option
2. 收集所有选中 option 对应的 route_configs 条目
3. 检查与全局 configs 无 key 冲突
4. merge 到全局 configs 命名空间
5. 删除 route_configs section
```

merge 后，节点的 `route_configs` 布线通过 `configs.<name>` 引用，
与后续的 `flatten_sub_dags()` 和 `wiring` 完全兼容。

**冲突规则**：
- `route_configs` 中的 key **不得与**全局 `configs` 中的 key 重名
- `meta_resolve()` 发现重名时抛出 `MetaResolveError`

### 2.2 多 route_key 场景

一个 meta YAML 可有多个 route_key（如 calibration_stack 有 4 个）。

当不同 route_key 选中的 option 共享 `route_configs` 条目时（如两个 route 都选中 `median`），
它们共享同一组参数（语义上一般正确——同一个 `chunk_rows` 值）。

若需要差异化，使用带前缀的 option key（如 `bias_median` / `main_median`），
对应不同的 `route_configs` 条目。

### 2.3 节点开关（`enabled` / `bypass`）

声明 `enabled: configs.<bool_key>` 的节点可被运行时开关。
`meta_resolve()` 解析 `enabled` 引用，若值为 `false`，执行编译期 bypass：

```
1. 确定 bypass 对：哪个 input → 哪个 output 做直通转发
2. 重写所有引用 node.output 的消费者 → node 的 input 源 link
3. 从 spec 中删除该节点
```

bypass 后该节点在 DAG 中彻底不存在，零运行时开销。

**bypass 对推断规则**：

1. 节点声明了 `bypass: <input_key>` → 使用该 input，配对第一个 output
2. 节点 `inputs:` section 只有一个 entry → 自动配对第一个 output
3. 多个 input 且未声明 `bypass` → `MetaResolveError`

类型幂等性（input type == output type）由 YAML 作者保证，引擎不做运行时检查。

**约束**：
- `enabled` 引用必须指向 `configs` 命名空间中的 bool 类型参数
- `configs:` section 中的参数不参与 bypass 推断（它们是配置，不是数据流）
- 值来源：通过 `global_configs` 传入，`meta_resolve()` 解析
- 值未提供时 fallback 到 YAML `configs` 中声明的 `default`

**示例**：

```yaml
configs:
  enable_resize: { type: bool, default: true }

nodes:
  resize:
    op: ImageResizeOp
    enabled: configs.enable_resize       # 可被关闭
    inputs:
      data: data_loader.result           # 唯一 input → 自动推断 bypass 对
    configs:
      scale: configs.resize_scale
    outputs:
      result: { type: sequence }         # type 与 inputs.data 一致 → 幂等

  # enable_resize=false 时:
  # 所有引用 resize.result 的地方被重写为 data_loader.result
  # resize 节点从 spec 中删除
```

多 input 场景需显式声明：

```yaml
  subtract:
    op: CalibrationSubtractOp
    enabled: configs.enable_bias
    bypass: data                         # 显式：inputs.data → outputs.result
    inputs:
      data: prev_step.result
      aux: other.result
    configs:
      reference: bias_stacker.result
    outputs:
      result: { type: sequence }
```

### 2.4 嵌套透传

执行顺序：`meta_resolve()` → `flatten_sub_dags()`。

由于 merge 在 flatten 之前完成，SubDAG 的 `configs.xxx` 引用能正确解析到
已 merge 的全局 configs 命名空间。

对于 Meta SubDAG（子图本身也是 meta），父图通过节点的 `routes` 字段透传选择：

```yaml
nodes:
  pipeline:
    op: advanced_sub.meta.yaml
    routes:
      inner_route: "sigma_clip"           # 字面量
      inner_route: routes.parent_route    # 引用父图已解析路由
```

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
    options: [<value>, ...]               # select 专属

# ─── 路由专属配置面板 ───
route_configs:
  <option_key>:
    <config_name>:
      label: "显示名称"
      description: "帮助文本"
      widget: <widget_type>
      hidden: true
      # widget 专属字段同上

# ─── 布局控制（可选） ───
layout:
  groups:
    - name: "分组名称"
      collapsed: false                    # 默认是否折叠
      keys: [config_key, ...]
  order: [route_key, config_key, ...]     # 面板元素排列顺序
```

### 3.1 Widget 类型

| widget | 适用 type | 专属字段 | 说明 |
|--------|-----------|---------|------|
| `switch` | bool | — | 开关 |
| `slider` | int, float | `min`, `max`, `step` | 单滑条 |
| `range_slider` | — | `min`, `max`, `step`, `bind` | 双端滑条 |
| `input` | int, float, str | `min`, `max`（数值可选）| 输入框 |
| `select` | str | `options: [...]` | 下拉选择 |
| `tabs` | — | — | 选项卡（仅用于 route） |
| `file_picker` | str, sequence | `accept` | 文件选择器 |
| `dir_picker` | str | — | 目录选择器 |

### 3.2 `range_slider` — 双端滑条

将两个独立参数绑定到同一控件。声明在其中一个参数上，`bind` 指向另一个。

```yaml
route_configs:
  sigma_clip:
    rej_low:
      label: "排异 σ 范围"
      widget: range_slider
      bind: rej_high          # 另一端参数名
      min: 0.5
      max: 6.0
      step: 0.1
    rej_high:
      hidden: true            # 被 bind 吸收
```

前端规则：
- 遇到 `range_slider` + `bind: X` → 渲染双端滑条
- 声明方 = 低端，`bind` 方 = 高端
- 提交时拆回两个独立参数

### 3.3 Fallback 规则

`.ui.yaml` 不存在或某 key 缺失时，前端按 type 推断：

| type | 默认 widget | 默认 label |
|------|-------------|-----------|
| bool | switch | config key 原样 |
| int | input | config key |
| float | input | config key |
| str | input | config key |
| dict | hidden | — |
| sequence (input) | file_picker | config key |

route 的默认 widget 为 `tabs`。

---

## 4. 引擎侧改动

仅修改 `meta.py` 的 `meta_resolve()`：

1. 解析顶层 `route_configs`
2. 将选中 option 对应的参数 merge 到 `configs`
3. 冲突检测
4. 删除 `route_configs` section

节点的 `route_key` / `route_inputs` / `route_configs` 布线逻辑**不变**。
`flatten.py`、`wiring.py`、`build.py` **不变**。

---

## 5. 完整示例

### 5.1 `stack.meta.yaml`

```yaml
description: "通用叠加算法"
version: "2"

inputs:
  fnames: { type: sequence, required: true }

routes:
  stacker:
    options:
      mean:       MeanStackerOp
      sigma_clip: sigma_clip.yaml
      median:     median_stack_core.yaml
      huber_mean: huber_mean.yaml
    default: mean

configs:
  int_weight:       { type: bool, default: true }
  exif_reduce_type: { type: str,  default: "sum" }
  output_filename:  { type: str,  default: "result.tif" }
  output_dtype:     { type: str,  default: "uint8" }
  loader_type:      { type: str,  default: "img_file_list" }
  loader_configs:   { type: dict, default: {} }

route_configs:
  sigma_clip:
    rej_high: { type: float, default: 3.0 }
    rej_low:  { type: float, default: 3.0 }
    max_iter: { type: int,   default: 5 }
  median:
    chunk_rows: { type: int, default: 32 }
  huber_mean:
    huber_c: { type: float, default: 1.345 }

nodes:
  data_loader:
    op: ImgDataLoaderOp
    inputs:
      src: inputs.fnames
    configs:
      loader_type: configs.loader_type
      configs: configs.loader_configs
    outputs:
      result: { type: sequence }

  stacker:
    route_key: stacker
    inputs:
      data: data_loader.result
    route_configs:
      mean:
        int_weight: configs.int_weight
      sigma_clip:
        int_weight: configs.int_weight
        rej_high: configs.rej_high
        rej_low: configs.rej_low
        max_iter: configs.max_iter
      huber_mean:
        int_weight: configs.int_weight
        huber_c: configs.huber_c
      median: {}
    outputs:
      result: { type: image }

  exif_loader:
    op: ExifReadOp
    inputs:
      fnames: inputs.fnames
    outputs:
      result: { type: sequence }

  exif_reducer:
    op: ExifReduceOp
    inputs:
      exifs: exif_loader.result
    configs:
      merge_method: configs.exif_reduce_type
    outputs:
      result: { type: object }

  image_saver:
    op: ImageSaveOp
    configs:
      image: stacker.result
      exif: exif_reducer.result
      output_filename: configs.output_filename
      output_dtype: configs.output_dtype
    outputs:
      return_code: { type: int }

outputs:
  result: image_saver.return_code
```

### 5.2 `stack.ui.yaml`

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
    description: "启用后使用整数累加，精度稍低但速度更快"
    widget: switch
  output_dtype:
    label: "输出位深"
    widget: select
    options: ["uint8", "uint16"]
  output_filename:
    label: "输出文件名"
    widget: input
  exif_reduce_type:
    hidden: true
  loader_type:
    hidden: true
  loader_configs:
    hidden: true

route_configs:
  sigma_clip:
    rej_low:
      label: "排异 σ 范围"
      description: "低于/高于均值多少个标准差视为异常值"
      widget: range_slider
      bind: rej_high
      min: 0.5
      max: 6.0
      step: 0.1
    rej_high:
      hidden: true
    max_iter:
      label: "最大迭代次数"
      widget: slider
      min: 1
      max: 20
      step: 1
  median:
    chunk_rows:
      label: "分块行数"
      description: "逐行分块处理，越大越快但内存占用越高"
      widget: slider
      min: 8
      max: 128
      step: 8
  huber_mean:
    huber_c:
      label: "Huber 常数 c"
      description: "越小越稳健（更多像素被降权），越大越接近普通均值"
      widget: slider
      min: 0.1
      max: 5.0
      step: 0.05

layout:
  groups:
    - name: "基础设置"
      keys: [int_weight, output_dtype, output_filename]
  order: [stacker, int_weight, output_dtype, output_filename]
```

### 5.3 `startrail.meta.yaml`（草案）

```yaml
description: "星轨叠加算法"
version: "2"

inputs:
  fnames: { type: sequence, required: true }

routes:
  mode:
    options:
      fifo: TrailStackerOp
      mix:  TrailStackerOp
    default: fifo

configs:
  fin:              { type: float, default: 0 }
  fout:             { type: float, default: 0 }
  int_weight:       { type: bool,  default: true }
  exif_reduce_type: { type: str,   default: "sum" }
  output_filename:  { type: str,   default: "result.tif" }
  output_dtype:     { type: str,   default: "uint16" }
  loader_type:      { type: str,   default: "img_file_list" }
  loader_configs:   { type: dict,  default: {} }

route_configs:
  mix:
    rej_high:   { type: float, default: 3.0 }
    rej_low:    { type: float, default: 3.0 }
    max_iter:   { type: int,   default: 5 }
    mask:       { type: str,   default: null }
    minus_only: { type: bool,  default: true }
    buffer_mode: { type: str,  default: "auto" }

nodes:
  data_loader:
    op: ImgDataLoaderOp
    inputs:
      src: inputs.fnames
    configs:
      loader_type: configs.loader_type
      configs: configs.loader_configs
    outputs:
      result: { type: sequence }

  weight_generator:
    op: WeightGeneratorOp
    inputs:
      sequence: inputs.fnames
    configs:
      fin: configs.fin
      fout: configs.fout
    outputs:
      result: { type: sequence }

  trailstacker:
    op: TrailStackerOp
    inputs:
      data: data_loader.result
      weight: weight_generator.result
    configs:
      int_weight: configs.int_weight
    outputs:
      result: { type: image }

  # ── mix 专属分支（fifo 模式时这些节点不存在） ──
  simgaclipstacker:
    route_key: mode
    route_configs:
      mix:
        op: sigma_clip.yaml
        int_weight: configs.int_weight
        rej_high: configs.rej_high
        rej_low: configs.rej_low
        max_iter: configs.max_iter
        buffer_mode: configs.buffer_mode
    # TODO: fifo 时此节点需要整体跳过 — 需要条件节点机制

  # ... 后续节点省略，此处标注设计问题
```

> **设计问题**：`startrail` 的 fifo/mix 模式不仅是替换一个 Op，
> 而是 mix 模式需要**额外的 DAG 分支**（sigma_clip + load_mask + mne）。
> 这超出了当前 route 机制（替换单节点 op）的能力。
>
> **可能的解决方向**：
> - 将 `fifo` 和 `mix` 分别定义为两个 SubDAG YAML，route 选择整个子图
> - 或引入条件节点（conditional node），但这增加了引擎复杂度
>
> 建议 startrail 暂不合并，保持两个独立 YAML，待条件节点机制成熟后再统一。

---

## 6. 前端 merge 算法

```python
def build_panel(meta: dict, ui: dict | None) -> Panel:
    ui = ui or {}
    panel = Panel()

    # 1. inputs
    for key, spec in meta.get("inputs", {}).items():
        panel.inputs[key] = {
            "type": spec["type"],
            "required": spec.get("required", True),
            **(ui.get("inputs", {}).get(key, {})),
        }
        panel.inputs[key].setdefault("widget", infer_widget(spec["type"], is_input=True))
        panel.inputs[key].setdefault("label", key)

    # 2. routes
    for route_key, route_def in meta.get("routes", {}).items():
        panel.routes[route_key] = {
            "options": list(route_def["options"].keys()),
            "default": route_def.get("default"),
            **(ui.get("routes", {}).get(route_key, {})),
        }
        panel.routes[route_key].setdefault("widget", "tabs")
        panel.routes[route_key].setdefault("label", route_key)

    # 3. configs
    for key, spec in meta.get("configs", {}).items():
        panel.configs[key] = {
            "type": spec["type"],
            "default": spec["default"],
            **(ui.get("configs", {}).get(key, {})),
        }
        panel.configs[key].setdefault("widget", infer_widget(spec["type"]))
        panel.configs[key].setdefault("label", key)

    # 4. route_configs
    for option_key, group in meta.get("route_configs", {}).items():
        for key, spec in group.items():
            panel.route_configs[option_key][key] = {
                "type": spec["type"],
                "default": spec["default"],
                **(ui.get("route_configs", {}).get(option_key, {}).get(key, {})),
            }
            entry = panel.route_configs[option_key][key]
            entry.setdefault("widget", infer_widget(spec["type"]))
            entry.setdefault("label", key)

    # 5. layout
    panel.layout = ui.get("layout", {})

    return panel


def infer_widget(typ: str, is_input: bool = False) -> str:
    if is_input:
        return "file_picker"
    return {
        "bool": "switch",
        "int": "input",
        "float": "input",
        "str": "input",
        "dict": "hidden",
    }.get(typ, "input")
```
