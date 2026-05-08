# UI Overlay 规范（`.ui.yaml`）

> 引擎侧 DAG/Meta YAML 格式已迁移至 [dag_node_definition.md](dag_node_definition.md)。
> 本文档仅描述前端 UI 渲染层。

---

## 1. 设计原则

- 引擎**只读** `.meta.yaml`，不 import `.ui.yaml`
- 前端通过 `deep_merge(meta, ui)` 生成面板；`.ui.yaml` 不存在时按 type 做 fallback 渲染
- `.ui.yaml` **不可新增** `.meta.yaml` 中不存在的参数 key——它只做 overlay，不扩展语义
- 内部组件 SubDAG（纯 `.yaml` 后缀）不需要 `.ui.yaml`，参数由引用方的 meta/ui 声明

---

## 2. `.ui.yaml` 完整结构

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

# ─── 布局控制（可选） ───
layout:
  groups:
    - name: "分组名称"
      collapsed: false                    # 默认是否折叠
      keys: [config_key, ...]
  order: [route_key, config_key, ...]     # 面板元素排列顺序
```

---

## 3. Widget 类型

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

### 3.1 `range_slider` — 双端滑条

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

---

## 4. Fallback 规则

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

## 5. 前端 merge 算法

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

    # 4. route_configs (route_key → option_key → params)
    for route_key, options in meta.get("route_configs", {}).items():
        for option_key, group in options.items():
            for key, spec in group.items():
                panel.route_configs[route_key][option_key][key] = {
                    "type": spec["type"],
                    "default": spec["default"],
                    **(ui.get("route_configs", {})
                       .get(option_key, {}).get(key, {})),
                }
                entry = panel.route_configs[route_key][option_key][key]
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

---

## 6. 完整示例：`stack.ui.yaml`

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
