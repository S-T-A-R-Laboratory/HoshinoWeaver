# DAG 计算图规范

本文档描述如何用 YAML 声明一个计算图（DAG, Directed Acyclic Graph），涵盖标准 DAG YAML 和 Meta YAML 两种格式。

处理流水线：

```
YAML ── meta_resolve() ──► 标准 spec ── flatten_sub_dags() ──► 展平 spec ── validate_and_build_order() ──► ValidatedDag ── wiring ──► 运行
```

- `meta_resolve()`：仅对 Meta YAML（含 `routes`/`route_key`/`enabled`）执行路由编译和节点开关
- `flatten_sub_dags()`：递归展开 `.yaml` SubDAG 引用为带命名空间前缀的扁平节点
- `validate_and_build_order()`：合法性检查 + 拓扑执行顺序推导（不实例化 Op）
- `wiring`：实例化 Op、连接队列、生成 feeder 协程

---

## 1. 文件约定

| 后缀 | 角色 | 消费方 |
|------|------|--------|
| `<name>.meta.yaml` | Meta YAML：含路由定义 + 参数声明 + 节点开关 | 引擎（meta_resolve → flatten → build） |
| `<name>.yaml` | 标准 DAG YAML（也用作 SubDAG 组件） | 引擎（flatten → build） |
| `<name>.ui.yaml` | 渲染 overlay（label/widget/min/max/group…） | 前端 |

内部组件 SubDAG（如 `sigma_clip.yaml`、`median_stack_core.yaml`）使用纯 `.yaml` 后缀，存放于 `hoshicore/dag/base/`。

---

## 2. 标准 DAG YAML 格式

### 2.1 顶层字段

```yaml
description: "图的描述"              # 可选，string
version: "1"                         # 可选，string

inputs:    { ... }                   # 全局输入声明
configs:   { ... }                   # 全局配置声明
nodes:     { ... }                   # 节点定义（必需）
outputs:   { ... }                   # 全局输出（必需）
```

### 2.2 `inputs`（全局输入）

类型：`object`（键值对）。声明图的序列输入入口。

```yaml
inputs:
  <input_name>:
    type: sequence                   # 必需，通常为 "sequence"
    required: true                   # 可选，默认 true
```

**约束**：全局 `inputs` 的 `type` 必须为 `sequence`（构建器强制校验）。

### 2.3 `configs`（全局配置）

类型：`object`（键值对）。声明图的标量配置入口。

```yaml
configs:
  <config_name>:
    type: bool | int | float | str | dict | list | image
    default: <value>                 # 可选
    required: false                  # 可选
```

**约束**：全局 `configs` 的 `type` 不建议使用 `sequence`。

### 2.4 `nodes`（节点定义）

类型：`object`。键为节点 id（`node_name`），值为 NodeSpec。

```yaml
nodes:
  <node_name>:
    op: <OpName>                     # 必需，非空字符串
    inputs:                          # 可选
      <arg_name>: <Link>            # 简写形式
      <arg_name>: { src: <Link> }   # 完整形式
    configs:                         # 可选
      <arg_name>: <Link>
      <arg_name>: { src: <Link> }
    outputs:                         # 必需
      <output_name>: { type: <type> }
```

#### `op` 字段

- 值为 Op 注册名（如 `MeanStackerOp`）时，wiring 层从注册表查找实例化
- 值以 `.yaml` 结尾时（如 `sigma_clip.yaml`），flatten 层自动展开为 SubDAG
- 值以 `.meta.yaml` 结尾时，flatten 层先调用 `meta_resolve` 再展开

#### `inputs` / `configs` 绑定

支持两种写法（等价）：

```yaml
# 简写：
inputs:
  data: data_loader.result

# 完整形式：
inputs:
  data: { src: data_loader.result }
```

`configs` 还支持字面量值（由 flatten 层处理 SubDAG 时使用）。

#### `outputs` 声明

每个输出字段必须声明 `type`：

```yaml
outputs:
  result: { type: image }
  statistics: { type: image }
```

### 2.5 `outputs`（全局输出）

类型：`object`（`name → link`）。声明图的完成条件——所有 link 指向的值就绪时，图执行完成。

```yaml
outputs:
  result: image_saver.return_code
  statistics: stacker.statistics
```

---

## 3. 链接语法（Link）

Link 是一个字符串，表示数据来源。构建器解析为三种类型：

| 语法 | 解析结果 | 语义 |
|------|----------|------|
| `inputs.<name>` | `("inputs", name)` | 全局序列输入 |
| `configs.<name>` | `("configs", name)` | 全局标量配置（支持多级 dotted path） |
| `<node>.<output>` | `("node", node, output)` | 节点输出 |

**多级 configs 路径**：Meta YAML 的 `route_configs` 经 resolve 后产生嵌套结构，引用时使用 dotted path（如 `configs.stacker.sigma_clip.rej_high`）。构建器的 `_parse_link` 使用 `configs.` 前缀匹配，整个后缀作为 config_name。

**节点引用解析**：`_parse_link` 对非 `inputs.`/`configs.` 前缀的 link 使用 `rsplit(".", 1)` 拆分，因此带命名空间的节点引用（如 `main_stacker.mean_stacker.statistics`）会被正确解析为 `node="main_stacker.mean_stacker"` `output="statistics"`。

---

## 4. SubDAG 展开

当节点的 `op` 字段以 `.yaml` 结尾时，`flatten_sub_dags()` 在编译期将子图展开为扁平拓扑。

### 4.1 展开规则

1. 子图节点添加 `{parent_name}.` 命名空间前缀
2. 子图内部 `inputs.*` 引用 → 映射到父图的实际源 link
3. 子图内部 `configs.*` 引用 → 映射到父图覆盖值或省略（由 Op 默认值补齐）
4. 子图内部节点引用 `node.output` → 添加前缀 `parent_name.node.output`
5. 父图中引用 SubDAG 输出的消费者 → 重写为展平后的实际 link
6. 支持递归展开（子图内部也可引用 `.yaml`），最大深度 10 层

### 4.2 特殊标记

- `__inactive__`：子图可选输入未被父图布线时的标记。wiring 层跳过此类队列的激活。

### 4.3 SubDAG YAML 格式

SubDAG 使用标准 DAG YAML 格式，其 `inputs`/`configs` 声明的是子图的接口：

```yaml
# sigma_clip.yaml — SubDAG 示例
description: "Sigma Clipping sub-DAG"
version: "1"

inputs:
  data: { type: sequence }
  fnames: { type: sequence, required: false }  # 可选输入

configs:
  int_weight: { type: bool, default: true }
  rej_high: { type: float, default: 3.0 }

nodes:
  mean_stacker:
    op: MeanStackerOp
    inputs:
      data: inputs.data
    configs:
      int_weight: configs.int_weight
    outputs:
      result: { type: image }
      statistics: { type: image }
  # ...

outputs:
  result: sigma_clip_iter.result
```

### 4.4 Meta SubDAG

子图本身也可以是 Meta YAML（含 `routes`/`route_key`）。flatten 检测到此情况时自动先调用 `meta_resolve`。父图通过节点的 `routes` 字段透传选择：

```yaml
sky_stacker:
  op: stacker.meta.yaml
  route_key: sky_stacker          # 父图自身的路由 key
  routes:
    stacker: routes.sky_stacker   # 透传：使用父图已解析的 sky_stacker 路由
  inputs:
    data: star_aligner.result
  outputs:
    result: { type: image }
```

`routes` 字段值的语法：
- 字面量字符串（如 `"sigma_clip"`）：直接传入子图的 route_choices
- `routes.<key>` 引用：从父图的 `_resolved_routes` 中解析后传入

---

## 5. Meta YAML 格式

Meta YAML 在标准 DAG YAML 基础上增加路由定义、路由专属配置、节点开关三项能力。文件后缀为 `.meta.yaml`。

### 5.1 完整结构

```yaml
description: "算法描述"
version: "2"

inputs:    { ... }                  # 同标准 YAML
routes:    { ... }                  # 路由定义（Meta 专属）
configs:   { ... }                  # 全局配置
route_configs: { ... }             # 路由专属配置（Meta 专属）
nodes:     { ... }                  # 节点定义（支持 route_key / enabled）
outputs:   { ... }                  # 同标准 YAML
```

### 5.2 `routes`（路由定义）

声明可选的处理路径。每个 route 有一组可选实现。

```yaml
routes:
  <route_key>:
    options:
      <option_a>: <OpName | sub.yaml | null>
      <option_b>: <OpName | sub.yaml | null>
    default: <option_key>           # 默认选择
```

**options 值**：
- Op 类名（如 `MeanStackerOp`）：直接填入节点 `op` 字段
- SubDAG 路径（如 `sigma_clip.yaml`）：填入 `op` 字段后由 flatten 展开
- `null`：fixed-op 模式——节点已显式声明 `op`，路由仅用于选择 `route_configs` 分支

### 5.3 `route_configs`（路由专属配置）

按 `route_key → option_key` 两级分组声明参数。

```yaml
route_configs:
  <route_key>:
    <option_key>:
      <param>: { type: ..., default: ... }
```

`meta_resolve()` 处理流程：
1. 确定每个 route_key 的选中 option
2. 将选中 option 对应的参数 merge 到 `configs.<route_key>.<option_key>` 嵌套命名空间
3. 删除顶层 `route_configs` section

**命名空间规则**：引用路径为三段式 dotted path `configs.<route_key>.<option_key>.<param>`，与全局 configs（单段 `configs.<name>`）天然隔离。

### 5.4 路由节点声明

节点使用 `route_key` 字段声明为路由节点：

```yaml
<node_name>:
  route_key: <route_key>            # 引用顶层 routes 定义
  op: <OpName>                      # 可选（fixed-op 模式时声明）
  routes:                           # 可选，透传路由到 Meta SubDAG
    <sub_route_key>: <choice | routes.xxx>
  inputs: { ... }                   # 所有选项共享
  route_inputs:                     # 可选，按 option 分组的专属输入
    <option_a>: { <slot>: <link> }
  configs: { ... }                  # 所有选项共享
  route_configs:                    # 可选，按 option 分组的专属配置布线
    <option_a>:
      <slot>: <link>               # 显式布线
      # 未列出的 slot → auto-wire
  outputs: { ... }                  # 必需
```

**fixed-op 模式**：当节点已声明 `op` 字段时，`meta_resolve` 不覆盖 `op`，仅执行 `route_configs` 分支选择。适用于节点 Op 固定但需要按路由切换配置参数的场景。对应 `routes` 定义中 option 值为 `null`。

**Auto-wire 规则**：顶层 `route_configs[route_key][choice]` 中声明的参数，若在节点 `route_configs[choice]` 中未显式布线，自动生成 `configs.<route_key>.<choice>.<param>` 引用。只需写**非默认布线**的条目。

### 5.5 节点开关（`enabled` / `bypass`）

声明 `enabled: configs.<bool_key>` 的节点可被运行时开关。

```yaml
configs:
  enable_resize: { type: bool, default: true }

nodes:
  resize:
    op: ImageResizeOp
    enabled: configs.enable_resize
    bypass: data                     # 多 input 时需显式声明
    inputs:
      data: prev_step.result
    configs:
      scale: configs.resize_scale
    outputs:
      result: { type: sequence }
```

`meta_resolve()` 解析 `enabled` 引用：
- 值为 `true` → 保留节点，移除 `enabled`/`bypass` 字段
- 值为 `false` → 编译期 bypass：重写所有消费者的 link 为 bypass 源，删除节点

**bypass 对推断规则**：
1. 声明了 `bypass: <input_key>` → 使用该 input，配对第一个 output
2. `inputs` 只有一个 entry → 自动配对第一个 output
3. 多个 input 且未声明 `bypass` → `MetaResolveError`

**enabled 值来源**：
- 优先从 `global_configs` 传入的运行时值
- 其次使用 YAML `configs` 中声明的 `default`
- 都未提供 → `MetaResolveError`

---

## 6. Wiring 层行为

`instantiate_and_wire()` 将 ValidatedDag 转化为可运行的异步管线。关键行为：

### 6.1 自动注入默认值

Op 声明的 `CONFIGS` 中有 `default` 但 YAML 未布线的键：
- 自动注入默认值（feeder 协程推送一次）
- 无 `default` 且未布线 → warning（可能导致节点永久挂起）

Op 声明的 `INPUTS` 中未布线的键：
- `required: false` → 标记队列为非活跃（`active=False`），`pre_execute` 跳过
- `required: true`（默认）→ 报错

### 6.2 Fan-out

同一个上游输出可连接多个下游队列。wiring 层通过 `outputs[name].append(target_queue)` 实现扇出。Sentinel 在每个消费者中独立传播。

### 6.3 变长源冲突检测

静态检测不同 `VARIABLE_OUTPUT` 源的序列输出汇入同一节点：
- 多个不同变长源 → 报错
- 固定长度 + 变长混合 → 报错

### 6.4 configs 解析

全局配置优先级：`用户显式传入` > `全局默认设置文件` > `YAML default` > `Op CONFIGS default`

嵌套 `route_configs`（`configs.stacker.sigma_clip.rej_high`）由递归展平后以 dotted key 查找。

---

## 7. Op 接口声明

```python
class BaseOp:
    INPUTS: dict[str, Any] = {}      # { name: { type, required? } }
    CONFIGS: dict[str, Any] = {}     # { name: { type, default?, required? } }
    OUTPUTS: dict[str, Any] = {}     # { name: { type } }
    VARIABLE_OUTPUT: bool = False    # 变长输出（Filter 类）
    DATA_PARALLEL: bool = False      # 允许数据并行段
    DECOMPOSABLE: bool = False       # 支持分布式归约
```

type 常见值：`sequence`、`image`、`object`、`int`、`float`、`str`、`bool`、`dict`、`list`

---

## 8. 完整示例

### 8.1 标准 SubDAG：`sigma_clip.yaml`

```yaml
description: "Sigma Clipping sub-DAG"
version: "1"

inputs:
  data: { type: sequence }
  fnames: { type: sequence, required: false }

configs:
  int_weight: { type: bool, default: true }
  mask: { type: image, required: false, default: null }
  rej_high: { type: float, default: 3.0 }
  rej_low: { type: float, default: 3.0 }
  max_iter: { type: int, default: 5 }
  early_converge_ratio: { type: float, default: 0.99 }
  buffer_mode: { type: str, default: "auto" }

nodes:
  mean_stacker:
    op: MeanStackerOp
    inputs:
      data: inputs.data
    configs:
      int_weight: configs.int_weight
      mask: configs.mask
    outputs:
      result: { type: image }
      statistics: { type: image }

  disk_buffer:
    op: DiskBufferWriterOp
    inputs:
      data: inputs.data
      fnames: inputs.fnames
    configs:
      buffer_mode: configs.buffer_mode
    outputs:
      buffer_handle: { type: image }

  sigma_clip_iter:
    op: SigmaClipIteratorOp
    configs:
      fgp_total: mean_stacker.statistics
      buffer_handle: disk_buffer.buffer_handle
      rej_high: configs.rej_high
      rej_low: configs.rej_low
      max_iter: configs.max_iter
      early_converge_ratio: configs.early_converge_ratio
    outputs:
      result: { type: image }
      statistics: { type: image }

outputs:
  result: sigma_clip_iter.result
  statistics: sigma_clip_iter.statistics
```

### 8.2 Meta YAML：`stack.meta.yaml`

```yaml
description: "均值/中值叠加算法"
version: "2"

inputs:
  fnames: { type: sequence, required: true }

routes:
  stacker:
    options:
      mean:       null
      sigma_clip: null
      median:     null
      huber_mean: null
    default: mean

configs:
  int_weight:       { type: bool, default: true }
  exif_reduce_type: { type: str,  default: "sum" }
  output_filename:  { type: str,  default: "result.tif" }
  output_dtype:     { type: str,  default: "uint8" }
  loader_type:      { type: str,  default: "img_file_list" }
  loader_configs:   { type: dict, default: {} }

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
    op: stacker.meta.yaml
    route_key: stacker
    routes:
      stacker: routes.stacker
    inputs:
      data: data_loader.result
    route_configs:
      mean:
        int_weight: configs.int_weight
      sigma_clip:
        int_weight: configs.int_weight
      median: {}
      huber_mean:
        int_weight: configs.int_weight
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

### 8.3 Meta YAML 带节点开关：`calibration_stack.meta.yaml`（节选）

```yaml
routes:
  bias_stacker:
    options:
      none: NoneOutputOp
      master: LoadSingleImageOp
      mean: MeanStackerOp
      median: median_stack_core.yaml
    default: none

nodes:
  bias_stacker:
    route_key: bias_stacker
    route_inputs:
      mean:   { data: inputs.bias_fnames }
      median: { data: inputs.bias_fnames }
    route_configs:
      master: { path: configs.bias_master_path }
      mean:   { int_weight: configs.int_weight }
    outputs:
      result: { type: image }
```

### 8.4 Meta SubDAG 透传：`sky_ground_stack.meta.yaml`（节选）

```yaml
routes:
  sky_stacker:
    options:
      mean:       null
      sigma_clip: null
      huber_mean: null
    default: sigma_clip

nodes:
  sky_stacker:
    op: stacker.meta.yaml               # Meta SubDAG
    route_key: sky_stacker              # 父图路由 key（选择 route_configs 分支）
    routes:
      stacker: routes.sky_stacker       # 透传到子图的 stacker route
    inputs:
      data: star_aligner.result
    route_configs:
      sigma_clip:
        int_weight: configs.int_weight
        mask: load_mask.result
        # rej_high, rej_low, max_iter — auto-wired
    outputs:
      result: { type: image }
```

---

## 9. 执行入口

### 9.1 `run_from_yaml()`

```python
results = await run_from_yaml(
    "hoshicore/dag/stack.meta.yaml",
    global_inputs={"fnames": ["a.tif", "b.tif", ...]},
    global_configs={"int_weight": True, "output_dtype": "uint16"},
    route_choices={"stacker": "sigma_clip"},
)
```

参数：
- `yaml_path`：YAML 文件路径
- `global_inputs`：全局输入数据（`name → Sequence`）
- `global_configs`：全局配置值（`name → value`）
- `route_choices`：路由选择（仅 Meta YAML 需要）。未提供的 route_key 使用 default
- `dag_search_paths`：SubDAG 搜索路径列表，默认 `[hoshicore/dag/base/, ...]`
- `tracker`：进度追踪器
- `cancel_event`：外部取消信号

---

## 10. 校验规则汇总

| 规则 | 层 | 错误类型 |
|------|-----|---------|
| `nodes` 必须存在 | build | DagSpecError |
| `outputs` 必须存在且为 dict | build | DagSpecError |
| 每个节点必须有 `op`（非空字符串） | build | DagSpecError |
| 每个节点必须有 `outputs` 声明 | build | DagSpecError |
| link 引用的全局 inputs/configs 必须已声明 | build | DagSpecError |
| link 引用的节点输出必须已声明 | build | DagSpecError |
| 全局 inputs 的 type 必须为 sequence | build | DagSpecError |
| 拓扑排序失败（环或阻塞） | build | DagSpecError |
| route_key 引用的路由未定义 | meta | MetaResolveError |
| route choice 无效 | meta | MetaResolveError |
| enabled 引用非 configs 命名空间 | meta | MetaResolveError |
| bypass 推断失败（多 input 未声明） | meta | MetaResolveError |
| SubDAG required input 未被父图布线 | flatten | ValueError |
| SubDAG config 无 default 且未被父图覆盖 | flatten | ValueError |
| Op 注册名未找到 | wiring | ValueError |
| required input 未布线 | wiring | ValueError |
| 变长源冲突 | wiring | ValueError |
