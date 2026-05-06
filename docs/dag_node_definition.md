# DAG 计算图规范（YAML）

本文档描述如何用 YAML 声明一个计算图（DAG, Directed Acyclic Graph）。

图的构建器只做“语义校验 + 推导执行顺序”。

## 总体语义

1. 图由若干个节点 `nodes` 组成，每个节点声明：
   - `op`：操作名称（占位用，构建器不实例化）
   - `inputs`：该节点在执行时需要的输入（来自全局 `inputs` 或其它节点输出）
   - `configs`：该节点在执行时需要的配置参数（来自全局 `configs` 或其它节点输出）
   - `outputs`：该节点执行完成后产生的输出字段（带 `type`）
2. 节点执行顺序由依赖关系决定：
   - 当某节点 `inputs/configs` 中引用的“其它节点输出”全部就绪时，该节点可执行。
   - 对引用全局 `inputs.*` 或 `configs.*` 的情况，它们视为从图的入口直接可用，不产生节点依赖。
3. 图的完成条件由顶层 `outputs` 决定：
   - 只要 `outputs` 列表中列出的所有“链接”（links）对应的值都获取到了，图就执行完。
   - 因此 `outputs` 支持多个目标结果。

## 顶层字段

### `description`（可选）

- string，图的描述。

### `version`（可选）

- string，图定义版本号，用于兼容不同 schema。

### `inputs`

- 类型：`object`（键值对）
- 含义：全局入口输入，且“序列性质”的输入应统一归到这里。
- 每个条目形如：
  ```yaml
  <input_name>:
    { "type": "<type>", "description": "...", "default": <optional> }
  ```
- 约定（建议）：全局 `inputs.*` 的 `type` 通常为 `sequence`。

### `configs`

- 类型：`object`（键值对）
- 含义：全局标量配置入口（单值）。
- 每个条目形如：
  ```yaml
  <config_name>:
    { "type": "<type>", "description": "...", "default": <optional> }
  ```
- 约定（建议）：全局 `configs.*` 的 `type` 通常为 `float/string/bool/...`（不建议用 `sequence`）。

### `nodes`

- 类型：`object`
- 键是节点 id（`node_name`），值是该节点的 `NodeSpec`。

### `outputs`

- 类型：`list<string>`
- 含义：图的目标完成条件，列出一个或多个 link。
- link 的语法见下节“链接方式”。

## NodeSpec（节点声明）

节点条目（`nodes.<node_name>`）结构：

```yaml
<node_name>:
  op: <OpName>
  inputs: # 可选
    <input_arg_name>:
      src: <Link>
  configs: # 可选
    <config_arg_name>:
      src: <Link>
  outputs: # 必需
    <output_name>: { "type": "<type>", ... }
```

### `op`

- 类型：string
- 含义：操作名称，用于后续把节点映射到具体 `Op` 实现。
- 当前阶段构建器不实例化 `Op`，只做字段校验。
- 值为 `.yaml` 后缀时，wiring 层自动创建 SubDagOp 封装子图。

### `inputs`

- 类型：`object`（可选）
- 含义：节点运行时需要的输入参数映射。
- 每个输入参数必须至少提供 `src`：
  ```yaml
  inputs:
    <input_arg_name>:
      src: <Link>
  ```
- `src` 用于指定输入来自哪里（全局 `inputs.*`、全局 `configs.*`、或其它节点的输出）。

### `configs`

- 类型：`object`（可选）
- 含义：节点运行时需要的配置参数映射。
- 同样使用 `src` 指定来源：
  ```yaml
  configs:
    <config_arg_name>:
      src: <Link>
  ```
- 约定（建议）：节点的 `configs` 只接收“标量/对象”来源（不接收全局 `inputs.*`）。

### `outputs`

- 类型：`object`（必需）
- 含义：节点执行完成后产生的输出字段。
- 每个输出字段必须提供至少 `type`：
  ```yaml
  outputs:
    <output_name>: { "type": "<type>" }
  ```
- `output_name` 用于被其它节点通过 `<node_name>.<output_name>` 进行引用。

## 链接方式（Link）

link 用一个字符串表示某个“数据来源”。

语法：

1. `inputs.<input_name>`
2. `configs.<config_name>`
3. `<node_name>.<output_name>`（引用其它节点的输出）

构建器在做合法性检查时会把 link 解析成来源类型，并验证：

- 是否存在对应的全局条目（`inputs/configs`）或节点输出（`<node>.<output>`）。
- link 指向的输出字段名是否已在声明中定义。

## Meta YAML 路由节点（扩展语法）

Meta YAML（`*.meta.yaml`）在标准 DAG YAML 基础上，为节点新增三个可选字段，用于在单一文件中声明多种可选处理路径。Meta YAML 需通过 `meta_resolve()` 预处理为标准 spec dict 后才能传入构建器。

### 路由节点声明

路由节点使用 `route` 替代 `op`：

```yaml
<node_name>:
  route:                              # 替代 op，列出可选实现
    <choice_a>: <OpName_or_YAML>
    <choice_b>: <OpName_or_YAML>
  inputs:                             # 可选，所有选项共享的输入
    <input_arg_name>: <Link>
  route_inputs:                       # 可选，按 route 选项分组的专属输入
    <choice_a>:
      <input_arg_name>: <Link>
  configs:                            # 可选，所有选项共享的配置
    <config_arg_name>: <Link>
  route_configs:                      # 可选，按 route 选项分组的专属配置
    <choice_a>:
      <config_arg_name>: <Link>
  outputs:                            # 所有选项共享的输出端口契约
    <output_name>: { "type": "<type>" }
```

### `route`

- 类型：`object`
- 含义：该节点位置可选的多种实现，key 是选项名（如 `mean`、`none`），value 是 Op 类名或子 DAG YAML 路径。
- 与 `op` 互斥：节点要么指定 `op`（标准节点），要么指定 `route`（路由节点）。

### `route_inputs`

- 类型：`object`（可选）
- 含义：按 route 选项分组的专属输入。只有选中的 route 对应的输入会被合并到 `inputs`，其余丢弃。
- 合并规则：`route_inputs[choice]` 中的条目追加/覆盖共享 `inputs`。

### `route_configs`

- 类型：`object`（可选）
- 含义：按 route 选项分组的专属配置。只有选中的 route 对应的配置会被合并到 `configs`，其余丢弃。
- 合并规则：`route_configs[choice]` 中的条目追加/覆盖共享 `configs`。

### SubDAG 兼容性

不同 route 选项的接口可能不完全一致（如 `MeanStackerOp` 接受 `int_weight`，但 `median_stack_core.yaml` 不声明该参数）。为避免 wiring 层因多余的参数报错，**所有可能不一致的参数应放入 `route_inputs` / `route_configs`**，而非共享区域。

### 预处理

```python
from hoshicore.engine.meta import meta_resolve

standard_spec = meta_resolve(meta_spec, route_choices={
    "bias_stacker": "none",
    "main_stacker": "sigma_clip",
})
# standard_spec 为标准 DAG spec dict，可传入 validate_and_build_order()
```
