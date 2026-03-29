# DAG 计算图规范（YAML）

本文档描述如何用 YAML 声明一个计算图（DAG, Directed Acyclic Graph）。

图的构建器只做“语义校验 + 推导执行顺序”，节点对应的 `Op` 实现可以在后续阶段再落地。

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
