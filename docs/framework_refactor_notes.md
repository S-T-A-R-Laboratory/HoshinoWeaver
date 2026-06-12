# 框架层改造建议

本文记录 HoshinoWeaver 框架层面的潜在缺陷、风险判断与建议改造方向。定位是后续排期参考，不要求一次性完成；建议优先处理会导致卡死、结果不一致或后续扩展成本快速上升的问题。

## 总体判断

当前架构的主线是合理的：GUI/CLI 收集输入与配置，YAML 描述 DAG，engine 负责编译、校验、布线和执行，ops/component 承载具体算法与数据结构。真正的复杂度集中在以下几处：

- 异步流式执行依赖队列、sentinel、CancellationToken、输出收集协程共同协作，生命周期语义较分散。
- Meta YAML、SubDAG flatten、route_configs、enabled/bypass 等编译逻辑跨多个模块互相了解内部约定。
- 配置默认值来源较多，GUI、CLI、YAML、Op、default_settings 的优先级不够显式。
- Op schema 仍是弱约束 dict，类型与接口约束很大程度依赖约定。

如果只选三件优先做，建议顺序是：

1. 重构执行、取消、队列关闭协议，降低卡死和难复现问题。
2. 引入统一 ConfigResolver，解决配置来源和默认值优先级问题。
3. 整理 YAML 编译 IR，让 meta、flatten、route、bypass 复杂度收束到编译层。

## P0: 执行、取消与队列关闭协议 ✅ 已完成

> 详见 `docs/dag_engine.md`。

已实现：
- ✅ 队列显式取消语义：`RichContextQueue.force_cancel(token)` + 入口守卫
- ✅ `BaseOp.execute()` 简化为 3 行，取消传播收束到 executor
- ✅ `DAGExecutor` 重写：统一 cancel_all + task.cancel，结构化 `DAGExecutionError`
- ✅ `run_dag()` 改造为 runtime group：watcher task + feeders/collector 容忍取消
- ✅ GUI/CLI 结构化根因展示
- ✅ 测试覆盖：force_cancel 唤醒、根因选择、上下游取消传播、外部取消、正常回归

待后续：
- `ParallelBaseOp._execute_concurrent()` 中 CancellationError 应 re-raise 而非转为 _STOP（影响 cancelled_nodes 统计准确性）
- 节点显式状态（pending/running/completed/failed/cancelled）和执行诊断日志可按需追加

## P0: 配置解析与默认值优先级

### 现状

配置来源包括：

- YAML `configs.default`
- YAML `route_configs`
- Op `CONFIGS.default`
- `hoshicore/default_settings.yaml`
- GUI 动态面板收集值
- OutputPanel 收集值
- CLI `--config`
- SubDAG 展开时的自动映射和省略
- wiring 中未布线 config 的 auto-inject

相关逻辑分布在：

- `hoshicore/engine/wiring.py`
- `hoshicore/engine/meta.py`
- `hoshicore/engine/flatten.py`
- `ui/panel_builder.py`
- `ui/UIUtils.py`
- `launcher.py`

### 风险

- GUI 显示值、CLI inspect 值和实际运行值不一致。
- route config 在父图、子图、flatten 后命名空间变化时出现覆盖或遗漏。
- 同名 config 在全局默认、YAML default、Op default 中优先级不透明。
- 问题定位困难：日志只有最终配置，不知道值来自哪里。

### 建议目标

引入统一 `ConfigResolver`，让所有入口都经过同一套解析逻辑。

建议输出不只是 `dict[str, Any]`，而是带来源信息的 resolved result：

```python
ResolvedConfig(
    values={"int_weight": True, "output_dtype": "uint16"},
    sources={
        "int_weight": "user.gui",
        "output_dtype": "default_settings",
    },
)
```

优先级建议明确为：

1. 用户显式输入：GUI、CLI、API caller。
2. 全局默认设置：`default_settings.yaml` 中 enabled 的条目。
3. 当前 YAML 或 route_configs 的 default。
4. Op `CONFIGS.default`。
5. 无默认且 required 时，在运行前报错。

### 可拆任务

1. 写文档明确配置优先级，先不改代码。
2. 抽出 `_resolve_configs()` 和 default settings 合并逻辑到独立模块。
3. 让 CLI `--inspect` 和 GUI 面板都读取同一个 resolved schema。
4. wiring 只消费 resolved configs，不再自行决定业务优先级。
5. 日志中打印配置来源，至少 debug 模式可见。

## P1: YAML 编译 IR

### 现状

当前编译链路大致是：

```text
raw YAML
  -> meta_resolve()
  -> flatten_sub_dags()
  -> validate_and_build_order()
  -> instantiate_and_wire()
```

Meta YAML、SubDAG、route 透传、route_configs 折叠、enabled/bypass、`__inactive__` 标记等约定分布在 `meta.py`、`flatten.py`、`build.py`、`wiring.py` 中。

### 风险

- 编译阶段和运行阶段边界不清，wiring 需要理解 flatten 产物。
- 子图作为独立 DAG 与嵌入父图时，配置命名规则可能不一致。
- 增加新的 YAML 能力时，需要同时修改多个阶段。
- 错误提示可能基于展开后的节点名，用户难以映射回原 YAML。

### 建议目标

定义一个明确的中间表示 IR。原始 YAML 的所有高级语法都编译成 IR，后续 build/wiring/executor 只理解 IR，不理解 meta 或 flatten 的历史产物。

IR 中建议包含：

- 节点列表、端口、边。
- 全局输入和配置 schema。
- route 解析结果。
- 原始 YAML 位置信息，用于错误提示。
- inactive optional input 的结构化表示，而不是字符串标记。
- 子图命名空间和 source map。

### 可拆任务

1. 先定义 IR dataclass，不急着替换全链路。
2. 让 `validate_and_build_order()` 支持从 IR 构造 ValidatedDag。
3. 将 `__inactive__` 替换成结构化字段。
4. 为 SubDAG 展开保留 source map，错误信息可以指回父 YAML 节点。
5. 最后收敛 `meta.py` 和 `flatten.py` 的输出格式。

## P1: Op Schema 强类型化

### 现状

Op 通过类属性 `INPUTS`、`CONFIGS`、`OUTPUTS` 声明接口，但结构是普通 dict，类型是字符串。DAG 校验主要检查 link 是否存在、结构是否满足约定，但较少进行运行前类型校验。

### 风险

- 错误到算法深处才暴露，例如 config 类型不符、image 为 None、sequence/scalar 混用。
- Op schema 字段逐渐增加后，缺少统一解析和校验。
- UI schema、DAG schema、Op schema 容易各自演化。

### 建议目标

逐步引入结构化 schema，不一定需要重依赖，可以先使用 dataclass。

建议统一字段：

- `type`
- `required`
- `default`
- `is_sequence`
- `global_key`
- `description`
- `resource_hint`
- `ui_hint`

### 可拆任务

1. 增加 schema normalize 层，把现有 dict 转成 dataclass，保持兼容。
2. build/wiring/preflight 统一消费 normalize 后的 schema。
3. 增加运行前类型校验，先覆盖 config，再覆盖 image/sequence。
4. 新 Op 要求使用结构化 schema，旧 Op 逐步迁移。

## P1: GUI 与后端运行服务解耦

### 现状

GUI 已经通过 `PanelSchema` 和 `DynamicConfigPanel` 走数据驱动方向，这是好的。但任务启动仍然在 UI 层直接拼 `global_inputs`、`global_configs`、`route_choices` 并调用 `run_from_yaml()`。

### 风险

- GUI 需要理解后端配置细节，后端 schema 改动会牵动 UI。
- CLI 和 GUI 的运行逻辑重复，默认值和预检行为可能分叉。
- 取消、日志、预检、结果预览等流程难以复用。

### 建议目标

引入后端 service 层，例如 `RunRequest` / `RunService`：

```python
RunRequest(
    workflow="startrail",
    input_files=[...],
    configs={...},
    route_choices={...},
    output={...},
)
```

GUI 和 CLI 都构造 `RunRequest`，由 service 负责：

- workflow path 解析
- config resolve
- inspect/preflight
- run/cancel
- progress/log/result 统一回调

### 可拆任务

1. 新增 `hoshicore/engine/service.py` 或 `hoshicore/app/run_service.py`。
2. 将 CLI 中的参数转换逻辑收束为 request builder。
3. GUI 只提交 request，不直接调用 `run_from_yaml()`。
4. 将 preflight callback、progress tracker、cancel_event 统一纳入 request/runtime。

## P1: 资源预检模型

### 现状

已有 `preflight.py` 和 Op `estimate_resources()`，方向正确。问题是实际资源占用受 DAG 拓扑、buffer mode、frame dtype、并发窗口、mask、缓存策略影响，单 Op 估计很容易偏乐观或与执行不一致。

### 风险

- 预检通过，但执行时仍然内存或磁盘不足。
- 用户看到 fallback 建议，但不知道由哪个节点触发。
- 子图展开后资源责任不清。

### 建议目标

资源估算提升为 DAG 级模型：

- 每个 Op 声明峰值内存、持有资源、释放时机、磁盘缓存行为。
- DAG 层按拓扑和并发关系估算峰值。
- preflight 报告包含节点来源和建议。

### 可拆任务

1. 在 preflight 报告中加入节点级资源明细。
2. 为 DiskBufferWriter、SigmaClip、Median、Mean 等核心 Op 补齐估算测试。
3. 引入保守系数，优先避免低估。
4. 与 ConfigResolver 打通，确保估算使用的配置就是实际运行配置。

## P2: 错误信息与可观测性

### 现状

当前日志较多，但面向用户的错误链路还可以更清晰。DAG 展开后节点名可能很长，异常可能经过队列和 executor 包装，GUI 最终弹窗未必能说明用户该改哪个参数或文件。

### 建议目标

- 每个运行任务有 task id。
- 每个节点失败能展示：节点名、原始 YAML 位置、输入来源、关键配置。
- debug/trace 模式可以导出 resolved DAG 和 resolved configs。
- GUI 弹窗显示用户可行动的信息，详细堆栈进日志文件。

### 可拆任务

1. 增加 `--dump-resolved-dag` 或 debug 日志输出。
2. 错误对象增加 source map 信息。
3. GUI 错误弹窗区分用户输入错误、资源不足、算法内部错误、未知错误。

## 建议路线图

### 第一阶段：降低运行风险

- 增加执行状态日志。
- 补死锁和取消相关测试。
- 抽出 ConfigResolver 的雏形。
- 明确配置优先级文档和日志。

### 第二阶段：收束编译复杂度

- 定义 YAML IR。
- 让 build/wiring 逐步消费 IR。
- 替换 `__inactive__` 等字符串协议。
- 加 source map，改善错误提示。

### 第三阶段：提升扩展能力

- Op schema dataclass 化。
- GUI/CLI 统一 RunService。
- DAG 级资源预检。
- 完善 inspect、dump、诊断工具。

## 非目标

以下内容不建议在上述改造初期同时推进：

- 大规模重写所有 Op。
- 一次性替换整个 executor。
- 同时改 UI 视觉和后端运行协议。
- 为 schema 引入过重依赖，除非已经确认收益大于迁移成本。

建议先用小步、兼容、可测试的方式把协议边界定清楚，再逐步替换内部实现。
