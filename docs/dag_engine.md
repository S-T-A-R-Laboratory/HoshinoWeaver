# DAG Engine 架构概览

本文描述当前 DAG 引擎的编译、布线、资源规划、队列通信和终止机制。YAML 字段与链接语法见 [DAG 节点定义规范](./dag_node_definition.md)，算子扩展接口见 [Operator 目录与设计](./op.md)。

## 编译链路

完整入口为 [run_from_yaml()](../hoshicore/engine/wiring.py)，执行顺序如下：

```text
读取 YAML + 合并全局默认设置
  → meta_resolve()（需要路由或节点开关解析时）
  → flatten_sub_dags()
  → validate_and_build_order() → ValidatedDag
  → 解析 YAML 配置默认值
  → run_preflight_checks() + 处理预检结果
  → plan_runtime() + apply_runtime_plan()
  → run_dag()
      → instantiate_and_wire()
      → 并发运行 feeders、DAGExecutor.execute()、输出收集
```

| 阶段 | 模块 | 输入 → 输出 |
| --- | --- | --- |
| 路由编译 | [meta.py](../hoshicore/engine/meta.py) | Meta spec + route_choices + 运行时配置 → 选中路由、解析节点开关后的标准 spec |
| 子图展开 | [flatten.py](../hoshicore/engine/flatten.py) | 节点的 `.yaml` 引用 → 命名空间化的 flat nodes；Meta 子图先解析再展开 |
| 校验与拓扑排序 | [build.py](../hoshicore/engine/build.py) | flat spec → `ValidatedDag`，包含节点、依赖、全局声明和目标输出所需节点的 `exec_order` |
| 资源与参数预检 | [preflight.py](../hoshicore/engine/preflight.py) | DAG + 有效配置 + 输入 → 内部资源报告与各项检查结果 |
| 运行时规划 | [runtime_plan.py](../hoshicore/engine/runtime_plan.py) | DAG + 资源预算 + 后端信息 → 配置覆写与后端规划提示 |
| 实例化与布线 | [wiring.py](../hoshicore/engine/wiring.py) | `ValidatedDag` + 数据 + 配置 → Op 实例、feeders、输出队列及变长输入节点集合 |
| 并发执行 | [executor.py](../hoshicore/engine/executor.py) + `wiring.py` | 节点执行、输入供给与结果收集 → 结果字典或异常 |

其余引擎模块：

| 模块 | 职责 |
| --- | --- |
| [registry.py](../hoshicore/engine/registry.py) | `@register_op()` 按类名或显式名称登记到 `REGISTERED_OP` |
| [inspect.py](../hoshicore/engine/inspect.py) | 仅读取 YAML 并浅层解析路由参数，提取 CLI `--inspect` 所需 schema；不实例化、布线或执行 Op |
| [visualize.py](../hoshicore/engine/visualize.py) | 生成路由解析后的 Mermaid 图，可选展开 YAML 子图 |

`run_dag()` 接受已校验的 `ValidatedDag`，负责布线和执行；全局默认设置的读取、Meta 编译、子图展开、预检和运行时规划由 `run_from_yaml()` 完成。手动调用时需自行完成所需的前置阶段。

### 关键编译约定

- **命名空间**：子图节点以 `parent.child` 命名，链接通过 `rsplit(".", 1)` 分离节点名与端口名。
- **目标裁剪**：校验器检查声明与链接，再从全局 `outputs` 回溯所需节点；仅对所需子图排序并检测环。
- **配置优先级**（初始解析，高 → 低）：运行时显式配置 > `default_settings.yaml` 的启用项 > 工作流 YAML 默认值 > Op 配置默认值。嵌套路由配置展平为 dotted key；带 `global: true` 的嵌套参数还可使用同名全局值。
- **有效配置一致性**：预检与 planner 接收已合并 YAML 默认值的配置；预检修复、planner 覆写应用后再布线。Op 默认值是未布线端口的后备值，并非覆盖上游计算结果。

### 实例化与布线

`instantiate_and_wire()` 按 `exec_order` 创建算子，连接上游输出与下游输入/配置队列，生成全局输入和配置的 feeder，并为目标输出创建收集队列。

- `__inactive__` 标记子图未接入的可选输入；布线时跳过并设 `queue.active = False`。一般可选输入未布线时也会标记为非活跃。
- 未布线配置优先使用带 `global: true` 声明的有效全局值，否则注入 Op 声明的 `default`。无默认值时当前仅记录警告，可能使 `pre_execute()` 一直等待，见「已知限制」。
- 必填输入未布线、实际使用的全局输入/配置缺失、算子未注册时抛出 `ValueError`。
- 静态检查序列输入的变长来源，拒绝不同 Filter 源汇合，或固定长度与变长源混合。运行时 `BaseOp.pre_execute()` 还会检查输入长度是否一致。

## 运行时执行模型

### Runtime Group 结构

当前引擎在同一 asyncio 事件循环中调度节点。算子可通过 `_run_cpu()` 将同步计算卸载到线程池；异步并发本身不意味着 CPU 计算自动并行。

`run_dag()` 将 feeder 协程预先注册为 Tasks，再启动三组并发工作和一个独立的取消监控 task：

```text
run_dag()
  ├─ watcher_task（独立，不加入 gather）
  │    └─ 等待 executor.cancel_event，调用 cancel_all() 并取消节点 Tasks
  └─ gather:
       ├─ _run_feeders()      ← 将全局输入/配置推入队列
       ├─ executor.execute() ← 管理节点 Tasks
       └─ _collect_outputs() ← 收集目标输出
```

外部事件的当前实现存在限制，详见「外部取消」；上图描述调用关系，不保证仅设置事件即可结束整个管线。

布线后按进度报告策略向选中的节点注入真实 tracker，其余保留 `DummyTracker`。各 Op 共用 executor 的取消事件；`run_dag()` 的 `finally` 取消并等待残留 feeder/监控 Tasks，关闭 tracker。

### DAGExecutor 职责

- 按拓扑顺序创建所有节点的 `asyncio.Task`，节点之间由队列等待和背压协调。
- 用 `gather(return_exceptions=True)` 等待节点完成，区分真实失败与取消。
- 节点真实失败时执行 `cancel_all()`，并对其他未完成节点调用 `task.cancel()`。
- 收集 `failed_nodes`、`cancelled_nodes`，补记在启动前就被取消的节点。
- 有真实失败时按 `exec_order` 从失败记录中选择根因，抛出 `DAGExecutionError`；仅有节点取消时 executor 本身正常返回。

### Op 执行生命周期

```python
async def execute(self) -> None:
    configs = await self.pre_execute()
    await self._async_execute(configs)
    await self._send_sentinel()
```

`pre_execute()` 先等待活跃输入的长度并校验，广播输出长度，然后读取一次性配置值。`_send_sentinel()` 仅向 **sequence 输出**发送结束标记；单图、整数、对象等输出不追加 sentinel。

`BaseOp.execute()` 不负责全局取消传播，异常交给 executor。子类可管理自己的清理逻辑；部分并行处理路径会吞掉取消异常，见「已知限制」。

### 输出收集

收集器按已知长度读取，或对未知长度读到 `StreamExhausted`。当前返回规则：单个结果直接返回值，多个结果返回列表，空序列不添加对应键。目标输出的全部结果保存在列表中，因此将大图像序列直接设为全局输出会累积内存；逐帧保存可通过 `BatchImageSaveOp` 完成。

## 队列通信协议

### 数据流

算子的输入/配置队列使用 [RichContextQueue](../hoshicore/component/queue.py)。`BaseOp.MAX_SIZE` 默认为 1，目标输出收集队列也使用 `maxsize=1`。

对于有限预读的保序等长流，Op 可通过 `required_lookahead(configs)` 声明输出位置所需后续输入帧数的上界。wiring 在启动任务前沿 sequence 路径传播该上界，在具有共同生产节点的分支汇合处扩大消费端输入队列。普通节点默认预读为 0。该策略不保证最小内存，但容量不随序列长度增长；预检将额外容量计入队列内存估算。

- **背压**：满队列会阻塞生产者，限制队列中的待处理数据数量；整体内存还包括算子状态、并发任务、帧缓存和输出收集。
- **Length 先行**：生产者先 `set_length()`，消费者通过 `get_length()` 等待长度就绪。整数表示已知长度，`None` 表示未知长度，不表示结束。
- **Sentinel**：正常结束使用专用对象 `BaseQueue._SENTINEL`；`get()` 回填该信号后抛出 `StreamExhausted`，其他并发消费者也能感知结束。数据值 `None` 可正常传递。
- **取消令牌**：`get()` 遇到 `CancellationToken` 时回填并抛出 `CancellationError`。
- **FileCacheQueue**：使用 pickle/json/numpy 将数据写入临时文件，内部队列传递路径；`get()` 反序列化并删除文件，调用方收到原数据。信号直接入队，跳过文件序列化。当前默认布线使用 `RichContextQueue`，不自动将大对象切换为文件队列。

完整长度契约、检查层次和边界问题见 [长度广播协议](./length_protocol.md)。

### 取消语义：`force_cancel(token)`

`DAGExecutor.cancel_all()` 遍历节点输入、配置和已连接输出队列，调用 `force_cancel()`。队列取消标记独立于普通数据，重复取消无效。

| 等待位置 | 取消机制 |
| --- | --- |
| 后续 `get()` / `put()` | `_check_cancelled()` 入口守卫立即抛出 |
| `await queue.get()` | 注入 token；队列满时丢弃一项后注入 |
| `await _length_event.wait()` | 设置长度事件，`get_length()` 唤醒后检查取消标记 |
| `await queue.put()` 或写锁等待 | `force_cancel()` 不保证解除已有写等待，节点 Task 的 `task.cancel()` 补充打断 |

节点失败时，顶层执行进入异常清理阶段，残留 feeder Tasks 也会被取消。取消 asyncio 等待不会强制终止已在线程中执行的同步计算。

## 终止协议

| 触发源 | 路径 | 对外表现 |
| --- | --- | --- |
| 节点真实异常 | `BaseOp.execute()` → executor | `DAGExecutionError`，携带根因与失败/取消记录 |
| 取消顶层运行 Task | `task.cancel()` → `run_dag()` 清理 | `asyncio.CancelledError`；CLI 的 Ctrl-C 由 `asyncio.run()` 和 launcher 的 `KeyboardInterrupt` 分支处理 |
| 仅设置外部取消事件 | watcher 取消节点 Tasks | 可能仍等待结果，或正常返回已有结果；详见下节 |
| 编译或布线错误 | 路由解析、展开、校验、实例化/布线 | `MetaResolveError` / `DagSpecError` / `ValueError` / 展开深度超限的 `RuntimeError` 等 |
| 预检中止 | 预检回调选择 abort | `PreflightAbortError` |

### 节点失败时序

```text
node_A raises ValueError
  → _run_node: 记录失败 → cancel_all() → 取消其他节点 Tasks
  → 队列 get/get_length 等待收到 CancellationError
    或节点等待被 task.cancel() 打断，收到 CancelledError
  → feeders/collector 捕获取消后退出
  → executor 等待节点收尾，按 exec_order 选择失败根因
  → run_dag() 传播 DAGExecutionError，并清理残留 Tasks
```

### 外部取消

当前 `run_dag()` 将外部 `cancel_event` 直接赋给 executor。watcher 等到事件被置位后调用 `cancel_all()`，但该方法的幂等守卫看到同一事件已置位就返回，**没有执行队列的 `force_cancel()`**。watcher 随后仅取消节点 Tasks。

因此，仅调用 `cancel_event.set()` 可能使 feeder 或 collector 保持等待；若其他工作已经完成，也可能返回已有结果，而非抛出 `CancelledError`。预期的「事件置位 → 所有队列取消 → 必然抛出 CancelledError」不符合当前实现。

GUI 的 `cancel_task()` 实际同时设置事件并取消顶层 Task，打断 `run_dag()` 的 gather 等待并进入清理。这与只设置事件的调用方式不同。事件守卫与队列取消应在后续代码修复中分别处理。

### `DAGExecutionError` 结构

异常在实例初始化时提供以下字段：

```python
root_cause: Exception
root_node: str
failed_nodes: list[tuple[str, Exception]]
cancelled_nodes: list[str]
```

根因是失败记录中 `exec_order` 最靠前的节点，不保证它与其他同时失败节点存在因果关系。GUI 弹窗和 CLI 日志读取这些字段展示失败详情。

## 资源预检与运行时规划

### 预检接口

`run_from_yaml()` 调用 `run_preflight_checks()`，先运行资源检查，再运行 `config_validity_check()`：

- 各 Op 通过 `estimate_resources()` 声明内存与磁盘需求；资源检查累加所需节点的估算，并加入 sequence 输出端口的队列开销。
- 内部 `PreflightReport` 保存资源预算、非分块内存和降级后的预算预测，交给 planner。
- 对外返回 `CheckResult` 列表；每项含 `PreflightIssue`，其中可附 `PreflightFix`。没有旧文中的 `warnings / proposed_fallbacks` 字段。
- 回调 `preflight_callback(result)` 返回 `apply / ignore / abort`：应用建议配置、忽略问题或抛出 `PreflightAbortError`。GUI 与 CLI 各自提供交互策略。
- 无回调时，有 fix 且有效配置 `auto_fallback` 为真才自动应用，否则记录警告。问题标为 error 本身不保证中止，是否中止取决于调用策略。

当前资源检查从 `global_inputs['fnames']` 的首个文件探测尺寸与 dtype。未提供非空 list/tuple 文件列表、或探测失败时，该项返回零估算；参数检查仍会运行。

### 分块规划

`runtime_planner` 启用时，`plan_runtime()` 生成后端提示，并为 `CHUNK_PLANNED` 算子规划共享的 `chunk_rows`：

- 从主机内存预算扣除预检估算的非分块内存与队列开销，累加分块算子每行成本。
- 对适用的 `cuda_host_io` 后端同时考虑可用显存、固定开销和每行成本。
- 按行对齐和配置的上下限约束块高；显式传入的数值 `chunk_rows` 保留，`"auto"` 允许规划。
- `apply_runtime_plan()` 将配置覆写写回有效配置，随后才实例化和布线。后端提示用于规划与日志，实际 kernel 分发仍由 custom-op 层决定。

## 已知限制

- **事件单独取消不完整**：队列取消可能被事件幂等守卫跳过，见「外部取消」。
- **未布线配置可能等待不结束**：缺少 default 的 CONFIGS 端口当前仅发出警告；必填输入缺失则已转换为硬错误。
- **并行节点的取消记录可能不完整**：`ParallelBaseOp._execute_concurrent()` 将部分 `CancellationError` 与 `StreamExhausted` 一同转为停止标记；该节点若正常返回，不会被登记到 `cancelled_nodes`。不能据此保证所有取消场景都完整终止。
- **资源估算是静态近似**：已累加各 Op 与队列成本，但未按拓扑生命周期精确计算同时存活的峰值，也依赖输入探测和各 Op 的估算完整性。
- **输出收集会累积序列结果**：有界队列不约束收集器列表；空序列目前缺少结果键。其他长度与发射数量边界问题见 [长度广播协议](./length_protocol.md)。

## 核查记录

2026-10-14 对照上述引擎模块、`ops/base.py`、`component/queue.py` 以及 GUI/CLI 调用代码修订。实际运行最小例确认普通 `None` 与 sentinel 的区别、事件单独取消的两种结果，以及取消顶层 Task 的清理路径。

相关既有测试（executor、queue、preflight、runtime_plan）共 73 项通过。这些测试通过不覆盖或消除上述已复现的事件取消限制。
