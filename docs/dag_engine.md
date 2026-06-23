# DAG Engine 架构概览

本文档描述 DAG 引擎的核心架构：编译链路、运行时执行模型、队列通信协议和终止机制。

## 编译链路

```
Raw YAML ─► meta_resolve() ─► flatten_sub_dags() ─► 

validate_and_build_order() ─► ValidatedDag ─► instantiate_and_wire() ─► 

execute(ops, feeders, output_queues)
```

| 阶段 | 模块 | 输入 → 输出 |
|------|------|-------------|
| 路由编译 | `meta.py` | Meta YAML + route_choices → 标准 spec（去除路由/开关语法） |
| 子图展开 | `flatten.py` | spec 中 `.yaml` 引用 → 命名空间化的 flat nodes |
| 校验+拓扑 | `build.py` | flat spec → `ValidatedDag`（含 `exec_order`、依赖图、全局声明） |
| 实例化+布线 | `wiring.py` | `ValidatedDag` + 数据 → Op 实例列表 + feeder 协程 + 输出队列 |

### 关键编译约定

- **命名空间**：子图节点展开后以 `parent.child` 点分隔命名，`rsplit(".", 1)` 解析 link
- **`__inactive__`**：标记子图可选输入未被父图布线，布线阶段跳过并设 `queue.active = False`
- **Config 解析优先级**（高→低）：用户显式输入 > `default_settings.yaml` > YAML `default` > Op `CONFIGS.default`
- **Auto-inject**：YAML 未布线但 Op 声明了 default 的 config 自动注入 feeder，避免 `pre_execute()` 挂起

## 运行时执行模型

### Runtime Group 结构

`run_dag()` 启动三组并发协程加一个独立监控 task：

```
run_dag()
  ├─ watcher_task (独立，不加入 gather)
  │     └─ await cancel_event → cancel_all + task.cancel
  │
  └─ gather:
       ├─ _run_feeders()      ← 将全局输入/配置推入队列
       ├─ executor.execute()  ← 管理所有节点 task
       └─ _collect_outputs()  ← 从输出队列收集结果
```

### DAGExecutor 职责

- 为每个 Op 按拓扑序创建 `asyncio.Task`
- `gather(return_exceptions=True)` 等待全部完成
- 节点异常分类：真实失败 → `failed_nodes`；取消 → `cancelled_nodes`
- 首个真实失败触发 `cancel_all()` + `task.cancel()` 所有其他节点
- 按拓扑序从 `failed_nodes` 选择根因，抛出 `DAGExecutionError`

### Op 执行生命周期

```python
async def execute(self) -> None:
    configs = await self.pre_execute()   # 等待所有 config 队列 + get_length
    await self._async_execute(configs)   # 算法主体
    await self._send_sentinel()          # 向下游发送结束标记
```

Op 只负责 raise，不处理取消传播。

## 队列通信协议

### 数据流

节点间通过 `RichContextQueue`（asyncio.Queue 封装，maxsize=1）通信：

- **流式单帧**：maxsize=1 保证有界内存，生产者-消费者逐帧流转
- **Length 先行**：`set_length()` / `get_length()` 在数据流之前传播序列长度，下游可预分配
- **Sentinel**：`None` 标记序列结束，下游据此退出循环
- **FileCacheQueue**：磁盘缓存变体，帧数据写入临时文件，队列传递路径字符串

### 取消语义：`force_cancel(token)`

由 executor 调用，标记队列为已取消并唤醒所有等待者：

| 等待位置 | 唤醒机制 |
|----------|----------|
| 尚未进入 `get()`/`put()`/`get_length()` | `_check_cancelled()` 入口守卫 |
| `await queue.get()` 内部 | `put_nowait(token)` 注入 |
| `await _length_event.wait()` | `_length_event.set()` |
| `await queue.put(item)` 内部 | executor `task.cancel()` 补充 |

`force_cancel()` 幂等——重复调用无效。

## 终止协议

三条终止路径：

| 触发源 | 入口 | 最终异常 |
|--------|------|----------|
| 节点内部异常 | `BaseOp.execute()` raise | `DAGExecutionError`（携带根因） |
| 外部取消（GUI/Ctrl-C） | `cancel_event.set()` | `asyncio.CancelledError` |
| 编译期错误 | `run_from_yaml()` 编译阶段 | `ValueError` / `DagSpecError` 等 |

### 节点失败时序

```
node_A raises ValueError
  → _run_node: failed_nodes.append + cancel_all() + task.cancel() others
  → 其他节点:
      卡在 get() → token 注入 → CancellationError → cancelled_nodes
      卡在 put() → CancelledError (task.cancel) → cancelled_nodes
      未启动    → gather 结果 CancelledError → cancelled_nodes
  → feeders/collector: CancellationError → 静默退出
  → executor.execute() raises DAGExecutionError
```

### 外部取消时序

```
cancel_event.set()
  → watcher_task 唤醒 → cancel_all() + task.cancel() 所有节点
  → 所有节点/feeders/collector 退出
  → gather 完成 → run_dag except 分支 raise CancelledError
```

### `DAGExecutionError` 结构

```python
class DAGExecutionError(Exception):
    root_cause: Exception               # 根因异常
    root_node: str                      # 根因节点名（拓扑序最上游）
    failed_nodes: list[(str, Exception)] # 所有真实失败
    cancelled_nodes: list[str]          # 因取消而终止的节点
```

GUI 弹窗和 CLI 日志均基于此结构展示根因和详情。

## 资源预检

`run_from_yaml()` 在执行前调用 `preflight_check()`：

- 各 Op 通过 `estimate_resources()` 声明峰值内存/磁盘需求
- 预检聚合报告：warnings + proposed_fallbacks（如切换 `buffer_mode`）
- 回调模式：GUI 弹窗确认 / CLI auto_fallback / 用户 abort

## 已知限制

- `ParallelBaseOp._execute_concurrent()` 中 `CancellationError` 被当作 `StreamExhausted`，导致被取消的并发 Op 不出现在 `cancelled_nodes` 中。不影响根因展示和终止完整性。
- 资源预检为单 Op 估算，尚未考虑 DAG 拓扑并发叠加的峰值。
