# Operator设计

Ref: [DAG节点定义](./dag_node_definition.md)

## 概述

DAG的节点的具体执行由Operator完成。Operator是异步执行的算子，通过队列机制实现节点间的数据流转和依赖管理。

## 数据流机制

### 队列初始化
节点初始化时，会按照节点定义中声明的`INPUTS`和`CONFIGS`创建对应的队列实例：
- `INPUTS`：输入数据队列（通常为序列数据）
- `CONFIGS`：配置参数队列（通常为标量配置）
- `OUTPUTS`：输出队列列表（支持广播到多个下游节点）

### 数据连接
数据通过将节点的`INPUTS`队列加入到其他节点的`OUTPUTS`列表中实现DAG的数据连接。
因此，一个`OUTPUT`可能对应多个节点的`INPUTS`（类似广播机制）。

### 异步执行
Operator的设计是异步的。图开始执行时，所有节点被启动，但被队列所阻塞。当前序节点开始按照进度处理数据时，后续节点可能会并发执行。

## 基类设计

节点的基类：[base.py](../hoshicore/ops/base.py)

### BaseOp

所有Operator的基类，提供基础的队列管理和执行流程。

**类属性：**
- `EXECUTOR`: 执行器类型（可选，如"cpu"）
- `INPUTS`: 输入定义字典，格式为 `{name: {type, description, ...}}`
- `CONFIGS`: 配置定义字典，格式为 `{name: {type, description, default, ...}}`
- `OUTPUTS`: 输出定义字典，格式为 `{name: {type, description}}`
- `MAX_SIZE`: 队列最大容量（默认为1）

**实例属性：**
- `config`: 配置队列字典 `{name: BaseQueue}`
- `inputs`: 输入队列字典 `{name: BaseQueue}`
- `outputs`: 输出队列列表字典 `{name: list[BaseQueue]}`
- `length`: 序列长度（可选，`None` 表示 sentinel 驱动）
- `name`: 节点名称
- `tracker`: 进度追踪器（`DummyTracker` 默认，由 wiring 注入 `ProgressTracker` 或 `ProxyTracker`）
- `_cancel_event`: 取消事件（`asyncio.Event` 或 `mp.Event`，由 wiring 注入）

**核心方法：**
```python
async def pre_execute() -> dict[str, Any]
    # 在执行前等待所有配置数据准备好
    # 返回配置字典

async def _async_execute(configs: dict[str, Any]) -> None
    # 子类必须实现的执行逻辑

async def execute() -> None
    # 执行入口：先pre_execute获取配置，再调用_async_execute
```

### ParallelBaseOp

并行执行的基类，用于处理序列数据中的每个元素可以独立并行处理的场景。

**类属性：**
- `CONCURRENCY`: 并发度（默认为1，串行执行；>1 时使用滑动窗口并发）
- `WINDOW_SIZE`: 滑动窗口大小（默认为 CONCURRENCY * 2）

**核心方法：**
```python
async def _async_execute(configs: dict[str, Any]) -> None
    # 已实现：循环处理length次，每次调用_async_execute_single
    # 将结果广播到所有outputs

async def _async_execute_single(
    data: Mapping[str, Awaitable[Any]],
    configs: dict[str, Any]
) -> dict[str, Any]
    # 子类必须实现：处理单个数据元素
    # data中的值是Awaitable，需要await获取实际值
    # 返回字典，键为OUTPUTS中定义的输出名称
```

**简化方法（可选）：**
```python
def _execute_single(data: dict[str, Any], configs: dict[str, Any]) -> dict[str, Any]
    # 同步版本，data已经是实际值
```

### FilterBaseOp

变长输出的过滤算子基类。输出序列长度不等于输入长度（如条件过滤），由 sentinel 信号驱动下游结束。

**类属性：**
- `VARIABLE_OUTPUT = True`：自动标记为变长输出
- `_infer_output_length()` 始终返回 `None`（sentinel 驱动）

**核心方法：**
```python
async def _async_execute(configs: dict[str, Any]) -> None
    # 子类实现：循环中选择性调用 _broadcast_outputs
    # 使用 self._input_range() 配合 except StreamExhausted: break
```

**使用模式：**
```python
class MyFilter(FilterBaseOp):
    INPUTS = {"data": {"type": "sequence", "required": True}}
    OUTPUTS = {"result": {"type": "sequence"}}

    async def _async_execute(self, configs):
        for i in self._input_range():
            data = self._async_convert_inputs()
            try:
                item = await data['data']
            except StreamExhausted:
                break
            if predicate(item):
                await self._broadcast_outputs({"result": item})
```

**静态冲突检测**：引擎在布线阶段检测变长源冲突：
- 多个不同 `VARIABLE_OUTPUT` 源的序列汇入同一节点 → 报错
- 固定长度 + 变长序列混合汇入 → 报错

## 队列类型

### BaseQueue

所有队列的抽象基类，定义统一接口。Op 代码仅依赖此接口，不直接依赖具体实现。

```python
class BaseQueue:
    _SENTINEL = object()         # 所有子类共享的正常结束信号
    active: bool
    async def put(self, item) -> None
    async def get(self) -> Any
    async def set_length(self, length: Optional[int]) -> None
    async def get_length(self) -> Optional[int]
```

### RichContextQueue
进程内异步队列，基于`asyncio.Queue`实现。继承 `BaseQueue`。

**特性：**
- 支持异步put/get操作
- 带锁保护的put操作
- 记录总数量`tot_num`

### FileCacheQueue
使用文件缓存的队列，继承自`RichContextQueue`。

**特性：**
- 支持pickle/json/numpy序列化
- 自动管理临时文件
- get时自动删除缓存文件
- 适用于大数据对象的传递

### IPCQueue
跨进程异步队列，继承 `BaseQueue`。用于多进程模式下不同进程的 Op 之间通信。

**传输策略（分层）：**

| 数据类型 | 传输方式 |
|---------|---------|
| `np.ndarray`（大于阈值） | `multiprocessing.shared_memory` 零拷贝 |
| `ShmTransportable` 对象（如 `FloatImage`） | 主数组走 shm + 元数据走 pickle |
| 其他对象（float、FGP 等） | `pickle` via `Pipe` |
| 控制帧（sentinel / cancel） | 标记帧 via `Pipe` |

**背压机制：** 双信号量（`mp.Semaphore`），与 `RichContextQueue(maxsize=N)` 语义一致。

**SharedMemory 生命周期：** producer 创建 → 发送描述符 → consumer 读取 + unlink → `cleanup()` 安全网兜底。

**ShmTransportable 基类：** 轻量包装类继承此 ABC 即可获得 SharedMemory 传输优化：
```python
class ShmTransportable(ABC):
    def shm_nbytes(self) -> int: ...
    def shm_pack_into(self, buf) -> bytes: ...
    @classmethod
    def shm_unpack_from(cls, buf, meta: bytes) -> Self: ...
```

## 实现示例

### 示例1：DataLoaderOp（并行算子）

```python
class DataLoaderOp(ParallelBaseOp):
    INPUTS = {"src": {"type": "sequence", "description": "数据源"}}
    CONFIGS = {
        "loader_type": {"type": "str", "description": "数据加载器类型"},
        "configs": {"type": "dict", "description": "加载器配置"}
    }
    OUTPUTS = {"result": {"type": "sequence", "description": "数据序列"}}

    async def _async_execute_single(self, data, configs):
        # data['src']是Awaitable，需要await
        src = await data['src']
        # 处理单个数据
        result = self.loader.load(src)
        return {"result": result}
```

### 示例2：TrailStackerOp（聚合算子）

```python
class TrailStackerOp(BaseOp):
    INPUTS = {
        "data": {"type": "sequence", "required": True},
        "weight": {"type": "sequence", "required": True}
    }
    CONFIGS = {"int_weight": {"type": "bool", "default": False}}
    OUTPUTS = {"result": {"type": "image"}}

    async def _async_execute(self, configs: dict[str, Any]) -> None:
        img_queue = self.inputs['data']
        weight_queue = self.inputs['weight']

        # 循环处理所有输入
        for i in range(self.length):
            img = await img_queue.get()
            weight = await weight_queue.get()
            # 聚合处理
            merger.merge(img, weight)

        # 输出最终结果
        result = merger.get_result()
        for output_queue in self.outputs['result']:
            await output_queue.put(result)
```

## 信号传播机制

### 正常结束信号（SENTINEL）
当节点完成所有数据处理后，会向所有输出队列发送`_SENTINEL`信号：
```python
async def _send_sentinel(self) -> None:
    """发送正常结束信号"""
    for queue_list in self.outputs.values():
        for queue in queue_list:
            await queue.put(BaseQueue._SENTINEL)
```

### 取消令牌（CancellationToken）
当节点执行失败时，会创建`CancellationToken`并传播到下游：
```python
class CancellationToken:
    def __init__(self, error: Exception, source_node: str):
        self.error = error
        self.source_node = source_node
```

### 队列自动信号处理
`BaseQueue.get()`（RichContextQueue 和 IPCQueue 均实现）会自动检测并处理信号：
- 遇到`_SENTINEL`：回填后抛出`StreamExhausted`（替代 PEP 479 禁止的 `StopIteration`）
- 遇到`CancellationToken`：回填后抛出`CancellationError`

**回填语义**：信号消费后会无条件回填到队列，确保同一队列的多个并发消费者都能收到终止信号。

### 异常传播流程
```python
async def execute(self) -> None:
    try:
        configs = await self.pre_execute()
        await self._async_execute(configs)
        await self._send_sentinel()  # 正常结束
    except CancellationError:
        await self._propagate_cancellation_from_upstream()  # 上游取消
        raise
    except Exception as e:
        await self._propagate_cancellation(e)  # 本节点异常
        raise
```

## DAG执行器

`DAGExecutor`负责启动和管理所有节点的执行，提供全局取消机制：

```python
class DAGExecutor:
    async def execute(self) -> None:
        tasks = [self._run_node(node) for node in self.nodes]
        try:
            await asyncio.gather(*tasks, return_exceptions=False)
        except Exception as e:
            self.cancel_event.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
```

## 多进程支持

### 进程分组

`multiprocess.compute_process_groups()` 使用贪心拓扑序算法将 Op 分配到进程组：

- **组 0**（主进程）：`EXECUTOR = None` 的 Op（I/O 型），以及 feeder 协程
- **组 1..N**（worker 进程）：`EXECUTOR = "cpu"` 的 Op，通过反亲和约束分散

分组评分函数：`score(p) = α × (上游在 p 中的数量) - β × (p 中同 EXECUTOR 节点数)`

### 跨进程进度追踪

Worker 进程中的 Op 使用 `ProxyTracker`（而非 `ProgressTracker`），将 `(method_name, *args)` 元组通过 `mp.Queue` 发送到主进程：

```
Worker 进程:  op.tracker.update(name)  →  ProxyTracker._q.put(("update", name, 1))
                                                    │
                                                    ▼  mp.Queue
Main 进程:  TrackerEventConsumer.run()  →  tracker.update(name, 1)  →  tqdm 更新
```

### 跨进程取消

- `cancel_event` 使用 `multiprocessing.Event`（跨平台），替代 `asyncio.Event`
- `CancellationToken` 通过 IPCQueue 的控制帧传播（`("cancel", error_str, source_node)` 标记帧）
- `_run_cpu` 的取消检查 `mp.Event.is_set()` 是线程安全的，直接兼容

## Meta YAML 路由预处理

`hoshicore/engine/meta.py` 提供 `meta_resolve()` 函数，将含路由字典的 Meta YAML 编译为标准 DAG spec dict。

**路由节点语法**：在节点中使用 `route` 替代 `op`，声明可选的实现方式：

```yaml
main_stacker:
  route:                          # 可选实现列表
    mean: MeanStackerOp
    median: median_stack_core.yaml
    sigma_clip: sigma_clip.yaml
  inputs:                         # 所有选项共享的输入
    data: flat_divide.result
  route_configs:                  # 按选项分组的专属配置
    mean:
      int_weight: configs.int_weight
    sigma_clip:
      int_weight: configs.int_weight
      rej_high: configs.rej_high
  outputs:
    result: { type: image }
```

**解析流程**：`meta_resolve(meta_spec, route_choices)` 遍历节点，对含 `route` 字段的节点：
1. 查找 `route_choices[node_name]` 获取用户选择
2. 将 `route[choice]` 填入 `op` 字段
3. 合并共享 `inputs` + `route_inputs[choice]`
4. 合并共享 `configs` + `route_configs[choice]`
5. 删除 `route` / `route_inputs` / `route_configs` 字段

**辅助算子**：`NoneOutputOp` 输出 `None`，配合 `CalibrationSubtractOp` / `CalibrationDivideOp` 的 passthrough 行为实现可选校准阶段。

## 设计问题与改进建议

#### 9. SubDagOp 多进程行为
**问题：** SubDagOp 内部创建子 DAG + event loop。如果 SubDag 跨越进程边界需要递归处理。

**当前约束：** 初版约束 SubDagOp 整体在单一进程内。

### 架构优势

✅ 异步设计支持流式处理和并发执行
✅ 队列机制实现了解耦和背压控制（`BaseQueue` 统一接口）
✅ 广播机制支持一对多的数据流
✅ `FileCacheQueue`支持大数据对象的内存优化
✅ 基类分层清晰（BaseOp / ParallelBaseOp / FilterBaseOp）
✅ 多进程对 Op 完全透明
✅ SharedMemory 零拷贝避免跨进程序列化开销
