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
- `config`: 配置队列字典 `{name: RichContextQueue}`
- `inputs`: 输入队列字典 `{name: RichContextQueue}`
- `outputs`: 输出队列列表字典 `{name: list[RichContextQueue]}`
- `length`: 序列长度（可选）
- `name`: 节点名称

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
- `CONCURRENCY`: 并发度（默认为4，当前未实现）

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

## 队列类型

### RichContextQueue
基础异步队列，基于`asyncio.Queue`实现。

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
            await queue.put(RichContextQueue._SENTINEL)
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
`RichContextQueue.get()`会自动检测并处理信号：
- 遇到`_SENTINEL`：抛出`StopIteration`
- 遇到`CancellationToken`：抛出`CancellationError`

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

## 设计问题与改进建议

### ✅ 已解决的问题

#### 1. 结束信号机制 ✅
**解决方案：** 实现了双信号机制（SENTINEL + CancellationToken）
- `RichContextQueue`在`get()`时自动检测信号
- `BaseOp.execute()`统一处理异常并传播信号
- `DAGExecutor`提供全局取消协调

#### 2. 序列长度传播机制 ✅
**解决方案：** 使用`asyncio.Event`实现阻塞式长度协调
- 生产者通过`queue.set_length()`设置长度并触发事件
- 消费者通过`queue.get_length()`等待长度就绪
- `BaseOp.pre_execute()`验证输入等长并向下游广播

#### 3. ParallelBaseOp并发执行 ✅
**解决方案：** 实现滑动窗口并发执行
- `CONCURRENCY=1`：串行执行（简化实现）
- `CONCURRENCY>1`：滑动窗口并发，保证输出有序
- `WINDOW_SIZE`可配置，默认为`CONCURRENCY * 2`

#### 4. 输入数据消费顺序 ✅
**解决方案：** 滑动窗口内顺序消费，窗口间顺序输出
- 每个窗口内按索引顺序调用`_async_convert_inputs()`
- 结果存储在预分配的列表中
- 窗口完成后按顺序广播结果

### 🟡 待优化问题

#### 5. 输出广播可能相互阻塞 ⚠️
**问题：** `_broadcast_result()`使用嵌套循环顺序`await`，如果某个队列满会阻塞其他队列。

**当前实现：**
```python
async def _broadcast_result(self, result: dict[str, Any]) -> None:
    for key, queue_list in self.outputs.items():
        for queue in queue_list:
            await queue.put(result[key])  # 顺序等待
```

**建议：** 使用`asyncio.gather`并发广播
```python
async def _broadcast_result(self, result: dict[str, Any]) -> None:
    tasks = []
    for key, queue_list in self.outputs.items():
        for queue in queue_list:
            tasks.append(queue.put(result[key]))
    await asyncio.gather(*tasks)
```

#### 6. 错误处理机制不统一
**问题：** 各算子的错误处理策略不一致
- `DataLoaderOp`中异常被捕获并continue
- 缺少统一的错误恢复策略

**建议：** 在BaseOp层面定义错误处理接口，支持可配置的错误策略（跳过/重试/中止）。

### 🟢 次要问题

#### 7. 类型标注不完整
- `_async_convert_inputs`返回`dict[str, Awaitable[Any]]`但未在类型中体现
- `_execute_single`和`_async_execute_single`的关系不清晰

#### 8. 队列容量管理
- `MAX_SIZE=1`作为默认值可能导致频繁阻塞
- 没有根据节点类型（生产者/消费者/转换器）动态调整队列大小的机制

#### 9. 资源清理
- `FileCacheQueue`有`clear()`方法，但BaseOp没有cleanup接口
- DAG执行失败时可能残留临时文件

### 架构优势

✅ 异步设计支持流式处理和并发执行
✅ 队列机制实现了解耦和背压控制
✅ 广播机制支持一对多的数据流
✅ `FileCacheQueue`支持大数据对象的内存优化
✅ 基类分层清晰（BaseOp/ParallelBaseOp）
