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

## 设计问题与改进建议

### 🔴 严重问题

#### 1. 结束信号机制缺失
**问题：** 当前框架缺少明确的流结束信号传递机制。
- `DataLoaderOp`中有`_SENTINEL`定义但未被正确处理（见L75注释）
- 下游节点无法判断上游是否已完成数据发送
- 可能导致下游节点永久阻塞在`queue.get()`

**影响：** 框架无法正确处理流式数据的终止，DAG执行可能挂起。

**建议：**
```python
# 在BaseOp中统一定义SENTINEL
class BaseOp:
    _SENTINEL = object()

    async def _async_execute(self, configs):
        # 执行完成后发送结束信号
        for output_name, queue_list in self.outputs.items():
            for queue in queue_list:
                await queue.put(self._SENTINEL)
```

#### 2. 序列长度传播机制不完善
**问题：** `length`属性的设置和传播机制不清晰。
- `ParallelBaseOp._async_execute`依赖`self.length`，但未说明如何设置
- `DataLoaderOp`中`_length`来自loader，但未调用`set_length()`
- 下游节点如何获知序列长度？

**影响：** 节点可能因length未设置而抛出异常，或处理错误的数据量。

**建议：** 在DAG构建时通过拓扑分析自动推导和设置length。

#### 3. ParallelBaseOp的并发未实现
**问题：** `CONCURRENCY=4`但`_async_execute`是串行循环（L72）。

**影响：** 性能未达到设计预期，并行算子实际串行执行。

**建议：**
```python
async def _async_execute(self, configs: dict[str, Any]) -> None:
    semaphore = asyncio.Semaphore(self.CONCURRENCY)

    async def process_item(i):
        async with semaphore:
            data = await self._async_convert_inputs()
            result = await self._async_execute_single(data, configs)
            for key, queue_list in self.outputs.items():
                for queue in queue_list:
                    await queue.put(result[key])

    await asyncio.gather(*[process_item(i) for i in range(self.length)])
```

### 🟡 设计缺陷

#### 4. 输入数据消费顺序问题
**问题：** `ParallelBaseOp._async_convert_inputs()`每次调用都从队列get，但并发执行时可能乱序。

**影响：** 如果实现真正的并发，数据处理顺序无法保证。

#### 5. 错误处理机制不统一
**问题：**
- `DataLoaderOp`中异常被捕获并continue（L64-68）
- `TrailStackerOp`中有`err_msg_collector`和`on_error_action`（已注释）
- `BaseOp`没有统一的错误处理策略

**建议：** 在BaseOp层面定义统一的错误处理接口。

#### 6. 输出广播的顺序性
**问题：** `ParallelBaseOp`中输出广播是嵌套循环（L76-78），如果某个队列阻塞会影响其他队列。

**建议：** 使用`asyncio.gather`并发广播到所有输出队列。

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
