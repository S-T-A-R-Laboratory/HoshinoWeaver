# hoshicore 技术文档

hoshicore 是 HoshinoWeaver (HNW) 的核心计算库，提供星空序列图像的通用预处理框架。用户通过 **YAML 声明式 DAG** 定义处理流程，框架自动完成依赖解析、异步调度和流式执行。

**核心能力**：

- 最大值 / 最小值 / 均值叠加（Max / Min / Mean Stacking）
- 迭代 Sigma Clipping 均值叠加
- 渐入渐出权重（Fade-in / Fade-out）
- 最大值叠加噪声均匀化（Max Noise Equalization）
- EXIF 读取与合并（曝光时间累加等）
- 多格式图像 I/O（TIFF 16-bit、JPEG、PNG 等）

---

## 1. 架构总览

```
┌───────────────────────────────────────────────────────────────────┐
│  用户层                                                           │
│    YAML DAG 定义（dag/*.yaml）+ Python API（run_from_yaml）        │
├───────────────────────────────────────────────────────────────────┤
│  引擎层  engine/                                                  │
│    build.py   ── YAML 解析、合法性校验、拓扑排序                     │
│    wiring.py  ── Op 实例化、队列布线、feeder 协程生成                │
│    executor.py── DAGExecutor 异步并发执行 + 全局取消                │
├───────────────────────────────────────────────────────────────────┤
│  算子层  ops/                                                     │
│    BaseOp / ParallelBaseOp 基类 + 各具体算子                       │
├───────────────────────────────────────────────────────────────────┤
│  组件层  component/                                               │
│    queue（异步背压队列）、merger（合并器）、dataloader（数据加载器）    │
│    imgfio（图像 I/O）、tagged_image（dtype 管理）、utils 等          │
└───────────────────────────────────────────────────────────────────┘
```

**执行流程**：

```
YAML 文件
   │
   ▼
build.validate_and_build_order()   →  ValidatedDag（校验通过的 DAG 结构 + 拓扑序）
   │
   ▼
wiring.instantiate_and_wire()      →  Op 实例列表 + feeder 协程 + output 收集队列
   │
   ▼
executor.DAGExecutor.execute()     →  所有节点并发启动，由队列背压自然调度
   │
   ▼
收集 output 队列结果                →  dict[str, Any]
```

---

## 2. 核心概念

### 2.1 DAG 计算图

处理流程通过 YAML 文件声明。一个 DAG 由四个顶层字段定义：

| 字段 | 含义 |
|------|------|
| `inputs` | 全局序列输入（如文件名列表），type 必须为 `sequence` |
| `configs` | 全局标量配置（如渐入比例、输出路径），支持 `default` |
| `nodes` | 节点定义，每个节点指定 `op`、`inputs`、`configs`、`outputs` |
| `outputs` | DAG 的目标输出，引用某个节点的输出端口 |

**Link 语法**：节点通过 link 字符串引用数据来源：

- `inputs.<name>` — 全局序列输入
- `configs.<name>` — 全局标量配置
- `<node_name>.<output_name>` — 其他节点的输出端口

**最小示例**（仅均值叠加）：

```yaml
description: "均值叠加"
version: "1"

inputs:
  fnames: { type: sequence, required: true }

configs:
  loader_type: { type: str, default: "img_file_list" }
  loader_configs: { type: dict, default: {} }
  int_weight: { type: bool, default: true }
  output_filename: { type: str, default: "result.tif" }
  output_dtype: { type: str, default: "uint8" }

nodes:
  data_loader:
    op: ImgDataLoaderOp
    inputs:
      src: inputs.fnames             # 引用全局输入
    configs:
      loader_type: configs.loader_type
      configs: configs.loader_configs
    outputs:
      result: { type: sequence }

  meanstacker:
    op: MeanStackerOp
    inputs:
      data: data_loader.result       # 引用上游节点输出
    configs:
      int_weight: configs.int_weight
    outputs:
      result: { type: image }

  image_saver:
    op: ImageSaveOp
    configs:
      image: meanstacker.result
      output_filename: configs.output_filename
      output_dtype: configs.output_dtype
    outputs:
      return_code: { type: int }

outputs:
  result: image_saver.return_code    # DAG 的最终输出
```

### 2.2 Op 算子体系

所有算子继承自两个基类：

| 基类 | 适用场景 | 核心方法 |
|------|---------|---------|
| `BaseOp` | 需要消费整个序列的聚合操作（如叠加） | `_async_execute(configs)` |
| `ParallelBaseOp` | 逐帧独立处理的并行操作（如 EXIF 读取） | `_async_execute_single(data, configs)` |

**Op 声明规范**：每个 Op 通过类属性声明输入输出：

```python
class MyOp(BaseOp):
    EXECUTOR = "cpu"                                    # 执行器标识（可选）
    INPUTS  = {"data": {"type": "sequence", "required": True}}
    CONFIGS = {"threshold": {"type": "float", "default": 0.5}}
    OUTPUTS = {"result": {"type": "image"}}
```

- `INPUTS`：序列数据输入，由队列流式传递
- `CONFIGS`：标量配置，在 `pre_execute()` 阶段一次性读取
- `OUTPUTS`：输出端口，结果通过广播推送到所有下游

**内置算子一览**：

| Op | 类型 | 功能 |
|----|------|------|
| `ImgDataLoaderOp` | BaseOp | 异步加载图像序列，支持文件列表和内存数组两种模式 |
| `WeightGeneratorOp` | BaseOp | 根据 `fin` / `fout` 参数生成渐入渐出权重序列 |
| `TrailStackerOp` | BaseOp | 最大值叠加（星轨），支持可选权重 |
| `MinStackerOp` | BaseOp | 最小值叠加 |
| `MeanStackerOp` | BaseOp | 均值叠加 |
| `SigmaClippingStackerOp` | BaseOp | 迭代 Sigma Clipping 均值叠加，输出均值图像和统计信息 |
| `MaxNoiseEqualizationOp` | BaseOp | 最大值叠加噪声均匀化，消除校正后的空间不均匀伪影 |
| `ExifReadOp` | ParallelBaseOp | 并行读取 EXIF 信息（CONCURRENCY=4） |
| `ExifReduceOp` | BaseOp | EXIF 信息聚合（如曝光时间累加） |
| `ImageSaveOp` | BaseOp | 图像保存，支持 dtype 转换和 EXIF 写入 |

### 2.3 数据流与队列机制

节点间通过 `RichContextQueue`（异步背压队列）连接，形成流式数据管道：

```
Feeder（全局数据注入）
    │
    ▼  put()
┌─────────────────┐    get()    ┌─────────────────┐
│ RichContextQueue │ ────────► │    Op 节点       │
└─────────────────┘            │                  │
                               │  处理后 put()     │
                               └───────┬──────────┘
                                       │ broadcast
                          ┌────────────┼────────────┐
                          ▼            ▼            ▼
                      Queue A      Queue B      Queue C
                    （下游节点）  （下游节点）  （输出收集）
```

**关键机制**：

- **背压控制**：队列 `maxsize=1`（默认），上游生产速度自动适配下游消费速度
- **长度协调**：生产者通过 `set_length()` 广播序列长度，消费者通过 `get_length()` 等待就绪
- **信号传播**：正常结束发送 `SENTINEL`，异常发送 `CancellationToken`，下游自动感知
- **文件缓存**：`FileCacheQueue` 支持将大对象自动序列化到磁盘，降低内存压力

---

## 3. 使用方式

### 3.1 快速上手

最简调用方式——通过 `run_from_yaml` 一行启动：

```python
from hoshicore.engine.wiring import run_from_yaml
import asyncio

results = asyncio.run(run_from_yaml(
    "hoshicore/dag/fifo_startrail.yaml",
    global_inputs={"fnames": ["img001.jpg", "img002.jpg", "img003.jpg"]},
    global_configs={
        "fin": 0.2,          # 渐入比例
        "fout": 0.2,         # 渐出比例
        "int_weight": True,  # 整型权重优化
        "output_filename": "result.tif",
    },
))
```

也可以分步调用以获得更多控制：

```python
from hoshicore.engine.build import _load_yaml, validate_and_build_order
from hoshicore.engine.wiring import run_dag
import asyncio

# 1. 加载并校验 DAG
spec = _load_yaml("hoshicore/dag/fifo_startrail.yaml")
dag = validate_and_build_order(spec)

# 2. 执行
results = asyncio.run(run_dag(
    dag,
    global_inputs={"fnames": file_list},
    global_configs={"fin": 0.1, "fout": 0.1},
    progress=True,  # 显示 tqdm 进度条
))
```

### 3.2 编写自定义 YAML DAG

编写步骤：

1. **定义 `inputs`**：声明序列输入（通常是文件名列表）
2. **定义 `configs`**：声明全局配置项及其默认值
3. **定义 `nodes`**：每个节点指定使用的 `op`，并通过 link 语法连接上游数据
4. **定义 `outputs`**：指定最终需要收集的节点输出

**布线规则**：

- YAML 中未布线但 Op 声明了 `default` 的 config 会被引擎自动注入
- 可选输入（`required: False`）未布线时会被标记为非活跃，Op 内部可通过 `self.inputs[key].active` 判断
- 引擎会自动校验 link 合法性（引用的节点/端口/全局字段是否存在）

**Op 注册表**：引擎内置的 `DEFAULT_OP_REGISTRY` 包含所有标准算子。在 YAML 的 `op` 字段中使用注册表中的键名即可引用对应的算子类。

### 3.3 编写自定义 Op

**并行算子示例**（逐帧独立处理）：

```python
from hoshicore.ops.base import ParallelBaseOp

class MyFilterOp(ParallelBaseOp):
    INPUTS = {"data": {"type": "sequence", "required": True}}
    CONFIGS = {"threshold": {"type": "float", "default": 0.5}}
    OUTPUTS = {"result": {"type": "sequence"}}
    CONCURRENCY = 4  # 并发度

    async def _async_execute_single(self, data, configs):
        img = await data['data']
        threshold = configs['threshold']
        result = img * (img > threshold)
        return {"result": result}
```

**聚合算子示例**（消费整个序列）：

```python
from hoshicore.ops.base import BaseOp

class MySumOp(BaseOp):
    INPUTS = {"data": {"type": "sequence", "required": True}}
    OUTPUTS = {"result": {"type": "image"}}

    async def _async_execute(self, configs):
        total = None
        for i in range(self.length):
            data = self._async_convert_inputs()
            img = await data['data']
            total = img if total is None else total + img
        for queue in self.outputs['result']:
            await queue.put(total)
```

**注册并使用**：

```python
from hoshicore.engine.wiring import DEFAULT_OP_REGISTRY, run_from_yaml
import asyncio

my_registry = {**DEFAULT_OP_REGISTRY, "MyFilterOp": MyFilterOp}

results = asyncio.run(run_from_yaml(
    "my_dag.yaml",
    global_inputs={...},
    global_configs={...},
    op_registry=my_registry,
))
```

### 3.4 子图嵌套（SubDagOp）

通过 `create_sub_dag_op()` 可以将一个完整的 YAML DAG 封装为单个 Op 节点，嵌入到父 DAG 中：

```python
from hoshicore.ops.sub_dag import create_sub_dag_op
from hoshicore.engine.wiring import DEFAULT_OP_REGISTRY

# 从 YAML 创建子图 Op 类
SigmaClipTrailOp = create_sub_dag_op(
    "hoshicore/dag/fix_startrail.yaml",
    op_name="SigmaClipTrailOp",
)

# 注册后即可在父 DAG 的 YAML 中使用
my_registry = {**DEFAULT_OP_REGISTRY, "SigmaClipTrailOp": SigmaClipTrailOp}
```

子图的 `INPUTS` / `CONFIGS` / `OUTPUTS` 自动从 YAML 推导。数据通过队列桥接（`_bridge_queue`）流式传输，无需缓存整个序列。支持多级嵌套。

---

## 4. 内置 DAG 工作流

### `fifo_startrail.yaml` — 渐入渐出星轨叠加

```
inputs.fnames ──┬──► ImgDataLoaderOp ──────────┐
                │                               ├──► TrailStackerOp ───┐
                ├──► WeightGeneratorOp ─────────┘                      │
                │                                                      ├─► ImageSaveOp
                └──► ExifReadOp ──────► ExifReduceOp ──────────────────┘
```

支持渐入渐出权重的最大值叠加，同时合并 EXIF 曝光时间。

**关键配置**：`fin`（渐入比例）、`fout`（渐出比例）、`int_weight`（整型权重优化）

### `fix_startrail.yaml` — 噪声均匀化星轨叠加

```
inputs.fnames ──┬──► ImgDataLoaderOp ──┬──► TrailStackerOp ─────────┐
                │                      │                            │
                │                      └──► SigmaClippingStackerOp ─┤
                │                             │ statistics           │
                │                             ▼                     │
                │                      MaxNoiseEqualizationOp ──────┤
                │                                                   ├─► ImageSaveOp
                └──► ExifReadOp ──────► ExifReduceOp ───────────────┘
```

在最大值叠加基础上，利用 Sigma Clipping 的统计信息（逐像素均值和方差）进行噪声均匀化校正，消除镜头校正引入的空间不均匀伪影。详见 [噪声均匀化方案](./noise-equalization.md)。

### `mean_only.yaml` — 纯均值叠加

```
inputs.fnames ──────► ImgDataLoaderOp ──────► MeanStackerOp ──► ImageSaveOp
```

最简工作流，仅做均值叠加和保存。

---

## 5. 项目目录结构

```
hoshicore/
├── engine/                 # 引擎层：DAG 构建、布线、执行
│   ├── build.py            #   YAML 解析 + 合法性校验 + 拓扑排序 → ValidatedDag
│   ├── wiring.py           #   Op 实例化 + 队列连接 + feeder 生成 + run_dag / run_from_yaml
│   └── executor.py         #   DAGExecutor：异步并发调度 + 全局取消机制
│
├── ops/                    # 算子层：所有 Op 定义
│   ├── base.py             #   BaseOp / ParallelBaseOp 基类
│   ├── dataloader.py       #   ImgDataLoaderOp（异步图像加载）
│   ├── weight_generator.py #   WeightGeneratorOp（渐入渐出权重）
│   ├── trailstacker.py     #   TrailStackerOp / MinStackerOp / MeanStackerOp
│   │                       #   SigmaClippingStackerOp / MaxNoiseEqualizationOp
│   ├── exif_op.py          #   ExifReadOp / ExifReduceOp
│   ├── image_saver.py      #   ImageSaveOp
│   └── sub_dag.py          #   SubDagOp + create_sub_dag_op()（子图嵌套）
│
├── component/              # 组件层：底层基础设施
│   ├── queue.py            #   RichContextQueue / FileCacheQueue / CancellationToken
│   ├── merger.py           #   MaxMerger / MinMerger / MeanMerger / SigmaClippingMerger
│   ├── dataloader.py       #   BaseLoader / ImgFileListLoader / ArrayLoader
│   ├── imgfio.py           #   图像读写（save_img 等）
│   ├── tagged_image.py     #   FloatImage / dtype 管理 / 缩放工具
│   ├── frame_buffer.py     #   DiskFrameBuffer（磁盘帧缓冲，用于 Sigma Clipping 多 pass）
│   ├── noise_equalization.py # equalize_noise()（噪声均匀化核心算法）
│   ├── exifdata.py         #   ExifData / 读写工具
│   ├── progress.py         #   ProgressTracker / DummyTracker（进度条管理）
│   └── utils.py            #   FastGaussianParam / dtype 工具 / 通用辅助函数
│
├── dag/                    # 预置 DAG 定义
│   ├── fifo_startrail.yaml #   渐入渐出星轨叠加
│   ├── fix_startrail.yaml  #   噪声均匀化星轨叠加
│   └── mean_only.yaml      #   纯均值叠加
│
├── ezlib/                  # 遗留工具库
│   └── arg.py              #   StackConfigArg / ImgInfo 等数据类
│
└── starlib/                # 星点工具库
    ├── stardetect.py       #   星点检测
    └── starshrink.py       #   星点缩小
```

---

## 6. 相关文档

| 文档 | 内容 |
|------|------|
| [DAG 节点定义规范](./dag_node_definition.md) | YAML DAG 的完整 schema 说明，包含字段定义、link 语法和校验规则 |
| [Op 算子设计](./op.md) | Op 基类设计、队列机制、信号传播、异常处理的详细说明 |
| [噪声均匀化方案](./noise-equalization.md) | 最大值叠加噪声均匀化的数学推导和算法流程 |
| [开发者日志](./dev-log.md) | 成像模型推导、噪声分析等技术背景 |
