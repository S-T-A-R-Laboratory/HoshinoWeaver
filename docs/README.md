# hoshicore 技术文档

hoshicore 是 HoshinoWeaver (HNW) 的核心计算库，是星空序列图像的通用预处理框架。用户通过 **YAML 声明式 DAG** 定义处理流程，框架自动完成依赖解析、异步调度和流式执行。

**核心能力**：

- 最大值 / 最小值 / 均值 / 中位数叠加（Max / Min / Mean / Median Stacking）
- 迭代 Sigma Clipping 均值叠加
- 校准流水线（Bias / Dark / Flat 减法与除法校正）
- 渐入渐出权重（Fade-in / Fade-out）
- 最大值叠加噪声均匀化（Max Noise Equalization）
- 图像预处理（缩放、裁切、排序、星点对齐）
- 批量序列保存（延时视频帧输出）
- EXIF 读取与合并（曝光时间累加等）
- 多格式图像 I/O（TIFF 16-bit、JPEG、PNG 等）
- Meta YAML 声明式路由（单一 YAML 描述多种可选处理路径）

**架构改造参考**：

- [框架层改造建议](./framework_refactor_notes.md)：记录执行引擎、配置解析、YAML 编译、Op schema、GUI/后端解耦等框架层风险与建议路线图。

---

## 1. 架构总览

```
┌───────────────────────────────────────────────────────────────────┐
│  用户层                                                            │
│    YAML DAG 定义（dag/*.yaml）+ Python API（run_from_yaml）         │
├───────────────────────────────────────────────────────────────────┤
│  引擎层  engine/                                                   │
│    build.py        ── YAML 解析、合法性校验、拓扑排序                 │
│    wiring.py       ── Op 实例化、队列布线、feeder 协程生成            │
│    executor.py     ── DAGExecutor 异步并发执行 + 全局取消            │
│    multiprocess.py ── 多进程执行引擎（进程分组、IPCQueue 注入、worker）│
│    meta.py         ── Meta YAML 预处理（路由解析 → 标准 spec dict）  │
├───────────────────────────────────────────────────────────────────┤
│  算子层  ops/                                                     │
│    BaseOp / ParallelBaseOp / FilterBaseOp 基类 + 各具体算子         │
├───────────────────────────────────────────────────────────────────┤
│  组件层  component/                                               │
│    queue（异步背压队列）、ipc_queue（跨进程队列）                     │
│    merger（合并器）、dataloader（数据加载器）                         │
│    image_io（图像 I/O）、data_container（dtype 管理 + 数据容器）、utils │
└───────────────────────────────────────────────────────────────────┘
```

**执行流程**：

```
标准 YAML                        Meta YAML（含 route 字典）
   │                                  │
   │                          meta.meta_resolve(spec, route_choices)
   │                                  │
   │                                  ▼
   │                            标准 spec dict
   │                                  │
   ▼──────────────────────────────────┘
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

所有算子继承自三个基类：

| 基类 | 适用场景 | 核心方法 |
|------|---------|---------|
| `BaseOp` | 需要消费整个序列的聚合操作（如叠加） | `_async_execute(configs)` |
| `ParallelBaseOp` | 逐帧独立处理的并行操作（如 EXIF 读取） | `_async_execute_single(data, configs)` |
| `FilterBaseOp` | 输出长度不等于输入长度的过滤操作 | `_async_execute(configs)` |

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
- `EXECUTOR`：执行器标识，`"cpu"` 表示 CPU 密集型（多进程模式下会被分配到 worker 进程），`None`（默认）表示 I/O 型或轻量型

**内置算子一览**：

| Op | 类型 | 功能 |
|----|------|------|
| **I/O** | | |
| `ImgDataLoaderOp` | BaseOp | 异步加载图像序列，支持文件列表和内存数组两种模式 |
| `LoadSingleImageOp` | BaseOp | 加载单张图像（如 master 校准帧） |
| `LoadMaskImageOp` | BaseOp | 加载并归一化遮罩图像 |
| `ImageSaveOp` | BaseOp | 图像保存，支持 dtype 转换和 EXIF 写入 |
| `BatchImageSaveOp` | ParallelBaseOp | 批量序列保存（逐帧带序号保存，用于延时视频帧输出） |
| **叠加核心** | | |
| `TrailStackerOp` | BaseOp | 最大值叠加（星轨），支持可选权重 |
| `MinStackerOp` | BaseOp | 最小值叠加 |
| `MeanStackerOp` | BaseOp | 均值叠加，输出结果图像和 FastGaussianParam 统计信息 |
| `DiskBufferWriterOp` | BaseOp | 将序列帧写入磁盘缓冲区，供下游多 pass 算法重放 |
| `SigmaClipIteratorOp` | BaseOp | 迭代式 Sigma Clipping，基于磁盘缓冲帧执行多 pass 裁剪 |
| `MedianReduceOp` | BaseOp | 中位数归约（消费 DiskFrameBuffer，分块计算逐像素中位数） |
| **校准** | | |
| `CalibrationSubtractOp` | ParallelBaseOp | 通用校准减法（暗场/偏置帧），reference=None 时 passthrough |
| `CalibrationDivideOp` | ParallelBaseOp | 通用校准除法（平场校正），reference=None 时 passthrough |
| **预处理** | | |
| `ImageResizeOp` | ParallelBaseOp | 图像缩放（按比例或目标尺寸） |
| `ImageCropOp` | ParallelBaseOp | 图像 ROI 裁切 |
| `SequenceSortOp` | BaseOp | 序列排序（自然排序 / 文件名 / 修改时间） |
| `StarAlignmentOp` | BaseOp | 星点对齐（接口预留，核心实现后续接入） |
| **后处理** | | |
| `MaxNoiseEqualizationOp` | BaseOp | 最大值叠加噪声均匀化，消除校正后的空间不均匀伪影 |
| **辅助** | | |
| `WeightGeneratorOp` | BaseOp | 根据 `fin` / `fout` 参数生成渐入渐出权重序列 |
| `ExifReadOp` | ParallelBaseOp | 并行读取 EXIF 信息（CONCURRENCY=4） |
| `ExifReduceOp` | BaseOp | EXIF 信息聚合（如曝光时间累加） |
| `NoneOutputOp` | BaseOp | 输出 None（用于 Meta YAML route="none"，触发下游 passthrough） |
| **元** | | |
| `SubDagOp` | BaseOp | 子 DAG 封装（从 YAML 动态创建） |

### 2.3 数据流与队列机制

节点间通过队列连接，形成流式数据管道。所有队列实现统一的 `BaseQueue` 接口：

```
Feeder（全局数据注入）
    │
    ▼  put()
┌─────────────────┐    get()    ┌─────────────────┐
│    BaseQueue     │ ────────► │    Op 节点       │
│ (RCQ 或 IPCQueue)│            │                  │
└─────────────────┘            │  处理后 put()     │
                               └───────┬──────────┘
                                       │ broadcast
                          ┌────────────┼────────────┐
                          ▼            ▼            ▼
                      Queue A      Queue B      Queue C
                    （下游节点）  （下游节点）  （输出收集）
```

**队列类型**：

| 队列 | 用途 |
|------|------|
| `BaseQueue` | 抽象基类，定义 `put/get/set_length/get_length` 接口 |
| `RichContextQueue` | 进程内异步队列，基于 `asyncio.Queue`（默认） |
| `IPCQueue` | 跨进程队列，基于 `SharedMemory` + `Pipe`（多进程模式自动注入） |
| `FileCacheQueue` | 文件缓存队列，大对象自动序列化到磁盘 |

**关键机制**：

- **背压控制**：队列 `maxsize=1`（默认），上游生产速度自动适配下游消费速度
- **长度协调**：生产者通过 `set_length()` 广播序列长度，消费者通过 `get_length()` 等待就绪
- **信号传播**：正常结束发送 `SENTINEL`，异常发送 `CancellationToken`，下游自动感知
- **Sentinel 驱动**：Filter 类 Op 输出长度未知时，下游通过 sentinel 信号感知序列结束（而非预知长度）
- **接口透明**：Op 代码仅依赖 `BaseQueue` 接口，无需区分进程内/跨进程队列

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

### 3.5 编写 Filter 算子（变长输出）

`FilterBaseOp` 用于输出序列长度不等于输入序列长度的场景（如条件过滤、去重等）。输出序列长度在执行前未知，由 sentinel 信号驱动下游结束。

```python
from hoshicore.ops.base import FilterBaseOp
from hoshicore.component.queue import StreamExhausted

@register_op()
class QualityFilter(FilterBaseOp):
    INPUTS = {"data": {"type": "sequence", "required": True}}
    CONFIGS = {"min_score": {"type": "float", "default": 0.5}}
    OUTPUTS = {"result": {"type": "sequence"}}

    async def _async_execute(self, configs):
        min_score = configs['min_score']
        for i in self._input_range():
            data = self._async_convert_inputs()
            try:
                img = await data['data']
            except StreamExhausted:
                break
            if compute_score(img) >= min_score:
                await self._broadcast_outputs({"result": img})
```

**设计要点**：

- `FilterBaseOp` 自动设置 `VARIABLE_OUTPUT = True`，输出长度广播为 `None`（sentinel 驱动）
- 下游节点通过 `_input_range()`（返回 `itertools.count()`）配合 `except StreamExhausted: break` 感知结束
- 引擎静态检测变长源冲突：多个不同 Filter 源的序列不能汇入同一节点，固定长度和变长序列不能混合

### 3.6 多进程执行

对于 CPU 密集型 DAG，可通过 `run_dag_multiprocess()` 启用多进程执行。CPU 型 Op（`EXECUTOR = "cpu"`）会被分配到 worker 进程，I/O 型 Op 留在主进程。跨进程边界自动使用 `IPCQueue`（SharedMemory 零拷贝传输）。

```python
from hoshicore.engine.multiprocess import run_dag_multiprocess
from hoshicore.engine.build import _load_yaml, validate_and_build_order
import asyncio

spec = _load_yaml("hoshicore/dag/fix_startrail.yaml")
dag = validate_and_build_order(spec)

results = asyncio.run(run_dag_multiprocess(
    dag,
    global_inputs={"fnames": file_list},
    global_configs={"fin": 0.1, "fout": 0.1},
    num_workers=2,   # worker 进程数
    progress=True,
))
```

**架构**（模型 B —— 独立 Event Loop 型）：

```
主进程 (asyncio loop)                    Worker 进程 (asyncio loop)
├── Feeder 协程                           ├── Op C ──RCQ──► Op D
├── Op A ──RCQ──► Op B                    │           (进程内)
│           (进程内)                       └── IPCQueue(consumer)
└── IPCQueue(producer) ──────────────────►
         SharedMemory + Pipe
```

**关键特性**：

- **Op 透明**：Op 代码无需任何修改
- **SharedMemory 零拷贝**：`np.ndarray` 通过 `multiprocessing.shared_memory` 传输，避免序列化开销
- **ShmTransportable 基类**：轻量包装类（如 `FloatImage`）继承 `ShmTransportable` ABC，通过 `shm_pack_into` / `shm_unpack_from` 实现直写 shm 传输
- **反亲和分组**：同 EXECUTOR 类型的 Op 分散到不同 worker 进程，避免 CPU 竞争
- **自动回退**：如果所有 Op 都是 I/O 型（无 CPU Op），自动回退到单进程 `run_dag()`
- **进度追踪**：worker 中的 `ProxyTracker` 通过 `mp.Queue` 将进度事件发回主进程显示
- **Barrier 同步**：所有 worker 就绪后主进程才开始推送数据，避免启动延迟传播

### 3.7 Meta YAML 与路由节点

当同一处理环节有多种可选实现（如叠加方法 mean/median/sigma_clip，校准帧来源 none/master/sequence）时，可使用 **Meta YAML** 在单一文件中声明所有可选路径，避免维护多份几乎相同的 YAML。

Meta YAML 在标准 YAML 基础上为节点新增三个可选字段：

| 字段 | 含义 |
|------|------|
| `route` | 替代 `op`，列出该节点位置可选的多种实现（key → Op 类名或子 DAG YAML） |
| `route_inputs` | 按 route 选项分组的专属输入布线 |
| `route_configs` | 按 route 选项分组的专属配置布线 |

**示例**（校准帧叠加器路由节点）：

```yaml
bias_stacker:
  route:
    none: NoneOutputOp              # 不使用 → 输出 None，下游 passthrough
    master: LoadSingleImageOp       # 从预计算文件加载
    mean: MeanStackerOp             # 从序列求平均
    median: median_stack_core.yaml  # 从序列求中位数（子图）
  route_inputs:
    mean:   { data: inputs.bias_fnames }
    median: { data: inputs.bias_fnames }
  route_configs:
    master: { path: configs.bias_master_path }
    mean:   { int_weight: configs.int_weight }
  outputs:
    result: { type: image }
```

**预处理层**：`meta_resolve(meta_spec, route_choices)` 将 Meta YAML 编译为标准 DAG spec dict，引擎零改动：

```python
import yaml
from hoshicore.engine.meta import meta_resolve
from hoshicore.engine.build import validate_and_build_order
from hoshicore.engine.wiring import run_dag

meta_spec = yaml.safe_load(open("calibration_stack.meta.yaml"))
standard_spec = meta_resolve(meta_spec, route_choices={
    "bias_stacker": "none",
    "dark_stacker": "mean",
    "flat_stacker": "master",
    "main_stacker": "sigma_clip",
})
dag = validate_and_build_order(standard_spec)
results = await run_dag(dag, global_inputs={...}, global_configs={...})
```

**合并规则**：选中 route 的 `route_inputs[choice]` 与共享 `inputs` 合并（route 专属覆盖共享），`route_configs[choice]` 同理。未选中的 route 专属参数被丢弃。

**SubDAG 兼容性**：所有可能不一致的参数（如 `int_weight` 只有部分 route 选项接受）一律放入 `route_configs`，`meta_resolve()` 只注入选中 route 的参数。

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
                │                      └──► sigma_clip.yaml ────────┤
                │                             │ statistics           │
                │                             ▼                     │
                │                      MaxNoiseEqualizationOp ──────┤
                │                                                   ├─► ImageSaveOp
                └──► ExifReadOp ──────► ExifReduceOp ───────────────┘
```

在最大值叠加基础上，利用 Sigma Clipping 子图的统计信息（逐像素均值和方差）进行噪声均匀化校正，消除镜头校正引入的空间不均匀伪影。详见 [噪声均匀化方案](./noise-equalization.md)。

### `sigma_clip.yaml` — Sigma Clipping 子图

```
inputs.data ──┬──► MeanStackerOp ──► statistics ───┐
              │                                    ├──► SigmaClipIteratorOp ──► result / statistics
              └──► DiskBufferWriterOp ──► handle ──┘
```

Sigma Clipping 被拆分为三阶段子图：MeanStackerOp 做初始均值叠加，DiskBufferWriterOp 将帧写入磁盘缓冲区，SigmaClipIteratorOp 基于均值统计和磁盘帧执行多 pass 迭代裁剪。作为子图可通过 `.yaml` 引用嵌入父 DAG。

### `median_stack_core.yaml` — 中位数叠加子图

```
inputs.data ──────► DiskBufferWriterOp ──► MedianReduceOp ──► result
```

中位数叠加子图：帧写入磁盘缓冲后分块计算逐像素中位数。作为子图可嵌入父 DAG 或 Meta YAML route 选项。

### `mean_only.yaml` — 纯均值叠加

```
inputs.fnames ──────► ImgDataLoaderOp ──────► MeanStackerOp ──► ImageSaveOp
```

最简工作流，仅做均值叠加和保存。

### `calibration_stack.meta.yaml` — 通用校准 + 叠加流水线（Meta YAML）

```
bias_fnames ──► bias_stacker ──────────────────────┬──► bias_subtract ──┐
dark_fnames ──► dark_stacker ──────────────────────┼──► dark_subtract ──┤
flat_fnames ──► flat_stacker ──► flat_bias_sub ────┼──► flat_divide ────┤
                                                    │                    │
light_fnames ──► light_loader ─────────────────────┘                    │
                                                                        ▼
                                                     main_stacker ──► image_saver
```

通过 Meta YAML 路由统一管理所有校准和叠加选项。每个 `*_stacker` 是路由节点（none / master / mean / median），`main_stacker` 路由叠加方法（mean / median / sigma_clip）。使用 `meta_resolve()` 编译后执行。

---

## 5. 项目目录结构

```
hoshicore/
├── engine/                    # 引擎层：DAG 构建、布线、执行
│   ├── build.py               #   YAML 解析 + 合法性校验 + 拓扑排序 → ValidatedDag
│   ├── wiring.py              #   Op 实例化 + 队列连接 + feeder 生成 + run_dag / run_from_yaml
│   ├── executor.py            #   DAGExecutor：异步并发调度 + 全局取消机制
│   ├── multiprocess.py        #   多进程引擎：进程分组 + IPCQueue 注入 + worker 管理
│   ├── meta.py                #   Meta YAML 预处理：meta_resolve()（路由解析 → 标准 spec dict）
│   └── registry.py            #   Op 注册表（@register_op 装饰器）
│
├── ops/                       # 算子层：所有 Op 定义
│   ├── base.py                #   BaseOp / ParallelBaseOp / FilterBaseOp 基类
│   ├── dataloader.py          #   ImgDataLoaderOp（异步图像加载）
│   ├── simple_ops.py          #   LoadSingleImageOp / LoadMaskImageOp / ImageResizeOp
│   │                          #   ImageCropOp / CalibrationSubtractOp / CalibrationDivideOp
│   │                          #   SequenceSortOp / NoneOutputOp
│   ├── weight_generator.py    #   WeightGeneratorOp（渐入渐出权重）
│   ├── trailstacker.py        #   TrailStackerOp / MinStackerOp / MeanStackerOp
│   │                          #   MaxNoiseEqualizationOp
│   ├── sigma_clip_ops.py      #   DiskBufferWriterOp / SigmaClipIteratorOp / MedianReduceOp
│   ├── alignment_ops.py       #   StarAlignmentOp（星点对齐，接口预留）
│   ├── exif_op.py             #   ExifReadOp / ExifReduceOp
│   ├── image_saver.py         #   ImageSaveOp / BatchImageSaveOp
│   └── sub_dag.py             #   SubDagOp + create_sub_dag_op()（子图嵌套）
│
├── component/                 # 组件层：底层基础设施
│   ├── queue.py               #   BaseQueue / RichContextQueue / FileCacheQueue / CancellationToken
│   ├── ipc_queue.py           #   IPCQueue（SharedMemory + Pipe 跨进程队列）
│   │                          #   SharedArrayRef / ShmTransportable 协议
│   ├── merger.py              #   MaxMerger / MinMerger / MeanMerger / SigmaClippingMerger
│   ├── dataloader.py          #   BaseLoader / ImgFileListLoader / ArrayLoader
│   ├── calibration.py         #   校准纯函数（calibration_subtract / calibration_divide / resize / crop）
│   ├── image_io.py              #   图像读写（save_img 等）
│   ├── data_container.py      #   dtype 基础设施 + FloatImage / FastGaussianParam / HuberMeanParam 等数据容器
│   ├── frame_buffer.py        #   DiskFrameBuffer（磁盘帧缓冲，用于 Sigma Clipping 多 pass）
│   ├── noise_equalization.py  #   equalize_noise()（噪声均匀化核心算法）
│   ├── exifdata.py            #   ExifData / 读写工具
│   ├── progress.py            #   ProgressTracker / DummyTracker / ProxyTracker / TrackerEventConsumer
│   └── utils.py               #   通用辅助函数（文件格式判断、resize、日志等）
│
├── dag/                       # 预置 DAG 定义
│   ├── fifo_startrail.yaml    #   渐入渐出星轨叠加
│   ├── fix_startrail.yaml     #   噪声均匀化星轨叠加
│   ├── sigma_clip.yaml        #   Sigma Clipping 子图（MeanStacker + DiskBuffer + Iterator）
│   ├── median_stack_core.yaml #   中位数叠加子图（DiskBuffer + MedianReduce）
│   ├── mean_only.yaml         #   纯均值叠加
│   └── calibration_stack.meta.yaml  # 通用校准 + 叠加 Meta YAML（路由节点）
│
├── ezlib/                     # 遗留工具库
│   └── arg.py                 #   StackConfigArg / ImgInfo 等数据类
│
└── starlib/                   # 星点工具库
    ├── star_detect.py          #   星点检测
    └── star_shrink.py          #   星点缩小
```

---

## 6. 相关文档

| 文档 | 内容 |
|------|------|
| [DAG 节点定义规范](./dag_node_definition.md) | YAML DAG 的完整 schema 说明，包含字段定义、link 语法和校验规则 |
| [Op 算子设计](./op.md) | Op 基类设计、队列机制、信号传播、异常处理的详细说明 |
| [噪声均匀化方案](./noise-equalization.md) | 最大值叠加噪声均匀化的数学推导和算法流程 |
| [开发者日志](./dev-log.md) | 成像模型推导、噪声分析等技术背景 |
