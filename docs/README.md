# hoshicore 技术文档

hoshicore 是 HoshinoWeaver (HNW) 的核心计算库，是星空序列图像的通用预处理框架。用户通过 **YAML 声明式 DAG** 定义处理流程，框架自动完成依赖解析、异步调度和流式执行。

**已实现的核心能力**：

- **工作流编排**：YAML DAG、可复用子图、Meta YAML 路由与节点开关
- **图像叠加**：最大值、最小值、均值、中位数、迭代 Sigma Clipping 和 Huber 加权均值
- **校准与预处理**：Bias/Dark/Flat 校正、缩放、裁切和序列排序
- **星点对齐与几何求解**：参考图对齐、共享相机参数与逐帧姿态的 Bundle Adjustment、几何重映射
- **窗口处理与区域合成**：窗口对齐叠加、天空/地面遮罩合成和有效帧筛选
- **星轨处理**：渐入渐出、最大值叠加噪声均匀化、滑窗与指数衰减星轨序列
- **星点与异常轨迹处理**：缩星、卫星等短暂亮轨迹抑制
- **图像与元数据** I/O：多格式图像加载、单图与序列保存、位深转换、EXIF 读取与合并

---

## 1. 架构总览

| 层次 | 目录 | 职责与参考 |
| --- | --- | --- |
| 引擎 | `hoshicore/engine/` | 编译与执行 DAG，见 [DAG Engine 架构](./dag_engine.md) |
| 算子 | `hoshicore/ops/` | 定义处理步骤与扩展接口，见 [Operator 目录与设计](./op.md) |
| 服务 | `hoshicore/services/` | 复用外部程序与模型，见 [服务与外部资源](./services.md) |
| 组件 | `hoshicore/component/` | 提供底层能力，如图像算法、几何求解、数据容器和基础设施等 |
| 工作流 | `hoshicore/dag/` | YAML 流程定义，见 [DAG 节点定义规范](./dag_node_definition.md) |
| 界面 | `ui/` | GUI、参数面板、进度与结果展示 |

编译链路、模块职责、配置解析、资源预检与运行时规划统一见 [引擎文档](./dag_engine.md#编译链路)。

### GUI 与工作流文件

GUI 入口为 `HoshinoWeaver desktop.py`，界面代码位于 `ui/`，使用 PySide6 和 `qasync` 将 Qt 与 asyncio 事件循环整合。`MODE_MAP` 为不同模式选择工作流文件， `ui/panel_builder.py` 中的 `PanelSchema` / `DynamicConfigPanel` 根据 YAML 动态生成参数面板。

- `<name>.meta.yaml`：顶层工作流，声明路由、参数和节点开关。
- `hoshicore/dag/base/<name>.yaml`：可复用子图；节点 `op` 可引用算子类名或子图文件名。
- `<name>.ui.yaml`：前端标签、控件、取值范围和分组等显示提示，与 Meta YAML 配对。

动态参数面板不提交标记为 `hidden: true` 的配置，包括其 YAML 默认值；执行时按 [配置解析约定](./dag_engine.md#关键编译约定) 取得有效值，与 CLI 未传对应 `--config` 时一致。输出路径、位深等配置由专门的输出面板提交。`visible_when` 仅控制显示，不改变参数提交规则。

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

算子使用以下四种扩展基类（后三者均继承自 `BaseOp`）：

| 基类 | 适用场景 | 核心方法 |
|------|---------|---------|
| `BaseOp` | 需要消费整个序列的聚合操作（如叠加） | `_async_execute(configs)` |
| `ParallelBaseOp` | 逐帧独立处理的并行操作（如 EXIF 读取） | `_async_execute_single(data, configs)` |
| `ChunkIteratorBaseOp` | 从可重放帧缓存中分块计算（如中位数、迭代裁剪） | `_init_chunk_state()` / `_merge_chunk()` / `_finalize_chunk()` / `_wrap_output()` |
| `FilterBaseOp` | 输出长度不等于输入长度的过滤操作 | `_async_filter(configs)` |

**Op 声明规范**：每个 Op 通过类属性声明输入输出：

```python
class MyOp(BaseOp):
    EXECUTOR = "cpu"                                    # 执行器标识（可选）
    INPUTS  = {"data": {"type": "sequence", "required": True}}
    CONFIGS = {"threshold": {"type": "float", "default": 0.5}}
    OUTPUTS = {"result": {"type": "image"}}
```

- `INPUTS`：序列数据输入，由队列流式传递
- `CONFIGS`：执行前一次性读取的值，可为标量、图像、统计量或几何方案
- `OUTPUTS`：输出端口，结果通过广播推送到所有下游
- `EXECUTOR`：算子执行类别标记；CPU 密集工作可通过 `_run_cpu()` 卸载到线程池，详见 [运行时执行模型](./dag_engine.md#运行时执行模型)

**内置算子一览**：

见 [op.md](./op.md)。

### 2.3 数据流与队列机制

队列类型、背压、长度先行和结束信号见 [队列通信协议](./dag_engine.md#队列通信协议)；任务调度、输出收集和异常处理见 [运行时执行模型](./dag_engine.md#运行时执行模型) 与 [终止协议](./dag_engine.md#终止协议)。

长度检查、新增算子的长度契约和边界问题见 [长度广播协议](./length_protocol.md)。

---

## 3. 使用方式

### 3.1 快速上手

最简调用方式——通过 `run_from_yaml` 一行启动：

```python
from hoshicore.engine.wiring import run_from_yaml
import asyncio

results = asyncio.run(run_from_yaml(
    "hoshicore/dag/startrail.meta.yaml",
    global_inputs={"fnames": ["img001.jpg", "img002.jpg", "img003.jpg"]},
    global_configs={
        "fin": 0.2,          # 渐入比例
        "fout": 0.2,         # 渐出比例
        "int_weight": True,  # 整型权重优化
        "output_filename": "result.tif",
    },
    route_choices={"mode": "fifo"},
))
```

需要手动编译或执行已校验的 DAG 时，按 [编译链路](./dag_engine.md#编译链路) 完成所需的前置阶段，再调用 `run_dag()`。

### 3.2 编写自定义 YAML DAG

编写步骤：

1. **定义 `inputs`**：声明序列输入（通常是文件名列表）
2. **定义 `configs`**：声明全局配置项及其默认值
3. **定义 `nodes`**：每个节点指定使用的 `op`，并通过 link 语法连接上游数据
4. **定义 `outputs`**：指定最终需要收集的节点输出

默认值注入、可选输入与变长源检查见 [实例化与布线](./dag_engine.md#实例化与布线)；字段和链接格式见 [DAG 节点定义规范](./dag_node_definition.md)。

**Op 注册表**：`hoshicore.engine.registry.REGISTERED_OP` 保存通过 `@register_op()` 注册的算子。在 YAML 的 `op` 字段中使用注册表中的键名即可引用对应的算子类；`op_registry=None` 时运行入口使用该注册表。

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

将以下内容保存为 `my_dag.yaml`：

```yaml
inputs:
  data: {type: sequence}
nodes:
  filter:
    op: MyFilterOp
    inputs: {data: inputs.data}
    outputs: {result: {type: sequence}}
outputs:
  result: filter.result
```

在定义 `MyFilterOp` 后执行：

```python
import asyncio
import numpy as np
from hoshicore.engine.wiring import run_from_yaml
from hoshicore.engine.registry import REGISTERED_OP

my_registry = {**REGISTERED_OP, "MyFilterOp": MyFilterOp}

results = asyncio.run(run_from_yaml(
    "my_dag.yaml",
    global_inputs={"data": [np.full((2, 3, 3), v, dtype=np.float32)
                            for v in (0.25, 0.75)]},
    global_configs={},
    op_registry=my_registry,
    progress=False,
))
```

复制注册表后添加自定义类，可保留标准算子且不修改全局注册表。也可从 `hoshicore.engine.registry` 导入 `register_op` 并用 `@register_op()` 注册；使用前须导入定义该类的模块。

**当前导入顺序限制**：在新 Python 进程中直接先导入 `engine.registry` 会触发循环导入。以上示例先导入 `run_from_yaml` 初始化标准算子，再读取注册表。

### 3.4 子图嵌套（编译期展开）

在节点的 `op` 中直接引用子图 YAML 文件，无需创建或注册 `SubDagOp`：

```yaml
inputs:
  data: {type: sequence}
nodes:
  stacker:
    op: sigma_clip_fused.yaml
    inputs: {data: inputs.data}
    outputs: {result: {type: image}}
outputs:
  result: stacker.result
```

子图的编译阶段见 [引擎编译链路](./dag_engine.md#编译链路)，命名空间与链接重写规则见 [SubDAG 展开](./dag_node_definition.md#4-subdag-展开)。

### 3.5 Meta YAML 与路由节点

当同一处理环节有多种可选实现（如叠加方法 mean/median/sigma_clip，校准帧来源 none/master/sequence）时，可使用 **Meta YAML** 在单一文件中声明所有可选路径，避免维护多份几乎相同的 YAML。

当前格式将路由定义放在顶层 `routes`，节点通过 `route_key` 引用；完整规范见 [Meta YAML 格式](./dag_node_definition.md#5-meta-yaml-格式)。

| 字段 | 含义 |
|------|------|
| 顶层 `routes` | 按路由名声明 `options`（选项 → Op 名或子图 YAML）与 `default` |
| 顶层 `route_configs` | 按路由名、选项名分组声明专属参数及默认值 |
| 节点 `route_key` | 引用顶层路由名；运行时选择按该名称传入，未指定时使用 `default` |
| 节点 `route_inputs` / `route_configs` | 按选项分组的专属输入 / 配置布线，覆盖共享布线 |

**示例**：将以下内容保存为 `stacker.meta.yaml`。`data` 是已加载的图像序列，而非文件名序列。

```yaml
inputs:
  data: {type: sequence}
routes:
  stacker:
    options:
      mean: MeanStackerOp
      median: median_stack_core.yaml
      sigma_clip: sigma_clip_fused.yaml
    default: mean
configs:
  int_weight: {type: bool, default: true}
route_configs:
  stacker:
    sigma_clip:
      rej_high: {type: float, default: 3.0}
      rej_low: {type: float, default: 3.0}
      max_iter: {type: int, default: 5}
nodes:
  stacker:
    route_key: stacker
    inputs: {data: inputs.data}
    route_configs:
      mean: {int_weight: configs.int_weight}
    outputs: {result: {type: image}}
outputs:
  result: stacker.result
```

**运行入口**：`run_from_yaml()` 自动完成路由解析、子图展开、校验、预检和执行：

```python
import asyncio
import numpy as np
from hoshicore.engine.wiring import run_from_yaml

results = asyncio.run(run_from_yaml(
    "stacker.meta.yaml",
    global_inputs={"data": [np.full((2, 3, 3), v, dtype=np.uint16)
                            for v in (10, 20, 30)]},
    global_configs={},
    route_choices={"stacker": "sigma_clip"},
    progress=False,
))
```

**编译链路**：见 [引擎编译链路](./dag_engine.md#编译链路)；路由参数合并与节点开关规则见 [Meta YAML 格式](./dag_node_definition.md#5-meta-yaml-格式)。

**合并规则**：节点选中分支的 `route_inputs[choice]` 与共享 `inputs` 合并，`route_configs[choice]` 与共享 `configs` 合并，分支布线优先。顶层专属参数写入 `configs.<route_key>.<choice>.<param>`，未显式布线的参数自动连接；未选中分支不注入。上例的 `rej_high` 等参数因此无需再写节点布线。

**SubDAG 兼容性**：仅部分选项接受的参数应放入节点的 `route_configs` 分支。上例只对 `mean` 传入 `int_weight`，Sigma Clipping 使用当前生产子图 `sigma_clip_fused.yaml`。节点级 `route: { ... }` 是旧格式，当前引擎不解析。

---

## 4. 内置 DAG 工作流

以下是可由 CLI 或 `run_from_yaml()` 直接运行的顶层工作流。前五项对应 GUI 的模式选择；校准流水线可通过 CLI/API 调用。

| 工作流 | 入口 YAML | 核心用途 |
| --- | --- | --- |
| 星轨叠加 | [startrail.meta.yaml](../hoshicore/dag/startrail.meta.yaml) | 最大值或 Mix 噪声均匀化叠加，支持渐入渐出 |
| 堆栈降噪 | [stack.meta.yaml](../hoshicore/dag/stack.meta.yaml) | 均值、中位数、Sigma Clipping 或 Huber 均值叠加 |
| 对齐叠加 | [aligned_stack.meta.yaml](../hoshicore/dag/aligned_stack.meta.yaml) | 逐帧或 BA 对齐后叠加，支持天空/地面分区处理 |
| 延时降噪 | [norma_bundle_window.meta.yaml](../hoshicore/dag/norma_bundle_window.meta.yaml) | BA 对齐的滑窗叠加，生成降噪图像序列 |
| 星轨延时 | [timelapse_startrail.meta.yaml](../hoshicore/dag/timelapse_startrail.meta.yaml) | 滑窗最大值、加权滑窗或指数衰减星轨序列 |
| 校准与叠加 | [calibration_stack.meta.yaml](../hoshicore/dag/calibration_stack.meta.yaml) | 组合 Bias/Dark/Flat 校准与多种叠加方法 |

工作流的路由、参数和默认值以对应 YAML 为准。

可复用子图及其算子组合统一见 [常见子 DAG](./op.md#常见子-dag)；子图引用和路由语法见 [DAG 节点定义规范](./dag_node_definition.md)。

---

## 5. 开发环境与测试

源码语法基线为 Python 3.10+（使用 `X | Y` 类型注解）；实际依赖的版本要求以
`requirements.txt` 和 `requirements-dev.txt` 为准。主要技术包括 NumPy、OpenCV、
SciPy、PyWavelets（图像计算），NetworkX（DAG 拓扑），PySide6 + qasync（GUI），
rawpy、tifffile、pyexiv2（图像 I/O 与 EXIF）。引擎使用 asyncio，部分 CPU 工作经
`asyncio.to_thread` 调度；可选 native 加速使用 pybind11、CMake、Ninja、OpenMP
和 CUDA，编译器与平台要求见 `csrc/README.md`。

测试位于 `tests/`，使用 pytest、pytest-asyncio 和 pytest-cov：

```powershell
python -m pytest tests/ -v --tb=short --cov=hoshicore --cov-report=term-missing -x
python -m pytest tests/test_yaml_loader.py -v
```

GitHub Actions 的 Python 3.11/3.12 与平台矩阵、native/NumPy 回退检查以
`.github/workflows/test.yaml` 为准。

## 6. 相关文档

| 文档 | 内容 |
|------|------|
| [DAG 节点定义规范](./dag_node_definition.md) | YAML DAG 的完整 schema 说明，包含字段定义、link 语法和校验规则 |
| [Meta YAML v2](./meta_yaml_v2_spec.md) | 路由、参数声明与节点开关的设计细节 |
| [DAG Engine 架构](./dag_engine.md) | 编译链路、模块职责、资源规划与运行时执行模型 |
| [Op 算子设计](./op.md) | Op 基类设计、队列机制、信号传播、异常处理的详细说明 |
| [长度广播协议](./length_protocol.md) | 序列长度如何在 DAG 中先行传播、四层一致检查、新增 Op 合同、缺陷清单与加固路线 |
| [Norma 组件工作原理](./norma.md) | 星点观测、射线匹配、相机与姿态求解、重映射、序列 BA 与窗口几何 |
| [噪声均匀化方案](./noise-equalization.md) | 最大值叠加噪声均匀化的数学推导和算法流程 |
| [对齐与稳定化](./bundle_adjustment_and_stabilization.md) | 星点对齐叠加的几何模型与 bundle adjustment |
| [开发者日志](./dev-log.md) | 成像模型推导、噪声分析等技术背景 |

**架构改造参考**：

- [框架层改造建议](./framework_refactor_notes.md)：记录执行引擎、配置解析、YAML 编译、Op schema、GUI/后端解耦等框架层风险与建议路线图。
- [统一打包入口](./packaging.md)：PyInstaller/Nuitka 后端、公共发行流程与验证。
- [Norma 小波星点恢复](./norma_wavelet_recovery.md)：全局开关、双图候选过滤比例触发、Bundle 整组一致的检测参数与固定边界内的焦距尺度补偿。
- [长度广播协议](./length_protocol.md)：`set_length()` / `get_length()` 的传播路径、四层一致检查、新增 Op 契约，以及 9 项已登记缺陷（含 1 项 P0 静默挂死）与加固顺序。
