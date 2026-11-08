# 服务与外部资源

`hoshicore/services/` 为 Op 提供跨节点复用的外部资源及具体能力。
职责分为资源管理、协议后端、能力适配和工作流调用；具体服务组合后端与适配器，
统一拥有执行边界和关闭入口，通用后端不认识具体模型。

## 职责边界与当前阶段

| 部分 | 职责 | 当前实现 |
| --- | --- | --- |
| 服务基础设施 | 注册、作用域、取消和关闭 | `ServiceRegistry`、`ServiceScope`、`SerialService` |
| 协议后端 | 创建外部资源、描述接口、执行协议、释放资源 | `component/onnx_backend.py`、Photoshop renderer |
| 能力适配器 | 模型契约校验、预处理、推理组合及输出解释 | `component/sky_segmentation.py` |
| 具体服务 | 组合后端与适配器，提供异步能力方法 | `OnnxService`、`SkySegmentationService`、`PhotoshopService` |
| Op | 队列、长度、进度及工作流策略 | 决定哪帧分割、如何反转或广播遮罩 |
| GUI | 参数编辑、任务启动和结果展示 | 分割尚未接入 GUI；独立预览任务应拥有自己的 scope |

模型的归一化、logits 解释等逻辑放在组件适配器中；服务可以在自己的 worker 内调用
适配器，提供完整的图像到遮罩能力。前后处理在服务线程上执行，不代表算法属于
通用 ONNX 服务。尺度和阈值由请求传入，模型路径和执行 provider 属于资源创建配置。

当前支持 Python 代码扩展：注册 `ServiceKey + factory` 后由 `require()` 调用。
注册体系仍要求 `SerialService`，尚未抽象独立于串行 worker 的生命周期协议。
配置式 exe 协议、能力描述驱动的 GUI、跨能力 Session 共享均未实现；后续出现真实用例
再扩展。路径和参数只能配置已支持的协议，不能自动描述任意程序的输入输出和错误语义。

## 作用域与生命周期

| 对象 | 职责 |
| --- | --- |
| `ServiceRegistry` | 注册服务工厂和创建配置选择器，不持有服务实例 |
| `ServiceKey[T]` | 标识服务及返回类型，支持编辑器提示和自定义扩展 |
| `ServiceScope` | 持有一个主图使用的服务，覆盖全部扁平化子图 |
| `SerialService` | 在资源专用的单个 worker 中执行阻塞调用与清理 |

`run_dag()` / `run_from_yaml()` 默认使用进程注册表，也可通过 `service_registry=`

注入应用自己的注册表或隔离测试。工厂只创建轻量适配器，外部程序或模型在首次请求时初始化；未使用的服务不会加载资源。配置选择器在主图入口选取并快照创建参数，不复制无关的 Loader 图像数据。

服务实例属于单个主图。同一图中的多个节点和全部扁平化子图共享同一个 `ServiceScope`，按服务 key 复用实例；单个 Op/Loader 结束不会释放它。主图结束时，等待节点及后台工作完成，再由 scope 关闭所有已创建的服务。下一张图按需创建新实例，不保留暖缓存。注册表可跨任务复用，但只保存注册信息。

调用方串行执行主图，等待前一次 `run_dag()` / `run_from_yaml()`（包括清理）返回后再开始下一次。服务层不协调跨图并发、资源交接或跨图配置冲突。scope 及其服务接口在所属图的事件循环中使用。

取消尚未执行的请求会跳过它，已经开始的请求会等待结束。清理失败向上传递，但不覆盖已有的节点根因异常。Op 不单独关闭共享服务；独立调用时用 `async with registry.scope(configs)`管理作用域。

## 内置服务

### Photoshop

`await services.photoshop.render(path)` 返回 BGR 像素和对应的 `ExifData`。RAW 使用 Photoshop 默认的相邻同名 XMP；返回的 EXIF/ICC 来自渲染后的 TIFF。`ImgFileListLoader` 只保留 `raw_backend` 读取策略，非 RAW 仍走普通读取器。

创建配置来自有效的全局配置，不通过 Op 配置队列或 `loader_configs` 传递：

| 配置 | 默认值 | 含义 |
| --- | --- | --- |
| `photoshop_path` | `None` | Windows `.exe` 或 macOS `.app` 路径 |
| `photoshop_timeout` | `180` | 单次渲染超时秒数 |
| `photoshop_hide` | `True` | 隐藏本任务创建的实例 |

路径可在全局设置中启用，或通过现有工作流的 CLI `--config photoshop_path=...` 提供；超时、隐藏选项可由程序调用的 `global_configs` 提供。同一图内所有 PS 请求共享一个服务和串行 worker，等待请求不占普通 IO 线程池线程。

已有 PS 时借用，不退出它，也不主动改变其窗口可见性；没有时才启动并记录归属。本任务创建的实例在释放时核对 PID 与创建时间，再请求正常退出；存在其他文档时拒绝退出，不强杀。启动期间无法确认身份时不接管，不需要 `photoshop_reuse_existing`。

渲染期间检查主进程，退出或被替换时报告错误，不自动重启或重试。若结果已完整写出，仍可读取；清理跳过已消失或被替换的进程。超时后脚本可能仍在执行，因此保留进程和诊断文件并报告错误。macOS 尚未实机验证，对齐流程尚未开放 GUI 选项。

### ONNX

`await services.onnx.infer(inputs, output_names=None)` 接收命名张量映射，返回请求的张量列表。
`await services.onnx.describe()` 返回 `ModelDescription`，包含输入输出名称、ONNX 类型和
shape、内嵌 metadata 及实际 providers；在 worker 上读取并复制描述，不暴露裸 Session。
首次请求才导入 `onnxruntime` 并创建 Session；请求串行执行，最后释放 Session 引用。
通用服务不提供默认模型和分割方法。调用形式参见 [ONNX Runtime API](https://onnxruntime.ai/docs/api/python/api_summary.html)。

| 全局配置 | 默认值 | 含义 |
| --- | --- | --- |
| `onnx_model_path` | `None` | 通用模型路径，首次调用 `infer` 或 `describe` 时必填 |
| `onnx_providers` | 自动选择 | 显式列表优先；省略或为 `None` 时自动选择可用加速后端 |
| `onnx_threads` | `4` | ONNX 单次计算线程数，须为正整数 |

配置由程序调用的 `global_configs` 提供；每张图按配置快照创建自己的 Session，图内复用，图结束后释放。
未指定 provider 时，首次请求才查询运行时可用后端，按 CUDA → DirectML → ROCm → CoreML → CPU 的顺序尝试。
自动模式下，加速后端初始化失败会记录警告并尝试下一后端；成功后记录选择模式、请求和实际 providers。
显式列表（包括 provider options）保持原样；不可用或首选后端初始化失败即报错，不切换整个会话。
推理期间的错误直接报错，不重选后端或重复推理。CUDA/DirectML 等执行计划仍允许由 CPU 承担部分算子。
DirectML 自动关闭 memory pattern 并使用顺序执行模式。

`OnnxService` 只包装同步的 `OnnxBackend`；后端不创建 worker，不注册为第二层服务。
模型适配器依赖 `InferenceBackend` 的 `describe()` / `infer()` 小接口，不读取 ORT Session。
后端的描述、推理和关闭必须在拥有它的服务 worker 上执行；扩展服务可组合自己的后端
与适配器，无需调用另一个独立管理生命周期的服务。

### SkySeg v2 天地分割

`services.segmentation` 是独立的 `SkySegmentationService`，复用 ONNX 后端代码，
持有自己的后端实例及 worker，与 `services.onnx` 不共享 Session。

| 全局配置 | 默认值 | 含义 |
| --- | --- | --- |
| `segmentation_model_path` | 内置 SkySeg v2 权重 | 旁置 `metadata.json` 声明接口；须实现 `sky_logits` 契约，任意张量模型使用通用 ONNX 服务 |
| `segmentation_providers` | 自动选择 | provider 选择与通用 ONNX 服务采用相同规则；显式配置优先 |
| `segmentation_threads` | `4` | ONNX 单次计算线程数，须为正整数 |

`await services.segmentation.predict_sky(image, scales=(512, 768, 1024), threshold=0.5, color_order='BGR', strategy='roi', working_long_edge=1536)`
返回 `SkyPrediction`：`probability` 是原图尺寸的 float32 天空概率，`mask` 是同尺寸
uint8 0/1 标签（1=天空，0=地面），`info` 记录版本、策略、实际 providers、尺度与耗时。
默认 `roi` 策略的 `raw_probability` 为细化前全局概率，`info['roi']` 包含工作图尺寸、
全部窗口坐标、实际扫描数、选区/覆盖比例等。`strategy='pyramid'` 可显式跳过局部细化，
此时 `raw_probability` 和 `info['roi']` 均为 `None`。
输入为 H×W×3 unsigned 整数、浮点数组或 `FloatImage`；默认 BGR，RGB 应显式指定。

模型契约和完整输入约定见 [模型部署元信息](model_metadata.md)。
调用完成前不得修改输入数组。

可选关键字参数 `progress=callback` 接收不可变 `SkyProgress` 事件：
`initializing`、`scale_started`、`scale_completed`、`roi_planned`、
`tile_started`、`tile_completed`、`fusing`、`complete`。
字段 `completed` / `total` 是当前阶段的计数：全局阶段为尺度数，局部阶段为实际选中的窗口数，
窗口数在 `roi_planned` 时才确定，可能为 0。`scale` 仅用于全局尺度，`strategy` 标识请求策略，
`seconds` 为刚完成尺度或窗口的耗时。`complete` 为输出就绪标记，不代表尺度/窗口计数，
不包含调用方缓存和作用域关闭。
回调在服务 worker 上同步执行，必须快速返回；GUI 必须通过 Qt queued signal 转发到主线程，
不能直接修改控件。回调异常记录警告，不中断推理。
INFO 日志记录请求输入尺寸、策略、尺度和阈值、各尺度及窗口进度/耗时、选区规模，以及包含初始化的总耗时；
失败记录原因。取消后 worker 可能仍会完成原生计算并记录实际完成情况，GUI 取消状态单独管理。

```python
from hoshicore.services import create_service_registry

# Op 内直接使用 self.services.segmentation；独立调用时显式管理 scope。
async with create_service_registry().scope() as services:
    result = await services.segmentation.predict_sky(bgr_uint8)
    sky_mask = result.mask
    ground_mask = 1 - sky_mask
```

模型放在 `models/skyseg_v2/skyseg_v2_fp16.onnx`，通过 Git LFS 保存，随两个打包后端
的公共资源清单分发，附实际读取的 `metadata.json` 和许可证。
兼容性检查依据具体预处理和张量接口，不要求上游 `format_version` 或固定权重版本。
v1 已取消跟踪、移除本地文件并排除打包；不重写 Git 历史。
ONNX Runtime 已列入 `requirements.txt`，运行时仍按需导入和初始化。
GPU 环境选择对应的 ORT 发行包，避免同时安装多个共享 `onnxruntime` 模块的发行包。
不依赖外部 `skyseg_onnx` 包、训练项目、PyTorch、MMCV 或 MMSegmentation。

`component/sky_segmentation.py` 实现模型适配，`component/sky_roi.py` 实现固定的选区细化：

1. 全局 512/768/1024 按长边保比例缩放，再将宽高向上取整到 32 倍数并二次缩放（不补边）。
   按 metadata 归一化 RGB；各尺度 logits 恢复原尺寸后 softmax，再平均天空概率。
2. 工作图长边限制为 1536，小图不放大；选取半径 32 的边界带，以及概率严格位于 `(0.2,0.8)` 的区域。
3. 中心 384、上下文 64、步长 256，仅扫描中心与选区相交的窗口。
   局部 RGB 不缩放，归一化后右/下补零到 32 倍数，输出 logits 裁去补边后才 softmax。
4. Hann 权重拼接（下限 0.05），sigma=8 软门控，局部融合权重 0.7；
   全局概率 ≥0.9 或 ≤0.1 且局部类别相反时，融合权重上限为 0.2。
5. 原尺寸融合后按 0.5 阈值二值化。

没有递归到原图、额外上下文复核、随机抽查、flip 或颜色修订。
整个请求仍在同一串行 worker 和 Session 上执行，窗口逐个处理，不缓存全部预测。
工作图限幅限制局部扫描成本；原图概率恢复及融合的内存仍随原图像素数增长。

GUI 蒙版编辑器已通过此服务生成并缓存二值蒙版，集成约定见 [GUI 专题](gui.md#生成任务与进度)，
操作步骤见 [用户手册附录 C](manual/manual_cn.md#附录-c遮罩文件制作说明)。
也提供独立诊断入口：

```sh
python -m tools.debug.debug_sky_segmentation input.jpg --output .test-tmp/skyseg
```

诊断工具处理 EXIF 方向与 RGB 图片的 ICC→sRGB 转换，输出原尺寸概率 `.npy`、概率 PNG、
天空二值 PNG（255=天空）与 `run.json`；它直接调用生产服务，不另写推理逻辑。
`--provider` 默认 `auto`，与 GUI/服务采用相同自动选择规则；可显式指定
`CPUExecutionProvider`、`CUDAExecutionProvider` 或 `DmlExecutionProvider`。
`--strategy` 默认 `roi`，`--working-long-edge` 默认 1536；`--strategy pyramid` 显式运行全局分支。

诊断工具只承担能力运行和结果检查。通用 ONNX 及分割服务的实例共享、串行执行、
重复结果一致性和释放行为由 `tests/component/test_services.py` 与
`tests/component/test_sky_segmentation.py` 维护，不要求诊断脚本重复实现这些检查。

## 注册自定义服务

继承 `SerialService`，通过 `_run()` 提交阻塞工作，在 `_close()` 释放资源；两者在同一专用 worker 上执行。工厂与配置选择器应在主图开始前注册，保持轻量且不执行外部 IO。

```python
from hoshicore.services import SerialService, ServiceKey, create_service_registry

class MyModelService(SerialService):
    def __init__(self, settings):
        super().__init__()
        self.model_path = settings['model_path']
        self.model = None

    def _infer(self, image):
        if self.model is None:
            self.model = load_my_model(self.model_path)  # 由扩展实现。
        return self.model(image)

    async def infer(self, image):
        return await self._run(self._infer, image)

    def _close(self):
        self.model = None

MY_MODEL = ServiceKey('my_extension.model', MyModelService)
registry = create_service_registry()  # 保留内置服务，也可创建空 ServiceRegistry。
registry.register(MY_MODEL, MyModelService,
                  settings=lambda c: {'model_path': c.get('my_model_path')})
# 将 registry 作为 service_registry= 传给 run_from_yaml()。
# Op 内：result = await self.services.require(MY_MODEL).infer(image)
```

`require(MY_MODEL)` 的返回类型为 `MyModelService`，无需修改核心容器的属性列表。未知 key、类型不匹配或重复注册会明确报错。

`settings=` 选择影响资源创建的参数，避免包含每次请求的阈值或整批图像。每个 scope 对同一个 key 只创建一个实例；创建参数在主图入口快照，每次请求的业务参数由服务方法传入。

一个主图需要多个模型实例时，为各实例注册不同 key 和配置选择器；仅复用后端代码，
不会因路径相同而自动共用资源。可将 `OnnxService` 注册到自定义 key 处理通用张量模型，
或像 `SkySegmentationService` 一样组合 `OnnxBackend` 与能力适配器。新增 Python 扩展无需
修改 scope 的便捷属性。GUI 应通过任务入口调用服务：工作流沿用主图 scope，独立预览
建立短期 scope，GUI 控件不直接持有 Session 或进程。
