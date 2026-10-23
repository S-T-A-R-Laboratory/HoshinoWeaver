# 服务与外部资源

`hoshicore/services/` 管理 Photoshop、ONNX 等需要跨节点复用的外部资源。
Op 通过 `self.services` 调用服务，服务负责创建配置、串行执行与资源释放。
图像预处理、阈值等业务逻辑留在 Op 或请求参数中。

## 作用域与生命周期

| 对象 | 职责 |
| --- | --- |
| `ServiceRegistry` | 进程内注册工厂，协调资源身份与使用计数 |
| `ServiceKey[T]` | 标识服务及返回类型，支持编辑器提示和自定义扩展 |
| `ServiceScope` | 持有一个主图使用的服务，覆盖全部扁平化子图 |
| `SerialService` | 在资源专用的单个 worker 中执行阻塞调用与清理 |

`run_dag()` / `run_from_yaml()` 默认使用进程注册表，也可通过 `service_registry=`
注入应用自己的注册表或隔离测试。工厂只创建轻量适配器，外部程序或模型在首次请求
时初始化；未使用的服务不会加载资源。配置选择器在主图入口选取并快照创建参数，
不复制无关的 Loader 图像数据。

同一资源在重叠执行的图之间共享，单个 Op/Loader 结束不会释放它。最后一个使用它的
主图结束后，等待节点及后台工作完成，再释放资源；后续任务重新按需创建，不保留暖缓存。
新图若在旧资源关闭期间到达，会等待关闭完成后再初始化。

取消尚未执行的请求会跳过它，已经开始的请求会等待结束。清理失败向上传递，但不覆盖
已有的节点根因异常。Op 不单独关闭共享服务；独立调用时用 `async with registry.scope(configs)`
管理作用域。

## 内置服务

### Photoshop

`await services.photoshop.render(path)` 返回 BGR 像素和对应的 `ExifData`。
RAW 使用 Photoshop 默认的相邻同名 XMP；返回的 EXIF/ICC 来自渲染后的 TIFF。
`ImgFileListLoader` 只保留 `raw_backend` 读取策略，非 RAW 仍走普通读取器。

创建配置来自有效的全局配置，不通过 Op 配置队列或 `loader_configs` 传递：

| 配置 | 默认值 | 含义 |
| --- | --- | --- |
| `photoshop_path` | `None` | Windows `.exe` 或 macOS `.app` 路径 |
| `photoshop_timeout` | `180` | 单次渲染超时秒数 |
| `photoshop_hide` | `True` | 隐藏本任务创建的实例 |

路径可在全局设置中启用，或通过现有工作流的 CLI `--config photoshop_path=...` 提供；
超时、隐藏选项可由程序调用的 `global_configs` 提供。相同规范化路径使用一个串行 worker，
重叠图若指定冲突的创建选项，会明确报错。等待请求不占普通 IO 线程池线程。

已有 PS 时借用，不退出它，也不主动改变其窗口可见性；没有时才启动并记录归属。
本任务创建的实例在释放时核对 PID 与创建时间，再请求正常退出；存在其他文档时拒绝
退出，不强杀。启动期间无法确认身份时不接管，不需要 `photoshop_reuse_existing`。

渲染期间检查主进程，退出或被替换时报告错误，不自动重启或重试。若结果已完整写出，
仍可读取；清理跳过已消失或被替换的进程。超时后脚本可能仍在执行，因此保留进程和
诊断文件并报告错误。macOS 尚未实机验证，对齐流程尚未开放 GUI 选项。

### ONNX

`await services.segmentation.infer(inputs, output_names=None)` 接收命名张量映射，
返回请求的张量列表。首次请求才导入可选的 `onnxruntime` 并创建 Session；请求串行执行，
最后释放 Session 引用。调用形式参见 [ONNX Runtime API](https://onnxruntime.ai/docs/api/python/api_summary.html)。

| 全局配置 | 默认值 | 含义 |
| --- | --- | --- |
| `segmentation_model_path` | `None` | 模型路径，实际推理时必填 |
| `segmentation_providers` | `['CPUExecutionProvider']` | 按优先级排列的执行 provider |

配置由程序调用的 `global_configs` 提供；模型路径或 provider 配置不同会得到不同资源。
当前是通用推理接口，尚未绑定具体分割模型、预处理、后处理或 GUI，也未引入强制 runtime 依赖。

## 注册自定义服务

继承 `SerialService`，通过 `_run()` 提交阻塞工作，在 `_close()` 释放资源；两者在同一
专用 worker 上执行。工厂与配置选择器应在主图开始前注册，保持轻量且不执行外部 IO。

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

`require(MY_MODEL)` 的返回类型为 `MyModelService`，无需修改核心容器的属性列表。
未知 key、类型不匹配或重复注册会明确报错。

`settings=` 选择影响资源创建的参数，避免包含每次请求的阈值或整批图像。
默认按选中的完整配置区分资源；可用 `resource_key=` 指定物理资源身份，
例如 Photoshop 按路径识别实例，并拒绝同一活动资源的配置冲突。
