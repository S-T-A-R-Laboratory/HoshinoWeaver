# 模型部署元信息

模型目录中的 `metadata.json` 是应用实际读取的部署契约；不是 ONNX 标准的一部分。
它不包含本机路径、训练 checkpoint、导出过程或未分发模型的信息。
当前每个目录对应一个权重，`model.file` 必须与所加载的文件名一致。
多个模型应放在不同目录，避免描述串用。

## 公共字段

| 字段 | 含义 |
| --- | --- |
| `schema_version` | 本项目 JSON 结构版本，目前为 1；不表示模型或 ONNX 版本 |
| `model.file` | 同目录权重文件名，必填 |
| `model.id` / `model.version` | 可选的模型身份和权重版本，不作为兼容性门槛 |
| `task` | `segmentation`、`detection` 或 `image_transform` |
| `adapter` | 解释张量和执行预后处理的适配器标识；不能是可执行代码路径 |
| `inputs` / `outputs` | 按名称描述的非空张量列表；每项必填 `name`、`dtype`、`layout` |

`dtype` 使用 `float16`、`float32` 等描述名称；`layout` 描述维度顺序。
通用读取器检查结构，任务适配器检查支持的 dtype、布局及 ONNX 签名。
元信息是部署者对语义的声明，无法仅凭文件证明模型真的学到了对应语义。
模型必须实现适配器约定；改名字或修改 JSON 不会使不兼容的模型自动兼容。

图像输入可进一步声明：`channels`（RGB/BGR/GRAY）、`signal`（linear/nonlinear）、
`value_range`（归一化前量程）、`normalization`（mean/std）和 `spatial`（缩放/补边约定）。
这里描述的是模型期待的输入，不是调用方原始数组的量程。
当前不引入色彩空间、ICC 或工作流全局图像类型。

## 各任务的语义边界

| 任务 | 任务描述需要补充的内容 |
| --- | --- |
| 分割 | logits/概率/类别标签、类别通道及顺序、输出网格与输入的空间关系 |
| 检测 | boxes/scores/labels 张量、框格式（xyxy/xywh 等）、像素或归一化坐标、类别映射、是否已含 NMS |
| 图变换（降噪等） | 图像输出的通道、量程、线性/非线性约定、空间比例；残差输出或完整图像输出 |

公共读取器允许以上任务描述，当前只接入 `sky_logits` 执行适配器。
检测及图变换尚无执行适配器，不因 metadata 中声明任务便获得推理能力。
未来任务专属字段由对应适配器明确后再实现，避免预先建立泛化执行框架。
降噪等图变换不应复用天空分割的 RGB8 转换而丢失数据精度。

## 当前 sky_logits 契约

参考实际文件 [`models/skyseg_v2/metadata.json`](../models/skyseg_v2/metadata.json)。

- 单输入、单输出，动态空间尺寸、batch=1、NCHW，输入 3 通道、输出 2 通道。
- 输入为非线性 RGB，量程 `[0,255]`，按声明的 mean/std 归一化。
- 输入/输出 dtype、名称须与 ONNX 签名一致。
- 全局输入按长边缩放，再向上取整尺寸并二次缩放；局部输入不缩放，
  在归一化后右/下补零到 `size_divisor` 倍数。
- 输出为与补边后输入同空间尺寸的浮点 logits；局部先裁去补边再 softmax。
- `classes` 为两个通道的标签，`sky_channel` 显式指定天空通道，可为 0 或 1。
- 尺度、阈值和 ROI 策略由调用参数及适配器实现控制，不从描述文件加载算法代码。

服务首次生成时在同一 worker 中读取文件并验证模型接口；作用域内复用预测器。
文件一旦存在，损坏、字段不兼容或权重文件名不匹配都会报错，不静默退回其他配置。
没有旁置文件时允许使用 ONNX 自带 `skyseg` 描述（mean/std、RGB、类别和 divisor）；
上游 `format_version` 不参与兼容性判断，输出仍按当前适配器的形状要求检查。
同时存在时，以旁置文件为准，避免两个配置来源混合覆盖。

## 当前调用方输入适配

`predict_sky` 接受 H×W×3 数组或 `FloatImage`：

| 输入 | 默认量程 |
| --- | --- |
| uint8 / uint16 / uint32 / uint64 | 对应类型 `[0, max]` |
| 裸浮点数组 | `[0,1]`；超出须显式传入 `value_range` |
| `FloatImage` | 使用其 `dtype` 表示的原始量程；浮点 dtype 使用 `[0,1]` |

`value_range=(black, white)` 优先于上述默认值，范围外裁切；不按每张图 min/max 推断。
`color_order` 默认为 BGR，RGB 须显式指定。拒绝 NaN、Inf、空图像和不支持的形状。
`linear=False` 表示不增加拉伸；`linear=True` 在量程归一化后应用
`x ** (1 / stretch_gamma)`，默认 gamma=2.2，再量化为 RGB8。
这是轻量的输入桥接，不是自动直方图拉伸，也不改变调用方原数据。
服务没有色彩管理；GUI 可继续使用现有显示参考图。

```python
result = await services.segmentation.predict_sky(
    float_pixels, value_range=(0, 65535), color_order='RGB',
    linear=True, stretch_gamma=2.2,
)
# FloatImage 无须重复声明其原始 dtype 对应的量程。
result = await services.segmentation.predict_sky(float_image, color_order='BGR')
```

## 验证记录

2026-11-08：输入适配、模型描述、分割/ROI、服务、provider、GUI 和打包相关测试共 133 项通过。
真实 DirectML 三尺度和 18 个局部窗口推理完成，与改动前同图的概率图及二值蒙版逐像素一致。
已检查实际公共打包清单包含 metadata，导入和服务构造不加载 ORT、不初始化 Session。
该验证检查接口与行为一致性，不评价模型质量；未重建发布包。
