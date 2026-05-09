<div align="center">

<h1><center>HoshinoWeaver | 织此星辰</center></h1>

[![GitHub release](https://img.shields.io/github/release/Designerspr/HoshinoWeaver.svg)](https://github.com/Designerspr/HoshinoWeaver/releases/latest) [![GitHub Release Date](https://img.shields.io/github/release-date/Designerspr/HoshinoWeaver.svg)](https://github.com/Designerspr/HoshinoWeaver/releases/latest) [![Github All Releases](https://img.shields.io/github/downloads/Designerspr/HoshinoWeaver/total.svg)](https://github.com/Designerspr/HoshinoWeaver/releases) 

[![license](https://img.shields.io/github/license/Designerspr/HoshinoWeaver)](./LICENSE) [![Tests](https://github.com/Designerspr/HoshinoWeaver/actions/workflows/test.yaml/badge.svg)](https://github.com/Designerspr/HoshinoWeaver/actions/workflows/test.yaml)

[**简体中文** | [English](./docs/README-en.md)]

</div>

## 简介

HoshinoWeaver 是一个面向天文与星轨摄影的图像堆栈工具，可用于快速从大量照片中创建星轨或对齐堆栈图像。


## 特性

- **多种星轨模式** -- 经典最大值、渐入渐出、去除叠加网格、最大值-均值混合叠加，覆盖常见星轨需求
- **专业堆栈模式** -- 支持多种均值和中位数算法，自动剔除异常值，降噪同时排异噪声
- **完整校准流程** -- 支持 Bias / 暗场 / 平场校准，支持加载 Master 帧或从拍摄序列自动生成
- **可选滤镜** -- 可选的缩星处理，减少改善深空及星轨图像的星点表现
- **设备友好** : 叠加速度与主流星轨叠加软件齐平；无需一次加载全部图像，低配设备也能处理大批量照片


## 发行版本

可以从[Github的Release页](https://github.com/Designerspr/HoshinoWeaver/releases)获取到HoshinoWeaver的所有发行版本。

## 环境要求

### 环境

至少在 Python >= 3.10 的环境运行该项目。

### 依赖包

见 [requirements.txt](./requirements.txt)。可通过在项目目录下运行 `pip install -r requirements.txt` 快速安装这些依赖包。

在部分设备和python版本上，可能需要手动编译 pyexiv2 以正常进行元数据读写。具体编译方法参见[pyexiv2的说明](https://github.com/LeoHsiao1/pyexiv2/blob/master/docs/Tutorial.md)。

## 用法

```bash
python "HoshinoWeaver desktop.py"
```

使用发行版本时直接运行同名可执行文件即可。首次启动会显示引导说明。


## HoshinoWeaver已实现

### 支持的文件格式

| 级别 | 格式 | 说明 |
|------|------|------|
| 完整支持 | TIFF, JPEG, PNG | 图像 + EXIF + 色彩配置文件 |
| RAW 支持 | CR2, CR3, ARW, NEF, DNG, RA2 | 图像 + 基础 EXIF（暂不支持 XMP 调整参数） |
| 基础支持 | BMP, GIF | 仅图像数据 |

### 支持的星轨类叠加模式

| 模式 | 适用场景 | 关键特性 |
|------|----------|----------|
| 最大值星轨 (fifo) | 经典星轨合成 | 支持渐入渐出效果 |
| 混合星轨 (mix) | 星轨 + 地景降噪 | 天空取最大值，地景取均值，自动亮度匹配+抑制噪声网格 |
| Threshold-Max 星轨 | 干净星轨，去除干扰 | 基于背景统计提取亮信号，抑制噪声和干扰 |


### 支持的堆栈类叠加模式

| 模式 | 适用场景 | 关键特性 |
|------|----------|----------|
| 均值 (mean) | 通用降噪、模拟慢门 | 支持整数权重加速 |
| Sigma 裁剪 (sigma_clip) | 有异常帧的降噪 | 迭代剔除异常像素后求均值 |
| 中位数 (median) | 简单稳健降噪 | 天然抗异常值，分块处理节省内存 |
| Huber 均值 (huber_mean) | 稳健降噪 | 基于 Huber 损失函数对异常值降权 |


## RoadMap

### v1.0.0 上线前需要做的
- 前端界面分离，通过配置文件生成可选参数【前端】
- 新的去网格叠加算法【待优化】

To Be Fixed
- Display P3 问题
- 对齐接缝黑边问题

### Further

1. 图形界面
  * 叠加预览
  * 蒙版绘制

2. 支持已知的叠加功能
  * 去除热燥 (暗场) / 去除暗角（平场）
  * 创建星轨延时序列
  * 实现星空对齐/星空地景分别对齐的常规堆栈降噪
  * 星轨断点的补齐(P0)
  * 支持创建时间切片

3. 输入和输出数据支持
  * 视频输入：支持视频抽帧叠加
  * 适当的连接断掉的星轨
  * 图像输入：更好支持各种数据类型（Raw的XMP等）
  * 视频导出：支持导出视频【mp4编码，编码参数配置】

4. 算子能力建设
  * 自动化天地分割
  * 流星Filter算子
  * 鱼眼对齐

5. 序列功能特性
  * 延时自动筛片去闪
  * 延时自动插值去闪
  * 分组对齐叠加

6. 实验性功能
  * 简化基于亮度估算方法中对图像方差的预估函数
  * 基于排异的混合叠加星轨算法
  * 星轨特殊排异（灯，飞机线？）/反排异（仅保留飞机/灯）(P0)
  * 从内存估算的并行数量，实现更高效的性能
  * 后期防抖: 弱化拍摄过程中小幅位移导致的星轨抖动造成的影响

7. 项目层面
  * 完善的测试流程
  * 日志系统
  * 合理的错误
  * 文档

## 许可证

HoshinoWeaver 基于 [Mozilla 公共许可证 2.0 (MPL-2.0)](https://www.mozilla.org/en-US/MPL/2.0/) 开源。您可以自由使用、修改和分发，修改后的源文件需保持同一许可证。

## 致谢

星点对齐功能基于 LoveDaisy 的 [star_alignment](https://github.com/LoveDaisy/star_alignment) 算法实现改进。

感谢所有参与 HoshinoWeaver 测试和提出建议的用户（目前还没有就是了🤔️）。

## 附录

### 为什么叫 HoshinoWeaver?

TO BE DONE


### Stargazers
[![Stargazers over time](https://starchart.cc/Designerspr/HoshinoWeaver.svg?variant=adaptive)](https://starchart.cc/Designerspr/HoshinoWeaver)

                    