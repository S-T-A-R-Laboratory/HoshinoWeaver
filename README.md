<div align="center">

<h1><center>HoshinoWeaver | 织此星辰</center></h1>

[![GitHub release](https://img.shields.io/github/release/Designerspr/HoshinoWeaver.svg)](https://github.com/Designerspr/HoshinoWeaver/releases/latest) [![GitHub Release Date](https://img.shields.io/github/release-date/Designerspr/HoshinoWeaver.svg)](https://github.com/Designerspr/HoshinoWeaver/releases/latest) [![license](https://img.shields.io/github/license/Designerspr/HoshinoWeaver)](./LICENSE) [![Github All Releases](https://img.shields.io/github/downloads/Designerspr/HoshinoWeaver/total.svg)](https://github.com/Designerspr/HoshinoWeaver/releases)

[**简体中文** | [English](./docs/README-en.md)]

</div>

## 简介

HoshinoWeaver 是一个基于 Python 的图像堆栈软件项目，可用于快速从大量照片中创建星轨图像。其针对星轨堆栈实现了优化，包含了针对星轨的实验性功能，还可用于各种模拟慢门场景。

HoshinoWeaver 的主要特性如下：

1. ![new-tag](https://img.shields.io/badge/new!-red) 用户友好的图形化界面：可以借助图形化界面快速预览和配置叠加相关设置。
2. 支持多种常用的叠加模式：常规的星轨叠加，有渐入/渐出效果的星轨，平均值叠加（暂不支持星点对齐），最大值+平均值的混合叠加模式等。
3. 运行快速，内存友好：支持多进程读入与叠加，叠加速度与主流星轨叠加软件齐平；无需一次性将所有照片加载到内存中，对低配置设备更友好。
4. 载入和输出图像时支持完整的EXIF信息和色彩配置信息，能够最大程度输入输出的效果一致性，并保存EXIF更丰富的图像。
5. 实验性功能：包含适用于星轨场景的实验性功能，帮助用户简便获得理想的图像。更多介绍见[已实现的实验性功能](#已实现的实验性功能)。

## 发行版本

目前可以从[Github的Release页](https://github.com/Designerspr/HoshinoWeaver/releases)获取到HoshinoWeaver的所有发行版本。

注意：推荐使用稳定的发行版本。附带有alpha, beta 或debug标识的版本是实验性版本或调试版本，可能仍存在一些缺陷。

## 环境要求

### 环境

至少在 Python >= 3.6 的环境运行该项目。推荐版本为3.9以上。

### 依赖包

见 [requirements.txt](./requirements.txt)。可通过在项目目录下运行 `pip install -r requirements.txt` 快速安装这些依赖包。

在部分设备和python版本上，可能需要手动编译 pyexiv2 以正常进行元数据读写。具体编译方法参见[pyexiv2的说明](https://github.com/LeoHsiao1/pyexiv2/blob/master/docs/Tutorial.md)。

## 用法

HoshinoWeaver支持两种方式启动，即图形界面模式(GUI)与命令行模式(CLI)。推荐从图形界面模式启动，可以更容易的预览与配置叠加任务。

### 图形界面模式(GUI)

要启动图形界面，可以直接运行`HoshinoWeaver desktop.py`:

```bash
python "HoshinoWever desktop.py"
```

如果您使用的发行版本，则直接运行与以上同名的可执行文件即可。当您首次启动图形界面，引导会说明相关用法。也可以随时从“帮助”中调出有关说明。

### 命令行模式(CLI)

如果希望从命令行启动HoshinoWeaver，则运行以下命令：

```sh
python launcher.py  --mode {mean,max,mask-mix,sigmaclip-mean} 
                    [--ground-mask GROUND_MASK]
                    [--fade-in FADE_IN]
                    [--fade-out FADE_OUT]
                    [--int-weight]
                    [--jpg-quality JPG_QUALITY]
                    [--png-compressing PNG_COMPRESSING]
                    [--output OUTPUT]
                    [--output-bits OUTPUT_BITS]
                    [--resize RESIZE]
                    [--num-processor NUM_PROCESSOR]
                    [--debug]
                   dirname
```

命令行参数释义如下：

#### 位置参数
  * `dirname` 图像文件夹目录。目前仅支持直接使用单个目录作为输入。

#### 可选参数
  * `--mode` 堆栈模式。这是一个 **必须的** 参数。目前 HoshinoWeaver 支持 4 中堆栈模式，分别为“最大值”、“平均值”、“Sigma裁切均值”、“最大值-裁切均值混合堆栈”。各个模式的工作原理见[堆栈模式](#已实现的叠加模式算法).
  * `--ground-mask` 地景掩模图像路径。该路径应当指向一个标注出天空区域的图像（黑白图像，但需要按RGB图像储存）。其中天空部分为白色，地景部分为黑色。该参数仅在使用需要地景区域的算法下生效。
  * `--fade-in` 星轨渐入比值。取值为0到1。仅在产生星轨的模式下生效。
  * `--fade-out` 星轨渐出比值。取值为0到1。需要和渐入比值求和小于1。仅在产生星轨的模式下生效。
  * `--int-weight` 叠加时使用整数代替浮点数。这可以带来较大的运行速度提升，并几乎对结果图像没有影响。大多数情况下推荐启用该项。
  * `--jpg-quality` 指定输出的JPEG图像质量。仅在输出JPEG图像时生效。默认值为80。
  * `--png-compressing` 指定输出的PNG图像压缩程度。仅在输出PNG图像时生效。默认值为0。
  * `--output` 指定输出图像文件名称。
  * `--output-bits` 指定输出图像位数。通常应从{8,16}中选择。当输出JPEG图像时，输出图像位数始终为8。
  * `--resize` 调整图像在运行过程中的尺寸。该值在输入阶段生效，因此推荐使用其进行快速叠加获得预览效果。不推荐直接使用该项放缩图像，因为可能损伤图像质量。
  * `--num-processor` 指定使用的并行进程数目。不推荐超过使用电脑的逻辑核心数量。
  * `--debug` 指定该项时，输出流将会打印debug级别信息。在调试场景下可启用。

#### 示例

运行以下指令，创建一个具有30%渐入和30%渐出的TIFF星轨图像：

```sh
python launcher.py /path/to/your/work_dir --mode max --fade-in 0.3 --fade-out 0.3 --int-weight --output "img.tif"
```

运行以下指令，创建一个平均值堆栈的JPEG图像：

```sh
python launcher.py /path/to/your/work_dir --mode mean --output "img.jpg"
```

## HoshinoWeaver已实现

### 已支持的数据类型

* 支持以下数据类型正常加载图像和元数据：
![TIFF/TIF](https://img.shields.io/badge/-TIFF%2FTIF-green) ![JPEG/JPG](https://img.shields.io/badge/-JPEG%2FJPG-green) ![PNG](https://img.shields.io/badge/-PNG-green)

* 支持以下数据类型加载图像和元数据，但还需要更多测试和调整[1]：
![CR2](https://img.shields.io/badge/-CR2-darkgreen) ![CR3](https://img.shields.io/badge/-CR3-darkgreen) ![ARW](https://img.shields.io/badge/-ARW-darkgreen) ![NEF](https://img.shields.io/badge/-NEF-darkgreen) ![DNG](https://img.shields.io/badge/-DNG-darkgreen) ![RA2](https://img.shields.io/badge/-RA2-darkgreen)

* 支持仅加载以下类型图像，暂不支持加载（或数据本身不支持）元数据。
![BMP](https://img.shields.io/badge/-BMP-yellow) ![GIF](https://img.shields.io/badge/-GIF-yellow) ![FITS](https://img.shields.io/badge/-FITS-yellow)

注：[1] 主要是不支持加载和后期相关的调整参数。例如，`HoshinoWeaver`支持读取RAW文件，但暂不支持将PS通用的XMP中的数据加载到RAW图像上。

### 已实现的叠加模式算法

* 最大值叠加： 最大值叠加模式下，结果图像的某一像素取值通过所有图像在该位置的最大值作为结果。通常的星轨叠加即采用该模式。 HoshinoWeaver 还支持在使用最大值叠加时配置渐入渐出比例，创建类似半岛雪人插件效果的渐变星轨。

* 平均值叠加： 平均值叠加模式下，结果图像的某一像素取值通过所有图像在该位置的平均值作为结果。平均值叠加通常被用于夜间地景降噪，日间模拟慢门等。

* Sigma裁剪均值：Sigma裁剪均值是平均值叠加的改进版本。在计算均值时，Sigma裁剪均值会排除显著异常的数据，用以获得对显著干扰的稳健性。当图像存在部分偏差较大的噪声时（如亮灯，过车等），使用Sigma裁剪均值可以获得更纯净的图像。

* 基于掩模的混合叠加 (实验性功能):参见[已实现的实验性功能](#已实现的实验性功能)部分。

### 已实现的叠加数据和计算支持

* 多进程读入和叠加：HoshinoWeaver 可以多进程并行的读入与叠加，叠加速度与主流星轨叠加软件齐平。

* 色彩配置文件/EXIF信息的读入与嵌入：载入和输出图像时支持完整的EXIF信息和色彩配置信息，能够最大程度保证输入输出的效果一致性，并保存EXIF更丰富的图像。

* 高位数的叠加工作流和输出：即使使用8位图像作为输入和输出，也会采用更高位数的叠加工作流，降低计算中的精度损失。

### 已实现的实验性功能

* 基于掩模的混合叠加 (实验性功能): 最终合成图像的天空部分将使用最大值叠加，地面部分则使用平均值叠加结果组合而成。使用该叠加需要额外提供一个用于框选天空部分的掩模图像(Mask)。最大值叠加和均值叠加的图像会存在亮度差异，该差异将通过 HoshinoWeaver 内置的算法修正到相同水平。

### HoshinoWeaver的未来计划

目前HoshinoWeaver仍然处在较早期的开发阶段。主要的未来开发方向有：

1. 图形界面
  * 叠加预览
  * 蒙版绘制

2. 支持已有的叠加算法/选项
  * 去除热燥
  * 创建星轨延时序列
  * 实现星空对齐/星空地景分别对齐的常规堆栈降噪
  * 实现星轨断点的补齐(P0)
  * 支持创建时间切片

3. 输入和输出数据支持
  * 支持视频抽帧叠加
  * 适当的连接断掉的星轨
  * 更好支持各种数据类型（Raw的XMP等）
  * 为max+mean增加并行叠加模式，减少IO次数

4. 实验性功能
  * 简化基于亮度估算方法中对图像方差的预估函数
  * 为混合叠加方法引入高光保护机制
  * 基于排异的混合叠加星轨算法
  * 星轨特殊排异（灯，飞机线？）/反排异（仅保留飞机/灯）(P0)
  * 减弱星点数量，以支持从密集星点图像创建稀疏星轨
  * 从内存估算的并行数量，实现更高效的性能
  * 规避/弱化星轨叠加时的网格问题(P0)
  * 基于蒙版对齐地景实现后期防抖，弱化拍摄过程中小幅位移导致的星轨抖动造成的影响

5. 项目层面
  * 完善的测试流程
  * 日志系统
  * 合理的错误
  * 文档

## 许可

HoshinoWeaver 根据 Mozilla 公共许可证 2.0 (MPL-2.0) 获得许可。这意味着您可以自由使用、修改和分发该软件，但须满足以下条件：

* 源代码可用性：您对源代码所做的任何修改也必须在 MPL-2.0 许可证下可用。这确保了社区可以从改进和变化中受益。
* 文件级 Copyleft：您可以将此软件与不同许可证下的其他代码结合使用，但对 MPL-2.0 许可文件的任何修改都必须保留在同一许可证下。
* 无保证：软件按“原样”提供，不提供任何形式的明示或暗示的保证。使用风险自负。

欲了解更多详细信息，请参阅[MPL-2.0许可证](https://www.mozilla.org/en-US/MPL/2.0/)。

## 附录

### 为什么叫 HoshinoWeaver?

TO BE DONE

### 特别鸣谢

感谢所有参与 HoshinoWeaver 测试和提出建议的用户（目前还没有就是了🤔️）。