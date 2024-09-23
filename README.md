# HoshinoWeaver

[![GitHub release](https://img.shields.io/github/release/Designerspr/HoshinoWeaver.svg)](https://github.com/Designerspr/HoshinoWeaver/releases/latest) [![GitHub Release Date](https://img.shields.io/github/release-date/Designerspr/HoshinoWeaver.svg)](https://github.com/Designerspr/HoshinoWeaver/releases/latest) [![license](https://img.shields.io/github/license/Designerspr/HoshinoWeaver)](./LICENSE) [![Github All Releases](https://img.shields.io/github/downloads/Designerspr/HoshinoWeaver/total.svg)](https://github.com/Designerspr/HoshinoWeaver/releases)

HoshinoWeaver is a Python-based software project designed for image stacking, specifically tailored for creating star trail images from extensive amount of photos. It is also suitable for a wide range of slow shutter simulation scenes.

HoshinoWeaver 是一款基于 Python 的图像堆栈软件项目，适用于从大量照片中创建星轨图像。它也适用于各种模拟慢门场景。

Currently implemented main features are:
1. Fast and memory-friendly: no need to load all photos into memory at once.
2. Can quickly create star trail images with fade-in/fade-out effects.

目前已实现的主要特性：
1. (NEW!) 用户友好的图形化界面，可以快速配置叠加选项。
1. 运行快速，且内存友好：无需一次性将所有照片加载到内存中。
2. 支持常见的最大值星轨叠加，平均值叠加（非对齐的）。
3. 支持创建有渐入/渐出效果的星轨图像。
4. 载入和输出图像时支持完整的EXIF信息和色彩配置信息。

## Usage

So far only the command-line launcher is fully developed and tested. To use HoshinoWeaver, run the following command:

目前只支持从命令行启动HoshinoWeaver。运行以下命令：

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

### Positional Arguments
  * `dirname` directory of images.

### Options
  * `--mode` stack mode. This is a **required** argument. For now, HoshinoWeaver supports 4 modes, respectively "max", "mean", "sigmaclip-mean", and "mask-mix". To see how these modes works, see [Stack mode](#Stack-Mode).
  * `--fade-in` Fade-in ratio. Ranges from 0 to 1. Only works when a star trail is required.
  * `--fade-out` Fade-out ratio. Ranges from 0 to 1. Only works when a star trail is required.
  * `--int-weight` Using integer weight instead of the float weight. This accelerates the program while having little affects on the result image. For most conditions, it is recommended to apply this.
  * `--jpg-quality` Specify the JPG_QUALITY when outputing JPEG image.
  * `--png-compressing` Specify the compression level of the output PNG image.
  * `--output` Specify the output filename.
  * `--debug` When this is activated, print logs with debug level.

### Example

To stack a star trail with 30% fade in and 30% fade out, run:

运行以下指令，创建一个具有30%渐入和30%渐出的星轨图像：

```sh
python launcher.py /path/to/your/work_dir --mode max --fade-in 0.3 --fade-out 0.3 --int-weight --output "img.tif"
```

To stack an average image, run:

运行以下指令，创建一个平均值堆栈图像：

```sh
python launcher.py /path/to/your/work_dir --mode mean --output "img.tif"
```

## Function

### Stack Mode

* 最大值： 最大值叠加模式下，结果图像的某一像素取值通过所有图像在该位置的最大值作为结果。最大值叠加预期将获得与常规星轨合成相同的结果。
* 平均值： 平均值叠加模式下，结果图像的某一像素取值通过所有图像在该位置的平均值作为结果。平均值叠加通常被用于日间模拟慢门，图像降噪等。
* Sigma裁剪均值：Sigma裁剪均值是平均值叠加的改进版本。在计算均值时，Sigma裁剪均值会排除显著异常的数据，用以获得对显著干扰的稳健性。
* 基于掩模的混合叠加 (Mask-mix): 在提供参数时，额外提供一个用于框选天空部分的掩模图像(Mask)。天空部分将使用最大值叠加以合成，地面部分则使用平均值。最大值叠加和均值叠加产生的图像会存在亮度差异将通过HoshinoWeaver算法修正到相同水平。

## Algorithm

(To be updated.)

## TODO List

### HoshinoWeaver已实现

1. 图形界面
  * 一个用户友好的图形化界面

2. 叠加算法
  * 常规最大值叠加/平均值叠加
  * 带渐入渐出的最大值叠加
  * 支持简单蒙版
  * 带排异的平均值叠加
  * 多进程读入和叠加
  * 支持全部读入到内存的叠加（可用于预览）

3. 输入和输出数据支持
  * 颜色配置文件读入与嵌入
  * 更高位数的叠加工作流和输出
  * 插入/合理的修改EXIF信息
  * EXIF检查和提示接口

4. 实验性功能
  * 基于亮度估算的混合叠加星轨算法
---

* 已实现支持的数据类型：![TIFF/TIF](https://img.shields.io/badge/-TIFF%2FTIF-green) ![JPEG/JPG](https://img.shields.io/badge/-JPEG%2FJPG-green) ![PNG](https://img.shields.io/badge/-PNG-green) ![CR2](https://img.shields.io/badge/-CR2-darkgreen) ![CR3](https://img.shields.io/badge/-CR3-darkgreen) ![ARW](https://img.shields.io/badge/-ARW-darkgreen) ![NEF](https://img.shields.io/badge/-NEF-darkgreen) ![DNG](https://img.shields.io/badge/-DNG-darkgreen) ![BMP](https://img.shields.io/badge/-BMP-yellow) ![GIF](https://img.shields.io/badge/-GIF-yellow) ![FITS](https://img.shields.io/badge/-FITS-yellow)

#### 备注

* ![color/green](https://img.shields.io/badge/-green-green) 代表能够正常加载图像和元数据。

* ![CR2](https://img.shields.io/badge/-darkgreen-darkgreen) 代表能够加载图像和元数据，但还需要更多测试和调整。例如，HoshinoWeaver支持读取RAW文件，但并不支持将XMP中的数据加载到RAW图像上。

* ![BMP](https://img.shields.io/badge/-yellow-yellow) 代表能够加载图像，不能加载（或数据本身不支持）元数据。

* pyexiv2在M1/M2 CPU设备上不能直接运行。（稍后补充相关文档）

### HoshinoWeaver的未来计划

目前HoshinoWeaver仍然处在非常早期的开发阶段。主要的未来开发方向有：

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
  * 增加从内存估算的并行数量，实现更高效的性能
  * 规避/弱化星轨叠加时的网格问题(P0)
  * 基于蒙版对齐地景实现后期防抖，弱化拍摄过程中小幅位移导致的星轨抖动造成的影响

5. 项目层面
  * 完善的测试流程
  * 日志系统
  * 合理的错误
  * 文档