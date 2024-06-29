# HoshinoWeaver

HoshinoWeaver is a Python-based software project designed for image stacking, specifically tailored for creating star trail images from extensive amount of photos. It is also suitable for a wide range of slow shutter simulation scenes.

HoshinoWeaver 是一款基于 Python 的图像堆栈软件项目，专门用于从大量照片中创建星轨图像。它也适用于各种慢速快门模拟场景。

Currently implemented main features are:
1. Fast and memory-friendly: no need to load all photos into memory at once.
2. Can quickly create star trail images with fade-in/fade-out effects.

目前已实现的主要特性：
1. 运行快速，且内存友好：无需一次性将所有照片加载到内存中。
2. 可以快速创建有渐入/渐出效果的星轨图像。

## Usage

So far only the command-line launcher is fully developed and tested. To use HoshinoWeaver, run the following command:

目前只支持从命令行启动HoshinoWeaver。运行以下命令：

```sh
python launcher.py [-h] --mode {mean,max} [--fade-in FADE_IN] [--fade-out FADE_OUT] [--int-weight] [--jpg-quality JPG_QUALITY]
                   [--png-compressing PNG_COMPRESSING] [--output OUTPUT] [--debug]
                   dirname
```

### Positional Arguments
  * `dirname` directory of images.

### Options
  * `--mode {mean,max}` stack mode. This is a **required** argument. Select from "mean" and "max".
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

## TODO List

### HoshinoWeaver已实现

* 常规最大值叠加/平均值叠加✅
* 渐入渐出最大值叠加✅
* 多进程读入和叠加✅
* 颜色配置文件读入与嵌入✅
* 已实现支持的数据类型：![TIFF/TIF](https://img.shields.io/badge/-TIFF%2FTIF-green) ![JPEG/JPG](https://img.shields.io/badge/-JPEG%2FJPG-green) ![PNG](https://img.shields.io/badge/-PNG-green) ![CR2](https://img.shields.io/badge/-CR2-darkgreen) ![CR3](https://img.shields.io/badge/-CR3-darkgreen) ![ARW](https://img.shields.io/badge/-ARW-darkgreen) ![NEF](https://img.shields.io/badge/-NEF-darkgreen) ![DNG](https://img.shields.io/badge/-DNG-darkgreen) ![BMP](https://img.shields.io/badge/-BMP-yellow) ![GIF](https://img.shields.io/badge/-GIF-yellow) ![FITS](https://img.shields.io/badge/-FITS-yellow)

#### 备注

* ![color/green](https://img.shields.io/badge/-green-green) 代表能够正常加载图像和元数据。

* ![CR2](https://img.shields.io/badge/-darkgreen-darkgreen) 代表能够加载图像和元数据，但还需要更多测试和调整。例如，HoshinoWeaver支持读取RAW文件，但并不支持将XMP中的数据加载到RAW图像上。

* ![BMP](https://img.shields.io/badge/-yellow-yellow) 代表能够加载图像，不能加载（或数据本身不支持）元数据。

* pyexiv2在M1/M2 CPU设备上不能直接运行。（稍后补充相关文档）

### HoshinoWeaver的未来计划

目前HoshinoWeaver仍然处在非常早期的开发阶段。主要的未来开发方向有：

1. 图形界面
  * 一个简单的图形界面
  * 叠加预览

2. 支持已有的叠加算法/选项
  * 去除热燥
  * 支持简单蒙版
  * 带排异的平均值叠加
  * 创建星轨延时序列
  * 支持创建时间切片

3. 输入和输出数据支持
  * 支持视频抽帧叠加
  * 16Bit的叠加工作流和输出
  * 适当的连接断掉的星轨
  * 插入/合理的修改EXIF信息
  * 更好支持各种数据类型（Raw的XMP等）

4. 实验性功能
  * 最大值/平均值混合叠加星轨算法
  * 星轨特殊排异（灯，飞机线？）/反排异（仅保留飞机/灯）
  * 减弱星点数量，以支持从密集星点图像创建稀疏星轨

5. 项目层面
  * 完善的测试流程
  * 日志系统
  * 合理的错误
  * 文档