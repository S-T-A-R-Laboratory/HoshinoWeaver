# HoshinoWeaver

Python implementation of an image stacker. Suitable for star trails and slow shutter simulation with large amounts of images.

基于Python的堆栈/模拟慢门工具，针对星轨场景进行了优化。

(Aim to be) the best startrail crafting tools!

做最好用的星轨叠加工具！

主要特性：
1. 运行快速，且内存友好：无需一次性将所有照片加载到内存中。
2. 支持渐入/渐出功能：可以快速创建有渐入/渐出效果的星轨图像。通常类似功能需要使用PS中的插件（如半岛雪人的堆栈），而主流星轨叠加软件不支持类似功能。

## Usage
```sh
python launcher.py [-h] --mode {mean,max} [--fade-in FADE_IN] [--fade-out FADE_OUT] [--int-weight] [--jpg-quality JPG_QUALITY]
                   [--png-compressing PNG_COMPRESSING] [--output OUTPUT] [--debug]
                   dirname
```

### Positional Arguments
  dirname: directory of images

### Options
  --mode {mean,max}     stack mode. Select from "mean" and "max".
  --fade-in Fade-in ratio. Ranges from 0 to 1.
  --fade-out Fade-out ratio. Ranges from 0 to 1.
  --int-weight
  --jpg-quality JPG_QUALITY
  --png-compressing PNG_COMPRESSING
  --output OUTPUT
  --debug               print logs with debug level.


### Example

To stack a star trail with 30% fade in and 30% fade out, run:

```sh
python launcher.py /path/to/your/work_dir --mode max --fade-in 0.3 --fade-out 0.3 --int-weight --output "img.tif"
```

To stack an average image, run:

```sh
python launcher.py /path/to/your/work_dir --mode mean --output "img.tif"
```

## TODO List

### EasyStacker已实现

* 常规最大值叠加/平均值叠加✅
* 渐入渐出最大值叠加✅
* 多进程读入和叠加✅
* 颜色配置文件读入与嵌入✅
* 已实现支持的数据类型：![TIFF/TIF](https://img.shields.io/badge/-TIFF%2FTIF-green) ![JPEG/JPG](https://img.shields.io/badge/-JPEG%2FJPG-green) ![PNG](https://img.shields.io/badge/-PNG-green) ![CR2](https://img.shields.io/badge/-CR2-darkgreen) ![CR3](https://img.shields.io/badge/-CR3-darkgreen) ![ARW](https://img.shields.io/badge/-ARW-darkgreen) ![NEF](https://img.shields.io/badge/-NEF-darkgreen) ![DNG](https://img.shields.io/badge/-DNG-darkgreen) ![BMP](https://img.shields.io/badge/-BMP-yellow) ![GIF](https://img.shields.io/badge/-GIF-yellow) ![FITS](https://img.shields.io/badge/-FITS-yellow)

#### 备注

* ![color/green](https://img.shields.io/badge/-green-green) 代表能够正常加载图像和元数据。

* ![CR2](https://img.shields.io/badge/-darkgreen-darkgreen) 代表能够加载图像和元数据，但还需要更多测试和调整。例如，Easystacker支持读取RAW文件，但并不支持将XMP中的数据加载到RAW图像上。

* ![BMP](https://img.shields.io/badge/-yellow-yellow) 代表能够加载图像，不能加载（或数据本身不支持）元数据。

* pyexiv2在M1/M2 CPU设备上不能直接运行。（稍后补充相关文档）

### EasyStacker的未来计划

目前EasyStacker仍然处在非常早期的开发阶段。主要的未来开发方向有：

1. 图形界面
  * 一个简单的图形界面
  * 叠加预览

2. 支持已有的叠加算法
  * 去除热燥
  * 支持简单蒙版
  * 带排异的平均值叠加
  * 创建星轨延时序列

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