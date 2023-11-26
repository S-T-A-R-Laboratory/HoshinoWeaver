# EZStacker

Python implementation of an image stacker. Suitable for star trails and slow shutter simulation with large amounts of images.

基于Python的堆栈/模拟慢门工具。拟提供类似半岛雪人的堆栈功能，但无需一次性将所有照片加载到内存中。

## Usage

To stack a star trail with 30% fade in and 30% fade out, run:

```sh
python launcher.py work_dir --fade-in 0.3 --fade-out 0.3
```

## TODO List

### EasyStacker已实现

* 普通叠加✅
* 渐入渐出叠加✅
* 多进程读入✅
* 已实现支持的数据类型：![TIFF/TIF](https://img.shields.io/badge/-TIFF%2FTIF-green) ![JPEG/JPG](https://img.shields.io/badge/-JPEG%2FJPG-green) ![PNG](https://img.shields.io/badge/-PNG-green) ![CR2](https://img.shields.io/badge/-CR2-darkgreen) ![CR3](https://img.shields.io/badge/-CR3-darkgreen) ![ARW](https://img.shields.io/badge/-ARW-darkgreen) ![NEF](https://img.shields.io/badge/-NEF-darkgreen) ![DNG](https://img.shields.io/badge/-DNG-darkgreen) ![BMP](https://img.shields.io/badge/-BMP-yellow) ![GIF](https://img.shields.io/badge/-GIF-yellow) ![FITS](https://img.shields.io/badge/-FITS-yellow)

#### 备注

* ![color/green](https://img.shields.io/badge/-green-green) 代表能够正常加载图像和元数据。

* ![CR2](https://img.shields.io/badge/-darkgreen-darkgreen) 代表能够加载图像和元数据，但还需要更多测试和调整。例如，Easystacker支持读取RAW文件，但并不支持将XMP中的数据加载到RAW图像上。

* ![BMP](https://img.shields.io/badge/-yellow-yellow) 代表能够加载图像，不能加载（或数据本身不支持）元数据。

### EasyStacker将计划支持

* 16Bit的叠加工作流
* 最大值/平均值混合叠加
* 颜色配置文件嵌入
* 一个简单的图形界面
* 适当的连接断掉的星轨
* 叠加预览
* 排异（灯，飞机线？）
* 视频抽帧叠加
* 去除热燥
* 更好支持各种数据类型（Raw的白平衡等）