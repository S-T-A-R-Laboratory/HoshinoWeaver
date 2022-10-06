# EasyStacker

Python implementation of an image stacker. Suitable for star trails and slow shutter simulation with large amounts of images.

基于Python实现的堆栈/模拟慢门工具。拟提供类似半岛雪人的堆栈功能，但无需一次性将所有照片加载到内存中。

## Usage

To stack a star trail with 30% fade in and 30% fade out, run:

```py
python stacker.py work_dir --fade-in 0.3 --fade-out 0.3
```

## TODO List

### EasyStacker已实现

* 普通星轨✅
* 渐入渐出星轨✅
* 已实现支持的数据类型：![TIFF/TIF](https://img.shields.io/badge/-TIFF%2FTIF-green) ![JPEG/JPG](https://img.shields.io/badge/-JPEG%2FJPG-green) ![PNG](https://img.shields.io/badge/-PNG-green) ![BMP](https://img.shields.io/badge/-BMP-yellow) ![GIF](https://img.shields.io/badge/-GIF-yellow) ![FITS](https://img.shields.io/badge/-FITS-yellow) ![CR2](https://img.shields.io/badge/-CR2-darkgreen) ![CR3](https://img.shields.io/badge/-CR3-darkgreen) ![ARW](https://img.shields.io/badge/-ARW-darkgreen) ![NEF](https://img.shields.io/badge/-NEF-darkgreen) ![DNG](https://img.shields.io/badge/-DNG-darkgreen)

（注：

![TIFF/TIF](https://img.shields.io/badge/-TIFF%2FTIF-green)代表经过一部分测试，且能够正常加载图像和元数据；![CR2](https://img.shields.io/badge/-CR2-darkgreen)代表能够加载图像，但还需要更多测试和调整；![BMP](https://img.shields.io/badge/-BMP-yellow)

代表能够加载图像，但并不保证能获得最好的图像性能。如果可能，请最好不使用这些数据类型作为输入。）

### EasyStacker将计划支持

* 16Bit的叠加工作流
* 最大值/平均值混合叠加
* 颜色配置文件嵌入
* 一个简单的命令行启动器和图形界面
* 适当的连接断掉的星轨
* 叠加预览
* 排异（灯，飞机线？）
* 视频抽帧叠加
* 去除热燥
* 更好支持各种数据类型（Raw的白平衡等）
* 多线程读入，改善速度性能
