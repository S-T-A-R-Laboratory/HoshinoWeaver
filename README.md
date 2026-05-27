<div align="center">

<h1><center>HoshinoWeaver | 织此星辰</center></h1>


[![GitHub release](https://img.shields.io/github/release/S-T-A-R-Laboratory/HoshinoWeaver.svg)](https://github.com/S-T-A-R-Laboratory/HoshinoWeaver/releases/latest) [![GitHub Release Date](https://img.shields.io/github/release-date/S-T-A-R-Laboratory/HoshinoWeaver.svg)](https://github.com/S-T-A-R-Laboratory/HoshinoWeaver/releases/latest) [![Github All Releases](https://img.shields.io/github/downloads/S-T-A-R-Laboratory/HoshinoWeaver/total.svg)](https://github.com/S-T-A-R-Laboratory/HoshinoWeaver/releases) 
[![license](https://img.shields.io/github/license/S-T-A-R-Laboratory/HoshinoWeaver)](./LICENSE) [![Tests](https://github.com/S-T-A-R-Laboratory/HoshinoWeaver/actions/workflows/test.yaml/badge.svg)](https://github.com/S-T-A-R-Laboratory/HoshinoWeaver/actions/workflows/test.yaml)

[**简体中文** | [English](./docs/README-en.md)]

</div>

## 简介

HoshinoWeaver (织此星辰, HNW) 是一个为天文摄影设计的通用图像预处理工具，支持创建渐隐星轨、图像对齐、图像堆栈等功能。在基础功能以外，HoshinoWeaver 还为上述功能添加了独创的优化与功能(去除星轨卫星线，消除星轨网格等)。

它也是一个灵活的计算图编排工具：你可以为自己的后期场景自定义计算流程，并通过 HoshinoWeaver 一键运行，而不必在多个软件之间流转。

访问 [官网](https://hoshinoweaver.springcitystudio.top/) 以了解更多最新信息。

## 功能一览

### 星轨叠加

将多张连续拍摄的图像合成为一张完整星轨照片。

| 模式 | 适用场景 | 说明 |
|------|----------|------|
| **最大值叠加** | ISO 较低、噪声少、未做镜头校正 | 逐像素取最大值，速度极快。支持渐入渐出效果 |
| **噪声均匀化** | 高ISO拍摄、有亮光干扰，应用镜头校正等场景 | 在最大值叠加基础上自动校正空间噪声不均，可融合均值地景 |

可选增强：去除卫星线 / 缩星

### 堆栈降噪

多张图像直接叠加求统计值，适合模拟慢门（流云、流水、平滑海面）以及固定场景的降噪叠加。支持均值、Sigma 裁剪、中位数、Huber 均值等多种稳健统计算法。

### 星点对齐叠加

将多张星空照片按星点对齐叠加，产出高信噪比的星空图像。

| 能力 | 说明 |
|------|------|
| **自动对齐** | 检测星点并对齐旋转误差，支持透视变换和畸变优化两种模型 |
| **天地分离** | 通过遮罩分别处理天空和地面，天空按星点对齐叠加降噪，地面独立叠加保留细节 |
| **灵活的叠加算法** | 天空和地面可分别选用不同的叠加算法（均值 / Sigma 裁剪 / 中位数 / Huber 均值） |

> 详细的参数说明和使用指引请参阅 [用户手册](./docs/manual/manual_cn.md)。

### 支持的文件格式

| 支持程度 | 格式类型 | 备注 |
| --- | --- | --- |
| **完整支持** | TIFF, JPEG, PNG | 保留 EXIF 信息与色彩配置文件 |
| **RAW 支持** | CR2, CR3, ARW, NEF, DNG, RA2 | 基础解析（不支持 XMP 调整） |
| **基础支持** | BMP, GIF, FITS | 仅读入像素数据 |


## 快速上手

### 运行发行版本

目前的最新版本是 `v1.0.0 "Vega"`。可以从 [官方网站](https://hoshinoweaver.springcitystudio.top/) 或 [GitHub Release 页](https://github.com/S-T-A-R-Laboratory/HoshinoWeaver/releases) 获取。下载安装后，双击运行 `HoshinoWeaver.exe` 即可启动图形界面。

> [!NOTE]
> 
> **基本使用流程：**
> 
> 1. **选择工作流** — 根据预期结果（星轨 / 降噪 / 星点对齐）选择对应工作流
> 2. **导入图像** — 选择待处理的图像序列
> 3. **准备遮罩**（按需） — 部分功能需要一张标注天空/地面分界的黑白遮罩图像
> 4. **调整参数** — 选择模式并配置参数，初次使用保持默认即可
> 5. **执行输出** — 设置输出路径和格式，点击执行
> 
> 详细的参数说明与操作指引请参阅 [用户手册](./docs/manual/manual_cn.md)。

### 从源码运行

- 至少需要 Python >= 3.10 的环境。
- 在项目目录下运行 `pip install -r requirements.txt` 安装依赖包。
- 运行 `python "HoshinoWeaver desktop.py"` 启动图形界面，或运行 `python launcher.py --help` 查看 CLI 的运行参数。
- 项目包含可选的 C++/CUDA 加速算子，可以通过 `python csrc/build_ops.py` 构建加速算子。无编译环境时会自动回退到 NumPy 实现。


## 技术特性

以下特性使 HoshinoWeaver 能够在普通硬件上高效处理大量高分辨率图像：

- **流式处理**：帧数据逐张流过管线，无需同时加载所有图片，低内存即可处理大批量高像素素材
- **并行计算加速**：通过 OpenMP 多线程和可选的 GPU 加速提高计算效率
- **DAG 算子引擎**：处理流程基于有向无环图驱动，通过 YAML 自由定义。你可以像搭建积木一样组合出自定义工作流，也可以开发新的算子扩展处理能力


### 相关文档

| 文档 | 内容 |
|------|------|
| [技术架构](./docs/README.md) | hoshicore 引擎层、算子层、队列机制、多进程执行等详细设计 |
| [DAG 节点定义规范](./docs/dag_node_definition.md) | YAML DAG 完整语法规范 |
| [C++ 算子构建](./csrc/README.md) | C++/CUDA 自定义算子的构建、平台策略、新增算子指南 |
| [性能基准测试](./bench/README.md) | Benchmark 套件使用说明 |
| [用户手册](./docs/manual/manual_cn.md) | 面向用户的完整功能与参数说明 |

## 附录

### 许可证

本项目基于 [MPL-2.0](./LICENSE) 协议开源。

### 致谢

* 星点对齐算法改进自 [LoveDaisy/star_alignment](https://github.com/LoveDaisy/star_alignment/)。
* 感谢所有为本项目提供样片与建议的摄影师。

### 为什么叫 HoshinoWeaver？

**Hoshino** 代表我们的目标（与致敬）；**Weaver** 代表我们的方式。

> "数字时代的摄影不再仅仅是捕捉，更是对数据的重新编织。我们希望通过这个工具，让每一位摄影师都能精密地控制每一根'数据经纬'，最终织出一幅属于自己的星河长卷。"

## Stargazers

[![Stargazers over time](https://starchart.cc/S-T-A-R-Laboratory/HoshinoWeaver.svg?variant=adaptive)](https://starchart.cc/S-T-A-R-Laboratory/HoshinoWeaver)
