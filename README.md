<div align="center">

<h1><center>HoshinoWeaver | 织此星辰</center></h1>


[![GitHub release](https://img.shields.io/github/release/Designerspr/HoshinoWeaver.svg)](https://github.com/Designerspr/HoshinoWeaver/releases/latest) [![GitHub Release Date](https://img.shields.io/github/release-date/Designerspr/HoshinoWeaver.svg)](https://github.com/Designerspr/HoshinoWeaver/releases/latest) [![Github All Releases](https://img.shields.io/github/downloads/Designerspr/HoshinoWeaver/total.svg)](https://github.com/Designerspr/HoshinoWeaver/releases) 

[![license](https://img.shields.io/github/license/Designerspr/HoshinoWeaver)](./LICENSE) [![Tests](https://github.com/Designerspr/HoshinoWeaver/actions/workflows/test.yaml/badge.svg)](https://github.com/Designerspr/HoshinoWeaver/actions/workflows/test.yaml)

[**简体中文** | [English](./docs/README-en.md)]

</div>

## 简介

HoshinoWeaver (织此星辰, HNW) 是一个为天文摄影设计的通用图像预处理工具。 通过创新的算子编排引擎，无论是简单的星轨还是复杂的分离堆栈合成，HoshinoWeaver都能为你织就理想的星辰影像。

它是一个合成软件，也是一个灵活的计算图编排工具：你可以为自己的后期场景自定义计算流程，并通过 HoshinoWeaver 一键运行，而不必在多个软件之间流转。

访问 [官网](https://hoshinoweaver.springcitystudio.top/) 以了解更多最新信息。


## 核心特性

### 🚀 性能与架构

* **流式处理 (Stream Processing)**：无需同时加载所有照片，低内存也能处理大量高像素的图片。
* **并行计算加速**：通过 OpenMP 和 GPU 加速技术，提高计算效率。
* **DAG 算子引擎**：基于有向无环图驱动计算引擎，计算流程通过 YAML 自由定义。这意味着你可以像搭建积木一样，组合出专属于你的处理工作流。

### 🌠 核心算法

* **星轨合成**：除了常规的渐入渐出星轨，还支持去除卫星线，缩星，抑制噪声网格等多种模式，让星轨创作更加简单
* **星点对齐**：支持按星点对齐堆栈星空降噪，还支持通过蒙版分离天空与地面，同时兼顾地面细节
* **专业堆栈算法**：支持 Sigma Clip、Huber Mean 等稳健统计算法，可以剔除卫星线、飞机线等异常干扰。

## 快速上手

### 运行发行版本

目前的最新版本是 `v1.0.0 "Vega"`。可以从 [官方网站](https://hoshinoweaver.springcitystudio.top/) 或 [Github的Release页](https://github.com/Designerspr/HoshinoWeaver/releases) 获取发行版本。当下载并安装完毕后， 双击运行 `HoshinoWeaver.exe` 即可运行图形界面版本。

> [!NOTE]
> 首次启动将进入引导模式，帮助你快速上手图形界面。

### 从源码运行

- 至少在 Python >= 3.10 的环境运行该项目。
- 在项目目录下运行 `pip install -r requirements.txt` 安装依赖包。
- 运行 `python "HoshinoWeaver desktop.py"` 启动图形界面，或运行 `python launcher.py --help` 查看 CLI 的运行参数。

## 开发说明

项目包含可选的 C++/CUDA 加速算子，无编译环境时自动回退到 NumPy 实现。

- `hoshicore/_custom_op/` — 算子 Python 接口
- `csrc/` — C++/CUDA 源码与构建脚本，详见 [csrc/README.md](./csrc/README.md)
- `bench/` — 性能基准测试，详见 [bench/README.md](./bench/README.md)

构建本地加速算子：

```bash
python csrc/build_ops.py
```


## HoshinoWeaver已实现

### 支持的文件格式

| 支持程度 | 格式类型 | 备注 |
| --- | --- | --- |
| **完整支持** | TIFF, JPEG, PNG | 保留 EXIF 信息与色彩配置文件 |
| **RAW 支持** | CR2, CR3, ARW, NEF, DNG, RA2 | 基础解析（不支持 XMP 调整） |
| **基础支持** | BMP, GIF | 仅读入像素数据 |

### 支持的工作流

#### 星轨类叠加

| 模式 | 适用场景 | 关键特性 |
|------|----------|----------|
| 最大值星轨 (fifo) | 经典星轨合成 | 支持渐入渐出效果 |
| 混合星轨 (mix) | 星轨 + 地景降噪 | 天空使用最大值叠加，地景取均值叠加，自动亮度匹配+抑制噪声网格 |

* 星轨类可选支持：去除卫星线 / 稀疏星轨

### 对齐类叠加

| 模式 | 适用场景 | 关键特性 |
|------|----------|----------|
| 星点对齐 | 图像绝大部分区域是星空 | 自动检测并对齐星点，消除旋转误差，叠加星空图像 |
| 天地分别对齐叠加 | 天空和地面都需要叠加降噪 | 自动分离及合并天空与地面的对齐结果，兼顾星空降噪与地面细节 |

## 附录

### 许可证

* 本项目基于 [MPL-2.0](https://www.google.com/search?q=./LICENSE) 协议开源。

### 致谢

* 星点对齐算法改进自 [LoveDaisy/star_alignment]()。
* 感谢所有为本项目提供样片与建议的摄影师。

### 为什么叫 HoshinoWeaver？

**Hoshino** 代表我们的目标（与致敬）；**Weaver** 代表我们的方式。

> [!Info]
> “数字时代的摄影不再仅仅是捕捉，更是对数据的重新编织。我们希望通过这个工具，让每一位摄影师都能精密地控制每一根“数据经纬”，最终织出一幅属于自己的星河长卷。”

### Project Stargazers
[![Stargazers over time](https://starchart.cc/Designerspr/HoshinoWeaver.svg?variant=adaptive)](https://starchart.cc/Designerspr/HoshinoWeaver)
