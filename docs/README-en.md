<div align="center">

<h1><center> <img height=30 src="../imgs/HNW.jpg"> HoshinoWeaver</center></h1>


[![GitHub release](https://img.shields.io/github/release/S-T-A-R-Laboratory/HoshinoWeaver.svg)](https://github.com/S-T-A-R-Laboratory/HoshinoWeaver/releases/latest) [![GitHub Release Date](https://img.shields.io/github/release-date/S-T-A-R-Laboratory/HoshinoWeaver.svg)](https://github.com/S-T-A-R-Laboratory/HoshinoWeaver/releases/latest) [![Github All Releases](https://img.shields.io/github/downloads/S-T-A-R-Laboratory/HoshinoWeaver/total.svg)](https://github.com/S-T-A-R-Laboratory/HoshinoWeaver/releases) 
[![license](https://img.shields.io/github/license/S-T-A-R-Laboratory/HoshinoWeaver)](../LICENSE) [![Tests](https://github.com/S-T-A-R-Laboratory/HoshinoWeaver/actions/workflows/test.yaml/badge.svg)](https://github.com/S-T-A-R-Laboratory/HoshinoWeaver/actions/workflows/test.yaml)

[[简体中文](../README.md) | **English**]

</div>

## Introduction

HoshinoWeaver (HNW) is a general-purpose image preprocessing tool designed for astrophotography. It supports fading star trails, image alignment, image stacking, and more. Beyond these core features, HoshinoWeaver introduces original enhancements such as satellite trail removal and star trail grid artifact elimination. You can also define custom processing pipelines for your own post-processing scenarios and run them with a single click, without switching between multiple applications.

Visit the [official website](https://hoshinoweaver.springcitystudio.top/) for the latest information.

## Features

### Star Trail Stacking

Combine multiple consecutively captured images into a single complete star trail photograph.

| Mode | Best For | Description |
|------|----------|-------------|
| **Maximum Stacking** | Low ISO, low noise, no lens correction applied | Per-pixel maximum value; extremely fast. Supports fade-in/fade-out effects |
| **Noise Equalization** | High ISO, light pollution, lens correction applied | Automatically corrects spatial noise non-uniformity on top of maximum stacking; supports blending mean-value foreground |

Optional enhancements: Satellite trail removal / Star shrinking

### Noise Reduction Stacking

Stack multiple images using statistical methods. Ideal for simulating long exposures (flowing clouds, silky water, smooth seascapes) and fixed-scene noise reduction. Supports mean, sigma clipping, median, Huber mean, and other robust statistical algorithms.

### Star-Aligned Stacking

Align and stack multiple starfield images by detected star positions to produce high signal-to-noise ratio images.

| Capability | Description |
|------------|-------------|
| **Auto Alignment** | Detects stars and corrects rotational errors; supports both perspective transform and distortion optimization models |
| **Sky/Ground Separation** | Uses a mask to process sky and ground independently — sky is aligned by stars for noise reduction, ground is stacked separately to preserve detail |
| **Flexible Stacking Algorithms** | Sky and ground can each use different stacking algorithms (Mean / Sigma Clipping / Median / Huber Mean) |

> For detailed parameter descriptions and usage guides, see the [User Manual](./manual/manual_cn.md).

### Supported File Formats

| Support Level | Format Types | Notes |
|---------------|--------------|-------|
| **Full Support** | TIFF, JPEG, PNG | Preserves EXIF metadata and color profiles |
| **RAW Support** | CR2, CR3, ARW, NEF, DNG, RW2, RAF, ORF | Basic decoding (XMP adjustments not supported) |
| **Basic Support** | BMP, GIF, FITS | Pixel data import only |


## Getting Started

### Running a Release Build

The current latest version is `v1.0.0-rc "Vega"`. Download it from the [official website](https://hoshinoweaver.springcitystudio.top/) or the [GitHub Releases page](https://github.com/S-T-A-R-Laboratory/HoshinoWeaver/releases). After installation, double-click `HoshinoWeaver.exe` to launch the GUI.

> [!NOTE]
> 
> **Basic workflow:**
> 
> 1. **Choose a workflow** — Select the workflow matching your goal (Star Trail / Noise Reduction / Star-Aligned Stacking)
> 2. **Import images** — Select the image sequence to process
> 3. **Prepare a mask** (if needed) — Some features require a black-and-white mask image marking the sky/ground boundary
> 4. **Adjust parameters** — Choose a mode and configure parameters; defaults work well for first-time use
> 5. **Execute** — Set the output path and format, then click Run
> 
> For detailed parameter descriptions and usage guides, see the [User Manual](./manual/manual_cn.md).

### Running from Source

- Requires Python >= 3.10.
- Run `pip install -r requirements.txt` in the project directory to install dependencies.
- Run `python "HoshinoWeaver desktop.py"` to launch the GUI, or run `python launcher.py --help` to view CLI options.
- The project includes optional C++/CUDA accelerated operators. Build them with `python csrc/build_ops.py`. The system automatically falls back to NumPy implementations when no compiler is available.


## Technical Highlights

The following features enable HoshinoWeaver to efficiently process large numbers of high-resolution images on ordinary hardware:

- **Streaming Pipeline**: Frames flow through the pipeline one at a time — no need to load all images simultaneously, enabling processing of large batches of high-resolution images with minimal memory
- **Parallel Computation**: OpenMP multi-threading and optional GPU acceleration for improved performance
- **DAG Operator Engine**: Processing pipelines are driven by a Directed Acyclic Graph defined via YAML. You can freely combine operators like building blocks to create custom workflows, or develop new operators to extend processing capabilities


### Documentation

| Document | Content |
|----------|---------|
| [Technical Architecture](./README.md) | hoshicore engine, operator system, queue mechanisms, multi-process execution |
| [DAG Node Definition Spec](./dag_node_definition.md) | Complete YAML DAG syntax reference |
| [C++ Operator Build Guide](../csrc/README.md) | C++/CUDA custom operator build process, platform strategies, adding new operators |
| [Benchmark Suite](../bench/README.md) | Benchmark usage guide |
| [User Manual](./manual/manual_cn.md) | Complete feature and parameter reference for end users |

## Appendix

### License

This project is open-sourced under the [MPL-2.0](../LICENSE) license.

### Acknowledgements

* The star alignment algorithm is improved from [LoveDaisy/star_alignment](https://github.com/LoveDaisy/star_alignment/).
* Thanks to all photographers who provided sample images and suggestions for this project.

### Why "HoshinoWeaver"?

**Hoshino** represents our goal (and tribute); **Weaver** represents our method.

> "Photography in the digital age is no longer just about capturing — it is about re-weaving data. We hope this tool empowers every photographer to precisely control each 'data thread', ultimately weaving their own tapestry of stars."

## Stargazers

[![Stargazers over time](https://starchart.cc/S-T-A-R-Laboratory/HoshinoWeaver.svg?variant=adaptive)](https://starchart.cc/S-T-A-R-Laboratory/HoshinoWeaver)
