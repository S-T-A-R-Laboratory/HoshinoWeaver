# Benchmark Suite

按职责拆成三层：

- `bench/cpu/`
  CPU kernel 与对齐阶段 benchmark
- `bench/gpu/`
  GPU 原型、CPU 基线与 CPU↔GPU compare benchmark
- `bench/data_tools/`
  raw cache、测试图片、synthetic starfield 数据生成工具

共享基础设施保留在：

- `bench/common.py`
  公共加载、计时、JSON 输出与输入选择逻辑
- `bench/data/`
  benchmark 输入数据目录

## 当前主入口

### CPU

- `python -m bench.cpu.kernels`
  算法内核微基准。默认直接对比 custom-op 相关 kernel 的 `numpy / compiled`，覆盖 `max / mean / FastGaussianParam / masked FGP / sigma clip fused / huber / median`。
- `python -m bench.cpu.max_stack`
  大尺寸 `max` 专项 benchmark。比较单进程 `NumPy in-place stream`、多进程 `NumPy local-reduce`、`custom op OpenMP stream`。
- `python -m bench.cpu.fgp_accumulate`
  大尺寸 `FastGaussianParam` 统计累加专项 benchmark。比较 Python、NumPy、`custom op OpenMP` 路径。
- `python -m bench.cpu.mean_stack`
  `fgp_accumulate` 的兼容 shim。
- `python -m bench.cpu.alignment`
  对齐 pipeline 分阶段 benchmark。当前覆盖 synthetic starfield / image-dir 输入下的 `detect / features / geometry / matching / warp / homography pipeline / optimization / remap / camera-model pipeline`。

### GPU

- `python -m bench.gpu.original_remap`
  当前 camera-model remap 口径集合。覆盖 `NumPy grid`、fused `camera_model_remap` custom-op、`cv2.remap` 与原主线路径。
- `python -m bench.gpu.triton_remap`
  camera-model remap 的 GPU 原型。当前覆盖零畸变 `grid generation + sampling`，其中 Triton 负责 grid，采样先走 `torch.grid_sample`。
- `python -m bench.gpu.compare_remap`
  统一对比 `original_remap` 与 `triton_remap`，直接输出 CPU→GPU 总时间差距，并保留 grid / sampling 分项；Triton 仅作热点验证，不作为正式运行时依赖。
- `python -m bench.gpu.original_homography`
  纯 homography warp 的 CPU 基线。只测 `cv2.warpPerspective`，不含 detect / features / match。
- `python -m bench.gpu.triton_homography`
  纯 homography warp 的 GPU 原型。当前覆盖 homography grid 生成与采样，Triton 负责 grid，采样先走 `torch.grid_sample`。
- `python -m bench.gpu.compare_homography`
  统一对比 `original_homography` 与 `triton_homography`，用于判断无相机参数路径里纯 warp kernel 的 GPU 价值。

### 数据工具

- `python -m bench.data_tools.generate_array_cache`
  生成 raw cache，供 benchmark 直接加载，避免重复图片解码。
- `python -m bench.data_tools.generate_dataset`
  生成图片目录输入，主要用于 smoke test 和输入扫描链路验证。
- `python -m bench.data_tools.generate_starfield_dataset`
  生成合成星点图数据集，供对齐 benchmark 或检测 smoke test 使用。
- `bench.data_tools.starfield`
  合成星点图公共 helper。生成带平移/轻旋转/噪声的 synthetic starfield 序列。

## 安装建议

benchmark 的最小依赖放在 [bench/requirements.txt](./requirements.txt)。

当前最小运行依赖只有：

- `numpy`
- `loguru`
- `opencv-python`

如果要本地跑测试或编译 `hoshicore._custom_op._C`，直接安装仓库根目录的 [requirements.txt](../requirements.txt) 即可。

如果要比较 GPU 原型，需自行准备带 `torch + triton` 的 Python 环境，并在 compare 脚本里通过 `--gpu-python` 指定对应解释器。

## 输入数据

benchmark 默认 `--input-mode auto`，会优先使用 raw cache。

扫描顺序：

1. `--input-dir`
2. `bench/data/cache/`
3. `bench/data/input/`
4. `bench/data/generated/`
5. 合成随机数据

输入模式说明：

- `auto`
  先扫 `raw_cache`，再扫 `images`，最后回退到 `synthetic`。
- `cache`
  推荐的默认性能输入。适合重复 benchmark，不需要图片解码。
- `images`
  保留给 smoke test 或输入链路检查，建议只保留小规模图片集。
- `synthetic`
  无本地数据时的兜底输入。

如果你希望“命令行写的小尺寸 synthetic 就一定测 synthetic”，显式传：

```bash
--input-mode synthetic
```

生成 raw cache：

```bash
python -m bench.data_tools.generate_array_cache --name max_u8_100x24mp_cache --frames 100 --height 4000 --width 6000 --dtype uint8
python -m bench.data_tools.generate_array_cache --name max_u8_1000x24mp_from_images --input-dir bench/data/generated/max_u8_100x24mp_jpg --frames 1000
```

生成图片目录：

```bash
python -m bench.data_tools.generate_dataset --name max_u8_100x24mp_jpg --frames 100 --height 4000 --width 6000 --dtype uint8 --format jpg
```

生成对齐用合成星点图：

```bash
python -m bench.data_tools.generate_starfield_dataset --name align_u16_32f --frames 32 --height 2048 --width 3072 --dtype uint16 --stars 1200 --format tif
```

## 运行建议

建议按下面顺序使用：

1. `python -m bench.cpu.kernels`
   先看当前 stack kernel 热点方向，也包含 `fgp_masked_mean_merge`、`sigma_clip_fused_*`、`fgp_add_partial_reduce` 的独立 `numpy / compiled` microbenchmark。
2. `python -m bench.cpu.max_stack` / `python -m bench.cpu.fgp_accumulate`
   跟进 `max / fgp_accumulate` 的 CPU kernel 优化。
3. `python -m bench.cpu.alignment`
   先确认对齐链阶段热点，避免直接 GPU 化错对象。
4. `python -m bench.gpu.compare_remap`
   对比 camera-model remap 的 CPU 原方案和 Triton GPU 原型。
5. `python -m bench.gpu.compare_homography`
   单独判断 homography warp kernel 的 GPU 价值。

当前对齐 GPU 试验要注意两点：

- `compare_remap` 的口径是“当前 CPU 原方案总时间”对比 “Triton GPU 原型总时间”。
- `compare_homography` 只覆盖纯 warp kernel，不代表整条无相机参数对齐路径。
- 当前正式保留的 GPU custom-op 只有 fused `camera_model_remap`；`homography` custom-op 试验已撤回。
- `median_reduce_chunk_baseline` 对齐当前 `MedianReduceOp` 主线的 chunk 处理：逐块分配 `float32` stack、逐帧拷贝切片、再执行 `np.median(axis=0)`。

示例：

```bash
python -m bench.cpu.kernels --frames 128 --height 1080 --width 1920 --dtype uint16 --input-mode synthetic
python -m bench.cpu.kernels --frames 64 --height 2048 --width 3072 --dtype uint16 --input-mode synthetic --cases fgp_masked_mean_merge_stream_numpy,fgp_masked_mean_merge_stream_compiled,sigma_clip_fused_merge_stream_numpy,sigma_clip_fused_merge_stream_compiled,sigma_clip_fused_masked_merge_stream_numpy,sigma_clip_fused_masked_merge_stream_compiled,fgp_add_partial_reduce_numpy,fgp_add_partial_reduce_compiled
python -m bench.cpu.kernels --frames 16 --height 2048 --width 3072 --dtype uint16 --input-mode synthetic --cases median_reduce_chunk_baseline --chunk-rows 32
python -m bench.cpu.max_stack --frames 100 --height 4000 --width 6000 --dtype uint8 --workers 4 --openmp-threads auto --input-mode cache
python -m bench.cpu.fgp_accumulate --frames 100 --height 4000 --width 6000 --dtype uint8 --openmp-threads auto --input-mode cache
python -m bench.cpu.max_stack --frames 1000 --input-dir bench/data/cache/max_u8_100x24mp_cache --output-json bench-results/max-1000.json
python -m bench.cpu.alignment --frames 16 --height 2048 --width 3072 --stars 1200 --input-mode synthetic
python -m bench.cpu.alignment --frames 16 --input-dir bench/data/generated/align_u16_32f --input-mode images
python -m bench.cpu.alignment --frames 16 --input-dir bench/data/generated/align_u16_32f --input-mode images --cases detect_stream match_stream
python -m bench.cpu.alignment --frames 16 --input-dir bench/data/generated/align_u16_32f --input-mode images --cases detect_wavelet_stream detect_extract_stream remap_stream
python -m bench.gpu.compare_remap --height 2048 --width 3072 --warmup 10 --repeat 30
python -m bench.gpu.compare_homography --height 2048 --width 3072 --warmup 10 --repeat 30
```

## 输出格式

benchmark 默认在终端打印简短摘要，主要展示各 case 的运行时间。

如果传入 `--output-json /path/to/report.json`，会写出完整 JSON 文件。

输出字段主要包括：

- `suite`
- `env`
- `custom_ops`
- `config`
- `input_source`
- `results`

`results` 中每个 case 通常包含：

- `samples_sec`
- `min_sec`
- `mean_sec`
- `median_sec`
- `max_sec`
