# CMake 构建架构

本文档描述 `csrc/` 的 CMake 构建设计，包括目标结构、preset 策略和多平台规划。

## 设计边界

构建系统只覆盖原生层：

- `csrc/` 下的 C/C++/CUDA 源码
- `hoshicore._custom_op._C` 模块的编译
- 工具链、OpenMP、preset、构建脚本

不涉及：

- Python API 层（由 `hoshicore/_custom_op/` 管理）
- 运行时 fallback 逻辑（由 Python facade 负责）
- PyInstaller 打包链（由 `make_package.py` 负责）

## 目录结构

```text
csrc/
  CMakeLists.txt
  CMakePresets.json
  build_ops.py
  cmake/
    HnwOptions.cmake
    HnwCompiler.cmake
    HnwOpenMP.cmake
    HnwPython.cmake
    HnwCuda.cmake
  module.cpp
  common/
  ops/
    fgp/
    max/
    median/
    noise/
    cuda/
```

## Target 设计

单一 Python 模块 + 多个内部静态库：

```text
_C (pybind11 module)
  ├─ hnw_ops_common
  ├─ hnw_ops_cpu_fgp
  ├─ hnw_ops_cpu_max
  ├─ hnw_ops_cpu_median
  ├─ hnw_ops_cpu_noise
  └─ hnw_ops_cuda       (CUDA 开启时)
```

约束：

- `_C` 模块名固定
- CPU/CUDA 编译边界清晰
- CUDA 关闭时 CPU-only 不受影响

## Preset

主 preset：

| Preset | 用途 |
|--------|------|
| `linux-gcc` | Linux 默认 |
| `linux-gcc-cuda` | Linux + CUDA |
| `linux-clang` | Clang 路径 |
| `macos-clang` | macOS |
| `windows-msvc` | Windows CPU-only |
| `windows-msvc-cuda` | Windows + CUDA |
| `linux-gcc-debug` | 调试 / smoke test |

原则：

- 默认优化配置使用 `RelWithDebInfo`
- debug 只作为诊断路径
- `binaryDir` 固定到 `csrc/build/<preset>/`

## build_ops.py

统一开发入口，不要求用户直接记 CMake 命令。职责：

- 选择 preset
- 探测或接收编译器路径
- 传递 `Python3_EXECUTABLE`
- 调用 `cmake --preset ...` + `cmake --build --preset ...`

## 多平台 CUDA 策略

| 平台 | 角色 | 说明 |
|------|------|------|
| Linux | 开发与验证 | 验证基线 CUDA 12.4 |
| Windows | 主要发布平台 | 发布目标 CUDA 12.8+，最低 driver 570.65+ |
| macOS | CPU fallback | GPU 路线走 Metal/MPS，不纳入 CUDA |

Windows 用户只需 NVIDIA driver，不需要 CUDA Toolkit。

## 后续方向

- macOS：评估 Metal/MPS 构建骨架，设计与 `_C` 并列的 GPU backend
- 打包链：待 custom-op 构建稳定后，评估是否接 `scikit-build-core`
- GPU 扩展：优先考虑大粒度数据链 / GPU-resident 路径，而非逐个小算子移植

## 验证

```bash
# 确认 _C 可导入
python -c "import hoshicore._custom_op._C as m; print(m.build_info())"

# 运行测试
python -m pytest tests/ -v

# 快速 benchmark
python -m bench.cpu.kernels --cases max_combine_stream_numpy,max_combine_stream_compiled
```
