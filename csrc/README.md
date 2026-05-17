# C++ Custom Ops

`csrc/` 负责 `hoshicore._custom_op._C` 的原生实现与本地构建。

设计边界：

- 只覆盖 custom-op 原生层
- Python 公共入口保持为 `hoshicore._custom_op`
- 运行时按 `compiled -> numpy` fallback
- 构建统一走 `CMake + Ninja`

## 目录

```text
csrc/
  build_ops.py
  CMakePresets.json
  CMakeLists.txt
  module.cpp
  common/
  ops/
    fgp/
    max/
    median/
    noise/
    cuda/
```

职责：

- `module.cpp`
  pybind11 模块入口，注册 `_C` 内的算子
- `ops/<name>/`
  单个算子的 C++/CUDA 实现与绑定
- `build_ops.py`
  统一本地构建入口
- `CMakeLists.txt` / `CMakePresets.json`
  custom-op 的 CMake/Ninja 构建骨架

## 构建环境

不要求必须使用 conda。只要当前解释器环境里具备以下组件即可：

- Python 3.10+
- `pybind11`
- Python development headers
- `cmake`
- `ninja`
- 可用的 C/C++ 编译器

如果当前环境缺少 CMake/Ninja，可额外安装：

```bash
pip install -r csrc/requirements.txt
```

这个文件只补 native 构建工具，不重复根目录 `requirements.txt` 里的项目依赖；
编译器和 OpenMP runtime 仍然是系统工具链要求。

推荐口径：

- 已激活目标环境时，直接运行 `python csrc/build_ops.py ...`
- 需要显式指定解释器时，直接用解释器路径调用脚本
- `build_ops.py` 会把当前 `sys.executable` 传给 `Python3_EXECUTABLE`

## 常用命令

默认稳定路径：

```bash
python csrc/build_ops.py
```

显式系统 GCC：

```bash
python csrc/build_ops.py --cc /usr/bin/gcc --cxx /usr/bin/g++
```

CUDA 路径：

```bash
python csrc/build_ops.py --cuda --cc /usr/bin/gcc --cxx /usr/bin/g++
```

debug smoke path：

```bash
python csrc/build_ops.py --preset linux-gcc-debug --cc /usr/bin/gcc --cxx /usr/bin/g++
```

显式解释器：

```bash
/path/to/python csrc/build_ops.py --cc /usr/bin/gcc --cxx /usr/bin/g++
```

只看配置：

```bash
python csrc/build_ops.py --dry-run
```

## 常用参数

- `--preset`
  指定 CMake preset；日常 Linux 路径通常不需要手动传
- `--cc / --cxx`
  显式指定编译器
- `--cuda`
  打开 CUDA preset/flags；Linux 下默认收敛到 `linux-gcc-cuda`
- `--compiler gcc|clang|msvc|auto`
  选择编译器家族
- `--no-openmp`
  关闭 OpenMP
- `--march-native`
  启用本机 CPU 指令集优化；只建议本机 benchmark 使用
- `--lto`
  启用 LTO
- `--omp-simd`
  为支持的 kernel 启用显式 OpenMP SIMD pragma
- `--clean`
  清理旧产物后全量重编
- `--verbose-build`
  打印完整 backend 输出

## 输出与中间产物

- 扩展模块输出到 `hoshicore/_custom_op/_C*.so|.pyd`
- `cmake` 中间产物默认在 `csrc/build/<preset>/`
- 唯一的 CUDA custom-op 为 fused `camera_model_remap`

## 打包约定

最终发布为 PyInstaller single-folder 模式。OpenMP 和 CUDA runtime 均**静态链接**到 `_C`
扩展模块内，打包时只需收集 `_C.so`（Linux）/ `_C.pyd`（Windows）本身，
无需额外携带 `libgomp.so`、`vcomp140.dll` 或 `cudart64_*.dll`。

### 静态链接策略

| 依赖 | 编译方式 | 说明 |
|------|----------|------|
| OpenMP (Linux + GCC) | `-static-libgomp` 或 link `libgomp.a` | 线程池实现烤进 `_C.so` |
| OpenMP (Windows + MSVC) | `/openmp` + 静态 CRT（`/MT`）| `vcomp140.dll` 依赖消除；若用 LLVM OpenMP 则嵌入 libomp |
| OpenMP (macOS) | 不启用 | macOS 包不启用 OpenMP |
| CUDA runtime | `CUDA_USE_STATIC_CUDA_RUNTIME=ON`，link `cudart_static` | NVIDIA 官方支持，消除 `libcudart.so` / `cudart64_*.dll` 依赖 |

### 无法静态链接的部分

- **NVIDIA driver**（`libcuda.so` / `nvcuda.dll`）：属于内核态接口，必须由用户机器提供
- 用户只需安装与本机 GPU 匹配的 NVIDIA 驱动，不需要 CUDA Toolkit

### 验证依赖是否干净

```bash
# Linux — 确认不再依赖 libgomp / libcudart
ldd hoshicore/_custom_op/_C*.so | grep -E “cudart|gomp”
# 预期：无输出

# Windows (Developer Command Prompt)
dumpbin /dependents hoshicore/_custom_op/_C*.pyd
# 预期：不出现 vcomp140.dll / cudart64_*.dll
```

### PyInstaller 收集

静态链接后，`_C` 无额外运行时库依赖。spec file 只需确保 `_C` 模块本身被收集：

```python
# PyInstaller spec — hiddenimports 确保 _C 被打包
hiddenimports=['hoshicore._custom_op._C']
```

若后续使用了 cuBLAS/cuFFT 等额外 CUDA 库且无法静态链接，再按需添加到
`binaries=[]` 中。

## CUDA 平台安排

CUDA 按”Linux 开发态 / Windows 发布态”拆分：

- Linux 作为开发与验证环境，已验证基线为 `CUDA 12.4`
- Windows 作为主要发布平台，正式 CUDA 发布收敛到 `CUDA 12.8+`
- macOS GPU 路线走 `Metal/MPS`，保留 CPU fallback

Windows 发布要求：

- `Windows 10/11 x64`
- `NVIDIA GPU with CUDA support`
- `NVIDIA Driver >= 570.65`（若发布构建使用 `CUDA 12.8 GA`）
- `CUDA Toolkit is NOT required`

Windows 构建与验证 TODO：

- 已提供 `windows-msvc-cuda` preset，作为 Windows CUDA 主构建路径
- 保留 `windows-msvc` 作为 CPU-only 基础路径
- 后续在 Windows 上验证 `_C` 导入、`build_info()`、`camera_model_remap` 对拍、CPU fallback、`numpy_grid/custom_op_fused` benchmark，以及 `.pyd` 依赖收集
- Windows 默认使用 `CMake + Ninja + MSVC`；不计划以 MinGW 作为正式支持路径
- 若在 Windows 上用 VSCode 进行构建/打包，优先使用 CMake Tools + Ninja，并从可用的 MSVC/Developer PowerShell 环境启动

说明：

- 终端用户只需要安装或升级 NVIDIA driver，不需要单独安装 CUDA Toolkit
- 若驱动过旧、没有可用 NVIDIA GPU，或 CUDA 初始化失败，运行时应提示并自动回退 CPU
- `cudaDriverGetVersion()` 用于判断驱动支持的 CUDA API 版本是否满足要求；若需要显示实际 NVIDIA driver 字符串，应额外走 NVML 或等效平台接口
- 后续建议把 CUDA 环境检查结果接入 UI，展示 GPU 型号、driver 状态、CUDA 可用性、是否已回退 CPU，以及 NVIDIA 官方驱动下载入口

macOS TODO：

- 设计 `Metal/MPS` 对应的 GPU custom-op 构建路径
- 明确 `_C` 与 macOS GPU backend 的二进制边界
- 保持与 Linux/Windows 一致的 fallback 语义

## 新增算子

最小流程：

1. 在 `csrc/ops/<name>/` 新增 `.h/.cpp`
2. 在 `module.cpp` 中注册 `bind_*_ops(m)`
3. 在 `hoshicore/_custom_op/api.py` 增加 Python 包装与 fallback
4. 接回主线调用点
5. 补 focused tests
6. 视需要补 microbenchmark

CUDA 算子沿用同样流程，但额外需要：

1. 在 `CMakeLists.txt` 的 `HNW_ENABLE_CUDA` 分支里接入 `.cu`/binding 源文件
2. 保持 CPU fallback 语义不变
3. 先补 focused correctness，再补最窄 benchmark

## 说明

- 上层统一通过 `hoshicore._custom_op` 调用，不直接依赖 `_C`
- 如需了解迁移设计细节，见 [CMAKE_MIGRATION.md](./CMAKE_MIGRATION.md)
