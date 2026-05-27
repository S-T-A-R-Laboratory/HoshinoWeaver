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
    sigma_clip/
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

## 调用方式

已激活目标环境时，直接运行：

```bash
python csrc/build_ops.py
```

`build_ops.py` 会把当前 `sys.executable` 传给 CMake 的 `Python3_EXECUTABLE`。
需要显式指定解释器时，直接用解释器路径调用脚本即可。

## 常用命令

默认构建：

```bash
python csrc/build_ops.py
```

显式系统 GCC：

```bash
python csrc/build_ops.py --cc /usr/bin/gcc --cxx /usr/bin/g++
```

CUDA 构建：

```bash
python csrc/build_ops.py --cuda --cc /usr/bin/gcc --cxx /usr/bin/g++
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
  指定 CMake preset；日常路径通常不需要手动传
- `--cc / --cxx`
  显式指定编译器
- `--cuda`
  打开 CUDA 构建，自动根据平台选择对应 CUDA preset
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
- CUDA custom-op 当前为 fused `camera_model_remap`

## 打包约定

最终发布为 PyInstaller single-folder 模式。CUDA runtime 静态链接到 `_C`；
OpenMP 在 Linux/Windows 为动态链接（PyInstaller 自动收集），macOS 为静态链接。

### 链接策略

| 依赖 | 链接方式 | 说明 |
|------|----------|------|
| OpenMP (Linux + GCC) | 动态（`libgomp.so`） | 系统自带，PyInstaller 自动收集到产物目录 |
| OpenMP (Windows + MSVC) | 动态（`vcomp140.dll`） | VC++ Redistributable 组件，PyInstaller 自动收集 |
| OpenMP (macOS) | 静态（Homebrew `libomp.a`） | 需先 `brew install libomp`，编译时自动检测并静态链接 |
| CUDA runtime | 静态（`cudart_static`） | 消除 `libcudart.so` / `cudart64_*.dll` 依赖 |

### 验证依赖

```bash
# Linux — 确认 cudart 已静态链接，libgomp 为动态（PyInstaller 会收集）
ldd hoshicore/_custom_op/_C*.so | grep -E "cudart|gomp"
# 预期：只看到 libgomp.so，不应出现 libcudart.so

# Windows (Developer Command Prompt)
dumpbin /dependents hoshicore/_custom_op/_C*.pyd
# 预期：出现 VCOMP140.DLL（正常），不应出现 cudart64_*.dll
```

### PyInstaller 收集

OpenMP 动态库由 PyInstaller 自动收集。spec file 确保 `_C` 模块被包含即可：

```python
# PyInstaller spec — hiddenimports 确保 _C 被打包
hiddenimports=['hoshicore._custom_op._C']
```

若后续使用了 cuBLAS/cuFFT 等额外 CUDA 库且无法静态链接，再按需添加到
`binaries=[]` 中。

## GPU 构建说明

### CUDA

构建 CUDA 算子需要安装 CUDA Toolkit，版本选择支持本机 GPU 的即可。
最终发布建议 12.8+，可覆盖所有架构的 NVIDIA GPU。

构建路径：

- Linux: `python csrc/build_ops.py --cuda`
- Windows: 使用 `windows-msvc-cuda` preset

Preset 参考：

| Preset | 平台 |
|--------|------|
| `linux-gcc-cuda` | Linux |
| `windows-msvc-cuda` | Windows |

### 其他 GPU 后端（规划中）

| 方向 | 状态 |
|------|------|
| AMD (ROCm/HIP) | 待评估 |
| macOS (Metal/MPS) | 待设计 |
| Vulkan (compute shader) | 待评估 |

所有 GPU 后端保持 CPU fallback 语义不变。

---

普通用户只需安装 NVIDIA 驱动（>= 570.65，对应发布构建的 CUDA 12.8），不需要 CUDA Toolkit。
驱动版本要求与 GPU 型号无关，只要驱动足够新即可；驱动过旧或无 NVIDIA GPU 时自动回退 CPU。

## 新增算子

最小流程：

1. 在 `csrc/ops/<name>/` 新增 `.h/.cpp`
2. 在 `CMakeLists.txt` 新增 static library target 并链接到 `_C`
3. 在 `module.cpp` 中注册 `bind_*_ops(m)`
4. 在 `hoshicore/_custom_op/ops/` 增加 Python 包装与 numpy fallback
5. 在 `hoshicore/_custom_op/api.py` + `__init__.py` 导出
6. 补 focused tests（`tests/test_custom_ops.py`）
7. 补 microbenchmark（`bench/cpu/kernels.py`）

CUDA 算子沿用同样流程，但额外需要：

1. 在 `CMakeLists.txt` 的 `HNW_ENABLE_CUDA` 分支里接入 `.cu`/binding 源文件
2. 保持 CPU fallback 语义不变
3. `.cpp` 绑定文件需 `#include "common/compat.h"`（MSVC `ssize_t` 兼容）

## 参考

- 构建架构与 preset 设计详见 [CMAKE_MIGRATION.md](./CMAKE_MIGRATION.md)
