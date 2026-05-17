# CMake Migration Blueprint

目标：用 `CMake + Ninja` 统一 custom-op 原生构建，为后续 CUDA、多平台、多工具链做准备，同时不改 Python 侧 `_C` 入口和 fallback 语义。

## 当前边界

本次迁移只覆盖原生构建层：

- `csrc/` 下的 C/C++/CUDA 源码组织
- `hoshicore._custom_op._C` 的构建
- 工具链、OpenMP、preset、构建脚本

本次不做：

- Python API 改名
- runtime / graph compile 改造
- 整个项目打包链切换
- 更大粒度的 GPU pipeline / runtime 改造

当前打包边界预设：

- `_C` 依赖的 OpenMP/CUDA 用户态 runtime 需要随包一起收集
- GPU driver 不随包分发，继续要求用户自行安装
- Linux 当前只作为 CUDA 开发与验证基线，Windows 作为后续主要 CUDA 发布平台
- macOS 的 GPU 支持后续改走 `Metal/MPS`，不纳入当前 CUDA 迁移范围

## Phase 1 状态

已落地：

- `csrc/CMakeLists.txt`
- `csrc/CMakePresets.json`
- `python csrc/build_ops.py ...`

当前形态：

- 单一 pybind11 模块 `_C`
- CPU/CUDA 共享同一 `_C` 入口
- 输出位置保持在 `hoshicore/_custom_op/`

## 目标结构

```text
csrc/
  CMakeLists.txt
  CMakePresets.json
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
    cuda/
```

## Target 设计

保持“单一 Python 模块 + 多个内部库”：

```text
_C
  ├─ hnw_ops_common
  ├─ hnw_ops_cpu_fgp
  ├─ hnw_ops_cpu_max
  └─ hnw_ops_cuda   # future
```

约束：

- `_C` 模块名不变
- CPU/CUDA 编译边界清楚
- CUDA 关闭时 CPU-only 不受影响

## Preset 口径

当前主 preset：

- `linux-gcc`
- `linux-gcc-cuda`
- `linux-clang`
- `macos-clang`
- `windows-msvc`
- `windows-msvc-cuda`

诊断 preset：

- `linux-gcc-debug`

原则：

- 默认优化配置使用 `RelWithDebInfo`
- debug 只作为 smoke/诊断路径
- `binaryDir` 固定到 `csrc/build/<preset>/`

## build_ops.py 角色

`build_ops.py` 保留为统一开发入口，不要求用户直接记 CMake 命令。

它负责：

- 选择 preset
- 探测或接收编译器路径
- 把当前 `sys.executable` 显式传给 `Python3_EXECUTABLE`
- 调用 `cmake --preset ...` / `cmake --build --preset ...`

## 后续阶段

### Phase 2

已完成：

- 增加 `HNW_ENABLE_CUDA`
- 接入最小 CUDA target
- 落地 fused `camera_model_remap` CUDA custom-op
- 保证 CUDA OFF 时 CPU-only 仍可单独构建

CUDA 平台策略：

- Linux：继续作为开发态，当前验证基线可先保持在 `CUDA 12.4`
- Windows：正式发布态目标收敛到 `CUDA 12.8+`
- Windows 用户要求以 NVIDIA driver 为准，不要求安装 CUDA Toolkit
- 若发布使用 `CUDA 12.8 GA`，用户侧最低 driver 要求为 `570.65+`
- 运行时需要做 CUDA 可用性检查；不满足条件时显式提示并回退 CPU
- Windows 后续需要验证 `.pyd + cudart + OpenMP runtime` 的打包收集
- Windows 当前推荐构建链为 `CMake + Ninja + MSVC`；VSCode 可以作为前端，但不改变底层工具链要求
- CUDA 环境检查结果后续建议接入 UI，用于展示 GPU/driver 状态、当前加速是否可用，以及驱动下载指引

macOS TODO：

- 评估 `Metal/MPS` 对应的原生构建骨架
- 设计与 `_C` 共享或并列的 macOS GPU backend 入口
- 明确打包时的动态库/框架收集策略

### Phase 3

- 在三平台路径稳定后，再评估是否接 `scikit-build-core`
- 若后续继续做 GPU，不再优先扩新的 host-in/host-out 小算子，而应转向更大粒度的数据链、batch 或 GPU-resident 路径

## 验证口径

每阶段至少回答：

- `_C` 是否可导入
- focused tests 是否通过
- CPU-only 构建是否稳定
- CUDA 关闭时是否仍可构建

建议命令：

```bash
python -c "import hoshicore._custom_op._C as m; print(m.build_info())"
python -m unittest discover -s test -p 'test_custom_ops.py'
python -m bench.cpu.kernels --cases max_combine_stream_numpy,max_combine_stream_compiled
```

## 当前结论

对当前项目来说，先把算子层迁到 `CMake + Ninja` 是合理的；  
把整个项目打包链一起迁移，则应后置到 custom-op 构建稳定之后。
