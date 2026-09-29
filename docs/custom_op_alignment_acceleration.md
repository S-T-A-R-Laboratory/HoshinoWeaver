# Custom-op 对齐加速记录

初始记录：2026-09-28。本批按六个功能/文档提交收尾；不将历史性能预测当作验收结果。

## 2026-09-29：Metal median 检测

在独立 `feature/macos-test` 分支验证后，已将单通道 uint16、`13×13` 的精确 Metal
背景中位数接入 Norma median 检测。CPU 后处理沿用原有的 response、float64
统计、阈值和形态学实现；其它 dtype/核大小继续走 OpenMP/NumPy。Metal runtime
或资源不可用时沿现有 typed fallback 回 CPU；CPU 后处理入口缺失时在分发前
改走 CPU/NumPy。GPU 两块 uint16 buffer 的 ledger 峰值为每像素 4 字节。

[macOS 实验 CI run 36512799638](https://github.com/S-T-A-R-Laboratory/HoshinoWeaver/actions/runs/36512799638)
8/8 通过。3 vCPU `Apple Paravirtual device` 上，合成 26MP 星场的完整检测
中位数 CPU 2220.3 ms → Metal+CPU 1567.8 ms（1.42×）；1.3MP 为
99.9 → 67.0 ms（1.49×）。五次采样的最终星点数组逐位一致，Metal median
背景也与 CPU 逐位一致；这组虚拟设备计时不外推实体 Mac。

本分支移植后的跨平台 CI 仍以对应推送结果为准。实验计时脚本和 branch-only
CI step 留在 `feature/macos-test`，常规 CI 只保留正确性与打包门禁。

## 已实现

| 工作 | 当前实现 |
|---|---|
| pywt 像素检测与 features | CPU/CUDA kernel、分发与资源回退；检测精确百分位选择和 staging 复用 |
| asterism 互为最近邻 | OpenMP/CUDA 网格；最近距离并列或不可网格化时回退 SciPy |
| asterism token 与投票 | OpenMP token 网格、按 anchor 分桶投票；log 保留 NumPy，邻距并列回退参考实现 |
| 方案 B：CUDA 原图转灰度 | `GraySource`、按 OpenCV 配置缓存兼容性自检、失败或不兼容时使用主机灰度 |
| median 去掉 float64 中间图 | `to_median_gray_u16` 保持原量化结果，其他 dtype/超范围仍走原路径 |
| 第 3 项：CPU 灰度 | `detection_gray_f64/u16`，OpenMP 并行转换、归一化和取整；颜色计算保留 OpenCV |

CPU 灰度只加速 uint8/uint16 的二维灰度和 BGR/BGRA 常见路径。uint16 单通道
median 图直接复用；float 输入沿用原来的最大值归一化语义。算子不可用或设置
`HNW_CUSTOM_OPS_FALLBACK=numpy` 时走参考实现，CPU native 真实错误传播。
OpenMP 使用已有线程配置、转换循环最多 8 线程；没有改全局线程策略。

## 本轮修复与数值约束

- asterism 网格 cell 下溢为零时转全扫描，避免无限扩格；有子进程超时回归测试。
- `asterism_match_ops.cpp` 在 GCC/Clang/Apple Clang 下局部使用
  `-ffp-contract=off`。分开写语句不足以禁止 GCC 合并 FMA；未全局修改编译选项。
- CUDA features 的近等 `vol*rho` 排序回退 CPU，避免设备 acos 舍入改变方向基准。
- OpenCV 灰度自检缓存包含 IPP/optimization 配置；真实 CUDA 错误不被吞掉。
- CPU 灰度直接调用当前 OpenCV 颜色转换，因此不依赖 GPU 路径的 IPP 算术反推。

## 验证与性能口径

Linux CUDA/OpenMP 扩展重编译通过。`auto` / `numpy` / `cpu` 三种模式全量
各 **860 passed / 363 subtests passed**（新增 22 项）；16 条既有 PyWavelets
边界层数 warning。clang-format 20.1.8 检查 **115 文件通过**，`git diff --check`
通过。Windows/macOS 尚未执行本批构建与运行验证。

本轮回归覆盖 IPP 开/关、uint8/uint16、单通道/BGR/BGRA、非连续输入、行尾宽度、
最近偶数取整、超范围/NaN、NumPy 回退、native 错误传播及线程数一致性。
真实 `_MG_0033.tif` / `_MG_0043.tif`（4160×6240 uint16 BGR）：两种灰度
输出均逐位一致；完整 median 检测的 positions/volumes/intensities 也逐位一致，
星点数分别为 22218 / 21863。

Linux GCC 11、8 线程，同一 `_MG_0033.tif`，每后端独立进程、warmup=1、
repeat=3 的中位数（共享机器的一轮测量，不外推 macOS 或完整检测加速比）：

| 灰度输出 | NumPy 前后转换 + OpenCV | OpenMP 前后转换 + OpenCV |
|---|---:|---:|
| normalized float64 | 582.8 ms | 163.5 ms |
| median uint16 | 504.1 ms | 218.5 ms |

复现时按单个 case 分别运行四次：

```bash
HNW_CUSTOM_OPS_THREADS=8 python -m bench.cpu.kernels \
  --input-mode images --input-dir image --frames 1 --dtype uint16 \
  --cases detection_gray_f64_openmp --warmup 1 --repeat 3
```

成对灰度 benchmark 在 `bench.cpu.kernels`：

- `detection_gray_f64_numpy` / `detection_gray_f64_openmp`
- `detection_gray_u16_numpy` / `detection_gray_u16_openmp`（uint16 输入）

每个 backend 使用独立进程，保持输入、线程数和 warmup/repeat 一致。
灰度转换耗时不等于完整检测耗时；共享机器上不依据小幅计时差异作收益结论。

## 剩余事项

1. Metal median 移植后，以主功能分支的 Windows/MSVC、macOS/Apple Clang CI
   结果完成最后验证；实验分支 8/8 绿不能替代目标分支的门禁。
2. 面向 `core-dev` 的最终集成审查；合并由 maintainer 决定。
3. 轮廓测量仅作为待剖析候选；没有批准以 connected-components 改变 contour 几何语义。

全局关闭 FMA、手写 SIMD、GEMM 匹配、Graph IR/ExecutionPlan、MPS 与其它
Metal 移植尚未立项。早期路线图和 `time_cost.md` 只作历史参考。
