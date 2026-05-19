# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**HoshinoWeaver (织此星辰)** is an astrophotography image preprocessing tool built around a DAG (Directed Acyclic Graph) operator engine. Users define image processing workflows via YAML, and the engine executes them as async streaming pipelines. Supports star trail stacking, sky/ground separation alignment, noise reduction stacking, and more.

## Commands

```bash
# Run GUI
python "HoshinoWeaver desktop.py"

# Run CLI pipeline
python launcher.py <config.yaml> [image_dir] [--route KEY=VALUE] [--input KEY=VALUE] [--config KEY=VALUE]

# Inspect a pipeline's parameter schema
python launcher.py <config.yaml> --inspect

# Run tests
pytest tests/ -v --tb=short --cov=hoshicore --cov-report=term-missing -x

# Run a single test
pytest tests/test_yaml_loader.py -v

# Build C++ custom ops
python csrc/build_ops.py                       # auto-detect compiler
python csrc/build_ops.py --cuda                # with CUDA support
python csrc/build_ops.py --dry-run             # inspect config only

# Run benchmarks
python -m bench.cpu.kernels --frames 64 --height 2048 --width 3072 --dtype uint16 --input-mode synthetic
python -m bench.cpu.max_stack --frames 100 --height 4000 --width 6000 --dtype uint8 --input-mode cache
python -m bench.cpu.alignment --frames 16 --height 2048 --width 3072 --input-mode synthetic

# Package for distribution (PyInstaller)
python make_package.py                         # auto-build _C if missing, fail if build fails
python make_package.py --no-build              # skip build attempt, fail if _C absent
python make_package.py --allow-numpy-only      # allow packaging without _C (dev/debug only)
```

## Architecture

### Processing Pipeline

```
Meta YAML ── meta_resolve() ──► Standard spec ── flatten_sub_dags() ──► Flat spec ── validate_and_build_order() ──► ValidatedDag ── wiring ──► DAGExecutor → results
```

### Engine Layer (`hoshicore/engine/`)

| Module | Role |
|--------|------|
| `meta.py` | Compiles Meta YAML (routes, enabled flags) into standard DAG spec |
| `flatten.py` | Recursively expands `.yaml` SubDAG references into namespaced flat nodes |
| `build.py` | Validates DAG spec, builds dependency graph (networkx), produces topological order |
| `wiring.py` | Instantiates Ops, connects async queues, creates feeder coroutines |
| `executor.py` | Runs all nodes concurrently with global cancellation propagation |
| `multiprocess.py` | Data-parallel execution: segments pipeline into workers for N-process parallelism |
| `registry.py` | `@register_op()` decorator → `REGISTERED_OP` dict mapping names to Op classes |
| `segment_detect.py` | Identifies parallelizable pipeline segments (I/O + Map chain + decomposable Reduce) |

### Operator System (`hoshicore/ops/`)

All operators inherit from `BaseOp` (in `ops/base.py`). Key base classes:

- **`BaseOp`** — declares `INPUTS`, `OUTPUTS`, `CONFIGS` dicts; implements `execute()` lifecycle with cancellation propagation
- **`ParallelBaseOp`** — frame-independent ops; implement `_async_execute_single()`; support sliding-window concurrency and `DATA_PARALLEL=True`
- **`FilterBaseOp`** — variable-length output (sentinel-driven); `VARIABLE_OUTPUT=True`

Operators communicate via async queues (`RichContextQueue`, `FileCacheQueue`, `IPCQueue`). Length metadata propagates before data flows, enabling downstream pre-allocation.

### DAG YAML Conventions (`hoshicore/dag/`)

| Pattern | Role |
|---------|------|
| `<name>.meta.yaml` | Top-level pipeline with route definitions, parameter declarations, node enable flags |
| `<name>.yaml` (in `base/`) | Reusable SubDAG components (sigma_clip, mix_core, etc.) |
| `<name>.ui.yaml` | Frontend rendering hints (labels, widgets, min/max, groups) |

Nodes reference operators by class name (e.g., `TrailStackerOp`) or by SubDAG filename (e.g., `sigma_clip.yaml`). Routes allow runtime algorithm selection (e.g., `mode: fifo | mix | tmax`).

### Custom Op Layer (`hoshicore/_custom_op/` + `csrc/`)

C++/pybind11 compiled extension (`_C`) with numpy fallback. Every wrapper in `_custom_op/ops/` follows `compiled → numpy` two-tier dispatch — the project always runs without compilation.

Key env vars: `HNW_CUSTOM_OPS_FALLBACK` (`auto`|`numpy`), `HNW_CUSTOM_OPS_THREADS` (`auto`|int), `HNW_CUSTOM_OPS_DEBUG` (`0`|`1`).

Build: `python csrc/build_ops.py`. Windows supports both MSVC and MinGW-w64 ucrt (`--compiler gcc`, CPU-only). See `csrc/README.md` for full build/platform/packaging details.

### Benchmark Suite (`bench/`)

CPU kernel microbenchmarks, GPU prototype comparisons, and data generators. See `bench/README.md` for commands, input modes, and output format.

### GUI (`ui/` + `HoshinoWeaver desktop.py`)

PySide6 with `qasync` event loop integration. The GUI dynamically generates parameter panels from `meta.yaml` + `ui.yaml` pairs via `PanelSchema` / `DynamicConfigPanel`. Mode switching loads different pipeline definitions from `MODE_MAP`.

### Key Design Invariants

- **Streaming + bounded memory**: queues default to `maxsize=1`; frames flow one-at-a-time. This is the core design constraint — never buffer all frames in an Op without using `FileCacheQueue` or disk-backed storage
- **Length before data**: `set_length()` / `get_length()` propagate sequence length through the graph *before* any frame data flows. Downstream Ops can pre-allocate accumulators. Filter ops return `None` (sentinel-driven, unknown length)
- **Cancellation propagation**: `CancellationToken` flows through output queues when any node fails; downstream nodes detect via `CancellationError` and propagate
- **Config priority** (high→low): runtime `global_configs` > `default_settings.yaml` > YAML pipeline `default` > Op class `CONFIGS` default
- **SubDAG namespacing**: flattened nodes get `parent.child` dot-separated names; link resolution uses `rsplit(".", 1)`
- **Custom-op fallback**: `_custom_op` layer always provides a numpy implementation; compiled `_C` is optional. The project must run correctly without building C++ extensions

### Adding a New DAG Operator

1. Create class in `hoshicore/ops/`, inherit `ParallelBaseOp` (frame-independent) or `BaseOp` (stateful reduction)
2. Declare `INPUTS`, `OUTPUTS`, `CONFIGS` class-level dicts with type/required/default specs
3. Decorate with `@register_op("YourOpName")` (imported from `hoshicore/engine/registry.py`)
4. Implement `_async_execute_single(data, configs)` (ParallelBaseOp) or `_async_execute(configs)` (BaseOp)
5. Reference by registered name in YAML node's `op` field

### Adding a New Custom Op (C++ kernel)

See `csrc/README.md` "新增算子" section for the full checklist. Key steps: C++ impl in `csrc/ops/<name>/` → bind in `module.cpp` → Python wrapper with fallback in `_custom_op/ops/` → tests in `test_custom_ops.py`.

## Tech Stack

- Python >= 3.10 (uses `X | Y` union syntax, `match` not used)
- numpy, opencv-python, scipy, PyWavelets
- pybind11 + CMake + Ninja (C++ extension build)
- OpenMP (CPU parallelism in compiled ops)
- CUDA 12.4+ (optional GPU ops, Linux dev / Windows release)
- networkx (DAG topology)
- PySide6 + qasync (GUI)
- rawpy, tifffile, pyexiv2 (image I/O with EXIF preservation)
- asyncio throughout the engine; `asyncio.to_thread` for CPU-bound work

## Testing

- Framework: pytest + pytest-asyncio + pytest-cov
- CI: GitHub Actions on Python 3.11 and 3.12
- Test location: `tests/`

## Key References

- `docs/dag_node_definition.md` — DAG YAML 完整语法规范（标准格式 + Meta 格式）
- `docs/meta_yaml_v2_spec.md` — Meta YAML v2 路由系统设计细节
- `docs/noise-equalization.md` — Mix 星轨噪声均衡算法原理
- `docs/bundle_adjustment_and_stabilization.md` — 星点对齐 / 天地分离的几何模型
- `docs/multi_process.md` — 多进程数据并行架构设计
- `csrc/README.md` — C++ custom-op 构建、平台策略、打包约定
- `csrc/CMAKE_MIGRATION.md` — CMake 构建架构与多平台策略
- `bench/README.md` — Benchmark 套件使用说明与运行建议

## Packaging

`make_package.py` generates a PyInstaller spec with MERGE for shared dependencies between CLI (`launcher.py`) and GUI (`HoshinoWeaver desktop.py`). Output goes to `dist/`.

**Build-check integration**: before generating the spec, `make_package.py` verifies that `_C` is importable. If absent, it automatically invokes `csrc/build_ops.py`. If build fails, packaging aborts (unless `--allow-numpy-only`). This prevents silent performance degradation in release builds.

**`_C` collection**: the spec dynamically detects `_C` via `importlib.util.find_spec` and adds it to `hiddenimports`. On Windows MSVC builds with OpenMP, `vcomp140.dll` is auto-collected. MinGW/Linux/macOS builds have no extra DLL dependencies.

Users only need an NVIDIA driver (no CUDA Toolkit) for GPU features.
