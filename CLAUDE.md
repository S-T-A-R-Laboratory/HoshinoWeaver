# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**HoshinoWeaver (织此星辰)** is an astrophotography image preprocessing tool built around a DAG (Directed Acyclic Graph) operator engine. Users define image processing workflows via YAML, and the engine executes them as async streaming pipelines. Supports star trail stacking, sky/ground separation alignment, noise reduction stacking, and more.

## Development Guidelines

When adding or modifying any feature, check whether the following need updating:
- `docs/` — algorithmic docs, design specs, API references
- `CLAUDE.md` — if architecture, base classes, or invariants changed
- Inline docstrings / YAML comments in affected files

### Production, Benchmark, and Debug Boundaries

- `hoshicore/` contains production implementations. Algorithm behavior tests
  should exercise these modules directly.
- `bench/` contains the existing kernel microbenchmarks. `benchmarks/` contains
  the tracked Norma alignment benchmark component and the CLI-level workflow
  performance baseline harness (`benchmarks/benchmark_workflow.py`, core in
  `benchmarks/workflow_bench.py`); keep local datasets and suites under
  `benchmarks/local/` and out of version control.
- `tools/debug/` contains tracked developer diagnostics that may call private
  implementation APIs. Notebooks and local reference files remain independent
  local diagnostics. None are stable project APIs or may be imported by the
  tracked test suite.
- Tests may depend only on tracked production code, tracked test helpers, or a
  tracked benchmark component. A fresh checkout must not require local debug
  scripts or private benchmark datasets to collect and run tests.
- Keep debug scripts thin. Test reusable behavior in its production module;
  do not add shared wrappers or framework abstractions solely to test a debug
  command-line interface.

## Agent Tooling Notes

- This repository uses UTF-8 source files. Do not assume Windows PowerShell 5.1 will display UTF-8 correctly.
- When terminal output looks garbled, prefer reading text files with Python using `encoding="utf-8"` or use `Get-Content -Encoding utf8` explicitly.
- Treat terminal mojibake as a display issue first. If Python can decode the file as UTF-8, trust the file bytes over the terminal rendering.
- For patching and search, prefer stable ASCII anchors such as function names, variable names, filenames, and numeric constants. Avoid relying on localized comments or mojibake text as patch context.
- In frequently edited debug/dev scripts, prefer English comments, docstrings, and log messages unless there is a strong reason to keep localized text.

## Commands

```bash
# Run GUI
python "HoshinoWeaver desktop.py"

# Run CLI pipeline
python launcher.py <config.yaml> [image_dir] [--route KEY=VALUE] [--input KEY=VALUE] [--config KEY=VALUE]
# --input VALUE forms: <dir> (sorted supported images), @<list file> (one path per line,
# '#' comments allowed), or a JSON array. Use @<list file> for long/explicit input sets.
# Automation flags: --preflight {ask,apply,ignore,abort} (ask = interactive default),
# --log-path PATH (log path is also printed once on stdout), --no-progress.

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

# Run benchmarks (see bench/README.md for full options)
python -m bench.cpu.kernels --frames 64 --height 2048 --width 3072 --dtype uint16 --input-mode synthetic

# Workflow performance baseline (see benchmarks/README.md)
python -m benchmarks.benchmark_workflow --dry-run    # validate suite + print commands
python -m benchmarks.benchmark_workflow              # run benchmarks/local/workflow_baseline.json
python -m benchmarks.benchmark_workflow --case <id> --repeat 3

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
| `meta.py` | Compiles Meta YAML (routes, enable flags) into standard DAG spec |
| `flatten.py` | Recursively expands `.yaml` SubDAG references into namespaced flat nodes |
| `build.py` | Validates DAG spec, builds dependency graph (networkx), produces topological order |
| `wiring.py` | Instantiates Ops, connects async queues, creates feeder coroutines |
| `executor.py` | Runs all nodes concurrently with global cancellation propagation |
| `registry.py` | `@register_op()` decorator → `REGISTERED_OP` dict mapping names to Op classes |
| `preflight.py` | Pre-execution resource estimation; compares peak memory/disk against system limits |
| `runtime_plan.py` | Computes `chunk_rows` for `ChunkIteratorBaseOp` nodes based on available memory |
| `inspect.py` | Extracts parameter schema from YAML for `--inspect` without instantiating Ops |
| `visualize.py` | DAG spec → Mermaid flowchart (pre- and post-flatten granularity) |

### Operator System (`hoshicore/ops/`)

All operators inherit from `BaseOp` (in `ops/base.py`). Key base classes:

- **`BaseOp`** — declares `INPUTS`, `OUTPUTS`, `CONFIGS` dicts; implements `execute()` lifecycle with cancellation propagation
- **`ParallelBaseOp`** — frame-independent ops; implement `_async_execute_single()`; `CONCURRENCY > 1` enables sliding-window concurrent execution
- **`ChunkIteratorBaseOp`** — multi-pass spatial chunking for buffer-backed ops (sigma clip, huber mean); `REPORTS_PROGRESS=True`; `CHUNK_PLANNED=True`
- **`FilterBaseOp`** — variable-length output (sentinel-driven); `VARIABLE_OUTPUT=True`

Operators communicate via async queues (`RichContextQueue`, `FileCacheQueue`). Length metadata propagates before data flows, enabling downstream pre-allocation.

### DAG YAML Conventions (`hoshicore/dag/`)

| Pattern | Role |
|---------|------|
| `<name>.meta.yaml` | Top-level pipeline with route definitions, parameter declarations, node enable flags |
| `<name>.yaml` (in `base/`) | Reusable SubDAG components (sigma_clip, mix_core, etc.) |
| `<name>.ui.yaml` | Frontend rendering hints (labels, widgets, min/max, groups) |

Nodes reference operators by class name (e.g., `TrailStackerOp`) or by SubDAG filename (e.g., `sigma_clip.yaml`). Routes allow runtime algorithm selection (e.g., `mode: fifo | mix | tmax`).

### Custom Op Layer (`hoshicore/_custom_op/` + `csrc/`)

C++/pybind11 compiled extension (`_C`) with CPU/NumPy fallbacks. Most wrappers in
`_custom_op/ops/` select a compiled CUDA/OpenMP backend and then fall back to
NumPy when that backend is explicitly unavailable. The
`star_detect_fused_pixel_components` wrapper has CUDA and OpenMP native backends
but intentionally no standalone NumPy implementation: its final production
fallback is Norma's OpenCV contour detector at the component layer, so the
project still runs without compilation.

Key env vars: `HNW_CUSTOM_OPS_FALLBACK` (`auto`|`cpu`|`numpy`), `HNW_CUSTOM_OPS_THREADS` (`auto`|int), `HNW_CUSTOM_OPS_DEBUG` (`0`|`1`). `cpu` disables CUDA while preserving OpenMP; GUI/runtime callers may use `set_backend_preference()` before starting a pipeline.

Build: `python csrc/build_ops.py`. Windows supports both MSVC and MinGW-w64 ucrt (`--compiler gcc`, CPU-only). See `csrc/README.md` for full build/platform/packaging details.

### Benchmark Suite (`bench/`)

CPU kernel microbenchmarks, GPU prototype comparisons, and data generators. See `bench/README.md` for commands, input modes, and output format.

### GUI (`ui/` + `HoshinoWeaver desktop.py`)

PySide6 with `qasync` event loop integration. The GUI dynamically generates parameter panels from `meta.yaml` + `ui.yaml` pairs via `PanelSchema` / `DynamicConfigPanel`. Mode switching loads different pipeline definitions from `MODE_MAP`.

### Key Design Invariants

- **Streaming + bounded memory**: queues default to `maxsize=1`; frames flow one-at-a-time. This is the core design constraint — never buffer all frames in an Op without using `FileCacheQueue` or disk-backed storage
- **Length before data**: `set_length()` / `get_length()` propagate sequence length through the graph *before* any frame data flows. Downstream Ops can pre-allocate accumulators. Filter ops return `None` (sentinel-driven, unknown length)
- **Cancellation propagation**: `CancellationToken` flows through output queues when the DAG is cancelled (external stop or node failure); downstream nodes raise `CancellationError` on consumption and propagate
- **Config priority** (high→low): runtime `global_configs` > `default_settings.yaml` > YAML pipeline `default` > Op class `CONFIGS` default
- **SubDAG namespacing**: flattened nodes get `parent.child` dot-separated names; link resolution uses `rsplit(".", 1)`
- **Custom-op fallback**: `_custom_op` layer always provides a numpy implementation; compiled `_C` is optional. The project must run correctly without building C++ extensions

### Adding a New DAG Operator

1. Create class in `hoshicore/ops/`, inherit `ParallelBaseOp` (frame-independent) or `BaseOp` (stateful reduction)
2. Declare `INPUTS`, `OUTPUTS`, `CONFIGS` class-level dicts with type/required/default specs
3. Decorate with `@register_op()` (imported from `hoshicore/engine/registry.py`; uses class name as the registered key)
4. Implement `_async_execute_single(data, configs)` (ParallelBaseOp) or `_async_execute(configs)` (BaseOp)
5. Reference by registered name in YAML node's `op` field

### Adding a New Custom Op (C++ kernel)

See `csrc/README.md` "新增算子" section for the full checklist. Key steps: C++ impl in `csrc/ops/cpu/<name>/` or `csrc/ops/cuda/<name>/` → bind in `module.cpp` → Python wrapper with fallback in `_custom_op/ops/` → focused tests in `tests/custom_ops/`.

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
- `docs/dag_engine.md` — DAG Engine 架构概览（编译链路、执行模型、队列协议、终止机制）
- `docs/op.md` — Operator 设计说明（队列机制、基类层次、数据流）
- `docs/noise-equalization.md` — Mix 星轨噪声均衡算法原理
- `docs/bundle_adjustment_and_stabilization.md` — 星点对齐叠加的几何模型
- `csrc/README.md` — C++ custom-op 构建、平台策略、打包约定
- `csrc/CMAKE_MIGRATION.md` — CMake 构建架构与多平台策略
- `bench/README.md` — Benchmark 套件使用说明与运行建议
- `benchmarks/README.md` — Norma 对齐基准 + 工作流性能基线（suite schema、指标口径、对标规范）

## Packaging

See `csrc/README.md` for full build/packaging details and `make_package.py --help` for CLI options.

`make_package.py` generates a PyInstaller MERGE spec for CLI (`launcher.py`) and GUI (`HoshinoWeaver desktop.py`). Before spec generation it verifies `_C` is importable, invoking `csrc/build_ops.py` automatically if absent. Packaging aborts on build failure unless `--allow-numpy-only` is passed.
