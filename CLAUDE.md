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

# Package for distribution (PyInstaller)
python make_package.py
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

### GUI (`ui/` + `HoshinoWeaver desktop.py`)

PySide6 with `qasync` event loop integration. The GUI dynamically generates parameter panels from `meta.yaml` + `ui.yaml` pairs via `PanelSchema` / `DynamicConfigPanel`. Mode switching loads different pipeline definitions from `MODE_MAP`.

### Key Design Invariants

- **Streaming + bounded memory**: queues default to `maxsize=1`; frames flow one-at-a-time. This is the core design constraint — never buffer all frames in an Op without using `FileCacheQueue` or disk-backed storage
- **Length before data**: `set_length()` / `get_length()` propagate sequence length through the graph *before* any frame data flows. Downstream Ops can pre-allocate accumulators. Filter ops return `None` (sentinel-driven, unknown length)
- **Cancellation propagation**: `CancellationToken` flows through output queues when any node fails; downstream nodes detect via `CancellationError` and propagate
- **Config priority** (high→low): runtime `global_configs` > `default_settings.yaml` > YAML pipeline `default` > Op class `CONFIGS` default
- **SubDAG namespacing**: flattened nodes get `parent.child` dot-separated names; link resolution uses `rsplit(".", 1)`

### Adding a New Operator

1. Create class in `hoshicore/ops/`, inherit `ParallelBaseOp` (frame-independent) or `BaseOp` (stateful reduction)
2. Declare `INPUTS`, `OUTPUTS`, `CONFIGS` class-level dicts with type/required/default specs
3. Decorate with `@register_op("YourOpName")` (imported from `hoshicore/engine/registry.py`)
4. Implement `_async_execute_single(data, configs)` (ParallelBaseOp) or `_async_execute(configs)` (BaseOp)
5. Reference by registered name in YAML node's `op` field

## Tech Stack

- Python >= 3.10 (uses `X | Y` union syntax, `match` not used)
- numpy, opencv-python, scipy, numba (JIT kernels), PyWavelets
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

## Packaging

`make_package.py` generates a PyInstaller spec with MERGE for shared dependencies between CLI (`launcher.py`) and GUI (`HoshinoWeaver desktop.py`). Output goes to `dist/`.
