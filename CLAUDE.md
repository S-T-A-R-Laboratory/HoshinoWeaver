# CLAUDE.md

This file provides guidance for agents working in this repository.

**HoshinoWeaver (织此星辰)** is an astrophotography image preprocessing tool built around a DAG (Directed Acyclic Graph) operator engine. Users define image processing workflows via YAML, and the engine executes them as async streaming pipelines. 

## Maintenance

- Keep this file concise: global constraints, common entry points and topic navigation. Update only if architecture, base classes, or invariants changed
- Put algorithm details, module APIs, build settings and validation records in existing
  topic documents (`docs/`), docstrings or YAML comments; update them with the affected behavior.

## Code Boundaries

- `hoshicore/` contains production code; behavior tests should exercise it directly.
- `bench/` contains kernel benchmarks; `benchmarks/` contains Norma and CLI workflow
  benchmarks. Local datasets and suites belong in ignored `benchmarks/local/`.
- `tools/debug/` contains thin developer diagnostics and may use private APIs;
  notebooks and local reference files are independent diagnostics, not stable APIs.
- Tests may use tracked production code, test helpers or benchmark components only.
  Fresh-checkout tests must not require debug scripts or private datasets. Do not
  introduce framework abstractions solely to test a debug CLI.
- `hoshicore/packaging/` is build-time code: applications must not import it,
  backends exclude it from distributions, and its `__init__.py` stays lightweight.

## Editing on Windows

- Files are UTF-8. Use `Get-Content -Encoding utf8` or Python with `encoding="utf-8"`.
  Treat mojibake as a display issue first; trust correctly decoded file bytes.
- Prefer stable ASCII names as search/patch anchors. In frequently edited debug/dev
  scripts, prefer English comments, docstrings and logs unless localization is needed.

## Common Commands

```sh
# Run GUI
python "HoshinoWeaver desktop.py"

# Run CLI pipeline
python launcher.py <pipeline.yaml> [image_dir] [--route KEY=VALUE] [--input KEY=VALUE] [--config KEY=VALUE]

# Inspect a pipeline's parameter schema
python launcher.py <pipeline.yaml> --inspect

# Run tests
python -m pytest tests/ -v --tb=short -x

# Build C++ custom ops
python csrc/build_ops.py                       # auto-detect compiler
python csrc/build_ops.py --cuda                # with CUDA support

# Run benchmarks (see bench/README.md for full options)
python -m bench.cpu.kernels --frames 64 --height 2048 --width 3072 --dtype uint16 --input-mode synthetic

# Package for distribution (PyInstaller)
python make_package.py --no-build --verify
# Package for distribution (Nuitka)
python make_package.py --backend nuitka --no-build --verify
```

CLI input forms and automation flags are documented in [README.md](README.md).
Native build options, benchmark commands and packaging policies are in the topic
documents below.

## Architecture and Invariants

Compilation: Meta YAML -> `meta_resolve()` -> `flatten_sub_dags()` ->
`validate_and_build_order()` -> `ValidatedDag` -> wiring -> `DAGExecutor`.

`engine/` compiles and schedules; `ops/` defines operators; `component/` supplies
algorithms and infrastructure; `dag/` holds workflows; `ui/` renders the GUI.
`_custom_op/` wraps native kernels from `csrc/`.

- **Bounded streaming:** queues default to `maxsize=1`. Do not accumulate all
  frames in an Op without `FileCacheQueue` or disk-backed storage.
- **Length before data:** `set_length()` / `get_length()` propagate length before
  frames. Variable-output filters announce `None` and terminate via sentinel.
- **Cancellation:** external stop or node failure propagates `CancellationToken`
  through queues; consumers raise `CancellationError` and propagate cancellation.
- **Config priority:** runtime `global_configs` > `default_settings.yaml` >
  pipeline YAML `default` > Op `CONFIGS` defaults.
- **SubDAG names:** flattened nodes use `parent.child`; resolve links with
  `rsplit(".", 1)`.
- **Optional native runtime:** production workflows must run without `_C`.
  Fallback may be NumPy or a component-level OpenCV path; not every native
  wrapper has its own NumPy implementation.
- **Release publication:** replace existing releases only with `--overwrite`;
  publish staged artifacts after requested verification, restoring the previous
  directory if publication fails.

For operator changes, consult the operator and length-protocol documents below;
native kernel additions follow the checklist in `csrc/README.md`.

## Topic Documents

| Topic | Reference |
|-------|-----------|
| Architecture, GUI, dependencies and testing | [Technical overview](docs/README.md) |
| DAG compilation and execution | [Engine](docs/dag_engine.md) |
| YAML and route syntax | [Node definition](docs/dag_node_definition.md), [Meta YAML v2](docs/meta_yaml_v2_spec.md) |
| Operator APIs and extension | [Operators](docs/op.md), [Length protocol](docs/length_protocol.md) |
| Native builds, dispatch and kernel extension | [Custom ops](csrc/README.md) |
| Kernel and workflow performance | [bench](bench/README.md), [benchmarks](benchmarks/README.md) |
| Packaging lifecycle and backends | [Packaging](docs/packaging.md)|

Algorithm documents are indexed in `docs/README.md`; native build design and
alignment acceleration details are linked from `csrc/README.md`.
