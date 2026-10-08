"""Run the workflow performance baseline suite.

Usage::

    python -m benchmarks.benchmark_workflow [suite.json]
    python -m benchmarks.benchmark_workflow --case stack_mean_disk --dry-run
    python -m benchmarks.benchmark_workflow --hardware-only

Every case is a JSON-described ``launcher.py`` invocation executed sequentially
in its own process. See ``benchmarks/README.md`` for the suite schema and for
how the reported metrics are defined.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from .workflow_bench import (DEFAULT_SUITE_PATH, PROJECT_ROOT,
                             RESULT_SCHEMA_VERSION, CaseSpec, PlannedRun,
                             SuiteError, SuiteSpec, load_suite, plan_case,
                             probe_backend, probe_hardware, probe_software,
                             probe_storage_paths, run_case, select_cases,
                             summarize_run_results, validate_case)

SUMMARY_COLUMNS = [
    "case_id", "run_index", "status", "expectation_met", "wall_seconds",
    "pipeline_seconds", "cpu_percent_avg_normalized", "cpu_time_seconds",
    "rss_peak_bytes", "peak_wset_bytes", "input_count", "output_bytes",
    "exit_code", "log_path",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the workflow performance baseline suite through "
                    "launcher.py and record timing/resource metrics.")
    parser.add_argument(
        "suite", nargs="?", default=str(DEFAULT_SUITE_PATH),
        help="Path to the baseline suite JSON "
             f"(default: {DEFAULT_SUITE_PATH})")
    parser.add_argument("--case", action="append", dest="case_ids",
                        metavar="ID", help="Run only this case (repeatable).")
    parser.add_argument("--label", action="append", dest="labels",
                        metavar="TAG", help="Run only cases with all labels.")
    parser.add_argument("--repeat", type=int, default=None,
                        help="Override repeats per case.")
    parser.add_argument("--timeout", type=float, default=None,
                        help="Override the per-run timeout in seconds.")
    parser.add_argument("--preflight", choices=("ask", "apply", "ignore",
                                                "abort"), default=None,
                        help="Override the launcher preflight policy.")
    parser.add_argument("--measure-interval", type=float, default=None,
                        help="Resource sampling interval in seconds.")
    parser.add_argument("--gpu-sample", type=float, default=None,
                        help="nvidia-smi sampling interval; 0 disables it.")
    parser.add_argument("--output-dir", default="benchmark_results/workflow",
                        help="Directory for the run folder.")
    parser.add_argument("--python", default=None,
                        help="Interpreter used to launch launcher.py.")
    parser.add_argument("--launcher", default=None,
                        help="Path to launcher.py.")
    parser.add_argument("--working-dir", default=None,
                        help="Working directory for the launched process.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate and print the planned invocations only.")
    parser.add_argument("--hardware-only", action="store_true",
                        help="Print the machine/backend block and exit.")
    parser.add_argument("--compare", default=None, metavar="RESULTS_JSON",
                        help="Compare against a previous results.json.")
    parser.add_argument("--fail-fast", action="store_true",
                        help="Stop after the first case that misses "
                             "expectations.")
    return parser.parse_args(argv)


# ── formatting helpers ─────────────────────────────────────────────────────


def _format_seconds(value: Any) -> str:
    return f"{value:.2f}s" if isinstance(value, (int, float)) else "n/a"


def _format_bytes(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(size) < 1024.0 or unit == "TiB":
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TiB"


def _format_percent(value: Any) -> str:
    return f"{value:.1f}%" if isinstance(value, (int, float)) else "n/a"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _machine_lines(hardware: Mapping[str, Any],
                   software: Mapping[str, Any]) -> list[str]:
    cpu = hardware.get("cpu", {})
    lines = [
        "[Machine]",
        f"  host        : {hardware.get('hostname')} "
        f"({hardware.get('platform')})",
        f"  cpu         : {cpu.get('model')} "
        f"[{cpu.get('physical_cores')}C/{cpu.get('logical_cores')}T"
        f"{', ' + str(cpu.get('max_frequency_mhz')) + ' MHz'
           if cpu.get('max_frequency_mhz') else ''}]"
        f" (source: {cpu.get('model_source')})",
        f"  ram         : {_format_bytes(hardware.get('ram_total_bytes'))}"
        f"   swap: {_format_bytes(hardware.get('swap_total_bytes'))}",
    ]
    gpus = hardware.get("gpus") or []
    if not gpus:
        lines.append("  gpu         : none reported by the runtime probe")
    for gpu in gpus:
        lines.append(
            f"  gpu[{gpu.get('index')}]     : {gpu.get('name') or 'unknown'} "
            f"vram={_format_bytes(gpu.get('memory_total_bytes'))} "
            f"cc={gpu.get('compute_capability') or 'n/a'} "
            f"driver={gpu.get('driver_version') or 'n/a'} "
            f"(source: {gpu.get('source')})")
    lines += [
        "[Software]",
        f"  product     : {software.get('product')} "
        f"{software.get('version')} ({software.get('release_name')})",
        f"  git         : {software.get('git_revision')} "
        f"[{software.get('git_branch')}] dirty={software.get('git_dirty')}",
        f"  python      : {hardware.get('python')} "
        f"({hardware.get('python_executable')})",
    ]
    return lines


def _case_headline(result: Mapping[str, Any]) -> str:
    resources = result.get("resources") or {}
    phases = result.get("phases") or {}
    parts = [
        f"status={result.get('status')}",
        f"wall={_format_seconds(result.get('wall_seconds'))}",
        f"pipeline={_format_seconds(phases.get('pipeline_seconds'))}",
        f"exec={_format_seconds(phases.get('execution_seconds'))}",
        f"cpu_avg_all_cores={_format_percent(resources.get('cpu_percent_avg_normalized'))}",
        f"cpu_time={_format_seconds(resources.get('cpu_time_seconds'))}",
        f"rss_peak={_format_bytes(resources.get('rss_peak_bytes'))}",
    ]
    return "  ".join(parts)


# ── reporting ──────────────────────────────────────────────────────────────


def _write_results(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2,
                   allow_nan=False),
        encoding="utf-8")


def _summary_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in payload.get("results", []):
        for run in case.get("runs", []):
            resources = run.get("resources") or {}
            phases = run.get("phases") or {}
            config = run.get("config") or {}
            output_bytes = sum(
                entry.get("bytes") or 0
                for entry in run.get("output_files", [])
                if isinstance(entry, dict))
            rows.append({
                "case_id": run.get("case_id"),
                "run_index": run.get("run_index"),
                "status": run.get("status"),
                "expectation_met": run.get("expectation_met"),
                "wall_seconds": run.get("wall_seconds"),
                "pipeline_seconds": phases.get("pipeline_seconds"),
                "cpu_percent_avg_normalized":
                    resources.get("cpu_percent_avg_normalized"),
                "cpu_time_seconds": resources.get("cpu_time_seconds"),
                "rss_peak_bytes": resources.get("rss_peak_bytes"),
                "peak_wset_bytes": resources.get("peak_wset_bytes"),
                "input_count": config.get("input_count"),
                "output_bytes": output_bytes or None,
                "exit_code": run.get("exit_code"),
                "log_path": run.get("log_path"),
            })
    return rows


def _write_summary(path: Path, payload: Mapping[str, Any]) -> None:
    rows = _summary_rows(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _compare_against(reference_path: Path,
                     payload: Mapping[str, Any]) -> None:
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    reference_cases = {case.get("case_id"): case
                       for case in reference.get("results", [])}
    print(f"[Compare] reference: {reference_path}")
    print(f"{'case_id':<40s} {'wall_ref':>10s} {'wall_now':>10s} "
          f"{'delta':>10s} {'ratio':>8s}")
    for case in payload.get("results", []):
        case_id = case.get("case_id")
        old = reference_cases.get(case_id)
        if old is None:
            continue
        old_wall = ((old.get("summary") or {}).get("wall_seconds") or {})
        new_wall = ((case.get("summary") or {}).get("wall_seconds") or {})
        old_mean = old_wall.get("mean")
        new_mean = new_wall.get("mean")
        if not isinstance(old_mean, (int, float)) \
                or not isinstance(new_mean, (int, float)):
            continue
        delta = new_mean - old_mean
        ratio = new_mean / old_mean if old_mean else float("nan")
        print(f"{case_id:<40s} {old_mean:>10.2f} {new_mean:>10.2f} "
              f"{delta:>+10.2f} {ratio:>8.3f}")


# ── preparation ────────────────────────────────────────────────────────────


def _apply_overrides(case: CaseSpec, args: argparse.Namespace) -> CaseSpec:
    changes: dict[str, Any] = {}
    if args.repeat is not None:
        if args.repeat < 1:
            raise SuiteError("--repeat must be >= 1")
        changes["repeats"] = args.repeat
    if args.timeout is not None:
        if args.timeout <= 0:
            raise SuiteError("--timeout must be > 0")
        changes["timeout_seconds"] = args.timeout
    if args.preflight is not None:
        changes["preflight"] = args.preflight
    if args.measure_interval is not None:
        if args.measure_interval <= 0:
            raise SuiteError("--measure-interval must be > 0")
        changes["measure_interval_seconds"] = args.measure_interval
    if args.gpu_sample is not None:
        if args.gpu_sample < 0:
            raise SuiteError("--gpu-sample must be >= 0")
        changes["gpu_sample_interval_seconds"] = args.gpu_sample
    return dataclasses.replace(case, **changes) if changes else case


def _prepare_run(case: CaseSpec, *, suite: SuiteSpec, output_dir: Path,
                 run_index: int, args: argparse.Namespace,
                 tolerate_missing_inputs: bool = False
                 ) -> tuple[PlannedRun | None, list[str]]:
    """Plan and validate one run; never raises for a case-level problem."""
    launcher_path = None
    if args.launcher:
        launcher_path = Path(args.launcher).expanduser()
        if not launcher_path.is_absolute():
            launcher_path = PROJECT_ROOT / launcher_path
    working_dir = None
    if args.working_dir:
        working_dir = Path(args.working_dir).expanduser()
        if not working_dir.is_absolute():
            working_dir = PROJECT_ROOT / working_dir
    try:
        planned = plan_case(case, suite_path=suite.path, output_dir=output_dir,
                            python_bin=args.python, launcher_path=launcher_path,
                            working_dir=working_dir, run_index=run_index,
                            require_inputs=not tolerate_missing_inputs)
        validate_case(case, pipeline_path=planned.pipeline_path,
                      routes=case.routes, configs=planned.configs)
    except SuiteError as exc:
        return None, [str(exc)]
    except Exception as exc:  # schema loading can fail on a broken pipeline
        return None, [f"{type(exc).__name__}: {exc}"]

    warnings = list(planned.inputs.warnings)
    if not planned.pipeline_path.is_file():
        warnings.append(f"pipeline YAML not found: {planned.pipeline_path}")
    if not Path(planned.launcher_path).is_file():
        warnings.append(f"launcher not found: {planned.launcher_path}")
    if planned.inputs.input_count == 0 and not tolerate_missing_inputs:
        return None, warnings + ["no input files resolved"]
    return planned, warnings


def _invalid_record(case: CaseSpec, run_index: int,
                    errors: Sequence[str]) -> dict[str, Any]:
    return {
        "case_id": case.id,
        "run_index": run_index,
        "description": case.description,
        "status": "invalid_input",
        "expect_success": case.expect_success,
        "expectation_met": False,
        "outputs_met": None,
        "command": [],
        "wall_seconds": None,
        "exit_code": None,
        "phases": {},
        "phases_source": "unavailable",
        "resources": {},
        "config": {"pipeline": case.pipeline,
                   "routes": dict(case.routes),
                   "configs": dict(case.configs)},
        "log_path": None,
        "output_files": [],
        "warnings": list(errors),
    }


def _decorate_run(record: dict[str, Any], planned: PlannedRun,
                  case: CaseSpec, *, backend: Mapping[str, Any],
                  storage_before: Mapping[str, Any],
                  storage_after: Mapping[str, Any]) -> dict[str, Any]:
    record["config"] = {
        "pipeline": str(planned.pipeline_path),
        "launcher": str(planned.launcher_path),
        "python": planned.python_bin,
        "working_dir": str(planned.working_dir),
        "routes": dict(case.routes),
        "configs": {key: _jsonable(value)
                    for key, value in planned.configs.items()},
        "buffer_mode": planned.configs.get("buffer_mode"),
        "temp_path": planned.configs.get("temp_path"),
        "input_count": planned.inputs.input_count,
        "input_list_sha1": planned.inputs.list_sha1(),
        "input_sources": [
            {"name": entry.name, "source": entry.source,
             "source_path": entry.source_path, "count": len(entry.files),
             "truncated": entry.truncated}
            for entry in planned.inputs.entries],
        "expected_outputs": [str(path) for path in planned.expected_outputs],
    }
    record["backend"] = dict(backend)
    declared = {key: case.storage.get(key) for key in ("input_media",
                                                       "cache_media")}
    record["declared_storage"] = {
        **declared,
        "kind": ("declared" if any(value for value in declared.values())
                 else "undeclared"),
    }
    record["storage"] = {"before": dict(storage_before),
                         "after": dict(storage_after)}
    return record


def _storage_paths(planned: PlannedRun) -> list[Path | None]:
    paths: list[Path | None] = []
    for entry in planned.inputs.entries:
        if entry.source == "dir" and entry.source_path:
            paths.append(Path(entry.source_path))
    paths.append(planned.paths.get("cache_root"))
    paths.append(planned.paths.get("output_root"))
    return paths


# ── commands ───────────────────────────────────────────────────────────────


def _print_hardware_only() -> int:
    backend = probe_backend({})
    hardware = probe_hardware(backend=backend)
    software = probe_software()
    for line in _machine_lines(hardware, software):
        print(line)
    print("[Backend]")
    compiled = backend.get("compiled_custom_ops", {}) if isinstance(backend,
                                                                   dict) else {}
    cuda = backend.get("cuda_runtime", {}) if isinstance(backend, dict) else {}
    print(f"  compiled    : {compiled.get('status')} "
          f"compiler={compiled.get('compiler')} "
          f"openmp={compiled.get('openmp')} cuda={compiled.get('cuda')}")
    print(f"  cuda        : {cuda.get('status')} "
          f"device={cuda.get('device')} "
          f"vram={_format_bytes(cuda.get('total_bytes'))}")
    print(f"  dependencies: {software.get('dependencies')}")
    return 0


def _dry_run(suite: SuiteSpec, cases: Sequence[CaseSpec],
             args: argparse.Namespace) -> int:
    output_dir = (PROJECT_ROOT / args.output_dir
                  if not Path(args.output_dir).is_absolute()
                  else Path(args.output_dir))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"{suite.suite_id}_dryrun_{timestamp}"
    print(f"[Dry run] suite={suite.suite_id} cases={len(cases)} "
          f"output={run_dir}")
    errors = 0
    had_warnings = False
    for index, case in enumerate(cases, 1):
        print(f"\n[{index}/{len(cases)}] {case.id}")
        if case.description:
            print(f"  description : {case.description}")
        planned, warnings = _prepare_run(case, suite=suite,
                                         output_dir=run_dir, run_index=1,
                                         args=args,
                                         tolerate_missing_inputs=True)
        for warning in warnings:
            had_warnings = True
            print(f"  WARNING     : {warning}")
        if planned is None:
            errors += 1
            continue
        print(f"  inputs      : {planned.inputs.input_count} file(s) "
              f"sha1={planned.inputs.list_sha1()}")
        for entry in planned.inputs.entries:
            print(f"    - {entry.name}: {len(entry.files)} file(s) "
                  f"from {entry.source} {entry.source_path or ''}")
        print(f"  output      : {planned.configs.get('output_filename')}")
        print(f"  buffer_mode : {planned.configs.get('buffer_mode')} "
              f"temp_path={planned.configs.get('temp_path')}")
        print(f"  command     : {' '.join(planned.command)}")
    print(f"\n[Dry run] errors={errors} "
          f"warnings={'yes' if had_warnings else 'none'} "
          f"(nothing was created)")
    return 2 if errors else 0


def _run_suite(suite: SuiteSpec, cases: Sequence[CaseSpec],
               args: argparse.Namespace) -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (PROJECT_ROOT / args.output_dir
                  if not Path(args.output_dir).is_absolute()
                  else Path(args.output_dir))
    run_dir = output_dir / f"{suite.suite_id}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    backend_cache: dict[tuple[tuple[str, str], ...], dict[str, Any]] = {}

    def backend_for(case: CaseSpec) -> dict[str, Any]:
        key = tuple(sorted(case.env.items()))
        if key not in backend_cache:
            backend_cache[key] = probe_backend(case.env)
        return backend_cache[key]

    default_backend = backend_for(cases[0]) if cases else probe_backend({})
    hardware = probe_hardware(backend=default_backend)
    software = probe_software()

    payload: dict[str, Any] = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "suite_id": suite.suite_id,
        "suite_path": str(suite.path),
        "suite_description": suite.description,
        "generated_at": datetime.now().astimezone().isoformat(),
        "generated_at_epoch": time.time(),
        "cli_args": sys.argv[1:],
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "environment": {
            "hardware": hardware,
            "software": software,
            "backend": default_backend,
        },
        "results": [],
    }

    for line in _machine_lines(hardware, software):
        print(line)

    results_path = run_dir / "results.json"
    summary_path = run_dir / "summary.csv"
    _write_results(results_path, payload)

    failed = 0
    for index, case in enumerate(cases, 1):
        runs: list[dict[str, Any]] = []
        case_warnings: list[str] = []
        for run_index in range(1, case.repeats + 1):
            print(f"\n[{index}/{len(cases)}] {case.id} "
                  f"(run {run_index}/{case.repeats})")
            planned, warnings = _prepare_run(case, suite=suite,
                                             output_dir=run_dir,
                                             run_index=run_index, args=args)
            case_warnings.extend(warnings)
            if planned is None:
                runs.append(_invalid_record(case, run_index, warnings))
                print("  input/config invalid:")
                for warning in warnings:
                    print(f"    - {warning}")
                break
            storage_before = probe_storage_paths(_storage_paths(planned))
            record = run_case(
                planned,
                timeout_seconds=case.timeout_seconds,
                measure_interval_seconds=case.measure_interval_seconds,
                gpu_sample_interval_seconds=case.gpu_sample_interval_seconds)
            storage_after = probe_storage_paths(_storage_paths(planned))
            record = _decorate_run(record, planned, case,
                                   backend=backend_for(case),
                                   storage_before=storage_before,
                                   storage_after=storage_after)
            if warnings:
                record["warnings"] = list(record.get("warnings", [])) + warnings
            runs.append(record)
            print(f"  {_case_headline(record)}")
            for warning in record.get("warnings", []):
                print(f"  WARNING: {warning}")

        case_payload = {
            "case_id": case.id,
            "description": case.description,
            "labels": list(case.labels),
            "pipeline": case.pipeline,
            "expect_success": case.expect_success,
            "repeats": case.repeats,
            "preflight_policy": case.preflight,
            "runs": runs,
            "summary": summarize_run_results(runs),
            "warnings": case_warnings,
        }
        payload["results"].append(case_payload)
        _write_results(results_path, payload)
        _write_summary(summary_path, payload)
        if not case_payload["summary"].get("expectations_met", False):
            failed += 1
            if args.fail_fast:
                print("\n[fail-fast] stopping after the first failing case")
                break

    print(f"\n[Summary] cases={len(payload['results'])} failed={failed}")
    print(f"  results : {results_path}")
    print(f"  summary : {summary_path}")
    if not failed:
        print("  mean wall per case:")
        for case_payload in payload["results"]:
            wall = (case_payload["summary"].get("wall_seconds") or {})
            if wall.get("mean") is not None:
                print(f"    {case_payload['case_id']:<40s} "
                      f"{wall['mean']:.2f}s "
                      f"(min {wall['min']:.2f}s, max {wall['max']:.2f}s)")

    if args.compare:
        _compare_against(Path(args.compare), payload)
    return 0 if failed == 0 else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.hardware_only:
        return _print_hardware_only()
    try:
        suite = load_suite(args.suite)
        cases = [_apply_overrides(case, args)
                 for case in select_cases(suite, case_ids=args.case_ids,
                                          labels=args.labels)]
    except SuiteError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        return _dry_run(suite, cases, args)
    return _run_suite(suite, cases, args)


if __name__ == "__main__":
    raise SystemExit(main())
