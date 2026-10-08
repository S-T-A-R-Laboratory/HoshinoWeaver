"""Core helpers for the CLI-level workflow performance benchmark.

The harness treats a HoshinoWeaver workflow as a black box: every case is a
JSON-described ``launcher.py`` invocation, run in its own process, while the
harness samples CPU/RAM/SWAP and records the machine, software and backend
context. It deliberately does not import the DAG engine to execute anything.

Layout of a run:

    <output_dir>/<suite_id>_<timestamp>/
        results.json                 # full structured payload
        summary.csv                  # one row per case run
        cases/<case_id>/run<k>/
            launcher.log             # --log-path target (copied evidence)
            stdout.txt, stderr.txt
            inputs_<name>.txt        # frozen input list passed as --input k=@file
            output/                  # pipeline output directory (default)
            cache/                   # disk-buffer temp_path (default)

Boundaries: the harness only ever reads pipeline YAML/JSON metadata and always
executes workflows through ``launcher.py``; nothing in ``hoshicore/`` is
imported to run a workflow.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # `import launcher` for schema validation
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_SUITE_PATH = (Path(__file__).resolve().parent / "local"
                      / "workflow_baseline.json")
DEFAULT_LAUNCHER = "launcher.py"
DEFAULT_OUTPUT_EXTENSION = "tif"
DEFAULT_PREFLIGHT = "ignore"
DEFAULT_TIMEOUT_SECONDS = 7200.0
DEFAULT_MEASURE_INTERVAL_SECONDS = 0.5
LAUNCHER_LOG_MARKER = "[Launcher] log file: "
SUITE_SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 1

_SUITE_KEYS = {"schema_version", "suite_id", "description", "defaults", "cases"}
_DEFAULT_KEYS = {
    "launcher", "python", "preflight", "timeout_seconds", "repeats",
    "measure_interval_seconds", "gpu_sample_interval_seconds", "env",
    "working_dir", "output_root", "cache_root",
}
_SUITE_ONLY_KEYS = {
    "launcher", "python", "working_dir", "output_root", "cache_root",
    "measure_interval_seconds", "gpu_sample_interval_seconds",
}
_CASE_KEYS = (_DEFAULT_KEYS - _SUITE_ONLY_KEYS) | {
    "id", "description", "pipeline", "inputs", "input_limit", "routes",
    "configs", "labels", "storage", "expect_success", "expected_outputs",
}
_STORAGE_KEYS = {"input_media", "cache_media"}
_PREFLIGHT_MODES = ("ask", "apply", "ignore", "abort")

# loguru's default file format starts with `YYYY-MM-DD HH:MM:SS.SSS`.
_LOG_TIMESTAMP = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})")
_LOG_RUN_FROM_YAML = re.compile(r"run_from_yaml time cost: ([\d.]+)s")
_LOG_FEEDER = re.compile(r"\[Feeder\] Global input '([^']+)': (\d+) items")
_DAG_START = "DAG execution starting"
_DAG_DONE = "DAG execution completed. Results collected."
_CAPABILITIES_MARK = "[Capabilities]"
_PREFLIGHT_MARK = "[Preflight]"

_BACKEND_PROBE_SNIPPET = (
    "import json;"
    "from hoshicore.component.runtime_diagnostics "
    "import probe_runtime_components;"
    "print('RESULT_JSON=' + json.dumps(probe_runtime_components(), default=str))"
)
_BACKEND_PROBE_MARKER = "RESULT_JSON="


class SuiteError(ValueError):
    """Raised when a suite file or case definition is not usable."""


# ────────────────────────────────────────────────────────────────────────────
# Suite schema
# ────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CaseSpec:
    id: str
    description: str
    pipeline: str
    inputs: Mapping[str, Any]
    input_limit: int | None
    routes: Mapping[str, str]
    configs: Mapping[str, Any]
    env: Mapping[str, str]
    labels: tuple[str, ...]
    storage: Mapping[str, Any]
    expect_success: bool
    expected_outputs: tuple[str, ...]
    timeout_seconds: float
    repeats: int
    preflight: str
    working_dir: str | None
    output_root: str | None
    cache_root: str | None
    launcher: str
    python: str | None
    measure_interval_seconds: float
    gpu_sample_interval_seconds: float


@dataclass(frozen=True)
class SuiteSpec:
    suite_id: str
    path: Path
    description: str
    cases: tuple[CaseSpec, ...]


def _require_mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SuiteError(f"{where} must be a JSON object")
    return value


def _reject_unknown_keys(payload: Mapping[str, Any], allowed: set[str],
                         where: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise SuiteError(
            f"{where} has unknown key(s): {', '.join(unknown)}. "
            f"Allowed: {', '.join(sorted(allowed))}")


def _string_list(value: Any, where: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(x, str)
                                              for x in value):
        raise SuiteError(f"{where} must be an array of strings")
    return tuple(value)


def _env_map(value: Any, where: str) -> dict[str, str]:
    if value is None:
        return {}
    raw = _require_mapping(value, where)
    result: dict[str, str] = {}
    for key, item in raw.items():
        if not isinstance(item, (str, int, float, bool)):
            raise SuiteError(f"{where}.{key} must be a scalar value")
        result[str(key)] = str(item) if not isinstance(item, bool) else (
            "true" if item else "false")
    return result


def _positive_number(value: Any, where: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SuiteError(f"{where} must be a number")
    if value < 0 or (value == 0 and not allow_zero):
        raise SuiteError(f"{where} must be "
                         f"{'>= 0' if allow_zero else '> 0'}")
    return float(value)


def _build_case(raw: Mapping[str, Any], defaults: Mapping[str, Any],
                *, index: int) -> CaseSpec:
    where = f"cases[{index}]"
    _reject_unknown_keys(raw, _CASE_KEYS, where)
    merged = {**defaults, **raw}

    case_id = merged.get("id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise SuiteError(f"{where}.id must be a non-empty string")
    case_id = case_id.strip()

    pipeline = merged.get("pipeline")
    if not isinstance(pipeline, str) or not pipeline.strip():
        raise SuiteError(f"{where}.pipeline is required")

    inputs = _require_mapping(merged.get("inputs", {}), f"{where}.inputs")
    if not inputs:
        raise SuiteError(f"{where}.inputs must declare at least one input")
    for name, value in inputs.items():
        if not isinstance(value, (str, list)):
            raise SuiteError(
                f"{where}.inputs.{name} must be a directory path, an "
                f"@list-file path, or an array of paths")

    routes = _require_mapping(merged.get("routes", {}), f"{where}.routes")
    for key, value in routes.items():
        if not isinstance(value, str):
            raise SuiteError(f"{where}.routes.{key} must be a string")

    configs = _require_mapping(merged.get("configs", {}), f"{where}.configs")
    for key, value in configs.items():
        if value is None:
            raise SuiteError(
                f"{where}.configs.{key} is null; omit the key instead")
        if not isinstance(value, (str, bool, int, float, list, dict)):
            raise SuiteError(
                f"{where}.configs.{key} must be a scalar, list or object")

    storage = _require_mapping(merged.get("storage", {}), f"{where}.storage")
    _reject_unknown_keys(storage, _STORAGE_KEYS, f"{where}.storage")

    preflight = merged.get("preflight", DEFAULT_PREFLIGHT)
    if preflight not in _PREFLIGHT_MODES:
        raise SuiteError(
            f"{where}.preflight must be one of {', '.join(_PREFLIGHT_MODES)}")

    input_limit = merged.get("input_limit")
    if input_limit is not None:
        if isinstance(input_limit, bool) or not isinstance(input_limit, int) \
                or input_limit <= 0:
            raise SuiteError(f"{where}.input_limit must be a positive integer")

    repeats = merged.get("repeats", 1)
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise SuiteError(f"{where}.repeats must be a positive integer")

    timeout = _positive_number(
        merged.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS),
        f"{where}.timeout_seconds")
    interval = _positive_number(
        merged.get("measure_interval_seconds",
                   DEFAULT_MEASURE_INTERVAL_SECONDS),
        f"{where}.measure_interval_seconds")
    gpu_interval = _positive_number(
        merged.get("gpu_sample_interval_seconds", 0),
        f"{where}.gpu_sample_interval_seconds", allow_zero=True)

    python_bin = merged.get("python")
    if python_bin is not None and not isinstance(python_bin, str):
        raise SuiteError(f"{where}.python must be a string path or null")

    return CaseSpec(
        id=case_id,
        description=str(merged.get("description", "")),
        pipeline=pipeline.strip(),
        inputs=inputs,
        input_limit=input_limit,
        routes=routes,
        configs=configs,
        env=_env_map(merged.get("env", {}), f"{where}.env"),
        labels=_string_list(merged.get("labels"), f"{where}.labels"),
        storage=storage,
        expect_success=bool(merged.get("expect_success", True)),
        expected_outputs=_string_list(merged.get("expected_outputs"),
                                      f"{where}.expected_outputs"),
        timeout_seconds=timeout,
        repeats=repeats,
        preflight=str(preflight),
        working_dir=merged.get("working_dir"),
        output_root=merged.get("output_root"),
        cache_root=merged.get("cache_root"),
        launcher=str(merged.get("launcher", DEFAULT_LAUNCHER)),
        python=python_bin,
        measure_interval_seconds=interval,
        gpu_sample_interval_seconds=gpu_interval,
    )


def load_suite(path: str | os.PathLike[str]) -> SuiteSpec:
    """Load and validate a workflow baseline suite (see README for schema)."""
    suite_path = Path(path).expanduser()
    if not suite_path.is_file():
        raise SuiteError(f"suite file does not exist: {suite_path}")
    try:
        payload = json.loads(suite_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise SuiteError(f"{suite_path} is not valid JSON: {exc}") from exc
    payload = _require_mapping(payload, "suite root")
    _reject_unknown_keys(payload, _SUITE_KEYS, "suite root")

    if payload.get("schema_version") != SUITE_SCHEMA_VERSION:
        raise SuiteError(
            f"suite schema_version must be {SUITE_SCHEMA_VERSION}, got "
            f"{payload.get('schema_version')!r}")

    defaults = _require_mapping(payload.get("defaults", {}),
                                "suite defaults")
    _reject_unknown_keys(defaults, _DEFAULT_KEYS, "suite defaults")

    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise SuiteError("suite cases must be a non-empty array")

    cases: list[CaseSpec] = []
    seen: set[str] = set()
    for index, raw_case in enumerate(raw_cases):
        case = _build_case(_require_mapping(raw_case, f"cases[{index}]"),
                           defaults, index=index)
        if case.id in seen:
            raise SuiteError(f"duplicate case id: {case.id}")
        seen.add(case.id)
        cases.append(case)

    suite_id = payload.get("suite_id", suite_path.stem)
    if not isinstance(suite_id, str) or not suite_id.strip():
        raise SuiteError("suite_id must be a non-empty string")

    return SuiteSpec(
        suite_id=suite_id.strip(),
        path=suite_path.resolve(),
        description=str(payload.get("description", "")),
        cases=tuple(cases),
    )


def select_cases(suite: SuiteSpec, *, case_ids: Sequence[str] | None = None,
                 labels: Sequence[str] | None = None) -> list[CaseSpec]:
    """Filter suite cases by id and/or label; order follows the suite file."""
    wanted_ids = {str(x) for x in (case_ids or ())}
    wanted_labels = {str(x) for x in (labels or ())}
    selected = [
        case for case in suite.cases
        if (not wanted_ids or case.id in wanted_ids)
        and (not wanted_labels or wanted_labels.issubset(set(case.labels)))
    ]
    if not selected:
        raise SuiteError("no case matched the requested id/label filters")
    missing_ids = wanted_ids - {case.id for case in suite.cases}
    if missing_ids:
        raise SuiteError(
            f"unknown case id(s): {', '.join(sorted(missing_ids))}")
    return selected


# ────────────────────────────────────────────────────────────────────────────
# Inputs, configs and command building
# ────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class InputEntry:
    name: str
    files: tuple[str, ...]
    source: str                      # "dir" | "list_file" | "array"
    source_path: str | None
    truncated: bool


@dataclass(frozen=True)
class ResolvedInputs:
    entries: tuple[InputEntry, ...]
    warnings: tuple[str, ...]

    @property
    def input_count(self) -> int:
        return sum(len(entry.files) for entry in self.entries)

    def list_sha1(self) -> str | None:
        """Fingerprint input names and paths in the order delivered to the DAG."""
        if not any(entry.files for entry in self.entries):
            return None
        digest = hashlib.sha1()
        for entry in self.entries:
            digest.update(json.dumps([entry.name, list(entry.files)],
                                     ensure_ascii=False).encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()


def _supported_image(name: str) -> bool:
    from hoshicore.component.utils import is_support_format
    return is_support_format(name)


def _expand_directory(directory: Path, *, limit: int | None) -> tuple[list[str],
                                                                    bool]:
    entries = sorted(x for x in os.listdir(directory)
                     if _supported_image(x))
    truncated = limit is not None and len(entries) > limit
    if limit is not None:
        entries = entries[:limit]
    return [str(directory / name) for name in entries], truncated


def _read_list_file(list_path: Path) -> list[str]:
    entries: list[str] = []
    with open(list_path, "r", encoding="utf-8-sig") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line and not line.startswith("#"):
                entries.append(line)
    return entries


def _resolve_user_path(value: str) -> Path:
    """Resolve a suite-declared input path against the repository root."""
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def resolve_inputs(case: CaseSpec, *,
                   require_exists: bool = True) -> ResolvedInputs:
    """Freeze every case input into an explicit, sorted file list.

    Directories are expanded at plan time (not by the launcher) so the recorded
    ``input_list_sha1`` describes exactly what the measured run consumed. With
    ``require_exists=False`` a missing directory or list file becomes a warning
    plus an empty entry, which is what ``--dry-run`` needs to validate a
    template whose data lives on another machine.
    """
    entries: list[InputEntry] = []
    warnings: list[str] = []
    for name, value in case.inputs.items():
        if isinstance(value, str) and value.startswith("@"):
            list_path = _resolve_user_path(value[1:])
            if not list_path.is_file():
                message = (f"input '{name}' list file does not exist: "
                           f"{list_path}")
                if require_exists:
                    raise SuiteError(f"case {case.id}: {message}")
                warnings.append(message)
                entries.append(InputEntry(name, (), "list_file",
                                          str(list_path), False))
                continue
            files = _read_list_file(list_path)
            truncated = False
            if case.input_limit is not None and len(files) > case.input_limit:
                files = files[:case.input_limit]
                truncated = True
            entries.append(InputEntry(name, tuple(files), "list_file",
                                      str(list_path), truncated))
        elif isinstance(value, str):
            directory = _resolve_user_path(value)
            if not directory.is_dir():
                message = f"input '{name}' directory does not exist: {directory}"
                if require_exists:
                    raise SuiteError(f"case {case.id}: {message}")
                warnings.append(message)
                entries.append(InputEntry(name, (), "dir", str(directory),
                                          False))
                continue
            files, truncated = _expand_directory(directory,
                                                 limit=case.input_limit)
            entries.append(InputEntry(name, tuple(files), "dir",
                                      str(directory), truncated))
        elif isinstance(value, list):
            if not all(isinstance(item, str) for item in value):
                raise SuiteError(
                    f"case {case.id}: input '{name}' array must contain only "
                    f"strings")
            files = [str(item) for item in value]
            truncated = False
            if case.input_limit is not None and len(files) > case.input_limit:
                files = files[:case.input_limit]
                truncated = True
            entries.append(InputEntry(name, tuple(files), "array", None,
                                      truncated))
        else:  # pragma: no cover - guarded by schema validation
            raise SuiteError(
                f"case {case.id}: input '{name}' has unsupported type")

        if not entries[-1].files:
            warnings.append(
                f"input '{name}' resolved to 0 files "
                f"(source: {entries[-1].source})")
        if entries[-1].truncated:
            warnings.append(
                f"input '{name}' truncated to input_limit="
                f"{case.input_limit}")
    return ResolvedInputs(tuple(entries), tuple(warnings))


def format_config_value(value: Any) -> str:
    """Render a JSON config value the way ``launcher.py`` expects on the CLI."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return value
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    raise SuiteError(f"unsupported config value type: {type(value).__name__}")


def substitute_placeholders(text: str, mapping: Mapping[str, str]) -> str:
    """Substitute ``{key}`` placeholders and reject anything left unresolved."""
    try:
        resolved = text.format(**mapping)
    except KeyError as exc:
        raise SuiteError(f"unknown placeholder {exc} in '{text}'") from exc
    if "{" in resolved or "}" in resolved:
        raise SuiteError(f"unresolved placeholder in '{text}' -> '{resolved}'")
    return resolved


def case_paths(case: CaseSpec, run_index: int, *, output_dir: Path,
               working_dir: Path) -> dict[str, Path]:
    """Resolve every per-run directory and file path for a case."""
    base = output_dir.resolve()
    case_dir = base / "cases" / case.id
    run_dir = case_dir / f"run{run_index}"
    mapping = {
        "case_id": case.id,
        "run_index": str(run_index),
        "output_dir": str(base),
        "run_dir": str(run_dir),
        "working_dir": str(working_dir),
    }
    output_root = substitute_placeholders(
        case.output_root or str(run_dir / "output"), mapping)
    cache_root = substitute_placeholders(
        case.cache_root or str(run_dir / "cache"), mapping)
    return {
        "case_dir": case_dir,
        "run_dir": run_dir,
        "output_root": _resolve_optional_path(output_root),
        "cache_root": _resolve_optional_path(cache_root),
    }


def effective_configs(case: CaseSpec, paths: Mapping[str, Path],
                      run_index: int) -> dict[str, Any]:
    """Configs for one run, including the injected output/cache defaults."""
    mapping = {
        "case_id": case.id,
        "run_index": str(run_index),
        "output_root": str(paths["output_root"]),
        "cache_root": str(paths["cache_root"]),
        "run_dir": str(paths["run_dir"]),
    }
    configs: dict[str, Any] = dict(case.configs)
    if "output_filename" not in configs:
        filename = f"{case.id}.{DEFAULT_OUTPUT_EXTENSION}"
        configs["output_filename"] = str(
            paths["output_root"] / substitute_placeholders(filename, mapping))
    else:
        configs["output_filename"] = substitute_placeholders(
            str(configs["output_filename"]), mapping)
    buffer_mode = configs.get("buffer_mode")
    if buffer_mode in ("disk", "replay") and not configs.get("temp_path"):
        configs["temp_path"] = str(paths["cache_root"])
    return configs


def validate_case(case: CaseSpec, *, pipeline_path: Path, routes: Mapping[str,
                                                                         str],
                  configs: Mapping[str, Any]) -> None:
    """Validate route/config names against the pipeline schema before launch."""
    from launcher import _resolve_config_overrides
    from hoshicore.engine.inspect import inspect_yaml

    inspect_result = inspect_yaml(str(pipeline_path),
                                 route_choices=dict(routes) or None)
    raw_overrides = {key: format_config_value(value)
                     for key, value in configs.items()}
    try:
        _resolve_config_overrides(raw_overrides, inspect_result)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SuiteError(f"case {case.id}: invalid config: {exc}") from exc

    known_routes = {route.name: route for route in inspect_result.routes}
    for key, value in routes.items():
        if key not in known_routes:
            raise SuiteError(
                f"case {case.id}: unknown route '{key}' "
                f"(available: {', '.join(sorted(known_routes)) or 'none'})")
        if value not in known_routes[key].options:
            raise SuiteError(
                f"case {case.id}: route '{key}' cannot be '{value}' "
                f"(available: {', '.join(known_routes[key].options)})")

    provided = set(case.inputs)
    for param in inspect_result.inputs:
        if param.required and param.name not in provided:
            raise SuiteError(
                f"case {case.id}: required input '{param.name}' is not "
                f"declared in the case")


def build_launcher_command(*, case: CaseSpec, pipeline_path: Path,
                           launcher_path: Path, python_bin: str,
                           log_path: Path, input_payloads: Mapping[str, str],
                           routes: Mapping[str, str],
                           configs: Mapping[str, Any]) -> list[str]:
    """Assemble the exact launcher argv used for a measured run."""
    command = [
        python_bin, "-u",
        str(launcher_path),
        str(pipeline_path),
        "--log-path", str(log_path),
        "--preflight", case.preflight,
        "--no-progress",
    ]
    for key, value in routes.items():
        command += ["--route", f"{key}={value}"]
    for key, value in configs.items():
        command += ["--config", f"{key}={format_config_value(value)}"]
    for key, payload in input_payloads.items():
        command += ["--input", f"{key}={payload}"]
    return command


# ────────────────────────────────────────────────────────────────────────────
# Machine / software context
# ────────────────────────────────────────────────────────────────────────────


def _cpu_model() -> dict[str, Any]:
    if sys.platform == "win32":
        try:
            import winreg
            key_path = (r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return {"model": str(name).strip() or None, "source": "winreg"}
        except (OSError, ImportError):
            pass
    elif sys.platform.startswith("linux"):
        try:
            text = Path("/proc/cpuinfo").read_text(encoding="utf-8",
                                                   errors="replace")
            for line in text.splitlines():
                if line.lower().startswith("model name"):
                    return {"model": line.split(":", 1)[1].strip(),
                            "source": "procfs"}
        except OSError:
            pass
    return {"model": platform.processor() or None, "source": "platform"}


def _query_nvidia_smi(timeout: float = 10.0) -> list[dict[str, Any]] | None:
    """Best-effort GPU names; the runtime probe already covers VRAM/CC."""
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,compute_cap,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True,
                                   encoding="utf-8", errors="replace",
                                   timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            index = int(parts[0])
        except ValueError:
            index = None
        try:
            memory_bytes: int | None = int(float(parts[2])) * 1024 * 1024
        except ValueError:
            memory_bytes = None
        rows.append({
            "index": index,
            "name": parts[1] or None,
            "memory_total_bytes": memory_bytes,
            "compute_capability": parts[3] or None,
            "driver_version": parts[4] or None,
        })
    return rows or None


def summarize_gpus(backend: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Describe GPUs from the structured runtime probe."""
    backend = backend or {}
    gpus: list[dict[str, Any]] = []
    cuda = backend.get("cuda_runtime") if isinstance(backend, dict) else None
    if isinstance(cuda, dict) and cuda.get("status") == "available":
        major = cuda.get("compute_capability_major")
        minor = cuda.get("compute_capability_minor")
        gpus.append({
            "index": cuda.get("device"),
            "name": None,
            "backend": "cuda",
            "memory_total_bytes": cuda.get("total_bytes"),
            "memory_free_at_probe_bytes": cuda.get("free_bytes"),
            "compute_capability": (f"{major}.{minor}"
                                   if major is not None and minor is not None
                                   else None),
            "driver_version": None,
            "source": "runtime",
        })
    metal = backend.get("metal_runtime") if isinstance(backend, dict) else None
    if isinstance(metal, dict) and metal.get("status") == "available":
        gpus.append({
            "index": None,
            "name": metal.get("device") or "metal",
            "backend": "metal",
            "memory_total_bytes": metal.get("recommended_max_working_set_bytes"),
            "memory_free_at_probe_bytes": None,
            "compute_capability": None,
            "driver_version": None,
            "source": "runtime",
        })
    return gpus


def probe_hardware(*, backend: Mapping[str, Any] | None = None,
                   query_gpu_names: bool = True) -> dict[str, Any]:
    """Describe the host: CPU, RAM, GPU (from the runtime probe), swap."""
    cpu = _cpu_model()
    physical = psutil.cpu_count(logical=False)
    logical = psutil.cpu_count(logical=True)
    freq = None
    try:
        freq_info = psutil.cpu_freq()
        freq = int(freq_info.max) if freq_info and freq_info.max else None
    except (OSError, NotImplementedError):  # pragma: no cover - platform
        freq = None
    gpus = summarize_gpus(backend)
    nvidia_rows = _query_nvidia_smi() if query_gpu_names else None
    if nvidia_rows:
        for row in nvidia_rows:
            match = next((gpu for gpu in gpus
                          if gpu.get("backend") == "cuda"
                          and gpu.get("index") == row["index"]), None)
            if match is None:
                gpus.append({
                    "index": row["index"], "name": row["name"],
                    "backend": "cuda",
                    "memory_total_bytes": row["memory_total_bytes"],
                    "memory_free_at_probe_bytes": None,
                    "compute_capability": row["compute_capability"],
                    "driver_version": row["driver_version"],
                    "source": "nvidia-smi",
                })
            else:
                match["name"] = row["name"]
                match["driver_version"] = row["driver_version"]
                if row["memory_total_bytes"] is not None:
                    match["memory_total_bytes"] = row["memory_total_bytes"]
                if row["compute_capability"]:
                    match["compute_capability"] = row["compute_capability"]
                match["source"] = "runtime+nvidia-smi"
    swap = psutil.swap_memory()
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "cpu": {
            "model": cpu["model"],
            "model_source": cpu["source"],
            "physical_cores": physical,
            "logical_cores": logical,
            "max_frequency_mhz": freq,
        },
        "ram_total_bytes": psutil.virtual_memory().total,
        "gpus": gpus,
        "swap_total_bytes": swap.total,
        "swap_used_at_probe_bytes": swap.used,
    }


def _git_metadata() -> dict[str, Any]:
    def _run(args: list[str]) -> str | None:
        try:
            completed = subprocess.run(
                ["git", "-c", f"safe.directory={PROJECT_ROOT.as_posix()}",
                 *args],
                cwd=PROJECT_ROOT, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=30)
        except (OSError, subprocess.SubprocessError):
            return None
        if completed.returncode != 0:
            return None
        return completed.stdout.strip()

    revision = _run(["rev-parse", "HEAD"])
    branch = _run(["rev-parse", "--abbrev-ref", "HEAD"])
    status = _run(["status", "--porcelain"])
    return {
        "git_revision": revision or None,
        "git_branch": branch or None,
        "git_dirty": bool(status) if status is not None else None,
    }


def _dependency_versions() -> dict[str, str | None]:
    from importlib import metadata

    packages = {
        "numpy": "numpy",
        "opencv": "opencv-python",
        "scipy": "scipy",
        "pywavelets": "PyWavelets",
        "rawpy": "rawpy",
        "tifffile": "tifffile",
        "psutil": "psutil",
        "loguru": "loguru",
        "networkx": "networkx",
    }
    versions: dict[str, str | None] = {}
    for label, package in packages.items():
        try:
            versions[label] = metadata.version(package)
        except Exception:  # pragma: no cover - depends on the environment
            versions[label] = None
    return versions


def probe_software() -> dict[str, Any]:
    """Describe the software under test: version, git state, dependencies."""
    from hoshicore.component.utils import RELEASE_NAME, VERSION

    build_info: dict[str, Any] | None
    try:
        from hoshicore._custom_op._dispatch import compiled_build_info
        build_info = compiled_build_info() or None
    except Exception as exc:  # pragma: no cover - optional native module
        build_info = {"status": "unavailable", "reason":
                      f"{type(exc).__name__}: {exc}"}

    payload = {
        "product": "HoshinoWeaver",
        "version": VERSION,
        "release_name": RELEASE_NAME,
        "dependencies": _dependency_versions(),
        "custom_ops_build": build_info,
    }
    payload.update(_git_metadata())
    return payload


def probe_backend(env: Mapping[str, str] | None = None,
                  *, timeout: float = 180.0) -> dict[str, Any]:
    """Structured backend report from a child process started with ``env``."""
    child_env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    if env:
        child_env.update({str(k): str(v) for k, v in env.items()})
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _BACKEND_PROBE_SNIPPET],
            cwd=PROJECT_ROOT, env=child_env, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"status": "probe_failed",
                "error": f"{type(exc).__name__}: {exc}"}
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(_BACKEND_PROBE_MARKER):
            raw = line[len(_BACKEND_PROBE_MARKER):]
            try:
                return json.loads(raw)
            except json.JSONDecodeError as exc:
                return {"status": "probe_failed",
                        "error": f"invalid probe JSON: {exc}"}
    tail = (completed.stderr or "").strip().splitlines()[-5:]
    return {"status": "probe_failed",
            "error": "probe produced no RESULT_JSON line",
            "stderr_tail": tail}


def probe_storage_paths(paths: Sequence[Path | None]) -> dict[str, Any]:
    """Free-space snapshot per path (no volume/media probing).

    Paths that do not exist yet resolve to their nearest existing parent, so a
    run-local cache/output directory still reports usable numbers.
    """
    import shutil

    result: dict[str, Any] = {}
    for index, path in enumerate(paths):
        key = str(path) if path is not None else f"path{index}"
        if path is None:
            result[key] = None
            continue
        candidate = path
        for _ in range(32):
            if candidate.exists():
                break
            parent = candidate.parent
            if parent == candidate:
                break
            candidate = parent
        try:
            usage = shutil.disk_usage(str(candidate))
        except OSError:
            result[key] = None
            continue
        result[key] = {
            "resolved_path": str(candidate),
            "total_bytes": usage.total,
            "free_bytes": usage.free,
        }
    return result


# ────────────────────────────────────────────────────────────────────────────
# Resource sampling
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class ResourceSamples:
    interval_seconds: float
    timestamps: list[float] = field(default_factory=list)
    process_cpu_percent: list[float] = field(default_factory=list)
    cpu_time_seconds: list[float] = field(default_factory=list)
    rss_bytes: list[int] = field(default_factory=list)
    pagefile_bytes: list[int] = field(default_factory=list)
    peak_wset_bytes: list[int] = field(default_factory=list)
    thread_count: list[int] = field(default_factory=list)
    process_count: list[int] = field(default_factory=list)
    system_mem_used_bytes: list[int] = field(default_factory=list)
    swap_used_bytes: list[int] = field(default_factory=list)
    gpu_mem_bytes: list[int] = field(default_factory=list)
    gpu_util_percent: list[float] = field(default_factory=list)
    gpu_available: bool | None = None


class ResourceSampler:
    """Sample a child process (plus its children) until :meth:`stop`."""

    def __init__(self, pid: int, *, interval_seconds: float,
                 gpu_interval_seconds: float = 0.0,
                 system_mem_used_before: int | None = None,
                 swap_used_before: int | None = None) -> None:
        self._pid = pid
        self._interval = max(0.05, float(interval_seconds))
        self._gpu_interval = max(0.0, float(gpu_interval_seconds))
        self.samples = ResourceSamples(interval_seconds=self._interval)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._children: dict[int, psutil.Process] = {}
        self._process: psutil.Process | None = None
        self._next_gpu_at = 0.0
        self._system_mem_used_before = system_mem_used_before
        self._swap_used_before = swap_used_before

    def start(self) -> None:
        try:
            self._process = psutil.Process(self._pid)
        except psutil.Error:  # pragma: no cover - process already gone
            self._process = None
        # Prime psutil's CPU delta and record the RSS/system baseline without
        # contributing psutil's always-zero first CPU reading to the average.
        self._sample_once(record_cpu=False)
        self._thread = threading.Thread(target=self._loop, name="hnw-sampler",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> ResourceSamples:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None
        self._sample_once()
        return self.samples

    # -- internals ---------------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            self._sample_once()

    def _sample_once(self, *, record_cpu: bool = True) -> None:
        process = self._process
        if process is None:
            return
        try:
            cpu_percent = process.cpu_percent(None)
            info = process.memory_info()
            rss = info.rss
            pagefile = getattr(info, "pagefile", None)
            peak_wset = getattr(info, "peak_wset", None)
            threads = process.num_threads()
            try:
                times = process.cpu_times()
                cpu_seconds = float(times.user + times.system)
            except psutil.Error:
                cpu_seconds = None
            process_count = 1
            for child in process.children(recursive=True):
                process_count += 1
                if child.pid not in self._children:
                    self._children[child.pid] = child
                    try:
                        child.cpu_percent(None)
                    except psutil.Error:
                        continue
                try:
                    cpu_percent += child.cpu_percent(None)
                    rss += child.memory_info().rss
                except psutil.Error:
                    self._children.pop(child.pid, None)
        except (psutil.Error, OSError):
            return

        now = time.monotonic()
        self.samples.timestamps.append(now)
        if record_cpu:
            self.samples.process_cpu_percent.append(float(cpu_percent))
        if cpu_seconds is not None:
            self.samples.cpu_time_seconds.append(cpu_seconds)
        self.samples.rss_bytes.append(int(rss))
        if pagefile is not None:
            self.samples.pagefile_bytes.append(int(pagefile))
        if peak_wset is not None:
            self.samples.peak_wset_bytes.append(int(peak_wset))
        self.samples.thread_count.append(int(threads))
        self.samples.process_count.append(int(process_count))
        self.samples.system_mem_used_bytes.append(
            psutil.virtual_memory().used)
        self.samples.swap_used_bytes.append(psutil.swap_memory().used)
        self._sample_gpu(now)

    def _sample_gpu(self, now: float) -> None:
        if self._gpu_interval <= 0 or self.samples.gpu_available is False:
            return
        if now < self._next_gpu_at:
            return
        self._next_gpu_at = now + self._gpu_interval
        command = ["nvidia-smi",
                   "--query-gpu=memory.used,utilization.gpu",
                   "--format=csv,noheader,nounits"]
        try:
            completed = subprocess.run(command, capture_output=True, text=True,
                                       encoding="utf-8", errors="replace",
                                       timeout=10)
        except (OSError, subprocess.SubprocessError):
            self.samples.gpu_available = False
            return
        if completed.returncode != 0 or not completed.stdout.strip():
            self.samples.gpu_available = False
            return
        self.samples.gpu_available = True
        mem_total = 0.0
        util_total = 0.0
        rows = 0
        for line in completed.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                mem_total += float(parts[0]) * 1024 * 1024
                util_total += float(parts[1])
            except ValueError:
                continue
            rows += 1
        if rows:
            self.samples.gpu_mem_bytes.append(int(mem_total))
            self.samples.gpu_util_percent.append(util_total / rows)


def summarize_resources(samples: ResourceSamples, *,
                        logical_cpu_count: int | None,
                        system_mem_used_before: int | None = None,
                        swap_used_before: int | None = None) -> dict[str, Any]:
    """Turn raw samples into the reported CPU/RAM/SWAP metrics."""

    def _mean(values: Sequence[float]) -> float | None:
        return statistics.fmean(values) if values else None

    def _peak(values: Sequence[float]) -> float | None:
        return max(values) if values else None

    cpu_percent_avg = _mean(samples.process_cpu_percent)
    normalized: float | None = None
    if cpu_percent_avg is not None and logical_cpu_count:
        normalized = cpu_percent_avg / logical_cpu_count
    cpu_time = max(samples.cpu_time_seconds) if samples.cpu_time_seconds else None
    if cpu_time is None and samples.process_cpu_percent:
        cpu_time = sum(value / 100.0 * samples.interval_seconds
                       for value in samples.process_cpu_percent)

    rss_first = samples.rss_bytes[0] if samples.rss_bytes else None
    rss_avg = _mean(samples.rss_bytes)
    rss_peak = _peak(samples.rss_bytes)
    delta_avg = (rss_avg - rss_first
                 if rss_avg is not None and rss_first is not None else None)
    delta_peak = (rss_peak - rss_first
                  if rss_peak is not None and rss_first is not None else None)

    system_delta_peak = None
    if system_mem_used_before is not None and samples.system_mem_used_bytes:
        system_delta_peak = (max(samples.system_mem_used_bytes)
                             - system_mem_used_before)
    swap_delta_peak = None
    if swap_used_before is not None and samples.swap_used_bytes:
        swap_delta_peak = (max(samples.swap_used_bytes) - swap_used_before)

    gpu: dict[str, Any] = {"sampled": bool(samples.gpu_mem_bytes)}
    if samples.gpu_mem_bytes:
        gpu["mem_peak_bytes"] = max(samples.gpu_mem_bytes)
        gpu["mem_avg_bytes"] = _mean(samples.gpu_mem_bytes)
    if samples.gpu_util_percent:
        gpu["util_avg_percent"] = _mean(samples.gpu_util_percent)
        gpu["util_peak_percent"] = _peak(samples.gpu_util_percent)

    return {
        "sample_interval_seconds": samples.interval_seconds,
        "sample_count": len(samples.timestamps),
        "logical_cpu_count": logical_cpu_count,
        "cpu_percent_avg": cpu_percent_avg,
        "cpu_percent_peak": _peak(samples.process_cpu_percent),
        "cpu_percent_avg_normalized": normalized,
        "cpu_time_seconds": cpu_time,
        "rss_baseline_bytes": rss_first,
        "rss_avg_bytes": rss_avg,
        "rss_peak_bytes": rss_peak,
        "rss_delta_avg_bytes": delta_avg,
        "rss_delta_peak_bytes": delta_peak,
        "peak_wset_bytes": _peak(samples.peak_wset_bytes),
        "pagefile_avg_bytes": _mean(samples.pagefile_bytes),
        "pagefile_peak_bytes": _peak(samples.pagefile_bytes),
        "thread_count_peak": _peak(samples.thread_count),
        "process_count_peak": _peak(samples.process_count),
        "system_mem_used_before_bytes": system_mem_used_before,
        "system_mem_used_after_bytes": (samples.system_mem_used_bytes[-1]
                                        if samples.system_mem_used_bytes
                                        else None),
        "system_mem_used_peak_bytes": _peak(samples.system_mem_used_bytes),
        "system_mem_delta_peak_bytes": system_delta_peak,
        "swap_total_bytes": psutil.swap_memory().total,
        "swap_used_before_bytes": swap_used_before,
        "swap_used_after_bytes": (samples.swap_used_bytes[-1]
                                  if samples.swap_used_bytes else None),
        "swap_used_peak_bytes": _peak(samples.swap_used_bytes),
        "swap_used_delta_peak_bytes": swap_delta_peak,
        "gpu": gpu,
    }


# ────────────────────────────────────────────────────────────────────────────
# Log parsing
# ────────────────────────────────────────────────────────────────────────────


def _log_timestamp(line: str) -> float | None:
    match = _LOG_TIMESTAMP.match(line)
    if not match:
        return None
    try:
        parsed = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        return None
    return parsed.timestamp()


def parse_log(log_path: Path) -> dict[str, Any]:
    """Extract coarse phase boundaries from the launcher log.

    Per-node ``_async_execute`` lines are intentionally not used as stage
    timings: in a streaming DAG a node is started early and then blocks on its
    upstream, so its wall time is not a comparable stage cost.
    """
    if not log_path.is_file():
        return {"source": "unavailable", "phases": {}, "preflight_lines": [],
                "capabilities_lines": []}
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {"source": "unavailable", "phases": {}, "preflight_lines": [],
                "capabilities_lines": []}

    pipeline_seconds: float | None = None
    dag_start: float | None = None
    dag_done: float | None = None
    feeder_items: dict[str, int] = {}
    preflight_lines: list[str] = []
    capabilities_lines: list[str] = []
    for line in text.splitlines():
        match = _LOG_RUN_FROM_YAML.search(line)
        if match:
            pipeline_seconds = float(match.group(1))
        match = _LOG_FEEDER.search(line)
        if match:
            feeder_items[match.group(1)] = int(match.group(2))
        if _PREFLIGHT_MARK in line:
            preflight_lines.append(line.strip())
            continue
        if _CAPABILITIES_MARK in line:
            capabilities_lines.append(line.strip())
            continue
        if _DAG_START in line:
            stamp = _log_timestamp(line)
            if stamp is not None and dag_start is None:
                dag_start = stamp
        elif _DAG_DONE in line:
            stamp = _log_timestamp(line)
            if stamp is not None:
                dag_done = stamp

    execution_seconds = None
    if dag_start is not None and dag_done is not None and dag_done >= dag_start:
        execution_seconds = dag_done - dag_start

    phases = {
        "pipeline_seconds": pipeline_seconds,
        "execution_seconds": execution_seconds,
        "feeder_items": feeder_items,
    }
    source = "log" if any(value is not None for value in
                          (pipeline_seconds, execution_seconds)) else \
        "unavailable"
    return {"source": source, "phases": phases,
            "preflight_lines": preflight_lines,
            "capabilities_lines": capabilities_lines}


def parse_printed_log_path(stdout_path: Path) -> str | None:
    """Read the single ``[Launcher] log file:`` line from captured stdout."""
    if not stdout_path.is_file():
        return None
    try:
        text = stdout_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    found: str | None = None
    for line in text.splitlines():
        if line.startswith(LAUNCHER_LOG_MARKER):
            found = line[len(LAUNCHER_LOG_MARKER):].strip()
    return found or None


# ────────────────────────────────────────────────────────────────────────────
# Planning and execution
# ────────────────────────────────────────────────────────────────────────────


def _suite_relative(path_value: str, suite_path: Path) -> Path:
    """Resolve a suite-declared asset path.

    Absolute paths win; everything else resolves against the repository root so
    a suite stored in ``benchmarks/local/`` can keep using repo-relative
    pipeline paths. ``suite_path`` is accepted so the helper stays explicit
    about which suite declared the path.
    """
    del suite_path
    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def _resolve_optional_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


@dataclass(frozen=True)
class PlannedRun:
    case: CaseSpec
    run_index: int
    pipeline_path: Path
    launcher_path: Path
    python_bin: str
    working_dir: Path
    paths: Mapping[str, Path]
    configs: Mapping[str, Any]
    inputs: ResolvedInputs
    command: tuple[str, ...]
    expected_outputs: tuple[Path, ...]
    input_payloads: Mapping[str, str]


def plan_case(case: CaseSpec, *, suite_path: Path, output_dir: Path,
              python_bin: str | None = None,
              launcher_path: Path | None = None,
              working_dir: Path | None = None,
              run_index: int = 0,
              require_inputs: bool = True) -> PlannedRun:
    """Resolve a case into the exact launcher invocation (no side effects)."""
    pipeline_path = _suite_relative(case.pipeline, suite_path)
    if working_dir is not None:
        working = working_dir
    else:
        working = _resolve_optional_path(case.working_dir) or PROJECT_ROOT
    launcher = launcher_path or _suite_relative(case.launcher, suite_path)
    resolution = resolve_inputs(case, require_exists=require_inputs)
    paths = case_paths(case, run_index, output_dir=output_dir,
                       working_dir=working)
    configs = effective_configs(case, paths, run_index)
    payloads = {
        entry.name: f"@{paths['run_dir'] / ('inputs_' + entry.name + '.txt')}"
        for entry in resolution.entries
    }
    command = build_launcher_command(
        case=case, pipeline_path=pipeline_path, launcher_path=launcher,
        python_bin=python_bin or case.python or sys.executable,
        log_path=paths["run_dir"] / "launcher.log",
        input_payloads=payloads, routes=case.routes, configs=configs)
    mapping = {
        "case_id": case.id,
        "run_index": str(run_index),
        "output_root": str(paths["output_root"]),
        "cache_root": str(paths["cache_root"]),
        "run_dir": str(paths["run_dir"]),
    }
    expected_values = (case.expected_outputs or
                       ((str(configs["output_filename"]),)
                        if case.expect_success else ()))
    expected = tuple(
        path if path.is_absolute() else working / path
        for path in (Path(substitute_placeholders(item, mapping))
                     for item in expected_values))
    return PlannedRun(
        case=case, run_index=run_index, pipeline_path=pipeline_path,
        launcher_path=launcher,
        python_bin=python_bin or case.python or sys.executable,
        working_dir=working, paths=paths, configs=configs,
        inputs=resolution, command=tuple(command), expected_outputs=expected,
        input_payloads=payloads)


def materialize_run(planned: PlannedRun) -> None:
    """Create every directory and frozen input list file for a run."""
    paths = planned.paths
    for key in ("case_dir", "run_dir", "output_root"):
        paths[key].mkdir(parents=True, exist_ok=True)
    if planned.configs.get("temp_path"):
        Path(str(planned.configs["temp_path"])).mkdir(parents=True,
                                                      exist_ok=True)
    for entry in planned.inputs.entries:
        list_path = paths["run_dir"] / f"inputs_{entry.name}.txt"
        body = "".join(f"{item}\n" for item in entry.files)
        list_path.write_text(body, encoding="utf-8")


def _sha1_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha1()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def collect_output_files(expected: Sequence[Path]) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in expected:
        exists = path.is_file()
        entry: dict[str, Any] = {
            "path": str(path),
            "exists": exists,
            "bytes": None,
            "sha1": None,
        }
        if exists:
            try:
                entry["bytes"] = path.stat().st_size
            except OSError:
                entry["bytes"] = None
            entry["sha1"] = _sha1_file(path)
        files.append(entry)
    return files


def _terminate_tree(pid: int) -> None:
    try:
        process = psutil.Process(pid)
    except psutil.Error:
        return
    targets = []
    try:
        targets = process.children(recursive=True)
    except psutil.Error:
        targets = []
    for child in targets:
        try:
            child.terminate()
        except psutil.Error:
            continue
    try:
        process.terminate()
    except psutil.Error:
        pass
    _gone, alive = psutil.wait_procs([*targets, process], timeout=10)
    for remaining in alive:
        try:
            remaining.kill()
        except psutil.Error:
            continue


def run_case(planned: PlannedRun, *,
             timeout_seconds: float | None = None,
             measure_interval_seconds: float | None = None,
             gpu_sample_interval_seconds: float | None = None) -> dict[str, Any]:
    """Execute one planned run and return its structured result."""
    case = planned.case
    timeout = float(case.timeout_seconds if timeout_seconds is None
                    else timeout_seconds)
    interval = float(case.measure_interval_seconds
                     if measure_interval_seconds is None
                     else measure_interval_seconds)
    gpu_interval = float(case.gpu_sample_interval_seconds
                         if gpu_sample_interval_seconds is None
                         else gpu_sample_interval_seconds)

    paths = planned.paths
    log_path = paths["run_dir"] / "launcher.log"
    stdout_path = paths["run_dir"] / "stdout.txt"
    stderr_path = paths["run_dir"] / "stderr.txt"

    warnings: list[str] = list(planned.inputs.warnings)
    result: dict[str, Any] = {
        "case_id": case.id,
        "run_index": planned.run_index,
        "description": case.description,
        "command": list(planned.command),
        "cwd": str(planned.working_dir),
        "status": "launch_error",
        "expect_success": case.expect_success,
        "expectation_met": False,
        "outputs_met": None,
        "wall_seconds": None,
        "exit_code": None,
        "started_at": datetime.now().astimezone().isoformat(),
        "phases": {},
        "phases_source": "unavailable",
        "resources": {},
        "log_path": str(log_path),
        "log_path_printed": None,
        "preflight_policy": case.preflight,
        "preflight_log_lines": [],
        "capabilities_log_lines": [],
        "output_files": [],
        "warnings": warnings,
    }

    if not planned.pipeline_path.is_file():
        warnings.append(f"pipeline YAML not found: {planned.pipeline_path}")
        result["warnings"] = warnings
        return result
    if not Path(planned.launcher_path).is_file():
        warnings.append(f"launcher not found: {planned.launcher_path}")
        result["warnings"] = warnings
        return result
    if planned.inputs.input_count == 0:
        result["status"] = "invalid_input"
        result["warnings"] = warnings + ["no input files resolved"]
        return result

    materialize_run(planned)

    system_mem_before = psutil.virtual_memory().used
    swap_before = psutil.swap_memory().used
    child_env = {**os.environ, "PYTHONIOENCODING": "utf-8",
                 "PYTHONUNBUFFERED": "1"}
    child_env.update({str(k): str(v) for k, v in case.env.items()})

    sampler: ResourceSampler | None = None
    started = time.perf_counter()
    with open(stdout_path, "wb") as stdout_handle, \
            open(stderr_path, "wb") as stderr_handle:
        try:
            process = subprocess.Popen(
                list(planned.command), cwd=str(planned.working_dir),
                env=child_env, stdout=stdout_handle, stderr=stderr_handle,
                stdin=subprocess.DEVNULL)
        except OSError as exc:
            result["wall_seconds"] = time.perf_counter() - started
            warnings.append(f"failed to start launcher: {exc}")
            result["warnings"] = warnings
            return result

        sampler = ResourceSampler(process.pid, interval_seconds=interval,
                                  gpu_interval_seconds=gpu_interval,
                                  system_mem_used_before=system_mem_before,
                                  swap_used_before=swap_before)
        sampler.start()
        timed_out = False
        try:
            exit_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_tree(process.pid)
            try:
                exit_code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                exit_code = None
        wall_seconds = time.perf_counter() - started
        samples = sampler.stop()

    result["wall_seconds"] = wall_seconds
    result["exit_code"] = exit_code
    result["resources"] = summarize_resources(
        samples,
        logical_cpu_count=psutil.cpu_count(logical=True),
        system_mem_used_before=system_mem_before,
        swap_used_before=swap_before)
    result["env"] = dict(case.env)
    result["env_effective_hnw"] = {
        key: value for key, value in child_env.items()
        if key.startswith("HNW_")}

    log_report = parse_log(log_path)
    result["phases"] = log_report["phases"]
    result["phases_source"] = log_report["source"]
    result["preflight_log_lines"] = log_report["preflight_lines"]
    result["capabilities_log_lines"] = log_report["capabilities_lines"]

    printed = parse_printed_log_path(stdout_path)
    result["log_path_printed"] = printed
    if printed is None:
        warnings.append("launcher did not print a log path line")
    elif Path(printed) != log_path:
        warnings.append(
            f"printed log path differs from --log-path "
            f"({printed} != {log_path})")

    result["output_files"] = collect_output_files(planned.expected_outputs)
    if planned.expected_outputs:
        result["outputs_met"] = all(entry["exists"]
                                    for entry in result["output_files"])

    if timed_out:
        result["status"] = "timeout"
    elif exit_code == 0:
        result["status"] = "success"
    else:
        result["status"] = "pipeline_failed"
    result["expectation_met"] = bool(
        (result["status"] == "success") == case.expect_success
        and (result["outputs_met"] is not False))
    result["warnings"] = warnings
    return result


def summarize_run_results(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate repeated runs of one case into headline metrics."""
    def _values(field_name: str) -> list[float]:
        return [float(run[field_name]) for run in runs
                if isinstance(run.get(field_name), (int, float))]

    def _resource_values(key: str) -> list[float]:
        values: list[float] = []
        for run in runs:
            payload = run.get("resources")
            if not isinstance(payload, dict):
                continue
            value = payload.get(key)
            if isinstance(value, (int, float)):
                values.append(float(value))
        return values

    def _stats(values: Sequence[float]) -> dict[str, Any] | None:
        if not values:
            return None
        ordered = sorted(values)
        return {
            "count": len(ordered),
            "min": ordered[0],
            "max": ordered[-1],
            "mean": statistics.fmean(ordered),
            "median": statistics.median(ordered),
        }

    success_count = sum(1 for run in runs if run.get("status") == "success")
    expectations = [run.get("expectation_met") for run in runs]
    return {
        "runs": len(runs),
        "success_count": success_count,
        "success_all": success_count == len(runs) and bool(runs),
        "expectations_met": all(bool(value) for value in expectations)
        if expectations else False,
        "wall_seconds": _stats(_values("wall_seconds")),
        "resources": {
            "cpu_time_seconds": _stats(_resource_values("cpu_time_seconds")),
            "cpu_percent_avg_normalized": _stats(
                _resource_values("cpu_percent_avg_normalized")),
            "rss_peak_bytes": _stats(_resource_values("rss_peak_bytes")),
            "peak_wset_bytes": _stats(_resource_values("peak_wset_bytes")),
        },
        "output_bytes": _stats([
            float(entry["bytes"]) for run in runs
            for entry in run.get("output_files", [])
            if isinstance(entry, dict)
            and isinstance(entry.get("bytes"), (int, float))]),
    }
