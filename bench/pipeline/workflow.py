"""End-to-end YAML/DAG workflow benchmark.

This suite intentionally calls the engine entry point instead of individual
component helpers. It measures meta/flatten/preflight/runtime plan/executor for
shipped stacker DAG workflows. Input image loading is done before timing so the
reported totals focus on engine compute behavior, not file I/O. The default
``auto`` backend preference is the production integration benchmark. ``cpu``
and ``numpy`` are intended for focused fallback validation rather than a full
backend performance matrix.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger

from bench.common import (
    PROJECT_ROOT,
    collect_env_info,
    discover_image_paths,
    load_benchmark_image,
    load_frames_from_paths,
    prepare_frames,
    print_or_save_report,
    summarize_samples,
)
from hoshicore._custom_op import build_info as custom_ops_build_info
from hoshicore.component.data_container import FloatImage
from hoshicore.component.image_io import peek_shape
from hoshicore.engine.wiring import run_from_yaml

import hoshicore.ops  # noqa: F401  # ensure built-in ops are registered


SUITE_ID = "pipeline.workflow"


@dataclass(frozen=True)
class WorkflowSpec:
    yaml_key: str
    route_choices: dict[str, str]
    configs: dict[str, Any] = field(default_factory=dict)
    input_kind: str = "memory_frames"
    requires_output: bool = False
    requires_mask: bool = False


WORKFLOWS: dict[str, WorkflowSpec] = {
    "stack_mean": WorkflowSpec(
        yaml_key="stacker",
        route_choices={"stacker": "mean"},
    ),
    "stack_median": WorkflowSpec(
        yaml_key="stacker",
        route_choices={"stacker": "median"},
    ),
    "stack_sigma_clip": WorkflowSpec(
        yaml_key="stacker",
        route_choices={"stacker": "sigma_clip"},
        configs={"rej_high": 3.0, "rej_low": 3.0, "max_iter": 5},
    ),
    "stack_huber_mean": WorkflowSpec(
        yaml_key="stacker",
        route_choices={"stacker": "huber_mean"},
        configs={"huber_c": 1.345},
    ),
    "stack_max": WorkflowSpec(
        yaml_key="stacker",
        route_choices={"stacker": "max"},
    ),
    "stack_file_mean": WorkflowSpec(
        yaml_key="stack",
        route_choices={"stacker": "mean"},
        input_kind="fnames",
        requires_output=True,
    ),
    "stack_file_median": WorkflowSpec(
        yaml_key="stack",
        route_choices={"stacker": "median"},
        input_kind="fnames",
        requires_output=True,
    ),
    "stack_file_sigma_clip": WorkflowSpec(
        yaml_key="stack",
        route_choices={"stacker": "sigma_clip"},
        configs={"rej_high": 3.0, "rej_low": 3.0, "max_iter": 5},
        input_kind="fnames",
        requires_output=True,
    ),
    "stack_file_huber_mean": WorkflowSpec(
        yaml_key="stack",
        route_choices={"stacker": "huber_mean"},
        configs={"huber_c": 1.345},
        input_kind="fnames",
        requires_output=True,
    ),
    "startrail_fifo": WorkflowSpec(
        yaml_key="startrail",
        route_choices={"mode": "fifo"},
        input_kind="fnames",
        requires_output=True,
    ),
    "startrail_mix": WorkflowSpec(
        yaml_key="startrail",
        route_choices={"mode": "mix"},
        configs={"rej_high": 3.0, "rej_low": 3.0, "max_iter": 5},
        input_kind="fnames",
        requires_output=True,
    ),
    "calibration_mean": WorkflowSpec(
        yaml_key="calibration_stack",
        route_choices={
            "bias_stacker": "none",
            "dark_stacker": "none",
            "flat_stacker": "none",
            "main_stacker": "mean",
        },
        input_kind="calibration_light",
        requires_output=True,
    ),
    "calibration_median": WorkflowSpec(
        yaml_key="calibration_stack",
        route_choices={
            "bias_stacker": "none",
            "dark_stacker": "none",
            "flat_stacker": "none",
            "main_stacker": "median",
        },
        input_kind="calibration_light",
        requires_output=True,
    ),
    "calibration_sigma_clip": WorkflowSpec(
        yaml_key="calibration_stack",
        route_choices={
            "bias_stacker": "none",
            "dark_stacker": "none",
            "flat_stacker": "none",
            "main_stacker": "sigma_clip",
        },
        configs={"rej_high": 3.0, "rej_low": 3.0, "max_iter": 5},
        input_kind="calibration_light",
        requires_output=True,
    ),
    "calibration_huber_mean": WorkflowSpec(
        yaml_key="calibration_stack",
        route_choices={
            "bias_stacker": "none",
            "dark_stacker": "none",
            "flat_stacker": "none",
            "main_stacker": "huber_mean",
        },
        configs={"huber_c": 1.345},
        input_kind="calibration_light",
        requires_output=True,
    ),
    "sky_ground_mean": WorkflowSpec(
        yaml_key="sky_ground_stack",
        route_choices={"sky_stacker": "mean", "ground_stacker": "max"},
        configs={
            "align_method": "homography",
            "same_camera": True,
            "enable_ground": True,
        },
        input_kind="fnames",
        requires_output=True,
        requires_mask=True,
    ),
}

CASE_NAMES = list(WORKFLOWS)
DEFAULT_CASES = [
    case_name for case_name in CASE_NAMES
    if case_name not in {"sky_ground_mean"}
]


def parse_cases(raw: str) -> list[str]:
    if raw == "all":
        return list(DEFAULT_CASES)
    if raw == "everything":
        return list(CASE_NAMES)
    return [item.strip() for item in raw.split(",") if item.strip()]


@contextmanager
def _custom_op_backend(preference: str):
    old_value = os.environ.get("HNW_CUSTOM_OPS_FALLBACK")
    if preference == "auto":
        os.environ.pop("HNW_CUSTOM_OPS_FALLBACK", None)
    else:
        os.environ["HNW_CUSTOM_OPS_FALLBACK"] = preference
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop("HNW_CUSTOM_OPS_FALLBACK", None)
        else:
            os.environ["HNW_CUSTOM_OPS_FALLBACK"] = old_value


def _validate_image_paths(paths: list[Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("workflow benchmark requires at least one input image")

    first_image = load_benchmark_image(paths[0])
    first_shape, first_dtype_bytes = peek_shape(str(paths[0]))
    for idx, path in enumerate(paths[1:], start=1):
        shape, dtype_bytes = peek_shape(str(path))
        if tuple(shape) != tuple(first_shape):
            raise ValueError(
                f"benchmark image shape mismatch: frame 0 {first_shape} vs "
                f"frame {idx} {shape}"
            )
        if dtype_bytes != first_dtype_bytes:
            raise ValueError(
                f"benchmark image dtype size mismatch: frame 0 {first_dtype_bytes} "
                f"vs frame {idx} {dtype_bytes}"
            )

    return {
        "mode": "images",
        "resolved_frames": len(paths),
        "resolved_shape": list(first_image.shape),
        "resolved_dtype": str(first_image.dtype),
        "sample_paths": [str(path) for path in paths[:3]],
    }


def _write_materialized_frames(frames: list[np.ndarray], output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fnames: list[str] = []
    for idx, frame in enumerate(frames):
        path = output_dir / f"frame_{idx:05d}.tif"
        ok = cv2.imwrite(str(path), frame)
        if not ok:
            raise RuntimeError(f"failed to materialize benchmark frame: {path}")
        fnames.append(str(path))
    return fnames


def _write_sky_ground_mask(frame: np.ndarray, path: Path) -> str:
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[: max(1, int(h * 0.65)), :] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), mask)
    if not ok:
        raise RuntimeError(f"failed to write workflow mask: {path}")
    return str(path)


def _prepare_workflow_inputs(
    args: argparse.Namespace,
    work_dir: Path,
) -> tuple[list[np.ndarray], list[str], dict[str, Any]]:
    if args.input_mode in {"auto", "cache"}:
        try:
            frames, source = prepare_frames(
                frames=args.frames,
                height=args.height,
                width=args.width,
                dtype=np.dtype(args.dtype),
                channels=args.channels,
                seed=args.seed,
                input_dir=args.input_dir,
                input_mode="cache",
            )
        except FileNotFoundError:
            if args.input_mode == "cache":
                raise
        else:
            materialized_dir = work_dir / "inputs"
            fnames = _write_materialized_frames(frames, materialized_dir)
            source = dict(source)
            source["engine_input"] = "in_memory_frames"
            source["engine_input_dir"] = str(materialized_dir)
            return frames, fnames, source

    if args.input_mode in {"auto", "images"}:
        root, image_paths = discover_image_paths(
            input_dir=args.input_dir,
            frames=args.frames,
        )
        if image_paths:
            source = _validate_image_paths(image_paths)
            source.update({
                "input_dir": str(root),
                "requested_input_mode": args.input_mode,
                "engine_input": "in_memory_frames",
            })
            return (
                load_frames_from_paths(image_paths),
                [str(path) for path in image_paths],
                source,
            )
        if args.input_mode == "images":
            raise FileNotFoundError(
                f"no image dataset found for frames={args.frames} under: "
                f"{args.input_dir or 'benchmark data dirs'}"
            )
        if args.input_dir is not None:
            raise FileNotFoundError(
                f"no raw cache or image dataset found for frames={args.frames} "
                f"under: {args.input_dir}"
            )

    frames, source = prepare_frames(
        frames=args.frames,
        height=args.height,
        width=args.width,
        dtype=np.dtype(args.dtype),
        channels=args.channels,
        seed=args.seed,
        input_dir=args.input_dir,
        input_mode=args.input_mode,
    )
    materialized_dir = work_dir / "inputs"
    fnames = _write_materialized_frames(frames, materialized_dir)
    source = dict(source)
    source["engine_input"] = "in_memory_frames"
    source["engine_input_dir"] = str(materialized_dir)
    return frames, fnames, source


def _workflow_global_inputs(
    spec: WorkflowSpec,
    *,
    frames: list[np.ndarray],
    fnames: list[str],
) -> dict[str, Any]:
    if spec.input_kind == "memory_frames":
        return {"data": frames, "fnames": fnames}
    if spec.input_kind == "fnames":
        return {"fnames": fnames}
    if spec.input_kind == "calibration_light":
        # 额外传 fnames 只服务 runtime planner 的统一 shape peek；DAG 本身消费 light_fnames。
        return {"light_fnames": fnames, "fnames": fnames}
    raise ValueError(f"unsupported workflow input kind: {spec.input_kind}")


async def _run_workflow_async(
    spec: WorkflowSpec,
    *,
    yaml_path: Path,
    frames: list[np.ndarray],
    fnames: list[str],
    temp_dir: Path,
    args: argparse.Namespace,
    input_source: dict[str, Any],
) -> dict[str, Any]:
    configs: dict[str, Any] = {
        "runtime_planner": True,
        # 保持基准指定的缓冲策略，不应用资源自动降级。
        "auto_fallback": False,
        "temp_path": str(temp_dir),
    }
    if args.buffer_mode != "auto":
        configs["buffer_mode"] = args.buffer_mode
    if spec.requires_output:
        configs["output_filename"] = str(temp_dir / "output.tif")
        configs["output_dtype"] = str(frames[0].dtype)
    if spec.requires_mask:
        configs["mask"] = _write_sky_ground_mask(frames[0], temp_dir / "mask.tif")
    configs.update(spec.configs)

    result = await run_from_yaml(
        str(yaml_path),
        global_inputs=_workflow_global_inputs(spec, frames=frames, fnames=fnames),
        global_configs=configs,
        progress=False,
        route_choices=spec.route_choices,
    )
    output = result.get("result")
    if isinstance(output, FloatImage):
        output_shape = list(output.data.shape)
        output_dtype = str(output.dtype)
        output_container = "FloatImage"
    elif isinstance(output, np.ndarray):
        output_shape = list(output.shape)
        output_dtype = str(output.dtype)
        output_container = "ndarray"
    elif isinstance(output, (int, np.integer)):
        output_shape = None
        output_dtype = "int"
        output_container = "return_code"
    else:
        raise RuntimeError(
            f"workflow returned unsupported result type: {type(output)!r}"
        )
    payload = {
        "output_shape": output_shape,
        "output_dtype": output_dtype,
        "output_container": output_container,
        "yaml": str(yaml_path),
        "route_choices": dict(spec.route_choices),
    }
    if spec.requires_output:
        output_filename = configs["output_filename"]
        payload["return_code"] = int(output)
        payload["output_file"] = output_filename
        payload["output_file_exists"] = Path(output_filename).exists()
    return payload


def _run_workflow_once(
    spec: WorkflowSpec,
    *,
    yaml_path: Path,
    frames: list[np.ndarray],
    fnames: list[str],
    temp_dir: Path,
    args: argparse.Namespace,
    input_source: dict[str, Any],
) -> dict[str, Any]:
    return asyncio.run(
        _run_workflow_async(
            spec,
            yaml_path=yaml_path,
            frames=frames,
            fnames=fnames,
            temp_dir=temp_dir,
            args=args,
            input_source=input_source,
        )
    )


def _time_workflow(
    case_name: str,
    spec: WorkflowSpec,
    *,
    yaml_path: Path,
    frames: list[np.ndarray],
    fnames: list[str],
    temp_root: Path,
    args: argparse.Namespace,
    input_source: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    case_temp_dir = temp_root / case_name
    case_temp_dir.mkdir(parents=True, exist_ok=True)

    last_payload: dict[str, Any] = {}
    for idx in range(args.warmup):
        last_payload = _run_workflow_once(
            spec,
            yaml_path=yaml_path,
            frames=frames,
            fnames=fnames,
            temp_dir=case_temp_dir,
            args=args,
            input_source=input_source,
        )

    samples: list[float] = []
    for idx in range(args.repeat):
        t0 = time.perf_counter()
        last_payload = _run_workflow_once(
            spec,
            yaml_path=yaml_path,
            frames=frames,
            fnames=fnames,
            temp_dir=case_temp_dir,
            args=args,
            input_source=input_source,
        )
        samples.append(time.perf_counter() - t0)

    result = summarize_samples(samples)
    result.update({
        "yaml": str(yaml_path),
        "route_choices": dict(spec.route_choices),
    })
    return result, last_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark shipped YAML/DAG workflows end to end. Auto is the "
            "production default; cpu and numpy are fallback integration checks."
        )
    )
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--dtype", type=str, default="uint16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument(
        "--input-mode",
        choices=["auto", "cache", "images", "synthetic"],
        default="auto",
    )
    parser.add_argument(
        "--cases",
        type=str,
        default="all",
        help="Comma-separated workflows, 'all' for default stable workflows, or 'everything'.",
    )
    parser.add_argument(
        "--buffer-mode",
        choices=["auto", "memory", "disk"],
        default="memory",
        help="Buffer mode passed to DAG configs. 'auto' uses project defaults.",
    )
    parser.add_argument(
        "--backend",
        dest="backend",
        choices=["auto", "cpu", "numpy"],
        default="auto",
        help=(
            "Backend preference for DAG integration: auto follows production "
            "dispatch (default); cpu disables CUDA; numpy forces custom-op "
            "fallbacks. Use pipeline.compute for detailed backend comparisons."
        ),
    )
    parser.add_argument(
        "--custom-op-backend",
        dest="backend",
        choices=["auto", "cpu", "numpy"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--log-level", type=str, default="WARNING")
    parser.add_argument(
        "--work-dir",
        type=str,
        default=None,
        help="Directory for benchmark inputs/outputs. Defaults to a temporary directory.",
    )
    parser.add_argument("--keep-work-dir", action="store_true")
    parser.add_argument("--output-json", type=str, default=None)
    return parser


@contextmanager
def _work_dir_context(args: argparse.Namespace):
    if args.work_dir:
        path = Path(args.work_dir)
        path.mkdir(parents=True, exist_ok=True)
        yield path
        return
    if args.keep_work_dir:
        path = Path(tempfile.mkdtemp(prefix="hnw-workflow-bench-"))
        yield path
        return
    temp = tempfile.TemporaryDirectory(prefix="hnw-workflow-bench-")
    try:
        yield Path(temp.name)
    finally:
        temp.cleanup()


def run(args: argparse.Namespace) -> dict[str, object]:
    logger.remove()
    logger.add(sys.stderr, level=args.log_level.upper())
    if args.repeat <= 0:
        raise ValueError("--repeat must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")

    requested_cases = parse_cases(args.cases)
    unknown = sorted(set(requested_cases) - set(CASE_NAMES))
    if unknown:
        raise ValueError(f"Unknown workflow case(s): {unknown}. Available: {CASE_NAMES}")

    with _work_dir_context(args) as work_dir:
        frames, fnames, input_source = _prepare_workflow_inputs(args, work_dir)
        yaml_paths = {
            "stacker": (
                PROJECT_ROOT / "hoshicore" / "dag" / "base" / "stacker.meta.yaml"
            ),
            "stack": PROJECT_ROOT / "hoshicore" / "dag" / "stack.meta.yaml",
            "startrail": PROJECT_ROOT / "hoshicore" / "dag" / "startrail.meta.yaml",
            "calibration_stack": (
                PROJECT_ROOT / "hoshicore" / "dag" / "calibration_stack.meta.yaml"
            ),
            "sky_ground_stack": (
                PROJECT_ROOT / "hoshicore" / "dag" / "sky_ground_stack.meta.yaml"
            ),
        }
        temp_root = work_dir / "temp"
        temp_root.mkdir(parents=True, exist_ok=True)

        results: dict[str, dict[str, Any]] = {}
        pipelines: dict[str, dict[str, Any]] = {}
        with _custom_op_backend(args.backend):
            for case_name in requested_cases:
                result, payload = _time_workflow(
                    case_name,
                    WORKFLOWS[case_name],
                    yaml_path=yaml_paths[WORKFLOWS[case_name].yaml_key],
                    frames=frames,
                    fnames=fnames,
                    temp_root=temp_root,
                    args=args,
                    input_source=input_source,
                )
                results[case_name] = result
                pipelines[case_name] = payload

        return {
            "suite": SUITE_ID,
            "benchmark_scope": "dag_workflow_end_to_end",
            "env": collect_env_info(),
            "custom_ops": custom_ops_build_info(),
            "config": {
                "frames": args.frames,
                "height": args.height,
                "width": args.width,
                "channels": args.channels,
                "dtype": args.dtype,
                "seed": args.seed,
                "input_dir": args.input_dir,
                "input_mode": args.input_mode,
                "cases": requested_cases,
                "buffer_mode": args.buffer_mode,
                "backend": args.backend,
                "backend_preference": args.backend,
                "warmup": args.warmup,
                "repeat": args.repeat,
                "log_level": args.log_level,
                "work_dir": str(work_dir),
                "keep_work_dir": args.keep_work_dir,
            },
            "input_source": input_source,
            "results": results,
            "pipelines": pipelines,
            "terminal_cases": requested_cases,
            "terminal_mode": "pipeline_totals",
        }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    print_or_save_report(run(args), args.output_json)


if __name__ == "__main__":
    main()
