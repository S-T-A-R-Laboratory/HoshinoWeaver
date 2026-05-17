"""Compare the current CPU remap path against the Triton GPU prototype."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def _run_and_check(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_ms(sec: float) -> str:
    return f"{sec * 1000.0:.3f} ms"


def _build_common_args(args: argparse.Namespace) -> list[str]:
    common = [
        "--height", str(args.height),
        "--width", str(args.width),
        "--channels", str(args.channels),
        "--src-focal-px", str(args.src_focal_px),
        "--dst-focal-px", str(args.dst_focal_px),
        "--yaw-deg", str(args.yaw_deg),
        "--pitch-deg", str(args.pitch_deg),
        "--roll-deg", str(args.roll_deg),
        "--seed", str(args.seed),
        "--warmup", str(args.warmup),
        "--repeat", str(args.repeat),
    ]
    if args.src_height is not None:
        common.extend(["--src-height", str(args.src_height)])
    if args.src_width is not None:
        common.extend(["--src-width", str(args.src_width)])
    return common


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=2048)
    parser.add_argument("--width", type=int, default=3072)
    parser.add_argument("--src-height", type=int, default=None)
    parser.add_argument("--src-width", type=int, default=None)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--src-focal-px", type=float, default=2400.0)
    parser.add_argument("--dst-focal-px", type=float, default=2400.0)
    parser.add_argument("--yaw-deg", type=float, default=0.30)
    parser.add_argument("--pitch-deg", type=float, default=0.15)
    parser.add_argument("--roll-deg", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--cpu-python", type=str, default=sys.executable)
    parser.add_argument("--gpu-python", type=str, default=sys.executable)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    common_args = _build_common_args(args)

    with tempfile.TemporaryDirectory(prefix="hnw-remap-compare-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        cpu_json = tmpdir_path / "original_remap.json"
        gpu_json = tmpdir_path / "triton_remap.json"

        cpu_cmd = [
            args.cpu_python,
            "-m", "bench.gpu.original_remap",
            *common_args,
            "--cases", "numpy_grid", "opencv_remap", "original_remap",
            "--output-json", str(cpu_json),
        ]
        gpu_cmd = [
            args.gpu_python,
            "-m", "bench.gpu.triton_remap",
            *common_args,
            "--image-dtype", "fp32",
            "--block-size", str(args.block_size),
            "--device-index", str(args.device_index),
            "--cases", "torch_grid", "triton_grid", "torch_remap", "triton_remap",
            "--output-json", str(gpu_json),
        ]

        _run_and_check(cpu_cmd)
        _run_and_check(gpu_cmd)

        cpu_report = _read_json(cpu_json)
        gpu_report = _read_json(gpu_json)

    cpu_total = float(cpu_report["results"]["original_remap"]["mean_sec"])
    gpu_total = float(gpu_report["results"]["triton_remap"]["mean_sec"])
    cpu_grid = float(cpu_report["results"]["numpy_grid"]["mean_sec"])
    gpu_grid = float(gpu_report["results"]["triton_grid"]["mean_sec"])
    cpu_sample = float(cpu_report["results"]["opencv_remap"]["mean_sec"])
    gpu_sample = float(gpu_report["results"]["torch_remap"]["mean_sec"])
    speedup = cpu_total / gpu_total

    summary = {
        "suite": "remap_compare",
        "cpu_python": args.cpu_python,
        "gpu_python": args.gpu_python,
        "config": {
            "height": args.height,
            "width": args.width,
            "src_height": args.src_height if args.src_height is not None else args.height,
            "src_width": args.src_width if args.src_width is not None else args.width,
            "channels": args.channels,
            "src_focal_px": args.src_focal_px,
            "dst_focal_px": args.dst_focal_px,
            "yaw_deg": args.yaw_deg,
            "pitch_deg": args.pitch_deg,
            "roll_deg": args.roll_deg,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "block_size": args.block_size,
            "device_index": args.device_index,
        },
        "results": {
            "cpu_original_total_sec": cpu_total,
            "gpu_triton_total_sec": gpu_total,
            "cpu_grid_sec": cpu_grid,
            "gpu_grid_sec": gpu_grid,
            "cpu_sample_sec": cpu_sample,
            "gpu_sample_sec": gpu_sample,
            "speedup_x": speedup,
            "delta_sec": cpu_total - gpu_total,
        },
        "gpu_accuracy": gpu_report.get("accuracy", {}),
    }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True),
                               encoding="utf-8")

    print("[remap_compare]")
    print(f"cpu_python={args.cpu_python} gpu_python={args.gpu_python}")
    print(f"cpu_original_total={_format_ms(cpu_total)}")
    print(f"gpu_triton_total={_format_ms(gpu_total)}")
    print(f"speedup={speedup:.2f}x delta={_format_ms(cpu_total - gpu_total)}")
    print(f"cpu_grid={_format_ms(cpu_grid)} gpu_grid={_format_ms(gpu_grid)}")
    print(f"cpu_sample={_format_ms(cpu_sample)} gpu_sample={_format_ms(gpu_sample)}")
    if args.output_json:
        print(f"json={args.output_json}")


if __name__ == "__main__":
    main()
