"""Compare CPU OpenCV homography warp against the Triton GPU prototype."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=2048)
    parser.add_argument("--width", type=int, default=3072)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--tx-px", type=float, default=12.0)
    parser.add_argument("--ty-px", type=float, default=-8.0)
    parser.add_argument("--rotation-deg", type=float, default=0.25)
    parser.add_argument("--scale", type=float, default=1.0005)
    parser.add_argument("--persp-x", type=float, default=1e-6)
    parser.add_argument("--persp-y", type=float, default=-8e-7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--cpu-python", type=str, default=sys.executable)
    parser.add_argument("--gpu-python", type=str, default=sys.executable)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    common_args = [
        f"--height={args.height}",
        f"--width={args.width}",
        f"--channels={args.channels}",
        f"--tx-px={args.tx_px}",
        f"--ty-px={args.ty_px}",
        f"--rotation-deg={args.rotation_deg}",
        f"--scale={args.scale}",
        f"--persp-x={args.persp_x}",
        f"--persp-y={args.persp_y}",
        f"--seed={args.seed}",
        f"--warmup={args.warmup}",
        f"--repeat={args.repeat}",
    ]

    with tempfile.TemporaryDirectory(prefix="hnw-homo-compare-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        cpu_json = tmpdir_path / "original_homography.json"
        gpu_json = tmpdir_path / "triton_homography.json"

        cpu_cmd = [
            args.cpu_python,
            "-m", "bench.gpu.original_homography",
            *common_args,
            "--cases", "opencv_warp",
            "--output-json", str(cpu_json),
        ]
        gpu_cmd = [
            args.gpu_python,
            "-m", "bench.gpu.triton_homography",
            *common_args,
            "--image-dtype", "fp32",
            f"--block-size={args.block_size}",
            f"--device-index={args.device_index}",
            "--cases", "torch_grid", "triton_grid", "torch_warp", "triton_warp",
            "--output-json", str(gpu_json),
        ]

        _run_and_check(cpu_cmd)
        _run_and_check(gpu_cmd)

        cpu_report = _read_json(cpu_json)
        gpu_report = _read_json(gpu_json)

    cpu_total = float(cpu_report["results"]["opencv_warp"]["mean_sec"])
    gpu_total = float(gpu_report["results"]["triton_warp"]["mean_sec"])
    torch_total = (float(gpu_report["results"]["torch_grid"]["mean_sec"]) +
                   float(gpu_report["results"]["torch_warp"]["mean_sec"]))
    gpu_grid = float(gpu_report["results"]["triton_grid"]["mean_sec"])
    speedup = cpu_total / gpu_total
    torch_speedup = cpu_total / torch_total

    summary = {
        "suite": "homography_compare",
        "cpu_python": args.cpu_python,
        "gpu_python": args.gpu_python,
        "config": {
            "height": args.height,
            "width": args.width,
            "channels": args.channels,
            "tx_px": args.tx_px,
            "ty_px": args.ty_px,
            "rotation_deg": args.rotation_deg,
            "scale": args.scale,
            "persp_x": args.persp_x,
            "persp_y": args.persp_y,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "block_size": args.block_size,
            "device_index": args.device_index,
        },
        "results": {
            "cpu_opencv_total_sec": cpu_total,
            "gpu_triton_total_sec": gpu_total,
            "gpu_torch_total_sec": torch_total,
            "gpu_triton_grid_sec": gpu_grid,
            "speedup_x": speedup,
            "torch_speedup_x": torch_speedup,
            "delta_sec": cpu_total - gpu_total,
        },
        "gpu_accuracy": gpu_report.get("accuracy", {}),
    }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True),
                               encoding="utf-8")

    print("[homography_compare]")
    print(f"cpu_python={args.cpu_python} gpu_python={args.gpu_python}")
    print(f"cpu_opencv_total={_format_ms(cpu_total)}")
    print(f"gpu_triton_total={_format_ms(gpu_total)}")
    print(f"gpu_torch_total={_format_ms(torch_total)}")
    print(f"speedup={speedup:.2f}x torch_speedup={torch_speedup:.2f}x")
    print(f"gpu_triton_grid={_format_ms(gpu_grid)}")
    if args.output_json:
        print(f"json={args.output_json}")


if __name__ == "__main__":
    main()
