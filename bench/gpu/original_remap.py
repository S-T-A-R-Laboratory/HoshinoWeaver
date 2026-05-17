"""CPU benchmark for the current camera-model remap path.

This mirrors the current zero-distortion data path:
- NumPy builds the per-pixel source map on CPU
- OpenCV remaps the image on CPU
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

import cv2
import numpy as np

from hoshicore._custom_op import build_info as custom_op_build_info
from hoshicore._custom_op import camera_model_remap as custom_camera_model_remap


CASE_NAMES = [
    "numpy_grid",
    "custom_op_fused",
    "opencv_remap",
    "original_remap",
]


def collect_env_info() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "cwd": str(Path(__file__).resolve().parents[2]),
    }


def summarize_samples(samples: list[float]) -> dict[str, Any]:
    return {
        "samples_sec": samples,
        "min_sec": min(samples),
        "max_sec": max(samples),
        "mean_sec": mean(samples),
        "median_sec": median(samples),
    }


def print_or_save_report(report: dict[str, Any], output_json: str | None) -> None:
    if output_json:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True),
                        encoding="utf-8")

    print("[original_remap]")
    print(
        " ".join([
            f"input={report['input_source']['mode']}",
            f"shape={report['input_source']['resolved_shape']}",
            f"dtype={report['input_source']['resolved_dtype']}",
        ]))
    for case_name in report["config"]["cases"]:
        payload = report["results"][case_name]
        print(
            f"{case_name}: mean={payload['mean_sec']:.6f}s "
            f"min={payload['min_sec']:.6f}s max={payload['max_sec']:.6f}s")
    if output_json:
        print(f"json={output_json}")


@dataclass(frozen=True)
class RemapConfig:
    height: int
    width: int
    src_height: int
    src_width: int
    fx_src: float
    fy_src: float
    cx_src: float
    cy_src: float
    fx_dst: float
    fy_dst: float
    cx_dst: float
    cy_dst: float
    rotation_dst_to_src: np.ndarray


def _make_rotation_matrix(yaw_deg: float,
                          pitch_deg: float,
                          roll_deg: float) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)

    cy = math.cos(yaw)
    sy = math.sin(yaw)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cr = math.cos(roll)
    sr = math.sin(roll)

    rz = np.array([
        [cy, -sy, 0.0],
        [sy, cy, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    ry = np.array([
        [cp, 0.0, sp],
        [0.0, 1.0, 0.0],
        [-sp, 0.0, cp],
    ], dtype=np.float32)
    rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cr, -sr],
        [0.0, sr, cr],
    ], dtype=np.float32)
    return rz @ ry @ rx


def _build_config(args: argparse.Namespace) -> RemapConfig:
    src_height = args.height if args.src_height is None else args.src_height
    src_width = args.width if args.src_width is None else args.src_width

    cx_src = (src_width - 1) * 0.5
    cy_src = (src_height - 1) * 0.5
    cx_dst = (args.width - 1) * 0.5
    cy_dst = (args.height - 1) * 0.5

    return RemapConfig(
        height=args.height,
        width=args.width,
        src_height=src_height,
        src_width=src_width,
        fx_src=args.src_focal_px,
        fy_src=args.src_focal_px,
        cx_src=cx_src,
        cy_src=cy_src,
        fx_dst=args.dst_focal_px,
        fy_dst=args.dst_focal_px,
        cx_dst=cx_dst,
        cy_dst=cy_dst,
        rotation_dst_to_src=_make_rotation_matrix(
            args.yaw_deg, args.pitch_deg, args.roll_deg),
    )


def _make_input_image(args: argparse.Namespace,
                      cfg: RemapConfig) -> np.ndarray:
    rng = np.random.default_rng(args.seed)
    image = rng.random((cfg.src_height, cfg.src_width, args.channels),
                       dtype=np.float32)
    return image


def build_grid_numpy(cfg: RemapConfig) -> tuple[np.ndarray, np.ndarray]:
    xs = np.arange(cfg.width, dtype=np.float32)
    ys = np.arange(cfg.height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")

    x = (grid_x - cfg.cx_dst) / cfg.fx_dst
    y = (grid_y - cfg.cy_dst) / cfg.fy_dst

    r = cfg.rotation_dst_to_src
    proj_x = r[0, 0] * x + r[0, 1] * y + r[0, 2]
    proj_y = r[1, 0] * x + r[1, 1] * y + r[1, 2]
    proj_z = r[2, 0] * x + r[2, 1] * y + r[2, 2]

    src_x = cfg.fx_src * (proj_x / proj_z) + cfg.cx_src
    src_y = cfg.fy_src * (proj_y / proj_z) + cfg.cy_src
    return src_x.astype(np.float32), src_y.astype(np.float32)


def remap_with_cv2(image: np.ndarray,
                   map_x: np.ndarray,
                   map_y: np.ndarray) -> np.ndarray:
    return cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def run_cpu_benchmark(func,
                      *,
                      warmup: int,
                      repeat: int) -> dict[str, Any]:
    for _ in range(warmup):
        func()

    samples: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        func()
        samples.append(time.perf_counter() - t0)
    return summarize_samples(samples)


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
    parser.add_argument("--cases", nargs="+", default=list(CASE_NAMES))
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    unknown_cases = [case for case in args.cases if case not in CASE_NAMES]
    if unknown_cases:
        raise ValueError(
            f"Unknown original remap benchmark case(s): {unknown_cases}. "
            f"Available: {list(CASE_NAMES)}")

    cfg = _build_config(args)
    image = _make_input_image(args, cfg)
    map_x, map_y = build_grid_numpy(cfg)

    runners = {
        "numpy_grid": lambda: build_grid_numpy(cfg),
        "custom_op_fused": lambda: custom_camera_model_remap(
            image=image,
            out_height=cfg.height,
            out_width=cfg.width,
            fx_src=cfg.fx_src,
            fy_src=cfg.fy_src,
            cx_src=cfg.cx_src,
            cy_src=cfg.cy_src,
            fx_dst=cfg.fx_dst,
            fy_dst=cfg.fy_dst,
            cx_dst=cfg.cx_dst,
            cy_dst=cfg.cy_dst,
            rotation_dst_to_src=cfg.rotation_dst_to_src,
        ),
        "opencv_remap": lambda: remap_with_cv2(image, map_x, map_y),
        "original_remap": lambda: remap_with_cv2(image, *build_grid_numpy(cfg)),
    }

    results = {
        case_name: run_cpu_benchmark(
            runners[case_name],
            warmup=args.warmup,
            repeat=args.repeat,
        )
        for case_name in args.cases
    }

    report = {
        "suite": "original_remap",
        "env": {
            **collect_env_info(),
            "cv2": cv2.__version__,
            "custom_op_build": custom_op_build_info(),
        },
        "config": {
            "height": args.height,
            "width": args.width,
            "src_height": cfg.src_height,
            "src_width": cfg.src_width,
            "channels": args.channels,
            "src_focal_px": args.src_focal_px,
            "dst_focal_px": args.dst_focal_px,
            "yaw_deg": args.yaw_deg,
            "pitch_deg": args.pitch_deg,
            "roll_deg": args.roll_deg,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "cases": args.cases,
        },
        "input_source": {
            "mode": "synthetic_numpy",
            "resolved_frames": 1,
            "resolved_shape": [cfg.src_height, cfg.src_width, args.channels],
            "resolved_dtype": "fp32",
        },
        "results": results,
    }
    print_or_save_report(report, args.output_json)


if __name__ == "__main__":
    main()
