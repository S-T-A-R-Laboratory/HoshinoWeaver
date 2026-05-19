"""CPU benchmark for pure homography warp."""

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


CASE_NAMES = [
    "opencv_warp",
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

    print("[original_homography]")
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
class WarpConfig:
    height: int
    width: int
    channels: int
    homography_src_to_dst: np.ndarray


def _make_homography(args: argparse.Namespace) -> np.ndarray:
    tx = args.tx_px
    ty = args.ty_px
    angle = math.radians(args.rotation_deg)
    scale = args.scale
    persp_x = args.persp_x
    persp_y = args.persp_y

    c = math.cos(angle) * scale
    s = math.sin(angle) * scale
    return np.array([
        [c, -s, tx],
        [s, c, ty],
        [persp_x, persp_y, 1.0],
    ], dtype=np.float32)


def _build_config(args: argparse.Namespace) -> WarpConfig:
    return WarpConfig(
        height=args.height,
        width=args.width,
        channels=args.channels,
        homography_src_to_dst=_make_homography(args),
    )


def _make_input_image(args: argparse.Namespace,
                      cfg: WarpConfig) -> np.ndarray:
    rng = np.random.default_rng(args.seed)
    return rng.random((cfg.height, cfg.width, cfg.channels),
                      dtype=np.float32)


def warp_with_cv2(image: np.ndarray,
                  h_src_to_dst: np.ndarray,
                  output_size: tuple[int, int]) -> np.ndarray:
    return cv2.warpPerspective(
        image,
        h_src_to_dst,
        output_size,
        flags=cv2.INTER_LINEAR,
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
    parser.add_argument("--cases", nargs="+", default=list(CASE_NAMES))
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    unknown_cases = [case for case in args.cases if case not in CASE_NAMES]
    if unknown_cases:
        raise ValueError(
            f"Unknown original homography benchmark case(s): {unknown_cases}. "
            f"Available: {list(CASE_NAMES)}")

    cfg = _build_config(args)
    image = _make_input_image(args, cfg)
    output_size = (cfg.width, cfg.height)

    runners = {
        "opencv_warp": lambda: warp_with_cv2(
            image, cfg.homography_src_to_dst, output_size),
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
        "suite": "original_homography",
        "env": {
            **collect_env_info(),
            "cv2": cv2.__version__,
        },
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
            "cases": args.cases,
        },
        "input_source": {
            "mode": "synthetic_numpy",
            "resolved_frames": 1,
            "resolved_shape": [cfg.height, cfg.width, cfg.channels],
            "resolved_dtype": "fp32",
        },
        "results": results,
    }
    print_or_save_report(report, args.output_json)


if __name__ == "__main__":
    main()
