"""Triton microbenchmark for camera-model remap.

This benchmark targets the hottest GPU candidate in the alignment pipeline:
per-pixel map generation for camera-model remap, followed by image sampling.

Scope is intentionally narrow:
- zero distortion only
- single-image remap
- Triton validates the map-generation kernel
- final production path is expected to become a dedicated CUDA kernel
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

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

CASE_NAMES = [
    "torch_grid",
    "triton_grid",
    "torch_remap",
    "triton_remap",
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

    print("[triton_remap]")
    print(
        " ".join([
            f"input={report['input_source']['mode']}",
            f"shape={report['input_source']['resolved_shape']}",
            f"dtype={report['input_source']['resolved_dtype']}",
            f"device={report['env']['device_name']}",
        ]))
    print(
        " ".join([
            f"grid_err={report['accuracy']['grid_max_abs_err']:.6e}",
            f"remap_err={report['accuracy']['remap_max_abs_err']:.6e}",
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
    rotation_dst_to_src: torch.Tensor
    grid_scale_x: float
    grid_scale_y: float


def _make_rotation_matrix(yaw_deg: float,
                          pitch_deg: float,
                          roll_deg: float) -> torch.Tensor:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)

    cy = math.cos(yaw)
    sy = math.sin(yaw)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cr = math.cos(roll)
    sr = math.sin(roll)

    rz = torch.tensor([
        [cy, -sy, 0.0],
        [sy, cy, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32)
    ry = torch.tensor([
        [cp, 0.0, sp],
        [0.0, 1.0, 0.0],
        [-sp, 0.0, cp],
    ], dtype=torch.float32)
    rx = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, cr, -sr],
        [0.0, sr, cr],
    ], dtype=torch.float32)
    return rz @ ry @ rx


def _build_config(args: argparse.Namespace) -> RemapConfig:
    src_height = args.height if args.src_height is None else args.src_height
    src_width = args.width if args.src_width is None else args.src_width

    fx_src = args.src_focal_px
    fy_src = args.src_focal_px
    fx_dst = args.dst_focal_px
    fy_dst = args.dst_focal_px

    cx_src = (src_width - 1) * 0.5
    cy_src = (src_height - 1) * 0.5
    cx_dst = (args.width - 1) * 0.5
    cy_dst = (args.height - 1) * 0.5

    return RemapConfig(
        height=args.height,
        width=args.width,
        src_height=src_height,
        src_width=src_width,
        fx_src=fx_src,
        fy_src=fy_src,
        cx_src=cx_src,
        cy_src=cy_src,
        fx_dst=fx_dst,
        fy_dst=fy_dst,
        cx_dst=cx_dst,
        cy_dst=cy_dst,
        rotation_dst_to_src=_make_rotation_matrix(
            args.yaw_deg, args.pitch_deg, args.roll_deg),
        grid_scale_x=2.0 / max(src_width - 1, 1),
        grid_scale_y=2.0 / max(src_height - 1, 1),
    )


def _make_input_image(args: argparse.Namespace,
                      cfg: RemapConfig,
                      device: torch.device) -> torch.Tensor:
    torch.manual_seed(args.seed)
    dtype = torch.float16 if args.image_dtype == "fp16" else torch.float32
    image = torch.rand(
        (1, args.channels, cfg.src_height, cfg.src_width),
        device=device,
        dtype=dtype,
    )
    return image


def build_grid_torch(cfg: RemapConfig,
                     device: torch.device) -> torch.Tensor:
    ys = torch.arange(cfg.height, device=device, dtype=torch.float32)
    xs = torch.arange(cfg.width, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    x = (grid_x - cfg.cx_dst) / cfg.fx_dst
    y = (grid_y - cfg.cy_dst) / cfg.fy_dst

    r = cfg.rotation_dst_to_src.to(device=device)
    proj_x = r[0, 0] * x + r[0, 1] * y + r[0, 2]
    proj_y = r[1, 0] * x + r[1, 1] * y + r[1, 2]
    proj_z = r[2, 0] * x + r[2, 1] * y + r[2, 2]

    src_x = cfg.fx_src * (proj_x / proj_z) + cfg.cx_src
    src_y = cfg.fy_src * (proj_y / proj_z) + cfg.cy_src

    norm_x = src_x * cfg.grid_scale_x - 1.0
    norm_y = src_y * cfg.grid_scale_y - 1.0
    return torch.stack((norm_x, norm_y), dim=-1).unsqueeze(0)


@triton.jit
def _remap_grid_kernel(
    out_ptr,
    height,
    width,
    fx_src,
    fy_src,
    cx_src,
    cy_src,
    fx_dst,
    fy_dst,
    cx_dst,
    cy_dst,
    r00,
    r01,
    r02,
    r10,
    r11,
    r12,
    r20,
    r21,
    r22,
    grid_scale_x,
    grid_scale_y,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = height * width
    mask = offs < total

    row = offs // width
    col = offs - row * width

    x = (col.to(tl.float32) - cx_dst) / fx_dst
    y = (row.to(tl.float32) - cy_dst) / fy_dst

    proj_x = r00 * x + r01 * y + r02
    proj_y = r10 * x + r11 * y + r12
    proj_z = r20 * x + r21 * y + r22

    inv_z = 1.0 / proj_z
    src_x = fx_src * proj_x * inv_z + cx_src
    src_y = fy_src * proj_y * inv_z + cy_src

    norm_x = src_x * grid_scale_x - 1.0
    norm_y = src_y * grid_scale_y - 1.0

    out_idx = offs * 2
    tl.store(out_ptr + out_idx, norm_x, mask=mask)
    tl.store(out_ptr + out_idx + 1, norm_y, mask=mask)


def build_grid_triton(cfg: RemapConfig,
                      device: torch.device,
                      block_size: int) -> torch.Tensor:
    out = torch.empty((cfg.height, cfg.width, 2),
                      device=device,
                      dtype=torch.float32)
    r = cfg.rotation_dst_to_src.to(device="cpu")
    total = cfg.height * cfg.width
    grid = lambda meta: (triton.cdiv(total, meta["BLOCK_SIZE"]),)
    _remap_grid_kernel[grid](
        out,
        cfg.height,
        cfg.width,
        cfg.fx_src,
        cfg.fy_src,
        cfg.cx_src,
        cfg.cy_src,
        cfg.fx_dst,
        cfg.fy_dst,
        cfg.cx_dst,
        cfg.cy_dst,
        float(r[0, 0]),
        float(r[0, 1]),
        float(r[0, 2]),
        float(r[1, 0]),
        float(r[1, 1]),
        float(r[1, 2]),
        float(r[2, 0]),
        float(r[2, 1]),
        float(r[2, 2]),
        cfg.grid_scale_x,
        cfg.grid_scale_y,
        BLOCK_SIZE=block_size,
    )
    return out.unsqueeze(0)


def remap_with_grid(image: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    # cudnn grid_sample expects input/grid dtypes to match on this path.
    return F.grid_sample(
        image,
        grid.to(dtype=image.dtype),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )


def run_cuda_benchmark(func,
                       *,
                       warmup: int,
                       repeat: int) -> dict[str, Any]:
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        func()
        torch.cuda.synchronize()
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
    parser.add_argument("--image-dtype",
                        choices=["fp16", "fp32"],
                        default="fp16")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--cases", nargs="+", default=list(CASE_NAMES))
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for bench_triton_remap.py")

    unknown_cases = [case for case in args.cases if case not in CASE_NAMES]
    if unknown_cases:
        raise ValueError(
            f"Unknown triton remap benchmark case(s): {unknown_cases}. "
            f"Available: {list(CASE_NAMES)}")

    device = torch.device(f"cuda:{args.device_index}")
    torch.cuda.set_device(device)

    cfg = _build_config(args)
    image = _make_input_image(args, cfg, device)

    torch_grid = build_grid_torch(cfg, device)
    triton_grid = build_grid_triton(cfg, device, args.block_size)
    torch_out = remap_with_grid(image, torch_grid)
    triton_out = remap_with_grid(image, triton_grid)

    runners = {
        "torch_grid": lambda: build_grid_torch(cfg, device),
        "triton_grid": lambda: build_grid_triton(cfg, device, args.block_size),
        "torch_remap": lambda: remap_with_grid(image, torch_grid),
        "triton_remap": lambda: remap_with_grid(
            image, build_grid_triton(cfg, device, args.block_size)),
    }

    results = {
        case_name: run_cuda_benchmark(
            runners[case_name],
            warmup=args.warmup,
            repeat=args.repeat,
        )
        for case_name in args.cases
    }

    report = {
        "suite": "triton_remap",
        "env": {
            **collect_env_info(),
            "torch": torch.__version__,
            "triton": triton.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_index": args.device_index,
            "device_name": torch.cuda.get_device_name(device),
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
            "image_dtype": args.image_dtype,
            "block_size": args.block_size,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "cases": args.cases,
        },
        "input_source": {
            "mode": "synthetic_torch",
            "resolved_frames": 1,
            "resolved_shape": [cfg.src_height, cfg.src_width, args.channels],
            "resolved_dtype": args.image_dtype,
        },
        "accuracy": {
            "grid_max_abs_err": float(
                (torch_grid - triton_grid).abs().max().item()),
            "remap_max_abs_err": float(
                (torch_out - triton_out).abs().max().item()),
        },
        "results": results,
    }
    print_or_save_report(report, args.output_json)


if __name__ == "__main__":
    main()
