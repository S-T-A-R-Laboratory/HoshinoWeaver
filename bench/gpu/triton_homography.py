"""Triton microbenchmark for pure homography warp."""

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
    "torch_warp",
    "triton_warp",
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

    print("[triton_homography]")
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
            f"warp_err={report['accuracy']['warp_max_abs_err']:.6e}",
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
    h_dst_to_src: torch.Tensor
    grid_scale_x: float
    grid_scale_y: float


def _make_homography(args: argparse.Namespace) -> torch.Tensor:
    tx = args.tx_px
    ty = args.ty_px
    angle = math.radians(args.rotation_deg)
    scale = args.scale
    persp_x = args.persp_x
    persp_y = args.persp_y

    c = math.cos(angle) * scale
    s = math.sin(angle) * scale
    h_src_to_dst = torch.tensor([
        [c, -s, tx],
        [s, c, ty],
        [persp_x, persp_y, 1.0],
    ], dtype=torch.float32)
    return torch.linalg.inv(h_src_to_dst)


def _build_config(args: argparse.Namespace) -> WarpConfig:
    return WarpConfig(
        height=args.height,
        width=args.width,
        channels=args.channels,
        h_dst_to_src=_make_homography(args),
        grid_scale_x=2.0 / max(args.width - 1, 1),
        grid_scale_y=2.0 / max(args.height - 1, 1),
    )


def _make_input_image(args: argparse.Namespace,
                      cfg: WarpConfig,
                      device: torch.device) -> torch.Tensor:
    torch.manual_seed(args.seed)
    dtype = torch.float16 if args.image_dtype == "fp16" else torch.float32
    return torch.rand((1, cfg.channels, cfg.height, cfg.width),
                      device=device,
                      dtype=dtype)


def build_grid_torch(cfg: WarpConfig,
                     device: torch.device) -> torch.Tensor:
    ys = torch.arange(cfg.height, device=device, dtype=torch.float32)
    xs = torch.arange(cfg.width, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    h = cfg.h_dst_to_src.to(device=device)
    denom = h[2, 0] * grid_x + h[2, 1] * grid_y + h[2, 2]
    src_x = (h[0, 0] * grid_x + h[0, 1] * grid_y + h[0, 2]) / denom
    src_y = (h[1, 0] * grid_x + h[1, 1] * grid_y + h[1, 2]) / denom

    norm_x = src_x * cfg.grid_scale_x - 1.0
    norm_y = src_y * cfg.grid_scale_y - 1.0
    return torch.stack((norm_x, norm_y), dim=-1).unsqueeze(0)


@triton.jit
def _homography_grid_kernel(
    out_ptr,
    height,
    width,
    h00,
    h01,
    h02,
    h10,
    h11,
    h12,
    h20,
    h21,
    h22,
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

    x = col.to(tl.float32)
    y = row.to(tl.float32)
    denom = h20 * x + h21 * y + h22
    src_x = (h00 * x + h01 * y + h02) / denom
    src_y = (h10 * x + h11 * y + h12) / denom

    norm_x = src_x * grid_scale_x - 1.0
    norm_y = src_y * grid_scale_y - 1.0

    out_idx = offs * 2
    tl.store(out_ptr + out_idx, norm_x, mask=mask)
    tl.store(out_ptr + out_idx + 1, norm_y, mask=mask)


def build_grid_triton(cfg: WarpConfig,
                      device: torch.device,
                      block_size: int) -> torch.Tensor:
    out = torch.empty((cfg.height, cfg.width, 2),
                      device=device,
                      dtype=torch.float32)
    h = cfg.h_dst_to_src.to(device="cpu")
    total = cfg.height * cfg.width
    grid = lambda meta: (triton.cdiv(total, meta["BLOCK_SIZE"]),)
    _homography_grid_kernel[grid](
        out,
        cfg.height,
        cfg.width,
        float(h[0, 0]),
        float(h[0, 1]),
        float(h[0, 2]),
        float(h[1, 0]),
        float(h[1, 1]),
        float(h[1, 2]),
        float(h[2, 0]),
        float(h[2, 1]),
        float(h[2, 2]),
        cfg.grid_scale_x,
        cfg.grid_scale_y,
        BLOCK_SIZE=block_size,
    )
    return out.unsqueeze(0)


def warp_with_grid(image: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
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
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--tx-px", type=float, default=12.0)
    parser.add_argument("--ty-px", type=float, default=-8.0)
    parser.add_argument("--rotation-deg", type=float, default=0.25)
    parser.add_argument("--scale", type=float, default=1.0005)
    parser.add_argument("--persp-x", type=float, default=1e-6)
    parser.add_argument("--persp-y", type=float, default=-8e-7)
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
        raise RuntimeError("CUDA is required for bench_triton_homography.py")

    unknown_cases = [case for case in args.cases if case not in CASE_NAMES]
    if unknown_cases:
        raise ValueError(
            f"Unknown triton homography benchmark case(s): {unknown_cases}. "
            f"Available: {list(CASE_NAMES)}")

    device = torch.device(f"cuda:{args.device_index}")
    torch.cuda.set_device(device)

    cfg = _build_config(args)
    image = _make_input_image(args, cfg, device)
    torch_grid = build_grid_torch(cfg, device)
    triton_grid = build_grid_triton(cfg, device, args.block_size)
    torch_out = warp_with_grid(image, torch_grid)
    triton_out = warp_with_grid(image, triton_grid)

    runners = {
        "torch_grid": lambda: build_grid_torch(cfg, device),
        "triton_grid": lambda: build_grid_triton(cfg, device, args.block_size),
        "torch_warp": lambda: warp_with_grid(image, torch_grid),
        "triton_warp": lambda: warp_with_grid(
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
        "suite": "triton_homography",
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
            "channels": args.channels,
            "tx_px": args.tx_px,
            "ty_px": args.ty_px,
            "rotation_deg": args.rotation_deg,
            "scale": args.scale,
            "persp_x": args.persp_x,
            "persp_y": args.persp_y,
            "image_dtype": args.image_dtype,
            "block_size": args.block_size,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "cases": args.cases,
        },
        "input_source": {
            "mode": "synthetic_torch",
            "resolved_frames": 1,
            "resolved_shape": [cfg.height, cfg.width, cfg.channels],
            "resolved_dtype": args.image_dtype,
        },
        "accuracy": {
            "grid_max_abs_err": float(
                (torch_grid - triton_grid).abs().max().item()),
            "warp_max_abs_err": float(
                (torch_out - triton_out).abs().max().item()),
        },
        "results": results,
    }
    print_or_save_report(report, args.output_json)


if __name__ == "__main__":
    main()
