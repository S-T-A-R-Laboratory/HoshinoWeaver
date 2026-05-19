"""生成 benchmark 用测试图片。

默认输出到 `bench/data/generated/<name>/`。
图片目录输入主要用于 smoke test 和输入链路验证，不作为默认性能基线。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "bench" / "data" / "generated"


def make_frame(
    *,
    height: int,
    width: int,
    dtype: np.dtype,
    channels: int,
    rng: np.random.Generator,
) -> np.ndarray:
    shape = (height, width, channels) if channels > 1 else (height, width)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return rng.integers(0, info.max + 1, size=shape, dtype=dtype)
    return rng.random(shape, dtype=np.float32).astype(dtype)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--height", type=int, default=4000)
    parser.add_argument("--width", type=int, default=6000)
    parser.add_argument("--dtype", type=str, default="uint8")
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--format", type=str, default="png", choices=["jpg", "png", "tif"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dtype = np.dtype(args.dtype)
    if args.format == "jpg" and dtype != np.dtype("uint8"):
        raise ValueError("JPEG dataset generation only supports uint8.")

    root = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_ROOT / args.name
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    for idx in range(args.frames):
        frame = make_frame(
            height=args.height,
            width=args.width,
            dtype=dtype,
            channels=args.channels,
            rng=rng,
        )
        path = root / f"frame_{idx:05d}.{args.format}"
        ok = cv2.imwrite(str(path), frame)
        if not ok:
            raise RuntimeError(f"Failed to write benchmark image: {path}")

    print(root)


if __name__ == "__main__":
    main()
