"""生成 benchmark raw cache。

默认输出到 `bench/data/cache/<name>/`，供 kernel benchmark 直接复用。
支持 synthetic 生成，也支持把现有图片目录转成 raw cache。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bench.common import discover_image_paths, load_benchmark_image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "bench" / "data" / "cache"


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


def resolve_image_paths(
    *,
    input_dir: str,
    frames: int,
) -> tuple[list[Path], Path]:
    root, paths = discover_image_paths(input_dir=input_dir, frames=frames)
    if root is None or len(paths) < frames:
        raise RuntimeError(
            f"Failed to resolve {frames} image frames from input_dir={input_dir!r}"
        )
    return paths, root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--height", type=int, default=4000)
    parser.add_argument("--width", type=int, default=6000)
    parser.add_argument("--dtype", type=str, default="uint8")
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_ROOT / args.name
    root.mkdir(parents=True, exist_ok=True)
    data_path = root / "frames.dat"
    meta_path = root / "meta.json"

    if args.input_dir:
        image_paths, source_root = resolve_image_paths(
            input_dir=args.input_dir,
            frames=args.frames,
        )
        first = load_benchmark_image(image_paths[0])
        shape = (args.frames, *first.shape)
        with data_path.open("wb") as f:
            first.tofile(f)
            for idx, path in enumerate(image_paths[1:], start=1):
                frame = load_benchmark_image(path)
                if frame.shape != first.shape:
                    raise ValueError(
                        f"image cache shape mismatch: frame 0 {first.shape} vs frame {idx} {frame.shape}"
                    )
                if frame.dtype != first.dtype:
                    raise ValueError(
                        f"image cache dtype mismatch: frame 0 {first.dtype} vs frame {idx} {frame.dtype}"
                    )
                frame.tofile(f)
        meta = {
            "frames": args.frames,
            "shape": list(first.shape),
            "dtype": str(first.dtype),
            "source_mode": "images",
            "source_input_dir": str(source_root),
        }
    else:
        dtype = np.dtype(args.dtype)
        shape = (args.frames, args.height, args.width, args.channels) if args.channels > 1 else (
            args.frames,
            args.height,
            args.width,
        )
        rng = np.random.default_rng(args.seed)
        with data_path.open("wb") as f:
            for _ in range(args.frames):
                frame = make_frame(
                    height=args.height,
                    width=args.width,
                    dtype=dtype,
                    channels=args.channels,
                    rng=rng,
                )
                frame.tofile(f)
        meta = {
            "frames": args.frames,
            "shape": list(shape[1:]),
            "dtype": str(dtype),
            "seed": args.seed,
            "source_mode": "synthetic",
        }

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(root)


if __name__ == "__main__":
    main()
