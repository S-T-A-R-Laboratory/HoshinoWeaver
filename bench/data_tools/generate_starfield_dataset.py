"""Generate a synthetic starfield image dataset for alignment benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from bench.data_tools.starfield import generate_starfield_frames


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "bench" / "data" / "generated"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--height", type=int, default=2048)
    parser.add_argument("--width", type=int, default=3072)
    parser.add_argument("--dtype",
                        type=str,
                        default="uint16",
                        choices=["uint8", "uint16"])
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--stars", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-shift", type=float, default=12.0)
    parser.add_argument("--max-rotation-deg", type=float, default=0.8)
    parser.add_argument("--noise-sigma", type=float, default=1.5)
    parser.add_argument("--format",
                        type=str,
                        default="png",
                        choices=["png", "tif"])
    args = parser.parse_args()

    root = (Path(args.output_dir)
            if args.output_dir else DEFAULT_OUTPUT_ROOT / args.name)
    root.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype(args.dtype)

    frames, transforms = generate_starfield_frames(
        frames=args.frames,
        height=args.height,
        width=args.width,
        stars=args.stars,
        seed=args.seed,
        dtype=dtype,
        channels=args.channels,
        max_shift=args.max_shift,
        max_rotation_deg=args.max_rotation_deg,
        noise_sigma=args.noise_sigma,
    )

    for idx, frame in enumerate(frames):
        path = root / f"frame_{idx:05d}.{args.format}"
        ok = cv2.imwrite(str(path), frame)
        if not ok:
            raise RuntimeError(f"Failed to write benchmark image: {path}")

    meta = {
        "frames": args.frames,
        "height": args.height,
        "width": args.width,
        "dtype": args.dtype,
        "channels": args.channels,
        "stars": args.stars,
        "seed": args.seed,
        "max_shift": args.max_shift,
        "max_rotation_deg": args.max_rotation_deg,
        "noise_sigma": args.noise_sigma,
        "transforms": transforms,
    }
    (root / "meta.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )
    print(root)


if __name__ == "__main__":
    main()
