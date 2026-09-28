"""Production median detection stage timings for the macOS experiment branch."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter
from unittest.mock import patch

import cv2
from loguru import logger
import numpy as np

from bench.common import run_benchmark, summarize_samples
from hoshicore._custom_op import build_info, set_backend_preference
from hoshicore._custom_op.backend_registry import resolve_backend
from hoshicore._custom_op.ops.filter import median_filter_2d_compiled
from hoshicore.component.norma import detection, geometry_view


def make_starfield(height: int, width: int, *, seed: int = 20260928) -> tuple[np.ndarray, np.ndarray]:
    """Noisy uint16 BGR field with about 800 Gaussian stars/MP and a sky mask."""
    if min(height, width) < 32:
        raise ValueError("starfield dimensions must be at least 32")
    rng = np.random.default_rng(seed)
    field = rng.normal(3000.0, 50.0, (height, width)).astype(np.float32)
    count = max(24, round(height * width * 800 / 1e6))
    yy, xx = np.mgrid[-7:8, -7:8]
    kernels = [np.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma)).astype(np.float32)
               for sigma in (1.2, 1.5, 1.8, 2.2)]
    for i in range(count):
        y, x = int(rng.integers(0, height)), int(rng.integers(0, width))
        y0, y1 = max(0, y - 7), min(height, y + 8)
        x0, x1 = max(0, x - 7), min(width, x + 8)
        kernel = kernels[i % len(kernels)][y0 - y + 7:y1 - y + 7, x0 - x + 7:x1 - x + 7]
        field[y0:y1, x0:x1] += np.float32(rng.uniform(3000, 30000)) * kernel
    image = np.empty((height, width, 3), dtype=np.uint16)
    for channel, scale in enumerate((0.9, 1.0, 1.1)):
        plane = field * np.float32(scale)
        np.clip(plane, 0, 65535, out=plane)
        image[..., channel] = np.rint(plane).astype(np.uint16)
    mask = np.ones((height, width), dtype=np.uint8)
    mask[height * 4 // 5:] = 0
    mask[height // 4:height // 3, width // 4:width // 3] = 0
    return image, mask


def profile_frame(image: np.ndarray, mask: np.ndarray) -> tuple[dict[str, float], detection.DetectedStars]:
    """Time disjoint production calls; the residual includes geometry, intensity and filtering."""
    if image.dtype != np.uint16 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("profile requires a uint16 BGR image")
    durations = {"gray_sec": 0.0, "pixel_sec": 0.0, "find_contours_sec": 0.0}
    counts: Counter[str] = Counter()

    def timed(name, fn):
        def wrapped(*args, **kwargs):
            started = perf_counter()
            result = fn(*args, **kwargs)
            durations[name] += perf_counter() - started
            counts[name] += 1
            return result
        return wrapped

    # Only three coarse calls are wrapped; per-contour timing would distort the loop.
    with ExitStack() as stack:
        for module, attribute, name in (
            (geometry_view, "to_median_gray_u16", "gray_sec"),
            (detection, "_detect_starmask_by_threshold_details", "pixel_sec"),
            (cv2, "findContours", "find_contours_sec"),
        ):
            stack.enter_context(patch.object(module, attribute, timed(name, getattr(module, attribute))))
        started = perf_counter()
        stars = geometry_view.StarDetectionCache.from_image(image, mask).median_stars
        total = perf_counter() - started
    if any(counts[name] != 1 for name in durations):
        raise RuntimeError(f"production stage layout changed: {dict(counts)}")
    durations["geometry_intensity_filter_sec"] = total - sum(durations.values())
    durations["total_sec"] = total
    return durations, stars


def summarize_profiles(samples: list[dict[str, float]]) -> dict[str, dict]:
    return {name.removesuffix("_sec"): summarize_samples([sample[name] for sample in samples])
            for name in samples[0]}


def _assert_same_stars(actual: detection.DetectedStars, expected: detection.DetectedStars) -> None:
    for name in ("positions", "volumes", "intensities"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))


def measure_case(height: int, width: int, *, seed: int, warmup: int, repeat: int) -> dict:
    image, mask = make_starfield(height, width, seed=seed)
    expected = geometry_view.StarDetectionCache.from_image(image, mask).median_stars
    if len(expected.positions) == 0:
        raise RuntimeError("benchmark starfield produced no detected stars")
    for _ in range(warmup):
        _, stars = profile_frame(image, mask)
        _assert_same_stars(stars, expected)
    samples = []
    for _ in range(repeat):
        sample, stars = profile_frame(image, mask)
        _assert_same_stars(stars, expected)
        samples.append(sample)

    gray = geometry_view.to_median_gray_u16(image)
    background = run_benchmark(lambda: median_filter_2d_compiled(gray, 13), warmup=warmup, repeat=repeat)
    fingerprint = hashlib.sha256()
    for name in ("positions", "volumes", "intensities"):
        fingerprint.update(np.ascontiguousarray(getattr(expected, name)).tobytes())
    return {
        "shape": list(image.shape), "dtype": str(image.dtype), "seed": seed,
        "masked_fraction": float(1 - np.count_nonzero(mask) / mask.size),
        "detected_stars": len(expected.positions), "stars_sha256": fingerprint.hexdigest(),
        "profile_matches_production_exactly": True,
        "stages": summarize_profiles(samples), "standalone_median": background,
        "note": "standalone_median is a separate microbenchmark, not an additive pipeline stage",
    }


def environment(require_metal: bool) -> dict:
    try:
        from hoshicore._custom_op import _metal
    except ImportError:
        metal = {"available": False, "reason": "Metal extension not installed"}
    else:
        metal = dict(_metal.metal_device_info())
    if require_metal and not metal.get("available"):
        raise RuntimeError(f"Metal runtime unavailable: {metal}")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    return {
        "commit": commit, "time_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "machine": platform.machine(),
        "python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__,
        "cpu_count": os.cpu_count(), "opencv_threads": cv2.getNumThreads(),
        "openmp_policy": os.environ.get("HNW_CUSTOM_OPS_THREADS", "auto"),
        "build": build_info(), "metal_device": metal,
    }


def markdown_summary(report: dict) -> str:
    lines = ["## macOS detection CPU baseline", "",
             f"Commit: `{report['environment']['commit']}`", "",
             "Median milliseconds; correctness is gated, timing is informational.", "",
             "| Image | Stars | Gray | Fused pixels | Find contours | Geometry/intensity/filter | Total |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for case in report["cases"]:
        timings = case["stages"]
        values = [timings[name]["median_sec"] * 1000 for name in
                  ("gray", "pixel", "find_contours", "geometry_intensity_filter", "total")]
        shape = "×".join(str(x) for x in case["shape"][:2])
        lines.append(f"| {shape} | {case['detected_stars']} | " + " | ".join(f"{v:.1f}" for v in values) + " |")
    lines += ["", "Fused pixels include median background, threshold and morphology.",
              "The residual includes Python orchestration and the small instrumentation overhead.",
              "Standalone median samples are in JSON and must not be added to the pipeline timings.",
              "No Metal median kernel is measured; hosted-runner timing is not a physical-Mac speed claim."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", default=["1024x1280", "4160x6240"])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--require-metal", action="store_true")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat < 1:
        parser.error("warmup must be nonnegative and repeat must be positive")
    sizes = []
    for value in args.sizes:
        try:
            h, w = (int(x) for x in value.lower().split("x"))
        except ValueError:
            parser.error(f"invalid size: {value}")
        if min(h, w) < 32:
            parser.error("dimensions must be at least 32")
        sizes.append((h, w))

    set_backend_preference("cpu")
    for name in ("detection_gray", "median_star_mask", "median_filter_2d"):
        if not resolve_backend(name, "cpu").native:
            raise RuntimeError(f"benchmark requires compiled CPU backend: {name}")
    logger.disable("hoshicore")
    report = {"environment": environment(args.require_metal), "completed": False,
              "config": {"warmup": args.warmup, "repeat": args.repeat, "median_ksize": 13,
                         "threshold_ratio": 1.0, "open_ksize": 3, "stars_per_megapixel": 800},
              "cases": []}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    for h, w in sizes:
        print(f"Profiling {h}x{w} uint16 BGR, CPU median detection", flush=True)
        report["cases"].append(measure_case(h, w, seed=args.seed, warmup=args.warmup, repeat=args.repeat))
        args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["completed"] = True
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = markdown_summary(report)
    args.output_json.with_suffix(".md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
