"""Benchmark 公共工具。

用途：
- 统一 benchmark 的计时和结果输出格式
- 生成合成 numpy 帧序列
- 加载 raw cache 或图片目录
- 为 OpenMP 线程数提供平台相关的自动选择
"""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import os
import platform
import time
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCH_DATA_DIR = PROJECT_ROOT / "bench" / "data"
BENCH_CACHE_DIR = BENCH_DATA_DIR / "cache"
DEFAULT_INPUT_DIRS = [
    BENCH_CACHE_DIR,
    BENCH_DATA_DIR / "input",
    BENCH_DATA_DIR / "generated",
]
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
_OMP_SETTER: Any | None = None


def collect_env_info() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "cwd": str(PROJECT_ROOT),
    }


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _format_seconds(value: float) -> str:
    return f"{value:.6f}s"


def _case_order(report: dict[str, Any]) -> list[str]:
    config = report.get("config", {})
    configured = config.get("cases")
    results = report.get("results", {})
    if isinstance(configured, list):
        return [str(case) for case in configured if case in results]
    return list(results.keys())


def render_terminal_summary(report: dict[str, Any], output_json: str | None) -> str:
    lines: list[str] = []
    suite = report.get("suite", "benchmark")
    input_source = report.get("input_source", {})
    custom_ops = report.get("custom_ops", {})

    lines.append(f"[{suite}]")

    if isinstance(input_source, dict):
        mode = input_source.get("mode")
        frames = input_source.get("resolved_frames")
        shape = input_source.get("resolved_shape")
        dtype = input_source.get("resolved_dtype")
        parts = []
        if mode:
            parts.append(f"input={mode}")
        if frames is not None:
            parts.append(f"frames={frames}")
        if shape:
            parts.append(f"shape={shape}")
        if dtype:
            parts.append(f"dtype={dtype}")
        if parts:
            lines.append(" ".join(parts))

    if isinstance(custom_ops, dict) and custom_ops.get("available"):
        compiler = custom_ops.get("compiler")
        openmp = custom_ops.get("openmp")
        omp_simd = custom_ops.get("omp_simd")
        ndebug = custom_ops.get("ndebug")
        parts = []
        if compiler:
            parts.append(f"compiler={compiler}")
        if openmp is not None:
            parts.append(f"openmp={openmp}")
        if omp_simd is not None:
            parts.append(f"omp_simd={omp_simd}")
        if ndebug is not None:
            parts.append(f"ndebug={ndebug}")
        if parts:
            lines.append(" ".join(parts))

    results = report.get("results", {})
    for case_name in _case_order(report):
        payload = results.get(case_name, {})
        if not isinstance(payload, dict):
            continue
        mean_sec = payload.get("mean_sec")
        min_sec = payload.get("min_sec")
        max_sec = payload.get("max_sec")
        if isinstance(mean_sec, (int, float)):
            summary = f"{case_name}: mean={_format_seconds(float(mean_sec))}"
            if isinstance(min_sec, (int, float)) and isinstance(max_sec, (int, float)):
                summary += f" min={_format_seconds(float(min_sec))} max={_format_seconds(float(max_sec))}"
            lines.append(summary)

    if output_json:
        lines.append(f"json={output_json}")
    return "\n".join(lines)


def print_or_save_report(report: dict[str, Any], output_json: str | None) -> None:
    payload = json.dumps(to_jsonable(report), indent=2, sort_keys=True)
    if output_json:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    print(render_terminal_summary(report, output_json))


def summarize_samples(samples: list[float]) -> dict[str, Any]:
    return {
        "samples_sec": samples,
        "min_sec": min(samples),
        "max_sec": max(samples),
        "mean_sec": mean(samples),
        "median_sec": median(samples),
    }


def run_benchmark(
    func: Callable[[], Any],
    *,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        func()

    samples: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        func()
        samples.append(time.perf_counter() - t0)
    return summarize_samples(samples)


def available_cpu_count() -> int:
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    return max(1, os.cpu_count() or 1)


def _discover_omp_library() -> str | None:
    candidates = [
        ctypes.util.find_library("gomp"),
        ctypes.util.find_library("omp"),
        ctypes.util.find_library("vcomp140"),
        "libgomp.so.1",
        "libomp.so",
        "libomp.dylib",
        "vcomp140.dll",
    ]
    for candidate in candidates:
        if candidate:
            return candidate
    return None


def set_omp_threads(num_threads: int) -> bool:
    global _OMP_SETTER
    if num_threads <= 0:
        return False
    if _OMP_SETTER is None:
        lib_name = _discover_omp_library()
        if lib_name is None:
            _OMP_SETTER = False
        else:
            try:
                runtime = ctypes.CDLL(lib_name)
                runtime.omp_set_num_threads.argtypes = [ctypes.c_int]
                runtime.omp_set_num_threads.restype = None
                _OMP_SETTER = runtime.omp_set_num_threads
            except Exception:
                _OMP_SETTER = False
    if _OMP_SETTER is False:
        return False
    _OMP_SETTER(num_threads)
    return True


def resolve_openmp_threads(
    raw_value: str,
    *,
    workers: int = 1,
) -> int:
    if raw_value != "auto":
        value = int(raw_value)
        if value <= 0:
            raise ValueError("openmp threads must be positive or 'auto'")
        return value
    return max(1, available_cpu_count() // max(1, workers))


def make_frames(
    *,
    frames: int,
    height: int,
    width: int,
    dtype: np.dtype,
    channels: int = 3,
    seed: int = 0,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    dtype = np.dtype(dtype)
    shape = (height, width, channels) if channels > 1 else (height, width)

    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        data = rng.integers(
            low=0,
            high=info.max + 1,
            size=(frames, *shape),
            dtype=dtype,
        )
    else:
        data = rng.random((frames, *shape), dtype=np.float32).astype(dtype)

    return [data[i].copy() for i in range(frames)]


def make_weights(frames: int) -> list[float]:
    if frames <= 1:
        return [1.0]
    return np.linspace(0.25, 1.0, frames, dtype=np.float32).tolist()


def _iter_dataset_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    if root.is_file():
        return []
    return [root] + sorted(path for path in root.rglob("*") if path.is_dir())


def _cache_meta_path(root: Path) -> Path:
    return root / "meta.json"


def _cache_data_path(root: Path) -> Path:
    return root / "frames.dat"


def _is_cache_dir(root: Path) -> bool:
    return _cache_meta_path(root).is_file() and _cache_data_path(root).is_file()


def _load_cache_meta(root: Path) -> dict[str, Any]:
    meta = json.loads(_cache_meta_path(root).read_text(encoding="utf-8"))
    required = {"frames", "shape", "dtype"}
    missing = required - set(meta)
    if missing:
        raise ValueError(f"benchmark cache metadata missing keys: {sorted(missing)}")
    return meta


def discover_cache_dataset(
    *,
    input_dir: str | None,
    frames: int,
) -> tuple[Path | None, dict[str, Any] | None]:
    roots: list[Path] = []
    if input_dir:
        roots.append(Path(input_dir))
    else:
        roots.extend(DEFAULT_INPUT_DIRS)

    for root in roots:
        if root.is_file():
            continue
        for dataset_dir in _iter_dataset_dirs(root):
            if not _is_cache_dir(dataset_dir):
                continue
            meta = _load_cache_meta(dataset_dir)
            if int(meta["frames"]) >= frames:
                return dataset_dir, meta
    return None, None


def open_cache_batch(
    root: Path,
    meta: dict[str, Any],
    *,
    frames: int,
) -> np.memmap:
    shape = tuple(int(v) for v in meta["shape"])
    if len(shape) < 2:
        raise ValueError(f"benchmark cache shape is invalid: {shape}")
    if int(meta["frames"]) < frames:
        raise ValueError(
            f"benchmark cache only has {meta['frames']} frames, requested {frames}"
        )
    full_shape = (int(meta["frames"]), *shape)
    batch = np.memmap(
        _cache_data_path(root),
        dtype=np.dtype(meta["dtype"]),
        mode="r",
        shape=full_shape,
    )
    return batch[:frames]


def discover_image_paths(
    *,
    input_dir: str | None,
    frames: int,
) -> tuple[Path | None, list[Path]]:
    roots: list[Path] = []
    if input_dir:
        roots.append(Path(input_dir))
    else:
        roots.extend(DEFAULT_INPUT_DIRS)

    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            candidates = [root] if root.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES else []
            if len(candidates) >= frames:
                return root.parent, candidates[:frames]
            continue

        search_dirs = _iter_dataset_dirs(root)
        for dataset_dir in search_dirs:
            candidates = sorted(
                p for p in dataset_dir.iterdir()
                if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
            )
            if len(candidates) >= frames:
                return dataset_dir, candidates[:frames]
    return None, []


def load_benchmark_image(path: Path) -> np.ndarray:
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to decode benchmark image: {path}")
    return image


def load_frames_from_paths(paths: list[Path]) -> list[np.ndarray]:
    frames = [load_benchmark_image(path) for path in paths]
    first = frames[0]
    for idx, frame in enumerate(frames[1:], start=1):
        if frame.shape != first.shape:
            raise ValueError(
                f"benchmark input shape mismatch: frame 0 {first.shape} vs frame {idx} {frame.shape}"
            )
        if frame.dtype != first.dtype:
            raise ValueError(
                f"benchmark input dtype mismatch: frame 0 {first.dtype} vs frame {idx} {frame.dtype}"
            )
    return frames


def prepare_frames(
    *,
    frames: int,
    height: int,
    width: int,
    dtype: np.dtype,
    channels: int = 3,
    seed: int = 0,
    input_dir: str | None = None,
    input_mode: str = "auto",
) -> tuple[list[np.ndarray], dict[str, Any]]:
    if input_mode not in {"auto", "cache", "images", "synthetic"}:
        raise ValueError(f"unsupported input_mode: {input_mode}")

    if input_mode in {"auto", "cache"}:
        cache_root, cache_meta = discover_cache_dataset(input_dir=input_dir, frames=frames)
        if cache_root is not None and cache_meta is not None:
            batch = open_cache_batch(cache_root, cache_meta, frames=frames)
            source = {
                "mode": "raw_cache",
                "input_dir": str(cache_root),
                "requested_input_mode": input_mode,
                "resolved_frames": int(batch.shape[0]),
                "resolved_shape": list(batch.shape[1:]),
                "resolved_dtype": str(batch.dtype),
                "cache_files": [
                    str(_cache_meta_path(cache_root)),
                    str(_cache_data_path(cache_root)),
                ],
            }
            return [batch[idx] for idx in range(batch.shape[0])], source
        if input_mode == "cache":
            raise FileNotFoundError(f"no raw cache dataset found for frames={frames} under: {input_dir or DEFAULT_INPUT_DIRS}")

    if input_mode in {"auto", "images"}:
        root, image_paths = discover_image_paths(input_dir=input_dir, frames=frames)
        if image_paths:
            loaded = load_frames_from_paths(image_paths)
            first = loaded[0]
            source = {
                "mode": "images",
                "input_dir": str(root),
                "requested_input_mode": input_mode,
                "resolved_frames": len(loaded),
                "resolved_shape": list(first.shape),
                "resolved_dtype": str(first.dtype),
                "sample_paths": [str(path) for path in image_paths[:3]],
            }
            return loaded, source
        if input_mode == "images":
            raise FileNotFoundError(f"no image dataset found for frames={frames} under: {input_dir or DEFAULT_INPUT_DIRS}")

    synthetic = make_frames(
        frames=frames,
        height=height,
        width=width,
        dtype=dtype,
        channels=channels,
        seed=seed,
    )
    source = {
        "mode": "synthetic",
        "input_dir": None,
        "requested_input_mode": input_mode,
        "resolved_frames": len(synthetic),
        "resolved_shape": list(synthetic[0].shape),
        "resolved_dtype": str(synthetic[0].dtype),
        "seed": seed,
    }
    return synthetic, source


def prepare_batch(
    *,
    frames: int,
    height: int,
    width: int,
    dtype: np.dtype,
    channels: int = 3,
    seed: int = 0,
    input_dir: str | None = None,
    input_mode: str = "auto",
) -> tuple[np.ndarray, dict[str, Any]]:
    frame_list, source = prepare_frames(
        frames=frames,
        height=height,
        width=width,
        dtype=dtype,
        channels=channels,
        seed=seed,
        input_dir=input_dir,
        input_mode=input_mode,
    )
    batch = np.stack(frame_list, axis=0)
    return batch, source
