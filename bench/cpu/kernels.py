"""算法内核微基准。

用途：
- 只测数值核，不测 DAG 调度、文件解码、EXIF、图像保存
- 用来判断哪些热点最值得优先做 C++ / pybind11 重写

覆盖内容：
- MaxMerger
- MeanMerger
- FastGaussianParam 自定义累加
- FastGaussianParam partial 归约
- Huber 融合加权累加
- FastGaussianParam mask 融合累加
- FastGaussianParam 累加
- SigmaClipping rejected 融合累加
- SigmaClipping rejected + mask 融合累加
- SigmaClippingMerger 单轮 pass
- HuberWeightedMerger 单轮 pass
- 中位数块计算
- 二维空间中值滤波
- 对齐星点特征提取与粗匹配
- 对齐小波 bandpass 重建 core

运行方式：
```bash
python -m bench.cpu.kernels --frames 128 --height 1080 --width 1920 --dtype uint16
python -m bench.cpu.kernels --frames 64 --height 2160 --width 3840 --dtype uint8 --repeat 3
```
"""

from __future__ import annotations

import argparse
import cv2
from typing import Any

import numpy as np

from bench.common import (
    collect_env_info,
    make_weights,
    prepare_frames,
    print_or_save_report,
    run_benchmark,
)
from hoshicore._custom_op import build_info as custom_ops_build_info
import hoshicore._custom_op.ops.alignment as alignment_ops
import hoshicore._custom_op.ops.fgp as fgp_ops
import hoshicore._custom_op.ops.filter as filter_ops
import hoshicore._custom_op.ops.max as max_ops
import hoshicore._custom_op.ops.median as median_ops
import hoshicore._custom_op.ops.noise as noise_ops
import hoshicore._custom_op.ops.sigma_clip as sigma_clip_chunk_ops
import hoshicore._custom_op.ops.wavelet as wavelet_ops
from hoshicore.component.data_container import DTYPE_MAX_VALUE


def parse_cases(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or None


def bench_max_combine(frames: list[np.ndarray], weights: list[float], int_weight: bool) -> None:
    from hoshicore.component.merger import MaxMerger

    merger = MaxMerger(int_weight=int_weight)
    for frame, weight in zip(frames, weights):
        merger.merge(frame, weight)
    _ = merger.merged_image


def bench_mean_merge(frames: list[np.ndarray], weights: list[float], int_weight: bool) -> None:
    from hoshicore.component.merger import MeanMerger

    merger = MeanMerger(int_weight=int_weight)
    for frame, weight in zip(frames, weights):
        merger.merge(frame, weight)
    _ = merger.merged_image
    _ = merger.result


def bench_fast_gaussian_add(frames: list[np.ndarray]) -> None:
    from hoshicore.component.data_container import FastGaussianParam

    total = FastGaussianParam(frames[0], source_dtype=frames[0].dtype)
    for frame in frames[1:]:
        total = total + FastGaussianParam(frame, source_dtype=frame.dtype)
    _ = total.mu
    _ = total.var


def bench_fgp_accumulate_stream(frames: list[np.ndarray]) -> None:
    from hoshicore.component.data_container import FastGaussianParam

    total = FastGaussianParam(frames[0], source_dtype=frames[0].dtype)
    for frame in frames[1:]:
        fgp_ops.fgp_accumulate(total, frame)
    _ = total.mu
    _ = total.var


def bench_max_combine_stream_backend(
    frames: list[np.ndarray],
    *,
    backend: str,
) -> None:
    result = np.array(frames[0], copy=True)
    combine = {
        "numpy": max_ops.max_combine_numpy,
        "compiled": max_ops.max_combine_compiled,
    }[backend]
    for frame in frames[1:]:
        combine(result, frame)


def build_threshold_max_stats(frames: list[np.ndarray]) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    frames_float = [frame.astype(np.float64, copy=False) for frame in frames]
    stack = np.stack(frames_float, axis=0)
    mean_img = np.mean(stack, axis=0, dtype=np.float64)
    std_img = np.std(stack, axis=0, dtype=np.float64)
    return frames_float, mean_img, std_img


def bench_threshold_max_merge_stream_backend(
    frames: list[np.ndarray],
    mean_img: np.ndarray,
    std_img: np.ndarray,
    weights: list[float],
    *,
    backend: str,
    n_sigma: float = 3.0,
) -> None:
    merge = {
        "numpy": max_ops.threshold_max_merge_numpy,
        "compiled": max_ops.threshold_max_merge_compiled,
    }[backend]
    result = np.array(mean_img, copy=True)
    for frame, weight in zip(frames, weights):
        merge(frame, mean_img, std_img, result, n_sigma, weight)


def build_equalize_noise_inputs(
    frames: list[np.ndarray],
) -> list[tuple[np.ndarray, np.ndarray, float, float, float, float]]:
    payloads: list[tuple[np.ndarray, np.ndarray, float, float, float, float]] = []
    for frame in frames:
        max_img = frame.astype(np.float64, copy=False)
        filled_std_img = np.sqrt(np.maximum(max_img, 1.0)).astype(np.float64, copy=False)
        payloads.append((
            max_img,
            filled_std_img,
            8.0,
            1.75,
            float(np.max(max_img)),
            0.9,
        ))
    return payloads


def bench_equalize_noise_correct_stream_backend(
    payloads: list[tuple[np.ndarray, np.ndarray, float, float, float, float]],
    *,
    backend: str,
) -> None:
    correct = {
        "numpy": noise_ops.equalize_noise_correct_numpy,
        "compiled": noise_ops.equalize_noise_correct_compiled,
    }[backend]
    for max_img, filled_std_img, sigma_ref, c_n_eff, max_value, highlight_preserve in payloads:
        correct(max_img, filled_std_img, sigma_ref, c_n_eff, max_value, highlight_preserve)


def bench_fgp_accumulate_stream_backend(
    frames: list[np.ndarray],
    *,
    backend: str,
) -> None:
    from hoshicore.component.data_container import FastGaussianParam

    total = FastGaussianParam(frames[0], source_dtype=frames[0].dtype)
    accumulate = {
        "numpy": fgp_ops.fgp_accumulate_numpy,
        "compiled": fgp_ops.fgp_accumulate_compiled,
    }[backend]
    for frame in frames[1:]:
        accumulate(total, frame)
    _ = total.mu
    _ = total.var


def clone_fgp(param):
    from hoshicore.component.data_container import FastGaussianParam

    return FastGaussianParam(
        sum_mu=np.array(param.sum_mu, copy=True),
        square_sum=np.array(param.square_sum, copy=True),
        n=np.array(param.n, copy=True),
        ddof=param.ddof,
        source_dtype=param.source_dtype,
        inplace_calc=True,
    )


def build_fgp_partials(frames: list[np.ndarray]):
    from hoshicore.component.data_container import FastGaussianParam

    return [
        FastGaussianParam(np.array(frame, copy=True), source_dtype=frame.dtype)
        for frame in frames
    ]


def bench_fgp_add_partial_reduce_backend(
    partials,
    *,
    backend: str,
) -> None:
    add = {
        "numpy": fgp_ops.fgp_add_numpy,
        "compiled": fgp_ops.fgp_add_compiled,
    }[backend]
    total = clone_fgp(partials[0])
    for partial in partials[1:]:
        add(total, partial)
    _ = total.mu
    _ = total.var


def build_accumulators(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from hoshicore.component.merger import _accum_dtypes

    sum_dt, sq_dt, n_dt = _accum_dtypes(frame.dtype, False)
    return (
        np.zeros(frame.shape, dtype=sum_dt),
        np.zeros(frame.shape, dtype=sq_dt),
        np.zeros(frame.shape, dtype=n_dt),
    )


def build_spatial_mask(frame: np.ndarray, mask_density: float) -> np.ndarray:
    plane = frame[..., 0] if frame.ndim == 3 else frame
    cutoff = float(np.quantile(plane, max(0.0, min(1.0, 1.0 - mask_density))))
    return plane >= cutoff


def build_sigma_clip_bounds(
    stats,
    *,
    rej_high: float,
    rej_low: float,
) -> tuple[np.ndarray, np.ndarray]:
    valid = stats.n > 0
    ref_mu = np.where(valid, stats.mu, 0)
    ref_std = np.where(valid, np.sqrt(np.maximum(stats.var, 0)), 0)
    rej_dtype = stats.source_dtype
    rej_high_img = np.array(
        np.floor(ref_mu + ref_std * rej_high).clip(
            min=0,
            max=DTYPE_MAX_VALUE[rej_dtype],
        ),
        dtype=rej_dtype,
    )
    rej_low_img = np.array(
        np.ceil(ref_mu - ref_std * rej_low).clip(
            min=0,
            max=DTYPE_MAX_VALUE[rej_dtype],
        ),
        dtype=rej_dtype,
    )
    return rej_high_img, rej_low_img


def build_huber_accumulators(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.zeros(frame.shape, dtype=np.float64),
        np.zeros(frame.shape, dtype=np.float64),
    )


def bench_huber_weighted_accumulate_stream_backend(
    frames: list[np.ndarray],
    ref_mean: np.ndarray,
    ref_std: np.ndarray,
    *,
    backend: str,
    frame_weight: float | None = None,
) -> None:
    from hoshicore.component.data_container import HuberMeanParam

    accumulate = {
        "numpy": fgp_ops.huber_weighted_accumulate_numpy,
        "compiled": fgp_ops.huber_weighted_accumulate_compiled,
    }[backend]
    weighted_sum, weight_total = build_huber_accumulators(frames[0])
    total = HuberMeanParam(
        weighted_sum=weighted_sum,
        weight_total=weight_total,
        source_dtype=frames[0].dtype,
    )
    for frame in frames:
        accumulate(total, frame, ref_mean, ref_std, 1.345, frame_weight)
    _ = total.mu


def bench_fgp_masked_mean_merge_stream(
    frames: list[np.ndarray],
    mask: np.ndarray,
) -> None:
    sum_mu, square_sum, count = build_accumulators(frames[0])
    for frame in frames:
        fgp_ops.fgp_masked_mean_merge(frame, mask, sum_mu, square_sum, count)


def bench_fgp_masked_mean_merge_stream_backend(
    frames: list[np.ndarray],
    mask: np.ndarray,
    *,
    backend: str,
) -> None:
    merge = {
        "numpy": fgp_ops.fgp_masked_mean_merge_numpy,
        "compiled": fgp_ops.fgp_masked_mean_merge_compiled,
    }[backend]
    sum_mu, square_sum, count = build_accumulators(frames[0])
    for frame in frames:
        merge(frame, mask, sum_mu, square_sum, count)


def bench_sigma_clip_fused_merge_stream(
    frames: list[np.ndarray],
    rej_high_img: np.ndarray,
    rej_low_img: np.ndarray,
) -> None:
    sum_mu, square_sum, count = build_accumulators(frames[0])
    for frame in frames:
        fgp_ops.sigma_clip_fused_merge(
            frame,
            rej_high_img,
            rej_low_img,
            sum_mu,
            square_sum,
            count,
        )


def bench_sigma_clip_fused_merge_stream_backend(
    frames: list[np.ndarray],
    rej_high_img: np.ndarray,
    rej_low_img: np.ndarray,
    *,
    backend: str,
) -> None:
    merge = {
        "numpy": fgp_ops.sigma_clip_fused_merge_numpy,
        "compiled": fgp_ops.sigma_clip_fused_merge_compiled,
    }[backend]
    sum_mu, square_sum, count = build_accumulators(frames[0])
    for frame in frames:
        merge(
            frame,
            rej_high_img,
            rej_low_img,
            sum_mu,
            square_sum,
            count,
        )


def bench_sigma_clip_fused_masked_merge_stream(
    frames: list[np.ndarray],
    mask: np.ndarray,
    rej_high_img: np.ndarray,
    rej_low_img: np.ndarray,
) -> None:
    sum_mu, square_sum, count = build_accumulators(frames[0])
    for frame in frames:
        fgp_ops.sigma_clip_fused_masked_merge(
            frame,
            mask,
            rej_high_img,
            rej_low_img,
            sum_mu,
            square_sum,
            count,
        )


def bench_sigma_clip_fused_masked_merge_stream_backend(
    frames: list[np.ndarray],
    mask: np.ndarray,
    rej_high_img: np.ndarray,
    rej_low_img: np.ndarray,
    *,
    backend: str,
) -> None:
    merge = {
        "numpy": fgp_ops.sigma_clip_fused_masked_merge_numpy,
        "compiled": fgp_ops.sigma_clip_fused_masked_merge_compiled,
    }[backend]
    sum_mu, square_sum, count = build_accumulators(frames[0])
    for frame in frames:
        merge(
            frame,
            mask,
            rej_high_img,
            rej_low_img,
            sum_mu,
            square_sum,
            count,
        )


def build_mean_stats(frames: list[np.ndarray], weights: list[float], int_weight: bool):
    from hoshicore.component.merger import MeanMerger

    merger = MeanMerger(int_weight=int_weight)
    for frame, weight in zip(frames, weights):
        merger.merge(frame, weight)
    merger.result.inplace_calc = False
    return merger.result


def bench_sigma_clip_pass(frames: list[np.ndarray], weights: list[float], int_weight: bool) -> None:
    from hoshicore.component.merger import SigmaClippingMerger

    stats = build_mean_stats(frames, weights, int_weight)
    clip = SigmaClippingMerger(ref_img=stats, rej_high=3.0, rej_low=3.0)
    for frame in frames:
        clip.merge(frame, None)
    _ = clip.result


def bench_huber_pass(frames: list[np.ndarray], weights: list[float], int_weight: bool) -> None:
    from hoshicore.component.merger import HuberWeightedMerger

    stats = build_mean_stats(frames, weights, int_weight)
    huber = HuberWeightedMerger(ref_stats=stats, huber_c=1.345)
    for frame in frames:
        huber.merge(frame, None)
    _ = huber.merged_image


def build_median_chunk_stacks(
    frames: list[np.ndarray],
    chunk_rows: int,
) -> list[np.ndarray]:
    first = frames[0]
    h, w = first.shape[:2]
    channels = first.shape[2] if first.ndim == 3 else None
    dtype = first.dtype
    stacks: list[np.ndarray] = []
    for row_start in range(0, h, chunk_rows):
        row_end = min(row_start + chunk_rows, h)
        actual_rows = row_end - row_start
        if channels is None:
            stack = np.empty((len(frames), actual_rows, w), dtype=dtype)
        else:
            stack = np.empty((len(frames), actual_rows, w, channels),
                             dtype=dtype)
        for frame_idx, frame in enumerate(frames):
            stack[frame_idx] = frame[row_start:row_end]
        stacks.append(stack)
    return stacks


def bench_median_reduce_chunk_backend(
    stacks: list[np.ndarray],
    *,
    backend: str,
) -> None:
    reduce_chunk = {
        "numpy": median_ops.median_reduce_chunk_numpy,
        "compiled": median_ops.median_reduce_chunk_compiled,
    }[backend]
    for stack in stacks:
        _ = reduce_chunk(stack)


def bench_median_filter_2d_backend(
    image: np.ndarray,
    *,
    ksize: int,
    backend: str,
) -> None:
    median_filter = {
        "numpy": filter_ops.median_filter_2d_numpy,
        "compiled": filter_ops.median_filter_2d_compiled,
    }[backend]
    _ = median_filter(image, ksize)


def build_alignment_match_inputs(
    n_points: int,
    *,
    k: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=(n_points, 3))
    vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)
    vec2 = vec + rng.normal(scale=1e-4, size=vec.shape)
    vec2 = vec2 / np.linalg.norm(vec2, axis=1, keepdims=True)
    vol = rng.random(n_points) * 10.0 + 1.0
    vol2 = vol * (1.0 + rng.normal(scale=1e-3, size=n_points))
    pts = rng.random((n_points, 2)) * 4000.0
    pts2 = pts + rng.normal(scale=1.0, size=pts.shape)
    return vec, vec2, vol, vol2, pts, pts2, k


def bench_extract_point_features_backend(
    vec: np.ndarray,
    vol: np.ndarray,
    k: int,
    *,
    backend: str,
) -> None:
    extract = {
        "numpy": alignment_ops.extract_point_features_numpy,
        "compiled": alignment_ops.extract_point_features_compiled,
    }[backend]
    _ = extract(vec, vol, k)


def bench_find_initial_match_backend(
    features1: np.ndarray,
    features2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    vectors1: np.ndarray,
    vectors2: np.ndarray,
    *,
    backend: str,
) -> None:
    find_match = {
        "numpy": alignment_ops.find_initial_match_numpy,
        "compiled": alignment_ops.find_initial_match_compiled,
    }[backend]
    _ = find_match(features1, features2, pts1, pts2, vectors1, vectors2)


def build_wavelet_input(frame: np.ndarray) -> np.ndarray:
    image = frame[..., 0] if frame.ndim == 3 else frame
    image = image.astype(np.float64, copy=False)
    if np.issubdtype(frame.dtype, np.integer):
        image = image / np.iinfo(frame.dtype).max
    return np.ascontiguousarray(image)


def bench_wavelet_dec_rec_core_backend(
    image: np.ndarray,
    level: int,
    *,
    backend: str,
) -> None:
    fn = {
        "numpy": wavelet_ops.wavelet_dec_rec_core_numpy,
        "compiled": wavelet_ops.wavelet_dec_rec_core_compiled,
    }[backend]
    _ = fn(image, level)


def bench_wavelet_dec_rec_backend(
    image: np.ndarray,
    resize_factor: float,
    *,
    backend: str,
) -> None:
    if backend == "auto":
        _ = wavelet_ops.wavelet_dec_rec(image, resize_factor)
        return
    if backend != "numpy":
        raise ValueError(f"Unknown wavelet backend: {backend}")
    level = wavelet_ops._wavelet_level(resize_factor)
    small = cv2.resize(image, None, fx=resize_factor, fy=resize_factor)
    rec = wavelet_ops.wavelet_dec_rec_core_numpy(small, level)
    _ = cv2.resize(rec, (image.shape[1], image.shape[0]))


def build_sigma_clip_chunk_stack(
    frames: list[np.ndarray], chunk_rows: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a 2D chunk stack + FGP totals for sigma_clip_chunk benchmarking."""
    first = frames[0]
    h, w = first.shape[:2]
    channels = first.shape[2] if first.ndim == 3 else 1
    # Use chunk_rows from the middle of the image
    row_start = max(0, h // 2 - chunk_rows // 2)
    row_end = min(h, row_start + chunk_rows)
    actual_rows = row_end - row_start
    plane_size = actual_rows * w * channels

    n_frames = len(frames)
    stack_2d = np.empty((n_frames, plane_size), dtype=first.dtype)
    for f, frame in enumerate(frames):
        stack_2d[f] = frame[row_start:row_end].reshape(-1)

    stack_f64 = stack_2d.astype(np.float64)
    total_sum = stack_f64.sum(axis=0)
    total_sq = (stack_f64 ** 2).sum(axis=0)
    total_n = np.full(plane_size, float(n_frames))
    return stack_2d, total_sum, total_sq, total_n


def bench_sigma_clip_chunk_backend(
    stack_2d: np.ndarray,
    total_sum: np.ndarray,
    total_sq: np.ndarray,
    total_n: np.ndarray,
    *,
    backend: str,
) -> None:
    fn = {
        "numpy": sigma_clip_chunk_ops.sigma_clip_iterative_chunk_numpy,
        "compiled": sigma_clip_chunk_ops.sigma_clip_iterative_chunk_compiled,
    }[backend]
    _ = fn(stack_2d, total_sum, total_sq, total_n, 3.0, 3.0, 5)


def bench_sigma_clip_chunk_full_backend(
    stack_2d: np.ndarray,
    *,
    backend: str,
) -> None:
    stack_f64 = stack_2d.astype(np.float64)
    total_sum = stack_f64.sum(axis=0)
    total_sq = (stack_f64 ** 2).sum(axis=0)
    total_n = np.full(stack_2d.shape[1], float(stack_2d.shape[0]))
    bench_sigma_clip_chunk_backend(
        stack_2d,
        total_sum,
        total_sq,
        total_n,
        backend=backend,
    )


def bench_sigma_clip_fused_chunk_backend(
    stack_2d: np.ndarray,
    *,
    backend: str,
) -> None:
    fn = {
        "numpy": sigma_clip_chunk_ops.sigma_clip_fused_chunk_numpy,
        "compiled": sigma_clip_chunk_ops.sigma_clip_fused_chunk_compiled,
    }[backend]
    _ = fn(stack_2d, 3.0, 3.0, 5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--dtype", type=str, default="uint16")
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--input-mode", choices=["auto", "cache", "images", "synthetic"], default="auto")
    parser.add_argument("--chunk-rows", type=int, default=32)
    parser.add_argument("--filter-ksize", type=int, default=25)
    parser.add_argument("--alignment-points", type=int, default=1000)
    parser.add_argument("--alignment-k", type=int, default=15)
    parser.add_argument("--wavelet-level", type=int, default=4)
    parser.add_argument("--wavelet-resize-factor", type=float, default=1.0)
    parser.add_argument("--mask-density", type=float, default=0.75)
    parser.add_argument("--sigma-rej-high", type=float, default=3.0)
    parser.add_argument("--sigma-rej-low", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--cases", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    requested_cases = parse_cases(args.cases)
    dtype = np.dtype(args.dtype)
    wavelet_registry_names = {
        "wavelet_dec_rec_core_numpy",
        "wavelet_dec_rec_core_compiled",
        "wavelet_dec_rec_numpy",
        "wavelet_dec_rec_auto",
    }
    wavelet_only = (
        requested_cases is not None
        and set(requested_cases).issubset(wavelet_registry_names)
    )
    frame_count = 1 if wavelet_only else args.frames
    frames, input_source = prepare_frames(
        frames=frame_count,
        height=args.height,
        width=args.width,
        dtype=dtype,
        channels=args.channels,
        seed=args.seed,
        input_dir=args.input_dir,
        input_mode=args.input_mode,
    )

    if wavelet_only:
        wavelet_input = build_wavelet_input(frames[0])
        registry: dict[str, Any] = {
            "wavelet_dec_rec_core_numpy": lambda: bench_wavelet_dec_rec_core_backend(
                wavelet_input,
                args.wavelet_level,
                backend="numpy",
            ),
            "wavelet_dec_rec_core_compiled": lambda: bench_wavelet_dec_rec_core_backend(
                wavelet_input,
                args.wavelet_level,
                backend="compiled",
            ),
            "wavelet_dec_rec_numpy": lambda: bench_wavelet_dec_rec_backend(
                wavelet_input,
                args.wavelet_resize_factor,
                backend="numpy",
            ),
            "wavelet_dec_rec_auto": lambda: bench_wavelet_dec_rec_backend(
                wavelet_input,
                args.wavelet_resize_factor,
                backend="auto",
            ),
        }
        cases = {
            case_name: run_benchmark(
                registry[case_name],
                warmup=args.warmup,
                repeat=args.repeat,
            )
            for case_name in requested_cases
        }
        report = {
            "suite": "kernels",
            "env": collect_env_info(),
            "custom_ops": custom_ops_build_info(),
            "config": {
                "frames": frame_count,
                "height": args.height,
                "width": args.width,
                "dtype": args.dtype,
                "channels": args.channels,
                "input_dir": args.input_dir,
                "input_mode": args.input_mode,
                "wavelet_level": args.wavelet_level,
                "wavelet_resize_factor": args.wavelet_resize_factor,
                "seed": args.seed,
                "warmup": args.warmup,
                "repeat": args.repeat,
                "cases": requested_cases,
            },
            "input_source": input_source,
            "results": cases,
        }
        print_or_save_report(report, args.output_json)
        return

    weights = make_weights(frame_count)
    spatial_mask = build_spatial_mask(frames[0], args.mask_density)
    fgp_partials = build_fgp_partials(frames)
    threshold_frames, threshold_mean_img, threshold_std_img = build_threshold_max_stats(frames)
    median_chunk_stacks = build_median_chunk_stacks(frames, args.chunk_rows)
    alignment_inputs = None
    alignment_features = None

    def get_alignment_inputs():
        nonlocal alignment_inputs
        if alignment_inputs is None:
            alignment_inputs = build_alignment_match_inputs(
                args.alignment_points,
                k=args.alignment_k,
                seed=args.seed,
            )
        return alignment_inputs

    def get_alignment_features():
        nonlocal alignment_features
        if alignment_features is None:
            vec, vec2, vol, vol2, _, _, k = get_alignment_inputs()
            alignment_features = (
                alignment_ops.extract_point_features_numpy(vec, vol, k),
                alignment_ops.extract_point_features_numpy(vec2, vol2, k),
            )
        return alignment_features

    def bench_alignment_extract(backend: str) -> None:
        vec, _, vol, _, _, _, k = get_alignment_inputs()
        bench_extract_point_features_backend(vec, vol, k, backend=backend)

    def bench_alignment_match(backend: str) -> None:
        vec, vec2, _, _, pts, pts2, _ = get_alignment_inputs()
        features1, features2 = get_alignment_features()
        bench_find_initial_match_backend(
            features1,
            features2,
            pts,
            pts2,
            vec,
            vec2,
            backend=backend,
        )

    wavelet_input = build_wavelet_input(frames[0])
    sc_chunk_stack, sc_chunk_sum, sc_chunk_sq, sc_chunk_n = build_sigma_clip_chunk_stack(
        frames, args.chunk_rows)
    equalize_noise_payloads = build_equalize_noise_inputs(frames)
    sigma_stats = build_mean_stats(frames, weights, True)
    huber_ref_mean = sigma_stats.mu.astype(np.float32)
    huber_ref_std = np.sqrt(np.maximum(sigma_stats.var, 0)).astype(np.float32)
    rej_high_img, rej_low_img = build_sigma_clip_bounds(
        sigma_stats,
        rej_high=args.sigma_rej_high,
        rej_low=args.sigma_rej_low,
    )

    registry: dict[str, Any] = {
        "max_combine_int_weight": lambda: bench_max_combine(frames, weights, True),
        "mean_merge_int_weight": lambda: bench_mean_merge(frames, weights, True),
        "fast_gaussian_add": lambda: bench_fast_gaussian_add(frames),
        "max_combine_stream_numpy": lambda: bench_max_combine_stream_backend(frames, backend="numpy"),
        "max_combine_stream_compiled": lambda: bench_max_combine_stream_backend(frames, backend="compiled"),
        "threshold_max_merge_stream_numpy": lambda: bench_threshold_max_merge_stream_backend(
            threshold_frames,
            threshold_mean_img,
            threshold_std_img,
            weights,
            backend="numpy",
        ),
        "threshold_max_merge_stream_compiled": lambda: bench_threshold_max_merge_stream_backend(
            threshold_frames,
            threshold_mean_img,
            threshold_std_img,
            weights,
            backend="compiled",
        ),
        "equalize_noise_correct_stream_numpy": lambda: bench_equalize_noise_correct_stream_backend(
            equalize_noise_payloads,
            backend="numpy",
        ),
        "equalize_noise_correct_stream_compiled": lambda: bench_equalize_noise_correct_stream_backend(
            equalize_noise_payloads,
            backend="compiled",
        ),
        "fgp_accumulate_stream_numpy": lambda: bench_fgp_accumulate_stream_backend(frames, backend="numpy"),
        "fgp_accumulate_stream_compiled": lambda: bench_fgp_accumulate_stream_backend(frames, backend="compiled"),
        "fgp_add_partial_reduce_numpy": lambda: bench_fgp_add_partial_reduce_backend(
            fgp_partials,
            backend="numpy",
        ),
        "fgp_add_partial_reduce_compiled": lambda: bench_fgp_add_partial_reduce_backend(
            fgp_partials,
            backend="compiled",
        ),
        "huber_weighted_accumulate_stream_numpy": lambda: bench_huber_weighted_accumulate_stream_backend(
            frames,
            huber_ref_mean,
            huber_ref_std,
            backend="numpy",
        ),
        "huber_weighted_accumulate_stream_compiled": lambda: bench_huber_weighted_accumulate_stream_backend(
            frames,
            huber_ref_mean,
            huber_ref_std,
            backend="compiled",
        ),
        "fgp_masked_mean_merge_stream_numpy": lambda: bench_fgp_masked_mean_merge_stream_backend(
            frames,
            spatial_mask,
            backend="numpy",
        ),
        "fgp_masked_mean_merge_stream_compiled": lambda: bench_fgp_masked_mean_merge_stream_backend(
            frames,
            spatial_mask,
            backend="compiled",
        ),
        "sigma_clip_fused_merge_stream_numpy": lambda: bench_sigma_clip_fused_merge_stream_backend(
            frames,
            rej_high_img,
            rej_low_img,
            backend="numpy",
        ),
        "sigma_clip_fused_merge_stream_compiled": lambda: bench_sigma_clip_fused_merge_stream_backend(
            frames,
            rej_high_img,
            rej_low_img,
            backend="compiled",
        ),
        "sigma_clip_fused_masked_merge_stream_numpy": lambda: bench_sigma_clip_fused_masked_merge_stream_backend(
            frames,
            spatial_mask,
            rej_high_img,
            rej_low_img,
            backend="numpy",
        ),
        "sigma_clip_fused_masked_merge_stream_compiled": lambda: bench_sigma_clip_fused_masked_merge_stream_backend(
            frames,
            spatial_mask,
            rej_high_img,
            rej_low_img,
            backend="compiled",
        ),
        "sigma_clip_pass": lambda: bench_sigma_clip_pass(frames, weights, True),
        "huber_pass": lambda: bench_huber_pass(frames, weights, True),
        "median_reduce_chunk_numpy": lambda: bench_median_reduce_chunk_backend(
            median_chunk_stacks,
            backend="numpy",
        ),
        "median_reduce_chunk_compiled": lambda: bench_median_reduce_chunk_backend(
            median_chunk_stacks,
            backend="compiled",
        ),
        "median_filter_2d_numpy": lambda: bench_median_filter_2d_backend(
            frames[0],
            ksize=args.filter_ksize,
            backend="numpy",
        ),
        "median_filter_2d_compiled": lambda: bench_median_filter_2d_backend(
            frames[0],
            ksize=args.filter_ksize,
            backend="compiled",
        ),
        "extract_point_features_numpy": lambda: bench_alignment_extract("numpy"),
        "extract_point_features_compiled": lambda: bench_alignment_extract("compiled"),
        "find_initial_match_numpy": lambda: bench_alignment_match("numpy"),
        "find_initial_match_compiled": lambda: bench_alignment_match("compiled"),
        "wavelet_dec_rec_core_numpy": lambda: bench_wavelet_dec_rec_core_backend(
            wavelet_input,
            args.wavelet_level,
            backend="numpy",
        ),
        "wavelet_dec_rec_core_compiled": lambda: bench_wavelet_dec_rec_core_backend(
            wavelet_input,
            args.wavelet_level,
            backend="compiled",
        ),
        "wavelet_dec_rec_numpy": lambda: bench_wavelet_dec_rec_backend(
            wavelet_input,
            args.wavelet_resize_factor,
            backend="numpy",
        ),
        "wavelet_dec_rec_auto": lambda: bench_wavelet_dec_rec_backend(
            wavelet_input,
            args.wavelet_resize_factor,
            backend="auto",
        ),
        "sigma_clip_chunk_numpy": lambda: bench_sigma_clip_chunk_backend(
            sc_chunk_stack, sc_chunk_sum, sc_chunk_sq, sc_chunk_n,
            backend="numpy",
        ),
        "sigma_clip_chunk_compiled": lambda: bench_sigma_clip_chunk_backend(
            sc_chunk_stack, sc_chunk_sum, sc_chunk_sq, sc_chunk_n,
            backend="compiled",
        ),
        "sigma_clip_iterative_chunk_numpy": lambda: bench_sigma_clip_chunk_backend(
            sc_chunk_stack, sc_chunk_sum, sc_chunk_sq, sc_chunk_n,
            backend="numpy",
        ),
        "sigma_clip_iterative_chunk_compiled": lambda: bench_sigma_clip_chunk_backend(
            sc_chunk_stack, sc_chunk_sum, sc_chunk_sq, sc_chunk_n,
            backend="compiled",
        ),
        "sigma_clip_chunk_full_numpy": lambda: bench_sigma_clip_chunk_full_backend(
            sc_chunk_stack,
            backend="numpy",
        ),
        "sigma_clip_chunk_full_compiled": lambda: bench_sigma_clip_chunk_full_backend(
            sc_chunk_stack,
            backend="compiled",
        ),
        "sigma_clip_fused_chunk_numpy": lambda: bench_sigma_clip_fused_chunk_backend(
            sc_chunk_stack,
            backend="numpy",
        ),
        "sigma_clip_fused_chunk_compiled": lambda: bench_sigma_clip_fused_chunk_backend(
            sc_chunk_stack,
            backend="compiled",
        ),
    }
    selected_cases = requested_cases or list(registry.keys())
    for case_name in selected_cases:
        if case_name not in registry:
            raise ValueError(f"Unknown case: {case_name}")

    cases: dict[str, dict[str, Any]] = {}
    for case_name in selected_cases:
        cases[case_name] = run_benchmark(
            registry[case_name],
            warmup=args.warmup,
            repeat=args.repeat,
        )

    report = {
        "suite": "kernels",
        "env": collect_env_info(),
        "custom_ops": custom_ops_build_info(),
        "config": {
            "frames": args.frames,
            "height": args.height,
            "width": args.width,
            "dtype": args.dtype,
            "channels": args.channels,
            "input_dir": args.input_dir,
            "input_mode": args.input_mode,
            "chunk_rows": args.chunk_rows,
            "filter_ksize": args.filter_ksize,
            "alignment_points": args.alignment_points,
            "alignment_k": args.alignment_k,
            "wavelet_level": args.wavelet_level,
            "wavelet_resize_factor": args.wavelet_resize_factor,
            "mask_density": args.mask_density,
            "sigma_rej_high": args.sigma_rej_high,
            "sigma_rej_low": args.sigma_rej_low,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "cases": selected_cases,
        },
        "input_source": input_source,
        "results": cases,
    }
    print_or_save_report(report, args.output_json)


if __name__ == "__main__":
    main()
