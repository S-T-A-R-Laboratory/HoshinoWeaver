"""大尺寸 FastGaussianParam 累加 benchmark。

用途：
- 固化“大图 + 多帧”的 `fgp_accumulate` 计算基线
- 比较 Python、NumPy 与 custom op/OpenMP 路径
"""

from __future__ import annotations

import argparse
from typing import Any, Callable

import numpy as np

from bench.common import (
    collect_env_info,
    prepare_batch,
    print_or_save_report,
    resolve_openmp_threads,
    run_benchmark,
)
from hoshicore._custom_op import build_info as custom_ops_build_info
import hoshicore._custom_op.ops.fgp as fgp_ops
from hoshicore.component.data_container import FastGaussianParam


def sequential_python_stream(batch: np.ndarray) -> FastGaussianParam:
    result = FastGaussianParam(np.array(batch[0], copy=True), source_dtype=batch[0].dtype)
    for idx in range(1, batch.shape[0]):
        result = result + FastGaussianParam(np.array(batch[idx], copy=True), source_dtype=batch[idx].dtype)
    return result


def sequential_numpy_stream(batch: np.ndarray) -> FastGaussianParam:
    result = FastGaussianParam(np.array(batch[0], copy=True), source_dtype=batch[0].dtype)
    for idx in range(1, batch.shape[0]):
        fgp_ops.fgp_accumulate_numpy(result, batch[idx])
    return result


def sequential_openmp_stream(batch: np.ndarray, openmp_threads: int) -> FastGaussianParam:
    module, _ = fgp_ops._load_compiled_module_result()
    if module is None or not hasattr(module, "fgp_accumulate"):
        raise RuntimeError("compiled fgp_accumulate backend is unavailable")
    if hasattr(module, "set_openmp_threads"):
        # 这个 benchmark 显式控制线程数，便于固定 OpenMP 线程配置。
        module.set_openmp_threads(int(openmp_threads))
    result = FastGaussianParam(np.array(batch[0], copy=True), source_dtype=batch[0].dtype)
    for idx in range(1, batch.shape[0]):
        fgp_ops.fgp_accumulate_compiled(result, batch[idx])
    return result


def assert_same_fgp(lhs: FastGaussianParam, rhs: FastGaussianParam) -> None:
    np.testing.assert_array_equal(lhs.sum_mu, rhs.sum_mu)
    np.testing.assert_array_equal(lhs.square_sum, rhs.square_sum)
    np.testing.assert_array_equal(lhs.n, rhs.n)


def parse_cases(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--height", type=int, default=4000)
    parser.add_argument("--width", type=int, default=6000)
    parser.add_argument("--dtype", type=str, default="uint8")
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--input-mode", choices=["auto", "cache", "images", "synthetic"], default="auto")
    parser.add_argument("--openmp-threads", type=str, default="auto")
    parser.add_argument(
        "--cases",
        type=str,
        default="single_python_stream,single_numpy_stream,single_openmp_stream",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    requested_cases = parse_cases(args.cases)
    openmp_threads = resolve_openmp_threads(args.openmp_threads)
    dtype = np.dtype(args.dtype)

    batch, input_source = prepare_batch(
        frames=args.frames,
        height=args.height,
        width=args.width,
        dtype=dtype,
        channels=args.channels,
        seed=args.seed,
        input_dir=args.input_dir,
        input_mode=args.input_mode,
    )
    # 用旧 Python 路径生成 reference，确保几种 backend 只比较性能不比较语义差异。
    reference = sequential_python_stream(batch)

    def case_single_python_stream() -> dict[str, Any]:
        result = sequential_python_stream(batch)
        assert_same_fgp(result, reference)
        return {"ok": True}

    def case_single_numpy_stream() -> dict[str, Any]:
        result = sequential_numpy_stream(batch)
        assert_same_fgp(result, reference)
        return {"ok": True}

    def case_single_openmp_stream() -> dict[str, Any]:
        result = sequential_openmp_stream(batch, openmp_threads)
        assert_same_fgp(result, reference)
        return {"ok": True}

    registry: dict[str, Callable[[], dict[str, Any]]] = {
        "single_python_stream": case_single_python_stream,
        "single_numpy_stream": case_single_numpy_stream,
        "single_openmp_stream": case_single_openmp_stream,
    }

    results: dict[str, dict[str, Any]] = {}
    for case_name in requested_cases:
        if case_name not in registry:
            raise ValueError(f"Unknown case: {case_name}")
        results[case_name] = run_benchmark(
            registry[case_name],
            warmup=args.warmup,
            repeat=args.repeat,
        )

    report = {
        "suite": "fgp_accumulate",
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
            "openmp_threads": openmp_threads,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "cases": requested_cases,
        },
        "input_source": input_source,
        "results": results,
    }
    print_or_save_report(report, args.output_json)


if __name__ == "__main__":
    main()
