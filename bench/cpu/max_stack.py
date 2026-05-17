"""大尺寸 max 叠加 benchmark。

用途：
- 固化“大图 + 多帧”的 `max` 算子计算基线
- 比较单进程 NumPy、多进程 NumPy local-reduce 与 custom op/OpenMP 路径
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import platform
from multiprocessing import shared_memory
from typing import Any, Callable

import numpy as np

from bench.common import (
    collect_env_info,
    prepare_batch,
    print_or_save_report,
    resolve_openmp_threads,
    run_benchmark,
    set_omp_threads,
)
from hoshicore._custom_op import build_info as custom_ops_build_info
from hoshicore._custom_op import max_combine as openmp_max_combine
import hoshicore._custom_op.ops.max as max_ops


_GLOBAL_BATCH: np.ndarray | None = None
_SHM_NAME: str | None = None
_SHM_SHAPE: tuple[int, ...] | None = None
_SHM_DTYPE: str | None = None


def sequential_numpy_stream(batch: np.ndarray) -> np.ndarray:
    result = np.array(batch[0], copy=True)
    for idx in range(1, batch.shape[0]):
        np.maximum(result, batch[idx], out=result)
    return result


def sequential_openmp_stream(batch: np.ndarray) -> np.ndarray:
    result = np.array(batch[0], copy=True)
    for idx in range(1, batch.shape[0]):
        openmp_max_combine(result, batch[idx])
    return result


def sequential_numpy_reduce(batch: np.ndarray) -> np.ndarray:
    return np.maximum.reduce(batch, axis=0)


def partition_frames(frame_count: int, workers: int) -> list[tuple[int, int]]:
    workers = max(1, min(workers, frame_count))
    base = frame_count // workers
    rem = frame_count % workers
    tasks: list[tuple[int, int]] = []
    start = 0
    for idx in range(workers):
        size = base + (1 if idx < rem else 0)
        end = start + size
        tasks.append((start, end))
        start = end
    return tasks


def init_fork_worker(batch: np.ndarray) -> None:
    global _GLOBAL_BATCH
    _GLOBAL_BATCH = batch


def init_shm_worker(
    shm_name: str,
    shape: tuple[int, ...],
    dtype_str: str,
) -> None:
    global _SHM_NAME, _SHM_SHAPE, _SHM_DTYPE
    _SHM_NAME = shm_name
    _SHM_SHAPE = shape
    _SHM_DTYPE = dtype_str


def load_batch_from_state() -> tuple[shared_memory.SharedMemory | None, np.ndarray]:
    if _GLOBAL_BATCH is not None:
        return None, _GLOBAL_BATCH
    if _SHM_NAME is None or _SHM_SHAPE is None or _SHM_DTYPE is None:
        raise RuntimeError("Worker batch state is not initialized.")
    shm = shared_memory.SharedMemory(name=_SHM_NAME)
    batch = np.ndarray(_SHM_SHAPE, dtype=np.dtype(_SHM_DTYPE), buffer=shm.buf)
    return shm, batch


def worker_numpy_reduce(task: tuple[int, int]) -> np.ndarray:
    shm, batch = load_batch_from_state()
    start, end = task
    try:
        return sequential_numpy_reduce(batch[start:end])
    finally:
        if shm is not None:
            shm.close()


def reduce_partials_numpy(partials: list[np.ndarray]) -> np.ndarray:
    return np.maximum.reduce(np.stack(partials, axis=0), axis=0)


def run_numpy_local_reduce(
    batch: np.ndarray,
    *,
    workers: int,
    start_method: str,
) -> np.ndarray:
    ctx = mp.get_context(start_method)
    tasks = partition_frames(batch.shape[0], workers)

    if start_method == "fork":
        with ctx.Pool(
            processes=len(tasks),
            initializer=init_fork_worker,
            initargs=(batch,),
        ) as pool:
            partials = pool.map(worker_numpy_reduce, tasks)
        return reduce_partials_numpy(partials)

    shm = shared_memory.SharedMemory(create=True, size=batch.nbytes)
    shared_batch = np.ndarray(batch.shape, dtype=batch.dtype, buffer=shm.buf)
    shared_batch[...] = batch
    try:
        with ctx.Pool(
            processes=len(tasks),
            initializer=init_shm_worker,
            initargs=(shm.name, batch.shape, batch.dtype.str),
        ) as pool:
            partials = pool.map(worker_numpy_reduce, tasks)
        return reduce_partials_numpy(partials)
    finally:
        shm.close()
        shm.unlink()


def default_start_method() -> str:
    if platform.system() == "Linux":
        return "fork"
    return "spawn"


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
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--start-method", type=str, default="auto", choices=["auto", "fork", "spawn", "forkserver"])
    parser.add_argument("--openmp-threads", type=str, default="auto")
    parser.add_argument(
        "--cases",
        type=str,
        default="single_numpy_stream,mp_numpy_local_reduce,single_openmp_stream",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    requested_cases = parse_cases(args.cases)
    start_method = default_start_method() if args.start_method == "auto" else args.start_method
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
    reference = sequential_numpy_reduce(batch)

    def case_single_numpy_stream() -> dict[str, Any]:
        result = sequential_numpy_stream(batch)
        if not np.array_equal(result, reference):
            raise AssertionError("single_numpy_stream mismatch")
        return {"ok": True}

    def case_mp_numpy_local_reduce() -> dict[str, Any]:
        result = run_numpy_local_reduce(
            batch,
            workers=args.workers,
            start_method=start_method,
        )
        if not np.array_equal(result, reference):
            raise AssertionError("mp_numpy_local_reduce mismatch")
        return {"ok": True}

    def case_single_openmp_stream() -> dict[str, Any]:
        set_omp_threads(openmp_threads)
        result = sequential_openmp_stream(batch)
        if not np.array_equal(result, reference):
            raise AssertionError("single_openmp_stream mismatch")
        return {"ok": True}

    registry: dict[str, Callable[[], dict[str, Any]]] = {
        "single_numpy_stream": case_single_numpy_stream,
        "mp_numpy_local_reduce": case_mp_numpy_local_reduce,
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
        "suite": "max_stack",
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
            "workers": args.workers,
            "start_method": start_method,
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
