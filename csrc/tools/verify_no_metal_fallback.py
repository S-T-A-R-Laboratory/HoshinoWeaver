"""Verify a --no-metal macOS build drops Metal and falls back to OpenMP.

Covers both a rebuild over a previous Metal build (stale extension or shader
sidecar would still be selected) and a plain CPU-only macOS build.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PACKAGE_DIR = PROJECT_ROOT / "hoshicore" / "_custom_op"
STALE_PATTERNS = ("_metal*.so", "_metal*.dylib", "_metal_kernels.metallib")
PARAMS = (3, "CIRCLE", 1, 1.0, 5)


def main() -> None:
    stale = sorted(
        str(path.relative_to(PROJECT_ROOT))
        for pattern in STALE_PATTERNS
        for path in PACKAGE_DIR.glob(pattern)
    )
    if stale:
        raise SystemExit(f"--no-metal build left stale Metal artifacts: {stale}")

    from hoshicore._custom_op import backend_registry
    from hoshicore._custom_op import build_info
    from hoshicore._custom_op._dispatch import load_metal_module
    from hoshicore._custom_op.ops import star_shrink as star_shrink_ops

    info = build_info()
    if info.get("openmp") is not True:
        raise SystemExit(f"expected an OpenMP-enabled build, got openmp={info.get('openmp')!r}")
    if info.get("cuda") is not False:
        raise SystemExit(f"expected a CUDA-free build, got cuda={info.get('cuda')!r}")

    module, _ = load_metal_module()
    if module is not None:
        raise SystemExit("Metal extension is still importable after a --no-metal build")

    selection = backend_registry.select_backend("star_shrink_process")
    if selection.backend != "openmp_cpu":
        raise SystemExit(
            f"expected the OpenMP fallback to be selected, got {selection.backend}"
        )

    image = np.arange(23 * 29 * 3, dtype=np.uint16).reshape(23, 29, 3) * 37
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[3:20, 4:26] = 1
    # Auto dispatch must be bit-identical to the direct OpenMP entry point,
    # which is only true if selection really landed on openmp_cpu.
    dispatched = star_shrink_ops.star_shrink_process(image, mask, *PARAMS)
    direct = star_shrink_ops.star_shrink_process_compiled(image, mask, *PARAMS)
    np.testing.assert_array_equal(dispatched, direct)
    np.testing.assert_array_equal(dispatched[mask == 0], image[mask == 0])

    print("HNW_NO_METAL_FALLBACK_OK")


if __name__ == "__main__":
    sys.exit(main())
