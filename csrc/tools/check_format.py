"""Check or apply the pinned C++/CUDA formatting policy."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


CLANG_FORMAT_VERSION = "20.1.8"
CSRC_ROOT = Path(__file__).resolve().parent.parent
SOURCE_SUFFIXES = frozenset(
    {".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".metal", ".mm"}
)


def _source_files() -> list[Path]:
    files = []
    for path in CSRC_ROOT.rglob("*"):
        relative = path.relative_to(CSRC_ROOT)
        if "build" in relative.parts or path.suffix not in SOURCE_SUFFIXES:
            continue
        files.append(path)
    return sorted(files)


def _clang_format_executable() -> str:
    executable = os.environ.get("CLANG_FORMAT") or shutil.which("clang-format")
    if executable is None:
        raise RuntimeError(
            "clang-format is unavailable; install requirements-dev.txt in the "
            "active environment"
        )
    version = subprocess.check_output(
        [executable, "--version"],
        text=True,
    ).strip()
    if CLANG_FORMAT_VERSION not in version:
        raise RuntimeError(
            f"clang-format {CLANG_FORMAT_VERSION} is required, found: {version}"
        )
    return executable


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fix",
        action="store_true",
        help="apply formatting instead of checking it",
    )
    args = parser.parse_args()

    executable = _clang_format_executable()
    files = _source_files()
    if args.fix:
        result = subprocess.run(
            [executable, "-i", *(str(path) for path in files)],
            check=False,
        )
        if result.returncode == 0:
            print(
                f"clang-format {CLANG_FORMAT_VERSION}: formatted {len(files)} files"
            )
        return result.returncode

    failures = []
    unexpected_error = ""
    for path in files:
        result = subprocess.run(
            [executable, "--dry-run", "--Werror", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            continue
        failures.append(path.relative_to(CSRC_ROOT.parent))
        if "code should be clang-formatted" not in result.stderr:
            unexpected_error = result.stderr
            break

    if unexpected_error:
        print(unexpected_error, end="")
        return 1
    if failures:
        print(
            f"clang-format {CLANG_FORMAT_VERSION}: {len(failures)} files need formatting"
        )
        for path in failures:
            print(f"  {path}")
        print("Run: python csrc/tools/check_format.py --fix")
        return 1

    print(f"clang-format {CLANG_FORMAT_VERSION}: checked {len(files)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
