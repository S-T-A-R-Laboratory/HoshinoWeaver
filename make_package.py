"""Unified HoshinoWeaver packaging entrypoint."""
from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path

from hoshicore.packaging import common


def main(argv: list[str] | None = None, *, default_backend: str = "pyinstaller") -> int:
    parser = argparse.ArgumentParser(description="Package HoshinoWeaver using PyInstaller or Nuitka.")
    parser.add_argument("--backend", choices=("pyinstaller", "nuitka"), default=default_backend,
                        help=f"Packaging backend (default: {default_backend}).")
    parser.add_argument("--apply-upx", action="store_true", help="Compress eligible binaries with UPX.")
    parser.add_argument("--apply-zip", action="store_true", help="Generate the existing release ZIP filename.")
    parser.add_argument("--debug-gui", action="store_true", help="Build GUI with a console window.")
    parser.add_argument("--no-build", action="store_true", help="Skip automatic C++ custom-op compilation.")
    parser.add_argument("--allow-numpy-only", action="store_true", help="Allow packaging without an importable _C.")
    parser.add_argument("--jobs", type=int, default=2, help="Nuitka C compiler concurrency (default: 2).")
    parser.add_argument("--build-dir", type=Path, help="Backend working directory; prefer an ASCII Windows path.")
    parser.add_argument("--dry-run", action="store_true", help="Generate commands/configuration without compiling.")
    parser.add_argument("--overwrite", action="store_true", help="Replace this version's release after successful verification.")
    parser.add_argument("--verify", action="store_true", help="Verify CLI/GUI before publishing.")
    parser.add_argument("--verify-only", action="store_true", help="Verify this version's existing release without building.")
    args = parser.parse_args(argv)
    if args.jobs <= 0:
        parser.error("--jobs must be positive")
    backend = importlib.import_module(f"hoshicore.packaging.{args.backend}")
    options = common.BuildOptions(root=common.ROOT, **vars(args))
    return common.run_packaging(options, backend)


def cli(argv: list[str] | None = None, *, default_backend: str = "pyinstaller") -> int:
    try:
        return main(argv, default_backend=default_backend)
    except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
        print(f"Packaging failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(cli())
