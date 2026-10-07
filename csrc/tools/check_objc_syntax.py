"""Parse the Metal Objective-C++ sources on any platform.

macOS CI is otherwise the first compiler these files ever see, so a typo costs a
full CI round trip. clang can parse Objective-C++ anywhere once it is pointed at
a runtime and given declarations for the Apple frameworks; ``objc_stubs/`` holds
just enough of Foundation and Metal for that. This is a syntax and name-lookup
gate only -- it proves nothing about Metal behaviour.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

CSRC = Path(__file__).resolve().parent.parent
OBJC_STUBS = Path(__file__).resolve().parent / "objc_stubs"
# Any real Apple runtime works; this only selects ABI details clang needs to
# accept ARC and @autoreleasepool off-platform.
OBJC_RUNTIME = "macosx-10.15"


def _sources() -> list[Path]:
    """Discovered, never listed: a new .mm must not be able to skip this gate."""
    return sorted(
        path
        for path in CSRC.rglob("*.mm")
        if "build" not in path.relative_to(CSRC).parts
    )


def _include_flags() -> list[str]:
    flags = [f"-I{OBJC_STUBS}", f"-I{CSRC}"]
    flags.append("-I" + sysconfig.get_paths()["include"])
    try:
        import pybind11
    except ImportError:
        print("pybind11 is required for the Objective-C++ syntax check", file=sys.stderr)
        raise SystemExit(2)
    flags.append("-I" + pybind11.get_include())
    return flags


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clang", default="clang++", help="Objective-C++ capable clang++")
    args = parser.parse_args()

    clang = shutil.which(args.clang)
    if clang is None:
        print(f"{args.clang} not found; skipping the Objective-C++ syntax check")
        return 0

    base = [
        clang,
        "-fsyntax-only",
        "-x",
        "objective-c++",
        "-std=c++17",
        "-fobjc-arc",
        f"-fobjc-runtime={OBJC_RUNTIME}",
        *_include_flags(),
    ]
    sources = _sources()
    if not sources:
        print("no Objective-C++ sources found under csrc/", file=sys.stderr)
        return 1

    failed = []
    for source in sources:
        relative = source.relative_to(CSRC)
        result = subprocess.run(
            [*base, str(source)], capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            failed.append(relative)
            print(f"--- {relative} ---", file=sys.stderr)
            print(result.stderr.strip(), file=sys.stderr)

    if failed:
        print(f"\nObjective-C++ syntax check failed for {len(failed)} file(s)", file=sys.stderr)
        return 1
    print(f"clang syntax check: {len(sources)} Objective-C++ files parsed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
