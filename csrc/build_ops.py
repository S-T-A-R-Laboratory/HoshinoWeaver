from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
SYSTEM = platform.system()
PACKAGE_DIR = PROJECT_ROOT / "hoshicore" / "_custom_op"
PRESETS_FILE = ROOT / "CMakePresets.json"


def _which(name: str) -> str | None:
    return shutil.which(name)


def _active_env_bin_dirs() -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()

    exe_dir = Path(sys.executable).resolve().parent
    if exe_dir.is_dir() and exe_dir not in seen:
        candidates.append(exe_dir)
        seen.add(exe_dir)

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        conda_bin = Path(conda_prefix) / "bin"
        if conda_bin.is_dir() and conda_bin not in seen:
            candidates.append(conda_bin)
            seen.add(conda_bin)

    return candidates


def _find_tool_in_dirs(directories: list[Path], names: list[str]) -> str | None:
    for directory in directories:
        for name in names:
            path = directory / name
            if path.is_file() and os.access(path, os.X_OK):
                return str(path)
    return None


def _run_capture(cmd: list[str]) -> str | None:
    try:
        proc = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception:
        return None
    return proc.stdout.strip() or None


def _ensure_env_tool(name: str, env_value: str | None) -> str | None:
    if not env_value:
        return None
    if os.path.isabs(env_value):
        return env_value if Path(env_value).exists() else None
    return _which(env_value)


def _resolve_linux_compiler(choice: str) -> tuple[str, str, list[str]]:
    hints: list[str] = []
    env_cc = _ensure_env_tool("CC", os.environ.get("CC"))
    env_cxx = _ensure_env_tool("CXX", os.environ.get("CXX"))
    if env_cc and env_cxx:
        return env_cc, env_cxx, hints

    env_bin_dirs = _active_env_bin_dirs()

    if choice in {"auto", "gcc"}:
        env_cc = _find_tool_in_dirs(env_bin_dirs, ["x86_64-conda-linux-gnu-gcc", "gcc"])
        env_cxx = _find_tool_in_dirs(env_bin_dirs, ["x86_64-conda-linux-gnu-g++", "g++"])
        if env_cc and env_cxx:
            return env_cc, env_cxx, hints
        cc = _which("gcc")
        cxx = _which("g++")
        if cc and cxx:
            return cc, cxx, hints
        if choice == "gcc":
            hints.append("Install gcc/g++ or export CC/CXX to a valid GCC toolchain.")

    if choice in {"auto", "clang"}:
        env_cc = _find_tool_in_dirs(env_bin_dirs, ["clang"])
        env_cxx = _find_tool_in_dirs(env_bin_dirs, ["clang++"])
        if env_cc and env_cxx:
            return env_cc, env_cxx, hints
        cc = _which("clang")
        cxx = _which("clang++")
        if cc and cxx:
            return cc, cxx, hints
        if choice == "clang":
            hints.append("Install clang/clang++ or export CC/CXX to a valid Clang toolchain.")

    hints.append("Linux build expects gcc/g++ or clang/clang++ on PATH, or explicit CC/CXX.")
    raise RuntimeError("\n".join(hints))


def _resolve_macos_compiler(choice: str) -> tuple[str, str, list[str], dict[str, str]]:
    hints: list[str] = []
    extra_env: dict[str, str] = {}

    env_cc = _ensure_env_tool("CC", os.environ.get("CC"))
    env_cxx = _ensure_env_tool("CXX", os.environ.get("CXX"))
    if env_cc and env_cxx:
        cc = env_cc
        cxx = env_cxx
    else:
        if choice not in {"auto", "clang"}:
            raise RuntimeError("macOS build currently supports Clang toolchains only.")
        cc = _run_capture(["xcrun", "--find", "clang"]) or _which("clang")
        cxx = _run_capture(["xcrun", "--find", "clang++"]) or _which("clang++")
        if not cc or not cxx:
            raise RuntimeError(
                "Failed to locate clang/clang++.\n"
                "Install Xcode Command Line Tools with `xcode-select --install`."
            )

    enable_openmp = os.environ.get("HNW_CSRC_OPENMP", "1") != "0"
    if enable_openmp and "HNW_LIBOMP_PREFIX" not in os.environ:
        libomp_prefix = _run_capture(["brew", "--prefix", "libomp"])
        if libomp_prefix:
            extra_env["HNW_LIBOMP_PREFIX"] = libomp_prefix
        else:
            raise RuntimeError(
                "OpenMP is enabled but libomp was not found.\n"
                "Install it with `brew install libomp`, or rerun with --no-openmp."
            )

    return cc, cxx, hints, extra_env


def _resolve_windows_compiler(choice: str) -> tuple[str | None, str | None, list[str]]:
    hints: list[str] = []
    env_cc = _ensure_env_tool("CC", os.environ.get("CC"))
    env_cxx = _ensure_env_tool("CXX", os.environ.get("CXX"))
    if env_cc and env_cxx:
        return env_cc, env_cxx, hints

    if choice in {"auto", "msvc"}:
        cl = _which("cl")
        if cl:
            return None, None, hints
        if choice == "msvc":
            hints.append(
                "Open 'x64 Native Tools Command Prompt for VS' or 'Developer PowerShell for VS' "
                "so that `cl` and `link` are on PATH."
            )

    if choice in {"auto", "gcc"}:
        cc = _which("gcc")
        cxx = _which("g++")
        if cc and cxx:
            return cc, cxx, hints
        if choice == "gcc":
            hints.append(
                "Install MinGW-w64 (ucrt variant recommended) and make gcc/g++ available on PATH."
            )

    if choice in {"auto", "clang"}:
        clang_cl = _which("clang-cl")
        if clang_cl:
            return clang_cl, clang_cl, hints
        if choice == "clang":
            hints.append("Install LLVM for Windows and make clang-cl available on PATH.")

    hints.append(
        "Windows build expects MSVC Build Tools in an activated Developer shell, "
        "MinGW-w64 gcc/g++ on PATH, or explicit CC/CXX."
    )
    raise RuntimeError("\n".join(hints))


def resolve_toolchain(choice: str) -> tuple[str | None, str | None, dict[str, str]]:
    if SYSTEM == "Linux":
        cc, cxx, _ = _resolve_linux_compiler(choice)
        return cc, cxx, {}
    if SYSTEM == "Darwin":
        cc, cxx, _, extra_env = _resolve_macos_compiler(choice)
        return cc, cxx, extra_env
    if SYSTEM == "Windows":
        cc, cxx, _ = _resolve_windows_compiler(choice)
        return cc, cxx, {}
    raise RuntimeError(f"Unsupported platform: {SYSTEM}")


def _render_cmd(cmd: list[str]) -> str:
    return " ".join(cmd)


def _shared_module_patterns() -> tuple[str, ...]:
    return ("_C*.so", "_C*.pyd", "_C*.dylib")


def _collect_mingw_deps(start: Path, bin_dir: Path, objdump: Path) -> list[str]:
    """Recursively collect all MinGW DLL dependencies of a PE binary."""
    visited: set[str] = set()
    queue: list[Path] = [start]
    while queue:
        target = queue.pop()
        result = subprocess.run(
            [str(objdump), "-p", str(target)],
            capture_output=True,
            text=True,
        )
        for line in result.stdout.splitlines():
            if "DLL Name:" not in line:
                continue
            dll_name = line.split("DLL Name:")[-1].strip()
            if dll_name in visited:
                continue
            src = bin_dir / dll_name
            if src.exists():
                visited.add(dll_name)
                queue.append(src)
    return sorted(visited)


def _copy_mingw_runtime_dlls(cc: str) -> None:
    """Copy MinGW runtime DLLs next to the .pyd so it imports without PATH changes.

    Only runs on Windows with a GCC compiler. Recursively resolves the full
    dependency tree via objdump so transitive deps (e.g. libdl.dll pulled in
    by libgomp-1.dll) are not missed.
    """
    if SYSTEM != "Windows":
        return
    if not cc or "gcc" not in Path(cc).name.lower():
        return

    bin_dir = Path(cc).parent
    objdump = bin_dir / "objdump.exe"
    pyd = next(PACKAGE_DIR.glob("_C*.pyd"), None)
    if not pyd or not objdump.exists():
        return

    deps = _collect_mingw_deps(pyd, bin_dir, objdump)
    for dll_name in deps:
        shutil.copy2(bin_dir / dll_name, PACKAGE_DIR / dll_name)
    if deps:
        print(f"mingw_dlls_copied={','.join(deps)}")


def _clean_shared_module_outputs() -> None:
    for pattern in _shared_module_patterns():
        for path in PACKAGE_DIR.glob(pattern):
            path.unlink()


def _cmake_binary_dir(preset: str) -> Path:
    return ROOT / "build" / preset


def _clean_cmake_outputs(preset: str) -> None:
    binary_dir = _cmake_binary_dir(preset)
    if binary_dir.exists():
        shutil.rmtree(binary_dir)
    _clean_shared_module_outputs()


def _ensure_required_tools() -> None:
    if _which("cmake") is None:
        raise RuntimeError("CMake backend requires `cmake` on PATH.")
    if _which("ninja") is None:
        raise RuntimeError("CMake backend requires `ninja` on PATH.")
    if not PRESETS_FILE.exists():
        raise RuntimeError(f"Missing CMake presets file: {PRESETS_FILE}")


def _default_cmake_preset(choice: str, *, enable_cuda: bool, resolved_cc: str | None = None) -> str:
    if SYSTEM == "Linux":
        if enable_cuda:
            if choice not in {"auto", "gcc"}:
                raise RuntimeError("Linux CUDA build currently expects GCC host compilers.")
            return "linux-gcc-cuda"
        if choice in {"auto", "gcc"}:
            return "linux-gcc"
        if choice == "clang":
            return "linux-clang"
    elif SYSTEM == "Darwin":
        if choice in {"auto", "clang"}:
            return "macos-clang"
    elif SYSTEM == "Windows":
        if enable_cuda:
            if choice == "gcc":
                raise RuntimeError("Windows CUDA build requires MSVC. MinGW is not supported as a CUDA host compiler.")
            if choice not in {"auto", "msvc"}:
                raise RuntimeError("Windows CUDA build currently expects MSVC host compilers.")
            return "windows-msvc-cuda"
        if choice == "gcc":
            return "windows-mingw"
        if choice == "auto" and resolved_cc and "gcc" in Path(resolved_cc).name.lower():
            return "windows-mingw"
        if choice in {"auto", "msvc"}:
            return "windows-msvc"
    raise RuntimeError(
        f"No default CMake preset for platform={SYSTEM} compiler={choice}. "
        "Pass --preset explicitly."
    )


def _setup_env_from_args(args: argparse.Namespace) -> tuple[dict[str, str], str | None, str | None]:
    explicit_cc = _ensure_env_tool("CC", args.cc)
    explicit_cxx = _ensure_env_tool("CXX", args.cxx)
    if (args.cc is None) != (args.cxx is None):
        raise RuntimeError("Both --cc and --cxx must be provided together.")
    if args.cc is not None and explicit_cc is None:
        raise RuntimeError(f"Failed to resolve --cc: {args.cc}")
    if args.cxx is not None and explicit_cxx is None:
        raise RuntimeError(f"Failed to resolve --cxx: {args.cxx}")

    if explicit_cc and explicit_cxx:
        cc, cxx, extra_env = explicit_cc, explicit_cxx, {}
    else:
        cc, cxx, extra_env = resolve_toolchain(args.compiler)

    env = os.environ.copy()
    if cc:
        env["CC"] = cc
    if cxx:
        env["CXX"] = cxx
    env.update(extra_env)
    if args.no_openmp:
        env["HNW_CSRC_OPENMP"] = "0"
    else:
        env.setdefault("HNW_CSRC_OPENMP", "1")
    if args.march_native:
        env["HNW_CSRC_ARCH"] = "1"
    env.setdefault("HNW_CSRC_ARCH", "0")
    env["HNW_CSRC_OMP_SIMD"] = "1" if args.omp_simd else env.get("HNW_CSRC_OMP_SIMD", "0")
    env.setdefault("HNW_CSRC_OMP_SIMD", "0")
    if args.extra_cflags:
        env["HNW_CSRC_EXTRA_CFLAGS"] = args.extra_cflags
    if args.extra_ldflags:
        env["HNW_CSRC_EXTRA_LDFLAGS"] = args.extra_ldflags
    if args.cuda:
        env["CCACHE_DISABLE"] = "1"
        if cxx:
            env["CUDAHOSTCXX"] = cxx
    return env, cc, cxx


def _cmake_build_commands(
    env: dict[str, str],
    args: argparse.Namespace,
    cc: str | None,
    cxx: str | None,
) -> tuple[list[str], list[str], str]:
    preset = args.preset or _default_cmake_preset(args.compiler, enable_cuda=args.cuda, resolved_cc=cc)
    configure_cmd = ["cmake", "--preset", preset]
    build_cmd = ["cmake", "--build", "--preset", preset]
    if args.verbose_build:
        build_cmd.append("--verbose")

    cache_vars = {
        "Python3_EXECUTABLE": sys.executable,
        "HNW_ENABLE_OPENMP": "OFF" if env.get("HNW_CSRC_OPENMP", "1") == "0" else "ON",
        "HNW_ENABLE_OMP_SIMD": "ON" if env.get("HNW_CSRC_OMP_SIMD", "0") == "1" else "OFF",
        "HNW_ENABLE_MARCH_NATIVE": "ON" if env.get("HNW_CSRC_ARCH", "0") == "1" else "OFF",
        "HNW_ENABLE_LTO": "ON" if args.lto else "OFF",
        "HNW_ENABLE_CUDA": "ON" if args.cuda else "OFF",
    }
    if env.get("HNW_CSRC_EXTRA_CFLAGS"):
        cache_vars["HNW_EXTRA_CXX_FLAGS"] = env["HNW_CSRC_EXTRA_CFLAGS"]
    if env.get("HNW_CSRC_EXTRA_LDFLAGS"):
        cache_vars["HNW_EXTRA_LINK_FLAGS"] = env["HNW_CSRC_EXTRA_LDFLAGS"]
    if cc:
        cache_vars["CMAKE_C_COMPILER"] = cc
    if cxx:
        cache_vars["CMAKE_CXX_COMPILER"] = cxx
        if args.cuda:
            cache_vars["CMAKE_CUDA_HOST_COMPILER"] = cxx
    if env.get("HNW_LIBOMP_PREFIX"):
        cache_vars["CMAKE_PREFIX_PATH"] = env["HNW_LIBOMP_PREFIX"]

    for key, value in cache_vars.items():
        # CMake parses -D values as cmake string literals on Windows; backslashes
        # are treated as escape sequences there, so \S in a path triggers an error.
        # Forward slashes are always accepted by CMake on all platforms.
        if SYSTEM == "Windows":
            value = value.replace("\\", "/")
        configure_cmd.append(f"-D{key}={value}")

    return configure_cmd, build_cmd, preset


def _print_filtered_output(output: str, backend: str, stage: str) -> None:
    if not output:
        return

    lines: list[str] = []
    for line in output.splitlines():
        if stage == "configure":
            if line.startswith("-- Configuring done") or line.startswith("-- Generating done") or "Build files have been written to:" in line:
                lines.append(line)
        else:
            if line.startswith("[") or "Linking CXX shared module" in line or line.startswith("FAILED:") or line.startswith("ninja:"):
                lines.append(line)
    if not lines:
        lines = [line for line in output.splitlines() if line.strip()]
    for line in lines:
        print(line)


def _run_command(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    verbose: bool,
    backend: str,
    stage: str,
) -> None:
    if verbose:
        subprocess.run(cmd, cwd=cwd, env=env, check=True)
        return

    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    _print_filtered_output(proc.stdout or "", backend=backend, stage=stage)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build HoshinoWeaver custom ops with the CMake/Ninja toolchain."
    )
    parser.add_argument(
        "--preset",
        default=None,
        help="CMake preset name. Overrides the platform/compiler default preset.",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Enable CUDA custom-op targets for supported presets/toolchains.",
    )
    parser.add_argument(
        "--compiler",
        choices=["auto", "gcc", "clang", "msvc"],
        default="auto",
        help="Preferred compiler family. Platform-incompatible choices fail fast.",
    )
    parser.add_argument(
        "--cc",
        default=None,
        help="Explicit C compiler path or name. Overrides --compiler auto-detection.",
    )
    parser.add_argument(
        "--cxx",
        default=None,
        help="Explicit C++ compiler path or name. Overrides --compiler auto-detection.",
    )
    parser.add_argument(
        "--no-openmp",
        action="store_true",
        help="Disable OpenMP for this build.",
    )
    parser.add_argument(
        "--march-native",
        action="store_true",
        help="Enable -march=native on non-Windows builds.",
    )
    parser.add_argument(
        "--lto",
        action="store_true",
        help="Enable link-time optimization for the selected backend.",
    )
    parser.add_argument(
        "--omp-simd",
        action="store_true",
        help="Enable explicit OpenMP SIMD pragmas in kernels that support them.",
    )
    parser.add_argument(
        "--extra-cflags",
        default=None,
        help="Append extra compiler flags.",
    )
    parser.add_argument(
        "--extra-ldflags",
        default=None,
        help="Append extra linker flags.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved build configuration without invoking the backend.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove previous build artifacts before invoking the backend.",
    )
    parser.add_argument(
        "--verbose-build",
        action="store_true",
        help="Show the full backend output instead of the default concise summary.",
    )
    args = parser.parse_args()

    try:
        _ensure_required_tools()
        env, cc, cxx = _setup_env_from_args(args)
        configure_cmd, build_cmd, resolved_preset = _cmake_build_commands(env, args, cc, cxx)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print("backend=cmake")
    print(f"platform={SYSTEM}")
    print(f"python={sys.executable}")
    print(f"cwd={ROOT}")
    print(f"compiler_preference={args.compiler}")
    if args.cc or args.cxx:
        print(f"explicit_cc={args.cc}")
        print(f"explicit_cxx={args.cxx}")
    print(f"resolved_cc={env.get('CC', '<default toolchain>')}")
    print(f"resolved_cxx={env.get('CXX', '<default toolchain>')}")
    print(f"openmp={env.get('HNW_CSRC_OPENMP', '1')}")
    print(f"march_native={env.get('HNW_CSRC_ARCH', '0')}")
    print(f"lto={args.lto}")
    print(f"omp_simd={env.get('HNW_CSRC_OMP_SIMD', '0')}")
    print(f"cuda={int(args.cuda)}")
    if "HNW_LIBOMP_PREFIX" in env:
        print(f"libomp_prefix={env['HNW_LIBOMP_PREFIX']}")
    if "CUDAHOSTCXX" in env:
        print(f"cudahostcxx={env['CUDAHOSTCXX']}")
    if args.extra_cflags:
        print(f"extra_cflags={args.extra_cflags}")
    if args.extra_ldflags:
        print(f"extra_ldflags={args.extra_ldflags}")
    print(f"clean={int(args.clean)}")
    print(f"preset={resolved_preset}")
    print("configure_command=" + _render_cmd(configure_cmd))
    print("build_command=" + _render_cmd(build_cmd))

    if args.dry_run:
        return 0

    try:
        if args.clean:
            _clean_cmake_outputs(resolved_preset)
        _run_command(
            configure_cmd,
            cwd=ROOT,
            env=env,
            verbose=args.verbose_build,
            backend="cmake",
            stage="configure",
        )
        _run_command(
            build_cmd,
            cwd=ROOT,
            env=env,
            verbose=args.verbose_build,
            backend="cmake",
            stage="build",
        )
        _copy_mingw_runtime_dlls(cc)
    except subprocess.CalledProcessError as exc:
        if not args.verbose_build:
            captured = getattr(exc, "stdout", None)
            if captured:
                print(captured, file=sys.stderr, end="" if captured.endswith("\n") else "\n")
        print(f"Build failed with exit code {exc.returncode}.", file=sys.stderr)
        return exc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
