"""Shared project requirements, verification and release lifecycle."""
from __future__ import annotations

import ctypes.util
import filecmp
import fnmatch
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from hoshicore.component.utils import SOFTWARE_NAME, VERSION

ROOT = Path(__file__).resolve().parents[2]
CLI_NAME = "launcher"
GUI_NAME = "HoshinoWeaver desktop"
BASE_EXCLUDES = ("tensorflow", "keras", "torch", "PyQt5", "PyQt6", "hoshicore.packaging")
UPX_EXCLUDES = ("hoshicore/_custom_op/_C*.pyd", "hoshicore/_custom_op/_C*.so",
                "hoshicore/_custom_op/_metal*.pyd", "hoshicore/_custom_op/_metal*.so",
                "vcruntime*.dll", "VCRUNTIME*.dll")


@dataclass(frozen=True)
class BuildOptions:
    root: Path = ROOT
    backend: str = "pyinstaller"
    apply_upx: bool = False
    apply_zip: bool = False
    debug_gui: bool = False
    no_build: bool = False
    allow_numpy_only: bool = False
    jobs: int = 2
    build_dir: Path | None = None
    dry_run: bool = False
    overwrite: bool = False
    verify: bool = False
    verify_only: bool = False

    @property
    def release_name(self) -> str:
        return f"{GUI_NAME}_{platform_name()}_{VERSION}{'-debug' if self.debug_gui else ''}"

    @property
    def release_dir(self) -> Path:
        return self.root / "dist" / self.release_name


@dataclass(frozen=True)
class PackageManifest:
    """Project requirements, independent of either packager's configuration."""
    native: bool
    data_dirs: tuple[tuple[Path, str], ...]
    data_files: tuple[tuple[Path, str], ...]
    custom_op_dir: Path
    custom_op_dlls: tuple[Path, ...] = ()
    metal_shader: Path | None = None
    turbojpeg: bool = False
    turbojpeg_libraries: tuple[Path, ...] = ()


@dataclass(frozen=True)
class BuildResult:
    directory: Path
    cli: Path
    gui: Path
    resource_dir: Path
    gui_log_dir: Path


def platform_name() -> str:
    name = {"win32": "win", "cygwin": "win", "linux": "linux", "darwin": "macos"}[sys.platform]
    if name == "macos" and int(platform.mac_ver()[0].split(".")[0]) >= 13:
        name += "13+"
    return name


def compiler_environment() -> dict[str, str]:
    """Keep unrelated credentials out of generated compiler diagnostics."""
    private = ("TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "AUTH", "API_KEY", "ACCESS_KEY")
    return {key: value for key, value in os.environ.items()
            if not any(part in key.upper() for part in private)}


def run_command(command: list[str], *, root: Path, dry_run: bool = False) -> None:
    print(subprocess.list2cmdline(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=root, env=compiler_environment(), check=True)


def validate_build_directory(directory: Path, options: BuildOptions) -> Path:
    """Compiler intermediates must never overwrite an existing release."""
    directory = directory.resolve()
    release_root = (options.root / "dist").resolve()
    if directory == options.root.resolve() or directory == release_root or release_root in directory.parents:
        raise RuntimeError(f"Build directory must be outside the release tree and distinct from the checkout root: {directory}")
    return directory


def ensure_custom_ops(root: Path = ROOT, *, skip_build: bool, allow_numpy_only: bool) -> bool:
    probe = [sys.executable, "-c", "import hoshicore._custom_op._C"]
    result = subprocess.run(probe, cwd=root, capture_output=True)
    if result.returncode == 0:
        return True
    if not skip_build:
        print("Custom ops missing; running csrc/build_ops.py ...", flush=True)
        built = subprocess.run([sys.executable, str(root / "csrc/build_ops.py")],
                               cwd=root, env=compiler_environment())
        if built.returncode == 0:
            result = subprocess.run(probe, cwd=root, capture_output=True)
            if result.returncode == 0:
                return True
    if allow_numpy_only:
        print("WARNING: _C unavailable; packaging with NumPy/contour fallbacks.")
        return False
    raise RuntimeError("Custom ops (_C) cannot be imported. Build with python csrc/build_ops.py "
                       "or use --allow-numpy-only.\n" + result.stderr.decode(errors="replace"))


def discover_manifest(root: Path, *, native: bool) -> PackageManifest:
    custom_op_dir = root / "hoshicore/_custom_op"
    metal = importlib.util.find_spec("hoshicore._custom_op._metal")
    shader = Path(metal.origin).with_name("_metal_kernels.metallib") if metal and metal.origin else None
    if shader and not shader.is_file():
        raise RuntimeError("Metal extension found without _metal_kernels.metallib; rebuild custom ops.")
    turbo = importlib.util.find_spec("turbojpeg")
    libraries = []
    if turbo and turbo.origin:
        for path in Path(turbo.origin).parent.glob("*turbojpeg*"):
            if path.is_file() and (path.suffix.lower() in {".dll", ".so", ".dylib"} or ".so." in path.name):
                libraries.append(path.resolve())
        library = ctypes.util.find_library("turbojpeg")
        if library:
            resolved = library if Path(library).is_file() else shutil.which(library)
            if not resolved and sys.platform == "linux" and shutil.which("ldconfig"):
                listing = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True)
                for line in listing.stdout.splitlines():
                    candidate = line.partition("=>")[2].strip()
                    if candidate and Path(candidate).name == library and Path(candidate).is_file():
                        resolved = candidate
                        break
            if resolved:
                libraries.append(Path(resolved).resolve())
            else:
                print(f"WARNING: Cannot resolve libturbojpeg source path: {library}")
    files = [(root / "hoshicore/default_settings.yaml", "hoshicore/default_settings.yaml"),
             (root / "LICENSE", "LICENSE")]
    if shader:
        files.append((shader, "hoshicore/_custom_op/_metal_kernels.metallib"))
    dirs = ((root / "hoshicore/dag", "hoshicore/dag"),)
    for source, _ in (*dirs, *files):
        if not source.exists():
            raise RuntimeError(f"Required packaging resource missing: {source}")
    return PackageManifest(native, dirs, tuple(files), custom_op_dir,
                           tuple(sorted(custom_op_dir.glob("*.dll"))) if native else (),
                           shader, turbo is not None, tuple(dict.fromkeys(libraries)))


def merge_distribution(source: Path, destination: Path) -> None:
    """Copy a distribution without overwriting conflicting dependencies."""
    if not source.is_dir():
        raise RuntimeError(f"Build distribution missing: {source}")
    for path in sorted(source.rglob("*")):
        target = destination / path.relative_to(source)
        if path.is_symlink():
            if target.is_symlink() and os.readlink(target) == os.readlink(path):
                continue
            if target.exists() or target.is_symlink():
                raise RuntimeError(f"Conflicting distribution symlink: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(os.readlink(path), target_is_directory=path.is_dir())
        elif path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif target.exists():
            if not target.is_file() or not filecmp.cmp(path, target, shallow=False):
                raise RuntimeError(f"Conflicting distribution file: {target}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def apply_upx(directory: Path) -> None:
    upx = shutil.which("upx")
    if upx is None:
        print("WARNING: UPX not found; leaving binaries uncompressed.")
        return
    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() not in {".exe", ".dll", ".pyd", ".so"}:
            continue
        if any(fnmatch.fnmatchcase(path.name.lower(), pattern.rsplit("/", 1)[-1].lower())
               for pattern in UPX_EXCLUDES):
            continue
        # Match Nuitka's UPX plugin compression settings.
        result = subprocess.run([upx, "-q", "--no-progress", "--best", "--lzma", str(path)],
                                capture_output=True, text=True)
        if result.returncode:
            print(f"WARNING: UPX skipped {path.name}: {result.stderr.strip()}")


def remove_owned_directory(path: Path, parent: Path) -> None:
    if path.is_symlink() or path.resolve().parent != parent.resolve():
        raise RuntimeError(f"Unsafe release directory: {path}")
    shutil.rmtree(path)


def publish_release(stage: Path, final: Path, *, overwrite: bool) -> None:
    """Replace a release only after successful verification, with rollback."""
    parent = final.parent
    if stage.is_symlink() or stage.resolve().parent != parent.resolve():
        raise RuntimeError(f"Unsafe staging directory: {stage}")
    backup = None
    if final.exists() or final.is_symlink():
        if not overwrite:
            raise RuntimeError(f"Release directory exists: {final}; use --overwrite to replace it.")
        if final.is_symlink() or final.resolve().parent != parent.resolve() or not final.is_dir():
            raise RuntimeError(f"Unsafe release replacement path: {final}")
        backup = parent / f".package-backup-{uuid.uuid4().hex}"
        final.rename(backup)
    try:
        stage.rename(final)
    except BaseException:
        if backup is not None:
            backup.rename(final)
        raise
    if backup is not None:
        remove_owned_directory(backup, parent)


def archive_release(directory: Path, archive: Path) -> None:
    # Write beside the destination and replace it only when compression succeeds.
    handle, temporary = tempfile.mkstemp(prefix=".package-zip-", suffix=".zip", dir=archive.parent)
    os.close(handle)
    temporary = Path(temporary)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as output:
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    output.write(path, path.relative_to(directory).as_posix())
        temporary.replace(archive)
    finally:
        temporary.unlink(missing_ok=True)


def run_packaging(options: BuildOptions, backend: ModuleType) -> int:
    """Both backends participate in this same release lifecycle."""
    final = options.release_dir
    if options.verify_only:
        if not final.is_dir():
            raise RuntimeError(f"Release directory does not exist: {final}")
        verify_distribution(backend.describe_release(final, options), native=not options.allow_numpy_only)
        return 0
    if final.exists() and not options.overwrite and not options.dry_run:
        raise RuntimeError(f"Release directory exists: {final}; use --overwrite to replace it.")
    started = time.monotonic()
    native = ensure_custom_ops(options.root, skip_build=options.no_build or options.dry_run,
                               allow_numpy_only=options.allow_numpy_only)
    manifest = discover_manifest(options.root, native=native)
    if options.dry_run:
        backend.build(options, manifest, None)
        print(f"Release: {final}")
        return 0
    final.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".package-", dir=final.parent))
    try:
        result = backend.build(options, manifest, stage)
        backend.postprocess(result, options)
        if options.verify:
            verify_distribution(result, native=native)
        publish_release(stage, final, overwrite=options.overwrite)
    finally:
        if stage.exists():
            remove_owned_directory(stage, final.parent)
    if options.apply_zip:
        archive = final.parent / f"{SOFTWARE_NAME}_{platform_name()}_{VERSION}.zip"
        archive_release(final, archive)
        print(f"ZIP: {archive}")
    size = sum(path.stat().st_size for path in final.rglob("*") if path.is_file())
    print(f"Release: {final}\nSize: {size / 1024**2:.1f} MiB\nTime: {time.monotonic() - started:.1f}s")
    return 0


def verify_distribution(result: BuildResult, *, native: bool) -> None:
    """Exercise the real executables away from the checkout and developer PATH."""
    import cv2
    import numpy as np

    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX",
                "CUDA_HOME", "CUDA_PATH", "INCLUDE", "LIB", "LIBPATH",
                "LD_LIBRARY_PATH", "LD_PRELOAD", "DYLD_LIBRARY_PATH",
                "DYLD_FALLBACK_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES",
                "QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH"):
        env.pop(key, None)
    if os.name == "nt":
        system = Path(env.get("SYSTEMROOT", r"C:\Windows"))
        env["PATH"] = os.pathsep.join(map(str, (system / "System32", system, system / "System32/Wbem")))
    else:
        env["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin"
    env["HNW_CUSTOM_OPS_FALLBACK"] = "cpu"
    env["QT_QPA_PLATFORM"] = "offscreen"
    cli, gui = result.cli, result.gui
    with tempfile.TemporaryDirectory(prefix="hnw-release-check-") as temporary:
        work = Path(temporary)
        # The Windows GUI writes preferences under expanduser("~")/AppData.
        if os.name == "nt":
            env["USERPROFILE"] = str(work)

        def run_cli(arguments: list[str], label: str) -> str:
            result = subprocess.run([str(cli), *arguments], cwd=work, env=env,
                                    capture_output=True, text=True, encoding="utf-8",
                                    errors="replace", timeout=120)
            (work / f"{label}.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
            if result.returncode:
                raise RuntimeError(f"Packaged CLI {label} failed:\n{result.stdout}\n{result.stderr}")
            return result.stdout + result.stderr

        help_text = run_cli(["--help"], "help")
        if "--preflight" not in help_text or "--input" not in help_text:
            raise RuntimeError("Packaged CLI does not expose the expected arguments")
        dag = result.resource_dir / "hoshicore/dag/startrail.meta.yaml"
        report = run_cli([str(dag), "--inspect", "--log-path", str(work / "inspect.log")], "inspect")
        if native and "custom_ops=available" not in report:
            raise RuntimeError(f"Packaged _C failed to load:\n{report}")
        if "pyexiv2=available" not in report:
            raise RuntimeError(f"Packaged EXIF backend failed to load:\n{report}")
        frames = []
        paths = []
        for index in range(3):
            frame = np.random.default_rng(index).integers(1, 65535, (32, 48, 3), dtype=np.uint16)
            path = work / f"frame-{index}.png"
            if not cv2.imwrite(str(path), frame):
                raise RuntimeError("Could not create packaging verification input")
            frames.append(frame)
            paths.append(str(path))
        listing = work / "inputs.txt"
        listing.write_text("\n".join(paths), encoding="utf-8")
        output = work / "stacked.png"
        run_cli([str(dag), "--input", f"fnames=@{listing}", "--route", "mode=fifo",
                 "--config", f"output_filename={output}", "--config", "output_dtype=uint16",
                 "--preflight", "abort", "--no-progress", "--log-path", str(work / "pipeline.log")],
                "pipeline")
        np.testing.assert_array_equal(cv2.imread(str(output), cv2.IMREAD_UNCHANGED),
                                      np.maximum.reduce(frames))
        startup = None
        if os.name == "nt":
            startup = subprocess.STARTUPINFO()
            startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startup.wShowWindow = 0
        with (work / "gui.txt").open("w", encoding="utf-8") as log:
            gui_log_root = result.gui_log_dir
            previous_logs = set(gui_log_root.glob("*_gui_*.log"))
            process = subprocess.Popen([str(gui)], cwd=work, env=env, startupinfo=startup,
                                       stdout=log, stderr=log)
            try:
                deadline = time.monotonic() + 90
                while time.monotonic() < deadline:
                    reports = [path.read_text(encoding="utf-8", errors="replace")
                               for path in set(gui_log_root.glob("*_gui_*.log")) - previous_logs]
                    if any("pyexiv2=available" in report
                           and (not native or "custom_ops=available" in report)
                           for report in reports):
                        break
                    code = process.poll()
                    if code is not None:
                        log.flush()
                        raise RuntimeError(f"Packaged GUI exited before initialization ({code}):\n"
                                           + (work / "gui.txt").read_text(encoding="utf-8"))
                    time.sleep(0.1)
                else:
                    raise RuntimeError("Packaged GUI did not report initialized runtime components within 90s")
                try:
                    code = process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    pass  # Survived after imports and runtime diagnostics.
                else:
                    log.flush()
                    raise RuntimeError(f"Packaged GUI exited during startup ({code}):\n"
                                       + (work / "gui.txt").read_text(encoding="utf-8"))
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=30)
                for path in set(gui_log_root.glob("*_gui_*.log")) - previous_logs:
                    path.unlink()
                if gui_log_root.is_dir() and not any(gui_log_root.iterdir()):
                    gui_log_root.rmdir()
    print("Package verification passed: CLI args, DAG, EXIF, native backend, exact stacking, GUI startup.")


