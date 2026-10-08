"""Nuitka configuration, compilation and distribution assembly."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

from .common import (BASE_EXCLUDES, CLI_NAME, GUI_NAME, ROOT, BuildOptions,
                     BuildResult, PackageManifest, apply_upx, merge_distribution, run_command,
                     validate_build_directory)

EXCLUDES = BASE_EXCLUDES + ("matplotlib", "pandas", "IPython", "tkinter",
                           "sympy", "mpmath", "pyarrow", "narwhals")
BYTECODE_PACKAGES = ("numpy", "scipy", "astropy", "tifffile", "PIL", "ui.resource")


def default_build_root(root: Path = ROOT) -> Path:
    """Keep Windows compiler intermediates off Unicode checkout paths."""
    if os.name == "nt" and not str(root).isascii():
        identity = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:12]
        return Path(tempfile.gettempdir()) / f"hnw-nuitka-{identity}"
    return root / "build/nuitka"


def package_configuration(manifest: PackageManifest) -> list[dict]:
    # Extension modules are included separately; DLLs undergo Nuitka analysis.
    config = [{
        "module-name": "pyexiv2.lib",
        "import-hacks": [{"global-sys-path": [""]}],
        "implicit-imports": [{"depends": ["exiv2api"]}],
        "dlls": [{"from_filenames": {"prefixes": ["exiv2", "libexiv2"],
                                    "suffixes": ["dll", "so", "dylib"]}}],
    }]
    if manifest.native:
        config.append({"module-name": "hoshicore._custom_op",
                       "dlls": [{"from_filenames": {"prefixes": ["*"], "suffixes": ["dll"]},
                                 "when": "win32"}]})
    if manifest.turbojpeg:
        dlls = [{"by_code": {"filename_code": repr(str(path))}, "dest_path": "."}
                for path in manifest.turbojpeg_libraries]
        config.append({"module-name": "turbojpeg", "dlls": dlls})
    return config


def export_report(source: Path, destination: Path) -> None:
    """Use the standard XML encoding name for reports with Unicode paths.

    Nuitka 4.1.3 writes 'utf8'; Expat's byte parser rejects non-ASCII content
    under that declaration, despite the file being valid UTF-8.
    """
    report = source.read_bytes()
    header, separator, body = report.partition(b"\n")
    header = header.replace(b"encoding='utf8'", b"encoding='utf-8'")
    destination.write_bytes(header + separator + body)


def build_command(name: str, build_root: Path, config_path: Path, *,
                  manifest: PackageManifest, options: BuildOptions) -> list[str]:
    root = options.root
    native, debug_gui, jobs = manifest.native, options.debug_gui, options.jobs
    gui = name == GUI_NAME
    command = [
        sys.executable, "-m", "nuitka", "--mode=standalone",
        f"--output-dir={build_root / ('gui' if gui else 'cli')}",
        f"--output-filename={name}{'.exe' if os.name == 'nt' else ''}",
        f"--report={build_root / ('gui-report.xml' if gui else 'cli-report.xml')}",
        f"--user-package-configuration-file={config_path}",
        "--assume-yes-for-downloads", "--file-reference-choice=runtime",
        "--no-prefer-source-code",
        "--noinclude-pytest-mode=nofollow",
        "--noinclude-setuptools-mode=nofollow",
        "--noinclude-IPython-mode=nofollow",
        "--include-package=scipy._lib.array_api_compat",
        # Cython extensions import helpers invisible to Python source analysis.
        "--include-module=av.utils",
        "--include-module=pyexiv2", "--include-distribution-metadata=pyexiv2",
    ]
    command.extend(f"--include-data-dir={source}={destination}" for source, destination in manifest.data_dirs)
    command.extend(f"--include-data-files={source}={destination}" for source, destination in manifest.data_files)
    for excluded in EXCLUDES + (() if gui else ("PySide6", "shiboken6")):
        command.append(f"--nofollow-import-to={excluded}")
    for package in BYTECODE_PACKAGES:
        command.append(f"--noinclude-custom-mode={package}:bytecode")
    if native:
        command.append("--include-module=hoshicore._custom_op._C")
    else:
        command.append("--nofollow-import-to=hoshicore._custom_op._C")
    if manifest.metal_shader:
        command.append("--include-module=hoshicore._custom_op._metal")
    if manifest.turbojpeg:
        command.append("--include-module=turbojpeg")
    if jobs is not None:
        command.append(f"--jobs={jobs}")
    if os.name == "nt":
        command.extend(["--msvc=latest",
                        # Legacy depends.exe corrupts non-ASCII paths. This
                        # scanner is available in the required Nuitka 4.1.3.
                        "--experimental=force-dependencies-pefile",
                        "--include-windows-runtime-dlls=yes",
                        f"--windows-console-mode={'disable' if gui and not debug_gui else 'force'}"])
        if not str(root).isascii():
            # clcache decodes diagnostics as mbcs, which can disagree with
            # localized MSVC output. Bypass it for Unicode checkouts.
            command.append("--disable-cache=ccache")
        if gui:
            # Nuitka accepts ICO/PNG, whereas the legacy spec uses HNW.jpg.
            from PIL import Image
            icon_path = build_root / "HNW.ico"
            if not icon_path.is_file():
                with Image.open(root / "imgs/HNW.jpg") as icon:
                    icon.save(icon_path, sizes=[(16, 16), (32, 32), (48, 48), (256, 256)])
            command.append(f"--windows-icon-from-ico={icon_path}")
    if gui:
        command.extend(["--enable-plugin=pyside6", "--include-qt-plugins=sensible"])
        if sys.platform == "darwin" and not debug_gui:
            command.extend(["--macos-create-app-bundle",
                            f"--macos-app-name={GUI_NAME}",
                            "--macos-signed-app-name=com.hoshinoweaver.desktop"])
    command.append(str(root / f"{name}.py"))
    return command


def describe_release(directory: Path, options: BuildOptions) -> BuildResult:
    suffix = ".exe" if os.name == "nt" else ""
    gui = directory / (GUI_NAME + suffix)
    resource = directory
    if sys.platform == "darwin" and not options.debug_gui:
        gui = directory / f"{GUI_NAME}.app/Contents/MacOS/{GUI_NAME}"
        resource = gui.parent
    return BuildResult(directory, directory / (CLI_NAME + suffix), gui,
                       directory, resource / "logs")


def build(options: BuildOptions, manifest: PackageManifest, stage: Path | None) -> BuildResult:
    if importlib.util.find_spec("nuitka") is None:
        raise RuntimeError("Nuitka is not installed; run python -m pip install -r requirements-dev.txt")
    build_root = validate_build_directory(options.build_dir or default_build_root(options.root), options)
    metadata_root = options.root / "build/nuitka"
    build_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)
    print(f"Compiler working directory: {build_root}", flush=True)
    config_path = metadata_root / "hnw.nuitka-package.config.yml"
    config_path.write_text(json.dumps(package_configuration(manifest), indent=2), encoding="utf-8")
    commands = [build_command(entry, build_root, config_path, manifest=manifest, options=options)
                for entry in (CLI_NAME, GUI_NAME)]
    (metadata_root / "commands.json").write_text(json.dumps(commands, indent=2), encoding="utf-8")
    for command in commands:
        run_command(command, root=options.root, dry_run=options.dry_run)
    if options.dry_run:
        return describe_release(options.release_dir, options)
    for report in ("cli-report.xml", "gui-report.xml"):
        export_report(build_root / report, metadata_root / report)
    if stage is None:
        raise RuntimeError("Missing release staging directory")
    merge_distribution(build_root / "cli" / f"{CLI_NAME}.dist", stage)
    gui_root = build_root / "gui"
    if sys.platform == "darwin" and not options.debug_gui:
        app = gui_root / f"{GUI_NAME}.app"
        merge_distribution(app, stage / app.name)
    else:
        merge_distribution(gui_root / f"{GUI_NAME}.dist", stage)
    return describe_release(stage, options)


def postprocess(result: BuildResult, options: BuildOptions) -> None:
    if options.apply_upx:
        apply_upx(result.directory)
