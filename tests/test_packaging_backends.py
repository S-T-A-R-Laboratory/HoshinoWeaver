"""Exercise backend configuration, compiler outputs and stage assembly together."""
from pathlib import Path
from types import SimpleNamespace
import importlib
import json

import pytest

from hoshicore.packaging import common


@pytest.mark.parametrize("name", ["pyinstaller", "nuitka"])
def test_backend_assembles_compiler_output_and_returns_explicit_layout(tmp_path, monkeypatch, name):
    backend = importlib.import_module(f"hoshicore.packaging.{name}")
    options = common.BuildOptions(root=tmp_path, backend=name, debug_gui=True,
                                  build_dir=tmp_path / "compiler")
    options.build_dir.mkdir()
    (options.build_dir / "HNW.ico").write_bytes(b"existing icon")
    manifest = common.PackageManifest(True, (), (), tmp_path / "ops")
    stage = tmp_path / "stage"
    stage.mkdir()
    commands = []
    monkeypatch.setattr(backend.importlib.util, "find_spec", lambda name: SimpleNamespace())

    def compile_output(command, *, root, dry_run):
        assert root == tmp_path and not dry_run
        commands.append(command)
        if name == "pyinstaller":
            output = Path(command[command.index("--distpath") + 1]) / options.release_name
            output.mkdir(parents=True)
            (output / "_internal").mkdir()
            suffix = ".exe" if common.os.name == "nt" else ""
            (output / (common.CLI_NAME + suffix)).write_bytes(b"CLI")
            (output / (common.GUI_NAME + suffix)).write_bytes(b"GUI")
            (output / "_internal/runtime.dll").write_bytes(b"shared runtime")
            compile(Path(command[-1]).read_text(encoding="utf-8"), "<spec>", "exec")
        else:
            output_dir = Path(next(c.split("=", 1)[1] for c in command if c.startswith("--output-dir=")))
            filename = next(c.split("=", 1)[1] for c in command if c.startswith("--output-filename="))
            output = output_dir / f"{Path(command[-1]).stem}.dist"
            output.mkdir(parents=True)
            (output / filename).write_bytes(b"entry")
            (output / "runtime.dll").write_bytes(b"shared runtime")
            report = Path(next(c.split("=", 1)[1] for c in command if c.startswith("--report=")))
            report.write_bytes(b"<?xml version='1.0' encoding='utf8'?>\n<report/>")

    monkeypatch.setattr(backend, "run_command", compile_output)
    result = backend.build(options, manifest, stage)
    assert result.directory == stage
    assert result.cli.is_file() and result.gui.is_file()
    assert (result.resource_dir / "runtime.dll").read_bytes() == b"shared runtime"
    assert result.gui_log_dir == result.resource_dir / "logs"
    metadata = tmp_path / "build" / name
    assert json.loads((metadata / "commands.json").read_text(encoding="utf-8")) == commands
    assert len(commands) == (1 if name == "pyinstaller" else 2)
    expected_flag = "--noconfirm" if name == "pyinstaller" else "--mode=standalone"
    assert expected_flag in commands[0]


def test_nuitka_configuration_keeps_pyexiv2_hack_and_describes_optional_binaries(tmp_path):
    from hoshicore.packaging import nuitka
    manifest = common.PackageManifest(True, (), (), tmp_path, turbojpeg=True,
                                      turbojpeg_libraries=(tmp_path / "turbojpeg.dll",))
    config = {item["module-name"]: item for item in nuitka.package_configuration(manifest)}
    assert config["pyexiv2.lib"]["import-hacks"] == [{"global-sys-path": [""]}]
    assert config["pyexiv2.lib"]["implicit-imports"] == [{"depends": ["exiv2api"]}]
    assert "hoshicore._custom_op" in config
    assert config["turbojpeg"]["dlls"][0]["by_code"]["filename_code"] == repr(str(tmp_path / "turbojpeg.dll"))
    fallback = common.PackageManifest(False, (), (), tmp_path)
    assert [item["module-name"] for item in nuitka.package_configuration(fallback)] == ["pyexiv2.lib"]
