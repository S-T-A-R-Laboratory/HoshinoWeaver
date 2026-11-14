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
    assert config["pyexiv2.lib"]["import-hacks"] == [
        {"global-sys-path": [""]},
        {"package-system-dlls": "yes", "when": "macos"},
    ]
    assert config["pyexiv2.lib"]["implicit-imports"] == [{"depends": ["exiv2api"]}]
    assert "hoshicore._custom_op" in config
    assert config["turbojpeg"]["dlls"][0]["by_code"]["filename_code"] == repr(str(tmp_path / "turbojpeg.dll"))
    fallback = common.PackageManifest(False, (), (), tmp_path)
    assert 'hoshicore._custom_op' not in {
        item['module-name'] for item in nuitka.package_configuration(fallback)}


@pytest.mark.parametrize('platform', ['darwin', 'win32', 'linux'])
@pytest.mark.parametrize('debug_gui', [False, True])
def test_nuitka_uses_one_modern_mode_for_cli_and_gui(tmp_path, monkeypatch, platform, debug_gui):
    from hoshicore.packaging import nuitka
    monkeypatch.setattr(nuitka.sys, 'platform', platform)
    # Windows command generation reuses this icon instead of reading a source image.
    (tmp_path / 'HNW.ico').write_bytes(b'icon')
    manifest = common.PackageManifest(False, (), (), tmp_path)
    options = common.BuildOptions(root=tmp_path, debug_gui=debug_gui)
    for entry in (common.CLI_NAME, common.GUI_NAME):
        command = nuitka.build_command(entry, tmp_path, tmp_path / 'config.yml',
                                       manifest=manifest, options=options)
        app = platform == 'darwin' and entry == common.GUI_NAME and not debug_gui
        assert [arg for arg in command if arg.startswith('--mode=')] == [
            '--mode=app' if app else '--mode=standalone']
        assert not set(command) & {'--standalone', '--onefile', '--macos-create-app-bundle'}
        assert any(arg.startswith('--macos-app-name=') for arg in command) is app
        assert '--include-package=astropy.constants' in command


@pytest.mark.parametrize('package', ['astropy.units.format', 'astropy.coordinates.angles'])
def test_astropy_ply_tables_ship_as_modules_and_files_without_regeneration(tmp_path, monkeypatch, package):
    from fnmatch import fnmatch
    import shutil
    from astropy.extern.ply import yacc
    from astropy.utils.parsing import _patch_ply_module
    from hoshicore.packaging import nuitka

    manifest = common.PackageManifest(False, (), (), tmp_path)
    config = next(c for c in nuitka.package_configuration(manifest) if c['module-name'] == package)
    folder = Path(importlib.import_module(package).__file__).parent
    patterns = config['data-files'][0]['patterns']
    sources = sorted({p for pattern in patterns for p in folder.glob(pattern)})
    assert sources, 'Installed Astropy must provide pre-generated PLY tables'
    modules = config['implicit-imports'][0]['depends']

    def forbid_regeneration(*args, **kwargs):
        pytest.fail('A packaged parser must not regenerate its tables')

    monkeypatch.setattr(Path, 'write_text', forbid_regeneration)
    for source in sources:
        assert any(fnmatch(f'{package}.{source.stem}', pattern) for pattern in modules)
        target = tmp_path / source.name
        shutil.copy2(source, target)
        with _patch_ply_module(yacc, target, package):
            if source.stem.endswith('_parsetab'):
                # Validate real tables through PLY without generating grammar.
                table = importlib.import_module(f'{package}.{source.stem}')
                assert yacc.LRTable().read_table(table)
        assert target.read_bytes() == source.read_bytes()
