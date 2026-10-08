from pathlib import Path
from types import SimpleNamespace
import importlib
import zipfile

import pytest

import make_package
from hoshicore.packaging import common


def fake_backend(events):
    def describe(directory, options):
        return common.BuildResult(directory, directory / "launcher", directory / "gui",
                                  directory, directory / "logs")

    def build(options, manifest, stage):
        events.append("build")
        if stage is None:
            return describe(options.release_dir, options)
        (stage / "launcher").write_bytes(b"new CLI")
        return describe(stage, options)

    return SimpleNamespace(build=build, describe_release=describe,
                           postprocess=lambda result, options: events.append("postprocess"))


def prepare_pipeline(monkeypatch):
    monkeypatch.setattr(common, "ensure_custom_ops", lambda *a, **kw: True)
    monkeypatch.setattr(common, "discover_manifest",
                        lambda root, *, native: common.PackageManifest(native, (), (), root))


def test_common_pipeline_verifies_before_replacing_and_writes_root_zip(tmp_path, monkeypatch):
    prepare_pipeline(monkeypatch)
    options = common.BuildOptions(root=tmp_path, overwrite=True, verify=True, apply_zip=True)
    final = options.release_dir
    final.mkdir(parents=True)
    (final / "launcher").write_bytes(b"old CLI")
    events = []

    def verify(result, *, native):
        events.append("verify")
        assert native
        assert (final / "launcher").read_bytes() == b"old CLI"
        assert result.cli.read_bytes() == b"new CLI"

    monkeypatch.setattr(common, "verify_distribution", verify)
    assert common.run_packaging(options, fake_backend(events)) == 0
    assert events == ["build", "postprocess", "verify"]
    assert (final / "launcher").read_bytes() == b"new CLI"
    archive = next(final.parent.glob("*.zip"))
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.namelist() == ["launcher"]
        assert zipped.read("launcher") == b"new CLI"
    assert not list(final.parent.glob(".package-*"))


def test_failed_verification_keeps_existing_release_and_cleans_stage(tmp_path, monkeypatch):
    prepare_pipeline(monkeypatch)
    options = common.BuildOptions(root=tmp_path, overwrite=True, verify=True, apply_zip=True)
    options.release_dir.mkdir(parents=True)
    (options.release_dir / "sentinel").write_text("existing release")

    def fail(*args, **kwargs):
        raise RuntimeError("bad runtime")

    monkeypatch.setattr(common, "verify_distribution", fail)
    with pytest.raises(RuntimeError, match="bad runtime"):
        common.run_packaging(options, fake_backend([]))
    assert (options.release_dir / "sentinel").read_text() == "existing release"
    assert sorted(p.name for p in options.release_dir.parent.iterdir()) == [options.release_name]


def test_publish_rolls_back_when_stage_rename_fails(tmp_path, monkeypatch):
    final, stage = tmp_path / "release", tmp_path / "stage"
    final.mkdir()
    stage.mkdir()
    (final / "sentinel").write_text("old")
    original_rename = Path.rename

    def fail_stage(path, target):
        if path == stage:
            raise OSError("rename failed")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_stage)
    with pytest.raises(OSError, match="rename failed"):
        common.publish_release(stage, final, overwrite=True)
    assert (final / "sentinel").read_text() == "old"
    assert not list(tmp_path.glob(".package-backup-*"))


def test_dry_run_does_not_create_release_or_compile_native(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(common, "ensure_custom_ops",
                        lambda root, **kw: calls.append(kw) or True)
    monkeypatch.setattr(common, "discover_manifest",
                        lambda root, *, native: common.PackageManifest(native, (), (), root))
    options = common.BuildOptions(root=tmp_path, dry_run=True)
    events = []
    common.run_packaging(options, fake_backend(events))
    assert calls == [{"skip_build": True, "allow_numpy_only": False}]
    assert events == ["build"]
    assert not (tmp_path / "dist").exists()


@pytest.mark.parametrize("backend_name", ["pyinstaller", "nuitka"])
def test_verify_only_uses_backend_layout_without_native_probe(tmp_path, monkeypatch, backend_name):
    backend = importlib.import_module(f"hoshicore.packaging.{backend_name}")
    options = common.BuildOptions(root=tmp_path, backend=backend_name, verify_only=True, debug_gui=True)
    options.release_dir.mkdir(parents=True)

    def unexpected(*a, **kw):
        pytest.fail("verify-only must not probe or compile native code")

    monkeypatch.setattr(common, "ensure_custom_ops", unexpected)
    calls = []
    monkeypatch.setattr(common, "verify_distribution", lambda result, *, native: calls.append((result, native)))
    assert common.run_packaging(options, backend) == 0
    result, native = calls[0]
    assert native
    assert result.resource_dir == options.release_dir / ("_internal" if backend_name == "pyinstaller" else "")
    assert result.gui_log_dir == result.resource_dir / "logs"


def test_manifest_rejects_missing_metal_shader(tmp_path, monkeypatch):
    monkeypatch.setattr(common.importlib.util, "find_spec",
                        lambda name: SimpleNamespace(origin=str(tmp_path / "_metal.so")) if name.endswith("._metal") else None)
    with pytest.raises(RuntimeError, match="_metal_kernels.metallib"):
        common.discover_manifest(tmp_path, native=True)


def test_manifest_keeps_data_and_native_binaries_separate(tmp_path, monkeypatch):
    monkeypatch.setattr(common.importlib.util, "find_spec", lambda name: None)
    (tmp_path / "hoshicore/dag").mkdir(parents=True)
    (tmp_path / "hoshicore/_custom_op").mkdir()
    (tmp_path / "hoshicore/default_settings.yaml").write_text("settings")
    (tmp_path / "LICENSE").write_text("license")
    dll = tmp_path / "hoshicore/_custom_op/runtime.dll"
    dll.write_bytes(b"native")
    for native in (True, False):
        manifest = common.discover_manifest(tmp_path, native=native)
        assert manifest.custom_op_dlls == ((dll,) if native else ())
        assert not any(path.suffix == ".dll" for path, _ in manifest.data_files)
        assert (tmp_path / "LICENSE", "LICENSE") in manifest.data_files


@pytest.mark.parametrize("relative", [".", "dist", "dist/existing-release"])
def test_build_directory_cannot_write_into_release_tree(tmp_path, relative):
    options = common.BuildOptions(root=tmp_path)
    with pytest.raises(RuntimeError, match="Build directory must be outside"):
        common.validate_build_directory(tmp_path / relative, options)
    assert common.validate_build_directory(tmp_path / "build/compiler", options) == tmp_path / "build/compiler"


@pytest.mark.parametrize("entry,arguments,expected", [
    (make_package.main, [], "pyinstaller"),
    (make_package.main, ["--backend", "nuitka"], "nuitka"),
])
def test_entrypoints_select_backend_and_share_options(monkeypatch, entry, arguments, expected):
    calls = []
    monkeypatch.setattr(common, "run_packaging", lambda options, backend: calls.append((options, backend)) or 0)
    assert entry(arguments + ["--no-build", "--apply-zip", "--verify", "--jobs=4"]) == 0
    options, backend = calls[0]
    assert options.backend == expected
    assert backend.__name__ == f"hoshicore.packaging.{expected}"
    assert options.no_build and options.apply_zip and options.verify and options.jobs == 4


def test_help_and_package_import_do_not_require_packager_dependencies():
    import subprocess
    import sys
    source = """
import sys
sys.modules['PyInstaller'] = None
sys.modules['nuitka'] = None
import hoshicore.packaging
assert 'hoshicore.packaging.common' not in sys.modules
import make_package
try:
    make_package.main(['--help'])
except SystemExit as exc:
    assert exc.code == 0
assert 'hoshicore.packaging.pyinstaller' not in sys.modules
assert 'hoshicore.packaging.nuitka' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", source], cwd=common.ROOT, check=True, capture_output=True)
