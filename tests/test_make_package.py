from types import SimpleNamespace
import sys

import pytest

from hoshicore.packaging import common, pyinstaller


@pytest.mark.parametrize("native", [True, False])
@pytest.mark.parametrize("debug_gui", [True, False])
def test_generated_spec_executes_with_manifest_and_correct_console_modes(monkeypatch, tmp_path, native, debug_gui):
    shader = tmp_path / "_metal_kernels.metallib"
    manifest = common.PackageManifest(
        native, ((tmp_path / "dag", "hoshicore/dag"),),
        ((shader, "hoshicore/_custom_op/_metal_kernels.metallib"),),
        tmp_path / "ops", (tmp_path / "libgomp-1.dll",) if native else (), shader, True,
        (tmp_path / "turbojpeg.dll",))
    options = common.BuildOptions(root=tmp_path, debug_gui=debug_gui, apply_upx=True)
    spec = pyinstaller.generate_spec(options, manifest)
    analyses, executables, merges, collections = [], [], [], []

    def analysis(scripts, **kwargs):
        analyses.append(kwargs)
        return SimpleNamespace(pure=(), scripts=scripts, zipfiles=[],
                               binaries=kwargs["binaries"], datas=kwargs["datas"])

    def exe(*args, **kwargs):
        executables.append(kwargs)
        return kwargs["name"]

    def collect(*args, **kwargs):
        collections.append(kwargs)
        return "collection"

    hooks = SimpleNamespace(collect_all=lambda name: ([], [], ["exiv2api"]),
                            copy_metadata=lambda name: [], collect_submodules=lambda name: [name])
    monkeypatch.setitem(sys.modules, "PyInstaller.utils.hooks", hooks)
    monkeypatch.setitem(sys.modules, "pyexiv2", SimpleNamespace())
    namespace = dict(Analysis=analysis, EXE=exe, COLLECT=collect, BUNDLE=lambda *a, **kw: None,
                     PYZ=lambda pure: pure, MERGE=lambda *args: merges.append(args))
    exec(compile(spec, "<generated.spec>", "exec"), namespace)
    assert [e["console"] for e in executables] == [debug_gui, True]
    assert [e["name"] for e in executables] == [common.GUI_NAME, common.CLI_NAME]
    assert all(e["upx"] for e in executables)
    assert all("hoshicore/_custom_op/_C*.pyd" in e["upx_exclude"] for e in executables)
    assert collections[0]["name"] == options.release_name
    assert len(merges[0]) == 2
    for a in analyses:
        assert ("hoshicore._custom_op._C" in a["hiddenimports"]) is native
        assert ("hoshicore._custom_op._C" in a["excludes"]) is not native
        assert "hoshicore._custom_op._metal" in a["hiddenimports"]
        assert "exiv2api" in a["hiddenimports"]
        assert "turbojpeg" in a["hiddenimports"]
        assert "hoshicore.packaging" in a["excludes"]
        assert (str(shader), "hoshicore/_custom_op") in a["datas"]
        assert (str(tmp_path / "turbojpeg.dll"), ".") in a["binaries"]
    assert "PySide6" in analyses[1]["excludes"]


def test_macos_spec_builds_app_only_for_release_gui(monkeypatch, tmp_path):
    monkeypatch.setattr(pyinstaller.sys, "platform", "darwin")
    monkeypatch.setattr(common.platform, "mac_ver", lambda: ("14.0", (), ""))
    manifest = common.PackageManifest(True, (), (), tmp_path)
    for debug, bundle in ((False, True), (True, False)):
        spec = pyinstaller.generate_spec(common.BuildOptions(root=tmp_path, debug_gui=debug), manifest)
        compile(spec, "<generated.spec>", "exec")
        assert ("app = BUNDLE(" in spec) is bundle
