from pathlib import Path

import pytest

from hoshicore.packaging import common


def test_shared_distribution_keeps_both_executables_and_one_runtime(tmp_path):
    gui, cli, output = (tmp_path / name for name in ("gui", "cli", "output"))
    gui.mkdir()
    cli.mkdir()
    (gui / "desktop.exe").write_bytes(b"gui")
    (cli / "launcher.exe").write_bytes(b"cli")
    for directory in (gui, cli):
        (directory / "runtime.dll").write_bytes(b"shared runtime")
    common.merge_distribution(gui, output)
    common.merge_distribution(cli, output)
    assert sorted(path.name for path in output.iterdir()) == [
        "desktop.exe", "launcher.exe", "runtime.dll"]
    assert (output / "runtime.dll").read_bytes() == b"shared runtime"



def test_shared_distribution_rejects_same_size_dll_collision(tmp_path):
    first, second, output = (tmp_path / name for name in ("first", "second", "output"))
    first.mkdir()
    second.mkdir()
    (first / "runtime.dll").write_bytes(b"first!")
    (second / "runtime.dll").write_bytes(b"second")
    common.merge_distribution(first, output)
    with pytest.raises(RuntimeError, match="Conflicting distribution file"):
        common.merge_distribution(second, output)
    assert (output / "runtime.dll").read_bytes() == b"first!"



def test_no_build_requires_importable_extension_unless_fallback_allowed(monkeypatch):
    calls = []

    def missing(command, **kwargs):
        calls.append(command)
        return common.subprocess.CompletedProcess(command, 1, stderr=b"missing _C")

    monkeypatch.setattr(common.subprocess, "run", missing)
    with pytest.raises(RuntimeError, match="Custom ops"):
        common.ensure_custom_ops(skip_build=True, allow_numpy_only=False)
    assert common.ensure_custom_ops(skip_build=True, allow_numpy_only=True) is False
    assert all(command[1] == "-c" for command in calls)



def test_successful_build_is_rechecked_before_packaging(monkeypatch):
    results = iter((1, 0, 1))

    def result(command, **kwargs):
        return common.subprocess.CompletedProcess(command, next(results), stderr=b"bad _C")

    monkeypatch.setattr(common.subprocess, "run", result)
    with pytest.raises(RuntimeError, match="Custom ops"):
        common.ensure_custom_ops(skip_build=False, allow_numpy_only=False)



def test_upx_preserves_custom_op_extensions_and_vc_runtime(tmp_path, monkeypatch):
    for name in ("_C.cp313.pyd", "_C.cpython.so", "_metal.so", "vcruntime140.dll", "VCRUNTIME140_1.dll", "launcher.exe"):
        (tmp_path / name).write_bytes(b"binary")
    monkeypatch.setattr(common.shutil, "which", lambda name: "upx")
    calls = []

    def compress(command, **kwargs):
        calls.append(Path(command[-1]).name)
        return common.subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(common.subprocess, "run", compress)
    common.apply_upx(tmp_path)
    assert calls == ["launcher.exe"]



def test_compiler_environment_keeps_toolchain_paths_and_drops_credentials(monkeypatch):
    monkeypatch.setenv("INCLUDE", "test-sdk")
    monkeypatch.setenv("TEST_API_KEY", "test-placeholder")
    monkeypatch.setenv("TEST_AUTH_TOKEN", "test-placeholder")
    env = common.compiler_environment()
    assert env["INCLUDE"] == "test-sdk"
    assert "TEST_API_KEY" not in env
    assert "TEST_AUTH_TOKEN" not in env

