"""CLI-level checks for the launcher flags used by automation tooling.

These cover the parts of ``launcher.py`` that other tools (the workflow
benchmark harness) depend on: input-list resolution, non-interactive
preflight policies, the single printed log-path line, and the exit-code
contract.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

import launcher
from hoshicore.engine.executor import DAGExecutionError
from hoshicore.engine.preflight import (CheckResult, PreflightAbortError,
                                        PreflightIssue)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STACK_YAML = PROJECT_ROOT / "hoshicore" / "dag" / "stack.meta.yaml"


@pytest.fixture
def log_env(monkeypatch, tmp_path):
    """Pin the log location and skip native capability probing.

    ``init_logger`` stays real so the log-path contract (exactly one printed
    line, the file actually created) is exercised without spawning a second
    interpreter.
    """
    default_log = tmp_path / "default" / "run.log"
    monkeypatch.setattr(launcher, "_make_log_filename",
                        lambda task: str(default_log))
    monkeypatch.setattr(launcher, "log_runtime_components", lambda logger: None)
    return default_log


# ── --input value resolution ────────────────────────────────────────────────

def test_resolve_input_value_plain_string():
    assert launcher._resolve_input_value("some_value") == "some_value"


def test_resolve_input_value_directory_is_sorted_and_filtered(tmp_path):
    for name in ("b.tif", "a.jpg", "notes.txt"):
        (tmp_path / name).write_bytes(b"x")
    resolved = launcher._resolve_input_value(str(tmp_path))
    assert [Path(entry).name for entry in resolved] == ["a.jpg", "b.tif"]


def test_resolve_input_value_list_file_skips_blank_and_comments(tmp_path):
    entries = tmp_path / "lights.txt"
    entries.write_text("\n# comment\na.tif\n\n   b.tif   \n", encoding="utf-8")
    assert launcher._resolve_input_value(f"@{entries}") == ["a.tif", "b.tif"]


def test_resolve_input_value_missing_list_file(tmp_path):
    with pytest.raises(ValueError):
        launcher._resolve_input_value(f"@{tmp_path / 'missing.txt'}")


def test_resolve_input_value_json_array():
    assert launcher._resolve_input_value('["a.tif", "b.tif"]') == [
        "a.tif", "b.tif"]


def test_resolve_input_value_non_array_json_passes_through():
    assert launcher._resolve_input_value('{"a": 1}') == '{"a": 1}'


# ── --preflight callback factory ───────────────────────────────────────────

def _check_result() -> CheckResult:
    return CheckResult(check_name="资源检查",
                       issues=[PreflightIssue(severity="warning",
                                              code="resource.ram",
                                              message="RAM 不足")])


@pytest.mark.parametrize("mode", ["apply", "ignore", "abort"])
def test_fixed_preflight_callback_returns_requested_action(mode):
    callback = launcher._make_preflight_callback(mode)
    assert callback(_check_result()) == mode


def test_ask_preflight_callback_stays_interactive():
    assert launcher._make_preflight_callback("ask") is launcher._cli_preflight_callback


# ── exit codes ─────────────────────────────────────────────────────────────

def _run_main(monkeypatch, argv: list[str]):
    monkeypatch.setattr(sys, "argv", ["launcher.py", *argv])
    launcher.main()


def test_main_exits_two_when_preflight_aborts(monkeypatch, log_env):
    def _abort(*args, **kwargs):
        raise PreflightAbortError("用户中止预检，执行已取消。")

    monkeypatch.setattr(launcher, "run_from_yaml", _abort)
    with pytest.raises(SystemExit) as excinfo:
        _run_main(monkeypatch, [str(STACK_YAML),
                                "--input", "fnames=a.tif",
                                "--preflight", "abort"])
    assert excinfo.value.code == 2


def test_main_exits_one_on_dag_execution_error(monkeypatch, log_env):
    def _fail(*args, **kwargs):
        raise DAGExecutionError(RuntimeError("boom"), "stacker",
                                [("stacker", RuntimeError("boom"))], [])

    monkeypatch.setattr(launcher, "run_from_yaml", _fail)
    with pytest.raises(SystemExit) as excinfo:
        _run_main(monkeypatch, [str(STACK_YAML), "--input", "fnames=a.tif"])
    assert excinfo.value.code == 1


def test_main_passes_progress_and_preflight_to_pipeline(monkeypatch, log_env):
    captured: dict[str, object] = {}

    async def _capture(*args, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(launcher, "run_from_yaml", _capture)
    _run_main(monkeypatch, [str(STACK_YAML), "--input", "fnames=a.tif",
                            "--no-progress", "--preflight", "ignore"])
    assert captured["progress"] is False
    assert captured["preflight_callback"](_check_result()) == "ignore"


# ── printed log path ───────────────────────────────────────────────────────

def test_default_log_path_is_printed_when_flag_missing(monkeypatch, log_env,
                                                       capsys):
    _run_main(monkeypatch, [str(STACK_YAML), "--inspect"])
    out = capsys.readouterr().out
    assert out.count("[Launcher] log file:") == 1
    assert str(log_env) in out
    assert log_env.is_file()
    assert log_env.stat().st_size > 0


def test_log_path_supports_nested_directories(monkeypatch, log_env, capsys,
                                              tmp_path):
    custom = tmp_path / "deep" / "dir" / "run.log"
    _run_main(monkeypatch, [str(STACK_YAML), "--inspect",
                            "--log-path", str(custom)])
    out = capsys.readouterr().out
    assert out.count("[Launcher] log file:") == 1
    assert str(custom) in out
    assert custom.is_file()
    assert custom.stat().st_size > 0
