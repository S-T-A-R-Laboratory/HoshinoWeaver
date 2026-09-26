"""Unit tests for the CLI-level workflow benchmark harness.

These cover the pure planning/reporting logic only: suite validation, input
freezing, command building, log phase extraction and resource aggregation. No
workflow is executed and no local dataset is required.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks import benchmark_workflow as bench_cli
from benchmarks.workflow_bench import (ResourceSamples, SuiteError,
                                       build_launcher_command, case_paths,
                                       effective_configs, format_config_value,
                                       load_suite, materialize_run, parse_log,
                                       parse_printed_log_path, plan_case,
                                       probe_storage_paths, resolve_inputs,
                                       select_cases, substitute_placeholders,
                                       summarize_resources, summarize_run_results)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_SUITE = PROJECT_ROOT / "benchmarks" / "workflow_bench.example.json"
STACK_YAML = PROJECT_ROOT / "hoshicore" / "dag" / "stack.meta.yaml"


def _write_suite(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _make_suite(inputs_dir: Path, *, defaults: dict | None = None,
                case: dict | None = None) -> dict:
    base_case = {
        "id": "case_a",
        "pipeline": "hoshicore/dag/stack.meta.yaml",
        "inputs": {"fnames": str(inputs_dir)},
        "routes": {"stacker": "mean"},
        "configs": {"int_weight": True},
    }
    if case:
        base_case.update(case)
    return {
        "schema_version": 1,
        "suite_id": "test_suite",
        "defaults": defaults or {},
        "cases": [base_case],
    }


def _lights_dir(tmp_path: Path, names: tuple[str, ...] = ("a.jpg", "b.tif",
                                                         "notes.txt")) -> Path:
    lights = tmp_path / "lights"
    lights.mkdir(exist_ok=True)
    for name in names:
        (lights / name).write_bytes(b"x")
    return lights


# ── suite schema ───────────────────────────────────────────────────────────

def test_load_suite_rejects_unknown_default_key(tmp_path):
    payload = _make_suite(tmp_path)
    payload["defaults"] = {"bogus": 1}
    with pytest.raises(SuiteError):
        load_suite(_write_suite(tmp_path, payload))


def test_load_suite_rejects_duplicate_case_ids(tmp_path):
    payload = _make_suite(tmp_path)
    payload["cases"] = [payload["cases"][0], payload["cases"][0]]
    with pytest.raises(SuiteError):
        load_suite(_write_suite(tmp_path, payload))


def test_case_values_override_defaults(tmp_path):
    payload = _make_suite(tmp_path,
                          defaults={"repeats": 3, "preflight": "ignore",
                                    "timeout_seconds": 60},
                          case={"repeats": 1})
    suite = load_suite(_write_suite(tmp_path, payload))
    case = suite.cases[0]
    assert case.repeats == 1
    assert case.preflight == "ignore"
    assert case.timeout_seconds == 60


def test_runtime_paths_are_suite_wide_only(tmp_path):
    payload = _make_suite(tmp_path, case={"working_dir": str(tmp_path)})
    with pytest.raises(SuiteError, match="unknown key"):
        load_suite(_write_suite(tmp_path, payload))


def test_null_config_value_is_rejected(tmp_path):
    payload = _make_suite(tmp_path, case={"configs": {"int_weight": None}})
    with pytest.raises(SuiteError):
        load_suite(_write_suite(tmp_path, payload))


def test_load_suite_tolerates_utf8_bom(tmp_path):
    payload = _make_suite(_lights_dir(tmp_path))
    path = tmp_path / "suite_bom.json"
    path.write_text(json.dumps(payload), encoding="utf-8-sig")
    suite = load_suite(path)
    assert suite.cases[0].id == "case_a"


def test_select_cases_filters_by_id_and_label(tmp_path):
    payload = _make_suite(tmp_path)
    payload["cases"] = [
        {**payload["cases"][0], "id": "alpha", "labels": ["stacker:mean"]},
        {**payload["cases"][0], "id": "beta", "labels": ["stacker:median"]},
    ]
    suite = load_suite(_write_suite(tmp_path, payload))
    assert [c.id for c in select_cases(suite, case_ids=["beta"])] == ["beta"]
    assert [c.id for c in select_cases(suite, labels=["stacker:mean"])] == [
        "alpha"]
    with pytest.raises(SuiteError):
        select_cases(suite, case_ids=["nope"])


# ── inputs ────────────────────────────────────────────────────────────────

def test_resolve_inputs_sorts_filters_and_limits(tmp_path):
    lights = _lights_dir(tmp_path)
    payload = _make_suite(lights, case={"input_limit": 1})
    case = load_suite(_write_suite(tmp_path, payload)).cases[0]
    resolved = resolve_inputs(case)
    entry = resolved.entries[0]
    assert entry.source == "dir"
    assert entry.truncated is True
    assert [Path(item).name for item in entry.files] == ["a.jpg"]
    assert resolved.input_count == 1
    assert resolved.warnings


def test_resolve_inputs_list_file_skips_comments(tmp_path):
    lights = _lights_dir(tmp_path)
    list_file = tmp_path / "lights.txt"
    list_file.write_text("# comment\na.jpg\n\nb.tif\n", encoding="utf-8")
    payload = _make_suite(lights, case={"inputs": {"fnames": f"@{list_file}"}})
    case = load_suite(_write_suite(tmp_path, payload)).cases[0]
    entry = resolve_inputs(case).entries[0]
    assert entry.source == "list_file"
    assert list(entry.files) == ["a.jpg", "b.tif"]


def test_input_fingerprint_includes_name_and_delivery_order(tmp_path):
    lights = _lights_dir(tmp_path)
    payload = _make_suite(lights, case={
        "inputs": {"fnames": ["a.jpg", "b.tif"]}})
    path = _write_suite(tmp_path, payload)
    original = resolve_inputs(load_suite(path).cases[0]).list_sha1()
    payload["cases"][0]["inputs"] = {"fnames": ["b.tif", "a.jpg"]}
    reversed_order = resolve_inputs(load_suite(_write_suite(tmp_path, payload)).cases[0]).list_sha1()
    payload["cases"][0]["inputs"] = {"lights": ["a.jpg", "b.tif"]}
    renamed = resolve_inputs(load_suite(_write_suite(tmp_path, payload)).cases[0]).list_sha1()
    assert len({original, reversed_order, renamed}) == 3


def test_resolve_inputs_missing_directory(tmp_path):
    payload = _make_suite(tmp_path, case={"inputs": {"fnames": str(
        tmp_path / "missing")}})
    case = load_suite(_write_suite(tmp_path, payload)).cases[0]
    with pytest.raises(SuiteError):
        resolve_inputs(case)
    resolved = resolve_inputs(case, require_exists=False)
    assert resolved.input_count == 0
    assert resolved.warnings


# ── planning and command building ─────────────────────────────────────────

def _planned(tmp_path, *, case_overrides: dict | None = None):
    lights = _lights_dir(tmp_path)
    payload = _make_suite(lights, case=case_overrides)
    suite_path = _write_suite(tmp_path, payload)
    suite = load_suite(suite_path)
    return plan_case(suite.cases[0], suite_path=suite.path,
                     output_dir=tmp_path / "out", run_index=1)


def test_plan_case_builds_expected_command(tmp_path):
    planned = _planned(tmp_path, case_overrides={
        "configs": {"int_weight": True, "output_dtype": "uint16"}})
    command = list(planned.command)
    assert command[0].endswith("python.exe") or "python" in command[0]
    assert command[1] == "-u"
    assert str(planned.launcher_path) == str(PROJECT_ROOT / "launcher.py")
    assert str(STACK_YAML) in command
    assert "--preflight" in command and "ignore" in command
    assert "--no-progress" in command
    assert "--route" in command and "stacker=mean" in command
    assert "int_weight=true" in command
    assert "--log-path" in command
    input_payload = command[command.index("--input") + 1]
    assert input_payload.startswith("fnames=@")
    payload_path = Path(input_payload.split("=", 1)[1][1:])
    assert payload_path.name == "inputs_fnames.txt"
    assert payload_path.parent == planned.paths["run_dir"]


def test_materialize_run_creates_dirs_and_list_file(tmp_path):
    planned = _planned(tmp_path)
    materialize_run(planned)
    assert planned.paths["run_dir"].is_dir()
    assert planned.paths["output_root"].is_dir()
    for index, entry in enumerate(planned.inputs.entries):
        list_file = planned.paths["run_dir"] / f"inputs_{entry.name}.txt"
        assert list_file.is_file()
        assert list_file.read_text(encoding="utf-8").splitlines() == [
            str(Path(item)) for item in entry.files], index


def test_output_filename_defaults_under_run_output_root(tmp_path):
    planned = _planned(tmp_path)
    output = Path(str(planned.configs["output_filename"]))
    assert output.parent == planned.paths["output_root"]
    assert output.parent.parent == planned.paths["run_dir"]
    assert output.name == "case_a.tif"
    assert planned.expected_outputs == (output,)


def test_temp_path_injected_only_for_disk_buffers(tmp_path):
    disk = _planned(tmp_path, case_overrides={"configs": {
        "buffer_mode": "disk"}})
    assert disk.configs["temp_path"] == str(disk.paths["cache_root"])
    memory = _planned(tmp_path, case_overrides={"configs": {
        "buffer_mode": "memory"}})
    assert "temp_path" not in memory.configs


def test_expected_output_placeholders_are_resolved(tmp_path):
    planned = _planned(tmp_path, case_overrides={
        "expected_outputs": ["{output_root}/{case_id}.tif"]})
    assert planned.expected_outputs[0].name == "case_a.tif"


def test_format_config_value_and_placeholder_errors():
    assert format_config_value(True) == "true"
    assert format_config_value(False) == "false"
    assert format_config_value(0.5) == "0.5"
    assert format_config_value([1, 2]) == "[1, 2]"
    with pytest.raises(SuiteError):
        substitute_placeholders("{unknown}", {"case_id": "x"})


def test_case_paths_and_effective_configs_are_consistent(tmp_path):
    planned = _planned(tmp_path)
    paths = case_paths(planned.case, 1, output_dir=tmp_path / "out",
                       working_dir=PROJECT_ROOT)
    configs = effective_configs(planned.case, paths, 1)
    assert Path(str(configs["output_filename"])).parent == paths["output_root"]


def test_build_launcher_command_is_reproducible(tmp_path):
    planned = _planned(tmp_path)
    rebuilt = build_launcher_command(
        case=planned.case, pipeline_path=planned.pipeline_path,
        launcher_path=planned.launcher_path, python_bin=planned.python_bin,
        log_path=planned.paths["run_dir"] / "launcher.log",
        input_payloads=planned.input_payloads, routes=planned.case.routes,
        configs=planned.configs)
    assert tuple(rebuilt) == planned.command


# ── log parsing ───────────────────────────────────────────────────────────

_LOG_TEXT = "\n".join([
    "2026-09-17 20:00:00.100 | INFO | x - [Capabilities] cuda=available",
    "2026-09-17 20:00:01.000 | INFO | x - [Feeder] Global input 'fnames': "
    "50 items \u2192 2 queue(s)",
    "2026-09-17 20:00:02.000 | INFO | x - DAG execution starting (5 nodes)...",
    "2026-09-17 20:00:03.500 | WARNING | x - [Preflight][--preflight=ignore] "
    "RAM low",
    "2026-09-17 20:00:12.000 | INFO | x - DAG execution completed. "
    "Results collected.",
    "2026-09-17 20:00:12.500 | INFO | x - run_from_yaml time cost: 11.50s.",
])


def test_parse_log_extracts_phases(tmp_path):
    log_path = tmp_path / "launcher.log"
    log_path.write_text(_LOG_TEXT, encoding="utf-8")
    report = parse_log(log_path)
    assert report["source"] == "log"
    assert report["phases"]["pipeline_seconds"] == 11.5
    assert report["phases"]["execution_seconds"] == pytest.approx(10.0)
    assert report["phases"]["feeder_items"] == {"fnames": 50}
    assert len(report["preflight_lines"]) == 1
    assert len(report["capabilities_lines"]) == 1


def test_parse_log_missing_file_is_unavailable(tmp_path):
    report = parse_log(tmp_path / "nope.log")
    assert report["source"] == "unavailable"
    assert report["phases"] == {}


def test_parse_log_without_boundaries_is_unavailable(tmp_path):
    log_path = tmp_path / "launcher.log"
    log_path.write_text("garbage line\n", encoding="utf-8")
    assert parse_log(log_path)["source"] == "unavailable"


def test_parse_printed_log_path_takes_last_marker(tmp_path):
    stdout = tmp_path / "stdout.txt"
    stdout.write_text("[Launcher] log file: first.log\nnoise\n"
                      "[Launcher] log file: second.log\n", encoding="utf-8")
    assert parse_printed_log_path(stdout) == "second.log"
    stdout.write_text("no marker\n", encoding="utf-8")
    assert parse_printed_log_path(stdout) is None


# ── resource aggregation ──────────────────────────────────────────────────

def test_summarize_resources_metrics():
    samples = ResourceSamples(
        interval_seconds=0.5,
        timestamps=[0.0, 0.5, 1.0],
        process_cpu_percent=[100.0, 200.0, 300.0],
        cpu_time_seconds=[0.5, 1.0, 1.5],
        rss_bytes=[100, 200, 400],
        pagefile_bytes=[10, 20, 30],
        peak_wset_bytes=[100, 250, 400],
        thread_count=[2, 4, 4],
        process_count=[1, 1, 1],
        system_mem_used_bytes=[1000, 1500, 1200],
        swap_used_bytes=[10, 30, 20],
    )
    result = summarize_resources(samples, logical_cpu_count=4,
                                 system_mem_used_before=900,
                                 swap_used_before=5)
    assert result["cpu_percent_avg"] == pytest.approx(200.0)
    assert result["cpu_percent_peak"] == pytest.approx(300.0)
    assert result["cpu_percent_avg_normalized"] == pytest.approx(50.0)
    assert result["cpu_time_seconds"] == pytest.approx(1.5)
    assert result["rss_avg_bytes"] == pytest.approx(700 / 3)
    assert result["rss_delta_avg_bytes"] == pytest.approx(700 / 3 - 100)
    assert result["rss_delta_peak_bytes"] == pytest.approx(300)
    assert result["peak_wset_bytes"] == pytest.approx(400)
    assert result["thread_count_peak"] == pytest.approx(4)
    assert result["system_mem_delta_peak_bytes"] == pytest.approx(600)
    assert result["swap_used_delta_peak_bytes"] == pytest.approx(25)


def test_summarize_run_results_aggregates_repeats():
    runs = [
        {"status": "success", "expectation_met": True, "wall_seconds": 10.0,
         "resources": {"cpu_percent_avg_normalized": 50.0, "rss_peak_bytes": 100},
         "output_files": [{"bytes": 10}]},
        {"status": "success", "expectation_met": True, "wall_seconds": 20.0,
         "resources": {"cpu_percent_avg_normalized": 70.0, "rss_peak_bytes": 300},
         "output_files": [{"bytes": 30}]},
    ]
    summary = summarize_run_results(runs)
    assert summary["success_all"] is True
    assert summary["expectations_met"] is True
    assert summary["wall_seconds"]["mean"] == pytest.approx(15.0)
    assert summary["wall_seconds"]["min"] == pytest.approx(10.0)
    assert summary["wall_seconds"]["max"] == pytest.approx(20.0)
    assert summary["resources"]["cpu_percent_avg_normalized"]["mean"] == pytest.approx(60.0)
    assert summary["resources"]["rss_peak_bytes"]["max"] == pytest.approx(300)
    assert summary["output_bytes"]["max"] == pytest.approx(30.0)


def test_probe_storage_paths_handles_missing_path(tmp_path):
    missing = tmp_path / "not" / "there"
    result = probe_storage_paths([tmp_path, missing, None])
    assert result[str(tmp_path)]["free_bytes"] > 0
    # Non-existent directories resolve to their nearest existing parent.
    assert result[str(missing)]["resolved_path"] == str(tmp_path)
    assert result["path2"] is None


# ── CLI surface ───────────────────────────────────────────────────────────

def test_dry_run_reports_and_creates_nothing(tmp_path):
    output_dir = tmp_path / "dry_out"
    exit_code = bench_cli.main([str(EXAMPLE_SUITE), "--dry-run",
                                "--output-dir", str(output_dir)])
    assert exit_code == 0
    assert not output_dir.exists()


def test_dry_run_rejects_unknown_case(tmp_path):
    exit_code = bench_cli.main([str(EXAMPLE_SUITE), "--dry-run",
                                "--case", "nope"])
    assert exit_code == 2


def test_missing_suite_returns_usage_error(tmp_path):
    assert bench_cli.main([str(tmp_path / "nope.json")]) == 2


def test_hardware_only_does_not_load_a_suite(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(bench_cli, "probe_backend",
                        lambda *a, **k: {"status": "stub"})
    monkeypatch.setattr(bench_cli, "probe_hardware", lambda **k: {
        "hostname": "host", "platform": "plat", "python": "3.13",
        "python_executable": "python",
        "cpu": {"model": "cpu", "physical_cores": 1, "logical_cores": 2,
                "max_frequency_mhz": 100, "model_source": "stub"},
        "ram_total_bytes": 1, "swap_total_bytes": 1, "gpus": [],
    })
    monkeypatch.setattr(bench_cli, "probe_software", lambda: {
        "product": "HoshinoWeaver", "version": "0", "release_name": "stub",
        "dependencies": {}, "git_revision": None, "git_branch": None,
        "git_dirty": None,
    })
    exit_code = bench_cli.main(["--hardware-only",
                                str(tmp_path / "missing_suite.json")])
    assert exit_code == 0
    assert "[Machine]" in capsys.readouterr().out


def test_decorate_run_marks_declared_storage(tmp_path):
    planned = _planned(tmp_path, case_overrides={
        "storage": {"input_media": "nvme", "cache_media": None}})
    record = {"warnings": []}
    decorated = bench_cli._decorate_run(record, planned, planned.case,
                                        backend={"status": "stub"},
                                        storage_before={"x": None},
                                        storage_after={"x": None})
    assert decorated["declared_storage"] == {
        "input_media": "nvme", "cache_media": None, "kind": "declared"}
    assert decorated["config"]["input_count"] == 2
    assert decorated["config"]["input_list_sha1"]
    assert decorated["backend"] == {"status": "stub"}


def test_decorate_run_marks_undeclared_storage(tmp_path):
    planned = _planned(tmp_path)
    decorated = bench_cli._decorate_run({}, planned, planned.case,
                                        backend={}, storage_before={},
                                        storage_after={})
    assert decorated["declared_storage"]["kind"] == "undeclared"


def test_summary_rows_read_run_level_config(tmp_path):
    planned = _planned(tmp_path, case_overrides={
        "labels": ["selftest"]})
    record = bench_cli._invalid_record(planned.case, 1, ["boom"])
    payload = {"results": [{"case_id": planned.case.id, "runs": [record]}]}
    rows = bench_cli._summary_rows(payload)
    assert rows[0]["case_id"] == "case_a"
    assert rows[0]["status"] == "invalid_input"


def test_report_headline_and_csv_use_only_primary_resource_metrics():
    record = {
        "status": "success", "wall_seconds": 10, "phases": {},
        "resources": {
            "cpu_percent_avg": 400, "cpu_percent_avg_normalized": 25,
            "cpu_time_seconds": 40, "rss_peak_bytes": 1024,
            "swap_used_delta_peak_bytes": 500,
        },
        "config": {},
    }
    headline = bench_cli._case_headline(record)
    assert "cpu_avg_all_cores=25" in headline
    assert "rss_peak=" in headline
    assert "swap" not in headline
    rows = bench_cli._summary_rows({"results": [{"runs": [record]}]})
    assert set(rows[0]) == set(bench_cli.SUMMARY_COLUMNS)
    assert "swap_used_delta_peak_bytes" not in rows[0]
