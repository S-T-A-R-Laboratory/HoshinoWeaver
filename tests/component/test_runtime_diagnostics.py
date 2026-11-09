from unittest.mock import Mock

from hoshicore.component import runtime_diagnostics


def test_runtime_component_probe_has_expected_sections():
    report = runtime_diagnostics.probe_runtime_components()

    assert set(report) == {
        "compiled_custom_ops", "cuda_runtime", "metal_runtime",
        "logical_ops", "modules",
    }
    assert report["compiled_custom_ops"]["status"] in {
        "available", "unavailable", "error",
    }
    assert "turbojpeg" in report["modules"]
    assert "pyexiv2" in report["modules"]
    assert report["logical_ops"]


def test_runtime_component_logging_runs_only_once(monkeypatch):
    report = {
        "compiled_custom_ops": {
            "status": "available", "compiler": "test", "openmp": True,
            "cuda": False,
        },
        "cuda_runtime": {
            "status": "not_built", "reason": "not built",
        },
        "metal_runtime": {
            "status": "unavailable", "reason": "not present",
        },
        "logical_ops": {
            "max_combine": {"backend": "openmp_cpu", "native": True,
                            "reason": None},
        },
        "modules": {
            "turbojpeg": {"status": "available"},
        },
    }
    monkeypatch.setattr(runtime_diagnostics, "_logged", False)
    monkeypatch.setattr(runtime_diagnostics, "_last_report", None)
    probe = Mock(return_value=report)
    monkeypatch.setattr(runtime_diagnostics, "probe_runtime_components", probe)
    logger = Mock()

    assert runtime_diagnostics.log_runtime_components(logger) is report
    assert runtime_diagnostics.log_runtime_components(logger) is None

    probe.assert_called_once_with()


def test_probe_reports_numpy_routes_when_compiled_module_is_absent(monkeypatch):
    monkeypatch.setattr(
        runtime_diagnostics, "load_compiled_module",
        lambda: (None, "module missing"))
    monkeypatch.setattr(
        runtime_diagnostics, "metal_device_info",
        lambda: {"available": False, "status": "unavailable"})
    monkeypatch.setattr(
        runtime_diagnostics, "_module_probe",
        lambda *args, **kwargs: {"status": "unavailable"})

    report = runtime_diagnostics.probe_runtime_components()

    assert report["compiled_custom_ops"]["status"] == "unavailable"
    assert report["cuda_runtime"]["status"] == "not_built"
    assert {item["backend"] for item in report["logical_ops"].values()} == {
        "numpy",
    }


def test_external_tools_use_configured_paths_and_report_partial_ffmpeg_support(monkeypatch, tmp_path):
    from hoshicore.component import photoshop, video_io

    app = tmp_path / 'Photoshop.exe'
    app.touch()
    monkeypatch.setattr(runtime_diagnostics.sys, 'platform', 'win32')
    resolve = Mock(return_value='configured-ffmpeg')
    monkeypatch.setattr(video_io, 'resolve_ffmpeg', resolve)
    probe = Mock(side_effect=[ValueError('libx264 missing'), ValueError('libx264 missing'), 'configured-ffmpeg'])
    monkeypatch.setattr(video_io, 'probe_ffmpeg', probe)
    renderer = Mock()
    monkeypatch.setattr(photoshop, 'PhotoshopRawRenderer', renderer)
    result = runtime_diagnostics.probe_external_components({'ffmpeg_path': 'custom-ffmpeg', 'photoshop_path': str(app)})
    resolve.assert_called_once_with('custom-ffmpeg')
    assert result['ffmpeg']['status'] == 'available'
    assert result['ffmpeg']['formats']['RGB / AVI']['status'] == 'available'
    assert result['ffmpeg']['formats']['H.264 / MP4']['reason'] == 'libx264 missing'
    assert result['photoshop'] == {'status': 'configured', 'path': str(app.resolve())}
    renderer.assert_not_called()


def test_missing_ffmpeg_does_not_hide_photoshop_configuration_error(monkeypatch):
    from hoshicore.component import video_io

    monkeypatch.setattr(video_io, 'resolve_ffmpeg', Mock(side_effect=ValueError('ffmpeg missing')))
    result = runtime_diagnostics.probe_external_components({})
    assert result['ffmpeg']['reason'] == 'ffmpeg missing'
    assert result['photoshop']['status'] == 'unavailable'
    assert result['photoshop']['reason']
