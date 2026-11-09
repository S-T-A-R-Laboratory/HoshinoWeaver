import subprocess
import sys

from ui.runtime_capabilities_format import format_runtime_capabilities_html


def test_capability_formatter_import_is_qt_free():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import ui.runtime_capabilities_format; "
                "assert not any(name.startswith('PySide6') for name in sys.modules)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_capability_dialog_shows_native_import_failure_reason():
    report = {
        "compiled_custom_ops": {
            "status": "unavailable",
            "reason": "ImportError: DLL initialization routine failed",
        },
        "cuda_runtime": {"status": "not_built"},
        "metal_runtime": {"status": "unavailable"},
        "logical_ops": {},
        "modules": {},
    }

    text = format_runtime_capabilities_html(report)

    assert "原生算子不可用" in text
    assert "DLL initialization routine failed" in text


def test_external_tool_details_are_escaped_and_distinguish_configured_from_verified():
    report = {
        'compiled_custom_ops': {}, 'cuda_runtime': {}, 'metal_runtime': {},
        'logical_ops': {}, 'modules': {},
        'external_components': {
            'ffmpeg': {'status': 'unavailable', 'reason': '<bad executable>'},
            'photoshop': {'status': 'configured', 'path': 'C:/Apps/<Photoshop>.exe'},
        },
    }
    text = format_runtime_capabilities_html(report)
    assert '&lt;bad executable&gt;' in text and '<bad executable>' not in text
    assert '&lt;Photoshop&gt;' in text


def test_external_probe_runs_off_gui_thread_and_finishes_after_dialog_close(monkeypatch):
    import threading
    import time

    import pytest
    pytest.importorskip('PySide6')
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtWidgets import QApplication
    from PySide6.QtTest import QTest
    from ui import runtime_capabilities_dialog as dialog_module

    app = QApplication.instance() or QApplication([])
    report = dict(compiled_custom_ops={}, cuda_runtime={}, metal_runtime={}, logical_ops={}, modules={})
    monkeypatch.setattr(dialog_module, 'get_runtime_components_report', lambda **kwargs: report)
    monkeypatch.setattr(dialog_module, '_load_default_settings', lambda: {'ffmpeg_path': 'selected'})
    entered, release = threading.Event(), threading.Event()
    worker_threads = []
    def probe(configs):
        assert configs['ffmpeg_path'] == 'selected'
        worker_threads.append(threading.get_ident())
        entered.set()
        assert release.wait(5)
        return {'ffmpeg': {'status': 'available'}, 'photoshop': {'status': 'unavailable'}}
    monkeypatch.setattr(dialog_module, 'probe_external_components', probe)
    dialog = dialog_module.RuntimeCapabilitiesDialog()
    try:
        assert entered.wait(2)
        assert not dialog.refresh_button.isEnabled()
        assert worker_threads != [threading.get_ident()]
        dialog.close()
    finally:
        release.set()
    deadline = time.monotonic() + 3
    while not dialog.refresh_button.isEnabled() and time.monotonic() < deadline:
        QTest.qWait(10)
    assert dialog.refresh_button.isEnabled()
    assert 'external_components' not in report  # Startup cache is not mutated.
    assert dialog._report['external_components']['ffmpeg']['status'] == 'available'
    dialog.deleteLater()
    app.processEvents()
