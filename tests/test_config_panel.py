from pathlib import Path

import numpy as np
import pytest
import tifffile

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import (
    QAbstractSpinBox, QApplication, QComboBox, QMainWindow, QStyle,
    QStyleOptionGroupBox,
)

from ui.output_panel import OutputPanel
from ui.panel_builder import DynamicConfigPanel, PanelSchema
from ui.widgets import ConfigSpec


@pytest.fixture(scope="module")
def qt_app():
    import os

    previous = os.environ.get("QT_QPA_PLATFORM")
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    app = QApplication.instance() or QApplication([])
    yield app
    if previous is None:
        os.environ.pop("QT_QPA_PLATFORM", None)
    else:
        os.environ["QT_QPA_PLATFORM"] = previous


@pytest.mark.asyncio
@pytest.mark.parametrize("align_mode", ["simple", "bundle"])
@pytest.mark.parametrize("enabled,override,expected_mode,expected_weight", [
    (True, {}, "native_relaxed", False),
    (False, {}, "auto", True),
    (True, {"star_detection_mode": "auto", "int_weight": True}, "auto", True),
])
async def test_hidden_configs_use_backend_defaults_like_cli(
        qt_app, tmp_path, monkeypatch, align_mode, enabled, override,
        expected_mode, expected_weight):
    import hoshicore.engine.wiring as wiring

    pipeline = Path(__file__).resolve().parents[1] / "hoshicore/dag/aligned_stack.meta.yaml"
    schema = PanelSchema.from_yaml(pipeline, pipeline.with_name("aligned_stack.ui.yaml"))
    schema.routes["align_mode"].default = align_mode
    panel = DynamicConfigPanel()
    panel.load_schema(schema)
    gui_configs = panel.collect_configs()
    hidden_keys = {spec.key for spec in schema.configs if spec.hidden}
    assert hidden_keys.isdisjoint(gui_configs)
    # Visible controls still submit values, even without a user edit.
    assert gui_configs["enable_ground"] is True

    raw_settings = {
        "star_detection_mode": {"enabled": enabled, "value": "native_relaxed"},
        "int_weight": {"enabled": enabled, "value": False},
        "star_detection_recovery": {"enabled": True, "value": False},
    }
    monkeypatch.setattr(wiring, "_read_raw_settings", lambda path=None: raw_settings)
    captured = []

    async def capture_run(dag, inputs, configs, *args, **kwargs):
        captured.append(configs.copy())
        return {}

    monkeypatch.setattr(wiring, "run_dag", capture_run)
    frame = tmp_path / "frame.tif"
    tifffile.imwrite(frame, np.zeros((10, 10, 3), dtype=np.uint16))
    output = {"output_filename": str(tmp_path / "output.tif")}
    routes = panel.collect_route_choices()
    # Exercise the shared GUI/CLI entry point; only image processing is stubbed.
    for configs in ({**gui_configs, **output, **override}, {**output, **override}):
        await wiring.run_from_yaml(
            str(pipeline), {"fnames": [str(frame)]}, configs,
            route_choices=routes, progress=False)

    for configs in captured:
        assert configs["star_detection_mode"] == expected_mode
        assert configs["int_weight"] is expected_weight
        assert configs["star_detection_recovery"] is False
    assert {key: captured[0][key] for key in hidden_keys} == {
        key: captured[1][key] for key in hidden_keys}
    panel.deleteLater()


def test_output_panel_submits_hidden_output_configs(qt_app):
    pipeline = Path(__file__).resolve().parents[1] / "hoshicore/dag/aligned_stack.meta.yaml"
    schema = PanelSchema.from_yaml(pipeline, pipeline.with_name("aligned_stack.ui.yaml"))
    panel = DynamicConfigPanel()
    panel.load_schema(schema)
    output = OutputPanel()
    output.load_specs(schema.outputs)
    output.apply_defaults({"output_format": "PNG", "output_dtype": "uint16",
                           "png_compressing": 3})
    output._regions[0]._path_edit.setText("output.png")
    configs = panel.collect_configs()
    assert "output_filename" not in configs
    assert "output_dtype" not in configs
    assert "png_compressing" not in configs
    configs.update(output.collect_outputs())
    assert configs["output_filename"] == "output.png"
    assert configs["output_dtype"] == "uint16"
    assert configs["png_compressing"] == 3
    panel.deleteLater()
    output.deleteLater()


def test_visible_when_does_not_omit_config(qt_app):
    schema = PanelSchema(meta_yaml_path="unused", configs=[
        ConfigSpec(key="show_option", type="bool", widget="switch", default=False),
        ConfigSpec(key="option", type="bool", widget="switch", default=True,
                   visible_when={"key": "show_option", "eq": True}),
    ])
    panel = DynamicConfigPanel()
    panel.load_schema(schema)
    assert panel._config_widgets["option"].isHidden()
    assert panel.collect_configs() == {"show_option": False, "option": True}
    panel.deleteLater()


@pytest.mark.parametrize("workflow", ["timelapse_startrail", "norma_bundle_window"])
def test_optional_video_output_panel(qt_app, workflow):
    pipeline = Path(__file__).resolve().parents[1] / "hoshicore/dag" / f"{workflow}.meta.yaml"
    schema = PanelSchema.from_yaml(pipeline, pipeline.with_name(f"{workflow}.ui.yaml"))
    assert len(schema.outputs) == 2
    panel = OutputPanel()
    panel.load_specs(schema.outputs)
    changes = QSignalSpy(panel.values_changed)
    assert not panel.is_ready()  # default image output needs its directory
    panel._enable_groups["enable_sequence_export"].setChecked(False)
    assert changes.count() == 1
    assert "至少" in panel.missing_reason()
    panel._enable_groups["enable_video_export"].setChecked(True)
    assert changes.count() == 2
    video = panel._regions[1]
    assert not panel.is_ready()
    video._path_edit.setText("video.mp4")
    assert changes.count() == 3
    assert panel.is_ready()
    values = panel.collect_outputs()
    assert values["enable_sequence_export"] is False
    assert values["enable_video_export"] is True
    assert values["video_fps"] == 25
    assert values["video_chroma"] == "420"
    assert "output_dir" not in values
    assert video._rows["video_bitrate_kbps"].isHidden()
    mode = video._rows["video_rate_control"].findChild(QComboBox)
    mode.setCurrentIndex(mode.findData("bitrate"))
    assert video._rows["video_crf"].isHidden()
    assert not video._rows["video_bitrate_kbps"].isHidden()
    panel.set_video_result({"path": str(Path("video.mp4").absolute())})
    assert not video._open_button.isHidden()
    panel.set_video_result(None)
    assert video._open_button.isHidden()
    panel.deleteLater()


@pytest.mark.parametrize("workflow", ["timelapse_startrail", "norma_bundle_window"])
def test_video_container_selection_updates_paths_and_preserves_encoder_values(qt_app, workflow, monkeypatch):
    from ui.output_presets import VIDEO_FORMAT_PRESETS

    pipeline = Path(__file__).resolve().parents[1] / "hoshicore/dag" / f"{workflow}.meta.yaml"
    schema = PanelSchema.from_yaml(pipeline, pipeline.with_name(f"{workflow}.ui.yaml"))
    panel = OutputPanel()
    panel.load_specs(schema.outputs)
    panel._enable_groups["enable_sequence_export"].setChecked(False)
    panel._enable_groups["enable_video_export"].setChecked(True)
    video = panel._regions[1]
    video._path_edit.setText("first.mp4")
    fps = video._rows["video_fps"].findChild(QAbstractSpinBox)
    fps.setValue(30)
    video._format_combo.setCurrentText("AVI")
    assert video._path_edit.text() == "first.avi"
    assert panel.is_ready()
    assert panel.collect_outputs()["video_fps"] == 30
    assert all(row.isEnabled() for row in video._rows.values())
    video._path_edit.setText("second.AVI")
    video._format_combo.setCurrentText("MP4")
    assert video._path_edit.text() == "first.mp4"
    video._format_combo.setCurrentText("AVI")
    assert video._path_edit.text() == "second.AVI"
    video._path_edit.setText("manual.mp4")
    assert video._format_combo.currentText() == "MP4"

    monkeypatch.setattr("ui.output_panel.QFileDialog.getSaveFileName",
                        lambda *args: ("chosen.mp4", "AVI 视频 (*.avi)"))
    video._browse()
    assert video._path_edit.text() == "chosen.avi"
    assert video._format_combo.currentText() == "AVI"
    video._path_edit.setText("invalid.mov")
    assert not panel.is_ready()
    assert ".avi" in panel.missing_reason()

    # Format presets control availability and submission together, even when
    # an encoder preset added later supports fewer controls than H.264 does.
    monkeypatch.setitem(VIDEO_FORMAT_PRESETS, "AVI", {
        **VIDEO_FORMAT_PRESETS["AVI"],
        "params": VIDEO_FORMAT_PRESETS["AVI"]["params"] - {"video_crf"},
    })
    video._format_combo.setCurrentText("MP4")
    video._format_combo.setCurrentText("AVI")
    assert not video._rows["video_crf"].isEnabled()
    assert "video_crf" not in video.collect()
    panel.deleteLater()


@pytest.mark.parametrize("workflow", ["timelapse_startrail", "norma_bundle_window"])
def test_uncompressed_avi_controls_follow_codec_capabilities(qt_app, workflow):
    from hoshicore.component.video_io import VideoExportOptions

    pipeline = Path(__file__).resolve().parents[1] / "hoshicore/dag" / f"{workflow}.meta.yaml"
    schema = PanelSchema.from_yaml(pipeline, pipeline.with_name(f"{workflow}.ui.yaml"))
    panel = OutputPanel()
    panel.load_specs(schema.outputs)
    video = panel._regions[1]
    codec = video._rows["video_codec"].findChild(QComboBox)
    assert codec.findData("raw_rgb") == -1
    crf = video._rows["video_crf"].findChild(QAbstractSpinBox)
    crf.setValue(23)
    video._format_combo.setCurrentText("AVI")
    codec.setCurrentIndex(codec.findData("raw_rgb"))
    video._path_edit.setText("output.avi")
    unsupported = {"video_chroma", "video_rate_control", "video_crf",
                   "video_bitrate_kbps", "video_preset", "video_color_range"}
    values = video.collect()
    assert values["video_codec"] == "raw_rgb"
    assert unsupported.isdisjoint(values)
    for key in unsupported:
        assert video._rows[key].isHidden()
        assert not video._rows[key].isEnabled()

    # The submitted GUI values resolve to the same backend codec and defaults.
    options = VideoExportOptions.from_mapping({
        key.removeprefix("video_"): value for key, value in values.items()
    })
    assert options.pixel_format == "bgr24" and options.color_range == "full"
    video._format_combo.setCurrentText("MP4")
    assert codec.currentData() == "h264"
    assert codec.findData("raw_rgb") == -1
    assert not video._rows["video_crf"].isHidden()
    assert video.collect()["video_crf"] == 23
    video._format_combo.setCurrentText("AVI")
    assert codec.currentData() == "raw_rgb"
    codec.setCurrentIndex(codec.findData("h264"))
    assert video.collect()["video_crf"] == 23
    panel.deleteLater()


@pytest.mark.parametrize("workflow", ["timelapse_startrail", "norma_bundle_window"])
@pytest.mark.parametrize("sidebar_width", [220, 250])
@pytest.mark.parametrize("codec", ["h264", "raw_rgb"])
def test_output_layout_in_desktop_sidebar(qt_app, workflow, sidebar_width, codec):
    import runpy

    from ui.UI import Ui_HNW

    root = Path(__file__).resolve().parents[1]
    desktop = runpy.run_path(str(root / "HoshinoWeaver desktop.py"), run_name="layout_test")

    class Host(QMainWindow, Ui_HNW):
        _setup_output_panel = desktop["HNW_window"]._setup_output_panel

    # Exercise the real tab's inherited styles and scroll area, without
    # starting the application's settings, task handlers or background work.
    host = Host()
    host.setupUi(host)
    host._setup_output_panel()
    host.args_area.setFixedWidth(sidebar_width)
    pipeline = root / "hoshicore/dag" / f"{workflow}.meta.yaml"
    schema = PanelSchema.from_yaml(pipeline, pipeline.with_name(f"{workflow}.ui.yaml"))
    panel = host.output_panel
    panel.load_specs(schema.outputs)
    panel._enable_groups["enable_video_export"].setChecked(True)
    if codec == "raw_rgb":
        video = panel._regions[1]
        video._format_combo.setCurrentText("AVI")
        combo = video._rows["video_codec"].findChild(QComboBox)
        combo.setCurrentIndex(combo.findData(codec))
    host.star_trail_option_box.setCurrentWidget(host.star_trail_output)
    host.show()
    qt_app.processEvents()
    try:
        assert panel.minimumSizeHint().width() <= panel.width()
        for region in panel._regions:
            group = panel._enable_groups[region.spec.enable_key]
            assert group.font().bold()
            assert not region.font().bold()
            option = QStyleOptionGroupBox()
            group.initStyleOption(option)
            label_rect = group.style().subControlRect(
                QStyle.CC_GroupBox, option, QStyle.SC_GroupBoxLabel, group)
            checkbox_rect = group.style().subControlRect(
                QStyle.CC_GroupBox, option, QStyle.SC_GroupBoxCheckBox, group)
            sample_point = checkbox_rect.topLeft()
            sample_point.setX(sample_point.x() + 4)
            sample_point.setY(sample_point.y() + 4)
            selected_color = group.grab().toImage().pixelColor(sample_point)
            assert selected_color.blue() > selected_color.red() + 40
            QTest.mouseClick(group, Qt.LeftButton, pos=label_rect.center())
            qt_app.processEvents()
            assert not group.isChecked()
            assert region.isHidden()
            unselected_color = group.grab().toImage().pixelColor(sample_point)
            assert unselected_color.red() > 180
            QTest.mouseClick(group, Qt.LeftButton, pos=label_rect.center())
            qt_app.processEvents()
            assert group.isChecked()
            assert not region.isHidden()
            for control in (QStyle.SC_GroupBoxLabel, QStyle.SC_GroupBoxCheckBox):
                title_rect = group.style().subControlRect(QStyle.CC_GroupBox, option, control, group)
                assert not title_rect.intersects(region.geometry())
            assert group.rect().contains(region.geometry())
            for combo in region.findChildren(QComboBox):
                if not combo.isVisible():
                    continue
                assert combo.width() > 50
                assert combo.parentWidget().rect().contains(combo.geometry())

        video = panel._regions[1]
        fps = video._rows["video_fps"].findChild(QAbstractSpinBox)
        assert fps.buttonSymbols() == QAbstractSpinBox.NoButtons
        previous = panel.collect_outputs()["video_fps"]
        fps.setFocus()
        QTest.keyClick(fps, Qt.Key_Up)
        assert panel.collect_outputs()["video_fps"] == previous + 1

        group = panel._enable_groups["enable_video_export"]
        group.setChecked(False)
        qt_app.processEvents()
        assert video.isHidden()
        group.setChecked(True)
        qt_app.processEvents()
        assert not video.isHidden()
        assert panel.collect_outputs()["video_fps"] == previous + 1
    finally:
        host.close()
        host.deleteLater()
