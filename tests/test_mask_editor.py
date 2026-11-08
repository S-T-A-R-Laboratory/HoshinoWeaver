"""Mask editing behavior without model weights or external image datasets."""
import asyncio
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QApplication, QMessageBox, QPushButton, QWidget

from ui.mask_canvas import MaskCanvas, image_rgb
from ui.mask_document import MaskDocument
from ui.mask_editor import MaskEditor, MaskEditorController, load_mask, load_mask_document, save_mask
from ui.mask_target import MaskTarget
from ui.panel_builder import DynamicConfigPanel, PanelSchema
from ui.widgets import ConfigSpec, RouteOptionSpec, RouteSpec


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


def reference(width=121, height=83):
    image = QImage(width, height, QImage.Format_RGB888)
    image.fill(QColor(30, 60, 100))
    return image


def stroke(doc, start=(4.5, 4.5), end=(20.5, 4.5), diameter=3, value=True):
    doc.begin_stroke()
    doc.paint_segment(start, end, diameter, value)
    doc.end_stroke()


def test_continuous_stroke_clipping_and_history():
    doc = MaskDocument(np.zeros((512, 512), np.bool_))
    stroke(doc, (-10.5, 4.5), (30.5, 4.5))
    assert doc.mask[4, :31].all()
    assert not doc.mask[12:].any()
    assert len(doc.undo_stack) == 1
    # Local history, not a full-image snapshot for each mouse event.
    assert doc.undo_stack[0].nbytes == 2 * 128 * 128
    original = doc.mask.copy()
    doc.mark_saved("mask.png")
    assert not doc.dirty
    doc.undo()
    assert not doc.mask.any() and doc.dirty
    doc.redo()
    np.testing.assert_array_equal(doc.mask, original)
    assert not doc.dirty
    stroke(doc, value=True)  # no change does not add history
    assert len(doc.undo_stack) == 1
    doc.undo()
    stroke(doc, (40.5, 40.5), (40.5, 40.5))
    assert not doc.redo_stack
    doc.invert()
    doc.undo()
    assert doc.mask[40, 40]


def test_history_budget_and_non_aliasing():
    initial = np.zeros((256, 256), np.bool_)
    doc = MaskDocument(initial, history_limit=32768)
    stroke(doc)
    stroke(doc, (140.5, 140.5), (150.5, 140.5))
    assert len(doc.undo_stack) == 1
    assert not initial.any()
    doc.undo()
    assert doc.mask[4, 4] and not doc.mask[140, 140]


def test_discard_restores_checkpoint_after_history_eviction():
    doc = MaskDocument(np.zeros((256, 256), np.bool_), history_limit=32768)
    stroke(doc)
    saved = doc.mask.copy()
    doc.mark_saved("mask.png")
    stroke(doc, (140.5, 140.5), (150.5, 140.5))
    doc.invert()
    assert doc.dirty and len(doc.undo_stack) == 1
    doc.discard_changes()
    np.testing.assert_array_equal(doc.mask, saved)
    assert not doc.dirty and not doc.undo_stack and not doc.redo_stack


def test_png_roundtrip_and_validation(qt_app, tmp_path):
    mask = np.indices((13, 17)).sum(axis=0) % 2 == 0
    path = tmp_path / "天空.png"
    save_mask(path, mask)
    loaded = load_mask(path)
    assert loaded.dtype == np.bool_
    np.testing.assert_array_equal(loaded, mask)
    image = reference()
    image.save(str(tmp_path / "color.png"))
    assert not load_mask(tmp_path / "color.png").any()
    gray = QImage(11, 9, QImage.Format_Grayscale8)
    gray.fill(QColor(127, 127, 127))
    gray.save(str(tmp_path / "gray.png"))
    assert not load_mask(tmp_path / "gray.png").any()
    transparent = QImage(11, 9, QImage.Format_ARGB32)
    transparent.fill(QColor(255, 255, 255, 0))
    transparent.save(str(tmp_path / "transparent.png"))
    with pytest.raises(ValueError, match="透明"):
        load_mask(tmp_path / "transparent.png")
    with pytest.raises(ValueError):
        load_mask(tmp_path / "missing.png")


def test_gray_threshold_and_jpeg_conversion(qt_app, tmp_path):
    pixels = np.array([[0, 127, 128, 255]], np.uint8)
    image = QImage(pixels.data, 4, 1, 4, QImage.Format_Grayscale8)
    image.save(str(tmp_path / "gray.png"))
    doc = load_mask_document(tmp_path / "gray.png")
    np.testing.assert_array_equal(doc.mask, [[False, False, True, True]])
    assert doc.dirty and not doc.path
    # JPEG compression may introduce intermediate gray levels near boundaries.
    pixels = np.zeros((48, 64), np.uint8)
    pixels[:, 29:] = 255
    image = QImage(pixels.data, 64, 48, 64, QImage.Format_Grayscale8)
    path = tmp_path / "天空.jpeg"
    assert image.save(str(path), "JPEG", 80)
    original = path.read_bytes()
    doc = load_mask_document(path)
    assert doc.mask.dtype == np.bool_
    assert not doc.mask[:, :20].any() and doc.mask[:, 40:].all()
    assert doc.dirty and not doc.path
    assert doc.suggested_path == str(path.with_suffix(".png"))
    editor = MaskEditor()
    applied = []
    target = MaskTarget("天空蒙版", str(path), lambda: True, applied.append)
    assert editor.open_session(reference(64, 48), "reference.jpg", target)
    output = tmp_path / "converted.png"
    from unittest.mock import patch
    with patch("ui.mask_editor.QFileDialog.getSaveFileName", return_value=(str(output), "")):
        editor.save_and_apply()
    assert applied == [str(output)] and path.read_bytes() == original
    np.testing.assert_array_equal(load_mask(output), editor.document.mask)
    editor.close()


def test_canvas_coordinates_stroke_and_render(qt_app):
    canvas = MaskCanvas()
    canvas.resize(800, 600)
    doc = MaskDocument(np.zeros((100, 120), np.bool_))
    canvas.set_document(doc, reference(120, 100))
    canvas.show()
    qt_app.processEvents()
    fit_center = canvas.image_point(QPointF(canvas.width() / 2, canvas.height() / 2))
    assert fit_center == QPointF(60, 50)
    canvas.scale = 3.
    canvas._fit_view = False
    canvas.offset = QPointF(51, 37)
    canvas.diameter = 3
    start = (QPointF(10.5, 20.5) * canvas.scale + canvas.offset).toPoint()
    end = (QPointF(40.5, 20.5) * canvas.scale + canvas.offset).toPoint()
    QTest.mousePress(canvas, Qt.LeftButton, pos=start)
    QTest.mouseMove(canvas, end)
    QTest.mouseRelease(canvas, Qt.LeftButton, pos=end)
    assert doc.mask[20, 11:40].all()
    assert not doc.mask[:15].any()
    assert len(doc.undo_stack) == 1
    canvas.mode = "mask"
    rendered = canvas.grab().toImage()
    assert rendered.pixelColor(end).red() == 255
    doc.undo()
    canvas.sync()
    assert canvas.grab().toImage().pixelColor(end).red() == 0
    canvas.close()


def test_right_drag_pans_without_painting(qt_app):
    canvas = MaskCanvas()
    canvas.resize(800, 600)
    doc = MaskDocument(np.zeros((100, 120), np.bool_))
    canvas.set_document(doc, reference(120, 100))
    canvas.show()
    qt_app.processEvents()
    initial_offset = QPointF(canvas.offset)
    QTest.mousePress(canvas, Qt.RightButton, pos=QPoint(300, 200))
    QTest.mouseMove(canvas, QPoint(350, 230))
    QTest.mouseRelease(canvas, Qt.RightButton, pos=QPoint(350, 230))
    assert canvas.offset == initial_offset + QPointF(50, 30)
    assert not doc.mask.any() and not doc.dirty
    assert not canvas._panning
    canvas.close()


def test_relative_brush_slider_and_screen_lock(qt_app):
    editor = MaskEditor()
    editor.open_session(reference(1000, 800), "small.jpg")
    first = editor.canvas.diameter
    assert first == pytest.approx(20, rel=.02)
    editor.open_session(reference(2000, 1600), "large.jpg")
    assert editor.canvas.diameter == pytest.approx(first * 2)
    editor.brush_size.setValue(editor.brush_size.value() + 50)
    assert editor.canvas.diameter > first * 2
    before_visual = editor.canvas.effective_diameter * editor.canvas.scale
    editor.screen_brush.setChecked(True)
    assert editor.canvas.effective_diameter * editor.canvas.scale == pytest.approx(before_visual, rel=.02)
    before_pixels = editor.canvas.effective_diameter
    editor.canvas.scale *= 2
    assert editor.canvas.effective_diameter == pytest.approx(before_pixels / 2)
    editor.screen_brush.setChecked(False)
    fixed_pixels = editor.canvas.effective_diameter
    editor.canvas.scale *= 2
    assert editor.canvas.effective_diameter == fixed_pixels
    editor.close()


def test_screen_fixed_brush_paints_fewer_image_pixels_after_zoom(qt_app):
    canvas = MaskCanvas()
    canvas.resize(800, 600)
    doc = MaskDocument(np.zeros((160, 200), np.bool_))
    canvas.set_document(doc, reference(200, 160))
    canvas.show()
    qt_app.processEvents()
    canvas.screen_brush = True
    canvas.diameter = 20
    canvas._fit_view = False
    canvas.offset = QPointF()
    counts = []
    for zoom in (2., 4.):
        canvas.scale = zoom
        point = QPointF(40.5, 40.5) * zoom
        QTest.mouseClick(canvas, Qt.LeftButton, pos=point.toPoint())
        counts.append(np.count_nonzero(doc.mask))
        doc.undo()
        canvas.sync()
    assert counts[0] > counts[1] * 3
    canvas.close()


def test_qt_rgb_copy_respects_stride(qt_app):
    image = reference(7, 5)
    rgb = image_rgb(image)
    assert rgb.shape == (5, 7, 3) and rgb.flags.c_contiguous
    np.testing.assert_array_equal(rgb[0, 0], [30, 60, 100])
    image.fill(QColor("red"))
    np.testing.assert_array_equal(rgb[0, 0], [30, 60, 100])


def mask_schema():
    return PanelSchema("unused", configs=[
        ConfigSpec("mask", widget="mask_picker", label="天空蒙版"),
        ConfigSpec("ordinary_file", widget="file_picker"),
    ], routes={"method": RouteSpec("method", options={
        "a": RouteOptionSpec("a"), "b": RouteOptionSpec("b")}, default="a")},
        route_configs={"method": {"a": [ConfigSpec("mask", widget="mask_picker")], "b": []}})


def test_explicit_mask_controls_and_revocable_targets(qt_app):
    panel = DynamicConfigPanel()
    panel.load_schema(mask_schema())
    spy = QSignalSpy(panel.mask_edit_requested)
    changes = QSignalSpy(panel.values_changed)
    assert len(panel.findChildren(QPushButton, "edit_mask")) == 2
    panel._config_widgets["mask"].findChild(QPushButton, "edit_mask").click()
    root = spy.at(0)[0]
    assert root.apply("saved.png")
    assert panel.collect_configs()["mask"] == "saved.png"
    assert changes.count() == 1
    route_button = panel._route_config_container["method"].findChild(QPushButton, "edit_mask")
    route_button.click()
    route = spy.at(1)[0]
    assert route.apply("route.png")
    assert panel.collect_configs()["method.a.mask"] == "route.png"
    panel._on_route_changed("method", "b")
    assert not route.apply("stale.png")
    assert root.is_current()
    panel.setEnabled(False)
    assert not root.apply("during-processing.png")
    panel.setEnabled(True)
    panel.load_schema(mask_schema())
    assert not root.apply("wrong-workflow.png")
    assert panel.collect_configs()["mask"] is None
    panel.deleteLater()


def test_save_before_apply_and_stale_target(qt_app, tmp_path, monkeypatch):
    editor = MaskEditor()
    paths = []
    current = [True]
    target = MaskTarget("天空蒙版", "", lambda: current[0], paths.append)
    editor.open_session(reference(), "reference.jpg", target)
    path = tmp_path / "saved.png"
    monkeypatch.setattr("ui.mask_editor.QFileDialog.getSaveFileName", lambda *a: (str(path), ""))
    stroke(editor.document)
    editor.save_and_apply()
    assert paths == [str(path)] and path.is_file()
    assert not editor.document.dirty
    current[0] = False
    editor.refresh()
    assert not editor.apply_button.isEnabled()
    editor.save_and_apply()
    assert len(paths) == 1
    # A failed write cannot mark clean or apply a new path.
    current[0] = True
    stroke(editor.document, (60.5, 60.5), (61.5, 60.5))
    errors = []
    monkeypatch.setattr(editor, "error", errors.append)
    def fail(*args):
        raise OSError("disk full")
    monkeypatch.setattr("ui.mask_editor.save_mask", fail)
    editor.save_and_apply()
    assert editor.document.dirty and errors and len(paths) == 1
    monkeypatch.setattr(editor, "confirm_discard", lambda: True)
    editor.close()


def test_reference_snapshot_and_controller_reuse(qt_app, monkeypatch):
    parent = QWidget()
    parent.img_view_label = SimpleNamespace(pixmap=QPixmap.fromImage(reference()), img_path="first.jpg")
    controller = MaskEditorController(parent)
    controller.open()
    editor = controller.editor
    qt_app.processEvents()
    parent.img_view_label.pixmap = QPixmap.fromImage(reference(33, 44))
    parent.img_view_label.img_path = "second.jpg"
    controller.open()
    assert controller.editor is editor
    assert editor.source == "second.jpg" and editor.document.mask.shape == (44, 33)
    assert not hasattr(editor, 'reference_button')
    assert editor.apply_button.isHidden()
    assert controller.close_for_parent()
    parent.close()


def test_discard_is_not_asked_again_after_reopening(qt_app, monkeypatch):
    parent = QWidget()
    parent.img_view_label = SimpleNamespace(pixmap=QPixmap.fromImage(reference()), img_path="first.jpg")
    controller = MaskEditorController(parent)
    controller.open()
    editor = controller.editor
    stroke(editor.document)
    questions = []
    def discard(*args):
        questions.append(True)
        return QMessageBox.Discard
    monkeypatch.setattr('ui.mask_editor.QMessageBox.question', discard)
    assert editor.close()
    assert not editor.document.dirty and not editor.document.mask.any()
    parent.img_view_label.img_path = "second.jpg"
    controller.open()
    assert len(questions) == 1 and editor.source == "second.jpg"
    assert not editor.document.dirty
    editor.close()
    parent.close()


def test_failed_preview_load_preserves_reference_identity(qt_app, tmp_path):
    from ui.UILibs import imgDisplayQFrame
    preview = imgDisplayQFrame()
    path = tmp_path / "valid.png"
    reference().save(str(path))
    preview.initImg(str(path))
    with pytest.raises(FileNotFoundError):
        preview.initImg(str(tmp_path / "missing.png"))
    assert preview.img_path == str(path)
    assert preview.pixmap.size() == reference().size()
    preview.close()


def test_replacement_size_mismatch_preserves_document(qt_app, tmp_path, monkeypatch):
    editor = MaskEditor()
    editor.open_session(reference(), "original.jpg")
    doc = editor.document
    stroke(doc)
    path = tmp_path / "mismatch.png"
    save_mask(path, np.zeros((10, 10), np.bool_))
    errors = []
    monkeypatch.setattr(editor, "error", errors.append)
    monkeypatch.setattr("ui.mask_editor.QFileDialog.getOpenFileName", lambda *a: (str(path), ""))
    editor.choose_mask()
    assert editor.document is doc and doc.dirty and errors
    monkeypatch.setattr("ui.mask_editor.QMessageBox.question", lambda *a: QMessageBox.Cancel)
    assert not editor.close()
    monkeypatch.setattr(editor, "confirm_discard", lambda: True)
    editor.close()


@pytest.mark.asyncio
async def test_generate_once_cache_and_cancel_preserves_edits(qt_app, monkeypatch):
    editor = MaskEditor()
    editor.open_session(reference(), "source.jpg")
    calls = []
    async def predict(image, progress=None):
        calls.append(image)
        return np.ones((image.height(), image.width()), np.bool_)
    monkeypatch.setattr("ui.mask_editor.predict_mask", predict)
    editor.generate()
    task = editor._task
    await task
    await asyncio.sleep(0)
    assert len(calls) == 1 and editor.document.mask.all()
    assert editor.progress_bar.value() == 100
    assert editor.document.mask.dtype == np.bool_ and editor.document.dirty
    for _ in range(3):
        editor.canvas.fit()
        editor.update_view()
        editor.refresh()
        qt_app.processEvents()
    assert len(calls) == 1
    previous = editor.document
    entered = asyncio.Event()
    drained = []
    async def slow(image, progress=None):
        entered.set()
        try:
            await asyncio.Future()
        finally:
            drained.append(True)
    monkeypatch.setattr("ui.mask_editor.predict_mask", slow)
    monkeypatch.setattr(editor, "confirm_discard", lambda: True)
    editor.generate()
    task = editor._task
    await entered.wait()
    editor.cancel_generation()
    await task
    await asyncio.sleep(0)
    assert drained and editor.document is previous and editor._task is None
    assert editor.progress_bar.value() != 100
    assert editor.progress_bar.format() == "已取消生成"
    editor.close()


@pytest.mark.asyncio
async def test_close_cancels_even_before_generation_starts(qt_app, monkeypatch):
    editor = MaskEditor()
    editor.open_session(reference(), "source.jpg")
    editor.show()
    monkeypatch.setattr(editor, "confirm_discard", lambda: True)
    editor.generate()
    task = editor._task
    assert not editor.close()
    assert editor.close_pending
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0)
    qt_app.processEvents()
    assert editor._task is None and not editor.isVisible()


@pytest.mark.asyncio
async def test_generation_failure_preserves_document(qt_app, monkeypatch):
    editor = MaskEditor()
    editor.open_session(reference(), "source.jpg")
    old = editor.document
    async def fail(image, progress=None):
        raise RuntimeError("model unavailable")
    errors = []
    monkeypatch.setattr("ui.mask_editor.predict_mask", fail)
    monkeypatch.setattr(editor, "error", errors.append)
    editor.generate()
    await editor._task
    await asyncio.sleep(0)
    qt_app.processEvents()
    assert errors == ["model unavailable"]
    assert editor.progress_bar.format() == "生成失败"
    assert editor.progress_bar.value() != 100
    assert editor.document is old and editor._task is None
    assert editor.generate_button.isEnabled()
    editor.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('num_tiles', [0, 2])
async def test_worker_progress_updates_gui_and_ignores_late_events(qt_app, monkeypatch, num_tiles):
    import threading
    from hoshicore.component.sky_segmentation import SkyProgress
    editor = MaskEditor()
    editor.open_session(reference(), "source.jpg")
    gui_thread = threading.get_ident()
    callbacks, worker_threads, values = [], [], []
    def emit(callback, event):
        worker_threads.append(threading.get_ident())
        callback(event)
    async def predict(image, progress=None):
        callbacks.append(progress)
        for completed, scale in enumerate((512, 768, 1024), 1):
            await asyncio.to_thread(emit, progress,
                                   SkyProgress('scale_completed', completed, 3, scale, .1))
            qt_app.processEvents()
            values.append(editor.progress_bar.value())
        await asyncio.to_thread(emit, progress, SkyProgress('roi_planned', 0, num_tiles))
        qt_app.processEvents()
        assert '局部细化' in editor.status.text()
        assert editor.progress_bar.value() == (40 if num_tiles else 95)
        for completed in range(1, num_tiles + 1):
            await asyncio.to_thread(emit, progress, SkyProgress('tile_completed', completed, num_tiles, seconds=.1))
            qt_app.processEvents()
            values.append(editor.progress_bar.value())
        await asyncio.to_thread(emit, progress, SkyProgress('fusing', num_tiles, num_tiles))
        qt_app.processEvents()
        assert editor.progress_bar.value() == 97
        await asyncio.to_thread(emit, progress, SkyProgress('complete', 1, 1))
        qt_app.processEvents()
        assert editor.progress_bar.value() == 99
        return np.ones((image.height(), image.width()), np.bool_)
    monkeypatch.setattr('ui.mask_editor.predict_mask', predict)
    editor.generate()
    await editor._task
    await asyncio.sleep(0)
    qt_app.processEvents()
    assert values == ([13, 27, 40, 68, 95] if num_tiles else [13, 27, 40])
    assert all(thread != gui_thread for thread in worker_threads)
    assert editor.progress_bar.value() == 100
    # Queued progress from a finished request must not replace its final state.
    callbacks[0](SkyProgress('scale_started', 0, 3, 512))
    qt_app.processEvents()
    assert editor.progress_bar.value() == 100
    entered = asyncio.Event()
    async def slow(image, progress=None):
        callbacks.append(progress)
        progress(SkyProgress('scale_completed', 1, 3, 512, .1))
        entered.set()
        await asyncio.Future()
    monkeypatch.setattr('ui.mask_editor.predict_mask', slow)
    monkeypatch.setattr(editor, 'confirm_discard', lambda: True)
    editor.generate()
    task = editor._task
    await entered.wait()
    qt_app.processEvents()
    assert editor.progress_bar.value() == 13
    # An earlier request's final event cannot advance this new request.
    callbacks[0](SkyProgress('complete', 1, 1))
    qt_app.processEvents()
    assert editor.progress_bar.value() == 13
    editor.cancel_generation()
    callbacks[1](SkyProgress('scale_completed', 3, 3, 1024, .1))
    qt_app.processEvents()
    assert editor.progress_bar.value() == 13
    assert '正在取消' in editor.progress_bar.format()
    await task
    await asyncio.sleep(0)
    assert editor.progress_bar.format() == '已取消生成'
    editor.close()


@pytest.mark.asyncio
async def test_parent_waits_for_generation_cleanup(qt_app, monkeypatch):
    parent = QWidget()
    parent.img_view_label = SimpleNamespace(pixmap=QPixmap.fromImage(reference()), img_path="source.jpg")
    parent.show()
    controller = MaskEditorController(parent)
    controller.open()
    editor = controller.editor
    entered = asyncio.Event()
    release = asyncio.Event()
    async def draining(image, progress=None):
        entered.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            await release.wait()
            raise
    monkeypatch.setattr("ui.mask_editor.predict_mask", draining)
    editor.generate()
    task = editor._task
    await entered.wait()
    assert not controller.close_for_parent()
    await asyncio.sleep(0)
    qt_app.processEvents()
    assert editor.isVisible() and parent.isVisible() and editor._task is task
    release.set()
    await task
    await asyncio.sleep(0)
    qt_app.processEvents()
    qt_app.processEvents()
    assert not editor.isVisible() and not parent.isVisible()


@pytest.fixture
def desktop_window(qt_app, monkeypatch):
    import importlib.util
    entry = Path(__file__).resolve().parents[1] / "HoshinoWeaver desktop.py"
    spec = importlib.util.spec_from_file_location("mask_editor_desktop_test", entry)
    desktop = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(desktop)
    monkeypatch.setattr(desktop.HNW_window, "read_config",
                        lambda self: self._CONFIG.update(guide_always_display=False))
    monkeypatch.setattr(desktop.HNW_window, "update_config_file", lambda self: None)
    window = desktop.HNW_window()
    window.show()
    qt_app.processEvents()
    yield window
    if window.mask_editor.editor is not None:
        monkeypatch.setattr(window.mask_editor.editor, 'confirm_discard', lambda: True)
    window.close()


def test_main_window_dual_entry_and_workflow_change(qt_app, desktop_window):
    window = desktop_window
    assert window.mask_editor_button.iconSize().width() == 24
    right = (window.mask_editor_button.mapToGlobal(QPoint()).x()
             - window.img_view_label.mapToGlobal(QPoint()).x() + window.mask_editor_button.width())
    assert window.img_view_label.width() - right <= 6
    assert window.mask_preview_tools.width() == window.img_view_label.width()
    assert window.mask_preview_tools.geometry().bottom() < window.img_view_label.geometry().top()
    assert not window.mask_preview_toggle.isHidden()
    assert not window.mask_preview_toggle.isEnabled()
    assert not window.mask_preview_toggle.icon().isNull()
    window.img_view_label.pixmap = QPixmap.fromImage(reference())
    window.img_view_label.img_path = "displayed.jpg"
    window.mask_editor_button.click()
    editor = window.mask_editor.editor
    assert editor.isVisible() and not editor.isModal() and editor.target is None
    window.load_mode_panel("对齐叠加")
    window.config_panel._config_widgets["mask"].findChild(QPushButton, "edit_mask").click()
    assert editor.target is not None and editor.target.is_current()
    assert not editor.apply_button.isHidden()
    window.load_mode_panel("堆栈降噪")
    assert not editor.target.is_current()
    assert editor.source == "displayed.jpg"
    window.close()
    assert not editor.isVisible()


def test_applied_mask_shared_across_frames_and_drafts_isolated(qt_app, desktop_window, tmp_path, monkeypatch):
    window = desktop_window
    window.load_mode_panel("对齐叠加")
    first, second = tmp_path / "first.png", tmp_path / "second.png"
    reference().save(str(first))
    reference().save(str(second))
    preview = window.img_view_label
    preview.initImg(str(first))
    window.config_panel._config_widgets['mask'].findChild(QPushButton, 'edit_mask').click()
    editor = window.mask_editor.editor
    editor.edit('invert')
    output = tmp_path / 'sequence_mask.png'
    monkeypatch.setattr('ui.mask_editor.QFileDialog.getSaveFileName', lambda *a: (str(output), ''))
    editor.save_and_apply()
    target = editor.target
    assert window.mask_editor.applied_target is target
    assert not window.mask_preview_toggle.isHidden()
    assert preview.mask_preview_matches
    center = preview.rect().center()
    assert preview.grab().toImage().pixelColor(center).green() > 100
    visible_icon = window.mask_preview_toggle.icon().cacheKey()
    window.mask_preview_toggle.click()
    assert preview.grab().toImage().pixelColor(center).green() == 60
    assert window.mask_preview_toggle.icon().cacheKey() != visible_icon
    assert '显示蒙版' in window.mask_preview_toggle.toolTip()
    window.mask_preview_toggle.click()
    assert window.mask_preview_toggle.icon().cacheKey() == visible_icon
    assert '隐藏蒙版' in window.mask_preview_toggle.toolTip()
    # Uncommitted strokes affect only the editor document, not the applied mask.
    editor.edit('invert')
    assert editor.document.dirty and not editor.document.mask.any()
    assert preview.grab().toImage().pixelColor(center).green() > 100
    doc = editor.document
    preview.initImg(str(second))
    assert preview.mask_preview_matches
    assert preview.grab().toImage().pixelColor(center).green() > 100
    window.mask_editor.open()
    assert editor.source == str(second) and editor.document is doc
    assert editor.document.dirty
    # Taking a reference snapshot must never bake in the preview overlay.
    assert editor.reference.pixelColor(0, 0).green() == 60
    monkeypatch.setattr('ui.mask_editor.QMessageBox.question', lambda *a: QMessageBox.Discard)
    editor.close()
    assert editor.document.mask.all() and not editor.document.dirty
    window.mask_editor.open()
    assert editor.document.mask.all() and editor.source == str(second)
    window.load_mode_panel("堆栈降噪")
    assert window.mask_editor_button.isHidden()
    assert not window.mask_preview_toggle.isHidden()
    assert not window.mask_preview_toggle.isEnabled()
    assert not preview.mask_preview_matches


def test_shared_preview_size_guard_and_parameter_invalidation(qt_app, desktop_window, tmp_path, monkeypatch):
    window = desktop_window
    window.load_mode_panel("对齐叠加")
    frame = tmp_path / 'frame.png'
    other_size = tmp_path / 'small.png'
    reference().save(str(frame))
    reference(64, 48).save(str(other_size))
    preview = window.img_view_label
    preview.initImg(str(frame))
    window.config_panel._config_widgets['mask'].findChild(QPushButton, 'edit_mask').click()
    editor = window.mask_editor.editor
    editor.edit('invert')
    output = tmp_path / 'sequence_mask.png'
    monkeypatch.setattr('ui.mask_editor.QFileDialog.getSaveFileName', lambda *a: (str(output), ''))
    editor.save_and_apply()
    target = editor.target
    preview.initImg(str(other_size))
    assert not preview.mask_preview_matches
    assert not window.mask_preview_toggle.isEnabled()
    errors = []
    monkeypatch.setattr(editor, 'error', errors.append)
    window.mask_editor.open()
    assert errors and editor.source == str(frame)
    preview.initImg(str(frame))
    assert preview.mask_preview_matches and window.mask_preview_toggle.isEnabled()
    target.set_path('other-mask.png')
    assert window.mask_editor.applied_target is None
    assert not preview.mask_preview_matches


def test_plain_save_does_not_apply_a_sequence_mask(desktop_window, tmp_path, monkeypatch):
    window = desktop_window
    window.load_mode_panel("对齐叠加")
    window.img_view_label.pixmap = QPixmap.fromImage(reference())
    window.img_view_label.img_path = 'reference.jpg'
    window.mask_editor.open()
    editor = window.mask_editor.editor
    editor.edit('invert')
    path = tmp_path / 'standalone.png'
    monkeypatch.setattr('ui.mask_editor.QFileDialog.getSaveFileName', lambda *a: (str(path), ''))
    assert editor.save()
    assert window.mask_editor.applied_target is None
    assert not window.img_view_label.mask_preview_matches
    assert not window.mask_preview_toggle.isHidden()
    assert not window.mask_preview_toggle.isEnabled()


@pytest.mark.parametrize("workflow", ["startrail", "aligned_stack", "timelapse_startrail", "norma_bundle_window"])
def test_workflow_mask_declarations(qt_app, workflow):
    root = Path(__file__).resolve().parents[1] / "hoshicore/dag"
    schema = PanelSchema.from_yaml(root / f"{workflow}.meta.yaml", root / f"{workflow}.ui.yaml")
    assert next(spec for spec in schema.configs if spec.key == "mask").widget == "mask_picker"
