"""Non-modal mask editor and its main-window integration controller."""
import asyncio
import math
from pathlib import Path

import numpy as np
from PySide6.QtCore import QIODevice, QSaveFile, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QColorSpace, QImage, QImageReader, QKeySequence, QShortcut
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFileDialog, QHBoxLayout,
                               QLabel, QMessageBox, QProgressBar, QPushButton, QSlider,
                               QVBoxLayout)

from ui.mask_canvas import MaskCanvas, image_rgb
from ui.mask_document import MaskDocument


def load_mask(path):
    return load_mask_document(path).mask


def load_mask_document(path):
    """Decode GUI mask formats and threshold grayscale at 128 into bool pixels."""
    reader = QImageReader(str(path))
    format_name = bytes(reader.format()).lower()
    if format_name not in (b"png", b"jpeg", b"jpg"):
        raise ValueError("编辑器支持 PNG 和 JPEG 蒙版。")
    image = reader.read()
    if image.isNull():
        raise ValueError(reader.errorString())
    if image.hasAlphaChannel():
        rgba = image.convertToFormat(QImage.Format_RGBA8888)
        pixels = np.frombuffer(rgba.constBits(), np.uint8).reshape(image.height(), rgba.bytesPerLine())
        pixels = pixels[:, :image.width() * 4].reshape(image.height(), image.width(), 4)
        if not np.all(pixels[:, :, 3] == 255):
            raise ValueError("请选择无透明区域的蒙版，白色表示天空、黑色表示地面。")
    grayscale = image.convertToFormat(QImage.Format_Grayscale8)
    rows = np.frombuffer(grayscale.constBits(), np.uint8).reshape(image.height(), grayscale.bytesPerLine())
    gray = rows[:, :image.width()]
    native = format_name == b"png" and image.allGray() and np.all((gray == 0) | (gray == 255))
    doc = MaskDocument(gray >= 128, str(path) if native else "")
    doc.suggested_path = str(Path(path).with_suffix(".png"))
    if not native:
        doc.mark_modified()
    return doc


def save_mask(path, mask):
    """Atomically replace the PNG only after successful encoding."""
    pixels = np.ascontiguousarray(mask, dtype=np.uint8) * 255
    height, width = pixels.shape
    image = QImage(pixels.data, width, height, pixels.strides[0], QImage.Format_Grayscale8)
    output = QSaveFile(str(path))
    if not output.open(QIODevice.WriteOnly):
        raise OSError(output.errorString())
    if not image.save(output, "PNG"):
        output.cancelWriting()
        raise OSError("无法编码 PNG 蒙版。")
    if not output.commit():
        raise OSError(output.errorString())


async def predict_mask(reference, progress=None):
    from hoshicore.engine.wiring import _load_default_settings
    from hoshicore.services import create_service_registry

    if reference.colorSpace().isValid():
        reference = reference.convertedToColorSpace(QColorSpace(QColorSpace.SRgb))
    rgb = image_rgb(reference)
    async with create_service_registry().scope(_load_default_settings()) as services:
        result = await services.segmentation.predict_sky(rgb, color_order="RGB", progress=progress)
        # Do not retain the native-size probability map beyond this request.
        return result.mask.astype(np.bool_)


class MaskEditor(QDialog):
    closed = Signal()
    mask_saved = Signal(str)
    mask_applied = Signal(object, str)
    inference_progress = Signal(int, object)

    def __init__(self, parent=None):
        super().__init__(parent, Qt.Window)
        self.setModal(False)
        self.resize(1150, 800)
        self.setWindowTitle("天空蒙版编辑器")
        self.reference = QImage()
        self.source = ""
        self.target = None
        self._task = None
        self._generation_id = 0
        self._progress_finished = True
        self._cancelling = False
        self._generated = False
        self._loaded_path = ""
        self.close_pending = False
        self._allow_close = False
        layout = QVBoxLayout(self)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.file_bar = QHBoxLayout()
        layout.addLayout(self.file_bar)

        def button(text, callback, bar):
            widget = QPushButton(text)
            widget.setAutoDefault(False)
            widget.clicked.connect(callback)
            bar.addWidget(widget)
            return widget

        self.load_button = button("加载蒙版", self.choose_mask, self.file_bar)
        self.generate_button = button("自动生成", self.generate, self.file_bar)
        self.save_button = button("保存", self.save, self.file_bar)
        self.save_as_button = button("另存为", lambda: self.save(force_dialog=True), self.file_bar)
        self.apply_button = button("保存并应用", self.save_and_apply, self.file_bar)
        self.file_bar.addStretch()
        self.cancel_button = button("取消生成", self.cancel_generation, self.file_bar)

        self.tools = QHBoxLayout()
        layout.addLayout(self.tools)
        self.tool_select = QComboBox()
        for label, key in (("画天空", "sky"), ("画地面", "ground")):
            self.tool_select.addItem(label, key)
        self.tools.addWidget(self.tool_select)
        self.tools.addWidget(QLabel("笔刷"))
        self.brush_size = QSlider(Qt.Horizontal)
        self.brush_size.setRange(0, 1000)
        self.brush_size.setValue(630)
        self.brush_size.setFixedWidth(120)
        self.tools.addWidget(self.brush_size)
        self.brush_label = QLabel()
        self.tools.addWidget(self.brush_label)
        self.screen_brush = QCheckBox("固定屏幕大小")
        self.screen_brush.setToolTip("缩放图片时保持笔刷屏幕直径，实际覆盖的原图像素随缩放变化")
        self.tools.addWidget(self.screen_brush)
        self.undo_button = button("撤销", lambda: self.edit("undo"), self.tools)
        self.redo_button = button("重做", lambda: self.edit("redo"), self.tools)
        self.invert_button = button("反相", lambda: self.edit("invert"), self.tools)
        self.tools.addStretch()
        self.view_select = QComboBox()
        for label, key in (("叠加预览", "overlay"), ("纯蒙版", "mask"), ("原图", "image")):
            self.view_select.addItem(label, key)
        self.tools.addWidget(self.view_select)
        self.tools.addWidget(QLabel("叠加透明度"))
        self.opacity = QSlider(Qt.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.setValue(45)
        self.opacity.setMaximumWidth(110)
        self.tools.addWidget(self.opacity)
        button("适应窗口", lambda: self.canvas.fit(), self.tools)
        self.canvas = MaskCanvas()
        layout.addWidget(self.canvas, 1)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)
        self.inference_progress.connect(self._update_progress, Qt.QueuedConnection)
        self.status = QLabel("白色＝天空，黑色＝地面；左键绘制，右键平移，滚轮缩放。")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.tool_select.currentIndexChanged.connect(self.update_view)
        self.view_select.currentIndexChanged.connect(self.update_view)
        self.brush_size.valueChanged.connect(self.update_view)
        self.screen_brush.toggled.connect(self.set_brush_mode)
        self.opacity.valueChanged.connect(self.update_view)
        self.canvas.changed.connect(self.refresh)
        for key, callback in ((QKeySequence.Undo, lambda: self.edit("undo")),
                              (QKeySequence.Redo, lambda: self.edit("redo")),
                              (QKeySequence.Save, self.save)):
            QShortcut(QKeySequence(key), self, activated=callback)
        self._target_timer = QTimer(self)
        self._target_timer.setInterval(300)
        self._target_timer.timeout.connect(self.refresh)
        self._target_timer.start()
        self.refresh()

    @property
    def document(self):
        return self.canvas.document

    def error(self, error):
        QMessageBox.warning(self, "蒙版操作失败", str(error))

    def confirm_discard(self):
        self.canvas.finish_stroke()
        if self.document is None or not self.document.dirty:
            return True
        answer = QMessageBox.question(
            self, "蒙版尚未保存", "是否保存当前蒙版的修改？",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Save)
        if answer == QMessageBox.Cancel:
            return False
        if answer == QMessageBox.Save:
            return self.save()
        if self.document.path:
            self.document.discard_changes()
            self.canvas.sync()
        else:
            self.canvas.set_document(MaskDocument(np.zeros_like(self.document.mask)), self.reference)
            self._generated = False
            self._loaded_path = ""
            self.update_view()
        self.refresh()
        return True

    def open_session(self, reference, source, target=None, shared_path=""):
        if self._task is not None:
            self.error("请先等待生成完成，或取消生成。")
            return False
        path = (target.path if target else "") or shared_path
        doc = self.document
        same_size = doc is not None and (reference.isNull() or
                    doc.mask.shape == (reference.height(), reference.width()))
        same_path = bool(path and self._loaded_path and
                         Path(path).resolve() == Path(self._loaded_path).resolve())
        same_reference = source == self.source and reference.size() == self.reference.size()
        reuse = same_size and (same_path or (not path and same_reference))
        try:
            if not reuse and path:
                doc = load_mask_document(path)
                self.check_size(doc.mask, reference)
            elif not reuse:
                doc = (MaskDocument(np.zeros((reference.height(), reference.width()), dtype=np.bool_))
                       if not reference.isNull() else None)
        except Exception as error:
            self.error(error)
            return False
        if not reuse and not self.confirm_discard():
            return False
        self.target = target
        self.progress_bar.hide()
        if not reuse:
            self._generated = False
            self._loaded_path = path
        self.reference = reference.copy()
        self.source = source
        self._allow_close = False
        if reuse:
            self.canvas.reference = self.reference
            self.canvas.update()
        else:
            self.canvas.set_document(doc, self.reference)
        self.update_view()
        self.status.setText("沿用当前蒙版；当前展示图作为编辑底图。" if reuse else
                            "已按灰度 128 转为二值蒙版，请保存为 PNG。" if doc and doc.dirty else
                            "白色＝天空，黑色＝地面；左键绘制，右键平移，滚轮缩放。")
        self.refresh()
        return True

    @staticmethod
    def check_size(mask, reference):
        if not reference.isNull() and mask.shape != (reference.height(), reference.width()):
            raise ValueError("蒙版尺寸与参考图不一致；请使用相同尺寸和方向的蒙版。")

    def choose_mask(self):
        if self._task is not None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "加载蒙版（灰度 ≥ 128 为天空）", "", "蒙版图像 (*.png *.jpg *.jpeg)")
        if not path:
            return
        try:
            doc = load_mask_document(path)
            self.check_size(doc.mask, self.reference)
            if not self.confirm_discard():
                return
            self.canvas.set_document(doc, self.reference)
            self._loaded_path = path
            self.update_view()
            self.progress_bar.hide()
            self._generated = False
            self.status.setText("已按灰度 128 转为二值蒙版，请保存为 PNG。" if doc.dirty
                                else f"已加载：{path}")
            self.refresh()
        except Exception as error:
            self.error(error)

    def save(self, checked=False, *, force_dialog=False):
        if self.document is None:
            return False
        self.canvas.finish_stroke()
        path = self.document.path
        if not path or force_dialog:
            suggested = path or self.document.suggested_path or (str(Path(self.source).with_name(Path(self.source).stem + "_sky.png"))
                                 if self.source else "sky_mask.png")
            path, _ = QFileDialog.getSaveFileName(self, "保存天空蒙版", suggested, "PNG 蒙版 (*.png)")
            if not path:
                return False
            if Path(path).suffix.lower() != ".png":
                path += ".png"
        if self.source and Path(path).resolve() == Path(self.source).resolve():
            self.error("请使用不同的文件名，避免覆盖参考图。")
            return False
        try:
            save_mask(path, self.document.mask)
            self.document.mark_saved(path)
            self._loaded_path = path
            self.mask_saved.emit(path)
            self.status.setText(f"已保存：{path}")
            self.refresh()
            return True
        except Exception as error:
            self.error(error)
            return False

    def save_and_apply(self):
        if self.target is None or not self.target.is_current():
            self.status.setText("原工作流参数已失效；仍可保存蒙版文件。")
            return
        if self.save():
            if self.target.apply(self.document.path):
                self.mask_applied.emit(self.target, self.document.path)
                self.status.setText(f"已保存并应用到：{self.target.label}")
            else:
                self.status.setText("文件已保存，原工作流参数已失效，未回填。")
        self.refresh()

    def edit(self, operation):
        if self._task is not None or self.document is None:
            return
        self.canvas.finish_stroke()
        getattr(self.document, operation)()
        self.canvas.sync()
        self.refresh()

    def update_view(self):
        self.canvas.finish_stroke()
        self.canvas.tool = self.tool_select.currentData()
        self.canvas.mode = self.view_select.currentData()
        base = (1000 if self.canvas.screen_brush else
                max(self.document.mask.shape) if self.document else 1000)
        self.canvas.diameter = max(1., base * .0002 * 1500 ** (self.brush_size.value() / 1000))
        self.brush_label.setText(f"{self.canvas.diameter:.0f} px")
        self.canvas.opacity = self.opacity.value() / 100
        self.canvas.update()

    def set_brush_mode(self, screen_fixed):
        self.canvas.finish_stroke()
        diameter = self.canvas.effective_diameter
        self.canvas.screen_brush = screen_fixed
        value = diameter * self.canvas.scale if screen_fixed else diameter
        base = 1000 if screen_fixed else max(self.document.mask.shape) if self.document else 1000
        position = round(1000 * math.log(max(value, .001) / (base * .0002)) / math.log(1500))
        self.brush_size.setValue(max(0, min(1000, position)))
        self.update_view()

    def refresh(self):
        doc = self.document
        busy = self._task is not None
        valid = self.target is not None and self.target.is_current()
        destination = (self.target.label if valid else "原目标已失效") if self.target else "独立编辑，仅保存文件"
        size = f"{doc.mask.shape[1]} × {doc.mask.shape[0]}" if doc else "无蒙版"
        self.summary.setText(f"参考图：{self.source or '无'}\n{size} | {destination}"
                             + (" | 有未保存修改" if doc and doc.dirty else ""))
        self.setWindowTitle("天空蒙版编辑器" + (" *" if doc and doc.dirty else ""))
        self.load_button.setEnabled(not busy)
        self.generate_button.setText("重新生成" if self._generated else "自动生成")
        self.generate_button.setEnabled(not busy and not self.reference.isNull())
        for widget in (self.save_button, self.save_as_button, self.invert_button,
                       self.tool_select, self.brush_size, self.screen_brush):
            widget.setEnabled(not busy and doc is not None)
        self.apply_button.setVisible(self.target is not None)
        self.apply_button.setEnabled(not busy and doc is not None and valid)
        self.undo_button.setEnabled(not busy and doc is not None and bool(doc.undo_stack))
        self.redo_button.setEnabled(not busy and doc is not None and bool(doc.redo_stack))
        self.cancel_button.setVisible(busy)
        self.canvas.setEnabled(not busy)

    def generate(self):
        if self._task is not None or self.reference.isNull():
            return
        if not self.confirm_discard():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self.error("当前没有可用的异步事件循环。")
            return
        self._generation_id += 1
        self._progress_finished = False
        self._cancelling = False
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFormat("正在初始化模型……")
        self.progress_bar.show()
        self._task = loop.create_task(self._generate(self._generation_id))
        self._task.add_done_callback(self._generation_done)
        self.status.setText("正在生成天空蒙版……")
        self.refresh()

    @Slot(int, object)
    def _update_progress(self, generation_id, event):
        if (generation_id != self._generation_id or self._task is None
                or self._progress_finished or self._cancelling):
            return
        if event.stage == 'initializing':
            self.progress_bar.setRange(0, 0)
            self.status.setText("正在初始化天空分割模型……")
            return
        self.progress_bar.setRange(0, 100)
        fraction = event.completed / max(1, event.total)
        if event.stage == 'scale_started':
            value = round(fraction * (40 if event.strategy == 'roi' else 100))
            text = f"全局定位：尺度 {event.scale}（{event.completed + 1}/{event.total}）"
        elif event.stage == 'scale_completed':
            value = round(fraction * (40 if event.strategy == 'roi' else 100))
            text = f"全局定位：已完成 {event.completed}/{event.total} 个尺度（{event.seconds:.2f} 秒）"
        elif event.stage == 'roi_planned':
            value = 40 if event.total else 95
            text = f"局部细化：待处理 {event.total} 个窗口" if event.total else "无需局部细化，正在融合"
        elif event.stage in ('tile_started', 'tile_completed'):
            value = 40 + round(55 * fraction)
            text = (f"局部细化：正在处理窗口 {event.completed + 1}/{event.total}"
                    if event.stage == 'tile_started' else
                    f"局部细化：已完成 {event.completed}/{event.total} 个窗口（{event.seconds:.2f} 秒）")
        elif event.stage == 'fusing':
            value = 97
            text = "正在融合全局与局部结果……"
        else:
            value = 99
            text = "正在缓存蒙版……"
        self.progress_bar.setValue(min(99, value))
        self.progress_bar.setFormat("%p% · " + text)
        self.status.setText(text)

    def _finish_progress(self, text, success=False):
        self._progress_finished = True
        was_busy = self.progress_bar.maximum() == 0
        self.progress_bar.setRange(0, 100)
        if success:
            self.progress_bar.setValue(100)
        elif was_busy:
            self.progress_bar.setValue(0)
        self.progress_bar.setFormat(text)

    async def _generate(self, generation_id):
        try:
            mask = await predict_mask(self.reference.copy(), progress=lambda event:
                                      self.inference_progress.emit(generation_id, event))
            self.check_size(mask, self.reference)
            doc = MaskDocument(mask)
            doc.mark_modified()
            self.canvas.set_document(doc, self.reference)
            self.update_view()
            self._generated = True
            self._finish_progress("生成完成 · 100%", success=True)
            self.status.setText("生成完成，结果已缓存；可使用画笔修正。")
        except asyncio.CancelledError:
            self._finish_progress("已取消生成")
            self.status.setText("已取消生成，原蒙版已保留。")
        except Exception as error:
            self._finish_progress("生成失败")
            self.status.setText(f"生成失败：{error}")
            if not self.close_pending:
                QTimer.singleShot(0, lambda message=str(error): self.error(message))

    def _generation_done(self, task):
        # Also runs if cancelled before the coroutine gets its first turn.
        if task.cancelled():
            self._finish_progress("已取消生成")
            self.status.setText("已取消生成，原蒙版已保留。")
        self._task = None
        self.refresh()
        if self.close_pending:
            self._allow_close = True
            QTimer.singleShot(0, self.close)

    def cancel_generation(self):
        if self._task is not None:
            self._cancelling = True
            self._finish_progress("正在取消，等待当前计算结束……")
            self.status.setText("正在取消，等待当前计算结束……")
            self._task.cancel()

    def reject(self):
        # Escape follows the same save/cleanup path as the window close button.
        self.close()

    def closeEvent(self, event):
        if not self._allow_close and not self.confirm_discard():
            event.ignore()
            return
        if self._task is not None:
            self.close_pending = True
            self.cancel_generation()
            event.ignore()
            return
        self.close_pending = False
        self._allow_close = False
        self._target_timer.stop()
        event.accept()
        self.closed.emit()

    def showEvent(self, event):
        self._target_timer.start()
        super().showEvent(event)


class MaskEditorController:
    """One editor per main window; no implicit global mask or workflow binding."""

    def __init__(self, window):
        self.window = window
        self.editor = None
        self._close_parent = False
        self.applied_target = None
        self.applied_path = ""
        preview = window.img_view_label
        if hasattr(preview, 'image_changed'):
            preview.image_changed.connect(self.preview_changed)

    def reference(self):
        preview = self.window.img_view_label
        image = preview.pixmap.toImage() if preview.pixmap is not None else QImage()
        return image, preview.img_path or ""

    def open(self, target=None):
        if self.editor is None:
            self.editor = MaskEditor(self.window)
            self.editor.closed.connect(self._closed)
            self.editor.mask_applied.connect(self._apply_preview)
            self.editor.mask_saved.connect(self._refresh_saved_preview)
        editor = self.editor
        if editor._task is None:
            self.refresh_workflow()
            reference, source = self.reference()
            if not editor.open_session(reference, source, target, shared_path=self.applied_path):
                return
        elif editor.target is not target:
            editor.error("请先等待生成完成，或取消生成。")
            return
        editor.show()
        editor.raise_()
        editor.activateWindow()

    def _target_registered(self):
        panel = getattr(self.window, 'config_panel', None)
        return (self.applied_target in panel.mask_targets() if panel is not None else
                self.applied_target is not None and self.applied_target.is_current())

    def _apply_preview(self, target, path):
        try:
            mask = load_mask(path)
        except Exception as error:
            self.editor.error(error)
            return
        self.applied_target = target
        self.applied_path = path
        self.window.img_view_label.set_mask_preview(mask)
        toggle = getattr(self.window, 'mask_preview_toggle', None)
        if toggle is not None:
            toggle.setChecked(True)
        self.refresh_workflow()

    def _refresh_saved_preview(self, path):
        if self.applied_path and Path(path).resolve() == Path(self.applied_path).resolve():
            try:
                self.window.img_view_label.set_mask_preview(load_mask(path))
            except Exception as error:
                self.editor.error(error)
            self.refresh_workflow()

    def refresh_workflow(self):
        panel = getattr(self.window, 'config_panel', None)
        supports_mask = panel is None or bool(panel.mask_targets())
        button = getattr(self.window, 'mask_editor_button', None)
        if button is not None:
            button.setVisible(supports_mask)
        if self.applied_target is not None and (not self._target_registered() or
                Path(self.applied_target.current_path or "").resolve() != Path(self.applied_path).resolve()):
            self.applied_target = None
            self.applied_path = ""
            self.window.img_view_label.set_mask_preview(None)
        self.preview_changed()

    def preview_changed(self):
        toggle = getattr(self.window, 'mask_preview_toggle', None)
        if toggle is not None:
            matches = self.window.img_view_label.mask_preview_matches
            has_image = self.window.img_view_label.pixmap is not None
            description = (f"已应用：{self.applied_target.label}\n{self.applied_path}\n"
                           if self.applied_target is not None else "")
            if self.applied_target is None:
                toggle.setChecked(False)
                reason = '尚未应用蒙版，请先在工作流蒙版参数中保存并应用。'
            elif not has_image:
                reason = description + '当前没有展示图像。'
            elif not matches:
                reason = description + '蒙版尺寸与当前图像不符，不拉伸叠加。'
            else:
                reason = description + '仅对相同像素尺寸叠加，不缩放蒙版。'
            self.window.mask_preview_tools.set_mask_state(self.applied_target is not None and matches, reason)

    def set_preview_visible(self, visible):
        self.window.img_view_label.mask_preview_visible = visible
        self.window.img_view_label.update()

    def close_for_parent(self):
        if self.editor is None or (not self.editor.isVisible() and self.editor._task is None):
            return True
        accepted = self.editor.close()
        self._close_parent = not accepted and self.editor.close_pending
        return accepted

    def _closed(self):
        if self._close_parent:
            self._close_parent = False
            QTimer.singleShot(0, self.window.close)
