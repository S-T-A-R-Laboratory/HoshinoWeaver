"""OutputPanel: renders one output region per OutputSpec declared in ui.yaml.

Responsibilities:
- Render format / path / dtype / format-specific params for each output
- Enforce intersection of physical constraints (IMAGE_FORMAT_PRESETS) and
  workflow constraints (OutputSpec.formats / dtype_options)
- Provide collect_outputs() → dict[config_key, value] for start_task injection
- Provide is_ready() + missing_reason() for detect_status()

Path caching: per-output, per-format. Switching format restores the last
path used for that format; switching back to a format with a cached path
avoids re-prompting.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtCore import QUrl
from PySide6.QtGui import QColor, QDesktopServices, QPainter, QPen
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QFrame, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QSizePolicy, QSlider, QStyle, QStyleOptionGroupBox, QVBoxLayout, QWidget,
)

from ui.output_presets import (
    IMAGE_FORMAT_PRESETS, VIDEO_CODEC_PRESETS, VIDEO_FORMAT_PRESETS,
    all_format_keys, detect_format_by_path,
)
from ui.panel_builder import OutputSpec
from ui.widgets import ConfigSpec, create_config_row
from ui.styles import GROUP_HEADER_STYLE, LABEL_STYLE


# ─── Single output region ────────────────────────────────────────────────────


def _select_row(label: str, values: list[str], parent: QWidget) -> tuple[QFrame, QComboBox]:
    row, _getter, _setter = create_config_row(
        ConfigSpec(key=label, label=label, widget="select",
                   options=[(value, value) for value in values]), parent)
    return row, row.findChild(QComboBox)


def _path_row(label: str, parent: QWidget, browse: Callable, *,
              on_change: Callable | None = None, read_only: bool = True,
              placeholder: str = "") -> tuple[QFrame, QLineEdit]:
    row, _getter, _setter = create_config_row(
        ConfigSpec(key=label, label=label), parent, on_change=on_change)
    line = row.findChild(QLineEdit)
    line.setReadOnly(read_only)
    line.setPlaceholderText(placeholder)
    button = QPushButton("...", row)
    button.setFixedWidth(28)
    button.setToolTip(f"选择{label}")
    button.clicked.connect(browse)
    row.layout().addWidget(button)
    return row, line


class _OutputRegion(QFrame):
    """Renders one output (format selector + path + dtype + format params)."""

    changed = Signal()

    def __init__(self, spec: OutputSpec, parent: QWidget | None = None,
                 path_cache: dict[str, str] | None = None):
        super().__init__(parent)
        self.spec = spec
        self._format_param_widgets: dict[str, tuple[QFrame, Callable[[], Any]]] = {}
        self._path_cache: dict[str, str] = path_cache if path_cache is not None else {}
        self._current_format: str | None = None

        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        # Label header
        if spec.label and not spec.enable_key:
            header = QLabel(spec.label, self)
            header.setStyleSheet(GROUP_HEADER_STYLE)
            outer.addWidget(header)

        # Resolve allowed formats
        self._allowed_formats = spec.formats or all_format_keys()
        self._allowed_formats = [
            f for f in self._allowed_formats
            if f in IMAGE_FORMAT_PRESETS
            and self._effective_dtypes(f)  # non-empty intersection
        ]
        if not self._allowed_formats:
            raise ValueError(
                f"OutputSpec '{spec.filename_key}': no usable format after "
                f"intersecting workflow constraints with IMAGE_FORMAT_PRESETS")

        # Row 1: format selector
        row_fmt, self._format_combo = _select_row("文件格式", self._allowed_formats, self)
        self._format_combo.currentTextChanged.connect(self._on_format_changed)
        outer.addWidget(row_fmt)

        # Row 2: path + browse
        row_path, self._path_edit = _path_row("输出路径", self, self._on_browse)
        outer.addWidget(row_path)

        # Row 3: dtype selector (optional)
        self._dtype_combo: QComboBox | None = None
        if spec.dtype_key:
            row_dtype, self._dtype_combo = _select_row("输出位深", [], self)
            self._dtype_combo.currentTextChanged.connect(lambda _: self.changed.emit())
            outer.addWidget(row_dtype)

        # Row 4+: format-specific params (one widget per preset_param declared
        # in format_params; visibility toggled by current format)
        self._params_container = QFrame(self)
        pc_layout = QVBoxLayout(self._params_container)
        pc_layout.setContentsMargins(0, 0, 0, 0)
        pc_layout.setSpacing(2)
        outer.addWidget(self._params_container)

        for preset_param, config_key in spec.format_params.items():
            row_p = self._build_param_row(preset_param, config_key)
            if row_p is not None:
                pc_layout.addWidget(row_p)

        # Initialize: pick first allowed format
        self._format_combo.setCurrentIndex(0)
        self._on_format_changed(self._format_combo.currentText())

    # ── Effective options ────────────────────────────────────────────────────

    def _effective_dtypes(self, fmt: str) -> list[str]:
        """Intersection of preset allowed_dtypes and spec.dtype_options."""
        preset_dtypes = IMAGE_FORMAT_PRESETS[fmt]["allowed_dtypes"]
        if self.spec.dtype_options is None:
            return list(preset_dtypes)
        return [d for d in preset_dtypes if d in self.spec.dtype_options]

    def _find_preset_for_param(self, preset_param: str) -> tuple[str, dict] | None:
        for fmt, preset in IMAGE_FORMAT_PRESETS.items():
            if preset_param in preset["params"]:
                return fmt, preset["params"][preset_param]
        return None

    def _build_param_row(self, preset_param: str, config_key: str):
        """Build a slider row for a format-specific param. Returns None if unknown."""
        found = self._find_preset_for_param(preset_param)
        if not found:
            return None
        owner_fmt, param_def = found

        row = QFrame(self._params_container)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(5, 0, 5, 0)
        layout.setSpacing(6)

        label = QLabel(param_def.get("label", preset_param), row)
        label.setMinimumWidth(60)
        label.setMaximumWidth(80)
        label.setStyleSheet(LABEL_STYLE)
        layout.addWidget(label)

        widget = param_def.get("widget", "slider")
        default = int(param_def.get("default", 0))
        mn = int(param_def.get("min", 0))
        mx = int(param_def.get("max", 100))

        if widget == "slider":
            slider = QSlider(Qt.Horizontal, row)
            slider.setMinimum(mn)
            slider.setMaximum(mx)
            slider.setValue(default)
            val_label = QLabel(str(default), row)
            val_label.setMinimumWidth(28)
            val_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            slider.valueChanged.connect(lambda v: val_label.setText(str(v)))
            slider.valueChanged.connect(lambda _: self.changed.emit())
            layout.addWidget(slider, 1)
            layout.addWidget(val_label)
            getter = slider.value
            setter = lambda v, s=slider: s.setValue(int(v))
        else:
            # Fallback: label + value only (YAGNI — other widgets not needed yet)
            lbl = QLabel(str(default), row)
            layout.addWidget(lbl, 1)
            getter = lambda v=default: v
            setter = lambda v: None

        # Attach owner format so we can toggle visibility
        row.setProperty("owner_fmt", owner_fmt)
        row.setProperty("config_key", config_key)
        self._format_param_widgets[preset_param] = (row, getter, setter)
        return row

    # ── Event handlers ───────────────────────────────────────────────────────

    def _on_format_changed(self, fmt: str):
        if fmt not in IMAGE_FORMAT_PRESETS:
            return
        self._current_format = fmt

        # Update dtype combo
        if self._dtype_combo is not None:
            current_dtype = self._dtype_combo.currentText()
            self._dtype_combo.blockSignals(True)
            self._dtype_combo.clear()
            effective = self._effective_dtypes(fmt)
            for d in effective:
                self._dtype_combo.addItem(d)
            # Preserve selection when possible
            if current_dtype in effective:
                self._dtype_combo.setCurrentText(current_dtype)
            elif effective:
                self._dtype_combo.setCurrentIndex(0)
            self._dtype_combo.setEnabled(len(effective) > 1)
            self._dtype_combo.blockSignals(False)

        # Toggle format-specific param rows
        for preset_param, (row, _getter, _setter) in self._format_param_widgets.items():
            owner_fmt = row.property("owner_fmt")
            row.setVisible(owner_fmt == fmt)

        # Restore cached path for this format
        cached = self._path_cache.get(fmt, "")
        self._path_edit.setText(cached)

        self.changed.emit()

    def _on_browse(self):
        # Build filter covering all allowed formats, with current as default
        filter_parts = []
        for fmt in self._allowed_formats:
            exts = " ".join(f"*{ext}" for ext in IMAGE_FORMAT_PRESETS[fmt]["ext"])
            filter_parts.append(f"{fmt} ({exts})")
        filter_str = ";;".join(filter_parts)
        current_fmt = self._format_combo.currentText()
        selected_filter = next(
            (fp for fp in filter_parts if fp.startswith(f"{current_fmt} ")),
            filter_parts[0] if filter_parts else "",
        )

        path, chosen_filter = QFileDialog.getSaveFileName(
            self, f"保存 {self.spec.label}", "", filter_str, selected_filter)
        if not path:
            return

        # Detect format from chosen filter label (e.g. "PNG (*.png)")
        chosen_fmt = chosen_filter.split(" ")[0] if chosen_filter else current_fmt
        if chosen_fmt not in IMAGE_FORMAT_PRESETS:
            chosen_fmt = detect_format_by_path(path) or current_fmt

        # Ensure extension matches chosen format
        if detect_format_by_path(path) != chosen_fmt:
            path = path + IMAGE_FORMAT_PRESETS[chosen_fmt]["default_ext"]

        self._path_cache[chosen_fmt] = path

        if chosen_fmt != current_fmt:
            self._format_combo.setCurrentText(chosen_fmt)
            # _on_format_changed restores cached path — writes it now
        else:
            self._path_edit.setText(path)
        self.changed.emit()

    # ── Collect / validate ───────────────────────────────────────────────────

    def collect(self) -> dict[str, Any]:
        """Return {config_key: value} for this output region."""
        result: dict[str, Any] = {}
        fmt = self._format_combo.currentText()
        path = self._path_edit.text()
        result[self.spec.filename_key] = path

        if self._dtype_combo is not None and self.spec.dtype_key:
            dtype = self._dtype_combo.currentText()
            if dtype:
                result[self.spec.dtype_key] = dtype

        for preset_param, (row, getter, _setter) in self._format_param_widgets.items():
            if row.property("owner_fmt") != fmt:
                continue
            config_key = row.property("config_key")
            result[config_key] = getter()

        return result

    def is_ready(self) -> bool:
        return bool(self._path_edit.text())

    def missing_reason(self) -> str | None:
        if not self._path_edit.text():
            return f"请选择{self.spec.label}路径"
        return None

    def apply_defaults(self, defaults: dict[str, Any]):
        """Apply output initialization defaults to this region's controls.

        Keys recognised:
          output_format — format combo (e.g. "TIFF")
          output_dtype  — dtype combo (e.g. "uint16")
          jpg_quality, png_compressing — format-specific param sliders
        """
        fmt = defaults.get("output_format")
        if fmt and fmt in self._allowed_formats:
            self._format_combo.setCurrentText(fmt)

        dtype = defaults.get("output_dtype")
        if dtype and self._dtype_combo is not None:
            effective = self._effective_dtypes(self._format_combo.currentText())
            if dtype in effective:
                self._dtype_combo.setCurrentText(dtype)

        for preset_param, (row, _getter, setter) in self._format_param_widgets.items():
            if preset_param in defaults:
                setter(defaults[preset_param])


# ─── Sequence output region ──────────────────────────────────────────────────


class _SequenceOutputRegion(QFrame):
    """Renders sequence output: directory picker + format + dtype + template."""

    changed = Signal()

    def __init__(self, spec: OutputSpec, parent: QWidget | None = None):
        super().__init__(parent)
        self.spec = spec
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        if spec.label and not spec.enable_key:
            header = QLabel(spec.label, self)
            header.setStyleSheet(GROUP_HEADER_STYLE)
            outer.addWidget(header)

        # Row 1: format selector
        allowed_formats = spec.formats or list(IMAGE_FORMAT_PRESETS.keys())
        self._allowed_formats = [
            f for f in allowed_formats if f in IMAGE_FORMAT_PRESETS
        ]

        row_fmt, self._format_combo = _select_row("文件格式", self._allowed_formats, self)
        self._format_combo.currentTextChanged.connect(self._on_format_changed)
        outer.addWidget(row_fmt)

        # Row 2: output directory + browse
        row_dir, self._dir_edit = _path_row("输出目录", self, self._on_browse_dir)
        outer.addWidget(row_dir)

        # Row 3: template name (editable)
        self._template_edit: QLineEdit | None = None
        if spec.template_key:
            row_tpl, _getter, _setter = create_config_row(
                ConfigSpec(key=spec.template_key, label="文件名模板"), self,
                on_change=self.changed.emit)
            self._template_edit = row_tpl.findChild(QLineEdit)
            self._template_edit.setPlaceholderText("frame_{index:05d}.png")
            outer.addWidget(row_tpl)

        # Row 4: dtype selector
        self._dtype_combo: QComboBox | None = None
        if spec.dtype_key:
            row_dtype, self._dtype_combo = _select_row("输出位深", [], self)
            self._dtype_combo.currentTextChanged.connect(lambda _: self.changed.emit())
            outer.addWidget(row_dtype)

        # Initialize format (also sets default template text)
        if self._allowed_formats:
            self._format_combo.setCurrentIndex(0)
            self._on_format_changed(self._format_combo.currentText())

    def _effective_dtypes(self, fmt: str) -> list[str]:
        preset_dtypes = IMAGE_FORMAT_PRESETS.get(fmt, {}).get("allowed_dtypes", [])
        if self.spec.dtype_options is None:
            return list(preset_dtypes)
        return [d for d in preset_dtypes if d in self.spec.dtype_options]

    def _on_format_changed(self, fmt: str):
        # Update template extension if user hasn't manually edited
        if self._template_edit is not None:
            old_text = self._template_edit.text()
            default = self._default_template(fmt)
            # Auto-update only if empty or still matches a default pattern
            if not old_text or self._is_default_template(old_text):
                self._template_edit.blockSignals(True)
                self._template_edit.setText(default)
                self._template_edit.blockSignals(False)

        if self._dtype_combo is not None:
            current_dtype = self._dtype_combo.currentText()
            self._dtype_combo.blockSignals(True)
            self._dtype_combo.clear()
            effective = self._effective_dtypes(fmt)
            for d in effective:
                self._dtype_combo.addItem(d)
            if current_dtype in effective:
                self._dtype_combo.setCurrentText(current_dtype)
            elif effective:
                self._dtype_combo.setCurrentIndex(0)
            self._dtype_combo.setEnabled(len(effective) > 1)
            self._dtype_combo.blockSignals(False)
        self.changed.emit()

    def _on_browse_dir(self):
        path = QFileDialog.getExistingDirectory(self, f"选择{self.spec.label}目录")
        if path:
            self._dir_edit.setText(path)
            self.changed.emit()

    def _default_template(self, fmt: str) -> str:
        ext = IMAGE_FORMAT_PRESETS.get(fmt, {}).get("default_ext", ".png")
        return "frame_{index:05d}" + ext

    def _is_default_template(self, text: str) -> bool:
        for fmt in self._allowed_formats:
            if text == self._default_template(fmt):
                return True
        return False

    def collect(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        result[self.spec.filename_key] = self._dir_edit.text()
        if self.spec.template_key:
            if self._template_edit and self._template_edit.text():
                result[self.spec.template_key] = self._template_edit.text()
            else:
                result[self.spec.template_key] = self._default_template(
                    self._format_combo.currentText())
        if self._dtype_combo is not None and self.spec.dtype_key:
            dtype = self._dtype_combo.currentText()
            if dtype:
                result[self.spec.dtype_key] = dtype
        return result

    def is_ready(self) -> bool:
        return bool(self._dir_edit.text())

    def missing_reason(self) -> str | None:
        if not self._dir_edit.text():
            return f"请选择{self.spec.label}目录"
        return None

    def apply_defaults(self, defaults: dict[str, Any]):
        fmt = defaults.get("output_format")
        if fmt and fmt in self._allowed_formats:
            self._format_combo.setCurrentText(fmt)
        dtype = defaults.get("output_dtype")
        if dtype and self._dtype_combo is not None:
            effective = self._effective_dtypes(self._format_combo.currentText())
            if dtype in effective:
                self._dtype_combo.setCurrentText(dtype)


class _VideoOutputRegion(QFrame):
    """Preset-driven video container, path and encoder controls."""

    changed = Signal()

    def __init__(self, spec: OutputSpec, parent=None):
        super().__init__(parent)
        self.spec = spec
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        self._allowed_formats = spec.formats or list(VIDEO_FORMAT_PRESETS)
        self._current_format = self._allowed_formats[0]
        self._path_cache = {}
        self._codec_cache = {}
        format_row, self._format_combo = _select_row("文件格式", self._allowed_formats, self)
        layout.addWidget(format_row)
        #self._description = QLabel(self)
        #self._description.setStyleSheet(LABEL_STYLE)
        #self._description.setWordWrap(True)
        #layout.addWidget(self._description)
        path_row, self._path_edit = _path_row(
            "视频文件", self, self._browse, on_change=self._on_path_change,
            read_only=False, placeholder="output.mp4")
        layout.addWidget(path_row)
        self._open_button = QPushButton("打开已导出视频", self)
        self._open_button.hide()
        self._result_path = None
        self._open_button.clicked.connect(self._open_result)
        layout.addWidget(self._open_button)
        self._getters = {}
        self._rows = {}
        for param in spec.params:
            row, getter, _setter = create_config_row(
                param, self, on_change=self._on_param_change)
            self._getters[param.key] = getter
            self._rows[param.key] = row
            if param.key == "video_codec":
                layout.insertWidget(1, row)
            else:
                layout.addWidget(row)
        self._format_combo.currentTextChanged.connect(self._on_format_changed)
        self._on_format_changed(self._current_format)

    def _on_format_changed(self, fmt):
        codec_getter = self._getters.get("video_codec", lambda: "h264")
        old_codec = codec_getter()
        self._codec_cache[self._current_format] = old_codec
        old_path = self._path_edit.text()
        if not old_path or detect_format_by_path(old_path, VIDEO_FORMAT_PRESETS) == self._current_format:
            self._path_cache[self._current_format] = old_path
        self._current_format = fmt
        preset = VIDEO_FORMAT_PRESETS[fmt]
        path = self._path_cache.get(fmt)
        if path is None:
            path = str(Path(old_path).with_suffix(preset["default_ext"])) if old_path else ""
        self._path_edit.blockSignals(True)
        self._path_edit.setText(path)
        self._path_edit.setPlaceholderText("output" + preset["default_ext"])
        self._path_edit.blockSignals(False)
        codec_row = self._rows.get("video_codec")
        if codec_row is not None:
            combo = codec_row.findChild(QComboBox)
            codec = self._codec_cache.get(fmt, old_codec)
            if codec not in preset["codecs"]:
                codec = preset["default_codec"]
            combo.blockSignals(True)
            combo.clear()
            for key in preset["codecs"]:
                combo.addItem(VIDEO_CODEC_PRESETS[key]["label"], key)
            combo.setCurrentIndex(combo.findData(codec))
            combo.setToolTip(combo.currentText())
            combo.blockSignals(False)
        self._update_visibility()
        self.changed.emit()

    def _on_path_change(self):
        path = self._path_edit.text()
        fmt = detect_format_by_path(path, VIDEO_FORMAT_PRESETS)
        if fmt in self._allowed_formats and fmt != self._current_format:
            self._path_cache[fmt] = path
            self._format_combo.setCurrentText(fmt)
        else:
            self.changed.emit()

    def _update_visibility(self):
        codec = self._getters.get("video_codec", lambda: "h264")()
        format_preset = VIDEO_FORMAT_PRESETS[self._current_format]
        codec_preset = VIDEO_CODEC_PRESETS[codec]
        supported = format_preset["params"] & codec_preset["params"]
        #self._description.setText(f"{format_preset['description']} · {codec_preset['description']}")
        for param in self.spec.params:
            self._rows[param.key].setEnabled(param.key in supported)
            visible = param.key in supported
            condition = param.visible_when
            if condition and "key" in condition and param.key in self._rows:
                getter = self._getters.get(condition["key"])
                visible = visible and getter is not None and getter() == condition.get("eq")
            self._rows[param.key].setVisible(visible)

    def _on_param_change(self):
        self._update_visibility()
        self.changed.emit()

    def _browse(self):
        filters = [f"{fmt} 视频 (*{VIDEO_FORMAT_PRESETS[fmt]['default_ext']})"
                   for fmt in self._allowed_formats]
        path, chosen_filter = QFileDialog.getSaveFileName(
            self, "导出视频", self._path_edit.text(), ";;".join(filters),
            filters[self._allowed_formats.index(self._current_format)])
        if path:
            fmt = chosen_filter.split(" ", 1)[0] if chosen_filter else self._current_format
            suffix = VIDEO_FORMAT_PRESETS[fmt]["default_ext"]
            if Path(path).suffix.lower() != suffix:
                path = str(Path(path).with_suffix(suffix))
            self._path_edit.setText(path)

    def _open_result(self):
        if self._result_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self._result_path))

    def set_result(self, receipt):
        self._result_path = receipt.get("path") if receipt else None
        self._open_button.setVisible(bool(self._result_path))

    def collect(self):
        codec = self._getters.get("video_codec", lambda: "h264")()
        supported = (VIDEO_FORMAT_PRESETS[self._current_format]["params"] &
                     VIDEO_CODEC_PRESETS[codec]["params"])
        return {self.spec.filename_key: self._path_edit.text(),
                **{key: getter() for key, getter in self._getters.items() if key in supported}}

    def is_ready(self):
        return self.missing_reason() is None

    def missing_reason(self):
        if not self._path_edit.text():
            return "请选择视频输出文件"
        fmt = detect_format_by_path(self._path_edit.text(), VIDEO_FORMAT_PRESETS)
        if fmt != self._current_format:
            return f"视频输出文件必须使用 {VIDEO_FORMAT_PRESETS[self._current_format]['default_ext']} 扩展名"
        return None

    def apply_defaults(self, defaults):
        pass


# ─── OutputPanel (container) ─────────────────────────────────────────────────


class _OutputGroup(QGroupBox):
    """Keep the checked indicator visible under the desktop's inherited QSS."""

    def paintEvent(self, event):
        super().paintEvent(event)
        option = QStyleOptionGroupBox()
        self.initStyleOption(option)
        rect = self.style().subControlRect(
            QStyle.CC_GroupBox, option, QStyle.SC_GroupBoxCheckBox, self).adjusted(1, 1, -1, -1)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        color = QColor("#218bb8") if self.isChecked() else QColor("#8b969e")
        painter.setPen(QPen(color, 1.2))
        painter.setBrush(color if self.isChecked() else QColor("white"))
        painter.drawRoundedRect(rect, 2, 2)
        if self.isChecked():
            painter.setPen(QPen(QColor("white"), 1.8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
            painter.drawLine(round(x + w * .22), round(y + h * .52),
                             round(x + w * .43), round(y + h * .73))
            painter.drawLine(round(x + w * .43), round(y + h * .73),
                             round(x + w * .80), round(y + h * .28))
        painter.end()


class OutputPanel(QWidget):
    """Container rendering one _OutputRegion or _SequenceOutputRegion per OutputSpec."""

    values_changed = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(4, 4, 4, 4)
        self._main_layout.setSpacing(4)
        self._main_layout.setAlignment(Qt.AlignTop)

        self._path_cache: dict[str, dict[str, str]] = {}
        self._regions: list[_OutputRegion] = []
        self._enable_groups: dict[str, QGroupBox] = {}

    def load_specs(self, specs: list[OutputSpec]):
        # Clear existing
        while self._main_layout.count():
            item = self._main_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self._regions.clear()
        self._enable_groups.clear()

        for spec in specs:
            if spec.type == "sequence":
                region = _SequenceOutputRegion(spec, parent=self)
            elif spec.type == "video":
                region = _VideoOutputRegion(spec, parent=self)
            else:
                region = _OutputRegion(spec, parent=self,
                                       path_cache=self._path_cache.setdefault(spec.filename_key, {}))
            region.changed.connect(self.values_changed.emit)
            if spec.enable_key:
                group = _OutputGroup(spec.label, self)
                font = group.font()
                font.setBold(True)
                group.setFont(font)
                region.setFont(self.font())
                group.setCheckable(True)
                group.setChecked(spec.enabled_default)
                group_layout = QVBoxLayout(group)
                # Reserve the title/checkbox line instead of placing content
                # underneath it. Account for the active font and DPI.
                group_layout.setContentsMargins(4, group.fontMetrics().height() + 12, 4, 8)
                group_layout.setSpacing(4)
                group_layout.addWidget(region)
                region.setVisible(spec.enabled_default)
                group.toggled.connect(region.setVisible)
                group.toggled.connect(lambda _checked: self.values_changed.emit())
                self._enable_groups[spec.enable_key] = group
                self._main_layout.addWidget(group)
            else:
                self._main_layout.addWidget(region)
            self._regions.append(region)

        self._main_layout.addStretch()

    def collect_outputs(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for region in self._regions:
            key = region.spec.enable_key
            if key:
                result[key] = self._enable_groups[key].isChecked()
                if not result[key]:
                    continue
            result.update(region.collect())
        return result

    def set_video_result(self, receipt):
        for region in self._regions:
            if isinstance(region, _VideoOutputRegion):
                region.set_result(receipt)

    def is_ready(self) -> bool:
        return self.missing_reason() is None

    def missing_reason(self) -> str | None:
        active = [region for region in self._regions
                  if not region.spec.enable_key or
                  self._enable_groups[region.spec.enable_key].isChecked()]
        if self._regions and not active:
            return "请至少选择一种输出"
        for region in active:
            reason = region.missing_reason()
            if reason:
                return reason
        return None

    def apply_defaults(self, defaults: dict[str, Any]):
        """Apply output initialization defaults to all regions."""
        for region in self._regions:
            region.apply_defaults(defaults)
