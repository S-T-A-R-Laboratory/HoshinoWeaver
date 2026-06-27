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

from typing import Any, Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QSizePolicy, QSlider, QVBoxLayout, QWidget,
)

from ui.output_presets import (
    IMAGE_FORMAT_PRESETS, all_format_keys, detect_format_by_path,
)
from ui.panel_builder import OutputSpec
from ui.styles import (
    COMBOBOX_STYLE, GROUP_HEADER_STYLE, LABEL_STYLE, LINEEDIT_STYLE,
)


# ─── Single output region ────────────────────────────────────────────────────


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
        if spec.label:
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
        row_fmt = QFrame(self)
        row_fmt_l = QHBoxLayout(row_fmt)
        row_fmt_l.setContentsMargins(5, 0, 5, 0)
        row_fmt_l.setSpacing(6)
        fmt_label = QLabel("文件格式", row_fmt)
        fmt_label.setMinimumWidth(60)
        fmt_label.setMaximumWidth(80)
        fmt_label.setStyleSheet(LABEL_STYLE)
        row_fmt_l.addWidget(fmt_label)
        self._format_combo = QComboBox(row_fmt)
        self._format_combo.setStyleSheet(COMBOBOX_STYLE)
        for fmt in self._allowed_formats:
            self._format_combo.addItem(fmt)
        self._format_combo.currentTextChanged.connect(self._on_format_changed)
        row_fmt_l.addWidget(self._format_combo, 1)
        outer.addWidget(row_fmt)

        # Row 2: path + browse
        row_path = QFrame(self)
        row_path_l = QHBoxLayout(row_path)
        row_path_l.setContentsMargins(5, 0, 5, 0)
        row_path_l.setSpacing(6)
        path_label = QLabel("输出路径", row_path)
        path_label.setMinimumWidth(60)
        path_label.setMaximumWidth(80)
        path_label.setStyleSheet(LABEL_STYLE)
        row_path_l.addWidget(path_label)
        self._path_edit = QLineEdit(row_path)
        self._path_edit.setReadOnly(True)
        self._path_edit.setStyleSheet(LINEEDIT_STYLE)
        row_path_l.addWidget(self._path_edit, 1)
        browse_btn = QPushButton("...", row_path)
        browse_btn.setMaximumWidth(28)
        browse_btn.clicked.connect(self._on_browse)
        row_path_l.addWidget(browse_btn)
        outer.addWidget(row_path)

        # Row 3: dtype selector (optional)
        self._dtype_combo: QComboBox | None = None
        if spec.dtype_key:
            row_dtype = QFrame(self)
            row_dtype_l = QHBoxLayout(row_dtype)
            row_dtype_l.setContentsMargins(5, 0, 5, 0)
            row_dtype_l.setSpacing(6)
            dtype_label = QLabel("输出位深", row_dtype)
            dtype_label.setMinimumWidth(60)
            dtype_label.setMaximumWidth(80)
            dtype_label.setStyleSheet(LABEL_STYLE)
            row_dtype_l.addWidget(dtype_label)
            self._dtype_combo = QComboBox(row_dtype)
            self._dtype_combo.setStyleSheet(COMBOBOX_STYLE)
            self._dtype_combo.currentTextChanged.connect(lambda _: self.changed.emit())
            row_dtype_l.addWidget(self._dtype_combo, 1)
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

        if spec.label:
            header = QLabel(spec.label, self)
            header.setStyleSheet(GROUP_HEADER_STYLE)
            outer.addWidget(header)

        # Row 1: format selector
        allowed_formats = spec.formats or list(IMAGE_FORMAT_PRESETS.keys())
        self._allowed_formats = [
            f for f in allowed_formats if f in IMAGE_FORMAT_PRESETS
        ]

        row_fmt = QFrame(self)
        row_fmt_l = QHBoxLayout(row_fmt)
        row_fmt_l.setContentsMargins(5, 0, 5, 0)
        row_fmt_l.setSpacing(6)
        fmt_label = QLabel("文件格式", row_fmt)
        fmt_label.setMinimumWidth(60)
        fmt_label.setMaximumWidth(80)
        fmt_label.setStyleSheet(LABEL_STYLE)
        row_fmt_l.addWidget(fmt_label)
        self._format_combo = QComboBox(row_fmt)
        self._format_combo.setStyleSheet(COMBOBOX_STYLE)
        for fmt in self._allowed_formats:
            self._format_combo.addItem(fmt)
        self._format_combo.currentTextChanged.connect(self._on_format_changed)
        row_fmt_l.addWidget(self._format_combo, 1)
        outer.addWidget(row_fmt)

        # Row 2: output directory + browse
        row_dir = QFrame(self)
        row_dir_l = QHBoxLayout(row_dir)
        row_dir_l.setContentsMargins(5, 0, 5, 0)
        row_dir_l.setSpacing(6)
        dir_label = QLabel("输出目录", row_dir)
        dir_label.setMinimumWidth(60)
        dir_label.setMaximumWidth(80)
        dir_label.setStyleSheet(LABEL_STYLE)
        row_dir_l.addWidget(dir_label)
        self._dir_edit = QLineEdit(row_dir)
        self._dir_edit.setReadOnly(True)
        self._dir_edit.setStyleSheet(LINEEDIT_STYLE)
        row_dir_l.addWidget(self._dir_edit, 1)
        browse_btn = QPushButton("...", row_dir)
        browse_btn.setMaximumWidth(28)
        browse_btn.clicked.connect(self._on_browse_dir)
        row_dir_l.addWidget(browse_btn)
        outer.addWidget(row_dir)

        # Row 3: template name (editable)
        self._template_edit: QLineEdit | None = None
        if spec.template_key:
            row_tpl = QFrame(self)
            row_tpl_l = QHBoxLayout(row_tpl)
            row_tpl_l.setContentsMargins(5, 0, 5, 0)
            row_tpl_l.setSpacing(6)
            tpl_label = QLabel("文件名模板", row_tpl)
            tpl_label.setMinimumWidth(60)
            tpl_label.setMaximumWidth(80)
            tpl_label.setStyleSheet(LABEL_STYLE)
            row_tpl_l.addWidget(tpl_label)
            self._template_edit = QLineEdit(row_tpl)
            self._template_edit.setStyleSheet(LINEEDIT_STYLE)
            self._template_edit.setPlaceholderText("frame_{index:05d}.png")
            self._template_edit.textChanged.connect(lambda _: self.changed.emit())
            row_tpl_l.addWidget(self._template_edit, 1)
            outer.addWidget(row_tpl)

        # Row 4: dtype selector
        self._dtype_combo: QComboBox | None = None
        if spec.dtype_key:
            row_dtype = QFrame(self)
            row_dtype_l = QHBoxLayout(row_dtype)
            row_dtype_l.setContentsMargins(5, 0, 5, 0)
            row_dtype_l.setSpacing(6)
            dtype_label = QLabel("输出位深", row_dtype)
            dtype_label.setMinimumWidth(60)
            dtype_label.setMaximumWidth(80)
            dtype_label.setStyleSheet(LABEL_STYLE)
            row_dtype_l.addWidget(dtype_label)
            self._dtype_combo = QComboBox(row_dtype)
            self._dtype_combo.setStyleSheet(COMBOBOX_STYLE)
            self._dtype_combo.currentTextChanged.connect(lambda _: self.changed.emit())
            row_dtype_l.addWidget(self._dtype_combo, 1)
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


# ─── OutputPanel (container) ─────────────────────────────────────────────────


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

    def load_specs(self, specs: list[OutputSpec]):
        # Clear existing
        while self._main_layout.count():
            item = self._main_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self._regions.clear()

        for spec in specs:
            if spec.type == "sequence":
                region = _SequenceOutputRegion(spec, parent=self)
            else:
                region = _OutputRegion(spec, parent=self,
                                       path_cache=self._path_cache.setdefault(spec.filename_key, {}))
            region.changed.connect(self.values_changed.emit)
            self._main_layout.addWidget(region)
            self._regions.append(region)

        self._main_layout.addStretch()

    def collect_outputs(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for region in self._regions:
            result.update(region.collect())
        return result

    def is_ready(self) -> bool:
        return all(region.is_ready() for region in self._regions)

    def missing_reason(self) -> str | None:
        for region in self._regions:
            reason = region.missing_reason()
            if reason:
                return reason
        return None

    def apply_defaults(self, defaults: dict[str, Any]):
        """Apply output initialization defaults to all regions."""
        for region in self._regions:
            region.apply_defaults(defaults)
