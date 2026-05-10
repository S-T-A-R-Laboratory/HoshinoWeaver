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


# ─── Styles (align with existing output tab look) ───────────────────────────

_LINEEDIT_STYLE = (
    "QLineEdit {"
    "  border: 1px solid rgba(220,220,220,200);"
    "  border-radius: 3px;"
    "  background-color: rgba(250,250,250,200);"
    "  padding: 2px 4px;"
    "  font-size: 11px;"
    "}"
)

_COMBOBOX_STYLE = (
    "QComboBox {"
    "  border: 0px solid rgba(199,199,199,100);"
    "  padding: 3px;"
    "  border-radius: 3px;"
    "  font-size: 11px;"
    "  color: rgba(30,30,30,200);"
    "}"
    "QComboBox::hover { background-color: rgba(0, 212, 254, 10); }"
    "QComboBox QAbstractItemView {"
    "  background-color: rgba(255,255,255,1);"
    "  border: 0px solid rgba(199,199,199,100);"
    "}"
)

_LABEL_STYLE = "font-size: 11px; color: rgba(30,30,30,200);"


# ─── Single output region ────────────────────────────────────────────────────


class _OutputRegion(QFrame):
    """Renders one output (format selector + path + dtype + format params)."""

    changed = Signal()

    def __init__(self, spec: OutputSpec, parent: QWidget | None = None):
        super().__init__(parent)
        self.spec = spec
        self._format_param_widgets: dict[str, tuple[QFrame, Callable[[], Any]]] = {}
        self._path_cache: dict[str, str] = {}
        self._current_format: str | None = None

        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        # Label header
        if spec.label:
            header = QLabel(spec.label, self)
            header.setStyleSheet(
                "font-weight: bold; font-size: 11px;"
                "color: rgba(255,255,255,220);"
                "background-color: rgba(90,90,90,200);"
                "border-top-left-radius: 3px;"
                "border-top-right-radius: 3px;"
                "padding: 3px 6px;"
            )
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
        fmt_label.setStyleSheet(_LABEL_STYLE)
        row_fmt_l.addWidget(fmt_label)
        self._format_combo = QComboBox(row_fmt)
        self._format_combo.setStyleSheet(_COMBOBOX_STYLE)
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
        path_label.setStyleSheet(_LABEL_STYLE)
        row_path_l.addWidget(path_label)
        self._path_edit = QLineEdit(row_path)
        self._path_edit.setReadOnly(True)
        self._path_edit.setStyleSheet(_LINEEDIT_STYLE)
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
            dtype_label.setStyleSheet(_LABEL_STYLE)
            row_dtype_l.addWidget(dtype_label)
            self._dtype_combo = QComboBox(row_dtype)
            self._dtype_combo.setStyleSheet(_COMBOBOX_STYLE)
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
        label.setStyleSheet(_LABEL_STYLE)
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
        else:
            # Fallback: label + value only (YAGNI — other widgets not needed yet)
            lbl = QLabel(str(default), row)
            layout.addWidget(lbl, 1)
            getter = lambda v=default: v

        # Attach owner format so we can toggle visibility
        row.setProperty("owner_fmt", owner_fmt)
        row.setProperty("config_key", config_key)
        self._format_param_widgets[preset_param] = (row, getter)
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
        for preset_param, (row, _getter) in self._format_param_widgets.items():
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

        for preset_param, (row, getter) in self._format_param_widgets.items():
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


# ─── OutputPanel (container) ─────────────────────────────────────────────────


class OutputPanel(QWidget):
    """Container rendering one _OutputRegion per OutputSpec."""

    values_changed = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(4, 4, 4, 4)
        self._main_layout.setSpacing(4)
        self._main_layout.setAlignment(Qt.AlignTop)

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
            region = _OutputRegion(spec, parent=self)
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
