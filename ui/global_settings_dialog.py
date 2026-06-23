"""GlobalSettingsDialog: UI for editing hoshicore/default_settings.yaml."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import yaml
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialogButtonBox, QFileDialog,
    QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

from ui.UILibs import uQDialog
from ui.styles import COMBOBOX_STYLE, LABEL_STYLE, LINEEDIT_STYLE

_HOSHICORE_ROOT = Path(__file__).parent.parent / "hoshicore"
_SETTINGS_PATH = _HOSHICORE_ROOT / "default_settings.yaml"
_SETTINGS_UI_PATH = _HOSHICORE_ROOT / "dag" / "global_settings.ui.yaml"


class GlobalSettingsDialog(uQDialog):
    """对话框：逐条管理 default_settings.yaml 中的全局设置。

    每行布局：[☑ 启用] [label] [value_widget]
    勾选启用时，value_widget 可编辑；取消勾选时变灰。
    确定时将当前状态写回 default_settings.yaml。
    """

    def __init__(self, parent=None,
                 settings_path: Path | None = None,
                 ui_yaml_path: Path | None = None):
        super().__init__(parent)
        self._settings_path = settings_path or _SETTINGS_PATH
        self._ui_yaml_path = ui_yaml_path or _SETTINGS_UI_PATH
        # (key, enabled_checkbox, value_getter, value_setter)
        self._rows: list[tuple[str, QCheckBox, Callable, Callable]] = []

        self.setWindowTitle("全局设置")
        self.setMinimumWidth(500)
        self.setMinimumHeight(300)

        self._build_skeleton()
        self._load_and_render()
        self._center()

    def _center(self):
        parent = self.parent()
        if isinstance(parent, QWidget):
            geo = parent.geometry()
            self.adjustSize()
            self.move(
                geo.center().x() - self.width() // 2,
                geo.center().y() - self.height() // 2,
            )

    # ─── UI skeleton ────────────────────────────────────────────────────────

    def _build_skeleton(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 8)
        root.setSpacing(8)

        # Scroll area for the rows
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(4)
        self._content_layout.addStretch()
        scroll.setWidget(self._content)
        root.addWidget(scroll, 1)

        # Button box
        btn_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        btn_box.button(QDialogButtonBox.Ok).setText("确定")
        btn_box.button(QDialogButtonBox.Cancel).setText("取消")
        btn_box.accepted.connect(self._on_accept)
        btn_box.rejected.connect(self.reject)
        root.addWidget(btn_box)

    # ─── Loading & rendering ────────────────────────────────────────────────

    def _load_and_render(self):
        # Load settings values
        if self._settings_path.exists():
            with open(self._settings_path, "r", encoding="utf-8") as f:
                raw_settings: dict[str, Any] = yaml.safe_load(f) or {}
        else:
            raw_settings = {}

        # Load UI hints
        if self._ui_yaml_path.exists():
            with open(self._ui_yaml_path, "r", encoding="utf-8") as f:
                ui_hints: dict[str, Any] = yaml.safe_load(f) or {}
        else:
            ui_hints = {}

        # Determine ordering: ui_hints order first, then remaining settings keys
        keys = list(ui_hints.keys())
        for k in raw_settings:
            if k not in keys:
                keys.append(k)

        self._rows.clear()
        # Insert rows before the stretch
        stretch_idx = self._content_layout.count() - 1
        for key in keys:
            hint = ui_hints.get(key, {})

            # Group header: render a separator label, no settings row
            if hint.get("type") == "group_header":
                sep = self._make_group_header(hint.get("label", key))
                self._content_layout.insertWidget(stretch_idx, sep)
                stretch_idx += 1
                continue

            entry = raw_settings.get(key, {})
            if not isinstance(entry, dict):
                continue
            enabled = bool(entry.get("enabled", False))
            value = entry.get("value")
            row_widget, cb, getter, setter = self._make_row(key, hint, value, enabled)
            self._content_layout.insertWidget(stretch_idx, row_widget)
            stretch_idx += 1
            self._rows.append((key, cb, getter, setter))

    def _make_group_header(self, label: str) -> QWidget:
        """Render a visual section separator label."""
        w = QLabel(label, self._content)
        w.setStyleSheet(
            "font-weight: bold; font-size: 11px;"
            "color: rgba(255,255,255,220);"
            "background-color: rgba(90,90,90,200);"
            "border-radius: 3px;"
            "padding: 3px 6px;"
            "margin-top: 4px;"
        )
        return w

    def _make_row(
        self,
        key: str,
        hint: dict,
        value: Any,
        enabled: bool,
    ) -> tuple[QWidget, QCheckBox, Callable, Callable]:
        """Build one settings row. Returns (row_widget, enable_cb, getter, setter)."""
        row = QFrame(self._content)
        row.setMinimumHeight(36)
        row.setMaximumHeight(36)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(6)

        # Enable checkbox
        cb = QCheckBox(row)
        cb.setChecked(enabled)
        cb.setToolTip("启用后此设置将在所有管线启动时生效")
        layout.addWidget(cb)

        # Label
        label_text = hint.get("label", key)
        description = hint.get("description", "")
        lbl = QLabel(label_text, row)
        lbl.setMinimumWidth(130)
        lbl.setMaximumWidth(160)
        lbl.setStyleSheet(LABEL_STYLE)
        if description:
            lbl.setToolTip(description)
        layout.addWidget(lbl)

        # Value widget
        widget_type = hint.get("widget", "input")
        getter, setter, value_widgets = self._make_value_widget(
            key, widget_type, hint, value, row)
        for w in value_widgets:
            layout.addWidget(w)
        if widget_type == "switch":
            layout.addStretch()

        # Wire checkbox → enable/disable value widgets
        def _update_enabled(checked: bool, widgets=value_widgets):
            for w in widgets:
                w.setEnabled(checked)

        cb.checkStateChanged.connect(lambda state: _update_enabled(state == Qt.Checked))
        _update_enabled(enabled)

        return row, cb, getter, setter

    def _make_value_widget(
        self,
        key: str,
        widget_type: str,
        hint: dict,
        value: Any,
        parent: QWidget,
    ) -> tuple[Callable, Callable, list[QWidget]]:
        """Create value widget(s). Returns (getter, setter, widget_list)."""

        if widget_type == "switch":
            w = QCheckBox(parent)
            w.setChecked(bool(value))
            return w.isChecked, w.setChecked, [w]

        elif widget_type == "select":
            combo = QComboBox(parent)
            combo.setStyleSheet(COMBOBOX_STYLE)
            combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            raw_opts = hint.get("options", [])
            # options: list of str OR list of {value, label}
            items: list[tuple[Any, str]] = []
            for opt in raw_opts:
                if isinstance(opt, dict):
                    items.append((opt.get("value"), str(opt.get("label", opt.get("value", "")))))
                else:
                    items.append((opt, str(opt)))
            for val, lbl in items:
                combo.addItem(lbl, val)
            values = [v for v, _ in items]
            if value is not None and value in values:
                combo.setCurrentIndex(values.index(value))
            getter = lambda: combo.currentData()
            setter = lambda v: combo.setCurrentIndex(values.index(v) if v in values else 0)
            return getter, setter, [combo]

        elif widget_type == "slider":
            from PySide6.QtWidgets import QSlider
            from PySide6.QtCore import Qt as _Qt
            mn = int(hint.get("min", 0))
            mx = int(hint.get("max", 100))
            step = int(hint.get("step", 1))
            init = int(value) if value is not None else mn
            slider = QSlider(_Qt.Horizontal, parent)
            slider.setMinimum(mn)
            slider.setMaximum(mx)
            slider.setSingleStep(step)
            slider.setValue(init)
            slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            val_lbl = QLabel(str(init), parent)
            val_lbl.setMinimumWidth(28)
            slider.valueChanged.connect(lambda v: val_lbl.setText(str(v)))
            getter = slider.value
            setter = lambda v, s=slider: s.setValue(int(v))
            return getter, setter, [slider, val_lbl]

        elif widget_type == "dir_picker":
            line = QLineEdit(parent)
            line.setReadOnly(True)
            line.setPlaceholderText("留空使用系统默认...")
            line.setStyleSheet(LINEEDIT_STYLE)
            line.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            if value:
                line.setText(str(value))
            btn = QPushButton("...", parent)
            btn.setFixedWidth(28)

            def _browse():
                path = QFileDialog.getExistingDirectory(
                    parent, hint.get("label", "选择目录"), line.text() or "")
                if path:
                    line.setText(path)

            btn.clicked.connect(_browse)
            getter = line.text
            setter = lambda v: line.setText(str(v) if v else "")
            return getter, setter, [line, btn]

        else:
            # Fallback: plain text input
            line = QLineEdit(parent)
            line.setStyleSheet(LINEEDIT_STYLE)
            line.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            if value is not None:
                line.setText(str(value))
            return line.text, line.setText, [line]

    # ─── Collect & save ──────────────────────────────────────────────────────

    def collect(self) -> dict[str, dict]:
        """Return {key: {enabled, value}} reflecting current UI state."""
        result: dict[str, dict] = {}
        for key, cb, getter, _ in self._rows:
            enabled = cb.isChecked()
            raw_val = getter()
            # Normalise empty string → None for path fields
            value = None if raw_val == "" else raw_val
            result[key] = {"enabled": enabled, "value": value}
        return result

    def save(self):
        """Write current state back to default_settings.yaml."""
        header_lines: list[str] = []
        existing: dict[str, Any] = {}
        if self._settings_path.exists():
            with open(self._settings_path, "r", encoding="utf-8") as f:
                raw = f.read()
            for line in raw.splitlines(keepends=True):
                if line.startswith("#") or line.strip() == "":
                    header_lines.append(line)
                else:
                    break
            existing = yaml.safe_load(raw) or {}
        header = "".join(header_lines)

        existing.update(self.collect())
        body = yaml.dump(existing, allow_unicode=True, default_flow_style=False, sort_keys=False)
        with open(self._settings_path, "w", encoding="utf-8") as f:
            f.write(header)
            f.write(body)

    def _on_accept(self):
        self.save()
        self.accept()
