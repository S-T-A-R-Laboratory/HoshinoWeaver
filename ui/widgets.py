"""Widget factory: ConfigSpec/RouteSpec → Qt widgets for dynamic panel."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QSizePolicy, QSlider, QSpinBox, QVBoxLayout, QWidget,
)

from ui.UILibs import DoubleSlider
from ui.styles import COMBOBOX_STYLE, LABEL_STYLE, LINEEDIT_STYLE
from ui.transforms import apply_inverse


# ─── Data Specs ──────────────────────────────────────────────────────────────


@dataclass
class ConfigSpec:
    key: str
    type: str = "str"
    default: Any = None
    widget: str = "input"
    label: str = ""
    description: str = ""
    hidden: bool = False
    min: float | None = None
    max: float | None = None
    step: float | None = None
    options: list | None = None
    bind: str | None = None
    accept: str | None = None
    # Named transform applied to the widget value before sending to backend.
    # For range_slider: dict like {"left": "negate", "right": "complement"}.
    # For other numeric widgets: a single string like "negate".
    transform: Any = None
    # For range_slider: backend default of the bind partner (right-handle key).
    # Populated by PanelSchema after parsing so the widget can initialize the
    # right handle from meta.yaml rather than a hard-coded max.
    bind_default: Any = None


@dataclass
class RouteOptionSpec:
    key: str
    label: str = ""
    description: str = ""


@dataclass
class RouteSpec:
    key: str
    label: str = ""
    widget: str = "tabs"
    options: dict[str, RouteOptionSpec] = field(default_factory=dict)
    default: str = ""




# ─── Widget Factory ──────────────────────────────────────────────────────────


def create_config_row(
    spec: ConfigSpec,
    parent: QWidget | None = None,
    on_change: Callable | None = None,
) -> tuple[QFrame, Callable[[], Any], Callable[[Any], None]]:
    """Create a labeled row widget for a config spec.

    Returns (row_frame, get_value, set_value).
    """
    row = QFrame(parent)
    row.setMinimumHeight(36)
    row.setMaximumHeight(36)
    layout = QHBoxLayout(row)
    layout.setContentsMargins(5, 0, 5, 0)
    layout.setSpacing(6)

    label = QLabel(spec.label or spec.key, row)
    label.setMinimumWidth(60)
    label.setMaximumWidth(80)
    label.setStyleSheet(LABEL_STYLE)
    label.setToolTip(spec.description)
    layout.addWidget(label)

    getter: Callable[[], Any]
    setter: Callable[[Any], None]

    if spec.widget == "switch":
        w = QCheckBox(row)
        w.setChecked(bool(spec.default))
        if on_change:
            w.stateChanged.connect(lambda _: on_change())
        getter = w.isChecked
        setter = w.setChecked
        layout.addWidget(w)
        layout.addStretch()

    elif spec.widget == "slider":
        slider, val_label, getter, setter = _make_slider(spec, row, on_change)
        layout.addWidget(slider, 1)
        layout.addWidget(val_label)

    elif spec.widget == "range_slider":
        ds, getter, setter, left_label, right_label = _make_range_slider(spec, row, on_change)
        layout.addWidget(left_label)
        layout.addWidget(ds, 1)
        layout.addWidget(right_label)

    elif spec.widget == "select":
        combo = QComboBox(row)
        combo.setStyleSheet(COMBOBOX_STYLE)
        items: list[tuple[Any, str]] = spec.options or []
        for value, label in items:
            combo.addItem(label, value)
        values = [v for v, _ in items]
        if spec.default is not None and spec.default in values:
            combo.setCurrentIndex(values.index(spec.default))
        if on_change:
            combo.currentIndexChanged.connect(lambda _: on_change())
        getter = lambda: combo.currentData()
        setter = lambda v: combo.setCurrentIndex(values.index(v) if v in values else 0)
        layout.addWidget(combo, 1)

    elif spec.widget == "file_picker":
        line = QLineEdit(row)
        line.setReadOnly(True)
        line.setPlaceholderText("点击浏览...")
        line.setStyleSheet(LINEEDIT_STYLE)
        if spec.default:
            line.setText(str(spec.default))
        btn = QPushButton("...", row)
        btn.setMaximumWidth(28)
        btn.setMinimumWidth(28)

        def _browse():
            accept = spec.accept or ""
            path, _ = QFileDialog.getOpenFileName(
                row, spec.label or "选择文件", "",
                f"支持的文件 (*{' *'.join(accept.split(','))} );;全部文件 (*)" if accept else "全部文件 (*)")
            if path:
                line.setText(path)
                if on_change:
                    on_change()

        btn.clicked.connect(_browse)
        getter = line.text
        setter = line.setText
        layout.addWidget(line, 1)
        layout.addWidget(btn)

    elif spec.widget == "dir_picker":
        line = QLineEdit(row)
        line.setReadOnly(True)
        line.setPlaceholderText("留空使用系统默认...")
        line.setStyleSheet(LINEEDIT_STYLE)
        if spec.default:
            line.setText(str(spec.default))
        btn = QPushButton("...", row)
        btn.setMaximumWidth(28)
        btn.setMinimumWidth(28)

        def _browse_dir():
            path = QFileDialog.getExistingDirectory(
                row, spec.label or "选择目录", line.text() or "")
            if path:
                line.setText(path)
                if on_change:
                    on_change()

        btn.clicked.connect(_browse_dir)
        getter = line.text
        setter = lambda v: line.setText(str(v) if v else "")
        layout.addWidget(line, 1)
        layout.addWidget(btn)

    elif spec.widget == "input" and spec.type == "int":
        spin = QSpinBox(row)
        spin.setMinimum(int(spec.min) if spec.min is not None else -999999)
        spin.setMaximum(int(spec.max) if spec.max is not None else 999999)
        spin.setSingleStep(int(spec.step) if spec.step else 1)
        spin.setValue(int(spec.default) if spec.default is not None else 0)
        if on_change:
            spin.valueChanged.connect(lambda _: on_change())
        getter = spin.value
        setter = spin.setValue
        layout.addWidget(spin)
        layout.addStretch()

    elif spec.widget == "input" and spec.type == "float":
        spin = QDoubleSpinBox(row)
        spin.setMinimum(spec.min if spec.min is not None else -999999.0)
        spin.setMaximum(spec.max if spec.max is not None else 999999.0)
        spin.setSingleStep(spec.step if spec.step else 0.1)
        spin.setDecimals(3)
        spin.setValue(float(spec.default) if spec.default is not None else 0.0)
        if on_change:
            spin.valueChanged.connect(lambda _: on_change())
        getter = spin.value
        setter = spin.setValue
        layout.addWidget(spin)
        layout.addStretch()

    else:
        line = QLineEdit(row)
        line.setStyleSheet(LINEEDIT_STYLE)
        if spec.default is not None:
            line.setText(str(spec.default))
        if on_change:
            line.textChanged.connect(lambda _: on_change())
        getter = line.text
        setter = line.setText
        layout.addWidget(line, 1)

    return row, getter, setter


def create_route_selector(
    spec: RouteSpec,
    parent: QWidget | None = None,
    on_changed: Callable[[str], None] | None = None,
) -> tuple[QFrame, Callable[[], str], Callable[[str], None]]:
    """Create a route selector as a labeled combobox.

    Returns (widget, get_selected_option, set_selected_option).
    """
    frame = QFrame(parent)
    frame.setMinimumHeight(44)
    frame.setMaximumHeight(44)
    layout = QHBoxLayout(frame)
    layout.setContentsMargins(5, 4, 5, 4)
    layout.setSpacing(8)

    title = QLabel(spec.label or spec.key, frame)
    title.setMinimumWidth(80)
    title.setMaximumWidth(80)
    title.setStyleSheet("font-weight: bold; font-size: 11px; color: rgba(50,50,50,220);")
    layout.addWidget(title)

    combo = QComboBox(frame)
    combo.setStyleSheet(COMBOBOX_STYLE)
    for opt_key, opt_spec in spec.options.items():
        combo.addItem(opt_spec.label or opt_key, opt_key)
        idx = combo.count() - 1
        combo.setItemData(idx, opt_spec.description, Qt.ToolTipRole)
    if spec.default and spec.default in spec.options:
        idx = list(spec.options.keys()).index(spec.default)
        combo.setCurrentIndex(idx)
    if on_changed:
        combo.currentIndexChanged.connect(
            lambda idx: on_changed(combo.itemData(idx)))
    getter = lambda: combo.currentData()
    setter = lambda v: combo.setCurrentIndex(
        list(spec.options.keys()).index(v) if v in spec.options else 0)
    layout.addWidget(combo, 1)

    return frame, getter, setter


# ─── Internal Helpers ────────────────────────────────────────────────────────


def _make_slider(
    spec: ConfigSpec, parent: QWidget, on_change: Callable | None
) -> tuple[QSlider, QLabel, Callable, Callable]:
    """Create a QSlider + value label for int/float config."""
    is_float = spec.type == "float"
    step = spec.step if spec.step else (0.1 if is_float else 1)
    multiplier = int(1.0 / step) if step < 1 else 1

    mn = spec.min if spec.min is not None else 0
    mx = spec.max if spec.max is not None else 100
    default = spec.default if spec.default is not None else mn

    slider = QSlider(Qt.Horizontal, parent)
    slider.setMinimum(int(mn * multiplier))
    slider.setMaximum(int(mx * multiplier))
    slider.setSingleStep(int(step * multiplier))
    slider.setValue(int(default * multiplier))
    slider.setMinimumWidth(40)

    val_label = QLabel(parent)
    val_label.setMinimumWidth(28)
    val_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

    def _update_label(v):
        quant_v = round((v - mn) / step) * step + mn
        slider.setValue(quant_v)
        if is_float:
            val_label.setText(f"{quant_v / multiplier:.2f}")
        else:
            val_label.setText(str(quant_v))
        if on_change:
            on_change()

    slider.valueChanged.connect(_update_label)
    _update_label(slider.value())

    def getter():
        v = slider.value()
        return v / multiplier if is_float else v

    def setter(val):
        slider.setValue(int(val * multiplier) if is_float else int(val))

    return slider, val_label, getter, setter


def _make_range_slider(
    spec: ConfigSpec, parent: QWidget, on_change: Callable | None
) -> tuple[DoubleSlider, Callable, Callable, QLabel, QLabel]:
    """Create a DoubleSlider for range_slider widget type."""
    mn = spec.min if spec.min is not None else 0
    mx = spec.max if spec.max is not None else 100
    step = spec.step if spec.step else 1
    default = spec.default if spec.default is not None else mn

    # Resolve UI-domain initial values via transform inverses, so meta.yaml
    # defaults (backend domain) drive both handles. Non-invertible transforms
    # (e.g. abs) on a range_slider are unsupported: split into two sliders
    # with their own defaults instead.
    tr = spec.transform if isinstance(spec.transform, dict) else {}
    left_ui = apply_inverse(tr.get("left"), default)
    if left_ui is None:
        raise ValueError(
            f"range_slider '{spec.key}': transform.left={tr.get('left')!r} is not invertible; "
            "split this into separate sliders instead of using a range_slider."
        )
    if spec.bind:
        right_backend = spec.bind_default if spec.bind_default is not None else mx
        right_ui = apply_inverse(tr.get("right"), right_backend)
        if right_ui is None:
            raise ValueError(
                f"range_slider '{spec.key}': transform.right={tr.get('right')!r} is not invertible; "
                "split this into separate sliders instead of using a range_slider."
            )
    else:
        right_ui = left_ui

    ds = DoubleSlider(parent)
    ds.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    multiplier = int(1.0 / step) if step < 1 else 1
    ds.min_value = int(mn * multiplier)
    ds.max_value = int(mx * multiplier)
    ds.left_value = int(left_ui * multiplier)
    ds.right_value = int(right_ui * multiplier)
    ds.update_slider()

    _val_style = "font-size: 10px; color: rgba(60,60,60,200); min-width: 20px;"
    left_label = QLabel(parent)
    left_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    left_label.setStyleSheet(_val_style)
    right_label = QLabel(parent)
    right_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    right_label.setStyleSheet(_val_style)

    def _update_labels(l, r):
        left_label.setText(f"{l / multiplier:.2f}")
        right_label.setText(f"{r / multiplier:.2f}")
        if on_change:
            on_change()

    ds.valueChanged.connect(_update_labels)
    _update_labels(ds.left_value, ds.right_value)

    def getter():
        return ds.left_value / multiplier, ds.right_value / multiplier

    def setter(vals):
        if isinstance(vals, (list, tuple)) and len(vals) == 2:
            ds.left_value = int(vals[0] * multiplier)
            ds.right_value = int(vals[1] * multiplier)
            ds.update_slider()
        _update_labels(ds.left_value, ds.right_value)

    return ds, getter, setter, left_label, right_label
