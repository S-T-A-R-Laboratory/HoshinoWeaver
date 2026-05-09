"""PanelSchema loader + DynamicConfigPanel: data-driven config UI from YAML."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable

import yaml
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame, QLabel, QLayout, QScrollArea, QSizePolicy,
    QVBoxLayout, QWidget,
)

from ui.widgets import (
    ConfigSpec, RouteOptionSpec, RouteSpec,
    create_config_row, create_route_selector,
)


# ─── PanelSchema ─────────────────────────────────────────────────────────────


@dataclass
class PanelSchema:
    meta_yaml_path: str
    routes: dict[str, RouteSpec] = field(default_factory=dict)
    configs: list[ConfigSpec] = field(default_factory=list)
    route_configs: dict[str, dict[str, list[ConfigSpec]]] = field(default_factory=dict)
    layout: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, meta_path: str, ui_path: str | None = None) -> "PanelSchema":
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = yaml.safe_load(f)

        ui: dict = {}
        if ui_path and os.path.isfile(ui_path):
            with open(ui_path, "r", encoding="utf-8") as f:
                ui = yaml.safe_load(f) or {}

        schema = cls(meta_yaml_path=meta_path)
        schema._parse_routes(meta, ui)
        schema._parse_configs(meta, ui)
        schema._parse_route_configs(meta, ui)
        schema.layout = ui.get("layout", {})
        return schema

    def _parse_routes(self, meta: dict, ui: dict):
        for route_key, route_def in meta.get("routes", {}).items():
            ui_route = ui.get("routes", {}).get(route_key, {})
            options: dict[str, RouteOptionSpec] = {}
            option_keys = list(route_def.get("options", {}).keys())
            ui_options = ui_route.get("options", {})
            for opt_key in option_keys:
                ui_opt = ui_options.get(opt_key, {})
                options[opt_key] = RouteOptionSpec(
                    key=opt_key,
                    label=ui_opt.get("label", opt_key),
                    description=ui_opt.get("description", ""),
                )
            self.routes[route_key] = RouteSpec(
                key=route_key,
                label=ui_route.get("label", route_key),
                widget=ui_route.get("widget", "tabs"),
                options=options,
                default=route_def.get("default", option_keys[0] if option_keys else ""),
            )

    def _parse_configs(self, meta: dict, ui: dict):
        for key, spec in meta.get("configs", {}).items():
            ui_cfg = ui.get("configs", {}).get(key, {})
            self.configs.append(_build_config_spec(key, spec, ui_cfg))

    def _parse_route_configs(self, meta: dict, ui: dict):
        meta_rc = meta.get("route_configs", {})
        ui_rc = ui.get("route_configs", {})

        if meta_rc:
            for route_key, options_dict in meta_rc.items():
                self.route_configs[route_key] = {}
                for option_key, params in options_dict.items():
                    specs: list[ConfigSpec] = []
                    ui_option = ui_rc.get(option_key, {})
                    for param_key, param_spec in params.items():
                        ui_param = ui_option.get(param_key, {})
                        specs.append(_build_config_spec(param_key, param_spec, ui_param))
                    self.route_configs[route_key][option_key] = specs
        elif ui_rc:
            # meta has no top-level route_configs but ui.yaml defines them.
            # Map ui route_configs (keyed by option_key) to route_keys.
            for route_key in self.routes:
                route_spec = self.routes[route_key]
                self.route_configs[route_key] = {}
                for option_key in route_spec.options:
                    ui_option = ui_rc.get(option_key, {})
                    if not ui_option:
                        continue
                    specs: list[ConfigSpec] = []
                    for param_key, ui_param in ui_option.items():
                        meta_spec = {"type": ui_param.get("type", "float"),
                                     "default": ui_param.get("default")}
                        specs.append(_build_config_spec(param_key, meta_spec, ui_param))
                    self.route_configs[route_key][option_key] = specs


def _build_config_spec(key: str, meta_spec: dict, ui_spec: dict) -> ConfigSpec:
    typ = meta_spec.get("type", "str")
    default = meta_spec.get("default")

    widget = ui_spec.get("widget", _infer_widget(typ))
    hidden = ui_spec.get("hidden", False)

    return ConfigSpec(
        key=key,
        type=typ,
        default=default,
        widget=widget,
        label=ui_spec.get("label", key),
        description=ui_spec.get("description", ""),
        hidden=hidden,
        min=ui_spec.get("min"),
        max=ui_spec.get("max"),
        step=ui_spec.get("step"),
        options=ui_spec.get("options"),
        bind=ui_spec.get("bind"),
    )


def _infer_widget(typ: str) -> str:
    return {
        "bool": "switch",
        "int": "input",
        "float": "input",
        "str": "input",
        "dict": "hidden",
        "list": "hidden",
        "image": "hidden",
    }.get(typ, "input")


# ─── DynamicConfigPanel ──────────────────────────────────────────────────────


class DynamicConfigPanel(QWidget):
    """Dynamically renders config controls based on a PanelSchema."""

    values_changed = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(4, 4, 4, 4)
        self._main_layout.setSpacing(2)
        self._main_layout.setAlignment(Qt.AlignTop)

        self._schema: PanelSchema | None = None
        self._config_getters: dict[str, Callable] = {}
        self._route_getters: dict[str, Callable] = {}
        self._route_config_getters: dict[str, Callable] = {}
        self._route_config_container: dict[str, QWidget] = {}
        self._bound_pairs: dict[str, str] = {}

    def load_schema(self, schema: PanelSchema):
        self._schema = schema
        self._config_getters.clear()
        self._route_getters.clear()
        self._route_config_getters.clear()
        self._route_config_container.clear()
        self._bound_pairs.clear()

        _clear_layout(self._main_layout)

        self._render_routes()
        self._render_configs()
        self._render_route_configs()

        self._main_layout.addStretch()

    def collect_route_choices(self) -> dict[str, str]:
        return {key: getter() for key, getter in self._route_getters.items()}

    # ── Rendering ────────────────────────────────────────────────────────────

    def _render_routes(self):
        for route_key, route_spec in self._schema.routes.items():
            frame, getter, setter = create_route_selector(
                route_spec, parent=self,
                on_changed=lambda opt, rk=route_key: self._on_route_changed(rk, opt),
            )
            self._main_layout.addWidget(frame)
            self._route_getters[route_key] = getter

    def _render_configs(self):
        layout_info = self._schema.layout
        groups = layout_info.get("groups", [])
        order = layout_info.get("order", [])

        rendered_keys: set[str] = set()

        # Keys that are route selectors (skip from config rendering)
        route_keys = set(self._schema.routes.keys())

        if groups:
            for group in groups:
                group_name = group.get("name", "")
                group_keys = group.get("keys", [])
                specs_in_group = [
                    s for s in self._schema.configs
                    if s.key in group_keys and not s.hidden and s.key not in route_keys
                ]
                if not specs_in_group:
                    continue
                self._add_group_header(group_name)
                for spec in specs_in_group:
                    self._add_config_widget(spec)
                    rendered_keys.add(spec.key)

        remaining = [
            s for s in self._schema.configs
            if s.key not in rendered_keys and not s.hidden and s.key not in route_keys
        ]
        if order:
            key_order = [k for k in order if k not in route_keys]
            remaining.sort(key=lambda s: key_order.index(s.key) if s.key in key_order else 999)

        for spec in remaining:
            self._add_config_widget(spec)

    def _render_route_configs(self):
        for route_key in self._schema.route_configs:
            container = QWidget(self)
            container_layout = QVBoxLayout(container)
            container_layout.setContentsMargins(0, 0, 0, 0)
            container_layout.setSpacing(0)
            self._main_layout.addWidget(container)
            self._route_config_container[route_key] = container
            current_option = self._route_getters[route_key]()
            self._rebuild_route_config_section(route_key, current_option)

    def _on_route_changed(self, route_key: str, option: str):
        if route_key in self._route_config_container:
            self._rebuild_route_config_section(route_key, option)
        self.values_changed.emit()

    def _rebuild_route_config_section(self, route_key: str, option: str):
        container = self._route_config_container[route_key]
        _clear_layout(container.layout())

        old_keys = [k for k in list(self._route_config_getters.keys())
                    if k.startswith(f"_rc_{route_key}_")]
        for k in old_keys:
            del self._route_config_getters[k]
            self._bound_pairs.pop(k, None)

        specs = self._schema.route_configs.get(route_key, {}).get(option, [])
        bound_targets: set[str] = set()
        for spec in specs:
            if spec.bind:
                bound_targets.add(spec.bind)

        for spec in specs:
            if spec.hidden or spec.key in bound_targets:
                continue
            row, getter, setter = create_config_row(
                spec, parent=container,
                on_change=self.values_changed.emit,
            )
            container.layout().addWidget(row)
            getter_key = f"_rc_{route_key}_{spec.key}"
            self._route_config_getters[getter_key] = getter
            if spec.bind:
                self._bound_pairs[getter_key] = spec.bind

    def _add_config_widget(self, spec: ConfigSpec):
        if spec.widget == "range_slider" and spec.bind:
            self._bound_pairs[spec.key] = spec.bind
        row, getter, setter = create_config_row(
            spec, parent=self,
            on_change=self.values_changed.emit,
        )
        self._main_layout.addWidget(row)
        self._config_getters[spec.key] = getter

    def _add_group_header(self, name: str):
        header = QLabel(name, self)
        header.setStyleSheet(
            "font-weight: bold; font-size: 11px; color: rgba(80,80,80,200); "
            "padding: 8px 5px 2px 5px;")
        self._main_layout.addWidget(header)

    # ── collect_configs override for range_slider bound pairs ─────────────

    def collect_configs(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for spec in self._schema.configs:
            if spec.hidden:
                if spec.default is not None:
                    result[spec.key] = spec.default
                continue
            if spec.key in self._config_getters:
                val = self._config_getters[spec.key]()
                if spec.key in self._bound_pairs:
                    if isinstance(val, (list, tuple)):
                        result[spec.key] = val[0]
                        result[self._bound_pairs[spec.key]] = val[1]
                    continue
                result[spec.key] = val

        for getter_key, getter in self._route_config_getters.items():
            # getter_key format: "_rc_{route_key}_{param_key}"
            # route_key may contain underscores, so we use known route_keys to parse
            param_key = None
            for rk in self._schema.route_configs:
                prefix = f"_rc_{rk}_"
                if getter_key.startswith(prefix):
                    param_key = getter_key[len(prefix):]
                    break
            if param_key is None:
                continue
            val = getter()
            if getter_key in self._bound_pairs:
                if isinstance(val, (list, tuple)):
                    result[param_key] = val[0]
                    result[self._bound_pairs[getter_key]] = val[1]
                continue
            result[param_key] = val

        return result


# ─── Utility ─────────────────────────────────────────────────────────────────


def _clear_layout(layout: QLayout | None):
    if layout is None:
        return
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget:
            widget.deleteLater()
        elif item.layout():
            _clear_layout(item.layout())
