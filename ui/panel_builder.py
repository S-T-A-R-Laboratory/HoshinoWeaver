"""PanelSchema loader + DynamicConfigPanel: data-driven config UI from YAML."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QFrame, QLabel, QLayout, QScrollArea,
                               QSizePolicy, QVBoxLayout, QWidget)

from ui.styles import GROUP_BOX_STYLE, GROUP_HEADER_STYLE
from ui.widgets import (ConfigSpec, RouteOptionSpec, RouteSpec,
                        create_config_row, create_route_selector)
from ui.yaml_loader import load_ui_yaml

# ─── PanelSchema ─────────────────────────────────────────────────────────────


@dataclass
class OutputSpec:
    """Output declaration for OutputPanel rendering."""
    filename_key: str
    label: str = "输出"
    type: str = "image"  # image | sequence | video
    dtype_key: str | None = None
    formats: list[str] | None = None  # workflow-allowed formats
    dtype_options: list[str] | None = None  # workflow-allowed dtypes
    format_params: dict[str, str] = field(
        default_factory=dict)  # preset_param → config_key


@dataclass
class PanelSchema:
    meta_yaml_path: str
    routes: dict[str, RouteSpec] = field(default_factory=dict)
    configs: list[ConfigSpec] = field(default_factory=list)
    route_configs: dict[str,
                        dict[str,
                             list[ConfigSpec]]] = field(default_factory=dict)
    layout: dict = field(default_factory=dict)
    outputs: list[OutputSpec] = field(default_factory=list)

    @classmethod
    def from_yaml(cls,
                  meta_path: str | Path,
                  ui_path: str | Path | None = None) -> "PanelSchema":
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = yaml.safe_load(f)

        ui: dict = {}
        if ui_path and os.path.isfile(ui_path):
            ui = load_ui_yaml(ui_path)

        schema = cls(meta_yaml_path=str(meta_path))
        schema._parse_routes(meta, ui)
        schema._parse_configs(meta, ui)
        schema._parse_route_configs(meta, ui)
        schema._parse_outputs(ui)
        schema._resolve_bind_defaults()
        schema.layout = ui.get("layout", {})
        return schema

    def _resolve_bind_defaults(self):
        """Inject `bind_default` into every range_slider-with-bind spec.

        For specs where `bind` references another key in the same group,
        copy that key's `default` into `bind_default` so the widget can
        initialize the right handle from meta.yaml rather than from `max`.
        """

        def _resolve(specs: list[ConfigSpec]):
            by_key = {s.key: s for s in specs}
            for s in specs:
                if s.bind and s.bind in by_key:
                    s.bind_default = by_key[s.bind].default

        _resolve(self.configs)
        for _route, opt_map in self.route_configs.items():
            for _opt, specs in opt_map.items():
                _resolve(specs)

    def _parse_outputs(self, ui: dict):
        from ui.output_presets import IMAGE_FORMAT_PRESETS
        for entry in ui.get("outputs", []) or []:
            if not isinstance(entry, dict) or "filename_key" not in entry:
                raise ValueError(
                    f"ui.yaml 'outputs' entry must be a dict with 'filename_key': {entry!r}"
                )
            spec = OutputSpec(
                filename_key=entry["filename_key"],
                label=entry.get("label", "输出"),
                type=entry.get("type", "image"),
                dtype_key=entry.get("dtype_key"),
                formats=entry.get("formats"),
                dtype_options=entry.get("dtype_options"),
                format_params=entry.get("format_params", {}) or {},
            )
            # Validate format_params keys exist in presets
            for preset_param in spec.format_params:
                found = any(preset_param in preset["params"]
                            for preset in IMAGE_FORMAT_PRESETS.values())
                if not found:
                    raise ValueError(
                        f"OutputSpec '{spec.filename_key}': format_params key "
                        f"'{preset_param}' is not declared in any IMAGE_FORMAT_PRESETS entry"
                    )
            # Validate formats list against presets
            if spec.formats is not None:
                for fmt in spec.formats:
                    if fmt not in IMAGE_FORMAT_PRESETS:
                        raise ValueError(
                            f"OutputSpec '{spec.filename_key}': format '{fmt}' "
                            f"is not declared in IMAGE_FORMAT_PRESETS")
            self.outputs.append(spec)

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
                default=route_def.get("default",
                                      option_keys[0] if option_keys else ""),
                visible_when=ui_route.get("visible_when"),
            )

    def _parse_configs(self, meta: dict, ui: dict):
        for key, spec in meta.get("configs", {}).items():
            ui_cfg = ui.get("configs", {}).get(key, {})
            self.configs.append(_build_config_spec(key, spec, ui_cfg))

    def _parse_route_configs(self, meta: dict, ui: dict):
        meta_rc = meta.get("route_configs", {})
        ui_rc = ui.get("route_configs", {})
        if not meta_rc:
            return
        for route_key, options_dict in meta_rc.items():
            self.route_configs[route_key] = {}
            for option_key, params in options_dict.items():
                specs: list[ConfigSpec] = []
                ui_option = ui_rc.get(option_key, {})
                for param_key, param_spec in params.items():
                    ui_param = ui_option.get(param_key, {})
                    specs.append(
                        _build_config_spec(param_key, param_spec, ui_param))
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
        options=_normalize_options(ui_spec.get("options")),
        bind=ui_spec.get("bind"),
        accept=ui_spec.get("accept"),
        transform=ui_spec.get("transform"),
        visible_when=ui_spec.get("visible_when"),
    )


def _normalize_options(raw: Any) -> list[tuple[Any, str]] | None:
    """Normalize `options` into list of (value, label) pairs.

    Accepts:
      - None
      - list of scalars (legacy):       ["a", "b"]      → [("a","a"), ("b","b")]
      - list of dicts (with alias):     [{value, label}, ...]
    """
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError(f"options must be a list, got {type(raw).__name__}")
    out: list[tuple[Any, str]] = []
    for item in raw:
        if isinstance(item, dict):
            if "value" not in item:
                raise ValueError(f"option dict must contain 'value': {item!r}")
            value = item["value"]
            label = str(item.get("label", value))
        else:
            value = item
            label = str(item)
        out.append((value, label))
    return out


# ─── Value Transforms ────────────────────────────────────────────────────────

from ui.transforms import apply_forward as _apply_transform  # noqa: E402


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
        self._route_config_getters: dict[tuple[str, str], Callable] = {}
        self._route_config_container: dict[str, QWidget] = {}
        self._bound_pairs: dict[str | tuple[str, str], str] = {}
        self._visibility_deps: dict[str, list[tuple[list[QWidget], dict]]] = {}
        self._config_widgets: dict[str, QWidget] = {}
        self._config_on_change_hooks: dict[str, list[Callable]] = {}

    def load_schema(self, schema: PanelSchema):
        self._schema = schema
        self._config_getters.clear()
        self._route_getters.clear()
        self._route_config_getters.clear()
        self._route_config_container.clear()
        self._bound_pairs.clear()
        self._visibility_deps.clear()
        self._config_widgets.clear()
        self._config_on_change_hooks.clear()

        _clear_layout(self._main_layout)

        self._render_routes()
        self._render_configs()
        self._render_route_configs()
        self._evaluate_all_visibility()

        self._main_layout.addStretch()

    def collect_route_choices(self) -> dict[str, str]:
        return {key: getter() for key, getter in self._route_getters.items()}

    # ── Rendering ────────────────────────────────────────────────────────────

    def _render_routes(self):
        for route_key, route_spec in self._schema.routes.items():
            frame, getter, setter = create_route_selector(
                route_spec,
                parent=self,
                on_changed=lambda opt, rk=route_key: self._on_route_changed(
                    rk, opt),
            )
            self._main_layout.addWidget(frame)
            self._route_getters[route_key] = getter

            # Place the route_config container immediately below its selector
            container: QWidget | None = None
            if route_key in self._schema.route_configs:
                container = QWidget(self)
                container_layout = QVBoxLayout(container)
                container_layout.setContentsMargins(0, 0, 0, 0)
                container_layout.setSpacing(0)
                self._main_layout.addWidget(container)
                self._route_config_container[route_key] = container

            if route_spec.visible_when:
                targets = [frame]
                if container is not None:
                    targets.append(container)
                self._register_visibility_dep(targets, route_spec.visible_when)

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
                    s for s in self._schema.configs if s.key in group_keys
                    and not s.hidden and s.key not in route_keys
                ]
                if not specs_in_group:
                    continue
                group_layout = self._add_group_box(group_name)
                for spec in specs_in_group:
                    self._add_config_widget(spec, target_layout=group_layout)
                    rendered_keys.add(spec.key)

        remaining = [
            s for s in self._schema.configs if s.key not in rendered_keys
            and not s.hidden and s.key not in route_keys
        ]
        if order:
            key_order = [k for k in order if k not in route_keys]
            remaining.sort(key=lambda s: key_order.index(s.key)
                           if s.key in key_order else 999)

        for spec in remaining:
            self._add_config_widget(spec)

    def _render_route_configs(self):
        for route_key in self._schema.route_configs:
            if route_key not in self._route_config_container:
                # Fallback: route_configs key has no matching route selector
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
        self._evaluate_visibility(route_key)
        self.values_changed.emit()

    def _rebuild_route_config_section(self, route_key: str, option: str):
        container = self._route_config_container[route_key]
        _clear_layout(container.layout())

        old_keys = [
            k for k in list(self._route_config_getters.keys())
            if k[0] == route_key
        ]
        for k in old_keys:
            del self._route_config_getters[k]
            self._bound_pairs.pop(k, None)

        specs = self._schema.route_configs.get(route_key, {}).get(option, [])
        bound_targets: set[str] = set()
        for spec in specs:
            if spec.bind:
                bound_targets.add(spec.bind)

        visible_specs = [
            s for s in specs if not s.hidden and s.key not in bound_targets
        ]
        if not visible_specs:
            return

        route_spec = self._schema.routes.get(route_key)
        route_label = (route_spec.label
                       or route_key) if route_spec else route_key
        if route_spec and option in route_spec.options:
            option_label = route_spec.options[option].label or option
        else:
            option_label = option
        group_name = f"{route_label} · {option_label}"

        group_layout = self._add_group_box(
            group_name,
            target_layout=container.layout(),
            parent_widget=container,
        )

        for spec in visible_specs:
            row, getter, setter = create_config_row(
                spec,
                parent=container,
                on_change=self.values_changed.emit,
            )
            group_layout.addWidget(row)
            getter_key = (route_key, spec.key)
            self._route_config_getters[getter_key] = getter
            if spec.bind:
                self._bound_pairs[getter_key] = spec.bind

    def _add_config_widget(self, spec: ConfigSpec, target_layout=None):
        if spec.widget == "range_slider" and spec.bind:
            self._bound_pairs[spec.key] = spec.bind
        layout = target_layout if target_layout is not None else self._main_layout

        def _on_config_change(key=spec.key):
            self.values_changed.emit()
            for hook in self._config_on_change_hooks.get(key, []):
                hook()

        row, getter, setter = create_config_row(
            spec,
            parent=self,
            on_change=_on_config_change,
        )
        layout.addWidget(row)
        self._config_getters[spec.key] = getter
        self._config_widgets[spec.key] = row

        if spec.visible_when:
            self._register_visibility_dep([row], spec.visible_when)

    def _add_group_box(
        self,
        name: str,
        target_layout=None,
        parent_widget: QWidget | None = None,
    ) -> QVBoxLayout:
        """Create a bordered group box, return its inner layout.

        target_layout  – layout to add the box to (defaults to _main_layout).
        parent_widget  – Qt parent for the frame (defaults to self).
        """
        parent = parent_widget if parent_widget is not None else self
        outer = QFrame(parent)
        outer.setObjectName("groupBox")
        outer.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        outer.setStyleSheet(GROUP_BOX_STYLE)
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(0, 0, 0, 4)
        outer_layout.setSpacing(0)

        header = QLabel(name, outer)
        header.setObjectName("groupHeader")
        header.setStyleSheet(GROUP_HEADER_STYLE)
        outer_layout.addWidget(header)
        layout = target_layout if target_layout is not None else self._main_layout
        layout.addWidget(outer)
        return outer_layout

    # ── Conditional Visibility ────────────────────────────────────────────

    def _register_visibility_dep(self, targets: list[QWidget], condition: dict):
        dep_key = condition.get("key", "")
        if not dep_key:
            return
        if dep_key not in self._visibility_deps:
            self._visibility_deps[dep_key] = []
        self._visibility_deps[dep_key].append((targets, condition))
        self._config_on_change_hooks.setdefault(dep_key, []).append(
            lambda k=dep_key: self._evaluate_visibility(k)
        )

    def _evaluate_visibility(self, changed_key: str):
        for targets, cond in self._visibility_deps.get(changed_key, []):
            current_val = self._get_current_value(changed_key)
            if "eq" in cond:
                visible = current_val == cond["eq"]
            elif "neq" in cond:
                visible = current_val != cond["neq"]
            else:
                visible = True
            for w in targets:
                w.setVisible(visible)

    def _evaluate_all_visibility(self):
        for dep_key in self._visibility_deps:
            self._evaluate_visibility(dep_key)

    def _get_current_value(self, key: str) -> Any:
        if key in self._route_getters:
            return self._route_getters[key]()
        if key in self._config_getters:
            return self._config_getters[key]()
        return None

    # ── collect_configs override for range_slider bound pairs ─────────────

    def collect_configs(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        _bound_pair_targets = set(self._bound_pairs.values())
        for spec in self._schema.configs:
            if spec.key in _bound_pair_targets:
                continue
            if spec.hidden:
                if spec.default is not None:
                    result[spec.key] = spec.default
                continue
            if spec.key in self._config_getters:
                val = self._config_getters[spec.key]()
                if spec.key in self._bound_pairs:
                    if isinstance(val, (list, tuple)):
                        left, right = val[0], val[1]
                        if isinstance(spec.transform, dict):
                            left = _apply_transform(spec.transform.get("left"),
                                                    left)
                            right = _apply_transform(
                                spec.transform.get("right"), right)
                        result[spec.key] = left
                        result[self._bound_pairs[spec.key]] = right
                    continue
                if isinstance(spec.transform, str):
                    val = _apply_transform(spec.transform, val)
                if val == "": val = None
                result[spec.key] = val

        for (route_key,
             param_key), getter in self._route_config_getters.items():
            current_option = self._route_getters[route_key]()
            option_specs = self._schema.route_configs.get(route_key, {}).get(
                current_option, [])
            spec = next((s for s in option_specs if s.key == param_key), None)

            val = getter()
            bound_key = (route_key, param_key)
            if bound_key in self._bound_pairs:
                if isinstance(val, (list, tuple)):
                    left, right = val[0], val[1]
                    if spec and isinstance(spec.transform, dict):
                        left = _apply_transform(spec.transform.get("left"),
                                                left)
                        right = _apply_transform(spec.transform.get("right"),
                                                 right)
                    result[param_key] = left
                    result[self._bound_pairs[bound_key]] = right
                continue
            if spec and isinstance(spec.transform, str):
                val = _apply_transform(spec.transform, val)
            if val == "": val = None
            # 暂不支持多层级路由参数键名
            result[f"{route_key}.{current_option}.{param_key}"] = val

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
