"""
YAML 参数内省层：从 DAG/Meta YAML 中提取参数 schema 供 CLI --inspect 使用。

不实例化 Op、不布线、不执行——仅读取 YAML 声明并做浅层 meta_resolve
以获取路由选择后的完整参数集。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .build import _load_yaml, _flatten_configs_for_validation
from .meta import meta_resolve
from .wiring import _spec_needs_meta_resolve


@dataclass
class InputParam:
    name: str
    type: str
    required: bool


@dataclass
class ConfigParam:
    name: str
    type: str
    default: Any = None
    has_default: bool = False
    required: bool = True
    source: str = "global"


@dataclass
class RouteInfo:
    name: str
    options: list[str]
    default: str | None


@dataclass
class InspectResult:
    yaml_path: str
    description: str
    inputs: list[InputParam]
    configs: list[ConfigParam]
    routes: list[RouteInfo]
    route_configs: list[ConfigParam]


def inspect_yaml(
    yaml_path: str,
    route_choices: dict[str, str] | None = None,
) -> InspectResult:
    """从 YAML 文件提取参数 schema。

    Args:
        yaml_path: YAML 文件路径。
        route_choices: 路由选择。提供后会运行 meta_resolve 以展示
                       route_configs 中的完整参数集。

    Returns:
        InspectResult 包含 inputs, configs, routes, route_configs。
    """
    raw_spec = _load_yaml(yaml_path)
    description = raw_spec.get("description", "")
    is_meta = _spec_needs_meta_resolve(raw_spec)

    # --- inputs ---
    inputs = _collect_inputs(raw_spec)

    # --- routes (before meta_resolve) ---
    routes = _collect_routes(raw_spec) if is_meta else []

    # --- configs (global, before route_configs merge) ---
    configs = _collect_configs(raw_spec.get("configs", {}))

    # --- route_configs (need meta_resolve with route_choices) ---
    route_cfgs: list[ConfigParam] = []
    if is_meta and route_choices:
        route_cfgs = _collect_route_configs(raw_spec, route_choices)

    return InspectResult(
        yaml_path=yaml_path,
        description=description,
        inputs=inputs,
        configs=configs,
        routes=routes,
        route_configs=route_cfgs,
    )


def _collect_inputs(spec: dict[str, Any]) -> list[InputParam]:
    raw = spec.get("inputs", {})
    if not isinstance(raw, dict):
        return []
    result = []
    for name, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        result.append(InputParam(
            name=name,
            type=entry.get("type", "?"),
            required=entry.get("required", True),
        ))
    return result


def _collect_configs(configs_raw: dict[str, Any]) -> list[ConfigParam]:
    if not isinstance(configs_raw, dict):
        return []
    result = []
    for name, entry in configs_raw.items():
        if not isinstance(entry, dict):
            continue
        if "type" not in entry and "default" not in entry:
            continue
        has_default = "default" in entry
        required = entry.get("required", not has_default)
        result.append(ConfigParam(
            name=name,
            type=entry.get("type", "?"),
            default=entry.get("default"),
            has_default=has_default,
            required=required,
            source="global",
        ))
    return result


def _collect_routes(spec: dict[str, Any]) -> list[RouteInfo]:
    routes_def = spec.get("routes", {})
    if not isinstance(routes_def, dict):
        return []
    result = []
    for name, info in routes_def.items():
        if not isinstance(info, dict):
            continue
        options = list(info.get("options", {}).keys())
        default = info.get("default")
        result.append(RouteInfo(name=name, options=options, default=default))
    return result


def _collect_route_configs(
    raw_spec: dict[str, Any],
    route_choices: dict[str, str],
) -> list[ConfigParam]:
    """通过 meta_resolve 获取选中路由后的 route_configs 参数。"""
    spec = copy.deepcopy(raw_spec)
    resolved = meta_resolve(spec, route_choices, {})

    resolved_routes = resolved.get("_resolved_routes", {})
    resolved_configs = resolved.get("configs", {})

    result: list[ConfigParam] = []
    for route_key, choice in resolved_routes.items():
        nested = resolved_configs.get(route_key, {}).get(choice, {})
        if not isinstance(nested, dict):
            continue
        for param_name, param_spec in nested.items():
            if not isinstance(param_spec, dict):
                continue
            has_default = "default" in param_spec
            full_name = f"{route_key}.{choice}.{param_name}"
            result.append(ConfigParam(
                name=full_name,
                type=param_spec.get("type", "?"),
                default=param_spec.get("default"),
                has_default=has_default,
                required=not has_default,
                source=f"route({route_key}={choice})",
            ))
    return result
