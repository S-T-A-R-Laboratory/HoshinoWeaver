"""
Meta YAML 预处理层：将含路由定义的 Meta YAML 编译为标准 DAG spec dict。

语法——顶层 ``routes`` 字段 + 节点 ``route_key``：

    routes:
      stacker_mode:
        options: { mean: MeanStackerOp, sigma_clip: sigma_clip.yaml }
        default: mean

    nodes:
      stacker:
        route_key: stacker_mode
        inputs: { data: loader.result }
        route_configs:
          sigma_clip: { rej_high: configs.rej_high }
        outputs: { result: { type: image } }

meta_resolve() 根据用户选择将 Meta YAML 编译为标准 spec dict，
可直接传入 validate_and_build_order() → instantiate_and_wire() 执行。

本层不递归处理子图 YAML——子图由 flatten 机制处理。
"""

from __future__ import annotations

import copy
from typing import Any


class MetaResolveError(ValueError):
    """Meta YAML 解析错误。"""
    pass


def meta_resolve(
    meta_spec: dict[str, Any],
    route_choices: dict[str, str],
) -> dict[str, Any]:
    """将 Meta YAML spec 编译为标准 DAG spec dict。

    Args:
        meta_spec:
            Meta YAML 文件经 yaml.safe_load() 后的 dict。
            此 dict 不会被修改（内部做深拷贝）。
        route_choices:
            route_key → 选中的 option key 映射。
            未提供的 route_key 使用 routes 定义中的 default 值。

    Returns:
        标准 DAG spec dict，可直接传入
        ``validate_and_build_order()``。
        结果 spec 中会包含 ``_resolved_routes`` 字段
        记录已解析的路由选择。

    Raises:
        MetaResolveError: route_choices 缺失、选项非法等。
    """
    spec = copy.deepcopy(meta_spec)
    nodes = spec.get("nodes")
    if not isinstance(nodes, dict):
        return spec

    # ── 处理顶层 routes 定义 ──
    routes_def: dict[str, dict[str, Any]] = spec.pop("routes", {})
    resolved_routes: dict[str, str] = {}

    for node_name, node_spec in nodes.items():
        if "route_key" not in node_spec:
            continue

        route_key: str = node_spec.pop("route_key")

        # 查找 routes 定义
        if route_key not in routes_def:
            raise MetaResolveError(
                f"Node '{node_name}' references route_key '{route_key}' "
                f"but no such route is defined in top-level 'routes'. "
                f"Available routes: {list(routes_def.keys())}")

        route_info = routes_def[route_key]
        options: dict[str, str] = route_info.get("options", {})
        default: str | None = route_info.get("default")

        # 获取用户选择（优先 route_choices，其次 default）
        choice = route_choices.get(route_key)
        if choice is None:
            choice = default
        if choice is None:
            raise MetaResolveError(
                f"Route choice missing for route_key '{route_key}' "
                f"(node '{node_name}'). "
                f"Available options: {list(options.keys())}")
        if choice not in options:
            raise MetaResolveError(
                f"Invalid route choice '{choice}' for route_key '{route_key}' "
                f"(node '{node_name}'). "
                f"Available options: {list(options.keys())}")

        # 设置 op
        node_spec["op"] = options[choice]

        # 应用 route 专属 extras
        _apply_route_extras(node_spec, choice)

        # 记录已解析的选择
        resolved_routes[route_key] = choice

    # ── 存储已解析的路由选择（供 flatten 向下传递）──
    if resolved_routes:
        spec["_resolved_routes"] = resolved_routes

    return spec


def _apply_route_extras(node_spec: dict[str, Any], choice: str) -> None:
    """合并 route 专属的 inputs 和 configs 到节点 spec 中。

    从 node_spec 中 pop ``route_inputs`` 和 ``route_configs``，
    将选中 choice 对应的条目合并到节点的 inputs/configs section。
    """
    route_inp: dict[str, dict[str, Any]] = node_spec.pop("route_inputs", {})
    route_cfg: dict[str, dict[str, Any]] = node_spec.pop("route_configs", {})

    extra_inputs = route_inp.get(choice, {})
    if extra_inputs:
        node_inputs = node_spec.setdefault("inputs", {})
        node_inputs.update(extra_inputs)

    extra_configs = route_cfg.get(choice, {})
    if extra_configs:
        node_configs = node_spec.setdefault("configs", {})
        node_configs.update(extra_configs)
