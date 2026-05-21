"""
Meta YAML v2 预处理层：将含路由定义的 Meta YAML 编译为标准 DAG spec dict。

语法——顶层 ``routes`` + ``route_configs`` + 节点 ``route_key``：

    routes:
      stacker:
        options:
          mean:       MeanStackerOp
          sigma_clip: sigma_clip.yaml
        default: mean

    route_configs:
      sigma_clip:
        rej_high: { type: float, default: 3.0 }

    nodes:
      stacker:
        route_key: stacker
        route_configs:
          sigma_clip: { rej_high: configs.rej_high }

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
    global_configs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """将 Meta YAML spec 编译为标准 DAG spec dict。

    Args:
        meta_spec:
            Meta YAML 文件经 yaml.safe_load() 后的 dict。
            此 dict 不会被修改（内部做深拷贝）。
        route_choices:
            route_key → 选中的 option key 映射。
            未提供的 route_key 使用 routes 定义中的 default 值。
        global_configs:
            运行时全局配置值。用于解析节点 ``enabled`` 引用。
            None 时所有 enabled 引用使用 YAML 声明的 default。

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

    # ── 1. 处理顶层 routes 定义 ──
    routes_def: dict[str, dict[str, Any]] = spec.pop("routes", {})
    top_route_configs: dict[str, dict[str, Any]] = spec.pop("route_configs", {})
    resolved_routes: dict[str, str] = {}

    for node_name, node_spec in nodes.items():
        if "route_key" not in node_spec:
            continue

        route_key: str = node_spec.pop("route_key")

        if route_key not in routes_def:
            raise MetaResolveError(
                f"Node '{node_name}' references route_key '{route_key}' "
                f"but no such route is defined in top-level 'routes'. "
                f"Available routes: {list(routes_def.keys())}")

        route_info = routes_def[route_key]
        options: dict[str, Any] = route_info.get("options", {})
        default: str | None = route_info.get("default")

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

        # fixed-op 节点：op 字段已由用户显式声明，跳过赋值
        # （options 值为 null 时作为纯标签，仅用于 route_configs 分支选择）
        if "op" not in node_spec:
            node_spec["op"] = options[choice]

        _apply_route_extras(node_spec, route_key, choice, top_route_configs)
        resolved_routes[route_key] = choice

    # ── 2. merge 选中 option 对应的 route_configs 到嵌套 configs ──
    if top_route_configs:
        spec_configs: dict[str, Any] = spec.setdefault("configs", {})
        for route_key, choice in resolved_routes.items():
            route_group = top_route_configs.get(route_key, {})
            option_params = route_group.get(choice, {})
            if option_params:
                spec_configs.setdefault(route_key, {})[choice] = option_params

    # ── 3. 处理节点 enabled/bypass/prune ──
    _resolve_enabled_nodes(spec, global_configs or {})

    # ── 4. 规范化 outputs（将 object 格式提取为纯 link 字符串） ──
    _normalize_outputs(spec)

    if resolved_routes:
        spec["_resolved_routes"] = resolved_routes

    return spec


def _resolve_enabled_nodes(
    spec: dict[str, Any],
    global_configs: dict[str, Any],
) -> None:
    """处理节点 ``enable`` 字段：根据是否声明 bypass 分流为穿透或断路。

    对每个声明了 ``enable: configs.<key>`` 的节点：
    1. 解析 enabled 值（优先 global_configs，其次 YAML default）
    2. 若为 True → 仅 pop ``enable`` 和 ``bypass`` 字段，保留节点
    3. 若为 False + 有 bypass → 穿透：重写下游 link，删除节点
    4. 若为 False + 无 bypass → 断路 (prune)：级联删除/标记 inactive
    """
    nodes = spec.get("nodes")
    if not isinstance(nodes, dict):
        return

    configs_spec: dict[str, Any] = spec.get("configs", {})

    bypass_nodes: list[tuple[str, str]] = []
    prune_set: set[str] = set()

    for node_name, node_spec in nodes.items():
        if not isinstance(node_spec, dict):
            continue
        enabled_ref = node_spec.pop("enable", None)
        if enabled_ref is None:
            continue

        bypass_key: str | None = node_spec.pop("bypass", None)

        enabled_val = _resolve_enabled_value(
            enabled_ref, configs_spec, global_configs, node_name)

        if enabled_val:
            continue

        if bypass_key is not None:
            bypass_nodes.append((node_name, bypass_key))
        else:
            prune_set.add(node_name)

    # ── bypass 穿透处理 ──
    for node_name, bypass_key in bypass_nodes:
        _apply_bypass(nodes, spec, node_name, bypass_key)

    # ── prune 断路处理 ──
    if prune_set:
        _cascade_prune(nodes, spec, prune_set)


def _apply_bypass(
    nodes: dict[str, Any],
    spec: dict[str, Any],
    node_name: str,
    bypass_key: str,
) -> None:
    """对单个节点执行 bypass 穿透：重写下游连线并删除节点。"""
    node_spec = nodes[node_name]
    node_inputs = node_spec.get("inputs", {})
    node_outputs = node_spec.get("outputs", {})

    bypass_input, bypass_output = _find_bypass_pair(
        node_name, node_inputs, node_outputs, bypass_key)

    source_link = node_inputs[bypass_input]
    bypass_ref = f"{node_name}.{bypass_output}"

    for other_name, other_spec in nodes.items():
        if other_name == node_name or not isinstance(other_spec, dict):
            continue
        for section in ("inputs", "configs"):
            sec = other_spec.get(section)
            if not isinstance(sec, dict):
                continue
            for key, val in sec.items():
                if isinstance(val, str) and val == bypass_ref:
                    sec[key] = source_link

    top_outputs = spec.get("outputs", {})
    for out_key, out_val in top_outputs.items():
        if isinstance(out_val, str) and out_val == bypass_ref:
            top_outputs[out_key] = source_link

    del nodes[node_name]


def _cascade_prune(
    nodes: dict[str, Any],
    spec: dict[str, Any],
    initial_prune: set[str],
) -> None:
    """级联处理 prune 节点：标记 __inactive__、级联删除、处理 optional outputs。

    级联规则：
    - 下游节点只有单个 input → 级联删除
    - 下游节点有多个 inputs → 对应连线标记 __inactive__
    - 级联到 outputs: required=false → 静默移除; required(默认) → 报错
    """
    output_links: dict[str, Any] = spec.get("outputs", {})
    prune_set = set(initial_prune)

    while prune_set:
        new_prune: set[str] = set()

        for pruned in prune_set:
            pruned_outputs = {
                f"{pruned}.{out}"
                for out in nodes[pruned].get("outputs", {})
            }
            if not pruned_outputs:
                continue

            for other_name, other_spec in list(nodes.items()):
                if other_name in prune_set or other_name in new_prune:
                    continue
                if not isinstance(other_spec, dict):
                    continue
                for section in ("inputs", "configs"):
                    sec = other_spec.get(section)
                    if not sec or not isinstance(sec, dict):
                        continue
                    for key, val in list(sec.items()):
                        if not isinstance(val, str) or val not in pruned_outputs:
                            continue
                        input_count = _count_inputs(other_spec)
                        if section == "inputs" and input_count <= 1:
                            new_prune.add(other_name)
                        else:
                            sec[key] = "__inactive__"

        # 处理 outputs 引用
        for out_key, out_val in list(output_links.items()):
            out_link = out_val if isinstance(out_val, str) else out_val.get("src", "")
            for pruned in prune_set:
                pruned_outputs = {
                    f"{pruned}.{out}"
                    for out in nodes[pruned].get("outputs", {})
                }
                if out_link in pruned_outputs:
                    out_required = True
                    if isinstance(out_val, dict):
                        out_required = out_val.get("required", True)
                    if out_required:
                        raise MetaResolveError(
                            f"Prune 导致全局输出 '{out_key}' 不可达 "
                            f"(节点 '{pruned}' 被断路)。"
                            f"如需允许此行为，请为该 output 声明 required: false")
                    else:
                        del output_links[out_key]
                    break

        for name in prune_set:
            del nodes[name]

        prune_set = new_prune


def _count_inputs(node_spec: dict[str, Any]) -> int:
    """统计节点 inputs section 中的条目数。"""
    inputs = node_spec.get("inputs", {})
    return len(inputs) if isinstance(inputs, dict) else 0


def _normalize_outputs(spec: dict[str, Any]) -> None:
    """将 outputs 中的 object 格式规范化为纯 link 字符串（供下游 build/wiring 使用）。"""
    outputs = spec.get("outputs", {})
    for key, val in list(outputs.items()):
        if isinstance(val, dict):
            outputs[key] = val["src"]


def _resolve_enabled_value(
    ref: str,
    configs_spec: dict[str, Any],
    global_configs: dict[str, Any],
    node_name: str,
) -> bool:
    """解析 enabled 引用为 bool 值。"""
    if not isinstance(ref, str) or not ref.startswith("configs."):
        raise MetaResolveError(
            f"Node '{node_name}': 'enable' must reference 'configs.<key>', "
            f"got '{ref}'")
    config_key = ref[len("configs."):]

    if config_key in global_configs:
        return bool(global_configs[config_key])

    cfg_def = configs_spec.get(config_key)
    if isinstance(cfg_def, dict) and "default" in cfg_def:
        return bool(cfg_def["default"])

    raise MetaResolveError(
        f"Node '{node_name}': enable references 'configs.{config_key}' "
        f"but no value provided and no default declared")


def _find_bypass_pair(
    node_name: str,
    node_inputs: dict[str, Any],
    node_outputs: dict[str, Any],
    bypass_key: str,
) -> tuple[str, str]:
    """确定 bypass 的 input→output 对。

    bypass_key 指定穿透的 input 名称，配对第一个 output。

    Returns:
        (input_key, output_key)
    """
    if not node_outputs:
        raise MetaResolveError(
            f"Node '{node_name}': no outputs declared, cannot bypass")

    first_output = next(iter(node_outputs))

    if bypass_key not in node_inputs:
        raise MetaResolveError(
            f"Node '{node_name}': bypass key '{bypass_key}' "
            f"not found in inputs: {list(node_inputs.keys())}")
    return bypass_key, first_output


def _apply_route_extras(
    node_spec: dict[str, Any],
    route_key: str,
    choice: str,
    top_route_configs: dict[str, dict[str, Any]],
) -> None:
    """合并 route 专属的 inputs 和 configs 到节点 spec 中。

    从 node_spec 中 pop ``route_inputs`` 和 ``route_configs``，
    将选中 choice 对应的条目合并到节点的 inputs/configs section。

    Auto-wire：顶层 ``route_configs[route_key][choice]`` 中声明但
    节点 ``route_configs[choice]`` 中未显式布线的参数，
    自动生成 ``configs.<route_key>.<choice>.<param>`` 引用。
    """
    route_inp: dict[str, dict[str, Any]] = node_spec.pop("route_inputs", {})
    route_cfg: dict[str, dict[str, Any]] = node_spec.pop("route_configs", {})

    extra_inputs = route_inp.get(choice, {})
    if extra_inputs:
        node_inputs = node_spec.setdefault("inputs", {})
        node_inputs.update(extra_inputs)

    extra_configs = route_cfg.get(choice, {})

    # auto-wire: 补全未显式布线的 route_configs 参数
    declared_params = top_route_configs.get(route_key, {}).get(choice, {})
    for param in declared_params:
        if param not in extra_configs:
            extra_configs[param] = f"configs.{route_key}.{choice}.{param}"

    if extra_configs:
        node_configs = node_spec.setdefault("configs", {})
        node_configs.update(extra_configs)
