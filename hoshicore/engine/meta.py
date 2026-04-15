"""
Meta YAML 预处理层：将含路由字典的 Meta YAML 编译为标准 DAG spec dict。

Meta YAML 在标准 DAG YAML 基础上扩展了三个可选字段：
    - route:         替代 op，列出该节点位置可选的多种实现
    - route_inputs:  按 route 选项分组的专属输入布线
    - route_configs: 按 route 选项分组的专属配置布线

meta_resolve() 根据用户选择将 Meta YAML 编译为标准 spec dict，
可直接传入 validate_and_build_order() → instantiate_and_wire() 执行。

本层不递归处理子图 YAML——子图由 wiring 层的 SubDagOp 机制处理。
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

    遍历 nodes 中的每个节点，若含 ``route`` 字段：

    1. 在 route_choices 中查找该节点的用户选择
    2. 将 ``route[choice]`` 填入 ``op`` 字段
    3. 将共享 ``inputs`` 与 ``route_inputs[choice]`` 合并
    4. 将共享 ``configs`` 与 ``route_configs[choice]`` 合并
    5. 删除 ``route`` / ``route_inputs`` / ``route_configs`` 字段

    不含 ``route`` 字段的普通节点原样保留。

    Args:
        meta_spec:
            Meta YAML 文件经 yaml.safe_load() 后的 dict。
            此 dict 不会被修改（内部做深拷贝）。
        route_choices:
            节点名 → 选中的 route key 映射。
            只需包含含 ``route`` 字段的节点。

    Returns:
        标准 DAG spec dict，可直接传入
        ``validate_and_build_order()``。

    Raises:
        MetaResolveError: route_choices 缺失、选项非法等。
    """
    spec = copy.deepcopy(meta_spec)
    nodes = spec.get("nodes")
    if not isinstance(nodes, dict):
        return spec  # 无节点，直接返回

    for node_name, node_spec in nodes.items():
        if "route" not in node_spec:
            continue

        route_dict: dict[str, str] = node_spec.pop("route")
        route_inp: dict[str, dict[str, Any]] = node_spec.pop(
            "route_inputs", {})
        route_cfg: dict[str, dict[str, Any]] = node_spec.pop(
            "route_configs", {})

        # ── 验证用户选择 ──
        choice = route_choices.get(node_name)
        if choice is None:
            raise MetaResolveError(
                f"Route choice missing for node '{node_name}'. "
                f"Available options: {list(route_dict.keys())}")
        if choice not in route_dict:
            raise MetaResolveError(
                f"Invalid route '{choice}' for node '{node_name}'. "
                f"Available options: {list(route_dict.keys())}")

        # ── 填入具体 op ──
        node_spec["op"] = route_dict[choice]

        # ── 合并 route 专属 inputs ──
        extra_inputs = route_inp.get(choice, {})
        if extra_inputs:
            node_inputs = node_spec.setdefault("inputs", {})
            node_inputs.update(extra_inputs)

        # ── 合并 route 专属 configs ──
        extra_configs = route_cfg.get(choice, {})
        if extra_configs:
            node_configs = node_spec.setdefault("configs", {})
            node_configs.update(extra_configs)

    # ── 完整性检查：不应遗留未解析的 route ──
    for node_name, node_spec in nodes.items():
        if "route" in node_spec:
            raise MetaResolveError(
                f"Node '{node_name}' still has unresolved route field.")

    return spec
