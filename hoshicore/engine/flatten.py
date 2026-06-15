"""
SubDAG 预展开：将 .yaml 引用的子图在编译期展平为顶层节点。

时机：在 validate_and_build_order() 之前、meta_resolve() 之后。
效果：所有 .yaml 子图引用被替换为带命名空间前缀的独立节点，
      使得段检测和数据并行可以穿透子图边界。

示例：
    展开前 (父图中):
        main_stacker:
          op: sigma_clip.yaml
          inputs: { data: flat_divide.result }
          configs: { int_weight: configs.int_weight, ... }

    展开后:
        main_stacker.mean_stacker:
          op: MeanStackerOp
          inputs: { data: flat_divide.result }
          configs: { int_weight: configs.int_weight }
        main_stacker.disk_buffer:
          op: DiskBufferWriterOp
          inputs: { data: flat_divide.result, fnames: __inactive__ }
          configs: { buffer_mode: "disk" }
        main_stacker.sigma_clip_iter:
          op: SigmaClipIteratorOp
          configs: { fgp_total: main_stacker.mean_stacker.statistics, ... }

设计要点：
    - rsplit(".", 1) 解析 link，因此 "main_stacker.mean_stacker.statistics"
      会被正确解析为 node="main_stacker.mean_stacker" output="statistics"
    - 子图的可选输入未被父图布线时标记为 __inactive__
    - 子图 configs 的默认值在父图未覆盖时省略该键，由 Op 自身 CONFIGS default 补齐
    - 递归展开：子图内部也可以引用 .yaml（当前无此场景但架构支持）
    - 子图若为 Meta YAML（含 routes/route_key），展开前自动调用 meta_resolve
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Optional

from loguru import logger

# 标记：表示子图可选输入未被父图布线
INACTIVE_MARKER = "__inactive__"

# 内部标记：表示子图 config 有 Op 默认值，不需要在 spec 中布线
_OMIT_SENTINEL = object()


def _collapse_route_configs(spec: dict[str, Any]) -> None:
    """将 meta_resolve 产生的嵌套 route_configs 坍缩为 flat config key。

    meta_resolve 将 route_configs 合并到 configs 嵌套命名空间：
        configs["stacker"]["sigma_clip"]["rej_high"] = {type: float, default: 3.0}
    内部节点引用为 configs.stacker.sigma_clip.rej_high。

    当子图作为 SubDAG 被展开时，父图以 flat key（如 "rej_high"）提供这些参数。
    本函数将嵌套结构提升回 flat key，并重写内部节点引用，
    使子图的 config 接口与父图提供的 key 对齐。

    仅在 SubDAG 展开上下文中调用（修改 spec 副本，不影响独立执行）。
    """
    configs = spec.get("configs")
    if not isinstance(configs, dict):
        return

    # 收集嵌套结构（无 type/default 的 dict entry = meta_resolve 生成的命名空间）
    nested_keys: list[str] = []
    promotions: dict[str, tuple[str, dict]] = {}  # flat_name → (dotted_path, leaf_spec)

    for key, value in list(configs.items()):
        if not isinstance(value, dict):
            continue
        if "type" in value or "default" in value:
            continue
        nested_keys.append(key)
        _collect_leaves(value, key, promotions, configs)

    if not promotions:
        return

    # 移除嵌套结构，添加 flat key
    for key in nested_keys:
        del configs[key]
    for flat_name, (_dotted_path, leaf_spec) in promotions.items():
        configs[flat_name] = leaf_spec

    # 构建重写映射：configs.stacker.sigma_clip.rej_high → configs.rej_high
    rewrite_map: dict[str, str] = {
        f"configs.{dotted_path}": f"configs.{flat_name}"
        for flat_name, (dotted_path, _) in promotions.items()
    }

    # 重写节点内部引用
    nodes = spec.get("nodes")
    if not isinstance(nodes, dict):
        return
    for node_spec in nodes.values():
        if not isinstance(node_spec, dict):
            continue
        for section in ("inputs", "configs"):
            sec = node_spec.get(section)
            if not isinstance(sec, dict):
                continue
            for arg_key, binding in list(sec.items()):
                if isinstance(binding, str) and binding in rewrite_map:
                    sec[arg_key] = rewrite_map[binding]
                elif isinstance(binding, dict) and binding.get("src") in rewrite_map:
                    binding["src"] = rewrite_map[binding["src"]]


def _collect_leaves(
    nested: dict[str, Any],
    prefix: str,
    promotions: dict[str, tuple[str, dict]],
    top_configs: dict[str, Any],
) -> None:
    """递归收集嵌套 dict 的叶节点（含 type/default 的 spec）。"""
    for key, value in nested.items():
        dotted = f"{prefix}.{key}"
        if isinstance(value, dict) and "type" not in value and "default" not in value:
            _collect_leaves(value, dotted, promotions, top_configs)
        else:
            flat_name = key
            if flat_name in top_configs:
                raise ValueError(
                    f"Route config '{dotted}' collides with existing config "
                    f"'{flat_name}' when collapsing for SubDAG expansion.")
            if flat_name in promotions:
                raise ValueError(
                    f"Route config '{dotted}' collides with another promoted "
                    f"config '{promotions[flat_name][0]}' (both map to '{flat_name}').")
            promotions[flat_name] = (dotted, value)


def flatten_sub_dags(
    spec: dict[str, Any],
    dag_search_paths: Optional[list[Path]] = None,
) -> dict[str, Any]:
    """递归展开 spec 中所有 .yaml SubDAG 引用为扁平拓扑。

    Args:
        spec: DAG 的原始 YAML spec dict（不会被修改，内部深拷贝）。
        dag_search_paths: 子图 YAML 搜索路径列表。None 使用 wiring 层默认值。

    Returns:
        展平后的 spec dict，所有 .yaml 引用被替换为带命名空间前缀的节点。
    """
    if dag_search_paths is None:
        from .wiring import DEFAULT_DAG_SEARCH_PATHS
        dag_search_paths = DEFAULT_DAG_SEARCH_PATHS

    spec = copy.deepcopy(spec)
    nodes = spec.get("nodes")
    if not isinstance(nodes, dict):
        return spec

    # 迭代展开：每次扫描一遍，展开所有 .yaml 引用。
    # 由于展开后可能引入新的 .yaml 引用（嵌套子图），循环直到无 .yaml 引用。
    max_depth = 10  # 防止无限递归
    for depth in range(max_depth):
        # 每轮重新读取 nodes 引用（_expand_one_sub_dag 会替换 spec["nodes"]）
        nodes = spec["nodes"]
        yaml_nodes = [
            (name, node_spec)
            for name, node_spec in list(nodes.items())
            if isinstance(node_spec, dict)
            and isinstance(node_spec.get("op"), str)
            and node_spec["op"].endswith(".yaml")
        ]
        if not yaml_nodes:
            break

        for parent_name, parent_node_spec in yaml_nodes:
            _expand_one_sub_dag(
                spec, parent_name, parent_node_spec, dag_search_paths)

        logger.info(
            f"[flatten] depth={depth}: expanded {len(yaml_nodes)} sub-DAG(s)")
    else:
        cur_nodes = spec["nodes"]
        yaml_remaining = [
            n for n, ns in cur_nodes.items()
            if isinstance(ns, dict)
            and isinstance(ns.get("op"), str)
            and ns["op"].endswith(".yaml")
        ]
        if yaml_remaining:
            raise RuntimeError(
                f"SubDAG expansion exceeded max depth ({max_depth}). "
                f"Remaining .yaml nodes: {yaml_remaining}")

    return spec


def _expand_one_sub_dag(
    parent_spec: dict[str, Any],
    parent_name: str,
    parent_node_spec: dict[str, Any],
    dag_search_paths: list[Path],
) -> None:
    """展开一个 .yaml SubDAG 引用，修改 parent_spec.nodes 就地替换。

    Steps:
        1. 加载子 YAML spec
        2. 构建 input/config 映射表（子图全局 → 父图实际源）
        3. 子图节点添加 "{parent_name}." 命名空间前缀
        4. 重写子图内部所有 link 引用
        5. 重写父图中引用此 SubDAG 输出的所有消费者
        6. 删除原 SubDAG 节点，插入展平后的子节点
    """
    from .build import _load_yaml
    from .wiring import _resolve_sub_dag_yaml

    parent_nodes = parent_spec["nodes"]
    sub_yaml_name = parent_node_spec["op"]

    # ── 1) 加载子 YAML ──
    resolved_path = _resolve_sub_dag_yaml(sub_yaml_name)
    sub_spec = _load_yaml(str(resolved_path))
    logger.debug(f"[flatten] expanding '{parent_name}' → {resolved_path}")

    # ── 1.5) Meta sub-DAG 检测：如果子图包含路由定义，先解析 ──
    sub_nodes_raw = sub_spec.get("nodes", {})
    has_routes_def = bool(sub_spec.get("routes"))
    has_route_nodes = any(
        isinstance(ns, dict) and "route_key" in ns
        for ns in sub_nodes_raw.values()
    ) if isinstance(sub_nodes_raw, dict) else False

    if has_routes_def or has_route_nodes:
        parent_route_choices = _resolve_parent_routes(
            parent_node_spec, parent_spec)
        from .meta import meta_resolve
        sub_spec = meta_resolve(sub_spec, parent_route_choices)
        _collapse_route_configs(sub_spec)
        logger.debug(
            f"[flatten] meta_resolve applied to sub-DAG '{parent_name}' "
            f"with choices: {parent_route_choices}")

    sub_nodes: dict[str, dict] = sub_spec.get("nodes", {})
    sub_inputs_spec: dict[str, dict] = sub_spec.get("inputs", {})
    sub_configs_spec: dict[str, dict] = sub_spec.get("configs", {})
    sub_outputs: dict[str, str] = sub_spec.get("outputs", {})

    # ── 2) 构建 input 映射：子图 "inputs.xxx" → 父图实际 src ──
    parent_inputs: dict[str, str] = {}
    parent_inputs_section = parent_node_spec.get("inputs", {})
    if isinstance(parent_inputs_section, dict):
        for key, binding in parent_inputs_section.items():
            if isinstance(binding, dict) and "src" in binding:
                parent_inputs[key] = binding["src"]
            elif isinstance(binding, str):
                parent_inputs[key] = binding

    # ── 3) 构建 config 映射：子图 "configs.xxx" → 父图实际 src 或字面量 ──
    parent_configs: dict[str, Any] = {}  # key → src_string or literal value
    parent_configs_section = parent_node_spec.get("configs", {})
    if isinstance(parent_configs_section, dict):
        for key, binding in parent_configs_section.items():
            if isinstance(binding, dict) and "src" in binding:
                parent_configs[key] = binding["src"]
            elif isinstance(binding, str):
                parent_configs[key] = binding
            else:
                # 字面量值（如 int, float, bool）
                parent_configs[key] = binding

    # ── 4) 构建输出映射：子图 output_name → 展平后的 "parent.node.output" ──
    output_rewrite: dict[str, str] = {}
    for out_name, out_link in sub_outputs.items():
        # out_link 形如 "sigma_clip_iter.result"
        rewritten = _rewrite_link(
            out_link, parent_name, parent_inputs, parent_configs,
            sub_inputs_spec, sub_configs_spec)
        output_rewrite[out_name] = rewritten

    # ── 5) 展开子图节点，添加前缀并重写内部 link ──
    expanded_nodes: dict[str, dict] = {}
    for child_name, child_spec in sub_nodes.items():
        prefixed_name = f"{parent_name}.{child_name}"
        new_spec: dict[str, Any] = {"op": child_spec["op"]}

        # 重写 inputs
        if "inputs" in child_spec and isinstance(child_spec["inputs"], dict):
            new_inputs: dict[str, Any] = {}
            for inp_key, inp_binding in child_spec["inputs"].items():
                src = _extract_src(inp_binding)
                rewritten = _rewrite_link(
                    src, parent_name, parent_inputs, parent_configs,
                    sub_inputs_spec, sub_configs_spec)
                if rewritten == INACTIVE_MARKER:
                    new_inputs[inp_key] = INACTIVE_MARKER
                else:
                    new_inputs[inp_key] = rewritten
            new_spec["inputs"] = new_inputs

        # 重写 configs
        if "configs" in child_spec and isinstance(child_spec["configs"], dict):
            new_configs: dict[str, Any] = {}
            for cfg_key, cfg_binding in child_spec["configs"].items():
                src = _extract_src(cfg_binding)
                rewritten = _rewrite_link(
                    src, parent_name, parent_inputs, parent_configs,
                    sub_inputs_spec, sub_configs_spec)
                # _OMIT_SENTINEL: 父图未覆盖且子图有默认值 → 省略该键，
                # 由 wiring 层的 auto-inject default 机制处理
                if rewritten is not _OMIT_SENTINEL:
                    new_configs[cfg_key] = rewritten
            new_spec["configs"] = new_configs

        # 保留 outputs 声明
        if "outputs" in child_spec:
            new_spec["outputs"] = copy.deepcopy(child_spec["outputs"])

        # 保留/继承 label：子节点自身 label 优先，否则继承父引用节点的 label
        child_label = child_spec.get("label")
        if child_label is None:
            child_label = parent_node_spec.get("label")
        if child_label is not None:
            new_spec["label"] = child_label

        expanded_nodes[prefixed_name] = new_spec

    # ── 6) 重写父图中引用此 SubDAG 输出的所有消费者 ──
    # 扫描父图所有节点（包括已展开的节点和尚未展开的节点），
    # 将 "parent_name.output_xxx" 替换为展平后的实际链接。
    _rewrite_consumers(parent_spec, parent_name, output_rewrite)

    # ── 7) 删除原 SubDAG 节点，插入展平后的子节点 ──
    # 保持插入顺序：在原节点位置插入展平节点
    new_nodes: dict[str, dict] = {}
    for name, node_spec in parent_nodes.items():
        if name == parent_name:
            # 替换为展平后的子节点
            new_nodes.update(expanded_nodes)
        else:
            new_nodes[name] = node_spec
    parent_spec["nodes"] = new_nodes

    # ── 8) 重写顶层 outputs 中引用此 SubDAG 的链接 ──
    top_outputs = parent_spec.get("outputs", {})
    for out_name, out_link in list(top_outputs.items()):
        if isinstance(out_link, str):
            # 检查是否引用了被展开的 SubDAG
            parts = out_link.rsplit(".", 1)
            if len(parts) == 2 and parts[0] == parent_name:
                sub_out_name = parts[1]
                if sub_out_name in output_rewrite:
                    top_outputs[out_name] = output_rewrite[sub_out_name]


def _extract_src(binding: Any) -> str:
    """从 YAML binding 中提取 src 字符串。"""
    if isinstance(binding, dict) and "src" in binding:
        return binding["src"]
    elif isinstance(binding, str):
        return binding
    else:
        # 字面量值 → 直接返回，由调用方处理
        return binding


def _rewrite_link(
    link: Any,
    parent_name: str,
    parent_inputs: dict[str, str],
    parent_configs: dict[str, Any],
    sub_inputs_spec: dict[str, dict],
    sub_configs_spec: dict[str, dict],
) -> Any:
    """重写子图内部的 link 引用。

    规则：
        - "inputs.xxx" → 查 parent_inputs 映射，若无则检查是否可选
        - "configs.xxx" → 查 parent_configs 映射，若无则用子图默认值
        - "node.output" → 添加 parent_name 前缀 → "parent.node.output"
        - 非字符串 → 直接返回（字面量值）
    """
    if not isinstance(link, str):
        # 字面量值（int, float, bool 等）
        return link

    link = link.strip()

    if link.startswith("inputs."):
        input_name = link[len("inputs."):]
        if input_name in parent_inputs:
            return parent_inputs[input_name]
        else:
            # 子图声明了此输入但父图未布线
            sub_spec = sub_inputs_spec.get(input_name, {})
            required = sub_spec.get("required", True)
            if not required:
                return INACTIVE_MARKER
            raise ValueError(
                f"SubDAG input '{input_name}' is required but not wired "
                f"by parent node '{parent_name}'.")

    if link.startswith("configs."):
        config_name = link[len("configs."):]
        if config_name in parent_configs:
            return parent_configs[config_name]
        else:
            # 父图未覆盖此 config → 省略该键，由 Op 自身 CONFIGS default 补齐
            sub_cfg = sub_configs_spec.get(config_name, {})
            if "default" in sub_cfg:
                return _OMIT_SENTINEL
            raise ValueError(
                f"SubDAG config '{config_name}' has no default and is not "
                f"provided by parent node '{parent_name}'.")

    # 子图内部节点引用（"node.output"）→ 添加 parent_name 前缀
    # 例："mean_stacker.statistics" → "main_stacker.mean_stacker.statistics"
    return f"{parent_name}.{link}"


def _rewrite_consumers(
    parent_spec: dict[str, Any],
    sub_dag_name: str,
    output_rewrite: dict[str, str],
) -> None:
    """重写父图中引用 SubDAG 输出的所有消费者节点。

    扫描所有节点的 inputs/configs section，将形如 "sub_dag_name.output_xxx"
    的引用替换为展平后的实际链接。
    """
    nodes = parent_spec.get("nodes", {})
    for node_name, node_spec in nodes.items():
        if not isinstance(node_spec, dict):
            continue
        for section in ("inputs", "configs"):
            sec_val = node_spec.get(section)
            if not isinstance(sec_val, dict):
                continue
            for arg_name, binding in list(sec_val.items()):
                src = _extract_src(binding)
                if not isinstance(src, str):
                    continue
                rewritten = _try_rewrite_sub_dag_ref(
                    src, sub_dag_name, output_rewrite)
                if rewritten is not None:
                    sec_val[arg_name] = rewritten


def _try_rewrite_sub_dag_ref(
    link: str,
    sub_dag_name: str,
    output_rewrite: dict[str, str],
) -> Optional[str]:
    """尝试将引用 SubDAG 输出的 link 重写为展平后的链接。

    link 形如 "sub_dag_name.output_xxx"。
    使用 rsplit(".", 1) 解析，匹配 sub_dag_name 后查 output_rewrite。

    Returns:
        重写后的链接字符串，或 None（不匹配时）。
    """
    if link.startswith("inputs.") or link.startswith("configs."):
        return None

    parts = link.rsplit(".", 1)
    if len(parts) != 2:
        return None

    node_name, output_name = parts
    if node_name == sub_dag_name and output_name in output_rewrite:
        return output_rewrite[output_name]

    return None


def _resolve_parent_routes(
    parent_node_spec: dict[str, Any],
    parent_spec: dict[str, Any],
) -> dict[str, str]:
    """从父图节点 spec 中提取路由选择，供子 Meta sub-DAG 的 meta_resolve 使用。

    父图引用 Meta sub-DAG 时可通过 ``routes`` 字段指定路由选择：

        pipeline:
          op: advanced_stacker.meta.yaml
          routes:                              # 独立字段
            stacker_mode: "sigma_clip"
          inputs: { data: loader.result }

    支持 ``routes.xxx`` 引用：将父图自身的已解析路由转发给子图：

        pipeline:
          op: sub.meta.yaml
          routes:
            inner_mode: routes.main_mode       # 引用父图 _resolved_routes

    Args:
        parent_node_spec: 父图中引用子 YAML 的节点 spec。
        parent_spec: 整个父图 spec（含 ``_resolved_routes``）。

    Returns:
        {route_key: choice_string} 字典，传入子图的 meta_resolve。
    """
    raw_routes = parent_node_spec.get("routes", {})
    if not isinstance(raw_routes, dict):
        return {}

    parent_resolved = parent_spec.get("_resolved_routes", {})
    result: dict[str, str] = {}

    for key, value in raw_routes.items():
        if isinstance(value, str) and value.startswith("routes."):
            # routes.xxx 引用：从父图的 _resolved_routes 中解析
            ref_key = value[len("routes."):]
            if ref_key in parent_resolved:
                result[key] = parent_resolved[ref_key]
            else:
                raise ValueError(
                    f"Route reference '{value}' cannot be resolved: "
                    f"'{ref_key}' not found in parent's _resolved_routes. "
                    f"Available: {list(parent_resolved.keys())}")
        elif isinstance(value, str):
            # 字面量选择
            result[key] = value
        else:
            raise ValueError(
                f"Invalid route value for key '{key}': {value!r}. "
                f"Expected a string (choice or 'routes.xxx' reference).")

    return result
