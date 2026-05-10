"""
DAG spec → Mermaid flowchart 可视化工具。

支持两种粒度：
    - pre-flatten：SubDAG 作为单节点显示，适合用户理解工作流
    - post-flatten：所有节点展开，适合调试布线

用法：
    函数式 API：
        from hoshicore.engine.visualize import spec_to_mermaid
        mermaid_str = spec_to_mermaid(spec)

    CLI：
        python -m hoshicore.engine.visualize dag.meta.yaml
        python -m hoshicore.engine.visualize dag.meta.yaml --flatten
        python -m hoshicore.engine.visualize dag.meta.yaml --route stacker=sigma_clip
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional


def spec_to_mermaid(
    spec: dict[str, Any],
    *,
    direction: str = "TD",
    title: Optional[str] = None,
) -> str:
    """将 DAG spec dict 转换为 Mermaid flowchart 字符串。

    Args:
        spec: YAML spec dict（meta_resolve / flatten 后均可）。
        direction: 图方向，"TD"（上到下）或 "LR"（左到右）。
        title: 可选标题，None 时使用 spec 的 description。

    Returns:
        Mermaid flowchart 语法字符串。
    """
    nodes: dict[str, dict[str, Any]] = spec.get("nodes", {})
    global_inputs: dict[str, Any] = spec.get("inputs", {})
    global_outputs: dict[str, Any] = spec.get("outputs", {})

    lines: list[str] = []
    graph_title = title or spec.get("description", "")
    if graph_title:
        lines.append(f"---")
        lines.append(f"title: {graph_title}")
        lines.append(f"---")

    lines.append(f"flowchart {direction}")

    # -- Global input nodes
    if global_inputs:
        lines.append("")
        lines.append("    %% Global Inputs")
        for inp_name, inp_spec in global_inputs.items():
            inp_type = inp_spec.get("type", "?") if isinstance(inp_spec, dict) else "?"
            node_id = f"input_{inp_name}"
            lines.append(f"    {node_id}[/\"{inp_name}\\n({inp_type})\"/]")
            lines.append(f"    class {node_id} inputNode")

    # -- Op nodes
    _CATEGORY_SHAPE: dict[str, tuple[str, str]] = {
        # category: (mermaid shape format, class name)
        # {id} = node_id, {label} = escaped label
        "map":    ('    {id}(["{label}"])',         "mapNode"),
        "reduce": ('    {id}[/"{label}"\\]',        "reduceNode"),
        "sink":   ('    {id}((("{label}")))',      "sinkNode"),
        "util":   ('    {id}["{label}"]',           None),
    }

    lines.append("")
    lines.append("    %% Nodes")
    tree = _build_subgraph_tree(list(nodes.keys()))
    _render_nodes_grouped(tree, nodes, lines, _CATEGORY_SHAPE)

    # -- Global output nodes
    if global_outputs:
        lines.append("")
        lines.append("    %% Global Outputs")
        for out_name in global_outputs:
            node_id = f"output_{out_name}"
            lines.append(f"    {node_id}([\"OUT: {out_name}\"])")
            lines.append(f"    class {node_id} outputNode")

    # -- Edges
    lines.append("")
    lines.append("    %% Edges")
    for node_name, node_spec in nodes.items():
        if not isinstance(node_spec, dict):
            continue
        target_id = _sanitize_id(node_name)

        for section in ("inputs", "configs"):
            sec_val = node_spec.get(section)
            if not isinstance(sec_val, dict):
                continue
            for arg_name, binding in sec_val.items():
                src = _extract_link(binding)
                if src is None or src == "__inactive__":
                    continue

                source_id, edge_label = _resolve_source(src, arg_name, section)
                if source_id:
                    lines.append(
                        f"    {source_id} -->|\"{ edge_label }\"| {target_id}")

    # -- Output edges
    for out_name, out_link in global_outputs.items():
        if not isinstance(out_link, str):
            continue
        source_id, _ = _resolve_source(out_link, out_name, "outputs")
        if source_id:
            out_id = f"output_{out_name}"
            lines.append(f"    {source_id} --> {out_id}")

    # -- Styles
    lines.append("")
    lines.append("    %% Styles")
    lines.append("    classDef inputNode fill:#e1f5fe,stroke:#0288d1")
    lines.append("    classDef outputNode fill:#e8f5e9,stroke:#388e3c")
    lines.append("    classDef subDagNode fill:#fff3e0,stroke:#f57c00")
    lines.append("    classDef routeNode fill:#f3e5f5,stroke:#7b1fa2")
    lines.append("    classDef mapNode fill:#e3f2fd,stroke:#1565c0")
    lines.append("    classDef reduceNode fill:#fce4ec,stroke:#c62828")
    lines.append("    classDef sinkNode fill:#efebe9,stroke:#4e342e")

    return "\n".join(lines)


def yaml_to_mermaid(
    yaml_path: str,
    *,
    flatten: bool = False,
    route_choices: Optional[dict[str, str]] = None,
    direction: str = "TD",
) -> str:
    """从 YAML 文件生成 Mermaid flowchart。

    Args:
        yaml_path: YAML 文件路径。
        flatten: 是否展开 SubDAG（True 显示所有底层节点）。
        route_choices: Meta YAML 路由选择。
        direction: 图方向。

    Returns:
        Mermaid flowchart 语法字符串。
    """
    from .build import _load_yaml
    spec = _load_yaml(yaml_path)

    if route_choices or _spec_needs_meta_resolve(spec):
        from .meta import meta_resolve
        spec = meta_resolve(spec, route_choices or {})

    if flatten:
        from .flatten import flatten_sub_dags
        spec = flatten_sub_dags(spec)

    return spec_to_mermaid(spec, direction=direction)


def _spec_needs_meta_resolve(spec: dict[str, Any]) -> bool:
    """检测 spec 是否需要 meta_resolve 预处理。"""
    if spec.get("routes") or spec.get("route_configs"):
        return True
    nodes = spec.get("nodes")
    if isinstance(nodes, dict):
        for ns in nodes.values():
            if isinstance(ns, dict) and ("route_key" in ns or "enable" in ns):
                return True
    return False


def _classify_node(node_spec: dict[str, Any]) -> str:
    """根据节点输入/输出类型推断节点类别。

    Returns:
        "map"    — 序列入序列出（流式处理）
        "reduce" — 序列入单值出（叠加/归约）
        "sink"   — 终端节点（无 sequence 输出，输出为 int/return_code 等）
        "util"   — 其他（单值变换、mask 生成等）
    """
    inputs = node_spec.get("inputs", {})
    outputs = node_spec.get("outputs", {})

    if not isinstance(inputs, dict):
        inputs = {}
    if not isinstance(outputs, dict):
        outputs = {}

    has_seq_input = any(
        _is_sequence_binding(inputs.get(k))
        for k in inputs
    )
    out_types = {
        (v.get("type") if isinstance(v, dict) else None)
        for v in outputs.values()
    }

    has_seq_output = "sequence" in out_types

    if has_seq_input and has_seq_output:
        return "map"
    if has_seq_input and not has_seq_output:
        if out_types & {"int", "return_code"}:
            return "sink"
        return "reduce"
    if not has_seq_input and not has_seq_output:
        if out_types & {"int", "return_code"}:
            return "sink"
        return "util"
    # no seq input but has seq output (source/loader)
    return "map"


def _is_sequence_binding(binding: Any) -> bool:
    """判断绑定目标是否来自 sequence 类型节点（通过命名启发式）。"""
    # 无法完全静态确定，但 inputs 字段存在即认为可能是 sequence 连接
    return binding is not None


def _build_subgraph_tree(
    node_names: list[str],
) -> dict[str, Any]:
    """将带 '.' 的节点名构建为嵌套层级树。

    返回结构：{"__nodes__": [顶层节点名], "prefix": {"__nodes__": [...], ...}}
    """
    tree: dict[str, Any] = {"__nodes__": []}
    for name in node_names:
        parts = name.split(".")
        if len(parts) == 1:
            tree["__nodes__"].append(name)
        else:
            cursor = tree
            for prefix in parts[:-1]:
                if prefix not in cursor:
                    cursor[prefix] = {"__nodes__": []}
                cursor = cursor[prefix]
            cursor["__nodes__"].append(name)
    return tree


def _render_nodes_grouped(
    tree: dict[str, Any],
    nodes: dict[str, dict[str, Any]],
    lines: list[str],
    category_shape: dict[str, tuple[str, str]],
    indent: int = 1,
    prefix: str = "",
) -> None:
    """递归渲染节点，嵌套层级用 subgraph 包裹。"""
    pad = "    " * indent

    for node_name in tree["__nodes__"]:
        node_spec = nodes.get(node_name)
        if not isinstance(node_spec, dict):
            continue
        _emit_single_node(node_name, node_spec, lines, category_shape, pad)

    for group_key, subtree in tree.items():
        if group_key == "__nodes__":
            continue
        full_path = f"{prefix}{group_key}" if not prefix else f"{prefix}_{group_key}"
        sg_id = _sanitize_id(full_path)
        lines.append(f"{pad}subgraph {sg_id} [\"{group_key}\"]")
        _render_nodes_grouped(
            subtree, nodes, lines, category_shape, indent + 1,
            prefix=full_path)
        lines.append(f"{pad}end")


def _emit_single_node(
    node_name: str,
    node_spec: dict[str, Any],
    lines: list[str],
    category_shape: dict[str, tuple[str, str]],
    pad: str,
) -> None:
    """渲染单个节点（形状 + class）。"""
    op_name = node_spec.get("op", "?")
    node_id = _sanitize_id(node_name)
    short_name = node_name.rsplit(".", 1)[-1]

    if isinstance(op_name, str) and op_name.endswith(".yaml"):
        label = f"{short_name}\\n[{op_name}]"
        lines.append(f"{pad}{node_id}[[\"{label}\"]]")
        lines.append(f"{pad}class {node_id} subDagNode")
    elif node_spec.get("route_key"):
        route_key = node_spec.get("route_key", "")
        label = f"{short_name}\\n{{{{{route_key}}}}}"
        lines.append(f"{pad}{node_id}{{{{{{\"{label}\"}}}}}}")
        lines.append(f"{pad}class {node_id} routeNode")
    else:
        category = _classify_node(node_spec)
        label = f"{short_name}\\n({op_name})"
        shape_fmt, cls_name = category_shape[category]
        # shape_fmt uses fixed 4-space indent; replace with current pad
        rendered = shape_fmt.format(id=node_id, label=label)
        lines.append(rendered.replace("    ", pad, 1))
        if cls_name:
            lines.append(f"{pad}class {node_id} {cls_name}")


def _sanitize_id(name: str) -> str:
    """将节点名转换为 Mermaid 合法 ID（替换特殊字符）。"""
    return name.replace(".", "_").replace("-", "_").replace(" ", "_")


def _extract_link(binding: Any) -> Optional[str]:
    """从绑定中提取 link 字符串。"""
    if isinstance(binding, dict) and "src" in binding:
        return binding["src"]
    elif isinstance(binding, str):
        return binding
    return None


def _resolve_source(
    link: str,
    arg_name: str,
    section: str,
) -> tuple[Optional[str], str]:
    """将 link 解析为 (source_node_id, edge_label)。"""
    link = link.strip()

    if link.startswith("inputs."):
        input_name = link[len("inputs."):]
        return f"input_{input_name}", arg_name

    if link.startswith("configs."):
        return None, ""

    # node.output
    parts = link.rsplit(".", 1)
    if len(parts) == 2:
        node_name = parts[0]
        source_id = _sanitize_id(node_name)
        label = arg_name if section == "inputs" else f"cfg:{arg_name}"
        return source_id, label

    return None, ""


def spec_to_mermaid_with_configs(
    spec: dict[str, Any],
    *,
    direction: str = "TD",
    title: Optional[str] = None,
) -> str:
    """同 spec_to_mermaid，但包含 configs 连线（更详细，适合调试）。

    configs 来源为节点输出时画为虚线。
    """
    nodes: dict[str, dict[str, Any]] = spec.get("nodes", {})
    global_inputs: dict[str, Any] = spec.get("inputs", {})
    global_outputs: dict[str, Any] = spec.get("outputs", {})

    lines: list[str] = []
    graph_title = title or spec.get("description", "")
    if graph_title:
        lines.append(f"---")
        lines.append(f"title: {graph_title}")
        lines.append(f"---")

    lines.append(f"flowchart {direction}")

    # -- Global inputs
    if global_inputs:
        lines.append("")
        lines.append("    %% Global Inputs")
        for inp_name, inp_spec in global_inputs.items():
            inp_type = inp_spec.get("type", "?") if isinstance(inp_spec, dict) else "?"
            node_id = f"input_{inp_name}"
            lines.append(f"    {node_id}[/\"{inp_name}\\n({inp_type})\"/]")
            lines.append(f"    class {node_id} inputNode")

    # -- Op nodes
    _CATEGORY_SHAPE_D: dict[str, tuple[str, str]] = {
        "map":    ('    {id}(["{label}"])',         "mapNode"),
        "reduce": ('    {id}[/"{label}"\\]',        "reduceNode"),
        "sink":   ('    {id}((("{label}")))',      "sinkNode"),
        "util":   ('    {id}["{label}"]',           None),
    }

    lines.append("")
    lines.append("    %% Nodes")
    tree = _build_subgraph_tree(list(nodes.keys()))
    _render_nodes_grouped(tree, nodes, lines, _CATEGORY_SHAPE_D)

    # -- Global outputs
    if global_outputs:
        lines.append("")
        lines.append("    %% Global Outputs")
        for out_name in global_outputs:
            node_id = f"output_{out_name}"
            lines.append(f"    {node_id}([\"OUT: {out_name}\"])")
            lines.append(f"    class {node_id} outputNode")

    # -- Edges
    lines.append("")
    lines.append("    %% Edges")
    for node_name, node_spec in nodes.items():
        if not isinstance(node_spec, dict):
            continue
        target_id = _sanitize_id(node_name)

        for section in ("inputs", "configs"):
            sec_val = node_spec.get(section)
            if not isinstance(sec_val, dict):
                continue
            for arg_name, binding in sec_val.items():
                src = _extract_link(binding)
                if src is None or src == "__inactive__":
                    continue

                src = src.strip()
                if src.startswith("inputs."):
                    input_name = src[len("inputs."):]
                    source_id = f"input_{input_name}"
                    lines.append(
                        f"    {source_id} -->|\"{arg_name}\"| {target_id}")
                elif src.startswith("configs."):
                    pass  # global configs: skip (no visual node)
                else:
                    parts = src.rsplit(".", 1)
                    if len(parts) == 2:
                        source_node = parts[0]
                        source_id = _sanitize_id(source_node)
                        if section == "configs":
                            lines.append(
                                f"    {source_id} -.->|\"{arg_name}\"| {target_id}")
                        else:
                            lines.append(
                                f"    {source_id} -->|\"{arg_name}\"| {target_id}")

    # -- Output edges
    for out_name, out_link in global_outputs.items():
        if not isinstance(out_link, str):
            continue
        out_link = out_link.strip()
        if out_link.startswith("inputs.") or out_link.startswith("configs."):
            continue
        parts = out_link.rsplit(".", 1)
        if len(parts) == 2:
            source_id = _sanitize_id(parts[0])
            out_id = f"output_{out_name}"
            lines.append(f"    {source_id} --> {out_id}")

    # -- Styles
    lines.append("")
    lines.append("    %% Styles")
    lines.append("    classDef inputNode fill:#e1f5fe,stroke:#0288d1")
    lines.append("    classDef outputNode fill:#e8f5e9,stroke:#388e3c")
    lines.append("    classDef subDagNode fill:#fff3e0,stroke:#f57c00")
    lines.append("    classDef routeNode fill:#f3e5f5,stroke:#7b1fa2")
    lines.append("    classDef mapNode fill:#e3f2fd,stroke:#1565c0")
    lines.append("    classDef reduceNode fill:#fce4ec,stroke:#c62828")
    lines.append("    classDef sinkNode fill:#efebe9,stroke:#4e342e")

    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────
# CLI entry point
# ────────────────────────────────────────────────────────────────

def main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="DAG YAML → Mermaid flowchart 可视化")
    parser.add_argument("yaml_path", help="DAG / Meta YAML 文件路径")
    parser.add_argument("--flatten", action="store_true",
                        help="展开 SubDAG 显示所有底层节点")
    parser.add_argument("--detailed", action="store_true",
                        help="包含 configs 连线（虚线）")
    parser.add_argument("--direction", default="LR", choices=["TD", "LR"],
                        help="图方向：TD（上到下）或 LR（左到右）")
    parser.add_argument("--route", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="路由选择（可多次指定），如 --route stacker=sigma_clip")
    parser.add_argument("-o", "--output", default=None,
                        help="输出文件路径（默认打印到 stdout）")

    args = parser.parse_args(argv[1:])

    route_choices: dict[str, str] = {}
    for item in args.route:
        if "=" not in item:
            print(f"错误：--route 格式应为 KEY=VALUE，得到：{item}", file=sys.stderr)
            return 1
        k, v = item.split("=", 1)
        route_choices[k.strip()] = v.strip()

    from .build import _load_yaml

    spec = _load_yaml(args.yaml_path)

    if route_choices or _spec_needs_meta_resolve(spec):
        from .meta import meta_resolve
        spec = meta_resolve(spec, route_choices)

    if args.flatten:
        from .flatten import flatten_sub_dags
        spec = flatten_sub_dags(spec)

    if args.detailed:
        result = spec_to_mermaid_with_configs(spec, direction=args.direction)
    else:
        result = spec_to_mermaid(spec, direction=args.direction)

    if args.output:
        if args.output.endswith(".md"):
            result = f"```mermaid\n{result}\n```"
        Path(args.output).write_text(result, encoding="utf-8")
        print(f"已写入：{args.output}")
    else:
        print(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
