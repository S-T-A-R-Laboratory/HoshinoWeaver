"""
DAG 构建器（仅负责：合法性检查 + 拓扑执行顺序推导）

当前阶段不实例化节点对应的 Op 实现，仅解析 YAML schema、做依赖校验。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Set, Tuple, cast, TypeAlias

import networkx as nx
import yaml

LinkDict: TypeAlias = dict[str, dict[str, Any]]


class DagSpecError(ValueError):
    pass


def _load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise DagSpecError("YAML 根节点必须是映射（object）。")
    data = cast(dict[str, Any], data)
    return data


def _require_type(obj: Any, typ: type, field: str) -> None:
    if not isinstance(obj, typ):
        raise DagSpecError(
            f"字段 `{field}` 期望类型 {typ.__name__}，但得到 {type(obj).__name__}。")


def _parse_link(link: str) -> Tuple[str, ...]:
    """
    返回 link 的解析结果：
    - ("inputs", input_name)
    - ("configs", config_name)
    - ("node", node_name, output_name)
    """
    link = link.strip()

    if link.startswith("inputs."):
        name = link[len("inputs."):]
        if not name:
            raise DagSpecError(f"非法 link: `{link}`")
        return ("inputs", name)

    if link.startswith("configs."):
        name = link[len("configs."):]
        if not name:
            raise DagSpecError(f"非法 link: `{link}`")
        return ("configs", name)

    parts = link.rsplit(".", 1)
    if len(parts) != 2:
        raise DagSpecError(
            f"link 语法不符合要求（期望 inputs.* / configs.* / node.output），但得到：`{link}`"
        )
    node_name, output_name = parts
    if not node_name or not output_name:
        raise DagSpecError(f"非法 link: `{link}`")
    return ("node", node_name, output_name)


def _iter_node_src_links(
        node_spec: dict[str, dict[str, Any]]) -> Iterable[Tuple[str, str]]:
    """
    遍历节点 spec 里 inputs/configs 下的所有 src link。
    返回 (location, src_link)，location 用于错误提示：
    - location 形如 "inputs.<arg>" 或 "configs.<arg>"
    """
    for section in ("inputs", "configs"):
        if section not in node_spec:
            continue
        sec_val = node_spec.get(section)
        if sec_val is None:
            continue
        if not isinstance(sec_val, dict):
            raise DagSpecError(f"`nodes[*].{section}` 必须是 object。")
        for arg_name, binding in sec_val.items():
            loc = f"{section}.{arg_name}"
            if isinstance(binding, dict) and "src" in binding:
                src = binding["src"]
            elif isinstance(binding, str):
                # 兼容一种更简写的写法：inputs: { x: "inputs.foo" }
                src = binding
            else:
                raise DagSpecError(
                    f"节点 {section} 绑定 `{loc}` 期望形如 {{src: <Link>}} 或简写为字符串，但得到：{repr(binding)}"
                )
            yield (loc, src)


def _collect_required_nodes(
    output_links: dict[str, Any],
    nodes: dict[str, dict[str, Any]],
    node_deps: dict[str, Set[str]],
) -> Set[str]:
    """
    计算从 outputs 出发“必须执行”的节点集合（递归依赖收集）。
    """
    required: Set[str] = set()
    stack: list[str] = []

    for link in output_links.values():
        kind = _parse_link(link)
        if kind[0] == "node":
            node_name = kind[1]
            if node_name not in nodes:
                raise DagSpecError(f"outputs 中引用的节点 `{node_name}` 不存在。")
            stack.append(node_name)

    while stack:
        nid = stack.pop()
        if nid in required:
            continue
        required.add(nid)
        for dep in node_deps.get(nid, set()):
            if dep not in required:
                stack.append(dep)

    return required


@dataclass(frozen=True)
class ValidatedDag:
    nodes: dict[str, dict[str, Any]]
    global_inputs: dict[str, dict[str, Any]]
    global_configs: dict[str, dict[str, Any]]
    output_links: dict[str, str]
    node_deps: dict[str, Set[str]]  # provider nodes -> dependent node
    exec_order: list[str]


def validate_and_build_order(
        spec: dict[str, Any],
        *,
        topo_seed: Optional[list[str]] = None) -> ValidatedDag:
    """
    - 环检测：topo 失败即环（在 required 子图内）
    - 阻塞检测：引用缺失会在字段校验阶段报错
    - 字段合法性：结构、类型、link 可解析性
    - 拓扑顺序：返回 required 节点的拓扑执行顺序
    """

    if "nodes" not in spec:
        raise DagSpecError("缺少顶层字段 `nodes`。")
    _require_type(spec["nodes"], dict, "nodes")
    nodes: dict[str, dict[str, Any]] = spec["nodes"]

    global_inputs: Optional[dict[str, str]] = spec.get("inputs", {})
    if global_inputs is None:
        global_inputs = {}
    _require_type(global_inputs, dict, "inputs")

    global_configs = spec.get("configs", {})
    if global_configs is None:
        global_configs = {}
    _require_type(global_configs, dict, "configs")

    if "outputs" not in spec:
        raise DagSpecError("缺少顶层字段 `outputs`。")
    _require_type(spec["outputs"], dict, "outputs")
    output_links = spec["outputs"]
    for key, link in output_links.items():
        if not isinstance(link, str):
            raise DagSpecError(f"`outputs的{key}` 必须是字符串 link，但得到 {str(link)}。")

    # ---------- 1) 校验全局 inputs/configs ----------
    global_inputs_types: dict[str, str] = {}
    for name, entry in global_inputs.items():
        if not isinstance(entry, dict):
            raise DagSpecError(f"`inputs.{name}` 必须是对象（包含 type 等）。")
        t = entry.get("type")
        if not isinstance(t, str) or not t:
            raise DagSpecError(f"`inputs.{name}.type` 必须是非空字符串。")
        global_inputs_types[name] = t

    global_configs_types: dict[str, str] = {}
    for name, entry in global_configs.items():
        if not isinstance(entry, dict):
            raise DagSpecError(f"`configs.{name}` 必须是对象（包含 type 等）。")
        t = entry.get("type")
        if not isinstance(t, str) or not t:
            raise DagSpecError(f"`configs.{name}.type` 必须是非空字符串。")
        global_configs_types[name] = t

    # ---------- 2) 校验每个节点字段 ----------
    # nodes 的原始声明顺序用于稳定拓扑顺序（Python 3.7+ 保留插入顺序）
    node_ids_in_order = list(nodes.keys()) if topo_seed is None else topo_seed
    for node_name, node_spec in nodes.items():
        if not isinstance(node_spec, dict):
            raise DagSpecError(f"`nodes.{node_name}` 必须是对象（NodeSpec）。")
        if "op" not in node_spec or not isinstance(node_spec["op"],
                                                   str) or not node_spec["op"]:
            raise DagSpecError(f"`nodes.{node_name}.op` 必须是非空字符串。")
        if "outputs" not in node_spec:
            raise DagSpecError(f"`nodes.{node_name}.outputs` 必须存在。")
        _require_type(node_spec["outputs"], dict, f"nodes.{node_name}.outputs")
        for out_name, out_def in node_spec["outputs"].items():
            if not isinstance(out_def, dict):
                raise DagSpecError(
                    f"`nodes.{node_name}.outputs.{out_name}` 必须是对象。")
            if "type" not in out_def or not isinstance(
                    out_def["type"], str) or not out_def["type"]:
                raise DagSpecError(
                    f"`nodes.{node_name}.outputs.{out_name}.type` 必须是非空字符串。")

        # nodes.inputs/configs 的结构
        for section in ("inputs", "configs"):
            if section not in node_spec:
                continue
            sec_val = node_spec.get(section)
            if sec_val is None:
                continue
            _require_type(sec_val, dict, f"nodes.{node_name}.{section}")
            for arg_name, binding in sec_val.items():
                if not isinstance(binding, (dict, str)):
                    raise DagSpecError(
                        f"`nodes.{node_name}.{section}.{arg_name}` 必须是 {{src: <Link>}} 或字符串简写。"
                    )

    # ---------- 3) 收集依赖与做 link 校验 ----------
    # node_deps: node -> set(provider_node)  (仅包含 provider 来自其它节点的依赖)
    node_deps: dict[str, Set[str]] = {nid: set() for nid in nodes}

    # 便于字段合法性检查：记录每个 node 的输出集合
    node_outputs: dict[str, Set[str]] = {}
    for nid, n_spec in nodes.items():
        outs: LinkDict = n_spec.get("outputs", {})
        node_outputs[nid] = set(outs.keys())

    def _validate_link_for_location(loc: str, src_kind: Tuple[str,
                                                              ...]) -> None:
        if src_kind[0] == "inputs":
            input_name = src_kind[1]
            if input_name not in global_inputs:
                raise DagSpecError(
                    f"{loc} 引用了全局输入 `inputs.{input_name}`，但它未在顶层 inputs 定义。")
            # 经验性约束：全局 inputs 应作为 sequence 提供
            if global_inputs_types.get(input_name) != "sequence":
                raise DagSpecError(
                    f"{loc} 引用了 `inputs.{input_name}`，但该输入的 type 不是 `sequence`。"
                )
        elif src_kind[0] == "configs":
            config_name = src_kind[1]
            if config_name not in global_configs:
                raise DagSpecError(
                    f"{loc} 引用了全局配置 `configs.{config_name}`，但它未在顶层 configs 定义。"
                )
            # 节点 configs 一般期望标量/对象来源；不强制，但至少不建议 sequence。
            if global_configs_types.get(
                    config_name) == "sequence" and loc.startswith("configs."):
                raise DagSpecError(
                    f"{loc} 引用了 `configs.{config_name}`，其 type 为 sequence，但节点 configs 不建议接收序列。"
                )
        else:
            _kind, node_name, out_name = src_kind
            if node_name not in nodes:
                raise DagSpecError(
                    f"{loc} 引用了节点输出 `{node_name}.{out_name}`，但节点 `{node_name}` 不存在。"
                )
            if out_name not in node_outputs.get(node_name, set()):
                raise DagSpecError(
                    f"{loc} 引用了节点输出 `{node_name}.{out_name}`，但该节点未声明此输出字段。")

    for node_name, node_spec in nodes.items():
        for loc, src in _iter_node_src_links(node_spec):
            # SubDAG 展开产生的 __inactive__ 标记：跳过校验
            if src == "__inactive__":
                continue
            src_kind = _parse_link(src)
            _validate_link_for_location(loc=f"nodes.{node_name}.{loc}",
                                        src_kind=src_kind)
            if src_kind[0] == "node":
                provider_node = src_kind[1]
                node_deps[node_name].add(provider_node)

    # ---------- 4) 从 outputs 推导 required 子图 ----------
    required_nodes = _collect_required_nodes(output_links, nodes, node_deps)
    if not required_nodes:
        # 没有 node 被 outputs 引用：这是一种“空图目标”，仍然算合法，但无执行项
        exec_order: list[str] = []
        return ValidatedDag(
            nodes=nodes,
            global_inputs=global_inputs,
            global_configs=global_configs,
            output_links=output_links,
            node_deps=node_deps,
            exec_order=exec_order,
        )

    # ---------- 5) 拓扑排序（仅在 required 子图上检测环/阻塞） ----------
    # 构建 provider -> dependent 的边
    provider_to_dependents: dict[str, Set[str]] = {
        nid: set()
        for nid in required_nodes
    }
    indegree: dict[str, int] = {nid: 0 for nid in required_nodes}
    for dependent in required_nodes:
        for provider in node_deps.get(dependent, set()):
            if provider not in required_nodes:
                # 依赖在 required 子图外，则说明输出并不真正需要它（但这通常不会发生，因为我们从依赖回溯生成了 required）
                continue
            provider_to_dependents[provider].add(dependent)
            indegree[dependent] += 1

    # 用稳定顺序选择可执行节点
    seed_order = node_ids_in_order
    idx = {nid: i for i, nid in enumerate(seed_order) if nid in required_nodes}

    # 使用 networkx 的 lexicographical_topological_sort 保持稳定 tie-breaker：
    # 每次从“当前可执行节点”里选择 idx 最小的那个。
    G: nx.DiGraph[str] = nx.DiGraph()
    for nid in required_nodes:
        G.add_node(nid)
    for provider, dependents in provider_to_dependents.items():
        for dependent in dependents:
            G.add_edge(provider, dependent)

    try:

        exec_order = list(
            nx.algorithms.dag.lexicographical_topological_sort(
                G, key=lambda n: idx.get(n, 10**9)))
    except nx.NetworkXUnfeasible:
        # 若在 required 子图上无法完成，通常就是存在环或阻塞
        remaining = [nid for nid in required_nodes if G.in_degree(nid) > 0]
        raise DagSpecError(f"拓扑排序失败（可能存在环/阻塞）。remaining nodes: {remaining}")
    if len(exec_order) != len(required_nodes):
        # 兜底：理论上不应发生，但保持和原实现一致的错误形态
        remaining = [
            nid for nid in required_nodes if nid not in set(exec_order)
        ]
        raise DagSpecError(f"拓扑排序失败（可能存在环/阻塞）。remaining nodes: {remaining}")

    return ValidatedDag(
        nodes=nodes,
        global_inputs=global_inputs,
        global_configs=global_configs,
        output_links=output_links,
        node_deps=node_deps,
        exec_order=exec_order,
    )


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("用法：python -m src.dag_builder <example.yaml>")
        return 2
    path = argv[1]
    spec = _load_yaml(path)
    validated = validate_and_build_order(spec)
    print("执行顺序（required 子图）：")
    for i, nid in enumerate(validated.exec_order, start=1):
        print(f"{i}. {nid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
