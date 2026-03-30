"""
DAG 实例化 + 布线层：将 ValidatedDag 转化为可运行的异步管线。

职责链：
    YAML ── build.py ──► ValidatedDag ── wiring.py ──► DAGExecutor + feeders
                                                            │
                                                            ▼
                                                       run_dag() → results

布线步骤：
    1. 按拓扑序实例化 Op（通过 Op 注册表查找类）
    2. 解析每个节点的 inputs/configs link，连接队列：
       - 节点输出 → 节点输入：将下游队列 append 到上游 outputs 列表
       - 全局输入 → 节点输入：收集目标队列，创建 feeder 协程
       - 全局配置 → 节点配置：收集目标队列，创建 feeder 协程
    3. 为全局 outputs 创建收集队列
    4. run_dag() 并发运行 feeders + DAGExecutor，最后收集结果
"""

import asyncio
from typing import Any, Awaitable, Optional, Sequence

from loguru import logger

from .build import ValidatedDag, _parse_link, _iter_node_src_links
from .executor import DAGExecutor
from ..ops.base import BaseOp
from ..component.queue import RichContextQueue

# ────────────────────────────────────────────────────────────────
# 默认 Op 注册表
# ────────────────────────────────────────────────────────────────

from ..ops.dataloader import ImgDataLoaderOp
from ..ops.weight_generator import WeightGeneratorOp
from ..ops.trailstacker import TrailStackerOp
from ..ops.image_saver import ImageSaveOp

DEFAULT_OP_REGISTRY: dict[str, type[BaseOp]] = {
    # YAML 中使用的 op 名称 → 实际类
    "DataLoaderOp": ImgDataLoaderOp,
    "ImgDataLoaderOp": ImgDataLoaderOp,
    "generate_weight": WeightGeneratorOp,
    "WeightGeneratorOp": WeightGeneratorOp,
    "TrailStackerOp": TrailStackerOp,
    "ImageSaveOp": ImageSaveOp,
}


# ────────────────────────────────────────────────────────────────
# Feeder 协程
# ────────────────────────────────────────────────────────────────


async def _feed_sequence(
    name: str,
    data: Sequence[Any],
    targets: list[RichContextQueue],
) -> None:
    """将全局序列输入逐项推送到所有目标队列。

    1. 向每个目标队列广播序列长度 (set_length)
    2. 逐项并发推送到所有目标队列 (backpressure-safe)
    """
    length = len(data)
    logger.info(
        f"[Feeder] Global input '{name}': "
        f"{length} items → {len(targets)} queue(s)"
    )
    for queue in targets:
        await queue.set_length(length)
    for item in data:
        await asyncio.gather(*(q.put(item) for q in targets))


async def _feed_config(
    name: str,
    value: Any,
    targets: list[RichContextQueue],
) -> None:
    """将全局标量配置推送到所有目标队列（每队列推送一次）。"""
    logger.debug(
        f"[Feeder] Global config '{name}' → {len(targets)} queue(s)"
    )
    for queue in targets:
        await queue.put(value)


# ────────────────────────────────────────────────────────────────
# 实例化 + 布线
# ────────────────────────────────────────────────────────────────


def instantiate_and_wire(
    dag: ValidatedDag,
    global_inputs: dict[str, Any],
    global_configs: dict[str, Any],
    op_registry: Optional[dict[str, type[BaseOp]]] = None,
) -> tuple[list[BaseOp], list[Awaitable[None]], dict[str, RichContextQueue]]:
    """
    根据 ValidatedDag 实例化 Op、连接队列、生成 feeder 协程。

    Args:
        dag:
            validate_and_build_order() 的输出。
        global_inputs:
            全局输入的实际数据 (name → Sequence)。
        global_configs:
            全局配置的实际值 (name → scalar)。
            未提供的配置项将自动从 YAML default 补齐。
        op_registry:
            op_name → Op class 的映射。None 时使用 DEFAULT_OP_REGISTRY。

    Returns:
        ops:
            按拓扑序排列的 BaseOp 实例列表。
        feeders:
            全局 inputs / configs 的 feeder 协程列表。
        output_queues:
            DAG 全局输出名 → 用于收集结果的 RichContextQueue。
    """
    registry = op_registry or DEFAULT_OP_REGISTRY
    nodes_spec = dag.nodes

    # ── 补齐 global_configs 默认值 ──
    effective_configs = _resolve_configs(dag.global_configs, global_configs)

    # ══════ 1) 实例化 Op ══════
    instances: dict[str, BaseOp] = {}
    for node_name in dag.exec_order:
        op_name = nodes_spec[node_name]["op"]
        if op_name not in registry:
            raise ValueError(
                f"Op '{op_name}' (node '{node_name}') not found in registry. "
                f"Available: {sorted(registry.keys())}"
            )
        instances[node_name] = registry[op_name](name=node_name)
        logger.debug(f"Instantiated '{node_name}' → {registry[op_name].__name__}")

    # ══════ 2) 布线：解析每个节点的 inputs/configs link ══════
    # 按 link 来源分类收集目标队列
    seq_targets: dict[str, list[RichContextQueue]] = {}  # global input  → queues
    cfg_targets: dict[str, list[RichContextQueue]] = {}  # global config → queues

    for node_name in dag.exec_order:
        node_spec = nodes_spec[node_name]
        for loc, src in _iter_node_src_links(node_spec):
            parsed = _parse_link(src)
            section, arg_name = loc.split(".", 1)

            # 获取下游目标队列
            op_inst = instances[node_name]
            if section == "inputs":
                target_queue = op_inst.inputs[arg_name]
            else:  # section == "configs"
                target_queue = op_inst.config[arg_name]

            if parsed[0] == "inputs":
                # 全局序列输入 → 目标队列
                seq_targets.setdefault(parsed[1], []).append(target_queue)
            elif parsed[0] == "configs":
                # 全局标量配置 → 目标队列
                cfg_targets.setdefault(parsed[1], []).append(target_queue)
            else:
                # 节点输出 → 目标队列
                provider_node, output_name = parsed[1], parsed[2]
                instances[provider_node].outputs[output_name].append(
                    target_queue
                )
                logger.debug(
                    f"Wired {provider_node}.{output_name} → "
                    f"{node_name}.{section}.{arg_name}"
                )

    # ══════ 2b) 自动注入：YAML 未布线但 Op 声明了 default 的 config ══════
    # pre_execute() 会 await 所有 CONFIGS 键的队列。
    # 如果 YAML 没布线某个键，队列永远为空 → 永久挂起。
    # 此处检测差集，有 default 的自动注入，无 default 的发出警告。
    unwired_feeders: list[tuple[str, Any, RichContextQueue]] = []

    for node_name in dag.exec_order:
        op_inst = instances[node_name]
        node_spec = nodes_spec[node_name]

        # 收集 YAML 中已布线的 config / input 键
        yaml_cfg_keys: set[str] = set()
        cfg_section = node_spec.get("configs")
        if isinstance(cfg_section, dict):
            yaml_cfg_keys = set(cfg_section.keys())

        yaml_inp_keys: set[str] = set()
        inp_section = node_spec.get("inputs")
        if isinstance(inp_section, dict):
            yaml_inp_keys = set(inp_section.keys())

        # 检查 Op CONFIGS
        for key, spec in op_inst.CONFIGS.items():
            if key not in yaml_cfg_keys:
                if "default" in spec:
                    unwired_feeders.append(
                        (f"{node_name}.{key}", spec["default"],
                         op_inst.config[key])
                    )
                    logger.debug(
                        f"Auto-inject default for unwired config "
                        f"'{node_name}.{key}': {spec['default']}"
                    )
                else:
                    logger.warning(
                        f"Config '{key}' of node '{node_name}' "
                        f"({op_inst.__class__.__name__}) is not wired in YAML "
                        f"and has no default — node may hang in pre_execute()."
                    )

        # 检查 Op INPUTS
        for key in op_inst.INPUTS:
            if key not in yaml_inp_keys:
                logger.warning(
                    f"Input '{key}' of node '{node_name}' "
                    f"({op_inst.__class__.__name__}) is not wired in YAML "
                    f"— node may hang in pre_execute()."
                )

    # ══════ 3) 校验全局数据齐备 ══════
    missing_inputs = [n for n in seq_targets if n not in global_inputs]
    if missing_inputs:
        raise ValueError(
            f"Global input(s) required but not provided: {missing_inputs}"
        )
    missing_configs = [n for n in cfg_targets if n not in effective_configs]
    if missing_configs:
        raise ValueError(
            f"Global config(s) required but not provided "
            f"(and no default): {missing_configs}"
        )

    # ══════ 4) 创建 feeder 协程 ══════
    feeders: list[Awaitable[None]] = []
    for name, targets in seq_targets.items():
        feeders.append(_feed_sequence(name, global_inputs[name], targets))
    for name, targets in cfg_targets.items():
        feeders.append(_feed_config(name, effective_configs[name], targets))
    # 未布线 config 的默认值注入
    for label, default_val, queue in unwired_feeders:
        feeders.append(_feed_config(label, default_val, [queue]))

    # ══════ 5) 创建全局输出收集队列 ══════
    output_queues: dict[str, RichContextQueue] = {}
    # NOTE: dag.output_links 运行时实际为 dict[str, str]（dataclass 标注为 list[str]）
    out_links: dict[str, str] = dag.output_links
    for out_name, out_link in out_links.items():
        parsed = _parse_link(out_link)
        if parsed[0] == "node":
            provider_node, output_name = parsed[1], parsed[2]
            if provider_node not in instances:
                raise ValueError(
                    f"Output '{out_name}' references node '{provider_node}' "
                    f"which is not in exec_order."
                )
            collector = RichContextQueue(maxsize=1)
            instances[provider_node].outputs[output_name].append(collector)
            output_queues[out_name] = collector
            logger.debug(
                f"Output '{out_name}' ← {provider_node}.{output_name}"
            )

    logger.info(
        f"DAG wired: {len(instances)} node(s), "
        f"{len(feeders)} feeder(s), {len(output_queues)} output(s)"
    )
    return list(instances.values()), feeders, output_queues


# ────────────────────────────────────────────────────────────────
# 执行入口
# ────────────────────────────────────────────────────────────────


async def run_dag(
    dag: ValidatedDag,
    global_inputs: dict[str, Any],
    global_configs: dict[str, Any],
    op_registry: Optional[dict[str, type[BaseOp]]] = None,
) -> dict[str, Any]:
    """
    端到端执行 DAG：实例化 → 布线 → 并发执行 → 收集结果。

    Args:
        dag:            validate_and_build_order() 的输出。
        global_inputs:  全局输入数据 (name → Sequence)。
        global_configs: 全局配置数据 (name → value)。
        op_registry:    Op 注册表，None 使用默认注册表。

    Returns:
        dict: 全局输出 name → value 的映射。
    """
    ops, feeders, output_queues = instantiate_and_wire(
        dag, global_inputs, global_configs, op_registry
    )
    executor = DAGExecutor(ops)

    logger.info(f"DAG execution starting ({len(ops)} nodes)...")

    # 结果收集协程 —— 必须与 executor 并发运行。
    # 原因：上游节点 _send_sentinel() 需要向 output_collector 队列
    # push SENTINEL，但该队列 maxsize=1 且已有实际结果占位。
    # 只有在结果被消费后 SENTINEL 才能入队，节点才能正常结束。
    # 如果把收集放在 gather 之后，就会形成死锁：
    #   gather 等待节点结束 → 节点等待 SENTINEL 入队 → 入队等待消费 → 消费在 gather 之后
    results: dict[str, Any] = {}

    async def _collect_outputs():
        for name, queue in output_queues.items():
            results[name] = await queue.get()

    # Feeders、Executor、结果收集 三者并发运行
    await asyncio.gather(*feeders, executor.execute(), _collect_outputs())

    logger.info("DAG execution completed. Results collected.")
    return results


async def run_from_yaml(
    yaml_path: str,
    global_inputs: dict[str, Any],
    global_configs: dict[str, Any],
    op_registry: Optional[dict[str, type[BaseOp]]] = None,
) -> dict[str, Any]:
    """
    便捷入口：从 YAML 文件加载、校验、并端到端执行 DAG。

    用法示例::

        results = await run_from_yaml(
            "hoshicore/dag/fifo_startrail.yaml",
            global_inputs={"fnames": ["a.jpg", "b.jpg", ...]},
            global_configs={"fin": 0.1, "fout": 0.1, ...},
        )
    """
    from .build import _load_yaml, validate_and_build_order

    spec = _load_yaml(yaml_path)
    dag = validate_and_build_order(spec)
    return await run_dag(dag, global_inputs, global_configs, op_registry)


# ────────────────────────────────────────────────────────────────
# 内部工具
# ────────────────────────────────────────────────────────────────


def _resolve_configs(
    dag_config_specs: dict[str, dict[str, Any]],
    user_configs: dict[str, Any],
) -> dict[str, Any]:
    """
    合并用户提供的配置与 YAML 声明的默认值。

    优先级：用户显式提供 > YAML default > 缺失（由后续校验捕获）。
    """
    resolved: dict[str, Any] = {}
    for name, spec in dag_config_specs.items():
        if name in user_configs:
            resolved[name] = user_configs[name]
        elif "default" in spec:
            resolved[name] = spec["default"]
            logger.debug(
                f"Config '{name}' not provided, "
                f"using default: {spec['default']}"
            )
    # 保留用户传入的额外配置（YAML 未声明但代码中可能需要）
    for name, value in user_configs.items():
        if name not in resolved:
            resolved[name] = value
    return resolved
