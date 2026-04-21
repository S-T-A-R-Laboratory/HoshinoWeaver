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
from pathlib import Path
from typing import Any, Awaitable, Optional, Sequence

from loguru import logger

from ..component.progress import DummyTracker, ProgressTracker
from ..component.queue import BaseQueue, RichContextQueue
from ..component.utils import time_cost_warpper
from ..ops.base import BaseOp
from .build import ValidatedDag, _iter_node_src_links, _parse_link
from .executor import DAGExecutor
from .flatten import INACTIVE_MARKER
from .registry import REGISTERED_OP

# ────────────────────────────────────────────────────────────────
# 子图 YAML 搜索路径
# ────────────────────────────────────────────────────────────────

# 内置 DAG 目录（基于包位置，不依赖工作目录）
_BUILTIN_DAG_DIR = Path(__file__).resolve().parent.parent / "dag"

# 默认搜索路径列表：op 字段以 .yaml 结尾时，按序搜索。
# 用户可通过 set_dag_search_paths() 追加自定义目录。
DEFAULT_DAG_SEARCH_PATHS: list[Path] = [_BUILTIN_DAG_DIR]


def set_dag_search_paths(paths: list[Path]) -> None:
    """覆盖全局 DAG 搜索路径（用于用户自定义子图目录）。

    注意：此函数修改模块级全局变量，非线程安全。
    同一进程中并发执行多个 DAG 时搜索路径会互相覆盖。
    当前单用户 CLI 场景无此问题；未来如需并发可改用 contextvars。
    """
    global DEFAULT_DAG_SEARCH_PATHS
    DEFAULT_DAG_SEARCH_PATHS = [Path(p) for p in paths]


def _resolve_sub_dag_yaml(op_name: str) -> Path:
    """将 .yaml 结尾的 op 引用解析为实际文件路径。

    解析规则：
        1. 绝对路径 → 直接使用
        2. 相对路径 → 按 DEFAULT_DAG_SEARCH_PATHS 顺序搜索，首个命中的文件生效

    Raises:
        FileNotFoundError: 所有搜索路径中均未找到该文件。
    """
    p = Path(op_name)
    if p.is_absolute():
        if p.exists():
            return p
        raise FileNotFoundError(f"Sub-DAG YAML not found: {p}")
    for search_dir in DEFAULT_DAG_SEARCH_PATHS:
        candidate = search_dir / op_name
        if candidate.exists():
            return candidate.resolve()
    searched = [str(d) for d in DEFAULT_DAG_SEARCH_PATHS]
    raise FileNotFoundError(
        f"Sub-DAG YAML '{op_name}' not found in search paths: {searched}")


# ────────────────────────────────────────────────────────────────
# Feeder 协程
# ────────────────────────────────────────────────────────────────
async def _feed_sequence(
    name: str,
    data: Sequence[Any],
    targets: list[BaseQueue],
) -> None:
    """将全局序列输入逐项推送到所有目标队列。

    1. 向每个目标队列广播序列长度 (set_length)
    2. 逐项并发推送到所有目标队列 (backpressure-safe)
    """
    length = len(data)
    logger.info(f"[Feeder] Global input '{name}': "
                f"{length} items → {len(targets)} queue(s)")
    for queue in targets:
        await queue.set_length(length)
    for item in data:
        await asyncio.gather(*(q.put(item) for q in targets))


async def _feed_config(
    name: str,
    value: Any,
    targets: list[BaseQueue],
) -> None:
    """将全局标量配置推送到所有目标队列（每队列推送一次）。"""
    logger.debug(f"[Feeder] Global config '{name}' → {len(targets)} queue(s)")
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
            op_name → Op class 的映射。None 时使用 REGISTERED_OP。

    Returns:
        ops:
            按拓扑序排列的 BaseOp 实例列表。
        feeders:
            全局 inputs / configs 的 feeder 协程列表。
        output_queues:
            DAG 全局输出名 → 用于收集结果的 RichContextQueue。
    """
    registry = op_registry or REGISTERED_OP
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
                f"Available: {sorted(registry.keys())}")
        instances[node_name] = registry[op_name](name=node_name)
        logger.debug(
            f"Instantiated '{node_name}' → {registry[op_name].__name__}")

    # ══════ 2) 布线：解析每个节点的 inputs/configs link ══════
    # 按 link 来源分类收集目标队列
    seq_targets: dict[str,
                      list[RichContextQueue]] = {}  # global input  → queues
    cfg_targets: dict[str,
                      list[RichContextQueue]] = {}  # global config → queues

    for node_name in dag.exec_order:
        node_spec = nodes_spec[node_name]
        for loc, src in _iter_node_src_links(node_spec):
            # SubDAG 展开产生的 __inactive__ 标记：跳过布线，标记队列非活跃
            if src == INACTIVE_MARKER:
                section, arg_name = loc.split(".", 1)
                op_inst = instances[node_name]
                if section == "inputs" and arg_name in op_inst.inputs:
                    op_inst.inputs[arg_name].active = False
                    logger.debug(
                        f"Skipped inactive input '{node_name}.{arg_name}' "
                        f"(from SubDAG flattening)")
                continue

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
                    target_queue)
                logger.debug(f"Wired {provider_node}.{output_name} → "
                             f"{node_name}.{section}.{arg_name}")

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
                         op_inst.config[key]))
                    logger.debug(f"Auto-inject default for unwired config "
                                 f"'{node_name}.{key}': {spec['default']}")
                else:
                    logger.warning(
                        f"Config '{key}' of node '{node_name}' "
                        f"({op_inst.__class__.__name__}) is not wired in YAML "
                        f"and has no default — node may hang in pre_execute()."
                    )

        # 检查 Op INPUTS
        for key, spec in op_inst.INPUTS.items():
            if key not in yaml_inp_keys:
                required = spec.get("required", True)
                if not required:
                    # 可选输入未布线 → 标记队列为非活跃，pre_execute 会跳过
                    op_inst.inputs[key].active = False
                    logger.debug(
                        f"Optional input '{key}' of node '{node_name}' "
                        f"({op_inst.__class__.__name__}) is not wired — marked inactive."
                    )
                else:
                    err_msg = (
                        f"Input '{key}' of node '{node_name}' "
                        f"({op_inst.__class__.__name__}) is not wired in YAML "
                        f"— node may hang in pre_execute().")
                    logger.error(err_msg)
                    raise ValueError(err_msg)

    # ══════ 3) 校验全局数据齐备 ══════
    missing_inputs = [n for n in seq_targets if n not in global_inputs]
    if missing_inputs:
        raise ValueError(
            f"Global input(s) required but not provided: {missing_inputs}")
    missing_configs = [n for n in cfg_targets if n not in effective_configs]
    if missing_configs:
        raise ValueError(f"Global config(s) required but not provided "
                         f"(and no default): {missing_configs}")

    # ══════ 4) 创建 feeder 协程 ══════
    feeders: list[Awaitable[None]] = []
    for name, targets in seq_targets.items():
        source = global_inputs[name]
        feeders.append(_feed_sequence(name, source, targets))
    for name, targets in cfg_targets.items():
        source = effective_configs[name]
        feeders.append(_feed_config(name, source, targets))
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
                    f"which is not in exec_order.")
            collector = RichContextQueue(maxsize=1)
            instances[provider_node].outputs[output_name].append(collector)
            output_queues[out_name] = collector
            logger.debug(
                f"Output '{out_name}' ← {provider_node}.{output_name}")

    # ══════ 6) 静态检测变长源冲突 ══════
    _check_variable_source_conflicts(dag, instances)

    logger.info(f"DAG wired: {len(instances)} node(s), "
                f"{len(feeders)} feeder(s), {len(output_queues)} output(s)")
    return list(instances.values()), feeders, output_queues


# ────────────────────────────────────────────────────────────────
# 执行入口
# ────────────────────────────────────────────────────────────────


async def run_dag(
    dag: ValidatedDag,
    global_inputs: dict[str, Any],
    global_configs: dict[str, Any],
    op_registry: Optional[dict[str, type[BaseOp]]] = None,
    progress: bool = True,
    dag_search_paths: Optional[list[Path]] = None,
    tracker: Optional[DummyTracker] = None,
    cancel_event: Optional[asyncio.Event] = None,
) -> dict[str, Any]:
    """
    端到端执行 DAG：实例化 → 布线 → 并发执行 → 收集结果。

    Args:
        dag:              validate_and_build_order() 的输出。
        global_inputs:    全局输入数据 (name → Sequence)。
        global_configs:   全局配置数据 (name → value)。
        op_registry:      Op 注册表，None 使用默认注册表。
        progress:         是否显示 tqdm 进度条（外部传入 tracker 时忽略）。
        dag_search_paths: 子图 YAML 搜索路径。None 使用 DEFAULT_DAG_SEARCH_PATHS。
                          传入后会覆盖模块级默认值（影响本次执行及嵌套子图）。
        tracker:          外部注入的进度追踪器。传入时优先使用，忽略 progress 参数。
        cancel_event:     外部取消事件。set() 后 Op 在下一个 _run_cpu 检查点退出。

    Returns:
        dict: 全局输出 name → value 的映射。
    """
    if dag_search_paths is not None:
        set_dag_search_paths(dag_search_paths)

    ops, feeders, output_queues = instantiate_and_wire(dag, global_inputs,
                                                       global_configs,
                                                       op_registry)

    # 注入进度追踪器：外部 tracker 优先，否则按 progress 参数创建 tqdm tracker
    if tracker is not None:
        for op in ops:
            op.tracker = tracker
    elif progress:
        tracker = ProgressTracker()
        for op in ops:
            op.tracker = tracker

    executor = DAGExecutor(ops)

    # 注入取消事件：外部传入优先，否则使用 executor 自身的 cancel_event
    if cancel_event is not None:
        executor.cancel_event = cancel_event
    for op in ops:
        op._cancel_event = executor.cancel_event

    logger.info(f"DAG execution starting ({len(ops)} nodes)...")

    # 结果收集协程 —— 必须与 executor 并发运行。
    # 原因：上游节点 _send_sentinel() 需要向 output_collector 队列
    # push SENTINEL，但该队列 maxsize=1 且已有实际结果占位。
    # 只有在结果被消费后 SENTINEL 才能入队，节点才能正常结束。
    # 如果把收集放在 gather 之后，就会形成死锁：
    #   gather 等待节点结束 → 节点等待 SENTINEL 入队 → 入队等待消费 → 消费在 gather 之后
    results: dict[str, Any] = {}

    async def _collect_outputs():

        async def _get_one(name, queue):
            results[name] = await queue.get()

        await asyncio.gather(
            *[_get_one(n, q) for n, q in output_queues.items()])

    try:
        # Feeders、Executor、结果收集 三者并发运行
        await asyncio.gather(*feeders, executor.execute(), _collect_outputs())
    except asyncio.CancelledError:
        logger.info("DAG cancelled by external request")
        raise
    finally:
        if tracker is not None:
            tracker.close_all()

    logger.info("DAG execution completed. Results collected.")
    return results


@time_cost_warpper
async def run_from_yaml(
    yaml_path: str,
    global_inputs: dict[str, Any],
    global_configs: dict[str, Any],
    op_registry: Optional[dict[str, type[BaseOp]]] = None,
    progress: bool = True,
    dag_search_paths: Optional[list[Path]] = None,
    tracker: Optional[DummyTracker] = None,
    cancel_event: Optional[asyncio.Event] = None,
    num_workers: Optional[int] = None,
    route_choices: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """
    便捷入口：从 YAML 文件加载、校验、并端到端执行 DAG。

    用法示例::

        results = await run_from_yaml(
            "hoshicore/dag/fifo_startrail.yaml",
            global_inputs={"fnames": ["a.jpg", "b.jpg", ...]},
            global_configs={"fin": 0.1, "fout": 0.1, ...},
        )

    Meta YAML 示例::

        results = await run_from_yaml(
            "hoshicore/dag/calibration_stack.meta.yaml",
            global_inputs={...},
            global_configs={...},
            route_choices={"main_stacker": "sigma_clip"},
        )

    Args:
        dag_search_paths: 子图 YAML 搜索路径列表。None 使用默认值
                          [hoshicore/dag/]。用户可追加自定义目录。
        tracker:          外部注入的进度追踪器。传入时优先使用，忽略 progress 参数。
        cancel_event:     外部取消事件。set() 后 Op 在下一个检查点退出。
        route_choices:    Meta YAML 路由选择。若 spec 包含顶层 ``routes``
                          或节点级 ``route`` 字段，传入路由选择进行编译期
                          拓扑决策。None 或 {} 表示使用各路由的默认值。
    """
    from .build import _load_yaml, validate_and_build_order
    from .flatten import flatten_sub_dags
    from .multiprocess import run_dag_multiprocess
    spec = _load_yaml(yaml_path)

    # Meta YAML 预处理：编译路由选择
    if route_choices is not None or _spec_has_routes(spec):
        from .meta import meta_resolve
        spec = meta_resolve(spec, route_choices or {})

    spec = flatten_sub_dags(spec, dag_search_paths=dag_search_paths)
    dag = validate_and_build_order(spec)
    return await run_dag_multiprocess(dag,
                                      global_inputs,
                                      global_configs,
                                      op_registry,
                                      progress=progress,
                                      dag_search_paths=dag_search_paths,
                                      tracker=tracker,
                                      cancel_event=cancel_event,
                                      num_workers=num_workers)


# ────────────────────────────────────────────────────────────────
# 内部工具
# ────────────────────────────────────────────────────────────────


def _spec_has_routes(spec: dict[str, Any]) -> bool:
    """检测 spec 是否包含路由定义（需要 meta_resolve 预处理）。"""
    if spec.get("routes"):
        return True
    nodes = spec.get("nodes")
    if isinstance(nodes, dict):
        for ns in nodes.values():
            if isinstance(ns, dict) and "route_key" in ns:
                return True
    return False


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
            logger.debug(f"Config '{name}' not provided, "
                         f"using default: {spec['default']}")
    # 保留用户传入的额外配置（YAML 未声明但代码中可能需要）
    for name, value in user_configs.items():
        if name not in resolved:
            resolved[name] = value
    return resolved


def _check_variable_source_conflicts(
    dag: ValidatedDag,
    instances: dict[str, BaseOp],
) -> None:
    """静态检测变长源冲突：不同 VARIABLE_OUTPUT 源的序列输出汇入同一节点时报错。

    沿拓扑序为每个 (node, output_port) 标记其变长源：
        - None: 固定长度
        - str:  变长源节点名（VARIABLE_OUTPUT=True 的节点）

    传播规则：
        - VARIABLE_OUTPUT 节点：自身即变长源
        - 普通节点：继承上游唯一变长源（若有）
        - 多个不同变长源汇入 → ValueError
        - 固定长度 + 变长源混合 → ValueError
    """
    nodes_spec = dag.nodes
    # (provider_node, output_port) → 变长源节点名 or None
    port_var_source: dict[tuple[str, str], Optional[str]] = {}

    for node_name in dag.exec_order:
        op_inst = instances[node_name]
        node_spec = nodes_spec[node_name]

        # ── 收集本节点序列输入的变长源 ──
        input_var_sources: set[str] = set()
        has_fixed_seq = False

        for loc, src in _iter_node_src_links(node_spec):
            if src == INACTIVE_MARKER:
                continue
            section, arg_name = loc.split(".", 1)
            if section != "inputs":
                continue
            if op_inst.INPUTS.get(arg_name, {}).get("type") != "sequence":
                continue

            parsed = _parse_link(src)
            if parsed[0] == "node":
                provider_node, output_name = parsed[1], parsed[2]
                src_var = port_var_source.get((provider_node, output_name))
                if src_var is not None:
                    input_var_sources.add(src_var)
                else:
                    has_fixed_seq = True
            else:
                # global inputs → 固定长度
                has_fixed_seq = True

        # ── 冲突检测 1：多个不同变长源 ──
        if len(input_var_sources) > 1:
            raise ValueError(
                f"Node '{node_name}' ({op_inst.__class__.__name__}) receives "
                f"sequence inputs from multiple variable-length sources: "
                f"{sorted(input_var_sources)}. "
                f"Use FilterGate pattern to align sequences from a single source."
            )

        # ── 冲突检测 2：固定 + 变长混合 ──
        if has_fixed_seq and input_var_sources:
            raise ValueError(
                f"Node '{node_name}' ({op_inst.__class__.__name__}) mixes "
                f"fixed-length and variable-length sequence inputs "
                f"(variable source: {next(iter(input_var_sources))}). "
                f"Use FilterGate pattern to align sequences before merging.")

        # ── 确定本节点序列输出的变长源 ──
        if op_inst.VARIABLE_OUTPUT:
            var_source = node_name
        elif input_var_sources:
            var_source = next(iter(input_var_sources))
        else:
            var_source = None

        for output_name, output_spec in op_inst.OUTPUTS.items():
            if output_spec.get("type") == "sequence":
                port_var_source[(node_name, output_name)] = var_source
