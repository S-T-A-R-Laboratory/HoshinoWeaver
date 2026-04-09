"""
SubDagOp —— 将完整子 DAG 封装为单个 Op 节点。

用法：
    1. 通过 create_sub_dag_op() 工厂从 YAML 创建 Op 类
    2. 将生成的类注册到 op_registry
    3. 在父 DAG YAML 中像普通节点一样使用

设计要点：
    - SubDagOp 的 inputs/configs 队列直接作为子 DAG 的 global_inputs/configs
      传给 instantiate_and_wire，后者通过 _bridge_queue 做流式转发
    - 子 DAG 的 output_queues 结果被收集后推入 SubDagOp 的 outputs
    - 天然支持多级嵌套：子 DAG 内部也可以包含 SubDagOp 节点

示例：
    # 从 YAML 创建一个子图 Op 类
    SigmaClipTrailOp = create_sub_dag_op(
        "hoshicore/dag/sigma_clip_trail.yaml",
        op_name="SigmaClipTrailOp",
    )
    # 注册到 op_registry
    my_registry = {**DEFAULT_OP_REGISTRY, "SigmaClipTrailOp": SigmaClipTrailOp}
    # 在父 DAG 中使用
    results = await run_dag(parent_dag, inputs, configs, op_registry=my_registry)
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Optional, Union

from loguru import logger

from ..component.queue import RichContextQueue
from .base import BaseOp


class SubDagOp(BaseOp):
    """将子 DAG 作为单个节点嵌入父图。

    子类需要设置类属性：
        SUB_DAG_SPEC: dict  —— 子 DAG 的原始 YAML spec（已解析的 dict）
        INPUTS / CONFIGS / OUTPUTS —— 与子 DAG 的全局 inputs/configs/outputs 对齐

    不要直接实例化此类，而是使用 create_sub_dag_op() 工厂函数。
    """

    SUB_DAG_SPEC: dict[str, Any] = {}

    async def _async_execute(self, configs: dict[str, Any]) -> None:
        # 延迟导入，避免循环引用（wiring → ops → sub_dag → wiring）
        from ..engine.build import validate_and_build_order
        from ..engine.wiring import instantiate_and_wire, _feed_config
        from ..engine.executor import DAGExecutor

        # ── 1) 构建子 DAG ──
        sub_dag = validate_and_build_order(self.SUB_DAG_SPEC)

        # ── 2) 组装子 DAG 的 global_inputs / global_configs ──
        # SubDagOp 的 self.inputs / self.config 队列直接作为数据源传入
        # instantiate_and_wire 会通过 isinstance 检测自动使用 _bridge_queue
        sub_global_inputs: dict[str, Any] = {}
        for key in self.INPUTS:
            sub_global_inputs[key] = self.inputs[key]

        sub_global_configs: dict[str, Any] = {}
        # configs 参数来自 pre_execute() 已解包的标量值
        for key, value in configs.items():
            sub_global_configs[key] = value

        # ── 3) 实例化 + 布线子图 ──
        # dag_search_paths 使用模块级默认值，无需显式传递
        sub_ops, sub_feeders, sub_output_queues = instantiate_and_wire(
            sub_dag, sub_global_inputs, sub_global_configs,
        )

        logger.info(
            f"[SubDag] '{self.name}': {len(sub_ops)} nodes, "
            f"{len(sub_feeders)} feeders, {len(sub_output_queues)} outputs"
        )

        # ── 4) 执行子图 + 收集结果 ──
        sub_executor = DAGExecutor(sub_ops)
        sub_results: dict[str, Any] = {}

        async def _collect_sub_outputs():
            for out_name, queue in sub_output_queues.items():
                sub_results[out_name] = await queue.get()

        await asyncio.gather(
            *sub_feeders,
            sub_executor.execute(),
            _collect_sub_outputs(),
        )

        # ── 5) 将子图结果推到 SubDagOp 的输出队列 ──
        # 过滤出本 Op 声明的输出端口（子图可能有额外内部输出）
        broadcast = {k: v for k, v in sub_results.items() if k in self.outputs}
        await self._broadcast_outputs(broadcast)

        logger.info(f"[SubDag] '{self.name}': execution completed.")


def create_sub_dag_op(
    yaml_path: Union[str, Path],
    op_name: str = "SubDagOp",
    op_registry: Optional[dict[str, type[BaseOp]]] = None,
) -> type[SubDagOp]:
    """从 YAML 文件动态创建一个 SubDagOp 子类。

    该函数会：
        1. 加载并解析 YAML spec
        2. 从 spec 的 inputs/configs/outputs 推导出 Op 的 INPUTS/CONFIGS/OUTPUTS
        3. 动态创建一个 SubDagOp 子类并返回

    Args:
        yaml_path:   子 DAG 的 YAML 文件路径。
        op_name:     生成的类名（也用于 op_registry 注册键）。
        op_registry: 子 DAG 内部使用的 Op 注册表。None 使用默认注册表。

    Returns:
        一个 SubDagOp 子类，可直接注册到 op_registry 使用。
    """
    from ..engine.build import _load_yaml

    yaml_path = Path(yaml_path)
    spec = _load_yaml(str(yaml_path))

    # 从 YAML spec 推导 INPUTS
    inputs_spec: dict[str, dict[str, Any]] = {}
    for name, entry in spec.get("inputs", {}).items():
        inputs_spec[name] = {
            "type": entry.get("type", "sequence"),
            "required": True,
        }

    # 从 YAML spec 推导 CONFIGS
    configs_spec: dict[str, dict[str, Any]] = {}
    for name, entry in spec.get("configs", {}).items():
        cfg: dict[str, Any] = {"type": entry.get("type", "scalar")}
        if "default" in entry:
            cfg["default"] = entry["default"]
        configs_spec[name] = cfg

    # 从 YAML spec 推导 OUTPUTS
    outputs_spec: dict[str, dict[str, Any]] = {}
    for name in spec.get("outputs", {}):
        # 子 DAG 输出默认为单值（非序列），子类可覆盖
        outputs_spec[name] = {"type": "image"}

    # 动态创建子类
    cls = type(op_name, (SubDagOp,), {
        "SUB_DAG_SPEC": spec,
        "INPUTS": inputs_spec,
        "CONFIGS": configs_spec,
        "OUTPUTS": outputs_spec,
    })

    return cls
