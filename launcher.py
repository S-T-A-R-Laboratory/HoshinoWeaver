import os
import sys

import argparse
import asyncio
import json

from loguru import logger as default_logger

from hoshicore.component.utils import init_logger, is_support_format
from hoshicore.engine.inspect import InspectResult, inspect_yaml
from hoshicore.engine.preflight import PreflightAction, PreflightReport
from hoshicore.engine.wiring import run_from_yaml


def _parse_kv_list(items: list[str], flag_name: str) -> dict[str, str]:
    """Parse a list of KEY=VALUE strings into a dict."""
    result: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            print(f"错误：{flag_name} 格式应为 KEY=VALUE，得到：{item}",
                  file=sys.stderr)
            sys.exit(1)
        k, v = item.split("=", 1)
        result[k.strip()] = v.strip()
    return result


def _coerce_value(raw: str, declared_type: str | None) -> object:
    """Coerce a CLI string value to the appropriate Python type."""
    if declared_type == "bool":
        return raw.lower() in ("true", "1", "yes")
    if declared_type == "int":
        return int(raw)
    if declared_type == "float":
        return float(raw)
    if declared_type in ("dict", "list"):
        return json.loads(raw)
    if declared_type == "str" or declared_type is None:
        return raw
    return raw


def _dir_to_file_list(dir_path: str) -> list[str]:
    """List supported image files in a directory, sorted."""
    entries = os.listdir(dir_path)
    entries.sort()
    return [os.path.join(dir_path, x) for x in entries if is_support_format(x)]


def _print_inspect(result: InspectResult) -> None:
    """Pretty-print the inspect result."""
    print(f"Pipeline: {os.path.basename(result.yaml_path)}")
    if result.description:
        print(f"Description: {result.description}")
    print()

    # Inputs
    if result.inputs:
        print("=== Inputs ===")
        for inp in result.inputs:
            req_str = "[REQUIRED]" if inp.required else "(optional)"
            print(f"  {inp.name:<20s} {inp.type:<12s} {req_str}")
        print()

    # Routes
    if result.routes:
        print("=== Routes ===")
        for route in result.routes:
            opts = []
            for opt in route.options:
                if opt == route.default:
                    opts.append(f"{opt}*")
                else:
                    opts.append(opt)
            print(f"  {route.name:<20s} options: {', '.join(opts)}")
        print("  (* = default)")
        print()

    # Global configs
    if result.configs:
        print("=== Configs ===")
        for cfg in result.configs:
            if cfg.has_default:
                info = f"default={_format_default(cfg.default)}"
            elif not cfg.required:
                info = "(optional, no default)"
            else:
                info = "[REQUIRED]"
            print(f"  {cfg.name:<24s} {cfg.type:<8s} {info}")
        print()

    # Route configs
    if result.route_configs:
        print("=== Route Configs ===")
        for cfg in result.route_configs:
            if cfg.has_default:
                info = f"default={_format_default(cfg.default)}"
            else:
                info = "[REQUIRED]"
            print(f"  {cfg.name:<40s} {cfg.type:<8s} {info}")
        print()


def _format_default(value: object) -> str:
    if isinstance(value, str):
        return f'"{value}"'
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _cli_preflight_callback(report: PreflightReport) -> PreflightAction:
    """CLI 预检回调：打印警告和建议，等待用户选择。"""
    print("\n" + "=" * 60)
    print("  资源预检警告")
    print("=" * 60)

    for w in report.warnings:
        print(f"  ⚠ {w}")

    has_fallback = bool(report.proposed_fallbacks)
    if has_fallback:
        print("\n建议降级方案：")
        for fb in report.proposed_fallbacks:
            print(f"  {fb.config_key}: {fb.current_value} -> {fb.proposed_value}"
                  f" ({fb.reason})")
    else:
        print("\n无可用的自动降级方案。")

    print("\n请选择操作：")
    if has_fallback:
        suffix = "（资源仍可能不足）" if report.budget_exceeded_after_fallback else ""
        print(f"  [A] 应用降级并继续{suffix}")
    print("  [I] 忽略警告，按原配置继续")
    print("  [Q] 中止执行")
    print()

    while True:
        try:
            choice = input("输入选择: ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            return "abort"
        if choice in ("A", "APPLY") and has_fallback:
            return "apply"
        if choice in ("I", "IGNORE"):
            return "ignore"
        if choice in ("Q", "QUIT", "ABORT"):
            return "abort"
        hint = "A、I 或 Q" if has_fallback else "I 或 Q"
        print(f"  无效输入，请输入 {hint}")


def main():
    parser = argparse.ArgumentParser(
        description="Run the DAG-based image processing pipeline.")
    parser.add_argument("config",
                        type=str,
                        help="Path to the YAML configuration file.")
    parser.add_argument("dir",
                        type=str,
                        nargs="?",
                        default=None,
                        help="Directory containing the input images "
                        "(shortcut for --input fnames=<dir>).")
    parser.add_argument(
        "--route",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Route selection (repeatable), e.g. --route stacker=sigma_clip")
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        dest="inputs",
        help="Global input (repeatable), e.g. --input light_fnames=/path/dir")
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        dest="configs",
        help="Global config (repeatable), e.g. --config int_weight=true")
    parser.add_argument("--inspect",
                        action="store_true",
                        help="Show parameter schema and exit.")
    log_group = parser.add_mutually_exclusive_group()
    log_group.add_argument("--debug",
                           action="store_true",
                           help="Enable DEBUG level logging.")
    log_group.add_argument(
        "--trace",
        action="store_true",
        help="Enable TRACE level logging (includes [MEM] diagnostics).")

    args = parser.parse_args()

    logger = init_logger(default_logger, args.debug, args.trace, None,
                         task=os.path.splitext(os.path.basename(args.config))[0])

    yaml_path = args.config
    if not os.path.isfile(yaml_path):
        logger.error(f"Config file does not exist: {yaml_path}")
        sys.exit(1)

    # Route choices
    route_choices = _parse_kv_list(args.route, "--route")

    # --inspect mode
    if args.inspect:
        result = inspect_yaml(yaml_path,
                              route_choices=route_choices or None)
        _print_inspect(result)
        return 0

    # Build global_inputs from dir shortcut + --input flags
    input_overrides = _parse_kv_list(args.inputs, "--input")
    global_inputs: dict[str, object] = {}

    if args.dir:
        if not os.path.isdir(args.dir):
            logger.error(f"Directory does not exist: {args.dir}")
            sys.exit(1)
        global_inputs["fnames"] = _dir_to_file_list(args.dir)

    for key, val in input_overrides.items():
        if os.path.isdir(val):
            global_inputs[key] = _dir_to_file_list(val)
        else:
            global_inputs[key] = val

    # Build global_configs from --config flags with type coercion
    config_overrides = _parse_kv_list(args.configs, "--config")
    inspect_result = inspect_yaml(yaml_path, route_choices=route_choices or None)
    type_map = {c.name: c.type for c in inspect_result.configs}
    for c in inspect_result.route_configs:
        type_map[c.name] = c.type

    global_configs: dict[str, object] = {}
    for key, raw_val in config_overrides.items():
        global_configs[key] = _coerce_value(raw_val, type_map.get(key))

    # Validate required inputs
    for inp in inspect_result.inputs:
        if inp.required and inp.name not in global_inputs:
            logger.error(
                f"Required input '{inp.name}' not provided. "
                f"Use positional dir argument or --input {inp.name}=<path>")
            sys.exit(1)

    try:
        asyncio.run(
            run_from_yaml(yaml_path,
                          global_inputs,
                          global_configs,
                          route_choices=route_choices,
                          preflight_callback=_cli_preflight_callback))
    except KeyboardInterrupt:
        print("\n已中止", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        root = e.__cause__ if e.__cause__ is not None else e
        logger.error(f"{type(root).__name__}: {root}")
        if args.debug or args.trace:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
