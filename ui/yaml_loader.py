"""YAML loader with !include + deep-merge override for ui.yaml files."""
import copy
from pathlib import Path
from typing import Any

import yaml

_INCLUDE_KEY = "!include"


def load_ui_yaml(path: str | Path) -> dict:
    """Load a ui.yaml file, resolving all !include directives."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        return raw or {}
    return _resolve(raw, base_dir=path.parent)


def _resolve(node: Any, base_dir: Path) -> Any:
    """Recursively walk the YAML tree and resolve !include directives."""
    if isinstance(node, dict):
        if _INCLUDE_KEY in node:
            frag_path = base_dir / node[_INCLUDE_KEY]
            fragment = _load_fragment(frag_path)
            overrides = {k: v for k, v in node.items() if k != _INCLUDE_KEY}
            merged = _deep_merge(fragment, overrides)
            return _resolve(merged, base_dir)
        return {k: _resolve(v, base_dir) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve(item, base_dir) for item in node]
    return node


def _load_fragment(frag_path: Path) -> Any:
    """Load a fragment file and recursively resolve its own !includes."""
    with open(frag_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if isinstance(raw, dict):
        return _resolve(raw, base_dir=frag_path.parent)
    return raw


def _deep_merge(base: Any, overrides: dict) -> Any:
    """Deep merge overrides into base. null values = delete key."""
    if not isinstance(base, dict):
        return base
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if value is None:
            result.pop(key, None)
        elif isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
