"""Tests for ui.yaml_loader: !include + deep-merge override."""
import textwrap
from pathlib import Path

import pytest

from ui.yaml_loader import load_ui_yaml, _deep_merge


@pytest.fixture
def tmp_yaml(tmp_path):
    """Helper to write yaml files under tmp_path and return paths."""
    def _write(name: str, content: str) -> Path:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(content), encoding="utf-8")
        return p
    return _write


class TestDeepMerge:
    def test_override_leaf(self):
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        result = _deep_merge(base, {"a": {"x": 10}})
        assert result == {"a": {"x": 10, "y": 2}, "b": 3}

    def test_null_deletes(self):
        base = {"a": 1, "b": 2, "c": 3}
        result = _deep_merge(base, {"b": None})
        assert result == {"a": 1, "c": 3}

    def test_null_deletes_nested(self):
        base = {"a": {"x": 1, "y": 2}}
        result = _deep_merge(base, {"a": {"y": None}})
        assert result == {"a": {"x": 1}}

    def test_add_new_key(self):
        base = {"a": 1}
        result = _deep_merge(base, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_replace_non_dict_with_value(self):
        base = {"a": "old"}
        result = _deep_merge(base, {"a": "new"})
        assert result == {"a": "new"}

    def test_base_not_mutated(self):
        base = {"a": {"x": 1}}
        _deep_merge(base, {"a": {"x": 99}})
        assert base == {"a": {"x": 1}}


class TestLoadUiYaml:
    def test_no_include(self, tmp_yaml):
        main = tmp_yaml("main.yaml", """\
            routes:
              stacker:
                label: "test"
            configs:
              weight:
                widget: switch
        """)
        result = load_ui_yaml(main)
        assert result["routes"]["stacker"]["label"] == "test"
        assert result["configs"]["weight"]["widget"] == "switch"

    def test_pure_include(self, tmp_yaml):
        tmp_yaml("fragments/rc.yaml", """\
            sigma_clip:
              rej_low:
                widget: slider
                max: 5.0
            huber_mean:
              huber_c:
                widget: slider
        """)
        main = tmp_yaml("main.yaml", """\
            route_configs:
              "!include": fragments/rc.yaml
        """)
        result = load_ui_yaml(main)
        assert result["route_configs"]["sigma_clip"]["rej_low"]["max"] == 5.0
        assert result["route_configs"]["huber_mean"]["huber_c"]["widget"] == "slider"

    def test_include_with_deep_override(self, tmp_yaml):
        tmp_yaml("fragments/rc.yaml", """\
            sigma_clip:
              rej_low:
                widget: slider
                min: 0
                max: 5.0
              max_iter:
                widget: slider
                min: 1
                max: 10
              early_converge_ratio:
                widget: slider
        """)
        main = tmp_yaml("main.yaml", """\
            route_configs:
              "!include": fragments/rc.yaml
              sigma_clip:
                max_iter:
                  max: 20
                early_converge_ratio: null
        """)
        result = load_ui_yaml(main)
        rc = result["route_configs"]
        assert rc["sigma_clip"]["rej_low"]["max"] == 5.0
        assert rc["sigma_clip"]["max_iter"]["max"] == 20
        assert rc["sigma_clip"]["max_iter"]["min"] == 1
        assert "early_converge_ratio" not in rc["sigma_clip"]

    def test_include_in_list(self, tmp_yaml):
        tmp_yaml("fragments/output.yaml", """\
            filename_key: output_filename
            label: "输出图像"
            dtype_options: ["uint8", "uint16"]
        """)
        main = tmp_yaml("main.yaml", """\
            outputs:
              - "!include": fragments/output.yaml
                dtype_options: ["uint8", "uint16", "uint32"]
        """)
        result = load_ui_yaml(main)
        out = result["outputs"][0]
        assert out["filename_key"] == "output_filename"
        assert out["dtype_options"] == ["uint8", "uint16", "uint32"]

    def test_nested_include(self, tmp_yaml):
        tmp_yaml("fragments/inner.yaml", """\
            huber_c:
              widget: slider
              min: 0.1
              max: 5.0
        """)
        # outer.yaml is in fragments/, so inner.yaml path is relative to fragments/
        tmp_yaml("fragments/outer.yaml", """\
            sigma_clip:
              rej_low:
                widget: slider
            huber_mean:
              "!include": inner.yaml
        """)
        main = tmp_yaml("main.yaml", """\
            route_configs:
              "!include": fragments/outer.yaml
        """)
        result = load_ui_yaml(main)
        assert result["route_configs"]["huber_mean"]["huber_c"]["widget"] == "slider"
        assert result["route_configs"]["sigma_clip"]["rej_low"]["widget"] == "slider"

    def test_empty_yaml(self, tmp_yaml):
        main = tmp_yaml("main.yaml", "")
        result = load_ui_yaml(main)
        assert result == {}
