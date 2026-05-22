"""Tests for the prune (conditional disconnect) mechanism in meta_resolve."""

import pytest

from hoshicore.engine.meta import MetaResolveError, meta_resolve
from hoshicore.engine.flatten import INACTIVE_MARKER


def _make_linear_spec(enable_val: bool):
    """A → B → C, where B has enable (no bypass → prune)."""
    return {
        "configs": {
            "do_b": {"type": "bool", "default": enable_val},
        },
        "inputs": {
            "src": {"type": "sequence"},
        },
        "nodes": {
            "A": {
                "op": "OpA",
                "inputs": {"data": "inputs.src"},
                "outputs": {"result": {"type": "sequence"}},
            },
            "B": {
                "op": "OpB",
                "enable": "configs.do_b",
                "inputs": {"data": "A.result"},
                "outputs": {"result": {"type": "sequence"}},
            },
            "C": {
                "op": "OpC",
                "inputs": {"data": "B.result"},
                "outputs": {"result": {"type": "sequence"}},
            },
        },
        "outputs": {
            "out": "C.result",
        },
    }


def _make_multi_input_spec(enable_val: bool):
    """A and B feed into C (multi-input). B has enable (no bypass → prune)."""
    return {
        "configs": {
            "do_b": {"type": "bool", "default": enable_val},
        },
        "inputs": {
            "src": {"type": "sequence"},
        },
        "nodes": {
            "A": {
                "op": "OpA",
                "inputs": {"data": "inputs.src"},
                "outputs": {"result": {"type": "sequence"}},
            },
            "B": {
                "op": "OpB",
                "enable": "configs.do_b",
                "inputs": {"data": "inputs.src"},
                "outputs": {"result": {"type": "sequence"}},
            },
            "C": {
                "op": "OpC",
                "inputs": {
                    "main": "A.result",
                    "aux": "B.result",
                },
                "outputs": {"result": {"type": "sequence"}},
            },
        },
        "outputs": {
            "out": "C.result",
        },
    }


class TestPruneSingleNode:
    """Test prune of a node whose downstream has multiple inputs."""

    def test_prune_marks_inactive(self):
        spec = _make_multi_input_spec(enable_val=False)
        result = meta_resolve(spec, {})
        nodes = result["nodes"]
        assert "B" not in nodes
        assert nodes["C"]["inputs"]["aux"] == INACTIVE_MARKER

    def test_enabled_true_preserves_node(self):
        spec = _make_multi_input_spec(enable_val=True)
        result = meta_resolve(spec, {})
        assert "B" in result["nodes"]
        assert result["nodes"]["C"]["inputs"]["aux"] == "B.result"


class TestPruneCascadeLinearChain:
    """Test that prune cascades through single-input linear chains."""

    def test_cascade_deletes_downstream(self):
        spec = _make_linear_spec(enable_val=False)
        # B pruned → C only has single input from B → C also pruned
        # C pruned → output 'out' references C.result → should raise
        with pytest.raises(MetaResolveError, match="不可达"):
            meta_resolve(spec, {})

    def test_cascade_with_optional_output(self):
        spec = _make_linear_spec(enable_val=False)
        spec["outputs"]["out"] = {"src": "C.result", "required": False}
        result = meta_resolve(spec, {})
        assert "B" not in result["nodes"]
        assert "C" not in result["nodes"]
        assert "out" not in result["outputs"]

    def test_enabled_true_no_cascade(self):
        spec = _make_linear_spec(enable_val=True)
        result = meta_resolve(spec, {})
        assert "B" in result["nodes"]
        assert "C" in result["nodes"]


class TestPruneReachesOutput:
    """Test behavior when prune cascade reaches outputs."""

    def test_required_output_raises(self):
        spec = {
            "configs": {"do_saver": {"type": "bool", "default": False}},
            "inputs": {"src": {"type": "sequence"}},
            "nodes": {
                "loader": {
                    "op": "LoaderOp",
                    "inputs": {"data": "inputs.src"},
                    "outputs": {"result": {"type": "sequence"}},
                },
                "saver": {
                    "op": "SaverOp",
                    "enable": "configs.do_saver",
                    "inputs": {"data": "loader.result"},
                    "outputs": {"code": {"type": "int"}},
                },
            },
            "outputs": {"result": "saver.code"},
        }
        with pytest.raises(MetaResolveError, match="不可达"):
            meta_resolve(spec, {})

    def test_optional_output_removed(self):
        spec = {
            "configs": {"do_saver": {"type": "bool", "default": False}},
            "inputs": {"src": {"type": "sequence"}},
            "nodes": {
                "loader": {
                    "op": "LoaderOp",
                    "inputs": {"data": "inputs.src"},
                    "outputs": {"result": {"type": "sequence"}},
                },
                "saver": {
                    "op": "SaverOp",
                    "enable": "configs.do_saver",
                    "inputs": {"data": "loader.result"},
                    "outputs": {"code": {"type": "int"}},
                },
                "other": {
                    "op": "OtherOp",
                    "inputs": {"data": "inputs.src"},
                    "outputs": {"result": {"type": "sequence"}},
                },
            },
            "outputs": {
                "opt_result": {"src": "saver.code", "required": False},
                "main_result": "other.result",
            },
        }
        result = meta_resolve(spec, {})
        assert "opt_result" not in result["outputs"]
        assert result["outputs"]["main_result"] == "other.result"
        assert "saver" not in result["nodes"]
        # loader is still in nodes (upstream of pruned saver) but will be
        # eliminated by build.py's _collect_required_nodes() since it has
        # no path to any remaining output
        assert "loader" in result["nodes"]


class TestBypassStillWorks:
    """Test that bypass (穿透) mechanism is unaffected."""

    def test_bypass_rewires_downstream(self):
        spec = {
            "configs": {"do_resize": {"type": "bool", "default": False}},
            "inputs": {"src": {"type": "sequence"}},
            "nodes": {
                "loader": {
                    "op": "LoaderOp",
                    "inputs": {"data": "inputs.src"},
                    "outputs": {"result": {"type": "sequence"}},
                },
                "resize": {
                    "op": "ResizeOp",
                    "enable": "configs.do_resize",
                    "bypass": "data",
                    "inputs": {"data": "loader.result"},
                    "outputs": {"result": {"type": "sequence"}},
                },
                "saver": {
                    "op": "SaverOp",
                    "inputs": {"data": "resize.result"},
                    "outputs": {"code": {"type": "int"}},
                },
            },
            "outputs": {"result": "saver.code"},
        }
        result = meta_resolve(spec, {})
        assert "resize" not in result["nodes"]
        assert result["nodes"]["saver"]["inputs"]["data"] == "loader.result"


class TestPruneWithConfigsRef:
    """Test prune when the pruned node's output is referenced in configs section."""

    def test_configs_ref_marked_inactive(self):
        spec = {
            "configs": {"do_exif": {"type": "bool", "default": False}},
            "inputs": {"src": {"type": "sequence"}},
            "nodes": {
                "exif_loader": {
                    "op": "ExifOp",
                    "enable": "configs.do_exif",
                    "inputs": {"data": "inputs.src"},
                    "outputs": {"result": {"type": "object"}},
                },
                "main_proc": {
                    "op": "MainOp",
                    "inputs": {"data": "inputs.src"},
                    "outputs": {"result": {"type": "image"}},
                },
                "saver": {
                    "op": "SaverOp",
                    "inputs": {"img": "main_proc.result"},
                    "configs": {"exif": "exif_loader.result"},
                    "outputs": {"code": {"type": "int"}},
                },
            },
            "outputs": {"result": "saver.code"},
        }
        result = meta_resolve(spec, {})
        assert "exif_loader" not in result["nodes"]
        assert result["nodes"]["saver"]["configs"]["exif"] == INACTIVE_MARKER


class TestOutputObjectFormat:
    """Test that outputs support {src, required} object format."""

    def test_object_format_normalized(self):
        spec = {
            "configs": {},
            "inputs": {"src": {"type": "sequence"}},
            "nodes": {
                "proc": {
                    "op": "ProcOp",
                    "inputs": {"data": "inputs.src"},
                    "outputs": {"result": {"type": "sequence"}},
                },
            },
            "outputs": {
                "main": {"src": "proc.result", "required": True},
            },
        }
        result = meta_resolve(spec, {})
        assert result["outputs"]["main"] == "proc.result"

    def test_string_format_unchanged(self):
        spec = {
            "configs": {},
            "inputs": {"src": {"type": "sequence"}},
            "nodes": {
                "proc": {
                    "op": "ProcOp",
                    "inputs": {"data": "inputs.src"},
                    "outputs": {"result": {"type": "sequence"}},
                },
            },
            "outputs": {
                "main": "proc.result",
            },
        }
        result = meta_resolve(spec, {})
        assert result["outputs"]["main"] == "proc.result"


class TestPruneGlobalConfigsOverride:
    """Test that global_configs can override enable value."""

    def test_runtime_enable_prevents_prune(self):
        spec = _make_linear_spec(enable_val=False)
        result = meta_resolve(spec, {}, global_configs={"do_b": True})
        assert "B" in result["nodes"]

    def test_runtime_disable_triggers_prune(self):
        spec = _make_linear_spec(enable_val=True)
        spec["outputs"]["out"] = {"src": "C.result", "required": False}
        result = meta_resolve(spec, {}, global_configs={"do_b": False})
        assert "B" not in result["nodes"]
        assert "C" not in result["nodes"]
