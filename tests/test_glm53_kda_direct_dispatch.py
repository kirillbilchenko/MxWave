"""Check the experimental dispatch schema without using a GPU or a native library."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

sys.path.insert(0, str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))
from benchmark_kda_direct_dispatch import summarize_graph_samples
from diagnose_kda_graph_topology import topology_signature
from kda_direct_dispatch import _validate, linear
from run_kda_dispatch_component import REVISION, command


def test_cpu_input_fails_before_any_native_library_access() -> None:
    x = torch.empty((4, 64), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="BF16 CUDA KDA"):
        _validate(x, 64, 64, 63)


def test_fused_schema_does_not_claim_to_mutate_unused_scratch() -> None:
    with FakeTensorMode():
        x = torch.empty((2, 2, 64), dtype=torch.bfloat16)
        pool = torch.empty((2, 4096), dtype=torch.bfloat16)
        buffers = [torch.empty(16, dtype=torch.uint8) for _ in range(7)]
        result = linear(x, buffers, pool, 64, 64, 1, 1, 120, 63)
        assert result.shape == (2, 2, 63)
        assert result.dtype == torch.bfloat16
        assert result is not x and result is not pool
    schema = str(torch.ops.mxwave_glm53.kda_fused_direct.default._schema)
    assert "pool" not in schema and "!" not in schema
    original = torch.ops.mxwave_glm53.kda_linear.default._schema
    scratch = next(arg for arg in original.arguments if arg.name == "pool")
    assert scratch.alias_info is not None and scratch.alias_info.is_write


def test_graph_gate_requires_stable_identical_controls_and_both_comparisons() -> None:
    order = ["reference_a", "reference_b", "direct"]
    unstable = summarize_graph_samples(
        {"reference_a": [100., 100.], "reference_b": [104., 104.], "direct": [100., 100.]}, order)
    assert not unstable["identical_controls_stable"]
    assert not unstable["performance_gate_passed"]
    slower = summarize_graph_samples(
        {"reference_a": [101., 101.], "reference_b": [100., 100.], "direct": [102.5, 102.5]}, order)
    assert slower["identical_controls_stable"]
    assert slower["direct_reference_ratios"]["reference_a"] < 1.02
    assert not slower["performance_gate_passed"]
    equal = summarize_graph_samples({name: [99., 101.] for name in order}, order)
    assert equal["performance_gate_passed"]


def test_graph_gate_rejects_incomplete_or_invalid_samples() -> None:
    order = ["reference_a", "reference_b", "direct"]
    with pytest.raises(ValueError, match="two identical controls"):
        summarize_graph_samples({"direct": [100., 100.]}, order)
    for bad in ([100.], [100., float("nan")], [0., 100.], [float("inf"), 100.]):
        with pytest.raises(ValueError, match="finite positive"):
            summarize_graph_samples({"reference_a": [100., 100.], "reference_b": [100., 100.],
                                     "direct": bad}, order)


def test_component_container_names_are_owned_by_the_existing_restoration_watchdog() -> None:
    for stage, mode in (("initial", "raw"), ("confirm", "aligned")):
        argv = command(stage, Path("/audit/kda-direct-v2"))
        assert argv[argv.index("--name") + 1] == f"mxwave-glm53-full-{REVISION}-{mode}-r1"
        assert argv[argv.index("--network") + 1] == "none"
        assert argv[argv.index("--memory") + 1] == "3g"
        assert argv[-1] == "--paired-graphs"
        assert "600" in argv and "/usr/bin/timeout" in argv
    with pytest.raises(ValueError, match="predeclared processes"):
        command("retry", Path("/audit/kda-direct-v2"))


def test_topology_signature_ignores_node_enumeration_but_retains_extra_dependency_nodes() -> None:
    def node(index: int, name: str, parents: list[int]) -> dict[str, object]:
        return {"index": index, "node_type": 0, "kernel_name": name, "grid": [1, 1, 1],
                "block": [128, 1, 1], "shared_memory": 0, "dependencies": parents}

    first = [node(0, "fill", []), node(1, "mm", [0]), node(2, "copy", [1])]
    permuted = [node(7, "copy", [8]), node(9, "fill", []), node(8, "mm", [9])]
    assert topology_signature(first) == topology_signature(permuted)
    changed = [node(0, "fill", []), node(1, "mm", [0]), node(2, "copy", [1]),
               node(3, "wait", [1])]
    assert topology_signature(first) != topology_signature(changed)
    with pytest.raises(ValueError, match="topology indices"):
        topology_signature([node(1, "mm", [0])])
