"""Guard graph coverage and overlapping device-time accounting in the C1 diagnosis."""

from __future__ import annotations

import gzip
import importlib
import json
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def analysis(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import the offline analyzer without importing a model runtime."""
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))
    return importlib.import_module("analyze_c1_runtime")


def test_incomplete_graph_cannot_be_a_target_or_drafter(analysis: ModuleType) -> None:
    rows = [{"name": "marlin_moe_wna16::Marlin"}] * 83
    assert analysis.graph_role(rows) == "unknown_graph"
    draft = ([{"name": "rotary_embedding_kernel"}] * 6
             + [{"name": "_grouped_conv_kernel"}] * 24)
    assert analysis.graph_role(draft) == "draft"
    assert analysis.graph_role(draft[:-1]) == "unknown_graph"
    native = (draft + [{"name": "rotary_embedding_kernel"}]
              + [{"name": "kernel_unified_attention"}] * 6
              + [{"name": "_selector_walk_kernel"}])
    assert analysis.graph_role(native) == "draft"
    assert analysis.graph_role(native[:-1]) == "unknown_graph"


@pytest.mark.parametrize("fragmented_annotations", [False, True])
def test_cycles_union_overlap_and_bind_head_to_cpu_shape(
    analysis: ModuleType, tmp_path: Path, fragmented_annotations: bool,
) -> None:
    events = []
    for index in range(6):
        start = index * 10.
        events.extend({"ph": "X", "cat": "kernel", "name": "marlin_moe_wna16::Marlin",
                       "ts": start, "dur": 1., "args": {"graph id": 1, "correlation": index}}
                      for _ in range(84))
        if fragmented_annotations:
            name = f"execute_6_generation_1(row{index})"
            events.append({"cat": "user_annotation", "name": name, "ts": start})
            events.extend({"cat": "gpu_user_annotation", "name": name,
                           "ts": start + offset, "dur": .25} for offset in (0., .75))
        events.extend([
            {"ph": "X", "cat": "kernel", "name": "nccl_allreduce", "ts": start + 2,
             "dur": 2., "args": {}},
            {"ph": "X", "cat": "kernel", "name": "dense_kernel", "ts": start + 3,
             "dur": 3., "args": {"correlation": 100 + index}},
            {"cat": "cuda_runtime", "args": {"correlation": 100 + index,
                                               "External id": 200 + index}},
            {"cat": "cpu_op", "name": "aten::mm", "args": {
                "External id": 200 + index, "Input Dims": [[6, 4096], [4096, 77440]]}},
        ])
    if fragmented_annotations:
        events.append({"cat": "user_annotation", "name": "execute_6_generation_1(missing)",
                       "ts": 35.})
    path = tmp_path / "trace.json.gz"
    with gzip.open(path, "wt") as stream:
        json.dump({"traceEvents": events}, stream)
    result = analysis.inspect_trace(path)
    summary = result["central_summary"]
    assert summary["cycles"] == (1 if fragmented_annotations else 2)
    assert len(result["all_cycles"]) == 5
    if fragmented_annotations:
        assert result["adjacent_graph_native_ordinal_gaps"] == {"1": 4, "2": 1}
    assert summary["median_span_ms"] == pytest.approx(.010)
    assert summary["median_gpu_kernel_union_ms"] == pytest.approx(.005)
    assert summary["median_no_traced_kernel_ms"] == pytest.approx(.005)
    assert summary["median_cross_group_overlap_ms"] == pytest.approx(.001)
    assert summary["median_group_union_ms"]["head/dense_vocabulary"] == pytest.approx(.003)
    assert result["graph_role_counts"] == {"target": 6}
