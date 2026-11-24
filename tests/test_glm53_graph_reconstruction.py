"""Device attribution must follow launch ownership even when GPU execution crosses CPU ranges."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))
from analyze_graph_reconstruction import DECODER, attribute


def test_asynchronous_graph_kernels_follow_correlation_and_keep_unmatched_events() -> None:
    trace = {"traceEvents": [
        {"cat": "user_annotation", "name": "execute_2048_context_8(sq2048)_generation_0(sq0)",
         "pid": 10, "tid": 10, "ts": 0, "dur": 10},
        {"cat": "user_annotation", "name": "execute_16_context_0(sq0)_generation_16(sq16)",
         "pid": 10, "tid": 10, "ts": 20, "dur": 10},
        {"cat": "cuda_driver", "name": "cuLaunchKernel", "pid": 10, "tid": 10,
         "ts": 5, "dur": 1, "args": {"correlation": 1}},
        {"cat": "cuda_runtime", "name": "cudaGraphLaunch", "pid": 10, "tid": 10,
         "ts": 25, "dur": 1, "args": {"correlation": 2}},
        # The eager kernel runs during the later CPU annotation. Time proximity would misattribute it.
        {"cat": "kernel", "name": DECODER + "eager>", "ts": 22, "dur": 4,
         "args": {"correlation": 1, "graph id": 0}},
        {"cat": "kernel", "name": DECODER + "graph>", "ts": 35, "dur": 3,
         "args": {"correlation": 2, "graph id": 7}},
        {"cat": "kernel", "name": "linear", "ts": 39, "dur": 2,
         "args": {"correlation": 2, "graph id": 7}},
        {"cat": "kernel", "name": DECODER + "unmatched>", "ts": 40, "dur": 2,
         "args": {"correlation": 999, "graph id": 0}},
    ]}
    result = attribute(trace)
    a, b = result["forwards"]
    assert not a["pure_decode"] and a["decompression_kernel_events"] == 1
    assert b["pure_decode"] and b["tokens"] == b["generation_requests"] == 16
    assert b["cuda_graph_launches"] == 1 and b["graph_decompression_kernel_events"] == 1
    assert b["graph_kernel_events"] == 2 and b["graph_kernel_span_us"] == 6
    assert result["unassigned"] == {"kernel_events": 1, "decompression_kernel_events": 1}


def test_overlapping_forwards_and_duplicate_correlations_fail_instead_of_guessing() -> None:
    annotation = {"cat": "user_annotation",
        "name": "execute_16_context_0(sq0)_generation_16(sq16)",
        "pid": 1, "tid": 1, "ts": 0, "dur": 10}
    launch = {"cat": "cuda_runtime", "name": "cudaGraphLaunch", "pid": 1, "tid": 1,
              "ts": 1, "dur": 1, "args": {"correlation": 1}}
    with pytest.raises(ValueError, match="Overlapping"):
        attribute({"traceEvents": [annotation, annotation, launch]})
    with pytest.raises(ValueError, match="Duplicate"):
        attribute({"traceEvents": [annotation, launch, launch]})
