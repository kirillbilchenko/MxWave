"""Offline trace attribution must exclude CPU annotations and unrelated GPU streams."""

from __future__ import annotations

import gzip
import importlib
import json
from pathlib import Path


def test_kda_trace_only_matches_complete_same_stream_gpu_work(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))
    analyzer = importlib.import_module("analyze_full_profiles")
    reviewer = importlib.import_module("analyze_kda_profiles")
    name = "execute_1_context_0(sq0sk0)_generation_1(sq1sk257)"
    events = [{"cat": category, "ph": "X", "name": name, "ts": 0, "dur": 200,
               "tid": 7} for category in ("gpu_user_annotation", "user_annotation")]
    events += [{"cat": "kernel", "ph": "X", "name": "marlin_moe_wna16::Marlin",
                "ts": index + 1, "dur": 1, "tid": 7, "args": {"stream": 7}}
               for index in range(84)]
    events += [{"cat": category, "ph": "X", "name": symbol, "ts": 80,
                "dur": duration, "tid": stream, "args": {"stream": stream}}
               for category, symbol, stream, duration in (
                   ("kernel", "BF16TripleBitmap_MM_Kernel", 7, 10),
                   ("cpu_op", "BF16TripleBitmap_MM_Kernel", 7, 10000),
                   ("kernel", "BF16TripleBitmap_Decompress_Kernel", 9, 50))]
    path = tmp_path / "trace.gz"
    with gzip.open(path, "wt") as output:
        json.dump({"traceEvents": events}, output)
    report = analyzer.gpu_steps(path)
    assert report["complete_expert_forward_steps"] == 1
    assert report["unmatched_kernels"] == 1
    assert report["kernel_groups_us"]["bf16_compressed_linear"] == 10
    assert report["kernel_groups_us"]["bf16_weight_decode"] == 50
    assert report["steps"][0]["kernel_us"]["bf16_compressed_linear"] == 10
    assert report["steps"][0]["kernel_union_us"] == 89
    assert report["steps"][0]["uncovered_stream_us"] == 111
    paired = reviewer.matched_steps(report["steps"], report["steps"])
    assert len(paired) == 1 and paired[0]["gpu_range_ratio"] == 1
    incomplete = {**report["steps"][0], "kernel_calls": {"routed_expert_gemm": 83}}
    assert reviewer.matched_steps(report["steps"], [incomplete]) == []
    unpaired = reviewer.shape_samples([*report["steps"], incomplete])
    assert len(unpaired) == 1 and unpaired[0]["samples"] == 1
    assert unpaired[0]["median_kernel_group_calls"]["bf16_compressed_linear"] == 1
