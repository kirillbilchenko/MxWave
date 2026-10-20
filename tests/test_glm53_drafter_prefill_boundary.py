"""Prefill observations must cover complete operands and exclude unread buffer tails."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from integrations.glm53_mxfp4.drafter_prefill_boundary_probe import PrefillHashTrace
from integrations.glm53_mxfp4.review_drafter_prefill_boundary import compare_traces


def test_digest_covers_all_bits_including_scalars_and_noncontiguous_values(tmp_path: Path) -> None:
    trace = PrefillHashTrace(tmp_path, 4)
    value = torch.zeros((8, 3), dtype=torch.bfloat16).T
    trace.record("before", "operand", (value, torch.tensor(8)))
    before = trace.events[-1]["tensors"]
    value[2, 7] = -0.0
    trace.record("after", "operand", (value, torch.tensor(8)))
    after = trace.events[-1]["tensors"]
    assert before[0]["sha256"] != after[0]["sha256"]
    assert before[1] == after[1]
    assert before[0]["bytes"] == 48 and before[1]["shape"] == []


def test_unused_routing_tails_and_scratch_are_excluded(tmp_path: Path) -> None:
    trace = PrefillHashTrace(tmp_path, 4)
    inputs = {"11": torch.arange(16), "12": torch.arange(2), "13": torch.tensor(8)}
    trace.record("before", "operand", inputs, {"moe_block_size": 8})
    before = trace.events[-1]["tensors"]
    inputs["11"][8:] = 42
    inputs["12"][1:] = 42
    trace.record("after", "operand", inputs, {"moe_block_size": 8})
    assert before == trace.events[-1]["tensors"]
    inputs["11"][7] = 42
    trace.record("used", "operand", inputs, {"moe_block_size": 8})
    assert before[0]["sha256"] != trace.events[-1]["tensors"][0]["sha256"]
    count = len(trace.events)
    trace.record("scratch", "workspace", torch.zeros(10))
    assert len(trace.events) == count
    inputs["13"] = torch.tensor(9)
    with pytest.raises(ValueError, match="routing extent"):
        trace.record("invalid", "operand", inputs, {"moe_block_size": 8})


def test_trace_requires_one_complete_selected_prefill_and_resets_per_request(tmp_path: Path) -> None:
    marker = tmp_path / "probe-marker.json"
    marker.write_text(json.dumps({"active": True}))
    trace = PrefillHashTrace(tmp_path, 4)
    calls = [trace.wrapper(f"model.layers.{layer}", layer=layer)(lambda positions: positions)
             for layer in trace.layers]
    for call in calls:
        call(torch.arange(4))
    first = trace.digest_report()
    # A child name ending in a layer number must not be mistaken for a decoder layer.
    trace.counts["model.layers.0.child.0"] = 20
    assert trace.digest_report() == first
    calls[0](torch.arange(4))
    with pytest.raises(ValueError, match="exactly once"):
        trace.digest_report()
    for call in calls[1:]:
        call(torch.arange(4))
    assert trace.digest_report() == first
    marker.write_text(json.dumps({"active": False}))
    calls[0](torch.arange(4))
    assert trace.digest_report() == first
    trace.max_bytes = trace.nbytes
    with pytest.raises(MemoryError, match="byte budget"):
        trace.record("extra", "operand", torch.ones(1))


def test_review_distinguishes_equal_input_drift_from_propagation() -> None:
    first = {"logical_rows": 3973, "layers": list(range(6)), "full_tensors_hashed": True,
             "scratch_contents_excluded": True, "unused_routing_tails_excluded": True,
             "events": [{"name": name, "category": category,
                         "tensors": [{"path": "/0", "sha256": "same"}]}
                        for name, category in [("op/input", "operand"), ("op/output", "result"),
                                               ("next/input", "operand"), ("next/output", "result")]]}
    second = deepcopy(first)
    assert compare_traces(first, second)["first_different_observation"] is None
    for event in second["events"][1:]:
        event["tensors"][0]["sha256"] = "changed"
    report = compare_traces(first, second)
    assert report["first_different_result"]["name"] == "op/output"
    assert report["first_different_result"]["matching_explicit_inputs"]
    assert not report["changes"][-1]["matching_explicit_inputs"]
    second["events"].reverse()
    with pytest.raises(ValueError, match="ordering"):
        compare_traces(first, second)
