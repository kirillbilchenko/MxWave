"""Separate expected future-row changes from changes in the shared target prefix."""

from __future__ import annotations

import numpy as np

from integrations.glm53_mxfp4.review_drafter_components import compare, row_view


def test_future_row_change_is_not_attributed_to_shared_prefix() -> None:
    events = [
        {"name": name, "category": "result", "parameters": {}, "tensors": [{
            "path": "", "payload_key": key, "dtype": "torch.float32",
        }]} for name, key in (("projection/output", "a"), ("attention/output", "b"))
    ]
    meta = {"components": {
        "events": events, "capture_bytes": 100, "python_wrappers_removed": True,
    }}
    left = {"meta": meta, "x": {
        "a": np.zeros((8, 2), dtype=np.float32), "b": np.zeros((8, 2), dtype=np.float32),
        "target_positions": np.arange(54, 62),
    }}
    right = {"meta": meta, "x": {k: v.copy() for k, v in left["x"].items()}}
    right["x"]["a"][7] = 99
    right["x"]["b"][1, 0] = 0.125
    result = compare(left, right, shared=7)
    first = result["first_changed_shared_row_result"]
    assert first["event"] == "attention/output"
    assert first["shared_rows"]["different_elements_per_row"] == [0, 1, 0, 0, 0, 0, 0]
    assert result["events"][0]["tensors"][0]["shared_rows"]["bitwise_equal"]


def test_sorted_dispatch_indices_are_not_treated_as_eight_token_rows() -> None:
    event = {"name": "layer3.moe_gemm#0/input", "category": "operand"}
    value = np.arange(8)
    assert row_view(value, event, {"path": "/11"}) is None
    assert row_view(value, event, {"path": "/0"}) is value
