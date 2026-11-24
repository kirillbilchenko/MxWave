"""Separate expected future-row changes from changes in the shared target prefix."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from integrations.glm53_mxfp4.audit_drafter_feature_export_numpy import arrays
from integrations.glm53_mxfp4.review_drafter_components import active_value, compare, row_view


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


def test_dispatch_uses_padded_count_instead_of_unused_capacity() -> None:
    leaves = [{"path": f"/{i}", "payload_key": str(i)} for i in (11, 12, 13)]
    event = {"name": "layer3.moe_gemm#0/input", "category": "operand",
             "parameters": {"moe_block_size": 8}, "tensors": leaves}
    frame = {"x": {"11": np.arange(64), "12": np.arange(8), "13": np.array([24])}}
    assert active_value(frame, event, leaves[0]).tolist() == list(range(24))
    assert active_value(frame, event, leaves[1]).tolist() == [0, 1, 2]


def test_fp8_trace_is_read_as_exact_bytes_without_numerical_conversion(tmp_path: Path) -> None:
    header = json.dumps({"x": {"dtype": "F8_E4M3", "shape": [4], "data_offsets": [0, 4]}}).encode()
    path = tmp_path / "fp8.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + bytes([0, 128, 126, 255]))
    assert arrays(path)["x"].tolist() == [0, 128, 126, 255]


def test_many_fields_share_one_readonly_mapping(tmp_path: Path) -> None:
    header = json.dumps({str(i): {"dtype": "I32", "shape": [1],
                                 "data_offsets": [i * 4, (i + 1) * 4]}
                         for i in range(1200)}).encode()
    path = tmp_path / "many.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header
                     + np.arange(1200, dtype="<i4").tobytes())
    result = arrays(path)
    assert len({id(x.base) for x in result.values()}) == 1
    assert [int(x[0]) for x in result.values()] == list(range(1200))
    assert not result["0"].flags.writeable


def test_expert_contributions_keep_token_and_route_order() -> None:
    event = {"name": "layer4.moe_gemm#0/output", "category": "result",
             "parameters": {"size_m": 8}}
    x = np.arange(64 * 2).reshape(64, 2)
    y = row_view(x, event, {"path": ""})
    assert y.shape == (8, 8, 2)
    assert np.array_equal(y[3, 5], x[3 * 8 + 5])
