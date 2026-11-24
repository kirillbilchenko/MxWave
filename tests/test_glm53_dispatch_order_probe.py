"""A dispatch-order ablation must preserve expert membership and exact source bytes."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file


@pytest.fixture(autouse=True)
def source_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Import the CPU-only contracts without initializing vLLM or CUDA."""
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))


def test_canonicalization_normalizes_only_within_each_expert_and_preserves_tail() -> None:
    module = importlib.import_module("dispatch_order_probe")
    left = [2, 0, 6, 6, 4, 1, 6, 6, 5, 3, 6, 6, 999]
    right = [0, 2, 6, 6, 1, 4, 6, 6, 3, 5, 6, 6, 999]
    expected = [0, 2, 6, 6, 1, 4, 6, 6, 3, 5, 6, 6, 999]
    for values in (left, right):
        actual = module.canonical_order(values, [0, 0, 2, 2, 5, 5], 12, 2)
        assert actual == expected
        assert values[-1] == 999 and actual[-1] == 999
        assert module.canonical_order(actual, [0, 0, 2, 2, 5, 5], 12, 2) == actual


@pytest.mark.parametrize("padded,block,experts", [
    (9, 2, [0] * 6), (14, 2, [0] * 7), (12, 0, [0] * 6),
    (12, 2, [0, 0, 2, 1, 1, 1]), (12, 2, [0, 0]), (12, 2, [-1] * 6),
])
def test_invalid_dispatch_bounds_and_noncontiguous_expert_groups_fail(
    padded, block, experts,
) -> None:
    module = importlib.import_module("dispatch_order_probe")
    with pytest.raises(ValueError):
        module.canonical_order(list(range(12)), experts, padded, block)


def test_source_reader_checks_metadata_payload_and_truncation(tmp_path) -> None:
    module = importlib.import_module("dispatch_order_probe")
    p = tmp_path / "model-00001-of-00001.safetensors"
    original = torch.arange(32, dtype=torch.uint8).reshape(4, 8)
    save_file({"weight": original}, str(p))
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"weight": p.name}}))
    expected = {index.name: hashlib.sha256(index.read_bytes()).hexdigest()}
    reader = module.TensorReader(tmp_path, expected)
    raw, shape = reader.raw("weight")
    assert raw == original.numpy().tobytes() and shape == [4, 8]
    p.write_bytes(p.read_bytes()[:-1])
    with pytest.raises(ValueError, match="Truncated source tensor"):
        reader.raw("weight")
    with pytest.raises(ValueError, match="metadata changed"):
        module.TensorReader(tmp_path, {index.name: "wrong"})


def test_source_reader_rejects_shard_path_escape(tmp_path) -> None:
    module = importlib.import_module("dispatch_order_probe")
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"weight": "../weights.safetensors"}}))
    reader = module.TensorReader(tmp_path, {index.name: hashlib.sha256(index.read_bytes()).hexdigest()})
    with pytest.raises(ValueError, match="Unsafe"):
        reader.raw("weight")
