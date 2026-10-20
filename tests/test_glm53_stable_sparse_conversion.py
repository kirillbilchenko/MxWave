"""Stable compaction must retain membership, invalid tails, and the native fast path."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))

from stable_sparse_conversion import convert_stable


def test_non_power_of_two_padding_is_invalid_and_output_is_contiguous() -> None:
    indices = torch.full((2, 2176), -1, dtype=torch.int32)
    indices[0, :3] = torch.tensor([4, 4, 0], dtype=torch.int32)
    requests = torch.tensor([0, 1], dtype=torch.int32)
    table = torch.tensor([[3, 7], [5, 11]], dtype=torch.int32)
    calls = []

    def native(
        req: torch.Tensor, blocks: torch.Tensor, tokens: torch.Tensor, **options: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert req is requests and blocks is table
        assert tokens.shape == (2, 4096) and options["NUM_TOPK_TOKENS"] == 4096
        assert bool((tokens[:, 2176:] == -1).all())
        calls.append(options)
        slots = torch.full_like(tokens, -1)
        slots[0, :3] = torch.tensor([28, 28, 12], dtype=torch.int32)
        return slots, torch.tensor([3, 0], dtype=torch.int32)

    slots, counts = convert_stable(
        native,
        requests,
        table,
        indices,
        BLOCK_SIZE=4,
        NUM_TOPK_TOKENS=2176,
        return_valid_counts=True,
    )
    assert len(calls) == 1 and slots.is_contiguous() and slots.shape == indices.shape
    assert slots[0, :3].tolist() == [28, 28, 12] and counts.tolist() == [3, 0]
    assert bool((slots[1] == -1).all())


def test_power_of_two_and_uncounted_calls_keep_original_objects() -> None:
    req = torch.tensor([0], dtype=torch.int32)
    table = torch.tensor([[1]], dtype=torch.int32)
    indices = torch.zeros((1, 128), dtype=torch.int32)
    result = object()

    def native(r: torch.Tensor, b: torch.Tensor, t: torch.Tensor, **options: Any) -> object:
        assert r is req and b is table and t is indices
        return result

    for counted in (False, True):
        assert (
            convert_stable(
                native,
                req,
                table,
                indices,
                NUM_TOPK_TOKENS=128,
                return_valid_counts=counted,
            )
            is result
        )


def test_bad_declared_width_is_rejected_before_native_execution() -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Native call must not run")

    with pytest.raises(ValueError, match="Declared sparse width"):
        convert_stable(
            forbidden,
            torch.zeros(1, dtype=torch.int32),
            torch.zeros((1, 1), dtype=torch.int32),
            torch.zeros((1, 2176), dtype=torch.int32),
            NUM_TOPK_TOKENS=2048,
            return_valid_counts=True,
        )


def test_installation_covers_loaded_aliases_and_future_imports(monkeypatch: Any) -> None:
    import hashlib
    from types import ModuleType

    import stable_sparse_conversion as adapter

    def native(*args: Any, **kwargs: Any) -> None:
        return None

    helper = ModuleType("vllm.test_sparse_helper")
    caller = ModuleType("vllm.test_prefill_and_decode")
    helper.triton_convert_req_index_to_global_index = native
    caller.prefill_conversion = native
    caller.decode_conversion = native
    monkeypatch.setitem(sys.modules, helper.__name__, helper)
    monkeypatch.setitem(sys.modules, caller.__name__, caller)
    monkeypatch.setattr(adapter.importlib, "import_module", lambda name: helper)
    monkeypatch.setattr(
        adapter, "NATIVE_SHA256", hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    )
    policy = adapter.install_stable_conversion()
    wrapped = helper.triton_convert_req_index_to_global_index
    assert wrapped is not native
    assert caller.prefill_conversion is caller.decode_conversion is wrapped
    assert len(policy["patched_aliases"]) == 3
    assert adapter.install_stable_conversion()["patched_aliases"] == policy["patched_aliases"]
    assert helper.triton_convert_req_index_to_global_index is wrapped
