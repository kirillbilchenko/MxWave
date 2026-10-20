"""The isolated MLA formats must describe identical cache values and valid sparse slots."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations/glm53_mxfp4"))
from diagnose_mla_workspace import HEADS, LENGTH, ROWS, TOPK_WIDTH, WIDTH, make_inputs


def test_formats_have_identical_logical_cache_and_query() -> None:
    """FP8 uses an exact power-of-two scale; BF16 must represent precisely those values."""
    fp8 = make_inputs(torch.device("cpu"), True)
    bf16 = make_inputs(torch.device("cpu"), False)
    assert torch.equal(fp8["q"], bf16["q"])
    assert torch.equal(fp8["cache"].to(torch.bfloat16) * .5, bf16["cache"])
    assert fp8["q"].shape == (ROWS, HEADS, WIDTH)
    assert fp8["q_pe"].shape == (ROWS, HEADS, 0)


def test_sparse_slots_cannot_read_another_requests_active_prefix() -> None:
    """Every active prefix belongs to that request, while padded tails address valid slot zero."""
    values = make_inputs(torch.device("cpu"), True)
    indices = values["indices"]
    assert indices.shape == (ROWS, TOPK_WIDTH)
    for row in range(ROWS):
        assert torch.equal(indices[row, :LENGTH], torch.arange(row * LENGTH, (row + 1) * LENGTH))
    assert not bool(indices[:, LENGTH:].any())
    assert int(indices.min()) >= 0 and int(indices.max()) < values["cache"].shape[0]
