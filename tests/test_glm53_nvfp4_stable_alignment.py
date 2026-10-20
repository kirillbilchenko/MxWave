"""Check token membership, uneven padding, and repeat-independent canonical order."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

SOURCE = Path(__file__).parents[1] / "integrations/glm53_nvfp4/stable_alignment.py"
spec = importlib.util.spec_from_file_location("nv_stable_alignment_test", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_uneven_expert_counts_and_heavy_padding_keep_exact_membership():
    # Three occupied experts, counts 1/5/2; block 4 introduces four masked slots.
    sentinel = 8
    a = torch.tensor([7, 8, 8, 8, 6, 3, 0, 1, 5, 8, 8, 8, 4, 2, 8, 8,
                      917, -99, 26, 88], dtype=torch.int32)
    b = torch.tensor([7, 8, 8, 8, 5, 6, 1, 3, 0, 8, 8, 8, 2, 4, 8, 8,
                      -2, 103, 8, 8], dtype=torch.int32)
    experts = torch.tensor([0, 2, 2, 5, -1], dtype=torch.int32)
    count = torch.tensor([16], dtype=torch.int32)
    expected = torch.tensor([7, 8, 8, 8, 0, 1, 3, 5, 6, 8, 8, 8, 2, 4, 8, 8,
                             8, 8, 8, 8], dtype=torch.int32)
    for source in (a, b):
        result = module.canonicalize(source, experts, count, token_count=sentinel,
                                     block_size=4, num_experts=6)
        assert torch.equal(result, expected)
        assert torch.equal(torch.sort(result[result < sentinel]).values, torch.arange(8))
    assert torch.equal(count, torch.tensor([16]))
    assert torch.equal(experts, torch.tensor([0, 2, 2, 5, -1]))
