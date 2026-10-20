"""Require complete finite warmup coverage instead of only common batch shapes."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def select():
    """Read the CPU planner without loading CUDA or the serving worker."""
    path = Path(__file__).parents[1] / "integrations/glm53_mxfp4/stable_dispatch_warmup.py"
    spec = importlib.util.spec_from_file_location("stable_dispatch_warmup_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.specializations


def test_every_supported_block_capacity_is_covered(select):
    cases = select()
    ordinary = (32, 64, 128, 256, 512, 1024, 2048)
    expected = {(b, c) for b in (8, 16, 32) for c in ordinary}
    expected |= {(48, c) for c in (64, 128, 256, 512, 1024, 2048, 4096)}
    expected |= {(64, c) for c in (64, 128, 256, 512, 1024, 2048)}
    assert {(c["block"], c["capacity"]) for c in cases} == expected
    assert len(cases) == len(expected) == 34
    for block in (8, 16, 32, 48, 64):
        covered = [n for c in cases if c["block"] == block
                   for n in range(c["first_tokens"], c["last_tokens"] + 1)]
        assert sorted(covered) == list(range(2, 2049))


def test_observed_cold_shapes_and_rounded_upper_boundary_have_representatives(select):
    cases = {(c["block"], c["capacity"]): c for c in select()}
    for block, capacity, observed in ((48, 2048, 1536), (16, 1024, 518)):
        case = cases[block, capacity]
        assert case["first_tokens"] <= observed <= case["last_tokens"]
    assert cases[48, 4096]["padded_tokens"] == 2064
    assert cases[48, 4096]["last_tokens"] == 2048


def test_bounded_small_limits_and_invalid_limits(select):
    assert select(1) == []
    assert all(c["last_tokens"] <= 17 for c in select(17))
    for invalid in (0, -1, 2049, 2.5):
        with pytest.raises(ValueError, match="token limit"):
            select(invalid)
