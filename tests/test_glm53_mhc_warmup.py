"""Coverage of split-policy warmup selection without requiring a GPU."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def select():
    """Import only the CPU policy adapter."""
    path = Path(__file__).parents[1] / "integrations/glm53_mxfp4/mhc_warmup.py"
    spec = importlib.util.spec_from_file_location("glm53_mhc_warmup_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.token_sizes


def test_non_power_of_two_split_variants_are_covered(select):
    def policy(tokens):
        # Synthetic policy with variants absent at every power-of-two size.
        return {1: 9, 2: 7, 3: 5, 4: 3}.get((tokens + 63) // 64, 1)

    sizes = select(512, policy)
    assert 129 in sizes
    assert {policy(n) for n in sizes} == {policy(n) for n in range(1, 513)}
    assert {1, 7, 8, 16, 17, 512} <= set(sizes)
    assert len(sizes) < 32


def test_small_token_limit_is_respected_and_invalid_policies_fail(select):
    assert select(1, lambda n: 1) == [1]
    with pytest.raises(ValueError, match="positive"):
        select(0, lambda n: 1)
    with pytest.raises(ValueError, match="non-positive"):
        select(10, lambda n: 0)
