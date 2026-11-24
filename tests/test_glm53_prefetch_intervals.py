"""Overlap evidence must measure interval unions rather than double-count GPU duration."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def prefetch(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import the CPU interval helper without loading a native CUDA library."""
    path = Path(__file__).parents[1] / "integrations/glm53_mxfp4/bf16_prefetch.py"
    spec = importlib.util.spec_from_file_location("glm53_prefetch_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(("left", "right", "expected"), [
    ([(0, 10), (2, 3), (0, 10)], [(5, 15)], 5),
    ([(0, 2), (4, 6)], [(1, 5)], 2),
    ([(0, 2)], [(2, 4)], 0),
    ([], [(0, 3)], 0),
])
def test_overlap_counts_each_instant_once(
    prefetch: ModuleType, left: list[tuple[float, float]], right: list[tuple[float, float]],
    expected: float,
) -> None:
    assert prefetch.interval_overlap(left, right) == expected
    assert prefetch.interval_overlap(right, left) == expected


def test_reversed_interval_cannot_be_reported_as_overlap(prefetch: ModuleType) -> None:
    with pytest.raises(ValueError, match="end before"):
        prefetch.interval_overlap([(5, 2)], [(0, 9)])
