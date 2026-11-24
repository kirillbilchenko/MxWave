"""Native rebucketing must use private configuration and leave the live manager intact."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))
from graph_capture_policy import candidates_for_sizes


def test_candidate_builder_receives_private_empty_state_and_copied_config() -> None:
    original = {(2, 0): [object()]}
    captures = [1, 2, 4, 8]

    class Manager(SimpleNamespace):
        def _init_candidates(self) -> None:
            assert self._candidates == {} and self._capture_descs == {}
            self._candidates = {"built_by_native": self.compilation_config.cudagraph_capture_sizes}

    manager = Manager(compilation_config=SimpleNamespace(
        cudagraph_capture_sizes=captures, max_cudagraph_capture_size=8),
        _candidates=original, _capture_descs={"original": object()})
    assert candidates_for_sizes(manager, {1, 4}) == {"built_by_native": [1, 4]}
    assert manager._candidates is original
    assert manager.compilation_config.cudagraph_capture_sizes is captures
    assert manager.compilation_config.max_cudagraph_capture_size == 8
    assert "original" in manager._capture_descs


def test_empty_or_nonpositive_policies_are_rejected() -> None:
    for allowed in (set(), {0}, {-1}):
        with pytest.raises(ValueError, match="positive"):
            candidates_for_sizes(SimpleNamespace(), allowed)
