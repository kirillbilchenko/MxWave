"""Select a retained subset of graph descriptors without recapture or graph-pool changes."""

from __future__ import annotations

import copy
from typing import Any


def candidates_for_sizes(manager: Any, allowed: set[int]) -> dict[Any, list[Any]]:
    """Ask the native builder to rebucket a private config copy; preserve all original state."""
    if not allowed or any(not isinstance(n, int) or n < 1 for n in allowed):
        raise ValueError("Require positive retained capture sizes")
    candidate = copy.copy(manager)
    candidate.compilation_config = copy.copy(manager.compilation_config)
    candidate.compilation_config.cudagraph_capture_sizes = sorted(allowed)
    candidate.compilation_config.max_cudagraph_capture_size = max(allowed)
    candidate._candidates = {}
    candidate._capture_descs = {}
    candidate._init_candidates()
    return candidate._candidates
