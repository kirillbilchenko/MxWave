"""Parse retained native speculative counters without assuming Prometheus suffix conventions."""

from __future__ import annotations

import re

_COUNTERS = ("num_drafts", "num_draft_tokens", "num_accepted_tokens",
             "num_accepted_tokens_per_pos")


def counters(text: str) -> dict[str, float]:
    """Retain named counter series and labels; reject duplicates and nonfinite values."""
    result = {}
    for line in text.splitlines():
        match = re.fullmatch(r'(vllm:spec_decode_[\w]+)(\{[^}]*\})?\s+([0-9.eE+\-]+)', line)
        if match is None:
            continue
        name = match[1].removesuffix("_total")
        if name.removeprefix("vllm:spec_decode_") not in _COUNTERS:
            continue
        key = name + (match[2] or "")
        value = float(match[3])
        if key in result or not 0 <= value < float("inf"):
            raise ValueError("Duplicate or invalid speculative metric")
        result[key] = value
    return result


def delta(before: str, after: str) -> dict[str, float]:
    """Keep new lazily created series and reject a counter reset within one arm."""
    first, second = counters(before), counters(after)
    if set(first) - set(second):
        raise ValueError("A speculative counter series disappeared")
    result = {name: value - first.get(name, 0.) for name, value in second.items()}
    if any(value < 0 for value in result.values()):
        raise ValueError("A speculative counter reset during measurement")
    return result
