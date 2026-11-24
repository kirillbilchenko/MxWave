"""Serving summaries must account for real arrival groups and reject incomplete streams."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations/glm53_mxfp4"))
from analyze_paired_speed import timing_metrics


def measurement() -> dict[str, Any]:
    """Give a complete stream with one latency spike outside the middle decode interval."""
    times = [.25 + i * .08 for i in range(64)]
    times[-1] += .3
    row = {"token_ids": list(range(64)), "elapsed_seconds": times[-1] + .1,
           "arrivals": [{"seconds": t, "tokens": 1} for t in times]}
    return {"requests": [row], "wall_seconds": 6., "aggregate_tokens_per_second": 64 / 6,
            "median_ttft_seconds": .25}


def test_middle_decode_excludes_first_and_drain_latency() -> None:
    """The client latency proxy must not hide request TTFT or conflate it with token rate."""
    result = timing_metrics(measurement(), 1)
    assert result["aggregate_tokens_per_second"] == pytest.approx(64 / 6)
    assert result["median_ttft_seconds"] == .25
    assert result["median_middle_interarrival_seconds"] == pytest.approx(.08)
    assert result["median_request_seconds"] > 5.


def test_partial_stream_cannot_look_faster() -> None:
    """A truncated stream must fail even if its stored group TPS looks plausible."""
    value = measurement()
    value["requests"][0]["token_ids"].pop()
    with pytest.raises(ValueError, match="64 generated tokens"):
        timing_metrics(value, 1)


def test_summary_disagreement_is_detected() -> None:
    """Corrupted aggregate statistics must not override retained request evidence."""
    value = measurement()
    value["aggregate_tokens_per_second"] *= 1.1
    with pytest.raises(ValueError, match="summary differs"):
        timing_metrics(value, 1)
