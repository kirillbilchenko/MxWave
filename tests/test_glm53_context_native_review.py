"""The native comparison must reject changing streams, counters and draft policies."""

from __future__ import annotations

import copy

import pytest

from integrations.glm53_mxfp4.review_context_native import (
    geometric_mean,
    metric,
    prefix_delivery,
    summarize_requests,
)


def requests() -> list[dict]:
    """Build six fixed-budget streams with ordinary Prometheus counter labels."""
    counters = {
        'vllm:spec_decode_num_drafts_total{engine="0"}': 50.0,
        'vllm:spec_decode_num_draft_tokens_total{engine="0"}': 350.0,
        'vllm:spec_decode_num_accepted_tokens_total{engine="0"}': 210.0,
    }
    return [{
        "trial": i, "counter_delta": dict(counters),
        "result": {"token_ids": list(range(256)), "usage": {"completion_tokens": 256},
                   "latency_seconds": 4.0, "ttft_seconds": 0.25},
    } for i in range(6)]


def test_native_stream_summary_and_paired_geometric_mean() -> None:
    result = summarize_requests(requests())
    assert result["median_end_to_end_tps"] == 64.0
    assert result["accepted_per_draft_call"] == 4.2
    assert result["acceptance_fraction"] == 0.6
    assert geometric_mean([0.5, 2.0]) == 1.0


@pytest.mark.parametrize("change", ["token", "counter", "trial"])
def test_reject_inexact_repeat(change: str) -> None:
    rows = copy.deepcopy(requests())
    if change == "token":
        rows[-1]["result"]["token_ids"][10] = 999
    elif change == "counter":
        rows[-1]["counter_delta"]['vllm:spec_decode_num_drafts_total{engine="0"}'] = 51.0
    else:
        rows[-1]["trial"] = 0
    with pytest.raises(ValueError):
        summarize_requests(rows)


def test_reject_wrong_draft_width_and_ambiguous_counter() -> None:
    rows = requests()
    for row in rows:
        row["counter_delta"]['vllm:spec_decode_num_draft_tokens_total{engine="0"}'] = 150
    with pytest.raises(ValueError, match="seven-token"):
        summarize_requests(rows)
    with pytest.raises(ValueError, match="one aggregate"):
        metric({
            'vllm:spec_decode_num_drafts_total{engine="0"}': 5,
            'vllm:spec_decode_num_drafts_total{engine="1"}': 5,
        }, "num_drafts")


def test_common_prefix_delivery_includes_the_enclosing_event() -> None:
    result = {"arrivals": [{"tokens": 1, "seconds": 0.2}, {"tokens": 4, "seconds": 0.5},
                           {"tokens": 2, "seconds": 0.7}]}
    assert prefix_delivery(result, 5) == (2, 0.5)
    assert prefix_delivery(result, 6) == (3, 0.7)
    with pytest.raises(ValueError, match="never delivered"):
        prefix_delivery(result, 8)
    with pytest.raises(ValueError, match="at least one"):
        prefix_delivery(result, 0)
