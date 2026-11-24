"""Requested client timing must not be accepted as proof of actual split admission."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))

from analyze_latency_probe import prompt_hash, split_membership


def _fixture() -> tuple[list[list[int]], list[dict[str, Any]]]:
    prompts = [[i] * 256 for i in range(8)]

    def row(index: int, scheduled: int) -> dict[str, Any]:
        return {
            "prompt_sha256": prompt_hash(prompts[index]),
            "prompt_tokens": 256,
            "scheduled_tokens": scheduled,
        }

    events = [
        {
            "sequence": 0,
            "prepared": {
                "row_order": [row(i, 256) for i in range(6)],
                "has_prefill": True,
                "num_tokens": 1536,
            },
        },
        {
            "sequence": 1,
            "prepared": {
                "row_order": [row(i, 1) for i in range(6)] + [row(i, 256) for i in (6, 7)],
                "has_prefill": True,
                "num_tokens": 518,
            },
        },
    ]
    return prompts, events


def test_six_plus_two_requires_the_actual_prompt_membership_and_chunk_sizes() -> None:
    prompts, events = _fixture()
    assert split_membership(events, prompts)["confirmed"]
    wrong = copy.deepcopy(events)
    wrong[0]["prepared"]["row_order"][0]["prompt_sha256"] = prompt_hash(prompts[7])
    assert not split_membership(wrong, prompts)["confirmed"]
    wrong = copy.deepcopy(events)
    wrong[1]["prepared"]["row_order"][-1]["scheduled_tokens"] = 255
    assert not split_membership(wrong, prompts)["confirmed"]


def test_missing_late_request_or_ambiguous_identity_is_not_qualified() -> None:
    prompts, events = _fixture()
    assert not split_membership(events[:1], prompts)["confirmed"]
    prompts[7] = prompts[0]
    with pytest.raises(ValueError, match="distinct"):
        split_membership(events, prompts)
