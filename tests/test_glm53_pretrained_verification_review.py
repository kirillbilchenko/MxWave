"""Review decisions at actual causal rows, not matching speculative step numbers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from integrations.glm53_mxfp4.review_pretrained_verification import decision


def frame(prefill: bool) -> dict[str, Any]:
    """Use different input and logit row indexing for prefill and decode."""
    return {
        "output_start": 0 if prefill else 2,
        "sampled": [12] if prefill else [20, 21],
        "x": {
            "target_logits_indices": np.array([1] if prefill else [0, 1]),
            "target_positions": np.array([0, 1] if prefill else [3, 4]),
            "target_input_ids": np.array([10, 11] if prefill else [13, 20]),
            "target_logits": np.array([[1., 2.]] if prefill else [[1., 2.], [3., 4.]]),
        },
        "meta": {"tensor_dtypes": {"target_logits": "torch.float32"}},
    }


def test_prefill_maps_sampled_row_to_last_prompt_input() -> None:
    result = decision([frame(True)], 0, [10, 11], [12, 13, 20, 21])
    assert result["offset"] == 0 and result["input_row"] == 1 and result["position"] == 1
    assert result["token"] == 12


def test_decode_uses_emitted_offset_not_frame_number() -> None:
    result = decision([frame(False)], 3, [10, 11], [12, 13, 20, 21])
    assert result["offset"] == 1 and result["input_row"] == 1 and result["position"] == 4
    assert result["logits"].tolist() == [3., 4.]


@pytest.mark.parametrize("wrong", ["position", "history", "sample"])
def test_rejects_unmatched_decision_evidence(wrong: str) -> None:
    value = frame(False)
    if wrong == "position":
        value["x"]["target_positions"][1] = 9
    elif wrong == "history":
        value["x"]["target_input_ids"][0] = 999
    else:
        value["sampled"][1] = 999
    with pytest.raises(ValueError):
        decision([value], 3, [10, 11], [12, 13, 20, 21])
