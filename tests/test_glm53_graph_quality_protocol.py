"""Expanded qualification may add evidence without replacing old tasks or tokenizer pins."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))
from prepare_graph_quality import expand_suite


def fixture() -> tuple[dict[str, Any], list[int], list[dict[str, str]]]:
    """Construct a complete legacy sample with an expanded token pool and unique source ids."""
    ids = list(range(110000))
    gsm = [{"question": f"Question {i}", "answer": f"Reasoning\n#### {i}"} for i in range(200)]
    legacy = {"wiki_total_tokens": len(ids), "wiki_offsets": [0, 2000],
              "wiki_windows": [ids[:1025], ids[2000:3025]], "datasets": {"pinned": True},
              "strict_control_stability_gate": {"mean_kl": 1e-5, "absolute_mean_nll": 1e-4},
              "tasks": [{"name": f"practical{i}", "kind": "exact"} for i in range(10)]
              + [{"name": f"gsm8k_{i:04d}", "kind": "math", "source_index": i,
                  "expected": str(i)} for i in range(24)]}
    return legacy, ids, gsm


def test_expansion_preserves_all_old_cases_and_adds_disjoint_reproducible_questions() -> None:
    legacy, ids, gsm = fixture()
    original = copy.deepcopy(legacy)
    suite = expand_suite(legacy, ids, gsm)
    assert legacy == original
    assert suite["tasks"][:34] == original["tasks"]
    assert len(suite["tasks"]) == 110 and len(suite["wiki_windows"]) == 100
    assert suite["wiki_scored_tokens"] == 102400
    assert len(set(suite["added_gsm8k_source_indices"])) == 76
    assert not (set(suite["added_gsm8k_source_indices"]) & set(range(24)))
    assert expand_suite(legacy, ids, gsm) == suite
    assert suite["strict_control_stability_gate"] == original["strict_control_stability_gate"]


def test_tokenizer_disagreement_and_duplicate_legacy_question_fail_closed() -> None:
    legacy, ids, gsm = fixture()
    wrong = ids.copy()
    wrong[11] += 1
    with pytest.raises(ValueError, match="tokenizer differs"):
        expand_suite(legacy, wrong, gsm)
    legacy["tasks"][-1]["source_index"] = 0
    with pytest.raises(ValueError, match="Legacy source questions"):
        expand_suite(legacy, ids, gsm)
