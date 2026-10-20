"""Check that equal aggregate task scores cannot hide repeat pass losses."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))
import analyze_task_repeats
from analyze_task_repeats import summarize_rounds


def test_same_total_still_records_both_directions_of_task_changes() -> None:
    base = [{"name": f"case{i}", "kind": "math", "passed": i != 0,
             "inference_completed": True, "answer_sha256": "original",
             "semantic_passed": i != 0} for i in range(34)]
    rounds = [copy.deepcopy(base) for _ in range(5)]
    for index, passed in ((0, True), (1, False)):
        rounds[1][index].update(passed=passed, answer_sha256="changed")
    result = summarize_rounds(rounds)
    assert result["strict_passes_by_round"] == [33] * 5
    pair = result["within_arm_pairs"][0]
    assert pair["a_pass_b_fail"] == ["case1"]
    assert pair["b_pass_a_fail"] == ["case0"]
    assert pair["different_answer_hashes"] == ["case0", "case1"]
    assert len(result["within_arm_pairs"]) == 10


def test_missing_round_or_changed_case_set_cannot_be_reported_complete() -> None:
    with pytest.raises(ValueError, match="five complete"):
        summarize_rounds([])
    base = [{"name": f"case{i}"} for i in range(34)]
    rounds = [copy.deepcopy(base) for _ in range(5)]
    rounds[4][0]["name"] = "different"
    with pytest.raises(ValueError, match="case sets differ"):
        summarize_rounds(rounds)


def test_legacy_scorer_completion_uses_raw_reply_not_task_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder = tmp_path / "control-r0"
    folder.mkdir()
    (folder / "real-work.json").write_text(json.dumps({"cases": [
        {"name": "bad_format", "response": {}, "passed": False},
        {"name": "api_error", "passed": False},
    ]}))
    monkeypatch.setattr(analyze_task_repeats, "task_scores", lambda *_: [
        {"name": "bad_format", "passed": False}, {"name": "api_error", "passed": False},
    ])
    rows = analyze_task_repeats.completed_scores(tmp_path, "control-r0", {})
    assert [row["inference_completed"] for row in rows] == [True, False]
