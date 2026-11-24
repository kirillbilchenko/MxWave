"""A validation continuation preserves the failed timing phase and its source-check gap."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))

from measure_output_validation import combined_candidate


def reports() -> tuple[dict, dict, dict]:
    """Make distinct original and fresh phase records to detect accidental phase replacement."""
    prior = {
        "status": "running",
        "label": "candidate",
        "plan_sha256": "frozen",
        "trials": [{"old_timing": i} for i in range(72)],
        "warmups": list(range(24)),
    }
    validation = {
        "status": "complete",
        "label": "candidate",
        "plan_sha256": "frozen",
        "trials": [],
        "warmups": ["fresh"],
        "quality": list(range(8)),
        "profiles": list(range(4)),
        "source_after": "fresh-source-check",
    }
    provenance = {"original_source_after_missing": True, "comparison_has_time_gap": True}
    return prior, validation, provenance


def test_derived_view_cannot_rewrite_original_timing_or_fresh_validation() -> None:
    prior, validation, provenance = reports()
    preserved = copy.deepcopy((prior, validation))
    result = combined_candidate(prior, validation, provenance)
    assert (prior, validation) == preserved
    assert result["trials"] == prior["trials"]
    assert result["quality"] == validation["quality"]
    assert result["derived_cross_model_load_view"]
    assert result["original_failed_run_preserved"]
    assert result["original_timing_source_after_missing"]
    result["trials"][0]["old_timing"] = -1
    result["quality"].append("mutation")
    assert (prior, validation) == preserved


@pytest.mark.parametrize(
    "change",
    [
        "new_timings",
        "missing_quality",
        "different_protocol",
        "hidden_source_gap",
        "hidden_time_gap",
        "old_after",
    ],
)
def test_continuation_rejects_incomplete_or_misleading_phase_combinations(change: str) -> None:
    prior, validation, provenance = reports()
    if change == "new_timings":
        validation["trials"] = ["new"]
    elif change == "missing_quality":
        validation["quality"].pop()
    elif change == "different_protocol":
        validation["plan_sha256"] = "changed"
    elif change == "hidden_source_gap":
        provenance["original_source_after_missing"] = False
    elif change == "hidden_time_gap":
        provenance["comparison_has_time_gap"] = False
    else:
        prior["source_after"] = "not-the-failed-run"
    with pytest.raises(ValueError):
        combined_candidate(prior, validation, provenance)
