"""User cancellation may unblock the alternative test only after verified restoration."""
from __future__ import annotations

import copy
from typing import Any

import pytest

from integrations.glm53_mxfp4.launch_after_capture_stop import verify_readiness


def fixture() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Build a restored explicit stop with matching independently retained chunks."""
    state = {"pid": 123, "phase": "failed_restored", "service_paused": False,
             "error": "RuntimeError('Diagnostic interrupted by signal 15')",
             "restoration": {"health_200": True, "originals_running": True,
                 "models": {"data": [{"id": "glm-5.3-flash", "max_model_len": 500000}]}}}
    request = {"status": "stop_requested_by_user", "capture_plan_sha256": "plan",
               "capture_controller_pid": 123, "data_deletion_requested": False,
               "training_started": False}
    row = {"status": "partial_chunks_independently_verified", "plan_sha256": "plan",
           "different_elements": 0, "training_dataset_complete": False,
           "training_started": False, "sealed_test_opened": False, "verified_prompts": 64,
           "paired_receipts": {"chunk": "hash"}, "accepted_bytes": 1000, "feature_rows": 200,
           "numpy_bf16_elements_compared": 4096000, "verified_rows": [{"sha256": "payload"}]}
    preserved = [dict(copy.deepcopy(row), rank=rank) for rank in (0, 1)]
    return state, request, preserved


def test_explicit_stop_accepts_preserved_partial_data() -> None:
    state, request, preserved = fixture()
    verify_readiness(state, request, "plan", preserved)


@pytest.mark.parametrize("problem", ["arbitrary_failure", "no_user_stop", "still_paused",
                                    "wrong_context", "peer_missing", "peer_changed", "full_label"])
def test_unverified_stop_cannot_launch_another_outage(problem: str) -> None:
    state, request, preserved = fixture()
    if problem == "arbitrary_failure":
        state["error"] = "Out of memory"
    elif problem == "no_user_stop":
        request.pop("status")
    elif problem == "still_paused":
        state["service_paused"] = True
    elif problem == "wrong_context":
        state["restoration"]["models"]["data"][0]["max_model_len"] = 8192
    elif problem == "peer_missing":
        preserved.pop()
    elif problem == "peer_changed":
        preserved[1]["verified_rows"][0]["sha256"] = "different"
    else:
        preserved[0]["training_dataset_complete"] = True
    with pytest.raises(ValueError):
        verify_readiness(state, request, "plan", preserved)
