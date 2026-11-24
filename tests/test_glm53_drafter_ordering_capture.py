"""Reject a capture whose sparse ordering policies differ from the qualified runtime."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from integrations.glm53_mxfp4.drafter_fp8_roundtrip import digest
from integrations.glm53_mxfp4.verify_drafter_feature_capture import verify_ordering


def fixture(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Construct a bound qualification receipt and both runtime installation records."""
    sources = {}
    for name in ("canonical_sparse_prefill.py", "canonical_sparse_decode.py"):
        path = root / name
        path.write_text(name)
        sources[name] = digest(path)
    proof = root / "ordering-qualification-review.json"
    proof.write_text(json.dumps({"plan_sha256": "qualified", "component_checks_passed": True,
                                 "target_repeat_passed": True, "production_restored": True}))
    plan = {"ordering_source_sha256": sources,
            "ordering_qualification_review_sha256": digest(proof),
            "ordering_qualification_plan_sha256": "qualified"}
    startup = [{"rank": rank,
                "canonical_sparse_prefill": {"status": "installed", "max_rows": 4096,
                    "k": 512, "scores_or_selected_membership_changed": False},
                "canonical_sparse_decode": {"status": "installed", "rows": 8, "next_n": 1,
                    "k": 512, "scores_or_membership_changed": False,
                    "other_shapes": "native fallback"},
                "canonical_decode_component": {"status": "inherited_verified",
                    "parent_review_sha256": digest(proof)}} for rank in (0, 1)]
    return plan, startup


def test_both_ranks_must_install_the_qualified_policy(tmp_path: Path) -> None:
    plan, startup = fixture(tmp_path)
    verify_ordering(tmp_path, plan, startup)
    verify_ordering(tmp_path, {}, [])  # Retained historical captures remain reviewable.
    for field, value in (("rows", 16), ("next_n", 8), ("status", "disabled")):
        changed = copy.deepcopy(startup)
        changed[1]["canonical_sparse_decode"][field] = value
        with pytest.raises(ValueError, match="numerical ordering"):
            verify_ordering(tmp_path, plan, changed)
    with pytest.raises(ValueError, match="one rank"):
        verify_ordering(tmp_path, plan, startup[:1])


def test_changed_policy_or_proof_cannot_qualify_capture(tmp_path: Path) -> None:
    plan, startup = fixture(tmp_path)
    proof = tmp_path / "ordering-qualification-review.json"
    proof.write_text(proof.read_text() + " ")
    with pytest.raises(ValueError, match="qualification"):
        verify_ordering(tmp_path, plan, startup)
    path = tmp_path / "canonical_sparse_decode.py"
    path.write_text("different policy")
    with pytest.raises(ValueError, match="source changed"):
        verify_ordering(tmp_path, plan, startup)
