"""Review the fixed development comparison with the same expert policy in every arm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .drafter_fp8_roundtrip import digest
from .review_context_native import review as review_native


def review(root: Path, output: Path) -> dict[str, Any]:
    """Validate the common target hook and retain all original development gates."""
    assert not output.exists()
    plan = json.loads((root / "plan.json").read_bytes())
    parent = json.loads((root / "development-parent-plan.json").read_bytes())
    assert digest(root / "development-parent-plan.json") == plan["parent_development_plan_sha256"]
    assert digest(root / "verification-review.json") == plan["verification_review_sha256"]
    assert plan["revision"] == "expert-fixed-development-v1"
    for name in ("arms", "variants", "modes", "calibration_cases", "warmup_case",
                 "context_payloads", "drafter_files", "selected_step", "requests", "output_tokens"):
        assert plan[name] == parent[name], name
    for name, sha in plan["expert_reduction_source_sha256"].items():
        assert digest(root / name) == sha
    assert plan["expert_reduction_policy"] == {
        "mode": "bounded_padding_default_occupancy", "target_rows_max": 16,
        "prefill_above_limit": "native", "same_in_both_arms": True,
    }
    coverage, replays = [], []
    for arm in plan["arms"]:
        for rank in (0, 1):
            startup = json.loads((root / f"{arm}-r{rank}/trial-startup-r{rank}.json").read_bytes())
            policy = startup["expert_reduction"]
            assert policy["installed"] and policy["policy"] == "bounded_padding_default_occupancy"
            assert policy["target_rows_max"] == 16
            shapes = {tuple(x["geometry"][:3]) for x in policy["dispatches"]}
            assert {(8, 2048, 4096), (64, 4096, 1024)} <= shapes
            coverage.append({"arm": arm, "rank": rank, **policy})
            counts = []
            for line in (root / f"{arm}-r{rank}.log").read_text().splitlines():
                if '"native_draft_replay"' not in line:
                    continue
                row = json.loads(line[line.index("{"):])
                assert row["rank"] == rank
                if row["native_draft_replay"] == [1, 8]:
                    counts.append(row["count"])
            assert counts and counts[0] == 1 and counts[-1] >= 64
            replays.append({"arm": arm, "rank": rank, "descriptor": [1, 8],
                            "observed_replays_lower_bound": max(counts)})
    # The existing reviewer retains failed output/timing gates in its report.
    result = review_native(root, output.with_name(output.stem + "-native.json"))
    state = json.loads((root / "controller.json").read_bytes())
    prepared = json.loads((root / "prepared.json").read_bytes())
    assert state["restoration"]["live"] == prepared["live"]
    assert state["restoration"]["originals"] == prepared["originals"]
    result.update(
        source_sha256=digest(Path(__file__)), native_reviewer_sha256=digest(
            Path(__file__).with_name("review_context_native.py")),
        policy_capture_coverage=coverage, observed_native_draft_replays=replays,
        expert_reduction_policy_identical=True,
        development_output_identity_pass=result["all_cross_arm_tokens_identical"],
    )
    result["limits"] += [
        "Both arms use padding; this does not measure padding overhead against the native target policy.",
        "Python expert-call counts establish capture coverage; draft replay counts are observed lower bounds.",
        "Target policies beyond 16 rows, C8/C16, full training and sealed evaluation remain outstanding.",
    ]
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    """Review archived development evidence after original production is restored."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = review(args.root, args.output)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("cases", "policy_capture_coverage")}, indent=2))


if __name__ == "__main__":
    main()
