"""Independently validate pool-order replay, masking, timings and full-model target repeats."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .audit_drafter_feature_export_numpy import arrays
from .drafter_capture_pairing import file_digest
from .review_drafter_long_repeat import review as review_target


def review(root: Path, output: Path) -> dict[str, Any]:
    """Check raw integer payloads independently and retain failed full-model or timing gates."""
    if output.exists():
        raise FileExistsError(output)
    target = review_target(root, output.with_name(output.stem + "-target.json"),
                           expected_revision="drafter-canonical-prefill-v1")
    components = []
    for rank in (0, 1):
        path = root / f"ct-r{rank}" / f"sparse-component-r{rank}.json"
        component = json.loads(path.read_bytes())
        fixture_path = root / f"sparse-fixture-r{rank}.safetensors"
        payload = path.with_suffix(".safetensors")
        if (component["status"] != "passed" or component["repeat_limit"] != 0.02
                or file_digest(payload) != component["payload_sha256"]
                or file_digest(fixture_path) != component["fixture_sha256"]):
            raise ValueError("Component identity, payload digest or timing gate changed")
        startup = json.loads((root / f"ct-r{rank}" / f"trial-startup-r{rank}.json").read_bytes())
        if (startup["canonical_sparse_component"] != component
                or startup["canonical_sparse_prefill"]["status"] != "installed"
                or startup["canonical_sparse_prefill"]["max_rows"] != 4096):
            raise ValueError("Full-model startup did not install the component-qualified policy")
        original, values = arrays(fixture_path), arrays(payload)
        if [r["rows"] for r in component["fixtures"]] != [8, 16, 64, 128, 3973]:
            raise ValueError("Missing mixed-padding or full-size component fixture")
        for row in component["fixtures"]:
            prefix = f"rows{row['rows']}_"
            source = values[prefix + "source_rows"].reshape(-1)
            for key, value in original.items():
                a, b = value[source], values[prefix + key]
                if (a.shape != b.shape or a.dtype != b.dtype
                        or not np.array_equal(a.view(f"u{a.dtype.itemsize}"), b.view(f"u{b.dtype.itemsize}"))):
                    raise ValueError("GPU replay operands differ from the frozen retained fixture")
            pools = values[prefix + "canonical_pools"]
            expected = np.sort(original["native_pools"][source], axis=1)
            expanded = values[prefix + "expanded_tokens"]
            if (not np.array_equal(pools, expected)
                    or not np.array_equal(np.sort(expanded, axis=1),
                                          np.sort(original["native_tokens"][source], axis=1))
                    or np.any(expanded < -1)
                    or np.any(expanded >= original["seq_lens"][source, None])):
                raise ValueError("Canonical output, expansion, invalid masking or causality differs")
            if row["canonical_repeats_exact"] != 32 or row["graph_replays_exact"] != 32:
                raise ValueError("Component repeats or graph replays are incomplete")
            for trial in row["trials"]:
                a, _, b = trial["milliseconds"]
                passed = abs(b / a - 1) <= 0.02
                if passed != trial["repeat_control_passed"]:
                    raise ValueError("Timing repeat control was misclassified")
            row["independent_integer_payload_checks"] = "passed"
        components.append({"rank": rank, **component})
    target_pass = target["repeat_gate_passed"] and all(
        pair["all_prefill_features_exact"] and pair["different_decision_logit_rows"] == 0
        and pair["shared_response_prefix_tokens"] == 128 and pair["shared_decision_positions"] == 128
        and all(field["different_elements"] == 0 for field in pair["shared_history_features"])
        for pair in target["pairs"]
    )
    result = {"status": "reviewed", "production_restored": True,
              "plan_sha256": file_digest(root / "plan.json"), "source_sha256": file_digest(Path(__file__)),
              "component_checks_passed": True, "target_repeat_passed": target_pass,
              "target_review": str(output.with_name(output.stem + "-target.json")),
              "target_repeat_summary": [{k: p[k] for k in ("rank", "policies", "all_prefill_features_exact",
                                          "shared_response_prefix_tokens", "shared_decision_positions",
                                          "different_decision_logit_rows")} for p in target["pairs"]],
              "components": components, "training_eligible": False, "sealed_test_opened": False,
              "limits": ["Component timings do not establish end-to-end serving overhead.",
                         "A single long-prompt repeat pass does not qualify all capture prompts or C8/C16."]}
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    """Review restored evidence on CPU before the next experiment or training step."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = review(args.evidence, args.output)
    print(json.dumps({k: v for k, v in result.items() if k not in ("components", "target_repeat_summary")}, indent=2))


if __name__ == "__main__":
    main()
