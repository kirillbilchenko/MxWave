"""Independently verify corrected decode selection, graph cost controls and full target repeats."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .audit_drafter_feature_export_numpy import arrays
from .drafter_capture_pairing import file_digest
from .review_drafter_long_repeat import difference
from .review_drafter_long_repeat import review as review_target


def review(root: Path, output: Path) -> dict[str, Any]:
    """Verify archived bytes, masked index expansion and repeat gates without promotion."""
    if output.exists():
        raise FileExistsError(output)
    target = review_target(root, output.with_name(output.stem + "-target.json"),
                           expected_revision="drafter-canonical-decode-v1")
    components = []
    for rank in (0, 1):
        path = root / f"ct-r{rank}" / f"decode-component-r{rank}.json"
        component = json.loads(path.read_bytes())
        fixture = root / f"decode-fixture-r{rank}.safetensors"
        payload = path.with_suffix(".safetensors")
        if (component["status"] != "passed" or component["repeat_limit"] != 0.02
                or file_digest(payload) != component["payload_sha256"]
                or file_digest(fixture) != component["fixture_sha256"]):
            raise ValueError("Decode component source, payload or gate changed")
        startup = json.loads((root / f"ct-r{rank}" / f"trial-startup-r{rank}.json").read_bytes())
        policy = startup["canonical_sparse_decode"]
        if (startup["canonical_decode_component"] != component
                or policy["status"] != "installed" or (policy["rows"], policy["next_n"], policy["k"]) != (8, 1, 512)
                or policy["other_shapes"] != "native fallback"):
            raise ValueError("Full-model startup did not install the component-qualified decode policy")
        raw, values = arrays(fixture), arrays(payload)
        if [f["variant"] for f in component["fixtures"]] != ["retained", "mixed_padding"]:
            raise ValueError("Missing retained or padding-stress fixture")
        for row in component["fixtures"]:
            prefix = row["variant"] + "_"
            scores, lengths, seq = (values[prefix + k] for k in ("scores", "lengths", "seq_lens"))
            if not difference(scores, raw["scores"])["exact"]:
                raise ValueError("Replayed scores differ from the retained native fixture")
            if row["variant"] == "retained":
                if (not np.array_equal(lengths, raw["lengths"].reshape(-1))
                        or not np.array_equal(seq, raw["seq_lens"].reshape(-1))):
                    raise ValueError("Retained bounds or positions changed")
            elif (lengths.tolist() != [0, 1, 7, 32, 511, 512, 993, 995]
                  or not np.array_equal(seq, lengths * 4 + np.arange(8) % 4)):
                raise ValueError("Padding-stress geometry differs from the frozen test")
            native = values[prefix + "native_reference"]
            expected = np.sort(native, axis=1)
            if (not np.array_equal(values[prefix + "canonical_pools"], expected)
                    or not np.array_equal(values[prefix + "canonical_graph"], expected)
                    or not np.array_equal(np.sort(values[prefix + "native_graph"], axis=1), expected)
                    or (row["variant"] == "retained" and not np.array_equal(expected, np.sort(raw["native_pools"], axis=1)))):
                raise ValueError("Canonical output or graph replay changes selected membership/order")
            expanded = values[prefix + "expanded_tokens"]
            for index in range(8):
                valid = native[index][native[index] >= 0]
                if (len(valid) != min(512, int(lengths[index])) or len(np.unique(valid)) != len(valid)
                        or np.any(native[index] < -1) or np.any(valid >= lengths[index])):
                    raise ValueError("Selected pool count, uniqueness or causality changed")
                wanted = np.concatenate(((valid[:, None] * 4 + np.arange(4)).reshape(-1),
                                         np.arange(int(lengths[index]) * 4, int(seq[index]))))
                padded = np.full(expanded.shape[1], -1, dtype=expanded.dtype)
                padded[:len(wanted)] = wanted
                if not np.array_equal(np.sort(padded), np.sort(expanded[index])):
                    raise ValueError("Independent pool expansion or padding mask differs")
            if (row["native_membership_repeats"] != 32 or row["canonical_exact_repeats"] != 32
                    or row["native_graph_replays"] != 32 or row["canonical_graph_exact_replays"] != 32
                    or row["calls_per_graph"] != 100 or len(row["trials"]) != 6
                    or row["unused_columns_poison_tests"] != ["large_finite", "nan"]):
                raise ValueError("Required repeat, poison or timing controls are missing")
            for trial in row["trials"]:
                a, _, b = trial["milliseconds_per_call"]
                if (abs(b / a - 1) <= 0.02) != trial["repeat_control_passed"]:
                    raise ValueError("Timing repeat control was misclassified")
            row["independent_numpy_check"] = "passed"
        components.append({"rank": rank, **component})
    passed = target["repeat_gate_passed"] and all(
        p["all_prefill_features_exact"] and p["shared_response_prefix_tokens"] == 128
        and p["shared_decision_positions"] == 128 and p["different_decision_logit_rows"] == 0
        and all(f["different_elements"] == 0 for f in p["shared_history_features"])
        and all(f["exact"] for f in p["initial_draft"].values()) for p in target["pairs"]
    )
    result = {"status": "reviewed", "production_restored": True,
              "plan_sha256": file_digest(root / "plan.json"), "source_sha256": file_digest(Path(__file__)),
              "component_checks_passed": True, "target_repeat_passed": passed,
              "repeat_gate_passed": target["repeat_gate_passed"], "components": components,
              "target_pairs": target["pairs"], "training_eligible": False, "sealed_test_opened": False,
              "limits": ["One long-prompt C1 self-repeat gate; no cross-drafter, C8/C16 or long-context qualification.",
                         "Component graph timings do not establish end-to-end serving overhead.",
                         "Fresh capture smoke still precedes full feature capture and training."]}
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    """Review completed and restored decode-correction evidence on CPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = review(args.evidence, args.output)
    print(json.dumps({k: v for k, v in result.items() if k not in ("components", "target_pairs")}, indent=2))


if __name__ == "__main__":
    main()
