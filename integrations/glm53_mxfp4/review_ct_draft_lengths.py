"""Independently review retained CT-only depth timing, counters, outputs and restoration."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from .measure_ct_draft_lengths import summarize_counters
from .mtp_metrics import delta


def digest(path: Path) -> str:
    """Hash a retained payload without loading it entirely into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def review(root: Path, output: Path) -> dict[str, Any]:
    """Verify evidence before reporting descriptive speed and acceptance comparisons."""
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((root / "evidence-manifest.json").read_bytes())
    listed = {row["path"] for row in manifest["files"]}
    assert len(listed) == len(manifest["files"]), "Duplicate manifest paths"
    for row in manifest["files"]:
        path = root / row["path"]
        assert path.resolve().is_relative_to(root.resolve())
        assert path.stat().st_size == row["bytes"] and digest(path) == row["sha256"]
    def retained(name: str) -> Any:
        assert name in listed, f"Unbound evidence: {name}"
        return json.loads((root / name).read_bytes())

    state = retained("controller.json")
    assert state["phase"] == "complete_restored" and not state["service_paused"]
    restored = state["restoration"]
    assert restored["health_200"] and restored["originals_running"]
    assert any(m["id"] == "glm-5.3-flash" and m["max_model_len"] == 500000
               for m in restored["models"]["data"])
    plan = retained("plan.json")
    assert not plan["sealed_test_opened"]
    native = plan["revision"] == "drafter-modal-native-v1"
    if native:
        assert plan["depths"] == {"original_a": 5, "adapted": 5, "original_b": 5}
        assert plan["trials"] == 6 and plan["requests"] == 96
        assert plan["selected_step"] == 16
        first, last, requests = "original_a", "original_b", 96
    else:
        assert plan["revision"] == "ct-draft-length-v1"
        assert plan["depths"] == {"s7a": 7, "s3": 3, "s5": 5, "s0": 0, "s7b": 7}
        first, last, requests = "s7a", "s7b", 48
    prepared = retained("prepared.json")
    assert digest(root / "plan.json") == prepared["plan_sha256"]
    cases = {c["id"]: c for c in plan["calibration_cases"]}
    assert len(cases) == (16 if native else 8)
    arm_results, first_outputs, first_counters = {}, {}, {}
    for arm, depth in plan["depths"].items():
        measurement = retained(f"{arm}-measurement.json")
        assert measurement["status"] == "complete" and measurement["depth"] == depth
        assert len(measurement["cases"]) == requests
        if native:
            assert measurement["plan_sha256"] == digest(root / "plan.json")
            assert measurement["source_sha256"] == digest(root / "measure_features.py")
        repeats, per_case = [], {}
        for case_id, case in cases.items():
            rows = [r for r in measurement["cases"] if r["case"] == case_id]
            assert [r["trial"] for r in rows] == list(range(6))
            first_outputs[arm, case_id] = rows[0]["result"]["token_ids"]
            first_counters[arm, case_id] = rows[0]["counter_delta"]
            for row in rows:
                assert row["domain"] == case["domain"]
                result = row["result"]
                assert len(result["token_ids"]) == result["usage"]["completion_tokens"] == 256
                assert result["usage"]["prompt_tokens"] == len(case["prompt_ids"])
                assert result["latency_seconds"] > result["ttft_seconds"] > 0
                counters = delta(row["metrics_before"], row["metrics_after"])
                assert counters == row["counter_delta"]
                assert summarize_counters(counters, depth) == row["speculation"]
            repeats.append({"case": case_id,
                "tokens_exact": all(r["result"]["token_ids"] == rows[0]["result"]["token_ids"]
                                    for r in rows),
                "counters_exact": all(r["counter_delta"] == rows[0]["counter_delta"] for r in rows)})
            per_case[case_id] = {
                "domain": case["domain"],
                "median_tps": statistics.median(256 / r["result"]["latency_seconds"] for r in rows),
                "median_ttft": statistics.median(r["result"]["ttft_seconds"] for r in rows),
                "trial_tps": [256 / r["result"]["latency_seconds"] for r in rows],
                "median_draft_rounds": statistics.median(
                    r["speculation"]["draft_rounds"] for r in rows),
                "median_accepted_per_round": statistics.median(
                    r["speculation"]["accepted_per_round"] for r in rows) if depth else None,
            }
        startup = retained(f"{arm}-startup.json")
        assert {r["rank"] for r in startup} == {0, 1}
        for entry in startup:
            assert entry["status"] == "complete"
            assert entry["context_probe"]["depth"] == depth
            assert entry["context_probe"]["drafter_loaded"] == bool(depth)
            if native:
                probe = entry["context_probe"]
                assert probe["checkpoint_files_verified"] and probe["fc_native_bitwise_equal"]
                assert probe["variant"] == plan["variants"][arm]
                assert probe["checkpoint_files"] == plan["drafter_files"][probe["variant"]]
                assert probe["quant_config"] == "Fp8Config" and probe["draft_layers"] == 6
                assert probe["context_width"] == 24576 and probe["causal"]
                assert probe["aux_layers"] == [24, 28, 32, 36, 40, 44]
            target = entry["trial"]["native_mxfp8_kda"]
            assert target["all_68_rounding_hashes_match_quality_run"]
            assert len(target["layers"]) == 68 and target["original_BF16_parameters_alive"] == 0
        domains = {}
        for domain in ("code", "chat", "explanation"):
            rows = [r for r in measurement["cases"] if r["domain"] == domain]
            counters = {k: sum(r["counter_delta"].get(k, 0) for r in rows)
                        for r in rows for k in r["counter_delta"]}
            domains[domain] = {
                **summarize_counters(counters, depth),
                "median_tps": statistics.median(256 / r["result"]["latency_seconds"] for r in rows),
                "median_ttft": statistics.median(r["result"]["ttft_seconds"] for r in rows),
                "median_tpot": statistics.median(r["result"]["tpot_seconds"] for r in rows),
                "geomean_case_median_tps": statistics.geometric_mean(
                    c["median_tps"] for c in per_case.values() if c["domain"] == domain),
                "median_draft_rounds": statistics.median(
                    r["speculation"]["draft_rounds"] for r in rows),
                "median_decode_seconds_per_draft_round": statistics.median(
                    (r["result"]["latency_seconds"] - r["result"]["ttft_seconds"])
                    / r["speculation"]["draft_rounds"] for r in rows) if depth else None,
            }
        arm_results[arm] = {"depth": depth, "requests": requests, "domains": domains,
                           "cases": per_case, "repeats": repeats,
                           "all_repeats_exact": all(r["tokens_exact"] and r["counters_exact"]
                                                    for r in repeats),
                           "startup_target_graph_rows": [
                               r["trial"]["physical_graph_token_counts"] for r in startup]}
    controls, comparisons = {}, {}
    for domain in ("code", "chat", "explanation"):
        a, b = [arm_results[arm]["domains"][domain]["median_tps"] for arm in (first, last)]
        drift = b / a - 1
        controls[domain] = {"relative_tps_drift": drift,
                           "within_2_percent": abs(drift) <= plan["speed_control_relative_tolerance"]}
        baseline = (a + b) / 2
        if native:
            comparisons[domain] = {}
            for arm in plan["arms"]:
                ratios = [arm_results[arm]["cases"][cid]["median_tps"] / statistics.mean(
                    arm_results[a]["cases"][cid]["median_tps"] for a in (first, last))
                    for cid, case in cases.items() if case["domain"] == domain]
                comparisons[domain][arm] = {
                    "tps_vs_bracketed_original":
                        arm_results[arm]["domains"][domain]["median_tps"] / baseline - 1,
                    "paired_geomean_tps_gain": statistics.geometric_mean(ratios) - 1,
                }
        else:
            off = arm_results["s0"]["domains"][domain]["median_tps"]
            comparisons[domain] = {
                arm: {"tps_vs_bracketed_seven": arm_results[arm]["domains"][domain]["median_tps"]
                      / baseline - 1,
                      "tps_vs_no_speculation": arm_results[arm]["domains"][domain]["median_tps"]
                      / off - 1}
                for arm in plan["arms"]
            }
    output_comparisons = []
    for arm in plan["arms"][1:]:
        for case_id in cases:
            a, b = first_outputs[first, case_id], first_outputs[arm, case_id]
            output_comparisons.append({"arm": arm, "case": case_id, "tokens_exact": a == b,
                "counters_exact": first_counters[first, case_id] == first_counters[arm, case_id],
                "first_different_output_index": next((i for i, (x, y) in enumerate(zip(a, b))
                                                      if x != y), None)})
    result = {
        "status": "reviewed", "plan_sha256": digest(root / "plan.json"),
        "source_sha256": digest(Path(__file__)), "verified_payloads": len(manifest["files"]),
        "arms": arm_results, "reload_controls": controls, "comparisons": comparisons,
        "output_comparisons": output_comparisons, "production_restored": True,
        "sealed_test_opened": False, "automatic_promotion": False,
        "limits": [f"{len(cases)} reused development prompts; not task accuracy or a sealed test.",
                   "Different output trajectories can affect acceptance and elapsed time.",
                   "Within-load repeats and reload drift are reported, never silently discarded.",
                   "Descriptive medians do not prove population speed or output equivalence.",
                   "Elapsed time per draft round includes draft, verify and host work; not GPU time.",
                   "This screen retains the existing CT tuning; production is not a measurement arm."],
    }
    if native:
        per_case_controls = {
            cid: arm_results[last]["cases"][cid]["median_tps"]
            / arm_results[first]["cases"][cid]["median_tps"] - 1 for cid in cases
        }
        repeats_ok = all(a["all_repeats_exact"] for a in arm_results.values())
        reload_exact = all(r["tokens_exact"] and r["counters_exact"]
                           for r in output_comparisons if r["arm"] == last)
        timing_ok = (all(c["within_2_percent"] for c in controls.values())
                     and all(abs(x) <= plan["speed_control_relative_tolerance"]
                             for x in per_case_controls.values()))
        thresholds = plan["development_thresholds"]
        gains = {d: comparisons[d]["adapted"]["paired_geomean_tps_gain"] for d in comparisons}
        result["development_gate"] = {
            "within_arm_exact": repeats_ok, "original_reload_exact": reload_exact,
            "timing_controls_pass": timing_ok, "per_case_reload_tps_drift": per_case_controls,
            "code_gain_pass": gains["code"] >= thresholds["code_geomean_tps_gain"],
            "chat_explanation_pass": all(
                gains[d] >= -thresholds["maximum_chat_explanation_tps_regression"]
                for d in ("chat", "explanation")),
            "generalization_proven": False,
        }
        result["development_gate"]["passed"] = all(result["development_gate"][k] for k in (
            "within_arm_exact", "original_reload_exact", "timing_controls_pass", "code_gain_pass",
            "chat_explanation_pass"))
        result["limits"].append(
            "These validation prompts selected the checkpoint; improvement cannot prove "
            "generalization. Sealed test remains closed.")
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    """Review a retained experiment without contacting serving or restarting a model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = review(args.root, args.output)
    print(json.dumps({"status": result["status"], "payloads": result["verified_payloads"],
                      "reload_controls": result["reload_controls"]}, indent=2))


if __name__ == "__main__":
    main()
