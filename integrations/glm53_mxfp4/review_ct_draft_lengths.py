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
    assert plan["revision"] == "ct-draft-length-v1" and not plan["sealed_test_opened"]
    assert plan["depths"] == {"s7a": 7, "s3": 3, "s5": 5, "s0": 0, "s7b": 7}
    prepared = retained("prepared.json")
    assert digest(root / "plan.json") == prepared["plan_sha256"]
    cases = {c["id"]: c for c in plan["calibration_cases"]}
    arm_results, first_outputs = {}, {}
    for arm, depth in plan["depths"].items():
        measurement = retained(f"{arm}-measurement.json")
        assert measurement["status"] == "complete" and measurement["depth"] == depth
        assert len(measurement["cases"]) == 48
        repeats, per_case = [], {}
        for case_id, case in cases.items():
            rows = [r for r in measurement["cases"] if r["case"] == case_id]
            assert [r["trial"] for r in rows] == list(range(6))
            first_outputs[arm, case_id] = rows[0]["result"]["token_ids"]
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
            }
        startup = retained(f"{arm}-startup.json")
        assert {r["rank"] for r in startup} == {0, 1}
        for entry in startup:
            assert entry["status"] == "complete"
            assert entry["context_probe"]["depth"] == depth
            assert entry["context_probe"]["drafter_loaded"] == bool(depth)
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
        arm_results[arm] = {"depth": depth, "requests": 48, "domains": domains,
                           "cases": per_case, "repeats": repeats,
                           "all_repeats_exact": all(r["tokens_exact"] and r["counters_exact"]
                                                    for r in repeats),
                           "startup_target_graph_rows": [
                               r["trial"]["physical_graph_token_counts"] for r in startup]}
    controls, comparisons = {}, {}
    for domain in ("code", "chat", "explanation"):
        a, b = [arm_results[arm]["domains"][domain]["median_tps"] for arm in ("s7a", "s7b")]
        drift = b / a - 1
        controls[domain] = {"relative_tps_drift": drift,
                           "within_2_percent": abs(drift) <= plan["speed_control_relative_tolerance"]}
        baseline = (a + b) / 2
        off = arm_results["s0"]["domains"][domain]["median_tps"]
        comparisons[domain] = {
            arm: {"tps_vs_bracketed_seven": arm_results[arm]["domains"][domain]["median_tps"]
                  / baseline - 1,
                  "tps_vs_no_speculation": arm_results[arm]["domains"][domain]["median_tps"]
                  / off - 1}
            for arm in plan["arms"]
        }
    output_comparisons = []
    for arm in ("s3", "s5", "s0", "s7b"):
        for case_id in cases:
            a, b = first_outputs["s7a", case_id], first_outputs[arm, case_id]
            output_comparisons.append({"arm": arm, "case": case_id, "tokens_exact": a == b,
                "first_different_output_index": next((i for i, (x, y) in enumerate(zip(a, b))
                                                      if x != y), None)})
    result = {
        "status": "reviewed", "plan_sha256": digest(root / "plan.json"),
        "source_sha256": digest(Path(__file__)), "verified_payloads": len(manifest["files"]),
        "arms": arm_results, "reload_controls": controls, "comparisons": comparisons,
        "output_comparisons": output_comparisons, "production_restored": True,
        "sealed_test_opened": False, "automatic_promotion": False,
        "limits": ["Eight reused development prompts; not task accuracy or a sealed quality test.",
                   "Different output trajectories can affect acceptance and elapsed time.",
                   "Within-load repeats and reload drift are reported, never silently discarded.",
                   "Descriptive medians do not prove population speed or output equivalence.",
                   "Elapsed time per draft round includes draft, verify and host work; not GPU time.",
                   "This screen retains the existing CT tuning; production is not a measurement arm."],
    }
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
