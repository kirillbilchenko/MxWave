"""Independently recompute restored routing sweeps without Torch or a GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

from verify_prefill_decode import checked_measurement

ARMS = ("qualified", "stock", "raw_dense", "fused_split1")


def digest(path: Path) -> str:
    """Bind retained source and evidence bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_blocks(row: dict[str, Any], *, queued: bool = False) -> dict[str, list[float]]:
    """Recompute before/after baselines from all original ABBA/BAAB event legs."""
    targets = ARMS[1:]
    iterations = 8 if queued else 4
    assert row["protocol"] == ("queued_bracketed_ABBA_BAAB" if queued else "bracketed_ABBA_BAAB")
    assert row["trials"] == 8 and row["iterations_per_leg"] == iterations
    assert len(row["samples"]) == 8
    if queued:
        assert row["untimed_primer_calls_per_leg"] == 1
        assert row["pressure_before_each_timed_event_pair"] == bool(row["pressure_bytes"])
    ratios: dict[str, list[float]] = {name: [] for name in targets}
    references: dict[str, list[float]] = {name: [] for name in targets}
    candidates: dict[str, list[float]] = {name: [] for name in targets}
    for index, sample in enumerate(row["samples"]):
        forward = index % 2 == 0
        assert sample["trial"] == index
        assert sample["direction"] == ("ABBA" if forward else "BAAB")
        assert sample["target_order"] == list(targets if forward else reversed(targets))
        assert set(sample["blocks"]) == set(targets)
        for target in targets:
            block = sample["blocks"][target]
            order = ["qualified", target, target, "qualified"] if forward else [
                target, "qualified", "qualified", target]
            assert block["order"] == order and len(block["legs"]) == 4
            reference, candidate = [], []
            for leg, arm in zip(block["legs"], order, strict=True):
                values = leg["times_us"]
                assert leg["arm"] == arm and len(values) == iterations
                if queued:
                    assert leg["untimed_primer_calls"] == 1
                    assert leg["synchronizations_after_enqueue"] == 1
                    assert leg["all_event_pairs_queued_before_synchronization"] is True
                assert all(math.isfinite(value) and value > 0 for value in values)
                (reference if arm == "qualified" else candidate).extend(values)
            first, second = statistics.median(reference), statistics.median(candidate)
            ratio = second / first
            assert [first, second, ratio] == [block["reference_median_us"],
                block["candidate_median_us"], block["ratio"]]
            references[target].append(first)
            candidates[target].append(second)
            ratios[target].append(ratio)
    decisions = {}
    for target in targets:
        combined = statistics.median(ratios[target])
        assert statistics.median(references[target]) == row["reference_median_us"][target]
        assert statistics.median(candidates[target]) == row["median_us"][target]
        assert combined == row["median_paired_ratio"][target]
        orders = [statistics.median(ratios[target][offset::2]) for offset in (0, 1)]
        assert orders == [row["order_ratios"][order][target] for order in ("ABBA", "BAAB")]
        decisions[target] = [combined, *orders]
    baseline = statistics.median(statistics.median(references[target][index] for target in targets)
                                 for index in range(8))
    assert baseline == row["median_us"]["qualified"]
    return decisions


def verify(report: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    """Reapply fixed numerical and both-order timing gates to all retained cases."""
    assert report["status"] == "complete"
    assert not report["full_model_loaded"] and not report["qualification_attempted"]
    assert not report["serving_adapter_changed"] and not report["serving_routing_qualified"]
    assert report["tokens"] == plan["tokens"] and report["arms"] == list(ARMS)
    blocked = plan.get("paired_blocks", False)
    queued = plan.get("queued_blocks", False)
    assert report.get("queued_blocks", False) == queued
    if queued:
        assert blocked and plan["iterations_per_leg"] == 8
        assert plan["untimed_primer_calls_per_leg"] == 1 and plan["tokens"] == [8, 16, 257]
    assert report.get("paired_blocks", False) == blocked
    if blocked:
        assert report["same_reference_graph_control"]
    assert [case["name"] for case in report["cases"]] == plan["cases"]
    assert report["peak_cuda_bytes"] <= plan["timing_cuda_allocator_cap_bytes"]
    leads, values = {}, []
    records = []
    for case in report["cases"]:
        assert case["weight_bits_exact"] and case["destination_guards_unchanged"]
        assert [row["tokens"] for row in case["measurements"]] == plan["tokens"]
        leads[case["name"]] = []
        for row in case["measurements"]:
            tokens = row["tokens"]
            padded = next((n for n in (8, 16, 32) if tokens <= n),
                          (tokens + 63) // 64 * 64)
            assert row["native_padded_tokens"] == padded
            error = row["arithmetic"]
            passed = bool(error["finite"] and error["relative_l2"] is not None
                          and 0 <= error["relative_l2"] < .01)
            assert error["passed"] == passed
            assert row["stock_outputs_bits_exact"] and row["fused_eager_graph_bits_exact"]
            assert set(row["timing"]) == {"eager", "graph"}
            controls, faster = [], []
            timings = {}
            for execution, groups in row["timing"].items():
                assert set(groups) == {"warm", "pressure"}
                for cache, measured in groups.items():
                    assert measured["pressure_bytes"] == (128 * 1024**2 if cache == "pressure" else 0)
                    ratios = checked_blocks(measured, queued=queued) if blocked else checked_measurement(
                        measured, 4, False, arms=ARMS)
                    control = all(.98 <= v <= 1.02 for v in ratios["stock"])
                    gain = all(v <= .95 for v in ratios["fused_split1"])
                    controls.append(control)
                    faster.append(gain)
                    timings[f"{execution}_{cache}"] = {
                        "reference_us": measured["reference_median_us"]["fused_split1"] if blocked
                            else measured["median_us"]["qualified"],
                        "raw_dense_us": measured["median_us"]["raw_dense"],
                        "fused_us": measured["median_us"]["fused_split1"],
                        "control_ratios": ratios["stock"], "fused_ratios": ratios["fused_split1"]}
            eligible = passed and all(controls) and all(faster)
            assert eligible == row["component_eligible"]
            if eligible:
                leads[case["name"]].append(tokens)
            values.append(passed)
            records.append({"case": case["name"], "tokens": tokens, "arithmetic_passed": passed,
                "relative_l2": error["relative_l2"], "controls_passed": all(controls),
                "fused_5_percent_faster_all_conditions": all(faster),
                "component_eligible": eligible, "timings": timings})
    shared = [n for n in plan["tokens"] if all(n in rows for rows in leads.values())]
    assert shared == report["component_eligible_tested_rows"]
    assert all(values) == report["correctness_passed"]
    return {"status": "verified", "all_arithmetic_passed": all(values),
        "per_matrix_eligible_tested_rows": leads, "shared_eligible_tested_rows": shared,
        "measurements": records,
        "event_samples": len(records) * 4 * 8 * 4 * (8 if queued else 4) * (3 if blocked else 1),
        "serving_routing_qualified": False}


def main() -> None:
    """Check immutable source, smoke, exact restoration and every original sample."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    root = args.root

    def read(name: str) -> dict[str, Any]:
        return json.loads((root / name).read_bytes())

    state, restoration = read("controller.json"), read("restoration.json")
    assert state["phase"] in ("complete", "failed_restored") and not state["service_paused"]
    assert restoration["status"] == "complete" and restoration["health"] == 200
    assert not state["full_model_loaded"] and not state.get("serving_adapter_changed", False)
    for row in restoration["containers"]:
        assert {k: v for k, v in row.items() if k != "rank"} == state["originals"][row["rank"]]
    prepared, plan = read("prepared.json"), read("component-plan.json")
    source = {str(p.relative_to(root / "src")): digest(p) for p in (root / "src").rglob("*")
              if p.is_file() and p.suffix in (".py", ".cuh")}
    assert source == prepared["source_sha256"] == plan["source_sha256"]
    assert digest(root / "src/run_prefill_decode.py") == state["controller_sha256"]
    assert digest(root / "component-plan.json") == prepared["plan_sha256"]
    smoke = read("smoke.json")
    assert smoke["status"] == "passed" and len(smoke["cases"]) == 3
    assert smoke["code_sha256"] == source["smoke_prefill_decode.py"]
    for case in smoke["cases"]:
        assert case["qualified_bits_exact"] and len(case["modes"]) == 3
        for mode in case["modes"]:
            assert mode["bits_exact"] and mode["nondefault_stream_exact"]
            assert mode["graph_replays_exact"] == 6 and mode["guards_unchanged"]
            assert mode["replay_allocation_growth"] == 0
    native = read("native-build.json")
    assert digest(root / "native-build.json") == prepared["native_manifest_sha256"]
    assert native["library_sha256"] == prepared["native_library_sha256"]
    assert source["prefill_decode.cuh"] == native["header_sha256"]
    routing = read("routing.json")
    assert routing["code_sha256"] == source["benchmark_prefill_routing.py"]
    assert routing["plan_sha256"] == prepared["plan_sha256"]
    assert routing["smoke_sha256"] == digest(root / "smoke.json")
    assert routing["library_sha256"] == plan["qualified_library_sha256"]
    result = verify(routing, plan)
    result.update(original_service_restored=True, source_files=len(source),
        verifier_sha256=digest(Path(__file__)), quality_qualified=False)
    args.report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "measurements"}))


if __name__ == "__main__":
    main()
