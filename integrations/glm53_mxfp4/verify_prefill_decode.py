"""Recompute a restored prefill component's decisions from its retained event samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

ARMS = ("qualified", "stock", "direct_pair", "row_coalesced")


def digest(path: Path) -> str:
    """Bind one retained file without importing GPU libraries."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_measurement(row: dict[str, Any], iterations: int,
                        confirmation: bool, arms: tuple[str, ...] = ARMS
                        ) -> dict[str, list[float]]:
    """Recompute paired medians and both order medians from every CUDA event sample."""
    assert row["trials"] == 8 and row["iterations"] == iterations
    samples = row["samples"]
    assert len(samples) == 8
    ratios: dict[str, list[float]] = {name: [] for name in arms}
    for index, sample in enumerate(samples):
        forward = (index + int(confirmation)) % 2 == 0
        assert sample["trial"] == index
        assert sample["order"] == list(arms if forward else reversed(arms))
        assert sample["direction"] == ("forward" if forward else "reverse")
        times = sample["times_us"]
        assert set(times) == set(arms)
        assert all(len(values) == iterations and all(math.isfinite(v) and v > 0 for v in values)
                   for values in times.values())
        reference = statistics.median(times["qualified"])
        for arm in arms:
            value = statistics.median(times[arm]) / reference
            assert value == sample["ratios"][arm]
            ratios[arm].append(value)
    decisions = {}
    for arm in arms:
        median = statistics.median(statistics.median(s["times_us"][arm]) for s in samples)
        combined = statistics.median(ratios[arm])
        orders = [statistics.median(value for value, sample in zip(ratios[arm], samples,
                   strict=True) if sample["direction"] == order)
                  for order in ("forward", "reverse")]
        assert median == row["median_us"][arm]
        assert combined == row["median_paired_ratio"][arm]
        assert orders == [row["order_ratios"][order][arm] for order in ("forward", "reverse")]
        decisions[arm] = [combined, *orders]
    return decisions


def verify_timing(report: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    """Reapply the original controls and all-case candidate gates without dropping samples."""
    assert report["status"] == "complete" and report["correctness_passed"]
    assert not report["full_model_loaded"] and not report["qualification_attempted"]
    assert report["tokens"] == plan["tokens"]
    assert [case["name"] for case in report["cases"]] == plan["cases"]
    pure, coupled = [], []
    for case in report["cases"]:
        assert case["weight_bits_exact"]
        assert set(case["decode"]) == {"warm", "pressure"}
        assert [row["tokens"] for row in case["coupled"]] == plan["tokens"]
        for cache, measured in case["decode"].items():
            assert measured["pressure_bytes"] == (128 * 1024**2 if cache == "pressure" else 0)
            pure.append(checked_measurement(measured, 8, report["confirmation"]))
        for row in case["coupled"]:
            assert row["outputs_bits_exact"] and set(row["timing"]) == {"warm", "pressure"}
            for cache, measured in row["timing"].items():
                assert measured["pressure_bytes"] == (128 * 1024**2 if cache == "pressure" else 0)
                coupled.append(checked_measurement(measured, 4, report["confirmation"]))
    parity = [all(.98 <= v <= 1.02 for row in rows for v in row["stock"])
              for rows in (pure, coupled)]
    assert parity == [report["pure_control_parity_passed"], report["coupled_control_parity_passed"]]
    controls = all(parity)
    assert controls == report["control_parity_passed"]
    candidates = {}
    for arm in ("direct_pair", "row_coalesced"):
        faster = all(v <= .95 for row in pure for v in row[arm])
        nonregression = all(v <= 1.02 for row in coupled for v in row[arm])
        candidates[arm] = {"decode_at_least_5_percent_faster": faster,
                          "all_coupled_cases_within_2_percent": nonregression,
                          "passed": controls and faster and nonregression}
    assert candidates == report["candidates"]
    assert any(row["passed"] for row in candidates.values()) == report["performance_passed"]
    return {"controls_passed": controls, "candidates": candidates,
            "pure_measurements": len(pure), "coupled_measurements": len(coupled),
            "event_samples": len(pure) * 8 * 8 * 4 + len(coupled) * 8 * 4 * 4}


def main() -> None:
    """Verify immutable code, exact original-service restoration and every completed stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    root = args.root

    def read(name: str) -> dict[str, Any]:
        return json.loads((root / name).read_bytes())

    state, restored = read("controller.json"), read("restoration.json")
    assert state["phase"] in ("complete", "failed_restored") and not state["service_paused"]
    assert restored["status"] == "complete" and restored["health"] == 200
    assert not state["full_model_loaded"]
    for row in restored["containers"]:
        assert {k: v for k, v in row.items() if k != "rank"} == state["originals"][row["rank"]]
    prepared, plan = read("prepared.json"), read("component-plan.json")
    code = {str(p.relative_to(root / "src")): digest(p) for p in (root / "src").rglob("*")
            if p.is_file() and p.suffix in (".py", ".cuh")}
    assert code == prepared["source_sha256"]
    assert digest(root / "component-plan.json") == prepared["plan_sha256"]
    assert digest(root / "src/run_prefill_decode.py") == state["controller_sha256"]
    assert digest(root / "native-build.json") == prepared["native_manifest_sha256"]
    assert read("native-build.json")["library_sha256"] == prepared["native_library_sha256"]
    smoke = read("smoke.json")
    wrapped_only = state["scope"] == "Wrapped-base correctness only"
    smoke_file = "smoke_prefill_wrap.py" if wrapped_only else "smoke_prefill_decode.py"
    assert smoke["code_sha256"] == code[smoke_file]
    reviews = {}
    if smoke["status"] == "passed" and wrapped_only:
        assert len(smoke["cases"]) == 5 and len(smoke["packing_preflight"]) == 5
        assert smoke["cpu_preflight_without_cuda"] and smoke["wrapped_bases_passed"] == 2
        assert smoke["full_bf16_word_space_tested"]
        assert smoke["peak_cuda_bytes"] <= 1024**3
        assert smoke["native_manifest"] == read("native-build.json")
        assert not (root / "timing.json").exists() and not (root / "confirmation.json").exists()
        for index, case in enumerate(smoke["cases"]):
            wrapped = index >= 3
            assert case["cpu_masked_exact"]
            assert case["qualified_skipped_for_wrapped_base"] == wrapped
            if wrapped:
                assert case["base"] == 255 and case["unmasked_mismatches"] > 0
                assert case["control_guards"] == {
                    "python_rejected": True, "native_error": 1, "output_unchanged": True}
            else:
                assert case["base"] <= 248 and case["qualified_bits_exact"]
                assert case["unmasked_mismatches"] == 0
            assert [row["mode"] for row in case["modes"]] == list(ARMS[2:] if wrapped else ARMS[1:])
            for row in case["modes"]:
                assert row["bits_exact"] and row["eager_exact"] and row["nondefault_stream_exact"]
                assert row["graph_replays_exact"] == 6 and row["guards_unchanged"]
                assert row["replay_allocation_growth"] == 0
        reviews["wrapped_correctness"] = {"safe_cases": 3, "wrapped_cases": 2,
            "all_bf16_words_covered": True, "unchanged_control_refuses_unsafe_base": True}
    elif smoke["status"] == "passed":
        assert len(smoke["cases"]) == 3
        oracle = read("pack-oracle.json")
        assert not oracle["cuda_initialized"] and oracle["supported_fixtures_exact"]
        assert oracle["all_wrapped_source_words_exact"]
        assert oracle["code_sha256"] == code["prefill_pack_oracle.py"]
        assert len(smoke["packing_preflight"]) == 3
        for row in smoke["packing_preflight"]:
            assert row["bits_exact"] and 0 <= row["metadata"][2] <= 248
        for case in smoke["cases"]:
            assert case["qualified_bits_exact"]
            assert [row["mode"] for row in case["modes"]] == list(ARMS[1:])
            for row in case["modes"]:
                assert row["bits_exact"] and row["nondefault_stream_exact"]
                assert row["graph_replays_exact"] == 6 and row["guards_unchanged"]
                assert row["replay_allocation_growth"] == 0
        for name in ("timing", "confirmation"):
            path = root / (name + ".json")
            if not path.exists():
                continue
            report = read(name + ".json")
            assert report["code_sha256"] == code["benchmark_prefill_decode.py"]
            assert report["plan_sha256"] == prepared["plan_sha256"]
            assert report["native_manifest"] == read("native-build.json")
            assert report["smoke_sha256"] == digest(root / "smoke.json")
            if report["status"] == "complete":
                reviews[name] = verify_timing(report, plan)
    winners = [arm for arm in ARMS[2:] if all(name in reviews and
        reviews[name]["candidates"][arm]["passed"] for name in ("timing", "confirmation"))]
    if state["phase"] == "complete":
        assert state["confirmed_component_winners"] == winners
        assert not state["serving_adapter_changed"] and not state["quality_qualified"]
    result = {"status": "verified", "controller_phase": state["phase"],
        "original_service_restored": True, "smoke_status": smoke["status"], "reviews": reviews,
        "confirmed_component_winners": winners, "quality_qualified": False,
        "verifier_sha256": digest(Path(__file__)), "source_files": len(code)}
    args.report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
