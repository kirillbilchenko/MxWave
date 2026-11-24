"""Independently retain C8/C16 source, timing, probability and device-trace evidence."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    """Hash a retained artifact with bounded memory."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def numerical(root: Path, first: str, second: str) -> dict:
    """Normalize raw full-vocabulary rows and compare all forty-eight actual decode steps."""
    a, b = [json.loads((root / (n + "-decode-quality.json")).read_bytes())
            for n in (first, second)]
    assert a["status"] == b["status"] == "complete"
    assert a["vocabulary"] == b["vocabulary"] == 154880
    assert a["concurrency"] == b["concurrency"] and a["concurrency"] in (8, 16)
    assert len(a["contexts"]) == len(b["contexts"]) == 16
    ids_equal, conditioning_equal, decoded, expected = True, True, 0, 0
    kls, nlls = [], []
    for x, y in zip(a["contexts"], b["contexts"], strict=True):
        assert x["index"] == y["index"] and x["prompt_sha256"] == y["prompt_sha256"]
        assert len(x["steps"]) == len(y["steps"]) == 4
        ids_equal &= x["token_ids"] == y["token_ids"]
        for i, j in zip(x["steps"], y["steps"], strict=True):
            assert i["position"] == j["position"] and i["kind"] == j["kind"]
            arrays = []
            for item in (i, j):
                path = root / item["file"]
                assert path.name == item["file"] and sha(path) == item["sha256"]
                value = np.fromfile(path, dtype="<f4").astype(np.float64)
                assert len(value) == a["vocabulary"] and not np.isnan(value).any()
                maximum = float(np.max(value))
                assert math.isfinite(maximum)
                arrays.append(value - maximum - math.log(float(np.exp(value - maximum).sum())))
            if i["kind"] != "decode":
                continue
            expected += 1
            if i["conditioning_token_ids"] != j["conditioning_token_ids"]:
                conditioning_equal = False
                continue
            decoded += 1
            p, q = arrays
            supported = np.isfinite(p)
            assert np.isfinite(q[supported]).all()
            kls.append(max(0., float(np.sum(np.exp(p[supported]) * (p[supported] - q[supported])))))
            token = x["token_ids"][i["position"]]
            nlls.append(float(p[token] - q[token]))
    kl = math.fsum(kls) / len(kls) if kls else None
    nll = math.fsum(nlls) / len(nlls) if nlls else None
    passed = (decoded == expected == 48 and ids_equal and conditioning_equal
              and kl is not None and nll is not None and kl <= 1e-5 and abs(nll) <= 1e-4)
    return {"reference": first, "candidate": second, "decode_steps": decoded,
        "expected_decode_steps": expected, "all_generated_ids_match": ids_equal,
        "all_decode_conditioning_matches": conditioning_equal,
        "mean_decode_kl": kl, "mean_decode_nll_difference": nll, "passed": passed}


def main() -> None:
    """Verify complete or failed-restored experiments without treating a failed model gate as success."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    inventory = json.loads((root / "evidence-inventory.json").read_bytes())["files"]
    entries = {row["path"]: row for row in inventory}
    assert len(entries) == len(inventory)
    for relative, row in entries.items():
        assert not Path(relative).is_absolute() and ".." not in Path(relative).parts
        path = root / relative
        assert path.is_file() and not path.is_symlink()
        assert path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], relative
    prepared = json.loads((root / "prepared.json").read_bytes())
    parent = json.loads((root / "parent-prepared.json").read_bytes())
    assert prepared["bindings"][0] == prepared["bindings"][1]
    binding = prepared["bindings"][0]
    assert len(binding["code"]) == 169 and len(parent["bindings"][0]["code"]) == 160
    for relative, expected in binding["code"].items():
        assert sha(root / "source" / relative) == expected
    changed = {name for name, expected in parent["bindings"][0]["code"].items()
               if binding["code"].get(name) != expected}
    assert changed == {"engine_plugin/pyproject.toml"}
    for filename, expected in prepared["protocol_files_sha256"].items():
        assert sha(root / "results" / filename) == expected
    assert sha(root / "source-overlay.tar.gz") == prepared["source_archive_sha256"]
    for name, key in (("native-CPU-queue-proof.json", "native_queue_CPU_proof_sha256"),
                      ("native-CPU-dispatch-proof.json", "native_dispatch_CPU_proof_sha256")):
        assert sha(root / name) == prepared[key]
    controller = json.loads((root / "controller.json").read_bytes())
    restored = json.loads((root / "post-restoration-check.json").read_bytes())
    assert controller["phase"] in ("complete", "failed_restored", "preparation_failed")
    assert not controller["service_paused"] and restored["health"] == 200
    assert restored["originals"] == prepared["originals"]
    plan = json.loads((root / "results/high-concurrency-plan.json").read_bytes())
    sys.path.insert(0, str(root / "source"))
    from high_concurrency_protocol import validate_plan, validate_snapshot

    validate_plan(plan)
    assert not plan["speculative_decoding"]
    assert plan["mean_decode_kl_floor"] == 1e-5 and plan["abs_mean_decode_nll_floor"] == 1e-4
    result = {"status": "passed", "verified_payloads": len(entries), "source_files": 169,
        "all_parent_arithmetic_sources_unchanged": True, "changed_parent_metadata": sorted(changed),
        "controller_phase": controller["phase"], "restored_health": 200,
        "quality_qualified": False, "previous_failed_gates_preserved": True,
        "verifier_sha256": sha(Path(__file__)), "worker_profiles": []}
    arms = {}
    source_groups = 0
    audit = json.loads((root / "results/source-audit.json").read_bytes())
    for arm in ("control", "candidate"):
        path = root / "results" / (arm + "-high-concurrency.json")
        if not path.exists():
            continue
        report = json.loads(path.read_bytes())
        arms[arm] = report
        for phase in ("source_before", "source_after"):
            if phase not in report:
                assert controller["phase"] != "complete"
                continue
            rows = report[phase]
            assert len(rows) == 2 and {r["tp_rank"] for r in rows} == {0, 1}
            for row in rows:
                expected = Counter((tuple(c["shape"]), c["raw_sha256"]) for c in audit["cases"]
                                   if c["rank"] == row["tp_rank"])
                actual = Counter((tuple(c["shape"]), c["sha256"])
                                 for c in row["source_scale_hashes"])
                assert row["status"] == "passed" and row["candidate"] == (arm == "candidate")
                assert expected == actual
                source_groups += sum(actual.values())
        for trial in report["trials"]:
            validate_snapshot(trial["before"])
            validate_snapshot(trial["after"])
            requests = trial["measurement"]["requests"]
            assert len(requests) == trial["concurrency"]
            wall = trial["measurement"]["wall_seconds"]
            for request in requests:
                assert len(request["token_ids"]) == sum(a["tokens"] for a in request["arrivals"]) == 64
                times = [a["seconds"] for a in request["arrivals"]]
                assert 0 < times[0] < times[-1] <= request["elapsed_seconds"] <= wall
                assert times == sorted(times)
            assert math.isclose(len(requests) * 64 / wall,
                trial["measurement"]["aggregate_tokens_per_second"], rel_tol=1e-12)
            assert math.isclose(statistics.median(r["arrivals"][0]["seconds"] for r in requests),
                trial["measurement"]["median_ttft_seconds"], rel_tol=1e-12)
    result["source_scale_groups_verified"] = source_groups
    if controller["phase"] == "complete":
        assert set(arms) == {"control", "candidate"} and source_groups == 672
        assert all(r["status"] == "complete" and len(r["trials"]) == 72 for r in arms.values())
        assert sum(len(t["measurement"]["requests"]) for r in arms.values() for t in r["trials"]) == 1272
        assert len([n for n in entries if n.startswith("results/") and n.endswith(".f32")]) == 1024
        from analyze_high_concurrency import quality_pairs, review

        independently_recomputed = review(root / "results")
        saved = json.loads((root / "results/quality-model-review.json").read_bytes())
        saved.pop("analysis_sha256")
        differences = []

        def compare_values(a, b, location=""):
            assert type(a) is type(b), location
            if isinstance(a, dict):
                assert a.keys() == b.keys(), location
                for key in a:
                    compare_values(a[key], b[key], location + "/" + str(key))
            elif isinstance(a, list):
                assert len(a) == len(b), location
                for index, (x, y) in enumerate(zip(a, b, strict=True)):
                    compare_values(x, y, location + "/" + str(index))
            elif a != b:
                assert (location.startswith("/numerical_comparisons/") and isinstance(a, float)
                        and isinstance(b, float) and abs(a - b) <= 1e-12), location
                differences.append(abs(a - b))

        compare_values(independently_recomputed, saved)
        independent_probabilities = [numerical(root / "results", a, b) for a, b in quality_pairs()]
        for actual, expected in zip(independent_probabilities, saved["numerical_comparisons"], strict=True):
            for key, value in actual.items():
                other = expected[key]
                if isinstance(value, float):
                    assert abs(value - other) <= 1e-12, key
                else:
                    assert value == other, key
        result.update(timed_groups=144, timed_requests=1272, timed_tokens=81408,
            numerical_comparisons=independent_probabilities, decisions=saved["decisions"],
            summaries=saved["summaries"], comparisons=saved["comparisons"],
            numerical_gate_passed=saved["numerical_gate_passed"],
            combined_cuda_saving_bytes=saved["combined_cuda_saving_bytes"],
            probability_recomputation_max_difference=max(differences, default=0.),
            probability_recomputation_only_tolerance=1e-12)
    else:
        result.update(inference_completed=False, failure_preserved=controller.get("error"),
                      retained_arm_status={arm: r["status"] for arm, r in arms.items()})
    for arm, report in arms.items():
        stage = "raw" if arm == "control" else "aligned"
        for rank in (0, 1):
            names = [name for name in entries if name.startswith(f"traces/rank{rank}/{stage}/")
                     and Path(name).name.startswith(f"dp0_pp0_tp{rank}_")
                     and name.endswith((".pt.trace.json.gz", ".pt.trace.json"))]
            names.sort(key=lambda name: entries[name]["mtime_epoch"])
            if controller["phase"] == "complete":
                assert len(names) == len(report["profiles"]) == 4
            for index, name in enumerate(names):
                path = root / name
                with (gzip.open(path, "rt") if name.endswith(".gz") else path.open()) as handle:
                    trace = json.load(handle)
                events = trace["traceEvents"]
                kernels = [e for e in events if e.get("cat") == "kernel"]
                decoder = [e for e in kernels if "BF16TripleBitmap_Decompress_Kernel<" in e.get("name", "")]
                value = {"arm": arm, "rank": rank, "path": name, "kernel_events": len(kernels),
                    "kernel_duration_sum_us": sum(e.get("dur", 0) for e in kernels),
                    "decompression_kernel_events": len(decoder),
                    "decompression_duration_sum_us": sum(e.get("dur", 0) for e in decoder),
                    "cuda_graph_launches": sum(e.get("cat") == "cuda_runtime" and
                        e.get("name", "").startswith("cudaGraphLaunch") for e in events),
                    "forward_annotations": [e["name"] for e in events if
                        e.get("cat") == "user_annotation" and e.get("name", "").startswith("execute_")]}
                if len(names) == len(report["profiles"]):
                    profile = report["profiles"][index]
                    value.update(concurrency=profile["concurrency"], window_ms=profile["window_ms"],
                        trace_time_matches_profile=(profile["started_epoch"] - 10 <=
                            entries[name]["mtime_epoch"] <= profile["finished_epoch"] + 30))
                    assert value["trace_time_matches_profile"]
                result["worker_profiles"].append(value)
                del trace, events, kernels, decoder
    result["profile_duration_limit"] = (
        "Kernel duration sums are separate diagnostic observations, not request time or a TP critical path.")
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "verified_payloads", "source_files", "controller_phase")}))


if __name__ == "__main__":
    main()
