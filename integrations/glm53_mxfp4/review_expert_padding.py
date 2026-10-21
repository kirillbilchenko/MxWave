"""Independently verify archived expert-padding outputs and timing controls using NumPy."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from .audit_drafter_feature_export_numpy import arrays
from .drafter_fp8_roundtrip import digest


def review(root: Path, components: Path, output: Path) -> dict[str, Any]:
    """Check source/restoration provenance, actual retained bits, and paired component overhead."""
    assert not output.exists()
    manifest = json.loads((root / "evidence-manifest.json").read_bytes())
    for item in manifest["files"]:
        path = root / item["path"]
        assert path.stat().st_size == item["bytes"] and digest(path) == item["sha256"]
    state = json.loads((root / "controller.json").read_bytes())
    prepared = json.loads((root / "prepared.json").read_bytes())
    assert state["phase"] == "complete_restored" and not state["service_paused"]
    restored = state["restoration"]
    assert restored["health_200"] and restored["originals_running"]
    assert restored["live"] == prepared["live"] and restored["originals"] == prepared["originals"]
    assert any(x["max_model_len"] == 500000 for x in restored["models"]["data"])
    plan = json.loads((root / "plan.json").read_bytes())
    report = json.loads((root / "result/report.json").read_bytes())
    assert report["status"] == "complete"
    assert report["source_sha256"] == digest(root / "replay_expert_padding.py")
    assert report["plan_sha256"] == digest(root / "plan.json") == prepared["plan_sha256"]
    for name, sha in plan["source_sha256"].items():
        assert digest(root / name) == sha
    assert report["output_sha256"] == digest(root / "result/outputs.safetensors")
    tensors = arrays(root / "result/outputs.safetensors")
    policies = plan["scope"]["policies"]
    graph_count = 4 * len(policies)
    cycles = plan["scope"]["timing_interleaved_cycles"]
    half = cycles // 2
    assert len(tensors) == 64 * len(policies)
    comparisons, timings, native_matches, stock_matches, repeats = [], [], 0, 0, 0
    full_k_padding = []
    parts = ("up", "activation", "down", "sum")
    for family, count in (("w13", 1152), ("w2", 576)):
        proofs = [json.loads((root / f"result/weight-sources-r{r}-{family}.json").read_bytes())
                  for r in (0, 1)]
        assert proofs[0] == proofs[1] and len(proofs[0]) == count
        assert len({p["name"] for p in proofs[0]}) == count
    for rank in (0, 1):
        r = report["ranks"][rank]
        assert r["rank"] == rank and len(r["cases"]) == 2 * graph_count
        assert len(r["repeat_checks"]) == 2
        for check in r["repeat_checks"]:
            assert check["all_bitwise_exact"] and check["comparisons"] == 20 * graph_count
            repeats += check["comparisons"]
        for case in r["cases"]:
            if case["policy"] == "native":
                assert case["native_matches"]
                continue
            for projection in case["projections"]:
                if case["policy"] == "full_k":
                    assert "padding" not in projection
                    assert projection["consumer"] in ("raw_full_k", "packed_full_k")
                    assert all(value > 0 for value in projection["tile"])
                    continue
                p = projection["padding"]
                assert p["real_entries_exact"] and p["padding_masked"]
                assert p["before"] <= p["after"] <= p["allocated"]
                assert p["after"] % (8 * p["quantum_blocks"]) == 0
        for position in (46, 54):
            shared, number = (8, 4) if position == 46 else (7, 5)
            for arm in ("a1", "b1"):
                path = components / f"{arm}-r{rank}" / (
                    f"context-opencoder-evol-15654-capture1-{number:04d}-r{rank}.safetensors"
                )
                metadata = json.loads(path.with_suffix(".json").read_bytes())
                assert digest(path) == metadata["payload_sha256"]
                original = arrays(path)
                events = {e["name"]: e for e in metadata["components"]["events"]}
                for part, event_name, leaf_path in (
                    ("up", "layer4.moe_gemm#0/output", None),
                    ("activation", "layer4.moe_gemm#1/input", "/0"),
                    ("down", "layer4.moe_gemm#1/output", None),
                ):
                    event = events[event_name]
                    leaf = (event["tensors"][0] if leaf_path is None else
                            next(x for x in event["tensors"] if x["path"] == leaf_path))
                    expected = original[leaf["payload_key"]].tobytes()
                    for implementation in ("stock", "packed"):
                        key = f"r{rank}_{position}_{arm}_native_{implementation}_{part}"
                        assert tensors[key].tobytes() == expected
                        native_matches += 1
                for policy in policies:
                    for part in parts:
                        prefix = f"r{rank}_{position}_{arm}_{policy}"
                        assert tensors[prefix + "_stock_" + part].tobytes() == (
                            tensors[prefix + "_packed_" + part].tobytes()
                        )
                        stock_matches += 1
                if "full_k" in policies:
                    for implementation in ("stock", "packed"):
                        for part in parts:
                            prefix = f"r{rank}_{position}_{arm}_"
                            suffix = f"_{implementation}_{part}"
                            full_k_padding.append({
                                "rank": rank, "position": position, "arm": arm,
                                "implementation": implementation, "part": part,
                                "bits_exact": tensors[prefix + "full_k" + suffix].tobytes()
                                == tensors[prefix + "pad_default" + suffix].tobytes(),
                            })
            for policy in policies:
                for implementation in ("stock", "packed"):
                    counts = {}
                    for part in parts:
                        a, b = (tensors[f"r{rank}_{position}_{arm}_{policy}_{implementation}_{part}"]
                                .reshape(8, -1) for arm in ("a1", "b1"))
                        counts[part] = np.count_nonzero(a != b, axis=1).tolist()
                        if policy != "native" or position == 46:
                            assert not any(counts[part][:shared]), (rank, policy, part, counts)
                    if position == 54 and policy == "native":
                        expected = ([0, 1, 0, 0, 0, 0, 0] if rank == 0 else
                                    [0, 0, 0, 0, 2, 0, 0])
                        assert counts["up"][:shared] == expected
                    comparisons.append({"rank": rank, "position": position, "policy": policy,
                                        "implementation": implementation, "shared_rows": shared,
                                        "different_elements_per_row": counts})
            samples = next(x["graphs"] for x in r["timings"] if x["position"] == position)
            assert len(samples) == graph_count
            assert all(len(x["samples_us"]) == cycles for x in samples.values())
            for arm in ("a1", "b1"):
                for implementation in ("stock", "packed"):
                    native = samples[f"{arm}:native:{implementation}"]
                    for policy in policies[1:]:
                        padded = samples[f"{arm}:{policy}:{implementation}"]
                        drifts = [statistics.median(x["samples_us"][half:]) /
                                  statistics.median(x["samples_us"][:half]) - 1
                                  for x in (native, padded)]
                        timings.append({
                            "rank": rank, "position": position, "arm": arm,
                            "implementation": implementation, "policy": policy,
                            "native_us": statistics.median(native["samples_us"]),
                            "padded_us": statistics.median(padded["samples_us"]),
                            "paired_median_overhead": statistics.median(
                                p / n - 1 for p, n in zip(padded["samples_us"],
                                                         native["samples_us"], strict=True)
                            ),
                            "native_padded_half_drifts": drifts,
                            "control_pass": all(abs(d) <= 0.02 for d in drifts),
                        })
    assert len(report["telemetry"]) == 4 * cycles
    telemetry_ok = all(t["returncode"] == 0 for t in report["telemetry"])
    result = {
        "status": "verified", "payloads_verified": len(manifest["files"]),
        "source_sha256": digest(Path(__file__)), "native_output_matches": native_matches,
        "stock_packed_matches": stock_matches, "reported_exact_graph_repeat_checks": repeats,
        "component_seconds": report["finished_epoch"] - report["started_epoch"],
        "comparisons": comparisons, "timings": timings, "telemetry_available": telemetry_ok,
        "full_k_versus_padding": full_k_padding,
        "timing_controls_passed": sum(t["control_pass"] for t in timings),
        "timing_controls_total": len(timings),
        "conclusion": "Every tested correction removes shared-row differences in this fixture.",
        "limits": [
            "Both TP partitions run separately on one GPU; no distributed or serving speed result.",
            "Timing ratios whose control_pass is false are descriptive only.",
            "This component result does not prove later target operations invariant.",
            "The adapted drafter remains unqualified and the sealed test remains closed.",
        ],
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    """Review archived evidence without importing PyTorch or accessing the service."""
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "components", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    result = review(args.root, args.components, args.output)
    print(json.dumps({k: v for k, v in result.items() if k not in ("comparisons", "timings")},
                     indent=2))


if __name__ == "__main__":
    main()
