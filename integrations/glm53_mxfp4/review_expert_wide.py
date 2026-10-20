"""Independently audit wider Marlin outputs, graph evidence and stable component timing."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from .audit_drafter_feature_export_numpy import arrays
from .drafter_fp8_roundtrip import digest

PARTS = ("up", "activation", "down", "sum")


def changed_prefix(a: np.ndarray, b: np.ndarray, rows_a: int, rows_b: int,
                   shared: int) -> int:
    """Count changed BF16 payloads per shared token, independent of route dispatch width."""
    if (a.dtype != np.uint16 or b.dtype != np.uint16 or min(rows_a, rows_b, shared) < 1
            or shared > min(rows_a, rows_b) or a.size % rows_a or b.size % rows_b):
        raise ValueError("Invalid BF16 token-major comparison")
    x, y = a.reshape(rows_a, -1)[:shared], b.reshape(rows_b, -1)[:shared]
    if x.shape != y.shape:
        raise ValueError("Per-token expert geometry changed")
    return int(np.count_nonzero(x != y))


def timing_pair(native: list[float], padded: list[float]) -> dict[str, Any]:
    """Retain failures of the original 2% stability limit without inventing a speed result."""
    if (len(native) != len(padded) or len(native) < 8 or len(native) % 2
            or any(not np.isfinite(x) or x <= 0 for x in [*native, *padded])):
        raise ValueError("Invalid paired timing samples")
    half = len(native) // 2
    drifts = [statistics.median(s[half:]) / statistics.median(s[:half]) - 1
              for s in (native, padded)]
    return {"native_us": statistics.median(native), "padded_us": statistics.median(padded),
            "paired_median_overhead": statistics.median(p / n - 1 for p, n in
                                                       zip(padded, native, strict=True)),
            "native_padded_half_drifts": drifts,
            "control_pass": all(abs(d) <= 0.02 for d in drifts)}


def review(root: Path, components: Path, output: Path) -> dict[str, Any]:
    """Verify immutable evidence, unchanged shared bits and separately gated overhead."""
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
    assert report["source_sha256"] == digest(root / "replay_expert_wide.py")
    assert report["plan_sha256"] == digest(root / "plan.json") == prepared["plan_sha256"]
    assert report["installed_marlin_sha256"] == plan["installed_marlin_sha256"]
    for name, sha in plan["source_sha256"].items():
        assert digest(root / name) == sha
    fixtures = json.loads((root / "fixtures/manifest.json").read_bytes())
    assert digest(root / "fixtures/manifest.json") == plan["fixture_manifest_sha256"]
    assert len(fixtures["files"]) == 36
    scope = plan["scope"]
    expected_groups = {(m, p) for m in scope["rows"]
                       for p in (["repeat"] if m == 8 else scope["patterns"])}
    comparisons, timings, native_matches, stock_matches, repeats = [], [], 0, 0, 0
    for family, count in (("w13", 1152), ("w2", 576)):
        proofs = [json.loads((root / f"result/weight-sources-r{r}-{family}.json").read_bytes())
                  for r in (0, 1)]
        assert proofs[0] == proofs[1] and len(proofs[0]) == count
        assert len({p["name"] for p in proofs[0]}) == count
    assert len(report["ranks"]) == 2
    for rank in (0, 1):
        row = report["ranks"][rank]
        assert row["rank"] == rank and row["scale_bytes_exact"]
        assert len(row["groups"]) == len(expected_groups)
        assert {(g["rows"], g["pattern"]) for g in row["groups"]} == expected_groups
        bases = {}
        for group in row["groups"]:
            rows, pattern = group["rows"], group["pattern"]
            path = root / "result" / group["output"]["path"]
            assert digest(path) == group["output"]["sha256"]
            tensors = arrays(path)
            assert len(tensors) == 34 and len(group["cases"]) == 8
            assert group["numerical_gate_passed"]
            assert group["repeat_checks"] == scope["graph_repeat_cycles"] * 8
            repeats += group["repeat_checks"]
            for case in group["cases"]:
                assert case["routing_block"] == 8
                f = next(f for f in fixtures["files"] if (f["rank"], f["arm"], f["rows"],
                         f["pattern"]) == (rank, case["arm"], rows, pattern))
                assert case["fixture_sha256"] == f["sha256"]
                for projection in case["projections"]:
                    if case["policy"] == "native":
                        continue
                    p, tile = projection["padding"], projection["tile"]
                    n = 2048 if projection["projection"] == 0 else 4096
                    assert p["real_entries_exact"] and p["padding_masked"]
                    assert p["before"] <= p["after"] <= p["allocated"]
                    assert p["after"] - p["before"] < 8 * p["quantum_blocks"]
                    assert p["after"] % (8 * p["quantum_blocks"]) == 0
                    assert p["after"] // 8 * (n // tile[1]) % (report["sms"] * tile[2]) == 0
            for arm in ("a1", "b1"):
                for implementation in scope["implementations"]:
                    assert np.array_equal(tensors[f"{arm}_native_{implementation}_sum"],
                                          tensors[f"{arm}_installed_native_sum"])
                    native_matches += 1
                if rows == 8:
                    original_path = components / f"{arm}-r{rank}" / (
                        f"context-opencoder-evol-15654-capture1-0005-r{rank}.safetensors"
                    )
                    metadata = json.loads(original_path.with_suffix(".json").read_bytes())
                    assert digest(original_path) == metadata["payload_sha256"]
                    original = arrays(original_path)
                    events = {e["name"]: e for e in metadata["components"]["events"]}
                    for part, event_name, leaf_path in (
                        ("up", "layer4.moe_gemm#0/output", None),
                        ("activation", "layer4.moe_gemm#1/input", "/0"),
                        ("down", "layer4.moe_gemm#1/output", None),
                    ):
                        event = events[event_name]
                        leaf = (event["tensors"][0] if leaf_path is None else
                                next(x for x in event["tensors"] if x["path"] == leaf_path))
                        for implementation in scope["implementations"]:
                            assert np.array_equal(tensors[f"{arm}_native_{implementation}_{part}"],
                                                  original[leaf["payload_key"]])
                            native_matches += 1
                for policy in scope["policies"]:
                    for part in PARTS:
                        assert np.array_equal(tensors[f"{arm}_{policy}_stock_{part}"],
                                              tensors[f"{arm}_{policy}_packed_{part}"])
                        stock_matches += 1
            for check in group["comparisons"]:
                policy, implementation = check["policy"], check["implementation"]
                counts = {}
                for part in PARTS:
                    if check["kind"] == "shared_arms":
                        a, b = (tensors[f"{arm}_{policy}_{implementation}_{part}"]
                                for arm in ("a1", "b1"))
                        count = changed_prefix(a, b, rows, rows, 7)
                    else:
                        assert check["kind"] == "shared_widths" and rows > 8
                        key = f"{check['arm']}_{policy}_{implementation}_{part}"
                        count = changed_prefix(bases[key], tensors[key], 8, rows, 8)
                    counts[part] = count
                assert counts == check["changed_elements"]
                assert (not any(counts.values())) == check["exact"]
                if policy != "native":
                    assert check["exact"]
                comparisons.append({"rank": rank, "rows": rows, "pattern": pattern, **check})
            assert len(group["comparisons"]) == (4 if rows == 8 else 12)
            if rows == 8:
                bases = tensors
            if rows in scope["timing_rows"]:
                samples = group["timings"]
                assert len(samples) == 8
                assert all(len(s["samples_us"]) == scope["timing_interleaved_cycles"]
                           for s in samples.values())
                for arm in ("a1", "b1"):
                    for implementation in scope["implementations"]:
                        timings.append({"rank": rank, "rows": rows, "pattern": pattern,
                                        "arm": arm, "implementation": implementation,
                                        **timing_pair(samples[f"{arm}:native:{implementation}"]
                                                      ["samples_us"],
                                                      samples[f"{arm}:pad_default:{implementation}"]
                                                      ["samples_us"])})
    assert len(report["telemetry"]) == 8 * scope["timing_interleaved_cycles"]
    result = {
        "status": "verified", "payloads_verified": len(manifest["files"]),
        "source_sha256": digest(Path(__file__)), "native_output_matches": native_matches,
        "stock_packed_matches": stock_matches, "exact_graph_repeat_checks": repeats,
        "component_seconds": report["finished_epoch"] - report["started_epoch"],
        "comparisons": comparisons, "timings": timings,
        "telemetry_available": all(t["returncode"] == 0 for t in report["telemetry"]),
        "timing_controls_passed": sum(t["control_pass"] for t in timings),
        "timing_controls_total": len(timings),
        "limits": "Synthetic layer-four fixtures; separate TP partitions on one GPU. No full-model C8/C16 invariance or serving-overhead claim. Failed timing controls remain failures. Sealed test closed.",
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    """Review complete archived evidence without GPU access or service changes."""
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "components", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    result = review(args.root, args.components, args.output)
    print(json.dumps({k: v for k, v in result.items() if k not in ("comparisons", "timings")},
                     indent=2))


if __name__ == "__main__":
    main()
