"""Offline exact-shape comparison of both-rank corrected-stock and selective-KDA traces."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from analyze_full_profiles import FORWARD, gpu_steps


def shape_samples(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Describe all complete same-count forwards without claiming matched context lengths."""
    groups: dict[tuple[int, int, int], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in steps:
        shape = FORWARD.match(row["name"])
        if shape is not None and row["kernel_calls"].get("routed_expert_gemm") == 84:
            groups[tuple(int(shape[i]) for i in (1, 2, 3))].append(row)
    result = []
    for shape, rows in sorted(groups.items()):
        kernel_groups = sorted({key for row in rows for key in row["kernel_us"]})
        result.append({
            "shape": {"tokens": shape[0], "context_requests": shape[1],
                      "generation_requests": shape[2]}, "samples": len(rows),
            "median_gpu_range_us": statistics.median(row["gpu_range_us"] for row in rows),
            "median_uncovered_stream_us": statistics.median(
                row["uncovered_stream_us"] for row in rows),
            "median_kernel_union_us": statistics.median(row["kernel_union_us"] for row in rows),
            "median_kernel_groups_us": {group: statistics.median(
                row["kernel_us"].get(group, 0.0) for row in rows) for group in kernel_groups},
            "median_kernel_group_calls": {group: statistics.median(
                row["kernel_calls"].get(group, 0) for row in rows) for group in kernel_groups}})
    return result


def matched_steps(stock: list[dict[str, Any]], packed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare complete 42-layer expert forwards only when detailed shape names agree."""
    arms = []
    for steps in (stock, packed):
        by_name: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for row in steps:
            if row["kernel_calls"].get("routed_expert_gemm") == 84:
                by_name[row["name"]].append(row)
        arms.append(by_name)
    result = []
    for name in sorted(arms[0].keys() & arms[1].keys()):
        shape = FORWARD.match(name)
        assert shape is not None
        groups = sorted({key for arm in arms for row in arm[name] for key in row["kernel_us"]})
        medians = {mode: {
            "samples": len(arm[name]),
            "gpu_range_us": statistics.median(row["gpu_range_us"] for row in arm[name]),
            "kernel_union_us": statistics.median(row["kernel_union_us"] for row in arm[name]),
            "uncovered_stream_us": statistics.median(
                row["uncovered_stream_us"] for row in arm[name]),
            "kernel_groups_us": {group: statistics.median(
                row["kernel_us"].get(group, 0.0) for row in arm[name]) for group in groups},
        } for mode, arm in zip(("stock", "kda"), arms, strict=True)}
        result.append({"annotation": name,
                       "shape": {"tokens": int(shape[1]), "context_requests": int(shape[2]),
                                 "generation_requests": int(shape[3])},
                       **medians,
                       "gpu_range_ratio": medians["kda"]["gpu_range_us"] /
                                          medians["stock"]["gpu_range_us"]})
    return result


def main() -> None:
    """Read captured evidence without starting an engine or issuing GPU requests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    profiles, pairs, rollups, unpaired = {}, [], [], {}
    hashes = {}
    for rank, location in ((0, "traces"), (1, "traces-worker")):
        arms = {}
        for mode in ("stock", "kda"):
            files = list((args.root / mode / location).glob("dp*.trace.json.gz"))
            if len(files) != 1:
                raise ValueError("Require exactly one worker GPU profile per arm and rank")
            path = files[0]
            profile = gpu_steps(path)
            hashes[str(path.relative_to(args.root))] = hashlib.sha256(path.read_bytes()).hexdigest()
            profile["path"] = str(path.relative_to(args.root))
            arms[mode] = profile.pop("steps")
            profiles[f"{mode}_rank{rank}"] = profile
            unpaired[f"{mode}_rank{rank}"] = shape_samples(arms[mode])
        matched = matched_steps(arms["stock"], arms["kda"])
        if not matched:
            raise ValueError("No complete forwards with matching detailed shape annotations")
        pairs.extend({"rank": rank, **row} for row in matched)
        grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = collections.defaultdict(list)
        for row in matched:
            shape = row["shape"]
            grouped[(shape["tokens"], shape["context_requests"],
                     shape["generation_requests"])].append(row)
        for shape, rows in sorted(grouped.items()):
            groups = sorted({g for row in rows for mode in ("stock", "kda")
                             for g in row[mode]["kernel_groups_us"]})
            rollups.append({"rank": rank, "shape": rows[0]["shape"], "matched_shapes": len(rows),
                "median_paired_gpu_range_ratio": statistics.median(
                    row["gpu_range_ratio"] for row in rows),
                "median_gpu_range_us": {mode: statistics.median(
                    row[mode]["gpu_range_us"] for row in rows) for mode in ("stock", "kda")},
                "median_uncovered_stream_us": {mode: statistics.median(
                    row[mode]["uncovered_stream_us"] for row in rows) for mode in ("stock", "kda")},
                "median_kernel_groups_us": {mode: {group: statistics.median(
                    row[mode]["kernel_groups_us"].get(group, 0.0) for row in rows)
                    for group in groups} for mode in ("stock", "kda")}})
    result = {"status": "complete", "profiles": profiles, "matched_steps": pairs,
              "shape_summary": rollups, "trace_sha256": hashes,
              "unpaired_shape_samples": unpaired,
              "analysis_code_sha256": {name: hashlib.sha256(
                  Path(__file__).with_name(name).read_bytes()).hexdigest() for name in
                  ("analyze_kda_profiles.py", "analyze_full_profiles.py", "compare_full_model.py")},
              "limitations": [
                  "Offline GPU annotations only; profiled durations differ from timing medians.",
                  "Exact shape names do not establish identical generated tokens or routing.",
                  "Kernel sums can overlap; GPU range is a separate elapsed measurement.",
                  "Name-based dense BF16 groups include the head and other unchanged projections.",
                  "Unpaired count groups can have different context lengths and request arrivals.",
                  "Uncovered stream spans can include host delays or waits; they do not prove CPU cause.",
                  "Unmatched kernels and incomplete forwards are retained in whole-trace totals."]}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "complete", "matched_shapes": len(pairs),
                      "rank_groups": len(profiles)}))


if __name__ == "__main__":
    main()
