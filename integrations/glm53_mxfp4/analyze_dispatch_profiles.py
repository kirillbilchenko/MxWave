"""Inspect host calls inside complete GLM forwards without adding nested wait durations."""

from __future__ import annotations

import argparse
import bisect
import collections
import gzip
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from analyze_full_profiles import FORWARD, kernel_group
from compare_full_model import trace_events

SCOPES = {
    "mxwave_glm53::kda_linear", "mxwave_glm53::moe_gemm", "mxwave_glm53::embedding_rows",
    "mxwave_glm53::moe_gemm_dispatch",
    "mxwave_zipserv::linear", "_moe_C::moe_wna16_marlin_gemm", "_C::marlin_gemm",
}


def interval_union_us(intervals: list[tuple[float, float]]) -> float:
    """Count overlapping nested CPU/CUDA intervals once."""
    covered, end = 0.0, float("-inf")
    for start, stop in sorted(intervals):
        covered += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return covered


def residual_us(parent: dict[str, Any], children: list[dict[str, Any]]) -> float:
    """Exclude contained child scopes/APIs, without calling the residue pure Python time."""
    start, stop = parent["ts"], parent["ts"] + parent["dur"]
    intervals = [(c["ts"], min(stop, c["ts"] + c["dur"])) for c in children
                 if c is not parent and c["dur"] > 0 and c["ts"] >= start
                 and c["ts"] + c["dur"] <= stop + .01
                 and (c["pid"], c["tid"]) == (parent["pid"], parent["tid"])]
    return max(0.0, parent["dur"] - interval_union_us(intervals))


def inspect(path: Path) -> dict[str, Any]:
    """Bind CPU annotations to GPU forwards that actually contain all 84 expert kernels."""
    threads: dict[tuple[int, int], list[dict[str, Any]]] = collections.defaultdict(list)
    cpu_ranges, gpu_ranges, kernels, activities, cpu_ops = {}, [], [], [], {}
    api_totals: dict[str, list[float]] = collections.defaultdict(list)
    with gzip.open(path, "rt") as handle:
        for event in trace_events(handle):
            cat, args = event.get("cat"), event.get("args", {})
            if event.get("ph") != "X":
                continue
            if cat in ("cpu_op", "cuda_runtime", "cuda_driver"):
                threads[(event["pid"], event["tid"])].append(event)
                if cat == "cpu_op" and "External id" in args:
                    cpu_ops[args["External id"]] = event
                if cat != "cpu_op":
                    api_totals[cat + "/" + event["name"]].append(event["dur"])
            elif cat == "user_annotation" and FORWARD.match(event["name"]):
                cpu_ranges[args["External id"]] = event
            elif cat == "gpu_user_annotation" and FORWARD.match(event["name"]):
                gpu_ranges.append(event)
            elif cat in ("kernel", "gpu_memcpy", "gpu_memset"):
                activities.append(event)
                if cat == "kernel" and kernel_group(event["name"]) == "routed_expert_gemm":
                    kernels.append(event)
    starts = {}
    for key, events in threads.items():
        events.sort(key=lambda event: event["ts"])
        starts[key] = [event["ts"] for event in events]
    rows = []
    for gpu in gpu_ranges:
        count = sum(k["args"].get("stream", k["tid"]) == gpu["tid"]
                    and gpu["ts"] <= k["ts"]
                    and k["ts"] + k["dur"] <= gpu["ts"] + gpu["dur"] + 1 for k in kernels)
        if count != 84:
            continue
        cpu = cpu_ranges[gpu["args"]["External id"]]
        shape = FORWARD.match(cpu["name"])
        assert shape is not None and cpu["name"] == gpu["name"]
        key = (cpu["pid"], cpu["tid"])
        events = threads[key]
        first = bisect.bisect_left(starts[key], cpu["ts"])
        last = bisect.bisect_right(starts[key], cpu["ts"] + cpu["dur"])
        contained = [e for e in events[first:last]
                     if e["ts"] + e["dur"] <= cpu["ts"] + cpu["dur"] + .01]
        custom: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        apis: dict[str, list[float]] = collections.defaultdict(list)
        for event in contained:
            if event["cat"] in ("cuda_runtime", "cuda_driver"):
                apis[event["cat"] + "/" + event["name"]].append(event["dur"])
            elif event["name"] in SCOPES:
                a = bisect.bisect_left(starts[key], event["ts"])
                b = bisect.bisect_right(starts[key], event["ts"] + event["dur"])
                custom[event["name"]].append({"inclusive_us": event["dur"],
                    "residual_us": residual_us(event, events[a:b])})
        kda_scopes = [e for e in contained if e["cat"] == "cpu_op"
                      and e["name"] == "mxwave_glm53::kda_linear"]
        kda_starts = [e["ts"] for e in kda_scopes]
        helpers: dict[str, dict[str, Any]] = {}
        for activity in activities:
            if (activity["args"].get("stream", activity["tid"]) != gpu["tid"]
                    or not gpu["ts"] <= activity["ts"]
                    or activity["ts"] + activity["dur"] > gpu["ts"] + gpu["dur"] + 1):
                continue
            op = cpu_ops.get(activity["args"].get("External id"))
            if op is None or (op["pid"], op["tid"]) != key:
                continue
            index = bisect.bisect_right(kda_starts, op["ts"]) - 1
            if index < 0:
                continue
            owner = kda_scopes[index]
            if op["ts"] + op["dur"] > owner["ts"] + owner["dur"] + .01:
                continue
            if activity["cat"] == "kernel" and kernel_group(activity["name"]) != "other":
                continue
            label = activity["cat"] + "/" + op["name"]
            item = helpers.setdefault(label, {"calls": 0, "sum_us": 0., "bytes": 0,
                                              "activity_names": []})
            item["calls"] += 1
            item["sum_us"] += activity["dur"]
            item["bytes"] += activity["args"].get("bytes", 0)
            if activity["name"] not in item["activity_names"]:
                item["activity_names"].append(activity["name"])
        rows.append({"annotation": cpu["name"], "expert_kernel_calls": count,
            "shape": [int(shape[i]) for i in (1, 2, 3)], "gpu_range_us": gpu["dur"],
            "cpu_range_us": cpu["dur"],
            "custom_scopes": {name: {"calls": len(values),
                "inclusive_us": sum(v["inclusive_us"] for v in values),
                "residual_us": sum(v["residual_us"] for v in values)}
                for name, values in custom.items()},
            "apis": {name: {"calls": len(values), "sum_us": sum(values)}
                     for name, values in apis.items()}, "kda_gpu_helpers": helpers})
    grouped: dict[tuple[int, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[tuple(row["shape"])].append(row)
    summary = []
    for shape, values in sorted(grouped.items()):
        def medians(samples: list[dict[str, Any]], field: str,
                    keys: tuple[str, ...]) -> dict[str, Any]:
            names = {name for row in samples for name in row[field]}
            return {name: {key: statistics.median(row[field].get(name, {}).get(key, 0.)
                                                 for row in samples) for key in keys}
                    for name in sorted(names)}
        summary.append({"shape": list(shape), "samples": len(values),
            "median_cpu_range_us": statistics.median(r["cpu_range_us"] for r in values),
            "custom_scopes": medians(values, "custom_scopes",
                                     ("calls", "inclusive_us", "residual_us")),
            "apis": medians(values, "apis", ("calls", "sum_us")),
            "kda_gpu_helpers": medians(values, "kda_gpu_helpers", ("calls", "sum_us", "bytes"))})
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "complete_forward_samples": len(rows), "shape_summary": summary, "steps": rows,
            "whole_trace_gpu_copies": {
                "calls": sum(e["cat"] == "gpu_memcpy" for e in activities),
                "sum_us": sum(e["dur"] for e in activities if e["cat"] == "gpu_memcpy"),
                "bytes": sum(e["args"].get("bytes", 0) for e in activities
                             if e["cat"] == "gpu_memcpy")},
            "whole_trace_apis": {name: {"calls": len(values), "sum_us": sum(values)}
                                 for name, values in api_totals.items()}}


def main() -> None:
    """Analyze saved stock/candidate worker traces on CPU only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profiles = {}
    for mode in ("stock", "kda"):
        for rank, folder in ((0, "traces"), (1, "traces-worker")):
            paths = list((args.root / mode / folder).glob("dp*.trace.json.gz"))
            if len(paths) != 1:
                raise ValueError("Require exactly one worker trace per arm and rank")
            profiles[f"{mode}_rank{rank}"] = inspect(paths[0])
    result = {"status": "complete", "profiles": profiles,
        "analysis_code_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()
            ).hexdigest() for name in ("analyze_dispatch_profiles.py", "analyze_full_profiles.py",
                                      "compare_full_model.py")},
        "limitations": ["Profiled CPU spans include waiting; they are not unprofiled latency.",
            "Different operation/category totals can overlap and must not be added.",
            "Residual time excludes traced children but is not a Python/ctypes causal breakdown.",
            "M4 count groups differ in context lengths and scheduling; not exact paired steps.",
            "First eager CPU annotation can include waiting for earlier queued GPU work."]}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "complete", "profiles": len(profiles)}))


if __name__ == "__main__":
    main()
