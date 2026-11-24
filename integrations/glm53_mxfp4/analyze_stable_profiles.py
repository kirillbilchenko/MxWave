"""Review stable-dispatch model traces without adding overlapping streams or nested forwards."""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from analyze_dispatch_profiles import interval_union_us
from analyze_full_profiles import FORWARD, kernel_group
from compare_full_model import trace_events


def group(name: str) -> str:
    """Keep canonical ordering separate and recognize the observed stock BF16 CUTLASS GEMMs."""
    lower = name.lower()
    if lower == "_canonicalize":
        return "stable_expert_order"
    if "cutlass" in lower and "bf16" in lower and "gemm" in lower:
        return "bf16_dense_linear"
    return kernel_group(name)


def complete_forwards(
    ranges: list[dict[str, Any]], kernels: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Use 84 main-stream expert GEMMs as coverage evidence, then union all contained streams."""
    result = []
    for annotation in ranges:
        start, stop = annotation["ts"], annotation["ts"] + annotation["dur"]
        contained = [
            k for k in kernels
            if start <= k["ts"] and k["ts"] + k["dur"] <= stop + 1
        ]
        experts = [
            k for k in contained
            if group(k["name"]) == "routed_expert_gemm"
            and k.get("args", {}).get("stream", k["tid"]) == annotation["tid"]
        ]
        if len(experts) != 84:
            continue
        shape = FORWARD.match(annotation["name"])
        if shape is None:
            raise ValueError("A forward annotation lost its shape")
        groups: dict[str, dict[str, Any]] = {}
        for k in contained:
            row = groups.setdefault(group(k["name"]), {"calls": 0, "sum_us": 0.})
            row["calls"] += 1
            row["sum_us"] += k["dur"]
        union = interval_union_us([
            (max(start, k["ts"]), min(stop, k["ts"] + k["dur"])) for k in contained
        ])
        result.append({
            "annotation": annotation["name"],
            "shape": [int(shape[i]) for i in (1, 2, 3)],
            "main_stream": annotation["tid"],
            "gpu_range_us": annotation["dur"],
            "kernel_union_all_streams_us": union,
            "uncovered_all_streams_us": max(0., annotation["dur"] - union),
            "kernel_groups": groups,
        })
    return result


def cpu_forwards(
    ranges: list[dict[str, Any]], operations: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Keep complete 68-projection CPU forwards and nested scopes separate from their parents."""
    names = ("mxwave_glm53::kda_linear", "mxwave_glm53::kda_fused_direct",
             "mxwave_zipserv::linear")
    result = []
    for annotation in ranges:
        contained = [op for op in operations if op["tid"] == annotation["tid"]
                     and annotation["ts"] <= op["ts"]
                     and op["ts"] + op["dur"] <= annotation["ts"] + annotation["dur"] + 1]
        groups = {name: {"calls": sum(op["name"] == name for op in contained),
                         "sum_us": sum(op["dur"] for op in contained if op["name"] == name)}
                  for name in names}
        if groups[names[0]]["calls"] + groups[names[1]]["calls"] != 68:
            continue
        shape = FORWARD.match(annotation["name"])
        if shape is None:
            raise ValueError("A CPU forward annotation lost its shape")
        result.append({"annotation": annotation["name"],
                       "shape": [int(shape[i]) for i in (1, 2, 3)],
                       "cpu_forward_span_us": annotation["dur"], "operations": groups})
    return result


def inspect(path: Path) -> dict[str, Any]:
    """Summarize one rank's trace; timestamps never cross node boundaries."""
    ranges, kernels, cpu_ranges, operations = [], [], [], []
    apis: dict[str, list[float]] = collections.defaultdict(list)
    copies = {"calls": 0, "sum_us": 0., "bytes": 0}
    with gzip.open(path, "rt") as stream:
        for event in trace_events(stream):
            if event.get("ph") != "X":
                continue
            category = event.get("cat")
            if category == "kernel":
                kernels.append(event)
            elif category == "gpu_user_annotation" and FORWARD.match(event["name"]):
                ranges.append(event)
            elif category == "user_annotation" and FORWARD.match(event["name"]):
                cpu_ranges.append(event)
            elif category == "cpu_op":
                operations.append(event)
            elif category in ("cuda_runtime", "cuda_driver"):
                apis[category + "/" + event["name"]].append(event["dur"])
            elif category == "gpu_memcpy":
                copies["calls"] += 1
                copies["sum_us"] += event["dur"]
                copies["bytes"] += event.get("args", {}).get("bytes", 0)
    forwards = complete_forwards(ranges, kernels)
    if not forwards:
        raise ValueError("No full forward with 84 routed expert GEMMs")
    shapes: dict[tuple[int, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in forwards:
        shapes[tuple(row["shape"])].append(row)
    summaries = []
    for shape, rows in sorted(shapes.items()):
        names = {key for row in rows for key in row["kernel_groups"]}
        summaries.append({
            "shape": list(shape), "samples": len(rows),
            **{
                "median_" + key: statistics.median(row[key] for row in rows)
                for key in ("gpu_range_us", "kernel_union_all_streams_us",
                            "uncovered_all_streams_us")
            },
            "kernel_groups": {
                name: {
                    "median_" + key: statistics.median(
                        row["kernel_groups"].get(name, {}).get(key, 0.) for row in rows
                    ) for key in ("calls", "sum_us")
                } for name in sorted(names)
            },
        })
    totals: dict[str, dict[str, Any]] = {}
    for event in kernels:
        row = totals.setdefault(event["name"], {"calls": 0, "sum_us": 0.})
        row["calls"] += 1
        row["sum_us"] += event["dur"]
    return {
        "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "complete_forward_count": len(forwards), "shape_summary": summaries,
        "forwards": forwards,
        "cpu_forwards": cpu_forwards(cpu_ranges, operations),
        "canonicalization": totals.get("_canonicalize", {"calls": 0, "sum_us": 0.}),
        "top_kernels": [{"name": name, "group": group(name), **row} for name, row in
                        sorted(totals.items(), key=lambda item: item[1]["sum_us"],
                               reverse=True)[:12]],
        "whole_trace_gpu_copies": copies,
        "whole_trace_apis": {
            name: {"calls": len(values), "sum_us": sum(values)}
            for name, values in sorted(apis.items())
        },
    }


def main() -> None:
    """Analyze explicit labelled traces, keeping inputs and interpretation limits attached."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graph-capture-sizes", type=int, nargs="+", default=[1])
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_bytes())
    result = {
        "status": "complete",
        "configured_cudagraph_capture_sizes": args.graph_capture_sizes,
        "profiles": {label: inspect(Path(path)) for label, path in manifest.items()},
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "analysis_code_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("analyze_stable_profiles.py", "analyze_dispatch_profiles.py",
                         "analyze_full_profiles.py", "compare_full_model.py")
        },
        "limitations": [
            "Profiled spans are not unprofiled request latency or throughput.",
            "Kernel category sums can overlap; only interval unions estimate covered time.",
            "Complete-forward anchors require 84 expert GEMMs on their annotated stream.",
            "All contained device streams are included; unattributed concurrent work can remain.",
            "Device-clock timestamps are never compared across the two nodes.",
            "Shape matches do not prove identical activations or routing between model arms.",
            "Uncovered time is not proof of a CPU bottleneck or a specific synchronization cause.",
            "CPU scope durations include blocked time; nested scopes must not be added to parents.",
            "Configured graph sizes do not prove replay; inspect actual graph launch events.",
        ],
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "complete", "profiles": len(result["profiles"])}))


if __name__ == "__main__":
    main()
