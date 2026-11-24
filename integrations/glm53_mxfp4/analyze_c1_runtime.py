"""Attribute native C1 trace cycles while keeping overlaps and unmeasured gaps explicit."""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import itertools
import json
import statistics
from pathlib import Path
from typing import Any

from analyze_dispatch_profiles import interval_union_us


def expert(name: str) -> bool:
    """Recognize the two retained Marlin expert implementations."""
    return "marlin_moe_wna16::Marlin" in name or "mxwave_grouped_" in name


def communication(name: str) -> bool:
    """Recognize native NCCL and the retained production RoCE kernels by name."""
    return any(x in name.lower() for x in ("nccl", "roce", "allreduce", "all_reduce",
                                          "allgather", "all_gather"))


def graph_role(kernels: list[dict[str, Any]]) -> str:
    """Require complete target coverage or the five/six-layer drafter signature."""
    experts = sum(expert(k["name"]) for k in kernels)
    if experts == 84:
        return "target"
    rotary = sum("rotary_embedding_kernel" in k["name"] for k in kernels)
    convolution = sum("_grouped_conv_kernel" in k["name"] for k in kernels)
    if experts == 0 and rotary in (5, 6) and convolution == rotary * 4:
        return "draft"
    attention = sum(k["name"] == "kernel_unified_attention" for k in kernels)
    selector = sum(k["name"] == "_selector_walk_kernel" for k in kernels)
    if (experts == 0 and attention in (5, 6) and convolution == attention * 4
            and rotary == attention + 1 and selector == 1):
        return "draft"
    return "unknown_graph"


def vocabulary_mm(operation: dict[str, Any] | None) -> bool:
    """Identify dense vocabulary products through recorded GEMM tensor dimensions."""
    if operation is None or operation["name"] not in ("aten::mm", "aten::addmm"):
        return False
    shapes = operation.get("args", {}).get("Input Dims", [])
    return any(isinstance(shape, list) and any(d in (77440, 154880) for d in shape)
               for shape in shapes)


def inspect_trace(path: Path) -> dict[str, Any]:
    """Analyze one worker independently; never compare timestamps across hosts."""
    with gzip.open(path, "rt") as stream:
        events = json.load(stream)["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"]
    if not kernels:
        return {"worker_trace": False, "path": str(path)}
    runtime = {e["args"]["correlation"]: e for e in events
               if e.get("cat") in ("cuda_runtime", "cuda_driver")
               and "correlation" in e.get("args", {})}
    cpu = {e["args"]["External id"]: e for e in events
           if e.get("cat") == "cpu_op" and "External id" in e.get("args", {})}
    graphs: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for event in kernels:
        if event.get("args", {}).get("graph id"):
            graphs[event["args"]["correlation"]].append(event)
    roles = {correlation: graph_role(rows) for correlation, rows in graphs.items()}
    blocks = []
    for correlation, rows in graphs.items():
        blocks.append({"correlation": correlation, "role": roles[correlation],
                       "graph_id": rows[0]["args"]["graph id"],
                       "kernel_count": len(rows), "expert_gemms": sum(
                           expert(k["name"]) for k in rows),
                       "start_us": min(k["ts"] for k in rows),
                       "end_us": max(k["ts"] + k["dur"] for k in rows)})
    blocks.sort(key=lambda b: b["start_us"])
    annotations = [e for e in events if e.get("cat") == "gpu_user_annotation"
                   and e.get("name", "").startswith("execute_")]
    # Native 0.32 emits one annotation fragment per graph node/stream, whereas
    # the older runtime emits enclosing ranges. Bind their shared unique name
    # to the CPU invocation, using the envelope only to locate its target graph.
    envelopes: dict[str, tuple[float, float]] = {}
    for event in annotations:
        name = event["name"]
        left, right = event["ts"], event["ts"] + event["dur"]
        previous = envelopes.get(name, (left, right))
        envelopes[name] = min(previous[0], left), max(previous[1], right)
    native_calls = sorted((e for e in events if e.get("cat") == "user_annotation"
                           and e.get("name", "").startswith("execute_")
                           and "_generation_1(" in e["name"]), key=lambda e: e["ts"])
    name_counts = collections.Counter(e["name"] for e in native_calls)
    ordinals = {e["name"]: i for i, e in enumerate(native_calls) if name_counts[e["name"]] == 1}
    for block in blocks:
        matching = [name for name, (left, right) in envelopes.items()
                    if left <= block["start_us"] + 1 and right >= block["end_us"] - 1]
        block["native_annotations"] = sorted(set(matching))
        indices = {ordinals[name] for name in matching if name in ordinals}
        block["native_cpu_ordinal"] = next(iter(indices)) if len(indices) == 1 else None
    top: dict[str, dict[str, Any]] = {}
    evidence = collections.Counter()
    for event in kernels:
        args, name = event.get("args", {}), event["name"]
        role = roles.get(args.get("correlation"), "outside")
        launch = runtime.get(args.get("correlation"), {})
        operation = cpu.get(launch.get("args", {}).get("External id"))
        if communication(name):
            group = role + "/communication"
        elif name == "_head_fp8_kernel":
            group = "head/draft_fp8"
        elif name in {"_xnorm_kernel", "_screen_kernel", "_select_kernel", "_exact_kernel",
                      "_final_kernel", "_block_argmax_kernel", "_gumbel_bounds_kernel"}:
            group = "head/certified_target"
        elif vocabulary_mm(operation):
            group = "head/dense_vocabulary"
            evidence["vocabulary_mm_from_launch_external_id_and_shape"] += 1
        elif role == "target" and expert(name):
            group = "target/expert_gemm"
        else:
            group = role + "/other_compute"
        event["analysis_group"] = group
        row = top.setdefault(name, {"name": name, "calls": 0, "sum_ms": 0., "groups": []})
        row["calls"] += 1
        row["sum_ms"] += event["dur"] / 1000
        if group not in row["groups"]:
            row["groups"].append(group)
    targets = [b for b in blocks if b["role"] == "target"]
    cycles = []
    for index, (current, following) in enumerate(itertools.pairwise(targets)):
        start, stop = current["start_us"], following["start_us"]
        buckets: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
        for event in kernels:
            left, right = max(start, event["ts"]), min(stop, event["ts"] + event["dur"])
            if right > left:
                buckets[event["analysis_group"]].append((left, right))
        all_intervals = [interval for values in buckets.values() for interval in values]
        active = interval_union_us(all_intervals)
        groups = {name: interval_union_us(intervals) / 1000 for name, intervals in buckets.items()}
        a, b = current["native_cpu_ordinal"], following["native_cpu_ordinal"]
        cycles.append({"index": index, "start_us": start, "end_us": stop,
                       "native_cpu_ordinal_gap": b - a if a is not None and b is not None else None,
                       "span_ms": (stop - start) / 1000,
                       "target_graph_ms": (current["end_us"] - start) / 1000,
                       "gpu_kernel_union_ms": active / 1000,
                       "no_traced_kernel_ms": (stop - start - active) / 1000,
                       "group_union_ms": groups,
                       "cross_group_overlap_ms": sum(groups.values()) - active / 1000,
                       "target_annotations": current["native_annotations"],
                       "draft_graphs": sum(b["role"] == "draft" and start <= b["start_us"] < stop
                                           for b in blocks)})
    # Retain all cycles; summarize central cycles to avoid initial CUPTI overhead and drain.
    central = [c for c in cycles[2:-1] if not native_calls or c["native_cpu_ordinal_gap"] == 1]
    if not central:
        raise ValueError("Too few complete target cycles for a warmed trace comparison")
    groups = sorted({name for row in central for name in row["group_union_ms"]})
    summary = {"cycles": len(central), **{
        "median_" + key: statistics.median(c[key] for c in central)
        for key in ("span_ms", "target_graph_ms", "gpu_kernel_union_ms", "no_traced_kernel_ms",
                    "cross_group_overlap_ms")},
        "median_group_union_ms": {name: statistics.median(c["group_union_ms"].get(name, 0.)
                                                         for c in central) for name in groups},
        "draft_graph_counts": dict(collections.Counter(c["draft_graphs"] for c in central))}
    return {"worker_trace": True, "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "kernel_count": len(kernels), "graph_count": len(graphs),
            "native_cpu_generation_annotations": len(native_calls),
            "target_graph_native_cpu_ordinals": [b["native_cpu_ordinal"] for b in targets],
            "adjacent_graph_native_ordinal_gaps": dict(collections.Counter(
                str(c["native_cpu_ordinal_gap"]) for c in cycles)),
            "graph_role_counts": dict(collections.Counter(roles.values())),
            "attribution_evidence": dict(evidence), "central_summary": summary,
            "graphs": blocks, "all_cycles": cycles,
            "top_kernels": sorted(top.values(), key=lambda r: r["sum_ms"], reverse=True)[:40],
            "limitations": [
                "Profiled spans are not ordinary request throughput or latency.",
                ("Central summary omits the first two and final cycle, and any pair that skips "
                 "a native CPU generation annotation; all cycles and coverage gaps are retained."),
                ("Target graphs require 84 expert GEMMs. Drafter labels require four grouped "
                 "convolutions per layer: legacy 5/6 rotary calls, or native 5/6 unified attention "
                 "calls with one extra rotary call and a selector walk. No expert GEMMs allowed."),
                "Other graph types are retained as unknown; a single graph is not a whole cycle.",
                "Kernel unions avoid overlap; group medians need not sum to the median span.",
                ("No-traced-kernel time is not proof of a CPU bottleneck: waits/copies/untraced "
                 "work and profiler overhead can contribute."),
                ("Head attribution uses observed production kernel names or CPU GEMM shapes; "
                 "unmatched head work may remain in other_compute."),
                ("GPU annotation fragments with a unique shared CPU invocation name are "
                 "enveloped to locate graphs. This is not an estimate of active GPU time."),
                ("A contiguous captured prefix can be summarized even when later native "
                 "generation annotations have no captured target graph. Coverage is explicit."),
                "Different outputs, routing, draft widths and runtime policies remain confounders.",
            ]}


def main() -> None:
    """Write a reproducible inventory from explicit saved trace paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "profiles": [inspect_trace(p) for p in args.traces]}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for row in result["profiles"]:
        print(json.dumps({"path": row["path"], "summary": row.get("central_summary")}))


if __name__ == "__main__":
    main()
