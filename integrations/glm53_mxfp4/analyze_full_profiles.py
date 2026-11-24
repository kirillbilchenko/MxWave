"""Attribute saved trial stalls to JIT logs and compare matched annotated GPU work.

This is offline analysis. It neither starts an engine nor changes the live service.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import datetime as dt
import gzip
import json
import re
import statistics
from pathlib import Path
from typing import Any

from compare_full_model import trace_events

FORWARD = re.compile(r"execute_(\d+)_context_(\d+).*_generation_(\d+)")
COMPILE = re.compile(
    r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d).*TileLang (begins|completes).*kernel `([^`]+)`"
)


def kernel_group(name: str) -> str:
    """Separate routed FP4 GEMMs from the unchanged generic Marlin linears."""
    lowered = name.lower()
    if "_row_gather_kernel" in lowered:
        return "embedding_lookup"
    if "bf16triplebitmap_decompress_kernel" in lowered:
        return "bf16_weight_decode"
    if "bf16triplebitmap_mm_kernel" in lowered:
        return "bf16_compressed_linear"
    if "splitk_reduction_bf16" in lowered:
        return "bf16_compressed_reduction"
    if ("nvjet_sm121" in lowered
            or "internal::gemvx" in lowered and "__nv_bfloat16" in lowered):
        return "bf16_dense_linear"
    if "mxwave_grouped_" in name or "marlin_moe_wna16::Marlin" in name:
        return "routed_expert_gemm"
    if "marlin::Marlin" in name:
        return "generic_marlin_linear"
    if any(key in lowered for key in ("nccl", "roce", "allreduce", "all_reduce", "allgather")):
        return "communication"
    if "mhc" in lowered or "hc_prenorm" in lowered:
        return "mhc"
    if any(key in lowered for key in ("kda", "mla", "attn", "attention", "flash")):
        return "attention"
    if any(key in lowered for key in ("moe_align", "topk", "silu", "router", "moe_sum")):
        return "routing_activation"
    if any(key in lowered for key in ("conv", "mamba", "gated_delta")):
        return "recurrent"
    if any(key in lowered for key in ("gemm", "gemv", "matmul")):
        return "other_gemm"
    return "other"


def gpu_steps(path: Path) -> dict[str, Any]:
    """Assign kernels to GPU annotations on the same stream, retaining unmatched work."""
    ranges: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    kernels = []
    launches: collections.Counter[str] = collections.Counter()
    with gzip.open(path, "rt") as stream:
        for event in trace_events(stream):
            if event.get("cat") == "gpu_user_annotation" and FORWARD.match(event["name"]):
                ranges[event["tid"]].append(event)
            elif event.get("cat") == "kernel":
                kernels.append(event)
            elif event.get("cat") == "cuda_runtime" and event["name"] in (
                "cudaGraphLaunch", "cudaLaunchKernel",
            ):
                launches[event["name"]] += 1
    indexes = {}
    for stream_id, events in ranges.items():
        events.sort(key=lambda event: event["ts"])
        indexes[stream_id] = [event["ts"] for event in events]
    steps: dict[tuple[int, int], dict[str, Any]] = {}
    totals: dict[str, float] = collections.defaultdict(float)
    covered_intervals: dict[tuple[int, int], list[tuple[float, float]]] = (
        collections.defaultdict(list))
    unmatched = 0
    for event in kernels:
        group = kernel_group(event["name"])
        totals[group] += event["dur"]
        stream_id = event["args"].get("stream", event["tid"])
        position = bisect.bisect_right(indexes.get(stream_id, []), event["ts"]) - 1
        matched = None
        for i in range(position, max(-1, position - 8), -1):
            candidate = ranges[stream_id][i]
            if event["ts"] + event["dur"] <= candidate["ts"] + candidate["dur"] + 1:
                matched = (i, candidate)
                break
        if matched is None:
            unmatched += 1
            continue
        i, annotation = matched
        row = steps.setdefault((stream_id, i), {"name": annotation["name"],
            "stream": stream_id, "gpu_range_us": annotation["dur"],
            "kernel_us": {}, "kernel_calls": {}})
        row["kernel_us"][group] = row["kernel_us"].get(group, 0) + event["dur"]
        row["kernel_calls"][group] = row["kernel_calls"].get(group, 0) + 1
        covered_intervals[(stream_id, i)].append((event["ts"], event["ts"] + event["dur"]))
    for key, row in steps.items():
        covered, last = 0.0, float("-inf")
        for start, end in sorted(covered_intervals[key]):
            covered += max(0.0, end - max(start, last))
            last = max(last, end)
        row["kernel_union_us"] = covered
        row["uncovered_stream_us"] = max(0.0, row["gpu_range_us"] - covered)
    main_steps = sum(row["kernel_calls"].get("routed_expert_gemm") == 84
                     for row in steps.values())
    return {"path": str(path), "launches": dict(launches), "kernel_count": len(kernels),
            "complete_expert_forward_steps": main_steps,
            "unmatched_kernels": unmatched, "kernel_groups_us": dict(totals),
            "steps": list(steps.values())}


def trial_compiles(report: dict[str, Any], logs: dict[str, Any]) -> list[dict[str, Any]]:
    """Match sequential client trials to timestamped HTTP logs and embedded JIT times."""
    times = [dt.datetime.fromisoformat(text).timestamp()
             for text in logs["completion_access_timestamps"]]
    compiles = []
    pending: dict[str, float] = {}
    for line in logs["tilelang_events"]:
        match = COMPILE.search(line)
        if match is None:
            continue
        timestamp = dt.datetime.strptime(match[1], "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=dt.UTC).timestamp()
        kind, name = match[2], match[3]
        if kind == "begins":
            pending[name] = timestamp
        elif name in pending:
            begin = pending.pop(name)
            compiles.append({"start_epoch": begin, "end_epoch": timestamp,
                             "kernel": name, "seconds_at_log_precision": timestamp - begin})
    offset = 0
    rows = []
    for case in report["cases"]:
        batch = case["batch"]
        offset += 2 * batch  # Frozen protocol: two warm-up groups per case.
        for index, trial in enumerate(case["trials"]):
            access = times[offset:offset + batch]
            if len(access) != batch:
                raise ValueError("HTTP access log is incomplete for the frozen timing protocol")
            offset += batch
            begin, end = min(access), min(access) + trial["wall_seconds"]
            overlaps = [event for event in compiles
                        if event["start_epoch"] < end and event["end_epoch"] > begin]
            rows.append({"batch": batch, "prompt_tokens": case["prompt_tokens"], "trial": index,
                "http_response_start_epoch": begin, "wall_seconds": trial["wall_seconds"],
                "tokens_per_second": trial["aggregate_tokens_per_second"],
                "ttft_seconds": [x["ttft_seconds"] for x in trial["requests"]],
                "tilelang_compiles": overlaps})
    return rows


def main() -> None:
    """Produce a compact analysis with retained cold samples and clearly labelled limits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--jit-logs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, logs = args.root, json.loads(args.jit_logs.read_bytes())
    profiles = {}
    for mode in ("raw", "aligned"):
        for location, rank in (("traces", 0), ("traces-worker", 1)):
            paths = list((root / mode / location).glob("dp*.gz"))
            if len(paths) != 1:
                raise ValueError("Expected exactly one both-request GPU profile per rank")
            profile = gpu_steps(paths[0])
            profile["path"] = str(paths[0].relative_to(root))
            profiles[f"{mode}_rank{rank}"] = profile
    pairs = []
    for rank in (0, 1):
        # The name includes the exact context length, request count and token
        # count. Compare identical one-request short-context decode steps.
        selected = {}
        for mode in ("raw", "aligned"):
            selected[mode] = {
                row["name"]: row for row in profiles[f"{mode}_rank{rank}"]["steps"]
                if row["name"].startswith("execute_1_context_0")
                and row["kernel_calls"].get("routed_expert_gemm") == 84
                and re.search(r"generation_1\(sq1sk(?:25[7-9]|26\d|27[01])\D", row["name"])
            }
        for name in sorted(selected["raw"].keys() & selected["aligned"].keys()):
            raw, packed = selected["raw"][name], selected["aligned"][name]
            pairs.append({"rank": rank, "annotation": name,
                "raw_expert_us": raw["kernel_us"]["routed_expert_gemm"],
                "aligned_expert_us": packed["kernel_us"]["routed_expert_gemm"],
                "raw_gpu_range_us": raw["gpu_range_us"],
                "aligned_gpu_range_us": packed["gpu_range_us"],
                "raw_kernel_groups_us": raw["kernel_us"],
                "aligned_kernel_groups_us": packed["kernel_us"]})
    matched = []
    for rank in (0, 1):
        rows = [row for row in pairs if row["rank"] == rank]
        if not rows:
            raise ValueError("No matched one-request GPU steps with complete expert coverage")
        matched.append({"rank": rank, "steps": len(rows),
            "raw_median_expert_us": statistics.median(row["raw_expert_us"] for row in rows),
            "aligned_median_expert_us": statistics.median(row["aligned_expert_us"] for row in rows),
            "median_paired_expert_duration_change_percent": 100 * statistics.median(
                row["aligned_expert_us"] / row["raw_expert_us"] - 1 for row in rows),
            "raw_median_gpu_range_us": statistics.median(row["raw_gpu_range_us"] for row in rows),
            "aligned_median_gpu_range_us": statistics.median(
                row["aligned_gpu_range_us"] for row in rows),
            "median_kernel_groups_us": {mode: {
                group: statistics.median(row[f"{mode}_kernel_groups_us"].get(group, 0)
                                         for row in rows)
                for group in profiles[f"{mode}_rank{rank}"]["kernel_groups_us"]}
                for mode in ("raw", "aligned")}})
    trials = {}
    for mode in ("stock", "raw", "aligned"):
        report = json.loads((root / "results" / f"{mode}-benchmark.json").read_bytes())
        trials[mode] = trial_compiles(report, logs[mode])
    output = {"status": "complete", "profiles": profiles, "matched_b1_steps": pairs,
        "matched_b1_summary": matched, "trial_compilation_overlaps": trials,
        "limitations": ["Profiles cover B1/256 and concurrent B4/2048, not slow B4/256 trials.",
                        "Slow B4/256 attribution uses timestamped HTTP/JIT logs and saved arrivals.",
                        "HTTP access timestamps approximate response start, not GPU execution start.",
                        "Embedded compile timestamps have one-second precision.",
                        "Kernel sums may overlap and are not elapsed request time.",
                        "Concurrent profiles have different forward-step counts; compare matched steps.",
                        "Matched steps share shapes, not verified identical routed activations.",
                        "Repeated-output controls fail; performance describes an unqualified workload.",
                        "No cold samples were removed from the original benchmark medians."]}
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"matched_b1": matched, "batch4_256_compiles": {
        mode: [{k: row[k] for k in ("trial", "tokens_per_second", "tilelang_compiles")}
               for row in values if row["batch"] == 4 and row["prompt_tokens"] == 256]
        for mode, values in trials.items()}}))


if __name__ == "__main__":
    main()
