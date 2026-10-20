"""Analyze retained full-model metrics and streaming-read both-rank torch traces."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TextIO

import numpy as np


def trace_events(stream: TextIO) -> Iterator[dict[str, Any]]:
    """Decode one trace event at a time with a bounded buffer, excluding the root object."""
    buffer = ""
    while '"traceEvents"' not in buffer:
        chunk = stream.read(65536)
        if not chunk or len(buffer) > 1024**2:
            raise ValueError("Trace events array is missing")
        buffer += chunk
    buffer = buffer.split('"traceEvents"', 1)[1]
    while "[" not in buffer:
        chunk = stream.read(65536)
        if not chunk:
            raise ValueError("Truncated trace events array")
        buffer += chunk
    buffer = buffer.split("[", 1)[1]
    decoder = json.JSONDecoder()
    while True:
        buffer = buffer.lstrip(" \n\r\t,")
        if buffer.startswith("]"):
            return
        if buffer:
            try:
                event, stop = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                pass
            else:
                if not isinstance(event, dict):
                    raise TypeError("A trace event is not a JSON object")
                yield event
                buffer = buffer[stop:]
                continue
        if len(buffer) > 32 * 1024**2:
            raise ValueError("One trace event exceeds the declared 32 MiB parsing bound")
        chunk = stream.read(65536)
        if not chunk:
            raise ValueError("Truncated trace event")
        buffer += chunk


def category(name: str) -> str:
    """Group obvious kernel families; retain original names for more precise interpretation."""
    lowered = name.lower()
    if any(key in lowered for key in ("nccl", "roce", "allreduce", "all_reduce", "allgather")):
        return "communication"
    if "marlin" in lowered:
        return "marlin_gemm"
    if any(key in lowered for key in ("kda", "mla", "attn", "attention", "flash")):
        return "attention"
    if any(key in lowered for key in ("moe_align", "topk", "silu", "router", "moe_sum")):
        return "routing_activation"
    if any(key in lowered for key in ("conv", "mamba", "gated_delta")):
        return "recurrent"
    if any(key in lowered for key in ("gemm", "gemv", "matmul")):
        return "other_gemm"
    return "other"


def profile(path: Path) -> dict[str, Any]:
    """Summarize GPU kernel time without confusing overlapping work with elapsed time."""
    counts: dict[str, dict[str, float]] = {}
    groups: dict[str, float] = defaultdict(float)
    intervals: dict[int, list[tuple[float, float]]] = defaultdict(list)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for event in trace_events(stream):
            if event.get("cat") != "kernel" or event.get("ph") != "X":
                continue
            name, duration = event["name"], float(event["dur"])
            row = counts.setdefault(name, {"calls": 0, "sum_duration_us": 0})
            row["calls"] += 1
            row["sum_duration_us"] += duration
            groups[category(name)] += duration
            device = int(event.get("args", {}).get("device", event.get("pid", 0)))
            start = float(event["ts"])
            intervals[device].append((start, start + duration))
    utilization = []
    for device, values in intervals.items():
        values.sort()
        first = last = values[0][0]
        covered = 0.0
        for start, stop in values:
            covered += max(0, stop - max(start, last))
            last = max(last, stop)
        utilization.append({"device": device, "span_first_last_kernel_us": last - first,
                            "kernel_union_us": covered,
                            "uncovered_us_including_between_request_gaps": last - first - covered})
    return {"path": str(path), "bytes": path.stat().st_size,
            "kernel_count": sum(x["calls"] for x in counts.values()),
            "kernel_sum_duration_us": sum(groups.values()), "kernel_groups_us": dict(groups),
            "device_spans": utilization,
            "kernels": sorted(({"name": name, **row} for name, row in counts.items()),
                              key=lambda row: row["sum_duration_us"], reverse=True),
            "limitations": ["Kernel durations can overlap and do not add to request wall time.",
                            "Uncovered spans include client gaps between profiled requests.",
                            "Name-based groups retain original symbols for later inspection."]}


def quality(reference: dict[str, Any], candidate: dict[str, Any], root: Path) -> dict[str, Any]:
    """Compare normalized full vocabulary distributions, token IDs and teacher-forced NLL."""
    rows = []
    if reference["vocabulary"] != candidate["vocabulary"]:
        raise ValueError("Quality vocabularies differ")
    for left, right in zip(reference["contexts"], candidate["contexts"], strict=True):
        data = []
        for item in (left, right):
            raw = (root / item["next_logprobs_file"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != item["sha256"]:
                raise ValueError("A retained full vocabulary distribution changed")
            values = np.frombuffer(raw, dtype="<f4")
            if len(values) != reference["vocabulary"] or np.isnan(values).any():
                raise ValueError("Invalid full vocabulary distribution")
            data.append(values)
        normalized = []
        for values in data:
            values = values.astype(np.float64)
            maximum = values.max()
            normalized.append(values - maximum - np.log(np.exp(values - maximum).sum()))
        p_log, q_log = normalized
        active = np.isfinite(p_log)
        kl = float((np.exp(p_log[active]) * (p_log[active] - q_log[active])).sum())
        rows.append({"index": left["index"], "kl": max(0, kl) if math.isfinite(kl) else None,
                     "next_logprobs_bits_exact": bool(np.array_equal(
                         data[0].view(np.uint32), data[1].view(np.uint32))),
                     "greedy_tokens_exact": left["next_token_ids"] == right["next_token_ids"],
                     "nll_difference": right["nll"] - left["nll"]})
    finite = all(row["kl"] is not None for row in rows)
    return {"contexts": rows, "mean_kl": float(np.mean([row["kl"] for row in rows]))
            if finite else None, "all_logprobs_bits_exact": all(
                row["next_logprobs_bits_exact"] for row in rows),
            "all_greedy_tokens_exact": all(row["greedy_tokens_exact"] for row in rows),
            "mean_nll_difference": candidate["mean_nll"] - reference["mean_nll"]}


def quality_gate(repeat: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Reject an unstable control before deriving any candidate tolerance from its noise."""
    repeat_kl, repeat_nll = repeat["mean_kl"], abs(repeat["mean_nll_difference"])
    stable = repeat_kl is not None and repeat_kl <= 1e-5 and repeat_nll <= 1e-4
    kl_limit = max(1e-7, 5 * (repeat_kl or 0)) if stable else None
    nll_limit = max(1e-5, 5 * repeat_nll) if stable else None
    passed = (stable and candidate["mean_kl"] is not None
              and candidate["mean_kl"] <= kl_limit
              and abs(candidate["mean_nll_difference"]) <= nll_limit
              and (candidate["all_greedy_tokens_exact"] or not repeat["all_greedy_tokens_exact"]))
    status = ("passed" if passed else "failed") if stable else "unqualified_unstable_control"
    return {"status": status,
            "raw_repeat_kl_limit": 1e-5, "raw_repeat_absolute_nll_difference_limit": 1e-4,
            "raw_repeat_stable": stable, "mean_kl_limit": kl_limit,
            "absolute_mean_nll_difference_limit": nll_limit, "passed": bool(passed)}


def main() -> None:
    """Produce a compact, reproducible result after the live model has been restored."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    results = root / "results"
    arms = {name: json.loads((results / f"{name}-benchmark.json").read_bytes())
            for name in ("stock", "raw", "aligned")}
    hashes = {arm["protocol_sha256"] for arm in arms.values()}
    if len(hashes) != 1 or any(arm["status"] != "complete" for arm in arms.values()):
        raise ValueError("Full-model arms are incomplete or used different frozen inputs")
    repeat = quality(arms["raw"]["quality"], arms["raw"]["quality_repeat"], results)
    if repeat["mean_kl"] is None:
        raise ValueError("The raw repeat control has a non-finite divergence")
    candidate = quality(arms["raw"]["quality"], arms["aligned"]["quality"], results)
    stock = quality(arms["stock"]["quality"], arms["raw"]["quality"], results)
    candidate["gates"] = quality_gate(repeat, candidate)
    timing = []
    keyed = {name: {(case["batch"], case["prompt_tokens"]): case for case in arm["cases"]}
             for name, arm in arms.items()}
    for key, raw in keyed["raw"].items():
        rates = {name: keyed[name][key]["median_aggregate_tokens_per_second"] for name in arms}
        timing.append({"batch": key[0], "prompt_tokens": key[1], "tokens_per_second": rates,
            "aligned_over_raw": rates["aligned"] / rates["raw"],
            "raw_over_stock": rates["raw"] / rates["stock"],
            "aligned_over_stock": rates["aligned"] / rates["stock"],
            "ttft_seconds": {name: keyed[name][key]["median_ttft_seconds"] for name in arms}})
    allocation = {name: arm["allocation_after"]["results"] for name, arm in arms.items()}
    saving = []
    for rank, (raw, packed) in enumerate(zip(allocation["raw"], allocation["aligned"], strict=True)):
        if raw["layers_seen"] != 42 or packed["layers_seen"] != 42:
            raise ValueError("The full-model scale adapter missed an eligible routed layer")
        if not all(scale.get("gpu_bytes_exact", False) for layer in packed["layers"]
                   for scale in layer["scales"] if scale["compressed"]):
            raise ValueError("A compressed scale tensor is missing the exact GPU decode proof")
        saving.append({"rank": rank,
            "cuda_allocated_saved_bytes": raw["cuda_allocated_bytes"] - packed["cuda_allocated_bytes"],
            "scale_payload_saved_bytes": raw["scale_retained_bytes"] - packed["scale_retained_bytes"],
            "raw_cuda_allocated_bytes": raw["cuda_allocated_bytes"],
            "aligned_cuda_allocated_bytes": packed["cuda_allocated_bytes"],
            "raw_scratch_bytes": raw["scratch_bytes"], "aligned_scratch_bytes": packed["scratch_bytes"]})
    traces = []
    for path in sorted(root.rglob("*.pt.trace.json*")):
        trace = profile(path)
        trace["path"] = str(path.relative_to(root))
        traces.append(trace)
    trace_groups = {f"{mode}/{directory}": any(trace["kernel_count"] and
                    trace["path"].startswith(f"{mode}/{directory}/") for trace in traces)
                    for mode in ("raw", "aligned") for directory in ("traces", "traces-worker")}
    value = {"status": "complete", "protocol_sha256": next(iter(hashes)), "timing": timing,
        "allocation": allocation, "saving": saving,
        "quality": {"raw_repeat": repeat, "raw_to_aligned": candidate, "stock_to_raw": stock},
        "profiles": traces, "profile_gpu_rank_groups": trace_groups,
        "gates": {"quality": candidate["gates"]["passed"],
                  "all_cases_throughput_within_2_percent_of_raw": all(
                      row["aligned_over_raw"] >= .98 for row in timing),
                  "all_cases_total_throughput_within_2_percent_of_stock": all(
                      row["aligned_over_stock"] >= .98 for row in timing),
                  "all_cases_raw_adapter_throughput_within_2_percent_of_stock": all(
                      row["raw_over_stock"] >= .98 for row in timing),
                  "both_ranks_profiled_in_both_arms": all(trace_groups.values())},
        "limitations": ["One imported AutoRound checkpoint; not BF16 or NVFP4 quality qualification.",
                        "An unstable repeat control invalidates the quality comparison.",
                        "Diagnostic authored contexts; not a standard perplexity benchmark.",
                        "TP2 W4A16 Marlin, no DFlash/MTP and fixed four-GiB KV per rank.",
                        "Only expert E8M0 scales are compressed; four-bit codes are unchanged.",
                        "Three sequential arms, five trials per case; not an interleaved thermal control."]}
    (results / "summary.json").write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    lines = ["# GLM full-model exact-scale pilot", "", ("| Prompt | Batch | Stock tok/s | "
             "Raw adapter tok/s | Compressed tok/s | Compressed / raw |"), "|---|---|---|---|---|---|"]
    for row in timing:
        rates = row["tokens_per_second"]
        lines.append(f"| {row['prompt_tokens']} | {row['batch']} | {rates['stock']:.2f} | "
                     f"{rates['raw']:.2f} | {rates['aligned']:.2f} | {row['aligned_over_raw']:.4f} |")
    lines.extend(["", "Exact allocations and teacher-forced/full-vocabulary quality are in summary.json.",
                  f"Quality gate passed: {candidate['gates']['passed']}.", "",
                  "Limitations:", "", *[f"- {note}" for note in value["limitations"]], ""])
    (results / "overview.md").write_text("\n".join(lines))
    print(json.dumps({"status": "complete", "quality_passed": candidate["gates"]["passed"],
                      "profile_files": len(traces), "saving": saving}))


if __name__ == "__main__":
    main()
