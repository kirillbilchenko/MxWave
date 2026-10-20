"""Attribute retained device kernels to actual model forwards through CUDA correlation IDs."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

FORWARD = re.compile(r"^execute_(\d+)_context_(\d+)\([^)]*\)_generation_(\d+)\(")
DECODER = "BF16TripleBitmap_Decompress_Kernel<"


def attribute(trace: dict[str, Any]) -> dict[str, Any]:
    """Follow launch correlations; keep unmatched kernels explicit instead of guessing by GPU time."""
    events = trace["traceEvents"]
    forwards = []
    for event in events:
        match = FORWARD.match(event.get("name", ""))
        if event.get("cat") != "user_annotation" or not match:
            continue
        forwards.append({"name": event["name"], "tokens": int(match[1]),
            "context_requests": int(match[2]), "generation_requests": int(match[3]),
            "pid": event["pid"], "tid": event["tid"], "start_us": event["ts"],
            "cpu_annotation_duration_us": event["dur"], "cuda_graph_launches": 0,
            "kernel_events": 0, "kernel_duration_sum_us": 0.,
            "decompression_kernel_events": 0, "decompression_duration_sum_us": 0.,
            "graph_kernel_events": 0, "graph_decompression_kernel_events": 0,
            "graph_kernel_first_us": None, "graph_kernel_last_end_us": None})
    forwards.sort(key=lambda row: row["start_us"])
    runtime = {}
    for event in events:
        if event.get("cat") not in ("cuda_runtime", "cuda_driver"):
            continue
        correlation = event.get("args", {}).get("correlation")
        if correlation is None:
            continue
        if correlation in runtime:
            raise ValueError("Duplicate CUDA runtime correlation makes ownership ambiguous")
        owners = [i for i, row in enumerate(forwards) if
            (row["pid"], row["tid"]) == (event["pid"], event["tid"])
            and row["start_us"] <= event["ts"] <=
                row["start_us"] + row["cpu_annotation_duration_us"]]
        if len(owners) > 1:
            raise ValueError("Overlapping model annotations make launch ownership ambiguous")
        owner = owners[0] if owners else None
        runtime[correlation] = owner
        if owner is not None and event.get("name", "").startswith("cudaGraphLaunch"):
            forwards[owner]["cuda_graph_launches"] += 1
    unassigned = Counter()
    total_kernels, total_decoders = 0, 0
    for event in events:
        if event.get("cat") != "kernel":
            continue
        total_kernels += 1
        is_decoder = DECODER in event.get("name", "")
        total_decoders += is_decoder
        owner = runtime.get(event.get("args", {}).get("correlation"))
        if owner is None:
            unassigned["kernel_events"] += 1
            unassigned["decompression_kernel_events"] += is_decoder
            continue
        row = forwards[owner]
        duration = float(event.get("dur", 0.))
        row["kernel_events"] += 1
        row["kernel_duration_sum_us"] += duration
        row["decompression_kernel_events"] += is_decoder
        row["decompression_duration_sum_us"] += duration if is_decoder else 0.
        if event.get("args", {}).get("graph id", 0):
            row["graph_kernel_events"] += 1
            row["graph_decompression_kernel_events"] += is_decoder
            first = row["graph_kernel_first_us"]
            last = row["graph_kernel_last_end_us"]
            row["graph_kernel_first_us"] = min(first, event["ts"]) if first is not None else event["ts"]
            end = event["ts"] + duration
            row["graph_kernel_last_end_us"] = max(last, end) if last is not None else end
    for row in forwards:
        first, last = row["graph_kernel_first_us"], row["graph_kernel_last_end_us"]
        row["graph_kernel_span_us"] = last - first if first is not None else None
        row["pure_decode"] = row["context_requests"] == 0 and row["generation_requests"] > 0
    if sum(row["kernel_events"] for row in forwards) + unassigned["kernel_events"] != total_kernels:
        raise ValueError("Kernel attribution dropped events")
    if sum(row["decompression_kernel_events"] for row in forwards) + unassigned[
            "decompression_kernel_events"] != total_decoders:
        raise ValueError("Kernel attribution dropped decompression events")
    return {"forwards": forwards, "kernel_events": total_kernels,
        "decompression_kernel_events": total_decoders, "unassigned": dict(unassigned),
        "ownership_method": "Device correlation ID -> CUDA runtime/driver call -> containing CPU model annotation, same PID/TID",
        "limits": ["Only the profiler's retained active window is analyzed; later request forwards may be absent.",
            "Kernel duration sums are not wall time, measured memory traffic or a TP critical path.",
            "A graph kernel span covers its first-to-last kernel, not the complete request or all copies.",
            "Unassigned kernels remain explicit; no nearest-time attribution or missing-event extrapolation."]}


def main() -> None:
    """Read only inventoried both-rank worker traces from the restored experiment archive."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.output.exists():
        raise FileExistsError("Preserve earlier device correlation analysis")
    inventory = json.loads((root / "evidence-inventory.json").read_bytes())["files"]
    rows = []
    for item in inventory:
        name = item["path"]
        if not name.startswith("traces/") or not name.endswith((".pt.trace.json.gz", ".pt.trace.json")):
            continue
        if not re.match(r"dp0_pp0_tp[01]_", Path(name).name):
            continue
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Trace inventory escapes its evidence directory")
        path = root / name
        with path.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if path.is_symlink() or actual != item["sha256"]:
            raise ValueError("Inventoried worker trace changed")
        with (gzip.open(path, "rt") if name.endswith(".gz") else path.open()) as handle:
            trace = json.load(handle)
        rows.append({"path": name, "sha256": actual, **attribute(trace)})
        del trace
    result = {"status": "complete", "worker_traces": rows,
        "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "worker_traces": len(rows)}))


if __name__ == "__main__":
    main()
