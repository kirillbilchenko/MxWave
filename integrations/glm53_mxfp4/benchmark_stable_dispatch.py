"""Qualify stable dispatch against the CPU ordering contract on real and synthetic metadata."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from analyze_repeat_model import read_tensor
from dispatch_order_probe import canonical_order
from stable_dispatch import canonicalize
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size


def expected(metadata: tuple[torch.Tensor, ...], block: int) -> torch.Tensor:
    """Build the CPU reference while checking every consumed token/expert membership."""
    tokens, experts, padded = metadata
    result = canonical_order(tokens.cpu().tolist(), experts.cpu().tolist(),
                             int(padded.item()), block)
    return torch.tensor(result, dtype=torch.int32, device=tokens.device)


def main() -> None:
    """Gate bytes, padding, streams, graphs, allocation and bounded kernel latency."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Preserve prior dispatch qualification")
    torch.set_num_threads(1)
    torch.manual_seed(83417)
    device = torch.device("cuda", 0)
    props = torch.cuda.get_device_properties(device)
    torch.cuda.set_per_process_memory_fraction(1024**3 / props.total_memory)
    report: dict[str, Any] = {"status": "running", "cases": [], "actual_cases": [],
        "started_epoch": time.time(), "source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("stable_dispatch.py", "stable_dispatch_gpu.py",
                         "benchmark_stable_dispatch.py", "dispatch_order_probe.py")},
        "limits": ["GPU metadata component, not full-model quality or throughput qualification.",
                   "Ascending TP-only expert bins with unique top-k experts per token."]}

    def save() -> None:
        p = args.report.with_suffix(".incomplete")
        p.write_text(json.dumps(report, indent=2) + "\n")
        p.replace(args.report)

    def check(metadata: tuple[torch.Tensor, ...], tokens: int, block: int) -> dict[str, Any]:
        wanted = expected(metadata, block)
        original, owners, padded = (v.clone() for v in metadata)
        ids, experts, count = metadata
        canonicalize(*metadata, num_tokens=tokens, num_experts=288, block=block)
        if not torch.equal(ids, wanted) or not torch.equal(experts, owners) \
                or not torch.equal(count, padded):
            raise RuntimeError("GPU canonical order differs or changes expert/padding metadata")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            ids.copy_(original)
            canonicalize(*metadata, num_tokens=tokens, num_experts=288, block=block)
        torch.cuda.current_stream().wait_stream(stream)
        if not torch.equal(ids, wanted):
            raise RuntimeError("Non-default-stream canonicalization changed bytes")
        with torch.cuda.stream(stream):
            ids.copy_(original)
            canonicalize(*metadata, num_tokens=tokens, num_experts=288, block=block)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            ids.copy_(original)
            canonicalize(*metadata, num_tokens=tokens, num_experts=288, block=block)
        before = torch.cuda.memory_allocated()
        for _ in range(5):
            ids.fill_(-1)
            graph.replay()
        torch.cuda.synchronize()
        growth = torch.cuda.memory_allocated() - before
        if growth or not torch.equal(ids, wanted):
            raise RuntimeError("Graph replay changed bytes or allocated persistent memory")
        timings = []
        for _ in range(10):
            canonicalize(*metadata, num_tokens=tokens, num_experts=288, block=block)
        for _ in range(20):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            canonicalize(*metadata, num_tokens=tokens, num_experts=288, block=block)
            end.record()
            end.synchronize()
            timings.append(start.elapsed_time(end) * 1000)
        graph_times = []
        for _ in range(20):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            graph_times.append(start.elapsed_time(end) * 1000)
        del graph
        return {"bits_exact": True, "expert_ids_and_padding_unchanged": True,
                "nondefault_stream": True, "graph_replay": True,
                "graph_allocation_growth_bytes": growth, "tokens": tokens, "block": block,
                "eager_event_median_us": statistics.median(timings),
                "graph_copy_plus_sort_median_us": statistics.median(graph_times),
                "eager_event_includes_cpu_enqueue_delay": True,
                "one_token_kernel_skipped": tokens <= 1}

    save()
    try:
        for label in ("default-0", "default-2"):
            root = args.traces / f"repeat-{label}-rank1"
            manifest = json.loads((root / "snapshots.json").read_bytes())
            event = next(e for e in manifest["events"]
                         if e["name"] == "language_model.model.layers.4.mlp/gemm3/input")
            items = {v["path"]: v for v in event["tensors"]}
            metadata = []
            for path in ("/11", "/12", "/13"):
                raw, _ = read_tensor(root, items[path])
                metadata.append(torch.frombuffer(bytearray(raw), dtype=torch.int32).to(device))
            result = check(tuple(metadata), 258, 8)
            result["capture"] = label
            report["actual_cases"].append(result)
            save()
            if label == "default-0":
                try:
                    with torch.profiler.profile(activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]) as profiler:
                        for _ in range(8):
                            canonicalize(*metadata, num_tokens=258, num_experts=288, block=8)
                        torch.cuda.synchronize()
                    trace = args.report.parent / "stable-dispatch-torch-trace.json"
                    profiler.export_chrome_trace(str(trace))
                    report["profile"] = {"file": trace.name, "sha256": hashlib.sha256(
                        trace.read_bytes()).hexdigest()}
                except RuntimeError as exc:
                    report["profile"] = {"error": repr(exc)}
                save()
        for tokens in (1, 2, 8, 32, 128, 258, 512, 1025, 2048):
            for pattern in ("balanced", "concentrated"):
                if pattern == "balanced":
                    topk = torch.topk(torch.rand((tokens, 288), device=device), 8, dim=1).indices
                else:
                    topk = torch.arange(8, device=device).expand(tokens, -1).contiguous()
                topk = topk.to(torch.int32)
                for block in (8, 16, 32, 48, 64):
                    metadata = moe_align_block_size(topk, block, 288, None,
                                                     ignore_invalid_experts=True)
                    result = check(metadata, tokens, block)
                    result["pattern"] = pattern
                    report["cases"].append(result)
                    save()
                del metadata, topk
                gc.collect()
                torch.cuda.empty_cache()
        report.update(status="passed", finished_epoch=time.time(),
                      cuda_peak_bytes=torch.cuda.max_memory_allocated())
        save()
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        save()
        raise


if __name__ == "__main__":
    main()
