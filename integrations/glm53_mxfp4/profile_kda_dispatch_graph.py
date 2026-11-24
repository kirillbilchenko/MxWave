"""Diagnose a graph timing discrepancy using identical-control graphs and saved GPU activities."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import statistics
import time
from functools import partial
from pathlib import Path
from typing import Any

import torch
from benchmark_full_model import write
from benchmark_kda_direct_dispatch import require_equal
from kda_direct_dispatch import linear
from kda_runtime import kda_linear, validate_runtime
from prepare_bf16_attention import tensor_digest
from safetensors.torch import load_file


def clocks() -> dict[str, Any]:
    """Read available NVML clocks without changing device settings or timing GPU work."""
    values: dict[str, Any] = {}
    try:
        import pynvml
    except ImportError as exc:
        return {"unavailable": repr(exc)}
    try:
        pynvml.nvmlInit()
        device = pynvml.nvmlDeviceGetHandleByIndex(0)
        for name, kind in (("sm_mhz", pynvml.NVML_CLOCK_SM), ("memory_mhz", pynvml.NVML_CLOCK_MEM)):
            try:
                values[name] = pynvml.nvmlDeviceGetClockInfo(device, kind)
            except pynvml.NVMLError as exc:
                values[name] = str(exc)
        try:
            values["temperature_c"] = pynvml.nvmlDeviceGetTemperature(device, pynvml.NVML_TEMPERATURE_GPU)
            values["pstate"] = pynvml.nvmlDeviceGetPerformanceState(device)
        except pynvml.NVMLError as exc:
            values["other_error"] = str(exc)
        pynvml.nvmlShutdown()
    except pynvml.NVMLError as exc:
        values["unavailable"] = repr(exc)
    return values


def main() -> None:
    """Keep the failed v1 gate unchanged while collecting paired graph/null-control evidence."""
    from mxwave_zipserv import runtime

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Preserve the first graph diagnostic")
    validate_runtime()
    prepared = json.loads((args.fixtures / "prepared.json").read_bytes())
    cases = [c for c in prepared["cases"] if c["name"].startswith("kda-")]
    if len(cases) != 2:
        raise ValueError("Require the same two real KDA fixtures")
    for name, expected in (("config.json", prepared["source_config_sha256"]),
                           ("model.safetensors.index.json", prepared["source_index_sha256"])):
        if hashlib.sha256((args.source / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Source metadata changed")
    torch.set_num_threads(1)
    torch.manual_seed(20261017)
    props = torch.cuda.get_device_properties(0)
    torch.cuda.set_per_process_memory_fraction(1.5 * 1024**3 / props.total_memory)
    pool = torch.empty((2, max(c["native_shape"][0] * c["native_shape"][1] for c in cases)),
                       dtype=torch.bfloat16, device="cuda")
    stream = torch.cuda.Stream()
    report: dict[str, Any] = {
        "status": "running", "diagnostic_only": True, "cases": [],
        "v1_performance_qualified_remains": False, "started_epoch": time.time(),
        "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()
                          ).hexdigest() for name in ("profile_kda_dispatch_graph.py",
                          "kda_direct_dispatch.py", "kda_runtime.py")},
        "device": {name: getattr(props, name, None) for name in
                   ("name", "major", "minor", "L2_cache_size", "clock_rate", "memory_clock_rate")},
        "limits": ["Graph/null-control diagnostic only; preserves the failed v1 gate.",
                   "Clock snapshots bracket timing blocks, not individual GPU kernels.",
                   "Profiled GPU activities are separate from unprofiled timing samples."]}
    write(args.report, report)
    try:
        for case in cases:
            artifact = args.fixtures / case["artifact"]
            if hashlib.sha256(artifact.read_bytes()).hexdigest() != case["artifact_sha256"]:
                raise ValueError("Source-bound artifact changed")
            loaded = load_file(artifact)
            packed = runtime.PackedWeight(tuple(loaded[n].cuda() for n in runtime.BUFFER_NAMES),
                                          *case["native_metadata"])
            del loaded
            decoded = runtime.decompress(packed).cpu()
            if tensor_digest(decoded) != case["padded_weight_sha256"]:
                raise RuntimeError("Source weights do not restore bit-for-bit")
            del decoded
            rows, width = case["logical_shape"]
            x = torch.randn((4, width), dtype=torch.bfloat16, device="cuda") * .1
            metadata = (packed.m, packed.k, packed.max_common, packed.max_rare, packed.start_exp, rows)
            buffers = list(packed.buffers)
            operations = {"reference_a": partial(kda_linear, x, buffers, pool, *metadata),
                          "reference_b": partial(kda_linear, x, buffers, pool, *metadata),
                          "direct": partial(linear, x, buffers, pool, *metadata)}
            graphs, outputs, dots = {}, {}, {}
            for label, fn in operations.items():
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(20):
                        fn()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                try:
                    graph.enable_debug_mode()
                except (AttributeError, RuntimeError) as exc:
                    dots[label] = {"debug_unavailable": repr(exc)}
                with torch.cuda.graph(graph, stream=stream):
                    output = fn()
                graph.replay()
                torch.cuda.synchronize()
                path = args.report.parent / (case["name"] + "-" + label + ".dot")
                try:
                    graph.debug_dump(str(path))
                    dots[label] = {"file": path.name,
                                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                except (AttributeError, RuntimeError, OSError) as exc:
                    dots[label] = {"debug_unavailable": repr(exc)}
                graphs[label], outputs[label] = graph, output
            require_equal(outputs["reference_b"], outputs["reference_a"])
            require_equal(outputs["direct"], outputs["reference_a"])
            for _ in range(30):
                for graph in graphs.values():
                    graph.replay()
            torch.cuda.synchronize()
            row: dict[str, Any] = {"fixture": case["name"], "output_bits_exact": True,
                                   "graph_dumps": dots, "timing_blocks": []}
            report["cases"].append(row)
            for order in (list(graphs), list(reversed(graphs))):
                before = clocks()
                values: dict[str, list[float]] = {name: [] for name in graphs}
                for _ in range(40):
                    for label in order:
                        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        begin.record()
                        graphs[label].replay()
                        end.record()
                        end.synchronize()
                        values[label].append(begin.elapsed_time(end) * 1000)
                medians = {name: statistics.median(v) for name, v in values.items()}
                row["timing_blocks"].append({"order": order, "clocks_before": before,
                    "clocks_after": clocks(), "samples_us": values, "medians_us": medians,
                    "null_reference_b_a_ratio": medians["reference_b"] / medians["reference_a"],
                    "direct_reference_a_ratio": medians["direct"] / medians["reference_a"]})
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as profiler:
                for iteration in range(3):
                    for label, graph in graphs.items():
                        with torch.profiler.record_function("dispatch_graph/" + label + "/" + str(iteration)):
                            graph.replay()
                            torch.cuda.synchronize()
            trace = args.report.parent / (case["name"] + "-trace.json")
            profiler.export_chrome_trace(str(trace))
            with trace.open("rb") as source, gzip.open(str(trace) + ".gz", "wb") as target:
                while chunk := source.read(1024**2):
                    target.write(chunk)
            trace.unlink()
            compressed = Path(str(trace) + ".gz")
            row["trace"] = {"file": compressed.name,
                             "sha256": hashlib.sha256(compressed.read_bytes()).hexdigest()}
            write(args.report, report)
            del packed, buffers, x, operations, graphs, outputs, graph, output, fn
            gc.collect()
            torch.cuda.empty_cache()
        report.update(status="complete", correctness_passed=True,
                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      finished_epoch=time.time())
        write(args.report, report)
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        write(args.report, report)
        raise


if __name__ == "__main__":
    main()
