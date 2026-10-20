"""Gate a single-dispatch BF16 host path against the unchanged qualified KDA operation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import torch
from benchmark_full_model import write
from diagnose_bf16_overlap import capture
from kda_direct_dispatch import linear
from kda_runtime import kda_linear, validate_runtime
from prepare_bf16_attention import rebuild_weight, tensor_digest
from safetensors.torch import load_file


def require_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Compare output storage bits, including exceptional BF16 values and all shape axes."""
    if actual.shape != expected.shape or not torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    ):
        raise RuntimeError("Single-dispatch output differs from the qualified KDA path")


def paired(
    operations: dict[str, Callable[[], torch.Tensor]], *, reverse: bool
) -> dict[str, Any]:
    """Separate host enqueue and CUDA-event latency; synchronize outside each host measurement."""
    samples: dict[str, list[dict[str, float]]] = {name: [] for name in operations}
    names = list(operations)
    for index in range(30):
        order = names[::-1] if (index + reverse) % 2 else names
        for name in order:
            torch.cuda.synchronize()
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            started = time.perf_counter_ns()
            output = operations[name]()
            host_us = (time.perf_counter_ns() - started) / 1000
            end.record()
            end.synchronize()
            samples[name].append({"host_enqueue_us": host_us,
                                  "gpu_event_us": begin.elapsed_time(end) * 1000})
            del output
    medians = {name: {key: statistics.median(row[key] for row in values)
                      for key in ("host_enqueue_us", "gpu_event_us")}
               for name, values in samples.items()}
    return {"samples": samples, "medians": medians,
            "direct_reference_ratio": {
                key: medians["direct"][key] / medians["reference"][key]
                for key in ("host_enqueue_us", "gpu_event_us")
            }}


def graph_time(graph: Any) -> float:
    """Time warmed graph replay independently of eager Python submission."""
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    values = []
    for _ in range(20):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        graph.replay()
        end.record()
        end.synchronize()
        values.append(begin.elapsed_time(end) * 1000)
    return statistics.median(values)


def summarize_graph_samples(
    samples: dict[str, list[float]], order: list[str],
) -> dict[str, Any]:
    """Require stable identical controls before applying the unchanged 2% graph limit."""
    names = {"reference_a", "reference_b", "direct"}
    if set(samples) != names or set(order) != names or len(order) != 3:
        raise ValueError("Require exactly two identical controls and the direct path")
    lengths = {len(values) for values in samples.values()}
    if len(lengths) != 1 or next(iter(lengths)) < 2 or any(
        not math.isfinite(value) or value <= 0 for values in samples.values() for value in values
    ):
        raise ValueError("Require equal-length finite positive graph timing samples")
    medians = {name: statistics.median(values) for name, values in samples.items()}
    a, b = medians["reference_a"], medians["reference_b"]
    stable = max(a, b) <= 1.02 * min(a, b)
    ratios = {name: medians["direct"] / medians[name] for name in ("reference_a", "reference_b")}
    return {"order": order, "samples_us_per_replay": samples, "medians_us_per_replay": medians,
            "identical_control_max_min_ratio": max(a, b) / min(a, b),
            "identical_controls_stable": stable, "direct_reference_ratios": ratios,
            "performance_gate_passed": stable and all(ratio <= 1.02 for ratio in ratios.values())}


def paired_graph_time(graphs: dict[str, Any]) -> list[dict[str, Any]]:
    """Interleave 20-replay batches with preallocated, warmed CUDA timing events."""
    if set(graphs) != {"reference_a", "reference_b", "direct"}:
        raise ValueError("Require two independently captured identical-reference graphs")
    names = ["reference_a", "reference_b", "direct"]
    events = {name: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
              for name in names}
    for begin, end in events.values():
        begin.record()
        end.record()
        end.synchronize()
        begin.elapsed_time(end)
    for _ in range(20):
        for name in names:
            graphs[name].replay()
    torch.cuda.synchronize()
    blocks = []
    for order in (names, names[::-1]):
        samples: dict[str, list[float]] = {name: [] for name in names}
        for _ in range(30):
            for name in order:
                begin, end = events[name]
                begin.record()
                for _ in range(20):
                    graphs[name].replay()
                end.record()
                end.synchronize()
                samples[name].append(begin.elapsed_time(end) * 1000 / 20)
        blocks.append(summarize_graph_samples(samples, order))
    return blocks


def main() -> None:
    """Verify two real projections, eager/stream/graph/compile behavior and bounded allocation."""
    from mxwave_zipserv import runtime

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--paired-graphs", action="store_true",
                        help="Prospective paired graph protocol; preserves legacy v1 measurements")
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Preserve previous component measurements")
    validate_runtime()
    prepared_path = args.fixtures / "prepared.json"
    prepared = json.loads(prepared_path.read_bytes())
    cases = [c for c in prepared["cases"] if c["name"].startswith("kda-")]
    if prepared["status"] != "packed_gpu_verification_pending" or len(cases) != 2:
        raise ValueError("Require the existing input/output projection fixtures")
    for name, expected in (("config.json", prepared["source_config_sha256"]),
                           ("model.safetensors.index.json", prepared["source_index_sha256"])):
        if hashlib.sha256((args.source / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Pinned checkpoint metadata changed")
    if hashlib.sha256(Path(__file__).with_name("prepare_bf16_attention.py").read_bytes()
                      ).hexdigest() != prepared["code_sha256"]:
        raise ValueError("Source fixture reconstruction implementation changed")
    index = json.loads((args.source / "model.safetensors.index.json").read_bytes())["weight_map"]
    torch.set_num_threads(1)
    torch.manual_seed(20261017)
    props = torch.cuda.get_device_properties(0)
    torch.cuda.set_per_process_memory_fraction(1.5 * 1024**3 / props.total_memory)
    stream = torch.cuda.Stream()
    max_words = max(c["native_shape"][0] * c["native_shape"][1] for c in cases)
    pool = torch.empty((2, max_words), dtype=torch.bfloat16, device="cuda")
    report: dict[str, Any] = {
        "status": "running", "cases": [], "started_epoch": time.time(),
        "scope": "Host dispatch only; unchanged native BF16 library and fused arithmetic",
        "graph_timing_protocol": ("paired-warmed-events-v2" if args.paired_graphs else
                                  "sequential-single-replay-v1"),
        "gates": {"current_path_output_bits_exact": True, "replay_allocation_growth_bytes": 0,
                  "m4_host_ratio_maximum": .9, "m4_eager_ratio_maximum": 1.02,
                  "m4_graph_ratio_maximum": 1.02},
        "prepared_sha256": hashlib.sha256(prepared_path.read_bytes()).hexdigest(),
        "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()
                          ).hexdigest() for name in ("benchmark_kda_direct_dispatch.py",
                          "kda_direct_dispatch.py", "kda_runtime.py", "prepare_bf16_attention.py")},
        "native_library_sha256": hashlib.sha256(
            Path(runtime._library()._name).read_bytes()).hexdigest(),
        "limits": ["Two real layer-zero/rank-zero projection fixtures; not full-model quality.",
                   "Eager CUDA-event latency includes CPU enqueue gaps.",
                   "Host enqueue measurements exclude synchronization after the call.",
                   "A passed component gate does not establish a model TPS improvement.",
                   "The prototype is not enabled in a serving adapter."]}
    write(args.report, report)
    try:
        for case in cases:
            artifact = args.fixtures / case["artifact"]
            if hashlib.sha256(artifact.read_bytes()).hexdigest() != case["artifact_sha256"]:
                raise ValueError("Packed fixture changed")
            cpu = rebuild_weight(args.source, index, case)
            if tensor_digest(cpu) != case["logical_weight_sha256"]:
                raise ValueError("Independent BF16 source bytes changed")
            del cpu
            packed_cpu = load_file(artifact)
            packed = runtime.PackedWeight(
                tuple(packed_cpu[name].cuda() for name in runtime.BUFFER_NAMES),
                *case["native_metadata"])
            del packed_cpu
            decoded = runtime.decompress(packed).cpu()
            if tensor_digest(decoded) != case["padded_weight_sha256"]:
                raise RuntimeError("Packed weights are not lossless")
            del decoded
            rows, width = case["logical_shape"]
            buffers = list(packed.buffers)
            metadata = (packed.m, packed.k, packed.max_common, packed.max_rare,
                        packed.start_exp, rows)
            for tokens, layout in [*[(n, "contiguous") for n in (0, 1, 2, 3, 4, 7, 8, 9, 16, 32)],
                                   (4, "strided"), (4, "rank3")]:
                x = torch.randn((tokens, width * (2 if layout == "strided" else 1)),
                                dtype=torch.bfloat16, device="cuda") * .1
                if layout == "strided":
                    x = x[:, ::2]
                if layout == "rank3":
                    x = x.reshape(2, 2, width)

                reference = partial(kda_linear, x, buffers, pool, *metadata)
                direct = partial(linear, x, buffers, pool, *metadata)

                expected, actual = reference(), direct()
                require_equal(actual, expected)
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    streamed_a, streamed_b = reference(), direct()
                torch.cuda.current_stream().wait_stream(stream)
                require_equal(streamed_a, streamed_b)
                row: dict[str, Any] = {"fixture": case["name"], "tokens": tokens,
                    "layout": layout, "eager_bits_exact": True, "stream_bits_exact": True}
                report["cases"].append(row)
                if tokens:
                    graphs, outputs = {}, {}
                    for name, fn in (("reference", reference), ("direct", direct)):
                        graphs[name], outputs[name] = capture(fn, stream)
                    before = torch.cuda.memory_allocated()
                    for _ in range(5):
                        x.add_(.125)
                        graphs["reference"].replay()
                        graphs["direct"].replay()
                        require_equal(outputs["direct"], outputs["reference"])
                    growth = torch.cuda.memory_allocated() - before
                    if growth:
                        raise RuntimeError("Graph replay grew retained CUDA memory")
                    row.update(graph_bits_exact=True, replay_allocation_growth_bytes=growth)
                    if tokens <= 8 and layout == "contiguous":
                        row["paired_orders"] = [paired({"reference": reference, "direct": direct},
                                                      reverse=reverse) for reverse in (False, True)]
                        if args.paired_graphs and tokens == 4:
                            second, second_output = capture(reference, stream)
                            require_equal(second_output, outputs["reference"])
                            row["graph_paired"] = paired_graph_time({
                                "reference_a": graphs["reference"], "reference_b": second,
                                "direct": graphs["direct"]})
                            require_equal(outputs["direct"], second_output)
                            del second, second_output
                        else:
                            row["graph_medians_us"] = {name: graph_time(graph)
                                                       for name, graph in graphs.items()}
                    del graphs, outputs
                    if tokens == 4 and layout == "contiguous":
                        def compile_target(value: torch.Tensor, storage: list[torch.Tensor] = buffers,
                                           cache: torch.Tensor = pool,
                                           info: tuple[int, ...] = metadata) -> torch.Tensor:
                            return linear(value, storage, cache, *info)

                        compiled = torch.compile(compile_target, fullgraph=True)
                        require_equal(compiled(x), reference())
                        row["inductor_bits_exact"] = True
                        del compiled, compile_target
                        torch._dynamo.reset()
                write(args.report, report)
                del x, expected, actual, streamed_a, streamed_b, reference, direct
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
            del buffers, packed
            gc.collect()
            torch.cuda.empty_cache()
        selected = [r for r in report["cases"] if r["tokens"] == 4
                    and r["layout"] == "contiguous"]
        passed = all(
            all(p["direct_reference_ratio"]["host_enqueue_us"] <= .9
                and p["direct_reference_ratio"]["gpu_event_us"] <= 1.02
                for p in row["paired_orders"])
            and (all(block["performance_gate_passed"] for block in row["graph_paired"])
                 if args.paired_graphs else
                 row["graph_medians_us"]["direct"] <= 1.02 * row["graph_medians_us"]["reference"])
            for row in selected
        )
        report.update(status="complete", correctness_passed=True, performance_gate_passed=passed,
                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      finished_epoch=time.time())
        write(args.report, report)
        print(json.dumps({k: report[k] for k in ("status", "correctness_passed",
                         "performance_gate_passed", "cuda_peak_allocated_bytes")}))
    except BaseException as exc:
        report.update(status="failed", error=repr(exc), finished_epoch=time.time())
        write(args.report, report)
        raise


if __name__ == "__main__":
    main()
