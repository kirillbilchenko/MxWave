"""Gate eager-only KDA dispatch while proving capture retains the existing operation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from functools import partial
from pathlib import Path
from typing import Any

import kda_runtime
import torch
from benchmark_full_model import write
from benchmark_kda_direct_dispatch import paired, require_equal
from diagnose_bf16_overlap import capture
from diagnose_kda_graph_topology import inspect_captures
from kda_eager_dispatch import linear
from prepare_bf16_attention import rebuild_weight, tensor_digest
from safetensors.torch import load_file


def main() -> None:
    """Keep 24 exactness cases and require stable, address-controlled graph parity at M4."""
    from mxwave_zipserv import runtime

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--fixtures", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Preserve the first eager-only dispatch attempt")
    kda_runtime.validate_runtime()
    prepared = json.loads((args.fixtures / "prepared.json").read_bytes())
    cases = [case for case in prepared["cases"] if case["name"].startswith("kda-")]
    if len(cases) != 2:
        raise ValueError("Require the same source-bound KDA fixtures")
    for name, expected in (("config.json", prepared["source_config_sha256"]),
                           ("model.safetensors.index.json", prepared["source_index_sha256"])):
        if hashlib.sha256((args.source / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Source metadata changed")
    index = json.loads((args.source / "model.safetensors.index.json").read_bytes())["weight_map"]
    torch.set_num_threads(1)
    torch.manual_seed(20261017)
    torch.cuda.set_per_process_memory_fraction(
        1.5 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    stream = torch.cuda.Stream()
    report: dict[str, Any] = {"status": "running", "cases": [], "started_epoch": time.time(),
        "scope": "Eager-only host dispatch; original custom operation during capture and compilation",
        "v1_v2_direct_graph_failures_preserved": True, "prototype_wired_into_adapter": False,
        "gates": {"output_bits_exact": True, "retained_replay_growth_bytes": 0,
                  "host_ratio_maximum": .9, "eager_event_ratio_maximum": 1.02,
                  "identical_graph_control_max_min_maximum": 1.02,
                  "graph_ratio_maximum_against_both_controls": 1.02,
                  "capture_uses_existing_operation": True,
                  "native_and_output_addresses_match_for_graph_control": True},
        "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()
            ).hexdigest() for name in ("benchmark_kda_eager_dispatch.py", "kda_eager_dispatch.py",
                "kda_direct_dispatch.py", "kda_runtime.py", "benchmark_kda_direct_dispatch.py",
                "diagnose_kda_graph_topology.py", "prepare_bf16_attention.py")}}
    write(args.report, report)
    try:
        for case in cases:
            artifact = args.fixtures / case["artifact"]
            if hashlib.sha256(artifact.read_bytes()).hexdigest() != case["artifact_sha256"]:
                raise ValueError("Source-bound packed fixture changed")
            source = rebuild_weight(args.source, index, case)
            if tensor_digest(source) != case["logical_weight_sha256"]:
                raise ValueError("Original BF16 source bytes changed")
            del source
            loaded = load_file(artifact)
            packed = runtime.PackedWeight(tuple(loaded[name].cuda() for name in runtime.BUFFER_NAMES),
                                          *case["native_metadata"])
            del loaded
            restored = runtime.decompress(packed).cpu()
            if tensor_digest(restored) != case["padded_weight_sha256"]:
                raise ValueError("GPU reconstruction is not bit-exact")
            del restored
            rows, width = case["logical_shape"]
            pool = torch.empty((2, packed.m * packed.k), dtype=torch.bfloat16, device="cuda")
            buffers = list(packed.buffers)
            metadata = (packed.m, packed.k, packed.max_common, packed.max_rare, packed.start_exp, rows)
            for tokens, layout in [*[(n, "contiguous") for n in (0, 1, 2, 3, 4, 7, 8, 9, 16, 32)],
                                   (4, "strided"), (4, "rank3")]:
                x = torch.randn((tokens, width * (2 if layout == "strided" else 1)),
                                dtype=torch.bfloat16, device="cuda") * .1
                if layout == "strided":
                    x = x[:, ::2]
                elif layout == "rank3":
                    x = x.reshape(2, 2, width)
                qualified_op = kda_runtime.kda_linear
                reference = partial(qualified_op, x, buffers, pool, *metadata)
                direct = partial(linear, x, buffers, pool, *metadata)
                expected = reference()
                require_equal(direct(), expected)
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    streamed_a, streamed_b = reference(), direct()
                torch.cuda.current_stream().wait_stream(stream)
                require_equal(streamed_a, streamed_b)
                row: dict[str, Any] = {"fixture": case["name"], "tokens": tokens, "layout": layout,
                                       "eager_bits_exact": True, "stream_bits_exact": True}
                report["cases"].append(row)
                if tokens:
                    captured_calls = []

                    def observe_qualified(
                        *args: Any, _calls: list[bool] = captured_calls, _original: Any = qualified_op,
                    ) -> torch.Tensor:
                        _calls.append(torch.cuda.is_current_stream_capturing())
                        return _original(*args)

                    graph_a, output_a = capture(reference, stream)
                    kda_runtime.kda_linear = observe_qualified
                    try:
                        graph_b, output_b = capture(direct, stream)
                    finally:
                        kda_runtime.kda_linear = qualified_op
                    if captured_calls != [True]:
                        raise RuntimeError("The eager-only policy did not retain the original capture op")
                    before = torch.cuda.memory_allocated()
                    for _ in range(5):
                        x.add_(.125)
                        graph_a.replay()
                        graph_b.replay()
                        require_equal(output_b, output_a)
                    growth = torch.cuda.memory_allocated() - before
                    if growth:
                        raise RuntimeError("Graph replay grew retained CUDA allocation")
                    row.update(graph_bits_exact=True, captured_qualified_operation=True,
                               replay_allocation_growth_bytes=growth)
                    del graph_a, graph_b, output_a, output_b
                    if tokens == 4 and layout == "contiguous":
                        row["eager_orders"] = [paired({"reference": reference, "direct": direct},
                                                     reverse=reverse) for reverse in (False, True)]
                        expected = reference()
                        row["shared_graph"] = inspect_captures(
                            {"reference_a": reference, "reference_b": reference, "direct": direct},
                            expected, stream, args.report.parent, case["name"], shared_pool=True)

                        def target(value: torch.Tensor, storage: list[torch.Tensor] = buffers,
                                   cache: torch.Tensor = pool,
                                   info: tuple[int, ...] = metadata) -> torch.Tensor:
                            return linear(value, storage, cache, *info)

                        compiled = torch.compile(target, fullgraph=True)
                        require_equal(compiled(x), reference())
                        explanation = torch._dynamo.explain(target)(x)
                        targets = [str(node.target) for graph in explanation.graphs
                                   for node in graph.graph.nodes if node.op == "call_function"]
                        if (not any("kda_linear" in name for name in targets)
                                or any("kda_fused_direct" in name for name in targets)):
                            raise RuntimeError("Compiled graph did not retain the qualified operation")
                        row.update(inductor_bits_exact=True, compiled_targets=targets)
                        del target, compiled, explanation
                        torch._dynamo.reset()
                write(args.report, report)
                del reference, direct, expected, streamed_a, streamed_b, x
                gc.collect()
                torch.cuda.empty_cache()
            del packed, pool, buffers
            gc.collect()
            torch.cuda.empty_cache()
        selected = [row for row in report["cases"] if row["tokens"] == 4
                    and row["layout"] == "contiguous"]
        passed = len(selected) == 2 and all(
            all(block["direct_reference_ratio"]["host_enqueue_us"] <= .9
                and block["direct_reference_ratio"]["gpu_event_us"] <= 1.02
                for block in row["eager_orders"])
            and row["shared_graph"]["all_native_arguments_match"]
            and row["shared_graph"]["all_logical_output_pointers_match"]
            and row["shared_graph"]["all_topology_signatures_match"]
            and all(block["performance_gate_passed"] for block in row["shared_graph"]["paired_timings"])
            for row in selected)
        report.update(status="complete", correctness_passed=True, performance_gate_passed=passed,
                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      finished_epoch=time.time())
        write(args.report, report)
    except BaseException as exc:
        report.update(status="failed", error=repr(exc), finished_epoch=time.time())
        write(args.report, report)
        raise


if __name__ == "__main__":
    main()
