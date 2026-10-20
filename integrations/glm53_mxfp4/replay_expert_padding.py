"""Test full-K routing padding on the exact retained two-projection expert fixture."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

from expert_reduction_policy import bind_padding, block_quantum
from replay_drafter_expert import bits_equal, compare, digest, frame, load_weights, weight_records


def telemetry() -> dict[str, Any]:
    """Retain GPU clock/thermal readings without treating host time as kernel time."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,pstate,power.draw",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False,
    )
    return {"epoch": time.time(), "returncode": result.returncode,
            "csv_sm_MHz_temperature_C_pstate_power_W": result.stdout.strip(),
            "error": result.stderr.strip()}


def verify_padding(original: tuple[Any, ...], padded: tuple[Any, ...],
                   block: int, quantum: int, routes: int) -> dict[str, Any]:
    """Require unchanged consumed entries and prove every appended route is masked."""
    a, b = [tuple(v.cpu() for v in x) for x in (original, padded)]
    count, after = int(a[2][0]), int(b[2][0])
    assert count <= after < count + block * quantum
    assert after % (block * quantum) == 0
    assert bits_equal(a[0][:count], b[0][:count])
    assert bits_equal(a[1][:count // block], b[1][:count // block])
    assert bool((b[0][count:] == routes).all()) and bool((b[1][count // block:] == 0).all())
    return {"before": count, "after": after, "allocated": b[0].numel(),
            "quantum_blocks": quantum, "real_entries_exact": True, "padding_masked": True}


def bind_pipeline(
    values: list[dict[str, Any]], parameters: list[dict[str, Any]],
    weights: list[Any], scales: list[Any], compressed: list[Any],
    raw: Any, packed: Any, implementation: str, policy: str,
) -> tuple[Any, Any, dict[str, Any]]:
    """Bind native/padded up, activation, down and reduction with owned persistent buffers."""
    import torch
    from moe_runtime import allocate_buffers
    from vllm import _custom_ops as ops
    from vllm.scalar_type import scalar_types

    sms = torch.cuda.get_device_properties(0).multi_processor_count
    x = values[0]["/0"]
    activation = torch.empty((64, 1024), dtype=x.dtype, device=x.device)
    output = torch.empty_like(x)
    calls, buffers, padding, owners, metadata = [], [], [], [], []
    for i, (w, s, p, args) in enumerate(zip(weights, scales, compressed, parameters, strict=True)):
        inp = x if i == 0 else activation
        routing = tuple(values[i][f"/{k}"] for k in (11, 12, 13))
        weights_for_routes = values[i]["/14"]
        m, n, k = (args[key] for key in ("size_m", "size_n", "size_k"))
        block, topk = args["moe_block_size"], args["top_k"]
        tile = raw.plan(m, n, k, block, topk, 288, 0)
        if policy == "pad_one":
            tile = (*tile[:2], 1)
        def pad() -> None:
            pass
        row = {"tile": list(tile), "projection": i}
        if policy != "native":
            quantum = block_quantum(n, tile[1], sms, tile[2])
            updated, pad = bind_padding(routing, m * topk, block, quantum)
            pad()
            torch.cuda.synchronize()
            row["padding"] = verify_padding(routing, updated, block, quantum, m * topk)
            routing = updated
        buf = allocate_buffers(m, n, topk, block, len(routing[0]), x.device)
        if implementation == "stock":
            kwargs = args | {"b_q_type": scalar_types.float4_e2m1f}
            if policy != "native":
                kwargs |= {"thread_k": tile[0], "thread_n": tile[1], "blocks_per_sm": tile[2]}

            def operation(
                inp: Any = inp, w: Any = w, s: Any = s, buf: Any = buf,
                routing: Any = routing, rw: Any = weights_for_routes, kwargs: Any = kwargs,
            ) -> Any:
                return ops.moe_wna16_marlin_gemm(
                    inp, buf.output, w, None, s, None, None, None, None, None, buf.locks,
                    *routing, rw, **kwargs,
                )
        else:
            operation = packed.bind(inp, w, p, buf, routing, weights_for_routes,
                                    topk, block, args["mul_topk_weights"], tile)
        calls.append(operation)
        padding.append(pad)
        buffers.append(buf)
        owners.append(routing)
        metadata.append(row)

    def launch() -> Any:
        padding[0]()
        calls[0]()
        ops.silu_and_mul(activation, buffers[0].output)
        padding[1]()
        calls[1]()
        torch.sum(buffers[1].output.view(8, 8, 4096), dim=1, out=output)
        return output

    return launch, (activation, output, buffers, owners, calls, padding), {
        "implementation": implementation, "policy": policy, "projections": metadata,
    }


def capture(launch: Any, owners: Any) -> tuple[Any, Any, Any]:
    """Warm/capture the complete two-GEMM sequence without allocations during replay."""
    import torch

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            launch()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = launch()
    torch.cuda.current_stream().wait_stream(stream)
    graph.replay()
    torch.cuda.synchronize()
    return graph, output, (launch, owners, stream)


def run(root: Path, source: Path, evidence: Path, frozen: Path) -> dict[str, Any]:
    """Gate equality before timing; retain both successful and failed policy outcomes."""
    import torch
    from moe_runtime import NativeMoe, pack_scales
    from safetensors.torch import save_file

    plan = json.loads((root / "plan.json").read_bytes())
    assert not (root / "report.json").exists()
    for name, expected in plan["frozen_bindings"].items():
        assert digest(frozen / name) == expected
    raw, packed = (NativeMoe(frozen / plan[key]) for key in ("raw_library", "packed_library"))
    assert not raw.compressed and packed.encoding == "sparse_palette" and packed.shared_stage
    report: dict[str, Any] = {"status": "running", "started_epoch": time.time(), "ranks": [],
                              "source_sha256": digest(Path(__file__)),
                              "plan_sha256": digest(root / "plan.json"), "telemetry": [],
                              "limits": "One captured layer; component timing, not serving speed."}
    outputs = {}
    for rank in (0, 1):
        weights, scales, compressed = [], [], []
        for family in ("w13", "w2"):
            w, s, proof = load_weights(source, weight_records(source, family), rank, family)
            (root / f"weight-sources-r{rank}-{family}.json").write_text(json.dumps(proof) + "\n")
            p = pack_scales(s, Path("/unused"), encoding="sparse_palette")
            assert bits_equal(packed.decode(p), s.view(torch.uint8))
            weights.append(w)
            scales.append(s)
            compressed.append(p)
        row: dict[str, Any] = {"rank": rank, "cases": [], "comparisons": [], "timings": []}
        for position in (46, 54):
            graphs, retained = {}, {}
            for arm in ("a1", "b1"):
                frames = [frame(evidence, arm, rank, position, i) for i in (0, 1)]
                values, references, parameters = [list(x) for x in zip(*frames, strict=True)]
                for policy in ("native", "pad_default", "pad_one"):
                    for implementation in ("stock", "packed"):
                        key = (arm, policy, implementation)
                        launch, owners, meta = bind_pipeline(
                            values, parameters, weights, scales, compressed, raw, packed,
                            implementation, policy,
                        )
                        graph, output, lifetime = capture(launch, owners)
                        activation, _, buffers, *_ = owners
                        snapshots = [buffers[0].output.cpu().clone(),
                                     activation.cpu().clone(), buffers[1].output.cpu().clone(),
                                     output.cpu().clone()]
                        assert all(bool(torch.isfinite(x).all()) for x in snapshots)
                        if policy == "native":
                            assert bits_equal(snapshots[0], references[0]), "Up replay differs"
                            assert bits_equal(snapshots[1], values[1]["/0"].cpu()), "Activation differs"
                            assert bits_equal(snapshots[2], references[1]), "Down replay differs"
                        name = f"r{rank}_{position}_{arm}_{policy}_{implementation}"
                        for part, tensor in zip(("up", "activation", "down", "sum"), snapshots,
                                                strict=True):
                            outputs[name + "_" + part] = tensor
                        graphs[key] = (graph, output, lifetime)
                        retained[key] = snapshots
                        row["cases"].append({"position": position, "arm": arm, **meta,
                                             "native_matches": policy == "native"})
                        save_file(outputs, root / "outputs.safetensors")
            shared = 8 if position == 46 else 7
            passing = []
            for policy in ("native", "pad_default", "pad_one"):
                equal = True
                for implementation in ("stock", "packed"):
                    a, b = (retained[arm, policy, implementation] for arm in ("a1", "b1"))
                    details = [compare(a[i], b[i], shared) for i in (0, 1, 2)]
                    summed = bits_equal(a[3][:shared], b[3][:shared])
                    equal &= all(x["shared_bits_exact"] for x in details) and summed
                    row["comparisons"].append({
                        "position": position, "policy": policy, "implementation": implementation,
                        "up_activation_down": details, "sum_shared_exact": summed,
                    })
                for arm in ("a1", "b1"):
                    assert all(bits_equal(a, b) for a, b in zip(
                        retained[arm, policy, "stock"], retained[arm, policy, "packed"], strict=True,
                    )), "Stock and compressed differ under the same correction"
                if equal:
                    passing.append(policy)
            (root / "numerical-progress.json").write_text(json.dumps(row, indent=2) + "\n")
            assert {"pad_default", "pad_one"} <= set(passing), "Padding did not fix the shared rows"
            # No performance test is reached unless the exact replay gate passes.
            for cycle in range(20):
                keys = list(graphs) if cycle % 2 == 0 else list(reversed(graphs))
                for key in keys:
                    graph, output, _ = graphs[key]
                    graph.replay()
                    torch.cuda.synchronize()
                    assert bits_equal(output.cpu(), retained[key][3]), "Replay drift"
            timings: dict[str, list[float]] = {":".join(k): [] for k in graphs}
            for graph, _, _ in graphs.values():
                for _ in range(100):
                    graph.replay()
            torch.cuda.synchronize()
            for cycle in range(8):
                report["telemetry"].append({"rank": rank, "position": position,
                                             "cycle": cycle, **telemetry()})
                keys = list(graphs) if cycle % 2 == 0 else list(reversed(graphs))
                for key in keys:
                    graph = graphs[key][0]
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                        enable_timing=True,
                    )
                    begin.record()
                    for _ in range(50):
                        graph.replay()
                    end.record()
                    end.synchronize()
                    timings[":".join(key)].append(begin.elapsed_time(end) * 1000 / 50)
            summary = {}
            for key, samples in timings.items():
                summary[key] = {
                    "samples_us": samples, "median_us": statistics.median(samples),
                    "first_last_half_drift": (
                        statistics.median(samples[4:]) / statistics.median(samples[:4]) - 1
                    ),
                }
            row["timings"].append({"position": position, "graphs": summary})
            print(json.dumps({"rank": rank, "position": position, "numerical_gate": "passed",
                              "medians_us": {k: v["median_us"] for k, v in summary.items()}}),
                  flush=True)
            del graphs, retained, launch, owners, lifetime, graph, output, frames, values
            gc.collect()
        report["ranks"].append(row)
        (root / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
        del weights, scales, compressed, w, s, p
        gc.collect()
        torch.cuda.empty_cache()
    save_file(outputs, root / "outputs.safetensors")
    report.update(status="complete", finished_epoch=time.time(),
                  output_sha256=digest(root / "outputs.safetensors"),
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    """Execute the frozen padding experiment and retain failures without replacing evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "source", "evidence", "frozen"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.root, args.source, args.evidence, args.frozen)
    except BaseException as exc:
        (args.root / "failure.json").write_text(json.dumps({"error": repr(exc)}) + "\n")
        raise


if __name__ == "__main__":
    main()
