"""Validate our bounded Marlin padding at wider, fixed-seven-draft component sizes."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path
from typing import Any

from replay_drafter_expert import bits_equal, digest, frame, load_weights, weight_records
from replay_expert_padding import bind_pipeline, capture, telemetry

PARTS = ("up", "activation", "down", "sum")


def snapshots(owners: Any) -> list[Any]:
    """Copy all four component stages outside the measured graph interval."""
    activation, output, buffers, *_ = owners
    return [value.detach().cpu().clone() for value in
            (buffers[0].output, activation, buffers[1].output, output)]


def prefix_check(a: list[Any], b: list[Any], rows_a: int, rows_b: int,
                 shared: int) -> dict[str, Any]:
    """Compare token-major raw bits even when the two dispatch widths differ."""
    import torch

    counts = {}
    for part, x, y in zip(PARTS, a, b, strict=True):
        x, y = x.reshape(rows_a, -1)[:shared], y.reshape(rows_b, -1)[:shared]
        assert x.shape == y.shape
        counts[part] = int((x.view(torch.int16) != y.view(torch.int16)).sum())
    return {"shared_rows": shared, "changed_elements": counts,
            "exact": not any(counts.values())}


def native_case(x: Any, ids: Any, route_weights: Any,
                weights: list[Any], scales: list[Any]) -> tuple[Any, Any, int]:
    """Call installed stock MoE and observe its actual routing-block choice and metadata."""
    from vllm.model_executor.layers.fused_moe.experts import marlin_moe
    from vllm.scalar_type import scalar_types

    original = marlin_moe.moe_align_block_size
    seen = []

    def observe(*args: Any, **kwargs: Any) -> Any:
        routing = original(*args, **kwargs)
        seen.append((routing, args[1]))
        return routing

    marlin_moe.moe_align_block_size = observe
    try:
        output = marlin_moe.fused_marlin_moe(
            hidden_states=x, w1=weights[0], w2=weights[1], bias1=None, bias2=None,
            w1_scale=scales[0], w2_scale=scales[1], topk_weights=route_weights,
            topk_ids=ids, quant_type_id=scalar_types.float4_e2m1f.id,
            global_num_experts=288,
        )
    finally:
        marlin_moe.moe_align_block_size = original
    assert len(seen) == 1
    routing, block = seen[0]
    return output.cpu().clone(), routing, block


def run(root: Path, source: Path, evidence: Path, frozen: Path,
        fixtures: Path) -> dict[str, Any]:
    """Gate exactness before timing; retain every wider fixture and actual planner choice."""
    import numpy as np
    import torch
    from moe_runtime import NativeMoe, pack_scales
    from safetensors.torch import save_file
    from vllm.model_executor.layers.fused_moe.experts import marlin_moe

    plan = json.loads((root / "plan.json").read_bytes())
    scope = plan["scope"]
    assert not (root / "report.json").exists()
    assert digest(Path(marlin_moe.__file__)) == plan["installed_marlin_sha256"]
    assert digest(fixtures / "manifest.json") == plan["fixture_manifest_sha256"]
    manifest = json.loads((fixtures / "manifest.json").read_bytes())
    for item in manifest["files"]:
        assert digest(fixtures / item["path"]) == item["sha256"]
    for name, expected in plan["frozen_bindings"].items():
        assert digest(frozen / name) == expected
    cycles, replays = scope["timing_interleaved_cycles"], scope["replays_per_timing"]
    assert cycles >= 8 and cycles % 2 == 0 and replays >= 50
    raw, packed = (NativeMoe(frozen / plan[key]) for key in ("raw_library", "packed_library"))
    assert not raw.compressed and packed.encoding == "sparse_palette" and packed.shared_stage
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    report: dict[str, Any] = {
        "status": "running", "started_epoch": time.time(), "ranks": [], "telemetry": [],
        "source_sha256": digest(Path(__file__)), "plan_sha256": digest(root / "plan.json"),
        "fixture_manifest_sha256": digest(fixtures / "manifest.json"),
        "installed_marlin_sha256": digest(Path(marlin_moe.__file__)), "sms": sms,
        "torch": str(torch.__version__), "cuda": str(torch.version.cuda),
        "limits": "Synthetic extensions of real layer-four operands; TP partitions replayed separately on one GPU. No full-model or serving-speed qualification.",
    }

    def progress() -> None:
        (root / "progress.json").write_text(json.dumps(report, indent=2) + "\n")

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
        row: dict[str, Any] = {"rank": rank, "scale_bytes_exact": True, "groups": []}
        report["ranks"].append(row)
        bases: dict[Any, Any] = {}
        for rows in scope["rows"]:
            for pattern in (["repeat"] if rows == 8 else scope["patterns"]):
                group: dict[str, Any] = {"rows": rows, "pattern": pattern, "cases": [],
                                        "comparisons": [], "repeat_checks": 0}
                row["groups"].append(group)
                graphs, retained, outputs = {}, {}, {}
                for arm in ("a1", "b1"):
                    item = next(f for f in manifest["files"] if (f["rank"], f["arm"], f["rows"],
                                f["pattern"]) == (rank, arm, rows, pattern))
                    with np.load(fixtures / item["path"], allow_pickle=False) as data:
                        x = torch.from_numpy(data["input_bf16_bits"].copy()).view(
                            torch.bfloat16,
                        ).cuda()
                        ids = torch.from_numpy(data["topk_ids"].copy()).cuda()
                        rw = torch.from_numpy(data["topk_weights"].copy()).cuda()
                    original_frames = [frame(evidence, arm, rank, 54, i) for i in (0, 1)]
                    native_values, references, parameters = map(list, zip(*original_frames,
                                                                        strict=True))
                    assert bits_equal(x[:8], native_values[0]["/0"])
                    assert bits_equal(rw[:8], native_values[0]["/14"])
                    expected, routing, block = native_case(x, ids, rw, weights, scales)
                    assert block == 8, "Outside the audited BF16 routing-block range"
                    outputs[f"{arm}_installed_native_sum"] = expected
                    for i, args in enumerate(parameters):
                        args.update(size_m=rows if i == 0 else rows * 8, moe_block_size=block)
                    values = [{"/0": x, "/11": routing[0], "/12": routing[1],
                               "/13": routing[2], "/14": rw} for _ in (0, 1)]
                    for policy in scope["policies"]:
                        for implementation in scope["implementations"]:
                            key = (arm, policy, implementation)
                            launch, owners, meta = bind_pipeline(
                                values, parameters, weights, scales, compressed, raw, packed,
                                implementation, policy,
                            )
                            graph, output, lifetime = capture(launch, owners)
                            saved = snapshots(owners)
                            assert all(bool(torch.isfinite(t).all()) for t in saved)
                            if policy == "native":
                                assert bits_equal(saved[3], expected), "Installed native sum differs"
                                if rows == 8:
                                    for actual, reference in zip(saved[:3],
                                            [references[0], native_values[1]["/0"].cpu(),
                                             references[1]], strict=True):
                                        assert bits_equal(actual, reference), "Retained M8 differs"
                            else:
                                for i, projection in enumerate(meta["projections"]):
                                    tile = projection["tile"]
                                    padding = projection["padding"]
                                    n, k = parameters[i]["size_n"], parameters[i]["size_k"]
                                    assert k % tile[0] == 0
                                    assert padding["after"] // block * (n // tile[1]) % (
                                        sms * tile[2]
                                    ) == 0
                            group["cases"].append({"arm": arm, "routing_block": block,
                                                   "fixture_sha256": item["sha256"], **meta})
                            graphs[key] = (graph, owners, lifetime)
                            retained[key] = saved
                            for part, tensor in zip(PARTS, saved, strict=True):
                                outputs["_".join(key) + "_" + part] = tensor
                path = root / f"r{rank}-m{rows}-{pattern}.safetensors"
                save_file(outputs, path)
                group["output"] = {"path": path.name, "sha256": digest(path)}
                passed = True
                for policy in scope["policies"]:
                    for arm in ("a1", "b1"):
                        assert all(bits_equal(a, b) for a, b in zip(
                            retained[arm, policy, "stock"], retained[arm, policy, "packed"],
                            strict=True,
                        )), "Stock versus packed differs"
                    for implementation in scope["implementations"]:
                        check = prefix_check(retained["a1", policy, implementation],
                                             retained["b1", policy, implementation], rows, rows, 7)
                        group["comparisons"].append({"kind": "shared_arms", "policy": policy,
                                                    "implementation": implementation, **check})
                        if policy != "native":
                            passed &= check["exact"]
                        for arm in ("a1", "b1"):
                            key = (arm, policy, implementation)
                            if rows == 8:
                                bases[key] = retained[key]
                            else:
                                check = prefix_check(bases[key], retained[key], 8, rows, 8)
                                group["comparisons"].append({"kind": "shared_widths", "arm": arm,
                                                            "policy": policy,
                                                            "implementation": implementation,
                                                            **check})
                                if policy != "native":
                                    passed &= check["exact"]
                group["numerical_gate_passed"] = bool(passed)
                progress()
                assert passed, "Wide padding does not preserve shared rows"
                for cycle in range(scope["graph_repeat_cycles"]):
                    keys = list(graphs) if cycle % 2 == 0 else list(reversed(graphs))
                    for key in keys:
                        graph, held, _ = graphs[key]
                        graph.replay()
                        torch.cuda.synchronize()
                        assert all(bits_equal(a, b) for a, b in zip(
                            snapshots(held), retained[key], strict=True,
                        )), "Wide graph replay drift"
                        group["repeat_checks"] += 1
                if rows in scope["timing_rows"]:
                    deadline = time.monotonic() + scope["timing_warm_seconds"]
                    while time.monotonic() < deadline:
                        for graph, _, _ in graphs.values():
                            for _ in range(50):
                                graph.replay()
                        torch.cuda.synchronize()
                    timings: dict[str, list[float]] = {":".join(k): [] for k in graphs}
                    for cycle in range(cycles):
                        report["telemetry"].append({"rank": rank, "rows": rows, "pattern": pattern,
                                                     "cycle": cycle, **telemetry()})
                        keys = list(graphs) if cycle % 2 == 0 else list(reversed(graphs))
                        for key in keys:
                            graph = graphs[key][0]
                            begin = torch.cuda.Event(enable_timing=True)
                            end = torch.cuda.Event(enable_timing=True)
                            begin.record()
                            for _ in range(replays):
                                graph.replay()
                            end.record()
                            end.synchronize()
                            timings[":".join(key)].append(begin.elapsed_time(end) * 1000 / replays)
                    group["timings"] = {key: {
                        "samples_us": samples, "median_us": statistics.median(samples),
                        "first_last_half_drift": statistics.median(samples[cycles // 2:]) /
                                                 statistics.median(samples[:cycles // 2]) - 1,
                    } for key, samples in timings.items()}
                progress()
                print(json.dumps({"rank": rank, "rows": rows, "pattern": pattern,
                                  "numerical_gate": "passed", "repeats": group["repeat_checks"],
                                  "timed": "timings" in group}), flush=True)
                del graphs, retained, outputs, original_frames, native_values, values
                del graph, owners, lifetime, launch, saved, output, held
                gc.collect()
        del weights, scales, compressed, w, s, p, bases
        gc.collect()
        torch.cuda.empty_cache()
    report.update(status="complete", finished_epoch=time.time(),
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    """Run one frozen wider component experiment, preserving any failure for independent review."""
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "source", "evidence", "frozen", "fixtures"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.root, args.source, args.evidence, args.frozen, args.fixtures)
    except BaseException as exc:
        (args.root / "failure.json").write_text(json.dumps({"error": repr(exc)}) + "\n")
        raise


if __name__ == "__main__":
    main()
