"""Measure source-exact dense output storage against the current compressed routes."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from benchmark_kda_memory import digest, telemetry
from dense_output_probe import PAIRS, assess, protocol, select_cases, summarize
from kda_policy import CONFIG_SHA, INDEX_SHA
from prepare_bf16_attention import LIBRARY_SHA, RUNTIME_SHA, rebuild_weight, tensor_digest
from safetensors.torch import save_file


def measure(parent: Callable[[], Any], dense: Callable[[], Any], pressure: torch.Tensor,
            *, evict: bool, tokens: int) -> dict[str, Any]:
    """Queue event pairs per leg, retaining controls for both unchanged operations."""
    p = protocol()
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
              for _ in range(p["samples_per_leg"])]
    calls = {"parent_repeat": (parent, parent), "dense_vs_parent": (parent, dense),
             "dense_repeat": (dense, dense)}
    samples = []
    for trial in range(p["trials"]):
        pairs = list(PAIRS) if trial % 2 == 0 else list(reversed(PAIRS))
        order = ["a", "b", "b", "a"] if trial % 2 == 0 else ["b", "a", "a", "b"]
        row: dict[str, Any] = {"trial": trial, "pair_order": pairs, "pairs": {},
                               "before": telemetry()}
        for pair in pairs:
            legs = []
            for arm in order:
                invoke = calls[pair][int(arm == "b")]
                invoke()  # Untimed primer; every retained event follows the same cache policy.
                for start, end in events:
                    if evict:
                        pressure.add_(1)
                    start.record()
                    invoke()
                    end.record()
                events[-1][1].synchronize()
                legs.append({"arm": arm, "us": [a.elapsed_time(b) * 1000 for a, b in events]})
            row["pairs"][pair] = legs
        row["after"] = telemetry()
        samples.append(row)
    return {"samples": samples, "summary": summarize(samples, tokens)}


def main() -> None:
    """Retain actual rank-specific packed fixtures and all paired timing attempts."""
    from mxwave_zipserv import runtime

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/audit"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True, choices=(0, 1))
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Preserve the first declared storage tradeoff measurement")
    plan = json.loads(args.plan.read_bytes())
    p = protocol()
    if plan["protocol"] != p:
        raise ValueError("Prospective storage-tradeoff protocol changed")
    parent_path = args.plan.with_name("parent-candidate.json")
    if digest(parent_path) != plan["selection_parent_sha256"]:
        raise ValueError("The qualified model's storage records changed")
    selected = select_cases(json.loads(parent_path.read_bytes()))
    if selected != plan["selected_cases"]:
        raise ValueError("Prospective rank-specific fixture selection changed")
    if (digest(Path(os.environ["MXWAVE_ZIPSERV_LIBRARY"])) != LIBRARY_SHA
            or digest(Path(runtime.__file__)) != RUNTIME_SHA):
        raise ValueError("Qualified native library or ABI changed")
    source = args.root / "source"
    if (digest(source / "config.json") != CONFIG_SHA
            or digest(source / "model.safetensors.index.json") != INDEX_SHA):
        raise ValueError("Source checkpoint metadata changed")
    torch.set_num_threads(1)
    torch.manual_seed(p["activation_seed"])
    props = torch.cuda.get_device_properties(0)
    if (props.major, props.minor) != (12, 1):
        raise ValueError("Require the qualified SM121 device")
    torch.cuda.set_per_process_memory_fraction(p["CUDA_allocation_cap_bytes"] / props.total_memory)
    report: dict[str, Any] = {
        "status": "running", "rank": args.rank, "protocol": p, "cases": [],
        "source_sha256": digest(Path(__file__)), "plan_sha256": digest(args.plan),
        "selection_parent_sha256": digest(parent_path), "qualified_library_sha256": LIBRARY_SHA,
        "runtime_sha256": RUNTIME_SHA, "torch": torch.__version__, "device": props.name,
        "started_epoch": time.time(), "model_loaded": False,
        "limits": ["Three actual rank-specific output matrices selected by packed storage size.",
                   "Random BF16 activations; whole-model numerical qualification remains required.",
                   "Dense weights replace packed storage in the proposed model, never duplicate it.",
                   "Both storage arms coexist only in this bounded comparison fixture.",
                   "The 50-us decode-cost screen is not an API-throughput guarantee.",
                   "Every repeat-control failure and event sample is retained."],
    }

    def save() -> None:
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    try:
        pressure = torch.zeros(p["pressure_bytes"] // 4, dtype=torch.int32, device="cuda")
        stream = torch.cuda.Stream()
        report["initial_warmup"] = {"before": telemetry()}
        deadline = time.monotonic() + p["warmup_seconds"]
        warm_calls = 0
        while time.monotonic() < deadline:
            pressure.add_(1)
            torch.cuda.synchronize()
            warm_calls += 1
        report["initial_warmup"].update(calls=warm_calls, after=telemetry())
        index = json.loads((source / "model.safetensors.index.json").read_bytes())["weight_map"]
        for case in [x for x in selected if x["rank"] == args.rank]:
            cpu = rebuild_weight(source, index, case)
            if (list(cpu.shape) != case["logical_shape"]
                    or tensor_digest(cpu) != case["logical_weight_sha256"]):
                raise ValueError("Actual rank-specific weight differs from the qualified model")
            packed_cpu = runtime.compress(cpu)
            if packed_cpu.storage_bytes != case["expected_packed_bytes"]:
                raise ValueError("Packing differs from the qualified model's storage record")
            artifact = args.report.parent / (case["name"] + ".safetensors")
            if artifact.exists():
                raise FileExistsError("Preserve each prepared packed fixture")
            save_file(dict(zip(runtime.BUFFER_NAMES, packed_cpu.buffers, strict=True)), artifact)
            buffer_digests = [digest_tensor_bytes(t) for t in packed_cpu.buffers]
            packed = packed_cpu.to("cuda")
            dense_weight = cpu.cuda()
            del cpu, packed_cpu
            arena = torch.full((packed.m * packed.k + 256,), 19.,
                               dtype=torch.bfloat16, device="cuda")
            destination = arena[128:-128].view(packed.m, packed.k)

            def decode(storage: Any = packed, output: torch.Tensor = destination) -> torch.Tensor:
                runtime._check(runtime._library().mxzip_decode(
                    torch.cuda.current_stream().cuda_stream,
                    *(t.data_ptr() for t in storage.buffers), storage.max_common,
                    storage.max_rare, storage.start_exp, output.data_ptr(), storage.m, storage.k))
                return output

            def check_weight(actual: torch.Tensor = destination,
                             expected: torch.Tensor = dense_weight,
                             guards: torch.Tensor = arena) -> None:
                if (not torch.equal(actual.view(torch.int16), expected.view(torch.int16))
                        or not bool((guards[:128] == 19.).all())
                        or not bool((guards[-128:] == 19.).all())):
                    raise RuntimeError("Reconstructed source words or destination guards changed")

            decode()
            check_weight()
            entry = {**case, "artifact": artifact.name, "artifact_sha256": digest(artifact),
                     "packed_bytes": packed.storage_bytes, "buffer_sha256": buffer_digests,
                     "native_metadata": [packed.m, packed.k, packed.max_common,
                                         packed.max_rare, packed.start_exp],
                     "weight_bits_exact": True, "measurements": []}
            report["cases"].append(entry)
            for tokens in p["tokens"]:
                x = torch.randn((tokens, packed.k), dtype=torch.bfloat16, device="cuda") * .1

                def parent(value: torch.Tensor = x, storage: Any = packed,
                           rebuild: Callable[[], torch.Tensor] = decode,
                           fused: bool = tokens in p["fused_tokens"]) -> torch.Tensor:
                    if fused:
                        return runtime.linear(value, storage, split_k=1)
                    return torch.nn.functional.linear(value, rebuild())

                def dense(value: torch.Tensor = x,
                          weight: torch.Tensor = dense_weight) -> torch.Tensor:
                    return torch.nn.functional.linear(value, weight)

                expected = torch.nn.functional.linear(x, dense_weight)
                first_parent, first_dense = parent(), dense()
                error = arithmetic(first_parent, expected)
                graphs, outputs = {}, {}
                for name, call in (("parent", parent), ("dense", dense)):
                    for _ in range(4):
                        call()
                    torch.cuda.synchronize()
                    stream.wait_stream(torch.cuda.current_stream())
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        outputs[name] = call()
                    graphs[name] = graph
                    graph.replay()
                    torch.cuda.synchronize()
                before = torch.cuda.memory_allocated()
                for _ in range(3):
                    for graph in graphs.values():
                        graph.replay()
                torch.cuda.synchronize()
                retained = torch.cuda.memory_allocated() - before
                row: dict[str, Any] = {
                    "tokens": tokens, "parent_route": "fused_split1" if tokens in
                    p["fused_tokens"] else "decode_then_dense", "arithmetic": error,
                    "parent_source_bits_exact": bits_equal(first_parent, expected),
                    "dense_source_bits_exact": bits_equal(first_dense, expected),
                    "parent_eager_graph_bits_exact": bits_equal(first_parent, outputs["parent"]),
                    "dense_eager_graph_bits_exact": bits_equal(first_dense, outputs["dense"]),
                    "graph_replay_retained_allocation_bytes": retained, "timing": [],
                }
                for mode, calls in (("eager", (parent, dense)),
                                    ("graph", (graphs["parent"].replay, graphs["dense"].replay))):
                    for cache in p["cache_regimes"]:
                        measured = measure(*calls, pressure, evict=cache == "pressure", tokens=tokens)
                        row["timing"].append({"mode": mode, "cache": cache, **measured})
                row["repeats_bits_exact"] = (
                    bits_equal(parent(), first_parent) and bits_equal(dense(), first_dense)
                    and bits_equal(outputs["parent"], first_parent)
                    and bits_equal(outputs["dense"], expected))
                check_weight()
                entry["measurements"].append(row)
                save()
                del parent, dense, call, calls, graphs, outputs, graph
                del x, expected, first_parent, first_dense
                gc.collect()
                torch.cuda.empty_cache()
            check_weight()
            entry["destination_guards_unchanged"] = True
            entry["packed_buffers_unchanged"] = [digest_tensor_bytes(t.cpu()) for t in
                                                  packed.buffers] == buffer_digests
            entry["dense_weight_sha256_after"] = tensor_digest(dense_weight.cpu())
            del packed, dense_weight, arena, destination, decode, check_weight
            gc.collect()
            torch.cuda.empty_cache()
            save()
        report.update(status="complete", CUDA_peak_bytes=torch.cuda.max_memory_allocated(),
                      finished_epoch=time.time())
        report["assessment"] = assess(report)
        save()
        print(json.dumps({key: report["assessment"][key] for key in
                          ("diagnostic_complete", "rank", "component_screen_passed", "events")}))
    except BaseException as exc:
        report.update(status="failed", error=repr(exc), finished_epoch=time.time())
        save()
        raise


def digest_tensor_bytes(value: torch.Tensor) -> str:
    """Hash arbitrary CPU fixture storage without converting its numerical type."""
    import hashlib

    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy()).hexdigest()


def bits_equal(first: torch.Tensor, second: torch.Tensor) -> bool:
    """Compare complete BF16 result words, including signed zero."""
    return bool(torch.equal(first.view(torch.int16), second.view(torch.int16)))


def arithmetic(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    """Preserve the existing real-projection relative-L2 screen and report exact word equality."""
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
    if not finite:
        return {"finite": False, "relative_l2": None, "max_absolute_error": None,
                "passed": False}
    difference = actual.float() - reference.float()
    relative = float(difference.norm() / reference.float().norm().clamp_min(1e-30))
    return {"finite": True, "relative_l2": relative,
            "max_absolute_error": float(difference.abs().max()),
            "passed": relative < protocol()["relative_l2_limit"]}


if __name__ == "__main__":
    main()
