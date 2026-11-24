"""Validate the actual input adapter with preserved outputs, compiled graphs and stream overlap."""

from __future__ import annotations

import argparse
import functools
import gc
import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from benchmark_fp32_split import raw
from benchmark_kda_memory import digest, telemetry
from fp32_input_policy import BUILD_SHA256, LIBRARY_SHA256, scratch_bytes
from fp32_input_probe import assess, protocol
from prepare_bf16_attention import LIBRARY_SHA, RUNTIME_SHA, tensor_digest
from safetensors.torch import load_file


def sha_tensor(value: torch.Tensor) -> str:
    """Hash actual output bytes without interpreting BF16 values as approximations."""
    return hashlib.sha256(raw(value)).hexdigest()


def graph_check(call: Any, x: torch.Tensor, stream: torch.cuda.Stream) -> tuple[dict[str, Any], str]:
    """Warm before capture and reject a hidden pool-sized clone or retained replay growth."""
    p = protocol()
    first = call(x)
    first_sha = sha_tensor(first)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call(x)
    stream.synchronize()
    before = torch.cuda.memory_allocated()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = call(x)
    graph.replay()
    torch.cuda.synchronize()
    extra = torch.cuda.memory_allocated() - before
    allocated = torch.cuda.memory_allocated()
    for _ in range(p["graph_replays"]):
        graph.replay()
    torch.cuda.synchronize()
    retained = torch.cuda.memory_allocated() - allocated
    after_sha = sha_tensor(output)
    return {"graph_matches_eager": after_sha == first_sha,
            "graph_retained_bytes": retained, "capture_extra_bytes": extra,
            "graph_output_sha256": after_sha}, first_sha


def main() -> None:
    """Exercise only frozen fixtures and restore original output routes for all untested sizes."""
    import kda_runtime as kda
    from fp32_input_runtime import install, prime
    from mxwave_zipserv import runtime

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Preserve the first integrated-wrapper attempt")
    p = protocol()
    plan = json.loads(args.plan.read_bytes())
    if (plan["protocol"] != p or plan["native_build_sha256"] != BUILD_SHA256
            or digest(Path(plan["native_build_path"])) != BUILD_SHA256):
        raise ValueError("Integrated protocol or native build changed")
    parent_path = args.plan.with_name(f"parent-rank{args.rank}.json")
    if digest(parent_path) != plan["parent_report_sha256"][str(args.rank)]:
        raise ValueError("Retained precision report changed")
    parent = json.loads(parent_path.read_bytes())
    if (digest(Path(os.environ["MXWAVE_ZIPSERV_LIBRARY"])) != LIBRARY_SHA
            or digest(Path(runtime.__file__)) != RUNTIME_SHA
            or digest(Path(os.environ["MXWAVE_GLM53_FP32_LIBRARY"])) != LIBRARY_SHA256):
        raise ValueError("Pinned native implementation changed")
    torch.set_num_threads(1)
    props = torch.cuda.get_device_properties(0)
    if (props.major, props.minor) != (12, 1):
        raise ValueError("Require SM121")
    torch.cuda.set_per_process_memory_fraction(p["CUDA_allocation_cap_bytes"] / props.total_memory)
    install()
    prime()
    report: dict[str, Any] = {"status": "running", "rank": args.rank, "protocol": p,
        "source_sha256": digest(Path(__file__)), "plan_sha256": digest(args.plan),
        "parent_report_sha256": digest(parent_path), "native_build_sha256": BUILD_SHA256,
        "qualified_library_sha256": LIBRARY_SHA, "FP32_library_sha256": LIBRARY_SHA256,
        "runtime_sha256": RUNTIME_SHA, "cases": [], "model_loaded": False,
        "started_epoch": time.time(), "initial_telemetry": telemetry()}
    payload_bytes = 0

    def save() -> None:
        temporary = args.report.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.report)

    def parent_tensor(item: dict[str, Any]) -> torch.Tensor:
        rel = Path(item["path"])
        if rel.is_absolute() or len(rel.parts) != 1:
            raise ValueError("Unexpected retained tensor path")
        data = (Path(f"/audit/fp32-split-v1/rank{args.rank}") / rel).read_bytes()
        if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError("Retained activation changed")
        return torch.frombuffer(bytearray(data), dtype=torch.uint8).view(torch.bfloat16).reshape(
            item["shape"]).clone().cuda()

    def retain(name: str, value: torch.Tensor) -> dict[str, Any]:
        nonlocal payload_bytes
        data = raw(value)
        payload_bytes += len(data)
        if payload_bytes > p["maximum_payload_bytes_per_rank"]:
            raise ValueError("Declared component payload bound exceeded")
        path = args.report.parent / (name + ".bf16")
        path.open("xb").write(data)
        return {"path": path.name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                "shape": list(value.shape)}

    save()
    try:
        for case in (c for c in plan["selected_cases"] if c["rank"] == args.rank):
            previous = next(c for c in parent["cases"] if c["name"] == case["name"])
            artifact = Path(case["artifact_path"])
            if digest(artifact) != case["artifact_sha256"]:
                raise ValueError("Packed source changed")
            cpu_buffers = load_file(artifact)
            packed = runtime.PackedWeight(tuple(cpu_buffers[n].cuda() for n in runtime.BUFFER_NAMES),
                                          *case["native_metadata"])
            hashes = [hashlib.sha256(cpu_buffers[n].numpy().tobytes()).hexdigest()
                      for n in runtime.BUFFER_NAMES]
            del cpu_buffers
            decoded = runtime.decompress(packed)
            exact = tensor_digest(decoded.cpu()) == case["padded_weight_sha256"]
            if not exact:
                raise ValueError("Packed source does not reconstruct exactly")
            del decoded
            words = 12608 * 4096
            arena = torch.full((2 * words + 256,), 13., device="cuda", dtype=torch.bfloat16)
            pool = arena[128:-128].view(2, words)
            kda._POOLS[0] = pool
            metadata = (*case["native_metadata"], case["logical_shape"][0])
            layer = SimpleNamespace(mxwave_kda_pool=pool, mxwave_kda_metadata=metadata)
            for name, value in zip(runtime.BUFFER_NAMES, packed.buffers, strict=True):
                setattr(layer, "mxwave_kda_" + name, value)
            call = functools.partial(kda.apply, layer)
            stream = torch.cuda.Stream()
            is_input = case["kind"] == "input"
            entry: dict[str, Any] = {"name": case["name"], "artifact_sha256": case["artifact_sha256"],
                "decoded_weight_bits_exact": exact, "known": [], "fallback": [], "compiled": [],
                "compiled_chain": None}
            report["cases"].append(entry)
            activations = p["activation_cases"] if is_input else ["787"]
            for tokens in p["tokens"]:
                for activation in activations:
                    prior = next(x for x in previous["checks"] if x["tokens"] == tokens
                                 and x["activation"] == activation and x["split"] == 2)
                    x = parent_tensor(prior["artifacts"]["activation"])
                    expected = prior["artifacts"]["candidate" if is_input else "original"]["sha256"]
                    checked, actual = graph_check(call, x, stream)
                    entry["known"].append({"tokens": tokens, "activation": activation,
                        "parent_sha256": expected, "output_sha256": actual,
                        "matches_parent": actual == expected, **checked})
                    if actual != expected:
                        raise ValueError("Integrated route changes the retained native output")
                    del x
                    save()
            if is_input:
                suffix = scratch_bytes(16) // 2
                entry["FP32_workspace_suffix_guards_unchanged"] = bool((pool[:, suffix:suffix+128] == 13.).all())
                if case["layer"] in p["compiled_layers"]:
                    compiled_call = torch.compile(call, fullgraph=True, dynamic=False)
                    for tokens in p["compiled_tokens"]:
                        prior = next(x for x in previous["checks"] if x["tokens"] == tokens
                                     and x["activation"] == "787" and x["split"] == 2)
                        x = parent_tensor(prior["artifacts"]["activation"])
                        checked, actual = graph_check(compiled_call, x, stream)
                        expected = prior["artifacts"]["candidate"]["sha256"]
                        entry["compiled"].append({"tokens": tokens, "output_sha256": actual,
                            "parent_sha256": expected, "matches_parent": actual == expected, **checked})
                        del x
                    del compiled_call
                    entry["FP32_workspace_suffix_guards_unchanged"] &= bool(
                        (pool[:, suffix:suffix+128] == 13.).all())
                for tokens in p["fallback_tokens"]:
                    torch.manual_seed(1949 + tokens)
                    x = torch.randn((tokens, packed.k), dtype=torch.bfloat16, device="cuda") * .1
                    expected = kda.kda_linear(x, list(packed.buffers), pool, *metadata)
                    actual = call(x)
                    checked, actual_sha = graph_check(call, x, stream)
                    name = f"{case['name']}-fallback-m{tokens}"
                    qualified_record = retain(name + "-qualified", expected)
                    actual_record = retain(name + "-actual", actual)
                    entry["fallback"].append({"tokens": tokens, "qualified": qualified_record,
                        "actual": actual_record, "matches_qualified": actual_sha == sha_tensor(expected),
                        **checked})
                    del x, expected, actual
                if case["layer"] in p["compiled_layers"]:
                    chain_inputs, chain_expected = [], []
                    for activation in ("787", "797"):
                        prior = next(x for x in previous["checks"] if x["tokens"] == 8
                                     and x["activation"] == activation and x["split"] == 2)
                        chain_inputs.append(parent_tensor(prior["artifacts"]["activation"]))
                        chain_expected.append(prior["artifacts"]["candidate"]["sha256"])
                    torch.manual_seed(1949 + 17)
                    middle = torch.randn((17, packed.k), dtype=torch.bfloat16, device="cuda") * .1
                    chain_inputs.insert(1, middle)
                    chain_expected.insert(1, next(x["qualified"]["sha256"] for x in entry["fallback"]
                                                  if x["tokens"] == 17))

                    def chain(a: torch.Tensor, mid: torch.Tensor, z: torch.Tensor,
                              invoke: Any = call) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                        return invoke(a), invoke(mid), invoke(z)

                    compiled_chain = torch.compile(chain, fullgraph=True, dynamic=False)
                    eager_outputs = compiled_chain(*chain_inputs)
                    eager_hashes = [sha_tensor(x) for x in eager_outputs]
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            compiled_chain(*chain_inputs)
                    stream.synchronize()
                    allocated = torch.cuda.memory_allocated()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        graph_outputs = compiled_chain(*chain_inputs)
                    graph.replay()
                    torch.cuda.synchronize()
                    extra = torch.cuda.memory_allocated() - allocated
                    allocated = torch.cuda.memory_allocated()
                    for _ in range(p["graph_replays"]):
                        graph.replay()
                    torch.cuda.synchronize()
                    retained = torch.cuda.memory_allocated() - allocated
                    graph_hashes = [sha_tensor(x) for x in graph_outputs]
                    entry["compiled_chain"] = {"physical_rows": [8, 17, 8],
                        "expected_output_sha256": chain_expected, "output_sha256": eager_hashes,
                        "graph_output_sha256": graph_hashes,
                        "graph_matches_eager": graph_hashes == eager_hashes,
                        "graph_retained_bytes": retained, "capture_extra_bytes": extra}
                    del compiled_chain, chain, chain_inputs, middle, eager_outputs, graph_outputs, graph
                # Keep the first captured stream pinned and let a second stream lease the other slot.
                retired = kda.retire_idle_eager_slots()
                if retired["retained_captured_stream_leases"] != 1:
                    raise ValueError("Expected exactly one pinned capture stream before overlap")
                pool.fill_(13.)
                torch.cuda.synchronize()
                other = torch.cuda.Stream()
                graphs, outputs, expected_hashes, activations_gpu = [], [], [], []
                for active, activation in zip((stream, other), ("787", "797"), strict=True):
                    prior = next(x for x in previous["checks"] if x["tokens"] == 8
                                 and x["activation"] == activation and x["split"] == 2)
                    x = parent_tensor(prior["artifacts"]["activation"])
                    activations_gpu.append(x)
                    active.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(active):
                        call(x)
                    active.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=active):
                        outputs.append(call(x))
                    graphs.append(graph)
                    expected_hashes.append(prior["artifacts"]["candidate"]["sha256"])
                allocated = torch.cuda.memory_allocated()
                for _ in range(p["two_stream_replays_per_graph"]):
                    for active, graph in zip((stream, other), graphs, strict=True):
                        with torch.cuda.stream(active):
                            graph.replay()
                torch.cuda.synchronize()
                retained = torch.cuda.memory_allocated() - allocated
                actual_hashes = [sha_tensor(x) for x in outputs]
                third = torch.cuda.Stream()
                third.wait_stream(torch.cuda.current_stream())
                rejected = False
                try:
                    with torch.cuda.stream(third):
                        call(activations_gpu[0])
                except ValueError as exc:
                    if "two inference streams" not in str(exc):
                        raise
                    rejected = True
                state = kda._STATES[(pool.device.index, pool.data_ptr())]
                entry["stream_stress"] = {"replays_per_graph": p["two_stream_replays_per_graph"],
                    "owned_slots": sorted(slot for slot, _ in state.streams.values()),
                    "captured_streams": len(state.captured_streams),
                    "parent_sha256": expected_hashes, "output_sha256": actual_hashes,
                    "outputs_match_parent": actual_hashes == expected_hashes,
                    "replay_retained_bytes": retained, "third_stream_rejected": rejected,
                    "workspace_suffix_guards_unchanged": bool((pool[:, suffix:suffix+128] == 13.).all())}
                del graphs, outputs, activations_gpu, graph, x, state, active, other, third
            torch.cuda.synchronize()
            entry["outer_pool_guards_unchanged"] = bool((arena[:128] == 13.).all() and (arena[-128:] == 13.).all())
            entry["packed_buffers_unchanged"] = hashes == [
                hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest() for t in packed.buffers]
            entry["telemetry"] = telemetry()
            del layer, call, packed, pool, arena, stream, value
            kda._POOLS.clear()
            torch._dynamo.reset()
            gc.collect()
            torch.cuda.empty_cache()
            save()
        report.update(status="complete", CUDA_peak_bytes=torch.cuda.max_memory_allocated(),
                      finished_epoch=time.time(), raw_payload_bytes=payload_bytes)
        report["assessment"] = assess(report)
        print(json.dumps(report["assessment"]))
    except BaseException as exc:
        report.update(status="failed", error=repr(exc), finished_epoch=time.time())
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
