"""Measure qualified KDA reconstruction, exact copies and unchanged prefill GEMMs."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl
from kda_memory_probe import assess, protocol, summarize
from prepare_bf16_attention import LIBRARY_SHA, RUNTIME_SHA, rebuild_weight, tensor_digest
from safetensors.torch import load_file


@triton.jit
def _copy_bits(source, destination, n: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(source + index, index < n, 0)
    tl.store(destination + index, value, index < n)


def digest(path: Path) -> str:
    """Hash a consumed fixture without materializing another complete byte copy."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def telemetry() -> dict[str, Any]:
    """Record sampled clock, temperature and process memory outside event intervals."""
    import pynvml as nvml

    nvml.nvmlInit()
    handle = nvml.nvmlDeviceGetHandleByIndex(0)
    result: dict[str, Any] = {"epoch": time.time()}
    for key, query in {
        "SM_MHz": lambda: nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_SM),
        "memory_MHz": lambda: nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_MEM),
        "temperature_C": lambda: nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU),
        "throttle_reasons": lambda: nvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle),
    }.items():
        try:
            result[key] = query()
        except nvml.NVMLError as exc:
            result[key] = {"unavailable": str(exc)}
    result["process_memory_kib"] = {
        key: int(value.split()[0])
        for line in Path("/proc/self/status").read_text().splitlines()
        for key, _, value in [line.partition(":")]
        if key in {"VmRSS", "VmHWM", "RssAnon", "RssFile", "RssShmem"}
    }
    return result


def timed(operation: Callable[[], Any], pressure: torch.Tensor, *, evict: bool
          ) -> list[dict[str, Any]]:
    """Time the identical callable as both control arms in balanced bracketed order."""
    p = protocol()
    records = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for trial in range(p["trials"]):
        row: dict[str, Any] = {"trial": trial, "legs": [], "before": telemetry()}
        for arm in (["a", "b", "b", "a"] if trial % 2 == 0 else ["b", "a", "a", "b"]):
            values = []
            for _ in range(p["samples_per_leg"]):
                if evict:
                    pressure.add_(1)
                start.record()
                operation()
                end.record()
                end.synchronize()
                values.append(start.elapsed_time(end) * 1000)
            row["legs"].append({"arm": arm, "us": values})
        row["after"] = telemetry()
        records.append(row)
    return records


def main() -> None:
    """Keep correctness, unprofiled timing and cold-replay counters explicitly separate."""
    from mxwave_zipserv import runtime

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/audit"))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Preserve every diagnostic attempt")
    if (digest(Path(os.environ["MXWAVE_ZIPSERV_LIBRARY"])) != LIBRARY_SHA
            or digest(Path(runtime.__file__)) != RUNTIME_SHA):
        raise ValueError("Qualified native library or ABI changed")
    torch.set_num_threads(1)
    torch.manual_seed(1907)
    props = torch.cuda.get_device_properties(0)
    if (props.major, props.minor) != (12, 1):
        raise ValueError("This bounded diagnostic requires SM121")
    p = protocol()
    torch.cuda.set_per_process_memory_fraction(p["CUDA_allocation_cap_bytes"] / props.total_memory)
    report: dict[str, Any] = {
        "status": "running", "protocol": p, "profile": args.profile, "operations": [],
        "source_sha256": digest(Path(__file__)), "qualified_library_sha256": LIBRARY_SHA,
        "runtime_sha256": RUNTIME_SHA, "torch": torch.__version__, "triton": triton.__version__,
        "device": props.name, "SMs": props.multi_processor_count, "started_epoch": time.time(),
        "limits": ["Same two layer-zero/rank-zero fixtures on both GPUs; not all model weights.",
                   "Logical payload rates are not measured LPDDR traffic or a hardware ceiling.",
                   "The best observed copy rate is a measured baseline, not a maximum-bandwidth proof.",
                   "NCU cold replay is separate from unprofiled eager/graph timing.",
                   "No serving runtime, weights, batch policy or arithmetic is changed."],
    }

    def save() -> None:
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def exercise(name: str, call: Callable[[], Any], check: Callable[[], None],
                 payload: int, flops: int, metadata: dict[str, Any]) -> None:
        for _ in range(4):
            call()
        torch.cuda.synchronize()
        check()
        graph = None
        if not args.profile:
            graph = torch.cuda.CUDAGraph()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.graph(graph, stream=stream):
                call()
            graph.replay()
            torch.cuda.synchronize()
            check()
            before = torch.cuda.memory_allocated()
            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize()
            if torch.cuda.memory_allocated() != before:
                raise RuntimeError("Graph replay retains additional allocations")
        item: dict[str, Any] = {"name": name, "exact_output_and_guards": True,
                               "logical_payload_bytes": payload, "useful_FLOPs": flops,
                               "metadata": metadata, "measurements": []}
        report["operations"].append(item)
        if args.profile:
            item["telemetry_before"] = telemetry()
            label = "kda_memory/" + name
            torch.cuda.cudart().cudaProfilerStart()
            torch.cuda.nvtx.range_push(label)
            call()
            torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            item["profile_label"] = label
            item["telemetry_after"] = telemetry()
        else:
            assert graph is not None
            for mode, invoke in (("eager", call), ("graph", graph.replay)):
                for cache in p["cache_regimes"]:
                    samples = timed(invoke, pressure, evict=cache == "pressure")
                    item["measurements"].append({"mode": mode, "cache": cache,
                                                 "samples": samples, "summary": summarize(samples)})
        check()
        save()
        del graph

    save()
    try:
        pressure = torch.zeros(p["pressure_bytes"] // 4, dtype=torch.int32, device="cuda")
        stream = torch.cuda.Stream()
        for size in p["copy_MiB"]:
            count = size * 1024**2 // 4
            source_bits = torch.arange(count, dtype=torch.int32, device="cuda")
            arena_bits = torch.full((count + 128,), -9317, dtype=torch.int32, device="cuda")
            target_bits = arena_bits[64:-64]

            def check_copy(source: torch.Tensor = source_bits, target: torch.Tensor = target_bits,
                           guards: torch.Tensor = arena_bits) -> None:
                if (not torch.equal(source, target)
                        or not bool((guards[:64] == -9317).all())
                        or not bool((guards[-64:] == -9317).all())):
                    raise RuntimeError("Copy output or guards changed")

            for block in p["copy_blocks"]:
                def copy(block_size: int = block, n: int = count,
                         source: torch.Tensor = source_bits,
                         target: torch.Tensor = target_bits) -> None:
                    _copy_bits[(triton.cdiv(n, block_size),)](
                        source, target, n, BLOCK=block_size, num_warps=4)
                exercise(f"copy-{size}MiB-b{block}", copy, check_copy, size * 1024**2 * 2, 0,
                         {"copy_bytes": size * 1024**2, "block": block})
            del copy, check_copy, source_bits, arena_bits, target_bits
            gc.collect()
            torch.cuda.empty_cache()
        fixture = args.root / "bf16-attention-v1/artifacts-v2"
        prepared = json.loads((fixture / "prepared.json").read_bytes())
        if prepared["code_sha256"] != digest(Path(__file__).with_name("prepare_bf16_attention.py")):
            raise ValueError("Fixture reconstruction source changed")
        index = json.loads((args.root / "source/model.safetensors.index.json").read_bytes())["weight_map"]
        for case in [r for r in prepared["cases"] if r["name"] in p["fixture_names"]]:
            artifact = fixture / case["artifact"]
            if artifact.name != case["artifact"] or digest(artifact) != case["artifact_sha256"]:
                raise ValueError("Packed fixture changed")
            cpu = rebuild_weight(args.root / "source", index, case)
            if tensor_digest(cpu) != case["logical_weight_sha256"]:
                raise ValueError("Checkpoint source words changed")
            rows, width = cpu.shape
            padded = torch.nn.functional.pad(cpu, (0, 0, 0, case["native_padding_rows"]))
            if tensor_digest(padded) != case["padded_weight_sha256"]:
                raise ValueError("Padded checkpoint words changed")
            dense = cpu.cuda()
            expected = padded.cuda()
            buffers = load_file(artifact)
            packed = runtime.PackedWeight(tuple(buffers[n].cuda() for n in runtime.BUFFER_NAMES),
                                          *case["native_metadata"])
            del buffers, padded, cpu
            arena = torch.full((packed.m * packed.k + 256,), 19., dtype=torch.bfloat16, device="cuda")
            output = arena[128:-128].view(packed.m, packed.k)
            expected_bits, output_bits = expected.view(torch.int32), output.view(torch.int32)

            def check_weight(actual: torch.Tensor = output_bits,
                             wanted: torch.Tensor = expected_bits,
                             guards: torch.Tensor = arena) -> None:
                if (not torch.equal(actual, wanted)
                        or not bool((guards[:128] == 19.).all())
                        or not bool((guards[-128:] == 19.).all())):
                    raise RuntimeError("Reconstructed weight or guards changed")

            def decode(storage: Any = packed, target: torch.Tensor = output) -> None:
                runtime._check(runtime._library().mxzip_decode(
                    torch.cuda.current_stream().cuda_stream,
                    *(t.data_ptr() for t in storage.buffers), storage.max_common, storage.max_rare,
                    storage.start_exp, target.data_ptr(), storage.m, storage.k))

            count = output_bits.numel()

            def copy_dense(n: int = count, source: torch.Tensor = expected_bits,
                           target: torch.Tensor = output_bits) -> None:
                _copy_bits[(triton.cdiv(n, 4096),)](
                    source, target, n, BLOCK=4096, num_warps=4)

            metadata = {"artifact_sha256": case["artifact_sha256"],
                        "source_weight_sha256": case["logical_weight_sha256"],
                        "padded_weight_sha256": case["padded_weight_sha256"],
                        "logical_shape": [rows, width], "padded_shape": [packed.m, packed.k],
                        "packed_bytes": packed.storage_bytes}
            decode_bytes = packed.storage_bytes + output.numel() * 2
            exercise(case["name"] + "/decode", decode, check_weight, decode_bytes, 0, metadata)
            exercise(case["name"] + "/copy-dense", copy_dense, check_weight,
                     output.numel() * 4, 0, metadata)
            for tokens in p["prefill_rows"]:
                x = torch.randn((tokens, width), dtype=torch.bfloat16, device="cuda") * .1
                reference = torch.nn.functional.linear(x, dense)
                y = torch.empty_like(reference)

                def dense_gemm(activation: torch.Tensor = x, weight: torch.Tensor = dense,
                               target: torch.Tensor = y) -> None:
                    torch.mm(activation, weight.T, out=target)

                def coupled(activation: torch.Tensor = x, weight: torch.Tensor = output,
                            target: torch.Tensor = y, nrows: int = rows,
                            reconstruct: Callable[[], None] = decode) -> None:
                    reconstruct()
                    torch.mm(activation, weight[:nrows].T, out=target)

                def check_gemm(actual: torch.Tensor = y, wanted: torch.Tensor = reference,
                               verify_weight: Callable[[], None] = check_weight) -> None:
                    verify_weight()
                    if not torch.equal(actual.view(torch.int16), wanted.view(torch.int16)):
                        raise RuntimeError("Unchanged GEMM differs from source-weight output")

                useful = 2 * tokens * rows * width
                payload = 2 * (tokens * width + rows * width + tokens * rows)
                for arm, call, extra in (("dense", dense_gemm, 0), ("decode-dense", coupled, decode_bytes)):
                    exercise(f"{case['name']}/M{tokens}/{arm}", call, check_gemm,
                             payload + extra, useful, {**metadata, "activation_rows": tokens})
                del call, dense_gemm, coupled, check_gemm, x, y, reference
            del decode, copy_dense, check_weight
            del dense, expected, expected_bits, output, output_bits, packed, arena
            gc.collect()
            torch.cuda.empty_cache()
        report.update(status="complete", CUDA_peak_bytes=torch.cuda.max_memory_allocated(),
                      finished_epoch=time.time())
        report["assessment"] = assess(report)
        save()
        print(json.dumps({"status": report["status"], "operations": len(report["operations"]),
                          "CUDA_peak_bytes": report["CUDA_peak_bytes"]}), flush=True)
    except BaseException as exc:
        report.update(status="failed", error=repr(exc), finished_epoch=time.time())
        save()
        raise


if __name__ == "__main__":
    main()
