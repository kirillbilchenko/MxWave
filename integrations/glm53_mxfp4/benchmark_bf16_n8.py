"""Qualify compact N8 activation staging and compare it to both original and rebuilt kernels."""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
import statistics
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import torch
from benchmark_bf16_kernels import digest
from diagnose_bf16_attention import write
from diagnose_bf16_cache import timed_call
from diagnose_bf16_overlap import capture
from prepare_bf16_attention import LIBRARY_SHA, RUNTIME_SHA, rebuild_weight, tensor_digest
from safetensors.torch import load_file

PRIMARY = {"kda-input-l0-r0", "kda-output-l0-r0", "head-r0"}
TOKENS = (0, 1, 2, 3, 4, 7, 8, 9, 16, 32, 128)
ORDER = ("warm", "evicted", "evicted", "warm", "evicted", "warm", "warm", "evicted")


def exact(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    """Require every BF16 output bit and every shape to remain unchanged."""
    passed = (actual.shape == expected.shape and bool(torch.isfinite(actual).all())
              and bool(torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))))
    if not passed:
        raise RuntimeError("N8 candidate changed output bits or produced a non-finite value")
    return passed


def load_library(runtime: Any, path: Path) -> Any:
    """Keep an independently owned local C ABI handle for each library without changing operators."""
    previous = os.environ["MXWAVE_ZIPSERV_LIBRARY"]
    try:
        os.environ["MXWAVE_ZIPSERV_LIBRARY"] = str(path)
        runtime._library.cache_clear()
        library = runtime._library()
    finally:
        os.environ["MXWAVE_ZIPSERV_LIBRARY"] = previous
        runtime._library.cache_clear()
    return library


def linear(library: Any, packed: Any, x: torch.Tensor, rows: int) -> torch.Tensor:
    """Call the unchanged split-one ABI with the same padding/cropping as the qualified adapter."""
    from mxwave_zipserv.runtime import padded_tokens

    if x.dtype != torch.bfloat16 or x.shape[-1] != packed.k or x.device.type != "cuda":
        raise ValueError("Require matching CUDA BF16 activations")
    tokens = x.numel() // packed.k
    if not tokens:
        return x.new_empty((*x.shape[:-1], rows))
    n = padded_tokens(tokens)
    flat = x.reshape(tokens, packed.k).contiguous()
    if n != tokens:
        padded = x.new_zeros((n, packed.k))
        padded[:tokens].copy_(flat)
        flat = padded
    output, workspace = x.new_empty((n, packed.m)), x.new_empty(0)
    status = library.mxzip_mm(torch.cuda.current_stream().cuda_stream,
        *(buffer.data_ptr() for buffer in packed.buffers), packed.max_common, packed.max_rare,
        packed.start_exp, flat.data_ptr(), output.data_ptr(), packed.m, n, packed.k,
        workspace.data_ptr(), 1)
    if status:
        raise RuntimeError(library.mxzip_error().decode())
    return output[:tokens, :rows].contiguous()


def decode(library: Any, packed: Any) -> torch.Tensor:
    """Verify the full decoded weight through each library's original decoder ABI."""
    output = torch.empty((packed.m, packed.k), device="cuda", dtype=torch.bfloat16)
    status = library.mxzip_decode(torch.cuda.current_stream().cuda_stream,
        *(buffer.data_ptr() for buffer in packed.buffers), packed.max_common, packed.max_rare,
        packed.start_exp, output.data_ptr(), packed.m, packed.k)
    if status:
        raise RuntimeError(library.mxzip_error().decode())
    return output


def gpu_digest(tensor: torch.Tensor) -> str:
    """Hash decoded GPU BF16 rows without a second full head-sized host allocation."""
    value = hashlib.sha256()
    for first in range(0, tensor.shape[0], 2048):
        value.update(tensor[first:first + 2048].cpu().view(torch.uint8).numpy())
    return value.hexdigest()


def resources(library: Any, packed: Any) -> dict[str, Any]:
    """Query compiled register use and CUDA's dynamic-shared-memory occupancy limit."""
    library.mxwave_n8_resources.argtypes = [ctypes.c_int, ctypes.c_int,
                                           ctypes.POINTER(ctypes.c_int)]
    library.mxwave_n8_resources.restype = ctypes.c_int
    values = (ctypes.c_int * 6)()
    status = library.mxwave_n8_resources(packed.max_common, packed.max_rare, values)
    if status:
        raise RuntimeError(f"CUDA kernel resource query failed: {status}")
    return dict(zip(("dynamic_shared_bytes", "registers_per_thread", "static_shared_bytes",
                     "maximum_active_blocks_per_sm", "threads_per_block", "compact"),
                    values, strict=True))


def timing(graphs: dict[str, Any], eviction: torch.Tensor) -> list[dict[str, Any]]:
    """Balance cache and arm orders; exclude cache-pressure work from event intervals."""
    records = []
    for reference in ("original", "control"):
        for block, first in enumerate(ORDER):
            for mode in (first, "evicted" if first == "warm" else "warm"):
                arms = (reference, "compact") if block % 2 == 0 else ("compact", reference)
                record: dict[str, Any] = {"reference": reference, "block": block,
                                          "mode": mode, "order": list(arms)}
                for arm in arms:
                    for _ in range(8):
                        graphs[arm].replay()
                    torch.cuda.synchronize()
                    evict: Callable[[], Any] = ((lambda: eviction.add_(1)) if mode == "evicted"
                                               else (lambda: None))
                    samples = [timed_call(graphs[arm].replay, evict) for _ in range(12)]
                    record[arm + "_samples_us"] = samples
                    record[arm + "_median_us"] = statistics.median(samples)
                record["ratio"] = record["compact_median_us"] / record[reference + "_median_us"]
                records.append(record)
    return records


def summaries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Retain each reference/cache regime and both timing orders in the review."""
    result = []
    for reference in ("original", "control"):
        for mode in ("warm", "evicted"):
            selected = [r for r in records if r["reference"] == reference and r["mode"] == mode]
            result.append({"reference": reference, "mode": mode,
                "reference_median_us": statistics.median(r[reference + "_median_us"] for r in selected),
                "compact_median_us": statistics.median(r["compact_median_us"] for r in selected),
                "median_paired_ratio": statistics.median(r["ratio"] for r in selected),
                "order_ratios": {first: statistics.median(r["ratio"] for r in selected
                    if r["order"][0] == first) for first in (reference, "compact")}})
    return result


def main() -> None:
    """Gate all fourteen fixtures before reporting speed, with a fresh confirmation process."""
    from mxwave_zipserv import runtime

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--confirmation", action="store_true")
    parser.add_argument("--candidate", default="compact")
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Preserve prior N8 measurements")
    original_path = Path(os.environ["MXWAVE_ZIPSERV_LIBRARY"])
    if digest(original_path) != LIBRARY_SHA or digest(Path(runtime.__file__)) != RUNTIME_SHA:
        raise ValueError("Qualified native control or runtime changed")
    build = json.loads((args.build / "native-build.json").read_bytes())
    prepared = json.loads((args.build / "fixtures/prepared.json").read_bytes())
    if (build["status"] != "built_pending_gpu_qualification"
            or prepared["status"] != "prepared_pending_gpu_qualification"
            or len(prepared["cases"]) != 14):
        raise ValueError("Finish the matched native builds and fourteen CPU fixtures first")
    for name, expected in prepared["source_metadata"].items():
        if digest(Path("/audit/source") / name) != expected:
            raise ValueError("CPU fixture source metadata changed")
    libraries = {"original": load_library(runtime, original_path)}
    selected = [a for a in build["arms"] if a["arm"] == args.candidate]
    if len(selected) != 1 or args.candidate == "control":
        raise ValueError("Select one built candidate distinct from the matched control")
    expected_shared_saving = 2048 if selected[0].get("compact", args.candidate == "compact") else 0
    for arm in build["arms"]:
        path = args.build / arm["library"]
        if path.name != arm["library"] or digest(path) != arm["library_sha256"]:
            raise ValueError("Native candidate or rebuilt control changed")
        # Retain the existing report schema: 'compact' names the selected candidate arm.
        if arm["arm"] in ("control", args.candidate):
            key = "control" if arm["arm"] == "control" else "compact"
            libraries[key] = load_library(runtime, path)
    if set(libraries) != {"original", "control", "compact"}:
        raise ValueError("Require the original, same-compiler control and compact candidate")
    torch.set_num_threads(2)
    torch.manual_seed(797 if args.confirmation else 787)
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    if (properties.major, properties.minor) != (12, 1):
        raise ValueError("Require SM121")
    torch.cuda.set_per_process_memory_fraction(2.5 * 1024**3 / properties.total_memory)
    source = Path("/audit/source")
    index = json.loads((source / "model.safetensors.index.json").read_bytes())["weight_map"]
    stream = torch.cuda.Stream()
    eviction = torch.zeros(128 * 1024**2, device="cuda", dtype=torch.uint8)
    report: dict[str, Any] = {"status": "running", "confirmation": args.confirmation,
        "candidate": args.candidate, "expected_shared_saving_bytes": expected_shared_saving,
        "code_sha256": digest(Path(__file__)), "cases": [], "tokens": list(TOKENS),
        "original_library_sha256": LIBRARY_SHA, "runtime_sha256": RUNTIME_SHA,
        "build_manifest_sha256": digest(args.build / "native-build.json"),
        "fixture_manifest_sha256": digest(args.build / "fixtures/prepared.json"),
        "eviction_bytes": eviction.numel(), "l2_cache_bytes": properties.L2_cache_size,
        "gates": {"all_output_bits_exact_to_original": True, "all_gpu_weight_bytes_exact": True,
                  "all_graph_replay_allocation_growth_bytes": 0},
        "limits": ["Component tests, not new model throughput or model quality results.",
            "Current fused arithmetic is the exact reference even where stock GEMM rounds differently.",
            "Timing uses graph replay; the full model currently uses eager M4 execution.",
            "Warm and 128 MiB cache-pressure conditions exclude eviction from event timings.",
            "N16+ calls check external-kernel regressions; selective KDA keeps its prefill fallback.",
            "CUDA occupancy queries are resource limits, not achieved-occupancy measurements."]}
    write(args.report, report)
    started = time.monotonic()
    try:
        for case in prepared["cases"]:
            artifact = Path(case["artifact_path"])
            if digest(artifact) != case["artifact_sha256"]:
                raise ValueError("Prepared packed fixture changed")
            cpu = rebuild_weight(source, index, case)
            if tensor_digest(cpu) != case["logical_weight_sha256"]:
                raise ValueError("Independent source reconstruction changed")
            del cpu
            buffers = load_file(artifact)
            packed = runtime.PackedWeight(tuple(buffers[n].cuda() for n in runtime.BUFFER_NAMES),
                                          *case["native_metadata"])
            del buffers
            entry: dict[str, Any] = {"name": case["name"], "rank": case["rank"],
                "logical_shape": case["logical_shape"], "source_weight_sha256":
                case["logical_weight_sha256"], "artifact_sha256": case["artifact_sha256"],
                "native_metadata": case["native_metadata"], "weight_checks": {},
                "resources": {}, "arithmetic": [], "graphs": [], "timings": []}
            report["cases"].append(entry)
            for name, library in libraries.items():
                decoded = decode(library, packed)
                passed = gpu_digest(decoded) == case["padded_weight_sha256"]
                entry["weight_checks"][name] = passed
                del decoded
                if not passed:
                    raise RuntimeError("Native decoder changed source BF16 bytes")
                if name != "original":
                    entry["resources"][name] = resources(library, packed)
            if (entry["resources"]["control"]["dynamic_shared_bytes"]
                    - entry["resources"]["compact"]["dynamic_shared_bytes"] != expected_shared_saving):
                raise RuntimeError("N8 resource change differs from the declared candidate")
            rows, width = case["logical_shape"]
            for tokens in TOKENS:
                x = torch.randn((tokens, width), device="cuda", dtype=torch.bfloat16) * .1
                operations = {name: partial(linear, lib, packed, x, rows)
                              for name, lib in libraries.items()}
                expected = runtime.linear(x, packed, split_k=1)[..., :rows].contiguous()
                record: dict[str, Any] = {"tokens": tokens, "random": {}}
                for name, operation in operations.items():
                    record["random"][name] = exact(operation(), expected)
                for kind in ("basis", "stress"):
                    value = torch.zeros_like(x) if kind == "basis" else x * 256
                    if kind == "basis" and tokens:
                        value[torch.arange(tokens), torch.arange(tokens)] = 1
                    reference = linear(libraries["original"], packed, value, rows)
                    record[kind] = {name: exact(linear(lib, packed, value, rows), reference)
                                    for name, lib in libraries.items()}
                    del value, reference
                entry["arithmetic"].append(record)
                if tokens == 4 or (case["name"] in PRIMARY and tokens in (1, 8)):
                    graphs, outputs = {}, {}
                    for name, operation in operations.items():
                        graphs[name], outputs[name] = capture(operation, stream)
                        exact(outputs[name], expected)
                        outputs[name].fill_(float("nan"))
                        before = torch.cuda.memory_allocated()
                        for _ in range(3):
                            graphs[name].replay()
                        torch.cuda.synchronize()
                        if torch.cuda.memory_allocated() != before:
                            raise RuntimeError("N8 graph replay grew GPU allocations")
                        exact(outputs[name], expected)
                        with torch.cuda.stream(stream):
                            streamed = operation()
                        torch.cuda.current_stream().wait_stream(stream)
                        exact(streamed, expected)
                        del streamed
                    entry["graphs"].append({"tokens": tokens, "bits_exact": True,
                        "nondefault_stream": True, "replay_allocation_growth_bytes": 0})
                    if case["name"] in PRIMARY:
                        records = timing(graphs, eviction)
                        entry["timings"].append({"tokens": tokens, "records": records,
                                                 "summaries": summaries(records)})
                    del graphs, outputs
                del x, operations, expected, operation
                gc.collect()
                torch.cuda.empty_cache()
                write(args.report, report)
            del packed
            gc.collect()
            torch.cuda.empty_cache()
        report.update(status="passed", seconds=time.monotonic() - started,
                      cuda_peak_bytes=torch.cuda.max_memory_allocated())
        write(args.report, report)
        print(json.dumps({k: report[k] for k in ("status", "seconds", "cuda_peak_bytes")}))
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        write(args.report, report)
        raise


if __name__ == "__main__":
    main()
