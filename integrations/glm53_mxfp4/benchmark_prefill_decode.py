"""Compare exact standalone decoding and unchanged stock GEMMs at observed prefill shapes."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from prefill_decode import MODES, NativeDecoder
from prepare_bf16_attention import (
    CONFIG_SHA,
    INDEX_SHA,
    LIBRARY_SHA,
    RUNTIME_SHA,
    rebuild_weight,
    tensor_digest,
)
from safetensors.torch import load_file
from smoke_prefill_decode import verify_words

TOKENS = (16, 128, 256, 257, 259, 512, 513, 514, 769, 1024, 2048)
ARMS = ("qualified", *MODES)


def decoder_calls(native: NativeDecoder, packed: Any, output: torch.Tensor
                  ) -> dict[str, Callable[[], torch.Tensor]]:
    """Bind the current qualified decoder and each separate-library variant to one output."""
    from mxwave_zipserv import runtime

    def qualified() -> torch.Tensor:
        runtime._check(runtime._library().mxzip_decode(
            torch.cuda.current_stream(output.device).cuda_stream,
            *(buffer.data_ptr() for buffer in packed.buffers),
            packed.max_common, packed.max_rare, packed.start_exp,
            output.data_ptr(), packed.m, packed.k))
        return output

    def make(mode: str) -> Callable[[], torch.Tensor]:
        def call() -> torch.Tensor:
            native.decode(mode, packed, output)
            return output
        return call

    return {"qualified": qualified, **{name: make(name) for name in MODES}}


def coupled_calls(calls: dict[str, Callable[[], torch.Tensor]], x: torch.Tensor, rows: int
                  ) -> dict[str, Callable[[], torch.Tensor]]:
    """Keep the stock GEMM and every activation/weight value identical across decoder arms."""
    def make(call: Callable[[], torch.Tensor]) -> Callable[[], torch.Tensor]:
        def operation() -> torch.Tensor:
            return torch.nn.functional.linear(x, call()[:rows])
        return operation
    return {name: make(call) for name, call in calls.items()}


def ratio_passes(measured: dict[str, Any], arm: str, lower: float, upper: float) -> bool:
    """Apply the declared bounds to both orders as well as the combined paired median."""
    values = [measured["median_paired_ratio"][arm]] + [
        row[arm] for row in measured["order_ratios"].values()]
    return all(lower <= value <= upper for value in values)


def digest(path: Path) -> str:
    """Bind small component inputs without reading the complete checkpoint."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(operation: Callable[[], Any], stream: torch.cuda.Stream) -> tuple[Any, Any]:
    """Warm and capture on one reusable stream with stable caller-owned weight storage."""
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(4):
            operation()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = operation()
    graph.replay()
    torch.cuda.synchronize()
    return graph, output


def measure(graphs: dict[str, Any], pressure: torch.Tensor, *, trials: int,
            iterations: int, evict: bool, reverse_first: bool = False) -> dict[str, Any]:
    """Retain alternating orders, same-graph controls and per-call CUDA-event timings."""
    records = []
    for trial in range(trials):
        forward = (trial + int(reverse_first)) % 2 == 0
        order = list(graphs) if forward else list(reversed(graphs))
        row: dict[str, Any] = {"trial": trial, "order": order, "times_us": {},
                               "direction": "forward" if forward else "reverse"}
        for arm in order:
            values = []
            for _ in range(iterations):
                if evict:
                    pressure.add_(1)
                first = torch.cuda.Event(enable_timing=True)
                last = torch.cuda.Event(enable_timing=True)
                first.record()
                graphs[arm].replay()
                last.record()
                last.synchronize()
                values.append(first.elapsed_time(last) * 1000)
            row["times_us"][arm] = values
        reference = statistics.median(row["times_us"]["qualified"])
        row["ratios"] = {arm: statistics.median(values) / reference
                         for arm, values in row["times_us"].items()}
        records.append(row)
    return {"trials": trials, "iterations": iterations, "pressure_bytes": pressure.numel() * 4
            if evict else 0, "samples": records,
        "median_us": {arm: statistics.median(statistics.median(r["times_us"][arm])
                                             for r in records) for arm in graphs},
        "median_paired_ratio": {arm: statistics.median(r["ratios"][arm] for r in records)
                                for arm in graphs},
        "order_ratios": {order: {arm: statistics.median(r["ratios"][arm] for r in records
                                     if r["direction"] == order)
                                 for arm in graphs} for order in ("forward", "reverse")}}


def main() -> None:
    """Require source bytes and stock-GEMM outputs before timing every predeclared case."""
    from mxwave_zipserv import runtime

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--confirmation", action="store_true")
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Preserve the first component timing report")
    if json.loads(args.smoke.read_bytes())["status"] != "passed":
        raise RuntimeError("Synthetic smoke must pass before real fixtures")
    plan = json.loads(args.plan.read_bytes())
    if (plan["tokens"] != list(TOKENS) or plan["trials"] != 8
            or plan["pure_iterations"] != 8 or plan["coupled_iterations"] != 4
            or plan["decode_gain"] != .05 or plan["nonregression"] != .02
            or plan["control_parity"] != .02):
        raise RuntimeError("Frozen component protocol changed")
    if (digest(Path(os.environ["MXWAVE_ZIPSERV_LIBRARY"])) != LIBRARY_SHA
            or digest(Path(runtime.__file__)) != RUNTIME_SHA):
        raise RuntimeError("Qualified runtime/library changed")
    for name, expected in (("config.json", CONFIG_SHA), ("model.safetensors.index.json", INDEX_SHA)):
        if digest(args.root / "source" / name) != expected:
            raise RuntimeError("Audited checkpoint metadata changed")
    torch.set_num_threads(1)
    torch.manual_seed(9417 + int(args.confirmation))
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    torch.cuda.set_per_process_memory_fraction(1.5 * 1024**3 / props.total_memory)
    native = NativeDecoder(args.build)
    source = args.root / "source"
    index = json.loads((source / "model.safetensors.index.json").read_bytes())["weight_map"]
    fixture = args.root / "bf16-attention-v1/artifacts-v2"
    prepared = json.loads((fixture / "prepared.json").read_bytes())
    if (prepared["status"] != "packed_gpu_verification_pending"
            or prepared["code_sha256"] != digest(Path(__file__).with_name(
                "prepare_bf16_attention.py"))):
        raise RuntimeError("Immutable KDA fixture preparation changed")
    cases = [row for row in prepared["cases"] if row["name"].startswith("kda-")]
    if [row["name"] for row in cases] != ["kda-input-l0-r0", "kda-output-l0-r0"]:
        raise RuntimeError("Require both audited real KDA shapes")
    report: dict[str, Any] = {"status": "running", "cases": [], "qualification_attempted": False,
        "full_model_loaded": False, "tokens": list(TOKENS), "trials": 8,
        "confirmation": args.confirmation,
        "plan_sha256": digest(args.plan),
        "code_sha256": digest(Path(__file__)), "native_manifest": native.manifest,
        "qualified_library_sha256": LIBRARY_SHA, "runtime_sha256": RUNTIME_SHA,
        "fixture_manifest_sha256": digest(fixture / "prepared.json"), "smoke_sha256":
            digest(args.smoke), "torch": torch.__version__, "started_epoch": time.time(),
        "limits": ["Two real layer-zero/rank-zero KDA matrices, not a complete GLM forward.",
            "No full-model reload, TP communication, serving throughput or quality qualification.",
            "Graph timings retain both orders and a rebuilt unchanged-decoder control.",
            "128 MiB pressure is outside each event pair and is not measured LPDDR traffic.",
            "Coupled operations use the same stock BF16 GEMM and caller-owned decoded weight."]}

    def save() -> None:
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    try:
        pressure = torch.zeros(32 * 1024**2, dtype=torch.int32, device="cuda")
        stream = torch.cuda.Stream()
        for case in cases:
            artifact = fixture / case["artifact"]
            if artifact.name != case["artifact"] or digest(artifact) != case["artifact_sha256"]:
                raise RuntimeError("Compressed source fixture changed")
            cpu = rebuild_weight(source, index, case)
            if tensor_digest(cpu) != case["logical_weight_sha256"]:
                raise RuntimeError("Independent source BF16 reconstruction changed")
            rows, width = cpu.shape
            dense = cpu.cuda()
            padded = torch.nn.functional.pad(cpu, (0, 0, 0, case["native_padding_rows"]))
            if tensor_digest(padded) != case["padded_weight_sha256"]:
                raise RuntimeError("Padded source weight changed")
            expected = padded.cuda()
            buffers = load_file(artifact)
            packed = runtime.PackedWeight(tuple(buffers[name].cuda() for name in runtime.BUFFER_NAMES),
                                          *case["native_metadata"])
            del cpu, padded, buffers
            arena = torch.full((packed.m * packed.k + 128,), 19., device="cuda",
                               dtype=torch.bfloat16)
            output = arena[64:-64].view(packed.m, packed.k)
            calls = decoder_calls(native, packed, output)
            entry: dict[str, Any] = {"name": case["name"], "source_weight_sha256":
                case["logical_weight_sha256"], "artifact_sha256": case["artifact_sha256"],
                "shape": [rows, width], "padded_shape": [packed.m, packed.k],
                "destination_address": output.data_ptr(), "resources": {
                    name: native.resources(name, packed) for name in MODES}, "coupled": []}
            report["cases"].append(entry)
            graphs = {}
            for arm, call in calls.items():
                output.fill_(-3)
                call()
                torch.cuda.synchronize()
                verify_words(output, expected, arena)
                graph, pure_result = capture(call, stream)
                output.fill_(-3)
                graph.replay()
                torch.cuda.synchronize()
                verify_words(output, expected, arena)
                graphs[arm] = graph
            entry["weight_bits_exact"] = True
            entry["decode"] = {cache: measure(graphs, pressure, trials=8, iterations=8,
                evict=cache == "pressure", reverse_first=args.confirmation)
                for cache in ("warm", "pressure")}
            del graphs, graph, pure_result
            save()
            for tokens in TOKENS:
                x = torch.randn((tokens, width), device="cuda", dtype=torch.bfloat16) * .1
                reference = torch.nn.functional.linear(x, dense)
                operations = coupled_calls(calls, x, rows)
                graphs, outputs = {}, {}
                for arm, operation in operations.items():
                    graph, result = capture(operation, stream)
                    if not torch.equal(result.view(torch.uint8), reference.view(torch.uint8)):
                        raise RuntimeError(f"Stock GEMM changed output bits: {arm}, M={tokens}")
                    graphs[arm], outputs[arm] = graph, result
                row = {"tokens": tokens, "outputs_bits_exact": True, "timing": {
                    cache: measure(graphs, pressure, trials=8, iterations=4,
                        evict=cache == "pressure", reverse_first=args.confirmation)
                    for cache in ("warm", "pressure")}}
                for graph in graphs.values():
                    output.fill_(-3)
                    graph.replay()
                torch.cuda.synchronize()
                verify_words(output, expected, arena)
                if any(not torch.equal(y.view(torch.uint8), reference.view(torch.uint8))
                       for y in outputs.values()):
                    raise RuntimeError("Coupled output changed after all timed graph replays")
                entry["coupled"].append(row)
                save()
                del graphs, outputs, graph, result, x, reference, operations, operation
                gc.collect()
                torch.cuda.empty_cache()
            del calls, output, arena, packed, dense, expected, call
            gc.collect()
            torch.cuda.empty_cache()
        pure_control = all(ratio_passes(value, "stock", .98, 1.02)
                           for case in report["cases"] for value in case["decode"].values())
        coupled_control = all(ratio_passes(value, "stock", .98, 1.02)
            for case in report["cases"] for row in case["coupled"]
            for value in row["timing"].values())
        controls = pure_control and coupled_control
        candidates = {}
        for arm in ("direct_pair", "row_coalesced"):
            faster = all(ratio_passes(value, arm, 0, .95)
                         for case in report["cases"] for value in case["decode"].values())
            nonregression = all(ratio_passes(value, arm, 0, 1.02)
                for case in report["cases"] for row in case["coupled"]
                for value in row["timing"].values())
            candidates[arm] = {"decode_at_least_5_percent_faster": faster,
                "all_coupled_cases_within_2_percent": nonregression,
                "passed": controls and faster and nonregression}
        report.update(status="complete", correctness_passed=True,
            control_parity_passed=controls, pure_control_parity_passed=pure_control,
            coupled_control_parity_passed=coupled_control, candidates=candidates,
            performance_passed=any(row["passed"] for row in candidates.values()),
            peak_cuda_bytes=torch.cuda.max_memory_allocated(), finished_epoch=time.time())
    except BaseException as exc:
        report.update(status="failed", error=repr(exc), finished_epoch=time.time())
        raise
    finally:
        save()
    print(json.dumps({key: report[key] for key in ("status", "correctness_passed",
        "control_parity_passed", "candidates", "peak_cuda_bytes")}))


if __name__ == "__main__":
    main()
