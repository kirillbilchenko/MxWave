"""Compare real-weight grouped MoE raw/control/async paths within declared CUDA caps."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import statistics
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import torch
from audit_checkpoint import digest, write_json
from moe_runtime import NativeMoe, PackedScales, allocate_buffers, pack_scales
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_permute_scales
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    mxfp4_marlin_process_scales,
)
from vllm.scalar_type import scalar_types


def same(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Compare floating tensors by their raw bytes, including signed zero."""
    return bool(torch.equal(a.view(torch.uint8), b.view(torch.uint8)))


def difference(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    """Describe a failed equality check without silently relaxing its contract."""
    changed = a.view(torch.int16) != b.view(torch.int16)
    error = (a.float() - b.float()).abs()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    return {"bits_exact": not bool(changed.any()), "changed_values": int(changed.sum()),
            "values": a.numel(), "finite": finite,
            "max_abs_error": float(error.max()) if finite else None,
            "relative_l2": float(torch.linalg.vector_norm(error) /
                                 torch.linalg.vector_norm(b.float())) if finite else None}


def load_fixture(
    directory: Path, manifest: dict[str, Any], tp: int, rank: int, device: torch.device,
    *, current_api: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Use stock repacking and permutation on exact source bytes for one TP partition."""
    records = {(t["expert"], t["projection"], t["suffix"]): t for t in manifest["tensors"]}
    count, hidden, intermediate = len(manifest["expert_ids"]), 4096, 2048 // tp
    lo, hi = rank * intermediate, (rank + 1) * intermediate

    def source(expert: int, projection: str, suffix: str) -> torch.Tensor:
        record = records[expert, projection, suffix]
        raw = (directory / record["path"]).read_bytes()
        if digest(raw) != record["sha256"]:
            raise ValueError("Full expert fixture payload changed")
        value = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(record["shape"])
        if projection == "down_proj":
            stride = 32 if suffix == "weight_scale" else 2
            return value[:, lo // stride : hi // stride].contiguous()
        return value[lo:hi].contiguous()

    weights, scales = [], []
    for family in ("w13", "w2"):
        n, k = (intermediate * 2, hidden) if family == "w13" else (hidden, intermediate)
        weight = torch.empty((count, k // 16, n * 2), dtype=torch.int32, device=device)
        scale = torch.empty((count, k // 32, n), dtype=torch.float8_e8m0fnu, device=device)
        for i, expert in enumerate(manifest["expert_ids"]):
            def projection(suffix: str, expert: int = expert, family: str = family) -> torch.Tensor:
                if family == "w13":
                    return torch.cat([source(expert, p, suffix)
                                      for p in ("gate_proj", "up_proj")], dim=0)
                return source(expert, "down_proj", suffix)

            raw_weight = projection("weight_packed").to(device)
            qweight = raw_weight.view(torch.int32).T.contiguous()
            if current_api:
                perm = torch.empty(0, dtype=torch.int32, device=device)
                repacked = ops.gptq_marlin_repack(qweight, perm, k, n, 4, False)
            else:
                repacked = ops.gptq_marlin_repack(qweight, k, n, 4, False)
            weight[i].copy_(repacked)
            raw_scale = projection("weight_scale").to(device)
            if (raw_scale == 255).any():
                raise ValueError("Reserved E8M0 NaN in real expert scales")
            floating = raw_scale.view(torch.float8_e8m0fnu).to(torch.bfloat16)
            permuted = marlin_permute_scales(floating.T, k, n, 32)
            scale[i].copy_(mxfp4_marlin_process_scales(permuted))
            del raw_weight, qweight, repacked, raw_scale, floating, permuted
        weights.append(weight)
        scales.append(scale)
    return weights[0], weights[1], scales[0], scales[1]


def capture(operation: Callable[[], torch.Tensor]) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    """Warm a side stream and capture one complete routed expert block."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            operation()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = operation()
    torch.cuda.current_stream().wait_stream(stream)
    return graph, output


def paired(
    raw: Callable[[], Any], compressed: Callable[[], Any], trials: int = 7, repeats: int = 20
) -> dict[str, Any]:
    """Record both arm orders with CUDA events, retaining individual trial values."""
    result: dict[str, Any] = {}
    for order in (("raw", "async"), ("async", "raw")):
        samples: dict[str, list[float]] = {"raw": [], "async": []}
        for _ in range(trials):
            for name in order:
                operation = raw if name == "raw" else compressed
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(repeats):
                    operation()
                end.record()
                end.synchronize()
                samples[name].append(start.elapsed_time(end) * 1000 / repeats)
        medians = {name: statistics.median(v) for name, v in samples.items()}
        result["then_".join(order)] = {
            "samples_us": samples, "median_us": medians,
            "async_over_raw": medians["async"] / medians["raw"],
        }
    return result


def profile(operation: Callable[[], Any], path: Path) -> dict[str, Any]:
    """Save a warmed torch trace and an explicit kernel-duration breakdown."""
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as profiler:
        for _ in range(8):
            operation()
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(path))
    events = json.loads(path.read_bytes())["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"]
    names: dict[str, dict[str, float]] = {}
    for event in kernels:
        entry = names.setdefault(event["name"], {"calls": 0, "total_us": 0})
        entry["calls"] += 1
        entry["total_us"] += event["dur"]
    return {"file": path.name, "sha256": digest(path.read_bytes()), "repeats": 8,
            "kernel_count": len(kernels), "kernel_sum_us": sum(e["dur"] for e in kernels),
            "kernels": names}


def check_decoder(native: NativeMoe, codec: Path, device: torch.device) -> dict[str, Any]:
    """Cover all 256 byte patterns, global escape indexes, streams and graph replay."""
    def stock_bf16_bits(raw: torch.Tensor) -> torch.Tensor:
        # Independently construct stock's pair ordering and exponent bit mapping.
        reordered = raw.reshape(-1, 4)[:, [0, 2, 1, 3]].to(torch.int32)
        bits16 = (reordered << 7).to(torch.int16).contiguous()
        return bits16.view(torch.uint8).reshape(*raw.shape[:-1], raw.shape[-1] * 2)

    if native.encoding in ("nibbles", "aligned_nibbles"):
        def verify_nibbles(raw: torch.Tensor) -> PackedScales:
            packed = pack_scales(raw.view(torch.float8_e8m0fnu), codec, encoding=native.encoding)
            if not same(native.decode(packed), raw):
                raise RuntimeError("Exact nibble GPU bytes differ")
            if not same(native.decode_bf16(packed), stock_bf16_bits(raw)):
                raise RuntimeError("Exact nibble BF16 fragment bits differ")
            return packed

        for base in range(0, 256, 16):
            values = torch.arange(16, dtype=torch.uint8, device=device) + base
            verify_nibbles(values.repeat(4096).reshape(1, 4096, 16))
        words = torch.arange(65536, dtype=torch.int32, device=device).reshape(-1, 1)
        values = ((words >> (torch.arange(4, device=device) * 4)) & 15).to(torch.uint8)
        base = 112 if native.encoding == "aligned_nibbles" else 120
        expected = (values.repeat(1, 4) + base).reshape(2, 32768, 16)
        packed = verify_nibbles(expected)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            streamed = native.decode(packed)
            streamed_bf16 = native.decode_bf16(packed)
        stream.synchronize()
        if (not same(streamed, expected)
                or not same(streamed_bf16, stock_bf16_bits(expected))):
            raise RuntimeError("Non-default stream nibble decoding differs")
        for operation, reference in ((native.decode, expected),
                                     (native.decode_bf16, stock_bf16_bits(expected))):
            graph, output = capture(lambda operation=operation: operation(packed))
            for _ in range(4):
                graph.replay()
            torch.cuda.synchronize()
            if not same(output, reference):
                raise RuntimeError("Captured nibble decoder differs")
        upper = torch.tensor([[253, 254, 255, 253] * 4, [254, 255] * 8, [255] * 16],
                             dtype=torch.uint8, device=device).reshape(1, 3, 16)
        verify_nibbles(upper)
        bad = torch.tensor([0, 16] + [0] * 14, dtype=torch.uint8, device=device).reshape(1, 1, 16)
        try:
            pack_scales(bad.view(torch.float8_e8m0fnu), codec, encoding=native.encoding)
        except ValueError:
            pass
        else:
            raise RuntimeError("Nibble packer accepted a non-representable exponent span")
        if native.encoding == "aligned_nibbles":
            crossing = torch.tensor([15, 16] * 8, dtype=torch.uint8, device=device).reshape(1, 1, 16)
            try:
                pack_scales(crossing.view(torch.float8_e8m0fnu), codec, encoding=native.encoding)
            except ValueError:
                pass
            else:
                raise RuntimeError("Aligned nibble packer accepted values crossing its base bucket")
        wrong = pack_scales(torch.zeros_like(bad).view(torch.float8_e8m0fnu), codec)
        try:
            native.decode(wrong)
        except ValueError:
            pass
        else:
            raise RuntimeError("Native decoder accepted the wrong scale encoding")
        return {"encoding": native.encoding, "all_byte_patterns": True, "escape_vectors": 0,
                "all_65536_nibble_words": True, "eager_stream_graph_exact": True,
                "upper_base_253_to_255_exact": True, "stock_bf16_fragment_bits_exact": True,
                "span_over_15_and_wrong_encoding_rejected": True}

    # Cross the CPU packer's 1 MiB chunk boundary with both normal and escape vectors.
    bits = torch.arange(256, dtype=torch.int32).to(torch.uint8).repeat(8192).reshape(4, 32768, 16)
    bits[:, :16384] = torch.arange(16, dtype=torch.uint8).remainder(4).reshape(1, 1, 16) + 120
    scales = bits.to(device).view(torch.float8_e8m0fnu)
    packed = pack_scales(scales, codec)
    if not same(native.decode(packed), scales.view(torch.uint8)):
        raise RuntimeError("Synthetic GPU scale bytes differ")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        streamed = native.decode(packed)
    stream.synchronize()
    if not same(streamed, scales.view(torch.uint8)):
        raise RuntimeError("Non-default stream decoder differs")
    graph, output = capture(lambda: native.decode(packed))
    for _ in range(4):
        graph.replay()
    torch.cuda.synchronize()
    if not same(output, scales.view(torch.uint8)):
        raise RuntimeError("Captured GPU decoder differs")
    if native.has_bf16_decoder:
        expected_bf16 = stock_bf16_bits(scales.view(torch.uint8))
        if not same(native.decode_bf16(packed), expected_bf16):
            raise RuntimeError("Stock BF16 fragment bits differ")
        bf16_graph, bf16_output = capture(lambda: native.decode_bf16(packed))
        bf16_graph.replay()
        torch.cuda.synchronize()
        if not same(bf16_output, expected_bf16):
            raise RuntimeError("Captured stock BF16 fragment bits differ")
    # Exhaust the 16-bit selector word used by the optimized PRMT decoder.
    words = torch.arange(65536, dtype=torch.int32, device=device).reshape(-1, 1)
    shifts = torch.arange(8, dtype=torch.int32, device=device) * 2
    values = ((words >> shifts) & 3).to(torch.uint8) + 120
    normal = torch.cat((values, values.flip((-1,))), dim=1).reshape(2, 32768, 16)
    upper = torch.tensor([[253, 254, 255, 253] * 4, [254, 255] * 8, [255] * 16],
                         dtype=torch.uint8, device=device).reshape(1, 3, 16)
    for expected in (normal, upper):
        packed_normal = pack_scales(expected.view(torch.float8_e8m0fnu), codec)
        if not same(native.decode(packed_normal), expected):
            raise RuntimeError("Optimized scale selector decoding differs")
        if native.has_bf16_decoder and not same(
            native.decode_bf16(packed_normal), stock_bf16_bits(expected)
        ):
            raise RuntimeError("BF16 fragment selector or upper-base bits differ")
    return {"all_byte_patterns": True, "escape_vectors": packed.exceptions.numel() // 16,
            "eager_stream_graph_exact": True, "all_65536_selector_words": True,
            "upper_base_253_to_255_exact": True,
            "stock_bf16_fragment_bits_exact": True if native.has_bf16_decoder else None}


def check_case(
    fixture: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    compressed: tuple[PackedScales, PackedScales], raw_native: NativeMoe, async_native: NativeMoe,
    m: int, skewed: bool, trace_dir: Path | None, *, timings: bool = True,
    diagnostic_path: Path | None = None,
    block_override: int | None = None,
    control_scales: tuple[PackedScales, PackedScales] | None = None,
    timing_function: Callable[..., dict[str, Any]] | None = None,
    current_api: bool = False,
    reference_native: NativeMoe | None = None,
) -> dict[str, Any]:
    """Check stock/control/candidate exactness and paired latency for real grouped GEMMs."""
    w13, w2, s13, s2 = fixture
    device, e, k = w13.device, w13.shape[0], 4096
    intermediate = w2.shape[1] * 16
    topk = 8
    block = block_override or next((b for b in (8, 16, 32, 48, 64)
                                   if m * topk / e / b < .9), 64)
    generator = torch.Generator().manual_seed(73 + m)
    logits = torch.randn((m, e), generator=generator)
    ids = logits.topk(topk, dim=-1).indices.to(torch.int32)
    if skewed:
        ids[:] = torch.arange(topk, dtype=torch.int32)
    top_ids = ids.to(device)
    routing_weights = torch.softmax(torch.randn(m, topk, generator=generator), -1).to(device)
    capacity = min(m * topk + e * (block - 1), m * topk * block)
    routing = (torch.empty(capacity, dtype=torch.int32, device=device),
               torch.empty((capacity + block - 1) // block, dtype=torch.int32, device=device),
               torch.empty(1, dtype=torch.int32, device=device))

    def route() -> None:
        ops.moe_align_block_size(top_ids, e, block, *routing, *([None] if current_api else [None, None]))

    route()
    x = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    n13 = intermediate * 2
    plans = (raw_native.plan(m, n13, k, block, topk, e, device.index),
             raw_native.plan(m * topk, k, intermediate, block, 1, e, device.index))
    arms = []
    arm_specs = [(raw_native, control_scales or (s13, s2)), (async_native, compressed)]
    if reference_native is not None:
        if control_scales is None:
            raise ValueError("A current-codec reference requires its packed control scales")
        arm_specs.append((reference_native, control_scales))
    for native, scales in arm_specs:
        # Each concurrently exercised arm owns its routing writes as well as GEMM scratch.
        arm_routing = tuple(torch.empty_like(t) for t in routing)
        arm_route = partial(ops.moe_align_block_size, top_ids, e, block, *arm_routing,
                            *([None] if current_api else [None, None]))
        arm_route()
        up = allocate_buffers(m, n13, topk, block, capacity, device)
        down = allocate_buffers(m * topk, k, 1, block, capacity, device)
        activation = torch.empty((m * topk, intermediate), dtype=torch.bfloat16, device=device)
        output = torch.empty_like(x)
        first = native.bind(x, w13, scales[0], up, arm_routing, routing_weights,
                            topk, block, False, plans[0])
        second = native.bind(activation, w2, scales[1], down, arm_routing, routing_weights,
                             1, block, True, plans[1])

        def operation(update_route: bool = True,
                      first: Any = first, second: Any = second, up: Any = up,
                      down: Any = down, activation: Any = activation,
                      output: Any = output, arm_route: Any = arm_route) -> torch.Tensor:
            if update_route:
                arm_route()
            first()
            torch.ops._C.silu_and_mul(activation, up.output)
            second()
            torch.sum(down.output.view(m, topk, k), dim=1, out=output)
            return output

        arms.append((operation, up, down, activation, output, arm_routing))
    reference_arm = arms.pop() if reference_native is not None else None
    stock_lock = torch.zeros_like(arms[0][1].locks)

    def stock_reference(alignment: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if current_api:
            # The pinned image's stable API adds explicit activation/global scales,
            # permutation and zero-point kind. Keep the legacy experiment calls below intact.
            common = {"b_bias": None, "a_scales": None, "global_scale": None, "b_qzeros": None,
                "g_idx": None, "perm": None, "workspace": stock_lock, "sorted_token_ids": alignment[0],
                "expert_ids": alignment[1], "num_tokens_past_padded": alignment[2],
                "topk_weights": routing_weights, "moe_block_size": block,
                "b_q_type": scalar_types.float4_e2m1f, "is_k_full": False, "use_atomic_add": True,
                "use_fp32_reduce": False, "is_zp_float": False}
            up = ops.moe_wna16_marlin_gemm(input=x, output=None, b_qweight=w13, b_scales=s13,
                top_k=topk, mul_topk_weights=False, size_m=m, size_n=n13, size_k=k,
                thread_k=plans[0][0], thread_n=plans[0][1], blocks_per_sm=plans[0][2], **common)
            activation = torch.empty_like(arms[0][3])
            torch.ops._C.silu_and_mul(activation, up)
            down = ops.moe_wna16_marlin_gemm(input=activation, output=None, b_qweight=w2,
                b_scales=s2, top_k=1, mul_topk_weights=True, size_m=m * topk, size_n=k,
                size_k=intermediate, thread_k=plans[1][0], thread_n=plans[1][1],
                blocks_per_sm=plans[1][2], **common)
            return up, down, down.view(m, topk, k).sum(dim=1)
        up = ops.moe_wna16_marlin_gemm(
            x, None, w13, None, s13, None, None, None, stock_lock,
            *alignment, routing_weights, block, topk, False, scalar_types.float4_e2m1f,
            m, n13, k, False, True, False, *plans[0],
        )
        activation = torch.empty_like(arms[0][3])
        torch.ops._C.silu_and_mul(activation, up)
        down = ops.moe_wna16_marlin_gemm(
            activation, None, w2, None, s2, None, None, None, stock_lock,
            *alignment, routing_weights, block, 1, True, scalar_types.float4_e2m1f,
            m * topk, k, intermediate, False, True, False, *plans[1],
        )
        return up, down, down.view(m, topk, k).sum(dim=1)

    def verify(arm: Any, expected: Any, context: str) -> None:
        if not torch.isfinite(expected[2]).all():
            raise RuntimeError("Stock routed expert output is not finite")
        outputs = (arm[1].output, arm[2].output, arm[4])
        if not all(same(a, b) for a, b in zip(outputs, expected, strict=True)):
            diagnostics = {"context": context, "tokens": m, "skewed": skewed,
                           "plans": plans, "difference": {
                               label: difference(a, b) for label, a, b in
                               zip(("up", "down", "output"), outputs, expected, strict=True)}}
            if diagnostic_path is not None:
                write_json(diagnostic_path, diagnostics)
            raise RuntimeError("Grouped output mismatch: " + json.dumps(diagnostics))
        if torch.count_nonzero(arm[1].locks).item() or torch.count_nonzero(arm[2].locks).item():
            raise RuntimeError("Grouped native kernel did not reset its managed locks")

    def arithmetic_reference(alignment: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if reference_arm is None:
            return stock_reference(alignment)
        for target, actual in zip(reference_arm[5], alignment, strict=True):
            target.copy_(actual)
        reference_arm[0](False)
        return reference_arm[1].output, reference_arm[2].output, reference_arm[4]

    reference = arithmetic_reference(routing)
    stock_output = reference[2]
    # The unchanged router may reorder tokens within experts. Fix its sorted output
    # for the direct stock/raw/async arithmetic comparison; do not loosen bit equality.
    for arm in arms:
        for target, original in zip(arm[5], routing, strict=True):
            target.copy_(original)
        arm[0](False)
        verify(arm, reference, "identical sorted routing")
    # Also verify real rerouting on every arm, using that arm's actual alignment in
    # the unchanged stock GEMMs. Timing below retains the normal routing operation.
    for _ in range(3):
        for arm in arms:
            arm[0]()
            verify(arm, arithmetic_reference(arm[5]), "dynamic eager routing")
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for stream, (operation, *_rest) in zip(streams, arms, strict=True):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            operation()
    for stream in streams:
        stream.synchronize()
    for arm in arms:
        verify(arm, arithmetic_reference(arm[5]), "dynamic non-default stream")
    graphs = [capture(a[0]) for a in arms]
    for _ in range(3):
        for graph, _ in graphs:
            graph.replay()
    torch.cuda.synchronize()
    for arm in arms:
        verify(arm, arithmetic_reference(arm[5]), "dynamic captured graph")
    result: dict[str, Any] = {
        "tokens": m, "concentrated_routing": skewed,
        "empty_experts": int((torch.bincount(ids.flatten().long(), minlength=e) == 0).sum()),
        "top_k": topk, "moe_block": block,
        "gate_up_plan": plans[0], "down_plan": plans[1],
        "stock_control_async_eager_stream_graph_bits_exact": True if reference_arm is None else None,
        "current_codec_control_candidate_eager_stream_graph_bits_exact": reference_arm is not None,
        "managed_locks_reset": True,
        "bit_equality_compares_identical_sorted_alignment": True,
        "dynamic_eager_stream_graph_matches_stock_for_own_alignment": True,
        "output_sha256": digest(stock_output.view(torch.uint8).cpu().numpy().tobytes()),
        "routed_counts": torch.bincount(ids.flatten().long(), minlength=e).tolist(),
    }
    if reference_arm is not None:
        qualified = arithmetic_reference(routing)
        stock = stock_reference(routing)
        result["installed_stock_diagnostic"] = {label: {
            "whole_output_bits_exact": same(actual, expected),
            "first_16_rows_difference": difference(actual[:16], expected[:16])}
            for label, actual, expected in zip(("up", "down", "output"), qualified, stock, strict=True)}
        result["arithmetic_reference"] = "original qualified compressed-codec library"
        del stock
    if timings:
        result["graph_latency"] = (timing_function or paired)(graphs[0][0].replay, graphs[1][0].replay)
    if timings and m in (1, 32) and not skewed:
        result["eager_latency"] = (timing_function or paired)(arms[0][0], arms[1][0])
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)
        result["profiles"] = {
            name: profile(arm[0], trace_dir / f"m{m}-{name}.json")
            for name, arm in zip(("raw", "async"), arms, strict=True)
        }
    return result


def main() -> None:
    """Run an isolated expert component, never instantiate or modify a serving engine."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--libraries", type=Path, required=True)
    parser.add_argument("--codec-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--full-layer", action="store_true")
    parser.add_argument("--profile-concentrated", action="store_true")
    parser.add_argument("--focus-regression", action="store_true")
    parser.add_argument("--candidate-library", type=Path)
    parser.add_argument("--prefill-moe-block", type=int, choices=(16, 32, 48, 64))
    args = parser.parse_args()
    if args.profile_concentrated and (
        not args.full_layer or args.diagnostic or args.resume_from is not None
        or args.focus_regression
    ):
        parser.error("Concentrated profiling requires --full-layer without diagnostic/resume")
    if args.focus_regression and (
        not args.full_layer or args.diagnostic or args.resume_from is not None
    ):
        parser.error("Focused optimization requires --full-layer without diagnostic/resume")
    if args.prefill_moe_block is not None and (
        not args.full_layer or args.diagnostic or args.resume_from is not None
    ):
        parser.error("Prefill tile tuning requires --full-layer without diagnostic/resume")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    device = torch.device("cuda", torch.cuda.current_device())
    total = torch.cuda.get_device_properties(device).total_memory
    allocation_cap = int((2.25 if args.full_layer else 1) * 1024**3)
    torch.cuda.set_per_process_memory_fraction(allocation_cap / total, device)
    started = time.monotonic()
    manifest_raw = (args.fixture / "manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    if args.full_layer and manifest["expert_ids"] != list(range(288)):
        raise ValueError("Full-layer qualification requires all 288 source experts")
    codec = args.codec_root / "integrations/marlin_scales/records.py"
    raw_library = args.libraries / "libmxwave_moe_raw.so"
    candidate_library = args.candidate_library or args.libraries / "libmxwave_moe_async.so"
    raw_native = NativeMoe(raw_library)
    async_native = NativeMoe(candidate_library)
    if raw_native.compressed or not async_native.compressed:
        raise ValueError("Expected a raw control and a compressed-scale candidate")
    report: dict[str, Any] = {
        "status": "running", "gpu": torch.cuda.get_device_name(device),
        "purpose": ("concentrated-routing traces only; no repeated latency baseline"
                    if args.profile_concentrated else "focused decoder optimization"
                    if args.focus_regression else "component qualification"),
        "torch": torch.__version__, "script_sha256": digest(Path(__file__).read_bytes()),
        "runtime_sha256": digest(Path(__file__).with_name("moe_runtime.py").read_bytes()),
        "cuda_allocation_cap_bytes": allocation_cap,
        "fixture_manifest_sha256": digest(manifest_raw), "source": manifest,
        "codec_sha256": digest(codec.read_bytes()),
        "libraries_sha256": {p.name: digest(p.read_bytes())
                             for p in (raw_library, candidate_library)},
        "candidate_library": str(candidate_library),
        "scale_encoding": async_native.encoding,
        "prefill_moe_block": args.prefill_moe_block,
        "decoder": check_decoder(async_native, codec, device), "partitions": [],
        "limitations": [
            f"{len(manifest['expert_ids'])} real layer-23 experts; synthetic top-8 routing, no model forward.",
            "TP2 weight partitions tested sequentially on one GPU, without inter-rank collectives.",
            "BF16 activations on both arms; not the source checkpoint's declared native W4A4 path.",
            "Paired latency includes routing, two grouped GEMMs, activation and top-k sum.",
            "Raw control uses preallocated workspaces and matches stock operator output bits.",
            "No whole-model throughput, quality or resident memory qualification.",
        ],
    }
    previous_report = None
    if args.resume_from is not None:
        previous_raw = args.resume_from.read_bytes()
        previous_report = json.loads(previous_raw)
        for key in ("fixture_manifest_sha256", "runtime_sha256", "codec_sha256", "libraries_sha256"):
            if previous_report[key] != report[key]:
                raise ValueError(f"Cannot reuse component results: {key} changed")
        report["resume_evidence"] = {
            "report_sha256": digest(previous_raw),
            "script_sha256": previous_report["script_sha256"],
            "reason": "Native kernels and graph operations unchanged; sorted-routing comparator fixed",
        }
    write_json(args.report, report)
    partitions = (((2, 0),) if args.profile_concentrated or args.focus_regression else
                  ((2, 0), (2, 1)) if args.full_layer else
                  (((1, 0),) if args.diagnostic else ((1, 0), (2, 0), (2, 1))))
    for tp, rank in partitions:
        fixture = load_fixture(args.fixture, manifest, tp, rank, device)
        compressed = tuple(pack_scales(s, codec, encoding=async_native.encoding) for s in fixture[2:])
        for raw, packed in zip(fixture[2:], compressed, strict=True):
            if not same(async_native.decode(packed), raw.view(torch.uint8)):
                raise RuntimeError("Real expert GPU scale reconstruction changed bytes")
        entry: dict[str, Any] = {
            "tp": tp, "rank": rank, "raw_scale_bytes": sum(s.numel() for s in fixture[2:]),
            "compressed_scale_bytes": sum(s.retained_bytes for s in compressed),
            "escape_vectors": sum(s.exceptions.numel() // 16 for s in compressed),
            "cases": [],
        }
        report["partitions"].append(entry)
        previous_partition = next((p for p in previous_report["partitions"]
                                   if p["tp"] == tp and p["rank"] == rank), None
                                  ) if previous_report is not None else None
        if args.diagnostic:
            # Replay only the preceding RNG draws, retaining the exact input that failed
            # in run-v2 without rerunning any of its already recorded smaller cases.
            for preceding_m in (1, 1, 1, 8, 8, 16, 16, 32, 32, 64, 64):
                torch.randn((preceding_m, 4096), device=device, dtype=torch.bfloat16)
        if (tp == 1 or (args.candidate_library is not None and rank == 0)) and not args.diagnostic:
            # Force wider exponent vectors in every expert to exercise the global escape
            # table through GEMM, separately from unchanged real-weight latency cases.
            if previous_partition is not None and "injected_escape_gemm_check" in previous_partition:
                entry["injected_escape_gemm_check"] = previous_partition["injected_escape_gemm_check"]
                torch.randn((1, 4096), device=device, dtype=torch.bfloat16)
            else:
                escaped_scales = []
                # A bounded subset also exercises exceptions in full-layer candidates.
                escaped_count = min(16, fixture[0].shape[0])
                for s in fixture[2:]:
                    s = s[:escaped_count]
                    escaped = s.view(torch.uint8).clone()
                    escaped.view(escaped.shape[0], -1)[:, :16] = torch.tensor(
                        [120, 125] * 8, dtype=torch.uint8, device=device
                    )
                    escaped_scales.append(escaped.view(torch.float8_e8m0fnu))
                escaped_fixture = (fixture[0][:escaped_count], fixture[1][:escaped_count],
                                   *escaped_scales)
                escaped_packed = tuple(pack_scales(s, codec, encoding=async_native.encoding)
                                       for s in escaped_scales)
                entry["injected_escape_gemm_check"] = check_case(
                    escaped_fixture, escaped_packed, raw_native, async_native, 1, False, None,
                    timings=False,
                )
                del escaped_scales, escaped_fixture, escaped_packed
        for m in ((1, 64, 128, 512) if args.focus_regression else
                  (64, 128) if args.profile_concentrated else
                  (1, 16, 64, 128, 512) if args.full_layer else
                  ((128,) if args.diagnostic else
                   ((1, 8, 16, 32, 64, 128) if tp == 1 else (1, 16, 64)))):
            for skewed in ((m != 1,) if args.focus_regression else
                           (True,) if args.profile_concentrated else
                           (False,) if args.diagnostic else (False, True)):
                cached = next((c for c in previous_partition["cases"]
                               if c["tokens"] == m and c.get("concentrated_routing", c.get(
                                   "skewed_eight_empty_experts")) == skewed),
                              None) if previous_partition is not None else None
                if cached is not None:
                    cached["reused_from_report_sha256"] = report["resume_evidence"]["report_sha256"]
                    if "eager_latency" in cached:
                        cached["prior_harness_eager_latency"] = cached.pop("eager_latency")
                    for saved in cached.get("profiles", {}).values():
                        name = saved["file"]
                        if Path(name).name != name:
                            raise ValueError("Cached trace must be a plain filename")
                        original_trace = args.resume_from.parent / "traces" / name
                        if digest(original_trace.read_bytes()) != saved["sha256"]:
                            raise ValueError("Cached trace hash changed")
                        args.traces.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(original_trace, args.traces / name)
                    entry["cases"].append(cached)
                    torch.randn((m, 4096), device=device, dtype=torch.bfloat16)
                    continue
                traces = (args.traces if not skewed and (
                    (tp == 1 and m in (1, 128)) or (args.full_layer and rank == 0 and m in (1, 64))
                ) else None)
                if args.profile_concentrated:
                    traces = args.traces
                if args.focus_regression:
                    traces = args.traces if m == 128 else None
                entry["cases"].append(check_case(fixture, compressed, raw_native, async_native,
                                                  m, skewed, traces,
                                                  timings=not (args.diagnostic or
                                                               args.profile_concentrated),
                                                  block_override=(args.prefill_moe_block
                                                                  if m >= 64 else None),
                                                  diagnostic_path=args.report.with_name(
                                                      f"diagnostic-tp{tp}-r{rank}-m{m}-{skewed}.json")))
                write_json(args.report, report)
                print(f"Grouped MoE: TP{tp} rank {rank}, M={m}, skewed={skewed} passed", flush=True)
        del fixture, compressed, entry
        gc.collect()
        torch.cuda.empty_cache()
    report.update(status="complete", seconds=time.monotonic() - started,
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
                  peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device))
    write_json(args.report, report)
    print(json.dumps({"status": "complete", "seconds": report["seconds"]}))


if __name__ == "__main__":
    main()
