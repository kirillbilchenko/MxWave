"""Qualify the opaque model adapter against the installed engine's complete MoE block."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from benchmark_moe import capture, load_fixture, same
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig, FusedMoEQuantDesc
from vllm.model_executor.layers.fused_moe.experts import marlin_moe
from vllm.scalar_type import scalar_types


def dispatch_timing(case, run, plugin, trials=7, repeats=20):
    """Pair unprofiled complete-MoE timings in both orders, keeping the same native kernel."""
    result = {}
    for order in (("custom", "dispatcher"), ("dispatcher", "custom")):
        samples = {mode: [] for mode in order}
        for _ in range(trials):
            for mode in order:
                plugin._GEMM_OP = plugin.moe_operation(mode)
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                before = time.perf_counter_ns()
                for _ in range(repeats):
                    run(case)
                submitted = time.perf_counter_ns()
                end.record()
                end.synchronize()
                complete = time.perf_counter_ns()
                samples[mode].append({
                    "cuda_elapsed_us": start.elapsed_time(end) * 1000 / repeats,
                    "host_submit_us": (submitted - before) / repeats / 1000,
                    "wall_us": (complete - before) / repeats / 1000,
                })
        medians = {mode: {metric: statistics.median(s[metric] for s in values)
                          for metric in ("cuda_elapsed_us", "host_submit_us", "wall_us")}
                   for mode, values in samples.items()}
        result["then_".join(order)] = {"samples": samples, "median_us": medians,
            "dispatcher_over_custom": {metric: medians["dispatcher"][metric] /
                                       medians["custom"][metric] for metric in medians["custom"]}}
    return result


def main() -> None:
    """Compare stock/opaque eager, stream and graph outputs with identical sorted routing."""
    import model_plugin

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--mode", choices=("raw", "aligned"), required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dispatch-comparison", action="store_true")
    args = parser.parse_args()
    os.environ["MXWAVE_MOE_MODE"] = args.mode
    model_plugin.install()
    torch.set_num_threads(1)
    torch.manual_seed(321)
    torch.cuda.set_per_process_memory_fraction(1.5 * 1024**3 /
                                              torch.cuda.get_device_properties(0).total_memory)
    manifest = json.loads((args.fixture / "manifest.json").read_bytes())
    # The archived fixture contains all 288 experts. This integration check
    # needs sixteen resident experts; full-model coverage is a separate gate.
    manifest["expert_ids"] = manifest["expert_ids"][:16]
    original_repack = ops.gptq_marlin_repack

    def repack(weight, k, n, bits, is_a_8bit):
        # This pinned installed engine added an explicit permutation argument.
        # No activation ordering is used for the source MXFP4 checkpoint.
        empty = torch.empty(0, dtype=torch.int32, device=weight.device)
        return original_repack(weight, empty, k, n, bits, is_a_8bit)

    ops.gptq_marlin_repack = repack
    try:
        w13, w2, s13, s2 = load_fixture(args.fixture, manifest, 2, 0, torch.device("cuda", 0))
    finally:
        ops.gptq_marlin_repack = original_repack
    layer = torch.nn.Module()
    layer.register_parameter("w13_weight_scale", torch.nn.Parameter(s13, requires_grad=False))
    layer.register_parameter("w2_weight_scale", torch.nn.Parameter(s2, requires_grad=False))
    config = FusedMoEQuantConfig(
        _a1=FusedMoEQuantDesc(), _a2=FusedMoEQuantDesc(),
        _w1=FusedMoEQuantDesc(scale=layer.w13_weight_scale),
        _w2=FusedMoEQuantDesc(scale=layer.w2_weight_scale))
    method = SimpleNamespace(use_cutlass_mxfp4=False,
                             moe=SimpleNamespace(layer_name="fixture.layers.23.mlp.experts"),
                             moe_quant_config=config, moe_kernel=None)
    align = marlin_moe.moe_align_block_size
    cases = []
    for m in ((1, 4, 8, 64, 128) if args.dispatch_comparison else (1, 64, 128)):
        for concentrated in (False, True):
            x = torch.randn((m, 4096), dtype=torch.bfloat16, device="cuda") * .01
            ids = torch.randn((m, 16), device="cuda").topk(8, dim=-1).indices.to(torch.int32)
            if concentrated:
                ids[:] = torch.arange(8, device="cuda", dtype=torch.int32)
            weights = torch.full((m, 8), 1 / 8, device="cuda", dtype=torch.float32)
            block = next(b for b in (8, 16, 32, 48, 64) if m * 8 / 288 / b < .9)
            routing = align(ids, block, 288, None, ignore_invalid_experts=True)
            cases.append({"m": m, "concentrated": concentrated, "x": x,
                          "ids": ids, "weights": weights, "routing": routing})

    def run(case):
        marlin_moe.moe_align_block_size = lambda *a, **kw: case["routing"]
        return marlin_moe.fused_marlin_moe(
            hidden_states=case["x"], w1=w13, w2=w2, bias1=None, bias2=None,
            w1_scale=config.w1_scale, w2_scale=config.w2_scale,
            topk_weights=case["weights"], topk_ids=case["ids"],
            quant_type_id=scalar_types.float4_e2m1f.id, global_num_experts=288,
            activation=MoEActivation.SILU)

    for case in cases:
        case["stock"] = run(case).clone()
    model_plugin.prepare_layer(method, layer, args.mode)
    results = []
    selected = model_plugin._GEMM_OP
    try:
        for case in cases:
            modes = ("custom", "dispatcher") if args.dispatch_comparison else (None,)
            checked = []
            for dispatch in modes:
                model_plugin._GEMM_OP = selected if dispatch is None else \
                    model_plugin.moe_operation(dispatch)
                if not same(run(case), case["stock"]):
                    raise RuntimeError(f"Opaque eager MoE output differs: M={case['m']}")
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    output = run(case).clone()
                stream.synchronize()
                if not same(output, case["stock"]):
                    raise RuntimeError("Opaque non-default stream output differs")
                graph, output = capture(lambda case=case: run(case))
                for _ in range(4):
                    graph.replay()
                torch.cuda.synchronize()
                if not same(output, case["stock"]):
                    raise RuntimeError("Opaque CUDA-graph MoE output differs")
                checked.append(dispatch or os.environ.get("MXWAVE_MOE_DISPATCH", "custom"))
                del graph, output
            item = {"tokens": case["m"], "concentrated": case["concentrated"],
                    "stock_eager_stream_graph_bits_exact": True, "dispatch_modes": checked}
            if args.dispatch_comparison:
                item["dispatch_timing"] = dispatch_timing(case, run, model_plugin)
            results.append(item)
        model_plugin._GEMM_OP = selected
        report = {"status": "complete", "mode": args.mode, "tp": 2, "rank": 0,
                  "experts_resident": 16, "routing_experts": 288,
                  "comparison": "Installed engine complete MoE block, identical sorted alignment",
                  "cases": results, "allocation": model_plugin.report(),
                  "torch": torch.__version__, "vllm": importlib.metadata.version("vllm"),
                  "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in Path(__file__).parent.glob("*.py")},
                  "library_sha256": {
                      key: hashlib.sha256(Path(os.environ[key]).read_bytes()).hexdigest()
                      for key in ("MXWAVE_MOE_RAW_LIBRARY", "MXWAVE_MOE_ALIGNED_LIBRARY")},
                  "limitations": ["Integration fixture, not full-model quality or throughput.",
                                  "Both fixture comparison arms intentionally remain resident."]}
        if args.dispatch_comparison:
            report["limitations"].extend([
                "Timing is a resident 16-expert component, not full-model throughput.",
                "CUDA elapsed and host submission times include queue stalls and must not be added.",
                "Only the host registration differs; kernel and scratch are identical.",
            ])
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        model_plugin._GEMM_OP = selected
        marlin_moe.moe_align_block_size = align


if __name__ == "__main__":
    main()
