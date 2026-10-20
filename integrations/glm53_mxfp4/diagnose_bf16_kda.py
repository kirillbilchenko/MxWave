"""Qualify selective KDA storage through the actual pinned vLLM TP2 linear classes."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import inspect
import json
import os
import time
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch
from diagnose_bf16_attention import write
from diagnose_bf16_overlap import capture
from kda_policy import KDA_PARTS, load_policy
from prepare_bf16_attention import rebuild_weight, tensor_digest
from safetensors import safe_open


def compare(actual: torch.Tensor, expected: torch.Tensor, *, exact: bool) -> dict[str, Any]:
    """Require exact fallback bytes or a predeclared 0.1% relative-L2 decode error bound."""
    a, b = actual.detach().cpu(), expected.detach().cpu()
    bits = a.shape == b.shape and torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    error = a.float() - b.float()
    relative = float(error.norm() / b.float().norm().clamp_min(1e-30)) if finite else None
    result = {"bits_exact": bits, "finite": finite, "relative_l2": relative,
              "passed": bits if exact else relative is not None and relative <= .001}
    if not result["passed"]:
        raise RuntimeError(f"Selective KDA arithmetic gate failed: {result}")
    return result


def result_tensor(value: Any) -> torch.Tensor:
    """Verify the actual vLLM return-bias contract rather than hiding tuple changes."""
    if not isinstance(value, tuple) or len(value) != 2 or value[1] is not None:
        raise RuntimeError("Bias-free vLLM linear must return (output, None)")
    return value[0]


def main() -> None:
    """Exercise real stock loading, replicated shards, release, streams, graphs and timing."""
    from benchmark_bf16_head import _paired
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor import parameter
    from vllm.model_executor.layers import linear
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )
    from vllm.model_executor.model_loader.utils import configure_quant_config
    from vllm.models.glm5next.nvidia.kda import _Glm5NextMergedColumnParallelLinear
    from vllm.models.glm5next.nvidia.model import Glm5NextForConditionalGeneration

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--fused-max-tokens", type=int, choices=(1, 8), default=1)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Preserve prior selective KDA qualification attempts")
    policy = load_policy(args.source)
    os.environ.update(MXWAVE_GLM53_ATTENTION_MAPPING="1", MXWAVE_GLM53_KDA_MODE="packed",
                      MXWAVE_GLM53_KDA_SOURCE=str(args.source),
                      MXWAVE_GLM53_KDA_FUSED_MAX_TOKENS=str(args.fused_max_tokens))
    import model_plugin

    model_plugin.install()
    import kda_adapter
    import kda_runtime

    torch.set_num_threads(2)
    torch.manual_seed(739)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(
        1.5 * 1024**3 / torch.cuda.get_device_properties(device).total_memory, device)
    config = json.loads((args.source / "config.json").read_bytes())
    index = json.loads((args.source / "model.safetensors.index.json").read_bytes())["weight_map"]
    quant = CompressedTensorsConfig.from_config(copy.deepcopy(config["quantization_config"]))
    configure_quant_config(quant, Glm5NextForConditionalGeneration)
    report: dict[str, Any] = {
        "status": "running", "cases": [], "rejections": [], "torch": torch.__version__,
        "fused_max_tokens": args.fused_max_tokens,
        "vllm": importlib.metadata.version("vllm"), "selected_prefixes": sorted(policy),
        "gates": {"source_and_restored_gpu_weight_bits_exact": True,
                  "maximum_single_token_relative_l2": .001, "fallback_output_bits_exact": True,
                  "maximum_shared_pool_bytes": 256 * 1024**2,
                  "fused_2_to_8_tokens_output_bits_exact": args.fused_max_tokens == 8},
        "code_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                        for name in ("kda_policy.py", "kda_runtime.py", "kda_adapter.py",
                                     "model_plugin.py", "diagnose_bf16_kda.py")},
        "runtime_source_sha256": {"linear": hashlib.sha256(
            Path(inspect.getfile(linear)).read_bytes()).hexdigest()},
        "limits": ["Real vLLM linear classes; TP2 source slices tested sequentially on one GPU.",
                   "Rank/world-size queries are patched only inside this private fixture.",
                   "Row-parallel collective reduction is disabled; full TP2 remains a gate.",
                   "Random and basis activations, not captured model activations.",
                   "Both dense control and packed candidate coexist for paired timing.",
                   "Components at layers 0, 22 and 44; whole-model quality is unqualified."]}
    write(args.report, report)
    started = time.monotonic()
    capture_stream = torch.cuda.Stream()
    try:
        for rank in (0, 1):
            with ExitStack() as stack:
                stack.enter_context(set_current_vllm_config(VllmConfig()))
                for module in (linear, parameter):
                    stack.enter_context(patch.object(module, "get_tensor_model_parallel_rank",
                                                     return_value=rank))
                    stack.enter_context(patch.object(module, "get_tensor_model_parallel_world_size",
                                                     return_value=2))
                for layer_id in (0, 22, 44):
                    for kind in ("input", "output"):
                        suffix = "in_proj_qkvbfg_a" if kind == "input" else "o_proj"
                        prefix = f"language_model.model.layers.{layer_id}.self_attn.{suffix}"
                        target = policy[prefix]
                        with torch.device(device):
                            if kind == "input":
                                layer = _Glm5NextMergedColumnParallelLinear(
                                    4096, [8192, 8192, 8192, 64, 128, 128],
                                    replicated_shard_ids=(4, 5), tp_size=2, bias=False,
                                    quant_config=quant, params_dtype=torch.bfloat16, prefix=prefix)
                            else:
                                layer = linear.RowParallelLinear(
                                    8192, 4096, bias=False, input_is_parallel=True,
                                    reduce_results=False, quant_config=quant,
                                    params_dtype=torch.bfloat16, prefix=prefix)
                        if not isinstance(layer.quant_method, kda_adapter.KdaLinearMethod):
                            raise TypeError("The actual vLLM constructor missed selective dispatch")
                        if tuple(layer.weight.shape) != (target.rows, target.width):
                            raise ValueError("The actual vLLM constructor used an unexpected TP shape")
                        try:
                            layer.quant_method.process_weights_after_loading(layer)
                        except ValueError:
                            pass
                        else:
                            raise RuntimeError("Incomplete source loading was accepted")
                        case: dict[str, Any] = {"prefix": prefix, "rank": rank,
                            "runtime_class": type(layer).__name__, "measurements": []}
                        source_parts = []
                        param = layer.weight
                        if rank == 0 and layer_id == 0 and kind == "input":
                            try:
                                param.weight_loader(param, torch.empty((1, 1)), 0)
                            except ValueError:
                                report["rejections"].append("implicit_source_dtype_cast")
                            else:
                                raise RuntimeError("The source loader accepted a dtype cast")
                        for part_id, part in enumerate(KDA_PARTS if kind == "input" else ("o_proj",)):
                            key = f"model.language_model.layers.{layer_id}.self_attn.{part}.weight"
                            with safe_open(args.source / index[key], framework="pt",
                                           device="cpu") as handle:
                                value = handle.get_tensor(key)
                                shard = part_id if kind == "input" else None
                                if kind == "input":
                                    param.weight_loader(param, value, shard)
                                    if rank == 0 and layer_id == 0 and part_id == 0:
                                        try:
                                            param.weight_loader(param, value, shard)
                                        except ValueError:
                                            report["rejections"].append("duplicate_fused_source")
                                        else:
                                            raise RuntimeError("Duplicate source loading was accepted")
                                    rows = value.shape[0] // 2 if part_id < 4 else value.shape[0]
                                    first = rank * rows if part_id < 4 else 0
                                    source_parts.append({"name": key, "axis": 0,
                                                         "range": [first, first + rows]})
                                else:
                                    param.weight_loader(param, value)
                                    source_parts.append({"name": key, "axis": 1,
                                                         "range": [rank * 4096, (rank + 1) * 4096]})
                            del value
                        del param
                        expected_cpu = rebuild_weight(args.source, index,
                                                      {"source_tensors": source_parts})
                        expected_sha = tensor_digest(expected_cpu)
                        if tensor_digest(layer.weight.detach().cpu()) != expected_sha:
                            raise RuntimeError("Real TP shard loader differs from independent source")
                        del expected_cpu
                        dense_weight = layer.weight.detach().clone()
                        layer.quant_method.process_weights_after_loading(layer)
                        gc.collect()
                        if (layer.weight is not None or layer.mxwave_kda_original_parameter() is not None
                                or layer.mxwave_kda_qualification["loaded_weight_sha256"] != expected_sha):
                            raise RuntimeError("Prepared KDA retains a dense parameter or wrong source")
                        case.update(qualification=layer.mxwave_kda_qualification,
                                    dense_parameter_released=True, independent_source_bits_exact=True)
                        report["cases"].append(case)
                        reference_method = linear.UnquantizedLinearMethod()
                        reference_layer = SimpleNamespace(weight=dense_weight)
                        sizes = ((0, 1, 8, 128, 512, 2048) if layer_id == 0 else (1, 8, 128))
                        if args.fused_max_tokens == 8:
                            sizes = ((0, 1, 2, 3, 4, 7, 8, 128, 512, 2048) if layer_id == 0
                                     else (1, 2, 3, 4, 7, 8, 128))
                        for tokens in sizes:
                            x = torch.randn((tokens, 4096), device=device, dtype=torch.bfloat16) * .1
                            baseline = partial(reference_method.apply, reference_layer, x)
                            candidate = partial(layer.quant_method.apply, layer, x)
                            actual = result_tensor(layer(x))
                            ref = baseline()
                            record: dict[str, Any] = {"tokens": tokens,
                                "eager": compare(actual, ref, exact=tokens != 1)}
                            del actual, ref
                            if tokens:
                                dense_graph, expected = capture(baseline, capture_stream)
                                packed_graph, output = capture(candidate, capture_stream)
                                record["graph"] = compare(output, expected, exact=tokens != 1)
                                before = torch.cuda.memory_allocated()
                                if tokens != 1:
                                    layer.mxwave_kda_pool.fill_(float("nan"))
                                for _ in range(3):
                                    packed_graph.replay()
                                record["replay"] = compare(output, expected, exact=tokens != 1)
                                record["replay_allocation_growth_bytes"] = (
                                    torch.cuda.memory_allocated() - before)
                                if record["replay_allocation_growth_bytes"]:
                                    raise RuntimeError("Repeated replay grew adapter storage")
                                with torch.cuda.stream(capture_stream):
                                    streamed = candidate()
                                torch.cuda.current_stream().wait_stream(capture_stream)
                                record["nondefault_stream"] = compare(streamed, expected,
                                                                     exact=tokens != 1)
                                del streamed
                                timed = ((1, 2, 3, 4, 7, 8, 128, 2048)
                                         if args.fused_max_tokens == 8 else (1, 128, 2048))
                                if tokens in timed:
                                    record["graph_timing"] = _paired(
                                        dense_graph.replay, packed_graph.replay, 5, 10,
                                        interleaved=True)
                                    record["eager_timing"] = _paired(baseline, candidate, 5, 5,
                                                                    interleaved=True)
                                del dense_graph, packed_graph, expected, output
                            case["measurements"].append(record)
                            write(args.report, report)
                            del x, baseline, candidate
                            torch.cuda.synchronize()
                            gc.collect()
                            torch.cuda.empty_cache()
                        # Real wrapper must preserve return_bias=False as well.
                        layer.return_bias = False
                        x = torch.randn((1, 4096), device=device, dtype=torch.bfloat16)
                        direct = layer(x)
                        if not isinstance(direct, torch.Tensor):
                            raise TypeError("vLLM return_bias=False contract changed")
                        compare(direct, reference_method.apply(reference_layer, x), exact=False)
                        case["return_bias_false_passed"] = True
                        if rank == 0 and layer_id == 0 and kind == "input":
                            compiled = torch.compile(layer, backend="eager", fullgraph=True,
                                                     dynamic=True)
                            case["dynamo_fullgraph"] = []
                            for count in (1, 3):
                                dynamic_x = torch.randn((count, 4096), device=device,
                                                        dtype=torch.bfloat16) * .1
                                case["dynamo_fullgraph"].append({
                                    "tokens": count, **compare(compiled(dynamic_x),
                                        reference_method.apply(reference_layer, dynamic_x),
                                        exact=count != 1)})
                            del compiled, dynamic_x
                            torch._dynamo.reset()
                        del direct, x, layer, dense_weight, reference_layer, reference_method
                        torch.cuda.synchronize()
                        gc.collect()
                        torch.cuda.empty_cache()
        report.update(status="passed", passed=True, seconds=time.monotonic() - started,
                      allocation=kda_runtime.report())
        write(args.report, report)
        print(json.dumps({"status": report["status"], "seconds": report["seconds"],
                          "prepared_layers": report["allocation"]["prepared_layers"],
                          "pool_bytes": report["allocation"]["pool_bytes"],
                          "peak": report["allocation"]["cuda_peak_allocated_bytes"]}))
    except BaseException as exc:
        report.update(status="failed", passed=False, error=repr(exc))
        write(args.report, report)
        raise


if __name__ == "__main__":
    main()
