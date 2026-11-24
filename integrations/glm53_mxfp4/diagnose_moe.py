"""Compare stock prepared MXFP4 routed experts with independently decoded CPU weights."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from diagnose_mxfp8 import CONFIG_SHA, INDEX_SHA, difference, memory, write
from safetensors import safe_open


def decode(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Decode low-nibble-first OCP E2M1 with independent power-of-two scales."""
    if (packed.dtype != torch.uint8 or scales.dtype != torch.uint8 or packed.ndim != 2
            or packed.shape[1] % 16 or scales.shape != (packed.shape[0], packed.shape[1] // 16)
            or bool((scales == 255).any())):
        raise ValueError("Expected finite group-32 packed E2M1/E8M0 tensors")
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(packed.shape[0], -1)
    levels = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6],
                          dtype=torch.float32)
    values = levels[codes.long()].reshape(*scales.shape, 32)
    result = torch.ldexp(values, scales.int().unsqueeze(-1) - 127).reshape(codes.shape)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("Non-finite expert reference")
    return result


def main() -> None:
    """Check top-eight expert arithmetic and stock repacking for both source TP2 slices."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Preserve prior numerical evidence")
    for name, want in (("config.json", CONFIG_SHA), ("model.safetensors.index.json", INDEX_SHA)):
        if hashlib.sha256((args.source / name).read_bytes()).hexdigest() != want:
            raise ValueError("Source metadata changed")
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        prepare_moe_fp4_layer_for_marlin,
    )
    from vllm.scalar_type import scalar_types

    torch.set_num_threads(2)
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(138)
    torch.cuda.set_per_process_memory_fraction(
        1.5 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    index = json.loads((args.source / "model.safetensors.index.json").read_bytes())["weight_map"]
    state: dict[str, Any] = {"status": "running", "started_epoch": time.time(), "partitions": [],
        "source_layer": 23, "source_experts": list(range(8)), "top_k": 8,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Stock preparation/GEMM on both TP2 weight slices sequentially; no collectives",
        "limits": ["Synthetic routing, not a full-model or actual loader coverage proof.",
                   "M512 compares sixteen selected rows; all output rows checked finite."]}

    def source(expert: int, projection: str, suffix: str, rank: int) -> torch.Tensor:
        name = f"model.language_model.layers.23.mlp.experts.{expert}.{projection}.{suffix}"
        with safe_open(args.source / index[name], framework="pt", device="cpu") as handle:
            raw = handle.get_tensor(name)
        if projection == "down_proj":
            stride = 32 if suffix == "weight_scale" else 2
            return raw[:, rank * 1024 // stride:(rank + 1) * 1024 // stride].contiguous()
        return raw[rank * 1024:(rank + 1) * 1024].contiguous()

    try:
        state["memory_before"] = memory()
        write(args.report, state)
        for rank in (0, 1):
            parts: dict[str, list[torch.Tensor]] = {key: [] for key in (
                "w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale")}
            references = []
            hashes = []
            for expert in range(8):
                values = {}
                for suffix in ("weight_packed", "weight_scale"):
                    values["w13", suffix] = torch.cat([
                        source(expert, projection, suffix, rank)
                        for projection in ("gate_proj", "up_proj")])
                    values["w2", suffix] = source(expert, "down_proj", suffix, rank)
                refs = []
                for family in ("w13", "w2"):
                    weight, scale = values[family, "weight_packed"], values[family, "weight_scale"]
                    hashes.append({"expert": expert, "family": family,
                        "weight": hashlib.sha256(weight.numpy().tobytes()).hexdigest(),
                        "scale": hashlib.sha256(scale.numpy().tobytes()).hexdigest()})
                    parts[family + "_weight"].append(weight)
                    parts[family + "_weight_scale"].append(scale)
                    refs.append(decode(weight, scale))
                references.append(refs)
            layer = torch.nn.Module()
            layer.params_dtype = torch.bfloat16
            layer.moe_config = SimpleNamespace(num_local_experts=8, hidden_dim=4096,
                                               intermediate_size_per_partition=1024)
            for name, tensors in parts.items():
                layer.register_parameter(name, torch.nn.Parameter(
                    torch.stack(tensors).cuda(), requires_grad=False))
            del parts, values, tensors
            prepare_moe_fp4_layer_for_marlin(layer)
            row: dict[str, Any] = {"rank": rank, "source_hashes": hashes, "cases": []}
            state["partitions"].append(row)
            for m in (1, 8, 64, 512):
                x = torch.randn((m, 4096), dtype=torch.float32).bfloat16() * .01
                selected = list(range(m)) if m < 512 else [*range(8), *range(m - 8, m)]
                reference_partials = []
                maximum_gate_up = 0.0
                for w13, w2 in references:
                    gate_up = (x[selected].float() @ w13.T).bfloat16()
                    maximum_gate_up = max(maximum_gate_up, float(gate_up.abs().max()))
                    gate, up = gate_up.float().chunk(2, dim=-1)
                    activated = (torch.nn.functional.silu(gate) * up).bfloat16()
                    output = (activated.float() @ w2.T * (2.5 / 8)).bfloat16()
                    reference_partials.append(output.float())
                reference = torch.stack(reference_partials).sum(0).bfloat16()
                if maximum_gate_up >= 10:
                    raise ValueError("Probe reached the model's clamp boundary")
                activation = x.cuda()
                ids = torch.arange(8, dtype=torch.int32, device="cuda").expand(m, -1).contiguous()
                routing_weights = torch.full((m, 8), 2.5 / 8, dtype=torch.float32, device="cuda")
                run = partial(fused_marlin_moe,
                    hidden_states=activation, w1=layer.w13_weight, w2=layer.w2_weight,
                    w1_scale=layer.w13_weight_scale, w2_scale=layer.w2_weight_scale,
                    bias1=None, bias2=None, topk_weights=routing_weights, topk_ids=ids,
                    quant_type_id=scalar_types.float4_e2m1f.id, global_num_experts=288,
                    activation=MoEActivation.SILU)
                actual, repeat = run().clone(), run()
                torch.cuda.synchronize()
                result = difference(actual[selected], reference)
                result.update(tokens=m, checked_rows=selected, maximum_gate_up=maximum_gate_up,
                    all_rows_finite=bool(torch.isfinite(actual).all()),
                    repeat_bits_exact=bool(torch.equal(actual.view(torch.uint8),
                                                       repeat.view(torch.uint8))))
                result["passed"] = bool(result["finite"] and result["all_rows_finite"]
                                         and result["relative_l2"] <= .01)
                row["cases"].append(result)
                write(args.report, state)
                print(json.dumps({"rank": rank, **result}), flush=True)
            del run, actual, repeat, layer, references, activation, reference_partials
            gc.collect()
            torch.cuda.empty_cache()
        passed = all(case["passed"] for row in state["partitions"] for case in row["cases"])
        state.update(status="complete" if passed else "numerical_failure", passed=passed,
                     cuda_peak_bytes=torch.cuda.max_memory_allocated(), finished_epoch=time.time())
        write(args.report, state)
        if not passed:
            raise RuntimeError("Stock routed-expert arithmetic differs from the CPU reference")
    except BaseException as exc:
        state.update(error=repr(exc), finished_epoch=time.time())
        if state["status"] == "running":
            state["status"] = "failed"
        write(args.report, state)
        raise


if __name__ == "__main__":
    main()
