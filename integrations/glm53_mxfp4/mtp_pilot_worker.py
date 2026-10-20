"""Observe native draft ownership and compare selected loaded draft experts with source bytes."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from stable_dispatch import StableWorker


class MtpPilotWorker(StableWorker):
    """Keep base source checks and expose draft-specific checks outside timed inference."""

    def mxwave_mtp_snapshot(self, verify_storage: bool = False) -> dict[str, Any]:
        """Report actual speculative config and graph descriptors for its verification row count."""
        from mtp_pilot_support import report
        from stable_dispatch import report as dispatch_report
        from vllm.distributed import get_tensor_model_parallel_rank

        runner = self.model_runner
        depth = int(os.environ["MXWAVE_MTP_DEPTH"])
        spec = runner.vllm_config.speculative_config
        drafter = getattr(runner, "speculator", None)
        draft = getattr(drafter, "model", None)
        if depth:
            if (spec is None or spec.method != "mtp" or spec.num_speculative_tokens != depth
                    or not spec.use_local_argmax_reduction
                    or draft is None or type(draft).__name__ != "Glm5NextMTP"):
                raise RuntimeError("The loaded native drafter differs from the frozen pilot")
            target = runner.model
            if callable(getattr(target, "get_language_model", None)):
                target = target.get_language_model()
            elif hasattr(target, "language_model"):
                target = target.language_model
            if (draft.model.embed_tokens is not target.model.embed_tokens
                    or draft.lm_head is not target.lm_head
                    or len(draft.model.layers) != 1
                    or any(layer.shared_head.head is not target.lm_head
                           for layer in draft.model.layers.values())):
                raise RuntimeError("Native MTP failed to share the compressed vocabulary modules")
        elif spec is not None or draft is not None:
            raise RuntimeError("The no-drafter control unexpectedly loaded a draft model")
        manager = runner.cudagraph_manager
        modes = {}
        for rows in sorted({1, 4, depth + 1}):
            descriptor = manager.dispatch(num_reqs=1, num_tokens=rows,
                                          uniform_token_count=rows, num_active_loras=0)
            modes[str(rows)] = str(descriptor.cg_mode)
        result = {"rank": get_tensor_model_parallel_rank(), "depth": depth,
            "method": None if spec is None else str(spec.method),
            "draft_class": None if draft is None else type(draft).__name__,
            "speculator_class": None if drafter is None else type(drafter).__name__,
            "draft_uses_local_argmax_reduction": bool(depth),
            "draft_vocabulary_modules_shared": bool(depth),
            "graphs_enabled": bool(manager._graphs_captured), "graph_modes_by_uniform_rows": modes,
            "retained_graphs": len(manager.graphs), "stable_dispatch": dispatch_report(),
            "cuda_allocated_bytes": torch.cuda.memory_allocated(),
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated()}
        if verify_storage:
            result["storage"] = report(depth)
        return result

    def mxwave_mtp_source_check(self) -> dict[str, Any]:
        """Repack experts 0/127/287 and both shared linears using the stock preparation functions."""
        import json

        from diagnostic_worker import compare_tensors
        from mtp_draft_loader import bf16_projection_slices
        from safetensors import safe_open
        from vllm.distributed import get_tensor_model_parallel_rank
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
            prepare_moe_fp4_layer_for_marlin,
        )
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
            prepare_mxfp8_layer_for_marlin,
        )
        from vllm.utils.torch_utils import set_default_torch_dtype

        rank = get_tensor_model_parallel_rank()
        snapshot = self.mxwave_mtp_snapshot(True)
        if not snapshot["depth"]:
            return {"status": "passed", "rank": rank, "draft_loaded": False, "records": []}
        draft = self.model_runner.speculator.model
        source = Path("/model")
        index = json.loads((source / "model.safetensors.index.json").read_bytes())["weight_map"]

        def load(name: str) -> torch.Tensor:
            with safe_open(source / index[name], framework="pt", device="cpu") as handle:
                return handle.get_tensor(name)

        records = []
        experts = [m for m in draft.modules() if hasattr(m, "w13_weight")]
        if len(experts) != 1:
            raise ValueError("Require exactly one loaded MTP expert module")
        actual = experts[0]
        for expert in (0, 127, 287):
            expected = torch.nn.Module()
            expected.params_dtype = torch.bfloat16
            expected.moe_config = SimpleNamespace(num_local_experts=1, hidden_dim=4096,
                                                  intermediate_size_per_partition=1024)
            for family in ("w13", "w2"):
                for suffix in ("weight_packed", "weight_scale"):
                    parts = []
                    projections = ("gate_proj", "up_proj") if family == "w13" else ("down_proj",)
                    for projection in projections:
                        value = load(f"model.language_model.layers.45.mlp.experts."
                                     f"{expert}.{projection}.{suffix}")
                        if family == "w13":
                            value = value[rank * 1024:(rank + 1) * 1024]
                        else:
                            stride = 32 if suffix == "weight_scale" else 2
                            value = value[:, rank * 1024 // stride:(rank + 1) * 1024 // stride]
                        parts.append(value.contiguous())
                    name = family + ("_weight" if suffix == "weight_packed" else "_weight_scale")
                    expected.register_parameter(name, torch.nn.Parameter(
                        torch.cat(parts).unsqueeze(0).cuda(), requires_grad=False))
            prepare_moe_fp4_layer_for_marlin(expected)
            for name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
                records.append({"name": f"expert{expert}.{name}", **compare_tensors(
                    getattr(actual, name)[expert], getattr(expected, name)[0])})
            del expected
        shared = [(name, module) for name, module in draft.named_modules()
                  if name.endswith((".mlp.shared_experts.gate_up_proj",
                                    ".mlp.shared_experts.down_proj"))]
        if len(shared) != 2:
            raise ValueError("Require both native MTP shared expert linears")
        for name, actual in shared:
            merged = name.endswith("gate_up_proj")
            weights, scales = [], []
            for projection in (("gate_proj", "up_proj") if merged else ("down_proj",)):
                prefix = "model.language_model.layers.45.mlp.shared_experts." + projection
                weight, scale = load(prefix + ".weight"), load(prefix + ".weight_scale")
                if merged:
                    width = weight.shape[0] // 2
                    weight, scale = weight[rank * width:(rank + 1) * width], \
                        scale[rank * width:(rank + 1) * width]
                else:
                    width = weight.shape[1] // 2
                    weight = weight[:, rank * width:(rank + 1) * width]
                    scale = scale[:, rank * width // 32:(rank + 1) * width // 32]
                weights.append(weight.contiguous())
                scales.append(scale.contiguous())
            weight, scale = torch.cat(weights), torch.cat(scales)
            expected = torch.nn.Module()
            expected.output_size_per_partition, expected.input_size_per_partition = weight.shape
            for key, value in (("weight", weight), ("weight_scale", scale)):
                expected.register_parameter(key, torch.nn.Parameter(value.cuda(), requires_grad=False))
            with set_default_torch_dtype(torch.bfloat16):
                prepare_mxfp8_layer_for_marlin(expected)
            for key in ("weight", "weight_scale"):
                records.append({"name": name + "." + key, **compare_tensors(
                    getattr(actual, key), getattr(expected, key))})
            del expected, weights, scales, weight, scale
        projections = [(name, module) for name, module in draft.named_modules()
                       if name.endswith(".self_attn.fused_qkv_a_proj")]
        if len(projections) != 1:
            raise ValueError("Require one native MTP fused BF16 attention input projection")
        name, actual = projections[0]
        q = load("model.language_model.layers.45.self_attn.q_a_proj.weight")
        kv = load("model.language_model.layers.45.self_attn.kv_a_proj_with_mqa.weight")
        config = draft.config
        geometry = {"q_rank": int(config.q_lora_rank), "kv_rank": int(config.kv_lora_rank),
                    "rope_rows": int(config.qk_rope_head_dim),
                    "hidden_size": int(config.hidden_size),
                    "nope": bool(getattr(config, "mla_nope", False))}
        checks = bf16_projection_slices(actual.weight, q, kv, **geometry)
        for suffix, value, expected in checks:
            records.append({"name": name + "." + suffix, **compare_tensors(value, expected)})
        passed = len(records) == 16 + len(checks) and all(r["bits_exact"] for r in records)
        return {"status": "passed" if passed else "failed", "rank": rank, "draft_loaded": True,
            "records": records, "projection_geometry": geometry,
            "limits": [("Three sampled routed experts, both shared linears and "
                "the complete fused BF16 attention input projection; other tensors use the native "
                "loader. This does not qualify numerical behavior.")]}
