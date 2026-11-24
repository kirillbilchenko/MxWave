"""Private-engine diagnostics: verify loaded tensors and compare graph/eager dispatch."""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from diagnose_mxfp8 import CONFIG_SHA, INDEX_SHA
from mhc_warmup import token_sizes
from safetensors import safe_open


def compare_tensors(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    """Compare exact bytes on CPU even when the prepared reference still lives on CUDA."""
    actual = actual.detach().to(device="cpu").contiguous()
    expected = expected.detach().to(device="cpu", dtype=actual.dtype).contiguous()
    a, b = actual.reshape(-1).view(torch.uint8), expected.reshape(-1).view(torch.uint8)
    return {"shape": list(actual.shape), "dtype": str(actual.dtype),
        "bits_exact": actual.shape == expected.shape and bool(torch.equal(a, b)),
        "loaded_sha256": hashlib.sha256(a.numpy().tobytes()).hexdigest(),
        "source_reference_sha256": hashlib.sha256(b.numpy().tobytes()).hexdigest()}


class DiagnosticWorker:
    """RPC mixin used only by disposable, loopback-only diagnostic engines."""

    def mxwave_set_eager(self, enabled: bool) -> dict[str, Any]:
        """Suspend graph dispatch on the pinned new runner, retaining captured graphs."""
        from vllm.config import CUDAGraphMode

        manager = getattr(self.model_runner, "cudagraph_manager", None)
        if manager is None or not hasattr(manager, "_graphs_captured"):
            return {"status": "unsupported", "eager": enabled,
                    "reason": "Expected the pinned runner's captured graph manager"}
        if not hasattr(self, "_mxwave_original_graph_state"):
            self._mxwave_original_graph_state = (manager, manager._graphs_captured)
        original, captured = self._mxwave_original_graph_state
        if manager is not original or not captured or not manager.graphs:
            return {"status": "unsupported", "eager": enabled,
                    "reason": "Expected the same manager with retained captured graphs"}
        before = manager._graphs_captured
        try:
            # dispatch() tests this flag before consulting its candidate cache.
            # Changing cudagraph_mode alone does not disable the new dispatcher.
            manager._graphs_captured = captured and not enabled
            descriptor = manager.dispatch(num_reqs=1, num_tokens=1,
                                          uniform_token_count=1, num_active_loras=0)
            expected = CUDAGraphMode.NONE if enabled else CUDAGraphMode.FULL
            if descriptor.cg_mode != expected:
                raise ValueError("The actual descriptor did not select the requested mode")
            return {"status": "complete", "eager": enabled,
                    "one_token_runtime_mode": str(descriptor.cg_mode),
                    "retained_graphs": len(manager.graphs),
                    "runner": type(self.model_runner).__module__ + "." +
                              type(self.model_runner).__name__}
        except Exception as exc:  # noqa: BLE001 -- RPC errors must not kill a loaded worker.
            manager._graphs_captured = before
            return {"status": "failed", "eager": enabled, "error": repr(exc)}

    def mxwave_checkpoint_check(self) -> dict[str, Any]:
        """Reprepare source experts and compare them to the actual loaded model, bit-for-bit."""
        from vllm.distributed import get_tensor_model_parallel_rank
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
            prepare_moe_fp4_layer_for_marlin,
        )
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
            prepare_mxfp8_layer_for_marlin,
        )
        from vllm.utils.torch_utils import set_default_torch_dtype

        rank = get_tensor_model_parallel_rank()
        source = Path("/model")
        for name, want in (("config.json", CONFIG_SHA), ("model.safetensors.index.json", INDEX_SHA)):
            if hashlib.sha256((source / name).read_bytes()).hexdigest() != want:
                raise ValueError("Diagnostic source metadata changed")
        index = json.loads((source / "model.safetensors.index.json").read_bytes())["weight_map"]
        records: list[dict[str, Any]] = []

        def load(name: str, rows: slice | None = None) -> torch.Tensor:
            with safe_open(source / index[name], framework="pt", device="cpu") as handle:
                return handle.get_tensor(name) if rows is None else handle.get_slice(name)[rows]

        def compare(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
            records.append({"name": name, **compare_tensors(actual, expected)})

        modules = dict(self.model_runner.model.named_modules())
        seen_layers = set()
        seen_storage = set()
        for name, actual in modules.items():
            match = re.search(r"layers\.(\d+)\.mlp\.experts$", getattr(actual, "layer_name", name))
            if match is None or not hasattr(actual, "w13_weight"):
                continue
            pointer = actual.w13_weight.data_ptr()
            if pointer in seen_storage:
                continue
            seen_storage.add(pointer)
            layer_index = int(match[1])
            if not 3 <= layer_index <= 44:
                continue
            seen_layers.add(layer_index)
            experts = (0, 127, 287) if layer_index in (3, 23, 44) else (0,)
            for expert in experts:
                expected = torch.nn.Module()
                expected.params_dtype = torch.bfloat16
                expected.moe_config = SimpleNamespace(num_local_experts=1, hidden_dim=4096,
                                                      intermediate_size_per_partition=1024)
                for family in ("w13", "w2"):
                    for suffix in ("weight_packed", "weight_scale"):
                        parts = []
                        for projection in (("gate_proj", "up_proj") if family == "w13"
                                           else ("down_proj",)):
                            key = (f"model.language_model.layers.{layer_index}.mlp.experts."
                                   f"{expert}.{projection}.{suffix}")
                            value = load(key)
                            if family == "w13":
                                value = value[rank * 1024:(rank + 1) * 1024]
                            else:
                                stride = 32 if suffix == "weight_scale" else 2
                                value = value[:, rank * 1024 // stride:(rank + 1) * 1024 // stride]
                            parts.append(value.contiguous())
                        parameter = family + ("_weight" if suffix == "weight_packed"
                                              else "_weight_scale")
                        expected.register_parameter(parameter, torch.nn.Parameter(
                            torch.cat(parts).unsqueeze(0).cuda(), requires_grad=False))
                prepare_moe_fp4_layer_for_marlin(expected)
                for attribute in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
                    compare(f"{name}.expert{expert}.{attribute}", getattr(actual, attribute)[expert],
                            getattr(expected, attribute)[0])
                del expected

        seen_linears = []
        for name, actual in modules.items():
            match = re.search(r"layers\.(0|3|44)\.mlp\.(shared_experts\.)?"
                              r"(gate_up_proj|down_proj)$", name)
            if match is None:
                continue
            prefix = f"model.language_model.layers.{match[1]}.mlp.{match[2] or ''}"
            merged = match[3] == "gate_up_proj"
            weights, scales = [], []
            for projection in (("gate_proj", "up_proj") if merged else ("down_proj",)):
                weight, scale = load(prefix + projection + ".weight"), load(
                    prefix + projection + ".weight_scale")
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
            expected.register_parameter("weight", torch.nn.Parameter(weight.cuda(), requires_grad=False))
            expected.register_parameter("weight_scale", torch.nn.Parameter(scale.cuda(), requires_grad=False))
            with set_default_torch_dtype(torch.bfloat16):
                prepare_mxfp8_layer_for_marlin(expected)
            for attribute in ("weight", "weight_scale"):
                compare(f"{name}.{attribute}", getattr(actual, attribute), getattr(expected, attribute))
            seen_linears.append(name)
            del expected, weights, scales, weight, scale

        # Replicated norm/mHC parameters and vocab-parallel head/embedding
        # samples do not undergo quantization packing. Test their actual loads.
        for name, parameter in self.model_runner.model.named_parameters():
            head = name.endswith("language_model.lm_head.weight")
            embedding = name.endswith("language_model.model.embed_tokens.weight")
            match = re.search(r"layers\.(\d+)\.(hc_(?:attn|ffn)_(?:fn|scale|base)|"
                              r"input_layernorm\.weight|post_attention_layernorm\.weight)$", name)
            if "language_model.model.layers." not in name:
                match = None
            if not head and not embedding and match is None:
                continue
            if head or embedding:
                key = "lm_head.weight" if head else "model.language_model.embed_tokens.weight"
                rows = parameter.shape[0]
                for row in (0, rows - 1):
                    reference = load(key, slice(rank * rows + row, rank * rows + row + 1))
                    compare(f"{name}.row{row}", parameter[row:row + 1], reference)
            elif match is not None:
                key = f"model.language_model.layers.{match[1]}.{match[2]}"
                if key not in index:
                    raise ValueError(f"No source for diagnostic parameter {key}")
                if parameter.ndim == 2:
                    for row in (0, parameter.shape[0] - 1):
                        compare(f"{name}.row{row}", parameter[row:row + 1],
                                load(key, slice(row, row + 1)))
                else:
                    compare(name, parameter, load(key))
        passed = bool(seen_layers == set(range(3, 45)) and len(seen_linears) == 6 and records
                      and all(row["bits_exact"] for row in records))
        result = {"status": "passed" if passed else "failed", "rank": rank,
                  "expert_layers": sorted(seen_layers), "records": records,
                  "mxfp8_linears": seen_linears,
                  "limits": ["One expert per layer; three experts at layers 3/23/44.",
                             "Head and embedding sampled at each rank's boundary rows.",
                             "This verifies selected loaded values, not model quality."]}
        self._mxwave_checkpoint_report = result
        return result

    def mxwave_mhc_warmup(self, maximum: int = 2048) -> dict[str, Any]:
        """Warm actual GLM layer methods for every selected split and small-token branch."""
        from vllm.model_executor.kernels.mhc.tilelang_kernels import compute_num_split

        start = time.monotonic()
        layers = [module for module in self.model_runner.model.modules()
                  if module.__class__.__name__ == "Glm5NextDecoderLayer" and hasattr(module, "hc_pre")]
        if len(layers) != 45:
            raise ValueError("Expected all 45 base GLM mHC layers")
        attributes = ("hidden_size", "n", "rms_norm_eps", "hc_eps", "mhc_post_mult_value",
                      "mhc_sinkhorn_iterations")
        signatures = {tuple(getattr(layer, key) for key in attributes) for layer in layers}
        if len(signatures) != 1:
            raise ValueError("The focused warmup does not cover differing layer specializations")
        layer = layers[0]
        hidden, hc = layer.hidden_size, layer.n
        sizes = token_sizes(maximum, lambda tokens: compute_num_split(
            64, hc * hidden, (tokens + 63) // 64))
        device = layer.hc_attn_fn.device
        residual = torch.full((maximum, hc, hidden), .125, dtype=torch.bfloat16, device=device)
        x = torch.full((maximum, hidden), .0625, dtype=torch.bfloat16, device=device)
        post = torch.full((maximum, hc, 1), .25, dtype=torch.float32, device=device)
        comb = torch.eye(hc, dtype=torch.float32, device=device).expand(maximum, -1, -1).contiguous()
        with torch.inference_mode():
            for size in sizes:
                for fn, scale, base, norm in (
                    (layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base,
                     layer.input_layernorm.weight),
                    (layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base,
                     layer.post_attention_layernorm.weight),
                ):
                    layer.hc_pre(residual[:size], fn, scale, base, norm, layer.rms_norm_eps)
                    layer.hc_fused_post_pre(x[:size], residual[:size], post[:size], comb[:size],
                                           fn, scale, base, norm, layer.rms_norm_eps)
        torch.cuda.synchronize()
        return {"status": "complete", "sizes": sizes, "seconds": time.monotonic() - start,
                "layer_signatures": [list(signature) for signature in signatures]}
