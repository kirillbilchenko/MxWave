"""Check loaded fused BF16 attention while retaining the qualified diagnostic RPCs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
from attention_mapping import FUSED_ATTENTION_COMPONENTS
from diagnostic_worker import DiagnosticWorker, compare_tensors
from safetensors import safe_open


def check_attention_loaded(
    model: Any, source: Path, rank: int, tp_size: int = 2,
) -> dict[str, Any]:
    """Verify BF16 dtype, TP/replicated shard boundaries and NoPE zero padding."""
    if tp_size <= 0 or not 0 <= rank < tp_size:
        raise ValueError("Invalid tensor parallel rank")
    config = json.loads((source / "config.json").read_bytes())["text_config"]
    index = json.loads((source / "model.safetensors.index.json").read_bytes())["weight_map"]
    expected = set()
    for layer in range(config["num_hidden_layers"]):
        prefix = f"model.language_model.layers.{layer}.self_attn."
        choices = [fused for fused, parts in FUSED_ATTENTION_COMPONENTS.items()
                   if fused != "wk_weights_proj"
                   and all(prefix + part + ".weight" in index for part in parts)]
        if len(choices) != 1:
            raise ValueError(f"Expected one source attention family at layer {layer}")
        expected.add((layer, choices[0]))
    records: list[dict[str, Any]] = []
    modules = []
    seen = set()

    def shape(key: str) -> list[int]:
        with safe_open(source / index[key], framework="pt", device="cpu") as handle:
            return handle.get_slice(key).get_shape()

    def row(key: str, offset: int) -> torch.Tensor:
        with safe_open(source / index[key], framework="pt", device="cpu") as handle:
            return handle.get_slice(key)[offset:offset + 1]

    for name, module in model.named_modules():
        match = re.fullmatch(r"language_model\.model\.layers\.(\d+)\.self_attn\."
                             r"(in_proj_qkvbfg_a|fused_qkv_a_proj)", name)
        if match is None:
            continue
        layer, fused = int(match[1]), match[2]
        if (layer, fused) not in expected:
            continue
        seen.add((layer, fused))
        weight = getattr(module, "weight", None)
        info = {"name": name, "dtype": str(getattr(weight, "dtype", None)),
                "quant_method": type(getattr(module, "quant_method", None)).__name__,
                "has_weight_scale": hasattr(module, "weight_scale")}
        modules.append(info)
        if (not isinstance(weight, torch.Tensor) or weight.dtype != torch.bfloat16
                or weight.ndim != 2 or info["has_weight_scale"]):
            info["passed"] = False
            continue
        prefix = f"model.language_model.layers.{layer}.self_attn."
        layout = []
        offset = 0
        for part_id, projection in enumerate(FUSED_ATTENTION_COMPONENTS[fused]):
            key = prefix + projection + ".weight"
            source_rows, width = shape(key)
            sharded = fused == "in_proj_qkvbfg_a" and part_id < 4
            if sharded and source_rows % tp_size:
                raise ValueError("Source output rows do not divide into TP shards")
            count = source_rows // tp_size if sharded else source_rows
            begin = rank * count if sharded else 0
            pad = (config.get("qk_rope_head_dim", 0)
                   if projection == "kv_a_proj_with_mqa" and config.get("mla_nope") else 0)
            layout.append((key, offset, begin, count, pad, width))
            offset += count + pad
        if (weight.shape[0] != offset
                or any(item[-1] != weight.shape[1] for item in layout)):
            info.update(passed=False, actual_shape=list(weight.shape), expected_rows=offset)
            continue
        first_record = len(records)
        for key, offset, begin, count, pad, width in layout:
            for local in sorted({0, count - 1}):
                records.append({"name": f"{name}:{key.rsplit('.', 2)[-2]}.row{local}",
                    **compare_tensors(weight[offset + local:offset + local + 1],
                                      row(key, begin + local))})
            if pad:
                for local in sorted({count, count + pad - 1}):
                    records.append({"name": f"{name}:NoPE_padding.row{local}",
                        **compare_tensors(weight[offset + local:offset + local + 1],
                                          torch.zeros((1, width), dtype=torch.bfloat16))})
        info["passed"] = all(record["bits_exact"] for record in records[first_record:])
    passed = bool(seen == expected and modules and records
                  and all(item["passed"] for item in modules))
    return {"status": "passed" if passed else "failed", "rank": rank, "tp_size": tp_size,
            "expected_fused_layers": len(expected), "seen_fused_layers": len(seen),
            "modules": modules, "records": records,
            "limits": ["All base fused attention dtypes and shapes; selected rows per source shard.",
                       "Includes replicated KDA f_a/g_a and MLA NoPE padding.",
                       "Loaded-weight qualification, not a full-model quality benchmark."]}


class AttentionDiagnosticWorker(DiagnosticWorker):
    """Extend source checks; inherit unchanged, already-qualified graph/warmup RPCs."""

    def mxwave_checkpoint_check(self) -> dict[str, Any]:
        """Run the existing source checks and require exact loaded fused attention too."""
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        result = super().mxwave_checkpoint_check()
        try:
            attention = check_attention_loaded(self.model_runner.model, Path("/model"),
                get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size())
        except Exception as exc:  # noqa: BLE001 -- preserve model until controller restoration.
            attention = {"status": "failed", "error": repr(exc)}
        result["attention"] = attention
        if attention["status"] != "passed":
            result["status"] = "failed"
        self._mxwave_checkpoint_report = result
        return result
