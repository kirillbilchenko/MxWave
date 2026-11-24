"""Reproduce stock V5 cache planning on CPU, without weights or runtime patches.

Run inside the pinned official image with no GPU access. Cache specifications
are reconstructed from the retained model configs and the installed backends.
The V5 negative control must reproduce both its 13.96 GiB requirement and 9,472
token estimate before any alternative is reported as passing the capacity gate.
This checks allocation arithmetic, not GPU serving or prefix-cache correctness.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def check(root: Path) -> dict[str, Any]:
    """Call installed vLLM planning functions for both native drafter backends."""
    import torch
    from vllm import platforms
    from vllm.platforms.cpu import CpuPlatform

    if torch.cuda.is_available():
        raise RuntimeError("This audit must run without GPU access")
    platforms.current_platform = CpuPlatform()

    from vllm.model_executor.layers.attention.attention import (
        Attention,
    )
    from vllm.v1.attention.backend import AttentionType, MultipleOf
    from vllm.v1.attention.backends.flashinfer import FlashInferBackend
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
        FlashInferMLASparseSM120Backend,
    )
    from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend
    from vllm.v1.core import kv_cache_utils as planner
    from vllm.v1.kv_cache_interface import (
        CircularBufferSpec,
        KVCacheLayout,
        MambaSpec,
        MLAAttentionSpec,
        SparseCacheRole,
        get_kv_quant_mode,
    )
    from vllm.v1.worker.utils import customize_attention_spec

    model = json.loads((root / "stock/model-config.json").read_text())
    model = model.get("text_config", model)
    draft = json.loads((root / "stock/drafter-config.json").read_text())
    original_draft = json.loads((root / "production/drafter-config.json").read_text())
    block_size, spec_depth, tp_size = 4608, 5, 2
    kv_budget = 6 * 1024**3
    linear = model["linear_attn_config"]
    num_heads, head_dim = linear["num_heads"], linear["head_dim"]
    conv_width = linear["short_conv_kernel_size"] - 1 + spec_depth
    recurrent_shape = (num_heads // tp_size, head_dim, head_dim)
    conv_shape = (3 * num_heads * head_dim // tp_size, conv_width)
    # The platform's page alignment uses the configured fp8_e4m3 MLA width.
    # The SM12 sparse backend then declares its actual 656-byte packed content.
    mamba_page = block_size * model["kv_lora_rank"]
    shapes = (conv_shape, recurrent_shape)
    mamba_raw = math.prod(conv_shape) * 2 + math.prod(recurrent_shape) * 4
    assert round(100 * (mamba_page - mamba_raw) / mamba_raw, 2) == 2.86
    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            mamba_cache_mode="align", block_size=block_size,
            skip_page_size_padded=None,
            get_resolved_kv_cache_layout=lambda: KVCacheLayout.BLHNC,
        ),
        model_config=SimpleNamespace(max_model_len=500000),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        attention_config=SimpleNamespace(hisparse_config=None),
        # Keep the annotation warning separate: it does not change pool math.
        speculative_config=None,
        max_in_flight_tokens=2 * 6919,
    )
    target: dict[str, Any] = {}
    for layer, kind in enumerate(model["layer_types"]):
        prefix = f"target.layers.{layer}"
        if kind == "linear_attention":
            target[prefix + ".kda"] = MambaSpec(
                block_size=block_size, shapes=shapes,
                dtypes=(torch.bfloat16, torch.float32), page_size_padded=mamba_page,
                num_speculative_blocks=spec_depth, num_prefill_checkpoint_blocks=1,
                mamba_cache_mode="align", prefill_checkpoint_alignment=64,
            )
        else:
            assert kind == "deepseek_sparse_attention"
            target[prefix + ".mla"] = customize_attention_spec(
                FlashInferMLASparseSM120Backend,
                MLAAttentionSpec(
                    block_size=block_size, num_kv_heads=1, max_tp_shards=1,
                    head_size=model["kv_lora_rank"], dtype=torch.uint8,
                    state_content_bytes=656, cache_dtype_str="fp8_ds_mla",
                    kv_quant_mode=get_kv_quant_mode("fp8_ds_mla"),
                    is_index_group_leader=True,
                ),
            )
            # FP8 indexer: 128 data bytes + one FP32 scale per pooled entry.
            # Its fixed 256-token kernel blocks require whole-page strides.
            target[prefix + ".indexer"] = MLAAttentionSpec(
                block_size=block_size, num_kv_heads=1, max_tp_shards=1,
                head_size=model["index_head_dim"] + 4, dtype=torch.uint8,
                tokens_per_state=model["index_kpool"], cache_role=SparseCacheRole.INDEXER,
                block_stride_alignment=MultipleOf(256),
            )
            span = model["index_kpool"] + spec_depth
            ring = model["index_kpool"] * 2 ** math.ceil(
                math.log2(math.ceil(span / model["index_kpool"]))
            )
            target[prefix + ".tail"] = CircularBufferSpec(
                block_size=ring, num_kv_heads=2, head_size=model["index_head_dim"],
                head_size_v=0, dtype=torch.bfloat16, dcp_sharded=False,
            )

    def evaluate(label: str, draft_config: dict[str, Any], backend: Any) -> dict[str, Any]:
        specs = dict(target)
        # Use the real attention method to select each backend's block size.
        layer = SimpleNamespace(
            attn_type=AttentionType.DECODER, attn_backend=backend,
            kv_cache_dtype="fp8_e4m3", kv_cache_torch_dtype=torch.float8_e4m3fn,
            num_kv_heads=draft_config["num_key_value_heads"] // tp_size,
            head_size=draft_config["head_dim"], head_size_v=draft_config["head_dim"],
            sliding_window=draft_config["sliding_window"],
        )
        draft_spec = customize_attention_spec(backend, Attention.get_kv_cache_spec(layer, config))
        for i in range(draft_config["num_hidden_layers"]):
            specs[f"draft.layers.{i}"] = draft_spec
        groups = planner.get_kv_cache_groups(config, specs)
        planner.validate_kv_cache_layout(KVCacheLayout.BLHNC, groups)
        pool_bytes = planner._pool_bytes_per_block(config, groups)
        needed = planner._max_memory_usage_bytes_from_groups(config, groups)
        maximum = planner._estimate_max_model_len_from_groups(config, groups, kv_budget - pool_bytes)
        detail = []
        for group in groups:
            spec = group.kv_cache_spec
            pages = spec.max_memory_usage_pages(config)
            detail.append({
                "kind": type(spec.first_spec).__name__, "layers": group.layer_names,
                "block_tokens": spec.block_size, "resident_blocks": pages,
                "group_page_bytes": spec.page_size_bytes, "pool_bytes_charged": pages * pool_bytes,
            })
        return {
            "label": label, "backend": backend.get_name(),
            "drafter_layers": draft_config["num_hidden_layers"],
            "draft_window": draft_spec.sliding_window, "draft_block_tokens": draft_spec.block_size,
            "pool_block_bytes": pool_bytes, "required_bytes": needed,
            "required_gib": needed / 1024**3, "usable_bytes": kv_budget - pool_bytes,
            "maximum_model_len_capped_at_500k": maximum, "fits_500k": maximum == 500000,
            "groups": detail,
        }

    results = [
        evaluate("v5_negative_control", draft, FlashInferBackend),
        evaluate("native_triton_candidate", draft, TritonAttentionBackend),
        evaluate("inco_geometry_on_stock_control", original_draft, FlashInferBackend),
    ]
    baseline, candidate, _ = results
    assert baseline["required_bytes"] == 14985265152
    assert baseline["maximum_model_len_capped_at_500k"] == 9472
    assert round(baseline["required_gib"], 2) == 13.96
    assert candidate["fits_500k"] and candidate["draft_block_tokens"] == 4608
    source_hashes = {}
    for obj in (Attention, FlashInferBackend, FlashInferMLASparseSM120Backend,
                TritonAttentionBackend, planner, MambaSpec, customize_attention_spec):
        path = Path(inspect.getfile(obj))
        source_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "status": "passed", "scope": "CPU allocation arithmetic only",
        "gpu_used": False, "weights_loaded": False, "runtime_source_modified": False,
        "production_service_changed": False, "prefix_caching": True,
        "max_model_len": 500000, "max_num_seqs": 32,
        "max_num_batched_tokens": 6919, "max_concurrent_batches": 2,
        "kv_budget_bytes_per_rank": kv_budget, "mamba_page_padding_percent": 2.86,
        "input_hashes": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (root / "stock/model-config.json", root / "stock/drafter-config.json",
                                   root / "production/drafter-config.json")},
        "source_hashes": source_hashes, "results": results,
        "remaining": ["DFlash group annotation", "GPU allocation and graph capture",
                      "repeated-prompt correctness and prefix reuse", "long-context probes",
                      "speed and quality qualification"],
    }


def main() -> None:
    """Write a reproducible read-only audit receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.root)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "results": [
        {k: v for k, v in row.items() if k != "groups"} for row in result["results"]
    ]}, indent=2))


if __name__ == "__main__":
    main()
