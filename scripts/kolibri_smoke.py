"""Exercise production Kolibri geometry with a small MXFP4 checkpoint on vLLM."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

from mxwave.engine import QuantizeConfig, quantize_model


def main() -> None:
    """Build eight-expert/two-layer weights and verify a real Marlin forward."""
    root = Path("/work")
    source = root / "smoke-source"
    output = root / "smoke-output"
    source.mkdir(exist_ok=True)
    config = json.loads((root / "source-bf16/config.json").read_text())
    config.update(
        num_hidden_layers=2, num_experts=8, num_experts_per_tok=6, vocab_size=256,
        layer_types=["sliding_attention", "full_attention"], max_position_embeddings=128,
        eos_token_id=255, pad_token_id=0,
    )
    (source / "config.json").write_text(json.dumps(config, indent=2))
    torch.manual_seed(20261004)
    hidden, intermediate = config["hidden_size"], config["moe_intermediate_size"]
    tensors: dict[str, torch.Tensor] = {}
    for layer in range(2):
        prefix = f"model.layers.{layer}"
        for expert in range(8):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                shape = (
                    (hidden, intermediate) if projection == "down_proj"
                    else (intermediate, hidden)
                )
                tensors[f"{prefix}.mlp.experts.{expert}.{projection}.weight"] = (
                    torch.randn(shape, dtype=torch.bfloat16) * 0.01
                )
        for projection, rows, cols in (
            ("q_proj", 6144, hidden), ("k_proj", 512, hidden),
            ("v_proj", 512, hidden), ("o_proj", hidden, 6144),
        ):
            tensors[f"{prefix}.self_attn.{projection}.weight"] = (
                torch.randn(rows, cols, dtype=torch.bfloat16) * 0.01
            )
        for projection in ("gate_proj", "up_proj", "down_proj"):
            shape = (
                (hidden, intermediate) if projection == "down_proj"
                else (intermediate, hidden)
            )
            tensors[f"{prefix}.mlp.shared_experts.{projection}.weight"] = (
                torch.randn(shape, dtype=torch.bfloat16) * 0.01
            )
        tensors[f"{prefix}.mlp.gate.weight"] = torch.randn(8, hidden, dtype=torch.bfloat16)
        tensors[f"{prefix}.moe.router.expert_bias"] = torch.linspace(
            -1, 1, 8, dtype=torch.bfloat16
        )
        for name in (
            "input_layernorm", "post_attention_layernorm", "post_attn_norm", "post_ffn_norm"
        ):
            tensors[f"{prefix}.{name}.weight"] = torch.ones(hidden, dtype=torch.bfloat16)
        for name in ("q_norm", "k_norm"):
            tensors[f"{prefix}.self_attn.{name}.weight"] = torch.ones(128, dtype=torch.bfloat16)
    tensors["model.embed_tokens.weight"] = torch.randn(256, hidden, dtype=torch.bfloat16) * 0.01
    tensors["lm_head.weight"] = torch.randn(256, hidden, dtype=torch.bfloat16) * 0.01
    tensors["model.norm.weight"] = torch.ones(hidden, dtype=torch.bfloat16)
    save_file(tensors, str(source / "model.safetensors"))
    del tensors
    quantize_model(QuantizeConfig(
        model_dir=source, output_dir=output, device="cuda", policy="kolibri1-routed-experts",
        method="mse", mse_clip_depth=4, verify_sqnr=True,
    ))
    torch.cuda.empty_cache()
    from vllm import LLM, SamplingParams

    started = time.time()
    model = LLM(
        model=str(output), skip_tokenizer_init=True, enforce_eager=True, dtype="bfloat16",
        max_model_len=128, max_num_batched_tokens=128, max_num_seqs=2,
        gpu_memory_utilization=0.04, kv_cache_memory_bytes=64 * 2**20,
        linear_backend="marlin", moe_backend="marlin",
    )
    result = model.generate(
        [{"prompt_token_ids": [1, 2, 3, 4]}],
        SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True, detokenize=False),
    )
    tokens = result[0].outputs[0].token_ids
    if len(tokens) != 8:
        raise RuntimeError(f"Expected eight output tokens, received {tokens}")
    record = {
        "status": "passed", "seconds": time.time() - started, "output_tokens": list(tokens),
        "hidden_size": hidden, "intermediate_size": intermediate,
        "layers": 2, "experts": 8, "top_k": 6,
        "runtime": "vllm-0.29.0", "plugin": "aleph-alpha-inference-1.0.0",
        "backends": {"linear": "marlin", "moe": "marlin"},
        "limitation": "Synthetic weights; establishes load/forward compatibility only",
    }
    (root / "smoke-result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
