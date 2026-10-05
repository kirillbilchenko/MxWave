"""Check native FP8 Kolibri kernels using a small subset of the official checkpoint."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main() -> None:
    """Load eight experts from one sliding and one full-attention reference layer."""
    root = Path("/work")
    source = root / "reference-fp8"
    output = root / "smoke-reference-fp8"
    output.mkdir(exist_ok=True)
    config = json.loads((root / "smoke-source/config.json").read_text())
    config["quantization_config"] = json.loads((source / "config.json").read_text())[
        "quantization_config"
    ]
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    index = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(str(root / "smoke-source/model.safetensors"), framework="pt") as small:
        keys = list(small.keys())
    tensors = {}
    for key in keys:
        original_key = key.replace("model.layers.1.", "model.layers.4.")
        for original in (original_key, original_key.removesuffix(".weight") + ".weight_scale_inv"):
            if original not in index:
                continue
            with safe_open(str(source / index[original]), framework="pt", device="cpu") as shard:
                if original in ("model.embed_tokens.weight", "lm_head.weight"):
                    tensor = shard.get_slice(original)[:256]
                elif original.endswith((".mlp.gate.weight", ".moe.router.expert_bias")):
                    tensor = shard.get_slice(original)[:8]
                else:
                    tensor = shard.get_tensor(original)
            tensors[original.replace("model.layers.4.", "model.layers.1.")] = tensor
    save_file(tensors, str(output / "model.safetensors"))
    del tensors
    from vllm import LLM, SamplingParams

    started = time.time()
    model = LLM(
        model=str(output), skip_tokenizer_init=True, enforce_eager=True, dtype="bfloat16",
        max_model_len=128, max_num_batched_tokens=128, max_num_seqs=2,
        gpu_memory_utilization=0.04, kv_cache_memory_bytes=64 * 2**20,
    )
    result = model.generate(
        [{"prompt_token_ids": [1, 2, 3, 4]}],
        SamplingParams(max_tokens=8, temperature=0, ignore_eos=True, detokenize=False),
    )
    tokens = result[0].outputs[0].token_ids
    if len(tokens) != 8:
        raise ValueError("Native FP8 reference did not complete eight decode steps")
    record = {
        "status": "passed", "seconds": time.time() - started, "output_tokens": list(tokens),
        "runtime": "vllm-0.29.0", "plugin": "aleph-alpha-inference-1.0.0",
        "gpu": torch.cuda.get_device_name(), "backends": "auto",
        "limitation": "Truncated real weights; kernel compatibility only, not model quality",
    }
    (root / "fp8-smoke-result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
