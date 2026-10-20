---
base_model: zai-org/GLM-5.3-Flash-BF16
library_name: transformers
license: mit
pipeline_tag: image-text-to-text
language:
- en
- zh
tags:
- glm
- multimodal
- vllm
- modelopt
- nvfp4
- w4a4
- dgx-spark
---

# GLM-5.3-Flash expert-only NVFP4 W4A4

This is a community conversion of
[`zai-org/GLM-5.3-Flash-BF16`](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16)
at revision `b1967181a3917ae70a437f4884748f6b8e3a1f4d`.

It was built for native NVFP4 execution on four NVIDIA GB10/DGX Spark nodes
with vLLM TP4+EP4. It is **not currently a drop-in checkpoint for an
unmodified upstream vLLM installation**: the tested deployment uses gated
compatibility patches for GLM-5.3's rope-free sparse MLA/KDA path on SM121a,
ProcessGroupNCCL collectives, mixed-precision `lm_head`, and deterministic
batch execution.

The complete conversion pipeline, vLLM compatibility patches, Docker/Compose
configuration, four-node launchers, acceptance tools, and pinned offline
dependencies are preserved in
[`shankinson/glm53-nvfp4-team`](https://github.com/shankinson/glm53-nvfp4-team).
The accepted source snapshot is commit
[`95e08fd`](https://github.com/shankinson/glm53-nvfp4-team/commit/95e08fd0b588ea51341b03ec80e95b9c76c2fa4d),
tagged `glm53-nvfp4-w4a4-accepted-20260829`.

## Quantization boundary

- NVIDIA Model Optimizer 0.46.0, commit `43fd41a58d52c4e6e5dec1d1ff5989ecc737ae1a`
- Base routed-expert layers 3–44 only
- 288 experts per layer
- `gate_proj`, `up_proj`, and `down_proj`: 36,288 weights total
- NVFP4 E2M1 weights, group size 16, with calibrated W4A4 activations
- MTP layer 45 remains BF16: 864 expert projection weights
- Vision, attention/KDA, sparse indexers, routers, shared experts, dense
  layers, embeddings, `lm_head`, and norms remain BF16

The activation-scale policy uses a 5% safety margin with:

- p99 across experts by default
- p98 for `w2` at layers 3–8
- p98.9 for `w2` at layers 24–44

The conversion and active-scale hashes are recorded in
`conversion_manifest.json`. The installed activation-scale shard SHA-256 is
`caacc401233a4d6bcae6fc4f830c1754b12f28d0f6c5b513ceb1e586b40f5eb1`.

## Validation

The final clean-cold four-node run used vLLM
`0.1.dev20051+g487ecf187`, FlashInfer 0.6.17, CUDA `sm_121a`, native
`FLASHINFER_CUTLASS` NVFP4 MoE, and `FLASHINFER_MLA_SPARSE_SM120`.

Constrained 400-item MMLU/ScienceQA regression evaluation:

| Checkpoint | Overall | Text | Image-text |
|---|---:|---:|---:|
| Official FP8 reference | 89.25% | 84.5% | 94.0% |
| W4A16 activation reference | 88.5% | 83.0% | 94.0% |
| This W4A4 checkpoint | 89.0% | 84.0% | 94.0% |

Unprofiled W4A4 phase throughput on four GB10 nodes:

| Concurrent width | Prefill prompt tok/s | Generation output tok/s |
|---:|---:|---:|
| 1 | 1,279.01 | 24.86 |
| 2 | 1,332.22 | 40.63 |
| 4 | 1,334.71 | 65.73 |
| 8 | 1,383.55 | 99.77 |
| 16 | 1,382.60 | 173.77 |

The long-prompt prefill case generated one token. The decode-dominant case
used a short prompt and 256 forced output tokens. Exact deterministic output,
reasoning, tools, image, four-image, 32-frame video, streaming, 7,820-token and
31,020-token prompts passed. A 901.5-second mixed-width soak completed 2,495
requests without failures or earlyoom intervention.

## Download

```bash
hf download shankinsonhf/GLM-5.3-Flash-NVFP4-W4A4-Experts \
  --local-dir /path/to/GLM-5.3-Flash-NVFP4-W4A4-Experts
```

The checkpoint is approximately 191 GiB (205 GB decimal). The repository uses
120 primary safetensors shards plus a separate activation-scale shard.

## Important runtime note

Use this checkpoint only with a runtime that recognizes ModelOpt
`MIXED_PRECISION` metadata and native NVFP4 expert layers. The validated GB10
deployment requires the
[companion vLLM compatibility patches and launch settings](https://github.com/shankinson/glm53-nvfp4-team);
vanilla vLLM behavior has not been claimed or validated. The checkpoint
retains the upstream multimodal processor, tokenizer, chat template, and MIT
license.

This conversion is not affiliated with Z.AI, NVIDIA, vLLM, or FlashInfer.
