---
license: mit
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
  - nvfp4
  - fp4
  - modelopt
  - sglang
  - glm5_next
---

# GLM-5.3-Flash-NVFP4

NVFP4 (4-bit float, group size 16) quantization of
[`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash), covering the
routed experts of **every** MoE layer — including layer 45, the multi-token-prediction
(MTP) layer — in a **single, uniform** scheme.

**328.4 GB → 194.6 GB**, weight-only (activations stay bf16).

**No calibration data is used in this quantization step**, and none is needed: the scheme
quantizes each group of 16 weights with a scale derived from those 16 values
(`memoryless_minmax`), so there is no forward pass for calibration data to inform. The
checkpoint contains no activation-scale tensors and declares `input_activations: null`.

## What is quantized

| | |
|---|---|
| format | `modelopt`, `quant_algo: NVFP4` — e2m1 weights, one fp8-e4m3 scale per group of 16, one fp32 global scale per tensor |
| scope | routed experts (`mlp.experts.*.{gate,up,down}_proj`) on layers **3–45** |
| activations | **none — weight-only (W4A16)**, so activations run in bf16 |
| left in bf16 | all attention (KDA + DSA + indexer), the MoE router, shared experts, the dense MLPs of layers 0–2, `lm_head`, embeddings, and the entire vision tower |
| MTP / layer 45 | **quantized identically to layers 3–44** |

`num_hidden_layers: 45` plus `num_nextn_predict_layers: 1` puts the MTP layer at index
**45**, and it is not a small head: 288 routed experts, ~7.25 B parameters, its own DSA
attention and `eh_proj`/`enorm`/`hnorm` plumbing. Keeping it in the same scheme as every
other expert layer means the checkpoint has one expert layout throughout — 37,152 packed
expert tensors (43 × 288 × 3).

Weight fidelity: mean relative reconstruction error against the dequantized FP8 source is
**0.0918** over sampled routed-expert weights, against NVFP4's intrinsic floor of ~0.093
(measured: gaussian 0.0951, laplace 0.0927) — i.e. at the format's limit.

## Evaluation

eval-gate medium tier, SGLang TP4 on 4×B200:

| benchmark | score | scored samples | truncated at `max_tokens` | score on completed only |
|---|---|---|---|---|
| `bfcl_v4` | 72.64 | 3,469 | 0 % | 72.64 |
| `hle` | 20.94 | 1,476 | **29.3 %** | **29.60** |
| `aa_lcr` | 77.00 | 100 | 1 % | 77.00 |
| `aime26` | 68.33 | 120 | **30.0 %** | **97.62** |

Scores are percentages. `bfcl_v4` (limit 300 per subset) and `hle` (limit 200 per subset) are
**bounded** runs and are bounds, not full benchmark scores. `aime26` uses `repeats: 4` over 30
problems.

**Read the last two columns before the second.** These runs used `max_tokens: 32768` with
`reasoning_effort: max`. On `hle` and `aime26` that budget is too small: a generation that hits
the cap never emits the graded answer block and scores **exactly zero** — verified, 36/36
truncated `aime26` samples had an empty answer extraction. So the `score` column is a blend of
real accuracy and token-budget attrition, and the honest reading of this model's `aime26` is
**97.62 on the problems it was allowed to finish**, not 68.33.

For scale: RadixArk's NVFP4 conversion of this model (the SGLang vendor's own, via NVIDIA ModelOpt) reports AIME 2026 at 92.45 %
using `max_tokens=131072`, with a p90 of 53,342 generated tokens — 63 % above the entire cap
used here.

**Do not compare the `hle` figure to the base model's published HLE.** They measure different
things, in four ways that all push this number down:

* **no tools.** The upstream number is *HLE with tools*; this run uses none.
* **`max_tokens` 32,768** vs upstream's 163,840 — the truncation above.
* **bounded prefix** (n=1,476) rather than the full ~2,500-sample set.
* **judge is GLM-5.2**, not the GPT-class judge upstream used.

Every one of these settings is held identical across the checkpoints this was compared
against, which is what makes them useful for *comparing* quantizations — and is exactly why
they are not absolute scores.

## Serving

**Image, digest-pinned** — `glm5_next` is a new architecture, so a generic SGLang tag will
not load it:

```
lmsysorg/sglang:glm-5.3-flash@sha256:e6f5482505e7502f791fe4615ad1fbec118cbbd6b44e98f2479b16b98b985ad6
```

In-image: sglang `0.0.0.dev1+g033446bb05`, torch `2.13.0+cu130`, transformers `5.16.0.dev0`.

```bash
docker run --gpus all --ipc=host --shm-size 32g -p 8000:8000 \
  -v /path/to/GLM-5.3-Flash-NVFP4:/models/GLM-5.3-Flash-NVFP4 \
  lmsysorg/sglang:glm-5.3-flash@sha256:e6f5482505e7502f791fe4615ad1fbec118cbbd6b44e98f2479b16b98b985ad6 \
  python3 -m sglang.launch_server \
    --model /models/GLM-5.3-Flash-NVFP4 \
    --served-model-name GLM-5.3-Flash \
    --quantization modelopt_fp4 \
    --tensor-parallel-size 4 \
    --kv-cache-dtype fp8_e4m3 \
    --moe-runner-backend flashinfer_trtllm \
    --disable-shared-experts-fusion \
    --reasoning-parser glm45 \
    --tool-call-parser glm47 \
    --chunked-prefill-size 32768 \
    --mem-fraction-static 0.80 \
    --cuda-graph-max-bs 64 \
    --max-running-requests 256 \
    --trust-remote-code \
    --host 0.0.0.0 --port 8000
```

**`--quantization modelopt_fp4` is required.** SGLang's `compressed-tensors` MoE path
supports only W4A4 NVFP4; a weight-only NVFP4 MoE checkpoint loads only through the
`modelopt` path (`ModelOptNvFp4FusedMoEMethod`).

vLLM reads the same modelopt NVFP4 format and is expected to work with a build supporting
`glm5_next`, but **was not tested here**:

```bash
vllm serve /path/to/GLM-5.3-Flash-NVFP4 --served-model-name GLM-5.3-Flash \
  --quantization modelopt_fp4 --tensor-parallel-size 4 \
  --kv-cache-dtype fp8_e4m3 --trust-remote-code --host 0.0.0.0 --port 8000
```

This is a reasoning model served with `--reasoning-parser glm45`: the chain of thought
arrives in `reasoning_content` and `content` stays empty until it finishes, so give
requests a real token budget (2048+).

## Notes

* Weight-only: activations are bf16, trading some W4A4 throughput for needing no
  calibration set.
* **Speculative decoding (MTP/EAGLE) works**, and needs two things beyond the defaults.
  Layer 45's experts are NVFP4 like every other expert layer, and the NextN draft model
  loads from them — measured on 4xB200 TP4:

  | | |
  |---|---|
  | draft model | `Glm5NextForConditionalGenerationNextN`, 1.68 GB/rank |
  | accepted tokens per verification step | **2.475** |
  | draft acceptance rate | **49.2 %** |
  | single-request decode | **288 tok/s** |

  1. **Declare the checkpoint as `modelopt_mixed`.** The `quantization_config` shipped here
     is flat (`quant_method: modelopt`, `quant_algo: NVFP4`), and SGLang's NextN path
     *discards* per-layer quantization metadata under `--quantization modelopt_fp4`, then
     reads the packed 4-bit layer-45 weights as if unquantized — the
     `size of tensor a (4096) must match the size of tensor b (2048)` you will otherwise
     hit (4096 = one byte per value, 2048 = two 4-bit values per byte). Replacing
     `config.json` with the provided **`config.modelopt_mixed.json`**, which names layers
     3–45 explicitly, fixes it. No weight changes.
  2. **Use a NextN-name-aware SGLang build.** Stock rewrites
     `model.layers.<n>` → `model.decoder` by *substring*, so this model's
     `model.language_model.layers.45.*` becomes `model.language_model.decoder.*` and the
     draft modules never bind. This is an upstream defect that affects every GLM-5.3-Flash
     conversion, not just this one.

  Without both, serve without `--speculative-algorithm` — which is how the scores above
  were produced, so they are unaffected either way.
* The vision tower is not quantized, so the size reduction applies to the text model.

## License

MIT, inherited from the base model.
