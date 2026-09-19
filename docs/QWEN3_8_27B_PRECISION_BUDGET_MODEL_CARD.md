---
license: apache-2.0
library_name: transformers
pipeline_tag: image-text-to-text
model_name: Qwen3.8-27B-MxWave-MXFP4-FP8
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
  - qwen3.8
  - qwen
  - mxfp4
  - fp8
  - mixed-precision
  - compressed-tensors
  - quantized
  - vllm
  - blackwell
  - dgx-spark
  - mxwave
  - multimodal
---

# Qwen3.8-27B — MxWave MXFP4/FP8 precision budget

This is an experimental quality-oriented mixed-precision conversion of
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B). It starts from the
published MxWave H64 MXFP4 checkpoint and uses MxWave's measured precision-budget
workflow to retain MXFP4 for 388 language projections while promoting 12 measured,
fusion-safe projections to channel-wise FP8.

Hugging Face repository:
`kirillbilchenko/Qwen3.8-27B-MxWave-MXFP4-FP8`.

This checkpoint is a qualified optional quality variant. It is about 2% larger
and 2.2% slower than H64 on the tested DGX Spark runtime. It has slightly better
paired perplexity and a positive, but statistically inconclusive, GSM8K result.
H64 remains the smaller and faster MxWave default.

This is not an official Qwen, NVIDIA, or AMD release.

| Property | Value |
|---|---:|
| Directory size | **20,250,144,780 bytes (18.86 GiB)** |
| Manifest-accounted tensor data | 20,226,686,160 bytes |
| Source tensor data | 55,562,855,904 bytes |
| Global compression ratio | 2.747x |
| Language projections | 388 MXFP4 + 12 FP8 |
| Source-precision passthrough tensors | 799 |
| Runtime format | `compressed-tensors` mixed MXFP4/FP8 |
| Primary text result | WikiText-2 prompt PPL **8.111578** |

## Quick start

Download with the current Hugging Face CLI:

```bash
HF_XET_HIGH_PERFORMANCE=1 hf download \
  kirillbilchenko/Qwen3.8-27B-MxWave-MXFP4-FP8 \
  --local-dir ./Qwen3.8-27B-MxWave-MXFP4-FP8
```

The tested runtime is official vLLM 0.29.0 on DGX Spark / SM121:

```bash
vllm serve ./Qwen3.8-27B-MxWave-MXFP4-FP8 \
  --load-format safetensors \
  --dtype bfloat16 \
  --linear-backend marlin \
  --max-model-len 16384 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.45 \
  --kv-cache-dtype bfloat16 \
  --no-enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-batched-tokens 8192
```

The FP8 and MXFP4 kernels are weight-only on this tested path; activations remain
BF16. Do not use the obsolete `VLLM_TEST_FORCE_FP8_MARLIN` compatibility switch
with vLLM 0.29.0.

## Precision allocation

The candidate promotes these runtime-safe groups:

- sequence-output projections in layers 8-11: three linear-attention `out_proj`
  tensors and one full-attention `o_proj` tensor;
- MLP input projections in layers 51-54: four `gate_proj` and four `up_proj`
  tensors.

The remaining 388 targeted language projections retain the original H64 OCP
MXFP4 E2M1 weights and E8M0 block-32 scales. Embeddings, language head, norms,
MTP tensors, vision tensors, and other non-target tensors remain source precision.
The exact allocation is recorded in [`mxwave-manifest.json`](./mxwave-manifest.json).

## Selection method

The allocation was fixed before final evaluation:

1. MxWave generated 12 semantic candidates spanning four runtime-operation
   families and three depth bands.
2. Each candidate was compared with H64 against the same BF16 next-token
   distributions on two disjoint 16-context splits.
3. A candidate was eligible only when mean forward KL improved on both splits.
4. Selection admitted at most four buckets and 1 GiB of added tensor data.
5. The combined allocation then had to improve both halves of an untouched
   96-context holdout without losing more than one BF16 top-1 agreement.

The final allocation promotes 12 of 400 quantized tensors and adds 393,854,976
tensor bytes (375.6 MiB). It does not use task labels for tensor selection.

The implementation and model-independent procedure are in
[MxWave](https://github.com/kirillbilchenko/MxWave) and its
[precision-budget runbook](https://github.com/kirillbilchenko/MxWave/blob/main/docs/PRECISION_BUDGET_RUNBOOK.md).

## Evaluation

All H64/candidate comparisons below are paired: same prompts, task definitions,
seeds, decoding, runtime image, and backend. The final runtime was
`vllm/vllm-openai:v0.29.0` at digest
`sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1`.

### WikiText-2 prompt perplexity

The primary deterministic evaluation used 316 independent 4,096-character
windows and 297,199 scored tokens.

| Model | Prompt PPL | Candidate comparison |
|---|---:|---:|
| BF16 source | **7.980864** | 1.6378% higher PPL |
| MxWave H64 | 8.119445 | **0.0969% lower PPL** |
| **MxWave MXFP4/FP8** | **8.111578** | baseline |
| AMD Quark-AWQ MXFP4 | 8.187544 | **0.9278% lower PPL** |

The paired candidate-versus-H64 95% bootstrap interval was
`[-0.1297%, -0.0637%]`, entirely below zero. Absolute perplexities are specific
to this character-window protocol and must not be compared directly with scores
from different token windows or strides.

### Untouched distribution holdout

| Metric | H64 | Candidate |
|---|---:|---:|
| Mean forward KL to BF16 | 0.047117 | **0.044942** |
| BF16 top-1 agreement | 89/96 | 89/96 |

Candidate-minus-H64 mean forward KL was `-0.002176` nats and was negative on
both predeclared holdout halves. The paired interval
`[-0.005939, +0.000523]` crossed zero, so this holdout passed the directional
gate but is not an independent statistically conclusive claim.

### Full deterministic GSM8K

This run used lm-evaluation-harness 0.4.13, all 1,319 test examples, five-shot
prompts, greedy decoding, temperature zero, and 1,024 maximum generated tokens.
The frozen `gsm8k_nothink` task SHA-256 was
`0a5c248412fa400d32386b4c8c65aeb025711650dda0e6c0ec39e67e86cda5c7`;
both runs completed with zero failed requests.

| Metric | H64 | Candidate | Difference |
|---|---:|---:|---:|
| Strict match | 1,208/1,319 (91.5845%) | **1,214/1,319 (92.0394%)** | **+0.4549 points** |
| Flexible extract | 1,208/1,319 (91.5845%) | **1,215/1,319 (92.1152%)** | **+0.5307 points** |

For strict matching, candidate-only wins were 25 and H64-only wins were 19.
Exact McNemar was `p=0.4514`; the paired interval was
`[-0.5307, +1.4405]` points. The direction is favorable and passes the
predeclared one-point non-inferiority gate, but it is not a statistically proven
GSM8K improvement.

### Serving tradeoff

The fixed workload used random 512-token inputs, 128 generated tokens,
temperature zero, ignored EOS, and disabled prefix caching.

| Model | Concurrency | Output tok/s |
|---|---:|---:|
| H64 | 1 | 12.484 |
| Candidate | 1 | 12.213 |
| H64 | 4 | 42.193 |
| Candidate | 4 | 41.230-41.250 |

The candidate was 2.17% slower at concurrency 1 and 2.24-2.28% slower at
concurrency 4, within the frozen 3% budget. The deterministic JSON, tool-use,
short-completion, and Python smoke suite passed all 12 repeated checks.

Machine-readable summaries:

- [`evaluation/precision-budget.json`](./evaluation/precision-budget.json)
- [`evaluation/qualification.json`](./evaluation/qualification.json)
- [`evaluation/gsm8k-paired-full.json`](./evaluation/gsm8k-paired-full.json)

## Limitations

- The measured improvements are small. This is a Pareto option, not a claim that
  every user should prefer it over H64.
- GSM8K favored the candidate but its paired uncertainty includes zero.
- Evaluation was text-only. Vision tensors are present and unchanged, but image
  capability was not independently evaluated.
- Long-context retrieval, coding benchmarks, and broad instruction-following
  suites were not used to make the claims above.
- Runtime support depends on vLLM, compressed-tensors, CUDA, and accelerator
  versions. The checkpoint is not a GGUF, MLX, llama.cpp, Ollama, or LM Studio
  model.
- The selected allocation is specific to Qwen3.8-27B. The method is
  architecture-adapter based, but another model must run its own frozen screen.

## Reproducibility and provenance

- Source model: `Qwen/Qwen3.8-27B` revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Public MxWave precision-budget implementation:
  [`6f7ab4ec`](https://github.com/kirillbilchenko/MxWave/tree/6f7ab4ec5871ad1cef766176c74eadf3b1d5a9ae).
- H64 parent manifest SHA-256:
  `8c9e4205941263bec879712fc790aabd8f75cc23dce53d9fb13daecb2161040f`.
- Mixed config SHA-256:
  `ff5cbabc9c487eea826373b209492e1b5e3d3eaa4b4216729cfe9f4fa4c81018`.
- Mixed index SHA-256:
  `d359434efbacc1a93c38221d5d554883d27554c2a77e55b41804ac09735db922`.
- Mixed manifest SHA-256:
  `ed3ed368221813a1c5bc9ab7dcc150506e2d254172446f1f493bba6923983263`.
- Ordered 18-shard hash-of-hashes:
  `2d84bd37a4c932e01c9f7305b4fd4a0191d4285f854f86a4fae07853c45db91b`.
- Full paired GSM8K report SHA-256:
  `aa4fab87c4fe98b36238f5e32e4115b43d1dbdc52343c652500725bbf2ed9c26`.

The detailed selection protocol, uncertainty, hashes, and runtime results are in
the
[Qwen precision-budget record](https://github.com/kirillbilchenko/MxWave/blob/main/docs/QWEN3_8_27B_PRECISION_BUDGET.md).
