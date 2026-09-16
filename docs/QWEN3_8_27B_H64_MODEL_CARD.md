---
license: apache-2.0
library_name: transformers
pipeline_tag: image-text-to-text
model_name: Qwen3.8-27B-MXFP4-MxWave
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
  - qwen3.8
  - mxfp4
  - compressed-tensors
  - quantized
  - vllm
  - mxwave
  - multimodal
---

# Qwen3.8-27B — MXFP4 by MxWave

This is a quality-oriented, post-training MXFP4 conversion of
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B), built from revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` with
[MxWave](https://github.com/kirillbilchenko/MxWave) `0.1.0`. The public quantizer
baseline is pinned to
[`0179e131`](https://github.com/kirillbilchenko/MxWave/tree/0179e131544f807ecdb6d04e7e914f181dd21c9f).

Hugging Face repository: `kirillbilchenko/Qwen3.8-27B-MXFP4-MxWave`.

`H64` is the calibration recipe, not a precision: **H** means block-Hessian scale
selection and **64** means 64 calibration sequences. The stored weights remain standard OCP
MXFP4 E2M1 with E8M0 scales and 32-value blocks. `d4` in the build record means an MSE
clipping search depth of four.

This checkpoint is experimental and is not an official Qwen or AMD release.

| Property | Value |
|---|---:|
| Tensor data | **19,832,831,240 bytes (18.47 GiB)** |
| Source tensor data | 55,562,855,904 bytes (51.75 GiB) |
| Reduction | **64.31%** / 2.8016x smaller |
| Format | `compressed-tensors` `mxfp4-pack-quantized` |
| Calibrated language projections | 400 |
| Passthrough tensors | 799 |
| Calibration | 64 × 512 tokens = 32,768 tokens |
| Primary text result | WikiText-2 prompt PPL **8.119445** |

## What was quantized

The base is a dense 64-layer hybrid model with 48 linear-attention layers and 16
full-attention layers. MxWave quantized these language-model projections:

| Component | Count | Stored precision |
|---|---:|---|
| MLP `gate_proj`, `up_proj`, `down_proj` | 192 | MXFP4 |
| Linear-attention `in_proj_qkv`, `in_proj_z`, `out_proj` | 144 | MXFP4 |
| Full-attention `q_proj`, `k_proj`, `v_proj`, `o_proj` | 64 | MXFP4 |
| **Total** | **400** | **MXFP4** |

Embeddings, language-model head, norms, linear-attention auxiliary tensors, MTP tensors,
vision encoder, and vision merger are among the 799 source-precision passthrough tensors.
The exact target and ignore lists are recorded in
[`mxwave-manifest.json`](./mxwave-manifest.json).

The vision tensors are unchanged, but image capability was **not** independently evaluated;
do not interpret the text-only results below as a multimodal quality claim.

## Quantization method

For every 32-value weight block, MxWave evaluates nearby shared E8M0 scale exponents.
Instead of minimizing raw weight MSE alone, H64 minimizes a block-Hessian proxy for output
error using real input second moments collected from calibration:

`error ≈ (W - W_quant) H (W - W_quant)^T`.

Calibration used the first 100 validation rows of `EleutherAI/pile_val_test`, frozen by
repository revision and corpus hash, to form 64 sequences of 512 tokens. All 400 target
projections received block-32 input second-moment statistics with damping `1e-6`.

After scale selection, values are assigned to their nearest E2M1 codes. This final recipe is
deliberately scale-only: it uses no sequential GPTQ rounding, error feedback, cross-block
recovery, rotation, mixed-precision layer promotion, or recovery training. Those experimental
variants did not improve the held-out result enough to justify inclusion.

## Evaluation

All comparisons used the same DGX Spark (GB10, SM121), model source, prompts, tokenizer,
and pinned vLLM image:

`vllm/vllm-openai:qwen38-flash-next@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8`

The comparison checkpoint was
[`amd/Qwen3.8-27B-Quark-AWQ-MXFP4`](https://huggingface.co/amd/Qwen3.8-27B-Quark-AWQ-MXFP4)
at revision `5233554c5fa56afda40150556b95573c2d7d29c0`.

The MXFP4 models used the Marlin weight-only kernel, so these runtime results are **W4A16**.
They are not native W4A4 measurements.

### WikiText-2 prompt perplexity

This is the primary deterministic quality comparison: 316 independent 4,096-character
windows from the pinned WikiText-2 raw test corpus and 297,199 scored tokens.

| Model | Prompt PPL | Relative to BF16 |
|---|---:|---:|
| BF16 source | **7.980864** | baseline |
| **MxWave** | **8.119445** | **+1.736%** |
| AMD Quark-AWQ MXFP4 | 8.187544 | +2.590% |

H64's perplexity was **0.832% lower than AMD's**. The paired 95% cluster-bootstrap interval
was `[-1.227%, -0.531%]`; H64 won 204 of 316 paired windows. Removing the single largest
favorable outlier left a 0.684% advantage.

Absolute PPL values are specific to this character-window protocol and should not be compared
with literature numbers produced using different token windows or strides. Full protocol:
[`evaluation/wikitext2-prompt-ppl.json`](./evaluation/wikitext2-prompt-ppl.json).

### Exact next-token distribution divergence

For 128 deterministic held-out WikiText contexts of up to 512 tokens, the evaluator collected
all **248,320** next-token log probabilities and compared each candidate with BF16.

| Metric | **MxWave** | AMD Quark-AWQ MXFP4 |
|---|---:|---:|
| Mean forward KL, nats | **0.049937** | 0.056735 |
| Forward-KL p95, nats | **0.175241** | 0.232540 |
| Mean reverse KL, nats | **0.044734** | 0.048565 |
| Mean Jensen-Shannon divergence, nats | **0.010946** | 0.012083 |
| Mean total variation | **0.078565** | 0.082362 |
| BF16 top-1 agreement | **93.75%** | 91.41% |
| Mean BF16 top-5 overlap | **4.359 / 5** | 4.297 / 5 |

H64's mean forward KL was about **12.0% lower** and it was lower on 70 of 128 contexts.
However, the paired 10,000-resample interval for `H64 - AMD` was
`[-0.02424, +0.00984]` nats, which crosses zero. This is positive evidence for H64, not a
statistically conclusive overall win. Full protocol and per-context metrics:
[`evaluation/next-token-divergence.json`](./evaluation/next-token-divergence.json).

### GSM8K pilot

This was a sampled 100-example, 5-shot screening run, not the primary ranking metric.

| Model | Flexible numeric match | Strict answer match |
|---|---:|---:|
| BF16 source | 88% | 85% |
| **MxWave** | **92%** | **91%** |
| AMD Quark-AWQ MXFP4 | 93% | 93% |

Strict matching required the standard GSM8K `####` final-answer marker.

Generation used temperature `0.7`, top-p `0.8`, top-k `20`, seed `1234`, and 16 concurrent
requests. H64's one-point flexible gap to AMD is smaller than the reported standard errors.
The paired trace audit found only one case where BF16 and AMD reasoned correctly while H64 did
not, plus one H64 strict-format-only failure. BF16 scoring below both quantized models further
shows why this small stochastic pilot must not be treated as a precise quantization ranking.
Protocol and raw-report hashes:
[`evaluation/gsm8k-pilot.json`](./evaluation/gsm8k-pilot.json).

### Reconstruction checks

| Check | Mean | Minimum | Coverage |
|---|---:|---:|---:|
| Sampled weight SQNR | 18.8208 dB | 17.3291 dB | 400 / 400 |
| Block-Hessian-weighted SQNR | 19.1403 dB | 17.2547 dB | 400 / 400 |

These checks detect packing, scale, and coverage failures; they are not substitutes for the
end-to-end PPL and distribution tests above.

## Serving with vLLM on DGX Spark

The tested SM121 path requires Marlin:

```bash
vllm serve . \
  --load-format safetensors \
  --dtype bfloat16 \
  --linear-backend marlin \
  --max-model-len 16384 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.45 \
  --kv-cache-dtype bfloat16 \
  --no-enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-batched-tokens 8192 \
  --no-enable-flashinfer-autotune
```

On this runtime, the expected startup log says `Using MarlinMxFp4LinearKernel`. It also warns
that activation quantization is ignored because Marlin is weight-only; inference therefore
uses packed MXFP4 weights with BF16 activations. Without the Marlin selection, the tested
FlashInfer `cute-dsl` FP4 backend rejected compute capability 12.1.

DGX Spark uses unified memory. Start with a conservative memory fraction, monitor
`MemAvailable`, and do not add `--enforce-eager`. Stop OpenWebUI and other inference traffic
when reproducing evaluation scores.

Runtime support is sensitive to the exact vLLM, compressed-tensors, FlashInfer, CUDA, and GPU
combination. A model that loads on this pinned SM121 image is not automatically portable to
every vLLM release or accelerator.

## Reproducibility and provenance

- Quantization tool and source:
  [MxWave on GitHub](https://github.com/kirillbilchenko/MxWave), commit
  [`0179e131544f807ecdb6d04e7e914f181dd21c9f`](https://github.com/kirillbilchenko/MxWave/tree/0179e131544f807ecdb6d04e7e914f181dd21c9f)
- Full commands and immutable hashes:
  [`REPRODUCIBILITY.md`](./REPRODUCIBILITY.md)
- Quantization manifest and per-tensor SQNR:
  [`mxwave-manifest.json`](./mxwave-manifest.json)
- Source model revision:
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- Calibration corpus SHA-256:
  `07ffd944ea4fc09f12e5d019633a77309199fdced749e3ddf13989e53586f36a`
- Calibration token IDs SHA-256:
  `e68f97cb4044312513df4c9c82bd758abe348a706fd7e1f11fe9ee15c83efa05`
- Published 18-shard/config/index hash-of-hashes:
  `6f81fb0182aca2a712b1d688d96871c81ef75e04bcc8622c79275aca60828d88`
- Published inference-payload hash-of-hashes:
  `0600d65150f8e44713897f6515af0c7ddf704d148e83c148217173bf1ff4917b`

The replay produced bitwise-identical calibration tensors and all 18 quantized weight shards.
For Hub schema compatibility, the published `config.json` omits a redundant group-level
`"format": null`; the global `mxfp4-pack-quantized` format is unchanged, and both
compressed-tensors 0.17 and the pinned vLLM parser resolve the omitted value to `None`.
Generated prose and telemetry-bearing provenance files are excluded from the payload hash.

## Limitations

- The main quality tests are text-only; vision and MTP behavior were not independently scored.
- The runtime benchmark used Marlin W4A16, even though the emitted config also describes
  dynamic group-32 MXFP4 activation quantization for compatible native kernels.
- The KL comparison favors H64 on aggregate but its paired confidence interval crosses zero.
- GSM8K used only 100 sampled examples and is directional evidence only.
- No controlled throughput result is reported; interactive OpenWebUI measurements are not a
  benchmark.
- Calibration can favor distributions resembling its corpus. Evaluate your own tasks before
  deployment.
- This is pure post-training quantization; it does not include recovery training or QAT.

## License

This derivative checkpoint inherits the Apache-2.0 license from the base model. See
[`LICENSE`](./LICENSE) and the [base model card](https://huggingface.co/Qwen/Qwen3.8-27B).
