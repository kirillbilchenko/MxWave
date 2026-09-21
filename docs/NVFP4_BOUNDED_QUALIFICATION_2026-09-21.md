# Qwen3.8-27B native-NVFP4 bounded qualification

Status: completed; rejected at the Phase 1 quality gate.

Date: 2026-09-21

## Question

Can an existing native NVFP4 W4A4 checkpoint improve the practical
size/quality/speed frontier over MxWave H64 on the DGX Spark strongly enough to
justify implementing a production NVFP4 backend in MxWave?

This is a diagnostic comparison, not a claim that the external checkpoint was
created by MxWave.

## Frozen artifacts

| Role | Artifact | Revision / value |
|---|---|---|
| Source family | `Qwen/Qwen3.8-27B` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| MxWave baseline | `kirillbilchenko/Qwen3.8-27B-MXFP4-MxWave` | `c0ffe6b8e5b099bf490395c047c8cbea021b100e` |
| Native-W4A4 candidate | `minima-ai/mnma_qwen3.8_27b_nvfp4` | `16e768e7d0461b0b86e565ecedd08a24eca53e9a` |
| Runtime | `vllm/vllm-openai:v0.29.0` | digest `c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1` |
| Hardware | NVIDIA DGX Spark | GB10 / SM121, 128 GiB unified memory |

The first run must use the stock image above. A patched image would answer a
different systems question and is not allowed to rescue Phase 0.

## Capability and size accounting

Raw repository size is not a fair comparison. The Minima checkpoint is a
text-only `Qwen3_5ForCausalLM` artifact and has neither vision tensors nor MTP
tensors. H64 retains both.

Measured tensor payloads:

| Artifact | Raw payload | Capability note |
|---|---:|---|
| MxWave H64 | 18.4706 GiB | language + vision + MTP |
| Minima NVFP4 | about 17.52 GiB | language only; no vision or MTP |
| H64 normalized to language-only | about 16.8213 GiB | subtracts 0.85818 GiB vision and 0.79106 GiB MTP |

The apparent raw Minima saving therefore reverses after normalizing capability:
the candidate is about 0.676 GiB, or roughly 4%, larger than language-only H64.
Minima cannot pass a claim of "more compression than H64". It can still pass as
a native-W4A4 runtime/quality result.

The old MxWave `research/nvfp4` artifact is not a candidate. It is 26.5907 GiB,
uses W4A16, covers only 256 targets, ignores 681 modules, and has no activation
scales. Its group-16 packing primitives may be reusable later, but the branch
must not be merged wholesale.

## Phase 0: structural, load, and execution validity

Run with:

- stock vLLM 0.29.0;
- tensor parallelism 1;
- MTP/speculative decoding disabled;
- BF16 KV cache, matching the H64 quality reference;
- no DFlash, SGLang, custom kernel, or patched model code;
- the pinned checkpoint revision above.

Required evidence:

1. Safetensors and quantization metadata load without repair.
2. The server or in-process evaluator completes a deterministic smoke request.
3. Logs identify the selected linear kernels.
4. Intended W4A4 projections execute through a native W4A4 path; a Marlin
   W4A16 fallback is a Phase 0 failure for the speed hypothesis.
5. The run records vLLM, compressed-tensors, FlashInfer, CUDA, driver, and image
   versions.

Stop on load failure, unsupported compressed-tensors metadata, numerical error,
or an execution path that makes the candidate W4A16 rather than W4A4.

## Phase 1: frozen quality screen

Use the already frozen inputs and reference artifacts on Spark:

- WikiText-2 prompt-PPL corpus:
  `/home/kirya/local-spark/runtime/benchmarks/wikitext2/wikitext-2-raw-v1-test.txt`
- H64 316-window result:
  `/home/kirya/local-spark/runtime/benchmarks/wikitext2/recovery-final-20260920/h64-v029-4k316.json`
- exact 128-context divergence directory:
  `/home/kirya/local-spark/runtime/benchmarks/next-token-kl/exact-128-20260916`

Frozen H64 anchors:

- prompt PPL: `8.120844647734497` over 297,199 scored tokens;
- mean forward KL from BF16: approximately `0.049937` nats.

The local recovered checkpoint is a second, stronger quality anchor. Its prompt
PPL is `8.0677605368`, although its prior divergence evidence was mixed and it
did not replace H64 as the public default.

Passing requires all of the following:

1. Candidate PPL is no higher than H64.
2. The paired 95% interval for candidate-minus-H64 per-window NLL has a
   non-positive upper bound.
3. Candidate mean forward KL is no higher than H64.
4. The paired 95% interval for candidate-minus-H64 forward KL has a
   non-positive upper bound.
5. No material regression appears in forward-KL p95, reverse KL, JS, TV, or
   top-token agreement.

These are intentionally strict promotion gates. A result may still be useful
scientifically while being rejected for production.

Two decisions are kept separate:

- **native-W4A4 feasibility:** non-inferior to H64 plus a verified native
  runtime gain;
- **actual project advance:** beat the recovered checkpoint on paired PPL
  without its KL/tail regressions, while also improving capability-normalized
  size. The text-only Minima artifact cannot pass the size part of this gate.

## Phase 2: runtime comparison

Run only if Phase 1 passes. Compare H64 and Minima one at a time under the same
stock image and workload definitions. Record:

- fresh-runtime-cache startup and warm restart;
- resident and peak host/accelerator memory;
- time to first token;
- prompt throughput;
- single-stream decode;
- concurrency-four aggregate throughput;
- exact kernel coverage by projection family.

A practical runtime win requires at least 15% improvement in one workload that
matters to the intended deployment without a material regression in the other
primary workload. A smaller isolated kernel gain is not enough.

## Phase 3: expensive qualification

Run only if the first two phases pass:

- full deterministic GSM8K;
- serving qualification and repetition stability;
- long-context non-inferiority.

Vision and MTP remain unsupported by this candidate and must be reported as
missing rather than inferred from the parent model.

## Decision rule

| Outcome | Decision |
|---|---|
| Load or native-W4A4 execution fails | Do not port NVFP4 from this evidence |
| Quality fails | Keep the result as a negative control; do not run expensive tasks |
| Quality passes but speed gain is below 15% | No production backend; native W4A4 is not valuable enough here |
| H64 quality and speed pass, but Recovery/normalized-size gate fails | Runtime feasibility only; not a new MxWave frontier |
| Quality, speed, Recovery, and normalized-size gates pass | Consider a new full-capability NVFP4 branch; do not reuse the old branch wholesale |
| Raw size looks smaller but normalized size does not | Describe it as a runtime alternative, never as better compression |

## How crowded the area is

NVFP4 checkpoint production is already commodity-scale. A 2026-09-21 public
Hugging Face API sample found more than 2,000 repositories tagged `nvfp4`, with
hundreds of search matches around this exact Qwen family. Those totals include
copies, conversions, and finetunes; they are adoption counts, not independent
methods.

The exact checkpoint already has several credible, distinct recipes:

- [NVIDIA mixed NVFP4/FP8](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4),
  using Local-Hessian calibration;
- [Red Hat AI AWQ/GPTQ](https://huggingface.co/RedHatAI/Qwen3.8-27B-NVFP4);
- [Minima all-backbone W4A4 PTQ](https://huggingface.co/minima-ai/mnma_qwen3.8_27b_nvfp4);
- [QUASAR all-W4A4 QAT](https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4);
- [RadixArk mixed NVFP4/FP8](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4);
- [Unsloth dynamic mixed precision](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4).

Plain W4A4, W4A16, mixed FP4/FP8 scopes, Local-Hessian scale selection,
AWQ/GPTQ, generic QAT, and generic sensitivity allocation are crowded. A new
MxWave backend implementing only those features would be useful engineering,
not a novel quantization contribution.

The most defensible remaining project direction is runtime-measured allocation:
combine held-out quality sensitivity with measured latency for each real
operator shape, precision, and selected SM121 kernel; enforce fused-group scale
legality; and include MTP acceptance or total speculative throughput when MTP is
present. This remains an experiment hypothesis, not a novelty claim.

## Result

### Phase 0: passed

The pinned Minima checkpoint loaded without metadata repair in the unchanged
vLLM 0.29.0 image. vLLM selected
`FlashInferCutlassNvFp4LinearKernel for NVFP4 GEMM`; this was a native W4A4
test, not a Marlin W4A16 fallback. Weight loading took 141.81 seconds and the
recorded compile step took 30.06 seconds. The runtime was:

- vLLM 0.29.0;
- compressed-tensors 0.17.0;
- FlashInfer 0.6.18;
- PyTorch 2.13.0+cu130 and CUDA 13.0;
- NVIDIA driver 580.142 on GB10, compute capability 12.1.

### Phase 1: failed

The candidate was worse on both frozen deterministic quality screens.

| Model | Prompt PPL | Relative to H64 |
|---|---:|---:|
| BF16 | 7.980864 | -1.724% |
| Recovery | 8.067761 | -0.654% |
| H64 | 8.120845 | baseline |
| Minima NVFP4 | 8.557928 | **+5.382%** |

All PPL variants used the same 316 windows and 297,199 scored tokens. The
paired cluster-bootstrap 95% interval for Minima relative to H64 was
`[+4.787%, +6.053%]`; Minima was better on only 18 of 316 windows. Relative to
Recovery it was `+6.076%`, with interval `[+5.472%, +6.758%]`, and was better
on only 7 windows. Removing its single most favorable window did not rescue
either comparison.

The exact next-token comparison used the same 128 contexts and complete
248,320-token output vocabulary for each model:

| Metric vs BF16 | H64 | Recovery | Minima NVFP4 |
|---|---:|---:|---:|
| Forward KL mean, nats | 0.049937 | 0.049088 | **0.072462** |
| Forward KL p95, nats | 0.175241 | 0.187826 | **0.320213** |
| Reverse KL mean, nats | 0.044734 | 0.048015 | **0.076037** |
| Jensen-Shannon mean, nats | 0.010946 | 0.011166 | **0.016406** |
| Total variation mean | 0.078565 | 0.079351 | **0.096146** |
| BF16 top-1 agreement | 93.750% | 92.969% | **91.406%** |

Minima's mean forward KL was 45.1% above H64 and its p95 was 82.7% above
H64. Expressed as candidate minus H64, the paired mean forward-KL difference
was `+0.022525` nats with bootstrap 95% interval
`[+0.007275, +0.038768]`. H64 had lower per-context KL on 88 contexts and
Minima on 40. This is a statistically resolved regression, not small-sample
ambiguity.

### Decision

Reject this NVFP4 candidate for the configured MxWave goal. It is about 5.27%
smaller by raw payload only because it omits vision and MTP. Against the
language-only portion of H64 it is about 4.02% larger, while also losing
substantial quality. The native kernel result establishes technical feasibility
on SM121, but not a size/quality frontier improvement.

Per the pre-registered stopping rule, throughput, GSM8K, and long-context tests
were not run. They cannot repair a deterministic quality and normalized-size
failure. No MxWave NVFP4 backend should be opened from this evidence, and no
public checkpoint should be derived from it.

The next public all-W4A4 candidate, QUASAR, is useful only as an optional QAT
quality ceiling: its full-capability weights are 19.147 GiB, already 3.66%
larger than H64. NVIDIA, RadixArk, and Unsloth full-capability variants are
larger still. None can satisfy the original "smaller and lower-loss" gate
before inference.

### Generated evidence

The generated reports remain on Spark under the frozen benchmark directories:

- PPL report:
  `wikitext2/minima-nvfp4-v029/minima-v029-4k316.json`, SHA-256
  `da1ffb27d6e820b30497cdeceb3d6b6049784fdb72c8cf355ece011e54caf106`;
- PPL versus H64:
  `wikitext2/minima-nvfp4-v029/minima-v029-vs-h64.json`, SHA-256
  `a1a70f5c55fccaf8733641088933fd533434c7b89548697e7bf50b78c32c7912`;
- PPL versus Recovery:
  `wikitext2/minima-nvfp4-v029/minima-v029-vs-recovery.json`, SHA-256
  `eff021f87a415e1e048e87f823060c2282b593306a257cbb34fc98bbbfb241c7`;
- next-token logits:
  `next-token-kl/exact-128-20260916/minima-nvfp4-v029-logprobs.safetensors`,
  SHA-256
  `772d2281f8f87379bcbd1d9ffea2177533f70b6ae265f4d33238a7f8f7706fdd`;
- divergence report:
  `next-token-kl/exact-128-20260916/minima-nvfp4-v029-divergence.json`,
  SHA-256
  `ffc0b5a054e91e230ae7d32e52d818b783aa2a5bc00a29fab20a0997c40a0dc3`.

Temporary experiment containers were removed after the reports and hashes were
captured. The pinned checkpoint and result files were retained.
