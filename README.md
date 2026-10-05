# MxWave

[![Hugging Face model](https://img.shields.io/badge/🤗%20Hugging%20Face-Qwen3.8--27B--MXFP4--MxWave-FFD21E)](https://huggingface.co/kirillbilchenko/Qwen3.8-27B-MXFP4-MxWave)
[![CI](https://github.com/kirillbilchenko/MxWave/actions/workflows/ci.yml/badge.svg)](https://github.com/kirillbilchenko/MxWave/actions/workflows/ci.yml)

**Quality-oriented MXFP4 post-training quantization for LLMs, with real
activation calibration, bounded-memory checkpoint processing, and verified
vLLM output.**

**Published model:**
[`kirillbilchenko/Qwen3.8-27B-MXFP4-MxWave`](https://huggingface.co/kirillbilchenko/Qwen3.8-27B-MXFP4-MxWave)
— an 18.47 GiB Qwen3.8-27B checkpoint with complete evaluation and reproducibility records.

New to the project? Start with the
[human-readable methodology guide](docs/METHODOLOGY_GUIDE.md): it explains the
quantization math, evaluation ladder, successful path, failed experiments, and
the lessons behind the current design.

MxWave converts floating-point safetensors checkpoints to vLLM's
`compressed-tensors` `mxfp4-pack-quantized` format. It reads source tensors in
bounded chunks, can collect activation statistics one decoder layer at a time,
and validates packing, reconstruction quality, and module coverage before an
artifact is considered complete.

The first fully evaluated target is `Qwen/Qwen3.8-27B` on NVIDIA DGX Spark
(GB10/SM121). The quantization math is model-independent; weight-streamed
calibration currently has a verified adapter for the Qwen3.5-text decoder
layout used by that model. Unsupported streaming layouts fail closed.

## Why this repo exists

Memoryless min/max round-to-nearest is a useful MXFP4 baseline, but it ignores
which input channels matter most on real data. MxWave is built around three
ideas:

1. **MSE-optimal, calibration-aware scale selection** — an explicit exponent
   candidate set per block, chosen by minimum reconstruction error and optionally
   weighted by real activation moments or a block Hessian.
2. **Bounded residency** — calibration loads one supported decoder layer at a
   time, while quantization reads and processes tensors in bounded row chunks.
3. **Verification-first output** — emitted shapes, dtypes, module coverage, and
   sampled reconstruction quality are checked instead of inferred from the
   source checkpoint.

## Status

🚧 **Experimental, with a working core and streaming engine.** Module-keyed calibration,
shard discovery, on-device quantization, output assembly, and strict output
verification are implemented and unit-tested. A paired 297k-token DGX Spark
likelihood screen of the historical published artifact found it 0.83%
lower in perplexity than AMD Quark-AWQ MXFP4 and 1.74% above BF16. On 128
overlapping WikiText contexts it also had 12.0% lower mean next-token forward KL from BF16
than AMD, although that paired interval crosses zero. The exact protocols and
uncertainty are recorded below; the earlier 100-item task screens remain
diagnostic rather than reportable benchmark scores.

The checked-in benchmark JSON was produced before the MxWave rename and retains
its original `mxstream-*` schema and model labels so the recorded SHA-256 hashes
remain valid. Current Python imports, console commands, and newly generated
artifacts use `mxwave` exclusively.

Current conversions use quantization revision `mxfp4-rne-v2`: exact E2M1 ties
round to the even neighbour, and Qwen's norm proxy uses `abs(1 + weight)`.
These corrections can change newly generated weights. Published artifacts and
their recorded scores describe the earlier implementation; reproducing their
exact bytes requires the pinned release commit. Resume rejects runs from the
earlier revision. Corrected models have not yet received whole-model quality
benchmarks on a reserved final corpus. The fixed validation comparison below
has completed. See the [evaluation record](docs/QUANTIZATION_FIXES_AND_VALIDATION.md).

The corrected-code Spark chunk benchmark has completed two sweep orders with
identical packed/scales hashes. For the tested H64 matrices, the current
1,048,576-element budget was faster than larger chunks. Two further sweep
orders included production output copies, disk writes and fsync, with the
same conclusion on those matrices. The
[dated control experiment](docs/CORRECTED_CONTROLS_2026-10-03.md) records both
timing scopes and the completed fresh quality comparison.

### Corrected-code validation controls

The Hugging Face link above points to the **published legacy-rounding H64**
artifact: its historical test-split PPL is **8.119445**. The **7.954455** below
belongs to a newly converted, **unpublished** `mxfp4-rne-v2` H64 checkpoint on
the validation split. These different evaluation splits are not an old-versus-new
rounding ablation. Exact checkpoint identities are recorded in the linked reports.

The fixed `mxfp4-rne-v2` comparison completed on Spark using stock vLLM 0.29.0:
128 frozen WikiText validation windows (121,284 scored tokens) for prompt PPL,
and 64 different windows with eight prefixes each for full-vocabulary KL.
No recipe tuning or model promotion occurred; another 89 windows remain reserved.

| Control | Prompt PPL ↓ | Mean forward KL from BF16 ↓ | Top-1 / 512 | Sampled SQNR, dB | Output tok/s, c1 | Output tok/s, c4 |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 7.831635 | reference | 512 | — | 4.65 | 17.55 |
| Corrected RTN | 8.059076 | 0.059035 | 455 | 18.911 | 12.98 | 44.18 |
| Unweighted MSE | 8.034145 | 0.062363 | 461 | 18.958 | 12.96 | 44.03 |
| Corrected norm fallback MSE | 8.040323 | 0.074429 | 456 | 18.956 | 12.81 | 43.38 |
| Local corrected H64 scale-only | **7.954455** | **0.054833** | 461 | 18.821 | 12.77 | 43.49 |
| AMD Quark-AWQ MXFP4 | 7.993200 | 0.060078 | 465 | — | 12.53 | 42.32 |

H64 is 0.485% lower in PPL than AMD (paired 95% interval: 0.151%–0.823% lower),
1.568% above BF16, and 1.298% below RTN. The corrected norm fallback did not
improve unweighted MSE: its PPL interval includes equality and its KL was worse.
H64's KL is 8.73% below AMD on this screen; top-1 agreement favors AMD.
These are scale-only gains on a fixed validation sample, with nominal paired
window intervals. They do not establish universal task superiority.

Decode rates are medians of three runs with 512 input tokens and 128 forced
output tokens, including prefill; c4 is aggregate output throughput. Small
differences among MXFP4 models are observations from this short run. SQNR is
the mean unweighted score of the first 16 rows of each target; AMD was not
remeasured for reconstruction. Downstream task accuracies were not rerun. H64
conversion took 255.46 seconds with 79.8 MiB peak CUDA allocation and 1.44 GiB
process RSS. The full protocol, uncertainty, conversion resources, and raw
reports are in the [dated control experiment](docs/CORRECTED_CONTROLS_2026-10-03.md).

### Fast serving and long-context pilots

A separate development screen used the same unpublished corrected H64,
12 frozen chat/math/code/prose prompts, three timing repetitions and c1:

| Serving profile | Output tok/s | Median TTFT | Median TPOT | Draft acceptance | Exact token parity |
|---|---:|---:|---:|---:|---:|
| MTP off | 13.18 | 106.58 ms | 75.65 ms | — | reference |
| Native MTP, 2 draft tokens | 21.27 | 212.46 ms | 47.33 ms | 73.55% | 9/12 unique prompts |

The paired median decode speedup was 1.60×, reaching 1.90× on code. TTFT was
worse, and three prompts diverged repeatably, so the strict deployment parity
gate failed. This is a measured serving opportunity requiring qualification;
production serving was restored to its prior configuration.

On 12 separate PG-19 validation books, H64 mean next-token KL from BF16 was
0.022531 at 512 tokens, 0.041915 at 2k and 0.066802 at 8k. The 8k-minus-512
interval includes zero: this small screen is inconclusive, with different
token positions at each length. The final WikiText windows remain reserved.
Protocols, exact report hashes, per-domain serving rates, parity diagnostics
and uncertainty are in the [dated pilot record](docs/FAST_PILOTS_2026-10-04.md).

A completed [follow-up](docs/MTP_FOLLOWUP_2026-10-04.md) reproduced the three
MTP divergences: the target scores changed to exact ties, with no non-argmax
selection in 4,608 traced tokens. Eager execution also failed exact parity.
All 32 paired requests on 16 new diagnostic tasks matched; corrected scoring
gave 8/8 Python tasks and 1/8 final-answer arithmetic tasks in both modes.
Code had a 1.87× paired total speedup, while short arithmetic replies were
slower and TTFT increased. On 24 different books with the same targets at
each length, KL was 0.037718/0.037700/0.036664 at 512/2k/8k; the paired
8k-minus-512 interval [−0.009941, +0.007150] includes zero. These sparse
checks do not establish general task quality or qualify production MTP.

### Kolibri-1 experimental release

The selected Kolibri-1 recipe uses softer routed-RMS scale selection for MXFP4
experts and preserves the official FP8 backbone. Its 32 safetensors shards total
**40.45 GiB**, with all **116,303 payloads** verified against the measured trial.
The standalone checkpoint serves through stock vLLM 0.29.0 and
`aleph-alpha-inference==1.0.0` on DGX Spark / GB10, using Marlin experts.

| Model / decode graphs | Prompt PPL | Mean forward KL | Sampled SQNR | Behavior checks | Output tok/s, c1 | Aggregate tok/s, c4 |
|---|---:|---:|---:|---:|---:|---:|
| Official Kolibri-1 FP8 | 24.201922 | reference | — | 6/6 | 43.61 | 113.18 |
| Softer RMS MXFP4 + FP8 backbone | 24.571060 | 0.055140 | not measured | 6/6 | 49.62 | 136.37 |

These are the standalone stock-loader measurements on 48 frozen passages and
26,488 scored tokens. PPL is 1.53% above the reference; the unchanged strict
KL target of **0.030 remains missed**, so this release is experimental.
Throughput uses the median of three runs with 512 input / 128 forced output
tokens; the candidate run overlapped a CPU/network weight upload.

The [release overview](docs/KOLIBRI1_RELEASE_2026-10-04.md) describes the recipe,
source pins, serving command, evidence, and limits. Detailed experiment data and
rejected-trial sources are preserved in a separate archive.

The separate [Kolibri GGUF exporter](docs/KOLIBRI1_GGUF.md) losslessly repacks
these experts and dequantizes the FP8 backbone to BF16, producing **42.30 GiB**.
It needs the pinned Kolibri llama.cpp architecture port. On the same Spark
SM121 and frozen evaluation tokens, its CUDA MMQ activation setting matters:

| GGUF MMQ activation setting | Prompt PPL | Mean forward KL | Behavior checks | Output tok/s, c1 |
|---|---:|---:|---:|---:|
| Default, four-bit | 24.830798 | 0.104662 | 6/6 | 31.40 |
| `GGML_CUDA_MMQ_PREC=q8` | 24.461075 | 0.061844 | 6/6 | 35.42 |

The eight-bit setting is recommended. Its PPL increase is 1.07% over the FP8
reference, but the strict KL target of 0.030 remains missed. Both formats are
experimental. GGUF SQNR and concurrency-four throughput were not measured;
the eight-bit screen overlapped CPU restoration downloads.

## Install

```bash
git clone https://github.com/kirillbilchenko/MxWave.git
cd MxWave
pip install -e ".[dev,calibrate]"
mxwave-calibrate --help
mxwave-quantize --help
mxwave-quantize-experts --help
```

### Experimental lossless GGUF export

MxWave can export a pure Qwen3.5 `compressed-tensors`
`mxfp4-pack-quantized` checkpoint to native GGUF MXFP4 without dequantizing or
requantizing its weights. The exporter preserves every four-bit code and E8M0
scale, applies the Qwen3.5 linear-attention V-head ordering required by ggml,
and delegates tokenizer, metadata, and unquantized tensors to llama.cpp.

The current adapter is validated against llama.cpp tag `b11232`, commit
`6f767fe960c3b97cf37fac4626c86400561ca1e4`. Put that checkout and its
`gguf-py` package on `PYTHONPATH`, then run:

```bash
PYTHONPATH=/path/to/llama.cpp:/path/to/llama.cpp/gguf-py \
  mxwave-export-gguf /path/to/mxwave-checkpoint \
    --outfile /path/to/model.gguf \
    --outtype bf16
```

For a complete Qwen3.5 multimodal package, publish the main GGUF above together
with a separate vision projector. The main GGUF already contains the embedded
MTP tensors used by runtimes such as Ollama:

```bash
# Vision tower/projector paired with that target.
PYTHONPATH=/path/to/llama.cpp:/path/to/llama.cpp/gguf-py \
  mxwave-export-gguf /path/to/mxwave-checkpoint \
    --mmproj --outfile /path/to/mmproj-model.gguf --outtype bf16
```

As an advanced, runtime-specific alternative, llama.cpp can split the embedded
MTP head into a standalone speculative-draft GGUF. This is not needed for
Ollama and duplicates the shared embedding and output tensors:

```bash
# Language target without embedded MTP tensors.
PYTHONPATH=/path/to/llama.cpp:/path/to/llama.cpp/gguf-py \
  mxwave-export-gguf /path/to/mxwave-checkpoint \
    --no-mtp --outfile /path/to/model-target.gguf --outtype bf16

# Standalone MTP draft for runtimes that accept a separate draft GGUF.
PYTHONPATH=/path/to/llama.cpp:/path/to/llama.cpp/gguf-py \
  mxwave-export-gguf /path/to/mxwave-checkpoint \
    --mtp --outfile /path/to/mtp-model.gguf --outtype bf16
```

The main target retains MxWave's strict, exact MXFP4 target-coverage check.
The current MxWave Qwen3.5 policy intentionally keeps both the vision tower and
MTP head in their source floating-point precision, so `--mmproj` and `--mtp`
delegate those selected tensors to llama.cpp. Those modes fail closed if a
future checkpoint contains selected `weight_packed` tensors; they will not
silently omit or dequantize an auxiliary MXFP4 weight. Prefer the default
embedded-MTP main GGUF plus `--mmproj`; use `--no-mtp` plus `--mtp` only when a
runtime explicitly accepts a separate draft model.

This command creates an inference artifact only. Runtime deployment, API
authentication, activation precision, context sizing, and Ollama/vLLM
lifecycle configuration intentionally remain outside MxWave.

## Activation calibration and quantization

```bash
mxwave-calibrate \
    --model-dir /path/to/float-model \
    --corpus /path/to/calibration.jsonl \
    --output /path/to/activation-stats.safetensors \
    --policy qwen3.8-27b-compatible \
    --statistics mean-abs,rms,block-hessian \
    --num-sequences 16 \
    --sequence-length 512 \
    --weight-loading streaming

mxwave-quantize \
    --model-dir /path/to/float-model \
    --output-dir /path/to/output \
    --policy qwen3.8-27b-compatible \
    --activation-stats /path/to/activation-stats.safetensors \
    --calibration-objective block-hessian \
    --mse-clip-depth 4 \
    --tensor-row-chunk-size 1024 \
    --device cuda \
    --verify-sqnr
```

`mean-abs` is the inexpensive diagnostic requested for comparing real inputs
with the LayerNorm-gamma fallback. `rms` is the diagonal approximation to
expected output reconstruction error. `block-hessian` retains correlations
within each 32-channel MXFP4 block and is the strongest available scale
objective. The Hessian affects scale selection only; code assignment remains
nearest-E2M1. The calibration file must cover every selected target exactly;
MxWave never silently mixes real statistics with gamma or unweighted MSE.

Without `--activation-stats`, MSE uses unweighted scale search by default.
The Qwen norm fallback requires explicit `--gamma-proxy`; `--no-gamma-proxy`
remains accepted. The corrected proxy did not improve plain MSE in the fixed
validation comparison, so it is retained as an experimental opt-in.

Hessian damping defaults to an absolute diagonal addition of `1e-6`.
Use `--hessian-damp-mode relative --hessian-damp 0.01` to add 1% of each block's
mean diagonal instead. This option is recorded in the artifact and is not a
newly validated H64 recommendation.

Calibration defaults to bounded-residency `--weight-loading streaming`. It
constructs the model on the meta device, loads the embedding and then one
decoder layer at a time, keeps the evolving hidden states on CPU, and releases
each layer before loading the next. Every checkpoint weight is still read once,
but the full float model is never resident. A clean Qwen3.8-27B `1x512` probe
peaked at 7.46 GiB process RSS and 3.28 GiB CUDA allocation while traversing all
64 layers; the BF16 checkpoint itself is 51.75 GiB. Linux may retain already-read
shard pages as reclaimable filesystem cache, so `docker stats` can temporarily
look larger than the tensor working set. The calibration artifact records both
peak RSS and peak accelerator allocation.

Sequential loading currently has a verified adapter for the Qwen3.5-text
decoder layout used by Qwen3.8-27B. Unsupported layouts fail closed. Use
`--weight-loading resident` explicitly only when a model has no streaming
adapter and enough memory is available.

Runtime-aware optimization uses a separate, model-independent operation graph.
Built-in adapters currently describe the verified Qwen3.5-text layout and the
standard dense Llama layout, including runtime-fused QKV and gate/up groups.
Adapters inspect only configuration and safetensors headers, validate every
required weight, and fail closed when an architecture is unknown or incomplete.

### Experimental routed-expert MoE conversion

`mxwave-quantize-experts` is an opt-in, adapter-driven path for fused MoE expert
banks. The engine consumes a model-independent layout IR, processes direct
expert/row slices under a host-memory cap, and preserves every non-target tensor
exactly. Architecture-specific config checks, tensor mapping, and runtime module
selectors are isolated in adapters. The first verified adapter covers the
Qwen3.5-MoE layout used by `nex-agi/Nex-N2.5-mini`; unknown or incomplete
architectures fail closed.

Inspect the complete plan before writing an artifact:

```bash
mxwave-quantize-experts \
    --model-dir /path/to/float-model \
    --output-dir /path/to/output \
    --method mse \
    --scale-percentile 99.5 \
    --mse-clip-depth 4 \
    --tensor-row-chunk-size 2048 \
    --host-tensor-cap-mib 1024 \
    --source-repository nex-agi/Nex-N2.5-mini \
    --source-revision 87420286149d9cce9bd46cd335ef9bda33c37c1b \
    --dry-run
```

Remove `--dry-run` to emit the checkpoint. If a run is interrupted, rerun the
identical command with `--resume`; MxWave binds the run to SHA-256 hashes of the
source shards and validates every completed output shard before reuse.
`--method rtn` is the reference default. `--method mse` performs fixed,
unweighted weight-reconstruction scale search; it does **not** use route,
activation, or Hessian calibration.

Source weight directories must remain immutable during conversion. Each write
or resume invocation performs one bounded sequential hash pass over all source
weight shards, trading additional I/O for cryptographically safe shard reuse.

The Qwen3.5-MoE adapter quantizes routed experts only. Routers, shared experts,
attention/GDN, embeddings, norms, the output head, vision weights, and all other
non-target tensors remain passthrough. The configured host cap bounds
planned output tensor payload per shard and is checked before loading payloads.
Current emission writes chunks directly to disk. Quantization inputs are also
bounded by `--tensor-chunk-max-elements`; device workspace, allocator, mmap,
page-cache, and filesystem overhead remain additional.

The output uses the standard weight-only `mxfp4-pack-quantized` contract. On DGX
Spark / SM121 it was validated with the official vLLM 0.29 image and both Marlin
backends forced:

```bash
vllm serve /path/to/output \
    --linear-backend marlin \
    --moe-backend marlin
```

The historical Nex-N2.5-mini preflight converted 30,720 logical expert matrices into
61,440 packed/scale tensors across 87 shards while process RSS remained about
1.3–1.9 GiB. Stock vLLM selected `MarlinExperts`, and text and vision smoke
forwards passed. These results establish bounded emission and runtime
compatibility; they are not a claim that RTN or unweighted MSE improves model
quality.

### Experimental precision budgets

`mxwave-precision-budget` builds bounded semantic interventions on top of an
existing MXFP4 checkpoint. It promotes explicitly selected, fusion-safe runtime
groups to channel-wise FP8 from a compatible floating-point donor while retaining
MXFP4 everywhere else. The planner reads checkpoint headers only; composition
streams primary shards and materializes one selected donor matrix at a time.

```bash
mxwave-precision-budget plan \
    --primary-model /path/to/mxfp4-model \
    --dense-donor /path/to/float-model \
    --output /path/to/precision-plan.json

mxwave-precision-budget compose \
    --plan /path/to/precision-plan.json \
    --bucket sequence-output-early-l08-11 \
    --bucket mlp-input-late-l51-54 \
    --output /path/to/mixed-model \
    --quant-device cuda
```

This remains opt-in: a plan defines candidates but does not establish that any
candidate improves the model. Promotion requires frozen reference and baseline
outputs, disjoint selection splits, a final split reserved before recipe selection, and whole-model task
checks. The first Qwen3.8-27B experiment improved prompt PPL from `8.119445` to
`8.111578` while adding 375.6 MiB. On the paired full GSM8K run it scored
`92.0394%` strict versus H64's `91.5845%`; the positive 0.455-point difference
was not statistically conclusive. The candidate is therefore a qualified
optional quality variant, not the new default. See the reusable
[precision-budget runbook](docs/PRECISION_BUDGET_RUNBOOK.md) and the complete
[Qwen experiment record](docs/QWEN3_8_27B_PRECISION_BUDGET.md). Its 96-context
confirmation suffix was disjoint from that experiment's selection prefix but
came from the previously evaluated WikiText test corpus. It was not globally
untouched evaluation data. The release
claims and serving instructions are frozen in the
[mixed-precision model card](docs/QWEN3_8_27B_PRECISION_BUDGET_MODEL_CARD.md).

Target matrices are read from safetensors and quantized in bounded row ranges;
`--tensor-row-chunk-size` controls the device working set. The production writer
predeclares each safetensors layout, writes packed values and scales directly to
their final offsets, and copies passthrough payloads in bounded raw-byte chunks.
It therefore retains neither a complete output tensor nor a complete output
shard in host memory. Activation statistics are opened lazily and one target's
validated statistic is released before the next target is loaded. Each shard is
flushed durably before its atomic rename and recorded in a run-bound SHA-256
ledger; resumed runs verify the complete payload rather than trusting headers.
Both engines hash source-shard contents, config and index files for run identity.
A durable pending-shard journal lets resume complete a crash between shard
rename and ledger update without manually deleting the completed shard.

### Archived DGX Spark experiments

The refinement implementations used for rejected rounding, feedback,
cross-block, local-proxy, layout, and small-recovery experiments are not part
of the production package. Their measurements and exact deployed source
snapshots remain recorded so negative results are not lost. The later scaled
recovery is a successful mechanism but remains local because its KL evidence
was mixed and full GSM8K tied H64.

- [Cross-block reconstruction](docs/QWEN3_8_27B_CROSSBLOCK_PROBE.md)
- [Selective BF16](docs/QWEN3_8_27B_SELECTIVE_BF16.md)
- [Selective FP8](docs/QWEN3_8_27B_SELECTIVE_FP8.md)
- [Static low-rank recovery](docs/QWEN3_8_27B_LOW_RANK_RECOVERY.md)
- [Operator-response selection](docs/OPERATOR_RESPONSE_PROBE_2026-09-18.md)
- [Small recovery-training probe](docs/QAT_RECOVERY_PROBE_2026-09-18.md)
- [Residual counteraction](docs/COUNTERACTION_PROBE_2026-09-19.md)
- [Suffix-JVP selection](docs/SUFFIX_JVP_PROBE_2026-09-19.md)
- [Final bounded layout/confirmation cycle](docs/FINAL_BOUNDED_EXPERIMENTS_2026-09-19.md)
- [Scaled recovery final qualification](docs/QWEN3_8_27B_RECOVERY_FINAL_QUALIFICATION_2026-09-21.md)
- [Native NVFP4 bounded qualification](docs/NVFP4_BOUNDED_QUALIFICATION_2026-09-21.md)
- [Canonical research tracker](docs/MXWAVE_RESEARCH_ROADMAP.md)

These are paired 100-item GSM8K samples with identical prompts and decoding,
not reportable benchmark scores. They are retained because they caught a
misleading weight-objective improvement.

| Artifact | Flexible | Strict | Calibration-weighted SQNR |
|---|---:|---:|---:|
| AMD Quark AWQ MXFP4 | 93% | 93% | not available |
| MxWave historical `abs(weight)` MSE (incorrect norm proxy) | 91% | 90% | 19.04 dB on 272/400 targets |
| MxWave block-Hessian scale selection, 16×512 | 92% | 91% | 19.15 dB |
| MxWave block-Hessian scale selection, 64×512 | 92% | 91% | 19.14 dB |
| MxWave block-local Hessian feedback, 64×512 | 94% | 94% | 19.68 dB |
| MxWave Hessian feedback + 1.125× MSE trust region, 64×512 | 93% | 93% | 19.26 dB |
| MxWave real-RMS scale selection | 90% | 88% | 19.08 dB |
| MxWave + one full Hessian rounding sweep | 86% | 86% | 19.69 dB |

The historical norm-proxy row used `abs(weight)` despite Qwen3.5 RMSNorm
applying `1 + weight`. Its task scores remain measurements of that artifact,
but it is not a valid test of the corrected gamma proxy. The local rounding
experiments used block Hessians and float-prefix inputs; they do not rule out
sequential full-Hessian GPTQ or combinations with channel scaling and rotations.

The rounding sweep lost eight paired items to AMD and gained one on both
extractors (two-sided exact McNemar `p=0.039`). This is direct evidence that
optimizing a local calibration quadratic more aggressively can hurt end-to-end
behavior even while its own reconstruction metric improves.
The real-RMS pass also failed to improve over block-Hessian scale selection
(one RMS-only versus four Hessian-only strict paired wins, exact McNemar
`p=0.375`). Block-Hessian scale-only therefore remains the current MxWave
default. The unconstrained feedback path led that pilot, but its
five/six feedback-only wins versus three scale-only wins are not significant
(`p=0.727` flexible, `p=0.508` strict). Against AMD it had four wins and three
losses on both extractors (`p=1.0`). It remains experimental until a larger,
preferably deterministic evaluation confirms the direction; neither local
reconstruction metrics nor this limited sampled screen are treated as proof.
The 1.125× ordinary-MSE trust region reduced the average unweighted SQNR cost
of feedback from 0.44 dB to 0.10 dB while retaining a 0.12 dB improvement in
the calibration-weighted objective over scale-only. It scored 93%/93%: two
paired wins and two losses versus AMD (`p=1.0`), and three/four wins versus
two losses against scale-only (`p=1.0` flexible, `p=0.688` strict). This is a
safer experimental candidate, but it was not retained.

A later held-out selector used a disjoint `32x512` split to accept feedback only
where both training and selection Hessians improved, subject to the 1.125× MSE
trust region. It improved held-out local SQNR, but on the deterministic
61,408-token likelihood pilot it was 0.146% worse than scale-only (paired 95%
bootstrap interval: 0.020% to 0.271% worse). This rejects feedback as the default
and demonstrates why local reconstruction metrics are not promotion criteria.

## Historical deterministic DGX Spark likelihood screen

Model: `Qwen/Qwen3.8-27B`. Hardware: NVIDIA GB10 / SM121 (DGX Spark). The source
is the pinned Salesforce WikiText-2 raw test parquet at revision
`b08601e04326c79dfdd32d625aee71d232d685c3`, SHA-256
`5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91`.

Rows were joined with two newlines and scored as all 316 non-empty independent
4,096-character windows through vLLM prompt logprobs, one request at a time.
The first token of each window is unscored, leaving 297,199 paired tokens. All
text hashes, token hashes, and token counts matched for every model. MXFP4
artifacts used the same Marlin A16 backend. Intervals are a fixed-seed 100,000
sample paired cluster bootstrap over windows.

This is an API prompt-perplexity comparison, not a literature-compatible
WikiText perplexity number: character windows and context resets differ from
the usual fixed-token sliding-window protocol. The absolute PPL should not be
compared with unrelated model cards; the paired differences below are the
intended result. The hashes, runtime settings, and full-precision values are in
the [machine-readable benchmark summary](benchmarks/qwen3.8-27b-wikitext2-prompt-ppl.json).
The complete pinned commands, configuration, hashes, and replay checks are in the
[Qwen3.8-27B H64 reproducibility record](docs/QWEN3_8_27B_H64_REPRODUCIBILITY.md).

Earlier recipe probes used windows from this test corpus. These intervals
describe paired sampling variation; they do not adjust for adaptive recipe
selection. The corrected controls above use a separate validation split and
include whole-model RTN and unweighted MSE. The reserved final windows remain
unscored; these historical intervals are not evidence from an untouched holdout.

| Artifact | Prompt PPL ↓ | Change vs BF16 | Paired window wins vs BF16 |
|---|---:|---:|---:|
| Qwen3.8-27B BF16 | 7.980864 | reference | — |
| MxWave block-Hessian scale-only, `64x512` calibration | **8.119445** | +1.736% `[+1.333%, +2.110%]` | 65 / 316 |
| AMD Quark-AWQ MXFP4 | 8.187544 | +2.590% `[+2.282%, +2.906%]` | 44 / 316 |

Directly against AMD, MxWave is 0.832% lower in PPL with a paired 95%
interval of 0.531% to 1.227% lower and wins 204/316 windows (two-sided sign test
`p=2.55e-7`). The largest favorable outlier is window 170; removing it still
leaves MxWave 0.684% lower. This supports a real advantage on this deterministic
likelihood screen, not a claim of universal task superiority. The earlier
sampled GSM8K screen was effectively tied (MxWave 92%/91%, AMD 93%/93%).

## Exact next-token distribution screen

The complementary distribution test uses 128 evenly spaced WikiText contexts
of up to 512 tokens and collects all 248,320 next-token log probabilities. Both
MXFP4 candidates are compared against the same BF16 reference distribution.
These contexts are prefixes of a subset of the PPL windows above, so this is
another metric on overlapping data. It scores one next-token position per
context, not 128 full sequence distributions; top-1 agreement is 120 versus
117 of 128 positions.
The complete inputs, per-context values, artifact hashes, and paired bootstrap
are in the
[machine-readable divergence report](benchmarks/qwen3.8-27b-next-token-divergence.json).

| Metric | MxWave H64 | AMD Quark-AWQ MXFP4 |
|---|---:|---:|
| Mean forward KL from BF16 ↓ | **0.049937** | 0.056735 |
| Forward-KL p95 ↓ | **0.175241** | 0.232540 |
| Mean Jensen-Shannon divergence ↓ | **0.010946** | 0.012083 |
| Mean total variation ↓ | **0.078565** | 0.082362 |
| BF16 top-1 agreement ↑ | **93.75%** | 91.41% |

H64's mean forward KL is 0.006798 nats (about 12.0%) lower and it wins 70 of
128 contexts. The paired 10,000-resample interval for `H64 - AMD` is
`[-0.024242, +0.009835]` nats, so this is favorable but not statistically
conclusive evidence. No controlled throughput result is reported; interactive
OpenWebUI observations were intentionally excluded from the quality benchmark.

The publication-ready artifact explanation, runtime caveats, and all three
quality screens are collected in the
[H64 model card](docs/QWEN3_8_27B_H64_MODEL_CARD.md).

## Project structure

```
MxWave/
├── mxwave/
│   ├── calibration.py Activation collectors + safe stats artifact contract
│   ├── calibration_cli.py Real-sequence forward calibration command
│   ├── calibration_stream.py One-decoder-layer-at-a-time calibration runner
│   ├── core.py      MXFP4 constants, MSE-optimal quantize_mxfp4()
│   ├── shard.py     safetensors shard discovery + streaming reads
│   ├── incremental_safetensors.py Direct, bounded output payload writer
│   ├── engine.py    GPU-streaming quantization orchestration
│   ├── format.py    input format detection from config.json (not suffix sniffing)
│   ├── rotate.py    Experimental block Hadamard / orthogonal primitives
│   ├── runtime_ir.py Model-independent runtime operations and fused groups
│   ├── runtime_adapters.py Validated architecture-adapter registry
│   ├── expert_ir.py Model-independent fused-expert bank/slice contract
│   ├── expert_engine.py Bounded adapter-driven expert planner and emitter
│   ├── expert_cli.py Installed `mxwave-quantize-experts` command
│   ├── gguf.py      Lossless MXFP4 repack + pinned llama.cpp adapter
│   ├── gguf_cli.py  Installed `mxwave-export-gguf` bridge command
│   ├── adapters/    Architecture-specific runtime and expert-layout builders
│   ├── mixed_precision.py Streaming MXFP4/FP8 checkpoint composition
│   ├── precision_budget.py Semantic byte-budget candidate planning
│   ├── qualification.py Whole-checkpoint verification + frozen evidence gates
│   ├── qualification_cli.py Installed `mxwave-qualify` command
│   ├── verify.py    SQNR, config-coverage verification (verification-first)
│   ├── output.py    compressed-tensors quantization_config assembly + coverage
│   └── cli.py       CLI entry point (wired to the engine)
├── scripts/
│   ├── evaluate_api_perplexity.py Paired OpenAI-API prompt-PPL evaluator
│   └── evaluate_next_token_kl.py Exact next-token distribution evaluator
├── tests/
├── pyproject.toml
└── README.md
```

## Roadmap

- [x] **Streaming engine** — tensor-row streaming, on-device quantize, output assembly
- [x] **Module-keyed calibration** — real mean-absolute, RMS, and block-Hessian
      input statistics from a bounded forward pass
- [x] **Weight-streamed calibration** — meta-device scaffold, CPU hidden states,
      and one resident decoder layer for verified architectures
- [ ] **Calibration-aware default** — choose and run a validated corpus by default
- [ ] **Rotation folding** — emit standard `compressed-tensors` `transform_config`
- [ ] **Auto per-layer precision** — Hessian-trace-driven MXFP4/FP8/BF16 assignment
- [x] **Measured precision-budget experiment** — adapter-driven, fusion-safe FP8
      allocation passed within-experiment selection/confirmation and full-corpus PPL gates;
      retained as an opt-in feature pending broader task validation
- [ ] **Sequential layer reconstruction** — calibrate each block on inputs from the
      already-quantized prefix and compensate errors across full input dimensions
- [x] **Held-out adaptive selection experiment** — evaluated with a disjoint
      selection split, rejected after deterministic PPL regressed, and archived
- [x] **Unified qualification contract** — `mxwave-qualify` verifies the whole
      checkpoint, runs bounded evidence hooks, imports versioned PPL/KL/serving/runtime
      reports, and fails incomplete when required runtime evidence is absent
- [x] **Deterministic likelihood screen** — paired BF16/MxWave/AMD comparison
      with exact input hashes and clustered uncertainty
- [x] **Layer-output-aware selection experiment** — operator response,
      counteraction, suffix-JVP, and exact-layout selectors were evaluated and
      rejected under their frozen Qwen scopes
- [x] **Sensitivity-guided mixed precision on Qwen** — a 12-tensor FP8 allocation
      passed the frozen Qwen gate; automatic cross-model allocation remains unproven

See the [qualification runbook](docs/QUALIFICATION_RUNBOOK.md) for the installed
command, runtime-evidence contract, and separation between MxWave decisions and
platform-specific vLLM lifecycle/telemetry.

## License

Apache-2.0. This project is an independent clean-room implementation based on
the public OCP MX specification and published quantization research. See
`CONTRIBUTING.md` for the provenance policy.
