# MxWave

The frozen eight-prompt drafter development rerun completes **20 October,
19:23 UTC**: all **144 requests** have matching output tokens across
original/adapted/original loads, and **8/8 reload timing controls pass**.
The small step-24 pilot is effectively speed-neutral on the same CT target:

| Development domain | Original drafter tok/s | Step-24 pilot tok/s | Difference |
| --- | ---: | ---: | ---: |
| Code, four prompts | 38.770 | 38.813 | +0.111% |
| Chat, two prompts | 29.932 | 29.914 | −0.060% |
| Explanation, two prompts | 31.523 | 31.419 | −0.329% |

Rates are geometric means of per-prompt medians; one candidate load is not final
speed qualification. Original production is restored with 500,000 context.
See the [full result and TTFT table](docs/GLM53_HANDOFF.md).
The separately added
[DeepInfer confidence experiment](docs/GLM53_DEEPINFER_TRANSFER.md) has an opt-in
training head and CPU scheduling reference; serving calibration and speed
remain unqualified.

**Expert reduction fix, 20 October 2026:** bounded routing padding makes shared
rows bit-identical in the retained 440-versus-448 dispatch, through both expert
projections and with stock/packed scales. The subsequent one-prompt full-model
check passes: 32 output tokens and all 32 decision logits on both ranks are
bit-identical across original/adapted drafters, measured KL 0. All eight request
controls and four capture-repeat comparisons pass. Original production is restored.

| Replay | Shared-row equality | Exact graph repeats | Timing stability controls |
|---|---|---:|---:|
| V2, two padding policies | Pass, both ranks | 960 | 18/32 pass |
| V3, default occupancy, longer warm-up | Pass, both ranks | 640 | 14/16 pass |

The two V3 timing failures remain failures at the unchanged 2% drift gate.
In the full-model check both drafters now propose identical inputs in aligned
shared-history frames, so differing proposals still need broader development
coverage. Both use nine draft rounds; no acceptance or serving-speed gain is
established. C8/C16 coverage and full 4,000-prompt training remain outstanding.
All 250 full-model evidence payloads verify on Mac/Spark2. See the current
[handoff](docs/GLM53_HANDOFF.md); the sealed test remains closed.

<!-- DRAFTER_TRAINING_PREFLIGHT -->
**Earlier CT drafter checkpoint, 20 October 2026:** the bounded context-projection pilot
selects step 24: validation draft prefix **code 4.25 → 4.375**, **chat 2.75 → 2.875**,
explanation unchanged. The independent review reproduces the result; native
export changes only FC and preserves the other 115 tensors/config exactly.
Native reload passes, but seven of eight development responses differ between
arms. All 144 requests finish; original reload outputs/counters repeat exactly
and timing controls pass. Descriptive throughput is code +5.91%, chat +1.61%,
explanation −4.57%, on changed responses. The unchanged-output gate fails; the
candidate is not promoted. Original production is restored at 15:14 UTC with
500,000 context; all 83 native evidence payloads verify on Mac and Spark2.
[Evidence, limits and current status](docs/GLM53_DRAFTER_CONTEXT_PILOT.md).

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

### GLM-5.3-Flash lossless runtime pilot

See [current state and next steps](docs/GLM53_CURRENT_STATE.md) for verified
findings, completed measurements and unresolved qualification.

**MXFP4/DFlash follow-up, 20 October 2026: complete and independently verified.**
The same-checkpoint comparison retains 48 timed requests and 9.36 GiB TP2
allocation savings (about 4.68 GiB/GPU):

| C1 workload | CT off tok/s | CT on tok/s | Change | Repeat control |
|---|---:|---:|---:|---|
| Diagnostic, 64 output tokens | 46.201 | 46.566 | +0.79% | Pass |
| Diagnostic, 256 output tokens | 64.439 | 65.657 | +1.89% | Pass |
| Code, 256 output tokens | 38.155 | 39.086 | +2.44% | **Fail: stock 2.85% spread** |
| Explanation, 256 output tokens | 23.304 | 23.950 | +2.77% | Pass |

TTFT adds 17.6–35.1 ms. All 24 paired token sequences and speculation counters
match; this is not a KL result. Four traces show zero decode rebuilds, while
compressed prefill still reconstructs 68 matrices/rank in 36.47/37.14 ms.
All 80 evidence payloads and 1,863 recomputed fields verify independently.
See [full results and limits](docs/GLM53_MXFP4_SERVING_V3.md).
The [reversible KDA MXFP8 quality screen](docs/GLM53_KDA_MXFP8_QUALITY.md) also
completes all five passes and restores original production at 05:51:18 UTC.
No checkpoint conversion or promotion has occurred.

| KDA weight-rounding quality screen, same BF16 execution | Result |
|---|---|
| Sampled WikiText PPL, 32,768 tokens/pass | 4.1150974 → 4.0857491 (−0.7132%); exact within-arm repeats |
| Full mean KL, three declared pairs | 1.23e-3–1.59e-3; **fails 1e-5** |
| Reference-repeat control | **Fails on the 4,096-token context**; other 208 positions repeat exactly |
| Cross-arm mean KL on those 208 positions | 1.22338e-3 in every pair |
| KDA weight SQNR, ranks 0/1 | 42.352 / 42.362 dB |
| Short tasks | Both functions pass; shared baseline format failures limit accuracy interpretation |
| Actual MXFP8 kernel speed in this quality screen | Not measured here; separate serving experiment below |

All 1,688 payloads verify independently; 5,414 recomputed fields agree on
Spark/Mac, maximum difference 7.11e-15.
This demonstrates changed distributions with slightly better sampled PPL,
not broad quality equivalence or an MXFP8 speed gain.

The [actual MXFP8 serving comparison](docs/GLM53_KDA_MXFP8_SPEED.md) completes
all nine loads and 108 timing groups on 20 October. All 115 archived payloads
verify; 7,441 independently recomputed fields match exactly on Spark/Mac.
Six final profiles are reviewed: all 232 target graph ranges are attributed.
Eight-row KDA GEMM sums fall from 22.7–22.8 to 13.0–13.1 ms. Both native
MXFP8 traces have zero weight rebuilds; CT still rebuilds in width-12 graphs.

| Completed serving comparison | Current CT lossless BF16 KDA | CT MXFP8 KDA | ModelOpt, matched settings | MXFP8 vs ModelOpt |
| --- | ---: | ---: | ---: | ---: |
| C1 diagnostic, 256 output tokens | 66.05 tok/s | 71.57 tok/s | 69.67 tok/s | +2.74%; repeats pass |
| C1 code | 39.24 tok/s | 44.86 tok/s | 49.09 tok/s | **−8.61%; repeats pass** |
| C1 explanation | 23.91 tok/s | 35.85 tok/s | 32.37 tok/s | +10.72%; ModelOpt repeat fails |
| Offered C8 | 64.04 tok/s | 89.14 tok/s | 79.30 tok/s | +12.41%; repeats fail |
| Offered C16 | 72.28 tok/s | 95.13 tok/s | 83.95 tok/s | +13.32%; candidate repeat fails |
| C1 code first token | 459.8 ms | 399.3 ms | 442.7 ms | −43.3 ms |

All modes admit at most six resident requests, so C8/C16 include queueing.
Generated outputs and speculative acceptance differ. These measurements do not
isolate kernel gains or establish a universal win over fully tuned production.
CT MXFP8 saves another 1.105/1.086 GiB per GPU versus current CT lossless BF16
KDA. Its allocation is 7.447/7.430 GiB per GPU below matched-settings ModelOpt,
including checkpoint-format differences. The earlier **1e-5 KL gate remains failed**.

The [full native MXFP8 GSM8K comparison](docs/GLM53_KDA_MXFP8_GSM8K.md) is
complete and independently verified: 1,319 questions per fresh CT arm, with
speculation off and the same frozen prompts/scoring.

| Completed GSM8K metric | Current CT lossless BF16 KDA | CT native MXFP8 KDA |
| --- | ---: | ---: |
| Numeric correct | 1,248/1,319 (94.617%) | **1,260/1,319 (95.527%)** |
| Strict-format correct | 1,226/1,319 (92.949%) | **1,245/1,319 (94.390%)** |
| API errors / truncations | 0 / 0 | 0 / 0 |
| C4 workload wall time, speculation off | 37m44.4s | **33m18.7s** |
| Workload output throughput | 41.90 tok/s | **48.18 tok/s** |

The candidate gains 22 answers and loses 10: **+12 correct / +0.910 percentage
points**, exact McNemar p=0.05010246. This is a higher observed task score,
not proof of general superiority; the **1e-5 KL gate remains failed**.
Workload time falls 11.73% and output TPS rises 14.98%, but these are single-pass
natural-response timings, separate from the interleaved serving test above.
The historical original CT score was 1,246/1,319 under an older runtime; no
production NVFP4 GSM8K comparison is implied.

All **2,690 payloads**, **2,638 regraded answers**, and **2,933 numerical fields**
verify on Spark/Mac with zero numerical difference. See the
[verification receipt](benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/kda-mxfp8-gsm8k-v2-final-independent-verification.json).
Original production is restored at **20 October 09:35:00 UTC**, healthy with
500,000 context. No candidate is promoted. The original CPU-only filename
collision remains retained. The [drafter audit](docs/GLM53_DRAFTER_COMPARISON.md)
also completes its read-only inventory; matching draft weight files leave shared
heads and target-conditioned histories for the next controlled diagnostic.
The [head diagnostic](docs/GLM53_DRAFTER_COMPARISON.md) subsequently completes:

| Paired packed / exact dense head on identical C1 draft states | Result |
| --- | ---: |
| Observed graph replays per rank | 129 (125 verified rounds + 4 post-cap lookaheads) |
| Proposed tokens changed | 0 / 903 |
| Retained raw local logit elements with different bits | 0 / 20,599,040 |
| Request output/counter controls | All six pass |

All 580 payloads and the corrected coverage review verify independently on
Spark/Mac. The original literal round-count failure remains retained. Original
production is restored at **11:22:59 UTC**, with 500,000 context. These results
clear the head for the tested states; the subsequent matched-history diagnostic
is reported below. No new speed or quality claim.
The paired CT/ModelOpt drafter follow-up also completes. All **88 recorded
drafter weight tensors per rank are identical**. At the same five-token code
prefix CT accepts six drafts and ModelOpt seven: only the seventh proposal
changes, while both targets choose the same next token. The correct token was
already in CT's candidate list. This supports adapting the drafter to CT's changed
features; it does not demonstrate a fix or account for the whole code-speed gap.
All 126 payloads and 1,922 numerical fields independently verify (maximum
absolute difference 2.24e-13). Original production is restored at **11:53:12 UTC**,
with 500,000 context; no training or experiment remains running.
[Details and limits](docs/GLM53_DRAFTER_COMPARISON.md).
The [handoff](docs/GLM53_HANDOFF.md) records the complete state.

**Service update, 20 October 2026, 04:46 UTC:** original production is restored
on port 8000 with **500,000 context**, with both original container/configuration
identities verified and both CT trial containers stopped. No new GPU experiment
has started. MXFP4/DFlash profiling found faster same-checkpoint decode but
prefill rebuild costs and a slower CT backbone; NVFP4 scale compression instead
preserves the production backbone. Its 3.16 GiB/GPU memory saving is verified,
but serving-speed and full-model numerical qualification remain unresolved.
See the [handoff and pending checks](docs/GLM53_HANDOFF.md). Earlier dated live-service
statements below are historical.

**MoE ordering diagnostic, 19 October 2026: captured prefill fixed.** The first
unstable operation is token ordering within experts. Canonical ordering preserves
routing membership and makes all three captured prefills exact on both ranks.

| Same original-production checkpoint, stable MoE ordering | Result | Scope |
|---|---|---|
| C1 first-token self-repeat KL | 0 | Before/after controls; captured prefill logits exact |
| Later C1 self-repeat KL | Up to 5.2904e-4 — fail | Four output tokens; unchanged 1e-5 gate |
| Eager 256-row expert-layer overhead | +0.38–1.77% | Six cells; all native controls pass 2% |
| C8 self-repeat | Unqualified | Batch layouts differ |

All 913 alignment payloads and six timing aggregations independently verify.
The combined MoE/sparse-MLA follow-up also verifies all 895 payloads and finds
first-token KL 0 at C1/C8. Later decode remains unqualified: ordinary C1 mean KL
9.11e-6–2.25e-5, matched C8 mean KL 2.47e-4. Recorded C1 inputs/embeddings match,
but graph outputs already differ at the first auxiliary boundary after layer 5.
C8 admits six requests then two under 6 GiB KV. These are repeatability
diagnostics, not compressed-versus-stock or serving-speed qualification. The
prior service is restored. See [verified results and next trace](docs/GLM53_NVFP4_MOE.md).
Earlier dated observations below are retained as history.

**Original-production repeat diagnostic, 19 October 2026, 20:47 UTC: first difference found.**
The same ModelOpt NVFP4/MXFP8 stack is tested without compression, at a fixed
256-token C1 prefill. The focused pair has identical expert inputs, actual
router logits and shared-expert output:

| Original-production TP2 diagnostic | First differing boundary | Elements changed there / rank | Raw first-token KL |
|---|---|---:|---:|
| Stock, one-token focused repeat | Layer 3 MoE expert output | 4 / 1,048,576 | 0.00537795 — fail |

The internal routing/GEMM/reduction cause is not isolated yet; another pair first
changes at layer 4. Raw logits agree with API probabilities, and ordinary
one-token controls also fail. No compression loss or speed verdict is changed.
All 4,290 payloads independently verify; the exact prior 8,192-context CT service
is restored. See [the bounded result and next test](docs/GLM53_NVFP4_REPEAT.md).

**Original-checkpoint NVFP4 serving test complete, 19 October 2026, 20:04 UTC.** Lossless
expert-scale storage saves **3.16 GiB/GPU of actual allocation**, preserving the
original production NVFP4/MXFP8 checkpoint and settings. No decode scale rebuild
appears in six retained traces. The observed rates below do **not** establish
a speed gain: all four stock-repeat controls fail the unchanged 2% gate.

| Workload | Stock A tok/s | Compressed tok/s | Stock B tok/s | Stock time drift (limit 2%) |
|---|---:|---:|---:|---:|
| 256-token C1 | 27.63 | 38.77 | 35.92 | 23.06% — fail |
| 256-token C8 | 68.40 | 73.62 | 69.97 | 2.24% — fail |
| 256-token C16 | 61.26 | 73.93 | 67.81 | 9.66% — fail |
| 2048-token C1 | 20.89 | 25.05 | 23.35 | 10.50% — fail |

C1/C8 stock and compressed self-repeat KL fail 1e-5, so full-model zero-loss
quality remains unqualified despite exact stored-scale recovery and 22 exact
isolated matched-routing checks. Cold-C1 TTFT is 631.4 / 481.8 / 480.7 ms
for A / compressed / B. PPL, broad task accuracy, SQNR and long-context quality
are unmeasured. All 154 evidence payloads independently verify; the previous
8,192-context CT service is restored. See [full result and evidence](docs/GLM53_NVFP4_MODEL_TEST.md).
Earlier dated component and CT results below remain historical.

**Production NVFP4 component, 19 October 19:03 UTC: complete.** The lossless
scale consumer preserves the original ModelOpt NVFP4 checkpoint. A full
288-expert layer on SM121 gives the following eight-row results against the
installed stock operator (32 interleaved trials, stable repeat controls):

| TP partition | Installed layer time, µs | Compressed layer time, µs | Time reduction | Output checks |
|---|---:|---:|---:|---|
| Rank 0 | 1,773.917 | 1,740.155 | 1.903% | Bitwise exact |
| Rank 1 | 1,755.222 | 1,701.288 | 3.073% | Bitwise exact |

Full-checkpoint encoded-scale storage, including metadata/raw fallback, saves
**3.250086 / 3.252740 GiB per rank**. This is not measured whole-model VRAM.
The broader 1/8/16/64/256-row component matrix is exact; accepted layer times are
2.06–4.48% lower than the matched rebuilt raw control. Real layer-23 weights use
synthetic activations/routing, and both partitions run sequentially on one GPU.
PPL, task accuracy, SQNR, full-model KL, TTFT and serving throughput are unmeasured
for this port. No serving adapter or deployment switch has occurred; all jobs
are terminal and the existing CT/DFlash service is unchanged. The
[result and evidence](docs/GLM53_NVFP4_COMPONENT.md) retain all failed attempts,
217 independently verified payloads and the next full-model integration step.
Earlier dated entries below remain historical.

**Production ModelOpt CPU audit, 19 October 17:59 UTC: complete.**
All 36,288 NVFP4 expert-scale tensors were scanned in the actual lossless8
checkpoint; this is a storage screen for preserving production's checkpoint.

| CPU result | Value |
|---|---:|
| Source expert scales, total / even TP2 rank | 17.71875 / 8.859375 GiB |
| Global / weighted per-tensor entropy | 3.773 / 3.359 bits/byte |
| Estimated palette storage saving per rank | 3.25–3.30 GiB |
| Reduction in expert weight-plus-scale bytes | about 4.1% |
| Exact sampled block-codec roundtrips | 1,008 / 1,008 |

GPU allocation, throughput, TTFT, KL, PPL, task accuracy and SQNR are unmeasured
for this NVFP4 route. Post-Marlin scale bytes differ from source FP8 bytes and
need an exact consumer prototype. Production's specialized head also retains
dense BF16 weights, so the vocabulary port requires more than loader changes.
Serving is unchanged. See [audit, evidence and next test](docs/GLM53_MODELOPT_SCALE_AUDIT.md).

**DFlash C1 isolation, 19 October 17:27 UTC: complete and independently verified.**
Seven trials per length/arm, 64 output tokens, TP2 SM121. The same CT checkpoint
with compression off/on isolates the runtime; ModelOpt provides replacement context.

| Prompt tokens / C1 | Existing ModelOpt tok/s | CT off tok/s | CT on tok/s | Compression change | CT TTFT off → on, ms |
|---|---:|---:|---:|---:|---:|
| 256 | 48.163 | 46.118 | 45.972 | -0.32% | 464.7 → 517.0 |
| 2,048 | 27.984 | 29.023 | 29.001 | -0.08% | 1,390.9 → 1,423.6 |

Compression reduces normalized decode time by 5.34% / 3.46%, but its prefill
cost cancels the gain for these 64-token answers. Isolated CUDA allocation saving
is **9.355840 GiB TP2**. All timed outputs match across all arm pairs; speculative
KL, PPL, task accuracy and SQNR remain unmeasured. Profiles prove no C1 decode
rebuilds and 68 prefill rebuilds/rank. Keeping only KDA dense could retain an
estimated 6.85–7.23 GiB saving; this selective option has not been measured.
The live candidate is restored with 8,192 context. Installed vLLM is
`0.1.dev20051+g487ecf187`. See [complete isolation result](docs/GLM53_DFLASH_C1.md).

**Serving replacement trial, 19 October: verified; candidate live with DFlash.**
This compares the current ModelOpt mixed NVFP4 checkpoint and our compressed CT
MXFP4 candidate under matching serving settings, plus unchanged production as
context. It is a checkpoint replacement comparison, not isolated compression.

| Prompt tokens / client concurrency | Production tok/s | Matched current tok/s | Candidate tok/s | Candidate vs matched |
|---|---:|---:|---:|---:|
| 256 / C1 | 51.770 | 48.399 | 45.754 | -5.46% |
| 256 / C8 | 101.841 | 74.383 | 75.704 | +1.78% |
| 256 / C16 | 107.402 | 74.885 | 75.512 | +0.84% |
| 2,048 / C1 | 30.459 | 27.683 | 28.898 | +4.39% |
| 2,048 / C8 | 38.926 | 32.151 | 34.643 | +7.75% |
| 2,048 / C16 | 38.938 | 32.440 | 35.180 | +8.45% |

**12.690092 GiB less post-startup CUDA allocation across TP2** under equal KV
budgets; all 150 timed output sequences match both references. Short C1 regresses,
short C8/C16 are near parity, and long prompts improve against the matched
checkpoint. Tuned production remains faster in all six cells. Full-stack KL,
PPL, task accuracy and SQNR are unmeasured. Repeated prose favors speculation;
these numbers do not predict the user's harness. The live trial uses 8,192
context and 4 GiB KV/rank, with substantial C8/C16 queueing.
See [latency tables, qualification limits, evidence and rollback](docs/GLM53_SERVING_TRIAL.md).

**Verified full-model result, 18 October 21:10 UTC:** startup warming works in
the measured workload. All **1,612 payloads / 207 sources / 12 worker traces**
independently verify on Spark and Mac. All **46 numerical comparisons** pass at
the unchanged KL <=1e-5 / absolute mean NLL <=1e-4 limits; all **300 paired timed
output sequences** match. Native mHC and matched quality grouping remain in both
arms. This is finite numerical qualification, not universal bitwise equivalence.

| Workload, 64 output tokens | Stock tok/s | Compressed tok/s | Change | Median TTFT, stock -> compressed |
|---|---:|---:|---:|---:|
| C1 / 256 prompt tokens | 14.784 | 16.748 | +13.28% | 0.384 -> 0.422 s |
| C1 / 2,048 prompt tokens | 12.518 | 13.999 | +11.82% | 1.183 -> 1.205 s |
| C8 / 256 prompt tokens | 60.757 | 62.564 | +2.97% | 1.169 -> 1.192 s |
| C8 / 2,048 prompt tokens | 31.488 | 31.530 | +0.13% | 6.586 -> 6.662 s |
| C16 / 256 prompt tokens | 80.870 | 81.108 | +0.29% | 1.838 -> 2.280 s |
| C16 / 2,048 prompt tokens | 36.628 | 36.477 | -0.41% | 11.395 -> 11.594 s |

Six trials/case/arm, 256/2,048 prompt tokens and 64 output tokens, two fresh loads
in fixed stock-then-compressed order. **The original overall speed gate still
fails**: C8/C16 miss the 5% throughput target, and short-C1/C16 exceed the TTFT
allowance. All trials, admission differences and historical failures are retained.
CUDA allocation is **9.367162 GiB lower across TP2** at every measured stage.
CPU process RSS is **158.78 MiB higher before quality / 166.83 MiB higher after
ordinary timing / 173.72 MiB higher after profiles** across TP2. These overlapping
Spark counters must not be added. No CPU-memory improvement is demonstrated.

The 34-variant startup warmup leaves dispatch caches unchanged through the full
workload; no recognized JIT event overlaps timing. All 576 before/after raw
output rows per arm are bitwise equal. The earlier 2.14 s C8 gap does not recur.
Current PPL, task accuracy and SQNR are unmeasured. Originals are restored healthy;
no configuration is promoted. See the [verified handoff](docs/GLM53_HANDOFF.md).

The subsequent two-GPU prefill diagnostic independently verifies 560 payloads /
210 sources and all 22 operations per GPU. Reconstruction approaches the tested
copy payload rate, suggesting modest decoder-only headroom. All unstable controls
remain; no new serving-speed claim is made. See the [counter findings and limits](docs/GLM53_SPARK_ROOFLINE.md#completed-kda-memory-diagnostic-18-october).

The subsequent dense-output replacement screen independently verifies 270
payloads / 213 sources and 221,184 timing samples. It passes numerical checks
but fails warm-cache C1 (56-68 us added per projection), despite faster prefill.
All 12 cost failures and six unstable controls remain. It is not integrated;
the warmed exact16 model above remains the candidate. See the
[dense-output findings](benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/dense-output-v1-final-findings.json).

**Native expert follow-up, 19 October: complete and independently verified.**
Same GLM-5.3-Flash CT MXFP4 source weights on SM121; complete routed layer 23,
TP rank 1, all 288 experts, actual captured inputs. The comparator is the current
aligned-scale Marlin path. Native includes FP8 activation quantization and
lossless scale reconstruction in each timed invocation.

| Rows | Current layer ms | Native layer ms | Native extra time | Native/current relative L2 |
|---|---:|---:|---:|---:|
| 1 | 0.246062 | 1.035762 | +320.93% | 4.084% |
| 8 | 1.558483 | 2.451117 | +57.28% | 4.174% |
| 16 | 2.249204 | 3.189554 | +41.81% | 4.146% |
| 256 | 6.001999 | 7.000208 | +16.63% | 4.134% |

Sixteen interleaved trials per arm/shape, 40 graph replays each; all reference
timing controls pass. The C8/C16 requirement of at least 15% time reduction
fails. Current replay and all repeats are bitwise exact; native repeats differ.
These are component results; full-model native KL, throughput, peak memory,
PPL, task accuracy and SQNR remain unmeasured. No native integration or promotion.
Scale rebuild costs about 0.7 ms/layer, and native expert kernels also take longer.
494 payloads independently verify on Spark/Mac; service restoration is complete.
See [traces, numerical limits and retained evidence](docs/GLM53_MOE_BACKENDS.md).

**4,096-token scheduler-budget follow-up, 19 October: independently verified.**
At fixed 10 ms admission, the larger cap **fails the prospective policy gate**.
Both caps share 4,096 allocation capacity; all original mean/p95 TTFT, ITL and
memory guards are preserved. No setting is promoted.

| Workload / cap | Stock tok/s | Compressed tok/s | Stock mean TTFT (s) | Compressed mean TTFT (s) |
|---|---:|---:|---:|---:|
| c1-short / 2,048 | 14.895 | 16.820 | 0.391 | 0.413 |
| c1-short / 4,096 | 14.937 | 16.826 | 0.383 | 0.412 |
| c16-short / 2,048 | 80.996 | 81.956 | 1.855 | 1.925 |
| c16-short / 4,096 | 84.228 | 85.851 | 2.032 | 2.056 |

Numerics: 42/42 matched comparisons pass KL <=1e-5 / abs mean
NLL <=1e-4; 52 cross-budget diagnostics remain separate. CUDA saving is
9.358188 GiB across TP2. PPL, task accuracy and SQNR are not remeasured.
See [full latency, memory, profile and quality evidence](docs/GLM53_PREFILL_BUDGET_4096.md).
Earlier experiments below keep their original metrics and decisions.

**Bounded scheduler-budget follow-up, 19 October: complete and independently verified.**
At fixed 10 ms admission, caps 2,048 / 2,080 are interleaved within each arm;
both use 2,080 worker capacity, unchanged warmed exact16 kernels and no BF16 cache.

| Short workload / token cap | Stock tok/s | Compressed tok/s | Throughput change | Headline TTFT, stock -> compressed | Mean TTFT, stock -> compressed |
|---|---:|---:|---:|---:|---:|
| c1-short / 2,048 | 14.863 | 16.736 | +12.605% | 0.392 -> 0.423 s | 0.390 -> 0.424 s |
| c1-short / 2,080 | 14.877 | 16.738 | +12.505% | 0.384 -> 0.417 s | 0.386 -> 0.422 s |
| c16-short / 2,048 | 80.943 | 84.297 | +4.144% | 1.669 -> 1.819 s | 1.800 -> 1.884 s |
| c16-short / 2,080 | 80.788 | 84.821 | +4.992% | 1.384 -> 1.430 s | 1.842 -> 1.870 s |

Twelve C16 / six C1 trials per cap and arm, 256 prompt / 64 output tokens.
Headline TTFT is the median of trial medians. The scheduling gate passes, but
compressed C16 mean improvement is only 13.7 ms (descriptive 95% interval
-46.3 to +14.4 ms); stock mean worsens by 42.0 ms. Both compressed profiles
still have three prefill passes / 204 reconstructions per rank. The remaining
latency cost is not solved. Memory saving remains **9.367162 GiB across TP2**;
extra worker capacity costs about 8.03 MiB total. All **42 matched same-budget
numerical comparisons pass**, with C1 KL 5.804e-6 and C16/2,080 KL zero on the
retained fixture. The 52 cross-budget diagnostics include differences above
1e-5 in both arms; there is no universal zero-loss or batch-invariance claim.
All 1,188 timed output pairs match. PPL, task accuracy and SQNR are not remeasured.
Spark/Mac verify 1,601 payloads / 216 sources / 16 traces. Original combined
compression speed/TTFT gates remain false; all previous failures are retained.
Original GLM is restored healthy. No configuration is promoted. See the
[full result, traces and evidence](docs/GLM53_PREFILL_BUDGET.md).

The preceding admission-only experiment below remains a separate historical result.

**Final C16 admission comparison, 19 October: independently verified.**
The 10 ms policy passes its prospective gate. All 94 numerical comparisons pass; all
1188 timed output pairs match: True.
The qualified warmed exact16 kernels are unchanged; input-FP32 is excluded.

| Short workload / window | Stock tok/s | Compressed tok/s | Throughput change | Median TTFT, stock -> compressed | TTFT gap |
|---|---:|---:|---:|---:|---:|
| c1-short / 3 ms | 14.856 | 16.745 | +12.71% | 0.376 -> 0.404 s | +27.6 ms |
| c16-short / 3 ms | 80.417 | 81.671 | +1.56% | 2.025 -> 2.133 s | +107.3 ms |
| c1-short / 10 ms | 14.835 | 16.691 | +12.50% | 0.380 -> 0.412 s | +32.7 ms |
| c16-short / 10 ms | 80.657 | 82.344 | +2.09% | 1.383 -> 1.694 s | +311.5 ms |

Twelve C16 and six C1 trials/window/arm, 256 prompt / 64 output tokens. CUDA
saving stays 9.367162 GiB across TP2. The actual C16 gap at
10 ms is +311.5 ms; this does not qualify every concurrency or prompt length.
The frozen median-of-trial-medians criterion remains unmet. Supplementary all-request
statistics give a +57.4 ms mean / +65.9 ms pooled median gap over all
192 requests per arm; these do not replace the frozen metric or its decision.
Historical compression speed gates remain retained. PPL, task accuracy and SQNR
are not remeasured. Both hosts verify 1603 payloads / 211 sources /
16 traces; original GLM is restored healthy. No configuration is promoted.
This completes compression experiments per user scope. See the
[full result, traces and limits](docs/GLM53_C16_ADMISSION.md).

**Input-only FP32 follow-up, 19 October: completed, C16 numerical gate fails.**
Spark/Mac independently verify 1,634 payloads / 229 source bindings / 12 traces.
C1/C8 KL pass at 5.390e-6 / 3.791e-6; C16 is **1.602e-5 > 1e-5** in all ten
cross-arm repeat/profile comparisons. Both arms repeat exactly and all 300
timed output pairs match. The preceding warmed exact16 result above remains the
numerically qualified fallback. No configuration is promoted.

| Workload, 64 output tokens | Stock tok/s | Input-FP32 tok/s | Change | Median TTFT, stock -> input-FP32 | Added TTFT |
|---|---:|---:|---:|---:|---:|
| C1 / 256 prompt tokens | 14.856 | 17.052 | +14.78% | 0.391 -> 0.432 s | +41.0 ms |
| C1 / 2,048 prompt tokens | 12.602 | 14.145 | +12.24% | 1.157 -> 1.222 s | +64.6 ms |
| C8 / 256 prompt tokens | 61.338 | 63.257 | +3.13% | 1.163 -> 1.201 s | +37.7 ms |
| C8 / 2,048 prompt tokens | 31.862 | 31.394 | -1.47% | 6.448 -> 6.765 s | +316.9 ms |
| C16 / 256 prompt tokens | 80.643 | 81.322 | +0.84% | 2.079 -> 2.211 s | +132.2 ms |
| C16 / 2,048 prompt tokens | 37.158 | 36.405 | -2.03% | 11.153 -> 11.672 s | +519.0 ms |

Six trials/case/arm, fixed stock-then-candidate order; original throughput/TTFT
gates still fail. CUDA saving stays **9.367162 GiB across TP2**; CPU RSS rises
242-251 MiB before profiles. KDA decode improves modestly to ~20-21 ms, while
expert computation dominates concurrency. The smaller C16 TTFT gap is
admission-confounded and is not transferred to the qualified fallback. PPL,
task accuracy and SQNR are unmeasured. GLM is restored healthy; verification
and retention are complete, with all original failures preserved. See the
[complete result and evidence](docs/GLM53_FP32_INPUT_MODEL.md).

The earlier FP32 integration fixture passed 120 retained-output and 66 fallback
comparisons, compiled graphs and six two-stream stress fixtures while reusing
the fixed pool. Component timing showed ~7-8% input improvement but retained two
repeat-control failures. Those component passes did not establish the full-model
qualification, which now fails above. Further compression-kernel optimization
stops per user scope. The final admission experiment is complete; see the
[verified result](docs/GLM53_C16_ADMISSION.md). MTP has not started.

The separate FP32 split-K precision prototype reduces M4/M8 kernel error about
22-26 times versus BF16 partial sums. All 576 comparisons pass the 0.001 relative
bound, but 216 M4/M8 exact-word checks fail and remain failed. Both hosts verify
2,227 payloads / 217 sources; speed and full-model KL are not measured for this
prototype. It is not integrated into the candidate. See the
[precision findings](benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/fp32-split-v1-final-findings.json).

**Historical matched C1/C8 result, 18 October:** compression passes all 33 numerical
comparisons with matched C1/C8 grouping and native mHC in both arms. Decode KL
is **5.804e-6 at C1 / 1.194e-6 at C8**, below the unchanged 1e-5 limit; all 102
timed paired output sequences match. All **1,168 payloads / 194 source bindings /
eight worker traces** independently verify on Spark/Mac.

| Workload, 64 output tokens | Stock tok/s | Compressed tok/s | Change |
|---|---:|---:|---:|
| C1 / 256 prompt tokens | 14.943 | 16.794 | +12.38% |
| C8 / 256 | 61.302 | 62.771 | +2.40% |
| C8 / 2,048 | 31.775 | 31.581 | -0.61% |

Six retained trials/case/arm, two fresh loads. CUDA allocation falls
**9.367 GiB across TP2** (175.908 -> 166.540 GiB); initial host RSS rises slightly.
C1 first-token latency rises **7.91%**; the overall speed gate fails, and C8
remains below the 5% gain target. Quality applies to the tested contexts/grouping;
inference is not universally bitwise exact. PPL, SQNR and task accuracy are not
remeasured in this run. Originals are restored healthy; no configuration is
promoted. See [quality, protocol and verified evidence](docs/GLM53_C8_TRACE.md).
MTP remains deferred.

**Exact16 follow-up, 18 October 19:08 UTC:** the full-model input/output route
passes all 46 matched C1/C8/C16 numerical comparisons and all 192 timed paired
output sequences match. All **1,581 payloads / 200 sources / eight traces**
independently verify on Spark/Mac. Six retained trials/case/arm:

| C16 workload, 64 output tokens | Stock tok/s | Compressed tok/s | Change | Median TTFT, stock -> compressed |
|---|---:|---:|---:|---:|
| 256 prompt tokens | 81.142 | 81.529 | +0.48% | 1.623 -> 2.211 s |
| 2,048 prompt tokens | 37.035 | 36.516 | -1.40% | 11.265 -> 11.601 s |

CUDA allocation saving remains **9.367 GiB/TP2**. Candidate physical-16 decode
replays graphs with 68 compressed KDA calls and zero reconstruction. The speed
gates still fail; short-C16 first-token latency rises 36.22% and needs analysis.
PPL, SQNR and task accuracy were not remeasured. Original GLM is restored;
no configuration is promoted. See the [verified handoff](docs/GLM53_HANDOFF.md).

**18 October startup-stall component:** tracing attributes reproduced 1.5–2.0 s
GPU gaps in both arms to first-use compilation in the shared deterministic
expert-order wrapper. Startup warming all 34 unchanged variants passes on both
SM121 GPUs:

| Component measure | Rank 0 | Rank 1 |
|---|---:|---:|
| One-time 34-variant warmup | 33.701 s | 34.498 s |
| Maximum subsequent call | 85.79 µs | 95.89 µs |
| RSS reclaimed by one-time heap trim | 12.80 MiB | 12.91 MiB |

Exact integer outputs/guards pass, reuse creates no additional compiler entries,
and scratch CUDA allocation returns to baseline. All 699 payloads / 203 sources
independently verify on Spark/Mac. This scratch-only component does not load the
model or measure serving speed, model CPU savings, PPL, SQNR or task accuracy.
The full-model follow-up is now complete and independently verified above; earlier
speed failures remain retained.
See the [handoff](docs/GLM53_HANDOFF.md) for current operational status.

**Historical matched comparison, 17 October:** the current candidate saves
**9.367 GiB / 5.33%** (175.866 → 166.499 GiB across TP2). Same imported
checkpoint, graph captures [1,4], eager C8, fixed KV and no drafter:

| Workload | Reference tok/s | Candidate tok/s | Change |
|---|---:|---:|---:|
| C1 / 256 prompt tokens | 14.994 | 16.848 | +12.36% |
| C1 / 2,048 | 12.642 | 13.961 | +10.43% |
| C4 / 256 | 40.372 | 41.099 | +1.80% |
| C4 / 2,048 | 26.263 | 26.312 | +0.19% |
| C8 / 256 | 57.022 | 58.071 | +1.84% |
| C8 / 2,048 | 31.189 | 31.167 | −0.07% |

Five retained trials/case, exactly 64 output tokens. TTFT is 2.3–5.9% higher.
Single-request throughput improves; concurrent serving remains near parity.
All 373 evidence payloads, 260 requests / 16,640 tokens and 672 source groups
verify independently on Spark/Mac; all 130 paired output sequences match.
Numerical qualification remains unresolved. PPL, task accuracy and SQNR are
not remeasured in this speed run; the separate full GSM8K result is below.
Both-rank C4 profiles show about 8% shorter decode forwards and 6.6–6.7%
slower prefill. Original GLM is restored healthy at 20:22:23 UTC. Read the
[paired result, ranges and next steps](docs/GLM53_PAIRED_SPEED.md).

**Component follow-up:** the direct-output decoder variants pass exact bytes
but fail their speed gates. The exponent-wrap mask fix now passes GPU tests
covering every BF16 bit pattern; all 136 actual KDA shards, both compressed
heads and 154,880 embedding rows have safe bases. Evidence independently
verifies on Spark/Mac; originals restored healthy at 21:43:57 UTC. No serving
library replacement or new model speed result. See
[reconstruction evidence](docs/GLM53_PREFILL_RECONSTRUCTION.md) and
[MTP, small-prefill routing and FP8 priorities](docs/GLM53_SPEED_OPTIONS.md).

Two subsequent [routing sweeps](docs/GLM53_PREFILL_ROUTING.md) retain 51,200
events with passing component numerical checks. The bracketed test identifies
24 rows as a shared component lead; control variance prevents a broad threshold
claim, and large prefills regress on the fused kernel. The candidate limit stays
eight. A [native MTP CPU audit](docs/GLM53_MTP_PILOT.md) repairs 870 mismatches
across all 872 draft selectors; all 1,756 source headers and native vocabulary
sharing pass. The newer runner also needs isolated CT policy; its corrected
retry loads the model, then fails an observer that assumed 64 rope rows instead
of this checkpoint's zero. Local validation now derives geometry from config.
No MTP speed result; exact original service restored at 18 October 05:50:54 UTC.

The actual-zero-rope CPU follow-up now preserves both complete source projection
matrices through native loading and the bridge. All fourteen payloads verify;
it checks two projections, with eighteen total comparisons required in the next
live observer. The full-pilot validator still needs updating before another MTP
GPU attempt. There is no measured MTP acceptance or speed yet.

Beyond MTP, the [prefill admission intervention](docs/GLM53_PREFILL_ADMISSION.md)
is complete and independently verified. On the same compressed mixed-MXFP4
target and two SM121 Sparks, fixed `[1,4]` graphs and no drafter, all twelve
short-C4 trials per window give:

| Idle collection window | Output TPS | Change | Median TTFT | Reconstructions/rank in separate profile |
|---|---:|---:|---:|---:|
| Zero | 40.873 | reference | 1.125 s | 136 |
| 3 ms | 43.631 | +6.75% | 0.778 s | 68 |
| 10 ms | 43.580 | +6.62% | 0.782 s | 68 |

Both frozen performance/mechanism gates pass; every control stays within its
throughput/latency guards. All 599 payloads / 108 groups / 450 requests /
28,800 timed tokens and six worker traces verify on Spark/Mac. All 300 paired
sequence comparisons match. Each profiled forward visits all 68 weights once;
device kernel counts agree. All three policies still fail repeat KL1e-5,
including zero wait; no configuration is promoted. PPL/task accuracy/SQNR
were not remeasured. This reduces prefill work for finite 64-output request
groups; sustained decode-only or continuous active traffic was not tested.
Exact original service restored healthy at 18 October **07:49:50 UTC**.

The [CUDA graph coverage test](docs/GLM53_GRAPH_EXPANSION.md) is complete and
independently verified. On the same compressed target, imported mixed MXFP4
format and two SM121 Sparks, six alternating trials per policy/case give:

| Case | Captures [1,4], TPS | Captures [1,2,4,8], TPS | Change |
|---|---:|---:|---:|
| C2 / P256 | 24.180 | 26.266 | +8.62% |
| C2 / P2048 | 18.166 | 19.795 | +8.97% |
| C8 / P256 | 57.764 | 59.710 | +3.37% |
| C8 / P2048 | 30.917 | 31.452 | +1.73% |

C2 already used padded C4 graphs; exact C2 reduces padding. Both-rank traces
confirm C8 changes eager decode to graph replay. The bundle misses its frozen
performance gate: short-C8 gain is below 5%, and short-C4 median TTFT regresses
34.8% alongside changed admission geometry. All 150 paired sequences match,
but both policies fail KL repeats. PPL/task accuracy/SQNR were not remeasured;
no release qualification or configuration promotion. All 466 evidence payloads
verify on Spark/Mac, with CPU probability roundoff explicitly bounded; exact
original service restored healthy at 07:00:20 UTC. C16 remains separate.

The [matched C8/C16 compression comparison](docs/GLM53_HIGH_CONCURRENCY.md)
is complete after repairing startup scratch ownership. Both targets use
captures [1,2,4,8,16], max sequences16, the same imported mixed MXFP4 checkpoint,
two SM121 Sparks and fixed 4 GiB KV/rank. Six trials per case and window use
finite 64-output cohorts. The stock control is reused from the first attempt
with an explicit time gap; these are not freshly interleaved model loads.

| Workload | Stock TPS, 0 / 3 ms | Compressed TPS, 0 / 3 ms | Change, 0 / 3 ms |
|---|---:|---:|---:|
| C1 / P256 | 14.876 / 14.824 | 16.705 / 16.689 | +12.30% / +12.58% |
| C4 / P256 | 40.553 / 42.854 | 40.798 / 43.461 | +0.60% / +1.42% |
| C8 / P256 | 58.094 / 61.084 | 59.566 / 62.400 | +2.53% / +2.15% |
| C8 / P2048 | 31.463 / 31.510 | 31.359 / 31.293 | −0.33% / −0.69% |
| C16 / P256 | 79.963 / 79.798 | 68.935 / 68.930 | −13.79% / −13.62% |
| C16 / P2048 | 36.745 / 36.705 | 33.940 / 33.948 | −7.64% / −7.51% |

Memory saving is 9.367 GiB across TP2. Both frozen performance gates fail,
including C1 TTFT regressions of 8.1–10.0%. All 636 paired timed sequences
match, but all24 full-vocabulary comparisons fail KL; four also fail NLL.
PPL/task accuracy/SQNR were not remeasured. All1,277 payloads,144 groups /
1,272 requests /81,408 tokens and sixteen worker traces verify on Spark/Mac.
Eight retained C16 pure-decode ranges across TP2 each reconstruct68 KDA
matrices, taking34–37 ms; the current fused limit is eight rows. These kernel
sums are not request latency. The startup repair preserves the two-slot
206,569,472-byte/rank pool and unchanged arithmetic. No configuration is promoted.
The [compact retained review](benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/high-concurrency-retry-v2-review.json)
keeps every trial, numerical failure, source binding and trace attribution.

The narrower [output-only sixteen-row follow-up](docs/GLM53_OUTPUT_ROW16.md)
retains the same checkpoint/format and recovers part of C16 throughput:

| Workload | Output-only TPS, 0 / 3 ms | Change versus stock, 0 / 3 ms |
|---|---:|---:|
| C1 / P256 | 16.959 / 16.809 | +14.00% / +13.39% |
| C4 / P256 | 40.229 / 43.857 | −0.80% / +2.34% |
| C8 / P256 | 60.115 / 63.108 | +3.48% / +3.31% |
| C8 / P2048 | 31.427 / 31.344 | −0.11% / −0.53% |
| C16 / P256 | 71.248 / 71.657 | −10.90% / −10.20% |
| C16 / P2048 | 34.455 / 34.391 | −6.23% / −6.30% |

Both performance gates still fail. All72 timing groups /636 requests /40,704
tokens independently verify on Spark/Mac. Disk exhaustion then interrupts
numerical collection: five groups finish, all17 available comparisons fail,
and candidate profiles/source-after are missing. Exact original service is
restored. The1,532-payload failed-run archive is verified; validation/profile
continuation reuses these timings with an explicit model-load gap. Stock and
compressed-parent controls also retain their time gaps. No new PPL/task/SQNR
result or release qualification is claimed. See the
[partial review](benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/output-row16-v1-partial-review.json).

The subsequent validation-only load completes all8 quality groups and4 profile
sessions. All1,293 payloads/180 bindings/16 traces independently verify on
Spark/Mac. Eight retained C16 graph ranges reconstruct34 matrices instead of68;
all24 KL comparisons and8 NLL comparisons fail. The original timing source-after
gap remains. That checkpoint restored the original service; the user has since
resumed experiments. The [completed review](benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/output16-validation-v1-review.json)
and [handoff](docs/GLM53_HANDOFF.md) retain the final state and next steps.


**18 October 15:43 UTC update:** V8 qualifies the exact16 input component
with zero arithmetic error: 435–487 µs fused versus 1.34–1.38 ms reconstruct+GEMM.
The stable sparse-compaction adapter gives exact stock C1 self-repeat.
The [C8 trace investigation](docs/GLM53_C8_TRACE.md) independently verifies
1,025 payloads / 190 bindings / twelve worker traces on Spark/Mac:

| C8 stock repeat condition | Mean decode KL | Gate |
|---|---:|---|
| Ordinary requests, unprofiled | 3.22e−5 / 5.37e−5 | Fail |
| Ordinary requests, profiled | 4.60e−5 / 7.01e−5 | Fail |
| Matched cohort, unprofiled | 0 / 0 | Pass; bitwise exact |
| Matched cohort, profiled | 0 / 0 | Pass; bitwise exact |

This establishes repeatability for matched C8 grouping. A general serving fix
and a fresh stock/compressed speed/quality comparison remain unqualified.
Exact original service identities and health 200 are restored. See the
[current handoff](docs/GLM53_HANDOFF.md) for the next component check.

**Earlier combined pilot:**

On two DGX Sparks (SM121), the same imported mixed MXFP4 checkpoint was tested
with exact expert-scale/KDA/head/embedding storage adapters. Expert four-bit
codes are unchanged; KV allocation is fixed and DFlash/MTP are disabled.

| Runtime | CUDA allocation, both nodes | Tok/s, c1/256 | Tok/s, c1/2048 | Tok/s, c4/256 |
|---|---:|---:|---:|---:|
| Stock imported MXFP4 | 175.87 GiB | 15.32 | 12.88 | 39.87 |
| Combined lossless storage | 168.72 GiB | 16.85 | 13.86 | 38.82 |
| Combined, lighter MoE dispatcher | 168.72 GiB | 16.76 | 13.94 | 38.89 |

Throughput is the median of five trials including prefill, against a saved
sequential control. All 30 timed outputs and 64 quality-check generated ids
match. Actual decode KL is 0.0000408 (original) / 0.0000425 (dispatcher)
versus 0.0000245 for stock repeating
itself; strict quality is unqualified because stock exceeds the predeclared
stability floor. Standard PPL, task accuracy and source-quantization SQNR were
not measured here. The [results and trace review](docs/GLM53_LOSSLESS_RUNTIME_RESULTS.md)
record the 7.15 GiB saving, four-request overhead and runtime requirements.
The dispatcher reduced host submission time but did not meaningfully improve
whole-model throughput; it remains opt-in.

A separate 17 October [U8 combined quality evaluation](docs/GLM53_BEST_CANDIDATE_QUALITY.md)
measures **9.37 GiB / 5.33%** total live CUDA saving over stock, with the same
imported checkpoint and no new throughput sweep:

| Runtime | CUDA allocation, TP2 | Sampled WikiText PPL, first / repeat | GSM8K sample |
|---|---:|---:|---:|
| Stock imported MXFP4 | 175.87 GiB | 4.30444 / 4.30203 | 22/24 |
| Combined, U8 shared-stage scales | 166.50 GiB | 4.29993 / 4.30106 | 22/24 |

Each PPL run scores 32,768 tokens in frozen test windows; this is not full-corpus
standard PPL. Both arms pass the same 28/32 completed strict practical cases,
with no reference pass lost. Two named tool cases are unmeasured because the
launch omits the flag that activates its configured parser. Storage checks
pass before/after inference, but **strict numerical quality remains unqualified**:
reference and candidate repeats exceed the unchanged KL/NLL stability floors.
The first visible reference divergence is around layer 4's MLP during prefill;
eager mode also varies. The linked report records evidence and the targeted
next diagnostic. These observations do not establish a general accuracy gain.

A later matched [stable-dispatch run](docs/GLM53_STABLE_DISPATCH_EXPERIMENT.md)
passes the unchanged graph/eager numerical gates. Both arms use the same
canonical expert-order patch; this reference is not an unmodified vLLM launch.

| Runtime, canonical dispatch | CUDA allocation, TP2 | PPL, first / repeat | Strict tasks | Tok/s, c1/256 | Tok/s, c1/2048 | Tok/s, c4/256 |
|---|---:|---:|---:|---:|---:|---:|
| Reference imported MXFP4 | 175.87 GiB | 4.30502 / 4.30502 | 30/34 | 14.99 | 12.50 | 38.77 |
| Combined, U8 shared-stage scales | 166.50 GiB | 4.30502 / 4.30502 | 30/34 | 16.77 | 13.93 | 39.02 |

This retains the **9.37 GiB / 5.33% saving**, improves single-request throughput
**11.4–11.9%**, and has near-parity concurrent throughput. TTFT increases
**2.3–7.1%**. All four 32,768-token held-out score sets are byte-identical;
graph candidate/reference KL is 7.22e-6 versus reference-repeat 7.38e-6.
Both real tool calls and both functional code tests pass. **Overall strict
qualification still fails**: the candidate loses two required output formats
while gaining two math passes. Math scores are 21/24 reference and 22/24
candidate; full autoregressive logits are not bit-identical. Five-trial speed
samples retain all outliers; only batch one has CUDA graphs. These are runtime
compression results on the imported quantized checkpoint, not an original-BF16
quantization/SQNR comparison. The linked report identifies BF16 prefill decode
as an introduced cost and records all 959 verified archive payloads.

Five subsequent [fixed task rounds](docs/GLM53_PRACTICAL_REPEAT_DIAGNOSTIC.md)
show 108/120 repeated math passes, 10/10 functional code checks and 10/10
tool calls in each arm. Strict totals are 152/170 versus 150/170, with the
aggregate gap in bare-code formatting. The same four cases vary in both arms;
these repetitions do not amend the earlier failed strict qualification.

A later [controlled-batch graph diagnostic](docs/GLM53_CONTROLLED_BATCH_EXPERIMENT.md)
compares eager and batch-four graphs on **one resident compressed v9 model**:

| Same compressed runtime, TP2 | CUDA allocation | C4 tok/s, two order blocks | TTFT, same blocks |
|---|---:|---:|---:|
| Eager original operation | 166.50 GiB | 38.96 / 37.57 | 1.134 / 1.093 s |
| Original operation, graphs [1,4] | 166.50 GiB | 41.03 / 40.66 | 1.117 / 1.114 s |

Graphs improve aggregate throughput **5.32% / 8.22%**, with TTFT
**-1.49% / +1.92%**. All twenty timing groups remain. In the sixteen-context,
four-token controlled diagnostic, eager/graph KL is **9.25e-6**, with self-repeat
KL **7.77e-6 / 7.29e-6**, passing unchanged limits and matching generated IDs.
Actual graph launches appear on both ranks. A separate direct-wrapper variant
fails KL and is rejected; the combined all-modes gate remains false. PPL/tasks
are not rerun here, and historical strict qualification stays failed. These
are graph-versus-eager gains on compressed weights, not an additional measured
compression-versus-stock gain. All 711 archive payloads verify.

The later [full GSM8K comparison](docs/GLM53_GSM8K_FULL_BENCHMARK.md) measures
all **1,319 test questions per arm**, zero-shot, low reasoning effort, C4,
graphs [1,4] and no drafter, on the same imported MXFP4 checkpoint:

| Runtime, canonical dispatch | CUDA allocation, TP2 | Numeric exact match | Strict answer format | Workload output tok/s | Workload time |
|---|---:|---:|---:|---:|---:|
| Reference imported MXFP4 | 175.87 GiB | 1,246/1,319 (94.47%) | 1,228/1,319 (93.10%) | 41.77 | 37m41s |
| Combined, U8 shared-stage scales | 166.50 GiB | 1,244/1,319 (94.31%) | 1,230/1,319 (93.25%) | 42.08 | 37m37s |

This confirms **9.37 GiB / 5.33% less allocation** with roughly equal workload
speed: +0.74% output tok/s, but only +0.16% questions/minute, as completion
lengths differ. The candidate loses 22 numeric passes and gains 20; paired
exact McNemar p=0.8776 does not establish equivalence. There are zero API
errors/truncations. All 2,638 raw answers, 672 source-scale groups and 2,960
archived payloads independently verify. Both original GLM containers are
restored healthy. No new PPL/KL/SQNR or TTFT is measured here; previous strict
qualification failures stay unchanged.

A subsequent two-Spark proxy-pinning component also
failed its follow-up gate: GPU-spaced forward latency changed within about
0.4%; a CPU-sleep-spaced secondary case improved 2.0–4.4%. These are synthetic
communication measurements, not updated model TPS. Both original GLM ranks
were restored and independently checked healthy. See the
[pinning component review](benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/roce-pinning-v2-review.json).

A candidate-only [Nsight diagnostic](docs/GLM53_NSYS_MODEL_DIAGNOSTIC.md) collected
both worker stacks and CUDA timelines without reloading stock. It identifies
21–22 ms of fused KDA kernel time plus a 2.39 ms head per M4 step, and a shared
attention-metadata synchronization. These include useful computation and queued
GPU waits; they are not removable-overhead estimates. The subsequent
[same-weight BF16 component test](docs/GLM53_BF16_KERNEL_DIAGNOSTIC.md) measured:

| M4 component, warm eager CUDA events | Stock BF16 | Compressed | Paired time ratio |
|---|---:|---:|---:|
| KDA input | 0.452 ms | 0.459 ms | 1.008 |
| KDA output | 0.133 ms | 0.150 ms | 1.134 |
| Vocabulary head | 2.813 ms | 2.438 ms | 0.868 |

Eight alternating-order trials per comparison; these are component times,
not new model TPS. A separate output graph check changes 12.5% slower warm
replay to 4.5% faster after a 128 MiB cache-eviction workload, with exact
output bytes. NCU identifies output grid size and input shared-memory
residency as tuning leads. Higher split-K fails the existing numerical
gates and remains disabled. GLM is restored; earlier model quality status
and adapter defaults are unchanged.

The compact N8 staging follow-up passed bit-exact component checks but
regressed input latency and was rejected. A read-only rANS screen of 27 GLM
expert matrices measured 3.915 bits per four-bit code: an illustrative parallel
layout exhausted that small saving in metadata. U8 scales compressed much
better. A complete 42-layer byte audit now measures a further **2.244 GiB**
of scale-payload saving over our existing codec across both TP2 ranks.
The fused GPU consumer reproduces existing-codec output bits in repeated
component tests, but concentrated prefill is slower and fails the latency
gate. A shared-stage follow-up passes all 56 exact-output cases and reduces
the affected M128 regression from 13.5–21.6% to 5.3–7.6%, with unchanged
compressed storage. It still fails the 2% gate; these are component results.
Two subsequent exact nibble-staging variants also fail: choosing the layout
before launch reduces concentrated M128 regression from 26.8–28.9% to
8.2–8.7%, preserving bytes but leaving U8 staging as the better result.
The [palette diagnostic](docs/GLM53_SCALE_PALETTE_DIAGNOSTIC.md)
separates this storage result from full-model allocation and quality. A
matched full-model follow-up measures another **2.220 GiB / 1.316%** live
allocation saving over the combined adapter, with identical final text in
all five answered practical tasks and matching short-window generated IDs.
Strict numerical qualification still fails: even the reference repeat
exceeds its KL floor. Tool calling was rejected by the launch configuration,
and sustained serving remains untested. These
are [component results and next gates](docs/GLM53_BF16_KERNEL_DIAGNOSTIC.md#paper-follow-up-and-actual-mxfp4-compressibility),
not updated model throughput or memory figures.

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

The subsequent [CPU drafter preflight](docs/GLM53_DRAFTER_PREFLIGHT.md) freezes
success thresholds, verifies a byte-identical FP8/BF16/FP8 checkpoint round trip,
strictly loads all 81 dense parameter tensors into the pinned SpecForge DFlash2
class, and prepares 4,000/200/200 train/validation/sealed-test prompts.
Production stays running. No optimizer, native parity or new speed result is
claimed; the earlier failed pilot below remains failed.

The [bounded CT drafter-calibration pilot](docs/GLM53_SELECTOR_CALIBRATION.md)
completes but **fails its held-out gate**. This is an offline projection-only fit
with fixed captured states; it is not serving speed or target-model task accuracy.

| Mean consecutive matching draft tokens / 7, averaged by prompt | Original projection | Selected projection |
| --- | ---: | ---: |
| Validation, eight prompts | 2.904 | 2.931 |
| Separate test, eight prompts | 2.394 | 2.383 |
| Test code prompts | 2.409 | 2.404 |
| Test explanation prompts | 2.348 | 2.320 |

All 3,682 capture payloads verify; NumPy independently reproduces 108 result fields.
Original production is restored with 500,000 context. The patch is not applied;
drafter recovery, production tuning/capacity, stable C8/C16 results and long-context
qualification remain open deployment gates. Work is committed and backed up on spark2.
