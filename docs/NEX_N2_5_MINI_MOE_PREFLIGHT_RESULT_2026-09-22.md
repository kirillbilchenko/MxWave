# Nex-N2.5-mini expert-MXFP4 preflight result — 2026-09-22

## Decision

The original RTN experiment succeeded as an engineering proof and failed as a
model-release candidate. A later, bounded unweighted-MSE scale-search follow-up
passed its paired-perplexity screen but failed the pre-registered exact
teacher-distribution gate. It is also stopped and should not be published.

MxWave converted the routed experts of Nex-N2.5-mini to standard
`mxfp4-pack-quantized` weights without materializing a fused expert bank, passed
the complete structural contract, and ran text and vision requests through
stock official vLLM 0.29 with `MarlinExperts`. This establishes a bounded
Qwen3.5-MoE adapter, emitter, verifier, and runtime path.

The raw round-to-nearest (RTN) artifact should not be published. It was 4.21%
smaller than the public NVFP4 control, below the owner's approximate 5% value
threshold, and has no demonstrated quality or speed advantage. On the frozen
no-thinking screen, MxWave passed 45/69 scored cases, BF16 passed 46/69, and
NVFP4 passed 47/69; all pairwise exact McNemar tests were unresolved. A one-run
throughput advantage of 3.61% over NVFP4 is only a diagnostic and is not a
defensible performance claim.

The admitted follow-up changed only legal per-block MXFP4 exponent selection:
it searched a fixed unweighted-MSE candidate set while retaining the same
22,902,614,752 tensor bytes, standard checkpoint format, and Marlin W4A16
runtime. On 64 paired WikiText-2 chunks / 61,408 scored tokens, it reached PPL
8.880923 versus 8.910400 for RTN, 9.009808 for public NVFP4, and 8.873219 for
BF16. That PPL gain was real on the frozen screen. However, on 64 separately
frozen terminal next-token distributions, mean BF16-forward KL was 0.034157 for
MSE versus 0.031049 for RTN: a 10.01% regression. Both fixed halves reversed,
so the candidate failed the directional gate even though the paired interval
crossed zero and teacher top-1 agreement was unchanged at 58/64.

The initial artifact was deliberately an RTN preflight. Neither it nor the
unweighted-MSE follow-up applied MxWave's
activation/Hessian calibration to routed experts. The result therefore rejects
publication of the raw-RTN artifact for lack of demonstrated added value. The
follow-up establishes a reproducible ground-truth likelihood improvement but
also demonstrates that lower local weight MSE and lower corpus PPL did not
preserve the full teacher distribution better. The candidate is rejected before
broader task, long-context, or throughput qualification. The bounded expert
adapter, emitter, verifier, and stock-runtime work remain valuable.

## Frozen inputs and scope

- BF16 source: `nex-agi/Nex-N2.5-mini`
- Source revision: `87420286149d9cce9bd46cd335ef9bda33c37c1b`
- Public control: `primitive-ai/Nex-N2.5-mini-NVFP4`
- Control revision: `ceaa4ce6d942c3d5456322d183b535a0c89a5fa6`
- Research branch: `research/nex-n2-5-moe-preflight`
- Runtime: official `vllm/vllm-openai:v0.29.0`
- Runtime digest:
  `sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1`
- Hardware: DGX Spark GB10 / SM121
- Spark record root:
  `/home/kirya/local-spark/experiments/nex-n2.5-moe-preflight-20260922`

The public control is also a weight round-to-nearest build with no calibration;
its card explicitly describes that method. Its representation differs from the
MxWave candidate: NVFP4 uses 16-value groups and a per-module global scale with
dynamic W4A4 activations, while MXFP4 uses 32-value blocks with E8M0
power-of-two scales and ran through the Marlin W4A16 path. Calibration therefore
does not explain the observed answer differences between the two RTN artifacts.

Only the 256 routed experts in each of 40 language layers were converted. The
adapter unfolded the source's 40 fused gate/up banks and 40 fused down banks
into the logical per-expert tensors expected by compressed-tensors/vLLM.
Routers, shared experts, attention/GDN, embeddings, norms, the head, and vision
weights remained passthrough tensors. The source config declares MTP, but the
pinned source index contains no separate MTP tensor payload; this experiment
does not claim that absent MTP weights were retained.

## Conversion and structural result

| Measurement | Result |
|---|---:|
| Conversion wall time | about 7 minutes 40 seconds |
| Source fused expert banks | 80 |
| Logical expert matrices | 30,720 |
| Packed/scale tensors | 61,440 |
| Passthrough tensors | 946 |
| Total output tensors | 62,386 |
| Output shards | 87 |
| MxWave tensor data | 22,902,614,752 bytes |
| Public NVFP4 tensor data | 23,909,493,472 bytes |
| MxWave difference | -1,006,878,720 bytes (-4.21%) |

The output was structurally qualified against the emitted index, config,
manifest, and per-shard integrity ledger. Recorded identities were:

- checkpoint: `eadbf27940485f0c9200e3189a8db03dee0a5380f274a80d182a63ca44abdcd5`
- config: `8ea02e392011595c2394a26862e51966608abab24fa2b750ce798b4ce67f6351`
- index: `69d4da6c13645e868454a7e22537e3f2ceef8883cf63817d67e2239145ad293c`
- manifest: `91a1a1c5ead3ba7a42fa0800aaa559b960010650bb8bd2010203f1c30fd2fb6c`

The converter processed direct expert/row slices rather than loading a fused
bank or whole model. During the full run, process RSS stayed around 1.3--1.9
GiB. In the focused one-bank probe, peak CUDA allocation was 26,640,384 bytes
and cgroup memory peaked at 2,386,784,256 bytes. Docker's larger apparent memory
growth during conversion was primarily filesystem page cache and must not be
reported as live tensor residency.

These observations support the bounded-memory design for this model. They are
not a universal memory upper bound: serializer, allocator, mmap, and filesystem
cache overhead remain outside the explicit 1 GiB materialized-output-payload
cap.

## Stock-runtime result

The emitted checkpoint loaded with both Marlin backends forced. The runtime log
selected `MarlinExperts` for the MXFP4 routed experts and loaded all 87 shards.

| Measurement | Result |
|---|---:|
| Model load memory | 21.47 GiB |
| Model load time | 140.746 seconds |
| Text smoke | exact response `MARLIN_OK` |
| Vision smoke | 64-by-64 red image answered `Red` |

This proves loader, tensor-layout, kernel-selection, and basic forward
compatibility. It does not by itself establish model quality.

## Frozen no-thinking screen

All three candidates received the same 82 frozen cases. Sixty-nine cases had
strict deterministic scoring; eight code cases and five summary cases were
retained as unscored parity observations. All candidates completed every
request with zero request errors.

| Category | MxWave expert RTN | BF16 | Public NVFP4 |
|---|---:|---:|---:|
| Arithmetic | 5/20 | 7/20 | 8/20 |
| Logic / ordering | 6/15 | 5/15 | 5/15 |
| Structured output | 15/15 | 15/15 | 15/15 |
| Short retrieval | 10/10 | 10/10 | 10/10 |
| 8K retrieval | 9/9 | 9/9 | 9/9 |
| **Strict total** | **45/69 (65.22%)** | **46/69 (66.67%)** | **47/69 (68.12%)** |
| Wall time | 47.881 seconds | 68.876 seconds | 47.832 seconds |
| Request errors | 0 | 0 | 0 |

Eight answers changed deterministically. The MxWave artifact lost
`arithmetic-04`, `arithmetic-05`, `arithmetic-07`, `arithmetic-12`, and
`ordering-07`, while gaining `arithmetic-08`, `ordering-06`, and
`ordering-10`. The net result is two fewer passes, or -2.90 percentage points,
relative to NVFP4. Relative to BF16, MxWave was lower by one case / 1.45
percentage points; NVFP4 was higher by one case / 1.45 points.

The paired outcomes do not establish a difference:

- BF16 to MxWave: three losses and two wins among five discordant cases; exact
  McNemar `p=1.0`;
- BF16 to NVFP4: three losses and four wins among seven discordant cases; exact
  McNemar `p=1.0`; and
- MxWave to NVFP4: five losses and three wins among eight discordant cases;
  exact McNemar `p=0.7266`.

This screen is intentionally small and is not a statistically precise estimate
of general capability. The three candidates are indistinguishable on it. The
arithmetic counts are useful diagnostics, while the perfect structured and
long-context retrieval results show that the artifact is operational rather
than generally broken. The BF16 wall-time difference is also not a controlled
throughput benchmark and must not be promoted as a speed claim.

## Matched single-run throughput diagnostic

The same 512-completion-token request was run once against each candidate, with
elapsed time including TTFT.

| Candidate | Elapsed | Output throughput |
|---|---:|---:|
| MxWave expert RTN | 13.6642 s | 37.4700 token/s |
| Public NVFP4 | 14.1576 s | 36.1644 token/s |
| Difference | -0.4934 s | +3.61% |

This was one run, without a repeated warm/cold protocol, confidence interval,
concurrency sweep, or separated prefill/decode timing. It is inconclusive and
must not be advertised as a speed advantage. Even if repeated, a 3.61% gain
would not by itself compensate for the missed size and quality gates.

## Bounded unweighted-MSE scale-search follow-up

Before building route-aware calibration, one cheap control tested whether the
existing MxWave scale search alone could improve the expert banks. It used no
calibration corpus or activation statistics. For each 32-value MXFP4 block it
evaluated the fixed percentile-99.5 candidate range with clip depth 4, always
including a no-clipping candidate, and selected the exponent with the lowest
unweighted weight MSE. These were existing MxWave defaults, not settings chosen
from the WikiText evaluation result.

The repository now exposes the measured configuration through the compatibility
expert CLI (whose original RTN name is retained so existing commands do not
break):

```bash
mxwave-quantize-expert-rtn \
  --model-dir /path/to/Nex-N2.5-mini-bf16 \
  --output-dir /path/to/Nex-N2.5-mini-mxwave-mse \
  --device cuda \
  --method mse \
  --scale-percentile 99.5 \
  --mse-clip-depth 4 \
  --tensor-row-chunk-size 2048 \
  --host-tensor-cap-mib 1024 \
  --source-repository nex-agi/Nex-N2.5-mini \
  --source-revision 87420286149d9cce9bd46cd335ef9bda33c37c1b
```

A 96-matrix probe covered layers 0, 13, 26, and 39, eight evenly spaced experts
per layer, and gate, up, and down matrices. Aggregate normalized reconstruction
MSE fell from 0.0128079936560 for RTN to 0.0126586091887, a 1.16634% reduction.
All eight layer/bank aggregates improved. Because local reconstruction is not a
promotion metric, this admitted exactly one full checkpoint and no parameter
sweep.

The full conversion completed in 517.7 seconds and emitted the same 87 shards,
62,386 tensors, and 22,902,614,752 tensor-data bytes as RTN. It loaded through
stock vLLM 0.29 with `MarlinExperts`. After correcting a stale descriptive RTN
label in its manifest, whole-checkpoint structural verification passed with:

- checkpoint: `1d134e5d5c704cc76fe85268ee1f65ece0107e932e9de856c141bdb5fa9e8738`;
- config: `8ea02e392011595c2394a26862e51966608abab24fa2b750ce798b4ce67f6351`;
- index: `69d4da6c13645e868454a7e22537e3f2ceef8883cf63817d67e2239145ad293c`;
- manifest: `29bd6b121e705e041f1e1520ce12d7e49b4fb4f07c66174907f6b63bd49a76c3`.

The clean repository CLI was then run with `--resume` against the measured
artifact. Its run identity matched, all 87 shards were accepted from their
integrity records without requantization, and the manifest hash remained
unchanged. This closes the temporary-driver reproducibility gap without a
second checkpoint build.

All four models were scored on the same first 64 non-empty 4,096-character
WikiText-2 chunks. Pairing was verified by chunk index, text hash, token hash,
prompt-token count, and scored-token count. Each model scored 61,408 tokens.

| Candidate | PPL | Relative to BF16 |
|---|---:|---:|
| BF16 | 8.8732193614 | baseline |
| MxWave unweighted MSE | 8.8809234683 | +0.086824% |
| MxWave RTN | 8.9103995255 | +0.419015% |
| Public NVFP4 | 9.0098083227 | +1.539339% |

The paired uncertainty analysis used 100,000 chunk-cluster bootstrap resamples
with seed `20260922`:

| Comparison | Relative PPL | First / last 32 | Paired 95% interval | Chunk wins / losses |
|---|---:|---:|---:|---:|
| MSE vs RTN | -0.330805% | -0.373023% / -0.286073% | [-0.576620%, -0.082547%] | 41 / 23 |
| MSE vs NVFP4 | -1.430495% | -1.485957% / -1.371722% | [-1.955160%, -0.930192%] | 45 / 19 |
| BF16 vs MSE | -0.086749% | -0.025395% / -0.151686% | [-0.514687%, +0.349066%] | 39 / 25 |

By point estimate, MSE recovered 79.28% of RTN's PPL gap to BF16 and 94.36% of
NVFP4's gap. The BF16-to-MSE interval crosses zero, so this experiment cannot
resolve a remaining likelihood difference between them. Conversely, both the
MSE-to-RTN and MSE-to-NVFP4 intervals exclude zero and agree across the two
fixed halves.

This is useful evidence, but PPL is only one metric. The raw PPL reports had
empty checkpoint-binding fields because the structural fingerprint was computed
after evaluation; their hashes and remote paths are retained in the compact
JSON record rather than silently rewriting them. The exact comparison below is
checkpoint-bound and supplies the pre-registered stopping decision.

### Exact next-token distribution divergence

The decisive follow-up used a separately frozen full-vocabulary protocol. It
selected 64 evenly spaced contexts from the 316 non-empty WikiText-2 test
chunks, including both endpoints, truncated each context to at most 512 Nex
tokens, and captured one terminal next-token distribution over all 248,320
model output IDs. Thus this is 64 prediction positions, not 64 times 512
positions. BF16, MSE, and RTN used identical token IDs and the same official
vLLM 0.29 image; the two MXFP4 checkpoints forced both Marlin backends.

- corpus SHA-256:
  `696cca6b65a171b0a358a4be6732cdfdf2dd6164a32e20fd70e3c13fc4dfae83`;
- context-manifest SHA-256:
  `9b72a2aa2b32c13dce3ff2e387d4f4a8d1383a240baf620a58d351566d8c9d3f`;
- comparison-protocol SHA-256:
  `e37d5803afa723cd7505baf407476f36e86c3168e9fbaa69251289e672013f38`;
- evaluator-script SHA-256:
  `c0f445f3b945786a2e86141dfb3a94fb82f0890b065d9a2187fce35b2c6b74a2`;
- bootstrap: 100,000 paired context resamples, seed `20260922`;
- report SHA-256:
  `fd86f5797bb452ec0997f5c25efbb3f0b28cb919c965e49cac499dcb889d97d7`.

| Metric against BF16 | MSE | RTN |
|---|---:|---:|
| Forward KL mean | 0.0341574002 | **0.0310492616** |
| Forward KL median | 0.0224198699 | **0.0156358342** |
| Forward KL p95 | 0.0937388614 | **0.0808118798** |
| Reverse KL mean | 0.0342553657 | **0.0317153584** |
| Jensen-Shannon mean | 0.0082373496 | **0.0075773702** |
| Total variation mean | 0.0702931608 | **0.0667302574** |
| Teacher top-1 agreement | 58/64 (90.625%) | 58/64 (90.625%) |
| Mean top-5 overlap | 4.453125 | 4.453125 |

Positive MSE-minus-RTN deltas favor RTN:

| Comparison | MSE mean | RTN mean | MSE minus RTN |
|---|---:|---:|---:|
| All 64 contexts | 0.0341574002 | 0.0310492616 | +0.0031081386 (+10.01%) |
| Contexts 0--31 | 0.0311781327 | 0.0250750375 | +0.0061030952 |
| Contexts 32--63 | 0.0371366678 | 0.0370234858 | +0.0001131820 |

MSE had lower forward KL on 34 contexts and RTN on 30, but MSE's larger losses
dominated the mean. The paired 95% interval was
`[-0.0014569435, +0.0075649953]` nats, so the 10.01% point regression is not a
resolved population estimate. It nevertheless fails the pre-registered gate:
MSE was not lower overall, was worse on both fixed halves, and the interval's
upper bound was not below zero. The paired teacher-top-1 table was 58 both
match, 0 MSE-only, 0 RTN-only, and 6 neither, so the top-1 guardrail passed.

PPL and terminal KL answer different questions. PPL scored the probability of
61,408 observed next tokens; this KL probe compared all vocabulary probabilities
at 64 terminal positions. The disagreement is therefore possible and useful:
unweighted scale search improved the observed-token likelihood while moving
the broader output distribution farther from BF16 at this bounded sample.

The raw exact artifacts remain under
`/home/kirya/local-spark/experiments/nex-n2.5-moe-preflight-20260922/next-token-kl/exact-64-20260922`.
The compact JSON record binds the context, BF16, MSE, RTN, and comparison hashes.
At this original stopping point, the optional public-NVFP4 distribution and a
128-context rerun were not run: both halves already reversed, so either would
have violated the predeclared stopping rule without changing the MSE-versus-RTN
decision. The owner later explicitly reopened only the public-NVFP4 comparison;
that result is recorded in the publication-gate extension below.

## Why route-aware calibration might still matter

The original preflight model uses plain RTN scale/value selection for every
routed expert. The admitted follow-up improves scale selection using weight MSE
but still does not observe routing or activations.
It minimizes local weight reconstruction without observing which tokens the
router sends to each expert or which input channels those tokens use. That is a
particularly weak approximation for sparse MoE weights:

1. each expert sees a conditional token distribution rather than the decoder's
   unconditional hidden-state distribution;
2. frequently and rarely selected experts do not have equal evidence or equal
   downstream importance; and
3. `down_proj` does not consume the decoder hidden state directly. It consumes
   the nonlinear product formed from the selected expert's gate and up paths.

These are plausible mechanisms, not causal findings from this screen. The BF16
control demonstrates that the one- and two-case score differences are ordinary
small-screen variation, not evidence of quantization damage. Task accuracy also
cannot measure distribution drift precisely enough to justify calibration.
The paired PPL result shows that generic MSE scale search removes most of the
observed-token likelihood deficit, while exact KL shows it does not improve the
full teacher distribution. This is objective mismatch, not evidence that a
larger unweighted-MSE search will help. Route-aware work remains a separate
hypothesis and must begin with a small held-out probe against RTN, not another
full checkpoint build.

The control's RTN method also means that route-aware calibration is a proposed
way to improve beyond both current artifacts, not a missing control feature that
already explains its score. NVFP4's smaller groups and finer scale hierarchy can
change reconstruction error independently of calibration; its W4A4 activation
path adds a separate numerical difference.

## Admitted follow-up and stopping rule

The single bounded unweighted-MSE scale-search follow-up is complete and stopped.
Do not sweep its percentile or clip depth, enlarge the exact-KL sample after both
halves reversed, run broader qualification, or publish the candidate because
PPL alone was favorable.

If route-aware calibration is revisited, use one separately pre-registered
small probe before any checkpoint build:

1. capture the actual input tokens routed to each expert, including observation
   counts and a fail-closed policy for under-observed experts;
2. capture the real post-SiLU gate/up product consumed by each expert's
   `down_proj`, rather than reusing the layer input statistic;
3. quantize with the existing activation/Hessian-aware objective while
   retaining the same standard checkpoint format and bounded expert streaming;
4. compare against RTN on disjoint held-out teacher-KL splits before rebuilding
   or running a large qualification suite.

That probe must show held-out teacher-relative improvement without sacrificing
the bounded-memory/runtime contract. The current result does not admit a full
route-aware conversion by itself.

## Final interpretation

- **Project value:** high as the first real bounded MoE emission/runtime proof.
- **Artifact value:** neither RTN nor MSE is a publication candidate; MSE is
  stopped after the exact-KL reversal.
- **Size:** real 4.21% advantage over the chosen public control, but below the
  stated materiality threshold.
- **Quality:** the original small task screen was unresolved. Paired PPL favors
  MSE over RTN and NVFP4, but exact terminal BF16-forward KL is 10.01% worse
  than RTN and reverses on both halves; the evidence is mixed and fails the
  frozen promotion rule.
- **Speed:** one favorable diagnostic run, not a claim.
- **Next action:** retain the MoE infrastructure and failure record. Do not do a
  full model rebuild or broader qualification for this MSE candidate. Any
  route-aware follow-up must first pass a new, small, disjoint teacher-KL probe.

## Owner-directed publication-gate extension

The owner later explicitly reopened one bounded comparison against the public
NVFP4 artifact. This did not change the MSE-versus-RTN directional failure
above; it answered the narrower question of whether the MSE artifact was still
a defensible Pareto publication candidate relative to the available public
quant.

The exact frozen no-thinking screen used the same 82 cases, request body,
tokenizer, vLLM 0.29 image, and `reasoning_effort=none` setting as the existing
controls. MSE completed every request and scored 45/69, versus 47/69 for public
NVFP4, 46/69 for BF16, and 45/69 for RTN. MSE retained 15/15 structured output,
10/10 short retrieval, and 9/9 8K retrieval. Its two-case deficit to NVFP4 was
unresolved: the exact McNemar value was `p=0.6875`, and the category-stratified
paired 95% interval for the pass-rate delta was `[-10.14, +4.35]` percentage
points. An initial MSE collection that used `chat_template_kwargs` instead of
the frozen `reasoning_effort=none` API field was excluded because its request
protocol and tokenization differed.

The public NVFP4 exact-distribution artifact was then collected for the same 64
contexts and all 248,320 output IDs. Mean BF16-forward KL favored MSE:
`0.0341574002` versus `0.0408951303`, a `-16.48%` point difference. The result
was not stable enough for a superiority or non-inferiority claim. The paired
relative 95% interval was `[-30.90%, +6.66%]`, the two fixed halves were
`+1.33%` and `-27.22%`, and teacher-top-1 agreement was 58/64 for MSE versus
60/64 for NVFP4. MSE had lower KL on 39 contexts and NVFP4 on 25. MSE's p95 KL
was better, while its median KL was worse; the favorable mean is partly driven
by NVFP4's larger tail errors.

Repeated serving used three paired seeds, prefix caching disabled, and the
artifacts' stock operating points: Marlin W4A16 experts for MSE and native
`FLASHINFER_CUTLASS` W4A4 experts for NVFP4. All 264 requests completed with no
errors, OOMs, or restarts.

| Workload | MSE | Public NVFP4 | MSE relative |
|---|---:|---:|---:|
| 8K prefill C1, median TTFT | 1240.51 ms | 1144.66 ms | +8.47% slower |
| 32-to-512 decode C1 | 37.395 tok/s | 36.440 tok/s | +2.63% |
| 512-to-128 balanced C4 | 125.315 tok/s | 119.619 tok/s | +4.77% |
| Model-load memory | 21.47 GiB | 22.41 GiB | -0.94 GiB |

The operational gate passed, but the result is workload-dependent rather than a
blanket speed win: NVFP4 prefills faster, while MSE decodes modestly faster.

### Publication decision after the extension

The classification is **mixed, not promoted**. MSE is a valid local Pareto
experiment: it is 4.21% smaller, has a resolved 1.43% paired-PPL advantage over
public NVFP4, uses 0.94 GiB less model-load memory, and has modest decode gains.
Those benefits do not clear the conservative publication gate because the size
win misses the stated 5% threshold, the task screen is two cases lower, exact
teacher-top-1 loses two net contexts, KL is split-unstable, and long prefill is
slower. It must not be described as generally higher quality or universally
faster than NVFP4.

The smallest credible corrective experiment is a four-layer route-aware
block-Hessian probe at layers 0, 13, 26, and 39. It should collect actual top-8
routed expert inputs, use the conditional hidden input for gate/up, use the real
post-activation gate-times-up input for down, weight contributions by squared
normalized router coefficients, and fall back to RTN for experts with effective
sample size below 64. Run two calibration replicas and two disjoint held-out
teacher-KL splits, patch only those eight source banks, and stop unless both
replicas improve both splits with a pooled paired interval below zero. Do not
build another full 40-layer checkpoint first.

The full machine-readable record is
[`benchmarks/nex-n2.5-mini-publication-gate-2026-09-22.json`](../benchmarks/nex-n2.5-mini-publication-gate-2026-09-22.json).
