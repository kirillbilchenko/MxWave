# MxWave methodology: a practical guide

This document explains the method behind MxWave, how to interpret its results,
and what the successful and failed experiments actually taught us. It is the
recommended starting point before reading the individual experiment records.

## Human-readable TL;DR

MxWave tries to make a floating-point language model much smaller while
preserving its behavior. It stores most selected weight values in MXFP4: a
4-bit floating-point format in which every group of 32 values shares one
power-of-two scale.

The difficult part is not merely rounding each weight to four bits. It is
choosing the shared scale. A scale that reconstructs the weights well on paper
can still damage the model if its error falls on input channels the model uses
heavily. MxWave therefore runs real text through the original model, observes
the inputs to each quantized projection, and uses their second moments when
choosing each block's scale.

The published `H64` recipe means:

- `H`: block-Hessian-weighted scale selection;
- `64`: 64 calibration sequences, each 512 tokens long;
- the stored values are still ordinary OCP MXFP4 E2M1, not “64-bit” values;
- it is scale selection only, followed by nearest-value rounding. It is not a
  full GPTQ sequential weight update.

The main result is good, but not lossless. On our paired WikiText-2 protocol,
the BF16 model had perplexity `7.980864`, H64 had `8.119445`, and AMD's
Quark-AWQ MXFP4 model had `8.187544`. H64 was therefore 1.736% worse than BF16
but 0.832% better than AMD on that exact protocol. It also used only 18.47 GiB
of tensor data versus 51.75 GiB for the source. Those numbers are strong
evidence for this model and harness, not a universal ranking of quantizers.

Most later ideas did **not** improve the final model. Several produced better
weight SQNR, lower layer-output error, or lower local teacher KL, then became
neutral or worse when inserted into the whole network. This is the central
lesson from the research: **a better local reconstruction score is a useful
diagnostic, not proof of a better language model**.

One pure-PTQ refinement passed the full process. A bounded mixed-precision candidate
kept 388 tensors in MXFP4 and promoted only 12 measured-sensitive tensors to
FP8. It reduced perplexity from `8.119445` to `8.111578`, added 375.6 MiB, and
was roughly 2.2% slower in the tested vLLM setup. Its full GSM8K score moved in
the favorable direction, but the paired confidence interval crossed zero. It
is therefore an optional quality/size trade-off, not a replacement for H64.

A later, materially larger recovery-training experiment changed legal E2M1
codes in 12 tensors while preserving H64's scales and standard checkpoint
format. It reduced PPL to `8.067761`, a 0.654% gain over H64 that recovered
37.92% of H64's gap to BF16. That mechanism worked, but the exact-distribution
metrics were mixed and full GSM8K was tied. It remains a local research
checkpoint rather than another public release.

In plain terms:

- **Best default:** H64, because it is the smaller and faster fully evaluated
  checkpoint.
- **Published optional Pareto point:** the 12-tensor precision-budget model, if
  a small size and speed premium is acceptable.
- **Lowest measured PPL:** the local recovery checkpoint, which is not promoted
  because broader behavioral evidence did not improve consistently.
- **Native NVFP4 result:** the tested Minima W4A4 checkpoint was 5.382% worse
  than H64 in PPL, 45.1% worse in mean forward KL, and about 4.02% larger after
  normalizing both artifacts to language-only capability. Native execution was
  verified, but it did not advance the size/quality frontier.
- **What succeeded scientifically:** real activation calibration, bounded
  streaming, strict output verification, paired end-to-end evaluation, and
  measured mixed-precision allocation.
- **What repeatedly failed:** selecting changes only because they improved a
  local proxy.
- **What a failed experiment means:** the frozen hypothesis failed under its
  recorded scope. It does not prove that the entire family of methods can
  never work on another model, dataset, budget, or training regime.
- **Current boundary:** within pure PTQ, unchanged all-MXFP4 size, standard
  vLLM kernels, and this Qwen checkpoint, the inexpensive variations appear
  close to a practical ceiling. Meaningful additional gains probably require
  spending precision selectively, adding recovery training, or changing the
  runtime/format constraint.

## The problem MxWave is solving

For one linear layer,

```text
Y = X W^T
```

quantization replaces `W` with `Wq`. The layer error is therefore

```text
DeltaY = X (Wq - W)^T = X DeltaW^T
```

Plain weight MSE minimizes `||DeltaW||^2`. That treats every input channel as
equally important and independent. Real activations are neither. Taking an
expectation over calibration inputs gives the output-error proxy

```text
E[||DeltaY||^2] = trace(DeltaW H DeltaW^T)
H = E[X^T X]
```

MxWave stores `H` block-diagonally, with one `32 x 32` matrix for each MXFP4
input block. This retains correlations within a block without constructing a
full-width matrix.

We call this a block Hessian because it is the usual input second-moment or
Gauss-Newton-style matrix used by second-order PTQ methods. It is **not** the
full Hessian of the language-model loss, and H64 does not perform GPTQ's
sequential inverse-Hessian error compensation.

## What MXFP4 stores

The relevant OCP MXFP4 rules are:

- each value uses the E2M1 4-bit format;
- positive magnitudes are `0, 0.5, 1, 1.5, 2, 3, 4, 6`;
- two codes are packed into each byte;
- 32 consecutive input-channel values share one E8M0 scale;
- the scale is a biased power-of-two exponent, not a normal floating-point
  number.

For a candidate scale `s`, MxWave reconstructs a block approximately as

```text
Wq = s * nearest_E2M1(clamp(W / s, -6, 6))
```

It evaluates a small, explicit set of nearby scale exponents, including a safe
non-clipping control, and selects the one with the lowest chosen objective.
The objective priority is:

1. block Hessian: `trace(DeltaW H DeltaW^T)`;
2. per-channel activation magnitude: weighted squared error;
3. unweighted weight MSE.

The old LayerNorm-gamma path is only a cheap proxy: a learned normalization
parameter is not the same thing as the empirical distribution reaching every
linear module. Real `mean-abs`, RMS, and block-Hessian statistics come from the
actual module inputs. Of those, H64's block Hessian was the strongest validated
choice for Qwen3.8-27B.

## End-to-end workflow

```text
pin model + corpus + runtime
            |
            v
collect real module inputs, one decoder layer at a time
            |
            v
choose a scale for every 32-value block
            |
            v
pack weights in compressed-tensors MXFP4 form
            |
            v
verify tensor shapes, dtypes, coverage, and reconstruction
            |
            v
serve with the same runtime used for qualification
            |
            v
evaluate distribution, likelihood, tasks, and system cost
```

### 1. Freeze the inputs

Record the source revision, tokenizer, corpus revision and hash, calibration
sequence construction, software image, quantization arguments, and random
seeds. Otherwise a later difference can be caused by data or runtime drift
rather than the quantization method.

### 2. Collect calibration statistics

The streaming calibrator constructs the model on the meta device, loads the
embedding and one decoder layer at a time, passes the hidden states forward,
stores only accumulated statistics, and releases the layer. Hidden states stay
on CPU between layers. This bounds tensor residency; filesystem cache can still
make host-memory monitoring look larger than the active tensor set.

Calibration is architecture-aware because the tool must know how to replay a
decoder layer correctly. The quantization math itself is architecture-neutral,
but a new model family needs a validated streaming adapter and policy.

### 3. Quantize in bounded chunks

Source matrices are read by row ranges and moved to the accelerator in bounded
chunks. Completed tensors accumulate only until the current output shard can be
saved atomically. This is how the converter can process a source checkpoint
larger than available accelerator memory.

### 4. Verify the artifact

Verification checks more than whether files exist:

- every intended linear module is targeted or explicitly ignored;
- packed weights and E8M0 scales have the correct shape and dtype;
- dequantization is finite and reconstruction scores are plausible;
- configuration and tensor names agree with the runtime format;
- fused runtime groups are kept compatible.

Coverage is critical. A valid-looking configuration that omits a module can
silently load the wrong representation or produce invalid output.

### 5. Evaluate behavior, not only tensors

The qualification ladder is deliberately expensive only after a candidate has
earned it:

```text
hypothesis
  -> frozen scope and pass/fail gate
  -> implementation invariants
  -> cheap local screen
  -> replication on a disjoint split
  -> candidate checkpoint
  -> untouched whole-model KL/PPL
  -> task, serving-smoke, size, and throughput checks
  -> promote, keep as optional, or reject
```

If a gate fails, the run stops. We do not weaken a threshold after seeing the
result, keep trying nearby layers on the same holdout, or build an expensive
checkpoint merely to search for a favorable metric.

## How to read the metrics

| Metric | What it answers | Appropriate use | Main limitation |
|---|---|---|---|
| Weight MSE / SQNR | Did packing or reconstruction become numerically better? | Unit checks and cheap screening | Ignores inputs and the rest of the model |
| Hessian-weighted SQNR | Is local linear-output error predicted to be lower on calibration data? | Choosing or checking scales | Still a local proxy and can overfit calibration |
| Operator-output NMSE | Does one replaced module/layer match its teacher output? | Stronger local screening | Later layers can amplify, cancel, or redirect the error |
| Teacher KL | Did the next-token distribution move away from BF16? | Sensitive behavioral comparison | Context-dependent; needs paired uncertainty |
| Prompt perplexity | How much probability did the model assign to known continuation tokens? | Primary deterministic language-quality gate | Corpus and window protocol matter |
| Top-1/top-k agreement | Did likely token choices remain the same? | Interpretability alongside KL | Discards most distribution information |
| Task accuracy | Can the model still solve a user-facing task? | Final capability/non-inferiority check | Noisy, prompt- and parser-dependent |
| Throughput, latency, memory, size | Is the artifact operationally useful? | Product decision | Runtime and hardware specific |

For paired comparisons, evaluate the same contexts with both candidates. The
resampling unit should be an independent context, window, or task item—not each
token inside a shared context. If a 95% paired interval crosses zero, the data
do not establish the sign of the mean effect. That is “inconclusive,” not “the
models are identical.” For paired binary task outcomes, McNemar's test examines
the examples on which only one model is correct.

## What succeeded

### H64: the successful all-MXFP4 path

H64 combined real block-Hessian calibration, scale-only MXFP4 quantization,
bounded processing, and strict verification. On the fixed 316-window,
297,199-token WikiText-2 comparison:

| Model | Prompt perplexity | Relative to BF16 |
|---|---:|---:|
| BF16 source | **7.980864** | baseline |
| MxWave H64 | **8.119445** | +1.736% |
| AMD Quark-AWQ MXFP4 | 8.187544 | +2.590% |

H64's PPL was 0.832% lower than AMD's on the same harness, with a paired
interval supporting the direction. Its mean next-token forward KL was also
lower, but that smaller 128-context KL comparison was not statistically
conclusive. The 100-item GSM8K run was a pilot, not a release-quality ranking.

This is a success because the artifact met the format, memory, size, runtime,
and end-to-end quality goals—not because every metric favored it.

### Precision budgeting: the successful optional path

The only later quality modification to pass the complete selection and holdout
process was measured mixed precision. Twelve of 400 target tensors were
promoted to channel-wise FP8 under a fixed 1 GiB budget; the other 388 remained
MXFP4.

It produced a small, credible PPL gain, passed task non-inferiority and serving
smoke tests, and stayed inside the 3% throughput-loss budget. Because it added
375.6 MiB, lost about 2.2% throughput, and did not establish a statistically
conclusive GSM8K gain, it is a new Pareto point rather than a new default.

The useful idea is not simply “use FP8.” It is: measure whole-model sensitivity,
require agreement on disjoint selection splits, allocate a strict byte budget,
and verify the combined allocation on untouched data.

### Recovery: a successful mechanism but rejected release candidate

The initial four-context recovery probe overfit and correctly stopped. A later
pre-registered run used 65,536 training tokens, legal fake-MXFP4 forwards, and
12 fixed targets. It improved paired PPL on 295/316 windows and recovered
37.92% of H64's PPL gap to BF16 without changing checkpoint size or runtime
format.

That positive likelihood result did not become a broad quality win. Mean
forward KL moved 1.70% favorably but its paired interval crossed zero; reverse
KL, JS, TV, and top-token agreement moved slightly unfavorably; full GSM8K was
a tie. The correct conclusion is narrower than either “recovery failed” or
“the recovered model is better”: scaled recovery can optimize this model's
likelihood, but this checkpoint is not sufficiently differentiated to publish.

### Process and engineering successes

- streaming calibration made a 51.75 GiB source practical on DGX Spark;
- row-chunked conversion bounded accelerator residency;
- format detection and coverage validation prevented silent incompatibilities;
- fused-module checks caught invalid partial precision changes;
- frozen gates prevented exploratory wins from turning into unsupported claims;
- machine-readable reports, hashes, and runbooks made positive and negative
  results reproducible.

## What failed, and why the failures matter

| Experiment | Local or exploratory signal | End result | Lesson |
|---|---|---|---|
| Real RMS weighting | Uses real activation magnitudes | Did not beat block Hessian | Diagonal magnitude loses within-block correlation information |
| Full Hessian rounding sweep | Better calibration-weighted SQNR | GSM8K pilot fell to 86/86 | Optimizing the calibration quadratic more aggressively can damage behavior |
| Block-local feedback | Better held-out local SQNR | Paired PPL was 0.146% worse | Local output recovery did not transfer through the network |
| Cross-block error propagation | Improved all six local probes by 1.4–15.8% | Layer-63 checkpoint was neutral-to-worse in PPL | A strong local gain can still be globally irrelevant |
| Static low-rank recovery | Rank 4/8 recovered some MXFP4 loss | Only 12.96%/15.61% recovery; failed the gate | The error was not concentrated enough in a cheap low-rank correction |
| Operator-response selector | Some candidates lowered exact local/suffix KL | Local winner matched final teacher winner only 2/6; whole model regressed | The best isolated replacement need not be best on the quantized trajectory |
| Layer-0 diagonal-Hessian patch | Passed an isolated fresh-split confirmation | Whole-model forward KL became 1.16% worse | Even replicated single-operation evidence may not compose |
| Small fake-MXFP4 recovery probe | Training NMSE improved 27.83% | Validation first worsened; fixed retry recovered only 6.44% vs a 10% gate | Tiny recovery training overfit; the later scaled run is a separate positive experiment |
| Residual counteraction selector | Tried to model signed cancellation with inherited error | Mean Spearman correlation was -0.0167 | A more elaborate proxy is not automatically more predictive |
| Suffix JVP selector | Differentiated a candidate through the remaining network | Mean Spearman was 0.333, below 0.50; only 1/3 stable late-layer signals | First-order downstream sensitivity was still too unstable |
| Layer-62 unweighted confirmation | Earlier splits favored unweighted MSE | Fresh split was 0.913% worse; interval crossed zero | Discovery-split wins must replicate before checkpoint construction |
| Exact-symmetry channel layout | Function-preserving permutations passed every invariant | Neither sort passed any layer on both splits | Valid implementation plus plausible theory can still yield no stable effect |
| Disagreement-weighted calibration | Reweighted all calibration Hessians using BF16-versus-H64 disagreement | Only 1/4 layers passed both splits; pooled interval crossed zero | Global hard-example weighting did not generalize reliably |
| Coupled gate/up rounding | Product NMSE improved 4.1--9.3% with legal codes | Every layer reversed direction across the two held-out splits | Multiplicative local cancellation was real but not predictive |
| FP8 `lm_head` compression | Removed 1.183 GiB / 6.406% | Forward KL regressed 4.670% with a positive paired interval | The byte reservoir is real, but naive per-row FP8 is not quality-neutral |
| MLX affine mixed-bit allocation | PPL improved 4.251% over affine 3-bit | Missed the 5% gate and had split-unstable KL | Format-specific sensitivity matters; splicing third-party weights is not MxWave quantization |

These outcomes fall into different categories:

- **Implementation failure:** invariants, reconstruction, or format checks fail.
  Fix the code; this says nothing about the hypothesis.
- **Hypothesis rejection:** implementation checks pass, but a frozen quality
  gate fails. Record it and stop.
- **Generalization failure:** selection improves, but a disjoint or untouched
  split reverses the effect.
- **Composition failure:** a local replacement helps in isolation but hurts the
  complete model trajectory.
- **Efficiency rejection:** quality improves, but not enough for its byte,
  memory, or speed cost.
- **Statistical inconclusiveness:** the point estimate looks favorable but the
  paired interval crosses zero. Keep claims conservative.
- **Runtime incompatibility:** the representation is valid in theory but the
  deployed kernel cannot execute it correctly or efficiently.

Negative results are useful here. They narrow the search space, expose weak
proxies, and prevent a larger, slower checkpoint from being called “better” on
the basis of one favorable sample.

## A template for the next experiment

Write this down before running the expensive measurement:

1. **Product question:** smaller, more accurate, faster, or more portable?
2. **Baseline and teacher:** exact model revisions and runtime.
3. **One hypothesis:** what mechanism should improve which behavior?
4. **Allowed scope:** tensors, layers, precision formats, and maximum candidates.
5. **Budgets:** additional bytes, peak memory, runtime, and wall-clock limits.
6. **Data partitions:** discovery, selection, and untouched confirmation.
7. **Primary metric:** one metric that decides the experiment.
8. **Guardrails:** task non-inferiority, top-1 agreement, format validity, and
   throughput limits.
9. **Frozen pass/fail rule:** include uncertainty and paired-win requirements.
10. **Stop rule:** state exactly what happens when a gate fails.
11. **Immutable record:** hashes, commands, logs, raw per-example values, and
    code revision.

A candidate should be promoted only for the claim its gate supports. Passing
PPL does not prove better reasoning; passing GSM8K does not prove better general
language modeling; passing a local reconstruction gate proves neither.

## Recommended internal reading path

### First: understand the released model

1. [README](../README.md) — project scope, commands, and headline results.
2. [H64 model card](QWEN3_8_27B_H64_MODEL_CARD.md) — what was quantized and the
   release claims.
3. [H64 reproducibility record](QWEN3_8_27B_H64_REPRODUCIBILITY.md) — exact
   inputs, commands, hashes, and environment.
4. [`core.py`](../mxwave/core.py) — the shortest path from the equations above
   to the actual scale search and packing.
5. [`calibration.py`](../mxwave/calibration.py) and
   [`calibration_stream.py`](../mxwave/calibration_stream.py) — statistic
   collection and bounded layer replay.

### Then: follow the successful refinement

1. [Selective BF16](QWEN3_8_27B_SELECTIVE_BF16.md) — useful signal, but too
   expensive.
2. [Selective FP8](QWEN3_8_27B_SELECTIVE_FP8.md) — a better quality/byte signal,
   but still below its promotion gate.
3. [Measured precision-budget experiment](QWEN3_8_27B_PRECISION_BUDGET.md) —
   the first complete positive result.
4. [Reusable precision-budget runbook](PRECISION_BUDGET_RUNBOOK.md) — how to
   repeat the methodology without tuning to this exact result.
5. [Final recovery qualification](QWEN3_8_27B_RECOVERY_FINAL_QUALIFICATION_2026-09-21.md)
   — a real PPL gain that was not broad enough for release.
6. [Unified qualification runbook](QUALIFICATION_RUNBOOK.md) — the fail-closed
   structural, quality, and runtime evidence contract.

### Finally: study why plausible ideas failed

1. [Cross-block reconstruction](QWEN3_8_27B_CROSSBLOCK_PROBE.md)
2. [Static low-rank recovery](QWEN3_8_27B_LOW_RANK_RECOVERY.md)
3. [Operator-response selection](OPERATOR_RESPONSE_PROBE_2026-09-18.md)
4. [Small recovery-training probe](QAT_RECOVERY_PROBE_2026-09-18.md)
5. [Residual counteraction](COUNTERACTION_PROBE_2026-09-19.md)
6. [Suffix-JVP selection](SUFFIX_JVP_PROBE_2026-09-19.md)
7. [Final bounded confirmation and exact-layout tests](FINAL_BOUNDED_EXPERIMENTS_2026-09-19.md)
8. [Research roadmap](MXWAVE_RESEARCH_ROADMAP.md) — retained ideas and current
   decision boundaries.

Some research records live only on their experiment branch until their code is
accepted or deliberately archived. A broken link on another branch means the
record was not merged; it should not be silently recreated from memory.

## External reading, in a useful order

1. [OCP Microscaling Formats (MX) v1.0 specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
   — authoritative E2M1, E8M0, block, and encoding rules.
2. [vLLM compressed-tensors](https://github.com/vllm-project/compressed-tensors)
   — the checkpoint representation MxWave emits.
3. [AWQ](https://arxiv.org/abs/2306.00978) — why activation-aware weight
   importance can preserve quality. MxWave is activation-aware, but H64 is not
   the AWQ search or rescaling algorithm.
4. [GPTQ](https://arxiv.org/abs/2210.17323) — the second-order PTQ lineage and
   sequential error compensation. MxWave borrows the second-moment motivation;
   H64 does not implement full GPTQ.
5. [SmoothQuant](https://arxiv.org/abs/2211.10438) — moving quantization
   difficulty between activations and weights. It is conceptually adjacent,
   mainly aimed at W8A8 rather than MxWave's W4A16 path.
6. [QuaRot](https://arxiv.org/abs/2404.00456) — rotations that reduce outliers
   while preserving the function. MxWave has rotation primitives, but the H64
   release uses no rotation.
7. [QERA](https://arxiv.org/abs/2410.06040) — activation-aware low-rank
   reconstruction of quantization error; useful context for the bounded
   low-rank probe.
8. [LLM Compressor quantization API](https://docs.vllm.ai/projects/llm-compressor/en/latest/api/llmcompressor/modifiers/quantization/)
   and [NVFP4 example](https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/quantization_w4a4_fp4/)
   — a production-oriented comparison for recipes, calibration, and formats.
9. [vLLM quantization documentation](https://docs.vllm.ai/en/latest/features/quantization/)
   — hardware and runtime support are part of the deployable method.
10. [lm-evaluation-harness task guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md)
    and [GSM8K task definition](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/gsm8k.yaml)
    — why prompt, few-shot, generation, and answer-extraction settings must be
    recorded with a task score.
11. [Efron's bootstrap overview](https://onlinelibrary.wiley.com/doi/pdf/10.2307/3314608)
    — statistical background for the paired resampling intervals used here.

## Final mental model

Think of quantization research as a chain of approximations:

```text
weight error
  -> predicted layer-output error
  -> measured local operator error
  -> final token-distribution error
  -> corpus likelihood
  -> task behavior
  -> deployed latency, memory, and reliability
```

Each arrow can break. MxWave's methodology is valuable because it makes those
breaks visible. The quantizer uses a stronger local approximation than plain
weight MSE, but the evaluation process never assumes that the approximation is
the final truth.
