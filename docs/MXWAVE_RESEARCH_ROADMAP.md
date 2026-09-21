# MxWave research directions and experiment tracker

> Working research note. This document records hypotheses and possible experiments;
> it is not a claim of novelty or measured improvement. Keep completed experiment
> results in separate reproducibility records. Last reviewed: 2026-09-21.

## Purpose

MxWave already provides a reproducible, streaming, calibration-aware MXFP4
baseline that emits standard `compressed-tensors` checkpoints. The next research
question is not whether another local scale-search variation can improve a
reconstruction metric. It is whether MxWave can select quantization errors that
the deployed architecture and runtime can absorb, while retaining:

- bounded host and accelerator memory;
- standard vLLM-compatible checkpoint formats;
- no custom inference kernel or online transform;
- model-family portability rather than Qwen-only rules;
- held-out, end-to-end promotion criteria.

The current north-star hypothesis is:

> Quantization should be optimized against the actual fused, nonlinear, and
> stateful inference operation, not only against an isolated weight matrix or
> linear layer.

## Current baseline and lessons

### What is established

- The Qwen3.8-27B H64 checkpoint is the reproducible MxWave MXFP4 baseline.
- Full block-Hessian scale selection is useful as a baseline, but scale-only
  Hessian optimization is now a crowded research area.
- The cross-block probe improved held-out layer-output normalized MSE by up to
  15.79%, but its targeted checkpoint made paired perplexity 0.0322% worse.
- Therefore, local weight MSE, SQNR, or block-output error cannot by itself be a
  promotion signal.
- The 2026-09-18 operator-response probe extended that result: local
  post-decoder-layer NMSE agreed with final teacher-KL winners on only 2/6
  measured MLP layers. Its pre-registered 70% gate became unreachable, so the
  replication stopped without a model build.
- Exact suffix teacher-KL argmin was also unstable across two disjoint
  eight-sequence splits: only 1/6 layer winners reproduced, and no non-baseline
  candidate passed a both-splits plus stratified-bootstrap rule.
- A residual-counteraction selector was numerically valid but failed every
  predictive gate: mean Spearman was -0.0167, cross-split improvement was 4/12,
  and only 3/6 selections reproduced.
- Propagating the exact candidate perturbation through the complete remaining
  network with native forward-mode AD recovered some late-layer signal, but
  still failed as a general selector: mean Spearman was 0.3333 versus 0.4000
  for weight NMSE, cross-split improvement was 4/6, and only 1/3 selections
  reproduced. SDPA Flash Attention lacked a forward-AD rule; eager attention
  completed with bounded memory.
- A fixed third-split confirmation isolated one reproducible exception: using
  diagonal-Hessian for layer-0 MLP weights reduced teacher KL by 13.29% on the
  fresh 16-context split and 12.46% over 32 pooled contexts, with positive
  paired intervals. Layer 14 did not pass its fixed gate.
- A new method must be accepted using disjoint data and an end-to-end signal such
  as teacher KL, perplexity, or a suitably validated proxy.

### Latest completed cycle: September 20--21

| Experiment | Result | Decision |
|---|---|---|
| Sequence-level disagreement-weighted calibration | Only 1/4 fixed layers improved both held-out splits; pooled paired interval crossed zero | Rejected; no control or checkpoint build |
| Coupled gate/up rounding | Legal joint rounding reduced calibration product NMSE by 4.1--9.3%, but 0/4 layers generalized across both splits | Rejected; local gated-product error is not a promotion signal |
| Large fake-MXFP4 recovery | PPL improved 0.654% versus H64 and recovered 37.92% of the BF16 gap; exact divergence was mixed and full GSM8K tied | Successful mechanism, local-only checkpoint; not an H64 replacement |
| FP8 `lm_head` compression | Saved 1.183 GiB / 6.406%, but mean forward KL regressed 4.670% with a positive paired interval | Rejected before PPL, tasks, MTP, or MXFP4-head escalation |
| MLX affine 3/4-bit allocation | Improved PPL 4.251% versus affine 3-bit but missed the frozen 5% gate; KL was split-unstable | Local Pareto evidence only; this spliced third-party weights and did not test MxWave quantization or export |

The scaled recovery result supersedes the earlier statement that recovery was
closed after the 6.44% local probe. It reopens recovery only as a materially
larger training project; it does not justify a small hyperparameter sweep. The
canonical decision record is the
[final recovery qualification](QWEN3_8_27B_RECOVERY_FINAL_QUALIFICATION_2026-09-21.md).

Detailed rejected-probe records remain intentionally isolated from production
code:

- DWC: `research/disagreement-weighted-calibration` at `18c9653`,
  `docs/DWC_PROBE_2026-09-20.md` and
  `benchmarks/qwen3.8-27b-dwc-probe.json`;
- coupled rounding: `research/coupled-gate-rounding` at `8c564b6`,
  `docs/COUPLED_ROUNDING_PROBE_2026-09-20.md` and
  `benchmarks/qwen3.8-27b-coupled-rounding-probe.json`;
- recovery implementation: `research/end-to-end-mxfp4-recovery` at `734db06`;
- FP8-head compression: `research/recovery-lm-head-compression` at `431f158`,
  `docs/QWEN3_8_27B_RECOVERY_FP8_LM_HEAD.md` plus its compact JSON reports.

The MLX affine-allocation record stays outside the production repository under
`mxwave-experiments/2026-09-21-mlx-affine4-budget`. Its durable conclusion is
captured here so the artifact cannot later be mistaken for a BF16-to-MxWave
conversion.

### Ideas already tested or rejected locally

| Idea | Local conclusion | Revisit only if |
|---|---|---|
| Larger/full Hessian alone | Insufficient evidence of end-to-end gain | Statistical stability changes |
| GPTQ-style rounding | Did not justify production promotion | Coupled to a better global objective |
| Feedback selector | Did not produce a useful model-level gain | A new trust region is available |
| Cross-block compensation | Local gain, paired PPL neutral-to-worse | End-to-end selection replaces local NMSE |
| Static low-rank recovery | Not competitive enough | Recovery is trained or operator-aware |
| Selective BF16/FP8 | Quality/size trade-off was insufficient | Sensitivity ranking becomes materially better |

These negative results remain valuable controls and should not be silently rerun
under a new name.

## Research activity snapshot

A broad arXiv all-fields search for `NVFP4 OR MXFP4`, sampled on 2026-09-18,
returned approximately 120 records from November 2024 onward. Of those records,
29 were from 2024-2025 and 91 were from 2026 through the sampling date. This is
an activity measure, not a count of independent or peer-reviewed methods.

Hugging Face searches for either term fill the 1,000-result search page. Those
results include mirrors, conversions, and derived checkpoints, so they measure
format adoption rather than original research.

- [arXiv activity search](https://arxiv.org/search/?query=NVFP4+OR+MXFP4&searchtype=all&abstracts=show&order=-announced_date_first&size=200)
- [Hugging Face NVFP4 search](https://huggingface.co/models?search=NVFP4)
- [Hugging Face MXFP4 search](https://huggingface.co/models?search=MXFP4)

The practical conclusion is that generic FP4 PTQ is no longer an uncrowded
topic. A useful contribution needs either a stronger objective, a new systems
constraint, or convincing evidence across architectures.

## Where current work is converging

### 1. Scale selection is becoming sophisticated but crowded

[H-Scale](https://arxiv.org/abs/2608.28113) uses hardware-valid,
diagonal-Hessian-guided NVFP4 scale refinement. [SOAR](https://arxiv.org/abs/2605.12245)
jointly optimizes global and block scales. [FAAR](https://arxiv.org/abs/2603.22370)
adds format-aware learned rounding and alignment.

Implication for MxWave: another nearby exponent candidate or larger Hessian is
unlikely to be a differentiated contribution by itself.

### 2. Calibration quality and Hessian reliability matter

[DASH-Q](https://arxiv.org/abs/2604.13806) reports that off-diagonal Hessian
terms can be dominated by sampling noise under limited calibration, while the
diagonal is more stable. [MaCa](https://arxiv.org/abs/2602.07465) shows that
sequence-length distribution affects calibration, and
[target-aware DPQ](https://arxiv.org/abs/2608.21019) argues against one universal
calibration recipe.

Implication for MxWave: full block Hessians should not automatically be treated
as more informative. Shrinkage, diagonal controls, multiple lengths, and held-out
selection are required.

### 3. Rotation, redistribution, and mixed precision are active areas

[Block Rotation](https://arxiv.org/abs/2511.04214) adapts rotations to
microscaling constraints. [MixQuant](https://arxiv.org/abs/2601.22347) performs
permutation-equivariant redistribution before rotation.
[dMX](https://arxiv.org/abs/2606.04115) and
[AdaMX](https://arxiv.org/abs/2608.03867) optimize mixed formats or precision.

Implication for MxWave: block permutation, rotation, or adaptive precision can
still be useful engineering, but none should be presented as a new category.

### 4. Objectives are moving beyond local reconstruction

[REAL-Q](https://arxiv.org/abs/2609.00049) aligns its surrogate more closely with
end-to-end behavior. [KronQ](https://arxiv.org/abs/2607.07964) adds output-gradient
covariance rather than assuming equal output sensitivity. Low-rank error repair
is also active in [ProjQ](https://arxiv.org/abs/2606.00494) and
[QERA](https://arxiv.org/abs/2410.06040).

Implication for MxWave: merely adding an output-side Hessian or low-rank residual
has clear prior art. Any new proposal needs to explain what the linearized
objectives fail to capture.

### 5. Runtime execution details can dominate quality

[Minima's Qwen3.8-27B NVFP4 work](https://arxiv.org/abs/2609.04098) quantizes all
496 backbone linear layers, including Gated DeltaNet, to W4A4. It reports that
some large local GEMM errors are attenuated by downstream nonlinearities and
that mismatched global scales in vLLM-fused projections materially damage the
model. Its [released checkpoint](https://huggingface.co/minima-ai/mnma_qwen3.8_27b_nvfp4)
is approximately 17.53 GiB.

LLM Compressor now documents
[observer fusion](https://docs.vllm.ai/projects/llm-compressor/en/stable/guides/observers/)
for tensor-group formats such as NVFP4. Consequently, supporting GDN or making
known fused scales consistent is important engineering, but is no longer enough
for a novelty claim.

## Opportunity map

| Direction | Research potential | Stock vLLM artifact | Expected effect | Initial effort |
|---|---:|---:|---|---:|
| Execution-aware nonlinear/state response | High | Yes | Better quality decisions | 2-4 weeks |
| Joint fused-group scale optimization | Medium-high | Yes | Quality; perhaps more safe FP4 coverage | 1-2 weeks |
| Robust multi-domain/length calibration | Medium | Yes | Reliability and fewer regressions | About 1 week |
| Learned rounding or recovery | High absolute quality | Yes after export | Quality, no inherent speed gain | 4-12 weeks plus compute |
| Exact-symmetry layout/permutation | Uncertain | Usually | Quality at unchanged format | 2-8 weeks |
| Custom adaptive FP4 format | High but risky | No | Possible size/quality gain | Large runtime project |
| Speculative-decoding co-design | Separate speed project | Possibly | Latency/throughput | Separate roadmap |

The recommended program has two connected tracks:

1. build execution-aware scoring and robust calibration using formats that vLLM
   already supports;
2. consider a custom adaptive FP4 representation only if the first track exposes
   a repeatable limitation that standard mixed precision cannot solve within the
   target byte and latency budgets.

Standard vLLM compatibility remains a hard requirement for the first track and
for distributable checkpoints. A custom format is an evidence-gated research
track, not part of the next experiment.

## Two-track research program

### Track A: execution-aware quantization compiler

Track A supplies the measurement system. It estimates which quantization errors
remain harmful after runtime fusion, nonlinearities, residual paths, and
recurrent state. Its candidates use existing MXFP4, NVFP4, FP8, and BF16
representations, and its output remains an ordinary vLLM-compatible checkpoint.

### Track B: adaptive FP4 representation

Track B begins only after Track A identifies stable failure shapes. Possible
adaptation dimensions include codebook, scale granularity, block size, outlier
handling, and representation choice by operation or block. Before implementing
a kernel, a numerical prototype must demonstrate a Pareto improvement over
standard-format mixed precision:

- lower held-out divergence at the same tensor-byte budget; or
- fewer tensor bytes at the same held-out divergence;
- with an estimated kernel path that does not erase the gain through metadata,
  unpacking, or dispatch overhead.

This ordering avoids designing a format around assumed failure modes. The
working combined thesis is:

> MxWave measures how the deployed architecture absorbs quantization error, then
> assigns the least expensive numerical representation that satisfies a bounded
> output-divergence target.

## Primary hypothesis: finite operator-response quantization

### Motivation

Most PTQ objectives estimate the effect of replacing a weight matrix with its
quantized approximation. They do not directly model whether the next softmax,
gate, residual path, or recurrent update attenuates or amplifies that error.
They may also calibrate source modules that are fused differently at inference.

MxWave can instead evaluate legal quantization candidates through a bounded
replay of the operation that vLLM will actually execute.

### Proposed procedure

1. Canonicalize source modules into runtime groups, such as fused QKV, gate/up,
   GDN QKV+Z, and GDN B+A, including shared-scale requirements.
2. Generate a small set of valid candidates using standard MXFP4, NVFP4, FP8,
   or BF16 representations and legal global/local scales.
3. Inject each candidate's actual dequantized error into captured activations.
4. Replay the enclosing operator or block, including its nonlinearities,
   residual path, and limited recurrent state where applicable.
5. Measure finite post-operator response rather than only a first-order or
   quadratic approximation.
6. Select on held-out block divergence or short teacher KL, subject to a
   memory/latency budget.
7. Emit an ordinary `compressed-tensors` checkpoint with no runtime transform.

Candidate response measurements include:

- normalized block-output error;
- teacher-student KL over next-token logits;
- attention-distribution divergence;
- gate-output and recurrent-state divergence;
- residual-relative error, rather than raw GEMM-relative error;
- sensitivity at multiple perturbation magnitudes to detect nonlinear response.

### Where this objective can influence the checkpoint

A scalar output sensitivity may cancel when independently selecting a scale for
one weight row. Finite response is therefore expected to be most useful for:

- shared global scales across a fused runtime group;
- format or precision allocation by operation;
- coupled rounding or transformations that span rows;
- selecting FP8/BF16 exceptions under a strict byte budget;
- nonlinear or stateful operations where local error is misleading.

It is not assumed to improve every independent 32-value microblock scale.

### Novelty boundary

The individual components overlap with existing work:

- KronQ models output sensitivity;
- REAL-Q pursues a more global objective;
- Minima analyzes nonlinear and recurrent error in Qwen;
- observer fusion captures known runtime grouping;
- MaCa and target-aware DPQ improve calibration composition.

The research question is whether their underexplored intersection can be made
general: finite nonlinear/state response, exact runtime grouping, robust
calibration, bounded streaming, and a standard output checkpoint. This requires
a more formal literature and patent review before using words such as "first"
or "novel" in public material.

## First bounded experiment

### Question

Does finite fused-operator response rank legal quantization candidates more like
held-out teacher KL than the current H64 objective or a diagonal-Hessian control?

### Scope

- One existing model: Qwen3.8-27B.
- 12-20 representative projections or fused groups.
- Early, middle, and late layers.
- Attention Q/O, MLP gate/up/down, and GDN QKV/Z/A/B/out coverage.
- 8-16 real calibration sequences.
- A disjoint 8-16-sequence held-out set.
- Multiple short and medium sequence lengths if the replay implementation allows.
- Three to five legal candidates per selected operation.
- No full checkpoint rebuild during the probe.

### Controls

Rank the same candidates with:

1. current H64 scale objective;
2. diagonal-Hessian objective;
3. full block-Hessian objective where stable;
4. finite fused-operator response.

Evaluate candidate rankings against:

1. held-out post-block error;
2. short teacher KL on disjoint contexts;
3. stability across calibration resamples and sequence lengths.

### Pre-registered promotion gate

Proceed to a full-model experiment only if all conditions hold:

- finite response improves Spearman rank correlation with held-out teacher KL by
  at least 0.20 over the strongest control;
- it selects a candidate with lower held-out KL on at least 70% of usable probes;
- the direction reproduces on two disjoint calibration/held-out splits;
- peak process memory remains bounded and no whole-model materialization occurs;
- the selected candidates remain legal in the intended vLLM format and kernel.

If the probe misses the gate, record it and stop. Do not weaken the gate after
observing results or build a complete 27B checkpoint to search for a signal.

### Estimated budget

- Implementation and local unit tests: 2-4 development days.
- Spark probe and one replication: 1-3 machine days, depending on replay cost.
- Full Qwen candidate after passing the gate: approximately 1-2 additional weeks.
- Multi-family, publication-quality validation: approximately 6-10 weeks.

These are planning estimates, not delivery commitments.

## Companion hypothesis: robust calibration portfolio

A single corpus and sequence length may overfit a quantizer to one activation
distribution. Maintain small, identified calibration shards for:

- general web or books;
- code;
- mathematics and structured reasoning;
- instruction/chat;
- short, medium, and long contexts.

Rather than minimizing the mean error across all samples, consider a robust
objective such as the worst normalized domain loss, a high percentile, or a
constrained Pareto score. Add hard examples where BF16 and the current quantized
model disagree, but preserve fixed anchor examples so the calibration set does
not collapse around one failure mode.

This is more likely to improve reliability than raw file size or throughput.
Its value should be measured by reduced variance and fewer domain regressions,
not only by aggregate perplexity.

## Current state after the bounded experiment cycle

Small all-MXFP4 scale, rounding, local-proxy, and layout variations are closed
for this checkpoint. DWC and coupled rounding reproduced the same central
failure: a plausible local objective moved, but held-out final behavior did not
move consistently. Exact-symmetry layout, counteraction, operator-response, and
suffix-JVP are also rejected under their frozen scopes.

The remaining work is no longer another inexpensive H64 variation:

1. **Qualification infrastructure.** Use one fail-closed scorecard for
   structure, PPL, multi-position distribution evidence, tasks, stock-vLLM
   load, memory, TTFT, decode, concurrency, long context, and MTP.
2. **Generality.** Complete one second dense architecture with a real streaming
   calibration adapter before describing MxWave as a general engine.
3. **Blackwell representation choice is evidence-gated.** The first existing
   native NVFP4 W4A4 checkpoint was evaluated on the same harness and rejected:
   it was worse than H64 in PPL and KL and larger after capability normalization.
   Do not implement an NVFP4 backend without a different candidate that first
   clears the same bounded external-checkpoint gate.
4. **Recovery at materially larger scale.** The large recovery run proves the
   mechanism can improve PPL, but another run is justified only with more
   diverse/longer data and a frozen goal expressed as BF16-gap recovery, not a
   nominal 5% PPL improvement.
5. **Passthrough compression.** Embeddings, `lm_head`, vision, and MTP are the
   remaining byte reservoir. Naive FP8 `lm_head` is closed; any retry needs a
   head-specific recovered or frequency-aware objective and capability-specific
   evaluation.
6. **MLX only as a product decision.** A true Apple backend requires
   BF16-to-MxWave quantization, MLX-specific sensitivity, and native baselines.
   Further tensor splicing does not answer that question.

Robust multi-domain and multi-length calibration remains a defensible
reliability experiment, but it is unlikely to create a material size or speed
gain. A custom adaptive format stays blocked until a numerical prototype shows
a large enough Pareto improvement to justify a runtime project.

## Generality requirements

A Qwen-only positive result establishes feasibility, not a general method. A
stronger claim requires at least:

1. one dense decoder-only transformer;
2. one hybrid recurrent/attention architecture such as Qwen3.8;
3. one MoE model if expert routing is claimed to be supported.

Architecture adapters may describe fused groups and operator boundaries, but
the scoring algorithm, calibration contract, streaming engine, and checkpoint
writer should remain shared. Model-name conditionals are evidence of missing
abstraction and must be documented explicitly.

## Whole-model evaluation requirements

No candidate should be described as better using perplexity alone. A complete
decision should include:

- paired perplexity on identical tokens;
- exact next-token KL or another teacher-distribution divergence;
- at least one reasoning task and one knowledge task;
- long-context degradation at multiple lengths;
- generation sanity and repetition checks;
- artifact size and precision coverage;
- cold-load time, time to first token, single-stream decode, and concurrent
  throughput;
- deterministic input hashes, artifact hashes, and paired uncertainty where
  applicable.

Task accuracy should be interpreted with seed and sample-size uncertainty.
Promotion should be based on a predefined scorecard rather than selecting the
metric that improved after the run.

## Research tracker

| ID | Work item | State | Evidence required for next state |
|---|---|---|---|
| R0 | Preserve H64 as frozen baseline | Complete | Existing reproducibility record |
| R1 | Literature/prior-art map | In progress | Review and categorize primary sources |
| R2 | Define runtime fused-group IR | In progress | Add a second dense-model adapter; Qwen header validation covers 128 operations and 496 weights |
| R3 | Finite operator-response scorer | Complete | Generic scoring, real MLP replay, diagonal/full-Hessian controls, and bounded suffix teacher KL implemented |
| R4 | Bounded Qwen local-NMSE probe | Rejected | 2/6 teacher-winner agreement; 70% gate became unreachable |
| R5 | Robust calibration portfolio | In progress | Two hashed disjoint splits measured; add positions/domains before selection |
| R6 | Full Qwen candidate from local NMSE | Rejected | Do not build from a selector that missed R4 |
| R6a | Layer-0 teacher-confirmed Qwen candidate | Rejected | Whole-model forward KL was 1.16% worse and top-1 agreement fell by 2/128 despite the isolated teacher-KL win |
| R6b | Quantized-baseline layer-0 replay gate | Rejected as selector; infrastructure retained | Two fresh splits chose opposite winners; pooled delta was -7.06e-6 with 8/16 wins and interval crossing zero |
| R6c | Small rank-16 fake-MXFP4 trajectory recovery | Rejected; infrastructure retained | Three-way retry selected step 16 / trust 0.5 and generalized positively, but hidden-gate recovery was 6.44% versus the fixed 10% requirement; final KL correctly skipped |
| R6d | Residual-counteraction selector | Rejected; infrastructure retained | Mean Spearman -0.0167, 4/12 cross-split improvements, and 3/6 stable selections |
| R6e | Complete-suffix forward-AD selector | Rejected; infrastructure retained | Mean Spearman 0.3333, below weight NMSE at 0.4000; 4/6 cross-split improvements and 1/3 stable selections |
| R6f | Layer-62 unweighted-MSE confirmation | Rejected | Fresh split was 0.91% worse than H64, won 9/16 contexts, and its paired interval crossed zero |
| R7 | Dense-model replication | Not started | Add and fully qualify one non-Qwen streaming-calibration adapter |
| R8 | MoE replication | Deferred | Only needed for an MoE support claim |
| R9 | Characterize standard-format failures | Blocked on R4 | Stable failure shape across splits and models |
| R10 | Adaptive FP4 numerical prototype | Blocked on R9 | Pareto gain over standard mixed precision |
| R11 | Adaptive FP4 kernel feasibility | Blocked on R10 | Projected end-to-end size/latency gain |
| R12 | Publication decision | Out of scope | Owner decision after complete evaluation |
| R13 | Exact-symmetry MXFP4 block layout | Rejected for Qwen3.8-27B; research branch retained | Both families passed 0/3 layers across two frozen splits; every pooled interval crossed zero |
| R14 | Sequence-level disagreement-weighted calibration | Rejected; research branch retained | One of four layers passed both splits and the pooled paired interval crossed zero |
| R15 | Coupled gate/up rounding | Rejected; research branch retained | Product NMSE improved locally, but zero of four layers improved both held-out splits |
| R16 | Large fake-MXFP4 recovery | Complete research result; local-only artifact | PPL improved 0.654% and recovered 37.92% of the BF16 gap; KL was mixed and GSM8K tied |
| R17 | FP8 `lm_head` compression | Rejected | Size fell 6.406%, but forward KL regressed 4.670% with a paired interval above zero |
| R18 | MLX affine mixed-bit allocation | Rejected as an MxWave model; local record only | Missed the 5% PPL gate, KL was split-unstable, and no MxWave quantization/export occurred |
| R19 | Unified qualification command | Complete locally; GPU integration pending | Packaged structural verifier, evidence hooks, capability inference, frozen gates, and atomic JSON/Markdown report |
| R20 | Existing native NVFP4 W4A4 checkpoint | Rejected before performance/tasks | Native SM121 execution passed, but PPL was 5.382% worse, mean forward KL was 45.1% worse, and normalized language-only size was about 4.02% larger than H64 |

Allowed states are `Not started`, `In progress`, `Blocked`, `Rejected`,
`Complete`, and `Deferred`. Every transition should link to a dated experiment
record or commit; do not store raw benchmark results only in this table.

## Decision log

Add one row whenever scope, promotion criteria, or the preferred research
direction changes.

| Date | Decision | Evidence | Consequence |
|---|---|---|---|
| 2026-09-18 | Prefer execution-aware finite response over another scale-only variant | Local cross-block result and current literature | Design a bounded correlation probe before any full rebuild |
| 2026-09-18 | Keep stock vLLM compatibility as a hard constraint for Track A and distributable artifacts | MxWave project goal | Defer custom formats and kernels until a measured standard-format limit |
| 2026-09-18 | Keep H64 as the frozen comparison baseline | Reproduced artifact and evaluations | New experiments must compare against it on paired inputs |
| 2026-09-18 | Pair execution-aware scoring with an evidence-gated adaptive FP4 track | Research review | Let measured failure shapes determine whether a new representation is warranted |
| 2026-09-18 | Start a format- and model-agnostic operator-response scorer on `research/operator-response` | Synthetic nonlinear, KL, reduction, and streaming tests | Define runtime fused-group replay without adding Qwen rules to the scoring core |
| 2026-09-18 | Add runtime-operation IR, a Qwen3.5 adapter, and non-mutating functional module replay | Synthetic gated-MLP replay and fused-group validation | Validate the adapter on the source checkpoint header, then replay selected real modules |
| 2026-09-18 | Validate the Qwen adapter against the Spark BF16 checkpoint header | 128 operations, 496 owned weights, 176 fused groups; complete text stack selected while ignoring MTP | Add a second architecture adapter and begin bounded real-module replay |
| 2026-09-18 | Reject local post-layer NMSE as the adaptive candidate selector | It matched exact suffix teacher-KL winners on 2/6 MLP layers; even perfect remaining probes could reach only 66.7% | Stop the replication, do not build a checkpoint, retain exact teacher KL as an oracle for a new proxy |
| 2026-09-18 | Do not use per-split teacher-KL argmin as a recipe | Only 1/6 winners reproduced on offset-64 and offset-72 splits; no alternative cleared the robust paired gate | Improve oracle sampling before proxy fitting or checkpoint construction |
| 2026-09-18 | Admit one narrow layer-0 diagonal-Hessian checkpoint experiment | Fixed offset-80 confirmation passed 12/16 wins and a positive paired interval; pooled three-split interval also positive | Keep block-Hessian elsewhere and require end-to-end H64 comparison before promotion |
| 2026-09-18 | Reject the layer-0 diagonal-Hessian recipe after the whole-model gate | Mean forward KL worsened 1.16%, all mean divergence metrics worsened, and BF16 top-1 agreement fell from 120/128 to 118/128 | Retain H64; do not run post-hoc perplexity or promote isolated suffix wins directly into recipes |
| 2026-09-18 | Stop the quantized-baseline selector after its bounded gate | The packed H64 control reconstructed exactly, but offsets 88 and 96 selected opposite winners and the pooled 95% interval crossed zero | Retain the streaming packed-checkpoint replay code; do not expand to a broad sweep or rebuild another model |
| 2026-09-18 | Reject the first fake-MXFP4 recovery configuration at its local gate | The H64 control reconstructed exactly and training NMSE improved 27.83%, but validation NMSE worsened 15.20% | Skip layer 62, final KL, and model rebuild; allow at most one pre-registered three-way-split trust-region retry |
| 2026-09-18 | Close the current fake-MXFP4 recovery track after its one admitted retry | Step 16 / trust 0.5 improved selection NMSE 4.64% and hidden-gate NMSE 6.44%, below the fixed 10% requirement | Preserve the generic recovery infrastructure on its research branch, retain H64, and do not sweep recovery hyperparameters or consume final splits |
| 2026-09-19 | Reject residual counteraction as a candidate selector | Numerically exact recurrence, but mean Spearman -0.0167 and only 4/12 cross-split improvements | Preserve the decomposition and packed-suffix replay; do not build a selected checkpoint |
| 2026-09-19 | Reject complete-suffix JVP as a general candidate selector | Native eager forward AD was valid and reached 4/6 cross-split improvements, but mean Spearman was 0.3333, weaker than weight NMSE, with 1/3 stable selections | Preserve forward-AD infrastructure on its research branch; do not narrow the claim to late layers post-hoc |
| 2026-09-19 | Prefer exact-symmetry block layout over another sensitivity proxy | Repeated proxy failures show that downstream-error estimation is not the current bottleneck; standard MXFP4 blocks remain layout-sensitive | If work continues, pre-register a one-day gated-MLP permutation probe before any checkpoint build |
| 2026-09-19 | Reject the layer-62 unweighted-MSE replacement on fresh confirmation | Mean exact KL was 0.91% worse than H64, with 9/16 wins and a confidence interval crossing zero | Do not build the isolated replacement or consume the immutable 128-context gate |
| 2026-09-19 | Close the exact-symmetry layout experiment for Qwen3.8-27B | Weight-norm and activation-weighted sorts each passed 0/3 layers; split-A layer-62 gains reversed on split B | Keep H64 unchanged, retain the bounded research branch, and do not merge the failed policy into main |
| 2026-09-20 | Reject sequence-level DWC | Only 1/4 layers improved on both splits; pooled interval `[-0.00034679,+0.00041532]` crossed zero | Do not build its control or checkpoint; retain the bounded record |
| 2026-09-20 | Reject coupled gate/up rounding | Legal candidates improved the calibration product objective, but every layer reversed direction across splits | Do not build a checkpoint or sweep optimizer settings |
| 2026-09-21 | Retain large recovery as a successful local mechanism, not a release | PPL improved 0.654% and recovered 37.92% of the BF16 gap, but exact distribution evidence was mixed and GSM8K tied | Keep H64 as the public default and preserve the recovery branch/results locally |
| 2026-09-21 | Reject naive FP8 `lm_head` compression | 6.406% size saving came with 4.670% worse forward KL and a paired interval above zero | Stop before PPL/tasks/MTP and do not escalate the same head to MXFP4 |
| 2026-09-21 | Reject the MLX affine mix as an MxWave artifact | 4.251% PPL improvement missed the 5% gate; KL was unstable; weights came from third-party affine checkpoints | Keep the result local and leave the true MxWave-to-MLX question unanswered |
| 2026-09-21 | Consolidate qualification behind one fail-closed command | Existing evidence was fragmented across structural, PPL, KL, serving, and platform reports | Use `mxwave-qualify`; missing runtime capabilities produce `incomplete`, never a silent pass |
| 2026-09-21 | Reject the tested Minima native-NVFP4 checkpoint at Phase 1 | Native W4A4 execution was verified, but paired PPL and KL both significantly favored H64 and the normalized artifact was larger | Stop before throughput, GSM8K, and long-context testing; do not open an MxWave NVFP4 backend from this evidence |

## Primary-source watchlist

Review these before changing the roadmap. Recent arXiv entries may not yet be
peer reviewed; cite their specific evidence rather than treating all claims as
established facts.

- [Benchmarking Microscaling Formats for LLM Quantization](https://arxiv.org/abs/2601.09555)
- [H-Scale](https://arxiv.org/abs/2608.28113)
- [SOAR](https://arxiv.org/abs/2605.12245)
- [FAAR](https://arxiv.org/abs/2603.22370)
- [DASH-Q](https://arxiv.org/abs/2604.13806)
- [MaCa](https://arxiv.org/abs/2602.07465)
- [Target-aware DPQ](https://arxiv.org/abs/2608.21019)
- [Block Rotation](https://arxiv.org/abs/2511.04214)
- [TORQ](https://arxiv.org/abs/2605.19561)
- [OAS/MBS for MXFP4](https://arxiv.org/abs/2603.08713)
- [MixQuant](https://arxiv.org/abs/2601.22347)
- [dMX](https://arxiv.org/abs/2606.04115)
- [AdaMX](https://arxiv.org/abs/2608.03867)
- [REAL-Q](https://arxiv.org/abs/2609.00049)
- [KronQ](https://arxiv.org/abs/2607.07964)
- [ProjQ](https://arxiv.org/abs/2606.00494)
- [QERA](https://arxiv.org/abs/2410.06040)
- [Minima Qwen3.8-27B NVFP4](https://arxiv.org/abs/2609.04098)
- [NVIDIA NVFP4 QAD report](https://research.nvidia.com/labs/nemotron/files/NVFP4-QAD-Report.pdf)
- [LLM Compressor NVFP4 guide](https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/quantization_w4a4_fp4/)
- [LLM Compressor observer fusion](https://docs.vllm.ai/projects/llm-compressor/en/stable/guides/observers/)
- [`compressed-tensors`](https://github.com/vllm-project/compressed-tensors)

## Explicit non-goals for the next experiment

- Do not create a proprietary FP4 format.
- Do not require a custom vLLM kernel.
- Do not rebuild the complete 27B model before the bounded gate passes.
- Do not claim that lower local MSE implies better model quality.
- Do not claim generality from Qwen alone.
- Do not optimize against the final evaluation set.
- Do not publish or upload an experimental checkpoint automatically.
