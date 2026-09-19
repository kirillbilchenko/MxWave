# Counteraction-preserving MXFP4 probe — 2026-09-19

Status: complete — the selector failed the frozen gates and no checkpoint was
built. The scope and gates below were committed before either registered result
was inspected.

## Decision

Reject `mean_resulting_hidden_nmse` as an MXFP4 scale-candidate selector for the
current Qwen3.8-27B H64 recipe.

The implementation was numerically correct: the block-Hessian control
reconstructed every selected H64 MLP exactly (`0.0` weight NMSE), and the
largest residual-recurrence error was `3.98e-16`. The selection signal did not
generalize to final next-token distributions:

| Frozen gate | Required | Observed | Result |
|---|---:|---:|---|
| Mean Spearman ρ, primary vs teacher KL | ≥ 0.50 | -0.0167 | fail |
| Primary advantage over strongest control | ≥ 0.20 | -0.0333 | fail |
| Cross-split KL improvements | ≥ 70% | 4/12 = 33.3% | fail |
| Same selection on both splits | ≥ 4/6 layers | 3/6 | fail |
| H64 reconstruction NMSE | ≤ 1e-12 | 0.0 | pass |
| Recurrence relative residual | ≤ 1e-10 | 3.98e-16 | pass |

The signed interaction is present and measurable, including negative
counteraction in early and middle blocks. But minimizing the error immediately
after one block does not predict the effect of the entire remaining quantized
suffix. Later blocks can counteract or amplify the candidate difference again.
This rejects the proposed one-block selector, not the residual-counteraction
mechanism itself.

| Layer | Offset 104: selected / KL winner / ρ | Offset 112: selected / KL winner / ρ |
|---:|---|---|
| 8 | RTN / H64 / -0.4 | RTN / RTN / 0.4 |
| 16 | RTN / diagonal / 0.2 | RTN / unweighted / -0.2 |
| 31 | H64 / H64 / 1.0 | diagonal / diagonal / 0.8 |
| 42 | H64 / unweighted / -0.6 | unweighted / RTN / -1.0 |
| 54 | H64 / diagonal / 0.4 | diagonal / diagonal / 0.8 |
| 62 | H64 / RTN / -0.8 | H64 / RTN / -0.8 |

The two registered splits completed in `562.30 s` and `560.19 s`. Peak
accelerator allocation was `5,543,043,072` bytes and peak process RSS was about
`10.55 GB`; the bounded streaming implementation therefore met its resource
goal. The compact machine-readable record is
[`benchmarks/qwen3.8-27b-counteraction-probe.json`](../benchmarks/qwen3.8-27b-counteraction-probe.json).

## Question

Can a legal MXFP4 scale candidate be chosen more reliably by measuring how its
block update interacts with the **actual inherited error from an MXFP4 prefix**,
rather than by minimizing isolated weight or operator error?

For a BF16 hidden state `h`, packed-prefix state `h_hat`, BF16 block output `y`,
and candidate output `y_hat` evaluated on `h_hat`:

```text
e_inherited = h_hat - h
e_update    = (y_hat - h_hat) - (y - h)
e_result    = y_hat - y = e_inherited + e_update

||e_result||² = ||e_inherited||² + ||e_update||²
               + 2 <e_inherited, e_update>
```

A negative interaction term is counteraction. A positive term amplifies the
inherited error. The proposed selector minimizes mean resulting hidden NMSE;
the signed interaction is retained to explain the selection.

The mechanism is motivated by *Why Does Post-Training Quantization Work?*
(arXiv:2609.11716), which measures residual-block counteraction in pretrained
models. That paper analyzes the mechanism; this probe tests a new MxWave
candidate-selection use of it. This document does not claim unique prior art.

## Fixed scope

- Source: Qwen3.8-27B BF16 checkpoint already used by the MxWave experiments.
- Execution baseline: the published H64/block-Hessian MxWave MXFP4 checkpoint.
- Calibration: the exact block-Hessian artifact used to build H64.
- Operation: the gated MLP inside each selected decoder block.
- Layers: `8,16,31,42,54,62`; layer 0 is intentionally excluded because it has
  no inherited decoder error.
- Candidates, in fixed order:
  1. RTN
  2. unweighted three-candidate MSE
  3. diagonal-Hessian three-candidate MSE
  4. block-Hessian three-candidate MSE (the H64 reconstruction control)
- Candidate format: ordinary OCP MXFP4, block size 32. No new runtime format,
  metadata, model size, or kernel is introduced.
- Data: two disjoint deterministic Pileval splits, each `8 × 512` tokens, at
  sequence offsets `104` and `112`.
- Final distribution check: exact teacher KL over the last 8 token positions of
  every sequence.
- Resource boundary: 90 minutes per split. The CLI writes an atomic report
  after every completed layer and stops before starting another layer once the
  boundary has elapsed. An external 100-minute process timeout is also used on
  Spark.
- No full 27B candidate checkpoint is built during this probe.

## Recorded controls

Every candidate records:

- BF16 weight NMSE;
- weight NMSE against the packed H64 reconstruction;
- isolated output NMSE against the unchanged H64 block;
- update-error NMSE without the signed inherited-error interaction;
- inherited hidden NMSE, signed interaction NMSE, resulting hidden NMSE, and
  error-growth NMSE;
- counteraction fraction and recurrence residual;
- final next-token teacher KL after the unchanged H64 suffix.

The H64/block-Hessian candidate must reconstruct the selected packed weights
with NMSE at most `1e-12`. The residual recurrence relative residual must be at
most `1e-10`. Either violation invalidates the run.

## Frozen evaluation

The primary metric is `mean_resulting_hidden_nmse`. The controls are
`weight_nmse`, `mean_operator_nmse`, and `mean_update_error_nmse`.

For each layer and split, candidate ranks under each local metric are compared
with final teacher-KL ranks using Spearman correlation. Ties receive average
ranks. Candidate-name order is only a deterministic final tie-break.

Selection is evaluated out of split:

- split A's primary metric selects one candidate per layer and that candidate's
  KL is evaluated on split B;
- split B selects and split A evaluates in the same way;
- if the primary metric ties within `rtol=1e-9, atol=1e-12`, H64/block-Hessian
  wins the tie.

The experiment passes only if all conditions hold:

1. both reports complete all six layers and use identical candidates, model,
   calibration, and distinct token hashes;
2. all H64 reconstruction NMSE values are at most `1e-12`;
3. all recurrence relative residuals are at most `1e-10`;
4. mean primary-metric Spearman correlation with teacher KL is at least `0.50`;
5. that correlation exceeds the strongest control correlation by at least
   `0.20`;
6. cross-split selections reduce teacher KL relative to H64 by more than
   `1e-12` in at least 70% of the 12 layer/direction decisions;
7. the selected candidate is identical between splits on at least four of six
   layers.

Failure of any gate means no full checkpoint build and no claim of improvement.
An inconclusive bounded stop is also a failure for this iteration, not a pass.

## Commands

The installed CLI is `mxwave-counteraction-probe`. The two commands differ only
in `--sequence-offset` and output path. Exact Spark paths, image digest, commit,
and resulting command lines are recorded alongside the machine-readable
reports before execution.

## Interpretation boundary

A pass demonstrates a useful selection signal on one Qwen-family checkpoint;
it does not establish architecture generality. The decomposition, MXFP4
candidates, and report are architecture-independent, while the first bounded
streaming execution backend is the existing Qwen3.5 text adapter. At least one
second architecture must pass a later confirmation before this can become a
default model-independent MxWave strategy.
