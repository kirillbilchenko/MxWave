# Suffix-JVP candidate probe — pre-registration (2026-09-19)

Status: complete — the general selector failed the frozen gates and no checkpoint was built. The
scope and gates below were committed before either registered quality result was inspected.

## Decision

Reject suffix-JVP teacher KL as a general MXFP4 candidate selector for the current Qwen3.8-27B
H64 recipe. Keep the implementation as research infrastructure on
`research/suffix-jvp-probe`; do not merge it as a production method or build a checkpoint from its
post-hoc choices.

The implementation itself was correct. Native forward AD crossed the complete 64-layer eager
suffix, the block-Hessian control reproduced H64 exactly, and the maximum zero-tangent teacher-KL
error was `0.0`. The predictive signal was materially better than the preceding one-block
counteraction score, but it was not reliable enough:

| Frozen gate | Required | Observed | Result |
|---|---:|---:|---|
| Mean Spearman rho, predicted vs exact teacher KL | >= 0.50 | 0.3333 | fail |
| Advantage over strongest local control | >= 0.20 | -0.0667 | fail |
| Cross-split KL improvements | >= 70% | 4/6 = 66.7% | fail |
| Same selection on both splits | >= 2/3 layers | 1/3 | fail |
| H64 reconstruction NMSE | <= 1e-12 | 0.0 | pass |
| Recurrence relative residual | <= 1e-10 | 3.24e-16 | pass |
| Zero-tangent teacher-KL error | <= 1e-8 | 0.0 | pass |

Simple weight NMSE had mean Spearman `0.4000`, slightly above suffix JVP. The earlier local
counteraction score was `-0.0167`, so differentiating the complete suffix recovered a real signal,
especially late in the network, but not a deployable selector.

| Layer | Offset 104: selected / exact winner / rho | Offset 112: selected / exact winner / rho |
|---:|---|---|
| 16 | RTN / block Hessian / 0.0 | block Hessian / RTN / -0.2 |
| 42 | unweighted MSE / RTN / 0.0 | unweighted MSE / unweighted MSE / 0.8 |
| 62 | unweighted MSE / unweighted MSE / 0.8 | RTN / unweighted MSE / 0.6 |

Layer 62 is an exploratory signal, not a recipe change: both split-selected alternatives improved
over H64 when evaluated on the other split, but the selected candidate differed. Restricting the
claim to late layers after seeing the result would be post-hoc and is not allowed by this record.

The two registered splits completed in `388.66 s` and `388.72 s`. Both peaked at
`8,041,931,264` accelerator bytes and about `10.93 GB` process RSS. The compact machine-readable
record is
[`benchmarks/qwen3.8-27b-suffix-jvp-probe.json`](../benchmarks/qwen3.8-27b-suffix-jvp-probe.json).

## Question

Can an inexpensive first-order model of the *remaining network* select a better legal MXFP4
candidate than local reconstruction metrics?

The existing counteraction probe measures the exact candidate perturbation at a selected MLP,
but its local hidden-error score was not predictive of final teacher KL on Qwen3.8-27B. This
probe keeps that exact perturbation and propagates its directional derivative through every
remaining packed decoder layer, the final normalization, and the language-model head.

For packed baseline trajectory `h` and candidate perturbation `v` at layer `k`:

```text
v[k]       = candidate_output - packed_baseline_output
v[l + 1]   = J(f[l], h[l]) v[l]
z[pred]    = z[packed] + J(head o norm, h[last]) v[last]
score      = KL(BF16 teacher || softmax(z[pred]))
```

The exact nonlinear candidate suffix is still run, but only as held-out ground truth. It is not
used by the selector. The method does not modify checkpoint bytes, model size, or inference.

## Scope

- Source: `Qwen/Qwen3.8-27B`, revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Packed baseline: the reproducible MxWave H64 checkpoint.
- Calibration: the existing `qwen3.8-27b-64x512.safetensors` block-Hessian artifact.
- Candidates: RTN, unweighted MSE, diagonal Hessian, and block Hessian.
- Primary metric: `mean_suffix_jvp_teacher_kl`.
- Ground truth: exact `mean_teacher_kl` after the full nonlinear packed suffix.
- Controls: weight NMSE, operator NMSE, update-error NMSE, and resulting-hidden NMSE.
- Stage-one layers: 16, 42, and 62.
- Split A: 8 sequences at corpus offset 104, length 512.
- Split B: 8 sequences at corpus offset 112, length 512.
- Final scoring: last 8 next-token positions per sequence.

The calculus is architecture-neutral. The current streaming runtime adapter is Qwen3.8-specific;
support for another model family requires only an adapter that exposes its sequential decoder
trajectory and output path.

## Compatibility gate

Before the quality run, execute one sequence of 32 tokens with only the block-Hessian candidate
through a complete suffix. The gate passes only if native forward-mode AD reaches the output head,
all tangents are finite, and the zero-perturbation candidate reproduces baseline teacher KL within
`1e-8`. An unsupported operator aborts the experiment. There is no finite-difference fallback.

Compatibility output is not a quality measurement and must not be used to change the frozen gates.

Compatibility outcome, recorded before either quality split: the SDPA configuration reached a
full-attention layer whose Flash-SDPA kernel has no PyTorch forward-AD rule and aborted. The
explicit eager-attention implementation then traversed all 64 layers and the head with native
forward AD. Its baseline reconstruction, recurrence residual, exact-candidate KL error, and
zero-tangent JVP KL error were all `0.0`. Both quality splits are therefore frozen to eager
attention; mixing implementations invalidates the experiment.

## Frozen quality gates

The stage-one result passes only if all gates pass across the two reports:

1. Both reports are complete, use `mxwave-suffix-jvp-probe-v1`, and use native `forward-ad`.
2. Artifact identities match and the two token hashes differ.
3. Both reports contain exactly the same three registered layers and four candidates.
4. The block-Hessian candidate reconstructs the packed baseline with weight NMSE at most `1e-12`.
5. Counteraction recurrence relative residual is at most `1e-10`.
6. Exact and JVP zero-tangent block-Hessian KL differ from baseline KL by at most `1e-8`.
7. Mean per-layer Spearman correlation between predicted and exact candidate KL is at least `0.50`.
8. That mean correlation exceeds the strongest local control by at least `0.20`.
9. A candidate selected on one split improves exact KL over block Hessian on the other split for at
   least 5 of 6 decisions (`>= 0.70`). Ties select block Hessian.
10. The selected candidate agrees between splits on at least 2 of 3 layers.

Passing stage one justifies a six-layer confirmation. It does not justify building or publishing a
checkpoint by itself. Failure rejects suffix-JVP candidate selection for this model/configuration;
it does not prove the idea cannot work on another architecture.

## Resource boundary

- Run inside a detached Spark container so SSH loss cannot terminate it.
- Preserve an atomic JSON report after every completed layer.
- Stop before another layer when the CLI boundary reaches 60 minutes.
- Wrap each split in an external 75-minute timeout so a single unsupported/hung layer cannot run
  indefinitely.
- Restore the daily vLLM service after success, failure, or timeout.

## Evaluation command

This historical command is available on the retained
`research/suffix-jvp-probe` branch, not in the production package.

```bash
mxwave-suffix-jvp-evaluate split-a.json split-b.json --output evaluation.json
```

The evaluator returns zero only when every frozen gate passes, two for a valid failed experiment,
and one for an invalid report.

## Interpretation and retained work

The failure is statistical rather than mechanical. A directional derivative is useful when the
candidate perturbation stays in a locally linear regime, which was more often true near the output.
Earlier candidates cross many gates, attention operations, residual additions, and recurrent GDN
updates; candidate rankings then depend on context and higher-order effects that one tangent cannot
represent reliably.

Retain:

- the device-agnostic `module_tensor_jvp` primitive and its nonlinear finite-difference tests;
- optional suffix-JVP fields in the counteraction report;
- the frozen two-split evaluator and compatibility failure diagnostics;
- the finding that native SDPA Flash Attention lacks a forward-AD rule in the tested PyTorch image,
  while eager attention works with bounded memory.

Do not infer:

- that H64 should change at layers 16, 42, or 62;
- that a late-layer-only rule has passed validation;
- that this Qwen result establishes behavior on another architecture;
- that a second-order or finite-difference proxy is automatically worthwhile. Exact suffix KL was
  already observed to be unstable on small splits in the earlier operator-response work.

Any next quantization experiment should address the demonstrated calibration/generalization
problem rather than add another local error proxy. The most defensible remaining bounded direction
is robust multi-length, multi-partition calibration with a predeclared end-to-end gate; rotation or
format changes are larger, separate tracks.
