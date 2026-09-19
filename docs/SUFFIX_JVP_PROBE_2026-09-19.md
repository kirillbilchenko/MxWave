# Suffix-JVP candidate probe — pre-registration (2026-09-19)

Status: implementation validated on a synthetic model; no Qwen3.8-27B results inspected yet.

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

```bash
mxwave-suffix-jvp-evaluate split-a.json split-b.json --output evaluation.json
```

The evaluator returns zero only when every frozen gate passes, two for a valid failed experiment,
and one for an invalid report.
