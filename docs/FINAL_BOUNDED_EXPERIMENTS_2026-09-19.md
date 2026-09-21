# Final bounded MXFP4 experiments — pre-registration and results (2026-09-19)

Status: complete. Both hypotheses were rejected by their frozen gates; no checkpoint was built.

These are the final two experiments in the current Qwen3.8-27B/H64 research cycle. They have hard
stopping rules and do not authorize additional candidate, layer, threshold, or data sweeps.

## Frozen artifacts

| Item | Identity |
|---|---|
| Source | `Qwen/Qwen3.8-27B` |
| Source revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Source config SHA-256 | `191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab` |
| Source index SHA-256 | `77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df` |
| H64 config SHA-256 | `d19a06baa5074b9f1f7b38296d846852fa2c071d2cebdd833e409206fc3feff0` |
| H64 index SHA-256 | `be4a2cd4e130058b23080c6f71172d1e0e37d703e27040deb60fe7f431a7331b` |
| Block-Hessian artifact SHA-256 | `4f9e726a77d2f9f9e304ef18fafc8ef103e12caccf00d3c9950369809824483f` |
| Corpus SHA-256 | `07ffd944ea4fc09f12e5d019633a77309199fdced749e3ddf13989e53586f36a` |
| Runtime | `vllm/vllm-openai:v0.29.0`, BF16 replay |

## Experiment A: layer-62 unweighted confirmation

### Hypothesis

Replacing only the three layer-62 MLP H64/block-Hessian weights with ordinary unweighted-MSE
MXFP4 improves exact final teacher KL on fresh data. The hypothesis comes from the completed
suffix-JVP experiment, where unweighted MSE was the exact teacher-KL winner on both observed
splits. The JVP selector itself remains rejected and is not used here.

### Frozen scope

- Layer: `62`, gated MLP only.
- Candidates: `unweighted-mse` and `block-hessian`.
- Candidate settings: percentile `99.5`, MSE clip depth `4`.
- Prefix and suffix: unchanged packed H64 execution trajectory.
- Attention: eager, matching the discovery run.
- Fresh split: offset `120`, `16×512` tokens.
- Token-ID SHA-256: `49ac6fcdd0f473d37d5258cd387c8e7a60d3ef5934810f66d901d2e9ed3aee81`.
- Final score: exact BF16-teacher KL over the last 8 token positions per sequence.
- Paired bootstrap: 100,000 resamples, seed `20260919`.

### Pass gate

All conditions must pass:

1. H64 reconstruction weight NMSE is at most `1e-12`.
2. Counteraction recurrence residual is at most `1e-10`.
3. The exact block-Hessian candidate and recorded baseline KL agree within `1e-8` per sample.
4. Unweighted MSE has lower mean teacher KL than block Hessian.
5. Unweighted MSE wins at least 12 of 16 paired contexts.
6. The 95% paired-bootstrap interval for `block_hessian_KL - unweighted_KL` has a strictly
   positive lower bound.

Failure ends Experiment A without a checkpoint. Passing permits one layer-62-only checkpoint
build followed by the already frozen 128-context full-vocabulary divergence gate. That checkpoint
is retained only if it lowers mean forward KL, has a strictly positive paired-bootstrap lower bound
for `H64 - candidate`, and does not reduce BF16 top-1 agreement.

## Experiment B: exact-symmetry MXFP4 block layout

### Hypothesis

A function-preserving permutation of gated-MLP intermediate channels can regroup the input
columns of the down projection into more favorable 32-value MXFP4 blocks without metadata,
runtime transforms, additional bytes, or custom kernels.

For one permutation `P`:

```text
W_gate' = P W_gate
W_up'   = P W_up
W_down' = W_down P^T
```

The gate/up packed values are the H64 values with rows reordered. Only the down projection is
re-quantized with unweighted MSE after its input columns are reordered. This isolates the layout
effect with an identity-unweighted-down control.

### Frozen candidates

1. `block-hessian`: unchanged H64 reconstruction.
2. `identity-unweighted-down`: H64 gate/up plus unweighted-MSE down projection, no permutation.
3. `weight-norm-sort`: stable ascending sort of BF16 down-projection column L2 norms.
4. `activation-weighted-sort`: stable ascending sort of column L2 norm multiplied by the square
   root of the down-input block-Hessian diagonal.

Permutation candidates use the same fixed percentile `99.5` and MSE clip depth `4`. Candidate
construction sees source weights and the registered calibration artifact only, never held-out KL.

### Frozen scope

- Layers: `16`, `42`, and `62` gated MLPs.
- Split A: offset `136`, `8×512`, token hash
  `d3873dbab129e1131e77beb000ee9ad6b021fe41b0746541f5687a3a347a08b9`.
- Split B: offset `144`, `8×512`, token hash
  `08715cfe3c0e33cb62cb017332f7f511d6b3779661f23bd535bed6735200ebb4`.
- Attention: SDPA; both reports must use the same implementation and BF16 dtype.
- Exact final BF16-teacher KL over the last 8 positions per sequence.
- Paired bootstrap: 100,000 stratified resamples, seed `20260919`.

Before quality execution, unit and real-model smoke tests must establish that every permutation is
bijective, inverse weight layout is exact, unquantized gated-MLP output is numerically equivalent,
and the unchanged H64 candidate reproduces the packed baseline.

### Pass gate

One fixed permutation family must pass on at least two of three layers. A family passes a layer only
when all conditions hold:

1. Its mean exact teacher KL is lower than both `block-hessian` and
   `identity-unweighted-down` on each split.
2. It wins at least 5 of 8 paired contexts against H64 on each split.
3. The stratified pooled 95% paired-bootstrap lower bounds for both
   `H64 - permutation` and `identity_unweighted - permutation` are strictly positive.

Global validity also requires exact registered identities, disjoint token hashes, H64 reconstruction
NMSE at most `1e-12`, and recurrence residual at most `1e-10`. Ties fail. If both families pass the
same number of layers, `weight-norm-sort` is the deterministic tie-break.

Passing permits one checkpoint containing only the winning family on the layers that passed. It
must then pass the same immutable 128-context whole-model divergence gate as Experiment A.
Failure ends the exact-layout track for this model; no new sort key or layer sweep is allowed.

## Resource and operational boundary

- Every GPU job runs detached on Spark with networking disabled.
- Experiment A: 20-minute external timeout.
- Experiment B: 45-minute timeout per split and atomic output after each layer.
- No full checkpoint is built before its bounded gate passes.
- Daily vLLM is restored after success, failure, or timeout.
- Publication/upload remains an owner decision and is outside this run.

## Results

### Experiment A — rejected

The fresh offset-120 confirmation completed in 94.51 seconds. All implementation and numerical
validity checks passed, but unweighted MSE did not reproduce its discovery-split improvement.

| Metric | H64 block Hessian | Layer-62 unweighted MSE |
|---|---:|---:|
| Mean exact teacher KL | 0.0326247318 | 0.0329225997 |
| Relative change versus H64 | — | **0.913% worse** |
| Paired context wins | — | 9/16 (required 12/16) |

The paired 95% bootstrap interval for `H64 - unweighted` was
`[-0.0011714758, 0.0003834154]`, crossing zero. H64 reconstruction NMSE was exactly zero, its
recorded and replayed sample KL values agreed exactly, and the maximum recurrence residual was
`2.40e-16`. This is a quality/generalization rejection, not an execution failure. Per the frozen
rule, no layer-62 checkpoint or 128-context follow-up was produced.

### Experiment B — rejected

Each split completed all three layers in about 265 seconds. Permutations were deterministic across
splits, bijective, exactly invertible, and numerically function-preserving before quantization
(real-model smoke relative error at most `1.82e-7`). H64 reconstruction NMSE was zero and the
maximum recurrence residual was `2.51e-16`.

Neither family passed any layer on both splits:

| Family | Layer | Split A candidate / H64 / identity KL | A wins | Split B candidate / H64 / identity KL | B wins | Pooled 95% CI, H64 − candidate | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| Weight-norm sort | 16 | 0.0443447 / 0.0442378 / 0.0443944 | 3/8 | 0.0187809 / 0.0183108 / 0.0185369 | 4/8 | [-0.0007576, 0.0001520] | No |
| Weight-norm sort | 42 | 0.0437374 / 0.0442378 / 0.0431219 | 4/8 | 0.0183948 / 0.0183108 / 0.0183576 | 3/8 | [-0.0003664, 0.0009009] | No |
| Weight-norm sort | 62 | 0.0435621 / 0.0442378 / 0.0439944 | 6/8 | 0.0186884 / 0.0183108 / 0.0185285 | 2/8 | [-0.0004873, 0.0009622] | No |
| Activation-weighted sort | 16 | 0.0441746 / 0.0442378 / 0.0443944 | 4/8 | 0.0185793 / 0.0183108 / 0.0185369 | 2/8 | [-0.0008174, 0.0008190] | No |
| Activation-weighted sort | 42 | 0.0436918 / 0.0442378 / 0.0431219 | 5/8 | 0.0183908 / 0.0183108 / 0.0183576 | 5/8 | [-0.0005482, 0.0010450] | No |
| Activation-weighted sort | 62 | 0.0436065 / 0.0442378 / 0.0439944 | 6/8 | 0.0185798 / 0.0183108 / 0.0185285 | 1/8 | [-0.0003672, 0.0007841] | No |

The useful signal was not stable: both sorts improved layer 62 on split A, then clearly regressed on
split B. Every pooled confidence interval crossed zero, and no family reached the required two of
three layers. No exact-layout checkpoint or whole-model follow-up was produced.

## Reproducibility and retained artifacts

- Frozen gate commit: `77dc5be`.
- Exact-layout implementation commit: `f55c442`.
- Spark artifact directory: `/home/kirya/local-spark/experiments/final-77dc5be`.
- Experiment A raw report SHA-256: `e8c4b1736a4572ece3981739f42aa7ee761275f9a702afac9ce361dea87f113a`.
- Experiment A evaluation SHA-256: `6bf8e322d506d7e47a3a0a78a9402018968d8f7eefa163b0f14aa4b1a64f3c6e`.
- Layout smoke SHA-256: `7d2ba5af1c9d0bf55b97b7d3080436b6773b75625a6b2ef3eec777f58de1f3a3`.
- Split A raw report SHA-256: `e1c11ef7353a2dc228fdd462df4a1a981a9694ab8ed78da6f809038f5f0b2a9f`.
- Split B raw report SHA-256: `8b3049e88e984633818a900a0e2dc90e0affeca18d65bdce0f0f97e6ecbb24ee`.
- Joint layout evaluation SHA-256: `10c6bf3f90f14975d9d50651fb962e417c9280e54b4bb5e5ec52fec605b104a6`.
- Compact machine-readable summary: `benchmarks/qwen3.8-27b-final-bounded-experiments.json`.

The research implementation remains isolated on `research/final-bounded-experiments`. These
negative results do not justify merging the exact-layout policy or evaluators into `main`; the
generic candidate-materialization refactor can be considered separately if another experiment
needs it.
