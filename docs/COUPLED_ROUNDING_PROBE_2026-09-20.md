# Coupled gate/up MXFP4 rounding probe — pre-registration (2026-09-20)

Status: complete. The hypothesis was rejected by the frozen gate; no checkpoint was built.

The protocol and implementation were frozen in commit `4db0c98` before any real-model result was
inspected.

This experiment tests whether the multiplicative structure of a gated MLP provides a useful
rounding degree of freedom that independent MXFP4 weight reconstruction misses. It is isolated on
`research/coupled-gate-rounding`. Failure ends this direction for the current checkpoint; it does
not authorize a layer, seed, learning-rate, step-count, or calibration-data sweep.

## Hypothesis and practical boundary

For a gated MLP,

```text
z = silu(X W_gate^T) * (X W_up^T)
```

the first-order product error contains two terms that can partially cancel. Standard H64 chooses
the legal MXFP4 values of `W_gate` and `W_up` independently. The candidate instead chooses their
rounding directions jointly, minimizing the exact gated-product error on fixed calibration
activations.

The candidate keeps every H64 E8M0 scale exponent unchanged and every weight at a legal E2M1
value. It does not change tensor shapes, checkpoint format, byte count, kernel selection, or serving
speed. A success can improve quality only; it cannot make this model smaller or faster.

Prior probability is deliberately low: roughly 15–25% for the strict held-out local gate and less
than 10% for a meaningful whole-model gain. A calibration-product improvement is expected and is
not sufficient evidence; the decision is based on exact held-out final teacher KL.

## Frozen identities

| Item | Identity |
|---|---|
| Source | `Qwen/Qwen3.8-27B` |
| Source revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Source config SHA-256 | `191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab` |
| Source index SHA-256 | `77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df` |
| H64 config SHA-256 | `d19a06baa5074b9f1f7b38296d846852fa2c071d2cebdd833e409206fc3feff0` |
| H64 index SHA-256 | `be4a2cd4e130058b23080c6f71172d1e0e37d703e27040deb60fe7f431a7331b` |
| Original H64 calibration SHA-256 | `5f7f6380d2ffd9731c0e4463d89be808c23a2b7101f5d3544eb4124f51c160de` |
| Corpus SHA-256 | `07ffd944ea4fc09f12e5d019633a77309199fdced749e3ddf13989e53586f36a` |
| Calibration token SHA-256 | `e68f97cb4044312513df4c9c82bd758abe348a706fd7e1f11fe9ee15c83efa05` |
| Held-out token SHA-256 | `4bcc89f68fb88e5cb64d9b83d0d6ad3ced954c102c74e5bb2e3f3d487db8d222` |
| Runtime | `vllm/vllm-openai:v0.29.0`, BF16 sequential replay |

The original and reproducibility-calibration safetensors contain the same statistics but differ in
metadata and therefore have different whole-file hashes. This probe is bound to the original file
currently present on Spark, whose hash is listed above.

## Candidate construction

- Layers: `8`, `16`, `42`, and `62`, gated MLP only.
- Gate and up scale selection: exact H64 block Hessian, percentile `99.5`, clip depth `4`.
- For each source weight and its fixed H64 scale, expose only the two legal E2M1 values that bracket
  it. Values already exactly representable have one effective choice.
- Initialize hard choices to the packed H64 reconstruction.
- Optimize the gate and up choices together using a straight-through hard forward pass. Every
  forward pass therefore uses legal fixed-scale MXFP4 values; a soft, non-representable checkpoint
  is never scored.
- Objective: per-intermediate-row normalized MSE of the exact float32 gated product against the
  BF16-source weights, using the frozen activation rows.
- Optimizer: Adam, 24 steps, learning rate `0.05`, initial logits `+/-0.5`, output-row chunks of
  `128`.
- Keep the lowest hard calibration-product MSE observed independently for each output row. A row
  that never improves stays exactly H64.
- The down projection and all weights outside the selected gate/up pair remain H64.

There is one candidate and one control: `coupled-rounding` and unchanged `h64`. There is no
hyperparameter or candidate-family selection.

## Calibration activation capture

The existing H64 Hessian statistics select the fixed scales. One additional bounded streaming pass
over the same registered `64 x 512` calibration sequences captures only the gate input at the four
selected layers. Eight deterministic midpoint-stratified positions per sequence are stored, giving
`512 x hidden_size` BF16 rows per layer. The pass retains one decoder layer at a time; it does not
load the whole model into RAM or VRAM.

Capture command inside the vLLM 0.29.0 container:

```bash
python -m mxwave.coupled_capture_cli \
  --model-dir /input \
  --corpus /calibration/mxstream-pileval-rows-0-99.json \
  --output /experiment/qwen3.8-27b-coupled-activation-samples.safetensors \
  --layers 8,16,42,62 \
  --policy qwen3.8-27b-compatible \
  --num-sequences 64 \
  --sequence-offset 0 \
  --sequence-length 512 \
  --samples-per-sequence 8 \
  --batch-size 1 \
  --device cuda \
  --dtype bfloat16 \
  --attention-implementation sdpa \
  --source-repository Qwen/Qwen3.8-27B \
  --source-revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
```

## Held-out execution

- One combined report uses contexts `152–167`: split A is offset `152`, eight contexts; split B is
  offset `160`, eight contexts.
- Each context is 512 tokens, disjoint from calibration.
- Prefix and suffix use the unchanged packed H64 execution trajectory.
- The source checkpoint provides the BF16 teacher trajectory.
- Exact forward KL is measured at the final eight positions of every context.
- A selected layer is changed in isolation. This is a local falsification probe, not a claim about
  the complete checkpoint.
- The paired bootstrap uses 100,000 resamples, seed `20260920`, stratified by layer and split.

Probe and frozen evaluation commands:

```bash
python -m mxwave.coupled_probe_cli \
  --model-dir /input \
  --baseline-model-dir /baseline \
  --corpus /calibration/mxstream-pileval-rows-0-99.json \
  --activation-stats /calibration/qwen3.8-27b-64x512.safetensors \
  --activation-samples /experiment/qwen3.8-27b-coupled-activation-samples.safetensors \
  --output /experiment/qwen3.8-27b-coupled-probe.json \
  --layers 8,16,42,62 \
  --source-repository Qwen/Qwen3.8-27B \
  --source-revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0

python -m mxwave.coupled_eval \
  /experiment/qwen3.8-27b-coupled-probe.json \
  --output /experiment/qwen3.8-27b-coupled-evaluation.json
```

## Pass gate

Every implementation and identity invariant must hold:

1. Exact registered source, H64, calibration, corpus, calibration-token, and held-out-token
   identities.
2. The H64 materialization exactly reproduces the packed baseline weights and all 16 final-KL
   values at every layer.
3. Every candidate value is a legal E2M1 code under the unchanged H64 scale; at least one code
   changes; calibration gated-product NMSE is strictly lower than H64.
4. Maximum counteraction recurrence residual is at most `1e-10`.

Quality then passes only when all of these hold:

1. At least three of four layers improve mean exact final teacher KL on both splits.
2. Every passing layer wins at least five of eight paired contexts on each split.
3. The pooled stratified 95% paired-bootstrap interval for `H64_KL - coupled_KL` has a strictly
   positive lower bound.

Ties fail. A bounded stop, incomplete report, or implementation-invariant failure also fails. No
full checkpoint is built before this gate passes.

## Results

The complete four-layer probe finished in 452.24 seconds. All artifact, execution, numerical, and
format invariants passed. The candidate reduced its calibration gated-product objective at every
layer while changing at most 0.106% of gate/up codes, so the implementation did find real legal
rounding alternatives rather than reproducing H64.

Those local gains did not generalize consistently. Every layer regressed on split A and improved
on split B; consequently, zero of four layers passed both splits.

| Layer | Product-NMSE reduction | Codes changed | Split A: H64 − candidate KL / wins | Split B: H64 − candidate KL / wins | Pass |
|---:|---:|---:|---:|---:|---:|
| 8 | 6.760% | 65,844 / 178,257,920 (0.0369%) | -0.0005225 / 2 of 8 | +0.0009234 / 5 of 8 | No |
| 16 | 4.185% | 31,598 / 178,257,920 (0.0177%) | -0.0002467 / 4 of 8 | +0.0012739 / 5 of 8 | No |
| 42 | 4.089% | 61,951 / 178,257,920 (0.0348%) | -0.0005427 / 4 of 8 | +0.0008002 / 5 of 8 | No |
| 62 | 9.338% | 188,511 / 178,257,920 (0.1058%) | -0.0003372 / 4 of 8 | +0.0001762 / 5 of 8 | No |

Across layers, the mean paired delta was `-0.00041228` nats on split A and `+0.00079343`
nats on split B, where positive favors coupled rounding. The pooled point estimate favored the
candidate by `+0.00019058` nats, but the pre-registered stratified 95% paired-bootstrap interval
was `[-0.00022316, +0.00062670]`, crossing zero. The maximum recurrence residual was
`3.17e-16`.

The failure is therefore statistical/generalization failure, not an invalid-format or execution
failure. Improving a local gated-product reconstruction objective is insufficient to predict final
teacher KL on this checkpoint. Per the frozen rule, there is no one-sided specificity control,
full-checkpoint build, perplexity run, or hyperparameter retry.

### Reproducibility artifacts

- Branch: `research/coupled-gate-rounding`.
- Frozen implementation/pre-registration commit: `4db0c98`.
- Spark artifact directory: `/home/kirya/local-spark/experiments/coupled-4db0c98`.
- Activation samples: 20,973,072 bytes, SHA-256
  `7d35abb6da2ba0d01b5b69105c09f2459a0e6c5d382e735cb571b3e4176e0832`.
- Raw probe report: 313,557 bytes, SHA-256
  `6e39038a50eb9d5baa67bb1c6c867a3bde1534d2dcd9bebad492a2190333c0f7`.
- Frozen evaluation: 2,266 bytes, SHA-256
  `609005ef98de251b713fba5668517c7f427ac40deb0ae013a67f1f70568af2f2`.
- Peak process RSS: 10,829,975,552 bytes; peak allocated accelerator memory:
  6,852,452,352 bytes.
- Compact machine-readable summary:
  `benchmarks/qwen3.8-27b-coupled-rounding-probe.json`.

## Interpretation and next gate

The observed failure means the gated-product coupling does not generalize strongly enough to
justify more work on this model. Calibration-product gains must not be presented as model-quality
gains.

Had the gate passed, that would only have established that this concrete candidate was useful
locally. It would not have proven that
joint coupling, rather than ordinary one-sided product-aware rounding, caused the gain. Before a
full checkpoint, a fresh-data specificity control must compare the frozen joint candidate with
equal-budget gate-only and up-only optimizers. Only if the joint candidate wins that control may a
single full checkpoint be built and subjected to the existing immutable 128-context divergence and
perplexity gates.

## Resource and operational boundary

- Jobs run detached on Spark with networking disabled.
- Activation capture and probe artifacts are written atomically.
- The probe has a 100-minute internal bound; an external timeout must also be used.
- No extra layers, data offsets, optimizer settings, or retries after a quality result.
- The daily model and observability stack are restored after success, failure, or timeout.
- Publication and merging remain owner decisions.
