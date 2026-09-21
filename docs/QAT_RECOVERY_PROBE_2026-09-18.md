# Fake-MXFP4 recovery probe record — 2026-09-18

> Working reproducibility record. This file is intentionally not committed yet.

## Decision

The first bounded quantization-aware recovery configuration is rejected. It
improved the selected layer's normalized output error by 27.83% on its training
contexts, but made the disjoint local-validation error 15.20% worse. The
pre-registered 10% validation-recovery gate therefore stopped the run after the
first target. The second target, final teacher-KL splits, and complete checkpoint
build were intentionally skipped.

This result rejects this exact small-sample rank-16 STE recipe. It does not by
itself reject quantization-aware recovery as a method. It does show that fitting
the accumulated trajectory error on four contexts generalizes poorly and must
not be promoted based on training loss.

## Frozen identities

| Item | Value |
|---|---|
| Branch | `research/qat-recovery` |
| Recovery implementation commit | `21298ca` |
| Preflight commit | `62b7e2d` |
| Source repository | `Qwen/Qwen3.8-27B` |
| Source revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Source config SHA-256 | `191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab` |
| Source index SHA-256 | `77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df` |
| H64 config SHA-256 | `d19a06baa5074b9f1f7b38296d846852fa2c071d2cebdd833e409206fc3feff0` |
| H64 index SHA-256 | `be4a2cd4e130058b23080c6f71172d1e0e37d703e27040deb60fe7f431a7331b` |
| Block-Hessian calibration SHA-256 | `5f7f6380d2ffd9731c0e4463d89be808c23a2b7101f5d3544eb4124f51c160de` |
| Corpus SHA-256 | `07ffd944ea4fc09f12e5d019633a77309199fdced749e3ddf13989e53586f36a` |
| Runtime image ID | `sha256:61952c22fc3e1b546a883248b34773219e992c233624f56bba4488c07e44f491` |

## Method

The experiment kept a BF16 master weight and trained temporary low-rank factors
through an exact legal MXFP4 quantize/dequant forward pass with an identity
straight-through gradient. The factors were training-only. A passing candidate
would have been merged into the master and reconstructed from standard MXFP4
packed values, so it would not change checkpoint size, vLLM kernels, or runtime
execution.

Targets were fixed from the archived static low-rank diagnostic, before this run:

1. `model.language_model.layers.57.linear_attn.out_proj.weight`;
2. `model.language_model.layers.62.linear_attn.out_proj.weight`.

The intended procedure was sequential: correct layer 57, then capture layer 62
on the already-corrected quantized trajectory. The first local gate failed, so
the second step was never executed.

Training configuration:

| Setting | Value |
|---|---:|
| Rank | 16 |
| Steps | 64 |
| Learning rate | 0.003 |
| Batch tokens | 128 |
| Maximum train/validation tokens | 1,024 / 1,024 |
| Relative delta regularization | 0.0001 |
| Gradient clip norm | 1.0 |
| Scale percentile / clip depth | 99.5 / 4 |
| Row chunk | 256 |
| Seed | 20260918 |

Frozen data protocol:

| Split | Contexts | Offset | Token-ID SHA-256 |
|---|---:|---:|---|
| Train | 4 × 512 | 64 | `9ed6e781f7a5e4921f3298bfff33388ad76b1b83d131b7825f74a63424e368bf` |
| Local validation | 4 × 512 | 72 | `93c7716de2d790a017a3e066e1ad2a68f582466303ff2fd9e6c8c11e02274512` |
| Final A, unused | 8 × 512 | 104 | `d00e42d3cd459d5fdfeab724122e1eb6d2862423d9b2412420fec4b165e34f44` |
| Final B, unused | 8 × 512 | 112 | `2aa77ff0648fc899965a1ab0064fd09c8138574aef000427d11ed79bb7fd90ff` |

Final promotion would have required lower H64-relative teacher KL on both final
splits and a positive lower bound for the pooled 100,000-sample paired-bootstrap
95% interval. Those data were not evaluated after the local failure.

## Result

| Layer-57 metric | H64 baseline | Recovered candidate | Change |
|---|---:|---:|---:|
| Training output NMSE | 0.05222430 | 0.03769095 | **27.83% recovered** |
| Validation output NMSE | **0.07886308** | 0.09085356 | **15.20% worse** |

Additional checks:

- regenerated block-Hessian candidate versus stored H64 weight NMSE: exactly
  `0.0`;
- combined captured H64 output NMSE versus BF16 teacher: `0.06086492`;
- learned full-delta norm: `15.58227`;
- candidate tensor SHA-256:
  `ec1d01630d55084cad4e29db3b0b4a410a60adf92f96df97365ce38cf0022682`.

The falling training objective alongside worse validation is classic
generalization failure. It is not explained by a wrong H64 reconstruction,
checkpoint-format mismatch, whole-model loading, OOM, or timeout.

## Resource bounds and artifacts

The job ran detached as `mxwave-recovery-gate-62b7e2d`, with networking
disabled, a 64 GiB RAM limit, a 72 GiB RAM-plus-swap limit, and a 60-minute GNU
`timeout`. It exited 0 without OOM after the gate rejected the candidate.

| Resource | Value |
|---|---:|
| Measured probe time | 79.51 s |
| Container wall time | about 89 s |
| Peak process RSS | 8,448,065,536 bytes (7.87 GiB) |
| Peak accelerator allocation | 4,308,979,712 bytes (4.01 GiB) |

| Artifact | SHA-256 |
|---|---|
| `preflight.json` | `2b816edf715d68a1f2940fad2be1de7b0097a9a76164e62bcce6fc6bfb8c6622` |
| `report.json` | `d8fa4e59c128771bfa84a46fc9c534d97f3140e02e1b7ce777f84db226e90435` |

Spark artifact directory:

```text
/home/kirya/local-spark/experiments/qat-recovery-62b7e2d
```

## Interpretation and next admissible test

Do not weaken the gate, run the untouched final splits, train layer 62, or build
a 27B checkpoint from this candidate.

If recovery is tested once more, the change should address the observed failure
rather than sweep ranks or learning rates. A defensible follow-up would use
three local partitions: training, candidate selection, and a still-hidden local
gate. Candidate selection could use a fixed trust-region path or early stopping,
while the hidden local gate and the existing offsets 104/112 remain untouched.
The retry should remain limited to layer 57 and stop unless the improvement
generalizes. Otherwise, close this recovery track and retain H64.

## Three-way trust-region retry

Commit `51c686d` implemented the single admitted retry. It preserved rank 16,
64 steps, learning rate 0.003, and the exact fake-MXFP4 forward. It expanded the
training sample and added a pre-registered candidate path rather than sweeping
model hyperparameters:

- checkpoints: steps 8, 16, 32, and 64;
- trust factors: 0.25, 0.5, and 1.0;
- selection rule: minimum selection-split NMSE across the twelve candidates,
  with H64 winning exact ties;
- hidden local gate: at least 10% recovery on an untouched third split.

The frozen three-way protocol was:

| Split | Contexts | Offset | Token-ID SHA-256 |
|---|---:|---:|---|
| Train | 8 × 512 | 64 | `7b92b256c36cfcefda7fa7212e5d089aae4f444ecdba8c0cc62cb16e4e4669f5` |
| Candidate selection | 8 × 512 | 72 | `1018b713b8240490424d40d5aa24f5b8362a3d68710be866de0ec8d88ee1648b` |
| Hidden local gate | 8 × 512 | 96 | `c92870e1f56b60540ccbdb064b854f95f0b828b8785c9a302fd9782707a8eb48` |
| Final A, unused | 8 × 512 | 104 | `d00e42d3cd459d5fdfeab724122e1eb6d2862423d9b2412420fec4b165e34f44` |
| Final B, unused | 8 × 512 | 112 | `2aa77ff0648fc899965a1ab0064fd09c8138574aef000427d11ed79bb7fd90ff` |

The selection split chose step 16 with trust factor 0.5. Unlike the first run,
the correction generalized in the same direction, but its effect remained below
the pre-registered gate:

| Metric | H64 baseline | Selected candidate | Recovery |
|---|---:|---:|---:|
| Training NMSE | 0.04810932 | 0.04321711 | 10.17% |
| Selection NMSE | 0.07115675 | 0.06785390 | 4.64% |
| Hidden-gate NMSE | 0.04890178 | 0.04575101 | 6.44% |

The regenerated H64 weight again matched exactly (`0.0` NMSE). The selected
candidate tensor SHA-256 was
`548158f200df5bd0e4aa68f8aac5002a2d2a8f3dfaa16bf302b130c54290b480`.
Because 6.44% is below the fixed 10% gate, final teacher KL was not run and no
checkpoint was built. The threshold was not changed after observing the result.

The detached container `mxwave-recovery-path-gate-51c686d` exited 0 without OOM:

| Resource | Value |
|---|---:|
| Measured probe time | 111.25 s |
| Container wall time | about 120 s |
| Peak process RSS | 9,405,734,912 bytes (8.76 GiB) |
| Peak accelerator allocation | 4,308,979,712 bytes (4.01 GiB) |

| Artifact | SHA-256 |
|---|---|
| `preflight.json` | `e16f55fb35905b852603a6a6758473244c65a76afaf01a9bd63b2b6b797333a0` |
| `report.json` | `68427b02c5b6069a70174926143f2c651d2eee747d33fb414166c50927287ccf` |

Spark artifact directory:

```text
/home/kirya/local-spark/experiments/qat-recovery-path-51c686d
```

### Final decision for this track

The fixed trust-region path corrected the first run's sign reversal, so its
selection mechanism is technically useful. It did not produce enough unseen
local recovery to justify spending the untouched final splits or a model build.
Per the pre-registered decision, close this recovery track for the current H64
model, retain the generic branch infrastructure, and do not start another rank,
learning-rate, checkpoint-step, or trust-factor sweep.
