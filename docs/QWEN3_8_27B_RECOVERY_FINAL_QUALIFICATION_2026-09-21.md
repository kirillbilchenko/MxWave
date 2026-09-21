# Qwen3.8-27B MXFP4 recovery: final qualification

Date: 2026-09-21  
Status: local research record; do not publish as an H64 replacement  
Checkpoint: `/home/kirya/local-spark/models/qwen3.8-27b-mxwave-h64-recovery-v2/model`

## Decision

The recovery checkpoint is technically valid and improves the primary deterministic
WikiText-2 perplexity result at effectively identical model size. It does not establish a
broad quality win over H64: exact-distribution evidence is mixed and full GSM8K is tied.

Keep the checkpoint and research branch locally. Do not replace or supersede the published
H64 model, and do not upload this checkpoint in its current form.

This is a successful bounded research experiment, not a sufficiently differentiated model
release.

## What changed

The model starts from the published H64 MXFP4 checkpoint. A bounded recovery-training pass
adjusted legal E2M1 codes in 12 selected tensors while preserving their original E8M0 scales:

- layers 8--10 linear-attention output projections;
- layer 11 attention output projection;
- gate/up projections in layers 51--54.

The recovery pass used rank-16 adapters with alpha 16, a BF16 teacher, legal MXFP4 rounding
in every forward pass, and 65,536 pinned training tokens. Packing changed 33,835,438 of
838,860,800 target values (4.0335%) and 37,676,520 packed codes. The emitted checkpoint is
ordinary `mxfp4-pack-quantized`; inference needs no adapter or custom kernel.

This procedure includes optimization through a quantized forward pass. It is
quantization-aware recovery training, not pure post-training quantization.

## Size

| Artifact | Bytes |
|---|---:|
| H64 tensor payload | 19,832,831,240 |
| Recovery tensor payload | 19,832,831,240 |
| H64 model directory | 19,856,330,600 |
| Recovery model directory | 19,856,338,703 |

The 8,103-byte directory difference is the recovery manifest. There is no material model-size
or inference-speed change because tensor shapes, quantization format, and runtime kernel are
unchanged.

## WikiText-2 prompt perplexity

Protocol: 316 paired 4,096-character windows, 297,199 scored tokens, vLLM 0.29.0, Marlin.

| Model | PPL | Relative to BF16 |
|---|---:|---:|
| BF16 | 7.980864 | baseline |
| H64 | 8.120845 | +1.754% |
| Recovery | **8.067761** | **+1.089%** |

Recovery improves PPL by 0.653677% relative to H64 and recovers 37.92% of H64's PPL gap to
BF16. It wins 295 of 316 paired windows. The 100,000-resample paired interval for the relative
change is `[-0.706400%, -0.601901%]`; the sign-test p-value is approximately `4.98e-63`.
Removing the single most favorable window leaves a 0.646358% improvement.

This is the strongest evidence in favor of the candidate: the result is broad, deterministic,
and statistically stable, but its magnitude is modest.

## Exact next-token divergence against BF16

Protocol: 128 deterministic contexts, up to 512 tokens, all 248,320 next-token log
probabilities, vLLM 0.29.0, Marlin.

| Metric | H64 | Recovery | Direction |
|---|---:|---:|---|
| Mean forward KL, nats | 0.049937 | **0.049088** | recovery better by 1.70% |
| Forward-KL p95, nats | **0.175241** | 0.187826 | recovery worse |
| Mean reverse KL, nats | **0.044734** | 0.048015 | recovery worse |
| Mean Jensen-Shannon, nats | **0.010946** | 0.011166 | recovery worse |
| Mean total variation | **0.078565** | 0.079351 | recovery worse |
| BF16 top-1 agreement | **93.750%** | 92.969% | recovery worse |
| Mean top-5 overlap | **4.359 / 5** | 4.336 / 5 | recovery worse |

Mean `H64 - recovery` forward KL is `+0.000849` nats, but its paired 95% interval is
`[-0.002154, +0.004125]`, which crosses zero. Recovery has lower forward KL on 59 contexts;
H64 has lower forward KL on 69.

Therefore the exact-distribution test does not support a general fidelity improvement. The
forward-KL mean moves favorably, while tail behavior and several complementary metrics move
slightly unfavorably.

## GSM8K

Protocol: full 1,319-example test split, five-shot, no-think template, greedy decoding,
temperature zero, 1,024-token generation limit, four concurrent requests, lm-eval 0.4.13.
Prompt, target, and sample hashes matched in the paired audit.

| Model | Strict match | Flexible extraction |
|---|---:|---:|
| BF16 | 88.400% | 90.447% |
| H64 control | **91.585%** | 91.585% |
| Recovery | 91.509% | **91.736%** |

Paired recovery-versus-H64 outcomes:

- strict: recovery-only correct 24, H64-only correct 25; delta `-0.076` percentage points;
  paired interval `[-1.137, +0.986]` points;
- flexible: recovery-only correct 27, H64-only correct 25; delta `+0.152` percentage points;
  paired interval `[-0.910, +1.213]` points.

This is a tie. The historical full H64 report did not embed its serving-image digest, so this
comparison is descriptive rather than the primary release gate.

Against BF16, recovery gains 3.108 points under strict extraction, with paired interval
`[+1.289, +4.928]`, but gains only 1.289 points under flexible extraction, with interval
`[-0.379, +2.957]`. The latter crosses zero. Quantized models outperforming BF16 on this task
also reinforces why GSM8K should not override the deterministic PPL and distribution results.

## Resource and compatibility result

The large recovery-training gate completed 512 steps in 2,174.04 seconds (36.23 minutes).
Peak accelerator allocation was 55.37 GiB and peak process RSS was 35.89 GiB. Its untouched
post-training confirmation passed the pre-registered divergence gate before checkpoint
packing.

The packed checkpoint loaded with stock vLLM 0.29.0 and the same Marlin W4A16 path as H64.
It structurally preserves the standard checkpoint format, config, tokenizer, vision tensors,
and MTP tensors. Vision behavior, MTP acceptance, and MTP throughput were not independently
qualified; tensor preservation must not be reported as capability validation.

## Provenance

| Item | Value |
|---|---|
| BF16 source revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| BF16 config SHA-256 | `191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab` |
| H64/recovery config SHA-256 | `d19a06baa5074b9f1f7b38296d846852fa2c071d2cebdd833e409206fc3feff0` |
| Recovery implementation record | `a55145c1354b082382545cac35a913739f43a480` |
| Final recovery branch | `research/end-to-end-mxfp4-recovery` at `734db06` |
| Adapter SHA-256 | `bdb5b529233b7e0b08aa014beb4469ead819affd494f84832ee34a62d1291cf6` |
| Recovery manifest SHA-256 | `ddcdb2d894e12ed61c1bacdaa11dd16936ac859f09bd6c9bffef304d8d697699` |
| Config SHA-256 | `d19a06baa5074b9f1f7b38296d846852fa2c071d2cebdd833e409206fc3feff0` |
| Index SHA-256 | `be4a2cd4e130058b23080c6f71172d1e0e37d703e27040deb60fe7f431a7331b` |
| 18 shards + config + index hash-of-hashes | `b5dc8edd0008e5f939eefac6fc745d017b352ca2e8439b4b4ab40024011000f5` |

Key result artifacts retained on Spark:

| Result | SHA-256 |
|---|---|
| Recovery PPL report | `2cd9cc4caecc55a82a4d4a445f52b9de76e2844369e2b2a6462bffbdee30187e` |
| Paired PPL comparison | `1814c7ebad9e5a741324dfc295c56d53f9ae47f6e46387dece6b09bff6337683` |
| Exact KL comparison | `75061e262f5861ed6edd23f91c82cdbaf9cdd7c649c182ded48bdd4ed6b01e8e` |
| Recovery GSM8K aggregate | `f7b936308697d35e49584684133187dd0d0a495c8d273de906df205fa7e530eb` |
| Recovery GSM8K samples | `ad3428abdb8645d806d6d7060286495ae274901cd605b260ce33f8ad18b44cb8` |
| BF16 GSM8K aggregate | `d40f07c9ebe41990f5f3c8a14e8dfb6f7c666f794ac562c058e24b2cf3e6bfda` |
| BF16 GSM8K samples | `b67ca1009aae9f5e319bbfc0aa75e002d582209178b123eb8a1e2c36d98bcd82` |
| H64 GSM8K aggregate | `d4bcabbabfe3367b96b22fb076509c9b2f2bd5aa58de15d2a47353be5f726830` |
| H64 GSM8K samples | `22eb587ca1347657958719f3b0a17a4b162180c79ee35c6e080407fd7eb6a942` |

## Final recommendation

Do not publish or overwrite H64 with this checkpoint. The recovery mechanism is real and the
PPL gain is reproducible, so retain the branch, checkpoint, and reports. However, a second
roughly 18.47-GiB public artifact would add user and maintenance cost for a 0.65% PPL gain
without a consistent KL or downstream-task advantage.

No further evaluation is required for the present no-publish decision.
