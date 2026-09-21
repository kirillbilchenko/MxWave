# Operator-response probe record — 2026-09-18

> Working reproducibility record. This file is intentionally not committed yet.

## Decision

The tested selector is rejected for model construction. Local post-decoder-layer
normalized MSE did not rank legal MXFP4 candidates consistently with final BF16
teacher KL. The pre-registered requirement was at least 70% teacher-winner
agreement. After six measured MLP layers, agreement was 2/6. With only six of
the planned twelve probes remaining, the maximum possible result was 8/12 =
66.7%, so the run was stopped without weakening the gate or building a model.

This rejects **local operator NMSE as the selector**. It does not invalidate the
streaming runtime IR, functional candidate replay, diagonal/full-Hessian
controls, or exact suffix teacher-KL probe.

## Frozen identities

| Item | Value |
|---|---|
| Branch | `research/operator-response` |
| Real-layer probe commit | `c215c7f` |
| Diagonal-Hessian control commit | `c463378` |
| Teacher-KL suffix probe commit | `3431938` |
| Source repository | `Qwen/Qwen3.8-27B` |
| Source revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Source `config.json` SHA-256 | `191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab` |
| Source index SHA-256 | `77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df` |
| Calibration | block-Hessian, 64 sequences × 512 tokens |
| Calibration SHA-256 | `4f9e726a77d2f9f9e304ef18fafc8ef103e12caccf00d3c9950369809824483f` |
| Held-out corpus SHA-256 | `07ffd944ea4fc09f12e5d019633a77309199fdced749e3ddf13989e53586f36a` |
| Held-out token IDs SHA-256 | `7b92b256c36cfcefda7fa7212e5d089aae4f444ecdba8c0cc62cb16e4e4669f5` |
| Held-out split | offset 64, 8 sequences × 512 tokens |
| Second-split token IDs SHA-256 | `1018b713b8240490424d40d5aa24f5b8362a3d68710be866de0ec8d88ee1648b` |
| Second held-out split | offset 72, 8 sequences × 512 tokens |
| Confirmation token IDs SHA-256 | `39407c5a8ba95e1ba2aed18d2237d384a5d88baee15a69778b7b5fdcfc173802` |
| Confirmation split | offset 80, 16 sequences × 512 tokens |
| Runtime image | `vllm/vllm-openai:qwen38-flash-next@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8` |

Spark paths used during this run:

```text
source:      /home/kirya/local-spark/models/qwen3.8-27b-bf16/huggingface
calibration: /home/kirya/local-spark/calibration/repro/qwen3.8-27b-64x512.safetensors
corpus:      /home/kirya/local-spark/calibration/mxstream-pileval-rows-0-99.json
reports:     /home/kirya/local-spark/experiments/operator-response-smoke
```

## Candidate set

All candidates emit legal MXFP4 E2M1 blocks with 32 values per E8M0 scale.
Candidate tensors were dequantized only for functional replay.

| Candidate | Scale choice |
|---|---|
| `rtn` | compressed-tensors-compatible memoryless min/max RTN |
| `unweighted-mse` | unweighted scale search, percentile 99.5, clip depth 4 |
| `diagonal-hessian` | same search weighted by the square root of the block-Hessian diagonal |
| `block-hessian` | same search using the complete 32×32 block Hessian |

## Commands

The container mounted the source model at `/input`, calibration data at
`/calibration`, the commit archive at `/opt/mxwave`, and reports at `/output`.
The substantive commands inside the pinned image were:

```bash
python3 -u -m mxwave.operator_probe_cli \
  --model-dir /input \
  --corpus /calibration/mxstream-pileval-rows-0-99.json \
  --activation-stats /calibration/repro/qwen3.8-27b-64x512.safetensors \
  --output /output/all-mlp-8x512.json \
  --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63 \
  --policy qwen3.8-27b-compatible \
  --num-sequences 8 --sequence-offset 64 --sequence-length 512 \
  --scale-percentile 99.5 --mse-clip-depth 4 \
  --tensor-row-chunk-size 128 --device cuda --dtype bfloat16 \
  --attention-implementation sdpa \
  --source-repository Qwen/Qwen3.8-27B \
  --source-revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0

python3 -u -m mxwave.operator_probe_cli \
  --model-dir /input \
  --corpus /calibration/mxstream-pileval-rows-0-99.json \
  --activation-stats /calibration/repro/qwen3.8-27b-64x512.safetensors \
  --output /output/representative-diagonal-8x512.json \
  --layers 0,1,2,3,5,7,10,11,14,15,31,42,58,63 \
  --policy qwen3.8-27b-compatible \
  --num-sequences 8 --sequence-offset 64 --sequence-length 512 \
  --scale-percentile 99.5 --mse-clip-depth 4 \
  --tensor-row-chunk-size 128 --device cuda --dtype bfloat16 \
  --attention-implementation sdpa \
  --source-repository Qwen/Qwen3.8-27B \
  --source-revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0

python3 -u -m mxwave.teacher_kl_probe_cli \
  --model-dir /input \
  --corpus /calibration/mxstream-pileval-rows-0-99.json \
  --activation-stats /calibration/repro/qwen3.8-27b-64x512.safetensors \
  --output /output/teacher-kl-layers63-31-0-8x512.json \
  --layers 63,31,0 \
  --policy qwen3.8-27b-compatible \
  --num-sequences 8 --sequence-offset 64 --sequence-length 512 \
  --logit-positions 8 \
  --scale-percentile 99.5 --mse-clip-depth 4 \
  --tensor-row-chunk-size 128 --device cuda --dtype bfloat16 \
  --attention-implementation sdpa \
  --source-repository Qwen/Qwen3.8-27B \
  --source-revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0

python3 -u -m mxwave.teacher_kl_probe_cli \
  --model-dir /input \
  --corpus /calibration/mxstream-pileval-rows-0-99.json \
  --activation-stats /calibration/repro/qwen3.8-27b-64x512.safetensors \
  --output /output/teacher-kl-replication-9layers-8x512.json \
  --layers 58,42,14,11,10,7,5,2,1 \
  --policy qwen3.8-27b-compatible \
  --num-sequences 8 --sequence-offset 64 --sequence-length 512 \
  --logit-positions 8 \
  --scale-percentile 99.5 --mse-clip-depth 4 \
  --tensor-row-chunk-size 128 --device cuda --dtype bfloat16 \
  --attention-implementation sdpa \
  --source-repository Qwen/Qwen3.8-27B \
  --source-revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0

python3 -u -m mxwave.teacher_kl_probe_cli \
  --model-dir /input \
  --corpus /calibration/mxstream-pileval-rows-0-99.json \
  --activation-stats /calibration/repro/qwen3.8-27b-64x512.safetensors \
  --output /output/teacher-kl-split3-layers0-14-16x512.json \
  --layers 0,14 \
  --policy qwen3.8-27b-compatible \
  --num-sequences 16 --sequence-offset 80 --sequence-length 512 \
  --logit-positions 8 \
  --scale-percentile 99.5 --mse-clip-depth 4 \
  --tensor-row-chunk-size 128 --device cuda --dtype bfloat16 \
  --attention-implementation sdpa \
  --source-repository Qwen/Qwen3.8-27B \
  --source-revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
```

Each long run used GNU `timeout`; the all-layer local probe had a 600-second
limit and teacher-KL runs had a 900-second limit. The replication was stopped
manually once passing the gate became mathematically impossible.

## Results

### Local MLP response map

The three-candidate 64-layer map completed in 232.84 seconds:

- full block-Hessian won 56/64 MLP layers;
- RTN won layers 0, 1, 5, 7, 8, 10, 13, and 14;
- unweighted MSE won 0/64;
- raw weight-error ranking matched the operator winner on 0/64 layers;
- one winner held across all eight sequences on 55/64 layers;
- choosing the local winner reduced summed local NMSE by only 0.0213% versus
  using block-Hessian for every MLP.

The 14-layer diagonal control never selected diagonal-Hessian. Full
block-Hessian was consistently 0.15–1.43% better than the diagonal control on
the selected layers, so off-diagonal block statistics affected the local
response result.

### Final teacher KL

Teacher KL used the exact BF16 suffix and final vocabulary head over the last
eight positions of each held-out sequence.

| Layer | Local operator winner | Teacher-KL winner | Spearman ρ | Teacher winner vs block-Hessian |
|---:|---|---|---:|---:|
| 63 | block-Hessian | block-Hessian | 0.8 | 0.00% |
| 31 | block-Hessian | RTN | -1.0 | 20.46% lower KL |
| 0 | RTN | unweighted MSE | -0.8 | 13.46% lower KL |
| 58 | block-Hessian | diagonal-Hessian | 0.6 | 8.96% lower KL |
| 42 | block-Hessian | block-Hessian | 0.4 | 0.00% |
| 14 | RTN | diagonal-Hessian | -0.8 | 1.44% lower KL |

Agreement was 2/6 and mean per-layer rank correlation was -0.133. The local
selector therefore failed both the 70% winner criterion and the intended
positive correlation behavior.

Layer 31 was the clearest non-tie: RTN beat block-Hessian on 7/8 sequences and
reduced mean teacher KL from 0.00085257 to 0.00067816. A paired bootstrap over
the eight sequences gave a positive 95% interval for the absolute improvement,
approximately `[0.0000339, 0.0003359]`.

### Second-split oracle stability

The same six layers and candidates were then evaluated on disjoint tokenized
sequences 72–79. The exact teacher-KL argmin reproduced on only layer 42, where
both splits selected the existing block-Hessian baseline.

| Layer | Split 1 teacher winner | Split 2 teacher winner | Combined-mean winner |
|---:|---|---|---|
| 63 | block-Hessian | RTN | RTN |
| 31 | RTN | block-Hessian | unweighted MSE |
| 0 | unweighted MSE | diagonal-Hessian | diagonal-Hessian |
| 58 | diagonal-Hessian | block-Hessian | block-Hessian |
| 42 | block-Hessian | block-Hessian | block-Hessian |
| 14 | diagonal-Hessian | unweighted MSE | unweighted MSE |

The pre-declared robust rule required a candidate to beat block-Hessian on both
splits and have a positive stratified paired-bootstrap 95% interval over all 16
sequences. No candidate passed. Layer 0 was the strongest exploratory signal:
diagonal-Hessian reduced mean KL by 12.07% and 11.11% on the two splits, but its
combined interval still crossed zero (`[-3.88e-5, 2.02e-4]`). It therefore
remains a hypothesis, not a recipe entry.

### Fixed third-split confirmation

Two comparisons were fixed before reading a fresh 16-context split:

- layer 0: diagonal-Hessian versus block-Hessian;
- layer 14: unweighted MSE versus block-Hessian.

Passing required lower mean KL, at least 12/16 paired wins, and a strictly
positive paired-bootstrap 95% interval.

Layer 0 passed: diagonal-Hessian reduced mean KL by 13.29%, won 12/16 contexts,
and had interval `[1.80e-5, 1.52e-4]`. Across all three splits it reduced mean
KL by 12.46%, won 22/32 contexts, and had stratified interval
`[1.33e-5, 1.51e-4]`.

Layer 14 failed: unweighted MSE reduced the fresh-split mean by 3.35% but won
only 8/16 contexts and its interval crossed zero. Its pooled 32-context interval
also crossed zero.

The only admissible recipe change is therefore narrow: diagonal-Hessian for the
three layer-0 MLP weights, with block-Hessian retained everywhere else. This is
eligible for an end-to-end checkpoint experiment, not yet for a quality claim.

### Whole-model confirmation checkpoint

Commit `b57ef57` added a model-agnostic, exact-tensor recipe mechanism. Recipes
are bound to the source repository/revision, hashes of declared source files,
the calibration objective and artifact hash, and the validated quantization
target set. Resume identity includes the normalized recipe. There is no Qwen
conditional in the selection engine.

The layer-0 recipe SHA-256 was
`61722ff203d2d85270a5345aafb3863b548f627f47fbd1b589fbb08cbfabf893`.
The complete 18-shard rebuild finished in 239.2 seconds with a 110 GiB memory
limit and two-hour timeout. It retained the standard compressed-tensors MXFP4
format and exactly the H64 artifact data size, `19,832,831,240` bytes.

A tensor-by-tensor comparison against frozen H64 found 1,593/1,599 tensor keys
bitwise identical. The only six changed keys were `weight_packed` and
`weight_scale` for layer-0 MLP `gate_proj`, `up_proj`, and `down_proj`, exactly
matching the declared override. The output manifest SHA-256 was
`9b76cd22706ce66722bc1aaff099f1855923344752fbaed2133758831189fe0e`.

The primary whole-model gate reused the immutable 128-context, 512-token
WikiText manifest and archived BF16/H64 full-vocabulary distributions. Candidate
collection took 350.63 seconds including model load, compilation, and 128
requests. The candidate log-probability artifact SHA-256 was
`f0101d0462209e4d0caaf83071bd28f47f9b0bcb4f065f98b893407e30d7aaef`;
the comparison report SHA-256 was
`48825567bbfddc3a87f8e708bf2d6e9d6ed5c3398431ce921048f06f2fda4278`.

| Metric | H64 | Layer-0 diagonal | Result |
|---|---:|---:|---|
| Mean forward KL from BF16 | **0.049937061** | 0.050514940 | candidate 1.16% worse |
| Forward-KL p95 | **0.175240615** | 0.182147583 | candidate worse |
| Mean reverse KL | **0.044734070** | 0.045129553 | candidate worse |
| Mean Jensen-Shannon divergence | **0.010945828** | 0.011038294 | candidate worse |
| Mean total variation | **0.078565230** | 0.079398165 | candidate worse |
| BF16 top-1 agreement | **120/128 (93.75%)** | 118/128 (92.19%) | candidate worse |

The paired mean `H64 - candidate` forward-KL difference was
`-0.000577879` nats with bootstrap 95% interval
`[-0.001503214, +0.000271706]`. H64 had lower KL on 62 contexts and the
candidate on 66, but the candidate's regressions were larger. Removing its
single largest regression still left the candidate worse by `0.000322501`
nats on average.

This rejects the recipe for promotion. The isolated teacher-KL effect did not
transfer through the complete network. A longer prompt-perplexity run was not
started after the primary gate failed; doing so would add compute and invite
post-hoc metric selection. The generic recipe mechanism remains useful research
infrastructure, while this specific layer-0 recipe is a negative result.

### Quantized-baseline replay diagnostic

Commit `11314d4` added bounded loading of a packed MXFP4 checkpoint into one
ordinary Transformers module at a time. The teacher remains BF16, while the
execution trajectory, candidate injection, suffix, normalization, and output
head come from the frozen H64 checkpoint. This tests a candidate in the error
environment it would actually encounter after quantization, without rebuilding
the full model.

The diagnostic was deliberately limited to the known layer-0 failure and two
fresh Pileval slices, offsets 88 and 96 (8 sequences of 512 tokens each). The
block-Hessian control reconstructed all three stored H64 MLP weights exactly:
baseline-weight NMSE and local operator NMSE were both `0.0` on both runs.

| Split | H64 mean teacher KL | Layer-0 diagonal mean teacher KL | Winner |
|---:|---:|---:|---|
| offset 88 | **0.026643729** | 0.026659108 | H64 by 0.058% |
| offset 96 | 0.035505111 | **0.035475605** | diagonal by 0.083% |
| pooled 16 sequences | 0.031074420 | 0.031067357 | practical tie |

For paired `diagonal - H64` sample KL, the pooled mean was
`-0.000007063`, wins were exactly 8/16 each, and a deterministic 100,000-sample
paired bootstrap interval was `[-0.000676111, +0.000634005]`. The two splits
disagreed and the interval spans effects far larger than the observed mean.
This diagnostic therefore does not reliably predict the whole-model regression;
no broader layer or candidate sweep is admitted from this result.

Both commands ran sequentially inside detached container
`mxwave-quantized-baseline-gate-11314d4`, which exited 0 without OOM. The
container had no network, a 110 GiB memory ceiling, and a 30-minute timeout.

## Resource bounds

| Run | Elapsed | Peak accelerator allocation | Peak process RSS |
|---|---:|---:|---:|
| 64-layer local MLP map | 232.84 s | 5.05 GiB | 7.72 GiB |
| 14-layer diagonal control | 96.73 s | 5.55 GiB | 7.75 GiB |
| 3-layer teacher-KL gate | 156.77 s | 3.17 GiB | 9.63 GiB |
| 6-layer second-split teacher KL | 301.97 s | 3.18 GiB | 9.63 GiB |
| 2-layer, 16-context confirmation | 201.35 s | 3.18 GiB | 9.99 GiB |
| Whole-model MXFP4 rebuild | 239.2 s | not recorded | bounded to 110 GiB |
| 128-context exact-KL collection | 350.63 s | vLLM runtime-managed | bounded to 110 GiB |
| Quantized-baseline replay, offset 88 | 89.55 s | 5.16 GiB | 10.26 GB |
| Quantized-baseline replay, offset 96 | 91.52 s | 5.16 GiB | 10.23 GB |

No run materialized the full model. Candidate weights were released after the
selected layer; suffix trajectories were held on CPU; one BF16 decoder layer
was resident at a time.

## Machine artifacts

| Artifact | SHA-256 | Status |
|---|---|---|
| `layers0-31-63-8x512.json` | `a1c7e54c876d51d3c649070b078306f3761c88874861280323af0c8cb29fec01` | Complete |
| `all-mlp-8x512.json` | `abdf27b8947f611dd1d386e912899170a19030436b5982e23d956b14f123f7c3` | Complete |
| `representative-diagonal-8x512.json` | `c75d0ad3d3b7f58341fb8cc628b9f8c650e54cd3c82610b3392666d5cc49fe7c` | Complete |
| `teacher-kl-layers63-31-0-8x512.json` | `10ff37046daf7279de96bdb91320443d9f91d9d8cf4ccb032eee18a164266cfc` | Complete |
| `teacher-kl-replication-9layers-8x512.json` | `04e28b16baccbb0dbb34ffcfa185cc7676eb60a437834e2bb49d024b6cdf1776` | Intentionally partial after layers 58, 42, and 14 |
| `teacher-kl-split2-6layers-8x512.json` | `c7d2493ef754378b43f80c603ec7858d046e83f5fc54fe9acd0633e70a2963f8` | Complete |
| `teacher-kl-split3-layers0-14-16x512.json` | `208a3d144595bb9d585e4b6de471b4328546db2e377cb754607a8a7751da931a` | Complete |
| `layer0-offset88-8x512.json` | `9185b5f38194d11e2c08973b2e7a7b9daa1251f2aac4f67bacfa935e4bac1965` | Complete quantized-baseline replay |
| `layer0-offset96-8x512.json` | `d55d8305700d175bf85814a7b567428b50c6600c498dc6950a0f3f596c81d124` | Complete quantized-baseline replay |

## Interpretation and next admissible experiment

The finite local response is informative—it robustly disproves the raw weight
MSE ranking—but it is not global enough to select a model. Error direction and
the BF16 suffix matter more than local response magnitude on several layers.

A next experiment must not build another locally selected checkpoint or run a
broad per-split teacher-argmin sweep. Quantized-baseline replay is now valid
infrastructure, but the layer-0 intervention was below its stable selection
resolution. The useful paths are:

1. use quantized-baseline replay only for predeclared interventions large enough
   to clear paired uncertainty on multiple disjoint splits;
2. test an accumulated or sequential correction objective, where earlier
   accepted quantized errors are present when a later operation is optimized;
3. retain block-Hessian whenever a candidate is a statistical tie, and require
   a bounded probe to predict a previously hidden whole-model result before
   another 27B rebuild.

Broad attention/GDN adaptation remains blocked. Neither BF16-suffix nor
quantized-baseline single-operation teacher KL has yet shown stable selection
power for a whole-model recipe.
