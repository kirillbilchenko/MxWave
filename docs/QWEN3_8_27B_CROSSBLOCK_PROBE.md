# Qwen3.8-27B cross-block reconstruction probe

> Archived experiment: its implementation is intentionally not shipped in the
> MxWave package. This record and the pinned Spark source-tree digests below
> preserve the result.

This record describes the Qwen-only go/no-go experiment for 128-channel
cross-block MXFP4 error compensation. The pre-registered complete-layer
promotion gate did not pass. A later, explicitly approved causal follow-up
patched only the repeatedly sensitive layer 63 `down_proj`; its paired
perplexity screen was neutral-to-worse and rejected production promotion.

## Question and controls

The production H64 quantizer selects scales independently for each 32-channel
MXFP4 storage block. The experimental candidate keeps that exact packed format
but propagates rounding error into later channels inside a 128-channel window
using a damped inverse Hessian.

For every selected decoder layer, the probe compares complete layer output on
held-out BF16-prefix inputs:

1. the existing 64x512 H64 checkpoint;
2. a matched 8x512 local 32-channel Hessian replacement;
3. the 8x512 128-channel cross-block replacement.

The matched candidate distinguishes wider reconstruction from a change in
calibration samples. The probe restores each of `gate_proj`, `up_proj`, and
`down_proj` to BF16 and applies reconstruction to the projection with the best
restoration result. `down_proj` was selected in all six comparisons.

The fixed promotion criterion was:

- cross-block held-out normalized MSE at most 90% of H64;
- lower MSE than the matched local-Hessian replacement;
- both conditions on at least two of three selected layers.

## Pinned inputs

| Item | Value |
|---|---|
| Source | `Qwen/Qwen3.8-27B` |
| Source revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Baseline | reproducible MxWave H64 scale-only checkpoint |
| Baseline manifest SHA-256 | `8ba884020e52ed92cc5902c2e9d5a65b60d8b0b53caa8965583ed1362cdea1a0` |
| Corpus SHA-256 | `07ffd944ea4fc09f12e5d019633a77309199fdced749e3ddf13989e53586f36a` |
| Policy | `qwen3.8-27b-compatible` |
| Sequence length | 512 |
| Train / held-out | 8 / 8 disjoint sequences per replication |
| Window | 128 input channels; four ordinary MXFP4 blocks |
| Hessian damping | `1e-6` collection + `1%` inverse-Hessian damping |
| Scale search | percentile `99.5`, clip depth `4` |
| Probe source-tree digest | `a7dbe535943e9b8170c396b938d3708e1023d068808ab1c8ef1ae4a4399e9e07` |
| Patch source-tree digest | `b44047a85355fa4a021311ec4a2e4aad553e071b45906f43af613af0ddbd5119` |

The source-tree digest was calculated with:

```bash
(find mxwave -type f -name '*.py' -print; printf '%s\n' pyproject.toml) \
  | LC_ALL=C sort \
  | xargs shasum -a 256 \
  | shasum -a 256
```

## Exact command

The following ran inside the pinned Qwen vLLM container on DGX Spark. It was
executed once with `OFFSET=64` and once with `OFFSET=80`.

```bash
OFFSET=64

python3 -u -m mxwave.qwen_probe \
  --model-dir /input \
  --baseline-model-dir /baseline \
  --corpus /calibration/mxwave-pileval-rows-0-99.json \
  --output "/probe/qwen3.8-27b-crossblock-8x8-l512-offset${OFFSET}.json" \
  --policy qwen3.8-27b-compatible \
  --num-train-sequences 8 \
  --num-heldout-sequences 8 \
  --sequence-offset "$OFFSET" \
  --sequence-length 512 \
  --batch-size 1 \
  --top-layers 3 \
  --window-size 128 \
  --scale-percentile 99.5 \
  --mse-clip-depth 4 \
  --hessian-damp 1e-6 \
  --gptq-damp-percent 1.0 \
  --tensor-row-chunk-size 512 \
  --device cuda \
  --dtype bfloat16 \
  --attention-implementation sdpa
```

The corpus provides enough tokens for both ranges. The split identities were:

| Offset | Train token-ID SHA-256 | Held-out token-ID SHA-256 |
|---:|---|---|
| 64 | `7b92b256c36cfcefda7fa7212e5d089aae4f444ecdba8c0cc62cb16e4e4669f5` | `1018b713b8240490424d40d5aa24f5b8362a3d68710be866de0ec8d88ee1648b` |
| 80 | `f4b44bc3c0520d39c281112281536936b82e17ffa69bfef6c0b2e894851a6dc0` | `c525a229ead27e99950d39b3b60ea20a384eb36749f6d17d1b04f91bbb7e8ab1` |

## Results

All layer indices below are zero-based. Lower normalized MSE is better.

| Split | Layer | H64 | Matched local | Cross-block | Cross vs H64 |
|---:|---:|---:|---:|---:|---:|
| 64 | 51 | 0.003557981 | 0.003560355 | 0.003506545 | **1.446% better** |
| 64 | 62 | 0.003275963 | 0.003283004 | 0.003210062 | **2.012% better** |
| 64 | 63 | 0.003371689 | 0.003382743 | 0.003088520 | **8.398% better** |
| 80 | 50 | 0.002738946 | 0.002738840 | 0.002695978 | **1.569% better** |
| 80 | 51 | 0.002890403 | 0.002889124 | 0.002838556 | **1.794% better** |
| 80 | 63 | 0.003052076 | 0.003044798 | 0.002570087 | **15.792% better** |

The direction reproduced on all six selected-layer comparisons. Layer 63 was
selected on both splits and showed the strongest result. However, only one of
six comparisons exceeded the fixed 10% threshold. The per-run gate therefore
reported zero of three clear wins for offset 64 and one of three for offset 80;
both required two.

Ordinary weight MSE increased for every cross-block candidate. Depending on the
layer, 15.98% to 29.66% of E2M1 codes changed relative to H64, and cross-block
weight MSE was approximately 18% to 78% higher. This is a meaningful risk even
though held-out layer outputs improved.

Each full replication took approximately 166 seconds and peaked at 4.94 GiB of
PyTorch accelerator allocation. The experiment streams one BF16 decoder layer
at a time and does not load the whole model.

## Decision

The experiment establishes a reproducible cross-block signal, so the idea is
not rejected as mathematically ineffective. It does **not** establish model
quality improvement. The pre-registered gate rejected automatic production
integration. The targeted PPL follow-up below then failed its end-to-end gate,
so the production H64 quantizer remains unchanged.

## Targeted checkpoint follow-up

The follow-up changed only zero-based decoder layer 63 `mlp.down_proj`, the one
projection that was repeatedly strongest across the two held-out probes. It
used all 32 sequences from offsets 64 through 95 for calibration; it did not
reuse WikiText evaluation text.

```bash
python3 -u -m mxwave.qwen_patch \
  --model-dir /models/qwen3.8-27b-bf16/huggingface \
  --baseline-model-dir /models/qwen3.8-27b-mxwave-hessian64-d4-repro/model \
  --corpus /calibration/mxwave-pileval-rows-0-99.json \
  --output-model-dir /models/qwen3.8-27b-mxwave-crossblock-l63/model \
  --layer 63 \
  --projection down_proj \
  --num-sequences 32 \
  --sequence-offset 64 \
  --sequence-length 512 \
  --batch-size 1 \
  --window-size 128 \
  --scale-percentile 99.5 \
  --mse-clip-depth 4 \
  --hessian-damp 1e-6 \
  --gptq-damp-percent 1.0 \
  --tensor-row-chunk-size 512 \
  --device cuda \
  --dtype bfloat16 \
  --attention-implementation sdpa
```

The calibration observed 16,384 projection inputs. Its token-ID SHA-256 was
`73ac56eca912fa8283ed668bc229116b204bd7e6c229139980a639b1ccd97b95`.
Relative to H64, 24.997846% of this projection's E2M1 codes and 10.507382% of
its E8M0 scales changed.

The patcher copied the affected shard and overwrote only the two existing
safetensors data ranges. The index, configuration, tensor shapes, dtypes,
offsets, and shard size remained unchanged. Baseline and candidate each contain
exactly `19,832,628,704` tensor bytes. The only size addition is the 2,345-byte
`mxwave-crossblock-patch.json` provenance record; it is not a model tensor.

## Paired 64-window PPL result

The candidate and H64 baseline were served one at a time with the frozen Marlin
configuration in the H64 reproducibility record. Both scored the same first 64
WikiText windows. All text hashes, token hashes, prompt-token counts, and
scored-token counts paired exactly.

| Artifact | Prompt PPL | Total NLL | Scored tokens |
|---|---:|---:|---:|
| H64 baseline | **8.165108133** | **128,948.815110** | 61,408 |
| Layer-63 cross-block patch | 8.167738441 | 128,968.593893 | 61,408 |

The candidate was `+0.002630307` PPL, or `+0.032214%`, worse. The fixed-seed
100,000-sample paired cluster-bootstrap interval was `[-0.004188%, +0.067512%]`.
It won 24 windows and lost 40 (two-sided exact sign test `p=0.05994`). Thus the
difference is not statistically distinguishable from zero, but there is also no
evidence of improvement. The 316-window evaluation was intentionally not run.

This result is the important control: a 8.40% to 15.79% improvement in the
selected layer's held-out output NMSE did not translate to end-to-end
likelihood. Future work must use an end-to-end acceptance signal or a stronger
trust region; layer-local improvement alone is insufficient.

## Raw reports

The Spark reports and hashes are:

```text
3484fa7f8edf68b12719fae8ea3697fb3642badb04a8095175a390b4f844b049  qwen3.8-27b-crossblock-8x8-l512-offset64.json
73109eea4c11384bca00d0bd36397e0d005ca4554ceb74dd9ac62216ded1aeac  qwen3.8-27b-crossblock-8x8-l512-offset80.json
c59c4e2a9fbf9455162cb7ed03466ca4462b676ae6bd7b8b7ef6ba7eac2c5021  mxwave-crossblock-patch.json
7e5f42779ef51d774677886af0f7240c707b4f19249eb5351d7fceef69242ac0  crossblock-l63-32x512-4k64.json
```

The two probe reports are under
`$HOME/local-spark/experiments/mxwave-crossblock-probe/`. The patch
record is inside its candidate model directory, and the PPL report is under
`$HOME/local-spark/runtime/benchmarks/wikitext2/`.
