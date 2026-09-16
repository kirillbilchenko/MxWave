# Qwen3.8-27B static low-rank recovery probe

> Archived experiment: its implementation is intentionally not shipped in the
> MxWave package. This record and the pinned Spark source snapshot below
> preserve the result.

## Outcome

A bounded experiment tested whether a tiny activation-weighted low-rank
correction could recover the quantization error of the same 64 attention output
projections used in the selective-BF16 and selective-FP8 experiments. It could
not recover enough held-out error to justify loading the model:

| Representation | Added tensor bytes | Held-out error recovery |
|---|---:|---:|
| Rank-4 BF16 correction | 5,767,168 | 12.9626% |
| Rank-8 BF16 correction | 11,534,336 | 15.6052% |
| Channel-wise FP8 reference | 945,029,120 over H64 | 94.3983% |

Rank 8 recovered only `16.5312%` as much held-out error as FP8. The
predeclared gate required both at least `50%` absolute recovery and at least
`75%` of FP8's recovery. Both quality checks failed, so the experiment stopped
before adapter compatibility, vLLM startup, PPL64, PPL316, or GSM8K. No adapter
was written.

This rejects a **static rank-at-most-8 matrix residual** for this scope. It does
not reject LR-QAT or quantization-aware distillation: those methods optimize
through the quantized model and can alter the quantized weights, whereas this
probe used one closed-form local correction and no training.

## Fixed scope and stop conditions

The limits were declared before execution:

- reuse the existing 64-sequence H64 calibration; do not recalibrate;
- select only all 48 `linear_attn.out_proj` and all 16
  `self_attn.o_proj` matrices;
- test only nested ranks 4 and 8;
- use the disjoint 32-sequence offset-64 artifact for selection;
- cap an emitted adapter at 250 MiB;
- run at most one 64-window PPL evaluation, and only after the offline gate;
- do not run the 316-window likelihood test or GSM8K in this experiment;
- do not expand to all attention matrices without a new decision.

Execution also had a 20-minute wall-clock timeout and a 24 GiB container-memory
limit. The successful offline pass completed in `10.1428` seconds. An initial
attempt stopped before reading any model matrix because the validator did not
recognize an omitted offset as the documented offset-zero default. That
validation bug was covered by a test and fixed without changing the experiment.

## Method

For dense donor weight `W`, H64 reconstruction `Wq`, and block-diagonal input
second moment `H`, the target residual is:

```text
E = W - Wq
```

For `H = L L^T`, MxWave computes a deterministic randomized SVD of `E L` and
maps the right factor back through `L^-1`:

```text
E L ~= U_r S_r V_r^T
E   ~= B A
```

The balanced `B` and `A` factors are cast to BF16 before scoring, matching the
intended LoRA representation. Rank 4 is the leading nested truncation of the
same rank-8 factorization, rather than a separately tuned candidate. Loss is
then measured on the disjoint Hessian as:

```text
sum((E - B A) H (E - B A)^T)
```

The implementation is model-agnostic. Checkpoint modules are selected by regex,
calibration artifacts are module-keyed, and the core math only requires
two-dimensional weights with 32-channel block Hessians.

## Reproduction

The exact source snapshot used on Spark is:

```text
$HOME/local-spark/deployment/mxwave-low-rank-probe-20260916
```

Its Python-plus-`pyproject.toml` digest is:

```text
76774d41b38047b6633a632574179ed8f728a723ec8bbe61865e0a000b4987b8
```

The command inside the pinned vLLM image was:

```bash
python3 -m mxwave.low_rank_probe \
  --quantized-model /models/qwen3.8-27b-mxwave-hessian64-d4-repro/model \
  --dense-donor /models/qwen3.8-27b-mxwave-h64-mlp-only/model \
  --training-stats /calibration/qwen3.8-27b-64x512.safetensors \
  --selection-stats /calibration/qwen3.8-27b-selection32-offset64-512.safetensors \
  --output /results/output-projections \
  --select-regex '^model\.language_model\.layers\.\d+\.linear_attn\.out_proj$' \
  --select-regex '^model\.language_model\.layers\.\d+\.self_attn\.o_proj$' \
  --expected-select-count 64 \
  --rank 4 \
  --rank 8 \
  --device cuda \
  --oversample 8 \
  --power-iterations 1 \
  --seed 20260916 \
  --row-chunk-size 1024 \
  --maximum-adapter-bytes 262144000 \
  --minimum-heldout-recovery 0.5 \
  --minimum-fp8-recovery-ratio 0.75
```

The complete per-module report remains at:

```text
$HOME/local-spark/experiments/low-rank-recovery-20260916/
  output-projections/low-rank-probe.json
```

Its SHA-256 is:

```text
c76defeaefbb894f5585d6ce8d7a40c4731ffffecfe30031d95fcaf5d838de09
```

The compact checked-in result is
`benchmarks/qwen3.8-27b-low-rank-recovery.json`.

## Interpretation

The aggregate held-out MXFP4 loss was `412.7630`. Rank 4 reduced it to
`359.2581`; rank 8 reduced it to `348.3505`. In comparison, channel-wise FP8
reduced it to `23.1219`.

Recovery was not uniformly zero. Rank 8 recovered `31.70%` of the error for
layer 62's linear-attention output and `50.79%` for layer 57, while layer 0 had
`97.23%` recovery but contributed only `0.572` of the aggregate baseline loss.
Most matrices retained a broad residual spectrum, so doubling rank from 4 to 8
added only `2.64` percentage points of aggregate recovery. That shape strongly
suggests that simply trying a slightly larger static rank would spend more
adapter compute without approaching the FP8 reference.

The next materially different recovery test, if pursued, should train through
fake MXFP4 numerics using an end-to-end or block-output objective. It should not
be presented as a continuation of this static-SVD rank sweep.
