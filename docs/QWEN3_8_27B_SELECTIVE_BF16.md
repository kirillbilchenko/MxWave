# Qwen3.8-27B selective-BF16 attention experiment

> Archived experiment: its composer is intentionally not shipped in the
> MxWave package. This record and the pinned Spark source snapshot below
> preserve the result.

This record tests whether a small BF16 attention subset can retain most of the
quality gain observed when every attention projection is left in BF16. The
candidate produced a statistically significant PPL improvement over full H64,
but failed the predeclared quality-per-byte gate. It was not promoted to the
316-window or GSM8K evaluations.

## Endpoints and gate

The two previously verified endpoints were:

| Artifact | Directory size | 64-window prompt PPL |
|---|---:|---:|
| Full H64 MXFP4 | 19,856,092,425 bytes | 8.165108133 |
| MLP-only MXFP4; all attention BF16 | 30,451,865,541 bytes | 8.094389899 |

The selective candidate initially retained three attention families in BF16:

- `linear_attn.out_proj`;
- `self_attn.o_proj`;
- `self_attn.v_proj`.

Header-only planning selected 80 BF16 modules and projected 22,912,820,704
tensor bytes. Runtime validation correctly rejected this plan: vLLM fuses
`q_proj`, `k_proj`, and `v_proj` and requires all three to use the same
quantization scheme. The exact error was:

```text
ValueError: Found a different quantization schemes for
['q_proj', 'k_proj', 'v_proj'] in
language_model.model.layers.3.self_attn.qkv_proj.
vLLM requires all to use the same scheme.
```

The invalid artifact was deleted after its manifest was preserved. This is an
important planning constraint for any architecture whose runtime fuses several
checkpoint modules into one linear operator.

After this runtime finding, the local composer gained a generic
`--coupled-module-regex` guard. A named `(?P<group>...)` capture declares fused
groups, and planning now rejects any group split across BF16 and MXFP4. The
post-experiment local source-tree digest is
`58a7abf1326f845b0d7259cc1f15b10d5486f02bfa5b22a933eca235da6f025c`;
the earlier deployed digest below remains the exact code that built the tested
artifact.

The corrected candidate retained only the 64 fusion-safe output projections:

- all 48 `linear_attn.out_proj` matrices;
- all 16 `self_attn.o_proj` matrices.

It left 336 matrices in the original H64 MXFP4 representation. The fixed gate
required both:

1. prompt PPL no higher than `8.1006`, recovering at least half of the observed
   all-attention-BF16 gain;
2. a paired 95% confidence-interval upper bound below zero versus full H64.

Only the second condition passed.

## Reproducible composition

The composition source snapshot is on Spark at:

```text
$HOME/local-spark/deployment/mxwave-selective-bf16-20260916
```

Its Python-plus-`pyproject.toml` digest, excluding macOS AppleDouble files, is:

```text
91591c9ea72f46fc19c1524a7d74c1ce01c43160718e9e921a9c6c69652fa297
```

The primary checkpoint was the reproducible full H64 artifact, and the dense
donor was the verified MLP-only artifact. Composition copied existing tensors;
it did not recalibrate or requantize any weight.

```bash
python3 -m mxwave.compose \
  --quantized-model /models/qwen3.8-27b-mxwave-hessian64-d4-repro/model \
  --dense-donor /models/qwen3.8-27b-mxwave-h64-mlp-only/model \
  --output /models/qwen3.8-27b-mxwave-h64-output-bf16/model \
  --keep-dense-regex '^model\.language_model\.layers\.\d+\.linear_attn\.out_proj$' \
  --keep-dense-regex '^model\.language_model\.layers\.\d+\.self_attn\.o_proj$' \
  --expected-keep-count 64
```

The composer validated donor shapes, per-shard keys and dtypes, target/ignore
coverage, exact projected bytes, and the correspondence between packed tensors
and configuration targets. It processed one shard at a time and completed the
18-shard composition in approximately 14 seconds.

The generated artifact contained:

| Property | Value |
|---|---:|
| Quantized modules | 336 |
| BF16 attention modules | 64 |
| Tensor data | 22,789,613,024 bytes |
| Shard files including headers | 22,789,808,408 bytes |
| Complete directory | 22,813,055,677 bytes |
| Added directory bytes versus H64 | 2,956,963,252 bytes |
| Saved directory bytes versus all-attention BF16 | 7,638,809,864 bytes |
| vLLM reported model memory | 20.77 GiB |

vLLM loaded the checkpoint with the frozen H64 serving configuration: Marlin,
BF16 KV cache, 4,096-token model length, two sequences, chunked prefill, prefix
caching disabled, FlashInfer autotuning disabled, and GPU memory utilization
`0.45`.

## Paired 64-window result

The candidate and both endpoints scored exactly the same first 64 deterministic
WikiText windows. Text hashes, token hashes, prompt-token counts, and scored-token
counts paired exactly.

| Artifact | Prompt PPL | Total NLL | Scored tokens |
|---|---:|---:|---:|
| MLP-only / all attention BF16 | **8.094389899** | **128,414.642071** | 61,408 |
| Selective output projections BF16 | 8.139823693 | 128,758.361306 | 61,408 |
| Full H64 | 8.165108133 | 128,948.815110 | 61,408 |

Versus full H64, the candidate improved PPL by `0.309664%`. The fixed-seed,
100,000-sample paired cluster-bootstrap interval was
`[-0.503382%, -0.118423%]`; it won 47 windows and lost 17. The improvement is
real on this screen.

However, it recovered only `35.7538%` of the PPL improvement obtained by leaving
all attention in BF16, below the required 50%. It also remained `0.561300%`
worse than the all-attention-BF16 endpoint, with paired interval
`[+0.295249%, +0.825320%]`.

## GSM8K and decision

PPL is not the only release metric. The intended survivor evaluation includes a
pinned deterministic GSM8K 5-shot run and AMD's published sampled thinking and
non-thinking recipes. GSM8K was deliberately not run here because this candidate
failed the earlier efficiency gate. Running a long stochastic generation suite
on every rejected precision plan would encourage noisy, expensive iteration.

The decision is **reject efficiency promotion**. Output projections account for
some of the MXFP4 degradation, but retaining all of them in BF16 costs 2.96 GB
and does not recover enough quality. A future follow-up should test a different
precision representation, such as FP8 for sensitive fused groups, rather than
adding more BF16 families one at a time.

The rebuildable model and its dedicated compile caches were deleted. The small
records remain at:

```text
$HOME/local-spark/experiments/selective-bf16-20260916/
$HOME/local-spark/runtime/benchmarks/wikitext2/mxwave-h64-output-bf16-4k64.json
```

Hashes:

```text
5d74229655bd7c9cb642b9ecbabf1fc2c53deb1a6215eb7587515d185c8399c6  mxwave-manifest.json
53ed237ebb54cdd9b30bd2862999567c02a910f56834fe7510c8346e62fd36a2  mxwave-h64-output-bf16-4k64.json
```
