# Qwen3.8-27B selective-FP8 attention experiment

> Archived experiment: its composer is intentionally not shipped in the
> MxWave package. This record and the pinned Spark source snapshot below
> preserve the result.

## Outcome

Replacing the same 64 attention output projections tested in BF16 with
channel-wise FP8 E4M3 produced a statistically credible but small improvement
over full H64 MXFP4:

| Artifact | Directory bytes | PPL64 |
|---|---:|---:|
| Full H64 MXFP4 | 19,856,092,425 | 8.165108133 |
| Selective FP8 output projections | 20,801,137,092 | 8.148518300 |
| Selective BF16 output projections | 22,813,055,677 | 8.139823693 |
| All-attention BF16 / MLP-only quantization | 30,451,865,541 | 8.094389899 |

The FP8 candidate improved PPL by `0.203180%` versus H64. The fixed-seed,
100,000-sample paired cluster-bootstrap interval was
`[-0.395995%, -0.014099%]`, with 41 winning and 23 losing windows. The effect is
small but distinguishable from zero on this screen.

It cost only 945,044,667 directory bytes over H64 and saved 2,011,918,585 bytes
versus the equivalent selective-BF16 plan. Its PPL gain per added decimal GB
was 2.05 times the BF16 plan's gain. It is therefore a valid Pareto signal.

It is not a release candidate. It recovered only `23.4591%` of the PPL gain
available from leaving all attention in BF16, and PPL `8.1485` missed the
predeclared `8.1006` promotion threshold. The 316-window and GSM8K evaluations
were not run.

## Representation

The mixed checkpoint contained:

- 336 matrices in the original H64 `mxfp4-pack-quantized` representation;
- 64 output-projection matrices as FP8 E4M3 values;
- one FP32 scale per output channel for each FP8 matrix;
- global compressed-tensors format `mixed-precision`, with explicit per-group
  `mxfp4-pack-quantized` and `float-quantized` formats.

This is ordinary channel-wise FP8 weight quantization, not OCP MXFP8. The
composer is architecture-neutral: selection is by concrete regex, while the
optional coupled-module guard prevents incompatible mixed schemes inside
runtime-fused projections.

Tensor data was 20,777,657,824 bytes. The 64 FP8 weights occupied
2,013,265,920 bytes and their scales 1,310,720 bytes. Replacing their MXFP4
representations added 945,029,120 tensor bytes.

## Reproduction

The exact build source snapshot is on Spark at:

```text
$HOME/local-spark/deployment/mxwave-selective-fp8-20260916
```

Its Python-plus-`pyproject.toml` digest is:

```text
9ec641a14b557544cbfffa18761c9b5b382c2169d54c7f398d8d05385cc18cf6
```

The composer streamed one primary shard and one selected donor tensor at a
time. GPU conversion produced the 18-shard checkpoint in approximately 55
seconds:

```bash
python3 -m mxwave.compose_fp8 \
  --quantized-model /models/qwen3.8-27b-mxwave-hessian64-d4-repro/model \
  --dense-donor /models/qwen3.8-27b-mxwave-h64-mlp-only/model \
  --output /models/qwen3.8-27b-mxwave-h64-output-fp8/model \
  --select-regex '^model\.language_model\.layers\.\d+\.linear_attn\.out_proj$' \
  --select-regex '^model\.language_model\.layers\.\d+\.self_attn\.o_proj$' \
  --expected-select-count 64 \
  --quant-device cuda
```

The local source later received a manifest-only correction restoring
`target_source_bytes` for FP8 targets. It does not change tensor values,
checkpoint configuration, or the measured result.

## Runtime compatibility

The installed vLLM build accepted the mixed checkpoint but did not select FP8
Marlin by default on SM121. Startup required the flag explicitly requested by
its kernel selector:

```bash
VLLM_TEST_FORCE_FP8_MARLIN=1 vllm serve /model ... --linear-backend marlin
```

With that flag, all weights loaded successfully in 122.60 seconds. vLLM
reported 18.95 GiB model memory and completed compilation, profiling, and CUDA
graph capture. It also warned that the GPU was using a weight-only FP8 Marlin
fallback rather than a native FP8 path, so this representation may reduce
compute-heavy throughput. The 64-window evaluation took 89.57 seconds versus
84.94 seconds for H64; those single-run timings are diagnostic, not a formal
throughput benchmark.

The Triton `tl.make_block_ptr` deprecation warning observed during warmup comes
from the installed runtime and does not affect current numerical execution.

## Evaluation and records

The candidate scored the exact same first 64 deterministic WikiText-2 windows
as the earlier H64 and BF16 screens: 262,144 characters, 61,472 prompt tokens,
and 61,408 scored tokens. Pairing was validated by window index, text hash,
token hash, prompt-token count, and scored-token count.

The compact checked-in result is
`benchmarks/qwen3.8-27b-selective-fp8-prompt-ppl.json`. The raw report remains
on Spark at:

```text
$HOME/local-spark/runtime/benchmarks/wikitext2/mxwave-h64-output-fp8-4k64.json
```

Recorded hashes before artifact cleanup:

```text
4988a88bbe1891e378071432fd9d10932ed95637d9d3a558b1209a494fc84187  mxwave-h64-output-fp8-4k64.json
8a22ddd544d5082f75247fab03b356426cc538185ef15eaada756ddf31c49232  mxwave-manifest.json
```

The large rebuildable checkpoint and its dedicated compile cache were deleted
after preserving the manifest, config, model card, and evaluation report.
