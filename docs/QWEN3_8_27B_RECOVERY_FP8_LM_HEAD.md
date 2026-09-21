# Qwen3.8-27B recovered H64: FP8 `lm_head` experiment

## Question

Can the recovered H64 checkpoint reduce storage by converting only
`lm_head.weight` from BF16 to channel-wise FP8 E4M3 without a meaningful quality
or serving regression?

This is a bounded compression experiment, not a new default. The complete
vision encoder, MTP predictor, token embedding, and all 400 recovered MXFP4
linear weights remain unchanged.

## Frozen variants

1. BF16 teacher: the immutable upstream source revision already recorded by
   the H64 reproducibility protocol.
2. H64: the existing 400-tensor H64 MXFP4 checkpoint.
3. Recovery parent: the checkpoint packed by MxWave commit `734db06` from the
   accepted large recovery adapter.
4. FP8-head candidate: variant 3 with only `lm_head.weight` converted from BF16
   to per-output-channel FP8 E4M3 plus one FP32 scale per row.

The FP8 candidate must use the standard compressed-tensors
`float-quantized` W8A16 representation. It must load on SM121 without a patched
runtime or custom kernel.

## Frozen conversion

```bash
mxwave-fp8-compress \
  --model-dir <recovery-parent> \
  --output-dir <fp8-head-candidate> \
  --target-module lm_head \
  --device cuda \
  --tensor-row-chunk-size 256
```

The converter must preserve all non-target tensors, tokenizer/processor files,
vision tensors, MTP tensors, and recovery metadata. It must emit an updated
safetensors index and a mixed-precision compressed-tensors configuration.

## Gates

The experiment stops before MXFP4 `lm_head` if any mandatory gate fails.

1. **Structure:** the output contains exactly one new FP8 weight representation,
   all other tensor values are unchanged, and the measured checkpoint reduction
   agrees with the header-only plan.
2. **Runtime:** stock `vllm/vllm-openai:v0.29.0` loads the model with Marlin on
   DGX Spark SM121 and selects the FP8 weight-only kernel for `lm_head`, both
   with ordinary decoding and with the existing two-token MTP proposer enabled.
3. **Exact distribution:** on the frozen 128-context/full-vocabulary protocol,
   mean forward KL from BF16 may not regress by more than 1% relative to the
   recovery parent. The paired 95% upper bound for `FP8 - parent` must be below
   `+0.002` nat per context.
4. **Likelihood:** fixed WikiText perplexity may not regress by more than 0.1%
   relative to the recovery parent. Because this boundary is intentionally
   tight, paired window values and their interval must be reported.
5. **Runtime cost:** steady-state decode throughput may not regress by more than
   5% in the same serving profile. MTP acceptance and throughput are measured
   separately because the proposer reuses `lm_head`; speculative verification
   must continue to preserve target-model results. Load time is recorded
   separately.

Top-1 agreement and task scores are reported as secondary diagnostics, not used
alone to accept a model. GSM8K is run only after the deterministic gates pass.

## Escalation to MXFP4

An MXFP4 `lm_head` experiment is authorized only if the FP8 candidate clears the
mandatory gates. It must use the same parent checkpoint, contexts, runtime, and
comparison protocol. Passing FP8 is evidence that the output projection has
some precision headroom; it is not evidence that a 4-bit output projection is
safe.

## Results — 2026-09-20

**Rejected at the exact-distribution gate.** The candidate produced a useful
storage and accelerator-memory reduction, but the BF16-relative distribution
regression was larger than the frozen tolerance. Per the gate, no perplexity,
GSM8K, MTP-throughput, or MXFP4-head run was started.

The converter was implemented at commit `fa69c8a`; commit `876b7f7` added the
anchored runtime target needed for a top-level Hugging Face `lm_head` to match
vLLM's nested `model.language_model.lm_head` path.

### Structure and size

- Recovery parent safetensors: `19,832,831,240` bytes (`18.470763` GiB).
- FP8-head safetensors: `18,562,426,176` bytes (`17.287607` GiB).
- Reduction: `1,270,405,064` bytes (`1.183157` GiB, `6.4056%`).
- `lm_head.weight`: FP8 E4M3, shape `248320 x 5120`.
- `lm_head.weight_scale`: FP32, shape `248320 x 1`.
- All 17 unaffected shard files were byte-identical. All 15 non-head tensors in
  the rewritten shard were value-identical, including all 15 `mtp.*` tensors;
  all 333 `model.visual.*` tensors were also unchanged.
- Output config SHA-256:
  `b2b7265637e721faf28f64a4a1c3af0f7a6f8e67acdac3b5e3656138741485dc`.
- Compression report:
  `benchmarks/qwen3.8-27b-recovery-fp8-head-compression.json` (SHA-256
  `2267d2f2bee362378fbb3fbe4621fa645bfda69233fa9272c1abba34b4838aa9`).

### Stock-runtime load

Stock `vllm/vllm-openai:v0.29.0` loaded all 18 shards with
`--linear-backend marlin`. Model weight allocation was `16.94` GiB, compared
with approximately `18.09` GiB for the recovery parent. vLLM selected
weight-only FP8 Marlin for the head and warned that this route is not native
FP8 compute on the tested platform, so a speed gain was not assumed. Ordinary
model load and exact inference passed. The two-token MTP runtime smoke was not
run after the quality gate failed; only preservation of its checkpoint tensors
was established.

### Exact full-vocabulary divergence

The frozen test used 128 identical WikiText contexts, up to 512 tokens each,
and all 248,320 next-token log probabilities. Every candidate was compared to
the same BF16 reference.

| Candidate | Mean forward KL | Reverse KL | JS | TV | Top-1 agreement |
|---|---:|---:|---:|---:|---:|
| H64 | 0.049937060 | **0.044734069** | **0.010945828** | **0.078565229** | 120/128 |
| Recovery parent | **0.049087645** | 0.048015111 | 0.011166077 | 0.079350967 | 119/128 |
| FP8 head | 0.051379769 | 0.050568224 | 0.011715412 | 0.082942708 | 120/128 |

Relative to the recovery parent, the FP8 candidate increased mean forward KL
by `0.002292124` nat, or `4.6695%`. The paired 95% interval for
`FP8 - recovery` was `[+0.000499349, +0.004360299]` nat. This fails both the
1% point-estimate limit and the `+0.002`-nat paired upper-bound limit. The
candidate also regressed by `2.8891%` against H64 on the point estimate,
although that paired interval crossed zero. Recovering one top-1 agreement did
not offset the consistent aggregate-divergence regressions.

The FP8 log-probability artifact SHA-256 is
`dd652345d420563abf76e7b290b39e88e3ae495c1d7db86eb3318fd3660d50cd`.
The complete machine-readable comparison is
`benchmarks/qwen3.8-27b-recovery-fp8-head-divergence.json` (SHA-256
`43082f42ac6c2fa7f53bf5a92aa8f4c7e89128e81386ff15a8f8d3a17d2e7d6f`).

### Decision

Do not publish or promote this checkpoint, and do not quantize its head to
MXFP4. The result is still useful: the output projection is large enough to
move whole-checkpoint size materially, but naive per-row FP8 is not a
quality-neutral compression for this model. A future retry would need a
different, pre-registered scale objective; rerunning downstream tests on this
same rejected candidate would not change that conclusion.
