# MxWave

[![Hugging Face model](https://img.shields.io/badge/🤗%20Hugging%20Face-Qwen3.8--27B--MXFP4--MxWave-FFD21E)](https://huggingface.co/kirillbilchenko/Qwen3.8-27B-MXFP4-MxWave)
[![CI](https://github.com/kirillbilchenko/MxWave/actions/workflows/ci.yml/badge.svg)](https://github.com/kirillbilchenko/MxWave/actions/workflows/ci.yml)

**Quality-oriented MXFP4 post-training quantization for LLMs, with real
activation calibration, bounded-memory checkpoint processing, and verified
vLLM output.**

**Published model:**
[`kirillbilchenko/Qwen3.8-27B-MXFP4-MxWave`](https://huggingface.co/kirillbilchenko/Qwen3.8-27B-MXFP4-MxWave)
— an 18.47 GiB Qwen3.8-27B checkpoint with complete evaluation and reproducibility records.

MxWave converts floating-point safetensors checkpoints to vLLM's
`compressed-tensors` `mxfp4-pack-quantized` format. It reads source tensors in
bounded chunks, can collect activation statistics one decoder layer at a time,
and validates packing, reconstruction quality, and module coverage before an
artifact is considered complete.

The first fully evaluated target is `Qwen/Qwen3.8-27B` on NVIDIA DGX Spark
(GB10/SM121). The quantization math is model-independent; weight-streamed
calibration currently has a verified adapter for the Qwen3.5-text decoder
layout used by that model. Unsupported streaming layouts fail closed.

## Why this repo exists

Memoryless min/max round-to-nearest is a useful MXFP4 baseline, but it ignores
which input channels matter most on real data. MxWave is built around three
ideas:

1. **MSE-optimal, calibration-aware scale selection** — an explicit exponent
   candidate set per block, chosen by minimum reconstruction error and optionally
   weighted by real activation moments or a block Hessian.
2. **Bounded residency** — calibration loads one supported decoder layer at a
   time, while quantization reads and processes tensors in bounded row chunks.
3. **Verification-first output** — emitted shapes, dtypes, module coverage, and
   sampled reconstruction quality are checked instead of inferred from the
   source checkpoint.

## Status

🚧 **Experimental, with a working core and streaming engine.** Module-keyed calibration,
shard discovery, on-device quantization, output assembly, and strict output
verification are implemented and unit-tested. A paired 297k-token DGX Spark
likelihood screen found the recommended block-Hessian scale-only artifact 0.83%
lower in perplexity than AMD Quark-AWQ MXFP4 and 1.74% above BF16. On 128
held-out contexts it also had 12.0% lower mean next-token forward KL from BF16
than AMD, although that paired interval crosses zero. The exact protocols and
uncertainty are recorded below; the earlier 100-item task screens remain
diagnostic rather than reportable benchmark scores.

The checked-in benchmark JSON was produced before the MxWave rename and retains
its original `mxstream-*` schema and model labels so the recorded SHA-256 hashes
remain valid. Current Python imports, console commands, and newly generated
artifacts use `mxwave` exclusively.

## Install

```bash
git clone https://github.com/kirillbilchenko/MxWave.git
cd MxWave
pip install -e ".[dev,calibrate]"
mxwave-calibrate --help
mxwave-quantize --help
```

## Activation calibration and quantization

```bash
mxwave-calibrate \
    --model-dir /path/to/float-model \
    --corpus /path/to/calibration.jsonl \
    --output /path/to/activation-stats.safetensors \
    --policy qwen3.8-27b-compatible \
    --statistics mean-abs,rms,block-hessian \
    --num-sequences 16 \
    --sequence-length 512 \
    --weight-loading streaming

mxwave-quantize \
    --model-dir /path/to/float-model \
    --output-dir /path/to/output \
    --policy qwen3.8-27b-compatible \
    --activation-stats /path/to/activation-stats.safetensors \
    --calibration-objective block-hessian \
    --mse-clip-depth 4 \
    --tensor-row-chunk-size 1024 \
    --device cuda \
    --verify-sqnr
```

`mean-abs` is the inexpensive diagnostic requested for comparing real inputs
with the LayerNorm-gamma fallback. `rms` is the diagonal approximation to
expected output reconstruction error. `block-hessian` retains correlations
within each 32-channel MXFP4 block and is the strongest available scale
objective. The Hessian affects scale selection only; code assignment remains
nearest-E2M1. The calibration file must cover every selected target exactly;
MxWave never silently mixes real statistics with gamma or unweighted MSE.

Calibration defaults to bounded-residency `--weight-loading streaming`. It
constructs the model on the meta device, loads the embedding and then one
decoder layer at a time, keeps the evolving hidden states on CPU, and releases
each layer before loading the next. Every checkpoint weight is still read once,
but the full float model is never resident. A clean Qwen3.8-27B `1x512` probe
peaked at 7.46 GiB process RSS and 3.28 GiB CUDA allocation while traversing all
64 layers; the BF16 checkpoint itself is 51.75 GiB. Linux may retain already-read
shard pages as reclaimable filesystem cache, so `docker stats` can temporarily
look larger than the tensor working set. The calibration artifact records both
peak RSS and peak accelerator allocation.

Sequential loading currently has a verified adapter for the Qwen3.5-text
decoder layout used by Qwen3.8-27B. Unsupported layouts fail closed. Use
`--weight-loading resident` explicitly only when a model has no streaming
adapter and enough memory is available.

Runtime-aware optimization uses a separate, model-independent operation graph.
Built-in adapters currently describe the verified Qwen3.5-text layout and the
standard dense Llama layout, including runtime-fused QKV and gate/up groups.
Adapters inspect only configuration and safetensors headers, validate every
required weight, and fail closed when an architecture is unknown or incomplete.

### Experimental precision budgets

`mxwave-precision-budget` builds bounded semantic interventions on top of an
existing MXFP4 checkpoint. It promotes explicitly selected, fusion-safe runtime
groups to channel-wise FP8 from a compatible floating-point donor while retaining
MXFP4 everywhere else. The planner reads checkpoint headers only; composition
streams primary shards and materializes one selected donor matrix at a time.

```bash
mxwave-precision-budget plan \
    --primary-model /path/to/mxfp4-model \
    --dense-donor /path/to/float-model \
    --output /path/to/precision-plan.json

mxwave-precision-budget compose \
    --plan /path/to/precision-plan.json \
    --bucket sequence-output-early-l08-11 \
    --bucket mlp-input-late-l51-54 \
    --output /path/to/mixed-model \
    --quant-device cuda
```

This remains opt-in: a plan defines candidates but does not establish that any
candidate improves the model. Promotion requires frozen reference and baseline
outputs, disjoint selection splits, an untouched holdout, and whole-model task
checks. The first Qwen3.8-27B experiment improved prompt PPL from `8.119445` to
`8.111578` while adding 375.6 MiB; its complete protocol, uncertainty, runtime
cost, and vLLM 0.29.0 validation are in the
[precision-budget record](docs/QWEN3_8_27B_PRECISION_BUDGET.md).

Target matrices are read from safetensors and quantized in bounded row ranges;
`--tensor-row-chunk-size` controls the device working set. Completed packed
tensors accumulate only inside the current output shard before its atomic save.

### Archived DGX Spark experiments

The refinement implementations used for the rejected rounding, feedback,
cross-block, selective-precision, and static low-rank experiments are not part
of the production package. Their measurements and exact deployed source
snapshots remain recorded so negative results are not lost.

- [Cross-block reconstruction](docs/QWEN3_8_27B_CROSSBLOCK_PROBE.md)
- [Selective BF16](docs/QWEN3_8_27B_SELECTIVE_BF16.md)
- [Selective FP8](docs/QWEN3_8_27B_SELECTIVE_FP8.md)
- [Static low-rank recovery](docs/QWEN3_8_27B_LOW_RANK_RECOVERY.md)

These are paired 100-item GSM8K samples with identical prompts and decoding,
not reportable benchmark scores. They are retained because they caught a
misleading weight-objective improvement.

| Artifact | Flexible | Strict | Calibration-weighted SQNR |
|---|---:|---:|---:|
| AMD Quark AWQ MXFP4 | 93% | 93% | not available |
| MxWave gamma-proxy MSE | 91% | 90% | 19.04 dB on 272/400 targets |
| MxWave block-Hessian scale selection, 16×512 | 92% | 91% | 19.15 dB |
| MxWave block-Hessian scale selection, 64×512 | 92% | 91% | 19.14 dB |
| MxWave block-local Hessian feedback, 64×512 | 94% | 94% | 19.68 dB |
| MxWave Hessian feedback + 1.125× MSE trust region, 64×512 | 93% | 93% | 19.26 dB |
| MxWave real-RMS scale selection | 90% | 88% | 19.08 dB |
| MxWave + one full Hessian rounding sweep | 86% | 86% | 19.69 dB |

The rounding sweep lost eight paired items to AMD and gained one on both
extractors (two-sided exact McNemar `p=0.039`). This is direct evidence that
optimizing a local calibration quadratic more aggressively can hurt end-to-end
behavior even while its own reconstruction metric improves.
The real-RMS pass also failed to improve over block-Hessian scale selection
(one RMS-only versus four Hessian-only strict paired wins, exact McNemar
`p=0.375`). Block-Hessian scale-only therefore remains the current MxWave
default. The unconstrained feedback path led that pilot, but its
five/six feedback-only wins versus three scale-only wins are not significant
(`p=0.727` flexible, `p=0.508` strict). Against AMD it had four wins and three
losses on both extractors (`p=1.0`). It remains experimental until a larger,
preferably deterministic evaluation confirms the direction; neither local
reconstruction metrics nor this limited sampled screen are treated as proof.
The 1.125× ordinary-MSE trust region reduced the average unweighted SQNR cost
of feedback from 0.44 dB to 0.10 dB while retaining a 0.12 dB improvement in
the calibration-weighted objective over scale-only. It scored 93%/93%: two
paired wins and two losses versus AMD (`p=1.0`), and three/four wins versus
two losses against scale-only (`p=1.0` flexible, `p=0.688` strict). This is a
safer experimental candidate, but it was not retained.

A later held-out selector used a disjoint `32x512` split to accept feedback only
where both training and selection Hessians improved, subject to the 1.125× MSE
trust region. It improved held-out local SQNR, but on the deterministic
61,408-token likelihood pilot it was 0.146% worse than scale-only (paired 95%
bootstrap interval: 0.020% to 0.271% worse). This rejects feedback as the default
and demonstrates why local reconstruction metrics are not promotion criteria.

## Deterministic DGX Spark likelihood screen

Model: `Qwen/Qwen3.8-27B`. Hardware: NVIDIA GB10 / SM121 (DGX Spark). The source
is the pinned Salesforce WikiText-2 raw test parquet at revision
`b08601e04326c79dfdd32d625aee71d232d685c3`, SHA-256
`5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91`.

Rows were joined with two newlines and scored as all 316 non-empty independent
4,096-character windows through vLLM prompt logprobs, one request at a time.
The first token of each window is unscored, leaving 297,199 paired tokens. All
text hashes, token hashes, and token counts matched for every model. MXFP4
artifacts used the same Marlin A16 backend. Intervals are a fixed-seed 100,000
sample paired cluster bootstrap over windows.

This is an API prompt-perplexity comparison, not a literature-compatible
WikiText perplexity number: character windows and context resets differ from
the usual fixed-token sliding-window protocol. The absolute PPL should not be
compared with unrelated model cards; the paired differences below are the
intended result. The hashes, runtime settings, and full-precision values are in
the [machine-readable benchmark summary](benchmarks/qwen3.8-27b-wikitext2-prompt-ppl.json).
The complete pinned commands, configuration, hashes, and replay checks are in the
[Qwen3.8-27B H64 reproducibility record](docs/QWEN3_8_27B_H64_REPRODUCIBILITY.md).

| Artifact | Prompt PPL ↓ | Change vs BF16 | Paired window wins vs BF16 |
|---|---:|---:|---:|
| Qwen3.8-27B BF16 | 7.980864 | reference | — |
| MxWave block-Hessian scale-only, `64x512` calibration | **8.119445** | +1.736% `[+1.333%, +2.110%]` | 65 / 316 |
| AMD Quark-AWQ MXFP4 | 8.187544 | +2.590% `[+2.282%, +2.906%]` | 44 / 316 |

Directly against AMD, MxWave is 0.832% lower in PPL with a paired 95%
interval of 0.531% to 1.227% lower and wins 204/316 windows (two-sided sign test
`p=2.55e-7`). The largest favorable outlier is window 170; removing it still
leaves MxWave 0.684% lower. This supports a real advantage on this deterministic
likelihood screen, not a claim of universal task superiority. The earlier
sampled GSM8K screen was effectively tied (MxWave 92%/91%, AMD 93%/93%).

## Exact next-token distribution screen

The complementary distribution test uses 128 evenly spaced WikiText contexts
of up to 512 tokens and collects all 248,320 next-token log probabilities. Both
MXFP4 candidates are compared against the same BF16 reference distribution.
The complete inputs, per-context values, artifact hashes, and paired bootstrap
are in the
[machine-readable divergence report](benchmarks/qwen3.8-27b-next-token-divergence.json).

| Metric | MxWave H64 | AMD Quark-AWQ MXFP4 |
|---|---:|---:|
| Mean forward KL from BF16 ↓ | **0.049937** | 0.056735 |
| Forward-KL p95 ↓ | **0.175241** | 0.232540 |
| Mean Jensen-Shannon divergence ↓ | **0.010946** | 0.012083 |
| Mean total variation ↓ | **0.078565** | 0.082362 |
| BF16 top-1 agreement ↑ | **93.75%** | 91.41% |

H64's mean forward KL is 0.006798 nats (about 12.0%) lower and it wins 70 of
128 contexts. The paired 10,000-resample interval for `H64 - AMD` is
`[-0.024242, +0.009835]` nats, so this is favorable but not statistically
conclusive evidence. No controlled throughput result is reported; interactive
OpenWebUI observations were intentionally excluded from the quality benchmark.

The publication-ready artifact explanation, runtime caveats, and all three
quality screens are collected in the
[H64 model card](docs/QWEN3_8_27B_H64_MODEL_CARD.md).

## Project structure

```
MxWave/
├── mxwave/
│   ├── calibration.py Activation collectors + safe stats artifact contract
│   ├── calibration_cli.py Real-sequence forward calibration command
│   ├── calibration_stream.py One-decoder-layer-at-a-time calibration runner
│   ├── core.py      MXFP4 constants, MSE-optimal quantize_mxfp4()
│   ├── shard.py     safetensors shard discovery + streaming reads
│   ├── engine.py    GPU-streaming quantization orchestration
│   ├── format.py    input format detection from config.json (not suffix sniffing)
│   ├── rotate.py    Hadamard / random-orthogonal rotation + folding
│   ├── runtime_ir.py Model-independent runtime operations and fused groups
│   ├── runtime_adapters.py Validated architecture-adapter registry
│   ├── adapters/    Architecture-specific runtime graph builders
│   ├── mixed_precision.py Streaming MXFP4/FP8 checkpoint composition
│   ├── precision_budget.py Semantic byte-budget candidate planning
│   ├── verify.py    SQNR, config-coverage verification (verification-first)
│   ├── output.py    compressed-tensors quantization_config assembly + coverage
│   └── cli.py       CLI entry point (wired to the engine)
├── scripts/
│   ├── evaluate_api_perplexity.py Paired OpenAI-API prompt-PPL evaluator
│   └── evaluate_next_token_kl.py Exact next-token distribution evaluator
├── tests/
├── pyproject.toml
└── README.md
```

## Roadmap

- [x] **Streaming engine** — tensor-row streaming, on-device quantize, output assembly
- [x] **Module-keyed calibration** — real mean-absolute, RMS, and block-Hessian
      input statistics from a bounded forward pass
- [x] **Weight-streamed calibration** — meta-device scaffold, CPU hidden states,
      and one resident decoder layer for verified architectures
- [ ] **Calibration-aware default** — choose and run a validated corpus by default
- [ ] **Rotation folding** — emit standard `compressed-tensors` `transform_config`
- [ ] **Auto per-layer precision** — Hessian-trace-driven MXFP4/FP8/BF16 assignment
- [x] **Measured precision-budget experiment** — adapter-driven, fusion-safe FP8
      allocation passed frozen selection, untouched KL, and full-corpus PPL gates;
      retained as an opt-in feature pending broader task validation
- [ ] **Sequential layer reconstruction** — calibrate each block on inputs from the
      already-quantized prefix and compensate errors across full input dimensions
- [x] **Held-out adaptive selection experiment** — evaluated with a disjoint
      selection split, rejected after deterministic PPL regressed, and archived
- [ ] **Automated runtime load-smoke** — structural checks, config coverage, and
      optional SQNR are wired into conversion; vLLM startup remains a recorded
      post-conversion check
- [x] **Deterministic likelihood screen** — paired BF16/MxWave/AMD comparison
      with exact input hashes and clustered uncertainty
- [ ] **Layer-output-aware selection** — select transformations by held-out
      decoder-layer output error rather than weight-local reconstruction alone
- [ ] **Sensitivity-guided mixed precision** — retain only the modules that
      account for most end-to-end loss in MXFP8/BF16

## License

Apache-2.0. This project is an independent clean-room implementation based on
the public OCP MX specification and published quantization research. See
`CONTRIBUTING.md` for the provenance policy.
