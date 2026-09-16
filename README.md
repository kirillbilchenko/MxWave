# mxstream

**GPU-streaming, calibration-aware, quality-oriented MXFP4 quantization** for
models larger than a single machine, targeting vLLM's `compressed-tensors`
`mxfp4-pack-quantized` format on Blackwell-class GPUs (e.g. NVIDIA DGX Spark).

> **Thesis:** a GPU-streaming, activation-aware MXFP4 engine that quantizes
> models bigger than any single machine, emits a standard, *verified* vLLM
> checkpoint, and measurably beats plain round-to-nearest MXFP4 — without a
> model fork.

## Why this repo exists

Stock MXFP4 tooling (e.g. `llm-compressor`'s `mxfp4-pack-quantized`) uses a
plain min/max round-to-nearest observer. That leaves quality on the table.
`mxstream` is built around three ideas that go beyond RTN:

1. **MSE-optimal, calibration-aware scale selection** — an explicit exponent
   candidate set per block, chosen by minimum reconstruction error and optionally
   weighted by real activation moments or a block Hessian.
2. **Rotation-based outlier suppression (QuaRot family)** — a fused Hadamard
   rotation makes activation outliers uniform before quantization, then is
   folded into LayerNorm + next-layer weights so inference is free.
3. **Verification-first** — the output is recomputed and validated (SQNR,
   config coverage, load smoke), never trusted-by-inheritance from the source.

## Status

🚧 **Working core + streaming engine.** Module-keyed activation calibration,
shard discovery, on-device quantization, output assembly, and strict output
verification are implemented and unit-tested. A paired 297k-token DGX Spark
likelihood screen found the recommended block-Hessian scale-only artifact 0.83%
lower in perplexity than AMD Quark-AWQ MXFP4 and 1.74% above BF16. The exact
protocol and uncertainty are recorded below; the earlier 100-item task screens
remain diagnostic rather than reportable benchmark scores.

## Install

```bash
cd mxstream
pip install -e ".[dev,calibrate]"
mxstream-calibrate --help
mxstream-quantize --help
```

## Activation calibration and quantization

```bash
mxstream-calibrate \
    --model-dir /path/to/float-model \
    --corpus /path/to/calibration.jsonl \
    --output /path/to/activation-stats.safetensors \
    --policy qwen3.8-27b-compatible \
    --statistics mean-abs,rms,block-hessian \
    --num-sequences 16 \
    --sequence-length 512 \
    --weight-loading streaming

mxstream-quantize \
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
objective. A positive `--hessian-rounding-sweeps` additionally refines each
E2M1 code against that same expected output-error objective after selecting
the block scale. That rounding path is research-only: one full-strength sweep
improved calibration-weighted SQNR but regressed the first downstream pilot,
so the quality-oriented default remains zero sweeps. The opt-in
`--hessian-error-feedback` path instead propagates each quantization error to
the remaining coordinates through a damped inverse-Hessian factor, with static
within-block activation ordering. It restores the original column order before
packing, so the emitted checkpoint remains standard MXFP4; it is still a
bounded block-local approximation and must pass downstream validation before
promotion. `--hessian-feedback-max-mse-ratio` can impose a generic per-row,
per-block trust region: feedback is retained only where it lowers Hessian loss
without increasing ordinary MSE beyond the requested ratio. The calibration
file must cover every selected target exactly;
mxstream never silently mixes real statistics with gamma or unweighted MSE.

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

Target matrices are read from safetensors and quantized in bounded row ranges;
`--tensor-row-chunk-size` controls the device working set. Completed packed
tensors accumulate only inside the current output shard before its atomic save.

### Initial DGX Spark screen

These are paired 100-item GSM8K samples with identical prompts and decoding,
not reportable benchmark scores. They are retained because they caught a
misleading weight-objective improvement.

| Artifact | Flexible | Strict | Calibration-weighted SQNR |
|---|---:|---:|---:|
| AMD Quark AWQ MXFP4 | 93% | 93% | not available |
| mxstream gamma-proxy MSE | 91% | 90% | 19.04 dB on 272/400 targets |
| mxstream block-Hessian scale selection, 16×512 | 92% | 91% | 19.15 dB |
| mxstream block-Hessian scale selection, 64×512 | 92% | 91% | 19.14 dB |
| mxstream block-local Hessian feedback, 64×512 | 94% | 94% | 19.68 dB |
| mxstream Hessian feedback + 1.125× MSE trust region, 64×512 | 93% | 93% | 19.26 dB |
| mxstream real-RMS scale selection | 90% | 88% | 19.08 dB |
| mxstream + one full Hessian rounding sweep | 86% | 86% | 19.69 dB |

The rounding sweep lost eight paired items to AMD and gained one on both
extractors (two-sided exact McNemar `p=0.039`). This is direct evidence that
optimizing a local calibration quadratic more aggressively can hurt end-to-end
behavior even while its own reconstruction metric improves.
The real-RMS pass also failed to improve over block-Hessian scale selection
(one RMS-only versus four Hessian-only strict paired wins, exact McNemar
`p=0.375`). Block-Hessian scale-only therefore remains the current mxstream
default. The unconstrained feedback path is the current pilot leader, but its
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
safer experimental candidate, not evidence of superiority.

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
| mxstream block-Hessian scale-only, `64x512` calibration | **8.119445** | +1.736% `[+1.333%, +2.110%]` | 65 / 316 |
| AMD Quark-AWQ MXFP4 | 8.187544 | +2.590% `[+2.282%, +2.906%]` | 44 / 316 |

Directly against AMD, mxstream is 0.832% lower in PPL with a paired 95%
interval of 0.531% to 1.227% lower and wins 204/316 windows (two-sided sign test
`p=2.55e-7`). The largest favorable outlier is window 170; removing it still
leaves mxstream 0.684% lower. This supports a real advantage on this deterministic
likelihood screen, not a claim of universal task superiority. The earlier
sampled GSM8K screen was effectively tied (mxstream 92%/91%, AMD 93%/93%).

## Project structure

```
mxstream/
├── mxstream/
│   ├── calibration.py Activation collectors + safe stats artifact contract
│   ├── calibration_cli.py Real-sequence forward calibration command
│   ├── calibration_stream.py One-decoder-layer-at-a-time calibration runner
│   ├── core.py      MXFP4 constants, MSE-optimal quantize_mxfp4()
│   ├── shard.py     safetensors shard discovery + streaming reads
│   ├── engine.py    GPU-streaming quantization orchestration
│   ├── format.py    input format detection from config.json (not suffix sniffing)
│   ├── rotate.py    Hadamard / random-orthogonal rotation + folding
│   ├── verify.py    SQNR, config-coverage verification (verification-first)
│   ├── output.py    compressed-tensors quantization_config assembly + coverage
│   └── cli.py       CLI entry point (wired to the engine)
├── scripts/
│   └── evaluate_api_perplexity.py Paired OpenAI-API prompt-PPL evaluator
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
- [ ] **Sequential layer reconstruction** — calibrate each block on inputs from the
      already-quantized prefix and compensate errors across full input dimensions
- [x] **Held-out adaptive selection experiment** — implemented with a disjoint
      selection split; rejected as the default after deterministic PPL regressed
- [ ] **Robust rounding optimization** — block-local feedback is experimental;
      full-strength coordinate rounding failed its pilot
- [ ] **Verification pass** — SQNR + config coverage + load-smoke wired into CLI
- [x] **Deterministic likelihood screen** — paired BF16/mxstream/AMD comparison
      with exact input hashes and clustered uncertainty
- [ ] **Layer-output-aware selection** — select transformations by held-out
      decoder-layer output error rather than weight-local reconstruction alone
- [ ] **Sensitivity-guided mixed precision** — retain only the modules that
      account for most end-to-end loss in MXFP8/BF16

## License

Apache-2.0 (pending). This project is a clean-room, quality-oriented build
inspired by the open MXFP4 ecosystem; see `CONTRIBUTING.md` for provenance.
