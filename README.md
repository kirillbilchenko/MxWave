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

1. **MSE-optimal, calibration-aware scale selection** — three candidate
   exponents per block, chosen by min reconstruction error, optionally weighted
   by real activation statistics (AWQ-style) or a block Hessian (GPTQ-style).
2. **Rotation-based outlier suppression (QuaRot family)** — a fused Hadamard
   rotation makes activation outliers uniform before quantization, then is
   folded into LayerNorm + next-layer weights so inference is free.
3. **Verification-first** — the output is recomputed and validated (SQNR,
   config coverage, load smoke), never trusted-by-inheritance from the source.

## Status

🚧 **Early scaffold.** The package skeleton, core quantization math, format
detection, rotation primitives, and verification helpers exist and are
unit-tested. The streaming engine (shard streaming, on-device quantization,
output assembly) is under construction.

## Install

```bash
cd mxstream
pip install -e ".[dev]"
mxstream-quantize --help
```

## Quick start (once the streaming engine lands)

```bash
mxstream-quantize \
    --model_dir /path/to/model \
    --output_dir /path/to/output \
    --workers 8 \
    --device cuda \
    --rotation hadamard \
    --verify
```

## Benchmark table (to be filled on the DGX Spark)

Same model, same `mxfp4-pack-quantized` format, measured on Blackwell (SM100).

| Engine | Method | Wiki PPL ↓ | GSM8K ↑ | MMLU ↑ | SQNR dB ↑ | Tok/s ↑ |
| ------ | ------ | ---------- | ------- | ------ | --------- | ------- |
| llm-compressor | RTN MXFP4 | — | — | — | — | — |
| NVIDIA ModelOpt | NVFP4 | — | — | — | — | — |
| mxstream | MSE MXFP4 | — | — | — | — | — |
| mxstream | + rotation | — | — | — | — | — |

## Project structure

```
mxstream/
├── mxstream/
│   ├── core.py      MXFP4 constants, MSE-optimal quantize_mxfp4()
│   ├── format.py    input format detection from config.json (not suffix sniffing)
│   ├── rotate.py    Hadamard / random-orthogonal rotation + folding
│   ├── verify.py    SQNR, config-coverage verification (verification-first)
│   └── cli.py       CLI entry point
├── tests/
├── pyproject.toml
└── README.md
```

## Roadmap

- [ ] **Streaming engine** — shard streaming through GPU VRAM, on-device quantize
- [ ] **Calibration-aware default** — streaming forward pass produces activation
      stats by default (no opt-in)
- [ ] **Rotation folding** — emit standard `compressed-tensors` `transform_config`
- [ ] **Auto per-layer precision** — Hessian-trace-driven MXFP4/FP8/BF16 assignment
- [ ] **Rounding optimization** — per-element rounding decision (AdaRound-style)
- [ ] **Verification pass** — SQNR + config coverage + load-smoke wired into CLI
- [ ] **Benchmark table** — populate the DGX Spark comparison above

## License

Apache-2.0 (pending). This project is a clean-room, quality-oriented build
inspired by the open MXFP4 ecosystem; see `CONTRIBUTING.md` for provenance.
