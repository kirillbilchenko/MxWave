# AGENTS.md — guidance for AI coding agents

## What this project is

MxWave is a bounded-memory, calibration-aware, quality-oriented MXFP4
quantization engine for LLMs, targeting vLLM's `compressed-tensors`
`mxfp4-pack-quantized` format on Blackwell-class GPUs (e.g. NVIDIA DGX Spark).

It is a clean-room build inspired by the open MXFP4 ecosystem. Core math follows
the public OCP MX spec and published methods (AWQ, GPTQ, QuaRot). Do not copy
code from qstream (or any repo) without a written license grant — record any
adapted code in `CONTRIBUTING.md`.

## Build / test commands

```bash
pip install -e ".[dev]"   # install with dev deps (pytest, ruff, mypy)
ruff check .              # lint — must pass
mypy mxwave             # type check — must pass (strict)
pytest                    # unit tests — must pass
```

## Architecture

- `mxwave/core.py` — MXFP4 E2M1 constants, `quantize_mxfp4()` with MSE-optimal
  candidate scale selection, optional gamma/Hessian weighting,
  `dequant_mxfp4()` round-trip.
- `mxwave/format.py` — input format detection from `config.json`
  (`quantization_config`), NOT tensor-name suffix sniffing. This is a core
  differentiator: the FP8 ecosystem has ≥3 incompatible packings.
- `mxwave/rotate.py` — experimental Hadamard / random-orthogonal rotation
  primitives; rotation is not wired into checkpoint emission.
- `mxwave/verify.py` — verification-first contract: `sqnr()`,
  `verify_config_coverage()`.
- `mxwave/cli.py` — thin CLI scaffold (`mxwave-quantize`).

## Conventions

- Python ≥ 3.12, strict typing. Use `from __future__ import annotations`.
- Line length 100 (ruff). Docstrings on all public functions.
- Quantization math must stay device-agnostic (tensor.device aware) — the
  streaming engine will move tensors between CPU disk and GPU VRAM.
- Tests live in `tests/`, one file per module, focused on the public contract.

## Pitfalls

- MXFP4 scales are e8m0 biased exponents (uint8, bias 127). Never treat the
  packed bytes as raw floats.
- Block size is 32 for MXFP4; last dim of a weight must be divisible by it.
- `verify_config_coverage` must be run before emitting a config — a Linear that
  is neither targeted nor ignored loads unquantized and can silently produce
  zero output.
- Rotation must be folded into LayerNorm + next-layer weights at output time for
  inference to be free; the emitted config should carry `transform_config`.

## Benchmarking (DGX Spark)

Benchmarks belong in the README table: same model, same `mxfp4-pack-quantized`
format, on SM100/SM121. Record PPL, task accuracies, SQNR, throughput. Prefer
deterministic metrics (perplexity) over noisy small-sample accuracies.
