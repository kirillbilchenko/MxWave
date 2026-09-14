# Contributing to mxstream

## Provenance note

`mxstream` is a quality-oriented, GPU-streaming build in the MXFP4 space. The
core math follows the public MX spec (OCP) and published methods (AWQ, GPTQ,
QuaRot). If you are adapting code from a source repo, get a written license
grant first and record it here.

## Setup

```bash
pip install -e ".[dev]"
```

## Quality gates

```bash
ruff check .            # lint
mypy mxstream           # type check
pytest                 # unit tests
```

All must pass before a PR.

## Adding a quantization feature

1. Implement the primitive in the appropriate module (`rotate.py`,
   `verify.py`, `core.py`, etc.).
2. Add a focused unit test in `tests/`.
3. Update the README roadmap and the benchmark table if it changes quality.

## Benchmarking on the DGX Spark

Benchmarks belong in the README table (same model, same `mxfp4-pack-quantized`
format). Record: model, hardware (SM100/SM121), PPL, task accuracies, SQNR,
throughput. Prefer deterministic metrics (e.g. `eval_ppl.py`-style perplexity)
over noisy small-sample accuracies.

## License

Apache-2.0. By contributing you agree to license your contribution under the
same terms.
