# Contributing to mxstream

## Provenance note

`mxstream` is a quality-oriented, GPU-streaming build in the MXFP4 space. The
core math follows the public MX spec (OCP) and published methods (AWQ, GPTQ,
QuaRot). If you are adapting code from a source repo, get a written license
grant first and record it here.

The block-Hessian coordinate rounding pass is a clean-room derivation of exact
single-coordinate minimization of the recorded quadratic reconstruction loss;
it was not adapted from qstream or another implementation.

The block-local Hessian error-feedback path is a clean-room implementation from
the GPTQ paper's published Algorithm 1 and the MR-GPTQ paper's published static
activation-order description. No implementation code was copied from either
project or from qstream.

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

For API likelihood comparisons, use `scripts/evaluate_api_perplexity.py` and
pin the corpus revision plus file hash. Compare identical window hashes, token
hashes, token counts, serving backend, and concurrency. Report paired
uncertainty over windows and describe the window/context-reset protocol; do not
present API prompt perplexity as directly comparable to a literature PPL that
uses different token windows or stride.

## License

Apache-2.0. By contributing you agree to license your contribution under the
same terms.
