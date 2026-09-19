# Contributing to MxWave

## Provenance note

MxWave is an independent clean-room implementation. Its numerical behavior is
derived from the public OCP MX Formats specification and published quantization
research, including AWQ, GPTQ, and QuaRot. Its serialized output follows the
public `compressed-tensors` contract used by vLLM.

The qstream project was consulted only as an ecosystem reference and comparison
target. No source code or prose was copied or adapted from qstream. Historical
experiments that were rejected and removed from the production package remain
documented under `docs/` so their results are not misrepresented as features.

Do not copy implementation code from qstream or any other repository without a
written license grant. Any permitted adaptation must be identified here with
its source, license, files, and nature of the changes. Independent
reimplementations should cite the public specification or paper they follow.

Primary references:

- [OCP Microscaling Formats specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [AWQ](https://arxiv.org/abs/2306.00978)
- [GPTQ](https://arxiv.org/abs/2210.17323)
- [QuaRot](https://arxiv.org/abs/2404.00456)

## Setup

```bash
pip install -e ".[dev]"
```

## Quality gates

```bash
ruff check .            # lint
mypy mxwave             # type check
pytest                  # unit tests
```

All must pass before a PR.

## Adding a quantization feature

1. Implement the primitive in the appropriate module (`rotate.py`,
   `verify.py`, `core.py`, etc.).
2. Add a focused unit test in `tests/`.
3. Update the README roadmap and the benchmark table if it changes quality.

## Adding a runtime adapter

1. Implement `matches(config)` and `build(config, tensor_names)` under
   `mxwave/adapters/`; discovery must use configuration plus checkpoint headers,
   never tensor payloads.
2. Register the adapter in `mxwave/runtime_adapters.py`.
3. Describe runtime-fused linear groups in their packing order and reject missing
   members so precision-aware features cannot silently split a fused operation.
4. Add synthetic conformance tests for matching, graph construction, fused-group
   membership, incomplete checkpoints, and auxiliary decoder stacks when relevant.
5. Record an end-to-end model smoke test before describing the adapter as verified.

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
