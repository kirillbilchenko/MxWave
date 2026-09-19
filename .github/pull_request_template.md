## Summary

Describe the problem, the chosen solution, and any important tradeoffs.

## Validation

- [ ] `ruff check .`
- [ ] `mypy mxwave`
- [ ] `pytest`
- [ ] Documentation is updated when behavior or commands change.
- [ ] New public behavior has focused tests.
- [ ] Quantization changes include deterministic, paired evaluation or are clearly
      marked experimental.
- [ ] No model weights, access tokens, generated checkpoints, or unrelated artifacts
      are included.

## Benchmark impact

For changes affecting quantization quality, checkpoint format, memory, or runtime,
record the model, artifact hashes, hardware, runtime versions, evaluation inputs,
quality metrics, throughput, peak memory, and uncertainty or paired deltas. Otherwise,
write `Not applicable`.
