# Kolibri-1 native MXFP4 GGUF

This exporter converts the selected softer-RMS Kolibri checkpoint into one GGUF
without another quantization pass over the routed experts. It retains their
MXFP4 codes and E8M0 scales exactly, dequantizes the FP8 attention and shared
experts to BF16, and preserves F32 router and normalization parameters. The
planned full-model tensor payload is 45,418,634,240 bytes (42.30 GiB), with a
small additional metadata header.

## Current status

The exporter, independent GGUF reader check, and CUDA inference harness are
implemented. The local suite passes 314 tests, lint, and strict type checks. A
synthetic Kolibri fixture loads and generates through native MXFP4 on Spark
SM121. Its frozen-token likelihood equals the analytically expected `ln(320)`,
and its full-vocabulary probability distribution is normalized.

Full-model conversion and quality measurements are pending. The published
vLLM checkpoint is experimental: its mean KL is 0.05514 against a target of
0.03. Those results do not qualify the GGUF or its separate inference backend.

## Runtime

Kolibri needs the external licensed architecture port described in
`CONTRIBUTING.md`. The reproducible build uses llama.cpp revision
`edd6e2bbdad5930899a93db8fa73c3b61c7b9bcc`, the recorded Kolibri patch, CUDA 13,
and architecture 121. Stock llama.cpp and Ollama compatibility has not been
established.

Put the patched llama.cpp directory and its `gguf-py` directory on `PYTHONPATH`,
along with MxWave. The exporter uses the port only for GGUF metadata and
tokenizer conversion. It reads and converts tensor payloads independently.

## Conversion

```bash
python scripts/export_kolibri_gguf.py CHECKPOINT OUTPUT.gguf --dry-run
python scripts/export_kolibri_gguf.py CHECKPOINT OUTPUT.gguf
```

The source checkpoint must include its MxWave manifest. Conversion validates
the declared quantization groups, every tensor shape, and complete config
coverage. It hashes all 116,303 source payloads against that manifest while
streaming, orders experts numerically, and checks all 903 output tensors with
an independent GGUF reader. The final file and JSON receipt appear only after
those checks pass. The receipt contains whole-file and per-tensor SHA-256
digests. An existing output or partial output is rejected.

The exporter needs the planned output size plus a 3 GiB disk reserve. It keeps
working memory bounded by expert matrices and FP8 row chunks; it does not
construct a full BF16 checkpoint.

## Full-model measurement

`scripts/kolibri_gguf_metrics.cpp` uses the public llama.cpp API to score the
same frozen token IDs as the native release evaluation. It saves likelihoods
for all 48 passages, 48 full-vocabulary distributions, six greedy behavior
checks, and three forced 128-token concurrency-one throughput repetitions.
Concurrency-four throughput is not measured by this harness.

`scripts/run_kolibri_gguf_check.py --root EXPERIMENT_ROOT` verifies the complete
GGUF receipt, runs the compiled harness, and applies the original frozen
comparison against the official FP8 reference. The quality target remains
unchanged. Completion of an evaluation is separate from passing that target.

The Spark workflow keeps the selected checkpoint, calibration, evaluation
data, and private experiment archive. After anonymous verification of the
public native release, it removes only the BF16 source, official FP8 reference,
and rejected parent checkpoint weight files to make room for the GGUF and the
previously deleted models. Old-model restoration continues even if GGUF
conversion or inference fails.
