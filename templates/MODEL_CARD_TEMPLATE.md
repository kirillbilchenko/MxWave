<!--
  MxWave MXFP4 model-card template.
  Replace every {{PLACEHOLDER}}. Delete <!-- OPTIONAL --> blocks that don't apply.
  Philosophy: lead with deterministic faithfulness metrics (PPL, SQNR), not noisy
  downstream task scores. State plainly what changed (weights) vs. what is
  bit-identical to the source (everything else).
-->
---
license: {{LICENSE}}                      # e.g. apache-2.0  /  other
# license_name: {{LICENSE_NAME}}          # only if license: other
# license_link: LICENSE                    # only if license: other
library_name: transformers
pipeline_tag: {{PIPELINE_TAG}}            # e.g. text-generation / image-text-to-text
base_model: {{BASE_MODEL_HF}}             # e.g. meta-llama/Llama-3.1-8B
base_model_relation: quantized
tags:
  - mxfp4
  - compressed-tensors
  - quantized
  - vllm
# - multimodal        # keep if the base is a VLM
---

# {{MODEL_NAME}} — MXFP4 (mixed precision)

A 4-bit **MXFP4** quantization of [{{MODEL_NAME}}]({{BASE_MODEL_URL}}), produced with
[**MxWave**](https://github.com/kirillbilchenko/MxWave) — a bounded-memory,
**calibration-aware** MXFP4 engine. Projections selected by the recorded policy
are quantized to MXFP4; explicitly ignored tensors stay {{REST_PRECISION}}.
<!-- REST_PRECISION: "BF16" (most models) or the source dtype. -->
<!-- OPTIONAL (only when prepending to the upstream card):
**The original model card follows in full [below](#original-model-card).** -->

| | |
|---|---|
| **Size** | **{{SIZE}}** (down from {{SOURCE_SIZE}} {{SOURCE_FORMAT}} source, ~{{PERCENT}}%) |
| **Format** | compressed-tensors `mxfp4-pack-quantized` |
| **Base** | {{BASE_ONE_LINER}} |
<!-- BASE_ONE_LINER: params, architecture (dense / MoE shape), layers, context,
     anything architecturally notable (vision, MTP, …). -->

## What is quantized to what

| Component | Precision | Why |
|---|---|---|
| Selected projection weights | **MXFP4** (4-bit) | primary source of the size reduction |
| Embeddings, lm_head{{VISION_BIT}}, norms | **{{REST_PRECISION}}** | sensitive / runs on every token — kept lossless from the source |
<!-- VISION_BIT: ", vision encoder, projector" if a VLM, else "". -->

## Quantization method (beyond round-to-nearest)

MxWave does **not** use a plain min/max RTN observer. Per block of 32, it
selects the scale that minimizes reconstruction error, choosing among candidate
exponents (MSE-optimal), optionally weighted by real activation statistics
(activation magnitudes) or a block Hessian. Values are then assigned to their
nearest E2M1 codes. The result is a standard `mxfp4-pack-quantized` checkpoint
with no runtime model fork.

## Quality & faithfulness

We report **deterministic, reproducible** faithfulness metrics rather than noisy
downstream task scores. (Task evals served over vLLM with continuous batching
are not bitwise-deterministic at `temperature=0` — batch-dependent reductions —
so small-sample accuracies are noisy and we don't quote them.)

| Metric | Result | What it shows |
|---|---|---|
| **Perplexity** (clean English) | **{{PPL}}** | language modeling intact — a broken quant lands in the hundreds |
| **Layer SQNR** | **≈ {{SQNR}} dB** | reconstruction error is just the unavoidable 4-bit rounding (MXFP4 vs the {{SOURCE_FORMAT}} source) |

What these checks establish:

- **The math path is verified.** Per-tensor SQNR is recomputed against the
  source for bounded row samples. Explicit passthrough tensors are copied from
  the source checkpoint.
- **Config coverage is verified.** Every real Linear in the checkpoint is
  targeted (or explicitly ignored) — none silently loads unquantized.
- **End-to-end quality is measured separately.** Reconstruction metrics do not
  replace paired perplexity or downstream evaluation.

PPL script: `scripts/evaluate_api_perplexity.py` in the MxWave repo. Record
the corpus revision/hash, windowing protocol, token count, serving backend, and
paired baseline reports; an absolute PPL is not comparable when those differ.

## Fidelity, footprint & provenance

<!-- OPTIONAL (VLM): -->
- **Vision is untouched:** the vision encoder + projector stay **{{REST_PRECISION}}**
  (bit-identical). This does not establish end-to-end image quality unless a
  multimodal evaluation is reported below.
- **Footprint:** ~{{WEIGHTS_GIB}} GiB of weights; fits a single ≥{{MIN_GPU}} GB GPU
  (e.g. DGX Spark, 128 GB).
- **Provenance:** built with [MxWave](https://github.com/kirillbilchenko/MxWave)
  `@{{COMMIT}}` from the `{{SOURCE_RELEASE}}` release.

## Serving with vLLM

Targets {{VLLM_IMAGE}}. The `config.json` here targets vLLM's *merged* runtime modules
(`qkv_proj`, `gate_up_proj`) so the fused linears load quantized.

```bash
docker run -d --name {{CONTAINER}} --gpus all --ipc=host -p 8000:8000 \
  -e VLLM_MXFP4_USE_MARLIN=1 \
  -v $(pwd):/model \
{{PATCH_MOUNTS}} \
  {{VLLM_IMAGE}} /model \
  --served-model-name {{SERVED_NAME}} \
{{SERVE_FLAGS}} \
  --gpu-memory-utilization {{GPU_MEMORY_UTILIZATION}} \
  --linear-backend marlin --trust-remote-code
```

<!-- On unified-memory systems such as DGX Spark, CPU and GPU allocations share
     the same 128 GB pool. Start around 0.45 and raise only while monitoring
     MemAvailable; 0.80 can starve the host even when the model itself fits. -->

<!-- OPTIONAL (only if the checkpoint needs a runtime patch — see vllm_patch/):
### Runtime patch ([`vllm_patch/`](./vllm_patch))

{{PATCH_REASON_ONE_PARAGRAPH}} See [`vllm_patch/README.md`](./vllm_patch/README.md).
-->

## How it was made

```bash
mxwave-calibrate \
  --model-dir <{{SOURCE_RELEASE}}> \
  --corpus {{CALIBRATION_CORPUS}} \
  --output ./activation-stats.safetensors \
  --policy {{POLICY}} \
  --statistics {{CALIBRATION_OBJECTIVE}} \
  --num-sequences {{CALIBRATION_SEQUENCES}} \
  --sequence-length {{CALIBRATION_SEQUENCE_LENGTH}} \
  --weight-loading streaming \
  --device cuda

mxwave-quantize \
  --model-dir <{{SOURCE_RELEASE}}> \
  --output-dir ./{{OUTPUT_DIR}} \
  --policy {{POLICY}} \
  --method mse \
  --scale-percentile 99.5 \
  --mse-clip-depth 4 \
  --tensor-row-chunk-size 1024 \
  --activation-stats ./activation-stats.safetensors \
  --calibration-objective {{CALIBRATION_OBJECTIVE}} \
  --device cuda \
  --source-repository {{BASE_MODEL_HF}} \
  --source-revision {{SOURCE_REVISION}} \
  --verify-sqnr \
  --sqnr-rows 16
```

`detect_input_format` reads the source format from `config.json`. MxWave processes
selected weight rows in bounded chunks, quantizes them to MXFP4, passes the
{{REST_PRECISION}} remainder through, and assembles a verified, drop-in
`config.json` + safetensors index.

## License

Inherits the {{LICENSE_BLURB}} from the base model. This is a derivative (quantized) work
of {{MODEL_NAME}}.

<!-- OPTIONAL (when prepending to upstream card):
---

# Original model card

{{PASTE UPSTREAM README HERE}}
-->
