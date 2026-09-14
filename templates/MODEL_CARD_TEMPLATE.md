<!--
  mxstream MXFP4 model-card template.
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
[**mxstream**](https://github.com/{{YOUR_USER}}/mxstream) — a GPU-streaming,
**calibration-aware** MXFP4 engine. Every Linear layer is quantized to MXFP4;
quality-sensitive layers (embeddings, lm_head, norms) stay {{REST_PRECISION}}.
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
| All Linear layers (`*.weight`) | **MXFP4** (4-bit) | the only place worth the size win |
| Embeddings, lm_head{{VISION_BIT}}, norms | **{{REST_PRECISION}}** | sensitive / runs on every token — kept lossless from the source |
<!-- VISION_BIT: ", vision encoder, projector" if a VLM, else "". -->

## Quantization method (beyond round-to-nearest)

mxstream does **not** use a plain min/max RTN observer. Per block of 32, it
selects the scale that minimizes reconstruction error, choosing among candidate
exponents (MSE-optimal), optionally weighted by real activation statistics
(AWQ-style gamma) or a block Hessian (GPTQ-style). Optionally, a fused Hadamard
rotation (QuaRot family) makes activation outliers uniform *before*
quantization, then is folded into LayerNorm + next-layer weights so inference
stays free. The result is a standard `mxfp4-pack-quantized` checkpoint with no
model fork.

## Quality & faithfulness

We report **deterministic, reproducible** faithfulness metrics rather than noisy
downstream task scores. (Task evals served over vLLM with continuous batching
are not bitwise-deterministic at `temperature=0` — batch-dependent reductions —
so small-sample accuracies are noisy and we don't quote them.)

| Metric | Result | What it shows |
|---|---|---|
| **Perplexity** (clean English) | **{{PPL}}** | language modeling intact — a broken quant lands in the hundreds |
| **Layer SQNR** | **≈ {{SQNR}} dB** | reconstruction error is just the unavoidable 4-bit rounding (MXFP4 vs the {{SOURCE_FORMAT}} source) |

Why this is enough to trust the checkpoint:

- **The math path is verified.** Per-tensor SQNR is recomputed against the
  source; the only residual is the ~{{SQNR}} dB 4-bit rounding on the Linear
  GEMMs. Everything else is bit-identical {{REST_PRECISION}}.
- **Config coverage is verified.** Every real Linear in the checkpoint is
  targeted (or explicitly ignored) — none silently loads unquantized.
- **Same format, better quality.** MXFP4 scales are MSE-optimal + optionally
  rotation-folded, so PPL is measurably closer to the base than plain RTN MXFP4
  at the same size.

PPL script: `evals/eval_ppl.py` in the mxstream repo.

## Fidelity, footprint & provenance

<!-- OPTIONAL (VLM): -->
- **Vision is untouched:** the vision encoder + projector stay **{{REST_PRECISION}}**
  (bit-identical), so image capability equals the base model. Verified working end-to-end.
- **Footprint:** ~{{WEIGHTS_GIB}} GiB of weights; fits a single ≥{{MIN_GPU}} GB GPU
  (e.g. DGX Spark, 128 GB).
- **Provenance:** built with [mxstream](https://github.com/{{YOUR_USER}}/mxstream)
  `@{{COMMIT}}` from the `{{SOURCE_RELEASE}}` release.

## Serving with vLLM

Targets {{VLLM_IMAGE}}. The `config.json` here targets vLLM's *merged* runtime modules
(`qkv_proj`, `gate_up_proj`) so the fused linears load quantized.

```bash
docker run -d --name {{CONTAINER}} --gpus all --privileged --ipc=host -p 8000:8000 \
  -e VLLM_MXFP4_USE_MARLIN=1 \
  -v $(pwd):/model \
{{PATCH_MOUNTS}} \
  {{VLLM_IMAGE}} /model \
  --served-model-name {{SERVED_NAME}} \
{{SERVE_FLAGS}} \
  --gpu-memory-utilization 0.97 --enforce-eager \
  --linear-backend marlin --trust-remote-code
```

<!-- OPTIONAL (only if the checkpoint needs a runtime patch — see vllm_patch/):
### Runtime patch ([`vllm_patch/`](./vllm_patch))

{{PATCH_REASON_ONE_PARAGRAPH}} See [`vllm_patch/README.md`](./vllm_patch/README.md).
-->

## How it was made

```bash
mxstream-quantize \
  --model_dir <{{SOURCE_RELEASE}}> \
  --output_dir ./{{OUTPUT_DIR}} \
  --workers 8 \
  --device cuda \
  --rotation {{ROTATION}} \
  --verify
```

`detect_input_format` auto-detects the source's {{SOURCE_FORMAT}}, streams each shard to
the GPU, quantizes the Linear weights to MXFP4 (MSE-optimal, optionally rotation-folded),
passes the {{REST_PRECISION}} remainder through, and assembles a verified, drop-in
`config.json` + safetensors index.

## License

Inherits the {{LICENSE_BLURB}} from the base model. This is a derivative (quantized) work
of {{MODEL_NAME}}.

<!-- OPTIONAL (when prepending to upstream card):
---

# Original model card

{{PASTE UPSTREAM README HERE}}
-->
