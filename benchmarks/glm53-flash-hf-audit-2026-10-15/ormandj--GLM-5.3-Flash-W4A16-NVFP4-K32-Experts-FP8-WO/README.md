---
base_model: zai-org/GLM-5.3-Flash-BF16
library_name: transformers
license: mit
pipeline_tag: image-text-to-text
tags:
  - glm5_next
  - safetensors
  - modelopt
  - fp8
  - nvfp4
  - sglang
---

# GLM-5.3-Flash, quantized for two RTX PRO 6000 Blackwell GPUs

A mixed-precision quantization of
[`zai-org/GLM-5.3-Flash-BF16`](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16)
(revision `f12e0fe1f6b2ea274c11a569582edfd99d993c5e`) that fits two 96 GB
SM120 GPUs with a 499,712-token context, four concurrent requests,
speculative decoding and vision input. It scores 96.9% on GSM8K (1,278 of
1,319, zero-shot, greedy), within repetition spread of the BF16 source
served the same way.

It is served by the matching
[`ormandj/sglang-glm53-flash-sm120`](https://github.com/ormandj/sglang-glm53-flash-sm120)
image. It is not an upstream Z.ai, ModelOpt or SGLang release.

## Run it

You need Linux x86_64, two visible SM120 GPUs, Docker with the NVIDIA
Container Toolkit and about 170 GB of disk.

```bash
pip install -U huggingface_hub
export MODEL_DIR=/srv/models/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO
HF_XET_HIGH_PERFORMANCE=1 hf download ormandj/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO --local-dir "$MODEL_DIR"

git clone https://github.com/ormandj/sglang-glm53-flash-sm120
cd sglang-glm53-flash-sm120
export IMAGE=ghcr.io/ormandj/sglang-glm53-flash-sm120:v0.1.1
export CACHE_DIR=/srv/cache/sglang-glm53-flash-sm120-v55
./examples/serve-glm53-flash.sh
```

The first boot compiles kernels into `CACHE_DIR` and takes about 10 to 20
minutes; later boots take about 8. The server is OpenAI-compatible on port
8000 with model name `glm-5.3-flash`. Measured throughput, capacity and
quality are in the repository's
[README](https://github.com/ormandj/sglang-glm53-flash-sm120#what-to-expect).

## What is quantized

- Routed experts in layers 3 through 45: ModelOpt W4A16 NVFP4, signed E2M1
  packed weights, E4M3FN scales over K=32, BF16 activations.
- 317 attention and shared-expert projections in layers 0 through 44: E4M3
  block-FP8 weight-only with FP32 inverse scales over 128x128 blocks.
- Kept in source precision: the layer 45 MTP draft weights (quantizing them
  silently collapses speculative acceptance to zero), the DSA indexer, the
  vision tower and projector, embeddings, LM head, routers, norms and
  ineligible linears.
- Tensor payload: 177,995,252,856 bytes in 90 safetensors shards.

Reconstruction against the BF16 source: routed experts aggregate relative L2
0.0851 (minimum matrix cosine 0.9949); FP8 tier aggregate relative L2 0.0225
(minimum cosine 0.9996). The full tensor selection, producer pins,
per-tensor measurements and per-shard hashes are in
`quantization-manifest.json`, `reconstruction-metrics-*.json` and
`model-sha256.json`.

## Runtime compatibility

The checkpoint uses the `modelopt_mixed` metadata contract and needs the
pinned SGLang and FlashInfer integration in the image above
(`--quantization modelopt_mixed`, shared-expert fusion disabled). Upstream
Transformers, vLLM and SGLang releases do not load it correctly today.

## Integrity

The publishing build refuses to upload unless its model files reproduce the
served artifact exactly:

- `config.json`: `00cf7f0ae5cbc1ca1593376730aa52dd918b59cab28552c61e9d8e3fd7a19d20`
- `model.safetensors.index.json`: `944bb41dd3dc2b6fde6b37465b9d9f7743ea89002302b7c38a87e0242270554d`
- `model-sha256.json`: `655a9b802f0629d1067f2350ee4e6d4c288c642846f2b1955760000ebf9bdfa5`

The last file holds the SHA-256 digest of every model shard.

## License and attribution

The source checkpoint is MIT-licensed. See the included license and the
upstream model card for the original model's usage guidance, limitations
and citation.
