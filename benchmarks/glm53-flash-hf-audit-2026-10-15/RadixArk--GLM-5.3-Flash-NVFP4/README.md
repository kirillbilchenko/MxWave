---
pipeline_tag: image-text-to-text
base_model:
- zai-org/GLM-5.3-Flash-BF16
license: mit
library_name: Model Optimizer
language:
- en
- zh
tags:
- RadixArk
- ModelOpt
- GLM-5.3-Flash
- quantized
- FP4
- fp4
- NVFP4
---

# Model Overview
## Description:

The RadixArk GLM-5.3-Flash-NVFP4 model is a quantized version of [zai-org/GLM-5.3-Flash-BF16](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16), pinned to source revision `a6c167b6`. RadixArk converted the 300B-class hybrid Mixture-of-Experts model to mixed-precision NVFP4 W4A4 using [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer). The conversion uses plain abs-max scaling with 256 tensor-scale normalization.

**Run on [SGLang](https://github.com/sgl-project/sglang)**: use the launch command below and see the [GLM-5.3-Flash cookbook](https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.3-Flash) for deployment guidance.

## Third-Party Community Consideration

This model is not owned or developed by RadixArk. It is a quantized derivative of Z.ai's model; see the upstream [GLM-5.3-Flash-BF16 model card](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16) for the source model's capabilities, training information, and license.

### License/Terms of Use:

[MIT License](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/LICENSE)

### Deployment Geography:

Global <br>

### Use Case: <br>

Developers looking to deploy an off-the-shelf, pre-quantized multimodal model for agentic engineering, coding, long-context reasoning, tool use, chat, and other AI-powered applications. <br>

### Release Date: <br>

Hugging Face 08/28/2026 via https://huggingface.co/RadixArk/GLM-5.3-Flash-NVFP4 <br>

## Model Architecture:

**Architecture Type:** Transformer (hybrid-attention sparse Mixture-of-Experts, natively multimodal) <br>
**Network Architecture:** GLM-5.3-Flash — 45 language-model layers comprising 3 dense MLP layers and 42 MoE layers, 288 routed experts per MoE layer, a shared expert, KDA linear attention, DSA sparse attention with an indexer, MLA, manifold-constrained hyper-connections, a vision encoder, and a native MTP/NextN draft layer <br>
**Number of Model Parameters:** 320B total, 18B activated per token <br>

## Input:

**Input Type(s):** Text, image, and video <br>
**Input Format(s):** String and visual media <br>
**Other Properties Related to Input:** Native context length up to 1,048,576 tokens. <br>

## Output:

**Output Type(s):** Text <br>
**Output Format:** String <br>

## Software Integration:

**Supported Runtime Engine(s):** <br>
* SGLang <br>

**Supported Hardware Microarchitecture Compatibility:** <br>
* NVIDIA Blackwell (validated on 4x GB300) <br>

**Preferred Operating System(s):** <br>
* Linux <br>

## Model Version(s):

Source: [zai-org/GLM-5.3-Flash-BF16](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16), revision `a6c167b6`. <br>
Quantized with [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) 0.46.0, commit `43fd41a58d52c4e6e5dec1d1ff5989ecc737ae1a`. <br>

## Training, Testing, and Evaluation Datasets:
### Calibration Data:

Calibration used 1,024 samples from the `abisee/cnn_dailymail` training split with sequence length 512 and batch size 1. <br>

### Training Dataset:

RadixArk did not train or fine-tune this checkpoint. Training information is inherited from the upstream [GLM-5.3-Flash-BF16 model card](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16). <br>

### Evaluation Dataset:

This checkpoint was evaluated on GSM8K, AIME 2026, and Terminal-Bench 2.1. <br>

## Post Training Quantization

The checkpoint uses plain abs-max NVFP4 W4A4 quantization with 256 tensor-scale normalization and group size 16. The quantized scope contains 36,423 NVFP4 bases: the `gate_proj`, `up_proj`, and `down_proj` tensors for all routed experts in 42 MoE layers (288 experts per layer), each shared expert, and the dense MLPs in layers 0-2.

All attention components (KDA, the DSA indexer, and MLA), hyper-connections, normalization layers, BF16 router weights and FP32 correction biases, the vision tower, the MTP/NextN layer, embeddings, and the language-model head remain at source precision. The checkpoint does not quantize the KV cache. Both FP8 (`fp8_e4m3`) KV cache with TRT-LLM DSA backends and BF16 KV cache with TileLang DSA backends are validated serving pairings; the recipe below defaults to FP8 KV with TRT-LLM DSA (roughly 1.8x KV token capacity), while BF16 KV with TileLang remains an alternative.

The checkpoint audit found 2,347 unchanged tensors content-identical to the pinned source; the checkpoint index independently contains all 889 MTP (layer-45) tensors. All scale values are finite and positive (range 3.49e-05 to 256.0). Full audit evidence is included in [`tensor-audit-b.json`](./tensor-audit-b.json) and [`precision-contract-b.json`](./precision-contract-b.json).

## Usage

Use the official SGLang GLM-5.3-Flash image and the following four-GPU configuration:

```sh
docker pull lmsysorg/sglang:glm-5.3-flash

docker run --gpus all \
  --shm-size 32g \
  -p 30000:30000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env "HF_TOKEN=<your-token>" \
  --ipc=host \
  lmsysorg/sglang:glm-5.3-flash \
  python3 -m sglang.launch_server \
    --model-path RadixArk/GLM-5.3-Flash-NVFP4 \
    --quantization modelopt_fp4 \
    --tp-size 4 \
    --dsa-prefill-backend trtllm \
    --dsa-decode-backend trtllm \
    --kv-cache-dtype fp8_e4m3 \
    --moe-runner-backend flashinfer_cutlass \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 6 \
    --speculative-adaptive \
    --reasoning-parser glm45 \
    --tool-call-parser glm47 \
    --host 0.0.0.0 \
    --port 30000
```

For other deployment topologies and hardware-specific configurations, see the [SGLang GLM-5.3-Flash cookbook](https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.3-Flash).

### Evaluation

The benchmark results below were produced with this NVFP4 checkpoint and must not be interpreted as results from a different quantization variant.

| Benchmark | Evaluation protocol | Score | Per-seed range | Stop rate |
|---|---|---:|---:|---:|
| GSM8K | Full 1,319-example split x 4 seeds; `temperature=1.0`, `top_p=0.95`, `max_tokens=32768` | **97.14%** | 96.89%-97.42% | 99.85%-100% |
| AIME 2026 | 30 problems x 16 repeats x 4 seeds (1,920 generations); `temperature=1.0`, `top_p=0.95`, `max_tokens=131072` | **92.45%** | 91.67%-93.54% | 94.58%-96.04% |
| Terminal-Bench 2.1 | 89 tasks, terminus-2 agent, effort max, pass@1; `temperature=1.0`, `top_p=0.95`, `max_tokens=32768` | **83.1% (74/89)** | - | - |

For AIME 2026, generated-token percentiles (total output tokens; this harness does not split reasoning from answer tokens) were p50 **8,696** and p90 **53,342**. The reported evaluations were text-only.

Terminal-Bench 2.1 was run on the `terminal-bench-2-1` dataset (revision `7d7bdc1c`) with concurrency 32 and task deadlines disabled (capability-first). Two tasks (`qemu-alpine-ssh`, `qemu-startup`) could not run because they require KVM, which was unavailable on the evaluation runner. Two trials (`extract-moves-from-video`, `regex-chess`) did not finish within the evaluation window, so the reported numerator can only rise. The Terminal-Bench 2.1 run was served with SGLang on 4x GB300 (TP4) using FP8 (`fp8_e4m3`) KV cache with TRT-LLM DSA attention backends, flashinfer_cutlass MoE, and NEXTN speculative decoding.

## Model Limitations:

Quantization can change model outputs relative to the source checkpoint. The base model may generate inaccurate, incomplete, irrelevant, biased, or otherwise undesirable responses. Developers should evaluate this checkpoint under their intended prompts, modalities, context lengths, sampling settings, and serving configuration, and apply appropriate safeguards.

## Ethical Considerations

RadixArk believes trustworthy AI is a shared responsibility. Developers should ensure that use of this model complies with the upstream license and meets the safety, privacy, security, and reliability requirements of their application.
