---
pipeline_tag: image-text-to-text
base_model:
- zai-org/GLM-5.3-Flash
license: mit
library_name: Model Optimizer
tags:
- nvidia
- ModelOpt
- GLM-5
- quantized
- FP4
- fp4
---

# Model Overview

## Description:
The NVIDIA GLM-5.3-Flash NVFP4 model is the quantized version of ZAI's GLM-5.3-Flash model, which is an auto-regressive language model that uses an optimized transformer architecture. GLM-5.3-Flash is a natively multimodal Mixture-of-Experts (MoE) model for reasoning, coding, and agentic tasks; it uses a hybrid sparse and linear attention architecture with Manifold-Constrained Hyper-Connections (mHC) to support a long context. For more information, please check [here](https://huggingface.co/zai-org/GLM-5.3-Flash). The NVIDIA GLM-5.3-Flash NVFP4 model is quantized with [Model Optimizer](https://github.com/NVIDIA/Model-Optimizer).

This model is ready for commercial or non-commercial use.  <br>

## Third-Party Community Consideration
This model is not owned or developed by NVIDIA. This model has been developed and built to a third-party's requirements for this application and use case; see link to Non-NVIDIA [(GLM-5.3-Flash) Model Card](https://huggingface.co/zai-org/GLM-5.3-Flash) from ZAI.

### License/Terms of Use:
[MIT](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md)

### Deployment Geography:
Global <br>

### Use Case:
Developers looking to take off-the-shelf, pre-quantized models for deployment in AI Agent systems, chatbots, RAG systems, and other AI-powered applications. <br>

### Release Date:
Huggingface 09/09/2026 via https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4 <br>

## References
Nvidia Model Optimizer: https://github.com/NVIDIA/Model-Optimizer

## Model Architecture:
**Architecture Type:** Transformers  <br>
**Network Architecture:** GLM-5.3-Flash (`Glm5NextForConditionalGeneration`) <br>
**Number of Model Parameters:** 320B in total and 18B activated <br>
**This model was developed based on [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)** <br>

## Input:
**Input Type(s):** Text, Image, Video <br>
**Input Format(s):** String, Red, Green, Blue (RGB), Video (MP4/WebM) <br>
**Input Parameters:** One-Dimensional (1D), Two-Dimensional (2D), Three-Dimensional (3D) <br>
**Other Properties Related to Input:** Context length up to 1M <br>

## Output:
**Output Type(s):** Text <br>
**Output Format:** String <br>
**Output Parameters:** 1D (One-Dimensional): Sequences <br>
**Other Properties Related to Output:** None <br>

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA's hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions. <br>

## Software Integration:
**Supported Runtime Engine(s):** <br>
* vLLM <br>
* SGLang <br>

**Supported Hardware Microarchitecture Compatibility:** <br>
* NVIDIA Blackwell <br>

**Preferred Operating System(s):** <br>
* Linux <br>

The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment.

## Model Version(s):
The model version is NVFP4 1.0 version and is quantized with nvidia-modelopt **v0.47.0** <br>

## Training and Evaluation Datasets:

## Calibration Dataset:
** Link: [cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail), [Nemotron-Post-Training-Dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2) <br>
** Data Collection Method by dataset: Automated. <br>
** Labeling Method by dataset: Automated. <br>
** Properties: The cnn_dailymail dataset is an English-language dataset containing just over 300k unique news articles as written by journalists at CNN and the Daily Mail. The Nemotron-Post-Training-Dataset-v2 is a post-training dataset curated by NVIDIA containing multi-turn conversations across diverse topics. <br>

## Training Dataset:
** Data Modality: Undisclosed <br>
** Data Collection Method by dataset: Undisclosed <br>
** Labeling Method by dataset: Undisclosed <br>
** Properties: Undisclosed

## Evaluation Dataset:
**Datasets:** GPQA Diamond, SciCode, MMMU Pro, AA-LCR, IFBench, Terminal Bench 2.1 <br>
**Data Collection Method by dataset:** Hybrid: Automated, Manually-Collected <br>
**Labeling Method by dataset:** Hybrid: Manually-Labeled, Automated <br>
**Properties:** We evaluated the model on text-based reasoning, coding, agentic tool-use, and multimodal benchmarks: GPQA Diamond contains 448 graduate-level multiple-choice questions written by domain experts in biology, physics, and chemistry. MMMU Pro is the more challenging version of the Massive Multi-discipline Multimodal Understanding benchmark, measuring college-level multimodal reasoning across diverse disciplines with expanded answer choices and a vision-only input setting.SciCode evaluates scientific coding capabilities. AA-LCR (Artificial Analysis Long Context Recall) evaluates a model's ability to accurately retrieve and recall information from long input contexts. IFBench is a benchmark for evaluating instruction-following capabilities across diverse and structured task constraints. Terminal-Bench 2.1 is an open-source evaluation framework designed to test AI agents on 89 complex, real-world tasks inside sandboxed command-line and container environments.<br>


## Inference:
**Acceleration Engine:** vLLM, SGLang <br>
**Test Hardware:** NVIDIA Blackwell GB200 <br>

## Post Training Quantization
This model was obtained by quantizing the weights and activations of GLM-5.3-Flash to NVFP4 data type, ready for inference with vLLM. Only the weights and activations of the linear operators within transformer blocks in sparse MoE shared experts and dense MLP are quantized. This optimization reduces the number of bits per parameter from 16 to 4, reducing the disk size and GPU memory requirements by approximately 3.33x.

** modelopt PTQ recipe: [nvfp4_experts_dense_mlp-kv_fp8_cast](https://github.com/NVIDIA/Model-Optimizer/blob/4956213d670c382385d9bc43e17379b9fc064e50/modelopt_recipes/models/zai-org/GLM-5.3-Flash/ptq/nvfp4_experts_dense_mlp-kv_fp8_cast.yaml)

## Usage

### vLLM

To serve this checkpoint with [vLLM](https://github.com/vllm-project/vllm), start from `vllm/vllm-openai:glm53-flash-arm64-cu130`   and run:

```sh
pip install -U "transformers>=5.16.1" && \
vllm serve /checkpoint \
  --served-model-name nvidia/GLM-5.3-Flash-NVFP4 \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 4 \
  --data-parallel-size 1 \
  --enable-expert-parallel \
  --enable-ep-weight-filter \
  --reasoning-parser glm45 \
  --kv-cache-dtype fp8 \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 128}' \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.90
```

### SGLang

To serve this checkpoint with [SGLang](https://github.com/sgl-project/sglang) on 4 NVIDIA Blackwell GPUs, run:

```sh
sglang serve \
  --model-path nvidia/GLM-5.3-Flash-NVFP4 \
  --quantization modelopt_fp4 \
  --tp-size 4 \
  --dsa-prefill-backend trtllm \
  --dsa-decode-backend trtllm \
  --kv-cache-dtype fp8_e4m3 \
  --moe-runner-backend flashinfer_cutlass \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --mem-fraction-static 0.85 \
  --host 0.0.0.0 \
  --port 30000
```

## Evaluation
The accuracy benchmark results are presented in the table below:
<table>
  <tr>
   <td><strong>Precision</strong>
   </td>
   <td><strong>GPQA Diamond</strong>
   </td>
   <td><strong>SciCode</strong>
   </td>
   <td><strong>MMMU Pro</strong>
   </td>
   <td><strong>AA-LCR</strong>
   </td>
   <td><strong>IFBench</strong>
   </td>
   <td><strong>Terminal Bench 2.1</strong>
   </td>
  </tr>
  <tr>
   <td>BF16
   </td>
   <td><strong>0.9217</strong>
   </td>
   <td><strong>0.5621</strong>
   </td>
   <td><strong>0.7688</strong>
   </td>
   <td><strong>0.71</strong>
   </td>
   <td><strong>0.613</strong>
   </td>
   <td><strong>0.8258</strong>
   </td>
  </tr>
  <tr>
   <td>NVFP4
   </td>
   <td><strong>0.9211</strong>
   </td>
   <td><strong>0.5769</strong>
   </td>
   <td><strong>0.763</strong>
   </td>
   <td><strong>0.7106</strong>
   </td>
   <td><strong>0.6054</strong>
   </td>
   <td><strong>0.8315</strong>
   </td>
  </tr>
</table>


> Baseline: [GLM-5.3-Flash-BF16](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16). Benchmarked with temperature=1.0, top_p=0.95. GPQA Diamond, SciCode, MMMU-Pro, AA-LCR and IFBench used max_new_tokens=327,680; Terminal Bench 2.1 used max_new_tokens uncapped.


## Model Limitations:
The base model was trained on data that contains toxic language and societal biases originally crawled from the internet. Therefore, the model may amplify those biases and return toxic responses especially when prompted with toxic prompts. The model may generate answers that may be inaccurate, omit key information, or include irrelevant or redundant text producing socially unacceptable or undesirable text, even if the prompt itself does not include anything explicitly offensive.

## Ethical Considerations

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. Developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

Please make sure you have proper rights and permissions for all input image and video content; if image or video includes people, personal health information, or intellectual property, the image or video generated will not blur or maintain proportions of image subjects included.

Please report model quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).
