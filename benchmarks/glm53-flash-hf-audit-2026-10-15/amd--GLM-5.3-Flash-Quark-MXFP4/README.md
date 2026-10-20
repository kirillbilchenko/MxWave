---
license: mit
base_model:
- zai-org/GLM-5.3-Flash
---
# Model Overview

- **Model Architecture:** GLM-5.3-Flash
  - **Input:** Text, Image, Video
  - **Output:** Text
- **Supported Hardware Microarchitecture:** AMD MI350/MI355
- **ROCm:** 7.2.0
- **PyTorch:** 2.9.1+rocm7.2.0.git7e1940d4
- **Transformers:** 5.12.1
- **Operating System(s):** Linux
- **Inference Engine:** [SGLang](https://docs.sglang.ai/)
- **Model Optimizer:** [AMD-Quark](https://quark.docs.amd.com/latest/index.html) (V0.12)
  - **Weight quantization:** MOE-only (shared experts quantized), OCP MXFP4, Static
  - **Activation quantization:** MOE-only, OCP MXFP4, Dynamic

This model was built with GLM-5.3-Flash model by applying [AMD-Quark](https://quark.docs.amd.com/latest/index.html) for MXFP4 quantization.

# Model Quantization

The model was quantized from [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) using [AMD-Quark](https://quark.docs.amd.com/latest/index.html). The weights and activations are quantized to MXFP4.

**Quantization scripts:**

```bash
from quark.torch import LLMTemplate, ModelQuantizer

EXCLUDE = [
    "*self_attn*",
    "*mlp.gate",
    "*mlp.gate_proj", "*mlp.up_proj", "*mlp.down_proj",  # dense MLP (layers 0-2)
    "*visual*",
    "*lm_head*",
    "*embed*",
    "model.language_model.layers.45.*",  # MTP block
]

template = LLMTemplate(
    model_type="glm5_next",
    kv_layers_name=["*self_attn.kv_b_proj"],
    q_layer_name="*self_attn.q_b_proj",
    exclude_layers_name=EXCLUDE,
)
LLMTemplate.register_template(template)

quant_config = template.get_config(scheme="mxfp4", exclude_layers=EXCLUDE)

quantizer = ModelQuantizer(quant_config)
quantizer.direct_quantize_checkpoint(
    pretrained_model_path="zai-org/GLM-5.3-Flash",
    save_path=OUTPUT_DIR,
    device="cuda",
    keep_excluded_layers_as_original_model_state=True,
)
```

# Deployment

### Use with SGLang

This checkpoint is validated for native MXFP4 deployment on AMD Instinct MI350X and MI355X GPUs (CDNA 4, `gfx950`). The accuracy measurements below were collected on MI350X. MI355X uses the same `gfx950` target, 288 GB HBM capacity per GPU, and ROCm `mi35x` software path.

The validation used:

- Model revision [`fc676278b68ba33f6b4724be286af9d7f6c814c2`](https://huggingface.co/amd/GLM-5.3-Flash-Quark-MXFP4/commit/fc676278b68ba33f6b4724be286af9d7f6c814c2)
- SGLang commit [`654df43cbee108a81fa1736c34ba8c701f199285`](https://github.com/sgl-project/sglang/commit/654df43cbee108a81fa1736c34ba8c701f199285), from [SGLang PR #36607](https://github.com/sgl-project/sglang/pull/36607)
- Container image `lmsysorg/sglang@sha256:6d68cd19206716cb3f1e31e2ad89cd0852d7ae614a792773c30a4277f8955c72`
- ROCm 7.2.0, PyTorch `2.9.1+rocm7.2.0.git7e1940d4`, Transformers `5.12.1`, and SGLang `0.5.18` with the source commit above mounted into the container

The pinned image predates the GLM-5.3 mixed Quark MXFP4/block-FP8 loader change, so use the exact SGLang source commit above with the image:

```bash
git clone https://github.com/sgl-project/sglang.git
git -C sglang checkout 654df43cbee108a81fa1736c34ba8c701f199285

hf auth login
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

docker run --rm -it \
  --network host \
  --ipc host \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v "$PWD/sglang:/sgl-workspace/sglang" \
  -v "$HF_HOME:/root/.cache/huggingface" \
  lmsysorg/sglang@sha256:6d68cd19206716cb3f1e31e2ad89cd0852d7ae614a792773c30a4277f8955c72 \
  bash
```

Inside the container, launch the validated 8-GPU TP8+EP8 configuration:

```bash
cd /sgl-workspace/sglang
export PYTHONPATH=/sgl-workspace/sglang/python:${PYTHONPATH:-}
export SGLANG_USE_AITER=1
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -m sglang.launch_server \
  --model-path amd/GLM-5.3-Flash-Quark-MXFP4 \
  --served-model-name amd/GLM-5.3-Flash-Quark-MXFP4 \
  --tp-size 8 \
  --ep-size 8 \
  --attention-backend dsa \
  --dsa-prefill-backend tilelang \
  --dsa-decode-backend tilelang \
  --linear-attn-backend triton \
  --kv-cache-dtype bfloat16 \
  --moe-runner-backend aiter \
  --disable-shared-experts-fusion \
  --disable-radix-cache \
  --context-length 65536 \
  --max-running-requests 64 \
  --cuda-graph-backend-decode full \
  --cuda-graph-max-bs-decode 64 \
  --mem-fraction-static 0.85 \
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":8}' \
  --watchdog-timeout 1200 \
  --trust-remote-code \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --host 127.0.0.1 \
  --port 30000
```

For a 4-GPU deployment, use GPUs `0,1,2,3`, set `--tp-size 4`, and omit `--ep-size 8`. Both configurations were validated with full decode graph capture through batch size 64 (`--cuda-graph-backend-decode full`).

## Evaluation

### Accuracy

The full 1,319-example GSM8K test set was evaluated with thinking enabled. These are accuracy results, not throughput or latency benchmarks.

| Benchmark | Measured hardware | Parallelism | Correct | Accuracy |
| --- | --- | --- | ---: | ---: |
| GSM8K | 4× AMD Instinct MI350X (`gfx950`) | TP4 | 1,282 / 1,319 | 97.19% |
| GSM8K | 8× AMD Instinct MI350X (`gfx950`) | TP8 + EP8 | 1,281 / 1,319 | 97.12% |

Both runs completed all 1,319 unique examples with a 100% `stop` finish rate and zero duplicate IDs, empty generations, evaluator errors, truncations, or server faults. MI355X is covered as the equivalent `gfx950` deployment target; the reported measurements were collected on MI350X.

### Reproduction

The results were produced with [`sgl-eval`](https://github.com/sgl-project/sgl-eval) at commit [`a231b7a439b235090ff7baa30778fa2b514309ae`](https://github.com/sgl-project/sgl-eval/commit/a231b7a439b235090ff7baa30778fa2b514309ae):

```bash
python -m pip install \
  "git+https://github.com/sgl-project/sgl-eval.git@a231b7a439b235090ff7baa30778fa2b514309ae"

python -m sgl_eval.cli run gsm8k \
  --base-url http://127.0.0.1:30000/v1 \
  --model amd/GLM-5.3-Flash-Quark-MXFP4 \
  --num-examples 1319 \
  --num-threads 64 \
  --max-tokens 32768 \
  --temperature 1.0 \
  --top-p 0.95 \
  --thinking \
  --out-dir ./gsm8k-full
```

Native MXFP4 execution requires a `gfx950` GPU. MI300X and MI325X use `gfx942` and are not supported by this native MXFP4 recipe.

# License
Modifications Copyright(c) 2026 Advanced Micro Devices, Inc. All rights reserved.
