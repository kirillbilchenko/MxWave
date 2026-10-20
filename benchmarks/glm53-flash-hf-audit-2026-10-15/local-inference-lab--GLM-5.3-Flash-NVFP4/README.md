---
base_model: zai-org/GLM-5.3-Flash-BF16
base_model_relation: quantized
library_name: transformers
pipeline_tag: text-generation
license: mit
language:
  - en
  - zh
tags:
  - quantized
  - nvfp4
  - mxfp8
  - modelopt
  - distillation
  - quantization-aware-distillation
  - conversational
  - multimodal
---

## Model Description

local-inference-lab/GLM-5.3-Flash-NVFP4 is a quantization-aware distilled version of
[zai-org/GLM-5.3-Flash-BF16](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16).
The student is trained against the original BF16 teacher with NVFP4 expert
weights in its forward pass, learning to compensate for quantization error
rather than relying on post-training quantization alone.

The architecture is unchanged: 45 decoder layers, including three dense
layers and 42 MoE layers with 288 routed experts each, eight active per
token, and one shared expert. Attention combines linear attention and
DeepSeek Sparse Attention. The release occupies approximately 199 GB on disk.

### What's quantized

| Component | Stored format | Distillation |
| --- | --- | --- |
| Routed decoder experts: gate, up and down projections | NVFP4 | Trained |
| Dense MLPs in layers 0–2 and shared experts | BF16 | Trained |
| Router projection matrices | BF16 | Trained |
| Decoder and final RMSNorm weights | BF16 | Trained |
| Routing correction biases | FP32 | Updated during distillation |
| Attention, residual-stream mixing, token embeddings and LM head | Original BF16/FP32 | Frozen |
| Vision encoder | BF16 | Unchanged; outside text distillation |
| Auxiliary MTP routed experts | MXFP8 | Unchanged; outside text distillation |

NVFP4 stores 4-bit E2M1 values with FP8 E4M3 block scales per 16 elements
and FP32 global scales. Gate/up projections share a global scale within
each expert; down projections have their own. The scales are recomputed
from the trained weights during quantization.

The multi-token prediction (MTP) module is retained from the mixed-precision
base checkpoint. Its routed experts use MXFP8 E4M3 values with power-of-two
block scales per 32 elements; its other parameters retain their original
precision. The vision encoder and MTP module were not distilled.

### Quantization-aware distillation

The BF16 teacher generates chat responses and supplies probability and
hidden-state targets for the student.

During training, the student's routed-expert weights are quantized to NVFP4
and reconstructed for each forward pass. Gradients update the underlying
BF16 weights so their low-precision representations better match the
teacher's next-token probabilities and final hidden states.

The student uses its own expert routing during training. Routers, shared
experts, dense MLPs and normalization weights are also adapted, allowing
the unquantized components to compensate alongside the NVFP4 experts.

### Training data

The distillation corpus contains approximately 200M prompt-and-response
tokens. Assistant responses come from the BF16 GLM teacher using its native chat template.

### Activation calibration

Distillation uses BF16 activations. For NVFP4 activation quantization at
inference, expert-specific input scales are calibrated separately on the
200M-token chat corpus.

Calibration uses natural routing and covers all 12,096 layer/expert pairs
without fallback scales. Gate/up share their expert's input range; down
uses the post-activation range. The serving configuration also retains
dynamic MXFP8 activation quantization for MTP routed experts.

### Requirements

Use a runtime supporting the `glm5_next` architecture and this mixed
NVFP4/MXFP8 ModelOpt layout, including per-expert activation scales.
The tokenizer, upstream chat template, generation settings, vision assets
and MTP tensors are included. KV cache and runtime buffers require memory
in addition to the model weights.
