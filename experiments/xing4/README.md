# Xing4.0-29B-A4B MXFP4 experiment

Status: experimental branch only. A local H16 artifact was built and served,
but it failed the first paired quality gate and is not a publication candidate.

## Pinned inputs

- BF16 source: `XingChen-AGI/Xing4.0-29B-A4B` at
  `baae3c3e813cad5f888f1f485cfff659c89076c5`.
- FP8 reference: `XingChen-AGI/Xing4.0-29B-A4B-FP8` at
  `5080375347768be8f18304df141a1aea5b8416c9`.
- Runtime base: `vllm/vllm-openai:v0.29.0` at image digest
  `sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1`.
  That manifest contains both `linux/arm64` and `linux/amd64` images.
- Xing4 vLLM support: upstream PR
  [vllm-project/vllm#57135](https://github.com/vllm-project/vllm/pull/57135),
  commit `25aa52a29131753c10e56528d3fa0c4464cead63`, cherry-picked onto the
  v0.29.0 source revision `98dff2a81d747d1dba01a47f939f48c3526d4206`.

The vLLM PR was not merged when this experiment began. The local image is a
reproducible compatibility overlay, not a supported release image.

## Quantization boundary

The `xing4-29b-a4b` policy validates the exact Xing4 config and targets 7,616
base-model matrices:

- all five attention projections in base layers 0 through 39;
- the three dense MLP projections in layers 0 and 1;
- gate, up, and down projections for all 64 routed experts in layers 2 through
  39; and
- the shared expert's three projections in layers 2 through 39.

The layer-40 MTP head, routers, mHC parameters, normalization weights,
embeddings, and LM head remain in their source precision. LayerNorm proxies
weight 5,104 input-facing targets when no activation-calibration artifact is
provided. The remaining 2,512 targets are selected with unweighted MSE.

The real header-only plan recorded 62,430,067,648 bytes of source tensor data.
The completed checkpoint contains 20,529,515,296 tensor bytes (19.1196 GiB),
a 3.0410x compression ratio. This is about 38% smaller than the official
30.89 GiB FP8 checkpoint, before filesystem metadata.

## Calibration record

The bounded first experiment uses 16 sequences of 512 tokens packed from the
same pinned Pile validation response used by the Qwen H64 recipe. It is disjoint
from the WikiText-2 test corpus reserved for evaluation.

```text
corpus SHA-256:    07ffd944ea4fc09f12e5d019633a77309199fdced749e3ddf13989e53586f36a
token IDs SHA-256: 35e4ac6ffb23353bad6e01d0d9efcdee5e7c0fc8cf281e3c28d14ae4f5472ad3
stats SHA-256:     abb2ffc7d011541341fd059abca84898435b1857dc60e7f8eb64f50dbcb040df
targets:           7,616
minimum samples:   121
maximum samples:   8,192
capture time:      64.77 seconds
peak process RSS:  7.67 GiB
peak CUDA alloc:   2.48 GiB
```

The minimum count is the least-routed expert projection, not a partial
calibration. Every selected target was observed.

## Runtime gates observed on DGX Spark

The pinned FP8 checkpoint loaded and generated successfully with the overlay.
vLLM selected DeepGEMM FP8 for dense and MoE paths and reported 29.06 GiB of
model memory.

A metadata-only MXFP4 load exposed one required serving option:

- automatic linear-backend selection chose FlashInfer `cute-dsl`, which rejects
  compute capability 12.1; and
- `--linear-backend marlin` reached a healthy server. Dense projections used
  `MarlinMxFp4LinearKernel`, and routed experts used `MarlinExperts`.

This is a W4A16 execution path: Marlin deliberately ignores the config's
dynamic activation-FP4 request. The real 41-shard checkpoint reached a healthy
server and completed deterministic generation. vLLM reported 16.4 GiB of model
memory and 109.34 seconds for weight loading, versus 29.06 GiB and 224.18
seconds for the official FP8 checkpoint. These startup measurements are not a
throughput benchmark and do not establish W4A4 speed or quality.

## H16 quality pilot

The first quality gate compared the H16 checkpoint against the pinned official
FP8 checkpoint on the same first 16 non-empty, 4,096-character WikiText-2 test
chunks. Both runs used one worker, identical text, identical tokenization, and
16,986 prompt tokens of which 16,970 received a next-token score.

| Checkpoint | Tensor data | Mean NLL | Perplexity |
|---|---:|---:|---:|
| Official FP8 | 30.89 GiB | 2.296674 | 9.941062 |
| MxWave H16 MXFP4 | 19.1196 GiB | 2.348834 | 10.473349 |

The H16 checkpoint regressed perplexity by 5.3544% and mean NLL by 0.052160
nats/token. It was worse on 15 of the 16 paired chunks. The result therefore
fails the initial quality gate despite the 38% checkpoint-size reduction.

The complete report hashes are:

```text
MxWave H16: 0f1e957b87be10e3b3b1b6ede610b86e955c15883230a6862f79b02d7b94076a
Official FP8: ed76c05bb68bb3175af101f657d7dd681c178e83e55fea742bbccb4beeb943f2
Corpus:       696cca6b65a171b0a358a4be6732cdfdf2dd6164a32e20fd70e3c13fc4dfae83
```

This pilot isolates quantization loss; it does not establish whether Xing4 is
preferable to another upstream model. Before spending more compute on H64,
mixed precision, or iterative calibration, the official Xing4 FP8 checkpoint
must first be compared with the incumbent Qwen checkpoint on representative
coding and agent tasks. If FP8 does not clear that model-level gate, further
Xing4 quantization work stops. An attempted H64 launch exited at argument
parsing because the container entrypoint was not overridden; it produced no
calibration artifact and is not a measured H64 result.

## Rebuild the temporary runtime image

Prepare a vLLM source tree containing the pinned overlay commit, then use that
tree as the Docker build context:

```bash
git clone https://github.com/vllm-project/vllm.git vllm-xing4-v029
cd vllm-xing4-v029
git checkout 98dff2a81d747d1dba01a47f939f48c3526d4206
git fetch origin pull/57135/head:xing4-pr
git cherry-pick 25aa52a29131753c10e56528d3fa0c4464cead63
docker build \
  -f /path/to/MxWave/experiments/xing4/Dockerfile.vllm-v0.29-overlay \
  -t local/mxwave-vllm-xing4:v0.29-pr57135 \
  .
```

`prepare_mxfp4_dummy_config.py` creates the broad metadata-only config used to
exercise both dense and fused-MoE MXFP4 initialization. It is not a model
conversion recipe.

Only after an artifact passes the acceptance gates should this Python-only
overlay be published as one multi-platform image:

```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -f /path/to/MxWave/experiments/xing4/Dockerfile.vllm-v0.29-overlay \
  -t ghcr.io/OWNER/mxwave-vllm:v0.29-xing4-pr57135 \
  --push \
  .
```

This does not add AMD GPU/ROCm support; `linux/amd64` here means an x86-64
host with an NVIDIA CUDA GPU.

## Conversion settings

The first real checkpoint uses the proven H64 scale-search knobs with the
smaller 16x512 calibration budget:

```bash
python -m mxwave.cli \
  --model-dir /input \
  --output-dir /output/model \
  --policy xing4-29b-a4b \
  --method mse \
  --scale-percentile 99.5 \
  --mse-clip-depth 4 \
  --tensor-row-chunk-size 1024 \
  --activation-stats /calibration/xing4-pileval-16x512-block-hessian.safetensors \
  --calibration-objective block-hessian \
  --device cuda \
  --source-repository XingChen-AGI/Xing4.0-29B-A4B \
  --source-revision baae3c3e813cad5f888f1f485cfff659c89076c5 \
  --verify-sqnr \
  --sqnr-rows 16
```

## Serving requirement

Until the SM121 FlashInfer path is fixed upstream, a converted checkpoint must
be launched with the explicit backend:

```bash
vllm serve /model \
  --served-model-name xing4-mxwave \
  --trust-remote-code \
  --dtype bfloat16 \
  --linear-backend marlin
```

## Acceptance gates before publication

1. Header-only planning must select exactly 7,616 matrices and exclude every
   layer-40 key.
2. Streamed calibration must observe every selected routed expert without
   loading the whole checkpoint into RAM.
3. The real converted checkpoint—not dummy weights—must load and generate via
   Marlin/MarlinExperts.
4. Size, deterministic perplexity, paired next-token divergence, at least one
   reasoning task, and throughput must be compared on identical inputs against
   the pinned FP8 reference and, where practical, BF16.
5. Publication is considered only if the measured quality/size result adds a
   useful Pareto point. Runtime compatibility alone is insufficient.
