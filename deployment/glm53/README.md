# GLM-5.3 Flash CT on stock vLLM 0.32.0

**Experimental release `glm53-v0.1.1`.** The pinned official vLLM image runs with
the MxWave plugin; no vLLM source files are replaced. Supported hardware:
**two Linux ARM64 DGX Sparks, SM121, TP2**. This is a specific checkpoint recipe,
not a general compressor or a promise of compatibility with later vLLM versions.

[Release downloads](https://github.com/kirillbilchenko/MxWave/releases/tag/glm53-v0.1.1)
· [Results and limitations](../../docs/GLM53_RELEASE.md)
· [Native sources](native/README.md)
· [Separate live setup and rollback](../../docs/GLM53_CT_LIVE.md)

The stock profile was GPU-screened on 21 October: 500k configured context starts,
prefix reuse works, DFlash accepts tokens, and three C1/C8 code/chat timing pairs
and a 300-question GSM8K subset complete. Against tuned custom production it is
about **22% slower at C1, 15–21% faster in aggregate at C8**. The owner's currently
live custom-runtime CT setup has different results. Numerical/repeatability
failures remain; 500k startup does not establish long-context answer quality.

## Selected profile

| Component | Setting |
|---|---|
| Target | Public Intel AutoRound CT MXFP4 experts, pinned revision |
| KDA projections | MXFP8 conversion during loading; **lossy** |
| Expert scales / embedding | Lossless packing; dense BF16 output head |
| Drafter | Public Modal BF16 DFlash converted to block FP8 on CPU; no training |
| Draft policy | 5 tokens at C1/C2, 3 at larger batches |
| Capacity | 500,000 context, 32 slots, 6 GiB KV/rank, 6,919 prefill budget |
| Cache / graphs | Prefix caching on; explicit tested graph sizes |
| Loader / draft attention | Run:ai streamer, 3 GiB buffer, concurrency 8; Triton |

Allow at least 200 GB free storage per node for target, drafter and packed assets,
plus Docker image/cache space. Both nodes need identical inputs. CPU preparation
was checked with an 8 GiB container memory limit and needs no GPU.

## 1. Download and build

Run from a clean directory on a Spark. `gh` is a download convenience; assets are
also available from the release page. Keep the release tag pinned.

```bash
gh release download glm53-v0.1.1 --repo kirillbilchenko/MxWave --dir release
cd release
sha256sum --check SHA256SUMS
tar -xzf mxwave-glm53-native-sm121-v0.1.1.tar.gz
docker build -f Dockerfile -t mxwave-glm53:0.1.1 .
```

The Dockerfile pins the official ARM64 vLLM 0.32.0 image by digest and installs
the wheel with `--no-deps`. It does not upgrade Torch or vLLM. The native archive
contains three pinned libraries, licenses and build sources, without model weights.
Library hashes are checked again before native loading.

## 2. Download public inputs

Using the Hugging Face CLI, download these exact revisions on each node, or
download once and copy with verification. Run from the release directory:

```bash
hf download INCModel3/GLM-5.3-Flash-MXFP4-Mixed-CT-AutoRound \
  --revision 35d1b0d5cb4a0db075fd2749c897e970dc15bfa5 --local-dir ct-model
hf download modal-labs/GLM-5.3-Flash-DFlash \
  --revision dae6d319510889a7f1ac5242d5b7fb2d5d95eb05 --local-dir modal-bf16
```

Retain downloaded licenses. MxWave did not create the AutoRound checkpoint or
train Modal's drafter. Origins and all 120 target shard hashes are in the wheel's
`checkpoints.json`.

## 3. Prepare and verify on CPU

```bash
mkdir prepared
docker run --rm --network none --cpus 2 --memory 8g \
  -e NVIDIA_VISIBLE_DEVICES=void -v "$PWD/ct-model:/model:ro" \
  mxwave-glm53:0.1.1 verify-model --model /model

docker run --rm --network none --cpus 2 --memory 8g \
  -e NVIDIA_VISIBLE_DEVICES=void -v "$PWD/ct-model:/model:ro" \
  -v "$PWD/native:/native:ro" -v "$PWD/prepared:/export:rw" \
  mxwave-glm53:0.1.1 build-bundle \
  --model /model --native /native --output /export/runtime

docker run --rm --network none --cpus 2 --memory 8g \
  -e NVIDIA_VISIBLE_DEVICES=void -v "$PWD/modal-bf16:/source:ro" \
  -v "$PWD/prepared:/export:rw" mxwave-glm53:0.1.1 \
  prepare-pretrained-drafter --source /source --output /export/modal-fp8

docker run --rm --network none -e NVIDIA_VISIBLE_DEVICES=void \
  -v "$PWD/prepared/runtime:/runtime:ro" mxwave-glm53:0.1.1 \
  doctor --bundle /runtime
```

All four vocabulary files must reproduce the GPU-tested hashes exactly. The
drafter command converts 42 backbone tensors to block FP8, preserves 54 other
tensors, verifies the exact weight hash, then corrects twelve BF16 convolution
projections' metadata. Without that correction the earlier native 0.32 trial
accepted zero drafts. Existing output directories are never overwritten.

Copy `prepared/runtime` and `prepared/modal-fp8` to the other Spark, retaining
licenses. `doctor` and `serve` recheck packed assets and drafter. Run `verify-model`
once on each target copy before serving. No private experiment directory or
training set is required. Legacy `prepare-bundle` and `prepare-drafter` commands
remain available for already prepared owner assets.

## 4. Launch TP2

Run rank 1 first, then rank 0. Set the master to rank 0's reachable private address
and each node's own `GLM_NODE_IP`. Configure NCCL/GLOO/RoCE interfaces for your
network; interface names are not hardcoded. Use a separate port while another
service is running. This command does not stop or replace that service.
Export `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`, `NCCL_IB_HCA` and
`NCCL_IB_GID_INDEX` if your network requires explicit selection; the command
passes through whichever of these variables are set.

```bash
export GLM_NODE_RANK=0
export GLM_MASTER_ADDR=192.168.192.1
export GLM_NODE_IP=192.168.192.1
export GLM_MODEL_DIR="$PWD/ct-model"
export GLM_DRAFTER_DIR="$PWD/prepared/modal-fp8"
export GLM_RUNTIME_DIR="$PWD/prepared/runtime"
export GLM_REPORTS_DIR="$PWD/reports-r$GLM_NODE_RANK"
export GLM_CACHE_DIR="$PWD/cache-r$GLM_NODE_RANK"
mkdir -p "$GLM_REPORTS_DIR" "$GLM_CACHE_DIR"

docker run --rm --name "mxwave-glm53-r$GLM_NODE_RANK" \
  --gpus all --network host --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --cap-add IPC_LOCK --device /dev/infiniband \
  -e VLLM_HOST_IP="$GLM_NODE_IP" -e OMP_NUM_THREADS=4 \
  -e NCCL_SOCKET_IFNAME -e GLOO_SOCKET_IFNAME -e NCCL_IB_HCA -e NCCL_IB_GID_INDEX \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v "$GLM_MODEL_DIR:/model:ro" -v "$GLM_DRAFTER_DIR:/drafter:ro" \
  -v "$GLM_RUNTIME_DIR:/runtime:ro" -v "$GLM_REPORTS_DIR:/reports:rw" \
  -v "$GLM_CACHE_DIR:/root/.cache:rw" \
  mxwave-glm53:0.1.1 serve \
  --bundle /runtime --model /model --drafter /drafter --reports /reports \
  --node-rank "$GLM_NODE_RANK" --master-addr "$GLM_MASTER_ADDR" \
  --host 127.0.0.1 --port 8046
```

Defaults match the tested profile above. Use `serve --help` for explicit options,
or append `--dry-run` to verify assets and inspect arguments without loading a
model. Rank 1 gets upstream `--headless`. The listener is loopback-only by default;
use your protected API access path when exposing it. Reports record actual
allocations after drafter, KV and graph loading.

Stop this recipe with `docker stop mxwave-glm53-r0` on rank 0 and
`docker stop mxwave-glm53-r1` on rank 1. The owner's existing live service has a
separate [exact-production rollback](../../docs/GLM53_CT_LIVE.md).

## Memory and limits

Stock CT allocates **87.202210 / 87.202180 GiB per rank**, including drafter, KV
and graphs. Lossless scale/embedding packing saves **6.94113 GiB TP2** after expert
scratch space. This is storage accounting, not a matched whole-service allocation
delta against production. The live custom-runtime comparison's **14.465825 GiB**
saving includes checkpoint/MXFP8 differences; do not add or substitute these numbers.

The recipe excludes production's certified argmax head, FP8 draft-head overlay,
admission hooks and custom RoCE all-reduce. Its hooks wrap version-pinned internal
vLLM APIs; source hash checks fail closed if those APIs change. New versions or
native libraries need separate qualification. The 500k capacity check is a startup
test; 32k/128k retrieval was not tested. Prefix reuse passes, but output identity
and the original strict KL gate do not. KDA and drafter conversion are lossy.

This release adds reproducible CPU preparation around the existing GPU-tested
runtime. It does not rerun benchmarks, relax gates or change the live service.
