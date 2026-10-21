# GLM CT on upstream vLLM 0.32.0

This is the default deployment target: the official vLLM image with the MxWave
wheel installed through vLLM's plugin mechanism. No vLLM source files are replaced.
The extension wraps version-pinned internal APIs and requires the verified bundle.

**20:23 UTC update:** the corrected packaged recipe starts at 500k context with
prefix caching and FP8 draft KV. Two 32-token probes accept 22/60 code and 21/50
chat proposals; actual allocation is 87.20221 / 87.20218 GiB per rank. Prefix reuse
is observed, but repeated-output identity still fails. The full timing/GSM8K
comparison is active; no production qualification or measured speed win yet.

**Status, 21 October 2026: the package now includes the GPU-tested drafter metadata
correction; full deployment qualification remains pending.** Native 0.32 applies FP8 loading to
twelve BF16 convolution weights with missing scales, producing NaN draft output.
A separate config with explicit BF16 exclusions restores 50–67.5% acceptance on
two short diagnostic prompts, with byte-identical weights and no vLLM source edits.
Run the new **`prepare-drafter`** step below before serving. `serve` rejects the
original incompatible config and verifies the corrected config and unchanged
weight hashes. Full timing and quality qualification remain pending; the isolated experiment used
[prepare_drafter_compat.py](prepare_drafter_compat.py) and a separately frozen launch.
[Diagnosis, settings and retained evidence](../../docs/GLM53_STOCK_DRAFTER_LOADING.md).

The earlier selected 8k Modal profile completed six C1 requests but accepted
**zero of 3,810 proposals**. Its 172.496 GiB TP2 allocation (+0.751 GiB versus the
historical selected profile) predates the correction; no fresh corrected-profile
memory delta is claimed. Native MTP separately needs a CT namespace/quantization
policy repair before comparison. Exact original 500k production is restored.

V6 starts at 500k context with 32 slots, 6 GiB KV per rank, a 6,919-token prefill
budget, prefix caching enabled and a dense target head. It uses native
`TRITON_ATTN` draft attention to avoid the default backend's small-page cache waste.
Actual capacity is 640,287 tokens; startup allocation is 87.157 GiB per rank after
model, drafter, KV and graphs. This is not a new saving versus production measurement.
The corrected 12,449-token V7 smoke confirms real prefix reuse (9,216 hit tokens),
but greedy messages differ cold-to-warm and warm-to-warm. The combined correctness
gate fails; V6 also differs without a reusable prefix. Fixed-input numerical tracing
is required before timing or quality claims. No 32k/128k probe has run.
[Current handoff and restoration status](../../docs/GLM53_HANDOFF.md).

## Supported profile

- Two DGX Sparks, Linux ARM64, SM121, one GPU per node, TP2.
- Official vLLM 0.32.0 ARM64 image, pinned by digest in [Dockerfile](Dockerfile).
- Audited AutoRound CT checkpoint: MXFP4 experts, with the measured MXFP8 KDA
  conversion at load time. KDA conversion is lossy; it failed the original 1e-5 KL
  gate. Expert-scale and vocabulary packing are lossless.
- Pretrained Modal FP8 DFlash2: five drafts at C1/C2, three at higher concurrency.
  No further drafter training is required for this profile.
- Separate model, drafter and verified runtime-bundle directories on both nodes.
  These are mounted read-only. Reports and CUDA caches are writable and unique
  to each launch.

The upstream recipe does not include production's certified argmax head, custom
FP8 draft head, admission/truncation hooks or custom RoCE all-reduce overlay.
Comparisons against that deployment must disclose those differences.

## Build and inspect

From the repository root, build the wheel using Python 3.12 or later:

```bash
python -m pip wheel . --no-deps --no-build-isolation -w dist
```

Build on each Spark using the same wheel:

```bash
docker build -f deployment/glm53/Dockerfile -t mxwave-glm53:vllm032 dist
```

The current exporter packages existing verified assets. It does not yet download
or generate all release assets from scratch. A public runtime-bundle distribution
and reproducible asset-build instructions remain release work; do not substitute
arbitrary libraries or model revisions to bypass the pinned hashes.

For the existing experiment owner, export once on each node:

```bash
docker run --rm --network none \
  -v /absolute/path/to/verified-experiment:/source:ro \
  -v /absolute/path/to/export-parent:/export:rw \
  mxwave-glm53:vllm032 prepare-bundle \
  --from-experiment /source --output /export/runtime
```

Copying the resulting `runtime` directory preserves its relocatable manifest.
Existing export destinations are never overwritten. Check installation and GPU:

```bash
docker run --rm --gpus all --network none \
  -v /absolute/path/to/runtime:/runtime:ro \
  mxwave-glm53:vllm032 doctor --bundle /runtime --gpu
```

## Launch on both nodes

Prepare a separate drafter directory once on each node, without GPU access:

```bash
docker run --rm --network none \
  -v /absolute/path/to/original-modal-fp8-checkpoint:/source:ro \
  -v /absolute/path/to/prepared-parent:/export:rw \
  mxwave-glm53:vllm032 prepare-drafter \
  --source /source --output /export/modal-fp8-vllm032
```

This checks the pinned source hashes and changes only `config.json`, preserving
twelve existing BF16 convolution matrices. Weight files are hard-linked when
possible, copied byte-for-byte across filesystems, and reverified. Existing output
directories are never overwritten. No model training or weight quantization occurs.
Keep the original and prepared files read-only when serving; hard links share data.

The following uses the **500k capacity settings that reached API readiness in V6**,
now with the corrected drafter metadata. The combined profile passes startup and
positive-acceptance checks; its repeated-output identity gate still fails.
Prefix correctness, output repeatability, the 32k/128k probes and the matched
performance/quality comparison remain required before replacing production.

Run on node 1 and then node 0, setting `GLM_NODE_RANK` to 1 and 0 respectively.
Use node 0's reachable private address for `GLM_MASTER_ADDR`, and each node's own
address for `GLM_NODE_IP`. Set checkpoint and writable paths for each host.
The wrapper adds upstream `--headless` on rank 1; only rank 0 runs the API server.

```bash
export GLM_NODE_RANK=0
export GLM_MASTER_ADDR=192.168.192.1
export GLM_NODE_IP=192.168.192.1
export GLM_MODEL_DIR=/absolute/path/to/ct-checkpoint
export GLM_DRAFTER_DIR=/absolute/path/to/prepared-parent/modal-fp8-vllm032
export GLM_RUNTIME_DIR=/absolute/path/to/runtime
export GLM_REPORTS_DIR=/absolute/path/to/new-reports
export GLM_CACHE_DIR=/absolute/path/to/cuda-cache
mkdir -p "$GLM_REPORTS_DIR" "$GLM_CACHE_DIR"

docker run --rm --gpus all --network host --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --cap-add IPC_LOCK --device /dev/infiniband \
  -e VLLM_HOST_IP="$GLM_NODE_IP" \
  -e OMP_NUM_THREADS=4 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v "$GLM_MODEL_DIR:/model:ro" \
  -v "$GLM_DRAFTER_DIR:/drafter:ro" \
  -v "$GLM_RUNTIME_DIR:/runtime:ro" \
  -v "$GLM_REPORTS_DIR:/reports:rw" \
  -v "$GLM_CACHE_DIR:/root/.cache:rw" \
  mxwave-glm53:vllm032 serve \
  --bundle /runtime --model /model --drafter /drafter --reports /reports \
  --node-rank "$GLM_NODE_RANK" --master-addr "$GLM_MASTER_ADDR" \
  --master-port 29719 --host 127.0.0.1 --port 8046 \
  --max-model-len 500000 --max-num-seqs 32 \
  --max-num-batched-tokens 6919 --kv-cache-gib 6 \
  --gpu-memory-utilization 0.85 --block-size 2304 \
  --draft-attention-backend TRITON_ATTN \
  --load-format runai_streamer --loader-memory-mib 3072 --loader-concurrency 8 \
  --prefix-caching --vocab-mode embedding \
  --graph-sizes 1 2 4 6 8 12 16 18 24 32
```

Provide NCCL/GLOO interface settings appropriate to the host's RoCE network.
The tested settings and image identities are retained with each operational trial;
interface names are not hardcoded in the package. Add `--dry-run` to the serving
command to verify assets and inspect arguments without launching a model.

Target and drafter KV use native `fp8_e4m3` settings. The 0.85 reservation fraction
matches production; the explicit 6 GiB KV budget remains independent of that
fraction. FlashInfer autotuning is disabled, matching the baseline configuration.

The example explicitly uses the tested native streaming loader:
`--load-format runai_streamer --loader-memory-mib 3072 --loader-concurrency 8`.
This exact 3 GiB buffer passes CPU byte checks on three complete representative
shards and loads the full target/drafter in the latest trials. Rank-0 logged model
loading is 123–126 seconds; controlled cold-start speed is not established. A 1 GiB buffer rejects
the largest tensor, and the 2 GiB screen stalls. Retain the explicit 3 GiB setting
when testing this loader. The default remains `auto`; production's custom fast
loader is not included in this upstream recipe.

`--vocab-mode embedding` keeps the target LM head dense. `--vocab-mode all` also
packs it and has different memory/performance characteristics. Explicit flags
above override the CLI's smaller development defaults (8k context, 16 slots,
prefix caching off). No failed capacity check silently reduces these settings.

Startup reports verify all 68 KDA projections, all 84 expert-scale tensors and
selected vocabulary consumers. They record actual allocated/reserved memory
following drafter, cache and graph initialization. Use those measurements for the
new recipe; the earlier 14.877 GiB saving against ModelOpt is not a measurement of
this dense-head, upstream-vLLM deployment.

V6/V7 receipts account for **3.47044 / 3.47069 GiB per rank** of lossless
scale/embedding storage savings after the reported expert workspace, **6.94113 GiB
across TP2**. These are resident tensor bytes, not a matched total CUDA allocation
comparison. The lossy KDA conversion separately reduces allocation by 2.10890 GiB
per rank during loading. Neither observation is a new saving-versus-production
measurement. Original 500k production was verified restored at 16:44:58 UTC.

## Qualification and release

The included `gpu_check.py` exercises native codecs, GEMMs and CUDA graph replay.
`smoke_client.py` checks the configured context, repeated-prefix reuse and output
identity. `control_stock_trial.py` is an operational harness for the owner's two
Sparks, with exact-container restoration and an independent watchdog; it is not
needed by consumers and must not be reused against another production deployment.

After startup passes, run the frozen realistic workloads and long-context probes,
then full GSM8K with speculation off. Keep failed trials and report runtime
configuration differences. Publish the wheel, source, notices, asset manifest and
reproduction commands together only after the release assets and GPU checks are
complete. The engineering report must separate lossless packing from lossy KDA
conversion and measured gains from estimates.
