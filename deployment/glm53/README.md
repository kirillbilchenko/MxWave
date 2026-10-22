# GLM CT on upstream vLLM 0.32.0

This is the default deployment target: the official vLLM image with the MxWave
wheel installed through vLLM's plugin mechanism. No vLLM source files are replaced.
The extension wraps version-pinned internal APIs and requires the verified bundle.

**22 October follow-up:** the same-production-runtime comparison is complete.
After exact restoration, the user explicitly selected its CT/Inco profile for
live use; that separate handover completed at 10:00 UTC. Its favorable CT medians use the custom runtime and Inco
drafter; they do not replace the official-image results below. Both repeatability
and the observed quality screen fail. Stock-vLLM compatibility remains mandatory;
no fast-path or KV change is applied to this recipe.
[Separate final report](../../docs/GLM53_SAME_RUNTIME_RESULTS.md).
[Live custom-runtime service and rollback](../../docs/GLM53_CT_LIVE.md).

**GPU-tested status, 21 October 2026:** the corrected recipe starts at **500k
configured context**, with prefix caching and useful DFlash acceptance. All three
paired timing blocks complete without API errors. Against the existing tuned
ModelOpt/Inco production deployment, this stock CT recipe measures **about 22%
lower C1 throughput**, **15% higher code / 21% higher chat aggregate throughput at
C8**, on the frozen short prompts. C8 first-token time improves substantially,
while mean individual completion latency increases by about 1–2 seconds. These
are complete deployments with different draft policies and runtime optimizations.
[Full ranges, latency, acceptance and retained evidence](../../docs/GLM53_HANDOFF.md).

The final 300-question speculation-off GSM8K screen is complete: production
**289/300 (96.33%)**, CT **287/300 (95.67%)**, zero API errors or truncations in
both arms. This small subset does not establish equivalence or a meaningful gap.
This is an **experimental opt-in recipe**, with no promotion. Original production
was restored before both Sparks were powered off; complete source and evidence
backups were independently verified first. No more experiments are queued.
The original strict KL and output-repeat failures remain recorded. A 500k startup
is not a 32k/128k retrieval or long-context quality qualification; those probes
were deliberately omitted from the final bounded screen.

Run **`prepare-drafter`** before serving. Native 0.32 otherwise applies FP8 loading
to twelve existing BF16 convolution matrices without scales. The config-only
correction preserves all weight bytes; `serve` verifies the corrected metadata
and pinned weights. No training is needed.
[Diagnosis and evidence](../../docs/GLM53_STOCK_DRAFTER_LOADING.md).

All three timing starts reproduce **87.202210 / 87.202180 GiB** allocated per rank,
including drafter, KV and graphs (**174.404390 GiB TP2**). Lossless scale/embedding
packing saves **6.941 GiB TP2** after expert scratch space. This is resident storage
accounting, not a newly measured total-allocation delta against production; the
historical 14.877 GiB result used a different profile.

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

The following uses the **500k capacity settings from the final three-pair screen**,
with the corrected drafter metadata. Startup, positive draft acceptance and actual
prefix reuse pass. Earlier repeated-output identity and strict KL failures remain;
the recipe is not promoted to production. Long-context retrieval probes were not run.

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
measurement. Restoration and final shutdown status are recorded in the handoff.

## Qualification and release

The included `gpu_check.py` exercises native codecs, GEMMs and CUDA graph replay.
`smoke_client.py` checks the configured context, repeated-prefix reuse and output
identity. `control_stock_trial.py` is an operational harness for the owner's two
Sparks, with exact-container restoration and an independent watchdog; it is not
needed by consumers and must not be reused against another production deployment.

The owner's final screen is deliberately limited to three short C1/C8 code/chat
pairs and a fixed 300-question GSM8K subset per arm, with speculation off for
quality. It retains earlier numerical failures and every observed result; it does
not certify general quality or long-context serving. No further experiments are
queued after that screen and the authorized shutdown.

The wheel and working owner recipe are retained. A self-contained public asset
release and reproducible asset-build instructions are still unavailable. Any future
publication should include source, notices, asset manifests and reproduction commands,
and distinguish lossless packing from lossy KDA conversion and measured gains from
estimates. The current recipe must not be advertised as a universally faster or
fully lossless replacement for the tuned production deployment.
