# Qwen3.8-27B H64 MXFP4 reproducibility record

This record freezes the procedure used for the recommended MxWave
`Qwen/Qwen3.8-27B` artifact: 64 real calibration sequences of 512 tokens,
block-Hessian scale selection, and no experimental rounding or error feedback.
It covers source acquisition, calibration, quantization, checkpoint verification,
serving, paired WikiText-2 likelihood, exact next-token divergence, and the
directional GSM8K pilot.

Recorded on 2026-09-15. Publishing is deliberately out of scope.

The numerical replay described below was completed immediately before the
project and Python package were renamed to MxWave. The rename changed imports,
console commands, and generated provenance filenames; it did not change the
quantization math. Historical benchmark JSON intentionally retains its original
schema labels so its recorded hashes remain valid. The public release baseline is
[MxWave commit `0179e131`](https://github.com/kirillbilchenko/MxWave/tree/0179e131544f807ecdb6d04e7e914f181dd21c9f),
and the payload checks in Sections 4–7 were repeated before publication.

## Reproduced status

The numerical procedure was replayed on the same DGX Spark with inference and
OpenWebUI traffic stopped:

- All 400 replayed block-Hessian tensors were bitwise equal to the original
  calibration tensors; the maximum absolute difference was `0.0`.
- All 18 quantized checkpoint shards, `config.json`, and
  `model.safetensors.index.json` reproduced exactly. Their hash-of-hashes was
  `e8675da3448ae399b84cff0d99cd3c246d678ba5883e91ad4e7a370a993385d7`.
- The complete numerical inference payload, including copied tokenizer and
  configuration assets but excluding generated provenance, evaluation, license,
  and model-card files, reproduced exactly. Its hash-of-hashes was
  `bbccb79edcaadbacc2f43609c3c35b51ca07b32865a19ac6e1a4fbaa82e35a11`.
- Calibration replay took 98.67 seconds and peaked at 8.07 GiB process RSS and
  3.28 GiB allocated CUDA memory. Quantization replay took about 235 seconds.

For Hub publication, a redundant group-level `"format": null` was omitted from
`config.json`. The global `"format": "mxfp4-pack-quantized"` is unchanged;
compressed-tensors 0.17 and the pinned vLLM parser both resolve the omitted field
to `None`. No tensor, index, tokenizer, or evaluation file changed. The published
release therefore has these metadata-normalized hashes:

- `config.json`: `d19a06baa5074b9f1f7b38296d846852fa2c071d2cebdd833e409206fc3feff0`;
- 18 shards plus config and index: `6f81fb0182aca2a712b1d688d96871c81ef75e04bcc8622c79275aca60828d88`;
- complete inference payload: `0600d65150f8e44713897f6515af0c7ddf704d148e83c148217173bf1ff4917b`.

The `.safetensors` calibration *file* is not expected to have a stable file hash:
its metadata intentionally records elapsed time and peak memory. Likewise,
`mxwave-manifest.json`, `mxwave-run.json`, and the generated model README can
change when provenance fields are added. Reproducibility is therefore checked at
three explicit levels:

1. calibration tensor equality;
2. inference-payload hashes;
3. paired evaluation inputs, tokenization, and likelihood results.

Do not use a whole-directory hash as the numerical reproducibility criterion.

## Frozen inputs and environment

| Item | Frozen value |
|---|---|
| Source model | `Qwen/Qwen3.8-27B` |
| Source revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Source checkpoint size | 55,562,855,904 bytes of tensor data; 51.75 GiB on disk |
| Calibration dataset | `EleutherAI/pile_val_test`, first 100 validation rows |
| Calibration repository revision | `05b327037e6301f256d8df32193756edc4c8e3bd` |
| Calibration corpus SHA-256 | `07ffd944ea4fc09f12e5d019633a77309199fdced749e3ddf13989e53586f36a` |
| Calibration token IDs SHA-256 | `e68f97cb4044312513df4c9c82bd758abe348a706fd7e1f11fe9ee15c83efa05` |
| Runtime image | `vllm/vllm-openai:qwen38-flash-next@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8` |
| Quantizer repository | `https://github.com/kirillbilchenko/MxWave` |
| Quantizer release baseline | `0179e131544f807ecdb6d04e7e914f181dd21c9f` |
| GPU | NVIDIA GB10, compute capability 12.1 / SM121 |
| Driver | 580.142 |
| Host | Ubuntu 24.04.4 LTS, Linux `6.17.0-1014-nvidia`, aarch64 |
| Docker | 29.2.1 |

The image contained Python 3.12.3, PyTorch `2.13.0+cu130`, CUDA 13.0,
Transformers 5.15.1, Accelerate 1.14.0, safetensors 0.8.0, NumPy 2.2.6,
vLLM `0.1.dev20073+g8e685d198`, compressed-tensors 0.17.0, and
FlashInfer 0.6.17.

The MxWave release-candidate source digest is computed as follows:

```bash
(find mxwave -type f -name '*.py' -print; printf '%s\n' pyproject.toml) \
  | LC_ALL=C sort \
  | xargs sha256sum \
  | sha256sum
```

Expected:

```text
f20799aecad6518995bfade0fb03c735d18edc6128745e5f57fabfbb88620bf7  -
```

The historical bitwise replay used the already synchronized Spark source digest
`b707b2c6fefcf87faad43b635191c464b031a4d977203ad62ef9354405c8fc5b`.
That digest predates the package/CLI rename and subsequent release cleanup, so it
is expected to differ from the release-candidate digest above. The stable
calibration-tensor and inference-payload hashes document the completed replay;
they do not substitute for the pinned public release commit.

## Resource and isolation requirements

- Stop every serving container before calibration, quantization, or evaluation.
- Do not send OpenWebUI requests during evaluation.
- For a fresh machine, reserve at least 80 GiB free disk for the 51.75 GiB source,
  approximately 18.47 GiB output, calibration artifact, and working files.
- DGX Spark CPU and GPU allocations share unified memory. Use the bounded
  `streaming` calibration mode and the `1024`-row quantization chunk below.
- Serve MXFP4 with Marlin and `--gpu-memory-utilization 0.45` for this comparison.
  Do not add `--enforce-eager`.

The commands below assume the existing `../local-spark` layout and SSH alias
`spark`. Commands marked **local** run from this MxWave repository. Commands
marked **Spark** run after `ssh spark`.

## 1. Synchronize the exact MxWave source

Clone the public tool, check out the pinned quantizer baseline, and verify the
source digest above:

```bash
# local
git clone https://github.com/kirillbilchenko/MxWave.git
cd MxWave
git checkout 0179e131544f807ecdb6d04e7e914f181dd21c9f
```

Create a new destination, then copy only the committed runtime files to it:

```bash
# local
ssh spark 'mkdir -p "$HOME/local-spark/deployment/mxwave-h64-repro"'
```

```bash
# local
git status --short
git archive --format=tar HEAD \
  pyproject.toml mxwave \
  scripts/evaluate_api_perplexity.py scripts/evaluate_next_token_kl.py \
  | ssh spark 'tar -xf - -C "$HOME/local-spark/deployment/mxwave-h64-repro"'
```

On Spark, define the paths used by all later commands:

```bash
# Spark
export SPARK_ROOT="${SPARK_ROOT:-$HOME/local-spark}"
export IMAGE='vllm/vllm-openai:qwen38-flash-next@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8'
export SOURCE="$SPARK_ROOT/models/qwen3.8-27b-bf16/huggingface"
export CODE_ROOT="$SPARK_ROOT/deployment/mxwave-h64-repro"
export CAL_ROOT="$SPARK_ROOT/calibration"
export REPRO_ROOT="$SPARK_ROOT/repro/qwen3.8-27b-h64"
export CAL_CORPUS="$REPRO_ROOT/mxwave-pileval-rows-0-99.json"
export STATS="$REPRO_ROOT/qwen3.8-27b-64x512.safetensors"
export OUTPUT="$REPRO_ROOT/model"
export EVAL_ROOT="$REPRO_ROOT/eval"
mkdir -p "$REPRO_ROOT" "$OUTPUT" "$EVAL_ROOT"
```

Confirm the image and machine before doing expensive work:

```bash
# Spark
docker image inspect "$IMAGE" --format '{{.Id}} {{index .RepoDigests 0}}'
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
docker --version
uname -a
```

No model-serving container should appear in this check:

```bash
# Spark
docker ps
free -h
df -h "$SPARK_ROOT"
```

## 2. Acquire and verify the BF16 source

The sibling local-spark downloader already pins the correct repository and
revision. It reads `HF_TOKEN` from local-spark's protected `.env`; never paste the
token into this record.

```bash
# local, from ../local-spark
SPARK_MODEL_PROFILE=qwen3.8-27b-bf16 bin/spark-llm download
```

Verify the model from Spark:

```bash
# Spark
cd "$SOURCE"
sha256sum config.json model.safetensors.index.json
sha256sum model-*.safetensors config.json model.safetensors.index.json \
  | LC_ALL=C sort -k2 \
  | sha256sum
```

Expected metadata hashes:

```text
191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab  config.json
77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df  model.safetensors.index.json
```

Expected source hash-of-hashes:

```text
44f35e79a08ef48880088b0de07cd0d06796060f8c257fc6f352bbde2f3ac2d3  -
```

Individual source shard hashes are in [Appendix A](#appendix-a-source-shard-hashes).

## 3. Acquire and verify the calibration corpus

The original run used the exact JSON response below. The repository revision is
recorded above, while the byte hash is the operational lock because the dataset
viewer URL itself does not expose a revision parameter.

```bash
# Spark
curl -fsSL --retry 3 \
  'https://datasets-server.huggingface.co/rows?dataset=EleutherAI%2Fpile_val_test&config=default&split=validation&offset=0&length=100' \
  -o "$CAL_CORPUS"
printf '%s  %s\n' \
  '07ffd944ea4fc09f12e5d019633a77309199fdced749e3ddf13989e53586f36a' \
  "$CAL_CORPUS" \
  | sha256sum --check
```

If that byte check fails, stop. Do not silently calibrate on a changed corpus.

## 4. Dry-run the calibration contract

This validates tokenizer output, target coverage, sequential decoder support, and
the corpus before weights are traversed:

```bash
# Spark
docker run --rm --gpus all --ipc host --network none \
  --user 1000:1000 --workdir /tmp --entrypoint python3 \
  -e HF_HUB_OFFLINE=1 \
  -e PYTHONPATH=/opt/mxwave \
  -e XDG_CACHE_HOME=/tmp/.cache \
  -v "$SOURCE:/input:ro" \
  -v "$REPRO_ROOT:/repro" \
  -v "$CODE_ROOT:/opt/mxwave:ro" \
  "$IMAGE" \
  -m mxwave.calibration_cli \
  --model-dir /input \
  --corpus /repro/mxwave-pileval-rows-0-99.json \
  --output /repro/qwen3.8-27b-64x512.safetensors \
  --policy qwen3.8-27b-compatible \
  --statistics block-hessian \
  --num-sequences 64 \
  --sequence-offset 0 \
  --sequence-length 512 \
  --batch-size 1 \
  --weight-loading streaming \
  --text-field text \
  --device cuda \
  --dtype bfloat16 \
  --attention-implementation sdpa \
  --hessian-damp 1e-6 \
  --source-repository Qwen/Qwen3.8-27B \
  --source-revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --dry-run
```

The dry-run must report:

```text
policy: qwen3.8-27b-compatible
targets: 400
objectives: [block-hessian]
sequences: 64
sequence_offset: 0
sequence_length: 512
tokens: 32768
weight_loading: streaming
corpus_sha256: 07ffd944ea4fc09f12e5d019633a77309199fdced749e3ddf13989e53586f36a
token_ids_sha256: e68f97cb4044312513df4c9c82bd758abe348a706fd7e1f11fe9ee15c83efa05
```

## 5. Capture streaming block-Hessian statistics

Run the same command without `--dry-run`:

```bash
# Spark
docker run --rm --gpus all --ipc host --network none \
  --user 1000:1000 --workdir /tmp --entrypoint python3 \
  -e HF_HUB_OFFLINE=1 \
  -e PYTHONPATH=/opt/mxwave \
  -e XDG_CACHE_HOME=/tmp/.cache \
  -v "$SOURCE:/input:ro" \
  -v "$REPRO_ROOT:/repro" \
  -v "$CODE_ROOT:/opt/mxwave:ro" \
  "$IMAGE" \
  -m mxwave.calibration_cli \
  --model-dir /input \
  --corpus /repro/mxwave-pileval-rows-0-99.json \
  --output /repro/qwen3.8-27b-64x512.safetensors \
  --policy qwen3.8-27b-compatible \
  --statistics block-hessian \
  --num-sequences 64 \
  --sequence-offset 0 \
  --sequence-length 512 \
  --batch-size 1 \
  --weight-loading streaming \
  --text-field text \
  --device cuda \
  --dtype bfloat16 \
  --attention-implementation sdpa \
  --hessian-damp 1e-6 \
  --source-repository Qwen/Qwen3.8-27B \
  --source-revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
```

Expected stable metadata:

| Field | Value |
|---|---:|
| targets | 400 |
| sequences | 64 |
| sequence length | 512 |
| observations per target | 32,768 |
| Hessian damping | `1e-6` |
| objective | `block-hessian` |
| weight loading | `streaming` |

The recorded original stats artifact is
`$CAL_ROOT/qwen3.8-27b-64x512.safetensors`, with historical file SHA-256
`5f7f6380d2ffd9731c0e4463d89be808c23a2b7101f5d3544eb4124f51c160de`.
The replay file SHA-256 was
`4f9e726a77d2f9f9e304ef18fafc8ef103e12caccf00d3c9950369809824483f`;
the difference is runtime telemetry in safetensors metadata.

Compare tensor payloads, not those whole-file hashes:

```bash
# Spark
docker run --rm -i --network none --entrypoint python3 \
  -v "$CAL_ROOT:/calibration:ro" \
  -v "$REPRO_ROOT:/repro:ro" \
  "$IMAGE" - <<'PY'
import torch
from safetensors import safe_open

original_path = "/calibration/qwen3.8-27b-64x512.safetensors"
replay_path = "/repro/qwen3.8-27b-64x512.safetensors"

with safe_open(original_path, framework="pt", device="cpu") as original:
    with safe_open(replay_path, framework="pt", device="cpu") as replay:
        assert list(original.keys()) == list(replay.keys())
        maximum = 0.0
        for key in original.keys():
            expected = original.get_tensor(key)
            actual = replay.get_tensor(key)
            assert expected.dtype == actual.dtype
            assert expected.shape == actual.shape
            assert torch.equal(expected, actual), key
            maximum = max(maximum, float((expected - actual).abs().max()))
        print(f"bitwise-equal tensors: {len(list(original.keys()))}")
        print(f"maximum absolute difference: {maximum}")
PY
```

Expected:

```text
bitwise-equal tensors: 400
maximum absolute difference: 0.0
```

## 6. Dry-run and execute quantization

The quality configuration is intentionally scale-only: the block Hessian
selects the shared exponent for each 32-value block, followed by ordinary
nearest-E2M1 code assignment.

First run the plan:

```bash
# Spark
docker run --rm --gpus all --ipc host --network none \
  --user 1000:1000 --workdir /tmp --entrypoint python3 \
  -e HF_HUB_OFFLINE=1 \
  -e PYTHONPATH=/opt/mxwave \
  -e XDG_CACHE_HOME=/tmp/.cache \
  -v "$SOURCE:/input:ro" \
  -v "$OUTPUT:/output" \
  -v "$REPRO_ROOT:/repro:ro" \
  -v "$CODE_ROOT:/opt/mxwave:ro" \
  "$IMAGE" \
  -m mxwave.cli \
  --model-dir /input \
  --output-dir /output \
  --policy qwen3.8-27b-compatible \
  --method mse \
  --scale-percentile 99.5 \
  --mse-clip-depth 4 \
  --tensor-row-chunk-size 1024 \
  --activation-stats /repro/qwen3.8-27b-64x512.safetensors \
  --calibration-objective block-hessian \
  --device cuda \
  --source-repository Qwen/Qwen3.8-27B \
  --source-revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --verify-sqnr \
  --sqnr-rows 16 \
  --dry-run
```

The plan must report 18 shards, 1,199 source tensors, 400 target tensors,
799 passthrough tensors, and projected output tensor data of 19,832,628,704
bytes with a 2.8016x global compression ratio.

Then execute the identical command without `--dry-run`:

```bash
# Spark
docker run --rm --gpus all --ipc host --network none \
  --user 1000:1000 --workdir /tmp --entrypoint python3 \
  -e HF_HUB_OFFLINE=1 \
  -e PYTHONPATH=/opt/mxwave \
  -e XDG_CACHE_HOME=/tmp/.cache \
  -v "$SOURCE:/input:ro" \
  -v "$OUTPUT:/output" \
  -v "$REPRO_ROOT:/repro:ro" \
  -v "$CODE_ROOT:/opt/mxwave:ro" \
  "$IMAGE" \
  -m mxwave.cli \
  --model-dir /input \
  --output-dir /output \
  --policy qwen3.8-27b-compatible \
  --method mse \
  --scale-percentile 99.5 \
  --mse-clip-depth 4 \
  --tensor-row-chunk-size 1024 \
  --activation-stats /repro/qwen3.8-27b-64x512.safetensors \
  --calibration-objective block-hessian \
  --device cuda \
  --source-repository Qwen/Qwen3.8-27B \
  --source-revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --verify-sqnr \
  --sqnr-rows 16
```

`--resume` may be added only when continuing the same validated partial output.
For a clean reproduction, start with a new empty output directory.

## 7. Verify the emitted checkpoint

Check the stable contract recorded in the manifest:

```bash
# Spark
jq '{
  policy,
  method,
  scale_percentile,
  mse_clip_depth,
  tensor_row_chunk_size,
  weight_scale_selection,
  source_tensors,
  target_tensors,
  passthrough_tensors,
  actual_output_bytes,
  global_compression_ratio,
  activation_calibration,
  sqnr_db: (.sqnr_db | del(.per_tensor)),
  calibration_weighted_sqnr_db: (.calibration_weighted_sqnr_db | del(.per_tensor))
}' "$OUTPUT/mxwave-manifest.json"
```

Expected key values:

| Field | Expected |
|---|---:|
| weight scale selection | `mse-activation-block-hessian` |
| source / target / passthrough tensors | 1,199 / 400 / 799 |
| actual output bytes | 19,832,831,240 |
| ordinary SQNR, mean / minimum | 18.820832 / 17.329132 dB |
| block-Hessian SQNR, mean / minimum | 19.140304 / 17.254692 dB |
| SQNR coverage | 400 / 400 |

Verify the core checkpoint payload:

```bash
# Spark
cd "$OUTPUT"
sha256sum model-*.safetensors model.safetensors.index.json config.json \
  | LC_ALL=C sort -k2 \
  | sha256sum
```

Expected:

```text
6f81fb0182aca2a712b1d688d96871c81ef75e04bcc8622c79275aca60828d88  -
```

Verify all inference files while excluding generated provenance and prose:

```bash
# Spark
cd "$OUTPUT"
find . -maxdepth 1 -type f \
  ! -name 'mxwave-manifest.json' \
  ! -name 'mxwave-run.json' \
  ! -name 'README.md' \
  ! -name 'REPRODUCIBILITY.md' \
  ! -name 'LICENSE' \
  -print0 \
  | LC_ALL=C sort -z \
  | xargs -0 sha256sum \
  | sha256sum
```

Expected:

```text
0600d65150f8e44713897f6515af0c7ddf704d148e83c148217173bf1ff4917b  -
```

Individual output shard hashes are in
[Appendix B](#appendix-b-output-shard-hashes).

## 8. Build the exact evaluation corpus

The evaluation source is `Salesforce/wikitext`, configuration
`wikitext-2-raw-v1`, test split, revision
`b08601e04326c79dfdd32d625aee71d232d685c3`. The pinned parquet is:

```text
https://huggingface.co/datasets/Salesforce/wikitext/resolve/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1/test-00000-of-00001.parquet
```

Its SHA-256 is
`5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91`.
The historical text file was created by joining all 4,358 viewer rows with two
newlines and no added trailing newline. This standard-library command recreates
that file and enforces its final byte hash:

```bash
# Spark
python3 - "$EVAL_ROOT/wikitext-2-raw-v1-test.txt" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
import urllib.parse
import urllib.request

destination = Path(sys.argv[1])
rows = []
base = "https://datasets-server.huggingface.co/rows"
for offset in range(0, 4358, 100):
    query = urllib.parse.urlencode(
        {
            "dataset": "Salesforce/wikitext",
            "config": "wikitext-2-raw-v1",
            "split": "test",
            "offset": offset,
            "length": min(100, 4358 - offset),
        }
    )
    with urllib.request.urlopen(f"{base}?{query}", timeout=120) as response:
        payload = json.load(response)
    rows.extend(item["row"]["text"] for item in payload["rows"])

assert len(rows) == 4358
content = "\n\n".join(rows).encode("utf-8")
digest = hashlib.sha256(content).hexdigest()
assert digest == "696cca6b65a171b0a358a4be6732cdfdf2dd6164a32e20fd70e3c13fc4dfae83"
destination.write_bytes(content)
print(f"rows={len(rows)} bytes={len(content)} sha256={digest}")
PY
```

Expected:

```text
rows=4358 bytes=1296370 sha256=696cca6b65a171b0a358a4be6732cdfdf2dd6164a32e20fd70e3c13fc4dfae83
```

The final corpus hash is the pass/fail lock. If the mutable viewer service ever
changes, read the pinned parquet with a parquet implementation and apply the same
ordered two-newline join.

## 9. Serve one model at a time with the evaluation configuration

Use a dedicated container and cache. The API key file is mounted, never printed.
For MxWave and AMD MXFP4, run:

```bash
# Spark; set MODEL_PATH and MODEL_ALIAS before each run
export MODEL_PATH="$OUTPUT"
export MODEL_ALIAS=qwen3.8-27b-mxwave-hessian64-d4
export MODEL_CACHE="$EVAL_ROOT/cache-$MODEL_ALIAS"
mkdir -p "$MODEL_CACHE"

docker run --rm -d --name qwen27-ppl \
  --gpus all --ipc host \
  -p 127.0.0.1:8000:8000 \
  --entrypoint bash \
  -e HF_HUB_OFFLINE=1 \
  -e SERVED_MODEL="$MODEL_ALIAS" \
  -e FLASHINFER_WORKSPACE_BASE=/cache/flashinfer \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor \
  -e TRITON_CACHE_DIR=/cache/triton \
  -v "$MODEL_PATH:/model:ro" \
  -v "$MODEL_CACHE:/cache" \
  -v "$SPARK_ROOT/runtime/secrets/llm_api_key:/run/secrets/llm_api_key:ro" \
  "$IMAGE" -lc '
IFS= read -r VLLM_API_KEY </run/secrets/llm_api_key
export VLLM_API_KEY
exec vllm serve /model \
  --served-model-name "$SERVED_MODEL" \
  --host 0.0.0.0 \
  --port 8000 \
  --load-format safetensors \
  --max-model-len 4096 \
  --max-num-seqs 2 \
  --gpu-memory-utilization 0.45 \
  --no-enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-batched-tokens 4096 \
  --kv-cache-dtype bfloat16 \
  --linear-backend marlin \
  --default-chat-template-kwargs "{\"enable_thinking\":false}" \
  --no-enable-flashinfer-autotune \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3'

until curl -fsS http://127.0.0.1:8000/health >/dev/null; do sleep 5; done
```

For the AMD comparison, use the same command after changing only:

```bash
# Spark
export MODEL_PATH="$SPARK_ROOT/models/qwen3.8-27b-amd-awq-mxfp4/huggingface"
export MODEL_ALIAS=qwen3.8-27b-amd-awq-mxfp4
export MODEL_CACHE="$EVAL_ROOT/cache-$MODEL_ALIAS"
mkdir -p "$MODEL_CACHE"
```

The frozen AMD artifact is
`amd/Qwen3.8-27B-Quark-AWQ-MXFP4@5233554c5fa56afda40150556b95573c2d7d29c0`.
Its `model.safetensors` is 19,798,196,184 bytes with SHA-256
`be1d745bc7312fdf1486059ec57cdeb514cc4d1aa06528c6677a0ebc0a0e1272`.
It can be acquired through local-spark with:

```bash
# local, from ../local-spark
SPARK_MODEL_PROFILE=qwen3.8-27b-amd-awq-mxfp4 bin/spark-llm download
```

For BF16, use this complete command. It omits `--linear-backend marlin` and uses
the recorded `0.65` memory fraction:

```bash
# Spark
export MODEL_PATH="$SOURCE"
export MODEL_ALIAS=qwen3.8-27b-bf16
export MODEL_CACHE="$EVAL_ROOT/cache-$MODEL_ALIAS"
mkdir -p "$MODEL_CACHE"

docker run --rm -d --name qwen27-ppl \
  --gpus all --ipc host \
  -p 127.0.0.1:8000:8000 \
  --entrypoint bash \
  -e HF_HUB_OFFLINE=1 \
  -e SERVED_MODEL="$MODEL_ALIAS" \
  -e FLASHINFER_WORKSPACE_BASE=/cache/flashinfer \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor \
  -e TRITON_CACHE_DIR=/cache/triton \
  -v "$MODEL_PATH:/model:ro" \
  -v "$MODEL_CACHE:/cache" \
  -v "$SPARK_ROOT/runtime/secrets/llm_api_key:/run/secrets/llm_api_key:ro" \
  "$IMAGE" -lc '
IFS= read -r VLLM_API_KEY </run/secrets/llm_api_key
export VLLM_API_KEY
exec vllm serve /model \
  --served-model-name "$SERVED_MODEL" \
  --host 0.0.0.0 \
  --port 8000 \
  --load-format safetensors \
  --max-model-len 4096 \
  --max-num-seqs 2 \
  --gpu-memory-utilization 0.65 \
  --no-enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-batched-tokens 4096 \
  --kv-cache-dtype bfloat16 \
  --default-chat-template-kwargs "{\"enable_thinking\":false}" \
  --no-enable-flashinfer-autotune \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3'

until curl -fsS http://127.0.0.1:8000/health >/dev/null; do sleep 5; done
```

Never run two model containers concurrently for this protocol.

## 10. Run the prompt-perplexity evaluation

With one server healthy, execute:

```bash
# Spark
python3 "$CODE_ROOT/scripts/evaluate_api_perplexity.py" \
  --input "$EVAL_ROOT/wikitext-2-raw-v1-test.txt" \
  --output "$EVAL_ROOT/$MODEL_ALIAS-4k316.json" \
  --model "$MODEL_ALIAS" \
  --api-key-file "$SPARK_ROOT/runtime/secrets/llm_api_key" \
  --base-url http://127.0.0.1:8000/v1 \
  --chunk-characters 4096 \
  --num-chunks 316 \
  --workers 1 \
  --timeout-seconds 600 \
  --dataset Salesforce/wikitext:wikitext-2-raw-v1:test \
  --dataset-revision b08601e04326c79dfdd32d625aee71d232d685c3 \
  --dataset-sha256 5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91
```

Stop the server before changing models; `--rm` removes this exact test container
after it stops:

```bash
# Spark
docker stop qwen27-ppl
```

Repeat Sections 9 and 10 for MxWave, AMD, and BF16. The stable expected
aggregate results are:

| Artifact | Prompt PPL | Mean NLL | Scored tokens |
|---|---:|---:|---:|
| BF16 | 7.980864117 | 2.077046691 | 297,199 |
| MxWave H64 scale-only | 8.119444612 | 2.094261754 | 297,199 |
| AMD Quark-AWQ MXFP4 | 8.187543970 | 2.102613971 | 297,199 |

Each report must also contain 316 chunks, 1,294,336 scored characters, 297,515
prompt tokens, and matching per-window text hashes, token hashes, prompt token
counts, and scored token counts. Elapsed time fields make whole-report hashes
non-reproducible; the historical report hashes in
`benchmarks/qwen3.8-27b-wikitext2-prompt-ppl.json` identify the archived runs but
are not rerun pass criteria.

## 11. Recompute the paired comparison

The following command validates pairing and repeats the fixed-seed, 100,000
sample cluster bootstrap used in the benchmark. Pass MxWave first and AMD
second:

```bash
# Spark
docker run --rm -i --network none --entrypoint python3 \
  -v "$EVAL_ROOT:/eval:ro" \
  "$IMAGE" - \
  /eval/qwen3.8-27b-mxwave-hessian64-d4-4k316.json \
  /eval/qwen3.8-27b-amd-awq-mxfp4-4k316.json <<'PY'
import json
import math
from pathlib import Path
import sys

import numpy as np

candidate = json.loads(Path(sys.argv[1]).read_text())
baseline = json.loads(Path(sys.argv[2]).read_text())
candidate_chunks = candidate["chunks"]
baseline_chunks = baseline["chunks"]
assert len(candidate_chunks) == len(baseline_chunks) == 316

pair_fields = ("index", "text_sha256", "token_sha256", "prompt_tokens", "scored_tokens")
for left, right in zip(candidate_chunks, baseline_chunks, strict=True):
    assert all(left[field] == right[field] for field in pair_fields)

candidate_nll = np.array([row["negative_log_likelihood"] for row in candidate_chunks])
baseline_nll = np.array([row["negative_log_likelihood"] for row in baseline_chunks])
tokens = np.array([row["scored_tokens"] for row in candidate_chunks])
candidate_mean = candidate_nll / tokens
baseline_mean = baseline_nll / tokens

candidate_ppl = math.exp(float(candidate_nll.sum() / tokens.sum()))
baseline_ppl = math.exp(float(baseline_nll.sum() / tokens.sum()))
relative = 100.0 * (candidate_ppl / baseline_ppl - 1.0)

wins = int(np.sum(candidate_mean < baseline_mean))
losses = int(np.sum(candidate_mean > baseline_mean))
ties = len(candidate_chunks) - wins - losses
trials = wins + losses
tail = min(wins, losses)
sign_p = min(
    1.0,
    2.0 * sum(math.comb(trials, k) for k in range(tail + 1)) / (2**trials),
)

rng = np.random.default_rng(20260915)
bootstrap = []
for _ in range(100):
    indices = rng.integers(0, len(tokens), size=(1000, len(tokens)))
    sampled_tokens = tokens[indices].sum(axis=1)
    candidate_samples = np.exp(candidate_nll[indices].sum(axis=1) / sampled_tokens)
    baseline_samples = np.exp(baseline_nll[indices].sum(axis=1) / sampled_tokens)
    bootstrap.extend(100.0 * (candidate_samples / baseline_samples - 1.0))
low, high = np.quantile(np.asarray(bootstrap), [0.025, 0.975])

largest = int(np.argmin(candidate_mean - baseline_mean))
keep = np.arange(len(tokens)) != largest
without_largest = 100.0 * (
    math.exp(float(candidate_nll[keep].sum() / tokens[keep].sum()))
    / math.exp(float(baseline_nll[keep].sum() / tokens[keep].sum()))
    - 1.0
)

print(
    json.dumps(
        {
            "candidate_ppl": candidate_ppl,
            "baseline_ppl": baseline_ppl,
            "relative_percent": relative,
            "paired_95_percent_ci": [float(low), float(high)],
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "two_sided_sign_test_p": sign_p,
            "largest_favorable_window_one_based": largest + 1,
            "relative_without_largest_percent": without_largest,
        },
        indent=2,
    )
)
PY
```

Expected MxWave-versus-AMD result:

```text
relative_percent: -0.831743%
paired_95_percent_ci: [-1.227040%, -0.531459%]
wins / losses / ties: 204 / 112 / 0
two_sided_sign_test_p: 2.5465891e-7
largest_favorable_window_one_based: 170
relative_without_largest_percent: -0.684463%
```

Negative relative PPL favors MxWave. The result supports this particular
deterministic likelihood protocol; it is not a claim of universal downstream
task superiority.

## 12. Run the exact next-token divergence evaluation

This evaluation compares complete next-token distributions rather than only the
probability assigned to the observed corpus token. It uses 128 evenly spaced
WikiText windows, the first 512 tokens of each selected window, and all 248,320
output token IDs. Raw matrices are about 122 MiB per model.

Prepare the immutable context manifest with the BF16 tokenizer:

```bash
# Spark
export KL_ROOT="$REPRO_ROOT/next-token-kl"
mkdir -p "$KL_ROOT"
cp "$CODE_ROOT/scripts/evaluate_next_token_kl.py" "$KL_ROOT/"

docker run --rm --network none --entrypoint python3 \
  -e HF_HUB_OFFLINE=1 \
  -v "$SOURCE:/model:ro" \
  -v "$EVAL_ROOT/wikitext-2-raw-v1-test.txt:/data/wikitext.txt:ro" \
  -v "$KL_ROOT:/run" \
  "$IMAGE" \
  /run/evaluate_next_token_kl.py prepare \
  --model /model \
  --corpus /data/wikitext.txt \
  --output /run/contexts.json \
  --num-contexts 128 \
  --context-tokens 512 \
  --chunk-characters 4096
```

Expected context contract:

```text
available non-empty chunks: 316
context manifest SHA-256: 9bac968222690a17af2c53983c900482fd0db4061df5a2b1403db9a4c786e7bf
selected contexts: 128
selected-context token digest: 9f5d0489d449bd0126bfc7f0327ecae7dfd7142b8167cc526b2f8c6722d59a76
```

Use this bounded helper to load one model at a time. The collector validates
complete token-ID coverage, the single repeated sampled-token entry returned by
this vLLM build, finite values, and log-probability normalization. Each model has
a 30-minute hard stop and a 110 GiB container memory limit.

```bash
# Spark
collect_kl() {
  model_path=$1
  model_label=$2
  linear_backend=$3
  cache_root="$KL_ROOT/cache-$model_label"
  mkdir -p "$cache_root/runtime" "$cache_root/engine"
  backend_args=()
  if [ -n "$linear_backend" ]; then
    backend_args=(--linear-backend "$linear_backend")
  fi

  timeout --signal=TERM --kill-after=60s 1800s \
    docker run --rm --gpus all --ipc host --shm-size 16g \
    --memory 110g --memory-swap 110g \
    --entrypoint python3 \
    -e HF_HUB_OFFLINE=1 \
    -e VLLM_TEST_FORCE_FP8_MARLIN=1 \
    -e TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor \
    -e TRITON_CACHE_DIR=/cache/triton \
    -e FLASHINFER_WORKSPACE_BASE=/cache/flashinfer \
    -v "$model_path:/model:ro" \
    -v "$cache_root/runtime:/cache" \
    -v "$cache_root/engine:/root/.cache/vllm" \
    -v "$KL_ROOT:/run" \
    "$IMAGE" \
    /run/evaluate_next_token_kl.py collect \
    --model /model \
    --model-label "$model_label" \
    --contexts /run/contexts.json \
    --output "/run/$model_label-logprobs.safetensors" \
    --runtime-image "$IMAGE" \
    --max-model-len 1024 \
    --gpu-memory-utilization 0.65 \
    "${backend_args[@]}"
}

collect_kl "$OUTPUT" mxwave-h64 marlin
collect_kl "$SOURCE" bf16 ''
collect_kl \
  "$SPARK_ROOT/models/qwen3.8-27b-amd-awq-mxfp4/huggingface" \
  amd-quark marlin
```

Compare the stored distributions on CPU:

```bash
# Spark
docker run --rm --network none --entrypoint python3 \
  -v "$KL_ROOT:/run" \
  "$IMAGE" \
  /run/evaluate_next_token_kl.py compare \
  --reference /run/bf16-logprobs.safetensors \
  --contexts /run/contexts.json \
  --candidate mxwave-h64=/run/mxwave-h64-logprobs.safetensors \
  --candidate amd-quark=/run/amd-quark-logprobs.safetensors \
  --output /run/next-token-divergence.json \
  --bootstrap-iterations 10000 \
  --bootstrap-seed 20260916
```

Historical artifact identifiers:

| File | SHA-256 |
|---|---|
| BF16 log probabilities | `f1f923777331ba518b78bdaefd5f840032e6eec08f7791683ab3e04564137462` |
| MxWave H64 log probabilities | `6be635e3dc99499f3ab4118c0dd6f964a4b585fa2cf384aa7d457ffc4dd35529` |
| AMD Quark log probabilities | `6ec912f6dc6edd38cc4cd3a6192d06f26ba8124b8af7fbf3f72425ee3c53cf4a` |
| Compact comparison report | `94692c3b1b9b975f6acf4cae06d9d3322570898f22e092dfa5dd37e075d15175` |

Expected aggregate result:

| Metric | MxWave H64 | AMD Quark |
|---|---:|---:|
| Mean forward KL from BF16 | 0.049937061 | 0.056735282 |
| Forward-KL p95 | 0.175240615 | 0.232539706 |
| Mean reverse KL | 0.044734070 | 0.048564505 |
| Mean Jensen-Shannon divergence | 0.010945828 | 0.012083333 |
| Mean total variation | 0.078565230 | 0.082361702 |
| BF16 top-1 agreement | 93.7500% | 91.4063% |

H64 has lower forward KL on 70 of 128 contexts and a 0.006798-nat lower
mean. The paired bootstrap interval for `H64 - AMD` is
`[-0.024242, +0.009835]` nats, so this evaluation favors H64 on aggregate but
does not establish a statistically conclusive advantage at 128 contexts.

## 13. Reproduce the GSM8K directional pilot

This run is intentionally labeled a pilot: it uses the first 100 GSM8K test
examples, sampled decoding, and 16 concurrent requests. It is useful for finding
large functional regressions but is not precise enough to rank a one-point gap.

After serving one model at a time with thinking disabled, run from the sibling
`local-spark` checkout:

```bash
# local, from ../local-spark; set MODEL_ALIAS for each served model
export MODEL_ALIAS=qwen3.8-27b-mxwave-hessian64-d4
export GSM_OUTPUT="$PWD/runtime/benchmarks/gsm8k-manual/$MODEL_ALIAS"
mkdir -p "$GSM_OUTPUT"

OPENAI_API_KEY="$(sed -n 's/^LLM_API_KEY=//p' .env)" \
uvx --from 'lm-eval[api]==0.4.13' lm-eval run \
  --model local-completions \
  --model_args \
  "model=$MODEL_ALIAS,base_url=http://127.0.0.1:18000/v1/completions,tokenizer_backend=none,num_concurrent=16,max_retries=5,timeout=900,tokenized_requests=False,max_length=16384,seed=1234" \
  --include_path evals/lm_eval \
  --tasks gsm8k_nothink \
  --gen_kwargs \
  max_gen_toks=1024 \
  temperature=0.7 \
  top_p=0.8 \
  top_k=20 \
  min_p=0.0 \
  presence_penalty=1.5 \
  repetition_penalty=1.0 \
  do_sample=True \
  --seed '0,1234,1234,1234' \
  --cache_requests true \
  --limit 100 \
  --output_path "$GSM_OUTPUT" \
  --log_samples
```

Repeat after changing `MODEL_ALIAS` to the BF16 and AMD served names. The task
hash must be
`567b35835d22441f13885c7e064da1b996f9d395724203056cebe01dcaa6e81c`.

| Model | Flexible numeric match | Strict `####` match | Raw report SHA-256 |
|---|---:|---:|---|
| BF16 | 88% | 85% | `3bda14da0132abc55b64c3f00ce4ef7641e1f432afe315593e5632abe177784d` |
| MxWave H64 | 92% | 91% | `1c5b238ff5cdb1629360a1956794babdc88fd6ddbea28a117d91ba099c4e73de` |
| AMD Quark | 93% | 93% | `331f88ddead14e5270bc22b4a5990a605905c52d4013321a996a61b31e368a63` |

The H64-versus-AMD difference is smaller than the reported standard errors.
Paired trace inspection found one BF16-and-AMD-correct/H64-wrong reasoning case
and one H64 strict-format-only failure. Treat the result as directional and use
Sections 10–12 as the primary fidelity evidence.

## Recorded quantization configuration

For avoidance of doubt, the complete quality-relevant configuration is:

```yaml
source:
  repository: Qwen/Qwen3.8-27B
  revision: 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
calibration:
  policy: qwen3.8-27b-compatible
  statistics: [block-hessian]
  sequences: 64
  sequence_offset: 0
  sequence_length: 512
  batch_size: 1
  weight_loading: streaming
  dtype: bfloat16
  attention_implementation: sdpa
  hessian_damp: 1.0e-6
quantization:
  policy: qwen3.8-27b-compatible
  method: mse
  scale_percentile: 99.5
  mse_clip_depth: 4
  tensor_row_chunk_size: 1024
  calibration_objective: block-hessian
  device: cuda
verification:
  sqnr: true
  sqnr_rows: 16
```

## Appendix A: source shard hashes

```text
ba0ce20aae489ad196733da5064bcdf159a1fe84f53336648196e1ebb7751b1c  model-00001-of-00018.safetensors
06a148c01bfbe3faa14a5f184a7ff29a706f7ae1c8b2705d2058e26d17a001fb  model-00002-of-00018.safetensors
2e1bf62cbcd406eaa64b60d10353e1f0ef4039d0976e56f05cabe953454f9968  model-00003-of-00018.safetensors
511e34063187882659753c4d93f3859f93c019fd438d8813071921c81d9a3f1a  model-00004-of-00018.safetensors
635cb53446dc74f219740fc59e18b774f877b803b9722e289ca62575a6efa701  model-00005-of-00018.safetensors
0bc5214fac607f0e6cc92eec3789d4b8559410ef9fce66621ba8158e8410dae0  model-00006-of-00018.safetensors
80b0c49033e9a0d5762562aa12f4acdb7f54da586f3d0110f28c48d91cf07892  model-00007-of-00018.safetensors
7192c5b66185d3592927daabee1cc19e6f6e0ce75988ee20e824b624765fda79  model-00008-of-00018.safetensors
af3c48cc37af44f3db6ae0579baf019180d48d9c527caa0a1f03ff85813a56d8  model-00009-of-00018.safetensors
163490a76f3bea3a40855b7efc04ce6d27afaf1a34f0bbde495b9491f76457c9  model-00010-of-00018.safetensors
5f3ae1b948aeee39da77aec558e8236cd65fe4d7cb7686a76bb007acc563c6d8  model-00011-of-00018.safetensors
a3de1c7114677a8f5ac5c4892c90e8238ea5c1e2038c80e757dfc87c3902ca55  model-00012-of-00018.safetensors
06ab79a41f74c9c5cb734816feb0c7fc364104b227165ee7391231e1155aa02a  model-00013-of-00018.safetensors
4138ed94603065ba884bbcadedb04d7718bb40117e85e6f5c6fc5b9c05b7a85b  model-00014-of-00018.safetensors
69224e27b9de4e7dbf6fc936c6eaae08447bda3b80a6c31a871ab451173afd22  model-00015-of-00018.safetensors
73cb9a1089fb6155cb648609478d6633be8a5c7d9ca5a05bc8925ce8a553cefe  model-00016-of-00018.safetensors
beb51f01056142ac4984bd800507b0dd0fd18de57f8e9ef6ea41d1a3598983a8  model-00017-of-00018.safetensors
1d3479509e21494658f9b64d317f5ea8e55c4025d28c702d6c4d0b356ce8ea06  model-00018-of-00018.safetensors
```

## Appendix B: output shard hashes

```text
bc83e46dad87ba8139ee911b92e75e87d905188e761ca43c1e8a3f46d4e2ae39  model-00001-of-00018.safetensors
0f798f1c5e0087047eafad648ad7ab09df74d0b3f58789868fc1413c3662df0a  model-00002-of-00018.safetensors
1e22bff91476ac31fd41fa6c5b6772286cacb458fcb0b13cd0d401d15504cf42  model-00003-of-00018.safetensors
5e7c67145b7b561c83945d7665be7c8ade7f0c36efb97822cd92761ba9e89c19  model-00004-of-00018.safetensors
1f45c4057ac61dba6675277af2e15d851bea4c7ec3da0d460fb111a487b2399b  model-00005-of-00018.safetensors
1fffa4cecbabbf0684b12a2ce62459881e79540b504359c643579179a8258557  model-00006-of-00018.safetensors
bf0393c8024cab1eef1cc707740d44a1a124a2ee1f435ebef73938b3bcca95af  model-00007-of-00018.safetensors
4a32f69500309784b286f5041b7a6b1b682cf0482d912a868f0a649b9c70c084  model-00008-of-00018.safetensors
e8baf29583bb580d85d068f46f20b64b3d231dd9018fe2ff13cb559962efaa61  model-00009-of-00018.safetensors
60303e0dd784bb0cd328828676e322db72face0f9569de4cba3f055a90348b03  model-00010-of-00018.safetensors
40954f425b94dd3d0b2c236cc3a46c6afcd64d1db711d8be3d9d551371f87e07  model-00011-of-00018.safetensors
6809ff898fe454aab2f98cd23cb9437b3e5a08d2881bd42fd7f6649315507ea3  model-00012-of-00018.safetensors
15bb397e3f6a74c6009d78d86a0648a245f5dac21520b9483a187445ce4bcf90  model-00013-of-00018.safetensors
7e9fb152349e5986b60738726d958f0701f79cad3d10f308443d56e032569599  model-00014-of-00018.safetensors
c7bea4f31fe3b1878ad1f68eb8c017a4016e6202a7aa2008af939c532edfb49d  model-00015-of-00018.safetensors
4f722ee0c31670694e56930e67f8c84845d42ff0084efa838ae598752b3a73aa  model-00016-of-00018.safetensors
3c625a09bda2e8038d29fc9d2a80fd28f429c0bd622738236e80bef14d28c2ed  model-00017-of-00018.safetensors
fe72369eebdd81b26cb9c22ea63aec33f003ddcbb5cf08e4e7165f7cbcc5628b  model-00018-of-00018.safetensors
```
