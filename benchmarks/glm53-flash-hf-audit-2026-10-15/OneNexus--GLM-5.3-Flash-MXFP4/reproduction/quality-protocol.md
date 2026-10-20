# OneNexus versus AMD Quark quality comparison

This records the public checkpoint-quality comparison. Both checkpoints used
the same question
sets, tokenizer, generation settings, and one attempt per question.
Output-cap truncations remain in the denominator; no failed examples were
replaced. No arm reported API errors.

## Evaluator and inputs

- Evaluator: `sgl-project/sgl-eval` at
  `a231b7a439b235090ff7baa30778fa2b514309ae`, Python 3.12.
- Dataset namespace recorded by sgl-eval:
  `645cf567ff08c0ae9cc3fc8e1edbb975b3067816`.
- GPQA source: `Idavidrein/gpqa`, revision
  `633f5ee89ab8ad4522a9f850766b73f62147ffdd`, Diamond subset.
- GPQA source CSV SHA-256:
  `41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305`.

| Input cache file | Questions evaluated | SHA-256 of full cached input |
| --- | ---: | --- |
| `gsm8k/test.jsonl` | 500 | `67e9046d913470477dfb321814d4e3240240a397cfc19c6c610af02dd2cb6456` |
| `mmlu/test.jsonl` | 500 | `a7a2c51253c4941c237f5ce4c662db2d56cf8ce8558c57555dbc026c64e99736` |
| `gpqa/diamond.jsonl` | 198 | `b0f7cc83215b28f9070723824ca25da4f96ec2dee43fdfa30c380b06b4343d19` |

Acquire any required dataset access from its owner. These hashes identify the
inputs; they do not grant redistribution rights. Gated prompts and reference
answers are not included in this card.

## Recorded generation settings

| Setting | GSM8K | MMLU | GPQA Diamond |
| --- | ---: | ---: | ---: |
| Questions | 500 | 500 | 198 |
| Concurrent evaluator workers | 32 | 32 | 32 |
| Output-token limit | 8,192 | 8,192 | 16,384 |
| Temperature / seed | 0 / 0 | 0 / 0 | 0 / 0 |
| `top_p` / `min_p` | 0.95 / 0 | 0.95 / 0 | 0.95 / 0 |
| `repetition_penalty` | 1.0 | 1.0 | 1.0 |
| `reasoning_effort` | max | max | max |
| Recorded chat-template kwargs | null | null | `{"thinking": true}` |

The GPQA `thinking` setting is recorded in evaluator `metrics.gen`; it was
not an extra CLI argument. With temperature zero, `top_p` does not alter
greedy token choice. The evaluator had no system-message override or extra
request body. Preserve the pinned evaluator defaults when reproducing.

The recorded commands used a local-cache override in the evaluator loader.
After obtaining the inputs above, place them under `inputs/` and verify their
hashes. The following commands retain the recorded evaluator arguments while
using portable local paths and a loopback endpoint:

```bash
QUALITY_EVAL_SHIM='from pathlib import Path; import sgl_eval.evals._loader as m; m._CACHE_ROOT = Path("inputs").resolve(); from sgl_eval import cli; raise SystemExit(cli.main(__import__("sys").argv[1:]))'

python3.12 -c "$QUALITY_EVAL_SHIM" run gsm8k \
  --num-examples 500 --num-threads 32 --max-tokens 8192 \
  --temperature 0 --seed 0 --reasoning-effort max --n-repeats 1 \
  --base-url http://127.0.0.1:30042/v1 \
  --model OneNexus/GLM-5.3-Flash-MXFP4 --out-dir results/gsm8k

python3.12 -c "$QUALITY_EVAL_SHIM" run mmlu \
  --num-examples 500 --num-threads 32 --max-tokens 8192 \
  --temperature 0 --seed 0 --reasoning-effort max --n-repeats 1 \
  --base-url http://127.0.0.1:30042/v1 \
  --model OneNexus/GLM-5.3-Flash-MXFP4 --out-dir results/mmlu

python3.12 -c "$QUALITY_EVAL_SHIM" run gpqa \
  --num-examples 198 --num-threads 32 --max-tokens 16384 \
  --temperature 0 --seed 0 --reasoning-effort max --n-repeats 1 \
  --base-url http://127.0.0.1:30042/v1 \
  --model OneNexus/GLM-5.3-Flash-MXFP4 --out-dir results/gpqa
```

Use a fresh result directory for each checkpoint. The checkpoint weights are
published. This bundle records the frozen quality runtime's source-manifest
hash below, but does not include its source snapshot; the commands and base
image alone do not recreate that runtime.

## Recorded server configuration

The checkpoint study uses BF16 KV with CUDA graphs and speculative decoding
off, intentionally separate from the FP8-KV/EAGLE performance configuration.
Both arms mount the same tokenizer; see [its exact hashes](tokenizer-provenance.json).

- Image ID: `sha256:3e11e65a3cc79aeca7be627b08c044400515ca0aa9b2f302954c3902fe35b9cc`.
- Comparison-source manifest SHA-256 recorded by GSM8K/MMLU:
  `bfc0636b92e89c38895cde6a6707365dda76779d8e991dce51b89220fd2923f1`.
- Four gfx950 GPUs; fixed device and CPU placement across checkpoint arms.
- Performance-affecting environment: `SGLANG_USE_AITER=1`,
  `SGLANG_OPT_DEEPGEMM_HC_PRENORM=0`, `SGLANG_ENABLE_WAR_BARRIER=1`,
  `SGLANG_FORCE_COARSE_WAR_BARRIER=1`, `SGLANG_OPT_USE_TOPK_V2=false`,
  `SGLANG_DSA_FUSE_TOPK=0`, `CHUNK_DELTA_ATTN_USE_FLASH_KDA=0`.
- Automatic affinity was disabled (`SGLANG_SET_CPU_AFFINITY=0`,
  `SGLANG_AUTO_NUMA_BIND=0`, `SGLANG_NUMA_BIND_V2=0`).

With each checkpoint mounted at `/model` and the common tokenizer at
`/tokenizer`, the recorded server arguments were:

```bash
python3 -m sglang.launch_server \
  --model-path /model --tokenizer-path /tokenizer \
  --served-model-name OneNexus/GLM-5.3-Flash-MXFP4 \
  --host 127.0.0.1 --port 30042 --tp-size 4 \
  --quantization quark --trust-remote-code --disable-cuda-graph \
  --context-length 65536 --mem-fraction-static 0.80 \
  --max-running-requests 32 --chunked-prefill-size 4096 \
  --max-prefill-tokens 16384 --dsa-prefill-backend tilelang \
  --dsa-decode-backend tilelang --kv-cache-dtype bfloat16 \
  --moe-runner-backend aiter --reasoning-parser glm45 \
  --tool-call-parser glm47 --mm-feature-transport cpu \
  --random-seed 0 --enable-metrics
```

The comparison runtime supports both MXFP4 checkpoints. PR #39273 enables
official FP8 and OneNexus MXFP4 serving and is not a snapshot of that runtime.
Results from a different source revision are new measurements, not exact
replays of this checkpoint study.

## Interpretation

Report all example outcomes and cap hits, and pair results by the same question
ID. Output files may not be byte-identical on a fresh run despite greedy
settings; recorded output hashes identify evidence, not a required output
checksum. Statistical non-significance of the observed paired differences
does not prove broad quality equivalence.
