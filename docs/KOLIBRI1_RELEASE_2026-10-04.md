# Kolibri-1 softer RMS MXFP4 release overview

The selected checkpoint combines calibrated MXFP4 routed experts with the
unchanged official FP8 backbone. Its 32 safetensors files total **40.45 GiB**
(43,433,901,512 bytes), 44.9% smaller than official FP8. The standalone checkpoint
loads with stock vLLM and needs no private trial worker or patch files.

**Experimental:** all likelihood bounds and six behavior checks passed, but mean
forward KL **0.055140** exceeds the unchanged **0.030** target.

## Final measurements

| Measurement | Official FP8 | Softer RMS standalone |
|---|---:|---:|
| Prompt PPL | 24.201922 | 24.571060 (+1.53%) |
| Mean forward KL from official FP8 | reference | 0.055140 |
| Behavior diagnostics | 6/6 | 6/6 |
| Output tok/s, c1 | 43.61 | 49.62 |
| Aggregate output tok/s, c4 | 113.18 | 136.37 |
| Weight files | 73.43 GiB | 40.45 GiB |

Quality uses 48 frozen passages and 26,488 scored tokens: WikiText-2 validation,
GermanQuAD test, and HumanEval prompts with canonical solutions. Code PPL is
likelihood, not pass@1. KL uses one full-vocabulary distribution per passage.
Behavior checks cover English/German arithmetic, retrieval, JSON, a tool call,
and retrieval near 8k tokens. This screen does not establish broad task accuracy.

Throughput is the median of three runs with 512 input / 128 forced output tokens,
BF16 KV, no prefix cache, and decode graph captures `[1,2,4]`, on NVIDIA GB10 /
DGX Spark (SM121). The candidate run overlapped a CPU/network weight upload.

## Sources and recipe

- Expert source: `Aleph-Alpha/Kolibri-1-BF16@7a8f290e7858825c3cf5e4c447ba68345de9f1d3`.
- Backbone/reference: `Aleph-Alpha/Kolibri-1@e52eb4627d11516b0c01de49210ab5a4e4061444`.
- All 50 × 384 × 3 routed expert matrices use MXFP4 E2M1, E8M0 scales, block size 32.
- MSE search: depth 4, percentile 99.5, round-to-nearest-even `mxfp4-rne-v2`.
- Routed RMS: 64 training windows, 32,768 WikiText-2/GermanQuAD/MBPP tokens,
  excluding the held-out evaluation passages.
- `alpha = 0.25 * n / (n + 128)`; the squared-error weight is
  `(1 - alpha) + alpha * clip(RMS² / mean(RMS²), 0.25, 4)`.
- 53,646 observed projections; 3,954 unobserved projections retain unweighted
  payloads. 53,591 projections changed, affecting 0.3033% of expert blocks. No rotation.

The adapter validates geometry, expert coverage, packed shapes, and every weighted
module before emitting its compact mixed-format config. Official FP8 backbone
payloads remain byte-identical; dynamic FP8 backbone activations are inherited.

## Serving

Tested with vLLM 0.29.0 and `aleph-alpha-inference==1.0.0`. The plugin supplies
Kolibri's routing, sandwich norms, and mixed attention and declares
`vllm>=0.29.0,<0.30.0`. vLLM 0.30 was not qualified.

```bash
python -m pip install 'vllm==0.29.0' 'aleph-alpha-inference==1.0.0'
VLLM_USE_DEEP_GEMM=0 VLLM_USE_DEEP_GEMM_E8M0=0 \
vllm serve kirillbilchenko/Kolibri-1-MXFP4-MxWave \
  --load-format safetensors --dtype bfloat16 --moe-backend marlin \
  --max-model-len 9216 --max-num-batched-tokens 1024 --max-num-seqs 4 \
  --kv-cache-dtype bfloat16 --kv-cache-memory-bytes 2147483648 \
  --gpu-memory-utilization 0.85 --no-enable-prefix-caching \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4]}' \
  --enable-auto-tool-choice --tool-call-parser kolibri1 --reasoning-parser kolibri1
```

Startup must select Marlin experts and automatic block-FP8 dense kernels. Runtime
memory also includes KV/workspaces. Full advertised context and reasoning quality
were not validated by this screen.

## Reproduction and evidence

The final path is expert conversion (`--policy kolibri1-routed-experts`), official
FP8 backbone assembly, routed calibration, softened retuning, exact export,
stock-loader qualification, and verified publication. Its scripts and focused
tests remain in the PR. Detailed records and rejected-trial code are preserved
in a separate full experiment archive outside the code-review diff.

Export took 223.841 seconds and verified all **116,303 payloads** against the
compact trial. Manifest SHA256:
`4d7b37059099d1306c30e1d20723d339e7b3121525ec541bfcd3e6688cb05a11`.
Resume rejects incompatible inputs and changed completed shards.

The [model repository](https://huggingface.co/kirillbilchenko/Kolibri-1-MXFP4-MxWave)
will include the model card, raw distributions, frozen protocol, payload/shard
hashes, calibration, dataset licenses, and exact licensed release source.

The compact trial measured PPL 24.482698 and KL 0.053467. Exported payloads match
it exactly, but the separately initialized stock run returned different
likelihoods; the cause is not established. The card uses standalone results.
Soft-RMS weight SQNR was not separately measured.

## GGUF follow-up

Native MXFP4 GGUF could preserve calibrated expert weights, at an estimated
42 GiB after converting the FP8 backbone. It needs a Kolibri exporter and
patched llama.cpp validation. Stock llama.cpp/Ollama compatibility is not claimed
for the current vLLM checkpoint.
