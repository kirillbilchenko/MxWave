# Native vLLM DFlash2 loading diagnosis

Recorded **21 October 2026, 19:56 UTC**. The zero-acceptance failure is causally
resolved in a bounded GPU diagnostic. Exact original 500k production was restored
at **19:55:15.227504 UTC**; no experiment is running or queued. This is a loading
compatibility result, not a production performance or general quality qualification.

## Finding and intervention

The pinned Modal FP8 DFlash2 checkpoint contains twelve **BF16** convolution
projection matrices, `layers.{0..5}.{attention_conv,mlp_conv}.kernel_projection.weight`,
each shaped `[1024,4096]`, with **no corresponding scale tensors**. Its quantization
config excludes only `model.fc` and `fc`. The older runtime explicitly constructed
these convolution linears with `quant_config=None`. Native vLLM 0.32 instead passes
the checkpoint's FP8 quantization config, and startup warns about loading BF16
tensors into FP8 parameters.

Native FP8 scale creation initializes missing scales to `torch.finfo(float32).min`.
Marlin preparation casts scales to the original BF16 dtype: this sentinel becomes
negative infinity. A CPU cast reproduces the overflow; zero times that scale is
NaN. The V1 GPU observer finds finite target auxiliary features and finite combined
context, followed by wholly NaN drafter hidden states/logits and candidate scores.
Both native graph replay and draft-only eager execution fail, ruling out graph
replay alone as the cause.

V2 changes **only the separate drafter config's `ignored_layers` metadata**. It adds
the twelve local paths and twelve native global aliases (`model.layers.45` through
`model.layers.50`). Explicit global aliases matter because native loading uses
global layer prefixes. The original directory remains untouched; the new directory
hard-links its weight file and verifies SHA256. Main attention projections still
use FP8. There is no training, weight conversion, target precision change or
replacement of installed vLLM source.

Both official-image CPU preflights instantiate the actual native convolution class
and verify all twelve projections are BF16 with `UnquantizedLinearMethod`.
The initial CPU fixture lacked TP rank/world-size metadata; its failure record is
retained. Both GPU ranks independently verify the same dtype/method pairs before
any diagnostic request. The CPU fixture's TP metadata mock is not a serving patch.

## Before and after

Two existing validation prompts are reused: `opencoder-evol-92378` (tagged code,
actually a Newton-method explanation) and `dolly-7475` (chat/travel question).
Each generates 32 tokens at C1, temperature zero, seed 42, five draft tokens.
The table applies separately to **both native replay and draft-eager** modes.

| Prompt | V1 accepted / proposed | V1 rounds | V2 accepted / proposed | V2 rounds |
| --- | ---: | ---: | ---: | ---: |
| `opencoder-evol-92378` | 0 / 155 (0%) | 31 | 27 / 40 (67.5%) | 8 |
| `dolly-7475` | 0 / 155 (0%) | 31 | 25 / 50 (50%) | 10 |

Within V2, native/eager counters and all 32 output token IDs agree for each prompt.
All eight trace files retain eight proposal frames each: **64/64 candidate-score
and combined-context frames are finite on both ranks**, as are all observed target
auxiliary tensors. All **32/32 eager final-hidden/logit frames** are finite.
V1 retained 64/64 wholly NaN score frames and 32/32 wholly NaN eager hidden/logit
frames, while its target/combined context remained finite.

Acceptance is `accepted proposals / proposed tokens`, not acceptance conditional
on reaching a position. Counters can include the terminal speculative block beyond
the 32-token output truncation; accepted tokens plus verification rounds need not
equal emitted tokens. CPU copies/synchronizations distort latency, so these runs
must **not** supply tok/s, TTFT, speedup or per-round performance claims.

## Configuration and scope

- Target: selected CT MXFP4 experts + MXFP8 KDA + lossless expert-scale, embedding
  and output-head packing. MXFP8 KDA remains a separate lossy change.
- Runtime: official vLLM 0.32.0 with the existing compression plugin; two DGX
  Sparks, TP2. Native Triton draft attention and **BF16 draft KV** (`auto`).
- Capacity: **8,192 context, 16 slots, 6 GiB KV per GPU, 4,096 prefill budget,
  prefix caching off**. This is the frozen V3 isolation profile, not production's
  500k/prefix-on profile. The policy remains C1:5/C2:5/C3–C16:3; only C1 is tested.
- Observer: first eight actual proposal boundaries per request, both ranks.
  The eager control uses native `is_profile=True` for draft execution throughout
  the request, leaving target execution unchanged.
- V1: source `ea930d3`, launch 19:38:16 UTC, exact restoration 19:45:42 UTC.
- V2: source `14c5c81`, launch 19:47:56 UTC, exact restoration 19:55:15 UTC;
  controller `3054345`, watchdog `3054999`, 25-minute independent restoration lease.
  `prepared.json` and `launch-rank{0,1}.json` define the authoritative launch.
  `profile-plan.json` is the inherited V3 prompt/source plan; its old scope text
  does not describe a new V2 KV change.

V2 plan SHA256: `a16f58dd37f0152e3b5e69aad228333d16b784d226c0e666253832f8cd940119`.
Its controller ends `complete_restored`, with both original container IDs, image,
Config/HostConfig/Mounts hashes and authenticated 500,000-token model endpoint
verified. No promotion or change to production configuration occurred.

## Evidence and preservation

Remote parent on both Sparks:
`/home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/full-model-v1/glm53-package-v1`.
Mac copies and independent reviews: `out/glm53-package-v1/`.

| Archive | Payloads | SHA256 |
| --- | ---: | --- |
| `deployment-compare-v1-stopped-evidence.tar.gz` | 96 | `dc1c0009f0665728df77d391204d29a9c1f9c6303ae6bc7e866dcdcb30c5717f` |
| `stock-draft-trace-v1-evidence.tar.gz` | 45 | `e9033efdc8d41b67089b253352e78c2173a8b08629c78aea45cdd044ce8c9d01` |
| `stock-draft-trace-v2-evidence.tar.gz` | 54 | `8e1cf2d6c872431e1bbab9aba958923937cc94241f3b149681cae4cc03f247b4` |

V2's complete payload manifest verifies independently on **Mac and Spark2**.
The Mac also reruns archived `review_draft_trace_v2.py` against all eight raw trace
files and reproduces `trace-review.json` exactly. Its receipt is
`stock-draft-trace-v2-independent-review.json`. The archive includes raw requests
and responses, per-position counters, both-rank traces, startup logs, CPU checks,
original/corrected drafter configs, header audit, launch inputs, controller and
watchdog records, and restoration receipts. Generated compile caches and Python
bytecode are excluded explicitly; source history is backed up separately.

Pinned original drafter config:
`c54c63c667baca52604519c5ac0d0d053210e3a3231dfb626d4758b1b6f8655e`.
Corrected config:
`28df83230b06fb5460eed2105357e81f09560124b0b4d65366d7afcd70dfa089`.
Unchanged `model.safetensors`:
`45f34ef126aa258d8e16bd605a3a05c36eb3fef3ed0f57aef20191db49f1ef46`.
The source bundle `glm53-stock-draft-trace-v2.bundle`, backed up on Spark2, hashes
to `be288fdca8c3b7f7ccfff515c074516bc01edc4b63332ae29b46d6403ffd930a`.
The immutable preparation receipt's `gpu_qualified: false` reflects its pre-GPU
creation time; the later trace review establishes only the diagnostic result above.

## What remains

1. Integrate the config correction into the packaged CLI with pinned validation.
   The current CLI still expects the original config; the isolated experiment
   used `prepare_drafter_compat.py` and a separately frozen native launch. Do not
   advertise the existing default command as corrected yet.
2. Recheck corrected serving under the intended deployment settings, including
   500k capacity, prefix caching and the selected draft KV precision. This test
   did not requalify those settings or measure a corrected-profile memory delta.
3. Address or explicitly retain the separate native target-repeat KL and prefix/
   output-repeat failures. Fixing the drafter loader does not clear those gates.
4. Freeze a new untraced production-versus-CT comparison and complete the requested
   speculation-off full 1,319-question GSM8K evaluation on both targets. The old
   interrupted comparison remains stopped, with all failures/outliers retained.

No new throughput, general quality, long-context correctness or memory-saving
claim is made from this screen. No additional drafter training is justified by
this particular failure: the same pretrained weight bytes now work.
