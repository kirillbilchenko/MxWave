# NVFP4 lossless scale component

This directory contains the component experiment and an opt-in ModelOpt serving
test adapter. The serving comparison is still experimental.
It retains the source NVFP4 codes, post-Marlin S0E5M3 scale bytes, compensating
global scales and BF16/FP32 grouped GEMM arithmetic. It does not convert the
production checkpoint to MXFP4 or replace the existing MXFP4 integration.

`build.py` stages the hash-pinned Apache-2.0 sources listed in
`../glm53_mxfp4/moe_sources.json`; the upstream license/notices remain in the
external build tree. Pass `--source` and `--output`. Compile `staged/host.cu`
for SM121 with the staged/Marlin/Torch include directories, ordinary `-O3`
CUDA math settings, and these definitions:

| Library | `NV_COMPRESSED` | `MARLIN_NAMESPACE_NAME` | Encoding |
|---|---:|---|---|
| `libnv_raw.so` | 0 | `nv_raw` | Unchanged scale loads |
| `libnv_direct5.so` | 1 | `nv_direct5` | Five-bit palette, direct expansion |
| `libnv_async5.so` | 2 | `nv_async5` | Five-bit palette, asynchronous bit-plane loads |
| `libnv_delta5.so` | 3 | `nv_delta5` | Five-bit exact byte offsets, asynchronous loads |

Do not enable `--use_fast_math`: that build failed the bitwise installed-stock
control on tiny intermediate values. Standard math restored the covered parity.
Final build commands and hashes are retained with the experiment evidence.

Run `benchmark.py --model <lossless8> --native <libraries> --output <new-directory>`.
Use `--candidate-library libnv_delta5.so --affine` together for the offset
consumer; generic palette indices cannot be decoded as offsets. The harness
defaults to 16 real experts, both TP2 partitions and rows 1/8/16/64/256.
`--experts 288` uses a complete layer. `--ranks`, `--rows` and `--trials` permit
explicit, separately retained repeat measurements. Each partition runs
sequentially on one GPU; this is not an end-to-end TP2 serving benchmark.
`--include-installed` times the installed operator alongside the rebuilt control;
`--shared-buffers` matches output/native scratch placement across the native arms
while preserving independent output snapshots for correctness checking.

Every expert/projection owns its low/high bit planes or exact raw fallback.
The offset codec requires a byte span of at most 32; other parts stay raw.
High planes have complete four-byte words for asynchronous reads. Metadata,
input/output, locks and scratch must remain alive through CUDA graph replay.
No complete dense scale reconstruction is launched inside the measured layer.

The harness bounds CUDA allocator usage, checks GPU byte reconstruction,
compares intermediate and final outputs against installed stock Marlin, and
retains individual timing trials, control repeats, clocks/thermals and traces.
Production preparation preserves scale arithmetic while avoiding the installed
scale-factor helper's large boolean-index temporary: positive finite E4M3 bytes
are monotonic, so it invokes the same helper on their exact maximum, then stages
one expert at a time. Live-service idle guards and terminal checks are retained
separately in the experiment controller evidence.

## Original-checkpoint serving test

`nv_adapter.py` hooks only the pinned ModelOpt NVFP4 expert method after stock
repacking. It verifies complete scale-byte recovery, replaces every known quant
scale owner and uses the existing production MoE tuning wrapper. `nv_worker.py`
records actual allocation, released raw owners and source coverage; `nv_plugin.py`
retains startup evidence. `nv_boot.py` registers this plugin only inside disposable
test containers. `MXWAVE_NVFP4_MODE=stock|packed` chooses the arm, and
`MXWAVE_NVFP4_LIBRARY` must identify the exact pinned library.

The adapter currently requires the tested GLM TP2 shapes and one visible SM121
GPU per worker host. It preserves original production routing, global scales,
activation/reduction policy, MXFP8 KDA and head/draft behavior. Its comparison
uses a restoration controller; it is not installed in the current live service.
The completed full-model test confirms 3.16 GiB/GPU of actual allocation saving
and graph replay without scale rebuilds. All timing controls and full-model
self-repeat KL gates fail, so speed and zero model loss remain unqualified.
See the [full-model result](../../docs/GLM53_NVFP4_MODEL_TEST.md).

The expanded large-row check found that a one-block-per-SM launch must reserve
space for the decoder's static shared palette before opting in dynamic shared
memory. The current `build.py` accounts for that space and validates the remaining
cache budget while preserving the original tile. Earlier component timings refer
to the retained earlier binary; the new full-model test measures the fixed build.

`smoke_adapter.py` covers real layer weights at production row counts with one
installed routing result shared between consumers. That isolation is explicit:
stock graph execution itself showed a repeat difference at 256 rows. Whole-model
serving retains stock routing and requires separate repeat/KL qualification.


## Original-production numerical capture

`repeat_worker.RepeatWorker` extends the diagnostic worker with `nv_capture`.
`repeat_capture.PrefillCapture` accepts the installed text path's precomputed
embeddings as well as token IDs. It clones the selected 256-token eager prefill
boundaries on the producer stream, capped at 768 MiB/rank, and writes exact
bytes only after the request completes. Optional layer focus records attention,
mHC, gate and shared-expert boundaries; raw target logits are captured separately.
It changes CPU/GPU scheduling and is not a performance instrument. Disabled
capture leaves forwarding wrappers installed.

The original-production repeat diagnostic first differs at the routed-expert
output with exact preceding captured values. Routing and arithmetic internals
still need separation; this is no deterministic fix or qualification pass.
See the [verified full result](../../docs/GLM53_NVFP4_REPEAT.md). Four CPU tests
cover bounds, immutable snapshots, ordinary IDs and precomputed embeddings.

## Internal MoE diagnostic

`moe_capture.py` / `moe_worker.py` extend the original-checkpoint observer with
bounded layers 3–5 internal snapshots and fixed-operand replay. They are opt-in
RPC tools for disposable containers, without changing expert arithmetic. See
[the active protocol](../../docs/GLM53_NVFP4_MOE.md); these intrusive runs do not
qualify performance or full-model compression quality.
