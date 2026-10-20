# GLM combined-adapter Nsight diagnostic

Completed 16 October 2026, revision `nsysv2`. Both original GLM containers were
restored; their identities/configuration digests, health 200 and absence of
active experiment containers were independently checked at **19:14 UTC**.

The next optimization target is the fused BF16 KDA/head kernels. Their GPU
time is substantial, but includes necessary projection computation; it is
not a measurement of compression overhead. A paired component benchmark
against the same reconstructed BF16 weights should establish the recoverable
part before another whole-model load.

## What ran

One load of the already-qualified combined dispatcher candidate:
`INCModel3/GLM-5.3-Flash-MXFP4-Mixed-CT-AutoRound`, revision
`35d1b0d5cb4a0db075fd2749c897e970dc15bfa5`, TP2 on the two Sparks. All four
storage paths stayed enabled: expert scales, KDA, output head and embeddings.
Expert FP4 codes, KV format, kernel arithmetic and the source weights stayed
unchanged. CUDA graphs remained size `[1]`; B4 executes eagerly. KV stayed
fixed at 4 GiB/rank, KDA fused limit eight, MoE dispatch `dispatcher`.

The detached controller waited for idle GLM and armed an independent watchdog
before pausing it. Nsight Systems 2025.3.2 wrapped both engines, tracing
CUDA/NVTX/OSRT and Python stacks at 500 Hz. vLLM used its `cuda` profiler
backend with `ignore_frontend=true`; PyTorch/Kineto remained disabled.

Two warm-up batches preceded five before-capture trials, one captured trial
and five after-capture trials. Each trial sent the same four 256-token prompts
and generated 64 tokens per request. Report generation finished on both ranks
before after-capture timing. No stock model was loaded or remeasured.

`nsysv1` stopped before loading weights: 108.54 GiB free was below vLLM's
109.46 GiB startup guard. It restored GLM successfully. `nsysv2` reduced the
startup utilization guard from 0.90 to 0.85, preserving the explicit KV budget.
Its checkpoint read took 39.95 seconds. Both snapshots and failure evidence
are retained; the retry did not alter the qualified adapter code.

## Timing and correctness

| Observation | Before capture | Captured | After capture |
|---|---:|---:|---:|
| Trials | 5 | 1 | 5 |
| Median aggregate tokens/s, including prefill/drain | 39.30 | 37.80 | 39.02 |
| Median of trial middle-decode interarrival medians | 84.03 ms | 90.28 ms | 83.21 ms |

Middle decode uses each request's one-token SSE arrivals at positions 8–55,
excluding prefill and the final drain. It is a client latency proxy, not a
CUDA-event measurement. The captured trial has **7.4–8.5% higher middle-decode
latency** and 3.1–3.8% lower end-to-end throughput. Prefill batching varied;
one captured trial cannot precisely isolate overhead. Before/after also retain
Nsight injection and CUDA NVTX annotations, so this is not an uninjected control.

Both-rank loaded-source checks and all three chat checks passed. All **44
timed completions / 2,816 generated IDs** match their corresponding requests
in the first before-capture trial. These compare the same candidate across
capture states, not against stock. Allocation stayed essentially constant at
**168.699 GiB across both nodes**. No new KL/accuracy screen ran; the existing
formal status remains **unqualified because the stock control is unstable**.

## Where the captured time goes

Each rank has 124,961 kernel events. The analysis verifies **61 complete M4
forwards per rank**, each with 84 routed-expert kernels and 91 communication
kernels. Medians below use those M4 forwards; prefill and tail shapes are excluded.

| Captured M4 observation | Rank 0 | Rank 1 |
|---|---:|---:|
| GPU model-forward span | 87.36 ms | 87.31 ms |
| Routed-expert kernel sum, 84 calls | 34.52 ms | 34.15 ms |
| Fused compressed KDA kernel sum, 68 calls | 21.36 ms | 21.61 ms |
| Compressed output-head kernel, outside forward | 2.389 ms | 2.394 ms |
| Embedding lookup | 8.9 µs | 9.5 µs |
| Communication kernel sum, including peer waiting | 2.00 ms | 4.19 ms |
| Main-stream gaps, including copy activities in coverage | 7.76 ms | 5.56 ms |
| Gap time before the next launch API enters | 2.72 ms | 0.50 ms |

Kernel sums may overlap across streams. The head is separately associated
through its launch thread and preceding M4 annotation: 61 head kernels/rank.
Do not add independent medians or both ranks as a causal latency breakdown.
No bandwidth, occupancy or roofline counters were collected.

**A specific shared-runtime synchronization is identified.** The existing
`FlashInferMLASparseSM90MetadataBuilder._kv_lens_host`, line 348, performs
`positions[:num_rows].cpu()` when it cannot use exact host metadata. Each M4
has one 32-byte D2H copy. Its GPU activity is only **1.44 / 1.54 µs**, while
the host API median is **0.483 / 10.250 ms**. Python samples inside those
API spans identify that exact source line: 213 / 433 samples.

The first M4 host copy waits roughly 289 ms behind the preceding prefill;
this is not a 289 ms physical transfer. Host wait includes queued GPU work
and overlaps the pipeline. Neither that outlier nor the 10 ms rank-1 median
is a recoverable-speedup estimate. This path plans sparse attention using
exact lengths; it does not compress KV or move compressed weights.

**GC and main-thread descheduling do not explain this capture's gaps.** Both
workers record zero GC collections/events during capture. Main-thread
runqueue wait is at most **1.44 / 1.05 ms for the entire capture bracket**,
including start/stop, versus many M4 steps. Native GPU waits and busy polling
still occur. Proxy-thread runqueue wait is higher during capture, so this
does not exonerate every thread or explain earlier runs.

The old Torch trace's 10 ms rank-0 late-launch median is not reproduced:
this capture has about 2.7 ms. Profiler type, sampled contexts and proxy
inheritance differ. This cannot be credited to a compression optimization.
Late Python samples span adapter launches and existing Torch/attention code;
they do not identify one dominant Python function as the cause.

## Next steps

**Step 1 is complete:** see the
[BF16 component comparison and Nsight Compute report](GLM53_BF16_KERNEL_DIAGNOSTIC.md).
Counter access works in an isolated container. The current head is faster,
KDA output has insufficient blocks, and input is limited by shared memory.
The unprofiled output cache-pressure follow-up changes 12.5% slower warm
replay into 4.5% faster evicted replay, so low occupancy is a tuning lead
rather than a demonstrated full-model compression penalty.
Higher split-K improves some timings but fails the unchanged numerical gates.
Prioritize a correctness-preserving KDA kernel lead before a model reload.

1. Profile representative KDA input/output and vocabulary-head kernels at
   M=1,4,8 in a bounded component fixture. Nsight Compute is installed
   at `/opt/nvidia/nsight-compute/2025.3.1/ncu` on both hosts, outside PATH.
   `RmProfilingAdminOnly=1`, so qualify isolated counter access before any
   outage. Paired CUDA-event timings can establish a lead while counter
   collection is prepared. Compare current
   fused execution with the same reconstructed BF16 weights and stock GEMM.
   Measure bandwidth, occupancy, register/shared-memory pressure and launch
   cost. Preserve the established byte, numerical, stream, graph and scratch
   gates. Establish a repeatable component lead before a full-model rerun.
2. Investigate exact host metadata reuse or a planning path that avoids the
   D2H round trip. The source deliberately avoids optimistic host lengths
   with async scheduling/speculation. Do not simply replace the GPU-derived
   positions: qualify mixed prefill/decode, EOS, preemption and speculation.
   Disabling async scheduling is a separate controlled configuration test
   that may lose useful overlap.
3. Keep embedding and helper-copy tuning behind these targets: embedding is
   about 9 µs, and previously measured KDA helpers total about 0.2 ms/step.
   Keep the bounded prefill fallback until an alternative measures better.
   Resolve the existing stock-repeat quality instability before a release.

## Evidence and recovery

[Compact measured review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/nsys-model-v2-review.json)
contains hashes, counters, timing trials and limits. Nsight warns some
CUDA/NVTX/OSRT events may be missing; the complete expert counts above were
independently checked. Exact OS scheduler tracing is unavailable. Python
samples were decoded from extended payloads, not inferred from marker counts.

On Spark1, `ROOT` is
`/home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/full-model-v1`.
State is `ROOT/nsys-controller-nsysv2.json`; restoration is
`ROOT/restoration-nsysv2.json`; independent check is
`ROOT/nsysv2-final-service-check.json`. Frozen source is
`ROOT/full-nsysv2-src/` on both nodes. Raw files are on head under
`ROOT/run-nsysv2/kda/{traces,traces-worker}/`, and timing is
`ROOT/run-nsysv2/results/nsys-measurement.json`.

Private archive `ROOT/nsysv2-evidence-v1.tar.gz`: 79,463,998 bytes, 123
locally hash-verified payload files, SHA-256
`8b608cc06798c28d21fdd625f989d7e66c9a16cd469b10815137415f8f4b4239`.
Use the `*-payload.sqlite` files for Python frames: they were exported with
`nsys export --include-json --include-blobs`. Keep raw traces private because
Nsight records process/environment metadata.

Reproduce the CPU-only analysis on Spark1 without loading a model:

```bash
MXWAVE_RUN_ROOT=/home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/full-model-v1
python3 -S "$MXWAVE_RUN_ROOT/agent-handoff-2026-10-16/analysis/nsysv2/analyze_nsight.py" \
  --inputs "$MXWAVE_RUN_ROOT/run-nsysv2/kda/traces/nsys-r0-payload.sqlite" \
           "$MXWAVE_RUN_ROOT/run-nsysv2/kda/traces-worker/nsys-r1-payload.sqlite" \
  --output /tmp/glm53-nsysv2-offline-review.json
```

The backend source SHA-256 is
`4449ea25921dcae1ec7988581ff26a6ff5a9e5d6d4a0ac4a68a0017ea108f136`.
Its host copy rule is existing runtime behavior; no upstream-vLLM equivalence
or production NVFP4/DFlash qualification is asserted by this MXFP4 diagnostic.
