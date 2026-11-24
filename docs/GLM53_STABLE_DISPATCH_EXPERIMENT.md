# GLM stable-dispatch experiment

**Complete, 17 October 2026: 9.37 GiB / 5.33% less live CUDA allocation,
11.4–11.9% higher single-request throughput, and identical held-out loss.**
Graph/eager repeatability and candidate numerical gates now pass unchanged.
Strict overall qualification still fails: two reference passes are lost to
output formatting, while two math passes are gained. Both arms score 30/34.
The original failed qualitybestv2 result and all numerical floors are preserved.

A later [five-round task diagnostic](GLM53_PRACTICAL_REPEAT_DIAGNOSTIC.md)
finds variability in both arms at the previously lost format cases. It retains
all 340 responses and preserves this run's strict failure. The separate
[single-dispatch KDA component](GLM53_KDA_DIRECT_DISPATCH.md) is not enabled in
these model results; its first performance confirmation failed.

The detached stablequalityv1 batch started at 10:19:11 UTC. Both original GLM
containers are restored with their exact identities/configuration digests,
health 200 and no remaining experiment/timing clients, independently verified
at 10:55:39 UTC. All 959 final archive payloads and the compact analysis inputs
were verified. No production defaults or published models changed.

## Whole-model results

Both arms use the same pinned imported mixed-MXFP4 checkpoint, TP2 image,
BF16 activations, 4 GiB FP8 KV per rank and canonical dispatch patch below.
The reference retains stock uncompressed runtime scales/KDA/head/embeddings;
it is not an untouched upstream vLLM launch. The candidate uses qualified v9
U8 shared-stage palette scales plus the existing BF16/vocabulary adapters.
FP4 weight codes and KV representation remain unchanged.

| Measurement | Reference | Candidate |
|---|---:|---:|
| Allocated CUDA memory, both ranks | 175.8658 GiB | 166.4987 GiB |
| Held-out PPL, first / repeat | 4.30501528 / 4.30501528 | 4.30501528 / 4.30501528 |
| Graph self-repeat KL | 7.376e-6 | 8.893e-6 |
| Eager self-repeat KL | 6.238e-6 | 8.828e-6 |
| Strict practical cases | 30/34 | 30/34 |
| GSM8K sample, strict answer extraction | 21/24 | 22/24 |
| Functional code tests / bare-code format | 2/2 / 2/2 | 2/2 / 1/2 |
| Tool calls with activated parser | 2/2 | 2/2 |

Memory saving is 10,057,913,856 bytes, **9.3672 GiB / 5.3263%** across TP2,
about 4.68 GiB per Spark. Each PPL run scores 32,768 target tokens in 32 frozen
test windows. All 128 retained float32 window arrays across the four runs are
byte-identical, not merely equal averages. This teacher-forced prefill result
does not establish identical autoregressive logits: decode distributions still
vary, including reference repeats. No original BF16 checkpoint comparison or
source-quantization SQNR is claimed.

The saving is mostly in the scales of already quantized experts, not only in
BF16 leftovers. Audited runtime component payloads across TP2 are:

| Component | Raw | Retained | Gross saving |
|---|---:|---:|---:|
| U8 expert scales | 8.859 GiB | 2.221 GiB | 6.639 GiB |
| Selected BF16 KDA weights | 8.649 GiB | 6.144 GiB | 2.506 GiB |
| BF16 head and embeddings | 2.363 GiB | 1.666 GiB | 0.697 GiB |

These are component payload totals, not three independently measured model
deltas. Added KDA/scale pools total 0.455 GiB across the two ranks, and other
allocation differences account for the remainder between gross payload saving
and the measured 9.367 GiB net CUDA reduction. U8 scales are compressed
directly; there is no prerequisite global BF16 scale expansion.

Candidate/reference graph decode KL is 7.222e-6 versus reference-repeat
7.376e-6; mean decode NLL difference is -7.173e-6. Eager KL is 6.043e-6
versus reference-repeat 6.238e-6, with NLL difference +4.499e-6. Conditioning
matches. Both dispatch modes pass the original numerical gates; all self-repeat
KL values stay below 1e-5 and absolute NLL deltas below 1e-4. The held-out
candidate limit remains 1e-5 and measured NLL delta is zero. No floor changed.

The two lost strict passes are slugify (functional code passes, but Markdown
fences violate the required bare-code response) and gsm8k_0978 (correct
arithmetic/answer, but the required final-answer marker is placed after it).
Of the two gained math passes, one repairs the answer-marker format and one
repairs a wrong numerical answer. These observations contextualize the result;
they do not replace the frozen grades or turn qualification into a pass.
The earlier qualitybestv2 reference scored 22/24 math, rather than this run's
21/24; retain both, with their different runtime controls and original failures.

### Unprofiled speed and latency

These are median aggregate API throughput over five retained trials, 64 generated
tokens, using the existing two warmup groups. Quality is scored before timing.
Both model loads are reused. Only batch one has CUDA graphs; concurrency four
is eager and is not an optimized production configuration.

| Prompt / concurrency | Reference tok/s | Candidate tok/s | Change | TTFT, reference → candidate |
|---|---:|---:|---:|---:|
| 256 / 1 | 14.99 | 16.77 | +11.87% | 372.8 → 399.4 ms (+7.14%) |
| 2048 / 1 | 12.50 | 13.93 | +11.45% | 1172.6 → 1199.3 ms (+2.27%) |
| 256 / 4 | 38.77 | 39.02 | +0.63% | 1065.9 → 1131.7 ms (+6.17%) |

The concurrent result is parity, not evidence of a statistically established
speedup. Five-trial ranges overlap: reference 30.35–38.92, candidate
30.75–39.59 tok/s. Each arm has one slow trial with a roughly 2.13–2.15 s
within-request token-arrival gap. No sample was removed. The retained JIT
warnings occur during warmup, not within those measured slow trials; the
available logs do not identify their cause. Single-request ranges do not
overlap in these five trials, but this is still one sequential paired batch.

## What the new traces show

Eight full-model Torch traces cover both ranks, both arms, and separate B1 and
concurrency-four captures. Profiling occurs after unprofiled measurements.
Complete forwards require all 84 routed expert GEMMs on the annotated stream;
the new analyzer unions all contained device streams so nested annotations
and concurrent Marlin linears are not miscounted as missing GPU work.

* The clearest introduced prefill cost is **68 BF16 KDA reconstruction kernels
  per rank**, 37.23 ms on rank 0 / 35.98 ms on rank 1 for the same 258-token
  prefill. Candidate forward ranges are 406.37 / 405.91 ms, versus reference
  360.72 / 359.83 ms. Above eight tokens the adapter decodes each selected
  matrix into bounded scratch and calls stock linear; these additional reads
  and writes explain a concrete source of prefill overhead. The profiled
  difference is not a promised removable application TTFT delta.
* Stable ordering costs only 42 kernels / 0.166–0.181 ms in that prefill.
  Single-token decode skips them. This is not the dominant latency target.
* B1 decode forward medians are 57.71 / 57.69 ms reference and
  56.54 / 56.47 ms candidate across three captured steps. Candidate KDA fused
  kernels account for 20.71–24.81 ms per step, alongside remaining BF16 GEMMs
  and expert work. Head processing sits outside these model-forward ranges;
  model-only ranges cannot be substituted for request TPS.
* In eager M4 decode, the reference medians are 90.62 / 90.59 ms and the
  candidate 97.10 / 97.11 ms under profiling. Candidate rank 0 has about
  21.36 ms without a kernel on any contained stream, while rank 1 has about
  16.54 ms of communication kernels. Host dispatch/scheduling is a lead;
  this does not prove a removable 21 ms adapter penalty. Unprofiled throughput
  is near parity and mixed-prefill scheduling differs between captures.
* Candidate M4 KDA CPU scopes span about 9.1–9.4 ms for 68 calls, with
  3.7–3.9 ms residual after traced children. Expert submission spans about
  1.54–1.55 ms for 84 calls. Nested ZipServ scopes overlap KDA scopes and must
  not be added. M4 copies/fills attributed to KDA total about 0.20–0.21 ms GPU
  time, including about 2.23 MB of copies; these are not the large lost time.
* Long CPU cudaMemcpyAsync spans include waiting for queued model work.
  They are not evidence that an equal duration of CPU computation can be
  deleted. Torch timings alone do not separate Python, driver backpressure,
  device stalls and profiler overhead.

The next performance target is the above-eight-token BF16 fallback, followed
by validating batch-four graph execution. Repeating the rejected nibble-stage,
register-decoder or compact-N8 sweeps without a new mechanism is unjustified.
No fresh full-model baseline is needed to analyze these saved traces.

## Next steps and acceptance

1. Preserve this strict failure and the older failures. Storage exactness,
   numerical parity and instruction-format reliability are separate results.
   A semantic secondary review may explain a format failure; it cannot silently
   override the frozen no-reference-pass-lost requirement.
2. Before another expensive quality batch, freeze a prospective protocol with
   separate semantic/format scores, working tool flags, a larger held-out/task
   sample and at least five repeat windows/runs where estimating noise. Keep
   the original KL/NLL floors and preserve failures. A statistical alternative
   contract must be explicitly documented as new; do not relabel this run.
   Do not keep rerunning the known tasks until one batch happens to pass.
3. Use bounded real KDA components first to evaluate tiled reconstruction
   fused with BF16 prefill GEMM, or a new overlap mechanism. Cover both selected
   projection shapes at M32/128/258/1025/2048, exact decoded weights, declared
   arithmetic tolerances, streams, allocation bounds and both execution orders.
   Existing prefetch/compact-N8 failures remain controls, not fresh leads.
4. For a concurrency claim, compare graphs at batches 1 and 4 in both arms,
   with representative mixed-prefill warmups. That configuration changes the
   reference, so a new matched control would be required; do not compare a
   graphed candidate against the saved eager reference as compression gain.
   Capture Nsight over an actual slow trial before attributing the 2 s gaps.
5. If bit-identical runtime arithmetic is required, isolate installed-stock
   versus qualified-native MoE and BF16 KDA/head kernels on identical captured
   inputs. The qualified v9 library is based on a different pinned kernel
   source revision from the installed image; current-codec parity is not proof
   of installed-stock parity. The present decode result stays numerical parity,
   not end-to-end bitwise equivalence.
6. Package validated adapters as an opt-in vLLM plugin under a dedicated
   MxWave integration folder, with source/native/layout/TP/architecture gates,
   bounded scratch and explicit fallback. It currently needs custom native
   libraries and a pinned engine integration. It is not stock vLLM/Ollama
   drop-in support. Before a llama.cpp port, audit GGUF's actual block layout,
   scale palettes, retained BF16 tensors and kernel access. This experiment
   already compresses U8 scales directly; it does not depend on global BF16
   scale inflation. Byte-sized GGUF scales can therefore still have palette
   headroom, but the measured whole-model saving cannot be transferred without
   that audit. Ollama also needs its runner to contain the corresponding kernels.

Free weight memory provides KV/concurrency headroom but does not automatically
increase the fixed KV budget in this experiment. KV compression remains a
separate untested extension. The BF16 source is not a cheap two-Spark baseline;
this comparison measures runtime compression on an already quantized checkpoint.

Compact independently checked results:

* [Quality, performance and source verification](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/stable-dispatch-full-model-v1-review.json)
* [Both-rank profile analysis](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/stable-dispatch-full-model-v1-profile-review.json)

Final private archives on each node contain all raw responses/arrays, frozen
source, logs, native qualification and traces. Rank 0: 181,914,553 bytes / 829
payloads, SHA256 bc31a0873700218e72b986cde6df1ae4e7612a9b7743f1f5206473e63d8b3461.
Rank 1: 14,586,610 bytes / 130 payloads, SHA256
0bd9efb86edbe906c15d4c8563306f7d751da27b703da5133acf9b7ae3f447a6.

## Mechanism established by a controlled test

The reference diagnostic found varying token permutations with identical
logical routes. A separate small Spark2 component reconstructed all 288 real
layer-4 down-projection experts for TP rank 1 from the pinned checkpoint.
Both reconstructed original outputs match the captured gold bit-for-bit.
Holding activations, weights, scales, routing weights, expert/padding metadata,
locks, output buffer and kernel parameters fixed while alternating only the
two captured permutations reproducibly changes one BF16 value by
4.76837e-7. All 12 alternating calls match their respective captured outputs.
Canonical per-expert ordering gives 12/12 bit-identical repeats.

This demonstrates order-dependent rounding in this real projection.
It does not prove that every full-model difference has the same cause or
that all downstream differences are harmless. GLM was never stopped for
the component, its activity counters stayed quiet, and health remained 200.
Peak allocated CUDA memory was 737,977,344 bytes, about 704 MiB.

## GPU implementation and qualification

stable_dispatch_gpu.py sorts each consumed expert segment in place. Binary
search locates the segment from the existing ascending expert-block IDs;
bounded sorting buckets cover concentrated groups without sorting every
expert at the maximum width. Expert assignment, weights, padding count and
unconsumed buffer tails stay intact. No GPU metadata is copied to the CPU
in model forward. Single-token dispatch skips the extra kernel.

The opt-in wrapper in stable_dispatch.py covers only the pinned Marlin
producer with 288 experts, TP without EP mapping, unique top-k expert IDs,
and at most 2048 input tokens. These constraints match this test engine.
It is not a generic replacement for arbitrary routing/EP producers.

GPU qualification passes **92 cases**: two real captures plus 90 combinations
of nine token counts (1–2048), five block sizes (8/16/32/48/64), and balanced
or concentrated routing. All match the independent CPU ordering contract,
preserve expert IDs and padding, pass non-default streams and five graph
replays, and show zero persistent CUDA allocation growth. Peak comparison
allocation is 2,555,904 bytes. The real-case eager CUDA-event medians are
14.96–16.85 us, including CPU enqueue delay. Graph copy-plus-sort medians
are 6.96–9.31 us. A separate Torch trace records eight sorting kernels:
median GPU duration **3.408 us**, including a slower first traced call.
These are component measurements, not full-model throughput claims.

The compact controlled/GPU review is
benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/stable-dispatch-component-v1-review.json,
SHA256 9182dbb550c711f5c3a51a0096b37bd94a71525109c1a2e1af8fa8b66cad11cc.
The full source tensor hashes, captured arrays and Torch trace stay on Spark.

## Frozen matched model protocol

Both reference and U8 combined candidate use this same dispatch patch,
CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialization, and the same
pinned image/source, TP2 transport, attention repair and 4 GiB FP8 KV/rank.
The reference keeps weights/scales uncompressed at runtime. The candidate
uses the qualified v9 U8 combined storage. All six new source files were
frozen before startup; the two nodes' 114 Python/TOML file inventories match:
a41d6a6b120e4d67f82bf9c35b0ea946f8416a1e0847be57a9a1b2cf13c7dd43.
The two GPU dispatch files match the passed component's hashes.

Both tool flags are enabled. All prior source checks, graph/eager repeats,
32 held-out windows twice and 34 practical tasks stay in the protocol.
The run stops before longer work if the unchanged decode-repeat KL/NLL
floors fail, or before practical tasks if the held-out repeat floor fails.
Full-model Torch traces are collected only after unprofiled quality scoring.

After each completed quality client, a separate coordinator keeps that
experiment engine loaded for the existing three timing cases: 256 tokens
at concurrency 1/4 and 2048 tokens at concurrency 1, 64 generated tokens,
five trials with the existing warmups. Timing is unprofiled; an additional
concurrency-4 trace follows it. This reuses the two loads. The benchmark
configuration captures graphs for batch 1 only; concurrency 4 is an eager
probe, not an optimized production deployment.

Only the owned orchestration controller is held during these extra checks;
the engines and quality clients continue running. An independent finite
resume guard is armed before any hold, resumes it on coordinator death or
deadline, and stops only the extra owned timing clients. The original
restoration watchdog and outage deadline remain in force. No frozen source,
quality gate or original production-container configuration is modified.

## Locations and monitoring

ROOT on either Spark:

    /home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/full-model-v1

Controlled/GPU components:

    ROOT/dispatch-order-v1/results/report.json                 # Spark2
    ROOT/dispatch-order-v1-controller.json                     # head
    ROOT/stable-dispatch-v1/results/report.json                # Spark2
    ROOT/stable-dispatch-v1/results/stable-dispatch-torch-trace.json
    ROOT/stable-dispatch-v1-controller.json                    # head

Model run:

    ROOT/full-stablequalityv1-src/
    ROOT/quality-model-controller-stablequalityv1.json
    ROOT/stablequalityv1-controller.log
    ROOT/run-stablequalityv1/results/{control,candidate}-quality-measurement.json
    ROOT/run-stablequalityv1/results/{control,candidate}-stable-performance.json
    ROOT/run-stablequalityv1/results/performance-coordinator.json
    ROOT/run-stablequalityv1/results/performance-src/
    ROOT/run-stablequalityv1/{stock,kda}/traces/
    ROOT/watchdog-stablequalityv1.json
    ROOT/restoration-stablequalityv1.json
    ROOT/stable-quality-v1-collection.json
    ROOT/stable-quality-v1-service-check.json
    ROOT/stable-quality-evidence-v1-rank{0,1}.tar.gz
    ROOT/stable-quality-independent-archive-check-v1.json
    ROOT/stable-quality-v1-handoff/RUNBOOK.md
    ROOT/stable-quality-v1-handoff/{quality-review,profile-review}.json

The head's handoff folder contains both-rank trace inputs and a ready CPU-only
reanalysis command. It does not require stopping GLM or loading model weights.
The frozen experiment directories remain immutable.

The completed controller was PID 1710340; independent collector 1710339;
speed coordinator 1768123. The independent archive/service verifier was
2074039 and completed successfully. All were detached from SSH. The finite
resume-guard receipt remains under results/performance-resume-guard.json.
Local code checks pass: ruff, strict mypy (35 core files), **500 tests passed,
one GPU test skipped locally**; the explicit Spark GPU cases above pass separately.

The Mac's owned caffeinate helper PID 57655 prevents idle/system sleep for
eight hours from about 11:58 Berlin time. Screen lock is allowed; the lid
must stay open. Receipt: /private/tmp/mxwave-experiment-awake.json.
