# GLM lossless runtime storage: combined TP2 result

**Current state and next steps:** [GLM experiment status](GLM53_CURRENT_STATE.md).
The [matched C1/C4/C8 comparison](GLM53_PAIRED_SPEED.md) is complete and
independently verified: C1 throughput **+10.4–12.4%**, C4/C8 **−0.07% to
+1.84%**, TTFT **+2.3–5.9%**, memory **9.367 GiB / 5.33% saved**. C4 profiles
show about 8% shorter decode forwards but 6.6–6.7% slower prefill, including
36.26 ms of explicit KDA reconstruction. All 373 payloads, 672 source groups,
260 requests and 16,640 tokens verify on Spark/Mac; all 130 paired output
sequences match. Exact original service identities restored healthy at
**20:22:23 UTC**. Next address prefill reconstruction and mixed admission.
Numerical qualification remains unresolved; roofline is deferred. Earlier
runs below retain their original results and limitations.

**Component follow-up, 21:45 UTC:** V3's direct-output decoder variants pass
bits and fail speed gates; all event samples independently verify. The
separate V4 mask fix passes the base-255 regression and every BF16 bit
pattern, with both unsafe-control guards working. Complete CPU audit finds
no unsafe bases in 136 actual KDA TP shards, both heads or 154,880 embedding
rows. Originals restored healthy at **21:43:57 UTC**; no serving library or
adapter replacement. Read the [verified component](GLM53_PREFILL_RECONSTRUCTION.md)
and [assessment of MTP, routing and FP8](GLM53_SPEED_OPTIONS.md). There is no
new model throughput or numerical-qualification claim from this component.

**Further follow-up, 22:15 UTC:** [V5/V6 routing sweeps](GLM53_PREFILL_ROUTING.md)
independently verify all 51,200 events. Small-row fused GEMMs are much faster;
only 8/24 rows pass the full bracketed gate on both matrices. Control variance
prevents a blanket threshold, and large rows regress. The candidate's limit
stays eight. A [CPU MTP selector audit](GLM53_MTP_PILOT.md) finds twelve of
fourteen native draft selectors differ from the checkpoint policy before
namespace/alias repair, zero afterward. Base policy is unchanged; no MTP model
has been loaded. Original service is healthy after 22:03:41 UTC restoration.

**Earlier, 17 October: [stablequalityv1](GLM53_STABLE_DISPATCH_EXPERIMENT.md)
is complete.** The best U8 candidate saves 9.37 GiB / 5.33%; single-request
throughput improves 11.4–11.9%, concurrency-four is near parity, and TTFT rises
2.3–7.1%. Reference/candidate numerical gates now pass unchanged; all four
held-out float32 score sets are byte-identical (PPL 4.30501528). Overall strict
qualification remains failed on two lost output-format passes, despite equal
30/34 aggregate scores. The linked report includes eight new both-rank traces,
the 36–37 ms KDA prefill reconstruction cost, complete source/archive verification
and next steps. Original GLM is restored; no experiment engine is running.

**Earlier full-model quality result, 17 October:** the best U8 combined candidate
saves **9.367 GiB / 5.326%** over stock, 175.866 → 166.499 GiB across TP2.
Sampled WikiText PPL is 4.30444 / 4.30203 for reference repeats and
4.29993 / 4.30106 for the candidate. Both score 22/24 on the fixed math sample
and the same 28/32 completed strict task cases; the two tool cases are API
rejections and remain unmeasured. Source bytes pass all scale checks before
and after inference. No new throughput result is collected.

Strict qualification remains failed: reference graph/eager repeat KL is
3.084e-5 / 3.115e-5 against the unchanged 1e-5 floor. Candidate versus reference
is 2.979e-5 / 2.680e-5, with all generated IDs matching. Both arms also exceed
the teacher-forced NLL repeat floor. There is no measured PPL/math-score
degradation in this sample, but numerical equivalence is not established.
The [quality report](GLM53_BEST_CANDIDATE_QUALITY.md) records all gates, output
fingerprints and the next targeted MLP-input/routing test. Earlier component
results below remain historical; they do not supply new U8 model TPS.

**Latest component follow-up, 17 October:** exact shared nibble staging
does not improve the best U8 staging result. Selecting the layout on-device
costs 26.8–28.9% at concentrated M128; selecting it before launch reduces
this to 8.2–8.7%, still failing the 2% gate. Both variants pass their 56
real-layer, eight synthetic fallback and 48 decoder cases each. Compressed
storage is unchanged; no new whole-model result or quality claim.
[Decision and hardware-counter evidence](GLM53_SCALE_PALETTE_DIAGNOSTIC.md#nibble-staging-and-static-layout-selection).
Both original GLM ranks and their configurations are independently verified
restored, health 200, at 08:13:40 UTC; no active batch containers remain.

**Preceding shared-stage result:** moving scale expansion into the
existing shared-memory stage passes all 56 exact-output cases and reduces
concentrated M128 regression to 5.3–7.6% and M512 to 2.2–2.8%. Uniform routing
is near parity. NCU confirms substantially fewer executed instructions,
with unchanged register/shared-memory allocation. The 2% latency gate still
fails. [Full component evidence](GLM53_SCALE_PALETTE_DIAGNOSTIC.md#shared-stage-pipeline-follow-up).
No whole-model performance, allocation or quality measurement below changes.
Both original GLM ranks are independently verified restored and healthy.

**17 October follow-up: a denser expert-scale codec saves another 2.220 GiB
of live CUDA allocation across both Sparks**, 168.719 → 166.499 GiB
(1.316%) relative to the existing combined adapter. Complete GPU scale
reconstruction before/after inference and all basic chat checks pass.
All five answered practical tasks return identical final text; all 64
short-window generated IDs match. This is a separate storage/quality run,
not an update to the throughput table below. The current component still
regresses concentrated prefill and fails latency nonregression.

Strict numerical qualification remains failed: reference-repeat decode
KL is 0.00005497 versus the unchanged 0.00001 floor; reference/candidate
KL is 0.00006405 and mean NLL difference +0.00013973. Full distributions
are not bit-identical. Both practical interval scores fail identically on
an ambiguous tuples/lists contract; tool calling is rejected because
the launch lacks `--tool-call-parser`. The
[palette diagnostic](GLM53_SCALE_PALETTE_DIAGNOSTIC.md#full-model-memory-answers-and-numerical-limits)
and [verified review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/expert-scale-palette-full-model-v1-review.json)
separate proven storage savings from unestablished accuracy/serving claims.
No production adapter or checkpoint changes. Both original GLM identities
and configurations are independently verified restored, health 200, with
no active containers from this run. The preceding experiment results follow.

**Latest diagnostics: `bf16kernelsv2` and `cachepressurev2` completed, with GLM
independently verified restored and healthy at 20:09 UTC.** The
[same-weight BF16 comparison and NCU counters](GLM53_BF16_KERNEL_DIAGNOSTIC.md)
show M4 input near parity, output about 13% slower and the head about 13%
faster in repeated warm replay. A subsequent output cache-pressure check
changes 12.5% slower warm replay to 4.5% faster after eviction; both preserve
exact bytes. Output grid size and input shared-memory residency are tuning
leads, not proof of a general compression penalty. Higher split-K fails all
27 numerical checks and remains disabled. No adapter or full-model
performance/quality result changed.

**Previous full-model diagnostic: `nsysv2` completed; GLM was restored and independently
verified healthy at 19:14 UTC.** One candidate load, no new stock baseline:
39.30 / 37.80 / 39.02 tokens/s before/during/after Nsight capture. All timed
sequences match. The [Nsight report](GLM53_NSYS_MODEL_DIAGNOSTIC.md) identifies
fused KDA/head kernel time as the first component-profiling target and a
shared attention-metadata D2H synchronization. GC and main-thread runqueue
wait do not explain this capture's gaps. Existing quality status is unchanged.

For continuation commands, code locations and operating constraints, see the
agent handoff.

Latest follow-up **`combineddispatchv1` is complete**. The lighter dispatcher
cut MoE host time, but the whole-model throughput gap remains. GLM is restored
and healthy at that follow-up's end. The original `combinedv2`
measurements below are retained for comparison; the follow-up is recorded below.

The combined adapter saved **7.15 GiB of live CUDA allocation across two
DGX Sparks (4.06%, about 3.57 GiB per node)**. Single-request throughput
improved 7.6–10.0%; four-request throughput was 2.6% lower. The experiment is
complete and the original GLM containers are restored, with unchanged
configuration digests, health 200 and no active experiment containers.

This measures the imported `INCModel3/GLM-5.3-Flash-MXFP4-Mixed-CT-AutoRound`
checkpoint at revision `35d1b0d5cb4a0db075fd2749c897e970dc15bfa5`.
It does not measure the serving ModelOpt NVFP4 checkpoint. Both arms use
TP2, Marlin W4A16 experts, fixed 4 GiB KV allocation per node and CUDA graph
capture for one request. DFlash and MTP are disabled.

## Whole-model measurements

| Workload | Stock tokens/s | Combined tokens/s | Change |
|---|---:|---:|---:|
| One request, 256-token prompt | 15.32 | 16.85 | +10.0% |
| One request, 2,048-token prompt | 12.88 | 13.86 | +7.6% |
| Four requests, 256-token prompts | 39.87 | 38.82 | −2.6% |

These are median aggregate throughput over five trials, including prefill.
All 30 timed completions match the stock token sequences. Median time to
first token increased from 0.373 to 0.408 seconds, 1.144 to 1.208 seconds,
and 1.021 to 1.110 seconds, respectively. Thus improved decode does not
imply faster prefill. The saved stock control uses identical settings and
prompts but is sequential rather than contemporaneous or interleaved.

Allocation after measurement was **175.87 → 168.72 GiB** across both nodes.
The exact saving is 7,674,083,840 bytes, after scratch and other allocations.
The component payload accounting below is separate from that allocation:

| Storage path | Original bytes, both ranks | Retained bytes, both ranks |
|---|---:|---:|
| Routed expert E8M0 scales | 9,512,681,472 | 4,794,092,128 |
| 68 KDA BF16 projections per rank | 9,287,237,632 | 6,596,544,432 |
| Untied BF16 head and row-addressable embedding | 2,537,553,920 | 1,788,921,568 |

KDA scratch totals 413,138,944 bytes; expert GEMM scratch totals 75,497,472
bytes. Coverage is complete: 42 expert layers, 68 KDA projections and two
vocabulary components on each rank. There are 83 compressed scale tensors
and one exact stock fallback per rank. Four-bit expert codes, the KV cache
and other MXFP8/BF16 tensors keep their existing representations.

## Correctness and qualification

Both ranks passed all 658 base source comparisons and the attention source
checks. Nine of nine graph/eager/graph chat checks passed. The four complete
vocabulary GPU shard hashes match their sources; KDA and eligible expert
scales also verify exact reconstruction. Original dense KDA/vocabulary
parameters are released.

Actual autoregressive quality checks generated four tokens in each of 16
frozen contexts. All 64 generated ids match; all 48 decode distributions
have identical conditioning. Candidate mean decode KL is **0.000040806**;
stock repeating itself gives **0.000024518**. Mean decode NLL difference is
+0.000054727. These are diagnostic windows, not a standard accuracy benchmark.

**Strict quality remains unqualified:** stock-repeat KL exceeds the
predeclared 0.00001 stability floor. No threshold was relaxed. Weight bytes
are restored exactly, while fused GEMM arithmetic can change rounding.
The experiment therefore supports storage and measured performance claims,
not a claim of identical logits or established accuracy equivalence.

## What both-rank traces show

Four worker traces were reviewed, with 32 matching complete forward shapes.
Matched one-token model-forward ranges decreased from about 59.0 to 55.2 ms.
The compressed KDA GEMMs are active; embedding lookup costs only about
7–9 microseconds in decode. Neither is silently falling back to a retained
dense copy.

Larger prefills use bounded reconstruction followed by the stock GEMM.
The explicit KDA weight decoder costs about **36–38 ms per prefill**. Sampled
256-token forward ranges rose from about 360 to 417 ms, and 2,048-token
ranges from about 1,130 to 1,200 ms. These profiled ranges are separate from
unprofiled request timing.

Four-request decode is eager because this comparison captures graph size
one only. In five complete M4 forwards per arm, rank-0 uncovered stream
time rose from about 6.4 to 30.7 ms, while rank-1 communication kernel time
rose from about 3.1 to 25.8 ms. A separate correlation check includes copy
activities: rank-0 gaps are 6.25 versus 30.64 ms, of which about **0.52 versus
23.99 ms occur before the next launch API is entered**. This points toward
host submission delays and rank imbalance, rather than slow row decoding.

The M4 groups have different detailed context lengths and scheduling;
they are not exact matched shapes. Profiling can also amplify CPU overhead.
Late API entry does not distinguish Python, driver work, instrumentation,
OS scheduling or CPU waits. The trace identifies an optimization target;
it does not prove all 24 ms are removable application overhead.

Additional host-scope attribution makes the next target more specific. Across
the same 84 expert calls per M4 forward, inclusive host time is **3.88 → 11.98 ms**
on rank 0 and **4.87 → 11.64 ms** on rank 1. Expert GPU kernel time remains
comparable. Investigate the Python custom-op/native binding before changing
the compression format; these profiled host spans do not isolate ctypes or
predict the size of an unprofiled improvement.

This motivated the dispatch intervention below. Its result did not confirm
host dispatch as the principal unprofiled throughput limit.

KDA also creates **170 helper activities per M4 step per rank**: 68 padding
fills, 68 device-to-device copies and 34 output compactions. They cost only
**0.20–0.21 ms of GPU time**. Their host launch/allocation overhead is a
secondary target; they are not evidence that GPU copy bandwidth explains the
regression. The host copy API's large whole-trace times include waits shared
by both arms. The handoff's trace breakdown
and [dispatch report](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/full-model-combined-dispatch-review-v1.json)
record counts, timing attribution, source binding and limitations.

## Completed dispatch intervention

`MXWAVE_MOE_DISPATCH=dispatcher` uses a direct PyTorch registration with the same
native kernel, mutation schema, fake behavior and dynamic tensor pointers.
Both dispatches matched stock bytes in all 20 raw/aligned component scenarios,
including eager, non-default streams and CUDA graphs. Aligned M=4 host submission
fell from about 279 to 70 microseconds per block; elapsed block time improved
about 1–2%. [Component evidence](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/moe-dispatch-review-v1.json).

The candidate-only full-model test reused the existing timing/decode controls:

| Workload | Stock tok/s | Previous combined | Dispatcher | Dispatcher vs stock |
|---|---:|---:|---:|---:|
| B1 / 256 prompt | 15.32 | 16.85 | 16.76 | +9.3% |
| B1 / 2,048 prompt | 12.88 | 13.86 | 13.94 | +8.2% |
| B4 / 256 prompt | 39.87 | 38.82 | 38.89 | −2.5% |

The B4 change over the previous candidate is only **+0.16%**. Saving is unchanged
at **7.15 GiB**; all 30 timed completions and 64 quality ids match stock. Mean
decode KL is 0.000042516 and mean NLL difference −0.000030619. The same unstable
stock control leaves strict quality unqualified. B4 TTFT remains 1.099 seconds,
versus 1.021 stock. Original container identities/configurations were independently
checked after restoration: health 200, no active experiment containers.

New traces show MoE host spans **11.98 → 1.59 ms** on rank 0 and **11.64 →
1.54 ms** on rank 1. This large reduction did not translate to whole-model speed.
Much of the host work can overlap GPU execution, and profiling/run conditions
can change apparent gaps. M4 rank-0 gaps remain 16.55 ms and rank-1 collective
spans 16.93 ms; unchanged KDA kernels also vary between runs. These are five
count-matched profiles, not identical context/scheduling samples; no live
clock/power or DRAM/SM counters were captured.

A post-hoc client-arrival diagnostic gives median middle-token intervals of
84.77 ms stock and 84.75 ms dispatcher; mean intervals are 85.58 and 86.80 ms.
Typical middle-token cadence is close, while prefill and slower intervals
remain relevant. This token-8–55 diagnostic is not a replacement acceptance
metric. [Full intervention review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/full-model-dispatch-v1-review.json),
[archive and analysis hashes](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/dispatch-evidence-v1.json).

## Synchronization follow-up: arrival skew confirmed

An additional offline review paired the same 91 main-stream RoCE all-reduces
by forward annotation and call order on both ranks. Five M4 forwards per arm
give 450 adjacent-collective interval pairs per arm. Across the five candidate
forwards, a rank's extra time between collectives correlates **0.9952–0.9996**
with the other rank's extra duration inside the next collective. Only local
durations/intervals are compared; unsynchronized cross-host timestamps are
not subtracted.

For example, one candidate collective takes **6.432 microseconds on rank 0**
and **1,040.485 microseconds on rank 1**. Rank 0 arrives 1,033.885 microseconds
later relative to the preceding collective, including 793.577 microseconds of
GPU gaps before the next launch API is entered. This is strong evidence of
peer-arrival waiting, rather than a millisecond spent transferring that message.
Stock also shows rank skew, sometimes in the opposite direction.

Source hashes were checked against the running pinned image. The RoCEnante
kernel stages data, rings a native CPU RDMA proxy, polls peer payload flags,
then reduces. Its elapsed GPU time includes waiting for peer/proxy progress.
Its single-channel protocol orders collectives; simply removing waits cannot
make dependent TP layers independent. Both arms have the same **84
`cudaStreamWaitEvent` calls per M4**, owned by `vllm::moe_forward_shared`, and
one short `cudaStreamSynchronize` in `aten::copy_` (about two microseconds).
Short event API calls can still enqueue longer GPU waits; they do not measure
the wait duration. No additional per-layer CPU synchronization was identified
in the compression adapters.

One concrete harness omission was also confirmed: production sets
`GLM_ROCE_PROXY_CPUS=auto` on both Sparks, while the completed stock/candidate
launches omitted it and the image does not provide it by default. Production
therefore requests performance-core pinning of the RDMA proxy; the experiments
did not request it. Both measured arms shared this omission, so it does not
alone invalidate their relative comparison or explain the regression.
Its performance effect remains unmeasured.

The local launcher now inherits the explicit public RoCE proxy/limit settings.
Timing and decode control reuse now reject a difference in either rank's
transport or OMP environment, before pausing production. Checks passed: ruff,
strict mypy and the full pytest suite. The offline review itself started no
GPU run; the subsequent pinning component is recorded below.
[Synchronization evidence](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/full-model-synchronization-review-v1.json).

## Completed pinning comparison: no consistent primary win

`pinningv2` ran two isolated TP2 processes using the installed RoCEnante
runtime, with no model weights. The first resident-GLM attempt, `pinningv1`,
failed during CUDA initialization before taking measurements. The user then
authorized an idle pause of both GLM ranks. The retry used a 6 GiB host cap
instead of 3.5 GiB and warmed the synthetic delay before calibration; these
changes do not isolate the exact cause of the startup OOM.

The completed test measured six cases in eight balanced ABBA/BAAB blocks.
Each message was BF16 `[4, 4096]`, or 32 KiB; forward-like cases contained
91 collectives with approximately 1 ms GPU or host spacing. Both ranks
completed **25,524 collectives**, with exact output bytes, zero transport
errors, and observed proxy affinity changing from all 20 CPUs to CPU 19.
Elapsed benchmark time was about 22.6 seconds per rank.

| Measurement | Effect of pinning, across both ranks/orders |
|---|---:|
| 91 collectives with GPU spacing, wall median | **−0.18% to +0.40%**, mixed direction |
| 91 collectives with CPU-sleep spacing, wall median | **2.00–4.40% faster**, secondary case |
| Back-to-back graph burst, wall median | **−11.86% to +5.71%**, mixed direction |
| Isolated GPU-spaced collective p95 | Inconsistent across ranks/orders |

Positive percentages indicate reduced latency. The primary gate required at
least 5% lower isolated GPU-spaced p95 on **both ranks in both order groups**,
without more than 2% median regression. A full-model follow-up also required
at least 1% lower GPU-spaced forward wall median in every group. **Both gates
failed.** No additional model comparison was launched.

The host-sleep case is a possible scheduling lead, not model throughput.
It deliberately includes OS timer/wakeup behavior. In the GPU-spaced case,
median proxy runqueue waiting increased from about 0.20–0.28 ms to 1.89–1.98 ms
per measured case region under pinning. These counters are not a scheduler
trace and do not establish the original model slowdown's cause. GPU SM clocks
were sampled outside timing regions; memory clocks were unavailable and no
DRAM/SM utilization counters were collected.

An independent restoration check at **18:11 UTC** verified both original GLM
identities/configurations running, health 200, and no active experiment
containers. The 102,745-byte archive retains 22 hash-verified payload files,
including both attempts, both-rank reports/logs, frozen source and restoration.
[Pinning measurements and evidence index](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/roce-pinning-v2-review.json).

## Adapter versus engine: additional offline attribution

The saved dispatcher traces were reviewed again, assigning each late-launch
interval once to an innermost CPU operator/API and separately marking whether
an adapter scope was active. All GPU streams, copies and memsets were checked
for overlapping work. This started no model/GPU run and did not pause GLM.

On candidate rank 0, the median 10.03 ms before launch API entry contains
1.07 ms inside adapter scopes and 8.96 ms outside. Across all five forwards,
the adapter-contained share is **9.6%**. Most temporal attribution is to stock
`moe_forward_shared`, attention/indexer calls, TorchDynamo and between-operator
time. No identified late-launch interval overlaps another traced GPU activity.
About 0.55 ms of the broader 16.35 ms main-stream gaps does overlap another
stream; main-stream gaps and device inactivity must therefore stay distinct.

This does **not** establish that the adapter is innocent or cap its possible
effect at 1 ms: extra work, object lifetimes or profiler callbacks can affect
later code. Conversely, a long stock-op span may include descheduling rather
than slow operator execution. The earlier stock trace also shows rank skew,
with most lateness on the opposite rank. Temporal scope ownership is not
a causal allocation of the unprofiled throughput difference.

Some simple explanations are unsupported by the counts:

| Per-M4 CPU operation count, same on both ranks | Stock | Candidate |
|---|---:|---:|
| TorchDynamo cache lookups | 149 | 149 |
| Shared-expert forward calls | 42 | 42 |
| `aten::empty` | 2,818 | 2,309 |
| `aten::contiguous` | 137 | 171 |
| `aten::copy_` | 306 | 408 |

The additional contiguity/copy wrappers fit the known KDA padding/compaction
path; CPU call counts are not device allocation or transfer counts. No new
lookup explosion was observed, and globally increased `aten::empty` count
does not explain the sampled regression. The existing traces lack Python/
native stacks and OS scheduling events, so they cannot separate interpreter,
driver, profiler and scheduler causes. Shape recording defaults to false in
this pinned vLLM; do not assume shape recording caused the overhead.

Nsight Systems **2025.3.2** is installed on Spark. CPU-only smoke traces
verified OS-runtime events and native call chains while GLM kept serving.
The later full-model `nsysv2` export includes independently decoded Python
frame contents; see the diagnostic linked above. `perf_event_paranoid=4` and denied tracefs
access mean precise scheduler events were not collected; OSRT-inferred
thread activity is not a replacement. The initial CPU-only probe preceded
the completed full-model diagnostic linked above.
[Attribution and capture plan](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/gap-ownership-v2-review.json),
[reproducible evidence archive](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/gap-ownership-evidence-v2.json).

## Next work, in order

1. Keep the dispatcher opt-in. Both the candidate-only Nsight diagnostic and
   the [same-weight BF16 kernel comparison](GLM53_BF16_KERNEL_DIAGNOSTIC.md) are
   complete. Prioritize KDA output work distribution preserving K accumulation,
   then input shared-memory footprint; do not enable the numerically rejected
   split-K settings. Establish a repeatable correctness-preserving component
   lead in both warm and streaming/cache-pressure conditions before another
   full-model load. The measured 21–22 ms KDA / 2.39 ms
   head includes useful projection computation, not pure compression overhead.
   Investigate exact host attention metadata reuse next. Its 32-byte D2H
   call waits on queued GPU work, so do not treat its full host duration as
   recoverable latency. Preserve the async/speculation correctness guard.
   The [detailed diagnostic](GLM53_NSYS_MODEL_DIAGNOSTIC.md#next-steps) records
   the order and limits. Do not repeat pinning or Python dispatch tweaks
   without a component lead; their completed tests did not close the gap.
2. A matched batch-four CUDA-graph trial is a separate configuration experiment.
   The harness currently forces graph size one and requires a saved control;
   add a tested fresh matched-stock mode before changing those settings or
   proxy placement. Old unpinned controls cannot qualify a pinned candidate.
3. Keep bounded prefill reconstruction until a measured replacement wins.
   Previously rejected fused-prefill and naive-overlap paths should not be
   repeated unchanged.
4. Investigate stock-repeat instability, then run a larger held-out quality
   screen before deployment or an accuracy claim. Package runtime adapters
   after qualification; a llama.cpp/Ollama port needs separate GGML integration.

The current prototype depends on a pinned vLLM image/native ABI and selective
selector, constructor and loading-context hooks. It is not an unmodified
stock-vLLM or Ollama feature. The missing embedding hook in `combinedv1`
was fixed and qualified before this candidate-only retry. Completed stock
timing and decode controls were reused with identity and file-hash checks.

Checkpoint reading improved from **629.52 to 40.06 seconds** by inheriting
the existing production slab-reader flag. Total model loading, including
adapter preparation, was 244 seconds; the API became healthy after about
365 seconds. Prepacking and startup cache work remain possible follow-ups.

## Evidence

[Measurements and restoration](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/full-model-combined-v2-review.json),
[both-rank trace review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/full-model-combined-trace-review-v1.json),
[launch-gap check](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/full-model-combined-launch-gaps-v2.json),
and [archive index](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/combined-evidence-v1.json).

The raw 115.18 MB archive retains 386 hash-verified reports, probability
files, both-rank traces, source snapshots and logs outside Git. Model weights,
native binaries and compile caches are excluded. Repository checks passed:
ruff, strict mypy and the full pytest suite.

## Related work and datacenter relevance

[DFloat11 on Hugging Face](https://huggingface.co/DFloat11) publishes losslessly
compressed BF16 models, reporting roughly 30% smaller weights.
[ZipServ](https://arxiv.org/abs/2603.17435) studies fused lossless decompression
and GEMM. [Approaching Shannon Bound with Lossless LLM Weight Compression](https://arxiv.org/abs/2606.15789)
also studies quantized formats and multi-GPU SGLang integration. The general
approach is established; this experiment is not evidence of conceptual novelty.

Our measured question is additional runtime storage compression on top of an
already quantized MXFP4 checkpoint. Datacenter value would depend on capacity,
throughput and tail latency at matched quality and service requirements.
Extra context/batch capacity or avoiding another GPU could matter even with
modest percentage savings; those benefits require measurement on the target
hardware. Spark results do not qualify datacenter GPUs or high-concurrency
serving. Current whole-model evidence remains 7.15 GiB saved, single-request
throughput improved, B4 throughput regressed and strict quality unqualified.

The [kernel follow-up](GLM53_BF16_KERNEL_DIAGNOSTIC.md#compact-n8-activation-staging-correctness-passes-speed-fails)
rejects the compact N8 staging candidate despite exact weights/outputs and
an improved resource limit. Its smaller shared allocation is not an additional
resident-model memory saving. The
[rANS feasibility screen](GLM53_BF16_KERNEL_DIAGNOSTIC.md#paper-follow-up-and-actual-mxfp4-compressibility)
uses actual MXFP4 expert codes/scales: codes show little independent-symbol
headroom, while scales warrant comparison with the existing compressed
runtime representation. Neither component work changes the measured
whole-model memory, throughput or quality results above.
