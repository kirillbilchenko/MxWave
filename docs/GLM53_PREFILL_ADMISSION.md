# Prefill admission: evidence and next serving test

**4,096-budget follow-up complete, 19 October 2026, 14:18 UTC.** The larger budget fails
the prospective scheduling gate. 42/42 matched numerical
checks pass; allocation saving is 9.358188 GiB across TP2. All latency and
memory failures remain recorded. No setting is promoted. See the
[measured result and evidence](GLM53_PREFILL_BUDGET_4096.md).
Older checkpoints below remain historical.

**Bounded latency follow-up complete, 19 October 2026, 11:33 UTC / 13:33 Berlin.** The 2,080-token
cap at 10 ms admission passes its prospective gate and preserves 9.367162 GiB
TP2 saving; all 42 matched numerical comparisons pass. Compressed C16 headline
TTFT improves 1.819 -> 1.430 s, but mean improves only 13.7 ms with uncertainty
including zero. Traces still show three prefill passes / 204 reconstructions
per rank. Remaining latency is not solved; no configuration is promoted.
Original service is restored and all jobs are finished. See the
[final result and retained evidence](GLM53_PREFILL_BUDGET.md).
Older checkpoints below retain their historical measurements and failures.

**Final C16 admission result, 19 October 2026, 09:01 UTC: complete.** The 10 ms policy
passes its prospective gate. Short-C16 TTFT is stock 1.383 s /
compressed 1.694 s (gap +311.5 ms).
This improves admission latency but leaves the C16 compression TTFT penalty unresolved.
All 94 numerical comparisons pass; CUDA saving is 9.367162 GiB across TP2.
Original GLM is restored healthy; no configuration is promoted. This concludes
compression experiments per user scope. See the [verified admission result](GLM53_C16_ADMISSION.md).
Older entries below are historical checkpoints.

**19 October 2026, 08:15 UTC: final C16 admission experiment is running.**
The user resumed this final compression experiment: short C16 in both arms,
3 versus 10 ms, with short-C1 delay controls and the qualified warmed exact16
kernel. All 207 parent sources are unchanged; four measurement additions are
frozen. Further compression-kernel tuning stops. No latency result is yet claimed.
See the [live admission status](GLM53_C16_ADMISSION.md). MTP has not started.
Older pending/paused statements below are historical checkpoints.

**19 October 2026, 07:43 UTC: full-model input-FP32 result is complete and fails C16.**
Spark/Mac verify 1,634 payloads / 229 sources / 12 traces. C16 KL is
1.6018e-5 > 1e-5 despite exact self-repeats and matched grouping; C1/C8 pass.
The preceding warmed exact16 route remains the qualified experimental fallback.
Original GLM is restored healthy, retention is complete and the awake lease is
released. No new experiment or configuration promotion. Next proposed phase is
3 versus 10 ms admission with the fallback, then MTP; neither has started.
See the [full result](GLM53_FP32_INPUT_MODEL.md) and [current handoff](GLM53_HANDOFF.md).
Older entries below describe historical checkpoints, superseded where pending.

**19 October 2026, 05:54 UTC: C16 follow-up audited, not yet run.** The later
warmed-model benchmark already uses **3 ms in all 72 timing groups**. In its
six short-C16 trials, first-cohort request counts are stock **8/4/1/8/8/7** and
compressed **4/8/6/5/3/5**. The first 2,048-token budget fills in 3/6 versus
1/6 trials; other windows stop at their deadline. These observations motivate
**3 versus 10 ms**, keeping both arms' native scheduler, token/KV limits,
graphs and kernels unchanged. They do not prove causation or a future TTFT gain.
Sixteen 256-token prompts total 4,096 tokens, so even complete collection still
requires at least two legal prefill forwards.

A CPU fixture against the actual frozen admission policy verifies the C16
token-budget stop with synthetic arrivals. For 0/0.5/2 ms spacing, 3 ms collects
8/7/2 requests and 10 ms collects 8/8/6, preserving deadlines and the remaining
queued messages. No API timing is inferred. A draft compares twelve short-C16
and six short-C1 repetitions per window/arm, with balanced window order;
quality/profile details, client/reviewer and source freeze remain to finish
after the active `fp32inputmodelv1` result. No additional model run is launched.
The [retained audit and CPU evidence](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/fp32-input-model-v1-TTFT-options-retention.json)
preserve all original failed gates. The old C4 result below does not establish
C16 behavior under the newer native-mHC and deterministic-order policy.

**Completed and independently verified, 18 October 2026, 07:52 UTC.**
Both 3 ms and 10 ms idle-only admission windows pass the frozen performance
and scheduling gates. Short-C4 throughput improves **6.75% / 6.62%**, median
TTFT falls **1.125 → 0.778 / 0.782 s**, and median large forwards fall
**two → one**. Exact original service identities/configs/mounts are restored,
HTTP 200 at **07:49:50 UTC**. No configuration was promoted: all three
policies, including zero wait, still fail numerical repeats.

All 108 groups / 450 requests / 28,800 generated tokens and all 599 evidence
payloads verify on Spark and Mac. Every one of the **300 paired timed output
sequences** matches. The same compressed mixed-MXFP4 target, two SM121 Sparks,
`[1,4]` graphs, KV allocation, native kernels and token limits are used throughout.

| Case | Zero wait TPS | 3 ms TPS | 3 ms change | 10 ms TPS | 10 ms change |
|---|---:|---:|---:|---:|---:|
| C4 / P256, primary | 40.873 | 43.631 | +6.75% | 43.580 | +6.62% |
| C1 / P256 | 16.806 | 16.713 | −0.55% | 16.709 | −0.57% |
| C8 / P256 | 57.585 | 59.685 | +3.65% | 60.000 | +4.19% |
| C4 / P2048 | 26.105 | 26.172 | +0.26% | 26.038 | −0.26% |
| C4 / P256, arrivals 2 ms apart | 40.899 | 41.044 | +0.36% | 43.640 | +6.70% |

Primary medians use all twelve trials/window; controls use all six. Both
windows meet the 5% primary gain, fewer-forwards, 2% control-throughput and
15 ms C1 TTFT guards. Observed C1 median TTFT differences are −11.63 / +2.42 ms;
these sample differences do not imply that waiting has no intrinsic latency cost.
Long C4 still needs four 2,048-row forwards and stops waiting at the token budget.
Three milliseconds does not consistently combine the staggered cohort; ten
reduces its median large-forward count to one.

The separate profiles settle the repeated-weight question within their scope:
zero wait runs two eager forwards, each visiting **all 68 KDA weight prefixes
once**, on both ranks. Each wait policy runs one. There are no duplicate
prefixes within these forwards. Actual device decompression kernels match
the observed **136 / 68 / 68** reconstruction calls per rank. All six native
worker traces verify; three API traces remain separate. Saved methods restore
exactly and graph capture/replay calls are excluded from Python weight counts.

Each numerical comparison has 48 real decode steps with matching conditioning
and IDs. Every NLL difference passes the unchanged absolute 1e-4 floor; every
KL fails the unchanged 1e-5 floor:

| Comparison | Mean decode KL | Mean NLL difference |
|---|---:|---:|
| Zero-wait repeat | 6.0946e-5 | −6.6952e-5 |
| 3 ms repeat | 3.0801e-5 | +6.5745e-5 |
| 10 ms repeat | 1.3036e-4 | −6.8646e-5 |
| Zero A → 3 ms A | 5.4175e-5 | −6.1683e-5 |
| Zero B → 3 ms B | 4.2851e-5 | +7.1014e-5 |
| Zero A → 10 ms A | 1.2261e-4 | +2.3326e-5 |
| Zero B → 10 ms B | 4.0238e-5 | +2.1631e-5 |

This measures complete finite requests with 64 output tokens. It establishes
less prefill work and higher throughput for those groups; sustained decode-only
speed and continuous arrivals during active decode were not tested. This is a
general serving-policy lead on the compressed target. Measure stock and
compressed targets with the same policy before claiming an increased compression
advantage. It is not a new PPL/task-accuracy qualification or an MLA correction.

All 336 source-scale groups and both 170-file source bindings pass. The only
parent change is plugin metadata; 159 other parent files remain byte-identical.
Archive `evidence-prefill-admission-v1.tar.gz`, 131,372,757 bytes, SHA256
`d63d6e69c7acda35d5d6f6cb83b38bbb1fea958b2681b1611a24930db64a9da7`.
CPU probability recomputation differs across machines by at most 1.7765e-15;
the verifier permits 1e-12 only for recomputation, leaving model gates unchanged.
All other review fields/decisions agree exactly. Reviews include explicit
roundoff metadata rather than claiming byte equality.

[Verified result, every trial and both-rank trace checks](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/prefill-admission-model-v1-review.json).
Source/results stay at `ROOT/full-prefilladmissionv1-src` and
`ROOT/run-prefilladmissionv1/results`; controller is complete and unpaused.
Local continuation/evidence: `/private/tmp/mxwave-prefill-admission-gpu-v1/`.
Next resolve actual MLA internals, freeze the corrected MTP validator after its
passing actual-geometry CPU proof, and qualify exact C2 separately. A combined
graph/admission policy has not been measured. Roofline remains deferred.

**Historical GPU launch, 18 October 2026, 07:19 UTC.** Revision
`prefilladmissionv1` uses one qualified compressed target and compares native
input collection windows **0 / 3 / 10 ms**, with `[1,4]` graphs, no MTP and
unchanged model arithmetic. Both-node source/native bindings and the actual
two-plugin CLI discovery pass. Full local tests, Ruff and strict core typing
pass. No new serving result yet.

The existing idle guard, independent watchdog and exact-container restoration
control this run. Controller **1248710**, watchdog **1251294**, collector
**1248709**; inspect `ROOT/quality-model-controller-prefilladmissionv1.json`
before assuming whether the original service is paused. Local source,
preparer, collector and verifier: `/private/tmp/mxwave-prefill-admission-gpu-v1/`.

| Case | Requests | Prompt length/request | Arrival spacing | Trials/window |
|---|---:|---:|---:|---:|
| Primary short C4 | 4 | 256 | 0 ms | 12 |
| Single-request control | 1 | 256 | 0 ms | 6 |
| C8 control | 8 | 256 | 0 ms | 6 |
| Long C4 control | 4 | 2,048 | 0 ms | 6 |
| Staggered C4 | 4 | 256 | 2 ms/request | 6 |

All six window-order permutations occur equally per case; workload order
also reverses. The plan retains **108 timed groups / 450 requests / 28,800
generated tokens**, exactly 64 outputs/request, plus two 16-token warmups
per case/window. Client wall time includes intended staggering; actual
request-launch offsets and every stream arrival are retained.

The wrapper waits only after the engine was idle and only for fresh text
prefills. It calls the native message handler, stops at the original fixed
deadline, and stops early at token/request budgets or a control/abort. Decode,
in-flight, paused, resumed, connector and streaming work is excluded. It
records elapsed collection time without clipping OS/message-handling overrun.
The original scheduler and token budget are unchanged.

The prospective gain gate requires short-C4 median TPS at least **1.05×**,
fewer large eager forwards, all control TPS at least **0.98×**, and C1 median
TTFT cost at most **15 ms**. Every window receives two C4 full-vocabulary
repeats on 16 contexts / four outputs; all seven repeat/cross comparisons
retain KL **1e-5**, absolute mean NLL **1e-4**, IDs and conditioning. This
does not qualify task accuracy or other concurrency levels.

After all timing, three separate C4 profiles observe weight identities by
`(forward ID, weight prefix)` on both ranks. The collector compares observed
reconstruction calls with actual BF16 decompression kernels. Python graph
capture/replay calls are excluded; saved methods must restore exactly.
A tiny GPU observer fixture runs before target loading and tests unchanged
BF16 output bits and capture exclusion; it does not requalify the KDA kernel.

All 159 other parent files stay byte-identical. The sole parent
change adds the admission plugin alongside `quality_worker:install`; ten new
modules produce 170 bound source files. Source archive SHA256
`6ea4b7b85a65eb9d846e365dee5477c0a88bce4be07dadf49838b356a82a7904`;
prepared SHA256 `6dfa66c454f05728a976e10c6993e1b1cd0e16c34304d8f75c380056936d2741`;
plan SHA256 `639475566d54800c38661d8bb545eb54edcc9723ee053e094e94b4b118a6b341`.
Source `ROOT/full-prefilladmissionv1-src`; results
`ROOT/run-prefilladmissionv1/results`. Historical MLA/repeat and format
failures remain open; no configuration is promoted by this experiment.

**Historical offline audit, 18 October 2026: no scheduling change measured yet.**
MTP is separate from this work. The analysis checks all **60** existing paired
timing groups and both ranks' eager-dispatch deltas, plus the two controlled
compressed-candidate traces. It leaves every serving result and numerical
gate unchanged. No model is loaded and the original service stays running.

At C4 with 256 prompt tokens/request, all 1,024 prompt tokens fit the existing
2,048-token scheduling limit. Nevertheless, admission sometimes produces two
or three large pure/mixed forwards. Every base forward contributes 42 expert
producer calls, and both ranks' observed call-count deltas agree.

| Large forwards | Trials per arm | Reference tok/s | Candidate tok/s | Candidate median TTFT, s | Inferred candidate reconstructions/rank |
|---|---:|---:|---:|---:|---:|
| 1 | 1 | 43.39 | 44.05 | 0.783 | 68 |
| 2 | 3 | 40.37 | 41.10 | 1.131 | 136 |
| 3 | 1 | 32.05 | 29.27 | 2.490 | 204 |

Two-forward entries are medians of three trials; other entries are individual
trials. These groups are selected after measurement and are not randomized
interventions. Do not report 44.05 versus 29.27 as an achieved optimization.
The reference slows with fragmented admission too: reconstruction alone does
not explain the pattern. Differences in queueing, GEMM shapes and complete
forward execution also need measurement.

The inferred reconstruction count follows the frozen candidate path: above
eight rows, each of 68 compressed KDA projections reconstructs its weight.
The Python counters do not directly count these device kernels and do not
count graph replays. Their timing-group totals are therefore an inference
from the unchanged operation, rather than measured weight-identity counts.

The separate controlled C4/P256 profile directly records **68 KDA operations
and 68 reconstruction kernels per rank**. Matching profiler external IDs
assigns exactly one reconstruction to every operation. Their kernel durations
sum to **36.259 / 36.263 ms** on ranks 0/1. Ranks run concurrently; do not add
these times. The trace lacks weight identities, so it cannot prove that the
same weight was never invoked twice within a forward.

At C4/P2048, all ten timing groups already have four 2,048-row forwards, plus
a small seven-row forward. Four chunks are required by the existing token
budget; the seven-row operation uses the compressed fused path. A wait window
alone cannot turn these 8,192 prompt tokens into one legal forward.

The next serving test should use the unchanged compressed target **without a
drafter**, with the following fixed comparison:

1. Inspect the pinned scheduler's admission boundary and implement a bounded
   opt-in wait only when a new prefill cohort starts. Freeze its maximum delay
   before timing; do not hold every decode step or change token limits together
   with the wait. A local opt-in prototype and native CPU fixture now pass;
   its GPU comparison is deferred for the C2/C8 graph experiment.
2. Count eager forwards, scheduled rows and reconstructions by
   `(forward ID, weight prefix)`. Observe actual device kernels in a separate
   profile. Python capture-time calls cannot stand in for graph replay counts.
3. Compare default and bounded-wait arms in alternating order on identical
   C1/C4/C8 requests and fixed output lengths. Retain every request, arrival,
   queue delay, TTFT and completion latency, including the slow groups. Use
   simultaneous and staggered arrivals to expose any batching tradeoff.
4. Require an actual reduction in split forwards/rebuilds, improved API results
   after the added wait, source checks and unchanged numerical gates. More
   merged batches alone do not qualify a serving improvement. Keep numerical
   failures visible if the baseline repeats still fail.

The [small-prefill routing sweep](GLM53_PREFILL_ROUTING.md) remains a different
intervention. It did not qualify a blanket crossover, and 257-row work regressed.
That matters because observed fragmented shapes include 257, 259, 513 and 769
rows; extending a tiny-row rule would not automatically resolve them.

[All-trial audit and trace correlations](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/prefill-admission-v1-review.json)
retain input hashes, every timing group and interpretation limits. The authored
analysis is `/private/tmp/mxwave-prefill-admission-v1/analyze.py`; original
verified evidence is `/private/tmp/mxwave-paired-speed-v1/`. This local analysis
has not been uploaded to Spark.

The separate source-only CPU fixture uses the real pinned input-drain loop
and message handler, with bounded scheduler/model stand-ins. Seven cases
pass: zero wait, collection, ongoing decode, full token/request budgets,
abort and wakeup. No CUDA or model allocation, and the original service stays
healthy. All 11 archived payloads verify locally; archive SHA256
`8e0bb635b34ffeefb10e0289e355d0f89eb07ed38049a710da352e785bbaecf6`.
This establishes adapter control flow, not full scheduler behavior or speed.
