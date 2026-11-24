# C8 repeatability: exact-workload trace investigation

Updated **18 October 2026, 18:50 UTC**. Both the native-mHC trace investigation
and the subsequent fresh stock/compressed pair are complete. The
[current handoff](GLM53_HANDOFF.md) records operational state and next work.

## Fresh matched stock/compressed comparison: matchedc8v1

All **33 numerical comparisons pass**, including eight self-repeats and every
cross/profile pair, with equal actual forward shapes, prompt/row order and
chunk boundaries. Both arms retain native mHC and stable sparse compaction.
C1 cross-arm mean decode KL is 5.803778255448176e-6, NLL difference
-3.6509906652337038e-6; C8 KL is 1.194444923172159e-6, NLL difference
2.5822470640270547e-6. The original gates remain KL <=1e-5 / absolute mean
NLL <=1e-4. All self-repeats have KL/NLL zero. Cross-arm rows are not all bitwise
equal: first C1 0/16 prefill and 0/48 decode rows; first C8 16/16 and 46/48.

C8 quality uses controlled queue/resume and verifies actual grouping;
ordinary API speed uses the 3 ms idle-only admission window without that control.
Six trials/case/arm give C1 +12.38%, short-C8 +2.40%, long-C8 -0.61% throughput.
C1 TTFT +7.91% exceeds its 5% ceiling, and C8 misses 5% throughput gain.
**The numerical gate passes; the overall speed gate fails.**
All 102 timed paired token sequences match (6,528 timing tokens per arm).
CUDA allocation before profiles falls 175.907657 -> 166.540494 GiB across TP2.
Host RSS does not improve; Spark memory counters overlap and must not be added.

One compressed short-C8 trial is slow (42.484 tok/s, TTFT 2.989 s) and initially
admits six of eight prompts. Endpoint clocks/temperatures do not show a large
drop for that trial; an unprofiled trial cannot establish the complete cause.
It remains in all declared summaries. Arms run stock then compressed with a time
gap and unlocked clocks; small speed differences retain that uncertainty.

All **1,168 payloads / 194 source bindings / eight worker traces** independently
verify on Spark/Mac; all decisions match, maximum CPU float64 difference
1.7759773686398805e-15. All profiled forwards are covered; controlled profiles
record six graph launches/rank, mixed-length profiles fifteen. Both actual CLI
checks, Ruff, strict mypy and pytest pass (one GPU skip). Exact original service
identities/configs/mounts and health 200 are restored. No configuration is promoted.
This is finite C1/C8 compression qualification, not general batch invariance or
universal zero inference error. Full-model C16 remains pending; MTP is deferred.

Archive: evidence-matched-c8-v1.tar.gz, 316,083,034 bytes, SHA256
eb244b63a6aeb58c337f89820de93cead3210ee0d7121a49677fcbb88ac8341b.
Local full evidence: /private/tmp/mxwave-matched-c8-v1/evidence/.
Retained repository [review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/matched-c8-v1-review.json),
[cross-host proof](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/matched-c8-v1-cross-host-verification.json)
and [supplemental timing/memory analysis](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/matched-c8-v1-supplemental-review-v2.json).
The [fresh offline trace breakdown](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/matched-c8-v1-fresh-profile-review.json)
finds no reconstruction in controlled M8 decode; compressed KDA takes
21.91-23.88 ms and expert GEMMs about 56 ms in kernel-duration sums/rank.
Both prefill shapes (2,048 and 23 rows) reconstruct 68 matrices in
35.82-36.90 ms/forward/rank. Sums may overlap; they are not API latency or
measured memory bandwidth. The original supplemental v1 name filter missed
native Decompress kernels; v2 corrects this derived field, preserving v1.
Every historical failed run and gate is retained below.

### Latency cost and the remaining outlier

The [arrival audit](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/matched-c8-v1-latency-review.json)
recomputes all 36 timing groups, 204 requests and 13,056 arrival events from
the inventory-verified results. Every event contains exactly one token, so
inter-token intervals require no interpolation. All stored TTFT/decode-rate
fields reproduce, and the 102 paired token sequences still match. Executed
[analysis source](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/matched-c8-v1-latency-analysis.py.txt)
and an [independent raw recalculation](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/matched-c8-v1-latency-review-check.json)
are retained. No new GPU experiment, measurement exclusion or gate change occurs.

| Case | Median first token, stock -> compressed | Median group completion, stock -> compressed | Late-stream mean interval, stock -> compressed |
|---|---:|---:|---:|
| C1 / 256 prompt tokens | 387.56 -> 418.21 ms | 4.283 -> 3.811 s | 61.18 -> 53.10 ms |
| C8 / 256 prompt tokens | 1.164 -> 1.196 s | 8.352 -> 8.157 s | 113.45 -> 109.85 ms |
| C8 / 2,048 prompt tokens | 6.489 -> 6.674 s | 16.113 -> 16.212 s | 103.83 -> 100.44 ms |

Group completion includes all 64 requested output tokens. The last column is
the median across trials of median per-request mean intervals for the final
32 gaps. It is an exploratory window, not an isolated GPU decode measurement
or replacement speed gate. There are six trials per case/arm; tokens and
requests within a trial are dependent.

Every long-prompt trial in both arms has exactly 7/5/4/3/2/1/0/0 gaps over
250 ms when requests are ordered by first-token arrival. Thus the early long
gaps are present in stock as well. The pattern and existing mixed prefill
traces support interference from processing other prompts; compressed prefill
adds reconstruction work. The ~31-32 ms median short-prompt TTFT increases
are consistent in magnitude with ~36 ms of profiled reconstruction per forward.
Repeated prefill work also explains why long-C8 can have slightly faster late
intervals yet a slightly slower whole group. This remains an inference across
different timed/profiled workloads, not a complete causal decomposition.

The candidate **short-C8 repeat 2** is the exception requiring further evidence.
It takes 12.052 s, compared with 8.134-8.190 s in its other five trials. Six
requests enter the first admission group and two enter later. The first six
have TTFT about 2.989 s, the other two about 5.232 s; six streams then have a
gap around 2.14 s. Their late intervals return to normal. Stock split-admission
trials also stall, but the observed delay is much smaller. Endpoint clocks and
temperatures do not establish the cause, and the original timed trial has no
simultaneous GPU/CPU trace. There is no evidence yet assigning it to compilation,
throttling, graph behavior or a particular compression kernel.

Including every trial, mean per-request completion is 8.408 s stock versus
8.787 s compressed at short-C8. The better median must not hide that tail cost.
Next investigate first-use versus repeated 6+2 admission with forward-level
CPU/GPU traces and clock/thermal telemetry in both arms. Actual grouping must
be checked; a requested client schedule alone is insufficient. New format/kernel
experiments are deferred until this anomaly and the running C16 comparison are
understood. The user accepts a modest explained compression cost; the original
speed targets remain recorded as failed rather than being silently relaxed.

### Offline alignment of the stock and compressed KDA calls

The [launch alignment](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/matched-c8-v1-linear-alignment-review.json)
rechecks all four controlled trace hashes and eight selected decode forwards.
Chronological and graph-node order agree; all 240 unchanged BF16 linear
signatures align, and all 68 replacement slots have the expected stock/fused
grids, alternating 34 input and 34 output projections.

| Rank / projection | Stock median kernel sum, ms | Compressed median kernel sum, ms | Median paired saving, ms |
|---|---:|---:|---:|
| 0 / input | 16.323 | 16.567 | -0.186 |
| 0 / output | 7.054 | 7.312 | -0.153 |
| 1 / input | 16.252 | 15.238 | +0.970 |
| 1 / output | 7.161 | 6.673 | +0.472 |

This supports the near-parity C8 diagnosis: these four forwards/rank do not
show a large consistent KDA saving. Projection attribution is inferred from
the checked ordering and geometry; the trace lacks tensor-pointer annotations.
Paired medians are not differences of medians. These are profiled device sums,
not API latency, DRAM bandwidth or a TP critical path.

The initial analyzer assumed logical-stream graph nodes could not overlap.
One control pair overlaps by 2.720 us; the final analysis retains that fact and
requires unique monotonic graph-node ordering instead. Initial source and
failure log remain alongside the executed source. This CPU-only follow-up
changes no original measurements or gates.

The [storage tradeoff](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/matched-c8-v1-storage-tradeoff-review.json)
uses exact per-matrix payload counts: making all output projections dense
would leave about 8.744 GiB estimated saving; adding the first five input
projections would leave 8.467 GiB. Packed copies would be removed. This is
an unimplemented memory estimate, with no measured speed or CPU RSS benefit.

## Historical evidence that motivated the trace investigation

The stable compaction adapter makes stock C1 self-repeat bitwise exact across
three runs on sixteen contexts. Stock C8 still fails KL at 6.80e−5 and 5.33e−5,
against the unchanged 1e−5 gate. Generated IDs match and NLL passes 1e−4.
Every C8 context already differs at its first prefill vocabulary output.

Admission initially collects 3, 1 and 8 requests across the three C8 runs.
Those logs do not prove actual forward shapes or identify a failing kernel.
Earlier mixed-length throughput profiles use separate prefill/decode attention
paths and confirm graph replay, but are not traces of the failed quality groups.


## Completed C8 trace investigation, 18 October 15:43 UTC

**c8tracev2 is complete and independently verified on Spark/Mac.**
All 12 quality groups, eight strict self-comparisons and six profiler sessions
finish; 12 worker traces cover the recorded forwards. Exact original
identities/configs/mounts and health 200 are restored. No GPU experiment is active.

**Controlled C8 repeats are bitwise exact**, both unprofiled and profiled:
all 16 prefill / 48 decode full-vocabulary rows match in all four comparisons;
KL = 0 and NLL difference = 0. Actual forward shapes and prompt order match.
Ordinary unprofiled repeats fail KL at 3.21805e−5 / 5.36556e−5; profiled repeats
fail at 4.59890e−5 / 7.00604e−5. All IDs/conditioning match and NLL passes.
Ordinary schedules and row membership differ. No gate is relaxed.

The complete archive has **1,025 payloads / 190 source bindings / 12 traces**,
288,213,015 bytes; SHA256
37f514033e9b39b32b14b24aae6a74aa37270cc05c36a8379b784d3f9b47672a.
Spark/Mac decisions match, with maximum CPU float64 difference 2.2175e−15.
CPU diagnostic tolerances remain separate from the model's KL ≤1e−5 /
absolute mean NLL ≤1e−4 gates. Every prior failure is retained.
Local evidence: /private/tmp/mxwave-c8-trace-v2/evidence/.
Remote terminal receipt: c8-trace-v2-evidence-check.json.

This establishes repeatability under matched C8 grouping. It does not fix
arbitrary continuous-serving batch invariance or qualify compression.
The remaining issue is strongly associated with batch layout. Most drift
(90–98% in the two ordinary unprofiled comparisons) comes from prompts with
unchanged own chunk boundaries; chunking alone is insufficient.

Device traces confirm different GEMM variants/grids and mHC paths across
batch layouts. Native mHC changes prenorm split counts, and fused post/pre
switches from small-row FMA at <=16 rows to DeepGEMM on larger/mixed steps.
Those are candidates, not a proven first numerical divergence. The next
targeted check should hold per-request mHC operands constant while varying the
observed batch shapes, quantify all output differences and timing, and test a
consistent native reduction path in an isolated fixture before model integration.

The fresh C1 stock/compressed comparison remains prepared locally and unlaunched.
The exact16 input/output integration remains unmeasured in a model. MTP remains
deferred and no configuration is promoted.

## Frozen diagnostic

One stock model load with the existing stable-compaction policy, pinned native
sources/libraries, BF16 execution, fixed transport, 2048-token scheduling budget,
capacity sixteen and captures [1,2,4,8,16]. All 186 parent sources remain
unchanged; four authored helpers provide CPU metadata and experiment orchestration.

Run three ordinary unprofiled repeats, then three controlled-admission unprofiled
repeats. Only then profile three repeats in each mode: twelve quality groups,
eight self-comparisons and six profiler sessions across both ranks.

All sixteen original 258-token prompts and four actual generated tokens remain
unchanged. Eight prompts total 2064 tokens, so native chunking remains necessary.
The diagnostic queues eight prompts under the existing keep-pause API on the
disposable service, then resumes in a finally block. Actual scheduler membership
and prepared row order prove whether it achieved a fixed cohort.

The observer copies only existing CPU metadata: actual scheduled tokens, prompt
hashes, prepared row order, padding and the selected graph descriptor. Per-step
profiler annotations allow correlation-based device-launch attribution.
No GPU tensor copying or arithmetic change occurs. Extra CPU observer work
and profiler effects remain explicit; this is not a throughput benchmark.

## Analysis contract

Compare original unprofiled and profiled KL/NLL, IDs, conditioning and raw
full-vocabulary hashes. Normalize transient request IDs/state slots while
preserving actual prompt identity, row order and execution shapes.
For each model forward, retain kernel names, grid/block dimensions, shared
memory, graph launches, and variant/count signatures. Keep unassigned kernels
explicit. Compare ordinary versus controlled batches and profiling effects
before choosing a runtime change.

A change in kernel variants alone does not locate the first numerical
divergence. If fixed batches still drift, the next measurement must capture
operands/results around the implicated operations. A controlled-cohort pass
would qualify this diagnostic condition, not arbitrary continuous serving.
The current C8 gate and every historical failure remain retained.

## First attempt and repair

c8tracev1 failed before its first forward when the observer converted the valid
native max_query_len=None descriptor field to int. The original service was
exactly restored, and all 212 payloads / 190 bindings independently verify on
Spark/Mac. No profile or numerical result was produced.
Archive SHA256: cb3466659af83151dadb81d02089d2321096c03670a0c9b2eadc02efae065c5d.

The corrected observer preserves optional fields. Four contract tests and a
CPU regression against the installed native eager/full descriptors pass,
with no CUDA initialization or model load. Original frozen sources remain
unchanged; c8tracev2 has a new source archive, plan and receipts.

MTP and the fresh C1 stock/compressed comparison remain deferred for this
investigation. No configuration is promoted.
