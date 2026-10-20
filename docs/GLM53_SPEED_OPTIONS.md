# GLM speed options after lossless storage compression

**Reference-repeat follow-up complete, 19 October 2026, 20:47 UTC.** The unchanged production
ModelOpt stack first differs at layer 3's MoE expert output with identical
inputs, actual gate logits and shared-expert output: 4/1,048,576 BF16 values
change on each rank, with final raw-logit KL 0.00537795. Raw logits agree with
API distributions. A different pair first changes at layer 4; the internal
routing/GEMM/reduction cause and fix remain unresolved. Full-model quality
stays unqualified at 1e-5; no speed or compression gate is relaxed. All 4,290
payloads verify independently. Exact prior CT/DFlash service is restored healthy
with 8,192 context; all diagnostic containers are stopped. See the
[complete diagnostic and next test](GLM53_NVFP4_REPEAT.md).

**NVFP4 full-model follow-up complete, 19 October 2026, 20:04 UTC.** The original production
checkpoint's lossless expert-scale adapter saves **3.1639 / 3.1665 GiB/GPU**
after graph capture, with no scale rebuild in retained request traces. Observed
short C1/C8/C16 rates are 38.77 / 73.62 / 73.93 tok/s versus stock B's
35.92 / 69.97 / 67.81, but all four timing controls fail 2%; speed is unqualified.
Stock and compressed C1/C8 self-repeat KL also fail 1e-5; zero model loss is
unqualified. All 154 payloads verify independently. The exact prior CT/DFlash
service is restored healthy with 8,192 context; no NVFP4 adapter was deployed.
See [complete results and next diagnostic](GLM53_NVFP4_MODEL_TEST.md).
Earlier dated checkpoints below remain historical.

**NVFP4 component follow-up complete, 19 October 2026, 19:03 UTC.** The original
production checkpoint's expert scales now have a lossless consumer: full-layer
M=8 time is 1.90% / 3.07% lower than installed stock, with bitwise-exact covered
outputs and passing repeat controls. Full-checkpoint encoded-storage accounting
saves 3.250086 / 3.252740 GiB per TP rank. Whole-model VRAM, KL, TTFT and serving
speed remain unmeasured for this port. This extends compression while retaining
production's NVFP4/MXFP8 checkpoint; it does not convert that checkpoint to MXFP4.
No serving adapter is installed. All experiments are terminal; live CT/DFlash
service is unchanged, healthy and limited to 8,192 context. Next: ModelOpt
adapter and matched full-model qualification. See the [component result](GLM53_NVFP4_COMPONENT.md).
Earlier dated checkpoints below remain historical.

**ModelOpt CPU scale screen complete, 19 October 2026, 17:59 UTC.**
Actual NVFP4 expert scales total 17.71875 GiB. Palette layouts estimate
3.25–3.30 GiB saved per TP2 rank, removing about 4.1% of expert weight-plus-scale
bytes. This supports a bounded post-Marlin component prototype while retaining
production's checkpoint and MXFP8 KDA; no GPU speed/quality gain is established.
The vocabulary port is less direct than initially estimated because production's
certified head retains dense BF16 weights for refinement. Serving is unchanged.
See [inventory, entropy, compatibility and evidence](GLM53_MODELOPT_SCALE_AUDIT.md).

**C1 follow-up complete, 19 October 2026, 17:27 UTC.** A three-arm DFlash
comparison isolates compression: CT off/on is essentially tied end-to-end at
64 output tokens, with faster decode and 32.7–52.3 ms additional TTFT.
Most short-C1 replacement regression exists before compression. Traces find zero
C1 decode rebuilds; prefill still reconstructs 68 matrices/rank. Leaving KDA
dense while retaining expert/vocabulary compression is an untested option that
could retain about 6.85–7.23 GiB TP2 saving and remove those rebuilds.
The exact candidate is restored healthy, still with 8,192 context. Live vLLM
reports `0.1.dev20051+g487ecf187`. See [full evidence and limits](GLM53_DFLASH_C1.md).

<!-- SERVING_TRIAL_STATUS_BEGIN -->
**Complete and independently verified, 19 October 2026, 15:21 UTC: compressed candidate is LIVE with DFlash.**
Same authenticated Spark listener `127.0.0.1:8000`, alias `glm-5.3-flash`.
Trial settings: **8,192 context**, 16 sequence slots, 2,048 prefill budget,
4 GiB KV per rank, 10 ms admission; original containers are retained stopped.
The user's harness can use the candidate now. Final auth and streaming checks pass.

Matched current → candidate short C1/C8/C16 throughput:
**48.399 → 45.754 / 74.383 → 75.704 / 74.885 → 75.512 tok/s**
(−5.46% / +1.78% / +0.84%). Long-prompt gains are +4.39% / +7.75% / +8.45%.
The unchanged tuned production reference is faster in all six cells; no
production speed upgrade is demonstrated. Common C16 logs include four running
and twelve waiting requests, so client latency includes substantial queueing.

Post-startup CUDA allocation is **12.690092 GiB lower across TP2** versus the
current checkpoint with equal KV budgets. This includes checkpoint differences;
the earlier isolated compression saving remains 9.367162 GiB.
All 150 timed output sequences match both references. **Full speculative-stack
KL, broad quality and long context remain unqualified.** Repeated diagnostic
prose yields ~99.8% candidate draft acceptance; harness behavior may differ.

All 37 evidence payloads and 11,477 numerical fields independently verify on
Spark/Mac with zero difference. V1's missing FP8-draft compatibility flag,
failed startup and successful rollback remain retained; v2 uses the existing
adapter in both arms, with CPU/source checks. Controller/watchdog completed;
candidate serving containers remain running. No new compression kernel or MTP
work was started. See [full tables, evidence and rollback](GLM53_SERVING_TRIAL.md).
<!-- SERVING_TRIAL_STATUS_END -->

**Native expert follow-up complete, 19 October 2026, 14:31 UTC.** The tested native path
takes 57.28% longer at C8 and 41.81% longer at C16, with about 4.1% component
relative L2 error and non-bitwise repeats. Full-model native KL/memory remain
unqualified. Traces show both scale-reconstruction overhead and slower expert
kernels; the 15% layer-time gate fails. Original service is restored, all jobs
are terminal and the owned lease is released. No integration or configuration
promotion. The bounded expert/latency investigation is closed; retain the
qualified warmed exact16 candidate. See [verified native result](GLM53_MOE_BACKENDS.md).

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

**19 October 2026, 05:36 UTC:** the fresh stock versus input-only FP32 model comparison
is launched (`fp32inputmodelv1`, controller 1258389). Both CPU
serving preflights pass with 229 frozen sources; 781 tests / one GPU skip, Ruff,
strict mypy and the full CPU protocol simulation pass. It measures C1/C8/C16
quality, ordinary throughput/TTFT, CPU/CUDA memory and actual kernel traces with
all original gates retained. No new model result is claimed. Verified relocation
of completed evidence provides the required disk reserve; model inputs and
canonical archives are unchanged. See the [handoff](GLM53_HANDOFF.md).

**18 October 2026, 22:50 UTC integration follow-up:** input-only FP32 split 2 passes
the independently verified GPU integration fixture (**434 payloads / 227
sources**): 120 retained-output checks, 66 native-fallback checks, compiled
captures and six two-stream stress fixtures. The existing pool is reused with
zero retained replay growth; a third stream is rejected. Both GPU jobs, retention
and exact original-service restoration are complete; the awake lease is released.
This adds no full-model quality/speed or CPU-saving result. The 216 prior exact-word
failures and two input timing repeat failures remain failed. Fresh matched model
qualification is next, after evidence-storage preparation. See the [handoff](GLM53_HANDOFF.md).

**18 October 2026, 22:31 UTC speed follow-up:** FP32 timing independently verifies on
Spark/Mac (**246 payloads / 220 sources / 884,736 events**). Input split 2 saves
about **7-8% of component time** (46/48 strict timing conditions pass; two M8
pressure repeat controls fail). Output gains shrink substantially under pressure;
the broad screen fails 78/288 conditions. Prior 216 exact-word failures remain
failed. No new full-model quality/speed or CPU-saving claim is made. GLM is
restored healthy, all jobs/retention complete and the owned awake lease released.
Next is a bounded input-only split-2 model probe under unchanged KL/NLL limits.
See the [handoff](GLM53_HANDOFF.md) for measurements, controls and provenance.

**18 October 22:12 UTC precision follow-up:** FP32 split partials are independently
verified on Spark/Mac (2,227 payloads / 217 sources). All 576 comparisons pass
the 0.001 relative-error bound; all 144 basis cases are exact. However, all 216
random M4/M8 cases fail the retained exact-word gate, despite about 22-26 times
lower relative error than the old BF16-partial route. Source/storage, guards,
repeats and graph checks pass. No speed, model-KL or CPU-saving result is measured,
and no route is promoted. GLM is restored healthy; all jobs and retention are
complete. Next comes paired component timing, then any justified exploratory
model test under the unchanged KL/NLL limits. See the [handoff](GLM53_HANDOFF.md).

**18 October 21:52 UTC follow-up:** dense-output replacement is rejected by its
frozen component screen. Spark/Mac independently verify **270 payloads / 213
sources / 221,184 timing samples**. All six matrices pass storage/execution and
numerical checks, but warm-cache C1 adds **56-68 us per projection**: 12 stable
cost failures plus six separately unstable conditions, all retained. Prefill
saves about **205-284 us/projection** in graph + pressure measurements; these
are not serving-speed results. No dense-only policy is integrated. The warmed
exact16 candidate below remains current. GLM is restored healthy, all jobs and
retention are complete, and the owned awake lease is released. An FP32 split-K
partial-storage idea is source-audited only; no new kernel result exists.
See the [handoff](GLM53_HANDOFF.md) for evidence and limitations.

**Verified full-model result, 18 October 21:10 UTC:** startup warming works in
the measured workload. All **1,612 payloads / 207 sources / 12 worker traces**
independently verify on Spark and Mac. All **46 numerical comparisons** pass at
the unchanged KL <=1e-5 / absolute mean NLL <=1e-4 limits; all **300 paired timed
output sequences** match. Native mHC and matched quality grouping remain in both
arms. This is finite numerical qualification, not universal bitwise equivalence.

| Workload, 64 output tokens | Stock tok/s | Compressed tok/s | Change | Median TTFT, stock -> compressed |
|---|---:|---:|---:|---:|
| C1 / 256 prompt tokens | 14.784 | 16.748 | +13.28% | 0.384 -> 0.422 s |
| C1 / 2,048 prompt tokens | 12.518 | 13.999 | +11.82% | 1.183 -> 1.205 s |
| C8 / 256 prompt tokens | 60.757 | 62.564 | +2.97% | 1.169 -> 1.192 s |
| C8 / 2,048 prompt tokens | 31.488 | 31.530 | +0.13% | 6.586 -> 6.662 s |
| C16 / 256 prompt tokens | 80.870 | 81.108 | +0.29% | 1.838 -> 2.280 s |
| C16 / 2,048 prompt tokens | 36.628 | 36.477 | -0.41% | 11.395 -> 11.594 s |

Six trials/case/arm, 256/2,048 prompt tokens and 64 output tokens, two fresh loads
in fixed stock-then-compressed order. **The original overall speed gate still
fails**: C8/C16 miss the 5% throughput target, and short-C1/C16 exceed the TTFT
allowance. All trials, admission differences and historical failures are retained.
CUDA allocation is **9.367162 GiB lower across TP2** at every measured stage.
CPU process RSS is **158.78 MiB higher before quality / 166.83 MiB higher after
ordinary timing / 173.72 MiB higher after profiles** across TP2. These overlapping
Spark counters must not be added. No CPU-memory improvement is demonstrated.

Warmup is now validated in both loaded model arms: unchanged dispatch caches
through the workload, no recognized compiler event during timed requests, and
576 before/after full-vocabulary rows per arm bitwise equal. The earlier 2.14 s
C8 gap does not recur. Startup takes 31.3-31.6 s; no CPU-memory gain results.
Exact16 decode replays graphs with zero reconstruction. Prefill retains about
36-38 ms/rank of reconstruction, while expert GEMMs dominate concurrent decode.
The next speed target is that measured prefill/admission cost under controlled
grouping. Original service is restored healthy; all evidence and analyses are
complete. See the [handoff](GLM53_HANDOFF.md) for recovery provenance and limits.

**18 October 19:38 UTC:** the focused replay reproduces the multi-second stalls
in both reference and compressed arms. Closed head traces and cache timestamps
identify first-use compilation of two specializations in the shared deterministic
expert-order wrapper; those GPU gaps fall below 1 ms on repetition. All 30 groups
finish, with all requested splits and 106 paired token sequences confirmed.
All **271 payloads / 200 source bindings / four worker traces** independently
verify on Spark/Mac. Original service identities/configs/mounts and health 200
are restored; experiment processes have exited. Both head traces/results bind
to this archive, while cache audits remain separate supplements.
A complete 34-variant startup warmup is CPU-prepared with unchanged
kernels; GPU intervention and its startup/memory/serving costs remain unmeasured.
See the [current handoff](GLM53_HANDOFF.md).

**18 October 19:08 UTC:** exact16 C16 is complete and independently verified on
Spark/Mac (1,581 payloads / 200 sources / eight traces), with the original
service restored. All 46 C1/C8/C16 numerical comparisons pass. C16 throughput
is +0.48% short / -1.40% long, while TTFT is +36.22% short / +2.98% long.
The original speed gate fails. Candidate physical-16 decode replays graphs
with zero KDA reconstruction, so the fallback is fixed; first-token latency
still needs analysis. CUDA allocation saving remains 9.367 GiB/TP2.
The user accepts an understood compression cost. The
[arrival audit](GLM53_C8_TRACE.md#latency-cost-and-the-remaining-outlier) supports
expected prompt-processing overhead as the main routine latency cost, but a
12.052 s short-C8 trial has an unexplained ~2.14 s streaming gap after split
admission. Investigating that tail cost takes priority over new kernel layouts;
the original failed speed gates and all outliers remain retained.
The [new C8 launch alignment](GLM53_C8_TRACE.md#offline-alignment-of-the-stock-and-compressed-kda-calls)
finds only small, rank-dependent KDA savings in the retained controlled traces.
A CPU-only fixed-mantissa layout oracle now preserves all 65,536 BF16 bit
patterns and does not increase aligned tile payload bytes at any exception
count. Its GPU implementation and speed are unmeasured. A separate synthetic
allocator probe passes; reclaimable model CPU memory is still unmeasured.
See the [handoff](GLM53_HANDOFF.md#cpu-only-follow-up-prepared-while-c16-runs)
for the exact evidence and next component gates. No runtime policy changed.

**Current checkpoint, 18 October 17:44 UTC:** the fresh matched native-mHC
C1/C8 pair passes all 33 numerical comparisons at the original limits, verified
on Spark/Mac (1,168 payloads / 194 sources / eight traces). C1 throughput +12.38%,
short-C8 +2.40%, long-C8 -0.61%; C1 TTFT +7.91%, so the overall speed gate fails.
CUDA allocation saving is 9.367 GiB across TP2. Original GLM is restored healthy.
Full-model exact16 input/output integration remains pending; MTP stays deferred.
See the [current handoff](GLM53_HANDOFF.md). Dated records below retain history.


**18 October 14:06 UTC operational update:** Work resumed at the user's request.
V8 qualifies the exact16 input component with zero arithmetic error. Stable
compaction gives exact stock C1 self-repeat; stock C8 still fails KL and differs
already at prefill. The independently verified parent result is retained, and
the targeted c8tracev1 workload is loading under a restoration watchdog.
Fresh stock/compressed quality and the full exact16 integration remain unmeasured.
Read [GLM53_HANDOFF.md](GLM53_HANDOFF.md) for live state; dated records below
preserve earlier failures and stopped-state snapshots. MTP remains deferred.

**18 October12:30 UTC: user resumed checks.** The actual MLA internal observer
is CPU prepared and loading one raw-reference model under independent restoration.
See the [current handoff](GLM53_HANDOFF.md). MTP remains deferred.

**Compression checkpoint18 October12:18 UTC:**
The [output-only continuation](GLM53_OUTPUT_ROW16.md) is independently verified:
34 rather than68 reconstructions per retained C16 graph range, partial API
recovery, both speed gates still failed and all24 KL comparisons failed.
Original service is restored and healthy. Resume from the
[current handoff](GLM53_HANDOFF.md); MTP stays deferred. Earlier assessments below
retain their original dates.

**18 October 10:37 UTC:** MTP remains deferred. The
[matched C8/C16 compression comparison](GLM53_HIGH_CONCURRENCY.md) is complete
and independently verified after repairing startup scratch ownership.
Memory saving is9.367 GiB/TP2; C1 throughput gains12.3–12.6%, C8 is near parity,
and C16 loses7.5–13.8%. Both speed gates and all24 KL comparisons fail; four
NLL comparisons also fail. Stock evidence is reused with an explicit time gap.
Both-rank traces identify68 reconstructions per retained C16 decode forward,
taking34–37 ms. The focused row16 V7 component is independently verified on
Spark/Mac, all45 payloads/18,432 events. The original service is restored healthy.
Both fixture outputs are exact, but only the
output matrix passes every unchanged timing gate; the larger input matrix's
repeat control fails. No shared row16 route or model speed improvement is
qualified. An output-only exact16 runtime follow-up is locally prepared for
full-model measurement; input reconstruction and the memory pool are unchanged.

Assessment, **18 October 2026**. Native MTP remains an unmeasured speed
hypothesis. Exact C2 graphs and bounded idle admission now have measured
serving gains. MTP execution shapes still need observation: verification
batches above eight rows reconstruct KDA matrices, and the completed routing
sweeps do not qualify a blanket crossover. Further expert-code compression
and standalone decoder occupancy changes have weak supporting evidence.

The [admission intervention](GLM53_PREFILL_ADMISSION.md) verifies all 108 groups /
450 requests / 28,800 tokens and 599 evidence payloads. Three/ten milliseconds
raises short-C4 TPS 6.75%/6.62%, TTFT 1.125 → 0.778/0.782 s; profiles and device
traces agree on 136 → 68 reconstructions/rank. Both frozen performance/mechanism
gates pass. Every policy still fails repeat KL, including zero wait. This is
finite 64-output request throughput; continuous traffic and sustained decode
were not tested. Original service is restored healthy at 07:49:50 UTC.

The [C2/C8 graph comparison](GLM53_GRAPH_EXPANSION.md) is now complete and
independently verified: exact C2 gains 8.6–9.0% TPS; graph C8 gains 1.7–3.4%.
The bundle fails the frozen speed/latency guard, and both policies fail KL
repeats despite matching generated IDs. Qualify C2 separately before promotion;
a combined graph/admission policy has not been measured. The separate matched
C8/C16 run now measures both policies on stock and compressed targets with
captures through16; it fails the compression speed gates. The MLA numerical
cause remains open.

## What is already measured

- [Paired serving runs](GLM53_PAIRED_SPEED.md): 9.367 GiB saved across TP2,
  C1 throughput +10.4–12.4%, C4/C8 near parity, first-token latency +2.3–5.9%.
  The experiment already uses FP8 KV and has no speculative drafter.
- [Expert-code entropy screen](GLM53_BF16_KERNEL_DIAGNOSTIC.md#paper-follow-up-and-actual-mxfp4-compressibility):
  27 stratified expert matrices, 3.915/4 bits per code. One rANS stream per
  tile saves 1.65%; 32 independent streams lose 0.30% including overhead.
  The approximately 2.1% zero-order entropy gap is not an exact bound for
  correlated coding or the whole checkpoint. No new entropy script is needed
  to establish that this particular coding direction has little headroom.
- [V3 reconstruction test](GLM53_PREFILL_RECONSTRUCTION.md): exact bytes,
  better occupancy capacity, no accepted speed gain. Logical packed reads
  plus padded BF16 writes give approximately 204–210 GB/s for qualified warm
  decoders. These are payload rates, not hardware-measured LPDDR traffic;
  a matched copy baseline is still absent.
- Whole-checkpoint base scan: all 136 actual KDA TP weight hashes agree with
  the preceding loaded weights; two compressed heads and both embedding
  base arrays also pass. No unsafe ZipServ base or embedding row was found.
  This defect does not explain current serving latency or the MLA divergence.

## Native MTP: strongest hypothesis, unmeasured gain

The imported checkpoint declares one next-token prediction layer and contains
1,756 tensor entries under layer 45. Native MTP is explicitly enabled in
the [official vLLM GLM recipe](https://recipes.vllm.ai/zai-org/GLM-5.3-Flash).
The pinned runtime's speculative configuration recognizes `glm5_next_mtp`.
These establish eligibility for investigation, not a working qualified launch
with our compressed-tensors adapter and Spark TP2 configuration.

A CPU-only inspection of the installed `vllm/models/glm5next` source also
finds the native `Glm5NextMTP` implementation and KDA speculative metadata
paths using `num_accepted_tokens` for state rollback. Those hooks are present
in the pinned image/overlays; their correctness with this CT checkpoint and
our adapters still needs execution testing. The source hashes and inspection
are retained in `ROOT/mtp-eligibility-v1/`. The retry enables a drafter and loads
it, but fails untimed validation before the first measured client request.

The [complete native CPU audit](GLM53_MTP_PILOT.md) now checks all 872 draft
selectors and 1,756 source tensor headers. Before namespace/alias repair,
870 selectors differ from source policy; after repair, zero do. Native
whole-module vocabulary sharing passes with a compressed target fixture.
The newer runner additionally needs an independent source CT config; the first
GPU attempt exposed reuse of already-mapped target metadata. That loader repair
now reaches readiness. The corrected observer's actual-zero-rope CPU preflight
also passes: both complete q/KV source matrices survive native binding and the
bridge, with no padding. All fourteen payloads verify on Spark/Mac. Only two
projection comparisons run here; the next live worker needs eighteen complete
checks. The full-pilot validator/preparer is now repaired, frozen and passes
native CPU preflight; the old nineteen-check/64-padding branch remains historical.
MTP is explicitly deferred. Actual GPU draft byte comparisons,
state rollback, acceptance and speed remain unverified. See the
[pilot state](GLM53_MTP_PILOT.md).

The current KDA operation computes `tokens = x.numel() // k` and uses the
fused compressed GEMM only up to eight rows. Larger calls reconstruct BF16
weights. For example, four requests verifying three positions can produce
twelve rows before any runtime padding. The actual execution shapes must be
recorded: speculation may otherwise reintroduce the reconstruction cost on
every verification step. The MTP layer and draft loading also sit outside the
base-layer adapter qualification already completed.

Do not budget a 1.5–2× gain. Measure emitted tokens per verification cycle,
acceptance by draft position, target verification time, draft time, scheduler
overhead and added memory. Compare no drafter against depths 1/2/3 at C1,
then C4/C8 with identical prompts and output limits. Retain ordinary API
throughput and latency, plus actual KDA operation row counts and rebuild
counts. Draft acceptance alone is not the speed result.
For this MoE model, extra verification positions can also touch additional
experts; a three-position check need not have the memory cost of one token.

Correct speculative rejection sampling preserves the target distribution in
the algorithm. Hardware precision and changed batch shapes can still alter
actual vLLM outputs, as its
[losslessness documentation](https://docs.vllm.ai/en/v0.18.0/features/speculative_decoding/#lossless-guarantees-of-speculative-decoding)
explains. Keep reference-repeat controls and full numerical qualification;
MTP does not resolve the open MLA cause or prove backend bit equality.

## Other options and corrections

| Proposal | Assessment for this experiment |
|---|---|
| Faster compressed KDA decode | Useful if measured; another 10% of whole-model speed is not established by a bandwidth estimate. Count its share of the critical path and retain existing numerical gates. |
| Small-prefill routing | V5/V6 and focused V7 retain numerical passes and large small-row component gains, but no shared row16 route passes all repeat controls. V7's output matrix alone passes at16. Keep the eight-row serving limit pending a separately qualified change;257 remains a measured regression. |
| Admission merging | Measured intervention passes performance/mechanism gates: short C4 +6.75%/+6.62% at 3/10 ms, two → one median large forwards. Both-rank profiles visit each weight once/forward; device counts 136 → 68. All policies fail KL repeats, so no promotion. Continuous active traffic and matched stock/compressed policies remain untested. |
| Rank communication imbalance | Earlier paired collective-interval analysis already demonstrated peer-arrival waiting; the subsequent proxy-pinning component failed its primary speed gates. Longer spans are not automatically recoverable transfer time. Recheck the current step's critical path before assigning a slow rank or retrying a rejected intervention. See the [synchronization evidence](GLM53_LOSSLESS_RUNTIME_RESULTS.md#synchronization-follow-up-arrival-skew-confirmed). |
| FP8 KDA weights | Separate lossy experiment. Actual loaded records contain 4,643,618,816 raw bytes per GPU and 3,298,566,000 / 3,297,978,432 packed bytes on ranks 0/1. Hypothetical FP8 is 2,321,809,408 bytes per GPU before scales: about 0.977 / 0.976 GB incremental saving, 1.953 GB across TP2 before scales. These are decimal GB. |
| Native FP8 GEMM | A backend and activation-format decision as well as weight storage. FP8 weights alone do not establish a supported fast FP8 compute path; activation quantization may add another quality change. A supported direct path could remove explicit BF16 reconstruction. |
| FP8 head / embeddings | Lower priority lossy work. Measure their contribution and quality impact separately; the existing lossless representations already shrink them. |
| FP8 KV cache | Already enabled; no new gain available from simply turning it on. Two-times capacity is a KV-only ideal, not a two-times serving-context guarantee with other memory and hybrid state. |

The decoder mask/guard component and complete base scan are independently
verified. The [completed routing sweep](GLM53_PREFILL_ROUTING.md) confirms
large small-row gains and large-prefill regressions; only 8/24 rows pass the
full bracketed component gate on both matrices. It does not qualify a broad
threshold. Original service is healthy. Capture actual MLA internal operands
and freeze MTP's full-pilot validator after the passing actual-geometry CPU proof,
then test C1 with the current eight-row limit. Qualify admission under broader
traffic after numerical work. Keep lossy FP8
changes behind the existing numerical gates and paired reference-repeat
controls. Overlap and selective BF16 caching remain later options.
