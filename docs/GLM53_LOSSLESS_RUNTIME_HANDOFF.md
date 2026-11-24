# Agent handoff: GLM lossless runtime storage

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

**ModelOpt CPU audit complete, 19 October 2026, 17:59 UTC.** The actual lossless8
NVFP4 expert scales contain 17.71875 GiB; palette layouts estimate 3.25–3.30 GiB
saved per TP2 rank, about 4.1% of expert weight-plus-scale bytes. All 36,288 scale
tensors were scanned; 1,008 sampled block-codec roundtrips are exact. A bounded
post-Marlin component prototype is justified, but GPU speed and full-model KL
are unmeasured. Production's certified head retains dense BF16 weights, so a
vocabulary port needs explicit refinement/draft support beyond loader changes.
No ModelOpt adapter or GPU experiment started; the existing 8,192-context CT
service remains unchanged and healthy. See the
[complete audit and evidence](GLM53_MODELOPT_SCALE_AUDIT.md).

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

**Current checkpoint, 18 October 21:10 UTC:** warmedmodelv1 and all final
analyses independently verify on Spark/Mac (1,612 payloads / 207 sources / 12
worker traces). All 46 numerical comparisons pass; all 300 paired timed output
sequences match. Startup warmup preserves all recorded outputs and dispatch
caches. CUDA saving is 9.367 GiB/TP2; CPU RSS is about 159-174 MiB higher.
C1 throughput +11.82-13.28%, C8 +0.13-2.97%, C16 -0.41 to +0.29%. Short-C1/C16
TTFT penalties remain and the original overall speed gate fails. Original GLM
is restored healthy; no experiment is running or configuration promoted.
See [current handoff](GLM53_HANDOFF.md) for the verified result and preserved
offline-verifier recovery. MTP stays deferred.

**Historical checkpoint, 18 October 17:44 UTC:** the fresh matched native-mHC
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

**Latest operational handoff:** [GLM53_HANDOFF.md](GLM53_HANDOFF.md).
The records below retain earlier snapshots and must not be read as live state.

**Final checkpoint18 October12:18 UTC: paused at user request.**
output16qualv1 completes and verifies1,293 payloads/180 bindings/16 traces on
Spark/Mac. Actual retained C16 graph ranges reconstruct34 matrices rather than68;
speed gates still fail, all24 KL comparisons fail,8 NLL failures. Original
service exact and healthy since12:14:05 UTC; no experiment containers running.
Archive d871cbac8d1d91332469c01434ce72549679974d8e1653d0198642fbf8c44dba.
No further experiments or cache cleanup started; MTP remains deferred.

[Current state, verified findings and next steps](GLM53_CURRENT_STATE.md)
provides the short entry point; the dated records below retain full history.

**CURRENT continuation: recovery verified; validation-only CPU preparation passes.**
Recovery collector3389440 is complete. Archive683fcb4b... (304,447,995bytes /
1,532payloads) verifies on Spark/Mac; all178 source bindings match. CPU timing
supplement independently verifies72 groups/636 requests/40,704tokens and17
available complete numerical comparisons. All17 fail; all636 timed sequences
match. Original source-after is missing, profiles0/4. Do not promote the result.

Storage recovery is complete:8,950 older generated payloads/archives totaling
7,593,681,896bytes backed up under worker
ROOT/storage-recovery-v1/offloaded-payloads/<original-relative-path>, then exact
head copies removed only after all hashes verified. Head-side offload-plan.json,
offload-backup-receipt.json and offload-cleanup-receipt.json retain every path,
hash and byte count. Manifest SHA5bdd74f0ef05cd297f8081484f99dad2e3f1a9e1f8ee778bf4796e2590a96c32.
Head free about7.7GB after retaining recovery archive. No models/checkpoints
or containers/images removed. Old raw payloads may now require the worker copy.

`output16qualv1` CPU preparation3474616 passes:180 bindings match on both nodes;
all178 output16 parent sources unchanged. Two validation helpers only, zero new
timing groups;8fresh quality groups,4short profiles/rank. Source archive
b6c3ef4a09b666c49bfe63c8f8dcea68d7abadc518ed71bbbea1cb8eb0e4883f;
prepared35039b55d34148b2544d1ca6a120219e1d4289e13b479dde2712a9c524456f1b;
plan cf68aa8a0630a82592f1b15da59e1965ab80807202aa5fc1bd74bf267721375b.
Do not overwrite first failed run or rerun72 timings. Fresh checks/profiles are
a separate model load; missing original source-after stays explicit. Launch
requires6GiB/node and each collection stage2GiB. Independent collector/verifier
must be frozen before GPU launch. Finish these compression profiles before MLA.
MLA frozen162-source observer remains unuploaded/unrun; needs extra SSD budget
before its136MiB/rank private-workspace replay images. MTP remains deferred.

Laptop caffeinate50382 is active until15:45:33UTC; renew before expiry.

**Historical11:24 UTC: output16 fails from head DISK FULL; exact originals restored.**
No GPU experiment is active. Controller2930080 is failed_restored, originals
exact/health200 at11:16:07 UTC. Client exits1, OOMKilledfalse. Combined
stdout/stderr reveals OSError28 writing candidate-c16-w0-b distribution data;
even saving failed measurement header hits ENOSPC. Thus main report remains
running at quality5/8 while controller is terminal. All72 timing groups intact;
no candidate profiles run. Do not claim completed validation or kernel failure.

First collector2930079 has died; receipt output-row16-v1-evidence-check.json
remains collecting because ENOSPC prevents its error write. Preserve it and
any partial extraction; use a new recovery receipt/base/archive for resumed
retention, not overwrite the first evidence attempt. Frozen verifier supports
failed-restored retention. A separate CPU supplement must validate complete72
timings and every available complete quality report despite missing full-model
review/source-after/profiles. Retain partial/truncated files explicitly.
Do not rerun72 timings merely to recover missing validation/profiles.

Head filesystem917GiB, unprivileged free0 initially; worker has1.2TB free;
Mac /private/tmp has509GB free. Containers occupy only1.4GB head, so no prune
or container removal is useful. Main experiment205GB includes183GB source
checkpoint. Original service mounts read-only a DIFFERENT33-shard lossless8
checkpoint; audited source has120 shards. Do not deduplicate or delete models.
Global model/cache dirs untouched.

Safely freed1,006,391,296 bytes by removing ONLY two regenerable extracted
directories after verifying every inventory entry, canonical archives on
head/worker and matching Mac copies. Removed high-concurrency-retry-v2-evidence
and high-concurrency-retry-v1-evidence; canonical archive SHA602e.../bb55... remain
on head and Mac, and added worker backup under
ROOT/storage-recovery-v1/verified-archives/. Full source, results and controllers
untouched. Receipt ROOT/storage-recovery-v1/regenerable-extractions.json.
No need restore those directories for current continuation; re-extract canonical
archives if needed later. Keep high-concurrency-v1-evidence intact because the
current collector copies its retained stock traces.

Next storage action: back up only old experiment-generated f32/raw-value/
profiler payloads and older large evidence archives to worker, verify every
checksum before removing head copies, keep metadata and current C816/output
sources/results/archives protected. Preview candidates first. Worker has ample
space. Data relocation is reversible; no user/model/checkpoint deletion.
Need enough room for retention immediately, and at least a measured diagnostic
budget before any next model pause. Add disk preflight to admin prepare/launch.
After full partial-run retention, likely run validation/profiles-only with
unchanged decoder and reused72 timing groups, explicitly labeling the new model
load/time gap. Finish compression profiles before the local MLA GPU diagnostic.

**11:11 UTC continuation:** output16v1 timing72/72 complete; last numerical
check2/8 groups, profiles0/4. Original service still paused under2930080 /
watchdog2932768. No error. Completed timing medians show only partial recovery:
shortC16+3.355%/+3.956% vs compressed parent but−10.899%/−10.202% vs stock;
longC16+1.519%/+1.304% vs parent but−6.232%/−6.305% vs stock.
ShortC8+3.480%/+3.313%, longC8−0.113%/−0.528% vs stock. C1TPS+14.002%/+13.392%
but TTFT+9.940%/+8.695%. Both original speed gates fail. Keep both reused
controls' time gaps explicit; these numbers precede archive verification.

Local retainer `/private/tmp/mxwave-output-row16-v1/retain_local.py` is ready,
expects remote receipt and evidence.tar.gz only after exact restoration and
Spark collection. It rechecks every payload, frozen verifier and cross-host
probability-only1e-12 roundoff bound, with exact discrete gates. The copied
correlation postprocessor is ready; after download run it over all16 worker
traces and inspect pure C16>8-row ranges for34 graph reconstructions, preserving
unassigned counts/absence of an actual range. No invented kernel ownership.

Actual MLA next source is now frozen LOCALLY ONLY in
`/private/tmp/mxwave-mla-internal-probe-v1/`. ArchiveSHA
`2747892df71900dc2ae99041f819fc4364a19a64f0f233d58c9c583092324a8f`;
planSHA `e2e4afbe4c28fc380631fcefcf176132cd17f88778f2ce150e797cb3fde26782`.
Seven overlay sources ->162 assembled from immutable156-binding boundary
parent. No upload or GPU controller. CPU preparer/script and nested programs
compile; it requires output16 terminal restoration/retention before executing.
Six focused tests, latest full suite(one skip) and whole Ruff pass.
Source guard now reserves8MiB transfer BEFORE every capture allocation.
Earlier local frozen draft1f55... is retained under frozen-first; it checked the
reserve only after capture. Initial local import-path failure also retained.
No GPU/source upload occurred for either draft.

Native eager-break inspection after all output16 timing confirms FULL mode
calls the original operation without leaving capture; fileSHA
`722683b44b4461ffd9a6498741da18945cbaac231c3bb01f0b0443dafe9e053e`.
Private inspection receipt local only; no native implementation copied to repo.
Current diagnostic captures planned-prefix bytes rather than masking to
conversion-valid counts, so actual clamped slot-zero reads remain visible.
Private workspace validation uses source-bound full runtime byte comparisons;
temporary complete private SSD images are not archived. Raw internal operands
and outputs are retained for independent numerical reconstruction.
Do not launch a second GPU controller while output16 owns the service pause.

**10:56 UTC continuation:** output16v1 is ready and unprofiled timing is
running; last check12/72 groups, all24warmups complete. Both worker snapshots
attest34 output-only exact16 exceptions, zero input exceptions, global limit8,
zero resident BF16 cache. No error. Quality/profiles follow all timing.
Controller2930080/watchdog2932768 still own the original service pause.

New actual-MLA observer is local only, not uploaded or GPU-run:
`mla_internal_capture.py`, `mla_internal_worker.py`,
`mla_internal_extension.py`, `mla_internal_gpu_fixture.py`,
`measure_mla_internal.py`, `run_mla_internal.py` in integrations.
The broad24MiB trace is replaced; copies+conservative gather staging+8MiB
transfer allowance must fit32MiB. Capture wraps actual layer3 KV writes,
query/value BMMs, sparse conversion, shared plan metadata, planned-prefix
selected FP8 bytes and attention output. Complete BMM hashes/strides and
current device/host scales are retained separately from captured scalar values.
Use planned lengths to mask selected cache, not conversion counts: an overlong
schedule can read real clamped slot-zero tails. CPU tests cover this.

Both proposed policies create SSD private images and reset shared device plan
buffers; only restore_private resets real float/int workspace before each
full-graph call. Four ABBA-ordered probes, five fixed calls/serve fifth. Tiny
authored CUDA graph fixture is written, GPU-unrun. No model source/plan archive
is frozen yet. Source must start from immutable156-binding
`full-decodeboundaryv1-src`, with broad worker replaced, preserving raw reference
kernels. Native source pins: MLA a309..., backend4449..., wrapper58d0...,
manager c183... full hashes in install(). New worker import/install separated
into extension so CPU tests don't import native vLLM. Five focused tests and
full suite pass one skip; whole Ruff passes before latest client/runner, their
focused Ruff also passes. No native implementation copied into repo.
Do not launch this while output16 runs. No numerical cause/correction proven.

**Current,18 October10:42 UTC: output-only exact16 model loading.**
`output16v1` controller2930080/collector2930079/watchdog2932768 are active.
CPU preparation passes178 bindings, both-node source equality and actual CLI.
The tiny ownership fixture passes; model loading is in progress. Original
service is paused under the independent watchdog. Do not end while paused.
Only exactly16-row square KDA output projections add a fused route; input
path/global limit8/native kernels/precision/two-slot pool unchanged, no BF16
cache or MTP. New local helpers/tests are in the worktree; isolated GPU source
starts from immutable174-binding `full-c816retryv2-src`, preserving unrelated
worktree exponent guards outside that source delta.

Local `/private/tmp/mxwave-output-row16-v1/`; remote full source
`full-output16v1-src`, prepared `output-row16-v1-prepared.json`,
controller `quality-model-controller-output16v1.json`, result
`run-output16v1/results/candidate-high-concurrency.json`, receipt
`output-row16-v1-evidence-check.json`.
Source archiveSHA `cb908fe66622d7c1bfe2aca4acc7eb16b7febff08ac4f5e77bbe61f610ce620e`;
preparedSHA `08bda6420ebc10f0c0cea6a0c979c7ca2c02b27f379a2fb3599523c19332b0e4`;
planSHA `f0ce0ea728a86f71b6cc9fa78268b6e6c6ce717b3059c26ac46067c75a4188f3`.
All original C8/C16 model gates,72 timing groups and separate profiles retained.
Stock and compressed-parent evidence reused with explicit time gaps.
Independent verifier/collector frozen/uploaded before launch; no numerical
or speed result yet. Full tests pass one skip, Ruff/core35 mypy pass.
CPU MLA internal-probe work may proceed locally while this GPU run executes;
do not launch a second GPU controller.

**Earlier,18 October10:32 UTC: C8/C16 complete; focused V7 fails shared row16.**
`c816retryv2` is complete and exactly restored. All1,277 payloads,174 source
bindings,144 groups/1,272 requests/81,408 tokens,672 probability groups and
sixteen worker traces verify on Spark/Mac. Archive362,781,198 bytes SHA256
`602e25daf5a498a2b6012c41276cb91f41695ccfd2093108a7eaacae0239d66a`.
Memory saving9.367GiB/TP2, C1+12.3–12.6%, shortC8+2.2–2.5%, longC8−0.3–0.7%,
shortC16−13.6–13.8%, longC16−7.5–7.6%. Both performance gates fail.
All636 timed paired sequences match, but all24 KL comparisons fail and four
NLL comparisons fail. Stock control is reused with a time gap, not a fresh
paired load. Never call completion qualification. Repo compact review:
`benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/high-concurrency-retry-v2-review.json`.

Both-rank correlation attribution finds eight pure-decode C16 ranges
(actual9/11/14/15, padded graph16), each rebuilding all68 KDA matrices in
34–37ms, with zero unassigned decoders. Allocated compressed bytes plus
padded BF16 writes imply213–231GB/s; this is a byte proxy, not measured LPDDR
traffic or request latency. The startup ownership repair passes292 worker
observations, pool206,569,472 bytes/rank, two leases/one captured, fusedmax8.
Do not tune the unchanged standalone decoder or resurrect naive overlap.

Focused `prefilldecodev7` controller2812316/collector2812315/watchdog2813957
is now complete, originals exactly restored and health200 at10:29:07 UTC.
All45 payloads,18source files and18,432 timing events independently verify
on Spark/Mac; independent reviews identical. Archive309,124 bytes SHA256
`ae9d4fd7a8e8173def9654c1ca0b53dc40ff79f1e59340c3d7d8d62711d663f3`.
Local `/private/tmp/mxwave-prefill-decode-v7/evidence/`.
Both real fixtures have relativeL2 zero and exact eager/graph outputs.
Output matrix qualifies at8 and16; input matrix qualifies at neither.
At input16 graph-warm, ABBA identical-control ratio0.9734604466852056
fails unchanged[.98,1.02]; all input16 fused gains pass. Shared lead[].
Output16 fused/reference ratios0.349–0.504 pass all conditions/orders.
257 remains slower for both shapes. No global threshold increase, interpolation
or serving change; no model speed gain claimed. Keep every failure.
Two CPU-preparation failures (nested-string syntax, worker-only build location)
are retained; both occurred without GPU launch or service pause.

Next: consider only the separately passing output-projection exact16 route,
retain the input reconstruction path, and qualify the changed runtime on
complete-model timing/quality/profiles. This can address part of C16 overhead,
not the remaining input-matrix tax or outstanding MLA numerical failure.
Do not retry V7 unchanged or loosen timing controls. MTP stays deferred.
Original service is currently healthy; no GPU controller active.
Owned caffeinate50382 prevents sleep until15:45:33 UTC/17:45Berlin, verified
with pmset. User explicitly requested keeping laptop awake and continuing work.

**Earlier, 18 October09:50 UTC: full captures pass; timing33/72.**
Both workers retain captures[1,2,4,8,16], exactlytwo KDA scratch leases with
one graph-owned lease, pool206,569,472 bytes/rank, fused limit8. All24warmups
complete;33/72 timing groups, no runtime error, quality0/8/profiles0/4.
Original service remains paused under retry2 controller2367788/watchdog2371083.
Sleep prevention verified with `pmset -g assertions`: ownedcaffeinate50382
expires **15:45:33 UTC /17:45Berlin**, receipt
`/private/tmp/mxwave-experiment-awake.json`; priorowned23172 remains harmless,
unrelated processes untouched. User explicitly says keep laptop awake and work.

Conditional future component `prefilldecodev7` is CPU-prepared locally only:
`/private/tmp/mxwave-prefill-decode-v7/`,18Python/CUDA sources +plan, archive
SHA256 `446415deb541b9babfee64f5562bf8244117d97319530c7b29dbdec1cd34809a`,
planSHA256 `4bee510d0cbe8af25ed89e06a407c37b860f73e1afebd4086fd92a56d7d1c6ca`.
No upload, remote preparation, controller or GPUrun. Trigger only after current
C16 profiles show material repeated KDA reconstruction. New timing queues an
untimed primer and eight event pairs per leg before synchronizing once; cache
pressure remains before each pair. Keep8ABBA/BAAB blocks, bothorders/eager/graph/
warm/pressure, bothreal matrices,2%repeat and5%gain gates,1%projection-L2.
Rows[8,16,257] retain currentpositive andnegative controls;18,432 timed events.
No automatic router change or interpolation to9..15. If16qualifies, first
propose exact16 in addition to current<=8 and still run fullmodel qualification.
Legacy V6 review recomputes identically with the extended verifier; arithmetic
function ASTs match. New queued-event/independent-verifier tests and Ruffpass.
Current GPU source174 remains frozen and unchanged by these local changes.

**Earlier, 18 October 09:34 UTC: corrected CUDA fixture passed; model loading.**
`c816retryv2` controller2367788 / collector2367787 / watchdog2371083 are active.
The original service is paused under their ownership; do not end a turn while
it remains paused. CPU preparation and CLI pass with174 bindings after retry1
restores exactly. New prepared SHA256
`80f1d1ba7205aa4ab8864e7f9f0b3e719fb80f25d8c39f81fc711b461c927e2f`.
Corrected real CUDA fixture passes: old third-owner guard reproduces, completed
eager leases retire, graph lease survives, eager slot0 and captured slot1 stay
disjoint; five extra graph replays match BF16 output bytes. Pool4096 fixture
bytes / two slots, allocator256MiB cap; native compressed kernels unchanged and
not called by this tiny test. Full-model C16 capture/timing still pending.

Retry1's failed fixture is fully retained:734 payloads,180,907,785 bytes,
archive SHA256 `bb55c71402ac04cded22be07851959435d2956d57e947e0486d381d0678225da`.
Spark and Mac frozen verifiers pass; local evidence under
`/private/tmp/mxwave-high-concurrency-retry-v1/evidence/`.
Optional additional stock-only GPU marker analysis is local at
`/private/tmp/mxwave-high-concurrency-v1/stock-kernel-breakdown-review.json`.
Pure-decode main-stream ranges containing84 expert kernels have expert GEMMs
about63–65ms at C8 and84–93ms for actual>8 rows in requested-C16 profiles.
Their main-stream ranges are111–121ms and137–148ms respectively. Only2–3
decode ranges/profile; duration sums are not request latency/TP critical path,
and uncovered intervals are not a CPU-launch-overhead measurement. No new
compressed result or broader throughput-fix claim follows.

**09:30 UTC update: first tiny GPU fixture failed; no retry model load.**
`c816retryv1` controller2300886/collector2300885/watchdog2303770 are restoring
the original service after its fixture asserted captured weight bytes before
executing the graph once. The fixture has now been corrected to replay once,
then check five additional replays interleaved with eager scratch reuse.
The allocator repair itself remains GPU-unqualified until this passes.
Preserve the failed run and its collector output; do not edit its frozen source.

New `c816retryv2` is frozen locally in
`/private/tmp/mxwave-high-concurrency-retry-v2/`, archive SHA256
`bf2cd68d4a5f9fb5a42e6bf1c15cadbf6ffc789b58f607bc20f2cacab4ed9308`.
Its initial CPU preparation completed assembly/control retention but reached
the healthy-service check while retry1 was still restoring; that precondition
failure is retained as `high-concurrency-retry-v2-preparation-first.json`.
Continuation2359733 waits for retry1's terminal restoration before repeating
bindings/native/CLI checks and completing preparation. No second GPU controller
exists. Full source has174 bindings; KDA arithmetic AST remains unchanged.
Compact first-attempt benchmark now exists at
`benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/high-concurrency-v1-review.json`.
All eight worker traces have local correlation-based attribution. The stock-only
supplement recomputes72 groups/636 requests/40,704 tokens and four self-repeats.
Status questions keep steering the existing compression goal; user has not
cancelled. Overall success gates remain failed; MTP explicitly deferred.

**Current, 18 October 09:22 UTC: C16 capture failure retained; service restored.**
`c816v1` is `failed_restored`, not running. Stock completes72 timing groups /
636 requests /40,704 tokens, eight quality sets and eight worker profiles.
Candidate fails before timing because `_slot` retains completed startup
streams and rejects the third stream. The exact pinned native graph manager
uses separate initialization, warmup and capture streams. All730 payloads
verify on Spark/Mac; archive180,955,887 bytes, SHA256
`879ee5dd8e956e36c19419625dc6a89056cff6303e913fa54ec7568e13081ef9`.
Four stock self-repeats fail KL; C8/window0 also fails NLL. IDs/conditioning
all match. No compression TPS result exists and throughput is not fixed.

`c816retryv1` CPU preparer2280632 is running without service pause. Local
`/private/tmp/mxwave-high-concurrency-retry-v1/`, source archive SHA256
`eadb74f57b23671c8ca69190b6ab994c0a182b8751c0f140d7e88db939ad223a`.
The isolated source starts from frozen169-source `c816v1`; only scratch lease
ownership/reporting changes in KDA. Six arithmetic function ASTs match exactly.
Worktree's extra exponent guards remain preserved locally, outside this
isolated known-checkpoint retry. New helper preserves graph-owned slots and
retires completed eager owners only before/after native startup capture.
Two slots/256 MiB and fused limit8 unchanged. Real tiny-CUDA regression must
pass before loading compressed model. Reuse completed stock arm verbatim,
explicitly disclose time gap; do not claim a fresh paired stock comparison.
Four separate short C8/C16 profiles/rank planned; long cases have API/queue
evidence only. MTP remains deferred. Original service healthy, no current pause.
Full tests (one skip), Ruff and core strict mypy pass. Owned awake lease23172
expires11:23 UTC; do not touch unrelated87929. Source-only upload approval
persists; optional docs/analysis bundles stay local.

**Historical, 18 October 08:24 UTC: compression C8/C16 takes priority; MTP deferred.**
User explicitly wants a working compression outcome before MTP testing.
`mtppilotv3` now passes native CPU preparation with the repaired eighteen-check
validator, 174 source bindings and zero padding; no new GPU controller exists.
Do not launch it while completing compression.

`c816v1` passes CPU preparation and its guarded paired GPU controller is launched.
Read [frozen high-concurrency protocol](GLM53_HIGH_CONCURRENCY.md) and live
`ROOT/quality-model-controller-c816v1.json`; source `full-c816v1-src` (169 bindings),
results `run-c816v1/results`, receipt `high-concurrency-v1-evidence-check.json`.
Both stock/compressed arms use full graphs [1,2,4,8,16], max sequences16,
4 GiB KV/rank, no MTP and interleaved 0/3 ms idle windows. Six cases with six
trials/window/target retain 144 groups / 1,272 requests / 81,408 tokens.
C16 retains current >8-row reconstruction; numerical floors and old failures
are unchanged. No new GPU result or throughput-fix claim yet.
Local `/private/tmp/mxwave-high-concurrency-v1/`; source archive SHA256
`d18d4952e409f464a8c82157c9bdca9ed8fd56d63c96807e311130bce31cdb27`.
Exact restoration must finish before evidence collection/final response.
Live observation at **08:54 UTC**: controller1769874 / collector1769873 /
watchdog1773534 are running; original service is paused under their ownership.
Stock has all72 timing groups / 636 requests / 40,704 tokens; every raw timing
group and both-rank C16 descriptor validates locally. Quality5/8, profiles0/4;
compressed arm has not started. These are timestamped counters, not final results.
Use `control-timing-raw-check.json` locally for stock-only provisional medians.
Do not infer a compression gain before the paired arm completes.

Local completion helpers are prepared: `retain_local.py` verifies receipt SHA,
safe archive members/inventory and runs the frozen verifier on Mac;
`summarize_local.py` compares Spark/Mac reviews (probability-only1e-12 CPU
roundoff allowance), retains all gates and writes the new compact benchmark.
Run the repository's `analyze_graph_reconstruction.py` on the finished evidence
into `graph-reconstruction-review.json` before summarizing. This new local
postprocessor is outside the frozen169-source GPU bundle; no runtime change.
It follows GPU correlations to CUDA API calls inside CPU model annotations,
retaining unassigned events. Two asynchronous/ambiguous-attribution tests pass;
all six old admission worker traces validate with zero unassigned decoder
kernels and exactly136/68/68 per rank, all in prefill/mixed forwards.
Current-run traces have not been collected yet.

**08:59 UTC update:** stock completes72 trials, eight quality sets and all
four profiles; source-after checks pass. Candidate is loading, originals
still paused under controller1769874/watchdog1773534. Stock rank-zero
profile snapshot verifies11 files locally, stored in
`stock-profile-snapshot/`; correlation output `stock-profile-correlation-review.json`.
Requested-C16 short profiles contain active decode batches11–14 using the
16-token graph. This covers >8 routing but is not a steady fully active C16
profile. No compressed speed result or final archive yet.

Full local suite (one skip), Ruff and strict core mypy pass.
Owned awake lease PID23172 expires **11:23 UTC**; prior owned97064 expires09:14,
unrelated87929 untouched. Approved source-only uploads persist; local docs stay local.

**Historical, 18 October 07:54 UTC: admission and actual-geometry CPU proof complete.**
Original GLM identities/configs/mounts restored exactly, health 200 at
**07:49:50 UTC**; no active GPU experiment and no configuration promoted.
`CURRENT_LOSSLESS_STATUS.json` now records the completed results and healthy
service. Controller 1248710 / watchdog 1251294 / collector 1248709 are completed.

[Admission result](GLM53_PREFILL_ADMISSION.md): 3/10 ms idle-only collection
gives primary short-C4 TPS **+6.75% / +6.62%**, TTFT **1.125 → 0.778 / 0.782 s**,
and median large forwards **two → one**. Both frozen performance/mechanism
gates pass. C8 TPS +3.65%/+4.19%; long C4 near parity, still four chunks;
staggered C4 +0.36%/+6.70%. These are finite 64-output request groups, not
sustained decode-only or continuous-active-traffic measurements. Graph/admission
policies have not been combined, and no new compression-advantage claim is made.

All **599 payloads / 108 groups / 450 requests / 28,800 tokens**, 336 source-scale
groups, 384 distributions and six worker profiles independently verify on
Spark/Mac. All **300 timed sequence comparisons** match. Per-weight observation
finds all 68 prefixes once per eager forward, no within-forward duplicate in
the three profiles. Device kernels match **136 → 68** reconstructions/rank.
Three API traces remain separate. All seven numerical comparisons fail KL1e-5;
NLL passes 1e-4, IDs/conditioning match. Self-repeat KL at 0/3/10 ms is
6.0946e-5 / 3.0801e-5 / 1.3036e-4. Historical failures remain open.

Archive `evidence-prefill-admission-v1.tar.gz`, 131,372,757 bytes, SHA256
`d63d6e69c7acda35d5d6f6cb83b38bbb1fea958b2681b1611a24930db64a9da7`.
Source `full-prefilladmissionv1-src` has 170 bindings: only parent plugin
metadata changes, 159 other parent files unchanged. Results
`run-prefilladmissionv1/results`; receipt `prefill-admission-v1-evidence-check.json`.
Local `/private/tmp/mxwave-prefill-admission-gpu-v1/{evidence,verify.py,mac-independent-review.json}`.
Mac CPU probability recomputation has 869 float differences, max 1.7765e-15;
verifier tolerance 1e-12 applies only there. Model thresholds are unchanged;
all other review fields/decisions agree exactly. No byte-equal review claim.
Compact repo `diagnostics-v1/prefill-admission-model-v1-review.json`.

[MTP actual-geometry CPU proof](GLM53_MTP_PILOT.md) also completes, no CUDA/model
load or service pause. Both complete q/KV source matrices bind exactly through
native `load_weights` and the bridge into 2,048×4,096; padding count zero.
All 14 payloads / four raw artifacts verify, Spark/Mac reviews byte-identical.
Only two projection checks execute; sixteen routed/shared checks are still
part of the next live eighteen-check observer. Source/worker are frozen and
uploaded, but the worker is not instantiated here. Completed CPU waiter 1475744.
Archive `evidence-mtp-geometry-v1.tar.gz`, SHA256
`1d7aa99458c4ccef6144e26f8bc2cf85e4d76f038c92dd10e9330e12cd13b4db`,
40,031,996 bytes; progress `mtp-geometry-v1-progress.json`.
Local `/private/tmp/mxwave-mtp-geometry-v1/`; source archive
`775f4cf1125a1c262fb28395cf297c16cf09716223b253fa1c45710d8204c23c`.
Full-pilot preparer/verifier still need repair/freeze: their old success branch
assumes nineteen checks and one 64-row padding event. No MTP acceptance/TPS yet.

Next: implement actual MLA internal operand/cache/private-state capture (local
source map prepared; GPU probe not implemented), then MTP validator/GPU work
and dedicated C2 qualification. Keep all numerical failures. C16/capacity,
continuous-traffic policy qualification and the deferred roofline remain.
Use the qualified `full-pairedspeedv1-src` parent, not the entire current
worktree's unqualified MoE runtime. Full local tests, Ruff and core mypy pass.
Owned awake 97064 expires 09:14 UTC; unrelated 87929 untouched. Source-only
permissions persist; denied optional docs/MLA bundles remain local.

**Historical launch, 18 October 07:19 UTC: admission experiment launched.**
Read [frozen protocol and queue/weight checks](GLM53_PREFILL_ADMISSION.md).
Revision `prefilladmissionv1`, controller 1248710 / watchdog 1251294 /
collector 1248709. Original service is temporarily stopped under the proven
idle guard and exact restoration watchdog; inspect actual controller phase.
Prepared/source bindings and the actual two-plugin CPU CLI pass on both nodes.
170 source bindings: 159 other parent files unchanged; only
the plugin metadata adds `admission_plugin:install` beside the qualified
`quality_worker:install`, plus 10 new modules. No MTP, graph-policy, KV or
native-kernel change. Handler/auditor hashes match the passing seven-case
CPU fixture. GPU observer fixture runs before full model loading.

0/3/10 ms fixed idle-only collection, 108 timing groups, 450 requests, 28,800
tokens, 30 warmups. Primary short C4 has 12 trials/window; C1/C8/long C4 and
2 ms-staggered C4 have six. All six window orders balanced per case. Every window
gets two C4 full-vocabulary repeats (384 distributions); strict seven numerical
comparisons and all historical failures preserved. Then three separate profiles
observe 68 distinct KDA weight prefixes per eager forward; compare with actual
`BF16TripleBitmap_Decompress_Kernel` events on both ranks, exclude graph calls,
and restore original methods. No timing instrumentation is added for weights.

Controller `ROOT/quality-model-controller-prefilladmissionv1.json`, prepared
`ROOT/prefill-admission-model-v1-prepared.json`, source
`ROOT/full-prefilladmissionv1-src`, results `ROOT/run-prefilladmissionv1/results`.
Local `/private/tmp/mxwave-prefill-admission-gpu-v1/`; source archive SHA256
`6ea4b7b85a65eb9d846e365dee5477c0a88bce4be07dadf49838b356a82a7904`;
prepared `6dfa66c454f05728a976e10c6993e1b1cd0e16c34304d8f75c380056936d2741`;
plan `639475566d54800c38661d8bb545eb54edcc9723ee053e094e94b4b118a6b341`.
Full local suite, Ruff/core mypy pass. New owned awake lease 97064 expires
09:14 UTC; old owned 43050 expires naturally 07:39; unrelated 87929 untouched.
Source-only permissions persist; optional denied docs/MLA bundles stay local.
MLA actual operands, dedicated C2 qualification, C16 and corrected MTP observer
remain pending. Roofline deferred. No numerical or speed result from this run.

**Completed state, 18 October 07:06 UTC: graph comparison complete, originals healthy.**
[Result and next decisions](GLM53_GRAPH_EXPANSION.md): exact C2 +8.6–9.0% TPS;
C8 +1.7–3.4%. Bundle fails the frozen 5% short-C8 gain / C4 TTFT guard
(C4 +34.8%), and both policies fail KL repeats. All 150 paired output
sequences match. Numerical means: baseline repeat 8.2816e-5, expanded repeat
3.8746e-5; unchanged KL floor 1e-5. No configuration promoted.

`quality-model-controller-graphexpandv1.json` is complete, service_paused=false.
Exact original IDs/configs/mounts and HTTP 200 restored at **07:00:20 UTC**.
Controller 882869 / watchdog 890911 / collector 882868 are completed-run IDs.
All 466 payloads, 72 groups, 300 requests, 19,200 timed tokens, all 336 source
scale groups and eight native worker profiles verify independently. C8 baseline
has no replay in its profile; expanded has four launches on each rank and no
eager eight-row calls. Four API traces are retained separately. Source 168 files
still preserve all 160 qualified parent files. Captures succeed on both ranks;
0.76/0.62 GiB totals do not measure incremental graph memory.

Archive `evidence-graph-expansion-v1.tar.gz`, 99,364,297 bytes, SHA256
`60eeecbd1e50bde9787e97319f01c65f47a21a739f8b0b515c560bc6b07d67d3`.
Local `/private/tmp/mxwave-graph-expansion-v1/{evidence,verify_v3.py,*review-v3.json}`;
compact repo `benchmarks/.../diagnostics-v1/graph-expansion-v1-review.json`.
Verifier v1 incorrectly counted API traces as worker profiles; preserved.
V2 Mac strict float equality fails only CPU probability recomputation.
V3 classifies native worker traces and allows absolute 1e-12 recomputation
roundoff only there; observed max 1.9985e-15, unchanged model KL/NLL floors.
Spark/Mac reviews agree except explicit roundoff metadata, not byte equality.
V3 source SHA256 `8b8d7a038e8b680a2aa5247b652c10a2d5e642ec8c2b60b01b520d446d707f66`.

Next non-MTP experiment is bounded admission collection on the unchanged
qualified target: the native CPU fixture/prototype passes, GPU plan/preparation
is not frozen yet. C4 geometry changes again correlate with latency, but do not
prove a wait-window gain. Also capture actual MLA internal cache/projection
operands; do not rerun the unchanged passing synthetic fixture. Dedicated C2
probability qualification and C16 capacity/KDA work remain pending. MTP still
needs actual zero-rope observer CPU proof and updated independent verifier
before another GPU run. Roofline remains deferred. Full local tests/Ruff/core
mypy pass. Source-only permissions persist; denied optional docs/MLA uploads
remain local.

**Historical launch, 18 October 06:34 UTC: C2/C8 graph comparison launched.**
Read [native dispatch proof and frozen GPU plan](GLM53_GRAPH_EXPANSION.md).
Controller 882869 / collector 882868, revision `graphexpandv1`; inspect the
actual controller phase before assuming whether originals are paused.
Prepared/source files live in `ROOT/full-graphexpandv1-src` (168 bindings,
all 160 parent sources unchanged), results in `ROOT/run-graphexpandv1/results`.
CPU native builder/dispatch fixture and both-node CLI/source preflights pass.
C2 was already FULL padded to4; C8 was NONE. No C8-specific exclusion failure
found. Exact C2/FULL C8 are the two changes; keep max sequences 8, native kernels,
KV 4 GiB/rank, no MTP or admission wait. Both policies retain all four graph allocations.
72 timing groups/300 requests/19,200 tokens, unchanged C8 numerical repeats,
four separate profiles and both-rank traces. Independent collector waits for
exact original restoration. C16 remains separate: max sequences 16 plus KDA >8 fallback.
Local helpers `/private/tmp/mxwave-graph-expansion-v1/`; do not copy the current
worktree's unqualified MoE runtime. CPUv1 failed filter preserved; CPUv2 passed
native rebucketing, all 19 combined evidence payloads verified locally.

Admission window prototype/nativeCPU audit is ready but its GPU run was
deferred for this graph test. Admission CPU v1 passes seven real input-loop/
handler cases with stand-in scheduler/model, no CUDA; archive SHA256
`8e0bb635b34ffeefb10e0289e355d0f89eb07ed38049a710da352e785bbaecf6`, 11 payloads
verified locally. No admission plugin is registered in the graph source.
Full local suite, Ruff and strict core mypy pass.

**Completed state, 18 October 05:59 UTC: original service restored; non-MTP admission
audit complete; MTP observer corrected locally.** There is no active GPU run
from the retry. `quality-model-controller-mtpretryv1.json` is `failed_restored`;
exact original identities/HTTP 200 restored at **05:50:54 UTC**. Controller
495392, watchdog 498189 and observer 506688 are completed-run identifiers.

All **338** retry payloads verify on Spark/Mac, independent reviews byte-identical.
Archive `evidence-mtp-retry-v1.tar.gz`, SHA256
`90428f9bc31f8e83f35253ec9948afc07352675b6e933ee01403076c10e985e6`.
Source `ROOT/full-mtpretryv1-src` has 173 files. Local evidence, verifier and
collection scripts: `/private/tmp/mxwave-mtp-retry-v1/`; compact
[retry review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/mtp-retry-v1-review.json).

Independent CT config fixes the actual newer loader and the MTP model reaches
readiness. The client then aborts at my observer's incorrect requirement for a
padding event. Actual source config has **zero rope rows**, q rank 1,536, KV
rank 512 and hidden width 4,096: fused projection **2,048 × 4,096**. The optional
padding branch correctly never ran. The older 64-row CPU destination was
artificial and did not establish this actual model geometry. Local
`mtp_draft_loader.py`/`mtp_pilot_worker.py` now derive source checks from config;
zero/nonzero-rope regression cases pass. Actual zero-rope draft has 18 checks,
not 19. These fixes have not been uploaded/frozen/GPU-tested. Before another
MTP launch, run actual-geometry CPU preflight and update that revision's
independent verifier; do not reuse its old hardcoded 19/64-row success branch.
All 1,756 source-consumption and declared-parameter guards remain unchanged.

Previous `mtppilotv2` failed MTP startup on inherited CT metadata; its completed
control remains **16.768 / 13.961 TPS** at P256/P2048, 10 trials/640 tokens.
Repeat KL **1.1901962e-5 > 1e-5**, matching IDs/conditioning; NLL delta
2.29481e-5 < 1e-4. Preserve this failure and the control's temporal reuse limit.
No MTP request timing or release qualification exists.

**Work beyond MTP:** [prefill admission audit](GLM53_PREFILL_ADMISSION.md) checks
all 60 saved paired timing groups and both rank deltas. Short-C4 one/two/three
large forwards associate with 44.05/41.10/29.27 candidate TPS, one/three/one
samples. In separate profiles each of 68 KDA operations has exactly one
correlated reconstruction, totaling 36.259/36.263 ms per rank. Timed-trial
rebuild counts are inferred from the frozen path; no weight identity is in
these traces. This is not a causal scheduling improvement. Long C4/P2048 already
requires four 2,048-row chunks; its additional seven-row forward is fused.
Source/report: `/private/tmp/mxwave-prefill-admission-v1/`; all-trial review in
`benchmarks/.../diagnostics-v1/prefill-admission-v1-review.json`.

Next prepare a bounded non-MTP admission intervention with queue-delay and
per-forward/per-weight counters; inspect the actual scheduler boundary before
choosing its wait mechanism. Also prepare real internal MLA operand capture
at layer 3, including projection/cache-write/value-projection state omitted
by the passing synthetic attention fixture. Do not rerun unchanged decoder,
N8, split-K, pinning or isolated-attention negatives. Roofline still deferred.
Full local tests, Ruff and strict core mypy pass after the observer fix.
Source-only permissions persist; denied optional docs/MLA bundles remain local.

**Current, 22:46 UTC: complete MTP CPU preflight passes; C1 pilot loading.**
All 872 native draft selectors agree after repair (870 differed before), all
1,756 source headers and native compressed-vocabulary sharing pass. The real
speculative config resolves `Glm5NextMTPModel` to the audited NVIDIA class,
preserving CT/BF16, TP2, FP8 KV and max context 8,192. Both CLIs pass. Twelve
CPU evidence payloads independently verify locally. No MTP speed/quality result.

Model controller `ROOT/quality-model-controller-mtppilotv2.json`, PID **3328686**,
is loading the no-drafter control. Originals are temporarily stopped and
watchdog **3332671** is armed. Observer **3345087** samples native draft counters
only during untimed quality/profile phases. Both arms use the frozen compressed
target; the second adds MTP depth two with local argmax reduction. Draft scales
and GEMMs stay stock; base scale ownership must remain all 42 layers. Retain
all trials, full-vocabulary live decode repeats and the existing KL/NLL floors.
Source `ROOT/full-mtppilotv2-src` (169 hash-bound files, qualified native/codec
unchanged), prepared proof `ROOT/mtp-pilot-v2-cpu-preflight.json`; local
`/private/tmp/mxwave-mtp-pilot-v2/`. Earlier failed CPU fixtures/preparers remain
preserved; no optional local docs/MLA bundle was uploaded. Local full suite:
610 passed, one skipped; Ruff and strict core mypy pass.

**Completed, 22:15 UTC: routing sweeps verified; representative MTP mapping passes.**
V5/V6 retain 51,200 events, all numerical checks pass, and every timing decision
independently verifies on Spark/Mac. Small-row fused GEMMs are much faster;
V6's shared component leads are [8,24]. Several other rows fail 2% control
parity, and 257/2,048 rows regress. Keep serving at the current eight-row limit.
Originals restored at **22:03:41 UTC**, exact identities and health 200 verified;
no experiment GPU process is active. Read [routing evidence](GLM53_PREFILL_ROUTING.md).

`mtp-dispatch-v1` finds twelve of fourteen representative native draft selectors
wrong before namespace/fused-alias metadata repair, zero afterward. The base
policy stays unchanged; no model or CUDA allocation. Eight CPU evidence
payloads independently verify locally. No native MTP inference has run.
[MTP compatibility and pilot requirements](GLM53_MTP_PILOT.md) record the next
work: complete CT loader coverage, an explicit stock draft-method scope,
CPU CLI preflight, then C1 depth-two versus no drafter. Use frozen
`ROOT/full-pairedspeedv1-src` (local extracted paired-speed `source/`), not the
unqualified current worktree MoE runtime. Last full suite: 595 passed, one skipped;
Ruff and strict core mypy pass. Preserve KL/NLL gates and the open MLA cause.

Completed V6 state: `ROOT/prefill-decode-controller-prefilldecodev6.json`;
source/results `ROOT/prefill-decode-v6/`, local `/private/tmp/mxwave-prefill-decode-v6/`.
MTP CPU source/results `ROOT/mtp-dispatch-v1/`, local
`/private/tmp/mxwave-mtp-dispatch-v1/`. No optional local analysis/docs bundle
has been uploaded; standing permission covers ongoing source-only revisions.

**Completed, 21:45 UTC: V4 mask regression and complete base scan verified.**
The fixed candidate masks exponent addition to eight bits. Both candidate
kernels pass the original 243-error base-255 fixture and the exhaustive BF16
fixture (99,200 errors without the mask), in eager, separate-stream and six
graph replays per mode. Both Python/C ABI stock guards refuse unsafe bases.
CPU audit covers 136 actual loaded KDA TP shards, both heads and 154,880
embedding rows, with no unsafe bases. V4's 34 outer / 13 native payloads and
the audit's twelve payloads independently verify locally; Spark/Mac V4 reviews
match bytes. Originals restored at **21:43:57 UTC**, health 200 and exact
identities verified. Qualified serving library and adapter unchanged.
State `ROOT/prefill-decode-controller-prefilldecodev4.json`; source/results
`ROOT/prefill-decode-v4/`; local `/private/tmp/mxwave-prefill-decode-v4/`.
Next measure small-prefill split-K=1 routing, including verification-batch
shapes above the eight-row limit. See [MTP/FP8 assessment](GLM53_SPEED_OPTIONS.md).
No full-model numerical qualification or new throughput win is claimed.

**Earlier completed component: `prefilldecodev3`.** The
[direct-output reconstruction test](GLM53_PREFILL_RECONSTRUCTION.md) uses two
real KDA matrices and the same stock GEMM, after synthetic exact-byte checks.
All synthetic and real fixture checks pass exact bytes. Neither candidate
meets the speed gates; rebuilt-control parity also fails. There is no accepted
speed win, confirmation or adapter change. All 36 outer / 13 native payloads
and 6,656 samples independently verify on Spark/Mac. Exact original identities
restored at **21:17:02 UTC**. Controller PID
**2710212**, state `ROOT/prefill-decode-controller-prefilldecodev3.json`.
The idle-only pause has an independent restoration watchdog. Do not start
another GPU/model job or finish while originals are paused. The first
unpaused v1 smoke failed at CUDA initialization before any case; retain it.
V2 stopped at the qualified decoder's raw-word source check, restored at
21:03:35 UTC and independently verifies on Spark/Mac (36 outer/13 native
payloads). CPU inversion locates an unsigned exponent-base wrap to 255;
masking exponent addition recovers all source bytes. V3 retains every raw
word in a representable fixture and rejects wrapped bases before GPU use;
it does not change the native library or numerical/performance gates.
Current source now masks candidate exponent addition to eight bits, with
writer/loader guards for the unchanged qualified decoder. V4 GPU qualification
is complete as above. The CPU-only `prefill-base-audit-v1` scanned all
136 actual KDA TP shards, both compressed heads and both embedding base
arrays without pausing service. Its report is on worker rank 1 under
`ROOT/prefill-base-audit-v1/results/base-scan.json`.
The user explicitly approved ongoing source-only revisions for this decoder
experiment. No rejected document/report bundle was uploaded.

**Complete and independently verified: paired serving speed, 17 October,
`pairedspeedv1`.** C1 throughput improves **10.4–12.4%**; C4/C8 changes are
**−0.07% to +1.84%**, with TTFT **2.3–5.9% higher**. Memory saving remains
**9.367 GiB / 5.33%**. All 130 paired 64-token sequences match; this does not
amend full-distribution/strict quality failures. Both original GLM identities
were restored healthy at **20:22:23 UTC**, with no experiment engine active.
All 373 payloads, 672 source groups and 260 requests / 16,640 tokens verify
on Spark/Mac. Three independent analysis JSON files match bytes. Read
the [paired result and next steps](GLM53_PAIRED_SPEED.md),
`ROOT/quality-model-controller-pairedspeedv1.json` and
`ROOT/paired-speed-v1-evidence-check.json` for authoritative state. Frozen source:
`ROOT/full-pairedspeedv1-src`; local continuation details:
`/private/tmp/mxwave-paired-speed-v1/LIVE_STATUS.md`.

Local raw archives, verifiers, three byte-identical Spark/Mac analyses,
admission counters, parser-error logs and completed status are retained under
`/private/tmp/mxwave-paired-speed-v1/`. The local analysis bundle is
`paired-speed-v1-independent-analysis.tar.gz`; raw archives remain separate
and SHA-bound. Compact review:
`benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/paired-speed-model-v1-review.json`.

Both-rank C4 profiles prove three graph launches and three complete decode
forwards/arm: median GPU span **85.0 → 78.2 ms**, about **8% shorter**.
Prefill instead increases **6.6–6.7%**, with 68 candidate KDA reconstructions
costing **36.26 ms** per profiled prefill. Slow ordinary short-C4 groups
correlate with mixed/split admission on both ranks; all remain in the medians.
Next target bounded prefill reconstruction and batch formation; do not blindly
repeat rejected N8, split-K, fused-prefill or naive-overlap variants. No new
accuracy qualification, GSM8K rerun or roofline. The exact MLA cause stays open.
Two offline parser errors were repaired append-only (frontend trace selection
and context-request count). Their code/logs remain; no inference was repeated.

An optional upload of the preceding MLA analysis bundle and updated project
documents was rejected by automatic approval review; these files remain local.
The original Spark evidence and its CPU verification remain on Spark. Do not
retry that upload without approval or new authorization evidence. Source-only
scripts required for this performance experiment were separately approved.

**Complete: isolated MLA operation, 17 October.** All six fresh-process cases /
thirty graph invocations repeat bytes and match eager output. Both cache formats,
plan histories and private workspace policies are covered. All 380 archive payloads
and 36 raw output tensors verify independently on Spark/Mac. Float/int workspace
is unchanged by these short-context calls; that limits the reset conclusion.
This synthetic latent-query fixture excludes the full module's projections,
KV writes and multi-layer interactions. Its negative reproduction result does
not fix or exonerate the full module. Original GLM identities restored healthy
at **19:40:08 UTC**. Read the
[verified component review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/mla-workspace-v1-review.json).

Next: matched reference/candidate C1/C4/C8 speed and latency plus separate C4
profiles, retaining graphs [1,4], KV budget, source/native bindings and historical
qualification failures. No new broad accuracy/GSM8K rerun. Roofline remains
deferred. The exact numerical cause in the full MLA module stays open; a future
capture of its real latent queries, cache and implicit state can narrow it.

**Complete: actual graph-boundary diagnostic, 17 October.** Two probes / twenty
graph invocations; 336 source groups, 4,511 archive payloads and 4,100 raw boundary
tensors verify independently on Spark and Mac. Every replay starts with exact
tracked bytes. In 14/16 non-first rank comparisons, the first changed module is
layer 3's inner `MLAAttention`; the other two first change at the TP output
projection while the peer already changes in MLA. Recorded module inputs match.
Resetting exposed stock-Marlin semaphores does not remove variation. All 32 IDs
match the saved reference (four distinct prompts). Original identities restored
healthy at **19:21:30 UTC**. No experiment remains active. Read the
[verified boundary review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/decode-boundary-model-v1-review.json).

The exact MLA-internal kernel cause is unresolved: latent projection, cache write,
sparse attention, value projection, implicit metadata and private workspace must
be separated. Next use an isolated operation fixture with fixed cache/plan and
private FlashInfer workspace, retaining the numerical limits. Do not launch
another broad quality or GSM8K rerun for this. No new serving KL or speed result;
strict qualification remains false. Then return to paired kernel profiling and
concurrency tests. Source: `ROOT/full-decodeboundaryv1-src`; analysis:
`ROOT/decode-boundary-v1-independent-analysis.tar.gz`. The diagnostic doc binds
the installed backend/FlashInfer wrapper and records all limitations.

**Earlier: reference active-value diagnostic, 17 October.** Four logical
probes / forty graph invocations, 336 source groups and all 454 archive payloads
verify. Independent Spark/Mac CPU analysis reconstructs forty raw hidden tensors
and 32 earliest changed active-cache pairs. Every non-first replay first changes
layer 4 / cache0; active layers 0–2 stay byte-identical. Device synchronization
does not remove the variation. All sixteen completions / 64 IDs match the saved
reference. Exact original service identities restored healthy at **18:43:50 UTC**;
no experiment remains active. Read the
[verified active-value review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/decode-values-model-v1-review.json).

Earliest active-cache relative L2 errors are 0.0973%–0.2264%; final hidden
relative L2 changes are 4.30%–39.35% under intrusive fixed-tracked-state replay.
Untracked internal workspace remains a confound. These values are not an
ordinary-serving KL estimate or proof of roundoff. No new accuracy or speed
benchmark ran. Strict qualification remains false; all previous failed gates
are unchanged. Next capture actual graph operands/outputs across layer 3's
MLA, MoE and mHC boundary into layer 4, then reconstruct the first differing
operation with identical operands and controlled workspace. Source is frozen in
ROOT/full-decodevaluesv1-src; retain the native-qualified v9 MoE runtime.

The preceding V3 diagnostic completed six probes / sixty graph invocations;
336 source groups and 342 archive payloads verify. It reproduced differing
hidden/cache fingerprints from identical tracked starting bytes, with 96/96
IDs matching the saved reference. Its active pre-state lead from layer 4 is
now narrowed by the raw active post-state evidence above. Full post-cache
changes at layer 0 include inactive rows. V3's collection/output-only repairs
are preserved and do not amend inference evidence or historical failed gates.

V3 launched with controller PID 1270278 and collector PID 1270277. Both-rank CPU
preflight passed the 145-file source, native bindings, actual CLI and schema.
V2 finished all five repeats: self-KL 0.000004644 / 0.000009817 /
0.000004663 / 0.000009554; every generated ID and prefill distribution matches.
It then failed before replay admission because rank 0 lacked room for the
8.56 GB GPU snapshots. No full-model replay ran. Exact originals restored
healthy; all 678 archive payloads and 320 distribution rows independently verify.
V3 reused those five repeats and moved the two temporary exact images to SSD
with 8 MiB transfers. Its tiny GPU fixture passed all four cases
before the one reference load for the six replay probes. No repeated GSM8K,
throughput benchmark or compressed candidate. Read
[the frozen scope, limits and exact paths](GLM53_DECODE_STATE_DIAGNOSTIC.md).
The preceding v1 fixture's CUDA-initialization failure is preserved; it never
ran graph probes or paused GLM. All completed benchmark results below remain.

**Roofline comparison deferred by user until the model work is finished.**
The user requested a paired reference/candidate log-log graph, matching the
provided example, and links to PNG/SVG plus remaining work afterward.
No roofline GPU workload has started. Existing operation timings and a new
read-only tool query identify the available L2 counters and absent DRAM-byte
definitions in the installed Nsight Compute version. Read
[the evidence and paired measurement plan](GLM53_SPARK_ROOFLINE.md).

**Complete and independently verified: full GSM8K benchmark, 17 October.**
All 1,319 pinned questions/arm: numeric scores **1,246/1,319 (94.47%) reference
versus 1,244/1,319 (94.31%) candidate**; strict format **1,228 versus 1,230**.
Primary pass changes are 22 lost / 20 gained, exact McNemar p=0.8776.
Throughput **41.770 →42.080 tok/s (+0.74%)**; the whole workload finishes
only **3.57 seconds sooner (+0.16% questions/minute)**. Memory
**175.866 →166.499 GiB across TP2**, **9.367 GiB / 5.33% saved**.
No request errors/truncations. Both arms retain captures [1,4], canonical
dispatch, low reasoning effort, 1,024-token cap, C4 and no drafter. The
candidate emits more tokens; this is roughly equal workload speed.

All 2,638 answers independently regrade, 672 source-scale groups and both
139-file source inventories verify, and all 2,960 archive payloads verify
on Spark and locally. Exact original GLM identities/configurations are
restored healthy at **16:57:33 UTC**. This batch does not attempt new KL
qualification or amend any older failure. Read
[full results, evidence and targeted next steps](GLM53_GSM8K_FULL_BENCHMARK.md).

**Authorized next: targeted reference decode/state diagnostic.** The user
confirmed proceeding after this benchmark. Prepare a new frozen revision,
compare persistent KDA/KV state and first differing decode inputs/routing,
then replay the identified operation. Preserve numerical floors and the
completed benchmark. Use ROOT/full-gsm8kfullv1-src as the tested source;
its moe_runtime.py is 960e5febe55af6b1e998b55e3665a59b0d0ac34b0d46a0fb5912e237f928cb62.
The current local worktree has another MoE runtime revision; assembling a
new run must retain the native-library-qualified implementation explicitly.

The preceding graphqualityv1 stopped after four **reference-only** legs:
KL 0.000006310838 / 0.000005206705 / 0.000011791085 against the fixed
0.000010000 ceiling. All generated ids matched; candidate and longer work
were never run. Originals were restored; Spark verified 587 archive payloads.
Its broader quality plan below is historical, not the active benchmark.

The [matched graph-four quality batch](GLM53_MATCHED_GRAPH_QUALITY.md) is
complete as a failed reference-only check. All 256 retained distribution rows
independently verify: prefill rows match bytes, while autoregressive decode
drifts. Its candidate, longer PPL/task phases and speed grid never ran.

**Complete: controlled batching / batch-four graph diagnostic, 17 October.**
The original KDA operation with graph captures [1,4] improves real API C4
throughput **5.32% / 8.22%** across two interleaved timing blocks, TTFT
**-1.49% / +1.92%**. Original eager self-repeat, graph self-repeat and their
cross-comparison pass the unchanged numerical limits. The direct-wrapper
addition fails two KL checks and is rejected; the all-modes combined gate
remains false. Full release qualification still retains its historical failure.

Both-rank traces prove three actual graph launches/rank and show M4 median
GPU spans falling from about 86.9 to 78.8 ms. All 29 controlled cohorts,
384 raw-response distributions, 336 source-scale groups, twenty timing groups
and 711 archived payloads independently verify. Memory is 166.4987 GiB across
TP2; this graph/eager comparison is on the same compressed model, not a new
compression-versus-stock benchmark. Exact original GLM configurations are
restored and healthy at **14:31:28 UTC**. No experiment remains active. Read
[the frozen protocol and exact paths](GLM53_CONTROLLED_BATCH_EXPERIMENT.md).
The earlier failed source preparation is preserved; no GPU inference or GLM
pause happened in that rejected preparation. Keep the graph path experimental;
next full qualification needs a matched graph-four raw-runtime control and a
prospective broader quality protocol, not another direct-wrapper timing loop.

**Completed: one-load eager dispatch model comparison, 17 October.**
GLM is restored and healthy at 13:20:25 UTC; all 555 archive payloads verify
on Spark and locally. Incremental throughput gain is 2.32% / 0.83% across
both timing orders; TTFT varies. Strict numerical checks fail, including
original-dispatch repeats (KL 5.258e-5 against 1e-5). All generated IDs match.
Only nine of sixteen quality groups actually prefill four requests together;
seven split. Both-rank profiles confirm different admission patterns. Next
control actual batch formation and validate batch-four graph execution in the
separate completed diagnostic above.

The eager-only component passed both fixed 24-case processes: host submission
is 44–50% lower, capture/compilation retain the original op, and the same 2%
timing bounds pass with explicit graph address controls. All 291 component
archive payloads verify. Neither earlier unrestricted-direct failure is amended.

The model comparison uses one resident v9 candidate, switches modes between
closed request groups, checks concurrent real-KV distributions first, then
interleaves timing and collects separate both-rank profiles. No stock model
baseline reload, PPL repeat or new task battery. Independent restoration is
armed before the authorized idle GLM pause. Read
[the frozen protocol, source bindings and exact monitoring paths](GLM53_EAGER_DISPATCH_EXPERIMENT.md).
ROOT/CURRENT_LOSSLESS_STATUS.json and the controller retain the completed job.
KDA CPU scope totals fall 23–35%, but M4 GPU spans stay about 101 ms. Read
the completed result before attributing the remaining latency to host wrappers.

**Completed: KDA capture dependency/address diagnostic, 17 October.**
The isolated component v2 finished both fixed processes: all 24 correctness
cases pass in each, and host enqueue is 45–51% lower, but some input graph
blocks remain 4–6% slower. Both performance gates fail; do not integrate the
direct wrapper. Original GLM configurations are restored with health 200 at
12:32:07 UTC. All 271 final evidence payloads are verified; earlier packaging
path failures remain preserved, without measurement reruns.

The completed component retains raw CUDA graphs, inspects all nodes and edges,
records actual native arguments, and matches addresses through a shared pool.
Both graph topologies match, with shared-address replay within 0.2%; no extra
wait/event nodes. GLM stayed running. The scoped eager-only component and its
model comparison supersede this diagnostic as the active work. Read
[the frozen protocols and monitoring paths](GLM53_KDA_DIRECT_DISPATCH.md#next-dependencyaddress-diagnostic),
ROOT/kda-topology-v1-controller.json for the completed measurements.

**Latest follow-ups complete, 17 October.** Five fixed practical-task rounds
per arm show reference variability at the previously lost format cases.
Strict totals are 152/170 reference and 150/170 candidate; math is **108/120
in both**, functional code and real tool calls are **10/10 each** in both.
Every case passing all five reference rounds also passes all five candidate
rounds. The aggregate strict gap is slugify's Markdown formatting. These are
34 repeated questions, not 170 independent benchmark cases, and the preceding
strict failure stays unchanged. All 340 raw responses were independently
regraded. A CPU scorer compatibility error after inference required an
append-only analysis repair; no inference rerun or frozen-source mutation.
Both original GLM configurations were independently verified restored at
**11:56:51 UTC**. Read the [results, repair bindings and exact paths](GLM53_PRACTICAL_REPEAT_DIAGNOSTIC.md).

The [single-dispatch KDA component](GLM53_KDA_DIRECT_DISPATCH.md) also completed:
24/24 exact cases in both fresh processes and roughly half the host enqueue
time, but confirmation fails the unchanged graph limit (+4.35% on one
projection). It remains unintegrated. Follow-up Torch traces show identical
GPU activity signatures; even two unchanged reference graphs differ by 2.67%
in one timing block. The next bounded component check should stabilize paired
graph timing before another adapter or full-model change. Original GLM was
not stopped for either component. The sealed stable-quality handoff folder
remains an immutable historical snapshot; use the current documents for follow-ups.

**Current, 17 October: stablequalityv1 is complete and independently verified.**
The best U8 combined candidate saves **9.37 GiB / 5.33%** across TP2,
raises single-request throughput **11.4–11.9%**, and has near-parity eager
concurrency-four throughput (+0.6%). TTFT is **2.3–7.1% higher**. Reference and
candidate use the same qualified canonical dispatch patch. The unchanged
graph/eager KL/NLL gates now pass; all four held-out runs have byte-identical
float32 scores over 32,768 tokens/run, PPL 4.30501528.

Strict overall qualification remains **failed**. Both score 30/34 strict tasks:
candidate loses slugify's bare-code format and one correct math answer's final
marker format, while gaining two math passes (one format repair, one arithmetic
repair). Both code functions execute correctly and both tools pass. Do not
replace these frozen failures with semantic reinterpretation or repeat until
a favorable batch appears. No end-to-end bitwise equivalence is claimed.

Eight new both-rank Torch traces identify **36–37 ms of BF16 KDA prefill
reconstruction** per rank at 258 tokens. Canonical dispatch adds only about
0.17 ms there and skips one-token decode. Eager M4 host scheduling remains a
lead, not a proven removable penalty. All five timing trials remain; a slow
trial in each arm contains a roughly 2.1 s gap with no matched JIT log evidence.

Both original GLM containers/configuration digests are restored, API healthy,
and no experiment/timing clients remain; independently checked at
**10:55:39 UTC**. All **959 final archive payloads**, 672 full source-scale
group comparisons, quality metrics and task scores are verified. No GPU job
is pending from this batch. The Mac sleep blocker remains separate.
Read [complete results, trace attribution, pinned paths and next steps](GLM53_STABLE_DISPATCH_EXPERIMENT.md)
before another launch. Analyze saved evidence first; a new batch-four graph
configuration would require its own matched reference.

**Earlier, 17 October at 09:56: reference diagnostic complete.**
Strict deterministic mode still fails the unchanged repeat-KL floor.
All 360 fixed-input expert GEMM replays are bit-exact. Logical routes and
MLP inputs match in layers 3–5, while the consumed token permutation varies;
first floating differences affect one BF16 value in 4–8 million values.
All 1,549 private archive payloads and independent decode calculations pass.
GLM is restored with matching original identities, health 200 and no active
experiment engines, independently checked at 09:56:58 UTC. No new throughput
baseline, candidate sweep or relaxed gate.
The subsequent controlled projection swap and canonical ordering qualification
are complete; their matched whole-model outcome is summarized above.
See [results, protocol and paths](GLM53_REFERENCE_REPEAT_DIAGNOSTIC.md).

**Earlier, 17 October at 09:22: original paired quality measurements complete.** Best U8
combined storage saves **9.367 GiB / 5.326%** over stock across TP2, with exact
scale reconstruction before/after inference. Sampled PPL is slightly lower,
math is 22/24 in both arms, and the same 28/32 completed strict cases pass.
No reference pass is lost; two tool requests are unmeasured API rejections.
Strict numerical qualification still fails because the reference and candidate
repeat beyond the unchanged KL/NLL floors. This is not proof of accuracy loss.
No throughput sweep was repeated. Both original GLM identities/configuration
digests match, health 200 and no active experiment containers, independently
checked at **09:22:22 UTC**. All 826 archive payload hashes, 672 source-group
comparisons, numerical metrics and task scores are independently verified.
Controller `ROOT/quality-model-controller-qualitybestv2.json` is `complete`.

Read the [quality results, source bindings and next diagnostic](GLM53_BEST_CANDIDATE_QUALITY.md).
First visible reference repeat divergence is the prefill output of layer 4's
MLP on both ranks; candidate repeat first differs at layer 5. The original
hooks retain output hashes only. The new diagnostic captures input and routing/component
boundaries in one targeted stock load, then replays actual fixed-input expert GEMMs;
do not rerun the throughput baseline to diagnose this. Two measured eager
repeats also vary, so disabling graphs alone does not remove it.

The local launcher now activates and validates both tool flags. Frozen v2
omits `--enable-auto-tool-choice`; a CPU test in the exact image demonstrates
that its tool parser remains absent despite `--tool-call-parser glm47`.
Do not count both API failures as matching model quality. The local analyzer
adds an explicit completed-inference requirement; the frozen v2 review already
fails numerical qualification. Future runs must use the corrected launcher.
The first failed attempt's TorchVersion metadata is also fixed and preflighted.
Earlier restoration statements below describe completed historical batches.

**Latest, 17 October: exact nibble staging was tested and rejected on latency.**
The device-selected layout costs 26.8–28.9% at concentrated M128; selecting
the layout once in `NativeMoe.bind()` reduces this to 8.2–8.7%, still worse
than the earlier U8 shared-stage result of 5.3–7.6%. At M512 the static
variant costs 5.3–7.2%. Both variants pass 56 real-layer exact-output cases,
eight synthetic range-crossing fallback GEMMs and 48 decoder stress cases
each. No new model load, model throughput, allocation or quality qualification.
Read the [decision, source bindings and counters](GLM53_SCALE_PALETTE_DIAGNOSTIC.md#nibble-staging-and-static-layout-selection)
and [compact review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/expert-scale-nibble-stage-v1-review.json).

Both batches are complete. Both original GLM identities/configuration and
mount digests match, health 200, with no active batch containers, independently
verified after the final batch at **08:13:40 UTC**. Raw evidence archives are
`ROOT/scale-nibble-evidence-v1.tar.gz` and
`ROOT/scale-nibble-static-evidence-v1.tar.gz`; hashes and source paths are
in the linked diagnostic and compact review. No experiment is running.

The best measured palette consumer remains `scale-sparse-native-v9`, U8
shared staging. The nibble builds `v10` and `v11` remain opt-in experiments;
do not select them for the next full-model run. The static variant requires
an explicitly bound exact U8 consumer for unsupported source ranges.
Next isolate operator-level repeat-control nondeterminism and fix the tool
parser/interval task contract before any targeted full-model quality run.
Do not repeat the throughput sweep for this failed component lead. Further
kernel work needs a producer-overlap or source-level stall experiment;
the counters do not establish occupancy loss or physical LPDDR saturation.

**Preceding shared-stage result:** all 56
real-layer GPU cases pass exact outputs. Concentrated M128 regression
improves to 5.3–7.6% (previous direct decoder 13.5–21.6%); M512 improves to
2.2–2.8% (previously 10.4–11.1%). Uniform routing is near parity. The strict
2% latency gate still fails. Source scale storage and FP4 codes are unchanged;
no new whole-model run or quality claim is made. NCU instruction counts
fall 16.6% / 15.1% for gate/up and down; registers/shared-memory allocation
remain unchanged. Read the
[pipeline result and next candidate](GLM53_SCALE_PALETTE_DIAGNOSTIC.md#shared-stage-pipeline-follow-up)
and [verified compact review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/expert-scale-shared-stage-v1-review.json).

The successful `sparsepipelinev2` profile/timing/profile batch uses one
GLM pause. Both original identities/configuration digests are independently
verified restored, health 200, at **07:39:52 UTC**, with no active batch
containers. `ROOT/scale-pipeline-evidence-v1.tar.gz` retains hash-verified
timing, counter, source/license, failed-preparation and restoration evidence.
The exact shared nibble follow-up is now evaluated above, including full
source-range checks and U8 fallback. It does not pass the latency gate.
No production or adapter default changes. The preceding full-model result follows.

**Measured, 17 October: denser expert scales save another 2.220 GiB in the full model.**
The complete 42-layer CPU audit reconstructs all 36,288 source scale tensors
exactly. Palette payload is 2,384,481,712 rather than 4,794,092,128 bytes for
our current codec: **2.244 GiB additional saving across TP2**. This is scale
payload; the separate full-model delta is below. Guarded GPU builds pass
all 28 cases plus 28 confirmation cases; metadata caching and an affine-only
consumer still fail latency nonregression. Small decode is near parity;
concentrated M128 is about 13–21% slower in the latest component build.
NCU attributes the earlier build's regression to increased dynamic work,
with unchanged register/shared-memory occupancy limits and no spills.

`palettemodelv1` completes a separate source/answer/quality check at the user's
request, without a new throughput sweep or performance acceptance claim.
Both arms pass loaded GPU source checks before/after inference and
graph/eager chats. Allocation is **168.719 → 166.499 GiB**, a 2.220 GiB /
1.316% additional saving over the existing combined adapter across both
Sparks. All five answered practical tasks have identical final text, and all
64 generated IDs match. The interval scorer fails identically on tuples
versus lists (the prompt's return container is ambiguous); the tool request
is rejected before inference because the launch omitted `--tool-call-parser`.
The full-vocabulary comparison remains **unqualified**: reference-repeat KL
is 0.00005497, above the unchanged 0.00001 floor; candidate KL is 0.00006405
and its mean NLL difference is +0.00013973. Storage is exact; end-to-end
bitwise identity and accuracy equivalence are not established.
Both arms keep the same combined KDA/vocabulary/dispatcher settings.
Both original GLM identities/configuration digests were independently
verified running, health 200, at **07:08 UTC**, with no active containers
from this run. No default adapter is changed.
State: `ROOT/palette-model-controller-palettemodelv1.json`; source:
`ROOT/full-palettemodelv1-src/`; reports: `ROOT/run-palettemodelv1/results/`;
restoration: `ROOT/restoration-palettemodelv1.json`. Read the
[palette diagnostic](GLM53_SCALE_PALETTE_DIAGNOSTIC.md) for source locations,
the terminal-prefetch fix, full audit, NCU evidence and qualification limits.
The hash-verified component archive is `ROOT/scale-palette-evidence-v1.tar.gz`;
its [index](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/expert-scale-palette-evidence-v1-index.json)
records the latest component gates, ratios, native source/license hashes and
the archive digest. Keep its older-build NCU attribution distinct from
latest-build unprofiled timings.
The [full-model review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/expert-scale-palette-full-model-v1-review.json)
preserves strict failures and practical task caveats. Its 11 JSON input
hashes and 256 full-vocabulary arrays verify locally; independent CPU
recomputation reproduces the metrics and flags. Do not redo throughput
baselines to resolve numerical nondeterminism. First isolate a deterministic
operator/control path, fix the tool launch contract and reduce component
prefill overhead. Sustained serving and longer contexts remain untested.

**User priorities, 17 October:** quality is a hard acceptance gate; evaluate
memory and latency tradeoffs without accepting an accuracy regression.
Matching a few answers or exactly reconstructing weight bytes is not a
substitute for the numerical and practical quality gates. The current
qualification failure means quality equivalence is unresolved, not that
an accuracy decline has been demonstrated.

Keep original GLM paused across a planned batch of variants; restore it
once the batch finishes or the user needs the service. Do not restart
production between each variant. `run_scale_pipeline.py` now implements a
prepared multi-job component batch with one pause/watchdog and final
restoration. Each job has its own frozen `bf16-kernels-*` source/report root
and kind `counters` or `timing`. Both native libraries must resolve and hash
correctly inside the CPU-only container preflight; use relative links within
`/audit`, not absolute host-only symlinks. Only exited owned component IDs
are removed between arms after logs/state are saved. Older controllers still
restore per invocation and should not be used to chain variant sessions.
No further model run is active.

**Earlier BF16 follow-up:** `n8compactv2` passes all 14-fixture bit-exact checks but
regresses M4 input latency about 9–12%. The 2-KiB shared-memory reduction
does improve CUDA's residency limit from three to four blocks; that did not
translate into speed. It is rejected and no default adapter changed.
The four-variant `n8mechanismv1` comparison is also complete: every candidate
passes correctness and fails the repeated performance gate. The narrow load
alone is near parity; compact staging remains slower with either load width.
Tail-first input scheduling gives only a small warm gain, without a matching
cache-pressure gain. No new full-model benchmark is justified for these variants.
Both original GLM identities/configuration digests are independently verified
running, health 200, with no active experiment containers. See the
[kernel follow-up](GLM53_BF16_KERNEL_DIAGNOSTIC.md#compact-n8-activation-staging-correctness-passes-speed-fails)
and [raw-sample review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/bf16-n8-mechanism-v1-review.json).

The user's paper suggestion now has a concrete CPU result. In 27 sampled
actual GLM expert matrices, FP4-code entropy is **3.915 bits/code**; an
illustrative 32-stream-per-2-KiB rANS layout saves **−0.30%** after metadata
and raw fallback. Scales save **83.12% relative to checkpoint U8 scales** in
that layout, not relative to model memory or our current compressed scales.
Prioritize a comparison with the existing aligned-nibble runtime representation
before writing a GPU rANS path. The [screen and limits](GLM53_BF16_KERNEL_DIAGNOSTIC.md#paper-follow-up-and-actual-mxfp4-compressibility)
explain why the paper's headline low-bit savings cannot be transferred to this
checkpoint. The read-only screen uses no GPU and never paused GLM.

Current evidence on Spark1: `ROOT/bf16-kernels-n8mechanismv1/{src,n8.json,n8-confirm.json}`,
`ROOT/bf16-kernels-controller-n8mechanismv1.json`,
`ROOT/restoration-n8mechanismv1.json` and
`ROOT/n8mechanismv1-final-service-check.json`.
The head's persistent `agent-handoff-2026-10-16/analysis/n8-entropy-v1/`
contains the latest offline analyzers and reviews. The CPU screen is at
`ROOT/entropy-screen-v1/{screen.py,results/entropy.json}`.

The palette comparison above implements the scale-storage follow-up. Its
component latency gate failed; the full-model diagnostic is solely to
establish live allocation, practical behavior and numerical limits.
For the input slowdown, reserving the original launch allocation with compact
internal buffers would distinguish layout from increased residency, if further
causal attribution is needed. Budget shared memory before a producer/consumer
decode/MMA experiment. These are proposed follow-ups, not implemented fixes.
Reuse existing throughput measurements until a component lead passes both
cache conditions and the unchanged quality/stream/graph gates.

**Preceding result, 16 October 2026: the BF16 component/NCU comparison
`bf16kernelsv2` and output cache check `cachepressurev2` are complete.**
Both original GLM identities/configuration
digests were independently verified running at **20:09 UTC**, health 200,
with no active GPU experiment. Read the
[kernel diagnostic](GLM53_BF16_KERNEL_DIAGNOSTIC.md) and
[compact review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/bf16-kernels-v2-review.json)
first. No full-model baseline or candidate was loaded, and no adapter changed.

At M4 in warm component replay, input is near stock (0.459 vs 0.452 ms), output is slower
(0.150 vs 0.133 ms) and the head is faster (2.438 vs 2.813 ms). Split-one
passes all 9 component cases; all 27 higher-split diagnostics fail the
existing numerical gates. NCU counter access works in an isolated container
without changing the host driver setting. Output's 64-block grid reaches
only 11.3% occupancy; input is limited to three resident blocks by shared
memory. Prioritize output work distribution preserving K accumulation,
then the input's shared-memory footprint. The head is lower priority.
The exported counter set does not expose direct LPDDR byte traffic.

**Cache-pressure qualification:** `cachepressurev2` changes the unchanged
output kernel from 12.5% slower when warm (138.35 vs 122.98 µs) to 4.5%
faster after a 128 MiB eviction workload (192.22 vs 200.79 µs). Both arm and
cache-treatment orders agree, and before/after output bytes are exact.
Output's low occupancy is a lead, not proof of a general compression penalty.
Require proposed kernels to win in both representative cache conditions.
The initial resident-GLM attempt failed CUDA context creation without touching
test weights; the successful follow-up used the restoration watchdog.
Source/state/report: `ROOT/bf16-kernels-cachepressurev2/{src,cache.json}` and
`ROOT/bf16-kernels-controller-cachepressurev2.json`.
Final restoration/check: `ROOT/restoration-cachepressurev2.json` and
`ROOT/cachepressurev2-final-service-check.json`; both original configurations
and the host profiling restriction are unchanged. The
[cache review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/bf16-output-cache-v2-review.json)
retains both cache and arm orders.

Source: `ROOT/bf16-kernels-bf16kernelsv2/src/` on both nodes; raw component
evidence on Spark2 in `results/` and copied to Spark1 in that run's `evidence/`.
State: `ROOT/bf16-kernels-controller-bf16kernelsv2.json`; restoration:
`ROOT/restoration-bf16kernelsv2.json`; independent health/identity check:
`ROOT/bf16kernelsv2-final-service-check.json`. The diagnostic gives offline
analysis commands. Establish a correctness-preserving component improvement
before another model load; quality remains unqualified.

**The preceding full-model trace follows.**

**Previous full-model result, 16 October 2026: the candidate-only `nsysv2` diagnostic is
complete.** Both original GLM identities/configuration digests were independently
verified running at **19:14 UTC**, health 200, with no active GPU experiment.
Read [the Nsight diagnostic](GLM53_NSYS_MODEL_DIAGNOSTIC.md) and
[compact measurements](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/nsys-model-v2-review.json)
first. It used one candidate load and no fresh stock control.

Before/captured/after B4 throughput was **39.30 / 37.80 / 39.02 tokens/s**.
Middle-decode streaming latency was **84.03 / 90.28 / 83.21 ms**; capture
perturbation is measurable. All 44 timed completion sequences and both source
checks pass. Formal quality remains unqualified from the earlier unstable
stock control. Per M4, fused KDA kernels take about **21–22 ms**, and the
separate head **2.39 ms**. These include useful computation; establish
recoverable overhead with a component benchmark before another model load.

Python stacks identify the existing sparse-attention planner's 32-byte
GPU-to-CPU position copy. Its device transfer is about 1.5 microseconds;
long host waits include preceding GPU work and are not free speedup. There
were no GC collections during capture, and main-thread runqueue wait was
only 1.44 / 1.05 ms across the whole capture bracket. Proxy/rank waiting and
profiling effects remain relevant. Prioritize BF16 kernel component profiling,
then exact attention-metadata reuse; do not blindly repeat Python/proxy tweaks.

On Spark1, `ROOT` is the experiment root in section 3. Current state:
`ROOT/nsys-controller-nsysv2.json`; source on both nodes:
`ROOT/full-nsysv2-src/`; timing: `ROOT/run-nsysv2/results/nsys-measurement.json`;
head/worker traces on head: `ROOT/run-nsysv2/kda/{traces,traces-worker}/`.
Restoration: `ROOT/restoration-nsysv2.json`; independent check:
`ROOT/nsysv2-final-service-check.json`. Private archive:
`ROOT/nsysv2-evidence-v1.tar.gz` (123 hash-verified payloads, SHA-256
`8b608cc06798c28d21fdd625f989d7e66c9a16cd469b10815137415f8f4b4239`).
The detailed diagnostic provides exact offline-analysis commands and limits.
Local ruff, strict core mypy and pytest pass (one existing GPU skip).

**Earlier experiments and interpretation follow.**

Updated 16 October 2026. **The pinning component `pinningv2` is complete and
failed its full-model follow-up gate.** GPU-spaced forward changes were within
about 0.4%; a secondary CPU-sleep-spaced case improved 2.0–4.4%. Both original
GLM identities/configurations were independently verified running at
**18:11 UTC**, health 200, with no active experiment containers.
No additional full-model comparison was launched.

**Further offline attribution:** only 9.6% of candidate rank-0 late-launch
time is temporally inside adapter scopes; most is in stock/compiled runtime
or between operators. This is not causal exoneration. A CPU-only Nsight probe
verified OSRT/native call chains; the completed full-model Nsight diagnostic is summarized above.
Read the adapter-versus-engine subsection before choosing another change.

The dispatcher follow-up `combineddispatchv1` is also complete; its throughput
regression remains. B4 throughput changed only
38.82 → 38.89 tokens/s, versus 39.87 stock, despite substantially lower host
submission time. Both original GLM identities/configurations were independently
verified running at **12:23 UTC**, health 200, with zero active experiment
containers at that check. The completed diagnostic is described above.

**Synchronization follow-up:** paired collectives strongly confirm rank-arrival
waiting. The model experiment omitted production's `GLM_ROCE_PROXY_CPUS=auto`
setting. Inheritance/control-reuse guards are fixed and checked. The completed
component does not demonstrate pinning as a fix for the model's slowdown.

A persistent copy of this handoff and its linked reports is on Spark1 at
`/home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/full-model-v1/agent-handoff-2026-10-16/`.
Read `docs/GLM53_LOSSLESS_RUNTIME_HANDOFF.md` there when the Mac is unavailable.

## 1. Objective and current verdict

Continue reducing the overhead introduced by MxWave's runtime storage adapters.
The dispatch intervention is complete. Prioritize prefill and slower decode
intervals, then quality qualification and packaging. The user wants a practical result and has asked that existing
measurements be reused instead of repeatedly loading stock baselines.

The current candidate combines expert E8M0 scale packing, KDA BF16 compression,
an untied compressed BF16 head and row-addressable BF16 embeddings. Expert FP4
codes and KV storage are unchanged. These are runtime storage adapters around
an already quantized checkpoint, not a new GLM quantization recipe.

| Measured aspect | Result |
|---|---|
| Live CUDA allocation, both nodes | 175.87 → 168.72 GiB; **7.15 GiB / 4.06% saved** |
| One request, 256-token prompt | 15.32 → 16.76 tokens/s; **+9.3%** |
| One request, 2,048-token prompt | 12.88 → 13.94 tokens/s; **+8.2%** |
| Four requests, 256-token prompts | 39.87 → 38.89 tokens/s; **−2.5%** |
| Correctness | Both-rank source checks, 9/9 chats, 30/30 timed token sequences pass |
| Actual decode quality | 64/64 generated ids match; 48 identically conditioned decode steps |
| Mean decode KL | Candidate 0.000042516; stock-repeat 0.000024518 |
| Formal quality status | **Unqualified: unstable stock control**, not a passed release gate |
| Checkpoint read | 629.52 → 40.06 seconds with the existing production slab reader |

Throughput includes prefill and uses five-trial medians. The stock control is
saved/sequential, not interleaved. TTFT increased; see the
[concise results](GLM53_LOSSLESS_RUNTIME_RESULTS.md) for the complete table.
The stock-repeat KL exceeds the predeclared 0.00001 stability floor. Do not
relax that floor, describe matching tokens as an accuracy benchmark, or equate
exact restored weight bytes with identical GEMM rounding.

## 2. Work in this checkout

On the Mac:

```bash
cd /Users/kyrylobilchenko/Documents/Projects/other/mxstream/.worktrees/glm53-flash-mxfp4
git branch --show-current
git status --short
cat AGENTS.md
```

Branch: `feat/glm53-flash-mxfp4`; base HEAD:
`a99b3621c19cc76cf89d4d81607d82789f78df0a`.
**Much of this experiment is untracked/uncommitted.** The branch commit alone
does not contain it. Preserve the working files; use this worktree rather than
the parent `mxstream` checkout. The qualified remote snapshots are an additional
recovery copy, not Git checkouts.

Read these first:

- [Results and interpretation](GLM53_LOSSLESS_RUNTIME_RESULTS.md).
- [Detailed investigation history](GLM53_FLASH_MXFP4_REUSE.md), when needed.
- [Measured result JSON](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/full-model-combined-v2-review.json).
- [Both-rank trace analysis](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/full-model-combined-trace-review-v1.json).
- [M4 launch correlation check](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/full-model-combined-launch-gaps-v2.json).
- [Host dispatch and helper-kernel analysis](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/full-model-combined-dispatch-review-v1.json).
- [Archive index and hashes](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/combined-evidence-v1.json).

The earlier Qwen work is in sibling `.worktrees/lossless-mxfp4`;
`docs/LOSSLESS_STORAGE_EXPERIMENT.md` there now includes the GLM follow-up.
Keep Qwen TP1 and GLM TP2 results separate.

## 3. Machines, model and runtime

| Item | Location / identity |
|---|---|
| Head | Mac SSH alias `spark`; `spark-2ed0`, `100.95.22.43` |
| Worker | Mac SSH alias `spark2`; `spark-7e25`, `100.113.61.32` |
| Private TP link | Head `192.168.192.1`; worker `192.168.192.2` |
| Host user | `kirya` |
| Experiment root, both hosts | `/home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/full-model-v1` |
| Imported checkpoint | `source/` under that root; already staged on both hosts |
| Completed qualified source | `full-combinedv2-src/` under that root |
| Results / traces | `run-combinedv2/results/`; `run-combinedv2/{stock,kda}/{traces,traces-worker}/` on head |
| Vocabulary artifacts | `vocab-v2/`, including `prepared.json` and all four shard artifacts |
| Native libraries | `native/` under experiment root |
| Architecture overlay | `/home/kirya/glm53-runtime-20261010-long` |
| Existing ZipServ/runtime workspace | `/home/kirya/local-spark/experiments/zipserv-vllm031-2026-10-14` |

The imported model is `INCModel3/GLM-5.3-Flash-MXFP4-Mixed-CT-AutoRound`,
revision `35d1b0d5cb4a0db075fd2749c897e970dc15bfa5`: 120 shards,
183,429,201,056 bytes. The running production model is a **different** ModelOpt
mixed NVFP4 checkpoint at:
`/home/kirya/local-spark/models/glm-5.3-flash/derived/glm-quant-mix/lossless8`.
The imported-model results do not qualify compression of that serving format.

Pinned image:
`sha256:39a03677234ea74e35544d301bd14716c1ae9695ad8d91b16e2c76545e26e8d2`.
Inside it use `/usr/bin/python3`, vLLM `0.1.dev20051+g487ecf187`, Torch
`2.13.0+cu130`. Do not substitute the older Qwen experiment's vLLM 0.31 venv.
The experimental engine listens on head loopback **18784**; production is **8000**.

Native pins are in `integrations/glm53_mxfp4/full_model_control.py::LIBRARIES`.
The BF16 library is `native/libmxwave_kda_zipserv.so`; raw/aligned expert
libraries are `libmxwave_moe_raw.so` and `libmxwave_moe_aligned.so`.
The external Python ABI is
`OLD/triple-v2/src/integrations/zipserv/mxwave_zipserv/runtime.py`, where `OLD`
means the existing ZipServ workspace above. `prepare_bf16_attention.py` binds
both BF16-library and runtime hashes. Preserve the external licenses recorded
in [CONTRIBUTING.md](../CONTRIBUTING.md); no external kernel source was vendored.

### Initial read-only check, run on Spark1

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 spark
```

Then, in that remote shell:

```bash
MXWAVE_RUN_ROOT=/home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/full-model-v1
python3 - "$MXWAVE_RUN_ROOT" <<'PY'
import json, pathlib, sys, urllib.request
r = pathlib.Path(sys.argv[1])
s = json.loads((r / 'full-controller-combineddispatchv1.json').read_bytes())
print({k: s[k] for k in ('phase', 'service_paused')})
print('restoration:', s['restoration']['status'])
print('production health:', urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status)
PY
docker ps --filter name=mxwave-glm53 --format '{{.Names}} {{.Status}}'
docker ps --filter name=local-spark-glm53 --format '{{.ID}} {{.Names}} {{.Status}}'
ssh -o BatchMode=yes -o ConnectTimeout=8 \
  -i /home/kirya/.ssh/id_glm_cluster kirya@192.168.192.2 \
  "docker ps --filter name=local-spark-glm53 --format '{{.ID}} {{.Names}} {{.Status}}'"
```

Original container names are `local-spark-glm53-tp2-long500k-20261013-r0` and
`...-r1`. Exact IDs are `full_model_control.py::ORIGINAL_IDS`; the completed
controller stores their original configuration digests. Use the controller's
identity checks rather than recreating those containers or dumping full envs.

## 4. Code map and contracts to preserve

All integration filenames below are under `integrations/glm53_mxfp4/`.

| Files | Responsibility / contract |
|---|---|
| `attention_mapping.py` | Registers missing fused attention aliases before CT method selection; essential correctness fix |
| `model_plugin.py`, `moe_runtime.py` | Post-Marlin aligned-nibble expert scales, opaque grouped GEMM, exact decode proof, ownership and scratch reporting |
| `kda_policy.py`, `kda_adapter.py`, `kda_runtime.py` | Exact 68-prefix selection, stock TP loading, fused BF16 execution up to 8 tokens, bounded fallback above 8 |
| `vocab_policy.py`, `vocab_adapter.py`, `vocab_runtime.py` | Exact head/embed selection, stock TP mask/reduce/gather, source-bound artifacts and original-weight release |
| `mxwave/lossless_bf16_rows{,_gpu}.py` at repo root | MxWave row format, CPU reference and Triton row decoder |
| `combined_worker.py`, `kda_worker.py` | Coverage/source RPCs and graph controls; reconstruct views so compression cannot silently remove checks |
| `run_kda_model.py`, `full_model_control.py` | Detached full-model controller, saved-control validation, idle pause, watchdog and exact restoration |
| `run_diagnostic_idle.py` | Small Spark2 GPU components; cancels its own job if production becomes busy |
| `measure_combined_arm.py`, `measure_kda_arm.py` | Ownership → source/chat → timing/profile → actual-decode gates |
| `decode_quality.py`, `compare_combined_model.py` | Hash-bound actual decode distributions, identical-conditioning comparisons and unchanged quality gates |
| `analyze_combined_profiles.py` | Offline component attribution and complete, matched forward shapes |
| `analyze_dispatch_profiles.py` | CPU/CUDA call scopes, nested-time correction and KDA helper attribution for complete forwards |

Per rank, require 42 expert layers, 83 compressed scale groups plus the
layer-44 `w2` stock fallback, 68 KDA projections and two vocabulary components.
The current KDA pool is **206,569,472 bytes per rank**, two stream slots, capped
at 256 MiB. Fused execution uses split-K=1 for up to eight tokens. Preserve
current-stream launches, graph behavior, fake-op shapes, mutation declarations
and delegated behavior for all unowned tensors.

Candidate flags are assembled by the controller: `MXWAVE_MOE_MODE=aligned`,
`MXWAVE_GLM53_KDA_MODE=packed`, `MXWAVE_GLM53_KDA_FUSED_MAX_TOKENS=8`,
`MXWAVE_GLM53_VOCAB_MODE=packed`, `MXWAVE_GLM53_ATTENTION_MAPPING=1`, and
hash-bound vocabulary/library paths. Keep `GLM_FAST_LOAD=1` inherited from
production. Read the saved sanitized launch metadata for the remaining flags.

## 5. Findings that determine the next work

The additional review uses the **same four hash-verified worker traces**.
Each contains 26 complete forwards, including five M4 decode forwards;
selection requires all 84 routed-expert GEMM kernels on the annotated stream.
That offline review ran no new model; the subsequent candidate-only test is
recorded below. Detailed original values are in the
[dispatch report](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/full-model-combined-dispatch-review-v1.json).

### GPU work and rank imbalance

| Profiled measurement | Stock | Combined | Interpretation |
|---|---:|---:|---|
| Matched one-token forward, rank 0 | 59.03 ms | 55.24 ms | Decode GPU path improves |
| M4 forward range, rank 0 | 93.00 ms | 104.16 ms | Eager execution regresses in this profile |
| M4 routed-expert kernels, rank 0 | 35.20 ms | 35.02 ms | Expert GPU computation is comparable |
| M4 routed-expert kernels, rank 1 | 36.71 ms | 33.47 ms | No observed expert-kernel collapse |
| M4 uncovered stream time, rank 0 | 6.37 ms | 30.74 ms | Additional submission/wait gaps |
| M4 communication kernels, rank 1 | 3.14 ms | 25.85 ms | Consistent with peer/rank imbalance |

The one-token row uses 14 matching detailed shapes. M4 rows are medians of
five count-matched forwards; their detailed context lengths/scheduling differ.
Kernel sums can overlap and must not be added to elapsed forward time.
Communication durations do not establish a physical network bandwidth limit.

After including copy activities, rank-0 M4 gaps are 6.25 versus 30.64 ms.
Of those, **0.52 versus 23.99 ms occur before the next launch API is entered**.
This supports investigating host submission and rank balance. It does not
separate Python, native binding, driver work, instrumentation, OS scheduling
or CPU waits, nor promise a 24 ms application speedup.

### Tested target: the MoE host-call path

| Per-M4 host scope, same 84 expert calls | Stock | Combined |
|---|---:|---:|
| Inclusive scope time, rank 0 | 3.88 ms | 11.98 ms |
| Inclusive scope time, rank 1 | 4.87 ms | 11.64 ms |
| Residual after traced children, rank 0 | 1.48 ms | 11.44 ms |
| Residual after traced children, rank 1 | 2.40 ms | 11.12 ms |

Combined totals include 83 `mxwave_glm53::moe_gemm` calls plus the one stock
fallback. The residual removes the **union** of nested CPU/CUDA intervals,
so runtime/driver nesting is not counted twice. It remains untraced scope
time, not a measurement of pure Python or ctypes cost. These are instrumented
host spans, not unprofiled request latency.

That difference motivated the completed dispatcher intervention below.
Preserve the same kernel, inputs, FP32 reduction, output bytes, stream and graph
behavior. `_plan` is already LRU-cached; another tile-plan cache is not justified.
Cache only immutable metadata, never dynamic input/output/routing pointers.

**Implemented follow-up:** `MXWAVE_MOE_DISPATCH=dispatcher` selects a direct
PyTorch dispatcher registration with the same native launch, mutation schema
and fake-tensor behavior. The default remains `custom`. The paired installed-MoE
fixture `dispatchv1` passed all 20 raw/aligned scenarios, including M=4, with
both dispatches matching stock bytes in eager, non-default-stream and graph
execution. In aligned M=4, CPU submission fell from about 279 to 70 microseconds
per complete MoE block; elapsed block time improved about 1–2% in both orders.
These are component results, not a 75% model speedup. See the
[dispatch component review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/moe-dispatch-review-v1.json).

The changed plugin also passed a refreshed full vocabulary fixture,
`vocabdispatchv1` (all four GPU source hashes, 1.16 GB peak allocation).
`full-combineddispatchv1-src/` and `diagnostics-v1/src-vocabdispatchv1/` are
frozen on both hosts. The completed candidate-only run reused stock timing
`kdamodelv2` and actual-decode control `combinedv1`.

**Whole-model outcome:** MoE host spans fell **11.98 → 1.59 ms** on rank 0 and
**11.64 → 1.54 ms** on rank 1, across the same 84 calls. Yet B4 throughput rose
only **0.16%**, remaining **2.48% below stock**. Allocation saving is unchanged
at 7.15 GiB. All 30 timed completions and 64 quality ids match stock; quality
remains `unqualified_unstable_control`. Keep the new path opt-in: this is a
component optimization, not a demonstrated throughput fix.

The new profiles retain rank-0 gaps (30.74 → 16.55 ms) and rank-1 collective
spans (25.85 → 16.93 ms). M4 ranges improve 104.16 → 100.05 ms, but unchanged
KDA kernels also vary (21.44 → 24.57 ms on rank 0). M4 groups differ in detailed
context/scheduling; no live clock/power or DRAM/SM counters were recorded.

A post-hoc client-arrival check gives median middle-token intervals of
**84.77 ms stock / 84.75 ms dispatcher**, with mean intervals **85.58 / 86.80 ms**.
Startup, prefill and slower intervals deserve attention. This uses token
positions 8–55 and is not a replacement acceptance metric. B4 TTFT remains
1.021 → 1.099 seconds. See the [completed intervention](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/full-model-dispatch-v1-review.json)
and [evidence index](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/dispatch-evidence-v1.json).

Latest controller: `full-controller-combineddispatchv1.json`; restoration:
`restoration-combineddispatchv1.json`; independent check:
`combineddispatchv1-final-service-check.json`. The 105.86 MB raw archive
`combineddispatchv1-evidence-v1.tar.gz` contains 286 locally verified files,
SHA-256 `6737aef261b1d660c53ab42d7720d2e7faf51facd6d34e153df30e61d9b9c2e4`.
Offline reports are in `agent-handoff-2026-10-16/analysis/combineddispatchv1/`.
The same analyzers accept root
`/private/tmp/mxwave-glm53-dispatchv1-review/extracted/run-combineddispatchv1`.

### Synchronization: the peer really is waiting

The [synchronization review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/full-model-synchronization-review-v1.json)
pairs 91 main-stream RoCE all-reduces in each of five M4 forwards per arm/rank.
For each adjacent pair, compare rank 0's extra interval before the collective
with rank 1's extra duration inside it. Candidate correlations are
**0.9952–0.9996**, with median absolute residuals **0.80–9.95 microseconds**
per forward. This uses only per-host intervals/durations, avoiding cross-host
clock alignment. A concrete candidate call is **6.432 versus 1,040.485
microseconds**; rank 0 arrives about **1,033.885 microseconds** later, including
**793.577 microseconds** of late host submission. Waiting direction can change;
stock also exhibits it. The trace supports arrival imbalance, not a proven
network bandwidth limit or a predicted application speedup.

The running image's transport sources match the inspected hashes. Relevant
files are `/opt/glm-roce-b12x/b12x/comm/roce/{roce_oneshot.py,_oneshot_cute.py,_roce_proxy.c}`.
The one-shot kernel stages, rings a native CPU RDMA proxy, spins for peer
payload flags, then reduces. The runtime has one channel, ordered across
streams; dependent decoder layers cannot simply run their reductions in
parallel. The proxy polls a doorbell and can sleep after idle polling, so
proxy scheduling belongs in the next trace. The current traces do not contain
doorbell/RDMA or CPU scheduler timestamps.

Both arms' **84 stream-wait calls per M4** map to
`vllm::moe_forward_shared`. Their host API spans total roughly 45–59 microseconds;
this is not GPU wait duration. The single CPU stream synchronization per M4
belongs to `aten::copy_` and lasts about two microseconds. No additional
per-layer CPU synchronization was found in the compression adapters.
Latest copy-inclusive rank-0 gaps are **16.35 ms**, including **10.03 ms before
launch API entry**; rank-1 RoCE kernel spans total **16.93 ms**. These quantities
overlap across ranks and must not be added as independent overhead.

**Confirmed configuration omission:** production requests
`GLM_ROCE_PROXY_CPUS=auto` on both ranks; saved `kdamodelv2` and
`combineddispatchv1` launches omit it, and the pinned image does not set it.
The local `full_model_control.py` now inherits explicit public proxy/limit
settings. `run_kda_model.py` checks both ranks' transport/OMP environment
before timing or decode-control reuse. A changed pinning setting must reject
the old control even when command lines match. The old arms shared the same
omission, so their relative results remain historical measurements; pinning
has not been demonstrated to explain or fix the regression.

The launcher patch passed ruff, strict mypy and all pytest tests (one existing
GPU skip). The offline investigation itself made no production change;
the subsequent authorized component is recorded below. **The combined
harness still requires `--reuse-stock-run`: add a tested fresh matched-control
mode before a changed-transport full-model run.** Do not bypass the new guard.
The small TP2 communication fixture has now completed.

### Pinning component complete: preserve the negative primary result

Read [the paired component review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/roce-pinning-v2-review.json).
Source is frozen on both Sparks at `ROOT/transport-pinningv2/src/`, where
`ROOT` is the full-model experiment root in section 3. Controller state is
`ROOT/transport-controller-pinningv2.json`; reports are
`ROOT/transport-pinningv2/attempt1/rank{0,1}/report.json` on the head.
The controller and independent watchdog restore the same original IDs.

`pinningv1` failed before measurement during CUDA initialization while GLM
remained resident. The user explicitly authorized stopping GLM if necessary.
`pinningv2` paused at an idle window, increased the component host cap to
6 GiB, and completed 25,524 BF16 32 KiB all-reduces per rank in about 22.6 s.
No model weights were loaded in the component. All outputs matched exact
bytes, both transport error counters were zero, and both proxy affinities
were observed as all 20 CPUs versus CPU 19.

Six cases ran in balanced ABBA/BAAB blocks. GPU-spaced 91-collective wall
medians changed **−0.18% to +0.40%**, with mixed signs. The secondary
CPU-sleep-spaced case improved **2.00–4.40%**; its OS timer behavior is not
a real model forward. Isolated-call p95 was inconsistent. Both predeclared
lead/follow-up gates failed; no full-model reload was triggered for pinning.
Do not present the secondary result as a measured TPS gain or widen the gates.

The 102,745-byte `ROOT/pinningv2-evidence.tar.gz` has 22 locally hash-verified
payload files, SHA-256
`922ea22ef1c734c664ba2f49227ee991cf22dd98f6cde4322d11100209d4030e`.
`ROOT/pinningv2-evidence-index.json` and
`ROOT/pinningv2-final-service-check.json` preserve archive/restoration checks.
Original GLM IDs/configurations were independently verified running at
18:11 UTC, health 200, with no active experiments. All repository checks pass.

### Adapter versus engine: earlier scope ownership and capture design

[The ownership review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/gap-ownership-v2-review.json)
uses the same four traces and 20 M4 forwards. Candidate rank 0 has median
10.03 ms before launch API entry: 1.07 ms inside adapter scopes, 8.96 ms
outside. Across the five samples, 9.6% is adapter-contained. No identified
late interval overlaps any other traced GPU activity. Only about 0.55 ms
of the broader 16.35 ms main-stream gaps overlaps other streams.

Large ownership buckets are stock shared-expert execution, attention/indexer
calls, TorchDynamo and between-operator time. This is temporal attribution,
not proof of where CPU cycles were spent or which code caused the delay.
Adapter-induced allocation/lifetime/profiler effects could appear later.
Stock also exhibits skew on the opposite rank. Do not conclude either that
the adapter caused every gap or that it is cleared by the 9.6% result.

Both arms have 149 TorchDynamo cache lookups and 42 shared-expert calls per
M4. Candidate `aten::empty` count falls 2,818 → 2,309; contiguous/copy wrappers
rise 137 → 171 and 306 → 408, consistent with KDA padding/compaction. These
are CPU operation counts, not actual allocations or GPU transfer counts.
The original profiler disables stacks; shape recording defaults to false.

Nsight is `/usr/local/bin/nsys`, version **2025.3.2**, with installation
`/opt/nvidia/nsight-systems/2025.3.2/`. CPU-only probes are preserved in
`ROOT/nsys-cpu-smoke-v{1,2}/`; v2 export has 2,234 OSRT events and 5,610 native
call-chain rows. Extended-payload exports subsequently verified real Python
frames, and `nsysv2` collected both model workers. `perf_event_paranoid=4`, and tracefs access
was denied. Precise CPU scheduling capture has not been qualified; OSRT
reports inferred activity and cannot establish asynchronous descheduling.

The completed `nsysv2` diagnostic implements this design; see its linked results.
The design uses **one candidate boot**, holding graph
and transport settings fixed. Time a warmed B4 workload with Nsight capture
inactive, collect a short two-rank CUDA/NVTX/OSRT range, stop capture and time
again. This measures active-capture overhead within an Nsight-launched process;
it is not a new stock-versus-compressed qualification or a clean process with
no profiling preload. Do not run PyTorch/Kineto concurrently with Nsight.

The pinned vLLM supports `--profiler-config '{"profiler":"cuda","ignore_frontend":true}'`.
Its profiler start/stop can drive Nsight's `--capture-range=cudaProfilerApi`
and `--capture-range-end=stop`. Verified CPU-probe flags are `--sample=none`,
`--cpuctxsw=none`, `--trace=osrt,nvtx` and Python sampling at 100 Hz. A model
capture must add CUDA tracing, cover both child workers and verify actual
stack/event contents. Per-thread scheduler counters can supplement missing
scheduler events but do not replace them. Native/Python blocking, profiler
amplification and remaining KDA host work are the questions to resolve.

**This capture has not been implemented/launched.** The existing full-model
controller hard-codes Torch profiling and the combined harness requires a
saved stock control. Add a tested diagnostic-only mode before using it;
do not bypass transport/quality reuse guards or overwrite qualified snapshots.
The existing idle pause, watchdog and exact restoration remain required.

### Secondary target: KDA padding and output compaction

The pinned ZipServ bridge rounds M=4 up to an eight-row tile. Each M4 model
forward has these KDA-owned helper activities on **each** rank:

| Activity | Count | GPU duration, rank 0 / rank 1 |
|---|---:|---:|
| Activation padding fills | 68 | 64 / 62 microseconds |
| Device-to-device activation copies | 68 | 65 / 74 microseconds |
| Cropped-output compaction kernels | 34 | 72 / 72 microseconds |

The 68 activation copies move 2,228,224 bytes (2.125 MiB) per step. All counts
were verified in every complete M4 sample. They total **170 helper activities
but only about 0.20–0.21 ms of GPU time**. Their GPU bandwidth cost cannot
explain the observed slowdown; removing allocations and host launches may
still help. Masked activation loads and valid-width output stores are possible
native-kernel experiments, not implemented optimizations.

Source inspection is bound to the deployed runtime SHA
`2a296929bf8363ec6bd9358c8eadbfcca47b77522bf030c8096ab22bd6441c35`.
Its `padded_tokens()` / `_linear()` perform padding and copying; the 34 padded
KDA input projections compact cropped outputs in `kda_runtime.py::kda_linear`.
KDA host scopes total about 8.70 / 8.28 ms per M4 forward. The nested
`mxwave_zipserv::linear` scopes account for 4.70 / 4.50 ms **within those**;
do not add the two or mislabel the nested calls as vocabulary-head work.

### Shared engine waits are not compression instructions

Whole rank-0 traces contain roughly 2.1 seconds in host `cudaMemcpyAsync`
calls in **both** arms, while GPU copy activity totals only 8.73 versus
8.80 ms. Host copy-call duration can include waiting for preceding GPU work;
it is not a measured 2.1-second DMA bottleneck. M1 host annotations likewise
contain about 60 / 56 ms in that API for stock/combined. Graph replay has one
`cudaGraphLaunch` per M1 forward and no per-layer Python custom-op scopes;
M4 has no graph launch and executes those scopes eagerly.

These instrumented timings can amplify CPU overhead. The live B4 medians regress 2.6%, with
overlapping five-trial ranges (stock 38.51–40.22, combined 37.43–39.39 tokens/s).
Do not claim a causal decomposition or statistical certainty from this trace.
No DRAM-bandwidth or SM-utilization counters were captured for GLM; these
traces alone cannot prove memory saturation or a hardware compute limit.

The intervention did not confirm host-call cost as the primary unprofiled
throughput limit. Much of that work can overlap GPU execution; profiling and
run variation also affect apparent gaps. Retained-memory saving does not imply
proportionally less traffic in every kernel.

**Second target: prefill.** The bounded reconstruction path adds 36–38 ms of
explicit KDA decoding per prefill. Keep the fallback until a replacement
measures better; earlier fused-prefill and naive-overlap attempts were already
negative. M4 uses fused KDA kernels, so it is not paying that full-weight
prefill reconstruction cost. Embedding decode is only 7–9 microseconds and
is not the first optimization target.

**Quality remains a separate gate.** Keep the 16 contexts / four generated
tokens / 48 actual decode-step comparison. Step zero is prefill, not decode.
Compare distributions only with identical generated prefixes. Investigate
stock-repeat drift before a larger held-out screen or any deployment claim.

## 6. Immediate next iteration

1. The Nsight whole-model diagnostic, MoE dispatch and proxy-placement tests
   are complete. Start with the [Nsight next steps](GLM53_NSYS_MODEL_DIAGNOSTIC.md#next-steps):
   profile representative fused KDA/head kernels against reconstructed BF16
   in a bounded component fixture, then investigate exact attention metadata
   reuse. Verify counter/tool availability before any GPU pause. Preserve the
   existing model controls and declared gates; do not reload for pinning alone.
2. Make the smallest change in the local worktree. Preserve the qualified
   remote snapshots. Run the local checks below.
3. Qualify affected components on Spark2 with the idle controller. Reuse a
   fixture only when its bound code/native hashes still match. KDA changes
   need the real-vLLM KDA fixture; vocabulary/selector changes need the
   four-shard vocabulary fixture, including the omitted-config embedding call.
   MoE execution changes additionally need the raw/aligned installed-MoE
   fixture in `run_plugin_idle.py` / `benchmark_model_plugin.py`: vocabulary
   selection coverage alone does not qualify a changed grouped GEMM.
4. For unchanged engine/protocol/transport settings, run **only the new candidate**,
   using stock timing `kdamodelv2` and stock decode `combinedv1` through reuse
   guards. Current Nsight runs inherit proxy auto-pinning, while those saved
   controls omit it. A future efficacy comparison must either explicitly
   reproduce the saved transport conditions or collect a fresh matched control.
   Do not bypass the guard or treat Nsight before/after timing as stock evidence.
5. If the component improves, compare allocation, B1/B4 throughput, TTFT,
   token sequences and decode quality; collect short both-rank profiles
   separately from timing. Restore production before heavy offline review
   and save hashes for all evidence.

A useful next performance target is eliminating the B4 regression while
retaining the B1 benefit and measured memory saving. This is a next-iteration
target, not a retroactive change to the previous quality gates.

### Local checks, from the GLM worktree

```bash
/Users/kyrylobilchenko/Documents/Projects/other/mxstream/.venv/bin/ruff check .
/Users/kyrylobilchenko/Documents/Projects/other/mxstream/.venv/bin/mypy mxwave
/Users/kyrylobilchenko/Documents/Projects/other/mxstream/.venv/bin/python -m pytest -q
```

All passed after the current implementation. Use `python -m pytest`, which
selects this worktree correctly. Tests cover control reuse, source comparisons,
row-codec integrity, graph switching, decode conditioning and trace attribution.

### Staging and launch conventions

Choose a **new alphanumeric revision** for each attempt. Full-model snapshots
must be identical on both nodes at `full-REV-src/`, containing integration
`.py` files, `engine_plugin/pyproject.toml`, and root `mxwave/*.py` as a package.
Do not rsync over `full-combinedv2-src/`. For GPU component fixtures, use
`diagnostics-v1/src-FIXTURE/` on both nodes, with the same code layout.
The current accepted fixtures are `kda8adapterv1` and `vocabfixturev3`;
native pins and artifact manifests must match too.

For example, freeze the current local code to a new full-model revision from
the Mac. Replace the revision before running; a destination that already
exists is deliberately refused:

```bash
MXWAVE_WORKTREE=/Users/kyrylobilchenko/Documents/Projects/other/mxstream/.worktrees/glm53-flash-mxfp4
MXWAVE_RUN_ROOT=/home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/full-model-v1
MXWAVE_REV=combineddispatchv1
for MXWAVE_NODE in spark spark2; do
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$MXWAVE_NODE" \
    "test ! -e '$MXWAVE_RUN_ROOT/full-$MXWAVE_REV-src' && mkdir -p '$MXWAVE_RUN_ROOT/full-$MXWAVE_REV-src/mxwave'" || exit 1
  rsync -a --prune-empty-dirs -e 'ssh -o BatchMode=yes -o ConnectTimeout=8' \
    --exclude='__pycache__/' --include='*/' --include='*.py' --include='*.toml' --exclude='*' \
    "$MXWAVE_WORKTREE/integrations/glm53_mxfp4/" \
    "$MXWAVE_NODE:$MXWAVE_RUN_ROOT/full-$MXWAVE_REV-src/" || exit 1
  rsync -a -e 'ssh -o BatchMode=yes -o ConnectTimeout=8' \
    --include='*.py' --exclude='*' "$MXWAVE_WORKTREE/mxwave/" \
    "$MXWAVE_NODE:$MXWAVE_RUN_ROOT/full-$MXWAVE_REV-src/mxwave/" || exit 1
done
```

For a changed component, copy that frozen directory to a new
`diagnostics-v1/src-FIXTURE/` on both hosts before running its controller.
Do not edit either snapshot after qualification; make a new revision instead.
The full-model preflight verifies matching source hashes between hosts.

Once a new fixture snapshot exists, the component command on Spark1 is:

```bash
MXWAVE_FIXTURE=replacewithnewfixturerevision
nohup python3 -u \
  "$MXWAVE_RUN_ROOT/diagnostics-v1/src-$MXWAVE_FIXTURE/run_diagnostic_idle.py" \
  --revision "$MXWAVE_FIXTURE" --probe bf16_vocab \
  > "$MXWAVE_RUN_ROOT/diagnostics-v1/controller-$MXWAVE_FIXTURE.log" 2>&1 < /dev/null &
```

Use `--probe bf16_kda_small` for the fused-up-to-eight KDA fixture. Both commands
are head-side controllers; they launch the component on Spark2. Fixture state
is `diagnostics-v1/controller-FIXTURE.json`. The component guard preserves GLM
and aborts only its own container when production activity resumes.

For MoE fixture changes, its source path is `plugin-fixture-REV-src/` under
the experiment root, with state `plugin-fixture-controller-REV.json`.
The latest `run_plugin_idle.py` uses `full-model-v1/native/`; older frozen
versions use `grouped-moe-v1/native/build/` and `grouped-moe-opt-v5/native/build/`.
The generic BF16 component launcher also has a historical library path.
If changing a native binary, explicitly point the fixture at the proposed
binary and verify its hash matches the full-model binary. A passing fixture
against an older library is not proof for a new build.

The **completed** full-model launch was the following. This is a reference;
`combinedv2` already exists and must not be reused as a new revision:

```bash
python3 -u "$MXWAVE_RUN_ROOT/full-combinedv2-src/run_kda_model.py" \
  --revision combinedv2 --combined --fixture-revision kda8adapterv1 \
  --fused-max-tokens 8 --reuse-stock-run kdamodelv2 \
  --reuse-decode-run combinedv1 --vocab-fixture-revision vocabfixturev3
```

For the next candidate, replace the source directory, revision and any changed
fixture revisions, and launch the command with `nohup`, redirected log/stdin
and `&` as above. The controller arms an independent restoration watchdog,
waits for idle and restores the exact originals. Budget roughly six minutes
startup, four minutes measurement and three minutes restoration from the last
run, plus idle wait and any new build/compilation work.

### A batch-four graph trial needs harness work first

Do not compare a graph-four candidate with the saved graph-one control.
`run_kda_model.py` currently hardcodes graph sizes `[1]` in both the actual
command and recorded metadata, and sets protocol `graph_capture_sizes=[1]`.
It overrides the broader default in `full_model_control.py::engine`.

The combined runner also currently **requires** `--reuse-stock-run`. To try
new graph settings, add and test an explicit fresh matched-stock mode first;
update both arms, protocol and metadata consistently. Do not bypass the reuse
checks. Prove actual M4 replay, partial-batch behavior and output correctness,
not merely that a capture flag was accepted. This is a legitimate reason for
a new stock run. A dispatch-only change under the same settings is not.

Native ABI/library changes likewise require new component proof and updated
pins. Current reuse checks bind all native library hashes; assess which
baseline dependencies changed rather than silently disabling those checks.

## 7. Offline reproduction, no GPU needed

Existing Mac extraction:
`/private/tmp/mxwave-glm53-combinedv2-review/extracted/run-combinedv2`.
The downloaded archive is in the same parent directory. Temporary Mac files
may disappear; persistent archives remain on Spark1.

```bash
cd /Users/kyrylobilchenko/Documents/Projects/other/mxstream/.worktrees/glm53-flash-mxfp4
/Users/kyrylobilchenko/Documents/Projects/other/mxstream/.venv/bin/python \
  integrations/glm53_mxfp4/analyze_combined_profiles.py \
  --root /private/tmp/mxwave-glm53-combinedv2-review/extracted/run-combinedv2 \
  --output /private/tmp/glm53-combined-review-next.json --model-only-graphs
/Users/kyrylobilchenko/Documents/Projects/other/mxstream/.venv/bin/python \
  integrations/glm53_mxfp4/analyze_dispatch_profiles.py \
  --root /private/tmp/mxwave-glm53-combinedv2-review/extracted/run-combinedv2 \
  --output /private/tmp/glm53-dispatch-review-next.json
```

`--model-only-graphs` is justified by the pinned runner: model forward is
captured while head/logits computation is outside it. It must not be assumed
for a different runner. Each arm has one worker trace per rank; API-server
CPU traces are separate. In `run-combinedv2`, the legacy mode name **`kda`
means the full combined candidate**, not KDA alone.

Persistent files under `MXWAVE_RUN_ROOT`:

| Evidence | File |
|---|---|
| Final controller / restoration | `full-controller-combinedv2.json`, `restoration-combinedv2.json` |
| Independent service check | `combinedv2-final-service-check.json` |
| Full raw archive, 386 files / 115.18 MB | `combinedv2-evidence-v1.tar.gz` and `.json` |
| Offline analysis and launch-gap helper | `combinedv2-analysis-v1.tar.gz` and `.json` |
| Additional host/helper analysis and reproducer | `combinedv2-dispatch-analysis-v1.tar.gz` and `.json` |
| Full source snapshot archive | `mxwave-glm53-full-combinedv2.tar.gz` |
| Composed comparison | `run-combinedv2/results/combined-comparison.json` |
| Actual decode reports / vectors | `run-combinedv2/results/{stock,kda}/` |
| Saved launch-gap helper | Inside the analysis archive, `analyze-m4-launch-gaps-v2.py` |

Raw archive SHA-256:
`d8b8f38c0a12497f2bd0d4d6856a07e818477d907a7242c50db9e840700ba6c0`.
Analysis archive SHA-256:
`9108e2f4245c212c7dcb6bfed65822bc79d77d6e2593ae89e089cd3ad3e06279`.
Dispatch-analysis supplement SHA-256:
`d017331c90826af4d4f519eefa153c1c5c1e3d6ad8b345391a2cad575642c9f7`.
The supplement preserves the complete per-forward CPU analysis, curated
report, analyzer sources and attribution tests. It does not replace the
original evidence or alter the frozen runtime source.
All 386 raw files were independently hash-verified locally. Archives exclude
model payloads, compile caches and native binaries; preserve the host artifacts.

If recovery is needed, copy the raw archive with `scp` from Spark1, verify its
hash, extract to a new directory and point the analyzer at its `run-combinedv2`.
No model reload is necessary to inspect this evidence.

## 8. Known mistakes and operating context

- `combinedv1` initialized but stopped before candidate measurement: GLM's
  embedding constructor omitted `quant_config`, bypassing the selector.
  `vocab_adapter.py` now injects the existing engine CT config only at the exact
  audited embedding prefix. `vocabfixturev3` reproduces that real constructor
  call. Keep this bridge and test when refactoring.
- Omitting the fused attention aliases previously produced wrong outputs.
  Keep `MXWAVE_GLM53_ATTENTION_MAPPING=1`; do not diagnose that historical
  broken load as compression quality loss.
- An earlier source checker compared CPU and CUDA tensors directly. The
  current byte comparison explicitly normalizes both to CPU. Preserve it.
- The slow 630-second read came from losing `GLM_FAST_LOAD` in the launch
  whitelist. Its generic reader is compatible with this checkpoint. Keep the
  fixed whitelist; startup preparation still accounts for additional time.
- `MxWave expert scales prepared: unknown (aligned)` has a missing display
  label; actual coverage/source checks use real module identities. The Marlin
  FP4 warning describes the selected W4A16 backend, not a model-load failure.
- Shutdown `EngineDeadError`/resource-tracker messages occurred on deliberate
  engine stops. Judge success from saved client/gate/controller states, not
  a clean-looking shutdown tail. In particular, `combinedv1` failed its gate
  despite an orderly stop.

The user already authorized idle-only pause/test/automatic restoration for
this experiment. Preserve that scope; do not interrupt active requests or
recreate production containers. All long-running control must be detached
on Spark, because the user closes the laptop while commuting. If the picking-up
agent itself uses the serving GLM backend, it cannot supervise a paused GLM;
the detached controller/watchdog must finish and restore independently.

For an active run, read `full-controller-REV.json`, the matching client log
(`docker logs mxwave-glm53-bench-REV-kda`) and `restoration-REV.json`.
The restoration function in `full_model_control.py` verifies original IDs
and digests and stops only that revision's experiment containers. Avoid
starting originals manually while a live controller could still launch an arm.

After overhead and quality work, package runtime modules and licensed native
dependencies behind one optional MxWave plugin. Keep controllers and diagnostic
history separate. The existing implementation requires pinned vLLM hooks;
llama.cpp/Ollama needs a distinct native GGML port and is a later task.

## 9. Expected hand-back

Provide the code change, matching fixture proof, a compact before/after table
for allocation/TPS/TTFT/quality, both-rank trace evidence with limitations,
exact restoration verification and updated report/archive hashes. Preserve
failed attempts. Keep raw traces/model artifacts outside Git. Do not publish
or deploy an unqualified adapter as part of this handoff.
