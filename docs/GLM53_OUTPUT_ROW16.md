# Output-only sixteen-row compression follow-up

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

Updated **18 October2026,12:30 UTC**. Compression collection is complete.
Work has resumed on the separate actual-MLA diagnostic; see the
[current handoff](GLM53_HANDOFF.md).
`output16qualv1` is complete and independently verified on Spark/Mac:
1,293 payloads/180 source bindings/16 worker traces, archive
`d871cbac8d1d91332469c01434ce72549679974d8e1653d0198642fbf8c44dba`
(362,942,944bytes). Exact originals and health200 restored at12:14:05 UTC;
no GPU experiment is active.

All8 fresh numerical groups and4 profile sessions finish on the separate
validation load. All24 KL comparisons fail and8 fail NLL; both speed gates
remain failed. All quality IDs/conditioning match, and CUDA saving is9.367GiB.
Independent both-rank attribution finds eight pure-decode ranges with10 or15
actual requests using the padded16 graph, each reconstructing34 matrices in
25.68–28.11ms. This verifies the intended count reduction from68, while the
retained API timings show only partial C16 recovery. Kernel sums are not request
latency or a TP critical path. The original source-after/time-gap limits remain.
See the [completed review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/output16-validation-v1-review.json)
and [handoff](GLM53_HANDOFF.md) for resumption details.

Historical first attempt: `output16v1` completes all72 timing
groups, then fails saving its sixth numerical group because the head disk fills.
Five groups complete; no candidate profiles run. The client exits1 and is not
OOM-killed. The failure also prevents writing the final failed measurement
header, so the saved header still says running; controller/logs are authoritative.
Controller2930080 is failed_restored, exact original identities/configs/mounts
and health200 restored at11:16:07 UTC. No GPU experiment is active.
MTP remains deferred. Recovery is independently verified on Spark/Mac:
1,532 payloads,178 source bindings,72 candidate timing groups/636 requests/
40,704 tokens. All636 paired timed sequences match; all17 available complete
numerical comparisons fail. Three numerical groups and all candidate profiles
remain missing. The original source-after check is absent and cannot be
retroactively supplied by a fresh model load.

Validation-only continuation `output16qualv1` subsequently completes and verifies.
All178 runtime/source files are unchanged; only two validation helpers are
added. It reuses all72 timings and collects fresh numerical repeats and four
short profiles/TP rank, preserving the model-load/time gap explicitly. It adds
6GiB pre-launch and2GiB per-stage disk checks. No new timing groups or MTP.

Completed, unprofiled medians show partial recovery against the retained
compressed parent: shortC16+3.36%/+3.96%, longC16+1.52%/+1.30% at0/3ms.
Against retained stock, shortC16 remains−10.90%/−10.20%, longC16−6.23%/−6.30%.
ShortC8 is+3.48%/+3.31%, longC8−0.11%/−0.53% relative to stock.
Both original performance gates therefore fail. C1 TPS is+14.00%/+13.39%,
but TTFT is+9.94%/+8.70%. These independently checked timing-only values lack
the original source-after check and complete numerical/profile collection.
Both control comparisons retain their time gaps; no configuration is promoted.

The completed [C8/C16 comparison](GLM53_HIGH_CONCURRENCY.md) loses7.5–13.8%
throughput at C16. Eight retained pure-decode ranges across TP2 each rebuild
all68 KDA weights in34–37ms. The current fused limit is eight rows, including
when actual9–15 rows dispatch to the padded16-row graph.

Focused V7 independently verifies all45 payloads/18,432 events. Both real
layer-zero/rank-zero fixtures have exact outputs, but the larger input matrix
fails the unchanged2% reference timing check. Only the4096×4096 output matrix
passes every gate at16 rows. Its fused/reference ratios range0.349–0.504
across eager/graph, warm/pressure and both timing orders. The257-row negative
controls are slower. This does not qualify a shared/global16-row threshold.
The [compact V7 review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/prefill-routing-v7-review.json)
retains every condition and failure.

This follow-up enables exactly16 rows only for the audited square KDA output
projections. Input projections, rows9–15 and larger prefills retain their
existing routes. The global fused limit stays8, native kernels and precision
are unchanged, and no resident BF16 weight cache is added. The fixed two-slot
pool remains206,569,472 bytes/rank. Pure16-row decode should consequently
reconstruct34 input matrices rather than68 total matrices; separate both-rank
profiles must verify the actual execution mechanism.

The complete original workload and model gates remain:

- C1/C4 short prompts, C8/C16 at256 and2,048 prompt tokens;0/3ms idle windows.
- Six trials/case/window,64 output tokens:72 new groups/636 requests/40,704
  tokens;24 warmup groups remain separate.
- After timing: eight full-vocabulary quality sets, then four short profiles
  per TP rank. No profiling inside timing groups.
- At least5% compression throughput gain in every C8/C16 case, C1/C4 ratio
  at least0.98, and no median TTFT ratio above1.05 relative to stock.
- KL≤1e-5, absolute mean NLL difference≤1e-4, matching conditioning/IDs,
  exact source reconstruction and complete source-scale coverage.
- Every failed gate and prior failure stays retained. No automatic promotion.

Stock control is reused from `c816v1`; compressed-parent observations come
from `c816retryv2`. Both comparisons have explicit time gaps. There is one
new model load, not a fresh interleaved control. Passing this narrower route
does not guarantee that remaining input reconstruction permits a C16 speed
gain over stock or resolves MLA repeatability.

Source is isolated from the verified174-binding parent;178 bindings now
match on both nodes. Only the parent's KDA runtime changes; four source helpers
are added. Local worktree exponent guards remain preserved separately.
All other parent functions have identical ASTs; the native library/ABI is unchanged.
Local full pytest passes with one skip, Ruff passes, and strict core mypy
passes35 files. CPU source, geometry routing and native CLI checks pass on Spark.

Source archive SHA256:
`cb908fe66622d7c1bfe2aca4acc7eb16b7febff08ac4f5e77bbe61f610ce620e`.
Prepared SHA256:
`08bda6420ebc10f0c0cea6a0c979c7ca2c02b27f379a2fb3599523c19332b0e4`.
Plan SHA256:
`f0ce0ea728a86f71b6cc9fa78268b6e6c6ce717b3059c26ac46067c75a4188f3`.

In the existing Spark root, read `quality-model-controller-output16v1.json`
for execution/restoration and `output-row16-v1-recovery-evidence-check.json`
for completed recovery. Source `full-output16v1-src`; results `run-output16v1/results`.
Local continuation `/private/tmp/mxwave-output-row16-v1/`.
The original collector2930079 failed with ENOSPC; its partial output is retained.
Recovery collector3389440 finished after exact restoration. Recovery archive
SHA256 `683fcb4bbc7b204d1a1b3a656766336a05efb389e0d17ed7ff0f641a26697ccb`,
304,447,995 bytes. The [partial review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/output-row16-v1-partial-review.json)
retains every completed timing, available numerical pair and missing stage.

Disk recovery retains canonical archives and moves only checksum-verified
older generated payloads to the existing worker under
`storage-recovery-v1/offloaded-payloads/<original-relative-path>`.
Head-side `storage-recovery-v1/offload-plan.json`, backup and cleanup receipts
retain all8,950 paths, bytes and hashes. About5.63GB remains free after final
continuation retention; no model/checkpoint files were moved or removed.
