# Compression throughput at C8 and C16

Updated **18 October 2026, 10:20 UTC**. `c816retryv2` completes and restores
the original service healthy. All **1,277 payloads**, 174 source bindings,
144 timing groups / 1,272 requests / 81,408 tokens and sixteen worker traces
verify independently on Spark and Mac. The C16 startup failure is fixed in
the experiment, but **both speed and numerical gates fail**. No configuration
is promoted. MTP remains deferred until compression has a working outcome.

The retry reuses the complete stock arm from `c816v1`; there is a time gap
and no fresh paired stock load. Both targets use the same captures through16,
max sequences16, precision, transport, 4 GiB KV/rank and 0/3 ms admission.
All six trials per workload/window are retained. TPS medians and changes:

| Workload | Stock TPS, 0 / 3 ms | Compressed TPS, 0 / 3 ms | Change, 0 / 3 ms |
|---|---:|---:|---:|
| C1 / P256 | 14.876 / 14.824 | 16.705 / 16.689 | +12.30% / +12.58% |
| C4 / P256 | 40.553 / 42.854 | 40.798 / 43.461 | +0.60% / +1.42% |
| C8 / P256 | 58.094 / 61.084 | 59.566 / 62.400 | +2.53% / +2.15% |
| C8 / P2048 | 31.463 / 31.510 | 31.359 / 31.293 | −0.33% / −0.69% |
| C16 / P256 | 79.963 / 79.798 | 68.935 / 68.930 | −13.79% / −13.62% |
| C16 / P2048 | 36.745 / 36.705 | 33.940 / 33.948 | −7.64% / −7.51% |

Allocated CUDA memory still falls **9.367 GiB across TP2**. C1 first-token
latency rises 8.12–10.04%; zero-window C4 rises7.99%, exceeding the5% guard.
All636 paired timed output sequences match. Nevertheless, all24 numerical
comparisons fail KL1e-5, including every self-repeat; four also fail absolute
NLL1e-4. All quality IDs/conditioning match. These failures remain unchanged.

Separate profiles cover short C8/C16 at both windows and both ranks, after
all unprofiled timing and quality. P2048 retains API/queue/worker evidence
without GPU traces. Correlation-based attribution finds **68 reconstructions
in each retained C16 pure-decode forward above eight active rows**, all inside
the captured graph: eight ranges across TP2, **34.38–37.33 ms** of decoder
kernel duration. Actual profile batches have9/11/14/15 rows using capture16;
they are not sustained fully active C16 measurements. No decoder is unassigned.
C8 profiles retain136/68 reconstructions per rank at0/3ms, in prefill only.

The rank-zero byte proxy is4,652,531,712 BF16 output bytes plus3,298,566,000
allocated compressed bytes per68 reconstructions. This implies approximately
213–231 GB/s in its retained decode ranges, **not measured LPDDR traffic**:
padding, caching and repeated loads are not accounted for. Kernel duration
sums and graph spans are not request latency or a TP critical path.

The next focused check is the existing fused kernel at **exactly16 rows**.
V7 is CPU-prepared with rows[8,16,257], a fresh seed and queued timing legs
to reduce synchronization-induced host gaps. It retains the old2% repeat,
5% gain and1% projection-L2 gates, all orders/cache/execution conditions,
both source matrices and18,432 timed events. No GPU controller or routing
change has been launched. Intermediate rows and full-model quality remain
unqualified. The prior V6 row16 control failures remain part of the record.

[Complete retained review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/high-concurrency-retry-v2-review.json)
includes all trials, numerical failures, both-rank profile attribution and
the reused-control declaration. Archive362,781,198 bytes, SHA256
`602e25daf5a498a2b6012c41276cb91f41695ccfd2093108a7eaacae0239d66a`.

## Startup repair and preserved attempts

The first `c816v1` attempt completes stock but fails during compressed C16
capture: initialization stream leases remain registered and exhaust the
two-slot scratch allocator. Its730-payload archive remains verified.
`c816retryv1` stops in a tiny fixture before model load because the fixture
inspects captured writes before the first replay; its734-payload archive
also remains verified. The corrected tiny fixture executes the graph before
checking ownership and passes five additional replays with eager reuse.

The repaired allocator retires completed eager leases only at synchronized
startup boundaries and preserves captured leases. Reconstruction/packing/GEMM
function ASTs match the previous frozen source. The full model passes292
worker snapshot checks: two leases, one captured lease,206,569,472 pool bytes
per rank and captures[1,2,4,8,16]. No inference-time synchronization is added.
The two-slot /256MiB bound and fused limit8 are unchanged.

## What is established

The [earlier matched stock/compressed result](GLM53_PAIRED_SPEED.md) saves
9.367 GiB across TP2 and improves C1 throughput 10.4–12.4%, while C8 is near
parity: +1.84% at P256 and −0.07% at P2048. Those arms used max sequences8,
captures [1,4], and eager C8. The completed C16 result is reported above.

The [graph-only comparison](GLM53_GRAPH_EXPANSION.md) improves C8 1.7–3.4%.
The [admission-only comparison](GLM53_PREFILL_ADMISSION.md) improves short C8
3.65–4.19%. These are separate serving-policy results on the compressed
target. Their gains cannot be added, or attributed to compression against a
stock target with a different policy.

Numerical repeatability remains unresolved, including stock self-repeats.
Matching greedy IDs does not pass the unchanged KL/NLL requirements.
The throughput issue cannot yet be declared fixed or the adapter qualified.

## Frozen comparison

The original two-load plan compares the imported MXFP4 checkpoint with stock
runtime storage against the qualified combined lossless-storage adapter.
Both use the same native libraries, TP2 transport, precision, FP8 KV budget,
prompt IDs, warmups, seed, streaming client and serving limits. There is no
speculative drafter or new kernel arithmetic. After the first attempt fails,
the completed retry reuses its verified stock arm and loads only compressed;
the temporal-control limitation is explicit in the review.

| Setting | Both targets |
|---|---|
| CUDA graph captures | [1,2,4,8,16], full decode only |
| Maximum sequences | 16 |
| KV budget | 4 GiB/rank, FP8 E4M3 |
| Maximum model length / batched tokens | 8,192 / 2,048 |
| Compressed KDA fused limit | 8; larger rows retain reconstruction + BF16 GEMM |
| Idle-only input collection | 0 and 3 ms, interleaved within each model load |
| Cases | C1/C4 at P256; C8/C16 at P256 and P2048 |
| Timing | Six groups per case/window/target; exactly 64 output tokens |
| Warmups | Two groups per case/window/target; 16 output tokens |
| Quality | C8 and C16, both windows, two repeats/target; 16 contexts × four outputs |
| Profiles | C8/C16 P256, both windows/target, after all unprofiled timing and quality |

The planned 144 timed groups / 1,272 requests / 81,408 tokens include
arrival times and generated IDs. The first eight C16 prompts are exactly the
old C8 prompts; eight additional distinct offsets use the same authored text
pool. Cases and window order reverse on alternate trials, identically in both
targets. Sequential target order still leaves clock/order uncertainty.

Both actual worker snapshots must show sequence capacity16 and exact FULL
descriptors for all five sizes. Snapshots establish eligibility; separate
trace annotations show actual batch formation. Startup and these finite
request groups test C16 capacity for P256/P2048 + 64 outputs, not sixteen
maximum-length contexts. Continuous active arrivals remain outside this test.

C16 deliberately retains the current >8-row KDA path. It may reconstruct
68 matrices per full decode forward. The experiment measures that risk and
retains device decoder counts separately from unprofiled API time; it does
not quietly raise the fused-kernel limit or promote a component variant.

The frozen performance screen asks for at least 5% compressed-vs-stock TPS
gain in all four C8/C16 cases under one matched window, no more than 2%
throughput regression in C1/C4 controls, and no more than 5% TTFT regression.
Every self-repeat, cross-target and cross-window full-vocabulary comparison
keeps KL1e-5 and absolute NLL1e-4. All 24 comparisons and 1,024 raw distribution
files are retained. Passing this finite performance screen would still leave
historical numerical/format failures and broader traffic qualification open.

## Conditional C16 follow-up

**Stock-only trace observation, 08:59 UTC:** all72 stock timing groups, eight
quality sets and four requested profiles finish. The four rank-zero worker
traces plus reports/plan verify against an eleven-file source snapshot.
Requested-C16 short profiles contain pure-decode active batches11–14; these
must use the retained16-token capture. They establish coverage of the >8-row
path, not sustained fully active C16. The compressed arm subsequently fails
during startup; no compressed timing or paired performance result exists.
The final failed-attempt archive retains 730 payloads / 180,955,887 bytes and
verifies independently on Spark and Mac: SHA256
`879ee5dd8e956e36c19419625dc6a89056cff6303e913fa54ec7568e13081ef9`.

All four stock C8/C16 self-repeat checks fail the unchanged KL1e-5 floor
(means 2.79e-5–1.13e-4). C8/window0 also fails the absolute NLL1e-4 floor
(1.04076e-4). All generated IDs and conditioning match. The retry preserves
these failures and numerical limits.

The local [device attribution tool](../integrations/glm53_mxfp4/analyze_graph_reconstruction.py)
follows device correlation IDs to CUDA runtime/driver calls contained by CPU
model annotations on the same PID/TID. It keeps unassigned events explicit
and does not associate a GPU kernel with a nearby CPU interval by timestamp.
It passes asynchronous/ambiguous-ownership tests and reproduces all old
admission decoder counts with zero unassigned reconstruction kernels. This
postprocessor is outside the frozen GPU bundle; current runtime code is unchanged.

The retained [V6 routing review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/prefill-routing-v6-review.json)
already has sixteen-row results for both source fixture shapes. Arithmetic
passes with recorded relative L2 zero, and fused execution is over 5% faster
than reconstruction + GEMM in every retained condition. However, identical
reference-control ratios exceed the unchanged ±2% limits in some blocks
(including approximately 0.976 and 1.045). Both sixteen-row component
eligibility decisions therefore remain false; this is not an accepted
runtime routing change.

The completed paired profiles do show material repeated C16 reconstruction.
Focused V7 tests8/16/257 rows with eight queued event pairs and one untimed
primer per leg, retaining both ABBA/BAAB orders and the unchanged2% reference
and5% gain gates. All18,432 events and45 payloads independently verify on
Spark/Mac, with identical reviews. Both real fixtures have relativeL2 zero,
but only the output matrix qualifies at16. Input16 graph-warm's identical
reference ABBA ratio0.9734604466852056 misses[.98,1.02]; its other numerical
and fused-gain checks pass. Shared eligible rows are empty. Both257-row
cases remain slower. Do not raise the global fused limit or rerun unchanged.

V7 is complete, exact originals/health200 restored at10:29:07 UTC; archive
SHA256 `ae9d4fd7a8e8173def9654c1ca0b53dc40ff79f1e59340c3d7d8d62711d663f3`.
The narrower output-projection exact16 route is a possible follow-up with
separate runtime qualification. It addresses only part of C16's cost.
A faster component result does not establish end-to-end speed or fix MLA
repeatability. MTP remains deferred.

## Source and execution state

Latest model source `ROOT/full-c816retryv2-src` has174 bindings. Controller
`quality-model-controller-c816retryv2.json` and collector receipt
`high-concurrency-retry-v2-evidence-check.json` are complete; exact original
service restoration and HTTP200 are verified. Source archive SHA256
`bf2cd68d4a5f9fb5a42e6bf1c15cadbf6ffc789b58f607bc20f2cacab4ed9308`.
Evidence archive SHA256
`602e25daf5a498a2b6012c41276cb91f41695ccfd2093108a7eaacae0239d66a`;
local continuation `/private/tmp/mxwave-high-concurrency-retry-v2/`.
The [compact review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/high-concurrency-retry-v2-review.json)
retains every trial and both-rank attribution. Native arithmetic is unchanged;
the isolated retry transplants only startup lifetime ownership and reporting.
Worktree exponent guards remain preserved separately.

Latest component `prefilldecodev7` is complete/restored, source18 files in
`ROOT/prefill-decode-v7/src`, controller
`prefill-decode-controller-prefilldecodev7.json`, receipt
`prefill-decode-v7-evidence-check.json`. Local
`/private/tmp/mxwave-prefill-decode-v7/`; no router change or model load.

The following source paths describe the preserved first failed attempt.

Assembled source `ROOT/full-c816v1-src` has 169 Python/TOML bindings. The
qualified `full-pairedspeedv1-src` parent has 160; only plugin registration
metadata changes among parent files. Native KDA/MoE arithmetic, loader guards,
source weights and precision are unchanged. CPU checks confirm the native
max16 graph descriptor, unchanged input-queue wrapper, real CLI in both arms,
and source equality on both nodes. They do not establish GPU capture safety,
cache capacity, speed or numerical qualification.

Source-only archive SHA256:
`d18d4952e409f464a8c82157c9bdca9ed8fd56d63c96807e311130bce31cdb27`.
Prepared SHA256:
`54f1928d2618df6dc901bc6489c6e192e7cf9bdc6c9c25936b66a6f14e04afe5`.
Runtime plan SHA256:
`0db9aed1112d8cc78f1e4085ccb7b86b470d55aee84181e630473ca8da342a0b`.

`ROOT` is the existing Spark experiment directory
`/home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/full-model-v1`.
Read `quality-model-controller-c816v1.json` for that attempt's execution/restoration
state, `run-c816v1/results` for retained partial measurements, and
`high-concurrency-v1-evidence-check.json` for independent collection status.
The existing idle guard and independent restoration watchdog own the service
pause. The collector waits for exact original identities and HTTP200 before
copying source/results/logs and both-rank traces and running the CPU verifier.

Local continuation: `/private/tmp/mxwave-high-concurrency-v1/`.
The [launcher](../integrations/glm53_mxfp4/run_high_concurrency.py),
[measurement](../integrations/glm53_mxfp4/measure_high_concurrency.py),
[analysis](../integrations/glm53_mxfp4/analyze_high_concurrency.py) and
[verifier](../integrations/glm53_mxfp4/verify_high_concurrency.py) retain the
full reproducible protocol. Full local tests, Ruff and strict core mypy pass.
