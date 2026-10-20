# Input-only FP32 model result: C16 numerical gate fails

**Final C16 admission result, 19 October 2026, 09:01 UTC: complete.** The 10 ms policy
passes its prospective gate. Short-C16 TTFT is stock 1.383 s /
compressed 1.694 s (gap +311.5 ms).
This improves admission latency but leaves the C16 compression TTFT penalty unresolved.
All 94 numerical comparisons pass; CUDA saving is 9.367162 GiB across TP2.
Original GLM is restored healthy; no configuration is promoted. This concludes
compression experiments per user scope. See the [verified admission result](GLM53_C16_ADMISSION.md).
Older entries below are historical checkpoints.

Updated **19 October 2026, 07:43 UTC**. `fp32inputmodelv1` is complete and independently
verified on Spark and Mac: **1,634 payloads, 229 source bindings, 12 worker
traces**. Keep the preceding warmed exact16 route as the numerically qualified
experimental fallback. The new FP32 route is not qualified or promoted.
No experiment is running; original GLM is restored healthy.

The input change modestly reduces decode kernel time, but its C16 probability
distributions exceed the frozen KL limit. Both arms repeat exactly under matched
grouping, so this failure is a reproducible cross-arm difference. It does not
invalidate the previous warmed exact16 qualification.

| Matched quality | Mean decode KL | Mean NLL difference | Decision at KL <=1e-5 and absolute mean NLL <=1e-4 |
|---|---:|---:|---|
| C1 | 5.390468235e-6 | -4.842326919e-6 | Pass |
| C8 | 3.791330145e-6 | -2.700732479e-6 | Pass |
| C16 | **1.601802484e-5** | -1.203978271e-6 | **Fail** |

All nine ordinary C16 cross-arm combinations and the profile pair reproduce
the same failure: **10 of 46 comparisons fail**. Every mean-NLL check passes.
All C1/C8/C16 within-arm repeats have zero KL/NLL differences. Conditioning,
generated IDs and actual grouping match; all **300 paired timed output
sequences** also match. Matching sampled tokens does not make the full
probability distributions equal. No PPL, task accuracy or SQNR was measured.

A separately hashed comparison with the preceding warmed run finds the new
stock output **640/640 full-vocabulary rows bitwise equal** to previous stock;
the new candidate matches its predecessor on **290/640 rows**. All ten groups
match prompts, conditioning, generated IDs and both-rank grouping. The review
rehashes 2,608 inputs. This supports a real arithmetic difference in the new
candidate; it does not locate the first differing layer or establish a kernel
implementation bug. One C16 row accounts for about 54% of total KL; it is retained.

| Workload, 64 output tokens | Stock tok/s | Input-FP32 tok/s | Change | Median TTFT, stock -> input-FP32 | Added TTFT |
|---|---:|---:|---:|---:|---:|
| C1 / 256 prompt tokens | 14.856 | 17.052 | +14.78% | 0.391 -> 0.432 s | +41.0 ms |
| C1 / 2,048 prompt tokens | 12.602 | 14.145 | +12.24% | 1.157 -> 1.222 s | +64.6 ms |
| C8 / 256 prompt tokens | 61.338 | 63.257 | +3.13% | 1.163 -> 1.201 s | +37.7 ms |
| C8 / 2,048 prompt tokens | 31.862 | 31.394 | -1.47% | 6.448 -> 6.765 s | +316.9 ms |
| C16 / 256 prompt tokens | 80.643 | 81.322 | +0.84% | 2.079 -> 2.211 s | +132.2 ms |
| C16 / 2,048 prompt tokens | 37.158 | 36.405 | -2.03% | 11.153 -> 11.672 s | +519.0 ms |

These are six-trial medians per case/arm from 72 ordinary timing groups,
600 requests and 38,400 generated tokens. Loads run in fixed stock-then-candidate
order. They establish the observed result, not statistical significance or a
clean isolated comparison against the preceding compressed kernel. The original
>=5% throughput / <=5% TTFT gate remains failed. The extra C1 throughput over
the preceding compressed run is only roughly 1-2% in raw measurements.

First-token latency is not solved. The candidate still pays prefill
reconstruction. Short-C16 first cohorts vary: stock **4/6/1/7/3/5**, candidate
**7/5/6/7/2/1**. All stop at the existing 3 ms deadline. Therefore the smaller
+132 ms gap, compared with the preceding run's +442 ms, is admission-confounded
and cannot be attributed to the FP32 kernel. For C1, all-trial median token
arrivals favor the new candidate from token **7 short / 6 long** onward; this is
descriptive, not a per-request guarantee.

The fallback's measured short-prompt TTFT remains **0.384 -> 0.422 s at C1**
and **1.838 -> 2.280 s at C16**. Its throughput is +11.82-13.28% at C1 and
near parity at C8/C16, with all 46 numerical comparisons passing. The failed
FP32 run's smaller C16 latency gap must not be transferred to that fallback.

The mixed traces show the intended decode route on both GPUs: 34 FP32 inputs,
34 original outputs and 34 FP32 reducers per covered M8/M16 graph, with graph
replay and zero reconstruction. Verification covers 56 M16 and 50 M8 ranges
across both arms. In candidate mixed profiles, per-rank full-size-step medians
are approximately **13.60-14.11 ms input**, **6.61-6.82 ms output** and
**0.084-0.128 ms reduction**. Median total KDA sums are **20.36-20.95 ms**,
versus about 22 ms in the preceding run. Expert kernel sums remain roughly
**60-61 ms at C8 / 97-98 ms at C16**. This explains why a small KDA component
gain has limited concurrent serving impact. Profiles are separate from timing;
kernel sums are not API wall time or measured memory bandwidth.

Startup dispatch caches remain unchanged and no recognized compilation event
overlaps an ordinary timing envelope. Boundary telemetry has 144 samples/arm:
stock **2,437-2,496 MHz / 66-76 C**, candidate **2,457-2,496 MHz / 68-79 C**.
No trial is filtered or clock-corrected; boundary samples cannot rule out
transient throttling.

CUDA allocation is **10,057,913,856 bytes / 9.367162 GiB lower across TP2**
at every measured stage, unchanged from the fallback. Candidate CPU RSS is
**242.18 MiB higher before quality / 250.52 MiB higher after ordinary timing**
across TP2. Its lower post-profile RSS is profiler-influenced and does not
establish CPU-memory savings. CPU and CUDA counters overlap on Spark and must
not be added. The proposed early memory observer skipped both-arm capture
under its original timing guard; anonymous-memory attribution remains open.

Both arms use native mHC, deterministic sparse/expert ordering, matched quality
cohorts, identical 34-variant startup warmup/heap trim, graph sizes 1/2/4/8/16,
4 GiB KV per rank, max 16 sequences and a 2,048-token budget, without MTP.
Only candidate input physical rows 1/4/8/16 use FP32 split 2. Output and other
input routes retain their parent behavior. The existing 206,569,472-byte/rank
two-slot pool is reused, without separate FP32 scratch or a resident BF16 cache.

Independent verification initially failed on a checker error: expected input
grid Y=197 omitted the split multiplier. The frozen native API gives
**12,608 * 2 / 64 = 394**, matching all six candidate traces. A separate offline
verifier corrects only this expectation; the original checker, failure logs,
all 229 experiment sources and every numerical/performance gate are preserved.
Seven negative fixtures reject wrong split/output geometry, missing reducers,
graphs or rank, and reconstruction. The edited worktree reviewer passes 14
focused tests and targeted Ruff. No GPU experiment was rerun. All 216 prior
exact-word failures and two input timing repeat failures remain failed.

Spark/Mac numerical decisions agree, with maximum float64 difference
**1.5543e-15**. Original service identities/configuration are restored, health
200 rechecked at about 07:39 UTC. Controller, collector, watchdog and retention
have finished; the owned awake lease and completed SSH transport are released.
No configuration is promoted and no new experiment has started.

Per user scope, stop compression-kernel optimization and retain the qualified
warmed exact16 fallback. The proposed last compression follow-up is **3 versus
10 ms admission** in both arms, with a short-C1 delay control, then MTP. The
3 ms window is already enabled. Sixteen 256-token prompts still need at least
two prefill forwards under the unchanged token budget. The three local admission
modules are unfinished/unvalidated and not frozen or launched. The latest
authorization covered observing this run's outcome; the broader goal remains
paused. See [admission evidence](GLM53_PREFILL_ADMISSION.md).

Evidence is linked from the [final retention manifest](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/fp32-input-model-v1-final-retention.json),
[cross-host proof](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/fp32-input-model-v1-cross-host-verification.json),
and [final findings](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/fp32-input-model-v1-final-findings.json).
Large full reviews are retained as lossless gzip files in that manifest.
Local archive: `/private/tmp/mxwave-fp32-input-model-v1/evidence.tar.gz`.
Remote archive: `ROOT/evidence-fp32-input-model-v1-recovery-v2.tar.gz`, where
`ROOT=/home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/full-model-v1`.
Archive: **465,586,181 bytes**, SHA256
`bfbbafa8f53bdbd2c82bd2e6cc3d6f48b678433b015beb0ff7863bce9ebca8c2`.
Frozen source archive SHA256:
`aecf6a332294e165dc6a60b4c7e3731e87944c41075d2304dbec82b6e8305cdc`.
Corrected verifier SHA256:
`d89d085a72172e09a6ec6f75c2c563ab0ccac33a9abf73499583197109639827`.
