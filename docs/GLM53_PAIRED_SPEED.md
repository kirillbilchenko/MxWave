# Matched GLM serving speed and latency

**Complete and independently verified, 17 October 2026, `pairedspeedv1`.**
Exact original GLM containers/configurations were restored with health 200 at
**20:22:23 UTC**. No experiment engine remains active from this run.

The storage candidate improves single-request throughput **10.4–12.4%**.
Concurrent C4/C8 throughput changes by **−0.07% to +1.84%**, with first-token
latency **2.3–5.9% higher** across the grid. CUDA allocation again falls by
**9.367 GiB / 5.33%** across TP2. This confirms a useful C1 benefit, but does
not establish a substantial concurrent-serving speed gain.

## All fixed-length workloads

Five trials per workload/arm, including prefill and completion drain. Prompt
tokens and 64-token output lengths are identical; all **130 paired generated
sequences / 8,320 output IDs** match. Matching IDs does not establish equal
full-vocabulary probabilities or amend earlier failed quality gates.

| Concurrency | Prompt tokens | Reference tok/s | Candidate tok/s | Change | TTFT reference → candidate, s |
|---|---:|---:|---:|---:|---:|
| C1 | 256 | 14.994 | 16.848 | +12.36% | 0.387 → 0.410 |
| C1 | 2,048 | 12.642 | 13.961 | +10.43% | 1.162 → 1.203 |
| C4 | 256 | 40.372 | 41.099 | +1.80% | 1.075 → 1.131 |
| C4 | 2,048 | 26.263 | 26.312 | +0.19% | 4.088 → 4.226 |
| C8 | 256 | 57.022 | 58.071 | +1.84% | 1.506 → 1.555 |
| C8 | 2,048 | 31.189 | 31.167 | −0.07% | 6.495 → 6.646 |

Each entry is a median of five retained groups. All ranges and individual
trial metrics remain in the [compact review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/paired-speed-model-v1-review.json).
C4/P256 varies substantially: reference **32.048–43.391 tok/s**, candidate
**29.272–44.047 tok/s**. No slow group was excluded. C1 ranges do not overlap
in this session; small concurrent changes retain fixed-order/clock uncertainty.

Client decode rates after the first token improve about **15.5% at C1**,
**2.9–3.3% at C4**, and **1.5–2.9% at C8**. These are arrival-based request
metrics, not GPU kernel rates. Higher TTFT offsets some decode benefit.

Allocation is **175.866 → 166.499 GiB** across TP2, with identical KV budgets.
Resident storage savings are not per-step LPDDR-traffic measurements: only
selected experts and embedding rows are consumed, and decompression has a cost.
No new PPL, SQNR, task-accuracy or serving-KL qualification was attempted.
The [full GSM8K comparison](GLM53_GSM8K_FULL_BENCHMARK.md) remains separate.

## Separate controlled C4 profile

All unprofiled timing finishes before profiling in each arm. One controlled
four-request cohort uses 256 prompt tokens/request and four output tokens.
All four worker traces prove **three actual CUDA graph launches**, one complete
prefill and three complete M4 decode forwards. Each forward contains all
84 routed-expert GEMMs on its annotated main stream.

| GPU forward | Rank | Reference, ms | Candidate, ms | Duration change |
|---|---:|---:|---:|---:|
| Prefill: 1,024 tokens / four requests | 0 | 725.976 | 774.001 | +6.62% |
| Prefill: 1,024 tokens / four requests | 1 | 725.366 | 774.103 | +6.72% |
| M4 decode, median of three | 0 | 85.010 | 78.219 | −7.99% |
| M4 decode, median of three | 1 | 85.043 | 78.213 | −8.03% |

The candidate adds **68 explicit KDA weight reconstructions**, totaling
**36.259 / 36.263 ms** per profiled prefill on ranks 0/1. The reference has
none. This is a concrete extra cost consistent with the TTFT penalty.
Other GEMM, communication and scheduling differences also exist; these
category sums are not an additive wall-time explanation.

Candidate decode still spends about **33.59 / 33.71 ms** in 84 expert GEMMs
and **21.79 / 21.81 ms** in 68 fused compressed KDA projections per forward.
Raw BF16 category totals include both KDA and other linears, so their difference
cannot be relabeled as KDA-only time. The later LM head is outside the complete
forward annotation. Ranks execute concurrently; never add their times.

Profiles contain one cohort per arm, not five independent profile trials.
The approximately 8% profiled forward reduction does not directly predict
unprofiled API throughput. Matched shapes/prompts and output IDs do not prove
identical routed activations. Profile durations do not replace serving metrics.

## Admission evidence retained with the slow trials

Both-rank Python eager-dispatch counter deltas agree for every timing group.
At C4/P256, the fastest reference/candidate trials each contain one 1,024-token
prefill (42 MoE-layer producer calls). The slowest reference trial instead
has eager steps of **256, 257 and 514 tokens**; the slowest candidate has
**256, 513 and 259 tokens**. These include split admission and mixed
prefill/decode work, rather than a single combined prefill.

This observed association is a useful next diagnostic, not a randomized proof
of the complete latency cause. The counters do not count CUDA graph replays;
those are proved separately by the traces. All original medians and ranges
remain unchanged. Do not select only the combined-prefill timing samples.

## What is next

1. **Address prefill and mixed admission.** Use the observed pure/mixed shapes
   to evaluate a bounded alternative to repeated KDA reconstruction. Require
   an actual mechanism improvement before another full-model load; earlier
   fused-prefill and naive-overlap variants already failed. Retain the existing
   scratch/memory contract and numerical gates.
2. **Keep kernel comparisons under matched batch formation.** The saved
   counters show why requested concurrency alone is insufficient. Preserve
   ordinary API latency as the serving outcome; controlled cohorts diagnose
   a mechanism rather than replacing that outcome.
3. **Retain the remaining decode targets.** Expert GEMMs and KDA projections
   dominate substantial device work. N8 variants were slower; split-K variants
   failed numerical gates. Neither is an accepted speed improvement.
4. **Resolve numerical qualification separately.** Layer 3's full MLA module
   is still the first observed divergent boundary. Its isolated synthetic
   attention fixture repeats exactly; real projections/cache writes/internal
   state remain open. Do not repeat broad GSM8K or relax the old tolerances.
5. **Then resume the requested roofline figure.** It remains deferred by the
   user. PNG/SVG links are outstanding, with LPDDR-counter limits documented.

## Reproduction and verified state

Same imported checkpoint/revision, pinned vLLM image, TP2, native libraries,
canonical expert order, BF16 activations, 4 GiB FP8 KV/rank, max sequences eight,
graph captures **[1,4]**, eager C8 and no drafter. The raw-runtime reference
includes the same existing shared fixes. Prompts come from the original authored
timing pool and reproduce the earlier three cases byte-for-byte before extension
to six cases. Two 16-token warmups/case precede five 64-token trials. Trial case
order alternates forward/reverse; the reference model loads before the candidate.
Generation uses temperature zero, seed 42 and ignored EOS for the fixed token count.
The checkpoint is `INCModel3/GLM-5.3-Flash-MXFP4-Mixed-CT-AutoRound`, revision
`35d1b0d5cb4a0db075fd2749c897e970dc15bfa5`.

Both source inventories (160 files), native hashes, all **672 source groups**,
all **260 timed requests / 16,640 tokens**, graph settings and exact service
identities verify independently on Spark and Mac. The three independent analysis
JSON files are **byte-identical across hosts**. All **373 archive payloads** verify.
Before freezing the run, the local suite passed **572 tests with one GPU skip**,
Ruff passed and strict core mypy passed all 35 source files. Those checks do not
replace the unresolved numerical qualification of the loaded model.

| Rank | Archive bytes | Payloads | Archive SHA256 |
|---|---:|---:|---|
| 0 | 6,066,895 | 200 | `e81b1335844606acd1516b058426e99a11ae0ad40a254c5bc7123ffc979175c9` |
| 1 | 4,983,651 | 173 | `009fceff64521a985702c3ef1567eb6f6a38774544bd6fe616bbbcf61decb85a` |

The first offline profile checker encountered a frontend trace in addition
to worker traces; its next revision misread context-request count as token
count. Both failed revisions/logs remain preserved. The final parser selects
worker traces by their `dp` rank prefix and correctly requires `[1024,4,0]`
prefill plus three `[4,0,4]` decode forwards. No inference, timing sample,
qualification gate or tolerance changed during these analysis-only repairs.

Spark ROOT:
`/home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/full-model-v1`.
Frozen source: `ROOT/full-pairedspeedv1-src`; raw results:
`ROOT/run-pairedspeedv1/results`; independent analysis:
`ROOT/paired-speed-v1-independent`. Local evidence/analysis:
`/private/tmp/mxwave-paired-speed-v1/`. See
[current state](GLM53_CURRENT_STATE.md) and the [handoff](GLM53_LOSSLESS_RUNTIME_HANDOFF.md).
