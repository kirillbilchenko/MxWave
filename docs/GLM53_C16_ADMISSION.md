# Final short-C16 admission experiment

**Completed and independently verified, 19 October 2026, 09:01 UTC.** The 10 ms admission
policy **passes its prospective gate**. Compressed short-C16 median TTFT at
10 ms is **1.694 s**, versus stock
**1.383 s**: a **+311.5 ms** gap.
The hoped-for gap of tens of milliseconds is not achieved under the frozen headline metric;
**the original short-C16 TTFT criterion remains unmet**. This headline is the
median of twelve trial medians. Supplementary statistics over all 192 requests
per arm at 10 ms give a **+57.4 ms mean** gap and **+65.9 ms pooled
median** gap. These describe the same complete data; they do not replace the
frozen headline metric or its decision.
Within the compressed arm, moving 3 -> 10 ms changes C16 TTFT by
**-20.56%** and throughput by
**+0.82%**. The C1 window delay is
**+8.9 ms**.

All 94 numerical comparisons pass at the unchanged KL <=1e-5 / absolute mean NLL <=1e-4 limits.
Timed output pairs match: **True** across
all **1188** retained comparisons. CUDA allocation
saving is **9.367162 GiB across TP2**. These measurements
cover short C1/C16, not every prompt length or concurrency.

| Short workload / window | Stock tok/s | Compressed tok/s | Throughput change | Median TTFT, stock -> compressed | TTFT gap |
|---|---:|---:|---:|---:|---:|
| c1-short / 3 ms | 14.856 | 16.745 | +12.71% | 0.376 -> 0.404 s | +27.6 ms |
| c16-short / 3 ms | 80.417 | 81.671 | +1.56% | 2.025 -> 2.133 s | +107.3 ms |
| c1-short / 10 ms | 14.835 | 16.691 | +12.50% | 0.380 -> 0.412 s | +32.7 ms |
| c16-short / 10 ms | 80.657 | 82.344 | +2.09% | 1.383 -> 1.694 s | +311.5 ms |

Each arm uses twelve short-C16 and six short-C1 repeats per window, all with
256 prompt / 64 output tokens. The 72 groups contain 792 requests and 50,688
output tokens. Policy order is balanced within each workload; the two fresh
model loads have fixed stock-then-compressed order. All trials are included.
No clock correction or outlier removal is applied.

| Matched concurrency | Cross-arm mean decode KL range | Numerical decision | Within-arm/window repeats bitwise equal |
|---|---:|---|---|
| C1 | 5.80377826e-06 to 5.80377826e-06 | Pass | True |
| C16 | 8.91934572e-07 to 8.91934572e-07 | Pass | True |

Quality includes 20 captures and all 94 same-concurrency arm/window/repeat/profile
comparisons. Actual worker grouping is checked. C16 quality captures use matched controlled
cohorts; ordinary timing uses native admission without pause/resume, full logprobs
or profiling. Native mHC and deterministic sparse/expert ordering remain in both
arms. Finite numerical qualification does not establish universal bitwise
equivalence; PPL, task accuracy and SQNR are not remeasured.

The policy gate requires at least 5% compressed C16 TTFT improvement, no more
than 5% stock C16 TTFT regression, at least 98% throughput retention and at most
15 ms added C1 TTFT, with quality and timed-output checks passing:

- `quality`: pass.
- `timed_outputs`: pass.
- `throughput_nonregression`: pass.
- `C1_delay_guard`: pass.
- `candidate_C16_TTFT_improves`: pass.
- `stock_C16_TTFT_nonregression`: pass.

The historical compression >=5% throughput / <=5% TTFT gates are retained
separately: **{'3': False, '10': False}**. User scope
accepts measured concurrency parity and ends further compression-kernel tuning.
A policy-gate pass alone does not establish that all serving latency is solved;
the actual compressed-versus-stock gaps are in the table.

Observed first admissions across all twelve C16 timing trials:

- control, 3 ms: first cohorts [3, 7, 6, 8, 7, 3, 2, 7, 4, 1, 8, 8]; first 2,048-token budget filled in 3/12 trials.
- control, 10 ms: first cohorts [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 3]; first 2,048-token budget filled in 11/12 trials.
- candidate, 3 ms: first cohorts [6, 8, 3, 3, 6, 8, 5, 7, 3, 3, 7, 1]; first 2,048-token budget filled in 2/12 trials.
- candidate, 10 ms: first cohorts [8, 8, 8, 6, 8, 8, 8, 8, 5, 8, 8, 8]; first 2,048-token budget filled in 10/12 trials.

| Arm / window / rank | Prefill-containing forwards | Prefill row counts | Reconstruction calls / kernel sum | Full-size decode graph launches |
|---|---:|---|---:|---:|
| candidate / 3 ms / 0 | 3 | [768, 2048, 1293] | 204 / 108.39 ms | 14 |
| candidate / 10 ms / 0 | 3 | [2048, 2048, 23] | 204 / 108.56 ms | 12 |
| control / 3 ms / 0 | 4 | [1024, 260, 2048, 785] | 0 / 0.00 ms | 14 |
| control / 10 ms / 0 | 3 | [2048, 2048, 23] | 0 / 0.00 ms | 12 |
| candidate / 3 ms / 1 | 3 | [768, 2048, 1293] | 204 / 108.95 ms | 14 |
| candidate / 10 ms / 1 | 3 | [2048, 2048, 23] | 204 / 109.49 ms | 12 |
| control / 3 ms / 1 | 4 | [1024, 260, 2048, 785] | 0 / 0.00 ms | 14 |
| control / 10 ms / 1 | 3 | [2048, 2048, 23] | 0 / 0.00 ms | 12 |

These are separate ordinary profile samples, one/window/arm on both GPUs.
They are not simultaneous traces of all twelve timing trials. The verifier
independently checks every recorded model forward, exact16 graph replay, 68
qualified fused KDA calls and zero reconstruction in covered full-size decode
ranges. Prefill may still contain a small final chunk: ongoing decode consumes
part of the shared 2,048-token budget. Kernel sums are not API wall time or
measured memory bandwidth. No recognized compilation event overlaps timing:
**True**.

At 10 ms, both ordinary profiles have the same prefill row counts
**2048 / 2048 / 23**. Each compressed prefill forward reconstructs 68 matrices,
costing **35.39-37.37 ms of kernel sum per rank per forward**; the three-forward
total is **108.56 ms on rank 0 / 109.49 ms on rank 1**. Covered full-size decode
continues to use graphs and fused kernels without reconstruction. This locates
a real prefill cost, without attributing the whole API gap to kernel sums.

The admission implementation stops collecting as soon as the 2,048-token budget
is full: eight 256-token prompts. Thus 10 ms is an upper wait bound and does not
guarantee collection of all sixteen requests. At 10 ms, stock fills that first
budget in 11/12 trials and compression in 10/12. Even filled-first-budget trials
have different later first-token arrival groups. The eighth/ninth arrival pair
sets each C16 trial median, making the median of trial medians sensitive to this
boundary. Across all requests, stock/compressed means are
**1.849240 / 1.906627 s**;
pooled medians are **1.574117 /
1.640005 s**. All twelve trials
remain in the frozen result. This is supplementary diagnosis, not proof that a
specific later scheduling change removes the gap. The
[remaining-latency review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/c16-admission-v1-remaining-latency-review-v2.json)
retains every trial, middle arrival, profile stage and source binding.

Both arms retain graphs 1/2/4/8/16, 4 GiB KV per rank, 16 sequences, the
2,048-token budget and identical 34-variant startup warmup/heap trim. The 207
qualified warmed exact16 parent sources are unchanged; four additions implement
the admission experiment. The failed input-FP32 variant is excluded. No kernel
implementation, model weights, speculative decoding or transport configuration
is changed. Local validation passes **796 tests / one GPU skip, Ruff and strict
core mypy**; the CPU protocol simulation rejects injected grouping, numerical
and compilation failures. Six trace fixtures reject wrong kernels/geometry,
missing rank/output or absent graph replay. No additional GPU experiment ran.

The measured CUDA saving is retained separately from CPU RSS. CPU process
counters overlap CUDA allocations on Spark and must not be added; post-profile
RSS is not evidence of a durable CPU-memory saving. Per-window clock/thermal
observations and all trial/request data are retained in the final findings.

The user's FP32 KL follow-up was investigated using retained evidence, without
another GPU experiment. The audit rehashes **787 files** and
reproduces the exact native source adaptations. The qualified fused kernel
already accumulates in FP32; the failed variant changes split-K from one to two,
stores two FP32 partial sums and combines them before the BF16 output conversion.
This avoids BF16 partial-store loss but still changes accumulation grouping.
Producer/reducer strides agree in the inspected source. No small reducer bug
or first differing model layer is established by this audit.

In the retained C16 case, 26/48 decode rows have higher KL with the FP32 variant;
one row accounts for 54.49% of its total KL. All rows remain included. The tested
recovery is the complete qualified route used in this admission experiment:
it preserves compression and restores the earlier passing numerical result.
Switching only physical row count 16 would need fresh qualification because
earlier smaller admitted batches can affect later state. Exact stock agreement
would require matching its GEMM arithmetic or using its GEMM on reconstructed
weights, with the associated latency/memory tradeoff. The
[retained KL audit](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/c16-admission-v1-fp32-kl-audit.json) records all per-row
values and source hashes; neither runtime nor numerical thresholds were changed.

Spark and Mac independently verify **1603 payloads / 211 source
bindings / 16 worker traces**, with identical decisions and maximum float64
difference **1.77598e-15**. All prior failures,
including the input-FP32 C16 failure, remain preserved. Original service identities
and configuration are restored, health 200; controller, collector and watchdog
are terminal and the owned awake lease is released. No configuration is promoted.
This completes the user's last compression experiment. No further compression
experiment or kernel tuning is planned; MTP is a separate later phase.

The initial evidence collector could hardlink completed result payloads but was
denied hardlinks to container-owned profiler files. Its failed receipt and log
remain in the archive. A separately recorded recovery collector verified the
partial evidence and copied head traces; it changed neither the 211 experiment
sources nor the verifier, measurements or gates. Six CPU recovery checks pass.
This required no new GPU measurement. See the
[archival recovery plan](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/c16-admission-v1-collection-recovery-plan.json).

Evidence: [final findings](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/c16-admission-v1-final-findings.json),
[cross-host proof](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/c16-admission-v1-cross-host-verification.json),
[final retention manifest](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/c16-admission-v1-final-retention.json), and
[frozen preparation](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/c16-admission-v1-preparation-retention.json).
Local archive: `/private/tmp/mxwave-c16-admission-v1/evidence.tar.gz`.
Remote: `ROOT/evidence-c16-admission-v1.tar.gz`, where
`ROOT=/home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/full-model-v1`.
Archive SHA256: `68499bcd530ef1a9ed337fa03ee1e90272f5c0ec0c099b8a9f7b64efbc236809` (466,028,470 bytes).
Frozen source archive SHA256:
`0ad0d869d7a8ac2810fa59ffc1715eeeedc24e5361f882816f957dd7f2ad4bcd`.
Plan SHA256: `b85013c785af0cd230f4739bb2faf527e495018f390dd4464b30e6a68716c9a4`.
