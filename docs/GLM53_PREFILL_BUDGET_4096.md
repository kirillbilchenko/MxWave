# 4,096-token prefill-budget result

**Complete and independently verified, 19 October 2026, 14:18 UTC.** The larger budget **fails
the prospective scheduling gate**. Matched same-budget numerical checks pass
in **42/42** comparisons at unchanged limits. The
stock-minus-compressed CUDA allocation saving is **9.358188 GiB across TP2**.
No configuration is promoted. Original service restoration is verified; the
separately prepared native expert screen is the remaining authorized experiment.

Stock C16 mean TTFT changes **1.855 -> 2.032 s (+176.7 ms)**; median throughput changes **80.996 -> 84.228 tok/s**.
Compressed C16 mean TTFT changes **1.925 -> 2.056 s (+130.7 ms)**; median throughput changes **81.956 -> 85.851 tok/s**.

## Protocol and timing

Fresh stock and warmed exact16 compressed loads compare scheduler caps 2,048 and
4,096, balanced and interleaved within each arm, at fixed 10 ms admission.
Both conditions allocate for 4,096 tokens. Native mHC, the qualified compression
kernels, graph sizes [1,2,4,8,16], and 4 GiB KV allocation per rank are unchanged.
No resident BF16 cache is used. There are twelve C16 and six C1 trials per cap
and arm, with 256 prompt and 64 output tokens: 72 groups and 792 requests.
Mean request TTFT is the prospective headline. Nearest-rank p95 TTFT and
single-token ITL have separate guards; all trials remain included.

| Arm | Workload | Cap | tok/s | Mean TTFT (s) | P95 TTFT (s) | Median trial TTFT (s) |
|---|---|---:|---:|---:|---:|---:|
| control | c1-short | 2,048 | 14.894944 | 0.391125 | 0.408521 | 0.386432 |
| control | c1-short | 4,096 | 14.936683 | 0.382983 | 0.390925 | 0.381922 |
| control | c16-short | 2,048 | 80.996113 | 1.855106 | 2.680751 | 1.793479 |
| control | c16-short | 4,096 | 84.228469 | 2.031805 | 2.383685 | 2.058789 |
| candidate | c1-short | 2,048 | 16.819571 | 0.413356 | 0.421520 | 0.412088 |
| candidate | c1-short | 4,096 | 16.826330 | 0.411863 | 0.413062 | 0.411745 |
| candidate | c16-short | 2,048 | 81.956079 | 1.925447 | 2.882643 | 1.666488 |
| candidate | c16-short | 4,096 | 85.851037 | 2.056102 | 2.458474 | 2.081585 |

| Arm | Workload | Cap | P95 ITL (ms) | Mean TPOT (ms) | Single-token streams / requests |
|---|---|---:|---:|---:|---:|
| control | c16-short | 2,048 | 171.033 | 169.316 | 192 / 192 |
| control | c16-short | 4,096 | 170.600 | 161.661 | 192 / 192 |
| control | c1-short | 2,048 | 67.965 | 61.907 | 6 / 6 |
| control | c1-short | 4,096 | 68.365 | 61.979 | 6 / 6 |
| candidate | c16-short | 2,048 | 168.818 | 165.604 | 192 / 192 |
| candidate | c16-short | 4,096 | 167.706 | 156.813 | 192 / 192 |
| candidate | c1-short | 2,048 | 62.100 | 53.843 | 6 / 6 |
| candidate | c1-short | 4,096 | 62.183 | 53.798 | 6 / 6 |

Coalesced streams are excluded from single-token ITL; coverage is reported above.

This is a measurable tradeoff: compressed C16 throughput rises 4.75% and p95
TTFT improves 2.883 -> 2.458 s, but mean TTFT worsens by 130.7 ms. The larger
cap therefore helps throughput and the slowest first-token arrivals in this
short workload; it fails the declared average-latency and memory requirements.
The compressed-versus-stock mean gap shrinks from 70.3 to 24.3 ms because both
arms become slower on the mean, with a larger stock regression. That smaller
gap does not establish an absolute latency improvement.

Failed prospective checks: `candidate_C16_TTFT_improves`, `stock_C16_TTFT_nonregression`, `candidate_budget_peak_bounded`, `candidate_capacity_allocation_bounded`.
Original compression speed/TTFT decisions by cap: `{"2048": false, "4096": false}`. These are separate from the scheduling gate.

| Within-arm change | Mean TTFT change (ms) | Descriptive paired 95% interval (ms) |
|---|---:|---:|
| control / c1-short | -8.14 | -16.49 to -0.57 |
| control / c16-short | +176.70 | +125.01 to +220.48 |
| candidate / c1-short | -1.49 | -4.81 to +1.27 |
| candidate / c16-short | +130.66 | +68.21 to +178.00 |

The paired bootstrap uses 20,000 resamples with seed 4096. It describes
within-arm trial uncertainty and does not replace any gate. The model loads
are separated in time; no paired cross-arm causal interval is claimed.

## Numerics

Twenty captures retain 94 comparisons: 42 matched same-budget comparisons
are gated, and 52 cross-budget comparisons are diagnostic because grouping
changes. Limits remain mean decode KL <=1e-5 and absolute mean NLL difference
<=1e-4. These finite contexts do not establish universal zero loss or serving
batch invariance. PPL, task accuracy and SQNR were not remeasured.

| Concurrency | Cap | Cross-arm mean decode KL range | Mean NLL difference range | All matched checks pass |
|---|---:|---:|---:|---|
| C1 | 2,048 | 5.80377826e-06 to 5.80377826e-06 | -3.65099067e-06 to -3.65099067e-06 | True |
| C16 | 2,048 | 8.91934572e-07 to 8.91934572e-07 | -1.41760571e-06 to -1.41760571e-06 | True |
| C1 | 4,096 | 5.80377826e-06 to 5.80377826e-06 | -3.65099067e-06 to -3.65099067e-06 | True |
| C16 | 4,096 | 8.19999075e-08 to 8.19999075e-08 | -3.83740389e-07 to -3.83740389e-07 | True |

Maximum matched self-repeat KL: **0**. Timed output equality: **True**, over 1,188 paired sequences.
Cross-budget control diagnostic KL reaches 4.0570844e-05. Cross-budget candidate diagnostic KL reaches 3.9894997e-05.

## Profiles and memory

Ordinary profiles are separate from unprofiled timing. The table retains both
ranks; reconstruction durations are kernel sums, not API latency or a TP
critical path. One profile per arm/cap does not identify every timed schedule.

| Arm | Cap | Rank | Prefill rows | Rebuild calls | Rebuild sum (ms) | Decode graph replays |
|---|---:|---:|---|---:|---:|---:|
| candidate | 2,048 | 0 | 2048 / 2048 / 23 | 204 | 110.182 | 16 |
| candidate | 4,096 | 0 | 2560 / 266 / 1291 | 204 | 108.390 | 15 |
| control | 2,048 | 0 | 2048 / 2048 / 23 | 0 | 0.000 | 16 |
| control | 4,096 | 0 | 4096 | 0 | 0.000 | 15 |
| candidate | 2,048 | 1 | 2048 / 2048 / 23 | 204 | 108.866 | 16 |
| candidate | 4,096 | 1 | 2560 / 266 / 1291 | 204 | 110.233 | 15 |
| control | 2,048 | 1 | 2048 / 2048 / 23 | 0 | 0.000 | 16 |
| control | 4,096 | 1 | 4096 | 0 | 0.000 | 15 |

The larger cap permits a single prefill pass, as the stock profile demonstrates,
but does not guarantee that all requests enter that pass. The compressed profile
still has three admission groups and 204 reconstructions per rank. Raising the
token budget alone therefore does not reliably remove the trailing passes.
These separate profiles describe observed mechanisms, not every timed trial.

All 16 worker traces verify; 120 retained exact16 decode ranges pass the unchanged fused-dispatch mechanism checks. No recognized JIT overlap: **True**.
control: boundary SM clocks 2437–2515 MHz; temperatures 57–75 C. candidate: boundary SM clocks 2431–2509 MHz; temperatures 61–78 C.
Boundary readings cannot exclude transient throttling; no clock normalization or trial filtering is applied.

Memory saving: **10,048,278,016 bytes / 9.358188 GiB across TP2**. Memory decisions: `{"TP2_saving_retained": true, "candidate_budget_peak_bounded": false, "candidate_capacity_allocation_bounded": false}`.
Rank 0 capacity allocation changes by **+290.350 MiB** from the frozen historical 2,048-capacity baseline. Rank 1 capacity allocation changes by **+298.375 MiB** from the frozen historical 2,048-capacity baseline.
Within the new capacity, changing the active cap from 2,048 to 4,096 raises the
maximum compressed allocated peak by **424.827 MiB per rank**.
The prospective capacity/peak allowance remains 128 MiB per rank. Both cap
conditions share the new allocation capacity, so the within-run cap comparison
alone cannot reveal that historical cost. Reserved CUDA and CPU RSS are retained
separately; overlapping unified-memory counters are never added.

## Retention and closure

Spark and Mac independently verify **1,639 payloads / 217 sources / 16 worker traces**. All decisions agree; maximum float64 difference is 1.776e-15. Archive SHA256:
`72366e1e4fc6180ad6a0183db1bd641160688d9f4b15d21f100a82169f1c1491`.

All 206 parent code bindings remain unchanged; one plugin metadata binding
changes, and ten modules are added, for 217 bindings. The dispatch capacity
extension passes 39 guarded startup variants. Validation: 814 tests pass,
one GPU skip, Ruff and strict core mypy pass. Actual layer-23 MoE inputs and
outputs are captured only after stock measurement, for a separate native
expert screen. They do not instrument the timing graphs.

The original containers are restored with exact identities and HTTP 200;
controller, collector and watchdog are terminal, and the owned awake lease
is released. Earlier failures and historical measurements remain retained.
The native expert screen has its own source freeze, bounded pause and numerical
requirements. It is not qualified by this scheduling result.

- [Final findings](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/prefill-budget-v3-final-findings.json.gz)
- [Independent verification](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/prefill-budget-v3-cross-host-verification.json)
- [Resource review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/prefill-budget-v3-resource-review.json)
- [Profile review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/prefill-budget-v3-fresh-profile-review.json.gz)
- [Uncertainty review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/prefill-budget-v3-uncertainty-review.json)
- [Operational closure](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/prefill-budget-v3-terminal-cleanup.json)
- [Final retention manifest](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/prefill-budget-v3-final-retention.json)
