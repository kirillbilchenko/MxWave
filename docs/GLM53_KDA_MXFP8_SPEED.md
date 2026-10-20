# Actual MXFP8 KDA serving comparison

<!-- KDA_MXFP8_SPEED_LIVE_BEGIN -->
**Final checkpoint, 20 October 2026, 09:35 UTC.** Native MXFP8 speed and
full GSM8K are complete and independently verified. Speed: all nine loads,
115 payloads and 7,441 numerical fields verify. GSM8K: both 1,319-question
passes, 2,690 payloads and 2,638 independently regraded answers verify;
2,933 numerical fields match exactly across Spark/Mac.

Fresh CT BF16 KDA scores **1,248/1,319 (94.62%)**; native MXFP8 KDA scores
**1,260/1,319 (95.53%)**. There are 22 gains and 10 losses, net +12 correct.
Both arms have zero API errors or truncations. The earlier **1e-5 KL gate
remains failed**; the higher observed task score does not establish losslessness.

Original production is restored on port 8000 with **500,000 context**, exact
original identities/configurations and healthy status verified at **09:35:00 UTC**.
No candidate is promoted. The read-only drafter inventory is also complete;
no further GPU experiment has started. See [the speed results](GLM53_KDA_MXFP8_SPEED.md),
[full GSM8K results](GLM53_KDA_MXFP8_GSM8K.md), and
[drafter findings](GLM53_DRAFTER_COMPARISON.md). This checkpoint supersedes older
service-state and interim-score statements below.
<!-- KDA_MXFP8_SPEED_LIVE_END -->

**Native MXFP8 serving comparison complete, 20 October 2026, 08:12 UTC.**
All nine loads and 108 timing groups finish. All **115 archived payloads**
verify; **7,441 recomputed fields match exactly** on Spark/Mac.
Six final profiles and all 232 target graph ranges are reviewed; source-order
KDA attribution has no unmatched graph ranges.
Original production was restored at **08:08:20 UTC**, with exact identities and
500,000 context, before the authorized GSM8K follow-up paused it again.

| Case | Current CT lossless BF16 KDA | CT MXFP8 KDA | ModelOpt, matched settings | CT MXFP8 vs ModelOpt | Repeat controls for that comparison |
| --- | ---: | ---: | ---: | ---: | --- |
| C1 diagnostic, 256 output tokens | 66.05 tok/s | 71.57 tok/s | 69.67 tok/s | +2.74% | Pass |
| C1 code | 39.24 tok/s | 44.86 tok/s | 49.09 tok/s | **−8.61%** | Pass |
| C1 explanation | 23.91 tok/s | 35.85 tok/s | 32.37 tok/s | +10.72% | **Fail: ModelOpt repeat** |
| Offered C8 | 64.04 tok/s | 89.14 tok/s | 79.30 tok/s | +12.41% | **Fail: both repeats** |
| Offered C16 | 72.28 tok/s | 95.13 tok/s | 83.95 tok/s | +13.32% | **Fail: candidate repeat** |

These are completed measurements, not a universal production speed qualification.
All modes run at most six resident requests; C8/C16 include queueing. Code is
14.31% faster than current CT, but remains 8.61% below matched-settings ModelOpt.
Mean code first-token time improves 459.8 → 399.3 ms versus current CT and
442.7 → 399.3 ms versus ModelOpt. The earlier code diagnostic shows lower
speculative acceptance requires more rounds; the final trace review below confirms the KDA component saving.

CT MXFP8 saves a further **1.105/1.086 GiB per GPU** versus current CT lossless
BF16 KDA. Its startup allocation is **7.447/7.430 GiB per GPU lower** than ModelOpt
under the same KV budget. That latter comparison includes different checkpoint
formats and is not solely a lossless-compression saving. The prior **1e-5 KL
failure remains**; the completed fresh GSM8K comparison scores 1,248 → 1,260 correct.
See [the complete speed table and evidence](GLM53_KDA_MXFP8_SPEED.md).

**Full GSM8K follow-up now starting (`kda-mxfp8-gsm8k-v2`).** Both CPU launch
checks pass and the control model is loading. The original v1 preparation failed
before inference because `queue.py` shadowed Python's standard module; that
failure is retained. V2 renames the launcher without changing prompts, scoring,
models or serving policy. Controller 110454 owns restoration; remote queue 108092,
local collector 9362 and live monitor 9363 track the run. Both arms answer all
1,319 questions with speculation off. Drafter audit 9364 waits until speed and
GSM8K artifacts are independently verified. No automatic promotion occurs.

**Final trace review complete.** All 232 retained target graph ranges pass
source-order attribution of all 68 KDA GEMMs; none are left unattributed. At
physical width 8, KDA input+output kernel sums are 22.739/22.795 ms for CT BF16,
12.977/13.132 ms for CT MXFP8 and 12.971/13.242 ms for ModelOpt (ranks 0/1).
Thus the expected roughly 9–10 ms KDA saving is realized in these traces.
Eight-row whole-graph median spans are 97.251/97.541 → 87.871/88.271 ms versus
CT BF16; ModelOpt is 91.152/91.427 ms. Kernel sums are not API latency.

Both MXFP8 traces have **zero weight reconstruction kernels across the full
recorded session**. CT BF16 has 5,848 reconstruction calls per rank across
captured/eager work, including **476 calls in seven width-12 graphs** (68 each).
The eager scopes are not all prefill by assumption. Native graph replay widths
are 6, 8, 12 and 16. Width-16 whole-graph spans remain 135.742/135.755 ms versus
ModelOpt's 124.310/124.722 ms; there is no claim that every graph becomes faster.
Output paths, routing and physical batch work differ. See the
[final profile summary](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/kda-mxfp8-speed-v1-final-profile-summary.json).

The frozen experiment uses A/B/P, B/P/A, P/A/B load order, with two trials per
case/load: six trials per mode/case, nested within three model loads. Common
settings are DFlash 7/5/3, 6 GiB KV per rank, 16 slots, a 2,048-token prefill
budget, disabled prefix cache, 10 ms idle admission and graphs `[1,2,4,8,16]`.
Context is 8,192 on private port 8014. ModelOpt uses the production checkpoint
with these matched settings, not its fully tuned production configuration.
Seeds, temperature zero, prompt token IDs and output budgets are fixed.

Every native MXFP8 load passes all 68 matrix checks per rank, matches the prior
rounding hashes and releases all original BF16 KDA parameters. The installed
consumer is Marlin W8A16. These component checks do not establish full-model
numerical equivalence. The checkpoint files remain unchanged.

The unchanged 2% repeat control measures the spread of the three load medians.
C1 diagnostic-256 and code pass in every mode. Explanation passes in both CT
modes, while ModelOpt fails at 4.11%. C8 spreads are 8.31%/3.01%/6.54% for
CT BF16/CT MXFP8/ModelOpt; C16 spreads are 4.75%/3.72%/0.11%. Short C1 diagnostic
also fails all three controls (3.12%/2.57%/9.71%). Preserve every failed control.
The short diagnostic medians are 47.64/52.01/50.78 tok/s, respectively.

Mean TTFT, CT BF16 / CT MXFP8 / ModelOpt:

| Case | CT BF16 | CT MXFP8 | ModelOpt |
| --- | ---: | ---: | ---: |
| C1 diagnostic-256 | 443.7 ms | 413.3 ms | 435.1 ms |
| C1 code | 459.8 ms | 399.3 ms | 442.7 ms |
| C1 explanation | 356.4 ms | 325.2 ms | 339.2 ms |
| Offered C8 | 4.975 s | 3.949 s | 4.103 s |
| Offered C16 | 17.868 s | 13.409 s | 14.790 s |

The earlier first-two-load code analysis finds 54 candidate draft rounds versus
47 ModelOpt rounds at the fixed 256-token budget, with acceptance 54.23% versus
63.22%. Dividing decode wall time by counted rounds gives 98.32 versus 101.27 ms:
a wall-time proxy, not kernel duration. More rounds outweigh cheaper rounds.
Outputs repeat within each mode but diverge between modes at token index 107.
Acceptance is not a task-accuracy score. The
[code diagnostic](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/kda-mxfp8-speed-v1-code-regression-review.json)
retains counters and limitations. The separate
[drafter audit](GLM53_DRAFTER_COMPARISON.md) follows full GSM8K.

Evidence: [complete recomputed measurements](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/kda-mxfp8-speed-v1-final-mac-review.json),
[independent verification](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/kda-mxfp8-speed-v1-final-independent-verification.json)
and [archive receipt](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/kda-mxfp8-speed-v1-final-archive-receipt.json).
The 373,823,476-byte archive has SHA-256
`2a5b53a90afc0f1ef2fe96f1d8648707d1e1cf840d296e2414cc5858625018a1`.
Local evidence: `/private/tmp/mxwave-kda-mxfp8-speed-v1-final`.
Earlier provisional reviews remain retained; the complete table above supersedes them.
