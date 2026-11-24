# Full GSM8K comparison for native MXFP8 KDA

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

**Complete and independently verified, 20 October 2026, 09:35 UTC.**
Both fresh arms answer all 1,319 questions. All **2,638 raw responses** are
independently regraded on Mac; **2,933 numerical fields** match Spark exactly.
All **2,690 archived payloads** verify. The exact original production containers,
configuration and **500,000 context** are restored, with authenticated health
confirmed at **09:35:00 UTC**. No candidate is promoted.

| Completed metric | Fresh CT lossless BF16 KDA | Native MXFP8 KDA |
| --- | ---: | ---: |
| Numeric correct | 1,248/1,319 | **1,260/1,319** |
| Numeric accuracy | 94.617% | **95.527%** |
| Strict-format correct | 1,226/1,319 | **1,245/1,319** |
| Strict-format accuracy | 92.949% | **94.390%** |
| API errors / truncations | 0 / 0 | 0 / 0 |
| Workload wall time | 37m44.4s | **33m18.7s** |
| Completion tokens | 94,888 | 96,298 |
| Workload output tokens/s | 41.90 | **48.18** |

The paired result is **22 wrong-to-correct, 10 correct-to-wrong**, a net
**+12 correct / +0.910 percentage points**. The exact two-sided McNemar
p-value is **0.05010246**. This is a higher observed score in one full pass,
not established general superiority or a pass at a strict p < 0.05 cutoff.
The prior **1e-5 KL gate remains failed**: this MXFP8 variant changes model
probabilities even though this task score improves.

The candidate finishes this workload **11.73% sooner**, with **14.98% higher
output-token throughput**, while generating 1.49% more tokens. These are
single-pass C4 GSM8K workload measurements with speculation off and natural
stopping; they do not replace the interleaved serving-speed experiment or
isolate kernel time.

Against the historical original CT run, the candidate scores **1,260 versus
1,246**: 25 previously failed questions recover and 11 previously correct
questions regress. The fresh control recovers 22 of the historical 73 failures
and introduces 20 failures, finishing at 1,248. Historical comparisons also
include runtime/admission changes and are contextual. These are CT comparisons,
not measured GSM8K scores for the NVFP4 production checkpoint.

The frozen grader is unchanged. Sixteen fresh-control failures and eleven
candidate failures contain a correct preceding number followed by an empty
trailing `####`; all retain their original failed scores. The strict-format
metric and these diagnostics are separate from the primary numeric metric.
Both ranks pass all 68 native KDA matrix checks, prior rounding hashes match,
and original BF16 KDA parameter references are released.

Evidence: [complete regraded results and changed-question IDs](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/kda-mxfp8-gsm8k-v2-final-mac-review.json),
[independent verification](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/kda-mxfp8-gsm8k-v2-final-independent-verification.json),
[archive receipt](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/kda-mxfp8-gsm8k-v2-final-archive-receipt.json),
and [restored service](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/kda-mxfp8-gsm8k-v2-final-final-service-review.json).
The 3,891,220-byte archive SHA-256 is
`aa911801d3369a51b13174e06fdad4fe89d320eb547b73ee6567fc40eb0f945a`.

The protocol was prepared **20 October 2026, 06:48 UTC** at the user's request.
It followed all nine [speed loads](GLM53_KDA_MXFP8_SPEED.md), their restoration
and evidence retention. Quality requests did not overlap the speed experiment.

The two arms in the completed comparison were:

| Arm | KDA | Other weights and runtime |
| --- | --- | --- |
| Fresh control | Current lossless BF16 KDA route | CT MXFP4 experts with packed scales; compressed vocabulary |
| Candidate | Native Marlin MXFP8 KDA | Identical to fresh control |

Speculation is off in both arms. The exact historical prompts and request bodies
are reused: zero-shot chat, temperature 0, seed 42, low reasoning effort,
1,024-token cap with natural stopping, and four concurrent requests. Both use
4 GiB KV per GPU, four sequence slots, graphs `[1,4]`, a 2,048-token prefill
budget, disabled prefix caching, 10 ms idle admission, and 8,192 context.

The historical uncompressed CT baseline scored **1,246/1,319 (94.47%)**; its
lossless candidate scored **1,244/1,319 (94.31%)**. Those runs used an older
runtime and admission configuration. They provide context, while the fresh
paired arms isolate the KDA change more closely. They are not scores for the
NVFP4 production checkpoint or the original BF16 source model.

Keep every answer, truncation and API error in the denominator, with no retries.
Report correct-to-wrong and wrong-to-correct pairs, the historical 73-failure
subset, primary numeric accuracy and the unchanged strict-format score. The
original graders reproduce all 2,638 historical responses exactly. A separate
CPU fixture exercises collection, raw/compact retention and independent
regrading using those responses, reproducing 1,246/1,244 and 22 lost/20 gained.
Its synthetic startup metadata checks the analysis contract only.

Nineteen historical reference failures had the right preceding number but
an empty trailing `####` marker. Preserve their original scores; list this
format issue separately rather than changing extraction after observing results.

The candidate passes all 68 native-kernel matrix checks on each rank,
matches the prior rounding hashes, and releases original BF16 KDA allocations.
These component checks do not establish full-model numerical equivalence.
The previous **1e-5 KL gate remains failed**. Neither the improved sampled
WikiText PPL nor the higher observed GSM8K score establishes losslessness.

CPU launch validation followed the speed run, then the comparison used private
port **8015**. Original production was paused and is now restored by the
controller/watchdog. The complete controller lifetime was approximately
83m40s, within its 3.5-hour bound. No checkpoint was rewritten or promoted.

Retained local staging: `/private/tmp/mxwave-kda-mxfp8-gsm8k-v2`.
Retained Spark staging: `full-model-v1/kda-mxfp8-gsm8k-v2`.
Local final evidence: `/private/tmp/mxwave-kda-mxfp8-gsm8k-v2-final`.
The queue, launch protocol, source hashes, raw responses and restoration receipt
are retained; every payload and answer has been independently checked on Mac.

The authorized [drafter comparison](GLM53_DRAFTER_COMPARISON.md) has now completed
its read-only checkpoint inventory: draft checkpoint/config files match across
both nodes, and matched-test speculative settings are identical. Effective
shared heads and target-conditioned histories remain differences to isolate.
No additional GPU experiment has started.

**Launch correction, 20 October 08:12 UTC.** V1 failed CPU preparation because
`queue.py` shadowed the standard library; no inference or service outage occurred
in that attempt. Its logs and source manifest are retained. V2 renames the
launcher to `launch_followup.py`, preserves all measurement/scoring settings,
and passes both CPU launch checks. Controller 110454 subsequently completed
both arms and restoration. The runtime binds the same 211 frozen source files per rank.
