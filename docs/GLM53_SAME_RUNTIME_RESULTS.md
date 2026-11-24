# Final same-runtime comparison results — 22 October 2026

All six timing stages and both 300-question quality stages are complete. Exact
original production is restored and independently verified healthy at 500k
configured context, **09:34:54 UTC**. Both original identities match; all 20
containers owned by the timing and quality runs are stopped. Both Sparks stay on;
no promotion, additional GPU work or shutdown is queued.

The frozen decision is **retain production; CT remains an experimental opt-in**.
CT has higher throughput medians and lower allocation, but all eight arm/cell
repeat controls miss the 2% limit and CT scores four fewer correct answers.
The small quality subset does not establish a general accuracy loss or equivalence.

## Scope

Production ModelOpt lossless8 versus CT MXFP4 experts + MXFP8 KDA + lossless
expert-scale/embedding packing; dense BF16 head in both. Both use the same
production image, overlays, Inco FP8 drafter, configured 7/5/3 draft policy,
500k context, 32 slots, 6 GiB KV/rank, 6,919-token budget, prefix caching and
original graph settings. Actual admission window is 4 ms. This compares different
quantized checkpoints and is not an isolated lossless-compression comparison.
It is a custom-production-runtime test, **not a new stock-vLLM plugin result**.
The distributable recipe must still use an unmodified official vLLM image.

Same enabled overlays do not guarantee the same dispatch: the dense fast hook
attaches to 53 MLA projections in production and zero in CT on both ranks, in all
three starts. No overlay or quantization fix was introduced during this test.

## Timing

Three interleaved A/B pairs, fixed seeds, realistic frozen short code/chat prompts,
256 output tokens, warmup per cell, intentionally cold per-request prefix salts.
All 24 measured cells and 108 requests return the required tokens without errors.
Units below are end-to-end output tok/s; C8 is aggregate across eight submitted
requests. Both arms reach six running requests with two waiting at C8.

| Workload | Production median [range] | CT median [range] | Median CT change |
|---|---:|---:|---:|
| Code C1 | 46.01 [39.48–51.96] | 49.35 [46.68–55.75] | +7.26% |
| Chat C1 | 39.06 [36.83–39.56] | 45.11 [43.77–46.77] | +15.49% |
| Code C8 | 74.35 [71.34–81.73] | 86.29 [85.73–90.79] | +16.07% |
| Chat C8 | 72.89 [72.84–77.20] | 83.71 [81.53–86.42] | +14.85% |

The normalized ranges `(max−min)/median` are 5.85–27.13%, exceeding 2% in every
arm/cell. Output token sequences also vary across trials despite fixed seeds.
CT wins all three paired trials for chat C1 and both C8 cells, but loses the second
code C1 pair. These are retained observations, not qualified stable speedups.

Latency in seconds, production / CT. C1 p95 is based on only three requests and
C8 p95 on 24; these are descriptive values for this screen.

| Workload | Mean first token | p95 first token | Mean request completion |
|---|---:|---:|---:|
| Code C1 | 0.704 / 0.465 | 0.870 / 0.486 | 5.658 / 5.087 |
| Chat C1 | 0.437 / 0.438 | 0.464 / 0.465 | 6.658 / 5.665 |
| Code C8 | 5.453 / 4.637 | 19.073 / 15.149 | 20.597 / 17.331 |
| Chat C8 | 5.321 / 5.056 | 18.922 / 16.652 | 20.723 / 18.378 |

Timestamp-matched telemetry has 21–25 observations per arm/rank: P0 throughout,
59–69°C, SM clocks 2,457–2,515 MHz, with no active clock-throttle reasons sampled.
Clocks were not locked; sampling cannot exclude transient host or device contention.
Raw clock, power, thermal and container samples are retained.

## Quality

Same frozen answer-independent **300-question GSM8K subset**, speculation off,
concurrency four, zero-shot, seed 42, temperature zero, max 1,024 output tokens.
No retries. All 600 stored answers independently re-score to the saved results.

| Metric | Production | CT |
|---|---:|---:|
| Correct numeric final answer | 291 / 300 (97.00%) | 287 / 300 (95.67%) |
| Strict output format + correct answer | 290 / 300 (96.67%) | 286 / 300 (95.33%) |
| API errors / truncations | 0 / 0 | 0 / 0 |

Paired outcomes: 284 both correct, seven production-only correct, three CT-only
correct, six both wrong. CT is −1.33 percentage points on this screen and misses
the predeclared `CT correct >= production correct` condition. The previous strict
KL failures remain unchanged; this is neither a KL pass nor a full 1,319-question
benchmark. Both quality summaries have the same protocol SHA256
`33d1fbf59bcd908edab19b79e48176b70c0914813580d38d145424fa7d09a793`.

## Memory and caching

Post-startup allocated GPU memory, including drafter/KV/graphs, repeats exactly
across all three starts: production **96.838109 GiB/rank**; CT **89.605322 / 89.605072
GiB** on ranks 0/1. Saving: **7.232787 / 7.233038 GiB**, **14.465825 GiB across TP2**.
This includes checkpoint-format and MXFP8 quantization differences; it is not all
lossless-packing savings and must not be added to earlier experiments. Reserved
memory, model-only accounting and system RSS are separate metrics.

All six cache probes pass the hit/miss sequence: 0 / 9,216 / 0 tokens for
initial / same-salt repeat / fresh-salt request. This establishes cache reuse,
not full numerical identity: the one-token same-salt outputs differ in the first
trial for both models and match in their later two trials. Earlier repeated-prefix
numerical concerns therefore remain. Ordinary timed requests deliberately produce
zero cache hits; GSM8K uses a different salt per question. This explains the 0%
cache-hit logs without implying caching is disabled.

## Interruptions, upload and provenance

V3 completes all timing, then stops at the first quality client's import because
the launcher omitted `numeric_scoring.py`. No quality question was sent. Production
restores at 09:07:26 UTC. The helper is copied unchanged from the original protocol;
a new isolated CPU import/scoring check passes before a separate quality-only
continuation. Timings are not rerun, no question is retried, and the original
10:27:14 UTC absolute watchdog deadline is preserved. The continuation finishes
both quality arms and restores production at 09:34:54 UTC.

The user's currently running Spark2 rsync processes started at **09:07:58 UTC**,
after all measured speed requests (08:30:33–09:01:59). They overlap quality A
(09:12:48–09:20:30) and B (09:24:18–09:32:03). This transfer does not explain the
recorded speed variation. It may affect quality-request latency; no causal link
to answer accuracy is established. Earlier unrelated transfers are not excluded.
The transfer was observed by process name/start time only and was not interrupted.

Evidence roots are `glm53-package-v1/same-runtime-final-v3`,
`same-runtime-final-quality-v1`, and `same-runtime-final-closeout-v1`.
The closeout includes `review.py`, raw-result `summary.json`, both-rank startup
reports, upload observation, restoration evidence and the archive manifest.
The CPU reviewer recalculates throughput from token counts/wall time, scores every
answer again, reconciles all startup configurations and memory, and retains all
failed repeat controls. Prior failed v1/v2 evidence remains separately preserved.

The [frozen protocol](GLM53_SAME_RUNTIME_FINAL.md),
[live/history handoff](GLM53_HANDOFF.md), and
[stock-vLLM recipe](../deployment/glm53/README.md) retain their distinct scopes.

Final evidence archive: **810 payloads**, **4,211,408 bytes**, SHA256
`6c7eadf167eb4280165e7b272b7b8911356cd95f601f31826220070ae401db4e`.
The archive is retained on Mac and both Sparks. Every payload independently
hash-verifies on Mac and Spark2; CPU recomputation reproduces the entire summary
exactly on both, including all 600 re-scored answers and 108 measured timing
requests. Local evidence/receipts:
`out/glm53-package-v1/same-runtime-final-closeout-v1/`.
Final results/source snapshot is committed at **`230185c`**; subsequent closeout
metadata is preserved in an incremental source bundle. No runtime change or
additional experiment accompanies this verification.
