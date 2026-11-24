# MXFP4 serving: compression and speculation on the same checkpoint

**Complete and independently verified, 20 October 2026, 05:39 UTC.** All six
loads and 48 timed requests complete. All 80 payloads verify locally; Spark/Mac
recomputation agrees on 1,863 numerical fields with zero difference. Original
production was restored at 05:37:55 UTC with its exact identities and 500,000
context. The separately authorized [KDA rounding quality diagnostic](GLM53_KDA_MXFP8_QUALITY.md)
subsequently completed and restored the same original service at 05:51:18 UTC.

Compression saves **9.36 GiB across TP2 (about 4.68 GiB/GPU)** in this setup.
The observed end-to-end gains are small; the code case fails its stock-repeat
control and remains descriptive. All 24 paired token sequences match, as do
the measured speculation counters. This does not establish full-model KL or
a speed upgrade over the differently configured production model.

| C1 case | CT off tok/s | CT on tok/s | Change | Mean TTFT change | Decode time change | Repeat controls |
|---|---:|---:|---:|---:|---:|---|
| Diagnostic, 64 output | 46.201 | 46.566 | +0.79% | +35.1 ms | -4.95% | Pass |
| Diagnostic, 256 output | 64.439 | 65.657 | +1.89% | +25.5 ms | -3.42% | Pass |
| Code, 256 output | 38.155 | 39.086 | +2.44% | +17.6 ms | -3.78% | **Fail: stock 2.85% spread** |
| Explanation, 256 output | 23.304 | 23.950 | +2.77% | +33.4 ms | -3.12% | Pass |

Six trials per arm/case across three alternating loads. TPS is the median of
all six complete-request rates. Decode time is normalized TPOT over speculative
bursts. A passing 2% load-repeat control does not establish statistical certainty
for a sub-2% throughput difference. The code stock control is retained unchanged.

The four final-pair code profiles each retain eleven eight-row C1 target graph
replays and one eager prefill. **Zero decode weight rebuilds** occur in all four.
Compressed prefill reconstructs 68 KDA matrices/rank in **36.47/37.14 ms** of
reconstruction-kernel sums. Profiled target graph medians do not improve in this
one trace; compressed head and drafter scopes are shorter. Profiled durations
cannot replace the unprofiled API measurements above.

Measured acceptance windows are identical off/on: 100% on both diagnostic
lengths, **50.77%** on code and **22.80%** on explanation. These are retained
counter-window ratios; snapshots may lag individual requests. Diagnostic text
substantially overstates acceptance on these varied-content examples.

The user clarified that the target is a working MXFP4 setup showing the measured
effect of compression. This test returns to that branch. The checkpoint is the
existing AutoRound CT MXFP4 model with the MxWave lossless storage/runtime
adapter; this work has not generated new MxWave-quantized GLM weights.

## Comparison

Six model loads alternated compression off/on: A1/B1/A2/B2/A3/B3. Each load
performs two trials per case after warmup, giving six trials per arm per case.
Both arms use the exact same checkpoint, drafter, graph sizes, admission policy,
8,192 context, 4 GiB KV/rank, 16 sequence slots and 2,048-token prefill budget.
This bounded test covers C1; it does not qualify concurrent capacity or 500k
context for the MXFP4 candidate.

| Case | Prompt tokens | Output tokens | Purpose |
|---|---:|---:|---|
| Retained diagnostic | 256 | 64 | Reproduce the earlier short-answer result |
| Retained diagnostic | 256 | 256 | Measure prefill amortization |
| Code review | 171 | 256 | Varied content; inspect drafter acceptance |
| Technical explanation | 107 | 256 | Varied content; inspect drafter acceptance |

Code/chat prompts are tokenized with the CT checkpoint's own tokenizer and
chat template, with thinking disabled. Fixed output lengths use the existing
greedy completion protocol. These are workload proxies, not the user's full
harness, and fixed-length generation can continue past natural completion.

Every request retains raw token IDs and stream-arrival times, plus before/after
speculation and scheduler metrics. Both-rank clock, temperature, power and
memory samples run during measurements. Timing precedes separate profiles in
the final A/B pair. Report TTFT, normalized decode time, full request throughput,
allocated memory and accepted tokens per draft call separately. Preserve all
failures and all cases. Same-arm load-repeat controls retain the 2% threshold;
generated-token agreement is checked before interpreting paired speed.
Finite greedy agreement does not establish full-stack KL qualification.

## Does MXFP4 need a different speculator?

Both production and CT use the same local DFlash2 FP8-block drafter. Its
45 target layers, hidden width 4,096, vocabulary 154,880 and selected target
layers match the CT model. Existing startup evidence verifies that the drafter
shares the actual target head and embedding, including the compressed
representations in the compression-on arm. The existing FP8 draft context-KV
adapter remains enabled in both arms.

The drafter consumes target hidden features. Different target quantization can
change those features and acceptance, even when the interface is compatible.
Structural compatibility does not prove optimal acceptance or speed. Changing
lossless storage on the same checkpoint does not itself call for a new drafter;
measure any effect of fused arithmetic and accepted verify paths.

In the earlier three-arm `dflash-c1-v1` test, the metrics window covering sanity,
warmup and timing records **899 accepted / 910 drafted tokens and 130 draft
calls in every arm**: ModelOpt, CT off and CT on. Thus acceptance did not explain
that measured regression. Those repeated diagnostic prompts are favorable to
speculation. This follow-up adds per-request counters for varied content before
deciding whether draft-length tuning or a different drafter is justified.

## Evidence and continuation

- [Independently recomputed results](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/mxfp4-serving-v3-review.json)
- [Archive and numerical verification](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/mxfp4-serving-v3-independent-verification.json)
- [Profile summary](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/mxfp4-serving-v3-profile-summary.json)
- [Original-production restoration](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/mxfp4-serving-v3-restoration.json)

- [Frozen protocol](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/mxfp4-serving-v3-plan.json)
- [Drafter/source compatibility record](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/mxfp4-serving-v3-drafter-compatibility.json)
- [CPU preflight and restoration identities](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/mxfp4-serving-v3-preflight-review.json)
- [Controller/client source bundle](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/mxfp4-serving-v3-source.tar.gz)
- [Source hashes](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/mxfp4-serving-v3-source-manifest.json)

Local staging: `/private/tmp/mxwave-mxfp4-serving-v3`.
Remote directory:
`/home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/full-model-v1/mxfp4-serving-v3`.
Read `controller.json` and per-block `*-measurement.json` for progress.
Restoration mock checks cover both-rank restart order, stopping only owned test
containers, idempotence after completion and rejection of changed originals.
Ruff and Python parsing pass for the staged bundle. Full project tests were
not rerun for this benchmark-only staging and documentation update.
