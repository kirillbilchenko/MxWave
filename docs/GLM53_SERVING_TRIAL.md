# GLM serving replacement: verified DFlash comparison

**22 October 2026, 10:00:06 UTC: the user-selected CT profile is live at 500k.**
`http://127.0.0.1:8000/v1`, alias `glm-5.3-flash`, now uses the final comparison's
CT/Inco profile on production's custom runtime. Authenticated generation and both
startup reports pass. Actual allocation retains the measured **14.465825 GiB TP2**
advantage, with unchanged 6 GiB KV/rank. Original containers are stopped and intact;
the detached handover is complete. [Current deployment and rollback](GLM53_CT_LIVE.md).
The older live-trial descriptions and rollback commands below are historical.

**21 October 2026, 12:49 UTC: original 500k production is healthy.** The user ends drafter training.
CT's next-launch profile selects unmodified Modal FP8 with C1:5 / C2:5 / C3–C16:3.
This is not a change to the live ModelOpt/Inco service. Use the selected
[speculative configuration](../integrations/glm53_mxfp4/ct_dflash_speculative_config.json)
with the pinned pretrained Modal checkpoint recorded in `full-model-v1/ct-serving-current.json`
on both Sparks. [Current evidence and remaining replacement gates](GLM53_HANDOFF.md).


**20 October 2026, 05:39 UTC:** the [matched MXFP4 comparison](GLM53_MXFP4_SERVING_V3.md)
is complete and independently verified. Original production was restored at
05:37:55 UTC. The separately authorized
[KDA weight-rounding quality diagnostic](GLM53_KDA_MXFP8_QUALITY.md) subsequently
completed and restored the same original 500,000-context service at 05:51:18 UTC. No MXFP8 variant
has been promoted to the production listener.

**Current service: original production restored, 20 October 2026, 04:45:58 UTC.**
The trial candidate is no longer serving. Both original containers are running
with identities/configuration hashes matching the saved pretrial record; both
candidate containers are stopped. Authenticated `http://127.0.0.1:8000/v1/models`
returns 200 for `glm-5.3-flash` with **500,000 context**. Existing original
production settings are restored, including DFlash, prefix caching, 6 GiB KV
per rank, 32 sequence slots and the 6,919-token prefill budget.
The [restoration receipt](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/production-restore-20261020-status.json)
and [identity review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/production-restore-20261020-local-review.json)
record this operational check; no new performance benchmark was run.
The dated trial measurements and rollback instructions below are retained as
history. Their statements that the 8,192-context candidate is live are superseded.

**C1 follow-up complete, 19 October 2026, 17:27 UTC.** A three-arm DFlash
comparison isolates compression: CT off/on is essentially tied end-to-end at
64 output tokens, with faster decode and 32.7–52.3 ms additional TTFT.
Most short-C1 replacement regression exists before compression. Traces find zero
C1 decode rebuilds; prefill still reconstructs 68 matrices/rank. Leaving KDA
dense while retaining expert/vocabulary compression is an untested option that
could retain about 6.85–7.23 GiB TP2 saving and remove those rebuilds.
The exact candidate is restored healthy, still with 8,192 context. Live vLLM
reports `0.1.dev20051+g487ecf187`. See [full evidence and limits](GLM53_DFLASH_C1.md).

Updated **19 October 2026, 15:21 UTC**. **Complete; candidate is live for the user's harness.**
The practical benefit is lower memory use. Under matched settings, short C1
throughput regresses 5.46%, short C8/C16 improve 1.78%/0.84%, and long prompts
improve 4.39–8.45%. The unchanged tuned production service remains faster in
all six measured cases. This trial does not demonstrate a production speed upgrade.

The existing authenticated Spark listener remains `http://127.0.0.1:8000/v1`,
with model alias `glm-5.3-flash` and DFlash enabled. Existing client forwarding
can continue using this listener. **Trial context limit is 8,192 tokens**;
maximum configured sequence slots are 16. Original production's 500,000-token
configuration and additional serving optimizations have not been ported or
qualified in this candidate. Original containers remain unchanged and stopped
for rollback. Controller and watchdog have finished the handover.

## End-to-end throughput

| Prompt tokens / client concurrency | Production tok/s | Matched current tok/s | Candidate tok/s | Candidate vs matched |
|---|---:|---:|---:|---:|
| 256 / C1 | 51.770 | 48.399 | 45.754 | -5.46% |
| 256 / C8 | 101.841 | 74.383 | 75.704 | +1.78% |
| 256 / C16 | 107.402 | 74.885 | 75.512 | +0.84% |
| 2,048 / C1 | 30.459 | 27.683 | 28.898 | +4.39% |
| 2,048 / C8 | 38.926 | 32.151 | 34.643 | +7.75% |
| 2,048 / C16 | 38.938 | 32.440 | 35.180 | +8.45% |

Three measured trials per cell/arm, 64 generated tokens per request. Throughput
is total output tokens divided by the client concurrency-group wall time,
including prefill and waiting; table reports the median of three trials.
Production is the unchanged service. Matched current and candidate share the
common settings below. Source checkpoint formats and required adapters differ,
so this is a replacement comparison, not an isolated compression experiment.

## First-token latency

| Prompt / client concurrency | Production mean (s) | Matched current mean (s) | Candidate mean (s) | Matched change (ms) | Matched p95: current → candidate (s) |
|---|---:|---:|---:|---:|---:|
| 256 / C1 | 0.464 | 0.483 | 0.505 | +21.1 | 0.504 → 0.533 |
| 256 / C8 | 1.907 | 2.662 | 2.543 | -118.9 | 4.469 → 4.270 |
| 256 / C16 | 4.232 | 6.095 | 5.935 | -159.9 | 11.369 → 11.110 |
| 2,048 / C1 | 1.571 | 1.589 | 1.429 | -160.3 | 1.611 → 1.448 |
| 2,048 / C8 | 7.384 | 8.550 | 7.778 | -772.2 | 13.858 → 12.556 |
| 2,048 / C16 | 14.038 | 16.469 | 14.961 | -1507.7 | 29.455 → 26.833 |

Means pool measured requests. Full trial ranges, mean normalized TPOT, request
latency and p95 stream-chunk gaps are in the
[verified metrics](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/serving-trial-v2-mac-independent-review.json).
DFlash returns token bursts; stream-chunk gaps are not single-token ITL.
Both common arms' logs show sampled intervals with four running and twelve
waiting requests at about 91% KV usage. Client C8/C16 includes substantial
queueing and does not establish performance of GPU batches of eight or sixteen.

## Memory, outputs and speculation

Post-startup Torch CUDA allocated memory across TP2 is **181.688551 GiB
current → 168.998459 GiB candidate**, a **12.690092 GiB reduction**
(6.346258 / 6.343833 GiB on ranks 0/1), with the same 4 GiB KV budget per rank.
This includes checkpoint-format differences. It is distinct from the earlier
9.367162 GiB isolated compression saving against the same CT checkpoint.
These are post-startup allocated bytes, not post-request allocation or a sum
of GPU and host counters. Reserved/peak and boundary cgroup/clock/thermal
records remain in the verified reports.

All **150/150 timed output sequences** match the candidate against both the
matched current and production references. Both chat checks pass in every arm;
the final authenticated streaming check passes. **Full DFlash-stack KL,
perplexity, broad task quality and long-context qualification were not measured.**
Exact finite greedy outputs do not establish universal numerical equivalence.

Speculative acceptance across chat sanity, warmups and timing is
**99.055% production,
99.670% matched current,
99.783% candidate**.
Accepted tokens per draft call are respectively
6.934,
3.080, and
3.083.
The inherited prompts are repeated diagnostic prose. Repetition favors
speculative acceptance; these real API times do not predict the user's
chat/coding harness throughput. Three trials and one load per common arm are
descriptive measurements, with no new statistical speed qualification.

## Common configuration and checks

Both arms use DFlash 7/5/3 by batch, 8,192 context, 16 sequence slots,
2,048 prefill budget, 4 GiB FP8 KV per rank, prefix caching off, 10 ms idle
admission, Marlin and nominal graph sizes [1,2,4,8,16]. Actual retained target
graph token counts are [4,6,8,12,16] in both arms. Both ranks prove shared
DFlash/target embeddings and output heads; the candidate shares its packed
vocabulary modules without retaining dense vocabulary weights. All 42 expert
layers, 68 KDA projections and two vocabulary tables pass startup checks.

The current checkpoint is the existing ModelOpt mixed NVFP4 `lossless8`
derivative. The candidate is the qualified compressed CT MXFP4 runtime over
`INCModel3/GLM-5.3-Flash-MXFP4-Mixed-CT-AutoRound`. Tokenizer and chat-template
bytes match. CLI commands match; checkpoint loader and compression opt-ins
are recorded separately. Production instead uses 500,000 context, 32 sequence
slots, 6,919 prefill budget, 6 GiB KV per rank, prefix caching and additional
custom optimizations. The production/common difference is a configuration
bundle; this run does not isolate the cost of each setting.

Each cell receives one warmup, then three timed trials; order reverses for the
middle trial. No recognized JIT-monitor warning overlaps the timed windows
(current: 0; candidate: 0). This log check is not a full device trace.
All 211 frozen runtime bindings match on both ranks before and after the runs.
No kernel arithmetic was changed for this serving integration.

## Retention, failed attempt and rollback

Spark and Mac independently verify **37 archive payloads and 11,477 numerical
fields with zero difference**. The complete raw measurements, source scripts,
startup proofs, logs and controller state are retained in the
[evidence archive](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/serving-trial-v2-evidence-v2.tar.gz) (705,780 bytes),
SHA-256 `265f56f6b7c991f14da84a9414de9cf0331a3a3bf7b558ce678d09ed661310ac`.
See the [archive verification](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/serving-trial-v2-mac-archive-review.json),
[output/allocation/log review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/serving-trial-v2-supplemental-mac-review.json),
and [live endpoint check](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/serving-trial-v2-final-service-check.json).
Authenticated models returns 200; unauthenticated models returns 401.
No API credentials are included in retained artifacts.

Attempt v1 completed the production reference but failed before current-arm
timing because the common environment omitted the existing production
`GLM_LV_DRAFT_FP8_KV=1` adapter. It restored exact originals with HTTP 200.
The [failed-attempt archive](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/serving-trial-v1-failed-evidence-v1.tar.gz)
retains 23 verified payloads, SHA-256
`77bc21d64b0ca796f17e295bd14ea9ad2ca1c48557ea768090df18521116fab8`.
V2 restores that adapter identically in both arms, with a source hash guard and
CPU FP8-buffer contract check. Its production reference is reused unchanged
with protocol and source-hash checks; failed timings are not substituted.

Local working evidence: `/private/tmp/mxwave-serving-trial-v2`.
Remote directory:
`/home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/full-model-v1/serving-trial-v2`.
Candidate containers: `mxwave-serving-trial-v2-candidate-r0/r1`.
The original containers retain their exact IDs/configuration. Trial containers
use `restart=no`; no service unit or original configuration was edited.
To restore the original setup:

```sh
ssh spark python3 -S -B /home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/full-model-v1/serving-trial-v2/control_trial.py --rollback
```

The rollback helper stops the trial containers and restores the originals;
its terminal phase is named `failed_restored` even for this explicit rollback.
