# Full GSM8K comparison

**Complete and independently verified, 17 October 2026.** Both original GLM
containers/configurations are restored and healthy at **16:57:33 UTC**.

| Full 1,319-question workload | Reference | Compressed candidate |
|---|---:|---:|
| Numeric answer exact match | 1,246 / 1,319 (94.47%) | 1,244 / 1,319 (94.31%) |
| Original strict answer format | 1,228 / 1,319 (93.10%) | 1,230 / 1,319 (93.25%) |
| Aggregate output tokens/s | 41.770 | 42.080 |
| Questions/minute | 35.002 | 35.058 |
| Benchmark wall time | 37 min 40.99 s | 37 min 37.41 s |
| Generated completion tokens | 94,441 | 94,993 |
| CUDA allocation, both ranks | 175.866 GiB | 166.499 GiB |
| API errors / truncated answers | 0 / 0 | 0 / 0 |

The measured allocation saving is **9.367 GiB / 5.33%**, including fixed
KV and runtime buffers. Numeric accuracy changes by **-0.1516 percentage
points**: 22 reference passes are lost and 20 reference failures become
passes. The paired exact McNemar p-value is 0.8776; it does not establish
equivalence. Strict-format totals improve by two. Output throughput changes
by about **+0.74%**, while the complete question workload finishes only
**3.57 seconds sooner (+0.16% questions/minute)**. The candidate generates
more tokens. This single sequential pair supports roughly equal workload
speed, not a demonstrated kernel speedup or a general quality-parity claim.
Both-rank loaded-weight and before/after source checks pass in each arm.

Launched detached on Spark at 15:24:04 UTC.
Controller PID 4103342; independent evidence collector PID 4103341. The
reference ran first, then the compressed candidate. Both original GLM
containers were restored once after this two-arm batch, with identity and
configuration checks plus HTTP health. A separate restoration watchdog armed
before the authorized idle pause.

## Verified evidence

All **2,638 raw answers** were independently regraded against the pinned
source questions, answers, prompts and generation settings. All **672 source
scale groups**, both loaded-weight checks per arm, protocol hashes, both
139-file source inventories, native-library hashes and exact original
container identities verify. All **2,960 archive payloads** verify on Spark
and locally. No experiment containers remain active from this batch.

The [compact review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/gsm8k-full-model-v1-review.json)
retains every question whose score changes, all speed totals, allocation and
peak/reserved counters, source bindings and archive digests. Strict-format
passes change in both directions: **29 lost / 31 gained**. Primary numeric
passes are **22 lost / 20 gained**; the net score alone does not describe
individual-answer preservation. Full assistant text agrees on 506/1,319
pairs, final text on 604, and extracted numeric values on 1,264. Ten pairs
have no extractable numeric answer in either arm. These are text comparisons,
not full-vocabulary logit or complete-inference bitwise checks.

Completion latency median is **6.531 / 6.505 seconds** (reference/candidate),
p95 **11.116 / 11.062 seconds**, p99 **14.639 / 14.540 seconds**. The maximum
is **23.299 / 31.729 seconds**; all cases and outliers remain. There is no
new TTFT, PPL, KL, source-BF16 SQNR or profiler measurement in this batch.
The earlier strict qualification failures are unchanged.

Archives under ROOT, retained on their source ranks and copied to the head:

| Rank | Payloads | Archive bytes | Archive SHA256 |
|---|---:|---:|---|
| 0 | 2,810 | 5,585,954 | `3ff4fd2f047a653d0650d892d7ab8b525b66d341223a0d6fdae74d857cb213d5` |
| 1 | 150 | 3,181,216 | `7b793191463ef8695f7e187bee588501306f590d210e17243fe2c972d331a90a` |

Local independent analysis is retained at
`/private/tmp/mxwave-gsm8k-full-v1/`. The primary verifier was frozen before
inference; supplementary workload/text/telemetry analysis is descriptive and
does not amend the declared scores. Use ROOT/full-gsm8kfullv1-src as the exact
tested snapshot. Its qualified moe_runtime.py hash is `960e5febe55af6b1e998b55e3665a59b0d0ac34b0d46a0fb5912e237f928cb62`;
another local worktree runtime revision is unqualified against this native
library and must not silently replace that snapshot.

## What is compared

- Reference: `INCModel3/GLM-5.3-Flash-MXFP4-Mixed-CT-AutoRound`, revision
  `35d1b0d5cb4a0db075fd2749c897e970dc15bfa5`, without MxWave runtime compression.
- Candidate: the same checkpoint with the qualified v9 scale palette / U8
  shared-expert staging, selected BF16 KDA projections, head and embeddings.
  Original FP4 weight codes are unchanged. Retain the original KDA operation;
  the rejected direct-wrapper addition is absent.
- Same pinned vLLM image, TP2, BF16 activations, 4 GiB FP8 KV per rank,
  canonical expert dispatch, graph captures [1,4], four concurrent chat
  requests, temperature 0, seed 42, low reasoning effort, 1,024-token cap.
  **No D2, DFlash, MTP or other speculative drafter in either arm.**

This is the complete pinned **GSM8K main/test set, 1,319 questions per arm**.
Every question is used once in original source order for submission. Requests
finish naturally; there is no forced token count, scheduler pause, retry,
favorable sample selection or repeated question round. Source questions,
answers, exact prompts and generation settings are frozen before either load.

The zero-shot chat prompt appends:

```text
Solve the problem. End your final answer with #### followed by the numeric answer.
```

The pinned native template renders the system directive `Reasoning Effort: Low`
and the generation prefix `<|assistant|><think>`. Reasoning tokens count toward
the output-token budget and measured completion throughput; scoring uses the
final-answer content.

This is a paired zero-shot evaluation, not a reproduction of a published
few-shot leaderboard setting. Natural variable-length groups can have
different batch admission and completion shapes between arms.

## Scores and limits

Report both of these prospectively declared metrics:

1. **Numeric answer exact match:** require a natural stop, extract the final
   number from the final-answer content (after `####` when present), and compare
   decimal values after comma normalization. Unfinished `<think>` text is not
   an answer. This flexible extraction can accept surrounding prose.
2. **Strict answer-format match:** retain the earlier bare-number / `####`-tail
   grader. Extra prose or Markdown may fail this metric even when the extracted
   numeric answer is correct. Never relabel the earlier strict failures.

All 1,319 questions remain in each denominator, including API failures and
truncated responses. Four consecutive request/response failures abort the
attempt without retries; wrong answers do not stop the benchmark. Save every
raw response separately with its request, prompt and payload hashes.

The final report includes both totals, descriptive Wilson 95% intervals,
question-level gains/losses, a paired exact McNemar statistic, truncations,
request errors and current CUDA allocations. Temporal nondeterminism is not
measured by Wilson intervals. Full-workload token rates include prefill,
variable completion lengths and client overhead; they are not the earlier
fixed-64-token TPS benchmark.

Each raw answer retains API elapsed time and token usage. Final analysis can
report total benchmark wall time, questions/minute, generated tokens/second,
and latency percentiles. This non-streaming client measures completion latency;
it does not provide a separate time-to-first-token measurement.

**This is an accuracy benchmark, not another strict KL qualification attempt.**
The preceding graph-four reference repeat reached KL 0.0000117910854 against
0.000010000 and stopped before candidate inference. All its generated ids
matched. That failure remains unchanged; it must not block collection of actual
GSM8K answer outcomes. Even a good GSM8K score cannot establish bitwise model
equivalence or resolve the reference-repeatability blocker. Both arms already
use imported MXFP4 weights, so this does not measure quantization loss versus BF16.

## Interpreting throughput and utilization

Allocated memory is capacity, not bytes transferred per generated token. The
adapters reduce expert-scale and selected BF16 storage while retaining the
main FP4 codes, the fixed KV representation and most projection work. Only
selected MoE experts execute for a token. Decompression consumes device work,
so a 5.33% allocation saving does not predict a 5.33% throughput improvement.

A brief read-only observation during this candidate run found CPU load
averages 3.86 / 2.71 on twenty cores, one-second CPU busy fractions 26.6% /
13.6%, and three GPU samples at 95–96% / 96% active time. No swap was used.
This does not establish CPU headroom on every critical thread or peak GPU
compute utilization. Active kernels can wait on memory or communication.
The reported memory-utilization counter was zero with memory clocks/sizes
unavailable; these observations cannot measure actual LPDDR GB/s or prove
memory-bandwidth saturation. There is no matched reference telemetry sample.

The preceding [controlled graph trace](GLM53_CONTROLLED_BATCH_EXPERIMENT.md)
measured about 79 ms per four-token model forward, before separately charged
head and request overhead. Its largest kernel groups were routed experts
(about 32 ms/rank), compressed KDA projections (22–24 ms) and other BF16
linears (about 9 ms). Categories can overlap and are not an additive latency
breakdown; the KDA number includes required projection math. Those three-step
traces are contextual evidence from an earlier fixed cohort, not a profile
of this natural-length GSM8K workload. Current token rates alone cannot
separate bandwidth, occupancy, compute and communication limits.

## Targeted repeatability follow-up

Finish and preserve this benchmark before changing inference. The preceding
graph-four reference-only check had byte-identical prefill distributions in
all four legs, but varying autoregressive decode distributions. Its largest
failed-leg step was context 4, decode position 1. That is a debugging lead,
not a replacement benchmark sample or evidence against this candidate.

A bounded follow-up should first reproduce that reference-only step, keeping
the checkpoint, native code, graphs [1,4] and actual request cohort fixed.
Record request row order and cache-block mappings. At the end of prefill,
compare KDA recurrent state and relevant KV pages as well as hidden outputs:
identical prefill logits do not establish identical persistent state. Then
capture inputs, state, router logits, selected experts and outputs around the
first differing decode operation. Replay that operation with identical
captured tensors to distinguish state/input drift from reduction-order noise.
Apply deterministic controls to the identified kernel; a global Torch flag
alone does not establish determinism in a custom native kernel.

Retain the original numerical floors and failures. Any later statistical
qualification contract needs its own frozen protocol and independent
reference-repeat calibration before candidate evaluation. A small practical
accuracy difference or a favorable rerun does not amend the old contract.

## Source and execution receipts

```text
ROOT=/home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/full-model-v1
Source: ROOT/full-gsm8kfullv1-src
Controller: ROOT/quality-model-controller-gsm8kfullv1.json
Results: ROOT/run-gsm8kfullv1/results
Progress: ROOT/run-gsm8kfullv1/results/{control,candidate}-gsm8k.json
Individual raw answers: ROOT/run-gsm8kfullv1/results/{control,candidate}/gsm8k/case-NNNN.json
Prepared: ROOT/gsm8k-full-v1-prepared.json
CPU proof: ROOT/gsm8k-full-v1-cpu-preflight.json
Launch: ROOT/gsm8k-full-v1-launch-receipt.json
Restoration: ROOT/restoration-gsm8kfullv1.json
Independent archive check: ROOT/gsm8k-full-v1-independent-evidence-check.json
```

```text
139 Python/TOML files
source inventory: fbc30821a34cd4270889bc04466869a5968b474f079535adec1c3c0caa073bdc
prepared: cd2cea32824b40445aac57c3155250a99656695c25c05b54ac074b9d8acf51ee
CPU proof: 45a5377575cc04ae6ec6e084c036ba43fd200138e45fecdd8942f171ca330b61
GSM8K protocol: 2b9eee3641a017860a549db1a554ab6d5f140a6766f7baa757a4f4f88962c25f
independent verifier: 6f781425f70740c4ba10585b838359a7d370263a2c199fabcb83919e8ccd2333
```

CPU preflight verifies the same qualified native library and dataset pins on
both nodes, both actual vLLM CLIs, message serialization and the installed chat
request schema. A complete synthetic 2,638-response fixture exercises file
hashes, extraction, scoring, paired analysis and retention of the prior failure.
Local lint, strict mypy and the unit suite pass, with one GPU-dependent skip.

Compatibility copies of the inherited quality-protocol files remain hash-bound
in the prepared results directory. The GSM8K client does not execute their KL,
WikiText or practical-task phases. Its active protocol is `gsm8k-protocol.json`.

The source starts from the bound graph-quality snapshot and overlays only the
benchmark helpers and controller selection. Qualified MoE/KDA codecs and native
libraries are unchanged. GPU work stays in the owned bounded containers; the
CPU HTTP client is bounded at four requests. Per-arm client deadline is three
hours; independent total outage watchdog is eight hours. Expected completion
is roughly two to three hours, subject to actual answer lengths.

Monitor from the laptop:

```bash
ssh spark 'python3 -S -m json.tool /home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/full-model-v1/quality-model-controller-gsm8kfullv1.json'
```

Both-rank archives, independent regrading and restoration checks are complete.
All previous strict qualification failures stay unchanged. The user authorized
the targeted reference decode/state follow-up after this completed benchmark;
it requires a new frozen revision and CPU preflight before GPU inference.
