# Original-checkpoint NVFP4 serving adapter test

**Reference-repeat follow-up complete, 19 October 2026, 20:47 UTC.** The unchanged production
ModelOpt stack first differs at layer 3's MoE expert output with identical
inputs, actual gate logits and shared-expert output: 4/1,048,576 BF16 values
change on each rank, with final raw-logit KL 0.00537795. Raw logits agree with
API distributions. A different pair first changes at layer 4; the internal
routing/GEMM/reduction cause and fix remain unresolved. Full-model quality
stays unqualified at 1e-5; no speed or compression gate is relaxed. All 4,290
payloads verify independently. Exact prior CT/DFlash service is restored healthy
with 8,192 context; all diagnostic containers are stopped. See the
[complete diagnostic and next test](GLM53_NVFP4_REPEAT.md).

<!-- NVFP4_MODEL_RESULT_BEGIN -->
**Complete, 19 October 2026, 20:04 UTC. Memory saving is verified; serving speed and full-model
zero-loss quality remain unqualified.** The prior CT/DFlash service has been
restored healthy, with the same container identities and 8,192-token limit.
The experimental NVFP4 adapter is not deployed.

## Observed serving results

The checkpoint and original production settings match in all three arms:
ModelOpt NVFP4 experts, MXFP8 KDA, DFlash, prefix caching, 6 GiB KV/rank,
32 sequence slots, 6,919-token budget and 500,000 configured context. Each cell
has five repetitions of 64 output tokens; concurrency is client concurrency
including queueing. Every arm logs at most six running and ten waiting requests.
No memory saving was reassigned to KV in this test.

| Workload | Stock A tok/s | Compressed tok/s | Stock B tok/s | Stock time drift (limit 2%) |
|---|---:|---:|---:|---:|
| 256-token C1 | 27.63 | 38.77 | 35.92 | 23.06% — fail |
| 256-token C8 | 68.40 | 73.62 | 69.97 | 2.24% — fail |
| 256-token C16 | 61.26 | 73.93 | 67.81 | 9.66% — fail |
| 2048-token C1 | 20.89 | 25.05 | 23.35 | 10.50% — fail |

All four prospective stock-repeat controls fail. Compressed is observed 5–9%
faster than stock B, but neither that comparison nor pooling the two drifting
stock runs establishes a repeatable speed gain. Stock A's short-C1 trials range
from 25.52 to 34.47 tok/s, so an early 40% gain claim would be misleading.

Mean first-token latency on the same timed requests:

| Workload | Stock A TTFT, ms | Compressed TTFT, ms | Stock B TTFT, ms |
|---|---:|---:|---:|
| 256-token C1 | 843.0 | 458.1 | 480.5 |
| 256-token C8 | 2489.5 | 2081.1 | 2248.1 |
| 256-token C16 | 6688.4 | 5195.6 | 5631.9 |
| 2048-token C1 | 2047.0 | 1421.9 | 1597.6 |

Separate cold-C1 means, with prefix cache reset, are **631.4 / 481.8 / 480.7 ms**
for A / compressed / B. The first cold requests can include first-use Triton
compilation. Compressed and B are essentially tied in this small cold sample;
there is no qualified TTFT improvement. All timed requests have **zero cached
prompt tokens**, despite prefix caching being enabled. Across the broader client
interval, draft acceptance is 32.78% / 32.48% / 32.98%, not the ~99% of the earlier
repetitive workload. These source-context prompts are not the user's harness.

## Memory and traces

Both stock loads allocate **96.8381 GiB/rank** after graph capture. Compressed
allocates **93.6743 / 93.6716 GiB**, saving **3.1639 / 3.1665 GiB** (6.3304 GiB
across TP2). The encoded buffers alone save 3.2501 / 3.2527 GiB; the measured
allocation includes 72 MiB/rank of runtime scratch and allocator overhead.
All 84 source-scale hashes/rank match, all GPU byte-recovery checks pass, dense
scale owners are released, and the 20 captured token sizes match both controls.
This saving is separate from the earlier CT/MXFP4 saving; do not add them.

Six retained rank traces cover one 32-token C1 request per arm. Every trace has
19 graph launches, 19,639 kernels and 840 expert GEMMs. Generated token sequences
and normalized kernel inventories match, allowing for the compressed ABI.
There are **zero standalone scale-reconstruction kernels** during the request.
Expert-GEMM GPU time totals in A / compressed / B are **591.15 / 584.27 / 608.02 ms**
on rank 0 and **607.84 / 572.67 / 596.04 ms** on rank 1. These are single profiled
requests, not a repeated kernel speed gate, and kernel sums can include overlap.

Timed-window telemetry is retained for both ranks. SM clocks span 2,437–2,515 MHz,
with overlapping arm medians; temperatures span 59–77°C. No clocks were locked
or changed. These samples do not establish throttling as the cause of the
failed repeat controls; the cause of the full API drift remains unresolved.

## Numerical qualification

The gate remains mean prefill **and** mean decode KL below **1e-5**, with matched
conditioning and actual forward layouts. Full 154,880-token distributions are
retained for two C1 and C8 repeats per arm, four generated tokens per request.
All probabilities in the C1 self-repeat pairs are finite: their divergence is
not a masked-token support artifact.

| Self-repeat | Mean prefill KL | Mean decode KL | Layout matched | 1e-5 gate |
|---|---:|---:|---|---|
| C1 stock_a | 5.589e-03 | 5.274e-04 | yes | fail |
| C1 packed | 5.646e-03 | 2.397e-04 | yes | fail |
| C1 stock_b | 1.072e-02 | 8.791e-04 | yes | fail |
| C8 stock_a | 9.693e-03 | 2.325e-03 | no | fail |
| C8 packed | 1.034e-02 | 1.849e-03 | no | fail |
| C8 stock_b | 8.576e-03 | 3.176e-03 * | no | fail |

* Stock-B C8 changes a generated conditioning token; its displayed decode mean
uses only still-matched positions and cannot qualify the comparison. All C8
layout comparisons fail. C1 self-repeat layouts match within each arm, but stock A
uses verification widths 7/2 and compressed 4/4. Cross-arm KL is therefore also
confounded by the actual speculative execution shape. All full-model quality
gates remain failed/unqualified; this run proves neither zero added model loss
nor that compression caused the observed divergence. The next focused diagnostic
should first establish stock repeatability with fixed forward layouts and
conditioning, then compare the compressed consumer under the same policy.

PPL, broad task accuracy, SQNR, long-context quality and larger-KV capacity were
not measured. The two simple functional sanity prompts pass in every arm.

## Retention and restoration

All **154 payloads** in the immutable 199,388,803-byte archive verify locally.
Archive SHA-256: `53b9b6d1c20c5f3c8832dee2d504f0c21ef77984cb859555d439fcae396c5f25`.
The raw archive remains on Spark at
`full-model-v1/nvfp4-model-v1/evidence.tar.gz` and locally at
`/private/tmp/mxwave-nvfp4-model-v1/evidence.tar.gz`. Spark and Mac independently
recompute 1,623 numerical fields with maximum difference 2.4404e-15; all gates
and categorical fields agree. This is independent evidence recomputation, not a
second GPU experiment.

Repository records: [numerical/timing review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/nvfp4-model-v1-independent-review.json),
[traces and telemetry](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/nvfp4-model-v1-supplement-review.json),
[archive verification](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/nvfp4-model-v1-mac-archive-review.json),
and [source/retention bindings](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/nvfp4-model-v1-retention.json).

Controller `complete_restored`: all six test containers are stopped, both exact
prior CT candidate identities are running, health/authenticated models return
200 and unauthenticated models returns 401. The live service has **8,192 context**;
the test's 500,000 configured context was not promoted. Original production
containers are unchanged and stopped. Earlier failed component/preflight attempts
remain retained below and in the archive.
<!-- NVFP4_MODEL_RESULT_END -->

The user approved the next test after the [NVFP4 component result](GLM53_NVFP4_COMPONENT.md).
The purpose is to add lossless scale storage to the original tuned ModelOpt
checkpoint without giving up its MXFP8 KDA or specialized head/draft paths.
This is an extension of the technique; the existing MXFP4 implementation remains.

## Integration and prerequisite checks

`nv_adapter.py` packs all 84 post-Marlin scale families on each rank after stock
preparation, verifies GPU reconstruction, replaces all known quant-descriptor
owners and records weak ownership plus actual allocation. The existing production
MoE tuning wrapper chooses the same tiles in both arms. Routing, activation,
GEMM reduction, global scales, KDA, vocabulary and drafter weights are preserved.

The extended screen retains these attempts under Spark `full-model-v1/nvfp4-model-v1`:

- `smoke1`: exact eager checks through 256 rows and exact small-row graphs;
  the 256-row graph differs from eager. No full-model load started.
- `smoke2`: stock itself changes 16 of 1,048,576 values across repeated 256-row
  graph execution (maximum absolute difference 1.52587890625e-5). Both eager
  repeats are exact. This does not identify the source within stock routing/GEMM.
- `smoke3`: retaining one installed routing result for both consumers yields
  exact results through 1,024 rows, including 256-row graphs, then the 6,919-row
  compressed launch fails with `invalid argument`.
- The one-block-per-SM launcher requested the whole device limit as dynamic
  shared memory while also owning a static decoder palette. The fix subtracts
  actual static shared bytes from that budget and checks the remaining cache
  requirement, without changing tile selection or arithmetic.
- `smoke4`: the fixed standard-math library passes 18 cells (both TP partitions,
  rows 1/4/8/12/16/32/256/1024/6919), including 14 graph cases with 16 replays each.
  Routing is deliberately shared only in this isolated consumer screen.
- `smoke5`: supplemental coverage of 512/1280 rows tests the remaining routing
  block sizes; all four additional cells pass on both TP partitions.

New library SHA-256:
`5fe6c21821b8f683a4db18a80523231e2dff8c99d7ef15fd4f196fade48c17da`.
The earlier component speed numbers refer to the earlier binary; this full-model
run measures the new adapter/binary independently.

CPU prompt freezing initially received a structured tokenizer result instead of
a token list. The failed preflights and sources remain retained; normalization
of that return value is preparation plumbing and changes no model kernels.

## Frozen protocol used for this comparison

Three separate loads: stock A, packed, stock B. Both TP ranks use the original
checkpoint and runtime image, with identical original `GLM_*` flags and tuning.
Keep DFlash, prefix caching, the original scheduler, 6 GiB KV per rank, 32 sequence
slots, 6,919 batched-token budget and 500,000 configured context. A configured
context limit does not establish long-context quality or capacity.

Declared differences from production are the private API/master ports, diagnostic
worker/plugin and profiler, seed 42, and permission to request full-vocabulary
logprobs. The latter are requested only after timing; timed requests retain the
normal fast head path. No inference service is promoted automatically.

The protocol freezes 16 varied repository-context prompts. It measures five
repeated-prompt trials per short C1, long C1, short C8 and short C16 cell, with 64
forced output tokens; a separate three-trial short-C1 check resets the prefix
cache for cold TTFT. Actual cached-token counts must be checked: the initial
short requests report zero hits. The first cold requests also trigger first-use
Triton JIT, so cold numbers include that startup effect. These are diagnostic source-context workloads, not the
user's harness or a task-accuracy benchmark. Client concurrency includes actual
queueing; trace/metrics determine the resident batch size.

After timing, collect two C1 and C8 repeats with four generated tokens and full
154,880-token distributions. Compare identical prompt/conditioning tokens and
actual forward batch layouts; retain stock self-repeat noise and the unchanged
1e-5 KL limit for both mean prefill and mean decode KL. A failed repeat control cannot qualify a compression KL result.
The stock-A/stock-B median time control retains the 2% limit; unstable cells are
reported as unqualified rather than improving the threshold after measurement.

The controller pauses only an idle live service, uses an independent restoration
watchdog and an explicit outage deadline, and restores the exact previously live
CT candidate identities on every exit. It leaves the original production
containers untouched and stopped. Its retained idempotent restoration command is:

```bash
ssh spark python3 -S -B /home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/full-model-v1/nvfp4-model-v1/control.py --restore
```

Local validation: ruff and strict mypy pass; 835 tests pass, one skipped, including
18 codec tests and three tests for release of shared/cloned scale owners.
