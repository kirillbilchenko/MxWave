# C1 runtime diagnosis, 22 October 2026

**Complete; exact original production restored at 06:21:40 UTC.** Both original
container/image/config/mount identities match, the API reports 500,000 context,
and all four profiling containers are stopped. No candidate was promoted. The
controller took **14.00 minutes** including loads and restoration. Both Sparks
remain powered on. No additional GPU experiment is queued.

The main observed difference is target computation, not communication alone.
This is a deployment diagnosis, not proof that compression itself causes the gap.

## Scope and controls

- A is a trace-enabled copy of unchanged production, including its original custom
  image, Inco seven-draft policy, certified target head, FP8 draft head and RoCE path.
- B is the final official-vLLM-0.32 CT/plugin recipe, including MXFP4 experts,
  MXFP8 KDA, packed expert scales/embeddings, dense head and corrected six-layer
  Modal FP8 drafter. Its C1 draft length is five. No vLLM source files were replaced.
- Both retain the final 500k serving settings. Private ports, native profiler
  configuration, trace bind mounts and container hostnames are the profiling changes.
- The same frozen short code/chat prompts, fixed seeds and 256 output tokens are
  used at C1. Each cell has one warmup, three ordinary controls, one separately
  labeled profiled request, then two ordinary controls. Cold prefix-cache salts
  are matched. All controls and output IDs, including variation, are retained.
- This is one load per arm, A then B, **not** the earlier three-pair speed screen.
  No GSM8K, training, new tuning or promotion is included. An independent watchdog
  and frozen identities protected exact original restoration.

## Ordinary controls

These exclude the profiled request. All 20 controls return 256 tokens without API
errors. Output IDs and native draft counts vary despite fixed seeds and greedy
sampling; no repeatability gate is cleared.

| Workload | Production median tok/s [range] | CT median tok/s [range] | Median TTFT A / B | Median request latency A / B |
|---|---:|---:|---:|---:|
| Code C1 | 47.78 [43.89–52.62] | 42.86 [39.20–44.07] | 443 / 460 ms | 5.36 / 5.97 s |
| Chat C1 | 39.21 [38.74–44.43] | 36.08 [33.61–38.09] | 438 / 433 ms | 6.53 / 7.09 s |

The median differences are approximately −10.3% code / −8.0% chat for this session.
They do not supersede the earlier final three-pair results (−21.7% / −21.5%);
there was no optimization between them. Different output trajectories and trial
variability prevent treating the smaller gap as an improvement.

Median native draft calls are code **72 / 64**, chat **98 / 81**, A / B.
CT uses fewer calls in these controls but still takes longer. Acceptance alone
therefore does not explain the deficit.

Both cache probes pass same-salt reuse and fresh-salt miss checks. All timing
requests have cold-cache salts. Active-request telemetry has 8–9 samples per
arm/rank: P0 throughout, no active slowdown or power-cap flags, temperatures
**58–65°C**, SM clocks **2,411–2,515 MHz**. B rank1's median clock is 2,483 MHz
versus A's 2,509 MHz; clocks were not locked. Sampling cannot rule out transient
events between observations. Raw XML, power readings and RSS are retained.

## Trace attribution

Rank0 medians below are milliseconds per captured target-to-next-target cycle.
Rank1 corroborates the direction. GPU interval unions account for overlap within
each category; category medians must not be added to predict wall time.

| Component | Code A | Code B | Chat A | Chat B |
|---|---:|---:|---:|---:|
| Whole target graph span | 61.12 | 73.58 | 53.13 | 69.47 |
| Target expert GEMMs | 31.13 | 37.93 | 22.94 | 33.68 |
| Other target compute | 25.33 | 31.02 | 24.65 | 31.21 |
| Target communication | 3.27 | 3.81 | 2.77 | 4.03 |
| Drafter compute (head placement differs) | 3.01 | 6.56 | 2.99 | 6.57 |
| Drafter communication | 0.64 | 0.93 | 0.46 | 0.90 |
| Certified target head / eager dense vocabulary GEMM | 1.63 | 2.64 | 1.63 | 2.64 |
| Production's separate FP8 draft head | 1.37 | inside draft graph | 1.34 | inside draft graph |
| Captured full cycle span | 71.16 | 86.03 | 63.33 | 81.29 |

Target spans on rank1 are code **61.39 → 73.50 ms**, chat **52.90 → 69.46 ms**.
The target gap is about **12–16 ms**, with both expert and other compute contributing.
Production's expert kernel is NVFP4 Marlin; CT uses the packed-scale MXFP4 bridge.
Outputs, routed experts, verification widths (8 versus 6), checkpoint formats,
runtime policies and kernel implementations all differ. These traces cannot
assign the expert gap specifically to scale compression or to runtime version.

The communication difference is roughly **0.5–1.3 ms** per target graph across
ranks/domains. That does not support the previous hypothesis that porting RoCE
alone recovers most of the deficit. The certified target head has about a 1 ms
observed component advantage, also too small to explain the whole gap.

Native CT captures draft sampling/head work inside its draft graph; the installed
`DFlashSpeculator._generate_draft` calls `sample_draft` after `_run_model`.
Production's FP8 draft head runs separately. Comparing only the drafter-compute
rows overstates the architectural difference. Both-rank raw traces retain the
BF16 GEMMs inside the native draft graph. No isolated draft-head speed claim is made.
The six native `kernel_unified_attention` calls total about **0.08 ms in one inspected
draft replay**; this offers little support for prioritizing a draft-attention port.

### Coverage limits

Every classified target graph contains 84 expert GEMMs. CPU generation annotation
names bind captured target graphs to invocation order. Native 0.32 emits fragmented
GPU annotations across graph nodes/streams; the analyzer joins their envelopes by
unique invocation name, while actual GPU time still comes only from kernel intervals.

Production captures a **contiguous prefix**: code 48 / 49 target graphs on ranks
0 / 1 versus 65 CPU generation annotations; chat 51 / 51 versus 97. The absent
tail is retained as an unresolved profiler-coverage limitation. CT captures
69 / 69 code and 81 / 81 chat target graphs, matching its CPU annotation counts.
Thus these medians are not equally complete whole-request samples or matched
token trajectories. They locate promising categories, not causal speedups.

Central summaries omit the first two and last cycle and any nonconsecutive native
invocation pair. Counts are code A **44 / 45**, B **65 / 65**; chat A **47 / 47**,
B **77 / 77**. All omitted/raw cycles and invocation ordinals remain in the JSON.
Time without a traced kernel is not labeled CPU overhead: waits, copies, profiler
overhead and untraced work can contribute. No dropped-event cause is established.

## Assessment of the proposed same-runtime comparison

Running CT and ModelOpt on the same production image, with the same Inco drafter
and all compatible optimizations, would be a stronger **deployment diagnostic**.
It would still compare different quantized checkpoints, not isolate lossless packing.
Hook activation must be verified; merely sharing an image does not prove that
checkpoint-specific head, KDA, expert or communication routes are identical.

The older [20 October experiment](GLM53_KDA_MXFP8_SPEED.md) records **6 GiB KV,
16 slots, 2,048 prefill budget, 8,192 context, no prefix cache, 10 ms admission,
graphs [1,2,4,8,16], and drafts 7/5/3**. The newer proposed summary's “4 GiB and
seven drafts everywhere” is not its frozen configuration. C1 code was −8.61%;
the C8/C16 and explanation repeat failures remain. Those results motivate a
comparison but do not prove that runtime alone explains the newer C1 gap.

If another experiment is chosen, first audit full-production CT hook compatibility
on CPU, then freeze a small same-runtime C1 comparison. Do not automatically add
GSM8K, C8 or another tuning campaign to this profiling session. A custom-image win
would justify investigating a specific portable optimization; it would not itself
fulfill the user's official-vLLM-plus-plugin recipe requirement. No such follow-up
has been launched or silently substituted for the stock recipe.

## Evidence and verification

Local root: `out/glm53-package-v1/c1-runtime-profile-v1/`.
Remote root on both nodes:
`/home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/full-model-v1/glm53-package-v1/c1-runtime-profile-v1`.

- `trace-review.json`: all eight worker trace hashes, graph inventories, raw cycles,
  coverage, interval-union summaries and attribution limits.
- `control-review.json` and `review_controls.py`: recomputed ordinary controls,
  retained profiled-request counters, acceptance per position and sampled telemetry.
- `rank0/`: final controls, both-arm telemetry, immutable preparation, logs,
  controller state and independent `final-verification.json`.
- `rank0/traces-{A,B}` and `rank1/traces-{A,B}`: native trace payloads.
- Reproducer: `analyze_c1_runtime.py` (not included in this branch).
  Eight focused tests pass, including fragmented annotations, missing invocation
  exclusion, incomplete graph rejection, head shape binding and overlap accounting.
  Repository Ruff and strict core mypy pass. The full suite passed before launch
  with one skip; no serving code changed after that run.

The source preparation is committed as `65474a9`; its incremental bundle verifies
on both nodes and Mac. Final archive and analysis/source receipts are recorded in
the handoff after backup verification. No new memory-saving or quality claim is made.
