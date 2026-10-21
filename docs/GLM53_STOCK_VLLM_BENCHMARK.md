# Prospective stock-vLLM deployment comparison

Prepared 21 October 2026. **Not run.** The upstream startup and prefix-cache
correctness checks must pass first. The user's latest requirement makes upstream
vLLM the candidate's default runtime. Consequently this is a comparison of two
complete deployments, not a compression-only experiment or an identical-runtime
comparison. Earlier gains must not be added to its results.

## Two arms

| Setting | A: unchanged production | B: upstream CT recipe |
| --- | --- | --- |
| Runtime | Original production image and overlays | Official vLLM 0.32.0 plus installed MxWave plugin |
| Target | ModelOpt lossless8 | Imported AutoRound CT MXFP4 experts, measured MXFP8 KDA, lossless expert-scale and embedding packing |
| Target head | Production dense head and certified argmax optimization | Dense BF16 head; certified optimization unavailable upstream |
| Drafter | Inco DFlash2 FP8 | Pretrained Modal DFlash2 FP8 |
| Draft policy | Live audit: 7 at C1, 5 at C2, 3 at C4/C8/C16, plus adaptive/truncation hooks | 5 at C1/C2; 3 at C3–C32; native DFlash policy |
| Context / slots | 500,000 / 32 | 500,000 / 32 |
| KV / prefill budget | 6 GiB per rank / 6,919 tokens | Same |
| Memory reservation / cache block | 0.85 / 2,304 | Same |
| Prefix cache / chunked prefill | Enabled / enabled | Same |
| Graph sizes | 1, 2, 4, 6, 8, 12, 16, 18, 24, 32 | Same requested sizes; actual captures/replays must be recorded |
| Target KV | FP8, native sparse-MLA representation | Upstream FP8, actual sparse-MLA representation recorded |
| Draft attention backend | Production custom path, retain effective backend in audit | Explicit native `TRITON_ATTN` under startup qualification; stock FlashInfer fails capacity through 64-token-page shared-pool waste |
| Draft KV | Production FP8 adapter | Explicit upstream `fp8_e4m3`; compatibility must pass |
| Draft head | Custom FP8 head | Native upstream implementation; custom optimization absent |
| Admission | Environment says 4 ms; scheduler-control override needs live audit; implementation caps at 5 ms | No custom admission-window implementation |
| Other custom kernels | Production retains its current model and RoCE overlays | None of those overlays |

There is no verified 10 ms production admission window. Do not change A to create
one. Inspect the effective scheduler controls immediately before the run and retain
the result. The stock recipe intentionally does not copy these production hooks.
Keep every additional discovered difference in the report, including tokenizer,
chat template, cache layout, graph padding and collective backend.

## Startup gates

1. Pin both images, wheel, model/drafter metadata, assets, launch arguments and
   source commit. Retain a sanitized production configuration snapshot and hashes
   binding the exact original containers, including private settings without
   exposing their contents.
2. Verify the requested 500k configuration starts, including the selected drafter,
   6 GiB cache and captured graphs. Do not reduce context, slots, KV, graph sizes
   or turn off prefix caching to make the benchmark start.
3. Exercise repeated real prompts; retain outputs and evidence of prefix reuse.
   If output identity or reuse fails, diagnose it before timing. Do not silently
   replace the check with a weaker criterion.
4. Record actual allocation per rank after initialization. B exposes Torch
   allocated/reserved bytes in its startup report. Establish a comparable source
   for A before claiming a memory delta. On Spark, host RSS or cgroup totals include
   unified memory and are not interchangeable with Torch allocation. If A cannot
   expose the same metric unchanged, report that limitation rather than subtracting
   unlike counters.

A configured limit is not proof that a request at that length fits. The 32k/128k
probes below establish only those tested lengths, not complete 500k qualification.

## Frozen timing protocol

- Realistic code, chat and explanation prompts; short around 256 input tokens and
  long around 2,048. Freeze prompt text, exact token counts/IDs per tokenizer and
  hashes before any measurement. Do not use repetitive diagnostic filler or the
  sealed drafter test set.
- C1/C4/C8/C16, 256 output tokens per request. Fixed seed 42 and identical request
  parameters. Retain outputs, finish reasons and actual token counts; generation
  length controls used for timing must be disclosed. Quality uses natural stops.
- Six interleaved **pairs**, A/B/A/B, totaling six trials per arm and cell. Each
  arm block covers the 24 domain/length/concurrency cells. Request seeds stay fixed;
  scheduler/grouping changes remain observable and are not hidden by seed claims.
- Warm up each cell. Use cold prefix-cache state for the timed cohort by resetting
  after warm-up, while leaving prefix caching enabled. Validate reset behavior on
  both engines and record before/after hit counters. If reset cannot be made
  equivalent, report a blocker; do not mix warm and cold measurements.
- Use the same fixed client machine and synchronized cohort launch. Poll running,
  waiting, KV and speculation counters throughout each cohort; retain raw timestamps.
  Poll GPU clocks, temperature, power and throttle/state reasons on both ranks.
- Keep all measurements and failures. Apply the existing repeat-control rules;
  label unstable cells unqualified instead of selecting convenient trials. Medians
  and full ranges include outliers. No historical numerical or speed gate changes.

Per cell report aggregate output tok/s (cohort tokens divided by cohort wall time),
median and range across trials; request TTFT mean and p95; full request latency;
peak/typical actual simultaneous running requests and queues; KV usage; proposed
and accepted drafts per position; verification rounds; memory per rank. Record raw
stream events so first-token timing excludes headers and empty role chunks.
Acceptance counters must reconcile; missing denominators are reported as missing.

After the timing cells, use frozen 32k and 128k contexts containing unique retrieval
facts at several positions. Report actual tokenizer length, retrieval answers,
TTFT, request latency, peak memory, cache usage and errors. A retrieval sanity check
is not a comprehensive long-context accuracy benchmark.

## Quality and restoration

After timing, run the complete pinned 1,319-question GSM8K test set on both targets
with speculation disabled. Use the same zero-shot prompt/extraction policy,
reasoning setting, seed and 1,024-token cap as the existing CT benchmark unless a
separately named protocol is frozen first. Retain every response, error and
truncation; report paired correct/wrong transitions and extraction limitations.
Production's historical GSM8K score is not known from the CT-only comparison.
Quality requires separate temporary launches; do not mutate original containers.

Every GPU stage runs under independent restoration controls. Restore the exact
original production containers and verify the authenticated 500k endpoint at the
end or on failure. No promotion. The code-only packaging work and CPU preparation
can continue while production is serving.

The suggested 2–3 hours for timing plus 1.5 hours for GSM8K is an estimate, not a
validated reservation. Timing alone contains 288 measured cohorts and 534,528
requested output tokens, before warm-ups, model reloads and long-context probes.
Estimate the actual outage from the first measured cells and observed upstream
load time before launching the full sequence; use an explicit watchdog bound.
