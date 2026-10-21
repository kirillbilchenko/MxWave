# Prospective stock-vLLM deployment comparison

Prepared 21 October 2026. **GPU comparison not yet started.** The user now requests
concrete outcomes and full GSM8K without further tuning beforehand. Collection is
diagnostic while numerical/output-identity gates remain failed; it cannot qualify
or promote the candidate. The user's latest requirement makes upstream
vLLM the candidate's default runtime. Consequently this is a comparison of two
complete deployments, not a compression-only experiment or an identical-runtime
comparison. Earlier gains must not be added to its results.

**Startup evidence, 21 October:** V6/V7 reach API readiness with the requested
capacity and native Triton draft attention. V7 observes real prefix reuse, but
cold/warm and warm/warm greedy output identity both fail. V6 also differs without
a reusable prefix. This fails the frozen correctness gate; timing, long-context
probes and GSM8K have not started. Retain the matched fixed-input replays;
do not reinterpret prefix hits as a numerical correctness pass.
Exact original production was verified restored at 16:44:58 UTC. The candidate's
87.157 GiB/rank startup allocation has no fresh matched production counterpart;
6.941 GiB of TP2 packed-tensor savings is a separate byte-accounting observation.

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
| Draft KV | Actual speculative config is `auto`; the old FP8 helper converts projection weights | Packaged upstream `fp8_e4m3`; isolated BF16-cache V3 also accepted zero proposals |
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
   Output identity already fails and remains a failed qualification gate. Requested
   diagnostic measurements must carry that failure. Actual prefix reuse and cold-state
   control must work; do not disable caching to obtain timings.
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
- Warm up each cell. Production exposes no cache-reset API, so use native per-request
  cache salts: identical between arms, distinct for each trial/cell/request and for
  warm-ups. Verify request-protocol and KV-hash support on CPU, then prove same-salt
  reuse and fresh-salt misses on GPU before timing. Prefix caching stays enabled.
  If this control fails, retain the timing blocker and continue with independent
  speculation-off GSM8K. Retain hit counters for every measured cohort.
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

## Prepared operational sequence

Remote root: `glm53-package-v1/deployment-compare-v1`. The implementation lives in
`deployment/glm53/compare_client.py`, `compare_control.py`, `prepare_compare_run.py`,
`prepare_quality_clone.py` and `compare_preflight.py`. CPU preparation validates
native CT CLI commands and creates stopped production quality clones.

The controller freezes 12 timing blocks followed by two full quality arms. A timing
failure skips remaining timing blocks, retains partial data and still attempts both
quality arms. It changes no quality gate. Its independent watchdog bounds the full
sequence at **10 hours** and restores the exact original containers on completion,
failure or laptop disconnection. This is a restoration limit, not a predicted runtime.

GSM8K uses distinct per-question salts, identical in both arms, the unchanged
existing grader, natural stops, no retries and all errors/truncations in the denominator.
Tokenizer JSON, tokenizer configuration and chat-template bytes match between arms.
Long timing inputs include disclosed combinations of distinct real training prompts
where natural inputs are too short; no repetitive diagnostics or sealed tests are used.
