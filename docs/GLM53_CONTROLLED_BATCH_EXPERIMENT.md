# GLM controlled batching and batch-four graph diagnostic

**Complete, 17 October 2026. Keep the graph path as an experimental
configuration; reject the direct-wrapper addition.** GLM's exact original
containers are restored and healthy at **14:31:28 UTC**, independently
confirmed. Both archives and all **711 payloads** verify on Spark and locally.
Launched at 14:10:07 UTC with controller PID 3516968 and collector 3516969. Revision
cohortdispatchv2 adds measured request admission and batch-four graph selection
to one resident best-v9 compressed model. Existing memory/quality results and
all failed gates remain unchanged. Read the preceding
[eager dispatch result](GLM53_EAGER_DISPATCH_EXPERIMENT.md) first.

The preceding experiment submitted four prompts together, but only nine of
sixteen groups actually prefetched together. Strict concurrent comparisons
therefore mixed batch shapes. Both-rank profiles also differed in admission
pattern, and reduced wrapper CPU scope time did not materially change the
approximately 101 ms M4 GPU forward. These observations motivate controlling
admission and testing actual graph replay before changing another kernel.

## Completed decision and measurements

All three smoke groups, twenty-four quality groups and both profile groups
pass independently observed batch coverage: **29 controlled cohorts**, each
verified on both ranks. All 96 prefill distributions (six legs × sixteen
contexts) are retained. Both graph managers retain the same two graphs; no
mode switch replaces weights, pools or graphs. All quality token IDs match.

| Decode comparison | Mean KL | Mean NLL delta | Unchanged numerical limits |
|---|---:|---:|---|
| Original eager self-repeat | 7.769e-6 | +3.133e-5 | Pass |
| Graph self-repeat | 7.294e-6 | +3.887e-6 | Pass |
| Original eager versus graph | 9.255e-6 | +3.241e-5 | Pass |
| Direct-wrapper self-repeat | 1.463e-5 | -7.185e-6 | **Fail: KL** |
| Original eager versus direct wrapper | 1.085e-5 | +1.916e-5 | **Fail: KL** |

Limits remain mean decode KL ≤1e-5, absolute mean NLL delta ≤1e-4 and
identical conditioning/generated IDs, with 48 real-KV decode steps/comparison.
The combined six-leg numerical_gate_passed flag remains **false**. Selecting
the original-operation graph path does not amend the failed direct-wrapper
result or turn the whole diagnostic into an all-modes pass. The absolute
differences are small; crossing a KL threshold is not an equivalent percentage
loss in accuracy. No end-to-end bitwise inference claim is made.

The observed batch-control result is useful: original eager self-repeat is
7.77e-6 here, compared with 5.26e-5 in the prior uncontrolled run. Different
launch/graph allocations and mode controls also changed; these two experiments
do not isolate batching as the sole cause of the improvement.

| Real API C4 / 256-token prompt, 64 generated tokens | Eager original | Graph original | Change |
|---|---:|---:|---:|
| First block, median aggregate tok/s | 38.956 | 41.030 | +5.32% |
| Reverse-first block, median aggregate tok/s | 37.573 | 40.664 | +8.22% |
| First block, median TTFT | 1.134 s | 1.117 s | -1.49% |
| Reverse-first block, median TTFT | 1.093 s | 1.114 s | +1.92% |

All twenty groups and all five trials/mode/block remain. These are closed-group
aggregate API measurements, not per-request token speed. All twenty groups
produce the same four 64-token sequences. The speed comparison is graph
versus eager **on the same compressed v9 model**; it cannot establish a new
compression-versus-stock speedup or be added to the prior B1 speed improvement.

Allocated CUDA memory in this graph-ready engine is **166.4987 GiB across
TP2**, about **83.25 GiB/rank**, effectively the same rounded allocation as
the earlier compressed engine. Both timing modes retain identical captures.
The earlier matched compression comparison measured 175.8658 →166.4987 GiB,
**9.3672 GiB / 5.3263%** saving. A new raw-runtime graph-four comparison is
required before claiming an independently measured new compression delta.

### What the matched traces establish

Four worker traces cover eager/graph on both ranks with the same four prompts,
1032-token prefill, three M4 decodes and identical generated IDs. Each graph
trace contains **three actual cudaGraphLaunch calls**; eager traces contain
zero. Graph-manager descriptors and canonicalization-counter evidence agree.

| Profile median over three complete M4 GPU forwards | Eager | Graph |
|---|---:|---:|
| Rank 0 forward | 86.87 ms | 78.83 ms |
| Rank 1 forward | 86.83 ms | 78.71 ms |

Rank 0 uncovered time falls from roughly 12–16 ms/forward to 1.6–2.1 ms;
rank 1 falls from roughly 5–6 ms to 1.5–2.2 ms. This supports addressing launch
gaps through graph replay. Three profiled steps do not replace the unprofiled
twenty-group throughput comparison, and uncovered time alone does not isolate
CPU computation or a specific synchronization cause.

Remaining graph M4 kernel sums are roughly 32 ms/rank for routed experts,
22–24 ms for compressed BF16 KDA projections, and 9 ms for other BF16 dense
linears. Category sums may overlap streams and must not be added into a new
wall-time estimate. Canonical ordering is about 0.4 ms/rank in graph replay.
Most remaining device time is useful kernel work; another Python wrapper
change is not justified by these traces.

PPL/task scores from the earlier full-model comparison are unchanged evidence,
not rerun by this experiment. Its strict practical-task failure is still open.
The graph-path numerical result covers sixteen contexts and four generated
tokens, not full release qualification or a new stock-runtime comparison.

Archives are ROOT/cohort-dispatch-model-v2-evidence-rank{0,1}.tar.gz:

- Rank 0: 419,281,776 bytes, 574 payloads, SHA256
  7865bf68fa88e6121906d8890ed37986facf5b263cd0616e33804ceda754a415.
- Rank 1: 5,409,653 bytes, 137 payloads, SHA256
  a94609b7d08d310ca6b35d2057a6a474bd4b3eebba879f13cf2ea6c2d1271236.

All original failed attempts remain preserved. No production configuration,
published model or native arithmetic kernel was changed.

The [compact independent review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/cohort-dispatch-model-v2-review.json)
reconstructs all **384 float32 rows** from raw API responses, checks all
**336 source-scale groups** in independent audit order, recomputes all five
KL/NLL comparisons and every timing median, and independently verifies all
29 cohort proofs. All 96 prefill rows match original bytes by context across
the six legs. Both rank source inventories match the frozen preparation.
The full raw-array verifier was retained before inference, SHA256
f5e0e22670cb97831afd03e310ba4454cdf2e4cabcb2edda90e236df0f0c164d;
its Spark copy is ROOT/verify-cohort-model-v2.py. Raw host and GPU timestamps
are never compared across nodes.

## Frozen protocol

Use the same pinned mixed-MXFP4 checkpoint, native v9 U8 consumer, canonical
expert ordering, BF16 KDA/head/embedding adapters, TP2 transport, BF16
activations, 4 GiB FP8 KV/rank and fixed source contexts. Capture graphs for
**[1, 4]**, with compilation mode zero and FULL_DECODE_ONLY. Both eager and
graph timing reuse those same captured allocations. Model weights, pools,
native libraries and FP4 codes are unchanged by mode selection.

1. Validate source/native/library/dataset bindings, real vLLM CLI/plugin/
   extension, actual four-prompt CompletionRequest schema and CPU client fixture
   before pausing GLM. Wait for an idle service window, arm the independent
   restoration watchdog, then pause the original containers once.
2. Load the compressed model once. Check the existing full source-scale audit.
3. Run one small admission smoke per mode: eager with original wrapper, eager
   with direct wrapper, and original wrapper with graph dispatch enabled.
   Use only the isolated **127.0.0.1:18784** test server's stock
   POST /pause?mode=keep&clear_cache=false, queue the four prompts, hold one
   second, then POST /resume in a finally block. Never call these endpoints on
   the original service. GET /is_paused must confirm each pause.
4. A one-second grace is not proof of admission. Before/after worker counters
   must show exactly **42 canonicalization calls at 1032 tokens** (4 × 258)
   on each rank, plus 126 calls at M4 for the three eager decode steps.
   Graph mode requires actual FULL M4 descriptors, retained graphs and zero
   Python M4 decode calls. Stop and retain evidence if a smoke group splits.
5. Run six fixed quality legs: eager-qualified-a, eager-direct-a,
   graph-qualified-a, graph-qualified-b, eager-direct-b, eager-qualified-b.
   Each leg has the same sixteen contexts, four groups of four, four generated
   tokens and full-vocabulary distributions. Every group must pass the measured
   batch proof. Never retry a split group. Keep all raw API responses, 384
   float32 distributions, token IDs, conditioning and admission receipts.
6. Compare three self-repeats plus eager-original versus eager-direct and
   eager-original versus graph-original. Keep **KL ≤1e-5**, absolute mean
   decode NLL delta ≤1e-4, 48 matched real-KV decode steps/comparison and
   identical generated IDs/conditioning. Do not revise the floors. Retain a
   numerical failure while collecting timing/profiles for diagnosis.
7. Compare eager-original and graph-original in the existing real API case:
   four independent streaming requests, 256-token prompts, 64 generated tokens,
   two 16-token warmups/mode/order, five trials/mode/order, two interleaved
   orders. Retain all twenty timed groups. **Do not pause admission for timed
   API requests**; wall time and TTFT keep their original definitions. Record
   worker counter snapshots around each group to show actual batching.
8. After timing, profile the two modes separately on both ranks with controlled
   four-request admission and the same first four quality contexts. Retain raw
   responses and batch proofs. Inspect actual CUDA graph launches and device
   kernels, rather than relying on a configuration flag. Check all source groups
   again, stop this experiment and restore the exact original containers once.

PPL and the 34-task battery are not repeated by this diagnostic. A passed
graph/host comparison cannot qualify the complete compression adapter or
supersede its historical strict task failures. Graph allocations differ from
the old batch-one-only baseline; report them and collect a matched reference
before claiming a new whole-model compression delta.

## Source and preparation evidence

The frozen snapshot contains **130 Python/TOML files**, inventory SHA256
f2df57a36d0cb7651b626c14b204535cd552aae1df2e2e314e9bc996ba9b93f2.
Source upload SHA256 is
68229982243aa6ff6c958a108fdc7f0e9346db9bb3b780560a5715c12d389b52.
Preparation SHA256 is
49e005eb35b32886a08940dcd47e909f8efb39d8dc053101af6755e6e0ab9889;
CPU preflight SHA256 is
783e294c2c3ef03e44d68c969b17b1fad4a1cd7d4f35a0f3cb1789a37b28bf70.
Native hashes match on both ranks; actual CLI/extension/schema and all four
controlled CPU fixture groups pass before launch. GLM remained healthy.
It starts from the verified eagerdispatchv1 snapshot and overlays only the
client/controller/observer/profile analyzer changes. Native consumer and codec
remain source-bound to the qualified v9 library. No new arithmetic is introduced.

The first preparation, cohortdispatchv1, accidentally packaged another local
MoE runtime revision. Native hash preflight rejected it before any service
pause or inference. Its source upload, prepared metadata and failed CPU proof
remain unchanged. Revision v2 fixes snapshot assembly; it does not discard a
GPU measurement. Local ruff, strict core mypy and the full unit suite pass,
including controlled-admission cleanup and both-rank batch-proof tests.

## Exact Spark paths

ROOT, identical on both Sparks:

    /home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/full-model-v1

Use ssh spark for the head. Read state with host stdlib Python, without Torch:

    ROOT/CURRENT_LOSSLESS_STATUS.json
    ROOT/full-cohortdispatchv2-src/
    ROOT/cohort-dispatch-model-v2-prepared.json
    ROOT/cohort-dispatch-model-v2-cpu-preflight.json
    ROOT/quality-model-controller-cohortdispatchv2.json
    ROOT/cohortdispatchv2-controller.log
    ROOT/run-cohortdispatchv2/results/cohort-plan.json
    ROOT/run-cohortdispatchv2/results/cohort-dispatch-model-review.json
    ROOT/run-cohortdispatchv2/results/*-batch-proofs.json
    ROOT/run-cohortdispatchv2/kda/traces/
    ROOT/watchdog-cohortdispatchv2.json
    ROOT/restoration-cohortdispatchv2.json
    ROOT/cohort-dispatch-model-v2-independent-evidence-check.json

The head controller is run_cohort_model_dispatch.py in the frozen snapshot;
its independent watchdog launches the same entrypoint with --watch. The shared
driver's defaults still describe eagerdispatchv1; do not run that default
entrypoint for this revision. Engine names are
mxwave-glm53-full-cohortdispatchv2-aligned-r{0,1}; the CPU client is
mxwave-glm53-bench-cohortdispatchv2-aligned.

Idle wait is bounded at thirty minutes, startup twenty minutes, client thirty
minutes and restoration outage ninety minutes. An independent collector saves
both-rank traces/logs/source/native bindings and verifies all archive payloads.
Spark controllers are detached from SSH and laptop state.
