# GLM eager KDA dispatch experiment

**Complete, 17 October 2026: small incremental throughput improvement, failed
strict numerical gate, and an observed request-batching confound.** The same
resident best-v9 compressed model exercised both modes. GLM's original
containers/configurations were restored and healthy at **13:20:25 UTC**.
All **555 archive payloads** verify on Spark and locally. Independent analysis
reconstructed all 256 float32 distributions from raw API responses, checked
336 source-scale groups and recomputed every timing sample and comparison.

This is a dispatch-only follow-up. The preceding model comparison's
**9.37 GiB saving and 11.4–11.9% single-request throughput gain** remain
documented separately; they are not the size or speed gain of this host tweak.

## Completed model results and trace findings

The [independent review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/eager-dispatch-model-v1-review.json)
retains both orders, all twenty timing groups and the original failed gates.

| C4 / 256-token prompt, 64 generated tokens | Original dispatcher | Eager direct | Change |
|---|---:|---:|---:|
| First timing block, median tok/s | 38.068 | 38.951 | +2.32% |
| Reverse-first block, median tok/s | 38.558 | 38.879 | +0.83% |
| First-block median TTFT | 1.129 s | 1.092 s | -3.33% |
| Reverse-first median TTFT | 0.848 s | 1.090 s | +28.59% |

Latency is variable; these two blocks do not establish a robust large gain.
The candidate and reference here are two dispatch modes of the same already
compressed model, rather than a new compression-versus-stock comparison.

All 64 generated IDs per quality leg and all 48 decode conditionings match.
Decode KL is **5.258e-5 for original self-repeat**, **3.232e-5 for eager
self-repeat**, and **4.586e-5 across modes**, all above the unchanged 1e-5
floor. Absolute mean decode NLL deltas are within 1e-4. The numerical gate
therefore fails, including the reference control. This does not establish
quality loss caused by compression, and it does not qualify this integration.
The preceding practical-task failure remains unchanged.

Both-rank stable-dispatch counters reveal that only **9 of 16** quality groups
prefilled all four 258-token prompts together. Seven groups split despite
being submitted as a four-prompt completion. A 1032-token forward increments
42 MoE calls; both ranks observed 378 such calls, proving nine full groups.
This is a concrete confound for comparing floating point outputs. It is not
yet proof that batching caused all the observed drift.

Four worker Torch traces confirm different admission patterns: the original
profile admits requests 1+1+2; the eager profile admits 1+3. Matched M4 GPU
forward spans remain approximately **100.7–100.9 ms** on both nodes. The 68
KDA CPU operation scopes sum to **9.19 → 5.96 ms** on rank 0 and
**8.72 → 6.70 ms** on rank 1 (35% / 23% lower). These scopes include blocked
time; they are not isolated CPU compute. Nested ZipServ scopes are kept
separate from their parents. Reduced wrapper scope time did not yield a
comparable full-forward improvement.

Peer communication and uncovered time shift between ranks/modes. The original
profile's M4 communication sums are about 0.96 / 18.65 ms; eager is about
19.87 / 2.02 ms. This does not prove a network or CPU bottleneck: different
batch formation and activations remain confounds. BF16 KDA prefill still
reconstructs 68 matrices, costing roughly 35.6–37.9 ms/rank in these traces.

Next, require measured batch-shape equality using the pinned server's
pause/keep/resume admission API. Then test actual batch-four graph replay
against eager on one resident model with the same captured graph allocations.
Keep the existing KL/NLL floors, preserve failures, and stop if batch control
cannot be proved. The current launch captured only batch one.

## Evidence motivating the narrower policy

The [unrestricted direct wrapper](GLM53_KDA_DIRECT_DISPATCH.md) reduced host
submission but failed its graph performance gates. Retained-graph inspection
then established matching complete topology: four nodes/three edges for the
input projection and three nodes/two edges for output, without extra event,
wait or empty nodes. With all 18 native arguments and logical output addresses
matched through a shared pool, direct/reference replay differs by less than
0.2% on both fixtures and timing orders. That supports an allocation/timing
explanation; it does not identify a specific memory subsystem or supersede
either preceding failed gate. The [topology review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/kda-capture-topology-v1-review.json)
retains both independent-pool and shared-pool measurements.

The narrower policy uses the simpler dispatcher eagerly and delegates to the
original mxwave_glm53::kda_linear operation while Torch compiles or CUDA captures.
Existing model graphs stay intact. It is opt-in experimental integration;
the default kda_runtime.py remains byte-for-byte unchanged. New selector RPCs
change only its apply function between completed request groups, synchronizing
first. They replace no weights, pools or captured graphs. Compilation retains
the original operation, so this optimization targets eager serving.

## Completed component gate

Two processes were declared in advance and both ran. Both pass all 24 original
cases: M0/1/2/3/4/7/8/9/16/32, strided M4 and rank-three M4 on each real
source-bound projection. Full source reconstruction, eager/stream output bits,
five changed-input graph replays, zero retained replay growth and real Inductor
execution pass. A capture observer proves every nonempty case uses the original
operation. Inductor graphs contain kda_linear and no kda_fused_direct.

| M4 component | Host enqueue ratio, both processes/orders | Eager CUDA-event ratio |
|---|---:|---:|
| Input projection | 0.547–0.564 | 0.939–0.948 |
| Output projection | 0.503–0.511 | 0.853–0.857 |

These are host submission improvements of about **44–50%**. Eager event timing
includes host enqueue gaps and does not isolate GPU arithmetic. The original
≥10% host improvement and ≤2% event-regression limits pass. Shared-pool graph
checks additionally require identical native arguments/output addresses,
matching full topology and stable identical controls; direct must stay within
2% of both controls in both timing orders. Both processes pass.

[Independent raw-sample recomputation](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/kda-eager-dispatch-v1-review.json)
checks every median/ratio, capture route, compile target and address match.
Full-model TPS, latency and concurrent decode quality are reported above;
the failed numerical gate keeps the component's model integration experimental.

Both nodes' 124 Python/TOML files match source inventory SHA256
8fc5f68232d4ab263be93cfe525ae93006f18e359fee28bd6eb6a5daed519c3a.
Component preparation SHA256 is
75ef95df864372b76050b265ab328f5bc3daf076b07f83bd4a4f350b030941e9.
The eager policy is 792a0086d7856ba20e5f999344e5c71cee06d44bb2de83aae24a560935e21afc;
the original KDA runtime remains a3ddf07e196c7a13f6fcc3798fc54e4d311275283436ad103abab00126a93449.
Native BF16 library remains 6d51630cd075a8c0dcbb957594341fb21e42d4b4634bb1c92735275d81a70f62.

All **291 final component/topology archive payloads** verify on Spark and locally, with
original GLM running and configuration identities unchanged. Archives are
ROOT/kda-eager-v1-evidence-rank{0,1}.tar.gz, SHA256
0efb88a878234ab4deedde319358665731fdff84360a43f04f2aee58070ed927
and e19e58072ca2b12b87b68e1d0ac559b906abd80a66eb1b1a923877b6675b8117.

## Frozen single-load model protocol

Both modes use the same pinned mixed-MXFP4 source, v9 U8 scale consumer,
BF16 KDA/head/embedding adapters, stable dispatch patch, TP2 transport,
4 GiB FP8 KV/rank, BF16 activations, seed 42 and batch-one-only CUDA graphs.
The existing memory/quality baseline is documented in
[the stable-dispatch experiment](GLM53_STABLE_DISPATCH_EXPERIMENT.md).

1. Load the compressed candidate once. Keep the default qualified operation
   during initialization/capture. CLI, source/native proofs, transport and the
   actual concurrent CompletionRequest schema pass CPU validation before pause.
2. Check all 168 actual source-scale groups against the independent CPU audit.
3. Run four fixed quality legs in order qualified-a, eager-a, eager-b,
   qualified-b. Each submits the same 16 authored contexts in groups of four,
   returning full-vocabulary distributions for four generated tokens. Keep
   raw API responses, all float32 rows, prompt hashes and generated conditioning.
   Each leg includes **48 real KV decode steps** after its prefills.
4. Compare qualified self-repeat, eager self-repeat and cross-mode distributions
   only with identical conditioning. Require all generated IDs/conditioning
   to match, mean decode KL ≤1e-5 and absolute mean NLL difference ≤1e-4.
   Preserve an unstable reference as failure. Continue collecting timing/traces
   for diagnosis without accepting a failed numerical gate.
5. Use the existing four-request/256-token prompt case, two 16-token warmup
   groups per mode/order and 64 generated timing tokens. Alternate modes in
   both orders, five trials per mode/order, retaining all **20 timed groups**.
   Switch only when all running/waiting request gauges are zero.
6. Collect separate short Torch profiles for each mode on both ranks after
   unprofiled timing, check all source groups again, reset the qualified mode,
   stop this engine and restore the original GLM containers once.

PPL and the 34-task battery are not repeated in this dispatch-only diagnostic.
The prior full-model strict task failure remains unchanged. This batch cannot
establish general quality qualification, publication readiness or benefit in a
fully compiled serving configuration. It measures the remaining eager host cost
with one model/graph cache and separates correctness from timing evidence.

The frozen 126-file source inventory is
fcb450d7372aa52e74384d2d41c3eabafbca98e82f862132ec37ecdb4c328d52;
preparation SHA256 is bf47f0750d0c5555dffef0a13e86fe0141e2539e8b6cb97895d2a300a15c1dd2.
Local ruff, strict core mypy and the full unit suite pass; new concurrent-choice
and idle-switch tests also pass. Native v9 source/library/dataset checks match
on both nodes. Saved v9 transport/OMP settings match. The actual pinned-image
CLI/plugin/extension and concurrent client fixture pass CPU validation.

## Spark paths and monitoring

ROOT on both nodes:

    /home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/full-model-v1

Use ssh spark for the head. Host stdlib Python is sufficient to inspect:

    ROOT/CURRENT_LOSSLESS_STATUS.json
    ROOT/quality-model-controller-eagerdispatchv1.json
    ROOT/eagerdispatchv1-controller.log
    ROOT/eager-dispatch-model-v1-prepared.json
    ROOT/eager-dispatch-model-v1-cpu-preflight.json
    ROOT/eager-dispatch-model-v1-launch-receipt.json
    ROOT/run-eagerdispatchv1/results/dispatch-plan.json
    ROOT/run-eagerdispatchv1/results/eager-dispatch-model-review.json
    ROOT/watchdog-eagerdispatchv1.json
    ROOT/restoration-eagerdispatchv1.json
    ROOT/eager-dispatch-model-v1-independent-evidence-check.json

Frozen source is ROOT/full-eagerdispatchv1-src. The engine containers are
mxwave-glm53-full-eagerdispatchv1-aligned-r{0,1}; the head client is
mxwave-glm53-bench-eagerdispatchv1-aligned. Actual trace mounts are
ROOT/run-eagerdispatchv1/kda/traces on each node. The evidence collector saves
both engine logs and the client log under the head's run-eagerdispatchv1/logs.
Final archives are ROOT/eager-dispatch-model-v1-evidence-rank{0,1}.tar.gz.
Their SHA256 values are
8e851cdca458a0a291c28f830517517db693da0e91c74e0d704cf3f0a4f274b6
and bc24dfe89e5ff11a115fb0d6d94498b6b2bee15c58d20272f5366546daf54a1c.
The former has 423 payloads and the latter 132. All raw responses, source,
worker/frontend traces, logs and restoration records remain retained.

Idle wait is bounded at 30 minutes; startup 20 minutes; client 30 minutes;
the restoration watchdog has a 90-minute outage deadline. It runs on Spark
independently of SSH, the laptop screen and the separate Mac sleep assertion.
Use bounded Docker workloads for GPU analysis; do not import Torch on a Spark
host. Preserve partial results and original failed attempts on any error.
