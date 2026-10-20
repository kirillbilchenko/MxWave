# Reference decode-state diagnostic

**Local implementation update,18 October10:56 UTC.** The bounded actual-MLA
observer is now implemented locally and passes five focused CPU tests plus
the full suite (one skip) and Ruff. It has not been uploaded or GPU-run.
The active `output16v1` compression timing source remains frozen and has no
MLA instrumentation; do not launch a second GPU controller.

The replacement observer wraps only layer3's real inner MLA operation chain:
cache-write inputs/touched bytes, query BMM input/output, actual sparse
conversion inputs/slots/counts, shared reserved indices/plan buffers,
planned-order selected FP8 bytes, attention output and value BMM input/output.
Complete BMM constant hashes and original strides/aliases are checked outside
capture. Empty NoPE views and captured scalar scales remain explicit.
Planned-prefix cache reads are retained even when conversion-valid counts
are shorter, including real clamped slot-zero reads.

This replaces the broad24MiB boundary trace. Retained copies, conservative
gather staging and an8MiB transfer allowance must fit32MiB together.
Both proposed replay policies allocate SSD images and reset shared device
plan buffers; only the private-workspace policy additionally restores the
actual float/int workspace. Four balanced probes use five fixed replays each
and serve the fifth. No favorable replay selection or new numerical tolerance.
A tiny authored CUDA graph fixture is written but still needs execution before
any actual-model probe. CPU tests verify call preservation/hook restoration,
byte masks, empty metadata, budget rejection and private-restore ordering.
This is implementation/preparation evidence, with no new cause, correction,
quality or speed claim.

Local files: `integrations/glm53_mxfp4/mla_internal_capture.py`,
`mla_internal_worker.py`, `mla_internal_extension.py`,
`mla_internal_gpu_fixture.py`, `measure_mla_internal.py`,
`run_mla_internal.py`; tests `tests/test_glm53_mla_internal_capture.py`.
The source-map record below predates this local implementation.

**Internal source map prepared, 18 October 07:26 UTC.** Read-only inspection
of the installed source on both ranks finds matching hashes. The inner
`MLAAttention` includes native KV write, BF16 query-projection `torch.bmm`,
top-k slot conversion, the FlashInfer call, and a BF16 value-projection
`torch.bmm`. The earlier passing synthetic fixture tested the attention call
with generated operands; it did not test these actual projections or writes.
The next probe must observe those stages on real full-model data and retain
constant-weight hashes, touched/selected cache bytes, indices, scales and
private plan state. A tracked-plus-private-workspace restoration control is
still needed on the actual full module. No new cause or correction is proven.

Source snapshots and the operation map are local under
`/private/tmp/mxwave-mla-internal-source-v1/`; the source map is
`capture-plan.json`. At that timestamp the probe was not implemented; the
local implementation is described above and remains GPU-unrun.
It adds no instrumentation to the active compression timing.
Native implementation snapshots remain private evidence.

**Isolated attention follow-up complete, 17 October.** `mlaworkspacev1` ran six
fresh bounded processes / thirty graph invocations, without loading model weights.
FP8 and equivalent BF16 caches at four rows / 32 heads / latent width 512,
length 259 / top-k width 2,176, repeat byte-for-byte. Graph outputs also match
eager output exactly. Both ordinary and length-2,176 capture followed by replan
to 259 work in this fixture; scalar `plan_info` is equal at both lengths.
Restoring 136 MiB of private float/int workspace makes no difference. Those
workspaces remain unchanged after these short-context calls, so this is not
evidence about every scratch-using schedule or multi-layer interference.

All 380 archive payloads and 36 raw outputs verify on Spark and Mac, with exact
analysis agreement. Original GLM configurations restored healthy at
**19:40:08 UTC**. An initial local file check preceded completion of its transfer;
the failed log and successful post-transfer check are retained. No inference
rerun. The full module's numerical cause remains open: this fixture excludes
its projections, KV write and multi-layer interactions. Strict qualification and
all historical failures are unchanged. Next perform matched C1/C4/C8 timing and
separate paired C4 profiles; no repeated GSM8K/quality battery or roofline work.
Read the [verified isolated review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/mla-workspace-v1-review.json),
SHA256 `31c21f4d1ef5fa2208680863184f71b4fb8d8d3c2b1a0e9d602df57bcee836c5`.
The 158-file source is `ROOT/full-mlaworkspacev1-src`; plan/results are under
`ROOT/mla-workspace-v1`. Analysis bundle `mla-workspace-v1-independent-analysis.tar.gz`
has SHA256 `fcebca4d3f68ecb906b736a90b4cfa38bedd4c105a74ba46728bef4efa2c744f`.

**Graph-boundary follow-up complete and independently verified, 17 October.**
Revision `decodeboundaryv1` ran two first-decode probes, five fixed replays per
rank, on one stock raw-runtime load. All 336 source groups, 4,511 archive payloads,
4,100 raw boundary tensors, twenty hidden tensors and sixteen earliest changed
active-cache pairs verify independently on Spark and Mac. All starting tracked
bytes match; both analyses agree exactly. The fifth replay is served. Both
original GLM configurations are restored, health 200 at **19:21:30 UTC**.

The first changed module is layer 3's inner `MLAAttention` in **14/16** non-first
rank comparisons. The other two first change at layer 3's output projection,
which includes TP reduction; the peer rank already changes in MLA. Every paired
replay therefore has an MLA change on at least one rank. Recorded inputs to each
first changed module match bytes, and the first changed active cache remains
layer 4 / cache0. Resetting the captured stock-Marlin semaphore buffers
(1,536 bytes/rank) does not remove the variation. The active KDA prestates also
match across the two policies. This narrows the boundary before the MoE GEMMs.

Changed MLA outputs have relative L2 error 0.00443%–0.3453%; final hidden changes
are 5.67%–10.39% in this intrusive diagnostic. These are not ordinary-serving KL.
The inner module includes latent projection, cache write, sparse attention and
value projection; its first changing output does not identify the exact kernel.
Implicit metadata and private native buffers remain outside the recorded inputs.
All eight completions / 32 IDs match V2, using four distinct prompts. No new
quality or speed measurement ran, and strict qualification remains false.

Next isolate those MLA internals with fixed explicit operands, cache, plan and
private FlashInfer workspace. The installed backend is
`flashinfer_mla_sparse_sm90.py`, SHA256
`4449ea25921dcae1ec7988581ff26a6ff5a9e5d6d4a0ac4a68a0017ea108f136`;
the installed FlashInfer wrapper is SHA256
`58d06e61aa753f938bf5682640767541dbd4cb6989340abcdc8f6f90792db6e7`.
The runtime audit after inference is explicitly separate from preregistered
checks. The layer-3 boundary uses four rows, 32 heads and top-k width 2,176.
Then return to paired performance profiling/concurrency tests; correctness
localization itself makes no tokens/s claim. Roofline remains deferred.

Read the [verified boundary review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/decode-boundary-model-v1-review.json).
Its SHA256 is
`5f034a29d3f681e539ee71b9c64fc19223b5e45d1f8e9e9eb3c3ad80075760aa`.
Frozen source is `ROOT/full-decodeboundaryv1-src` (156 files); results are
`ROOT/run-decodeboundaryv1/results`. Independent analysis bundle
`decode-boundary-v1-independent-analysis.tar.gz` has SHA256
`64428b6db14f501ea0f4ec80f16c5d98b7945f680bb89775ac228b4b1142f18d`.
Raw rank-0/rank-1 archive hashes are respectively
`01072bd275c81a7273df08ae901e6ad2d10accadd3664a75b08862a00108d424` and
`4d7dfa5882152e4749a2b27804accfd2a5354e99ddb06706b52e25429af6faf9`.
Controller/collector PIDs were 1771533/1771532. The eight-case CUDA fixture and
CPU preflight passed before the model load; the 23 focused tests, full local
suite, lint and strict core mypy pass. All earlier failed gates remain unchanged.

**Earlier active-value follow-up complete and independently verified, 17 October.**
Revision `decodevaluesv1` ran four logical probes / eight rank reports / forty
fixed-state graph invocations on one raw-runtime reference load. All 336
before/after source-scale groups and 454 archived payloads verify. Independent
CPU verifiers on Spark and Mac reconstruct forty raw hidden tensors and 32 raw
current/reference active-cache pairs. Their discrete results agree; the largest
float64 reduction difference is 3.55e-15. Exact original service identities are
restored, health 200 at **18:43:50 UTC**, with no active experiment container.

In all 32 non-first replay comparisons, both active states at layers 0–2 stay
byte-identical. The first changed active view is **layer 4 / cache0**, followed
by all 62 active KDA views from layer 4 onward. The earliest observed active
cache error has max absolute value 0.0078125–0.03125 and relative L2 error
0.0973%–0.2264%. Final hidden relative L2 changes are much larger,
**4.30%–39.35%**, with maximum absolute error up to 4.375. Explicit device
synchronization before and after replay does not eliminate variation.

These are **intrusive fixed-tracked-state replay errors**, not ordinary-serving
KL or a new task-quality score. Native workspace and other internal state
outside registered KV/input/bounded metadata storages are not reset. Do not
label the large hidden changes as proven floating-point roundoff. The two
completion policies also have different probe histories; their magnitudes do
not establish a comparative improvement. All sixteen completions / 64 generated
IDs match the saved V2 reference, using only eight distinct prompts. Matching
IDs does not qualify the candidate or amend any historical failure.

The next model check is to record actual graph operands and outputs across
layer 3's MLA, MoE and mHC path into layer 4, then reproduce the first differing
operation from identical operands and controlled workspace. The active-cache
boundary does not yet identify that causal operation. No new KL, PPL, accuracy
or throughput measurement ran. Roofline work remains deferred by user.
Read the [verified active-value review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/decode-values-model-v1-review.json).

The frozen plan used the same two targets, normal/device-synchronized replay,
five attempts with the fifth served, SSD images and the five reused V2 repeat
legs. Policy order reverses on the second target. The full unit suite, lint and
strict core mypy pass, as do all eight actual tiny CUDA-graph fixture cases.
Source inventory SHA256
`d80e6588de59dbdec80c02eceb96dcf0d444ed197b96817d19de83cc20ebc5f7`;
source archive
`de7cd6dea63527f752834d954f66c635ee983d5d17d6929928f9e3b7d3a95b1c`;
prepared proof
`a0d08475a3de6b74a1139394207ef96ec8d4a2d2f34a189dbeea81cf511d7346`;
plan `81263c5fb60c04e67d41668e7368cd75577a91f5d677d1ba9a620c53602b6ba0`;
frozen independent verifier
`e7ee1f2f3e3dbbd862a2c6d6ae9589d9d61ad194255fdde6929f68e9293290c2`.
The compact review SHA256 is
`2e06e3b01a78926f62bc31a377f007c5a0982077bba759ce557b4587e9cee6e2`;
independent analysis bundle `decode-values-v1-independent-analysis.tar.gz`,
SHA256 `34c788c8dfb366ed64a29b5f63c6b4b1e75c0778a5560bd385020cab5a1b06d9`.
Raw archives: rank 0
`740d98fece542c9d9dda8810526f5992ff820c208fd3dfc0b559ef5c8e5dff0f`;
rank 1 `6587dcb73c72f20a4c7c72565998b3f782a5d31012a159af11f40c7b9b9e6507`.
Results are under `ROOT/run-decodevaluesv1/results`; the 150-file frozen source
is `ROOT/full-decodevaluesv1-src`. Controller/collector PIDs were 1496998/1496997.

**V3 complete and independently verified, 17 October, 18:19 UTC.** Six
logical probes / twelve rank reports / sixty graph invocations complete, with
336 before/after scale groups checked and 342 archive payloads verified.
Exact original service identities are restored, health 200 at 18:18:23 UTC,
and no experiment container remains active. Every replay starts with identical
tracked storage bytes, yet all twelve worker probes have five distinct hidden
output fingerprints and differing resulting storage. All 24 completions /
96 generated IDs still match the saved reference. No candidate or new KL,
accuracy or throughput benchmark ran; strict qualification remains false.

Active KDA pre-state is identical across all three first-decode probes for
contexts 4–7, including the same prompt order and generated IDs. At the third
decode of contexts 0–3, repeated probes differ in **62/68** active state hashes,
starting at layer 4 on both ranks; layers 0–2 remain identical before that probe.
Full post-cache views differ as early as layer 0, but include inactive/padding
state. Their names do not identify the earliest causal operation. The current
fingerprints do not quantify hidden-value changes or separate active post-state
from unused cache rows. Read the
[verified V3 review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/decode-state-model-v3-review.json).

The next useful check is bounded active post-state / raw hidden-value recording
and a device-completion control on the same restored graph. If drift persists,
record the first differing operation's operands around the layer-3/layer-4
boundary and replay it independently. The reference's execution/untracked
scratch is now a reproduced lead; compression is absent in this diagnostic.
Neither this lead nor matching generated tokens qualifies the candidate.

Authorized after the independently verified full GSM8K comparison on
17 October 2026. **Revision v2 finished with a snapshot-admission failure.**
All five unobserved reference repeats completed, but rank 0 could not reserve
8,563,791,048 bytes for the two GPU images. No full-model replay executed.
Both original services were independently verified restored and healthy;
all 678 archived payloads verify locally and on Spark.

The 320 retained float32 rows independently reconstruct from 20 raw API groups.
Mean decode self-KL is **0.000004644025 / 0.000009817437 / 0.000004663354 /
0.000009553701**. All generated IDs match and every prefill distribution is
byte-identical. These four comparisons pass the unchanged limits; this batch
does not resolve or replace the earlier failed reference repeat. Source-before
checks cover 168 groups; source-after was not reached. Read
[the partial-evidence review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/decode-state-model-v2-review.json).

**Revision v3 launched detached at 18:05:22 UTC and is now complete.**
Controller PID 1270278, independent collector PID 1270277. Both-rank CPU
preflight passes, including native bindings, actual CLI and msgspec schema.
It reuses those
five verified reference repeats. Complete before/reference-after images live
in owned SSD files, with 8 MiB byte-preserving transfers and comparisons;
no GPU snapshot images remain resident. Local alias/NaN/stride/chunk/cleanup
tests, full unit suite, ruff and strict core mypy pass. The tiny actual CUDA
fixture passed all four cases before the reference load, with 55,808 bytes
of Torch CUDA peak allocation and no retained GPU snapshot images. Roofline work is
deferred at the user's request until model qualification is addressed.
The completed benchmark remains 9.367 GiB / 5.33% less allocation, nearly
equal workload speed and 1,246 versus 1,244 numeric passes out of 1,319.

The preceding graph-four reference repeat reached KL 0.000011791085 against
the unchanged 0.000010000 ceiling. Its prefill distributions were identical,
but decode varied. A global CUBLAS workspace setting is already pinned to
`:4096:8`; setting it again is not a new determinism experiment.

## Frozen scope

One raw-runtime reference load on the same pinned image/checkpoint and TP2
transport, canonical dispatch, BF16 activations, 4 GiB FP8 KV/rank and graphs
[1,4]. No compression candidate, drafter, accuracy battery or speed claim.

1. Verify native/source/dataset bindings and the actual CPU CLI before an idle
   pause. Arm the restoration watchdog, then pause the exact originals once.
   Run a tiny CUDA graph fixture in a bounded 6 GiB container before loading
   model weights. A fixture failure restores the original service immediately.
2. V2 ran five fixed unobserved repeat legs on all sixteen original contexts,
   four generated tokens each. Save full probability distributions and every
   cohort proof. Record numerical failures without selecting or retrying a
   favorable repeat; this is diagnosis rather than a qualification attempt.
   V3 reuses this hash-bound evidence and performs no new repeat legs.
3. Run six controlled four-request probes: three at the first decode of
   contexts 4–7, and three at the third decode of contexts 0–3. These contain
   the two largest previously observed drifting steps. Selection is from
   reference-only failure evidence, not a new accuracy sample.
4. At the selected actual graph invocation, snapshot all unique owned KV
   storage allocations, input buffers and bounded metadata tensors. Restore
   exact bytes before each of five graph replays. Compare the resulting hidden
   outputs and named cache views. Serve the fifth fixed replay and retain its
   state, rather than selecting an output or advancing the cache five times.
5. Record logical active KDA state before each probe, input IDs, positions,
   prompt hashes, request order and state indices. Cross-probe comparisons
   require identical logical conditioning/order. Finish all before/after source
   checks, release temporary images, and restore the original service once.

Two SSD images cover at most 4.5 GiB of owned KV/input storage per rank, plus
at most 128 MiB for metadata images. Capacity checks precede admission; image
pages are flushed/released as chunks transfer. This is temporary diagnostic
I/O, not model-serving offload or a new runtime memory-saving measurement.

Native internal workspaces outside those registered allocations are not reset.
The observer and restoration copies perturb execution; a reproduced difference
localizes the problem to fixed-state execution or untracked workspace behavior,
not automatically to one kernel's reduction order. Different cache views are
leads rather than proof of the earliest causal operation. No reproduction in
these fixed probes would leave the older failure unchanged.

## Paths and current validation

Use `ssh spark` for the head. On both ranks:

```text
ROOT=/home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/full-model-v1
ROOT/full-decodestatev3-src/
ROOT/decode-state-v3-prepared.json
ROOT/quality-model-controller-decodestatev3.json
ROOT/run-decodestatev3/results/decode-state-plan.json
ROOT/run-decodestatev3/results/decode-state-model-review.json
ROOT/run-decodestatev3/stock/traces/decode-state-*-r*.json
```

The controller is run_decode_state.py, using the source-bound single-load
driver with ARM_LABEL=control. Its independent watchdog uses that same
entrypoint. Preserve the qualified v9 MoE runtime from the completed GSM8K
snapshot; another local revision is not interchangeable with its native library.

Local byte/alias/NaN restoration tests, full unit suite, ruff and strict core
mypy pass. The tiny CUDA graph fixture must additionally establish actual
worker hooks, both stream modes, final-state advancement, method cleanup and
detection of deliberately changing untracked scratch before full-model launch.

V2's both-rank CPU preflight passed **144 source files**, native/dataset bindings,
the actual raw-reference flags, CLI imports and schema. Inventory SHA256 is
`c0b1cad41db36ff16541f518f118b67f99ec0d160fb1cbb9b7e19e90d3a4a8d7`;
prepared SHA256 `3769e6b4f0ef446d8161e7f401e4e2d0509fc4298b8537872dc1f2005a039659`;
plan SHA256 `7405234898117b770416c3aa6987e18d2ceec771ba83fa6bfdeb888cbfd92ea4`.
The independent verifier is frozen before inference, SHA256
`775c5f7fac6d48952177719d7e8a0d90f6b823bebe0a5c6f4aae43d051a5247b`.
It remains unchanged. A separate post-inference partial-evidence verifier,
`09a2372e27a1f3b4d267ed3f8afcd5e0667472e3ac173b2f54451cbbca116b1f`,
checks the completed distributions and failed admission without claiming that
the full replay protocol completed.

V3 freezes **145 source files**, retaining the same native-qualified v9 runtime.
Inventory SHA256 is
`d4f50e4cd436c5a004b3810012ddcad4ace7530a543623099154467f798ad2f2`;
source archive `dcd08973b73cf815bb0dd8cac08190a223b49f8629c80ee573d9fa51c97ef333`.
The independent verifier is frozen before inference, SHA256
`dac29a1f5c2667a47311f286ba6ca787459b77c765ba2d0ba27f1104d60e3828`.
Prepared proof SHA256
`eb7a796245fd308c84985eb18887e16304c6f33073ab1bced6fc5807fce19c95`;
plan SHA256 `0601645d8494c516760aa09c98b509cd22932829e2e80ea7aba7d793c6f80392`.

Collection initially lacked the prepared-metadata mirror on rank 1. The exact
head copy restored that mirror; append-only collector resume verifies both
archives without rerunning inference or changing measured source. The original
failed collection proof is retained. The frozen analysis verifier completed
all assertions, then hit an undefined output-only `comparisons` variable left
from the reused V2 repeat analysis. An append-only helper repairs that output
field to an empty list; numerical/source/probe assertions remain unchanged.
Both original and repaired helpers, their logs and hashes are archived.

Raw archive SHA256: rank 0
`4b81ebc7c6e7994a54716fdae2bd47e84bc2a6f227da723ea09c1bd38f2b5f62`;
rank 1 `c8fb1d5a9d9e4e7250555fef0fe17e4a9dc4e1858ac6012854d103c91d57fe7a`.
The independent analysis bundle is
`decode-state-v3-independent-analysis.tar.gz`, SHA256
`705dcd82fca040c7278b30f46ea0a874e30b7616b217116d127444df54132a81`.

Revision v1's tiny fixture failed at the first CUDA allocation while GLM was
still running. Its graph hook and full-model inference never ran, and GLM was
never stopped. The 3 GiB container was not marked OOM-killed; the CUDA failure's
exact resource cause was not isolated. Preserve its source, proof and recovered
stderr. Revision v2 moves the unchanged tiny fixture into the guarded pause,
raises its container budget to 6 GiB and sets IPC/memlock limits before launch.
