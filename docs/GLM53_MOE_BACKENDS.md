# SM121 expert-backend investigation

**Final result, 19 October 2026, 14:31 UTC: no native integration.**
The bounded `nativeactualv4` screen completed all four actual-input cases and
failed every complete-layer speed gate. Spark and Mac independently verify
494 payloads, 68 source/dependency bindings, 217 baseline bindings and four
traces, with identical decisions (maximum float64 difference 6.94e-18).
Original service restoration, terminal jobs and the owned awake-lease release
are verified. No model, backend or serving configuration is promoted.

## Complete-layer measurements

Same GLM-5.3-Flash CT `mxfp4-pack-quantized` checkpoint and exact source weights
on SM121, layer 23, all 288 experts, TP rank 1. Actual C1/C8/C16/P256 activations
and routes were captured from the stock model. The current comparator is the
qualified aligned-scale Marlin path; native uses pinned FlashInfer MXFP8×MXFP4
with its default heuristic (`tune=False`). Each arm has 16 interleaved timing
trials, 40 graph replays per trial. Timing includes routing, quantization,
both GEMMs, activation/reduction, output zeroing and native scale reconstruction.

| Rows | Current layer ms | Native layer ms | Native extra time | Native/current relative L2 |
|---|---:|---:|---:|---:|
| 1 | 0.246062 | 1.035762 | +320.93% | 4.084% |
| 8 | 1.558483 | 2.451117 | +57.28% | 4.174% |
| 16 | 2.249204 | 3.189554 | +41.81% | 4.146% |
| 256 | 6.001999 | 7.000208 | +16.63% | 4.134% |

All four reference timing controls pass the unchanged 2% stability limit
(second-half/first-half ratios 1.01484 / 0.99516 / 1.00600 / 0.98346).
The required C8/C16 time reduction was at least 15%; both fail substantially.
These are component measurements, not full-model throughput or TTFT.
PPL, task accuracy and SQNR were not measured by this screen.

## What the traces establish

Each trace contains exactly one replay per arm, all GPU events attributed by
their host launch correlation, and zero unattributed GPU events. The values
below are kernel-duration sums from separate profiled replays, not the ordinary
timing medians above. Native FC1/FC2 kernels include fused ancillary work.

| Rows | Current complete kernel sum, ms | Native scale rebuild, ms | Native remainder, ms |
|---|---:|---:|---:|
| 1 | 0.279 | 0.772 | 0.312 |
| 8 | 1.532 | 0.707 | 1.695 |
| 16 | 2.308 | 0.727 | 2.773 |
| 256 | 6.265 | 0.764 | 6.773 |

Native reconstructs both full expert-scale arrays on every invocation to retain
the existing lossless scale storage. That costs 0.707–0.772 ms per tested layer.
The two native expert kernels themselves total 1.690 / 2.766 ms at C8/C16,
versus current Marlin's 1.518 / 2.292 ms. Even subtracting reconstruction leaves
the native profile above the current complete path at every tested size.
This subtraction is a diagnostic, not a measured cache or fused-decoder variant.
Removing reconstruction alone therefore has no support here as a sufficient
route to the 15% gate. No further GPU tuning or full-model port is launched.

Actual routing is sparse: only 54 / 78 experts are active at C8/C16, with
46 / 50 receiving a single token. That explains why concurrency alone should
not be assumed to produce large dense expert batches. It does not establish
a hardware limit or rule out a different kernel, layout or tuning strategy.

## Numerical and memory limits

The current complete path reproduces every captured model output bit for bit,
and all 20 repeats per shape are exact. Native outputs are finite but differ
from current by 4.084–4.174% relative L2. None of its 20 repeats per shape are
bitwise identical to its first output. Maximum native self-repeat relative L2
is 0.3892% / 0.1794% / 0.1590% / 0.0400% for M=1/8/16/256.
The native FC2 BF16 atomic reduction is a plausible repeat-noise mechanism;
activation quantization and arithmetic order also differ. Their individual
contributions were not isolated. No component error is converted into model KL.
The unchanged full-model KL <=1e-5 gate remains mandatory and untested for native.

Both component layouts retain 1,868,562,464 bytes of weights/scales. Native
additionally needs 113,246,208 bytes (108 MiB) of reusable raw-scale scratch;
GPU reconstruction is exact for both arrays. Its reported total scratch is
113,333,828–125,981,828 bytes across shapes. Both benchmark layouts and graph
allocations coexist: the measured 4,039,118,336-byte benchmark CUDA peak is
not a deployment peak or a full-model memory qualification. Detailed per-shape
allocation and graph deltas are retained in the final findings.

Every timing boundary records telemetry for both arms. C8 SM clocks span
2457–2470 MHz at 48–52 C, and C16 2418–2463 MHz at 51–53 C in both arms;
all samples are P0. Across shapes clocks span 2418–2509 MHz and 48–58 C.
Memory-clock readings are unavailable. Boundary samples cannot exclude transient
throttling, but stable interleaved controls do not suggest a large arm-specific
clock/thermal imbalance. Raw samples and power readings remain archived.

## Evidence and operational closure

- [Final findings](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/moe-native-actual-v4-actual-final-findings.json)
- [Independent Spark/Mac verification](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/moe-native-actual-v4-actual-cross-host-verification.json)
- [Attributed trace review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/moe-native-actual-v4-actual-profile-review.json)
- [Service restoration check](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/moe-native-actual-v4-actual-final-service-check.json)
- [Final retention manifest](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/moe-native-actual-v4-final-retention.json)
- [Preserved v3 setup failure](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/moe-native-actual-v3-final-retention.json)

Archive: `moe-native-actual-v4-evidence.tar.gz`, 85,825,023 bytes, SHA256
`b6fde7108cd6bc91ce7cb4f6240a5fac9e4e5be6de66ad6a59fd0e36b2779c84`.
It is retained at `/private/tmp/mxwave-moe-native-actual-v4/actual-evidence.tar.gz`
and under the remote audit `full-model-v1` root. Source archive SHA256
`f3dfc0f6af612d47b5bcb2f947e3ad91142131d9c90a5d6d199cd4c0bebebfc2`.
All 58 upstream FlashInfer files remain unchanged and external to the repository;
their license and provenance are recorded in `CONTRIBUTING.md` and the archive.
V3's loader setup failure is preserved separately; it yielded no timing result.

The existing warmed exact16 compression candidate remains numerically qualified
on its measured fixtures, with about 9.37 GiB TP2 saving and about 13% faster C1.
The [4,096-budget test](GLM53_PREFILL_BUDGET_4096.md) improves C16 throughput/tail
latency but worsens mean TTFT and extra memory, so it is not adopted. Remaining
first-token latency is not solved. The authorized bounded investigation is
complete; no BF16 cache, new kernel project or MTP work has started.

## Historical preparation and checkpoints

The entries below preserve the sequence and original failures. Their pending
or running language is superseded by the verified result above.

**Corrected native screen launched, 19 October 2026, 14:23 UTC.** Controller
735466 owns revision `nativeactualv4`; the live handoff records its watchdog
and pause state. Revision `nativeactualv3` stopped during fixture loading before any
case or timing. The harness passed unsupported `current_api=True` to the
archived fixture loader. Original service restoration completed at 14:21:47 UTC
with exact identities and HTTP 200. This is not a native-kernel result.
Revision `nativeactualv4` uses the existing explicit-empty-permutation repack
adapter, with a new CPU call-signature and cleanup check. It compares the complete routed layer 23 on captured
C1/C8/C16/P256 inputs, all 288 experts, TP rank 1. Current replay must reproduce
captured outputs exactly before native timings are interpreted. Both arms use
the exact source weight values; native activation quantization and reduction
rounding are measured separately. Require at least 15% complete-layer time
reduction at C8/C16, acceptable component numerics and preserved memory before
considering a full-model integration. Full-model KL <=1e-5 remains mandatory.
The failed attempt is independently retained and operationally closed before
this corrected screen's separate bounded pause. Its archive SHA256 is
`fcb04fd179d31832e6e20866d02cb2d9e1b609dd233da2be55581474d5e35f50`.

The [4,096-budget experiment](GLM53_PREFILL_BUDGET_4096.md) is complete and
independently verified: all 42 matched numerical comparisons pass, but mean
C16 TTFT and extra-memory gates fail. Its original service restoration and
terminal jobs were verified before this separate native screen launched.
No configuration is promoted. BF16 caching, new fused-prefill kernels and MTP
remain deferred. See [current handoff](GLM53_HANDOFF.md) for the live phase.
Working evidence: `/private/tmp/mxwave-moe-backend-v1`. Older launch/preparation
statements below are historical; their evidence and failures remain retained.

The 15% layer threshold is a screening requirement, not an end-to-end speed
claim. Complete-layer timing includes activation conversion, expert grouping,
two expert GEMMs, activation and output reduction. Preserve source bytes,
actual routing, all timing trials, control-repeat checks, peak/resident memory,
and clock/thermal telemetry. Full-model integration is conditional on the screen;
finite component arithmetic checks cannot substitute for full-model KL.

The installed CUTLASS and OAI Triton MXFP4 routes do not support SM121 in this
image. A pinned newer FlashInfer source path executed successfully without
upgrading installed packages. Historical evidence was byte-verified on worker
storage before duplicate head copies were removed; canonical archives remain.

## Compatibility checkpoint: 19 October, 13:14 UTC

**19 October 2026, 13:14 UTC — authorized expert/latency investigation active.**
Installed CUTLASS MXFP4 MoE rejects SM121 on both ranks. A pinned newer
FlashInfer SM12x MXFP8×MXFP4 path imports with the installed compiler and has
now executed/replayed graphs at M=1/8/16/256 on Spark2 (synthetic compatibility
inputs only; no speed/quality claim). Its M=1/8/16 outputs were not bitwise
self-repeatable; quantify this on actual operands before any integration.
The 4096-token dispatch extension passes all 39 guarded GPU specializations,
including block48/capacity8192. Preparing one 2048/4096 prefill-budget pair with
10 ms admission, mean/p95 TTFT, ITL and unchanged memory/KL gates. Capture
layer23 operands in separate eager forwards only after all measured work.
Original service remains healthy and unchanged. No full-model run has started.
Working evidence: `/private/tmp/mxwave-moe-backend-v1` and
`/private/tmp/mxwave-prefill-budget-v3`. See [backend audit](GLM53_MOE_BACKENDS.md).

Pinned upstream FlashInfer commit: `f64b4ba8762ded12a4f4b9a2092974c9a395e7ca`.
The installed CUTLASS predicate is false for120/121 and true for100/103;
the generic FP4 predicate being true on121 does not imply MXFP4 MoE support.
The installed OAI Triton backend also excludes SM12x. The newer native path
requires no installed-package edit for this isolated source probe. Public
package extras specify CuTe DSL4.7.1; the bounded probe used installed4.6.2.
Import/execution success is limited to these tested shapes and operations.
Native source archive SHA256:
`af0483fb787dff9fb1e23064af4725e449f7cf285370fcd0cb4b739573cf9e2d`.
GPU smoke source archive SHA256:
`c3f15cdc5438f5cf7dfbb51938f581da6f1670fcc00ae2738e5209eb607158f3`.

Both checks ran in bounded disposable containers beside idle serving, without
pausing GLM. Native telemetry stayed at48 C and about2411–2418 MHz. No timing
from this compatibility check qualifies speed; fixture uses8 synthetic experts,
not the full288 or captured activations. Native component benchmarking remains
pending actual operands and memory-accounted scale reconstruction.

The current memory saving includes about4.394 GiB of expert-scale compression.
A backend that retains all raw E8M0 expert scales would surrender that portion;
a reusable layer scratch may preserve it, but conversion time and scratch must
be included in the component screen. No backend has been promoted.

## Full-model test launched: 19 October, 13:21 UTC

Revision `prefillbudgetv3` is running with217 frozen source bindings, plan
SHA256 `952642d1105f557397c7ecb062f977eaa6521632c53893dcd12f7630aeabfdc1`.
Controller150509, collector150508, independent watchdog154619. Both original
containers are paused under the finite restoration deadline. The experiment
contains two model loads,72 timing groups and16 worker traces. Mean/p95 TTFT,
ITL, per-trial allocation peaks, unchanged numerical gates and actual graph
replays are independently reviewed. Stock-only operand capture occurs after
all measurements and uses explicitly eager C1/C8/C16/P256 forwards; it does
not change timed graphs or claim to reproduce graph intermediates.

Head storage was prepared by byte-verifying completed v2 and C16 payloads
on worker storage before removing duplicate head copies. Canonical archives,
protocols, reports and model files remain on head. Backup roots are
`ROOT/retained-prefill-budget-v3` and `ROOT/retained-prefill-budget-v3-extra`;
all copy manifests and receipts are retained under those experiment names.

Source inspection explains a plausible source of native self-repeat noise:
the FC2 finalizer uses BF16 global atomic addition (`red.global.add.noftz.v4.bf16x2`).
This is a mechanism finding, not yet a measured full-model KL diagnosis.

## Actual-operand comparison prepared: 19 October, 13:37 UTC

Revision `nativeactualv3` freezes 66 source/dependency bindings, including all
58 unchanged upstream files from the executed compatibility probe. Plan SHA256:
`472d49524ae8268252b7d7e32ad6835f042f7a6194c6408b45371f62ec14fe50`.
Source archive SHA256:
`36d89c1a9bb971c254050ce1c365531cf5586c165b961b3dd0da47175c882365`.
It has not launched. It requires all four actual operand shapes, the completed
full-model experiment, and a healthy restored original service before its own
bounded pause. This screen uses all 288 experts of layer 23 on TP rank 1.

The current Marlin replay must reproduce the captured model output bit for bit,
including the model's SILU clamp of 10 and its expert-sum operation. Both graph
arms retain 20 repeat outputs, 16 interleaved timing trials of 40 replays each,
clock/thermal telemetry, and a trace per shape. Native scale reconstruction is
inside every timed invocation, using reusable scratch and the same lossless
scale codec where supported. Persistent storage and scratch are reported
separately; keeping both benchmark layouts resident is not deployment memory.

The independent CPU verifier passes valid-evidence tests and rejects corrupt
payloads, wrong trial order and falsely stable controls. Its metric comparison
tolerances check the audit arithmetic; they do not relax the model KL gate.
Component numerical errors and atomic-reduction repeat noise remain pending.
No speed, quality or full-model memory claim follows from compatibility alone.

Prelaunch v1 was retained and rejected before any GPU job: its container omitted
the explicit aligned-adapter environment. V2 sets and asserts that environment;
the declared computation, source weights and performance/numerical gates are unchanged.

## Real operands retained: 19 October, 13:47 UTC

Stock capture completed on both ranks for M=1/8/16/256, with owned wrappers
removed and 4,621,888 payload bytes per rank. The activation is SILU, global
expert count 288, no expert map, and router weighting occurs after FC2 as required.
The prepared actual-input check passes all 217 baseline source bindings.
Prepared SHA256: `3b244135221ac7d5969a521200275af2c7fb7d67546ef01eeaa71bc3980521bf`.
Rank-1 operand manifest SHA256:
`efae5e268f7a332ae87e88b1d7adfeb25b742e8bcf21c754e330ddf3cfe303db`.
The component still awaits full-model completion and restoration before launching.

CPU inspection of the actual routing gives 8 / 54 / 78 / 195 active experts
at M=1 / 8 / 16 / 256. At C8, 46 of the 54 active experts receive just one token;
at C16, 50 of 78 do. Mean tokens per active expert are 1.185 / 1.641 at C8/C16.
These sparse expert batches are the workload the native screen must accelerate.
Logical selected weight payloads are 339,738,624 / 490,733,568 bytes per layer
and rank at C8/C16; these are unique-weight footprints, not measured DRAM traffic.

Stock budget timing is complete: C16 mean TTFT 1.855106 -> 2.031805 seconds
at 2048 -> 4096, while median throughput rises 80.996 -> 84.228 tok/s.
The ordinary trace changes from [2048, 2048, 23] prefill rows to one [4096]
forward. Thus one pass is achieved, but stock mean TTFT worsens by 176.7 ms.
These stock-arm observations await paired compressed results and full archival
verification; do not promote the larger budget from the pass-count reduction.

CPU scale audit verifies all 864 rank-1 source scale tensors for layer 23.
Native-layout w13 exponents span 118–122 (75,497,472 bytes), and w2 spans
119–122 (37,748,736 bytes). Both fit the existing aligned-nibble codec at
base 112. This establishes codec eligibility for this component, not whole-model
native memory qualification; GPU decode equality and timed reconstruction remain
part of the pending screen. Evidence: `actual-native-scale-ranges.json`.

Prelaunch v2 is also retained without a GPU run. Source review established that
FlashInfer consumes FC1 rows as [up, gate], whereas Marlin consumes [gate, up].
V3 permutes complete weight rows and their scale rows together; every source bit
is preserved. An asymmetric CPU SILU contract test verifies the correct output
and rejects the swapped branch order. The source freeze now contains 66 files,
including that test and its proof. This is an adapter correction, not a measured
native-kernel numerical failure. Only the corrected v3 screen is prepared to run.

## Executed setup failure and corrected v4: 19 October, 14:22 UTC

V3's immutable source and error are retained separately. It completed zero
cases and collected no native timing. The loader adapter is already established
in the repository's `benchmark_model_plugin.py`: temporarily supply the empty
permutation tensor required by this runtime's Marlin repack API, and restore the
owned wrapper after loading. V4 uses that same contract, without modifying
the archived loader, source weight bits, native kernels or gates.

A CPU test checks the pinned loader/repack/complete-MoE signatures, rejects the
failed keyword, and verifies restoration on both success and loader failure.
All earlier layout and verifier checks pass. V4 freezes 68 files, including the
same 58 unchanged upstream files. Source archive SHA256:
`f3dfc0f6af612d47b5bcb2f947e3ad91142131d9c90a5d6d199cd4c0bebebfc2`.
Plan SHA256: `2acc35a565c9aa05fd5a830551c8a867b9160a8ec5e102012faa1a433ddaeaac`.
Prepared SHA256: `52cf900fd465eaee79fc7a7e149ac354994779714e5403ea0512f852d411ccef`.
Working evidence: `/private/tmp/mxwave-moe-native-actual-v4`.
