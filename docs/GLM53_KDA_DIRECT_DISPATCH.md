# Single-dispatch KDA component prototype

**Component complete; not integrated into the adapter.** Both fresh processes
pass all 24 correctness cases, stream/graph replay, Inductor compilation and
zero retained replay growth. Host enqueue roughly halves. Initial performance
passed, but confirmation failed the unchanged 2% graph limit on the output
projection (**+4.35%**). The prototype is therefore **not performance-qualified**.
No serving model or native kernel was modified; original GLM remained idle and
running, health 200 with matching container identities after the component.

| M4 projection | Initial host ratio, both orders | Confirm host ratio, both orders | Initial graph, reference → direct | Confirm graph, reference → direct |
|---|---|---|---|---|
| KDA input | 0.545 / 0.549 | 0.535 / 0.537 | 431.824 → 433.072 µs | 603.168 → 606.480 µs |
| KDA output | 0.488 / 0.492 | 0.482 / 0.484 | 142.384 → 142.592 µs | 175.408 → 183.040 µs |

Eager CUDA-event ratios are 0.943–0.998 for input and 0.845–0.856 for output.
Peak allocated CUDA memory is 384,593,920 bytes in both processes. These are
component results, not a measured whole-model TPS improvement. Read the
[compact measured review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/kda-direct-dispatch-v1-review.json).

## Follow-up graph traces

kda-graph-diag-v1 captured two independent copies of the unchanged reference
plus the direct wrapper, using fixed M4 inputs and both timing orders.
An identical-reference comparison itself reaches **+2.67%** in one input
projection block. Direct/reference ratios range from 0.998 to 1.025 for input
and 0.999 to 1.023 for output. These results expose measurement/control
variation at the gate's scale; they do not replace the failed v1 qualification.

All nine profiled replays per projection have identical GPU activity signatures
across all three graphs: input has fill, copy, the same BF16 matrix kernel and
output slice copy; output has fill, device copy and the same BF16 matrix kernel.
Grid, block, register and shared-memory metadata match. Variable duration is
inside that same native kernel, with no extra GPU activity in the direct path.
This excludes an obvious extra kernel as the explanation, not every possible
CUDA graph dependency difference. Graph DOT export was unavailable.

GPU SM clock snapshots were 2,411–2,470 MHz, temperature 45–46 °C, Pstate 0;
memory-clock reads were unsupported. Snapshots do not attribute per-kernel
variation. The v1 timer creates events for each single replay and its original
gate times the arms sequentially. A cleaner prospective component experiment
should preallocate/warm events, time repeated replay batches in interleaved
orders, and require two identical controls to agree within the same 2% bound.
It must preserve v1's failure and native/prototype hashes. If isolating device
contention requires an authorized idle GLM pause, arm the independent restoration
watchdog first. Do not repeat a full-model baseline for this component check.

## Trace-backed mechanism

The completed stablequalityv1 M4 traces show 68 KDA CPU scopes taking about
9.1–9.4 ms per forward. KDA enters the nested ZipServ custom operation, repeats
buffer/shape validation and reconstructs a packed-weight descriptor. These
profiled scopes include waiting and instrumentation overhead, so their full
duration is not a predicted removable latency.

The prototype uses one explicitly registered CUDA operation for up to eight
tokens and calls the same hash-pinned mxzip_mm C ABI with the same padded
eight-token geometry and split-K=1. Its schema is read-only and does not claim
to mutate an unused prefill pool. Above eight tokens the original mutable-pool
KDA fallback remains the operation. It records cross-stream input lifetimes
and preserves logical output shape/contiguity. No compiled BF16 or expert-MoE
kernel source changes. This is an independent host wrapper around the existing
qualified bridge, not another compact-N8/kernel variant.

The native BF16 library remains SHA256
6d51630cd075a8c0dcbb957594341fb21e42d4b4634bb1c92735275d81a70f62,
and the old KDA runtime source remains
a3ddf07e196c7a13f6fcc3798fc54e4d311275283436ad103abab00126a93449.
Both nodes' 118 Python/TOML files match inventory SHA256
645987318f6de545122a0fd25f667ee0d4f3ffe226ac84f2316041f02af083b8.
The two-file prototype/benchmark overlay is SHA256
e263e9b88adc30036bf72f0b6263c32bcc218785897ae04ae4409868c5bfcae7.

## Prospective isolated component v2

**Complete; both predeclared processes fail the performance gate.** Launched
17 October at **12:28:44 UTC**, head controller PID 2816570. Both original GLM
configurations are restored with health 200 at **12:32:07 UTC**. Before launch,
both nodes' 120 Python/TOML files match inventory
SHA256 a59eb61404b4f1d2de16dea1c791ceabe79f5c27608c0ad99c0a74bab4bfa6b0;
preparation SHA256 is bcd14788a9f71d77c7e6d473ce7430f4f5c7af2744fb3c593393de44b6f87635.
The native library, direct wrapper and old KDA runtime retain their v1 hashes.
Pinned-image CPU imports/contract checks and local lint, strict core typing
and the full unit suite pass before launch.

The frozen protocol declares two fresh processes, initial and confirm, both
run regardless of their performance score. Correctness failures abort.
All 24 original cases, streams, changed-input graphs, Inductor execution and
allocation checks remain. Eager timing is unchanged. At each contiguous M4
projection, graph timing now uses three captures: reference_a, a separately
captured identical reference_b, and direct. Timing events are allocated and
warmed in advance. Each retained sample averages 20 graph replays; 30 samples
per arm are interleaved in forward and reverse orders. No samples are dropped.

For each order, identical-control max/min medians must be ≤1.02 and direct
must be ≤1.02 times **both** controls. The original ≥10% host reduction and
≤2% eager regression gates also remain in both orders for both projections.
Both processes must pass; neither the prior failure nor these limits change.

The controller acquires the component lock, waits for unchanged idle GLM
counters, arms and checks an independent restoration watchdog, then stops both
original GLM containers once. Only the bounded component runs on Spark2; no
experimental model is loaded. A 30-minute outage deadline and 600-second
per-container limits apply. Restoration starts after both processes or any
error and verifies the same original identities/configuration digests.

On the head, read ROOT/kda-direct-v2-controller.json,
ROOT/kda-direct-v2-prepared.json and ROOT/watchdog-kdadirectv2.json. Frozen
source is ROOT/kda-direct-v2-src; Spark2 reports are
ROOT/kda-direct-v2/results/{initial,confirm}/report.json. The head retains
each report in its controller and ROOT/kda-direct-v2/{initial,confirm}.json.
No adapter integration is enabled by this measurement.

Both processes again pass all 24 exact-output, stream, changed-input graph,
Inductor and zero-growth checks. Host enqueue ratios are 0.542–0.550 for the
input projection and 0.486–0.490 for output. Eager event ratios are 0.931–0.943
and 0.852–0.861 respectively. Nevertheless, input graph replay is **5.65–6.05%
slower than the two stable controls** in the initial reverse block, and
**4.38–4.42% slower** in the confirmation forward block. The confirmation's
output forward block additionally fails control stability (2.47% max/min).
Other blocks are near parity. All blocks remain; the unchanged component
qualification fails in both processes. CPU inspection independently recomputes
every median, ratio and gate from retained samples in the
[v2 review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/kda-direct-dispatch-v2-review.json).

Independent archive/service checking verifies **271 payloads** in
ROOT/kda-dispatch-v2-evidence-v3-rank{0,1}.tar.gz. Archive SHA256 digests are
58b05636f5d04bd38ef1ce0202d6d29bdbac24a978fda74c64caa92bb9efbb8d
and 2bdfaaab9121e00e76f06c2f165d134fb21915eedf5a022ef1d4b4c94e8ba7db.
The check is ROOT/kda-direct-v2-independent-evidence-check-v3.json. Two earlier
packaging attempts assumed fixture/older preparation manifests existed on both
nodes; they failed on missing metadata, not model execution. Their scripts,
failures and partial head archive are preserved. The final collector follows
actual rank ownership. No measurements or frozen source were rerun/changed.

## Next dependency/address diagnostic

The installed Torch API exposes the raw graph only with keep_graph=True;
the earlier debug-mode attempt used the default destroyed raw graph. A separate
bounded diagnostic now retains the graph, records all kernel/copy/event/empty
nodes and dependency edges through public CUDA APIs, and logs the 18 actual
native-call arguments. It compares independent captures, then a shared-pool
attempt. Shared-pool timings are interpreted as an address control only when
all recorded native arguments and logical output addresses match. Neither
previous performance failure is amended, and there is no native kernel change.

Frozen source is ROOT/kda-topology-v1-src, preparation is
ROOT/kda-topology-v1-prepared.json, and the head controller is
ROOT/kda-topology-v1-controller.json. Spark2's
ROOT/kda-topology-v1/results/diagnostic holds report.json and raw DOT graphs.
The original service stays running; the idle guard aborts only its own component
if production requests resume. It does not reload a full-model reference.

## Predeclared component gate

Use the two existing real layer-0/rank-0 BF16 KDA input/output projections.
Reconstruct source bytes independently, hash-check packed artifacts and verify
full GPU reconstruction before comparing operations. Keep the current runtime
as the control; installed-stock arithmetic equivalence is a separate question.

There are 24 cases: each projection at M0/1/2/3/4/7/8/9/16/32 plus strided and
rank-three M4 inputs. Require exact output bytes eagerly and on a non-default
stream, and five graph replays with changing input values and zero retained
allocation growth for nonempty cases. Real Inductor fullgraph execution must
match at contiguous M4. The first source-bound projection includes its real
logical/native row-padding difference.

Timing follows correctness: 30 interleaved samples per variant in both arm
orders, with host enqueue and CUDA-event latency recorded separately. Graph
replay medians are measured separately. To advance, both projection M4 cases
must reduce median host enqueue by at least 10% in both orders, with no more
than 2% eager-event or graph regression. A fresh process confirms an initial
pass. If the initial performance gate fails, retain it and stop; do not rerun
until a favorable result appears. Every source/benchmark/cache binding and
attempt remains available privately.

The component has two CPUs, a 3 GiB host-memory limit, a 1.5 GiB PyTorch CUDA
allocator limit and a 600-second container timeout. Require at least 4 GiB
host headroom and unchanged production counters before launch. If GLM wakes,
stop only the owned component and reject its interrupted timing. There is no
production pause or model reload for this component.

## Locations

ROOT on either Spark:

    /home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/full-model-v1

Head controller and preparation:

    ROOT/run-kda-direct-v1.py
    ROOT/kda-direct-v1-controller.json
    ROOT/kda-direct-v1-controller.log
    ROOT/kda-direct-v1-prepared.json

Immutable source is ROOT/kda-direct-v1-src. Spark2 results are:

    ROOT/kda-direct-v1/results/initial/report.json
    ROOT/kda-direct-v1/results/confirm/report.json

The head controller retains each complete report. It waited for
ROOT/behavior-repeat-independent-archive-check-v1.json to pass, then acquired
the component lock and the existing quiet-window guard. The diagnostic source
is ROOT/kda-graph-diag-v1-src; Spark2's ROOT/kda-graph-diag-v1/results/diagnostic
contains both compressed Torch traces and report.json. Both components are
complete. A prospective component pass would justify
a candidate-only integration using saved matched reference data where the
configuration stays identical. It would not itself establish whole-model TPS,
quality qualification, or an upstream/default adapter change.
