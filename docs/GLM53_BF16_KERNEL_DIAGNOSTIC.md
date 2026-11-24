# GLM BF16 kernel comparison and hardware counters

**Final C16 admission result, 19 October 2026, 09:01 UTC: complete.** The 10 ms policy
passes its prospective gate. Short-C16 TTFT is stock 1.383 s /
compressed 1.694 s (gap +311.5 ms).
This improves admission latency but leaves the C16 compression TTFT penalty unresolved.
All 94 numerical comparisons pass; CUDA saving is 9.367162 GiB across TP2.
Original GLM is restored healthy; no configuration is promoted. This concludes
compression experiments per user scope. See the [verified admission result](GLM53_C16_ADMISSION.md).
Older entries below are historical checkpoints.

**19 October 2026, 05:36 UTC:** the fresh stock versus input-only FP32 model comparison
is launched (`fp32inputmodelv1`, controller 1258389). Both CPU
serving preflights pass with 229 frozen sources; 781 tests / one GPU skip, Ruff,
strict mypy and the full CPU protocol simulation pass. It measures C1/C8/C16
quality, ordinary throughput/TTFT, CPU/CUDA memory and actual kernel traces with
all original gates retained. No new model result is claimed. Verified relocation
of completed evidence provides the required disk reserve; model inputs and
canonical archives are unchanged.

**18 October 2026, 22:50 UTC integration follow-up:** input-only FP32 split 2 passes
the independently verified GPU integration fixture (**434 payloads / 227
sources**): 120 retained-output checks, 66 native-fallback checks, compiled
captures and six two-stream stress fixtures. The existing pool is reused with
zero retained replay growth; a third stream is rejected. Both GPU jobs, retention
and exact original-service restoration are complete; the awake lease is released.
This adds no full-model quality/speed or CPU-saving result. The 216 prior exact-word
failures and two input timing repeat failures remain failed. Fresh matched model
qualification is next, after evidence-storage preparation.

**18 October 2026, 22:31 UTC speed follow-up:** FP32 timing independently verifies on
Spark/Mac (**246 payloads / 220 sources / 884,736 events**). Input split 2 saves
about **7-8% of component time** (46/48 strict timing conditions pass; two M8
pressure repeat controls fail). Output gains shrink substantially under pressure;
the broad screen fails 78/288 conditions. Prior 216 exact-word failures remain
failed. No new full-model quality/speed or CPU-saving claim is made. GLM is
restored healthy, all jobs/retention complete and the owned awake lease released.
Next is a bounded input-only split-2 model probe under unchanged KL/NLL limits.


**18 October 22:12 UTC precision follow-up:** FP32 split partials are independently
verified on Spark/Mac (2,227 payloads / 217 sources). All 576 comparisons pass
the 0.001 relative-error bound; all 144 basis cases are exact. However, all 216
random M4/M8 cases fail the retained exact-word gate, despite about 22-26 times
lower relative error than the old BF16-partial route. Source/storage, guards,
repeats and graph checks pass. No speed, model-KL or CPU-saving result is measured,
and no route is promoted. GLM is restored healthy; all jobs and retention are
complete. Next comes paired component timing, then any justified exploratory
model test under the unchanged KL/NLL limits.

**Follow-up:** the compact N8 candidate `n8compactv2` preserves every tested
output bit and removes the input residency limit, but is about 9–12% slower
at M4. It is rejected. All four `n8mechanismv1` variants also pass correctness
and fail their repeated performance gate. The
[rANS screen](#paper-follow-up-and-actual-mxfp4-compressibility) below is complete.

Completed on Spark2 GB10/SM121, 16 October 2026, run **`bf16kernelsv2`**.
Both original GLM ranks/configuration digests were independently verified
running at **19:52 UTC**, health 200, with no active experiment containers.
The controller completed at 19:50:59 UTC after automatically restoring GLM.
The cache-pressure follow-up is also complete; both original configurations
were independently checked healthy again at **20:09 UTC**, with no active
experiment and the profiling driver restriction unchanged on both hosts.

**The compressed head is already faster; KDA output performance depends on
cache state.** At M4 in repeated component replay, input is near stock,
output is about 13% slower, and the head is about 13% faster. A subsequent
unprofiled cache-pressure check changes output from 12.5% slower when warm
to 4.5% faster after eviction. Low output occupancy remains a tuning lead,
but it is not proof of a general compression penalty. Larger split-K fails
every existing numerical gate tested. No adapter default or model weight
was changed, and no full-model benchmark baseline was reloaded.

The [compact machine-readable review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/bf16-kernels-v2-review.json)
contains both timing orders, numerical errors and the attributed counters.

## Same-weight component results

These use the installed vLLM `UnquantizedLinearMethod.apply`/default CUDA
GEMM against the existing fused runtime, with identical source BF16 weights.
Each pair has eight alternating-order trials, twelve calls per arm per
trial. Timings use CUDA events without a profiler; synchronized wall timings
are retained separately. The direct fused call includes padding/cropping
but excludes the outer selective-KDA custom-op wrapper and TP collectives.

| Shape / execution | Stock BF16, ms | Compressed, ms | Median paired time ratio |
|---|---:|---:|---:|
| KDA input, M1 graph | 0.602 | 0.429 | 0.712 |
| KDA output, M1 graph | 0.200 | 0.141 | 0.706 |
| Head, M1 graph | 3.676 | 2.410 | 0.656 |
| KDA input, M4 eager | 0.452 | 0.459 | 1.008 |
| KDA output, M4 eager | 0.133 | 0.150 | 1.134 |
| Head, M4 eager | 2.813 | 2.438 | 0.868 |
| KDA input, M8 eager | 0.457 | 0.438 | 0.958 |
| KDA output, M8 eager | 0.136 | 0.145 | 1.076 |
| Head, M8 eager | 2.871 | 2.465 | 0.867 |

Ratios are medians of paired ratios, rather than ratios of the rounded
columns. M1 graph and M4/M8 eager correspond to the prior full-model pilot's
graph-size-one configuration; both eager and graph results exist at all sizes.
This is one component session. Some cases vary with trial/order: the M4
input eager ratio is 1.053 when dense runs first and 0.987 when fused runs
first. Treat that case as parity, not a demonstrated slowdown or speedup.
The output/head M4 conclusions agree in both orders.

Repeated component calls reuse one matrix. In particular, the 23.81 MB
packed output fits within the GPU's 24 MiB L2, while its 33.55 MB dense
control does not. Other model layers may evict those bytes between calls.
The paired timing is therefore a warm component comparison, not a controlled
cold-cache replay of a whole forward. The follow-up below measures this
sensitivity. The separate NCU run flushes caches, but its replayed durations
must not replace unprofiled timing medians.

As a **scale estimate only**, repeating the layer-zero KDA pair 34 times and
adding the head gives:

| Execution | Stock sum, ms | Compressed sum, ms | Difference, ms |
|---|---:|---:|---:|
| M1 graph | 30.923 | 21.786 | −9.137 |
| M4 eager | 22.687 | 23.171 | +0.484 |
| M8 eager | 23.012 | 22.277 | −0.735 |

This estimate excludes other layers' weight distributions, outer adapter
dispatch, communication, attention, scheduling and concurrent GPU work.
It is not measured whole-model throughput and cannot explain the whole B4
regression. It does show why the previous trace's approximately 24 ms of
KDA/head time should not be called removable compression overhead.

## Output cache-pressure follow-up

Run **`cachepressurev2`** tests the unchanged split-one output kernel and
stock GEMM on the same source bytes and M4 activations. Eight paired blocks
exercise both arm orders and both warm/evicted treatment orders, with twelve
per-call CUDA-event measurements per arm. Graph replay minimizes host launch
differences. Before each evicted measurement, a 128 MiB GPU buffer is read
and written on the same stream **before** the start event. Eviction work is
excluded from the measured interval; this is an explicit cache-pressure
workload, not an API guarantee that every cache line is invalidated.

| M4 output, graph replay | Stock BF16 | Compressed | Median paired ratio |
|---|---:|---:|---:|
| Warm | 122.98 µs | 138.35 µs | 1.1247 |
| After 128 MiB eviction workload | 200.79 µs | 192.22 µs | 0.9549 |

Every warm block is slower (ratios 1.115–1.135); every evicted block is
faster (0.947–0.964), with the same direction in both arm orders. GPU decoded
weight hashes and before/after output bytes match exactly. Peak component
CUDA allocation is 259,411,456 bytes. This establishes cache sensitivity
within this session, not a new model TPS result or a measured adapter change.
Do not use the warm 13% penalty as a full-forward fact, or replace it with
the evicted 4.5% gain as a whole-model claim. Kernel changes need to improve
both representative conditions.

The first guarded attempt, `cachepressurev1`, failed during CUDA context
creation while GLM was resident, before allocating test weights. GLM stayed
running. The successful v2 run used the independently watched, previously
authorized idle pause and automatic restoration. Its
[compact review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/bf16-output-cache-v2-review.json)
retains both orders, exactness, source hashes and the raw-report hash.

## Correctness, representation and rejected settings

Every decoded GPU weight byte matches its independently reconstructed source
SHA-256. Logical shapes are input `[12576, 4096]` (native rows 12608), output
`[4096, 4096]`, head `[77440, 4096]`; these are actual TP2 rank-zero shards.
The prepared layer-zero fixtures and head artifact were reused without repacking.

| Component | Dense bytes | Packed bytes | Saving |
|---|---:|---:|---:|
| KDA input | 103,022,592 | 73,394,904 | 28.76% |
| KDA output | 33,554,432 | 23,809,768 | 29.04% |
| Head | 634,388,480 | 447,464,872 | 29.47% |

All **9 current split-K=1 cases** pass random/basis activation, nondefault
stream, CUDA-graph capture/replay and zero replay-allocation-growth checks.
M4/M8 random outputs match stock byte-for-byte in this fixture. M1 relative
L2 is 0.000153–0.000168, below the unchanged 0.001 gate. Exact restored
weights still do not imply identical GEMM rounding in every configuration.

All **27 diagnostic cases with split-K=2,4,8 fail**: their random-activation
relative L2 is roughly 0.0025–0.0029, above 0.001, and M4/M8 KDA also require
byte-identical output. At M4, output graph time improves from about 145 µs
at split one to 98 µs at split four and 88 µs at split eight, but these are
rejected settings. The native split-K workspace/reduction rounds partial
results to BF16, so changing the split is a numerical change even though
the weights remain lossless. Do not enable it by relaxing the gates.

This deliberate paired benchmark retains dense and compressed representations
and temporarily reconstructs the full weight for verification. Its peak CUDA
allocation is **1,752,287,744 bytes** (profile process: 1,776,327,680); these
are component-test peaks, not a new runtime memory-saving measurement or a
change to the adapter's 256 MiB persistent scratch bound. Model quality
remains **unqualified because of the earlier unstable stock control**.

## Nsight Compute findings

Nsight Compute **2025.3.1** successfully collected counters in a separate
disposable container with `SYS_ADMIN`. Host `RmProfilingAdminOnly=1` was not
changed. The production model was stopped during collection. CUDA-profiler
start/stop and NVTX labels identify all **20 launches**, including nine main
GEMMs across three components and dense/current/split-four arms. The source
and native ABI are pinned to the same hashes as the qualified adapter.

The seven sections are SpeedOfLight, LaunchStats, Occupancy, SchedulerStats,
WarpStateStats, MemoryWorkloadAnalysis and InstructionStats. Kernel replay
uses **17 passes**, cold-cache control and unmodified clocks. Profiled times
differ from unprofiled medians; use them for attribution, not model TPS.

| Fused M4 kernel | Blocks, split 1 → 4 | Achieved occupancy, 1 → 4 | Allocated shared memory/block | Shared-memory block limit/SM |
|---|---:|---:|---:|---:|
| KDA input | 197 → 788 | 21.86% → 24.08% | 26,240 B | 3 |
| KDA output | 64 → 256 | 11.31% → 36.80% | 17,280 B | 5 |
| Head | 1,210 → 4,840 | 41.10% → 41.29% | 17,152 B | 5 |

The GPU has **48 SMs**, 48 maximum resident warps/SM, 24 MiB L2 and
102,400 bytes shared memory/SM. Fused blocks contain four warps (128 threads)
and use 68 registers/thread, allocated as 72. Shared memory restricts
residency before the register limit does.

**Output has low grid parallelism.** Only 64 blocks spread across 48 SMs
leave much of the potential residency unused (0.27 capacity waves).
Split four raises this to 1.07 waves and substantially increases achieved
occupancy. The unprofiled improvement and these counters identify parallel
work distribution as a credible target. They do not qualify split four's
numerics or prove that occupancy alone determines latency.

**Input has a shared-memory limit.** Three four-warp blocks permit 25%
theoretical occupancy; measured occupancy is 21.86%. Reducing allocated shared
memory from 26,240 to at most **25,600 bytes** would remove the shared-memory
barrier to four resident blocks, subject to all other constraints. This is a
specific footprint target, not a promised speedup. Four blocks would permit
33.3% theoretical occupancy, and tail scheduling still matters.

**Head already fills its five-block residency.** Occupancy is near its
41.7% theoretical bound and larger split-K barely changes unprofiled time.
The head should remain behind KDA in the tuning order.

Long-scoreboard waits are material, and the fused kernels execute substantially
more instructions than stock GEMM. However this export exposes compute/cache
pipeline metrics, **not DRAM byte counters or direct LPDDR bandwidth**.
For example, the head's 26% compute-memory pipeline throughput is not 26%
LPDDR utilization. Logical payload divided by latency is also not measured
DRAM traffic. Do not conclude that the memory controller has spare bandwidth
from this counter set, or add warp-stall percentages to host wait times.

## Original diagnostic priorities and gates

The N8 storage audit in step 2 now has the rejected candidates described
below. Do not repeat those changes as an untested next step.

1. Prioritize KDA output work distribution while preserving the full K
   accumulation. Audit a smaller row tile/cooperative layout against the
   codec's existing 64-row indexing. Changing a template parameter alone is
   not sufficient proof of a valid compressed layout. Another option is an
   FP32 split-K accumulator/reduction prototype, but reordering can still
   change output bits; it must pass the same gates before proceeding.
2. Audit input decoder shared-memory storage for the 25,600-byte residency
   target. The pinned launch code separately doubles the global common-byte
   and rare-BF16 maxima when allocating staging space. Audit whether their
   per-tile bounds/lifetimes permit shared storage; keep copy alignment and
   double-buffer ordering valid. This is a source-based hypothesis, not a
   tested layout. Keep all rare-exponent exceptions exact; do not cap buffers
   or alter exponent coverage to achieve the footprint.
3. The unprofiled output cache-pressure check is complete and reverses the
   warm-only timing conclusion. Qualify proposed kernels in both conditions
   on the existing component fixtures:
   both TP shards and representative layers 0/22/44, random/basis/stress
   activations, exact GPU bytes, existing numerical limits, nondefault
   streams, graph replay, scratch reuse and memory bounds. Require a
   repeatable gain in both timing orders and an independent confirmation
   session before another full-model load.
4. Retain the current split-one adapter and bounded prefill fallback until
   that gate passes. Attention-metadata reuse remains a separate runtime
   investigation with async/speculation correctness requirements. Resolve
   stock-repeat quality instability before a release claim.

## Evidence and continuation commands

On Spark1/2, `ROOT` is
`/home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/full-model-v1`.
The frozen component source is `ROOT/bf16-kernels-bf16kernelsv2/src/`.
Controller state is `ROOT/bf16-kernels-controller-bf16kernelsv2.json`,
restoration `ROOT/restoration-bf16kernelsv2.json`, independent service check
`ROOT/bf16kernelsv2-final-service-check.json` (head). Raw reports are on
Spark2 at `ROOT/bf16-kernels-bf16kernelsv2/results/` and copied to the head's
`ROOT/bf16-kernels-bf16kernelsv2/evidence/` for continuation without the Mac.

| Private evidence | Bytes | SHA-256 |
|---|---:|---|
| `timing.json` | 281,371 | `f86936b9764fbf62010421bf36cf087a7b4eac5450c36005830fc671f3b0d29b` |
| `profile.json` | 6,649 | `2c25376bfe8489ac07e9acc7d2aa991b10677852e41aee68d94283aa7b3883dc` |
| `kernels.ncu-rep` | 13,543,546 | `19718f88549d0b9923390422f836df21552330dae031f1d5ed0432ecb75c48be` |
| `ncu-labeled.csv` | 134,299 | `dd6302363b560a84959e65886e8b1fa8e66ddbcc3ba9c3cc85b522a04b0fa122` |

The CPU-only review was run on Spark1 from the retained timing and Nsight CSV
evidence. The experiment scripts for this study are not included in this branch.

The first `bf16kernelsv1` controller failed a helper import before creating
state or stopping GLM. Its failure log is retained. The dependency was removed
and a new immutable v2 directory was prepared; v2 CPU import/CLI, local lint,
strict core typing and tests pass. No runtime adapter implementation changed.

The main diagnostic's private archive is `ROOT/bf16kernelsv2-evidence-v1.tar.gz`
(1,866,186 bytes, 123 hash-bound payloads, SHA-256
`eea0b4c52e8357c88086e3974ac8d4a33b9874f6444e24ef75bea1a9ea0a5524`).
It includes frozen source, raw timing/NCU reports, failed-start log and service
checks; the review reproduces on Spark1 without a GPU. The cache follow-up
has a separate source snapshot at `ROOT/bf16-kernels-cachepressurev2/src/`,
state `ROOT/bf16-kernels-controller-cachepressurev2.json`, raw report
`ROOT/bf16-kernels-cachepressurev2/cache.json` (head) and
`results/cache.json` (worker). Raw report: 17,593 bytes, SHA-256
`708c12a53192ecfa2e446372f00b78699c26274dace1c875f50ca615dfc5f15b`.
Cache restoration and independent check are
`ROOT/restoration-cachepressurev2.json` and
`ROOT/cachepressurev2-final-service-check.json`.
The cache archive is `ROOT/cachepressurev2-evidence-v1.tar.gz` (302,005 bytes,
119 payloads, SHA-256
`5eec4a666759e88f099aad927ffe00a2af03df3c0f952adeab2a5849740d4246`).
Both archives were downloaded and every payload hash checked locally against
their manifests. They retain the failed attempts as well as successful runs.

## Compact N8 activation staging: correctness passes, speed fails

`n8compactv2` tests eight-row activation staging and `ldmatrix.x2` in the
existing BF16 fused kernel. The baseline stages only eight useful rows but
reserves sixteen and loads four fragments, of which N8 consumes two.
The candidate removes 2,048 bytes of double-buffer storage and loads only
the consumed fragments. N16+ retains the original layout and load. The
weight codec, buffers, decoder, MMA order and split-one accumulation are unchanged.

CUDA's input resource limit improves from **3 to 4 resident blocks per SM**:
dynamic shared memory **25,120 → 23,072 B**, registers/thread **68 → 64** for
layer 0/rank 0. Both shards at layers 0/22/44 meet the four-block resource
target. This is a resource query, not an achieved-occupancy measurement.

All 14 fixtures (input/output at three layers and both ranks, plus both head
shards) pass full GPU decoded-weight hashes and bit-exact arithmetic against
the original fused kernel. Each has 11 token counts, random/basis/stress
activations, nondefault streams and graph replay with zero allocation growth.
The exact reference is the original fused kernel, including its rounding;
these checks do not resolve the separate full-model quality gate.

| M4 input, compared with original fused kernel | Main paired ratio | Fresh-process paired ratio |
|---|---:|---:|
| Warm | 1.102 | 1.116 |
| After 128 MiB cache pressure | 1.086 | 1.103 |

Both timing orders and the independently rebuilt same-compiler control agree
on the regression. Output is approximately at parity in confirmation; head
gains are around 0–1%, with cache/order noise. The input lead and nonregression
gates fail. **Do not install this candidate or load the full model for it.**
Timings use graph replay; the model's M4 path was eager, so even a positive
component result would require that additional check before a serving claim.

The two GPU processes took roughly 48 seconds each, with peak allocated CUDA
memory about 1.22 GB. Original GLM IDs/configuration digests were restored
unchanged, health 200. Frozen source is
`ROOT/bf16-kernels-n8compactv2/src/`; reports are `n8.json` and
`n8-confirm.json` on the head, with raw worker copies in `results/`.
Build/fixtures are retained at `ROOT/bf16-kernels-n8compactv1/build/`,
referenced by the v2 run's relative `build` symlink. The unsuccessful v1
preflight stopped before a service pause: root-owned pack artifacts were not
readable by the controller. Ownership of only those generated fixtures was
corrected; their bytes and hashes did not change.

### Isolating the mechanism

`n8mechanismv1` separates four hypotheses using the same 14 fixtures and
the original/rebuilt controls:

1. `narrow`: eight-column fragment load with the original shared allocation.
2. `compact_x4`: compact storage with the original four-fragment load; upper
   lanes address valid duplicate rows, and only the original first two
   fragments are consumed. The old sixteen-row addresses would be unsafe.
3. `tailfirst`: original storage/load, but schedule the final five input row
   tiles first for the exact 12,608 × 4,096, N8, split-one geometry.
4. `compact_tailfirst`: compact eight-row/load kernel plus that permutation.

The metadata lead is exception skew: input row tile 196 has about 155–172k
rare BF16 values versus a typical 6–7k, and currently launches last. This
counts encoded values, including padding, not measured decode cycles or
total traffic. The permutation preserves every output tile and its complete
K accumulation. Other shapes and larger token paths remain unchanged.
Two separate containers run fresh per-candidate processes, reversing candidate
order in confirmation. Original GLM is paused only when idle and restored by
the controller plus an independent watchdog. No full-model experimental
checkpoint is loaded.

Both passes completed, taking 193.6 and 191.5 seconds respectively including
fresh per-candidate Python/CUDA contexts. All four candidates pass the exact
weight/output, stream, graph and allocation gates; **none passes the input
full-model follow-up gate**. Ratios below compare with the original fused
kernel; main and confirmation are shown separately rather than pooled.

| M4 input variant | Main warm / evicted | Confirmation warm / evicted |
|---|---:|---:|
| Narrow load, original buffers | 1.002 / 1.009 | 1.001 / 0.998 |
| Compact buffers, original load | 1.126 / 1.111 | 1.119 / 1.108 |
| Original buffers/load, tail first | 0.981 / 1.012 | 0.973 / 1.018 |
| Compact buffers/load, tail first | 1.053 / 1.066 | 1.106 / 1.096 |

The regression follows compact staging rather than load width. This does
not yet distinguish layout effects from four-block residency/contention.
Tail scheduling alone gives a small warm improvement but fails the 3%
input lead required across both cache conditions, controls, timing orders
and processes. Some order-specific comparisons also exceed the 2%
nonregression gate. Do not select a median while ignoring those checks.
No new full-model run is justified and no native adapter default changed.
The [independent review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/bf16-n8-mechanism-v1-review.json)
recomputes summaries from raw per-call samples.

A further causal diagnostic could reserve the original launch allocation
while retaining compact internal buffers, isolating layout from increased
residency. It is not run or claimed as a fix here. Inspect compiled
instructions/counters for an exposed decode bottleneck before a larger
pipeline rewrite.

Both original GLM ranks were restored and independently checked running
with unchanged IDs/image/configuration digests, health 200, and no active
experiment containers. State, restoration and independent proof are
`ROOT/bf16-kernels-controller-n8mechanismv1.json`,
`ROOT/restoration-n8mechanismv1.json` and
`ROOT/n8mechanismv1-final-service-check.json`. The head retains the two full
matrix reports at `ROOT/bf16-kernels-n8mechanismv1/{n8.json,n8-confirm.json}`;
worker `results/` also retains each candidate's original standalone JSON.
In the legacy report schema the key `compact` identifies the selected
candidate; the top-level `candidate` field and native manifest identify its
actual mechanism. The offline analyzer validates this binding across both
processes and recomputes timing summaries from samples.

## Paper follow-up and actual MXFP4 compressibility

[Approaching Shannon Bound with Lossless LLM Weight Compression](https://arxiv.org/abs/2606.15789)
uses tile-addressable rANS, a layer-shared codebook and decode into GEMM's
shared-memory layout. Its modified SGLang/CUTLASS evaluation uses A100/H200;
larger-batch throughput results do not establish same-batch gains on GB10/Marlin.
Our current fused path already avoids full-layer decoded-weight materialization.

The read-only CPU `entropy-screen-v1` samples 27 actual MXFP4 expert matrices:
layers 3/22/44, experts 0/143/287, all three projections. Each code and scale
tensor contributes four aligned 16-KiB windows: 54 cases, 3,538,944 bytes total.
Config and header checks establish MXFP4 coverage, group size and U8 storage.
This is a stratified sample, not all experts or a whole-checkpoint projection.

| Sampled payload | Zero-order bits/symbol | One rANS stream/tile | 32 independent streams/tile |
|---|---:|---:|---:|
| Four-bit codes | 3.915 / 4 | 1.65% saving | −0.30% saving |
| Eight-bit E8M0 scales | 0.579 / 8 | 91.56% saving | 83.12% saving |

These are CPU-encoded, exactly decoded sizes, including a frequency table,
initial state per stream, tile offsets/tags, lane lengths and raw fallback.
Tiles are 2,048 checkpoint bytes; 32 independent streams are an illustrative
parallel layout, **not the paper's implementation**. They show how parallel
state/index overhead can consume small entropy gains. A different tile size
or parallel coder may behave differently. The entropy model is per tensor
and cannot rule out compression exploiting longer correlations.

The original codes/scales are restored exactly. This screen introduces no
GPU path, runtime adapter or quality change. Scale percentages are relative
to checkpoint U8 scales, not to total model memory or the current compressed
runtime scales. GPU decode tables, allocator alignment, runtime Marlin layout
and scratch are excluded. Codes dominate MXFP4 storage; do not average the
equally sized sampled code/scale windows into a model-saving claim.

Next, evaluate a scale-only candidate against the **existing aligned-nibble
runtime representation**, with actual Marlin tiles, codebooks, states and
decoder resource costs. Compare a smaller-alphabet rANS table with simpler
constant/base-plus-exception coding; the extra table competes with scarce
SM121 shared memory. Prototype one fused expert tile only if measured
incremental resident savings justify it, then repeat exact-byte, arithmetic,
stream, graph, scratch and paired warm/cache-pressure gates. A general FP4-code
rANS port is lower priority on this evidence. No new full-model baseline is
needed to perform these CPU/component decisions.

For KDA kernel work, audit whether a producer/consumer split can overlap
software decoding with MMA beyond the current copy pipeline. Our decoder
currently reconstructs A fragments in registers; adding decoded-A shared
tiles and synchronization can reduce SM121 residency. Budget those resources
and identify exposed decode stalls before implementing one component variant.
The rejected compact candidate shows why a better resource limit alone is
insufficient. This is a proposed experiment, not a measured fix.

Reproduce the CPU screen on Spark1 (no GPU and no GLM pause):

```bash
python3 -S "$MXWAVE_RUN_ROOT/entropy-screen-v1/screen.py" \
  --source "$MXWAVE_RUN_ROOT/source" \
  --report /tmp/glm53-mxfp4-entropy-screen.json
```

Raw report: `ROOT/entropy-screen-v1/results/entropy.json`. The source is an
independent equation-based CPU rANS reference, not copied DietGPU/paper code.
Its stream length and expected symbol count are external metadata; it is a
feasibility screen, not a versioned/checksummed checkpoint artifact format.

The combined private evidence archive is `ROOT/n8-entropy-evidence-v1.tar.gz`
(1,656,375 bytes; 343 payloads; SHA-256
`f8ce5f6042ecb0e86066d09ab9fadb8f067c6baa27042e66f99da99dfab4c870`).
It retains frozen sources, both N8 sessions, the CPU screen, licenses,
controller/restoration checks and offline reviews, excluding model weights
and native binaries. All three reviews reproduce byte-for-byte on Spark1
without CUDA, and every archive payload was hash-verified locally.
The [evidence index](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/n8-entropy-evidence-v1.json)
binds the archive and independent final service proof. Ruff, strict core
mypy and the full pytest suite pass, with one existing GPU skip.
