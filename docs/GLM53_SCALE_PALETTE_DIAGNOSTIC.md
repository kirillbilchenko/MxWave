# Exact palette compression of Marlin expert scales

Updated 17 October 2026. The component comparison and full-model
storage/answer diagnostic are complete. The codec remains experimental:
latency and strict numerical qualification fail. The existing combined GLM
measurement remains **7.15 GiB saved across two Sparks**, with a B4
throughput regression and unqualified formal quality. See the
[combined results](GLM53_LOSSLESS_RUNTIME_RESULTS.md).

**Latest: shared-stage decoding improves the component result but still
fails latency nonregression.** All 56 real-layer GPU cases pass exactness.
Concentrated M128 now takes 5.3–7.6% longer than the current codec, versus
13.5–21.6% in the preceding affine-only experiment; M512 takes 2.2–2.8%
longer, versus 10.4–11.1%. Uniform routing stays near parity. The storage
format and FP4 weight codes are unchanged. No new full-model run is made.
See [the pipeline follow-up](#shared-stage-pipeline-follow-up).

## Storage result and scope

The source is layer 23 of
`INCModel3/GLM-5.3-Flash-MXFP4-Mixed-CT-AutoRound`, revision
`35d1b0d5cb4a0db075fd2749c897e970dc15bfa5`: all 288 experts, both
projections and both TP2 partitions. CPU reconstruction follows the actual
Marlin scale permutation. The GPU fixture separately checks its complete
scale bytes against those CPU hashes.

| TP rank / scales | Current aligned nibbles, bytes | Palette records, bytes | Additional saving |
|---|---:|---:|---:|
| 0 / gate+up | 37,748,752 | 18,891,112 | 49.956% |
| 0 / down | 18,874,384 | 9,437,400 | 49.999% |
| 1 / gate+up | 37,748,752 | 18,889,864 | 49.959% |
| 1 / down | 18,874,384 | 9,437,384 | 49.999% |
| Total | 113,246,272 | 56,655,760 | 49.971% |

These counts include record, exception and descriptor payloads. They are
**scale storage savings relative to our current compressed codec**, not
50% of the model. Original U8 scale payload here is 226,492,416 bytes.
The first GPU partition measures 56,624,128 → 28,329,984 allocated bytes,
including allocator rounding, and exact reconstructed source bytes.
The subsequent `scale-palette-all-v1` CPU audit passes **all 36,288 source
scale tensors in all 42 main expert layers**, reconstructed into 168 TP
scale groups. Original U8 payload is 9,512,681,472 bytes. The current codec
retains 4,794,092,128 bytes, including its two raw fallback groups; the
palette format retains 2,384,481,712 bytes. That is **2,409,610,416 bytes /
2.244 GiB / 50.26% additional scale-payload saving**. This remains payload
accounting, not a measured additional full-model allocation saving.
166 groups select palette 119/120/121; two select 120/121/122. Both are
data-selected consecutive palettes, without a model-specific value rule.

The preceding CPU tile screen covers complete scale groups in layers
3, 23 and 44. Its first run stopped on a baseline-accounting assumption:
layer-44 down scales cross an aligned-nibble bucket and use the existing
raw fallback. The corrected layer-44 run passes. Do not treat the first
partial report as a successful whole screen or compare that fallback
against a fictitious nibble representation.

## Representation

`integrations/scale_tiles.py` implements independent, model-agnostic U8
codecs. The selected prototype, `mxwave-palette16-v2`, chooses each input
tensor's three most frequent bytes and sorts that palette. It stores one
32-bit two-bit-selector record for each 16 source bytes, in fragment order.
A vector containing another byte instead stores `0xc0000000 | index`,
pointing directly to its exact 16-byte exception. Normal selector records
cannot have those top two bits set. The descriptor is 24 bytes.

There is no hash table, separate vector-index array, extra GPU decode
launch, clipped exponent or change to FP4 weight codes. An affine palette
can be converted directly into the stock BF16 fragment bit layout;
arbitrary palettes and all 256 U8 values have an exact fallback. The
palette is selected from data, without hardcoded GLM layers or exponents.
The first fused consumer is specific to the pinned Marlin BF16/MXFP4
backend on SM121; that implementation is not a generic vLLM/Ollama plugin.
Automatic full-model codec selection is not implemented.

This is a simpler exception format inspired by the entropy screen, not
an implementation of the paper's GPU rANS decoder.

## Correctness, integration failure and diagnostic

CPU round trips pass for every complete source group. Six standalone GPU
decoder cases pass: every byte, all 65,536 selector-word patterns, upper
exponents, a non-affine palette, a constant palette and exceptions across
the 1 MiB chunk boundary. Independent BF16 fragment bits, non-default
streams and CUDA graph replay pass without replay allocation growth.

The first direct-record integration (`sparsescalesv4`) nevertheless fails
with an illegal memory access during repeated eager execution, after its
first M1 equality checks and graph timings. No complete case was committed;
that run supplies no usable performance qualification.

The launch-blocking `sparsescalesv5` diagnostic reproduces the fault in the
candidate gate/up GEMM on the first cache-evicted eager sample. The control
passes that sequence. This uses `CUDA_LAUNCH_BLOCKING=1` and arm/sample
progress records, not Nsight timing or Compute Sanitizer. The original
serving containers restore automatically after each attempt.

Source inspection identifies an unused final register prefetch: the
Marlin pipeline requests the next stage even when the current stage is
the last computation. Inline scale bits tolerate an unused stale read;
an exception record can dereference those bits as an invalid address.
The guarded build suppresses only that unused sparse-codec prefetch;
all consumed loads and the control paths retain their original order.
`sparsescalesv6` passes all 28 cases and all 28 fresh-confirmation cases
with the guard and the same record format. The observed fault no longer
reproduces; its exact address was not confirmed with Compute Sanitizer.

The repeated comparison **passes exactness and memory, fails latency**.
Uniform graph routing stays near parity, while concentrated routing gives:

| Tokens | Candidate/current graph-latency ratios, both runs/ranks/cache modes |
|---|---:|
| 1 | 0.989–1.006 |
| 4 | 0.994–1.024 |
| 32 | 1.011–1.100 |
| 128 | 1.213–1.228 |
| 512 | 1.117–1.160 |
| 2048 | 1.067–1.100 |

These ranges summarize medians; the unchanged acceptance gate also checks
each arm order. The comparison deliberately keeps all representations;
its 3,344,635,904-byte peak is not the deployed adapter's memory footprint.
The [offline review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/expert-scale-palette-v2-review.json)
retains all raw-sample-derived cases, allocation accounting and counter inputs.

## NCU attribution and follow-up

`sparsencuv1` profiles four GEMMs per arm: gate/up and down at M1 and M128,
with the complete 288-expert layer resident, rank zero and concentrated
top-8 routing. Both arms pass exact current-codec outputs before/after
profiling. NCU performs 17 replay passes, flushes caches and leaves clocks
unchanged; its durations are diagnostic rather than new paired timing.

At M128 gate/up, executed instructions increase **136.16 → 173.85 million
(27.7%)**, and NCU duration increases **592.96 → 726.94 µs**. Down changes
**72.54 → 90.70 million instructions (25.0%)**, with **336.80 → 386.14 µs**.
Both allocate 96 registers/thread and 33,792 shared bytes/block, with
similar 24% achieved occupancy, a three-block shared-memory limit and
zero reported register spills. The counters support extra decoder work
as the next target, rather than an occupancy reduction. They do not
measure physical LPDDR traffic or establish complete causal attribution.

Offline SASS exports expose repeated descriptor/palette loads in the
generic decoder. `sparsescalesv7` snapshots read-only metadata once per
kernel; both complete processes still show the concentrated-prefill regression.
It is not an accepted improvement.

The subsequent build, `scale-sparse-native-v6`, adds an explicit **affine-only
consumer policy** without changing the format. It removes generic palette
branches and always uses exact addition for consecutive palettes. The CPU
packer records the verified palette property; `NativeMoe.bind()` rejects
nonconsecutive palettes before any CUDA launch. The standalone decoder
retains its generic all-byte path. `sparsescalesv8` completes all 28 cases
and all 28 confirmation cases with exact current-codec output bits and the
memory gate passed. The latency gate still fails: concentrated M128
typically takes about 13–21% longer and M512 about 10–11% longer. Neither
metadata caching nor the affine policy is an accepted speed improvement.
No defaults change. The full-model hook separately validates this policy
during preparation because its dispatcher bypasses `NativeMoe.bind()`.

The installed image's stock GEMM also differs numerically from the
original qualified compressed-codec library. The comparison records that
difference separately. Matching the existing codec is not proof of
installed-stock arithmetic equivalence or a passed full-model quality gate.

## Shared-stage pipeline follow-up

`sparsepipelinev2` profiles the latest direct decoder, runs a shared-stage
comparison and fresh confirmation, then profiles the shared-stage consumer.
The entire successful batch uses one GLM pause and final restoration.
`scale-sparse-native-v9` is an opt-in `--shared-stage` build: it fetches the
same four-byte records, expands each into the existing 16-byte U8 scale
slot after the issuing thread's async-copy wait, and publishes it with
the existing barrier. Consumers use the stock scale-conversion path.
No expanded global-memory copy, extra decode launch, changed FP4 codes,
new shared-memory allocation or changed matrix/reduction arithmetic is
introduced. Wind-down stages are not decoded after their valid input ends.

The [verified review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/expert-scale-shared-stage-v1-review.json)
recomputes timing from raw samples. Six decoder stress cases per process,
all 28 layer-23 cases per process and all 28 confirmation cases pass exact
bytes/outputs, dynamic routing, non-default streams, graph replay, lock
reset and allocation-growth checks. Both complete TP2 partitions are tested
sequentially. This is not a new all-42-layer GPU consumer qualification.

| Tokens, concentrated top-8 routing | Previous direct decoder / current codec | Shared stage / current codec |
|---|---:|---:|
| 1 | 0.993–1.001 | 0.991–1.019 |
| 4 | 0.982–1.002 | 0.960–1.024 |
| 32 | 1.017–1.032 | 0.990–1.011 |
| 128 | 1.135–1.216 | 1.053–1.076 |
| 512 | 1.104–1.111 | 1.022–1.028 |
| 2048 | 1.040–1.060 | 0.990–0.996 |

Ranges are graph median ratios over both runs, ranks and cache conditions;
the unchanged gate additionally checks each arm order and eligible eager
cases. Previous-direct timings come from its earlier unprofiled comparison,
not a contemporaneous direct/shared timing pair. Shared-stage uniform
routing ranges 0.987–1.013 across all tested sizes. The at-most-2% gate
still fails. Storage/allocator accounting remains the same compressed
format; the earlier 2.220 GiB full-model delta has not been remeasured
with this consumer, and the earlier quality status remains unqualified.

NCU now covers the actual latest direct library and the shared-stage
library, each against the exact same current-codec binary. At M128:

| Counter | Latest direct decoder | Shared-stage decoder |
|---|---:|---:|
| Gate/up executed instructions | 169.73 million | 141.62 million |
| Down executed instructions | 88.57 million | 75.19 million |
| Gate/up Tensor Core active cycles, % of peak | 20.97% | 26.87% |
| Allocated registers per thread | 96 | 96 |
| Shared memory per block | 33,792 B | 33,792 B |

The current-codec instruction counts are approximately 136.16 / 72.54
million. Moving decoding out of repeated fragment loads cuts candidate
instructions by 16.6% / 15.1%, leaving about 4.0% / 3.6% above control.
This intervention supports decoding/pipeline work as a real overhead source.
Shared-stage gate/up still has a higher barrier-stall ratio (1.24 versus
0.76 in its control). Producer expansion, synchronization and remaining raw
U8 scale conversion are leads; their individual time contributions have
not been isolated. NCU replay durations are diagnostic, not replacements
for unprofiled paired timing.

These metrics answer the utilization question: instruction issue is about
53–60% of peak for the profiled gate/up variants, while occupancy is around
24%. Occupancy is an active-warp measure, not GPU utilization. CUDA-event
kernel intervals exclude host submission. The counters do not establish
physical LPDDR bandwidth saturation or whole-model CPU saturation.

The concrete next candidate is exact **nibble staging** for tensors whose
complete source range fits one aligned 16-value bucket. Expand normal
two-bit selectors directly into four-bit offsets once per shared stage,
and use the current fast aligned-nibble fragment loader. Validate every
source byte, including rare exceptions; checking only the three palette
values is insufficient. Keep the U8-stage path for unsupported ranges.
The complete source-range audit confirms **166 of 168 groups** satisfy
`minimum // 16 == maximum // 16`, including their exception values.
The two unsupported groups are layer 44's down projection on TP ranks 0
and 1, with full ranges 118–128 and 119–128 respectively. They retain an
exact fallback; no values may be clipped to make them eligible. Eligibility
must be checked from each packed tensor, rather than hardcoded from this audit.
This is a generic data-property policy, not a hardcoded layer/exponent
rule or further quantization. The subsequent nibble-stage experiment below
implements and evaluates this proposal.
Require the same exactness and latency gates before another model load;
quality remains a hard acceptance gate.

Two earlier CPU build attempts fail before GPU work on missing tool paths;
the successful build records both the CMake launcher and CUDA compiler
paths. The initial `sparsepipelinev1` launch fails before GPU work because
an absolute host-only library symlink is invisible under `/audit`. The
corrected batch uses relative links and a CPU-only container path/hash
preflight before pausing GLM. Failed attempts are retained in the archive.

Both original GLM identities and configuration digests independently match,
both run, health is 200, and no batch containers remain active at
**07:39:52 UTC**. Evidence under `ROOT`:

| Item | Path |
|---|---|
| Latest-direct NCU reports | `bf16-kernels-sparseaffinencuv3/results/` on worker |
| Shared timing and confirmation | `bf16-kernels-sparsesharedv2/results/` on worker; JSON/logs copied to head |
| Shared NCU reports | `bf16-kernels-sparsesharedncuv2/results/` on worker |
| Frozen native source/provenance | `scale-sparse-native-v9/{src,upstream-vllm,native}/` on worker |
| Batch plan and state | `scale-pipeline-plan-v2.json`, `scale-pipeline-controller-sparsepipelinev2.json` |
| Restoration and independent check | `restoration-sparsepipelinev2.json`, `sparsepipelinev2-final-service-check.json` |
| Complete private archive | `scale-pipeline-evidence-v1.tar.gz` on head |

Archive: 9,597,020 bytes, SHA-256
`71da3325ea235ee14c6a19650077f9ce77d46e61567ec905bb98cb906cba66df`.
All 33 outer manifest payloads and the nested 526-file worker archive
verify locally, including both selected native binaries. It retains
raw samples, 16 counter profiles and reports, source/licenses, failed
preparations and restoration evidence without model weights. Ruff, strict
core mypy and the full pytest suite pass (one existing GPU skip).

## Nibble staging and static layout selection

The 17 October follow-up implements `--shared-stage --nibble-stage` without
changing global selector/exception storage. `moe_scale_policy.py` validates
the complete source range, puts the verified base and layout flag in the
upper bits of the existing 24-byte descriptor, and retains U8 staging for
unsupported ranges. The GPU producer expands each record into eight bytes
of exact nibbles in the existing 16-byte shared slot. The consumer uses the
aligned-nibble BF16 loader. Shared allocation, FP4 codes and matrix/reduction
arithmetic remain unchanged.

`sparsenibblev1` completes two independent processes, reversing the TP shard
order in confirmation. All 56 real layer-23 cases pass exact up/down/final
outputs, dynamic routing, nondefault streams, graph replay and managed-lock
checks against the original qualified codec. Both processes additionally
pass 24 decoder stress cases (all aligned buckets, arbitrary palettes, all
selector words and rare out-of-bucket values) and four synthetic fallback
GEMMs each. Those eight fallback cases keep real weight codes but edit one
scale per tensor to force a range crossing; they are safety coverage rather
than additional checkpoint qualification. Unverified descriptors fail closed.

The first implementation fails the unchanged latency gate:

| Concentrated routing | Candidate/current-codec graph ratio, both processes/ranks/cache modes |
|---|---:|
| M128 | 1.268–1.289 |
| M512 | 1.162–1.184 |
| M2048 | 1.067–1.082 |

The ratio ranges use paired median ratios; the gate separately retains
both arm-order medians and eager cases. Most uniform cases are near parity,
but M2048 reaches 1.041. Counter captures at concentrated M128 measure
136.16 → 170.92 million instructions for gate/up and 72.54 → 89.37 million
for down. Both arms allocate 96 registers per thread and 33,792 shared
bytes per block, with about 24% active-warp occupancy. Increased dynamic
work is a lead; reduced occupancy is not supported for these profiled kernels.
NCU replay durations are diagnostic, not paired timing measurements.

The controlled `--static-nibble-stage` follow-up moves layout selection
into `NativeMoe.bind()`, using the CPU-verified complete range once before
capture/timing. Its nibble GEMM omits the U8 branch. Unsupported groups
bind the existing exact U8 consumer; a missing fallback raises an error.
The standalone decoder still checks all ranges. This isolates the device
layout branch without changing storage, packing math or matrix arithmetic.
Its two processes also pass all 56 real-layer and eight synthetic fallback
GEMMs, plus 48 decoder stress cases in total. Every unsupported GEMM selects
the U8 consumer before launch, and missing/unverified policies fail closed.

| Concentrated routing | Static-layout candidate/current-codec graph ratio |
|---|---:|
| M128 | 1.082–1.087 |
| M512 | 1.053–1.072 |
| M2048 | 1.026–1.045 |

Uniform M1–M512 paired medians range 0.989–1.005; uniform M2048 reaches
1.039. The unchanged gate checks both arm orders and eager cases as well
and still fails. Allocation in each paired-comparison process peaks at
3,382,383,616 / 3,382,384,640 bytes with deliberately retained source,
control and candidate arms. This is not the component-only retained size
or a whole-model allocation result. Per-shard gate/up scale allocations
remain 37,749,248 → 18,891,776 bytes, and down 18,874,880 → 9,438,208.

At concentrated M128, the static gate/up/down instruction counts fall to
142.28 / 75.51 million, versus the same-control pair's 136.16 / 72.54 million.
The kernel change substantially removes the first version's extra dynamic
work; it does not remove all scale-pipeline overhead. Register allocation
increases to 120 per thread versus 96, while shared allocation stays 33,792
bytes. The register occupancy bound is four blocks rather than five, but
the shared-memory bound stays three and measured active-warp occupancy
remains approximately 24%. Do not attribute this result to an observed
occupancy loss. Gate/up barrier-stall ratio is 1.36 versus 0.77 in its
control, a remaining scheduling lead rather than a measurement of exactly
recoverable latency. Bank conflicts and per-source-line stall costs remain
unmeasured. Counter captures use 17 replay passes with profiler cache
treatment; unprofiled paired timing remains authoritative.

**Decision:** neither nibble variant passes the 2% gate. The previous U8
shared-stage consumer remains the best measured palette implementation:
M128 1.053–1.076 and M512 1.022–1.028 in its earlier paired comparison.
These older and newer variant/control comparisons ran in separate windows,
with the same pinned aligned-codec control binary; they are not a new
interleaved U8-versus-nibble pair. No default or production configuration
changes. Reuse the existing model-throughput baseline. Prioritize
operator-level repeat-control stability and the practical-quality launch
contract before another full-model session. Further pipeline work should
have a concrete producer-overlap or source-stall attribution experiment.

The [compact two-variant review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/expert-scale-nibble-stage-v1-review.json)
retains exactness counts, every tested size/routing ratio range, both-order
gate outcomes, hardware counters and raw-evidence hashes. Native builds
are `scale-sparse-native-v10` (device-selected) and `v11` (static). Component
roots are `bf16-kernels-sparsenibblev1`, `bf16-kernels-sparsenibblencuv1`,
`bf16-kernels-sparsenibblestaticv1` and `bf16-kernels-sparsenibblestaticncuv1`.

The first run restores both original containers with matching identity
digests and health 200 at **08:05:51 UTC**. Its verified raw archive is
`ROOT/scale-nibble-evidence-v1.tar.gz`, 9,039,078 bytes, SHA256
`6bb9a2a26737923cad3d178a0c7c46f4ae534a39028891431c40d42d7efbf411`.
All 24 outer payload hashes, the nested 546-file worker archive hash and
the candidate native binary hash are verified locally. The static variant
is a separate prepared follow-up batch after that restoration; each batch
owns one pause, its own watchdog and final restoration.

`sparsenibblestaticv1` also completes restoration: both original identities,
configuration digests and mount digests match, health is 200, and no batch
containers remain active, independently verified at **08:13:40 UTC**.
`ROOT/scale-nibble-static-evidence-v1.tar.gz` is 9,144,770 bytes, SHA256
`dd24da1ebd5994f9d2d87b1bc661fcaea561cf07285f2d2ad013b4137ea4155d`.
All 24 outer payload hashes, its nested 546-file archive hash, both job
manifest bindings and native binary hashes are verified locally. Candidate
library hashes are `a2ae635d287025bd471dec076ace973af8c27de2b87d4c92b2f33d3b2e670b68`
for v10 and `0b7931a131dfc6d942f422b5bbad5b6749aef702d0b28484198b7eaa1e085043`
for v11. Raw timings, counters/replay reports, source/licenses, build and
restoration evidence remain on Spark and in the local verified archives.
Full derived reviews and their final analyzer are retained in
`ROOT/scale-nibble-analysis-v1/` and `ROOT/scale-nibble-static-analysis-v1/`.
The repository retains the compact review linked above; model
weights and raw profiling payloads are not copied into the repository.
Local Ruff, strict core mypy and the full pytest suite pass, with one existing
GPU-only test skipped locally. Native SM121 correctness and timing are
measured in the separate Spark component containers described above.

No new whole-model load, throughput sweep or model-quality qualification
is part of these component trials. The previously measured 2.220 GiB
full-model saving belongs to the same compressed storage with the earlier
consumer; it has not been remeasured with either nibble consumer.

## Gate and next steps

The unprofiled component comparison retains the declared gates:

1. Exact source bytes and exact current-codec up/down/final outputs under
   identical sorted routing, dynamic routing, side streams and graphs.
2. At least 20% additional allocated scale-memory saving.
3. At most 2% paired latency regression in both cache conditions and both
   arm orders, including small eager decode and graph replay.
4. A fresh confirmation process, reversing TP-partition order.

Each process tests M=1, 4, 8, 32, 128, 512 and 2048 with uniform and
concentrated top-8 routing, on both TP shards sequentially. It loads a
complete layer, not the full model or TP collectives. CUDA event intervals
exclude the 128 MiB eviction workload. Allocation growth and managed locks
are checked. Debug timings cannot pass the offline performance analyzer.

If correctness fails again, obtain a Compute Sanitizer address report
before another format change. If exactness passes but latency fails,
profile selected representative calls with NCU before changing the
decoder. Reuse the collected full-model Nsight Systems/PyTorch traces;
a component win is required before treating this as a performance improvement.

The latency gate has not passed. At the user's explicit request to check
real answers and losslessness, `palettemodelv1` completed a separate
full-model storage/quality diagnostic. It is not performance
acceptance or a new throughput sweep. Both arms retain the same combined
KDA, vocabulary, dispatcher, graph-size-one and fixed-KV settings; only the
expert-scale consumer changes. Its plugin owns its descriptors/native
library and validates the affine policy. The original serving containers
are restored automatically after both arms or any failure.

Both arms complete source checks, graph/eager chats, two 16-context
full-vocabulary decode passes and practical tasks. Practical coverage is
six authored tasks, including a 4,261-token retrieval prompt; it is not a
long-context or sustained mixed-serving qualification. Do not publish a
speed or quality claim from scale entropy.

The user's requested real lossless check is mandatory: verify every
actually replaced tensor on both GPU workers against source bytes, then
run a fresh current-codec repeat control and candidate on identical
conditioning, seeds, graph modes and batches. Compare full log-probability
distributions, not just generated token IDs. An unstable control leaves
arithmetic/quality unqualified; do not relax the existing stability floor.
Exercise functional coding tests, structured JSON, mocked tool calls,
retrieval within the configured context limit and sustained mixed serving.
The reference is the existing MXFP4 checkpoint, whose earlier quantization
error is retained. CPU scale losslessness is established; **full-model
arithmetic/quality for this new consumer remains unqualified**.

## Full-model memory, answers and numerical limits

The [full-model review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/expert-scale-palette-full-model-v1-review.json)
preserves the original result and an independently recomputed CPU review.
All 11 JSON input hashes and all 256 retained full-vocabulary arrays verify;
the recomputed flags and numerical metrics match the device-side review.
Each arm uses the same already quantized checkpoint, KDA/vocabulary
compression, graph configuration, TP2 transport and fixed 4 GiB KV per rank.

| After identical inference/check sequences, both ranks | Current expert-scale codec | Palette codec |
|---|---:|---:|
| Live CUDA allocation, bytes | 181,160,413,696 | 178,776,583,680 |
| Live CUDA allocation, GiB | 168.719 | 166.499 |

The additional measured saving is **2,383,830,016 bytes / 2.220 GiB /
1.316% of the reference runtime allocation**, about 1.11 GiB per Spark.
This is a matched full-model allocation delta, distinct from the 2.244 GiB
payload estimate. The control already contains the previous combined
adapter; it is not an uncompressed or BF16 reference. Do not add old
cross-run allocation deltas and label the sum a new measured stock result.
Fixed KV allocation is unchanged, so additional context/concurrency capacity
was not demonstrated by this run.

Both workers reconstruct every replaced expert-scale group exactly before
and after inference, matching the independent 42-layer CPU source audit.
The other loaded-source and graph/eager/graph chat checks also pass.

| Authored task | Reference | Candidate | Interpretation |
|---|---|---|---|
| JSON, three-line instruction, slugify, retrieval | Pass | Pass | Identical final text; slugify functional tests pass |
| Merge intervals | Strict scorer fails | Same failure | Identical code returns tuples; scorer expects lists |
| Named weather tool | HTTP 400 | HTTP 400 | Request rejected before model evaluation |

All **five requests that produced final answers have identical final text**;
both code artifacts have matching hashes. The interval prompt did not
explicitly require list-valued output, making that strict return-type check
ambiguous. Its original failed score is retained. The slugify scorer strips
Markdown fences; both outputs include fences despite the plain-code prompt.
The separate post-quality admission probe captures the tool rejection:
`tool_choice=function "get_weather" requires --tool-call-parser to be set`.
This is an experiment launch configuration gap; tool quality remains untested.

All 64 generated token IDs match in the quality windows, with identical
conditioning for all 48 actual decode distributions in each comparison.
The full distributions are not bit-identical, including repeat controls:

| Comparison | Mean actual-decode KL | Mean decode NLL difference |
|---|---:|---:|
| Current codec repeats itself | 0.0000549694 | +0.0000051721 |
| Current codec versus palette | 0.0000640503 | +0.0001397310 |
| Palette repeats itself | 0.0000408545 | −0.0000438332 |

The unchanged repeat floor is KL ≤ 0.00001. It fails, and the candidate's
NLL difference also exceeds the relative diagnostic window. Strict
numerical qualification therefore remains failed; identical short answers
do not prove identical accuracy. Exact storage is established relative to
the MXFP4 checkpoint. It does not recover that checkpoint's prior
quantization error or establish end-to-end bitwise losslessness.

The practical outcome is a reproducible memory benefit with unchanged
answers in a small test. Costs are a custom pinned Marlin/vLLM integration,
unresolved component prefill slowdown and incomplete accuracy/serving
qualification. No default or production checkpoint is changed.
Quality is a hard acceptance gate: memory/latency benefits cannot justify
an accuracy regression. Failed numerical qualification leaves equivalence
unresolved; it is not itself proof of an accuracy decline.

Next work should isolate repeat-control nondeterminism without another
throughput sweep, clarify the interval return contract and configure tool
parsing before any future practical run. Performance work must address
the concentrated-prefill case; metadata caching and affine decoding alone
were insufficient. Sustained mixed requests and longer contexts remain
unmeasured. These are follow-ups, not completed gates.

## Locations and continuation

`ROOT` on both Sparks is:

```text
/home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/full-model-v1
```

| Evidence | Location under ROOT |
|---|---|
| Initial tile screen, partial | `scale-tiles-cpu-v1/results/screen.json` on head |
| Corrected layer-44 accounting | `scale-tiles-cpu-v2/results/screen.json` on head |
| Direct-record CPU proof | `scale-sparse-cpu-v2/results/screen.json` on both |
| Current direct CPU artifacts | `scale-sparse-cpu-v2/results/*.npz` on head |
| Failing native build | `scale-sparse-native-v3/native/` on worker |
| Guarded native build | `scale-sparse-native-v4/native/` on worker |
| Launch diagnostic | `bf16-kernels-sparsescalesv5/` |
| Guarded comparison | `bf16-kernels-sparsescalesv6/` |
| Detached controller state | `bf16-kernels-controller-sparsescalesv6.json` on head |
| Independent restoration | `restoration-sparsescalesv6.json` on head |
| Complete 42-layer byte audit | `scale-palette-all-v1/results/screen.json` on head |
| NCU reports / CSV / offline SASS | `bf16-kernels-sparsencuv1/results/` on worker; reports/counters copied to head |
| Metadata-snapshot comparison | `bf16-kernels-sparsescalesv7/`; native build `scale-sparse-native-v5/` |
| Consecutive-palette comparison | `bf16-kernels-sparsescalesv8/`; native build `scale-sparse-native-v6/` |
| Full-model source/quality diagnostic | `full-palettemodelv1-src/`, `run-palettemodelv1/results/` |
| Full-model state / restoration | `palette-model-controller-palettemodelv1.json`, `restoration-palettemodelv1.json` |
| Frozen component evidence archive | `scale-palette-evidence-v1.tar.gz` on head |

Each run has immutable `src/`; each build preserves the source hashes,
Apache-2.0 license and native library hashes in `native/manifest.json`.
Worker raw reports are in the run's `results/`; completed reports and logs
are copied to the head. Do not edit a frozen source snapshot or overwrite
previous evidence. Old hash-index experiments are a different format.

For a completed pair, copy its reports and run the CPU-only review:

```bash
python3 integrations/glm53_mxfp4/analyze_sparse_scales.py \
  --reports sparse.json sparse-confirm.json \
  --cpu-audit all-scale-audit.json --ncu-directory ncu \
  --output sparse-review.json
```

Here `ncu/` contains its own two profile JSONs and corresponding `*-ncu.csv`
files. Keep those distinct from the two unprofiled benchmark JSONs with
the same basenames. The linked review combines v6 timing and its matching
native-build NCU attribution; it does not describe later-build performance.

The component archive is 16,502,461 bytes, SHA-256
`70251f23c195d39cd5ffc2a394b915e4356cbb1b610f5010e235b794c750cb72`.
Its 686 manifest-listed payloads verify locally, including the nested worker
archive, SHA-256
`b422a0ddf031ab596e9f8b39c18bbf8ad2ed69f96e116d0fa12660a3dc5ac384`.
It preserves failed attempts, native builds, source licenses, raw timing
samples, NCU reports and offline SASS exports without model weights.

Full-model evidence is separately frozen in
`ROOT/palette-model-evidence-v1.tar.gz`: 81,323,921 bytes, 414 manifest
payloads, SHA-256
`27d541993d4ae343dc252889a2df653111701929bb10bd28cfd36e8ab8bcbd87`.
It includes both arms' full-vocabulary distributions, practical code and
sandbox scores, complete GPU checks, frozen source/native libraries,
engine/client logs and restoration receipt. Both original identities and
configuration digests independently match, health is 200, and no containers
from this run remain active (`ROOT/palettemodelv1-final-service-check.json`,
07:08 UTC). Future variants should share one service pause per planned
batch, per the user's updated operating preference in the handoff.

The existing idle guard, component lock, fixed container identities and
independent restoration watchdog remain mandatory. The user has already
authorized idle GLM pauses/restoration for these experiments. No serving
container is recreated, and no production adapter or checkpoint changes.
