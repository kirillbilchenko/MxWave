# CUDA graph coverage: C2, C8 and C16

**18 October 10:30 UTC follow-up:** the separate
[matched C8/C16 compression comparison](GLM53_HIGH_CONCURRENCY.md) is complete
and independently verified. Startup-only lease retirement fixes C16 capture
while preserving the two-slot pool and graph-owned storage. Both targets now
use captures [1,2,4,8,16], max sequences16, fixed KV and0/3 ms admission.
Compression gives short-C8+2.2–2.5%, long-C8−0.3–0.7%, short-C16−13.6–13.8%
and long-C16−7.5–7.6%; both speed gates fail. All24 KL comparisons fail, four
also fail NLL. Stock control is reused with an explicit time gap.
Both-rank C16 traces each show68 reconstructions per retained decode forward,
taking34–37 ms. The existing eight-row fused limit is now the measured next
compression issue. MTP remains deferred; no configuration is promoted.

Updated **18 October 2026, 07:05 UTC**. GPU comparison `graphexpandv1`
is complete and independently verified on Spark/Mac. Exact C2 improves
throughput by 8.6–9.0%; C8 gains 1.7–3.4%. The combined performance gate
fails, and both policies fail numerical repeats. Exact original service
identities and HTTP 200 were restored at **07:00:20 UTC**.

## Why the existing captures are [1,4]

The saved paired runs show the same native dispatch on both TP ranks:

| Requested decode batch | Captures [1,4] | Captures [1,2,4,8] |
|---|---|---|
| C1 | FULL, 1 token | FULL, 1 token |
| C2 | FULL, padded to 4 tokens | FULL, 2 tokens |
| C3 | FULL, padded to 4 tokens | FULL, padded to 4 tokens |
| C4 | FULL, 4 tokens | FULL, 4 tokens |
| C8 | Eager | FULL, 8 tokens |
| C16 with max sequences 8 | Eager | Eager |

`[1,4]` originated in the controlled C4 validation and was retained for
matched comparisons. No recorded C8-specific memory failure or correctness
rejection was found. Historical numerical failures remain open already under
`[1,4]`; they are not evidence of an additional C8-specific failure.
The explicit override was in the graph-quality launcher (`run_graph_quality.py`);
the paired-speed launcher inherited it. These experiment scripts are not
included in this branch. The underlying full-model scaffold already declares
`[1,2,4,8]` before the quality launcher's overrides.

Exact C2 therefore reduces padding work. C8 enables a graph where decode was
eager, making it the clearer launch-overhead hypothesis. This is a serving
runtime tuning mechanism that can generalize; its benefit and safe capture
sizes still depend on model, backend, memory and workload.

CUDA capture/replay is a general serving technique. Each model instance creates
its own graphs; the captured graph is not an artifact to reuse with a different
model or runtime. The reusable MxWave workflow is to inspect native dispatch,
measure capture sizes under fixed workloads, retain numerical controls and
record the resulting policy with its model/hardware/runtime identities. This
experiment validates only GLM on the pinned Spark runtime. Its adapter uses
version-bound private APIs, so it is not yet a general framework interface.
See [NVIDIA's CUDA Graph overview](https://docs.nvidia.com/dl-cuda-graph/latest/cuda-graph-basics/cuda-graph.html)
for the general launch-overhead mechanism.

A graph-policy gain on the compressed target is a serving-configuration gain.
It does not establish a larger advantage from compression: that comparison
would require stock and compressed targets using the same expanded captures.

## Native CPU proof

The CPU fixture uses the pinned native `CudaGraphManager` constructor,
candidate builder and dispatch. Pipeline discovery and graph-pool allocation
are stand-ins; no model or CUDA graph is constructed. All four configurations
pass: `[1,4]`/max8, `[1,2,4,8]`/max8, adding16/max8, and adding16/max16.
Adding capture16 while leaving max sequences8 does not make C16 eligible.

The first fixture failed safely because filtering expanded exact buckets
removed C2 instead of remapping it to C4. The repaired helper asks the native
builder to rebuild the smaller policy on private manager/config copies.
The failed fixture is preserved. Both completed archives were independently
checked locally against every retained file hash:

- CPU v1: 9 payloads; archive SHA256
  `9c0e5ac1ad2bd19129e978cafae0d4d0da80848a48b0dc39340caf1c03ef31ae`.
- CPU v2: 10 payloads; archive SHA256
  `08ddd4712d72485c265c108505279ae4adc0f9f8ddde0a46c9a8ac895903290e`.
- Passing report SHA256
  `e0c739a0c98d1870de0e753fdf2f9b6277b45ce95b1650cbb632b7aa016034c3`.

## Frozen GPU comparison

One resident compressed target captures `[1,2,4,8]`. Between completed request
groups, the worker selects the native baseline `[1,4]` mapping or the expanded
mapping. All graph allocations stay resident in both arms; target arithmetic,
transport, KV, scheduler limits and native libraries are unchanged. This tests
dispatch performance with equal allocations, not incremental graph memory
against a separate two-capture model load.

- Six cases: C1/C2/C4/C8 at P256 and C2/C8 at P2048.
- Six trials per policy/case, alternating case/policy order: 72 timed groups,
  300 requests, 19,200 generated tokens. Two 16-token warmups per policy/case.
- Ordinary HTTP requests, 64 output tokens, max sequences8, max batch2048,
  context8192, fixed 4 GiB KV/rank; no MTP or admission wait.
- After timing: 16 full-vocabulary contexts, four generated tokens,
  requested C8, two repeats/policy. Keep KL1e-5/NLL1e-4 and IDs/conditioning.
- After all timing: separate C2/C8 profiles per policy, both rank traces,
  actual descriptors and eager-forward counter deltas. Requested concurrency
  does not force identical scheduler batch formation.
- Prospective speed gate: short-C8 median TPS at least1.05×; every workload
  at least0.98×; short C1/C4 median TTFT at most1.03×. Preserve every sample.

All 160 qualified parent source files remain byte-identical. Eight new
experiment/helper modules produce a 168-file binding on both nodes. The
source archive is `graph-expansion-v1-source.tar.gz`, SHA256
`f9fe3960b3f3f6cf4a92d10b159f89c6f9054ccf53d5c17d7e6fcc6ff2817bce`.
Prepared SHA256 `d2b19f209a4094d340856e8409169ebea1276124376a2366e624a309004fb2b3`;
plan SHA256 `ce9701ab9ce0c112affad9905c4993500d4a9ed8a4bde519cd857836064d335e`.

Completed controller PID 882869, collector PID 882868. Source/results are
`ROOT/full-graphexpandv1-src` and `ROOT/run-graphexpandv1/results`;
controller `ROOT/quality-model-controller-graphexpandv1.json`. `ROOT` is the
same Spark full-model experiment directory documented in the handoff.
Local preparation, source, collector and verifier are in
`/private/tmp/mxwave-graph-expansion-v1/`.

## Measured result

All 72 groups, 300 requests and 19,200 generated tokens verify. Every one of
the 150 paired output sequences matches. Medians include all six trials per
case and policy; profiles do not contribute to these timings.

| Case | Baseline TPS | Expanded TPS | TPS change | Median TTFT, baseline → expanded |
|---|---:|---:|---:|---:|
| C1 / P256 | 16.843 | 16.758 | −0.51% | 0.406 → 0.410 s |
| C2 / P256 | 24.180 | 26.266 | +8.62% | 0.578 → 0.619 s |
| C4 / P256 | 41.221 | 40.976 | −0.60% | 0.840 → 1.133 s |
| C8 / P256 | 57.764 | 59.710 | +3.37% | 1.559 → 1.588 s |
| C2 / P2048 | 18.166 | 19.795 | +8.97% | 1.863 → 1.849 s |
| C8 / P2048 | 30.917 | 31.452 | +1.73% | 6.725 → 6.726 s |

The short-C8 gain misses the frozen 5% gate; short-C4 median TTFT increases
34.8%, exceeding its 3% guard. Both-rank counters associate C4 latency with
different admission geometry. Baseline often starts with two requests
(512 rows), while expanded usually starts with one (256 rows); the median
request changes accordingly. Forward counts alone do not explain the entire
latency difference. These observations do not establish a causal benefit
from adding an admission window. Every slow group remains in the result.

All eight native worker traces verify independently. Both C2 profiles replay
graphs. Both C8 baseline profiles have zero graph launches; both expanded C8
profiles have four. Expanded C8 records no eager eight-row forwards. Four
additional API-process traces are retained but excluded from worker-profile
classification. Graph capture succeeds on both ranks: logs report 29 seconds
and 0.76/0.62 GiB. These totals do not isolate incremental graph memory.

All 336 before/after source-scale groups pass. Each probability comparison
has all 48 real decode steps, matching conditioning and generated IDs:

| Comparison | Mean decode KL | Mean NLL difference | Strict gate |
|---|---:|---:|---|
| Baseline repeat | 8.2816e-5 | −9.2890e-5 | Fail KL |
| Expanded repeat | 3.8746e-5 | +2.8734e-6 | Fail KL |
| Baseline A → expanded A | 6.4483e-5 | −7.4892e-5 | Fail KL |
| Baseline B → expanded B | 3.9679e-5 | +2.0872e-5 | Fail KL |

The KL1e-5/NLL1e-4 floors are unchanged. Matching IDs do not qualify model
probabilities. C2 has matching timed IDs; dedicated C2 probability repeats
remain pending. Historical MLA and strict-format failures are still open.

The archive contains 466 payloads, 99,364,297 bytes, SHA256
`60eeecbd1e50bde9787e97319f01c65f47a21a739f8b0b515c560bc6b07d67d3`.
The first independent verifier counted API traces as worker profiles; the
original review is preserved. V3 classifies only native worker traces and
passes on both systems. Mac probability recomputation differs from Spark
by at most 1.9985e-15. The verifier allows 1e-12 CPU recomputation roundoff
only for probability fields; all hashes, timings, IDs and gate decisions
agree exactly. This does not relax the model's numerical floors. Reviews
therefore agree with explicit roundoff metadata, rather than byte equality.

[Compact result, every trial's C4 geometry and both-rank trace checks](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/graph-expansion-v1-review.json)
and [native CPU proof](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/graph-expansion-cpu-v2-review.json)
retain the decision and evidence bindings. No original-service configuration
was changed or promoted.

## C16 remains a separate decision

C16 needs max sequences16 and cache-capacity validation. It also exceeds
the current KDA fused limit of eight rows: a 16-row decode would reconstruct
the 68 BF16 matrices and run the larger-row path. Capturing that path removes
launch work but retains reconstruction traffic. CPU eligibility is not GPU
memory, numerical or throughput qualification. Measure C8 first, then test
C16 with its additional scope explicitly recorded.

Admission merging, actual MLA internal-state diagnosis and the corrected MTP
geometry preflight remain pending. Exact C2 is a measured lead requiring
dedicated numerical/latency qualification. C8 graphs are functional but do not
provide the proposed large speed unlock in this test. See
[current state](GLM53_CURRENT_STATE.md).
