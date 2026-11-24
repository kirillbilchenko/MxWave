# GLM-5.3-Flash: reuse existing MXFP4 before considering another quantization

Checked 2026-10-15. Preparation worktree: `feat/glm53-flash-mxfp4`.
The header audit, selected-weight comparison, scale screen and GPU components
finished on Spark2. The full layer-23 component covers all 288 experts, with
both TP2 weight partitions tested sequentially. The original record decoder
was bit-exact but regressed by 5–18% under concentrated routing. A later exact
aligned-nibble decoder passes the focused rank-0 check: about 2.3% lower
single-token latency and at most 1.13% higher concentrated-routing latency,
while retaining the 50% scale-storage saving. Its broader two-partition
component confirmation was interrupted in all six attempts. The subsequent
whole-model TP2 pilot completed: **4.30 GiB less allocated CUDA memory than stock
globally** (4.37 GiB versus the raw adapter control),
but no qualified speed or model-quality result. That initial stock execution produces
bad output, and cold mHC compilations contaminate one timing case. The same
original GLM containers were restored and returned health 200. See the
[completed pilot and next steps](#completed-whole-model-pilot-and-trace-findings).

Follow-up diagnostics confirm 1,316 exact selected loaded-tensor comparisons
across both ranks and real-layer mHC warmup. The graph/eager switch passes a
tiny-engine test on the actual pinned runner, observing 3/0/3 graph replays.
The initial corrected-harness stock comparison completed: **0/9 sanity checks passed**
across graph/eager/graph phases, with compression disabled. The restored NVFP4
deployment passes the same three prompts. These are different quantized
checkpoints; quality remains unqualified for the imported MXFP4 execution path.
See [follow-up evidence](#follow-up-components-and-one-load-diagnostic).

A subsequent CPU audit finds a concrete runtime compatibility defect: all 45
fused attention input projections incorrectly select MXFP8 even though their
checkpoint weights are BF16 with no scales. The opt-in mapping repair now passes
both the CPU selector audit and selected loaded-attention checks on both GPUs.
The corrected stock emits coherent reasoning and diagnostic NLL falls from about
12.09 to 0.745. Its short chat budget still truncates final answers, so this is
not a completed quality qualification. Three real BF16 projection components
also pass exact GPU decoding with 28.76–29.18% retained-weight savings. Their
single-token graph timings improve, but large prefill timings regress. See the
[fused-attention finding](#fused-attention-dispatch-defect).

## Decision

Start compatibility screening with
[INCModel3/GLM-5.3-Flash-MXFP4-Mixed-CT-AutoRound](https://huggingface.co/INCModel3/GLM-5.3-Flash-MXFP4-Mixed-CT-AutoRound).
It is the closest existing storage format to our vLLM integration. This is a
practical integration choice; the published evaluations do not establish a
quality winner among all MXFP4 and NVFP4 releases. The numerical check below
also finds byte-exact agreement with MxWave's default MSE on the selected expert
rows, giving no reconstruction benefit from repeating that quantization there.

Reuse the upstream checkpoint and its original quantization metadata. Add a
lossless storage/runtime adapter rather than passing those weights through the
MxWave quantizer again. Preserve upstream author, source, license, and recipe
attribution. A storage wrapper is not a new MxWave quantization recipe.

The current quantization CLI deliberately rejects quantized input. Importing
this mixed checkpoint requires a separate validated path; changing its
`quant_method` string alone is not a valid conversion.

## Verified source layout

The public API, config, checkpoint index and all 120 safetensors headers were
audited; a separate comparison read selected weight rows. Evidence is stored under
`benchmarks/glm53-flash-hf-audit-2026-10-15/`; `sources.json` pins the reviewed
revisions and downloaded metadata hashes.

| Property | AutoRound source |
| --- | --- |
| Revision | `35d1b0d5cb4a0db075fd2749c897e970dc15bfa5` |
| Shards | 120 |
| Total safetensors file bytes, including headers | 183,429,201,056 (183.43 GB) |
| Quantization method / container format | `compressed-tensors` / `mixed-precision` |
| Expert weights | `mxfp4-pack-quantized`, group 32, packed uint8 nibbles and uint8 E8M0 scales |
| Expert targets | 37,152, including NextN layer 45 |
| Other quantized linears | `mxfp8-quantized`, group 32, FP8 values and uint8 E8M0 scales |
| Declared activation schemes | Dynamic MXFP4 for experts; dynamic MXFP8 for other quantized linears |
| Observed head and embedding headers | BF16, 1,268,776,960 bytes each |

For example, a down projection has packed shape `[4096, 1024]` and scale shape
`[4096, 64]`; gate/up have packed shape `[2048, 2048]` and scale shape
`[2048, 128]`. These match canonical MXFP4 dimensions. All-shard index ownership,
expert coverage, tensor intervals, shapes and scale dtypes passed. Full payload
hashes and actual inference-runtime arithmetic have not been verified.

The source index advertises `metadata.total_size: 0`. Do not use that field for
sizing or copy it into derived metadata. The public API's per-file sizes give
the total above. The completed header inventory recomputes 183,418,338,168
tensor bytes across 76,060 tensors; the difference is shard headers.

The original BF16 checkpoint contains 642,646,653,816 tensor bytes. Reusing this
183.43 GB source avoids that BF16 download and a complete quantization pass.
This is not an estimate of additional runtime compression savings.

## Existing reported quality

These are author-reported results, not measurements reproduced on our Sparks.
Rows with different evaluation protocols cannot be ranked against one another.

| Source | Reported evidence | Practical interpretation |
| --- | --- | --- |
| [NVIDIA NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) | BF16 → NVFP4: GPQA 92.17% → 92.11%; IFBench 61.30% → 60.54% | Strong existing reference; generation limits reach 327,680 tokens for these tasks. |
| [AMD Quark MXFP4](https://huggingface.co/amd/GLM-5.3-Flash-Quark-MXFP4) | GSM8K 1,282/1,319 (97.19%) on TP4; 97.12% on TP8/EP8 | Full GSM8K runs, 32,768-token cap and no truncations. No universal PPL/KL target is established. |
| [AutoRound mixed MXFP4](https://huggingface.co/INCModel3/GLM-5.3-Flash-MXFP4-Mixed-CT-AutoRound) | BF16 → mixed: GSM8K 97.35% → 97.35%, MMLU 86.66% → 85.81%; four-task average 83.99% → 83.66% | RTN recipe and closest storage format. The card lacks enough evaluator detail to set a precise equivalence margin. |
| [OneNexus MXFP4](https://huggingface.co/OneNexus/GLM-5.3-Flash-MXFP4) | Matched subsets: GSM8K 98.0% vs AMD 98.2%; MMLU 85.6% vs 84.6%; GPQA 70.2% vs 66.7% | Differences are not statistically established; 55/198 and 61/198 GPQA answers are truncated. |
| [Four-Spark NVFP4 community release](https://huggingface.co/shankinsonhf/GLM-5.3-Flash-NVFP4-W4A4-Experts) | Fixed 400-item MMLU/ScienceQA screen: official FP8 89.25%, W4A16 88.5%, W4A4 89.0% | Relevant GB10 evidence, but TP4 native NVFP4 with compatibility patches; not a two-Spark prediction. |

OneNexus publishes a
[paired evaluation protocol](https://huggingface.co/OneNexus/GLM-5.3-Flash-MXFP4/blob/880fe3b301de033fb497c412af394a5bb976098f/reproduction/quality-protocol.md)
and aggregate outcomes. It uses 8,192 output tokens for GSM8K/MMLU and 16,384
for GPQA, greedy decoding, BF16 KV and no speculation. The GPQA cap-hit rates
mean its lower score cannot be compared directly with NVIDIA's long-budget
score as a measure of MXFP4 versus NVFP4 error. Completed-only accuracy must
not replace all-example accuracy; it conditions on a model-dependent subset.

Search snippets described an older OneNexus mixed recipe with protected BF16
expert layers. Its reviewed current config has no routed-expert exclusions
outside NextN and its payload is 195,550,624,632 bytes. Do not plan from the
stale 227 GB snippet. The current quality artifact identifies its own evaluated
weight revision, separately from the model-card revision.

AMD is another useful reference: its payload is 185,077,182,400 bytes, with
FP8 side layers and a Quark container. OneNexus uses BF16 side layers and the
same group-32 uint8 expert scales in sampled headers. Their `.weight` tensor
names differ from compressed-tensors' `.weight_packed`. Any importer must
verify nibble order, scales, per-layer precision and complete coverage before
claiming lossless repacking.

## Execution gates

1. **Completed: compatibility and CPU scale-codec screen.** Full header
   coverage, the selected numerical comparison and the complete-scale probe
   below passed. The fast codec saves 50% after the actual Marlin permutation
   in both TP1 and TP2 partitions. This result is separate from GPU latency.
2. **Grouped MoE and full 288-expert layer measured; performance work remains.**
   The results below cover real gate/up/down matrices, synthetic top-8 routing,
   empty experts, both TP2 partitions, non-default streams and CUDA graphs.
   Resolve the concentrated-routing regression before promoting the adapter.
   Actual inter-rank execution and inference-loader integration also remain.
   Compare identical AutoRound weights with
   compression disabled; do not use NVIDIA weights as the lossless control.
3. **Preserve the mixed checkpoint.** Retain the MXFP8 side-layer groups,
   BF16 head/embedding, ignored modules and NextN handling. The generic
   `Linear` group must not override explicitly targeted expert schemes.
4. **One whole-model confirmation after the component passes.** Compare
   retained memory, startup peak, throughput, prompt likelihoods and matched
   next-token distributions with the identical uncompressed source. Reuse
   immutable controls when unchanged; do not repeat them for every codec edit.
5. **Compare upstream quantizations separately.** A new MXFP4-versus-NVFP4
   quality/performance comparison uses matched prompts, tokenizer, reasoning
   mode, output limits, scoring, KV policy and disclosed activation precision.
   It does not validate lossless storage by itself.

The current live GLM uses explicit Marlin backends. Existing native NVFP4 W4A4
examples show a separate compute opportunity, not a feature proved lost by
changing this deployment's current format. A source declaring W4A4 can execute
as W4A16 through Marlin; exact stored weights do not guarantee identical
probabilities across those arithmetic paths. Record actual selected kernels
and qualify quality rather than silently changing or misdescribing schemes.

## How to choose numbers economically

There is no transferable published GLM KL target in the reviewed MXFP4 cards.
Do not reuse the old Kolibri 0.030 threshold or choose one after seeing a run.
For lossless storage, bytes must be restored exactly; compare output drift
against an unchanged-runtime repeatability control instead of allowing a
fresh quantization error budget. Use held-out prompt NLL and full-vocabulary KL
to catch loader, decoder and arithmetic regressions.

For any later quantization recipe, predeclare calibration, development and
final held-out sets. Save natural-routing activations and expert hit counts
once, screen a small fixed set of scale recipes by activation-weighted expert
output error, and evaluate only the surviving complete candidate. Neither a
good weight SQNR nor a high GSM8K score establishes general model quality.
Record coding/reasoning outcomes and truncations separately; two-Spark runs
must not adopt a low token cap merely to reproduce a convenient headline.

The coverage, selected-weight, CPU scale-codec and GPU MoE measurements are
complete, including a full 288-expert layer. The original record decoder missed
its preferred latency gate; the aligned decoder and full-model pilot followed.
Both checkpoint copies and engine integration are complete. Whole-model quality
remains unqualified. A new BF16 quantization's time estimate does not describe
this reuse route.

## Completed coverage and numerical checks — 2026-10-15

The [full audit](../benchmarks/glm53-flash-hf-audit-2026-10-15/INCModel3--GLM-5.3-Flash-MXFP4-Mixed-CT-AutoRound/full-audit-report.json)
validated all 120 headers, exact expert coverage and ownership, all tensor
byte intervals, and every quantized matrix's scale shape/dtype. No unowned
scales or unclassified floating weight matrices were found. Generic `Linear`
selector dispatch and fused aliases still require an inference-loader test.

| Component | Matrices | Upstream storage | Proposed experts-only BF16-source recipe |
| --- | ---: | --- | --- |
| Base routed experts, layers 3–44 | 36,288 | MXFP4 | Same selected modules and representation |
| MTP routed experts, layer 45 | 864 | MXFP4 | Preserve BF16 |
| Leading dense MLP, layers 0–2 | 9 | MXFP8 | Preserve BF16 |
| Base shared experts | 126 | MXFP8 | Preserve BF16 |
| MTP shared experts | 3 | MXFP8 | Preserve BF16 |

All quantized scales are uint8 E8M0 with group size 32. The 138 MXFP8 matrices
use 1,535,115,264 weight bytes and 47,972,352 scale bytes. Keeping them in BF16
would add 1,487,142,912 bytes; keeping MTP routed experts in BF16 adds another
10,645,143,552 bytes. These differences exactly explain the gap to the earlier
195,550,624,632-byte estimate. MTP file storage is not necessarily resident
target storage when using the separate DFlash drafter.

Attention, indexers, routers, norms, vision weights, head and embedding have
BF16 weighted-module headers. The full source has 9,787,146,240 scale bytes,
including MTP and MXFP8. These are checkpoint bytes, not measured GPU savings.

The [numerical summary](../benchmarks/glm53-flash-hf-audit-2026-10-15/INCModel3--GLM-5.3-Flash-MXFP4-Mixed-CT-AutoRound/row-summary.json)
covers 12 expert matrix slices from layers 3, 23, 44 and 45, all three
projections and 32 rows each. It compares 1,310,720 FP4 values, 655,360 packed
bytes and 40,960 E8M0 scale bytes against fresh MxWave output from original
BF16 slices. No real activation statistics were supplied.

| Method | Packed-byte agreement with upstream | Scale-byte agreement | Base-expert pooled SQNR |
| --- | ---: | ---: | ---: |
| Upstream AutoRound | reference | reference | 18.984964 dB |
| MxWave default unweighted MSE | **100%** | **100%** | 18.984964 dB |
| MxWave RTN control | 92.9247% | 92.7563% | 18.893237 dB |

Byte agreement includes MTP; SQNR pools only the nine base-expert slices. MSE
used unchanged defaults: percentile 99.5, clip depth 1, no gamma/Hessian.
The card's RTN label therefore does not imply MxWave RTN's scale-selection
rule. The exact sampled match does not establish whole-checkpoint equality
or identify the upstream implementation as MxWave's algorithm.

Nine additional MXFP8 slices cover dense, shared and MTP-shared projections.
Their pooled SQNR is 32.1854 dB and relative L2 error is 2.4588% against BF16;
they would differ from an experts-only recipe's BF16 passthrough. All three
BF16 controls (head, embedding and one attention projection) are byte-identical
to the official BF16 reference.

The BF16 reference revision is `f9fed3bbc82291073373d67452e9f25317ef7577`;
the upstream card does not identify its original BF16 source revision. The
unchanged MxWave core is revision `mxfp4-rne-v2`, SHA-256
`3531ce8234c4051507c0479d5c5b363249bedb8a5bc5ad007f45ac6116a9083a`.
This was a sparse, deterministic sample, not a random statistical estimate.
No activation replay, perplexity, KL, task accuracy or inference speed was
measured. The CPU comparison peaked at about 552 MB RSS in a container capped
at 2 GB and two CPUs, with no GPU device requests.

Raw evidence on Spark2:
`/home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/evidence-v1.tar.gz`.
The 7,924,481-byte archive has SHA-256
`ccbd705b689fb2023719f22e3ede31ff1f2ef9d1418da45884dff46e128f10a7`.
Local copy: `/private/tmp/mxwave-glm53-audit-evidence-v1.tar.gz`. All 120 cached
header hashes, all 69 sampled payload hashes and both copied reports were
verified. GLM remained on its original containers/start times; health was 200.
Ruff, strict mypy (33 core files) and pytest (328 passed, one skipped) passed.

## Completed base-expert scale screen — 2026-10-15

The [CPU scale report](../benchmarks/glm53-flash-hf-audit-2026-10-15/INCModel3--GLM-5.3-Flash-MXFP4-Mixed-CT-AutoRound/scale-probe.json)
uses nine complete scale matrices: gate/up/down of layer 3 expert 0, layer 23
expert 143 and layer 44 expert 287. It downloads only 2,359,296 payload bytes.
The installed, SHA-256-pinned vLLM layout functions execute on CPU, including
gate/up fusion, the column/row TP partitions, scale permutation and E8M0
conversion for BF16 activations. Converted scale bits match the expected
permutation exactly. Both codecs reconstruct every byte in all 18 layout cases.

| CPU scale storage | TP1 | TP2, both ranks combined |
| --- | ---: | ---: |
| Uncompressed bytes | 2,359,296 | 2,359,296 |
| Fast records plus exact exceptions | 1,179,648 | 1,179,648 |
| Fast saving | **50.00%** | **50.00%** |
| Range codec, including its serialized metadata | 599,808 | 600,000 |
| Range saving | 74.58% | 74.57% |

There are no fast-codec exceptions in these samples. Descriptor bytes and GPU
allocator rounding are excluded. Three deterministic expert samples do not
establish the ratio on all experts. If 50% holds across the base experts'
9,512,681,472 scale bytes, the saving would be **4,756,340,736 bytes**
(4.76 GB / 4.43 GiB), approximately 2.21 GiB per rank with balanced TP2
ownership. This remains a projection, not measured resident GPU memory.

The range codec's higher compression does not select it for deployment:
earlier dense-kernel profiling exposed its latency cost. Start the new MoE
component with the fast asynchronous records. The CPU probe peaked at
546,328,576 bytes RSS and had no GPU device access. It does not establish the
declared W4A4 checkpoint's arithmetic behavior under another activation backend.

## Runtime component protocol

Base routed experts are the first scope. MTP, MXFP8 side linears and DFlash
remain independent components. Preserve their upstream weights and schemes
when assembling a later whole model.

1. Stage a hash-pinned grouped Marlin MoE kernel beside the existing dense
   adapter; retain the dependency's license and record any adapted code in
   `CONTRIBUTING.md`. The existing `NativeMarlin.gemm` bridge only handles
   dense GEMM and cannot stand in for expert dispatch.
2. Build one bounded fixture from complete base-expert matrices, retaining
   GLM's hidden/intermediate sizes and top-8 routing. Include both gate/up
   and down GEMMs, uneven and empty experts, and each TP2 partition. A reduced
   expert-count fixture must be labeled as such; it is not the full 288-expert
   routing workload.
3. Compare identical BF16-activation Marlin arithmetic with raw scales and
   compressed scales. Test exact decoder bytes first, then GEMM output against
   the raw control and its repeatability. Cover eager, non-default streams and
   captured graph replay. Do not compare different quantization sources as a
   lossless control.
4. Measure retained allocation, startup peak, decode/prefill latency and a
   kernel trace. Cache the unchanged raw component result for subsequent edits.
   The Qwen dense control cannot validate a new grouped MoE workload. Prefer
   a median latency regression below 2%, confirmed in both measurement orders;
   do not promote a larger regression without a documented tradeoff.
5. Run only in a guarded idle window on Spark2, with at most 1 GiB of experiment
   CUDA allocations and a 2 GiB container cap for the 16-expert fixture.
   The subsequent 288-expert fixture uses streaming weight loading and a
   chunked CPU scale packer, with a 2.25 GiB Torch CUDA cap, 3 GiB container cap
   and 5.5 GiB minimum available memory. The controller must cancel its
   own experiment if GLM becomes active. Do not stop GLM or download the entire
   AutoRound checkpoint for this component. Full-layer/TP2 and whole-model
   tests follow only after this bounded gate passes.

The completed components below apply this protocol. Their qualification jobs
and the earlier CPU probes have exited.

## Completed grouped MoE component — 2026-10-15

The [runtime report](../benchmarks/glm53-flash-hf-audit-2026-10-15/grouped-moe-v1/report.json)
and [compact summary](../benchmarks/glm53-flash-hf-audit-2026-10-15/grouped-moe-v1/summary.json)
use 16 real layer-23 experts (96 complete code/scale tensors, 213,909,504
downloaded bytes). Hidden size 4096 and intermediate size 2048 remain intact;
TP2 partitions use intermediate size 1024. All runs execute on Spark2/GB10
SM121 beside idle GLM. TP2 partitions run sequentially on one GPU, without
inter-rank communication. This is not a full 288-expert or whole-model test.

The isolated Apache-2.0 grouped Marlin dependency is pinned to vLLM revision
`db9527a46873454610df6dbedf79a36d6bf1a7f6` in
[the build manifest](../benchmarks/glm53-flash-hf-audit-2026-10-15/grouped-moe-v1/build-manifest.json).
Only scale loads/expert pointer advancement and the component C ABI change.
Both arms use the same tile plan, BF16 activations and non-atomic FP32 partial
reduction. This does not test native W4A4 arithmetic. The installed vLLM engine
and serving containers are untouched; the C ABI is not an inference plugin.

**Correctness:** all 24 real-weight cases pass (12 TP1, six per TP2 rank).
They cover varied top-8 assignments and concentrated routing that leaves eight
experts empty, eager execution, non-default streams and captured graph replay.
Every GPU scale byte reconstructs exactly; the independent decoder also covers
all 256 byte values and global exception indexes. A separate injected-exception
GEMM check passes. Native managed locks return to zero. Matching sorted routing
produces bit-identical gate/up, down and final outputs against stock vLLM.
For newly run cases, real rerouting and graph outputs also match stock GEMMs
when the reference consumes that run's actual sorted alignment.

An initial M=128 equality failure exposed an evaluation issue: the unchanged
router changes token order within experts. In the
[focused diagnostic](../benchmarks/glm53-flash-hf-audit-2026-10-15/grouped-moe-v1/router-order-diagnostic.json),
955 sorted entries differed between raw and compressed runs. This slightly
changes the floating-point reduction path. Reusing identical sorted routing
made **all** stock/raw/compressed output differences zero, without changing the
kernel or adding an error tolerance. Correctness now compares the same sorted
alignment; paired latency still executes the real router. Ten already completed
graph comparisons are reused with source/report hashes, rather than repeated.

**Retained scale storage:**

| Partition | Raw scale bytes | Records + exceptions + descriptors | Reduction |
| --- | ---: | ---: | ---: |
| TP1, 16 experts | 12,582,912 | 6,291,488 | 50% less 32 descriptor bytes |
| TP2 rank 0 | 6,291,456 | 3,145,760 | 50% less 32 descriptor bytes |
| TP2 rank 1 | 6,291,456 | 3,145,760 | 50% less 32 descriptor bytes |

There are zero exceptions in the real scale samples. Peak CUDA allocated memory
is 367,045,632 bytes (350.04 MiB), reserved 387,973,120 bytes (370 MiB), including
both comparison arms and scratch. The tensor byte reduction is measured; the
peak is not a separate whole-model before/after memory comparison. The earlier
approximately 4.76 GB base-expert projection remains a projection across all
layers/experts.

**Graph replay latency:** ranges below are the observed percentage change in
median latency in both measurement orders (seven trials, 20 calls per trial).
Negative values mean lower latency. Timed work includes routing, both grouped
GEMMs, SwiGLU and the top-k sum. Earlier eager values remain labeled as prior
harness evidence because the comparator changed.

| Partition and routing | M=1 | M=16 | M=64 | M=128 |
| --- | ---: | ---: | ---: | ---: |
| TP1, varied | -1.84% to -1.69% | -3.98% to -3.35% | -3.82% to -2.58% | -0.73% to -0.61% |
| TP1, concentrated | -2.40% to -2.28% | -2.04% to -1.47% | **+2.53% to +2.80%** | **+2.06%** |
| TP2 rank 0, varied | -2.50% to -1.78% | -2.60% to -2.50% | -1.61% to -0.89% | not tested |
| TP2 rank 1, varied | -2.64% to -2.43% | -2.75% to -2.68% | -1.57% to -1.28% | not tested |
| TP2 rank 0, concentrated | -2.61% to -2.44% | -2.43% to -1.94% | +1.52% to +1.54% | not tested |
| TP2 rank 1, concentrated | -2.48% to -2.45% | -3.42% to -1.95% | +1.41% to +1.42% | not tested |

Absolute timings vary between orders; use the paired comparisons rather than
unpaired trace durations as the performance evidence. These results support
continuing the **TP2** path. They do not establish a universal speedup: the TP1
concentrated M=64/128 cases exceed the preferred 2% regression gate and remain
unqualified for that gate. No full-model tokens/sec figure follows from them.

**Traces:** four hash-bound torch traces cover TP1 M=1 and M=128 for both arms.
They show two Marlin launches per block and no standalone scale-decompression
launch. Grouped GEMMs account for approximately 95.6–96.2% of traced GPU kernel
time; the remaining routing/activation/sum operations account for the rest.
Unpaired profiler durations do not measure the speedup or establish DRAM
saturation. No Nsight Compute bandwidth/occupancy measurements were collected
for this new MoE component.

The [idle guard](../benchmarks/glm53-flash-hf-audit-2026-10-15/grouped-moe-v1/guard-controller.json)
accepted the final run with unchanged request/token counters and exit 0.
[The reused cases' guard](../benchmarks/glm53-flash-hf-audit-2026-10-15/grouped-moe-v1/reused-cases-guard.json)
also has unchanged counters; its later exit 1 was the resolved alignment check,
not GLM activity. Both serving ranks retain their original container IDs/start
times. The component ran under the planned 2 GiB container and 1 GiB Torch CUDA
allocation caps and has exited.

Evidence archive on Spark2:
`/home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/grouped-moe-v1/evidence-v1.tar.gz`.
It contains reports, traces, exact executed sources, the fixture manifest and
native libraries, excluding the weight payloads. Size 2,644,598 bytes; SHA-256
`d3ce69ab4d2f4acc344a87898a584b5b3c552507fa2333ab266c7e61f01e12f9`.
Local copy: `/private/tmp/mxwave-glm53-moe-evidence-v1.tar.gz`.

## Complete 288-expert layer — 2026-10-15

The [full-layer report](../benchmarks/glm53-flash-hf-audit-2026-10-15/grouped-moe-full-layer-v1/report.json)
and [summary](../benchmarks/glm53-flash-hf-audit-2026-10-15/grouped-moe-full-layer-v1/summary.json)
cover every gate/up/down matrix of layer 23: **1,728 code/scale tensors,
3,850,371,072 downloaded bytes**. They reuse the same native libraries as the
16-expert component. Both TP2 partitions run sequentially on Spark2, without
collectives. Routing is synthetic, with varied top-8 assignments and an extreme
concentrated case that sends every token to the same eight experts. These are
not routing statistics from a real model forward.

**Correctness:** all 20 cases pass, covering M=1/16/64/128/512, both routing
patterns and both TP2 partitions. Every real scale byte reconstructs exactly.
Stock/raw/compressed GEMMs and final outputs are bit-identical for identical
sorted alignment; independently rerouted eager, stream and graph outputs match
the stock reference using their own alignment. The synthetic decoder check
also passes all 256 byte patterns and 65,536 escape vectors across the CPU
packer's 1 MiB chunk boundary. The chunked packer rebases exception indexes
globally and bounds temporary CPU expansion.

| Partition | Raw scale bytes | Compressed bytes, including descriptors | Bytes saved |
| --- | ---: | ---: | ---: |
| TP2 rank 0 | 113,246,208 | 56,623,136 | 56,623,072 |
| TP2 rank 1 | 113,246,208 | 56,623,136 | 56,623,072 |
| Both partitions | 226,492,416 | 113,246,272 | **113,246,144** |

Real scales have no exceptions. This is 50% of scale storage, less descriptor
overhead; codes remain unchanged. It saves about **2.94% of this layer's total
code-plus-scale payload**, not 50% of the model. The earlier 4.76 GB / 4.43 GiB
base-expert saving remains a projection over all 42 routed layers, supported
by this complete layer and the earlier three-layer scale sample. It is not a
whole-model resident-memory measurement.

Peak Torch CUDA allocation was 2,269,271,552 bytes (2.11 GiB), reserved
2,321,547,264 bytes (2.16 GiB), including both comparison arms and scratch.
The job completed under the 3 GiB container and 2.25 GiB allocation caps.
These comparison peaks are not compressed-only resident memory.

**Graph replay latency:** percentage change in the median, both arm orders,
seven trials of 20 calls. Negative values mean lower latency. Both arms use
the same source weights, routing policy, tile plans and BF16 activations.

| Partition and routing | M=1 | M=16 | M=64 | M=128 | M=512 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rank 0, varied | -4.75% to -2.17% | -2.71% to -2.46% | -3.51% to -2.79% | -3.16% to -2.94% | -3.90% to -2.69% |
| Rank 1, varied | -2.21% to -0.55% | -3.13% to -1.95% | -3.82% to **+5.54%** | -4.12% to -2.09% | -2.96% to -2.75% |
| Rank 0, concentrated | -1.92% to -0.12% | -0.46% to +0.23% | **+10.10% to +10.50%** | **+12.07% to +17.63%** | **+5.78% to +7.32%** |
| Rank 1, concentrated | -4.52% to -1.32% | -2.27% to -1.87% | **+8.48% to +10.00%** | **+9.88% to +11.41%** | **+5.29% to +6.77%** |

Most varied-routing cases improve by roughly 2–4%, but rank 1 M=64 changes sign
between orders and does not establish a stable speedup. Absolute timings also
drift between orders. The concentrated M>=64 regressions are consistent in
both orders and ranks and exceed the preferred 2% gate. The full-layer result
therefore does **not** qualify this implementation for deployment. Do not carry
forward the smaller fixture's favorable TP2 result as the current verdict.

Four hash-bound Torch traces cover rank 0 varied routing at M=1 and M=64.
They contain two grouped Marlin launches per forward and no standalone scale
decode launch. Grouped GEMMs account for about 96.3% and 99.5% of traced kernel
time respectively. Raw/compressed launches use 92/94 registers per thread,
32 KiB shared memory and the same estimated occupancy (25%, shared-memory
limited). These are launch-resource estimates, not Nsight counter measurements.
The initial traces cover varied routing; the separate concentrated-routing
collection below narrows the regression without repeating the latency suite.

The [guard record](../benchmarks/glm53-flash-hf-audit-2026-10-15/grouped-moe-full-layer-v1/guard-controller.json)
accepted the run with unchanged request/token counters and exit 0. The
[worker verification](../benchmarks/glm53-flash-hf-audit-2026-10-15/grouped-moe-full-layer-v1/worker-verification.json)
confirms no OOM and an exited experiment. Both GLM container IDs and start times
are unchanged, and head health is 200.

Frozen evidence on Spark2:
`/home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/grouped-moe-full-layer-v1/evidence-v1.tar.gz`.
Size 2,827,012 bytes, 22 members, SHA-256
`3da7ecdd8635dcdd739f028f9eb21e20cd6c59306e246a71d6ada96c19b70e19`.
Local copy: `/private/tmp/mxwave-glm53-moe-full-layer-evidence-v1.tar.gz`.
The fixture manifest, executed benchmark/runtime, libraries and traces match
the report hashes. Weight payloads are excluded from the archive.

### Focused concentrated-routing traces

The [focused report](../benchmarks/glm53-flash-hf-audit-2026-10-15/grouped-moe-full-layer-v1/concentrated-profile.json)
covers rank 0 at M=64 and M=128, with all tokens routed to the same eight
experts. Both cases pass the same eager/stream/graph bit checks. No new paired
latency baseline was run; only the four missing Torch traces were collected.

Both grouped GEMMs show longer durations in the compressed traces. The
following medians are **unpaired profiler observations**, useful for locating
the work, not replacements for the paired regression percentages above:

| Tokens | Gate/up raw → compressed | Down raw → compressed |
| --- | ---: | ---: |
| 64 | 238.5 → 289.2 µs | 128.4 → 158.7 µs |
| 128 | 439.8 → 506.2 µs | 245.5 → 280.4 µs |

All four traces retain the same kernel count, grid, 128-thread blocks,
32 KiB shared memory and estimated 25% occupancy. Registers remain 92 versus
94. There is no extra decode launch or changed occupancy threshold explaining
the regression. The next target is scale loading/expansion inside both GEMMs,
not a Python launch bottleneck. Repeated per-fragment expansion is a candidate
cost, **not a hardware-counter-confirmed cause**. Both tested workloads use an
eight-token MoE block despite many tokens reaching each active expert; future
experiments should examine decode reuse and tile choice while preserving the
same exact-output comparator.

GLM activity interrupted the first attempt; the controller stopped only its
own container. The second attempt completed with unchanged activity counters,
exit 0 and no OOM. Both GLM ranks retain their original IDs/start times, and
head health remains 200. The trace controller and all its GPU jobs have exited.

Focused evidence on Spark2:
`/home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/grouped-moe-full-layer-v1/concentrated-profile-evidence-v1.tar.gz`.
Size 162,385 bytes, 12 members, SHA-256
`87b78b43b59bf9716419a0721775ae70f248cd63da2d0534d4d630abd3040385`.
Local copy: `/private/tmp/mxwave-glm53-concentrated-profile-evidence-v1.tar.gz`.
The earlier full-layer archive supplies the identical native libraries.

### Decoder optimization and hardware counters

The focused optimization keeps the original raw library and source weights,
testing rank 0 at M=1 with varied routing and M=64/128/512 with concentrated
routing. Both arm orders use seven trials of 20 graph replays. Each variant
passes byte reconstruction and matched stock GEMM/output checks, including
streams and graphs. These are component measurements with BF16 activations,
not full-model tokens/sec or the checkpoint's declared dynamic W4A4 execution.

| Decoder change | M=128 concentrated latency change, both orders |
| --- | ---: |
| Byte-permutation record decode | +6.57% to +10.96% |
| Decode once before the shared-memory barrier | +7.36% to +7.50% |
| Decode earlier during prefetch | +22.38% to +22.51% |
| Decode directly into shared BF16 fragments | +8.37% to +13.54% |
| Shared decode with a 64-token prefill tile | +7.63% to +8.79% |
| Tensor-base nibble offsets, additive reconstruction | +3.48% to +4.71% |
| **Aligned-base nibble offsets** | **+1.09% to +1.10%** |

The 64-token experiment changes the tile for both arms; it is not a speed
comparison against the default eight-token raw kernel. Its 255 registers and
99 KiB shared allocation also make it a poor default. The other variants keep
the raw tile plan. No additional standalone decode kernel is introduced.

Nsight Compute 2025.3.1 profiled the raw and shared-stage kernels at M=128,
gate/up, rank 0. Only the eight active experts are resident for profiling;
the routing metadata still has 288 experts and the original launch plan.
Each five-pass application replay passes the raw-reference bit check.
Raw → shared-stage counters show warp instructions +4.99%, integer thread
instructions +2.50%, long-scoreboard stalls 14.17% → 6.81%, math-pipe stalls
7.10% → 8.51% and barrier stalls 17.17% → 20.97%. This supports targeting
decoder instruction/barrier costs. Stall percentages are sampled activity,
not additive fractions of forward latency. The requested DRAM-read metric
was unavailable; these results do not establish achieved memory bandwidth.
Nsight's replay durations are diagnostic, not paired latency measurements.

All 864 real scale matrices in this layer have byte values within 118–122.
The newer layout packs exact four-bit offsets under a single base per scale
tensor, eliminating per-record base headers and exception branches. For this
layer the aligned base is 112; offsets occupy disjoint exponent bit fields,
allowing direct reconstruction of the stock BF16 fragments with shifts and
bitwise selection. The MXFP4 weight codes and GEMM arithmetic stay identical.
Packing fails closed if values cross the selected 16-value bucket. Other
layers must be screened; this fast layout is not assumed universally valid.

The focused aligned result is -2.40% to -2.29% at M=1, -1.83% to +0.90% at
M=64, +1.09% to +1.10% at M=128 and +1.00% to +1.13% at M=512. Absolute
M=512 times drift substantially, so no stable throughput gain is established.
The accepted focused attempt completed with unchanged GLM counters after four
attempts were interrupted by new serving activity. Interrupted measurements
are excluded. The subsequent 20-case, two-partition confirmation was interrupted
six times; its controller exited with no accepted run. No GPU experiment is
left running. This is an uncompleted qualification gate, not a decoder failure
or a passing performance result.
The [confirmation outcome](../benchmarks/glm53-flash-hf-audit-2026-10-15/grouped-moe-optimization-v1/confirmation-outcome.json)
records all six interruptions. Its head health check returned 200 and the
[worker check](../benchmarks/glm53-flash-hf-audit-2026-10-15/grouped-moe-optimization-v1/confirmation-worker-verification.json)
confirms the original GLM IDs/start times and no running experiment containers.

The storage opportunity is still **4.43 GiB globally, about 2.21 GiB per TP2
rank**, projected across the base experts. That is roughly 2.5% of the current
87.2 GiB/rank loading footprint, not a measured saving in the live NVIDIA
checkpoint. It justifies a bounded integration experiment once the component
gate passes; it does not yet establish a deployment or whole-model speed win.

The [focused summary](../benchmarks/glm53-flash-hf-audit-2026-10-15/grouped-moe-optimization-v1/summary.json)
retains all seven variants, paired samples and correctness results. The
[hardware report](../benchmarks/glm53-flash-hf-audit-2026-10-15/grouped-moe-optimization-v1/hardware-profile.json)
and [guard records](../benchmarks/glm53-flash-hf-audit-2026-10-15/grouped-moe-optimization-v1/guard-controllers.json)
bind the two successful counter collections and every accepted focused run.
Native sources, libraries, executed benchmark/runtime hashes and all 148
archive members were independently verified; see the
[evidence verification](../benchmarks/glm53-flash-hf-audit-2026-10-15/grouped-moe-optimization-v1/evidence-verification.json).
The frozen archive is `grouped-moe-opt-v1/optimization-evidence-v1.tar.gz`
under the Spark2 audit directory, with local copy
`/private/tmp/mxwave-glm53-opt-evidence/optimization-evidence-v1.tar.gz`.
It is 12,575,484 bytes, SHA-256
`0294070e6c008da3aade7d9843eb07393d50a0ef8097636ab5f35dd19ed8d5e0`.

## Next gates after the full-layer measurement

1. Confirm the aligned-nibble result across the complete 20-case component
   matrix and both sequential TP2 partitions. Preserve the completed original
   baseline and rejected variants. Screen all layers for the aligned exponent
   bucket, retaining the exact record representation as a fallback. Real model
   routing statistics remain a later full-model input.
2. Wrap the validated grouped kernels as Torch custom operations and integrate
   scale ownership with vLLM's `MarlinExperts` loader/quantization path. Preserve
   generic MXFP8 Linear dispatch, MTP exclusions and fused aliases. Qualify
   Dynamo, graphs, allocator ownership and real TP2 collectives. The component
   C ABI alone cannot be advertised as working out of the box in stock vLLM.
3. Run one matched full-model confirmation with the identical imported
   AutoRound checkpoint, compression disabled/enabled. Measure actual retained
   allocation, startup peak, prompt likelihoods, KL and end-to-end tokens/sec.
   Establish repeatability with identical sorted routing when diagnosing
   numerical drift. Current NVIDIA-derived GLM is a separate source comparison.
4. Keep MTP, MXFP8 side linears, DFlash and GGUF integration separate until
   their own component gates pass. For TP1 concentrated routing, profile that
   specific regression before choosing another decoder/tile configuration.

### Whole-model measurement effort

The benchmark itself is straightforward once the loader exists. A rough
engineering estimate is half a day to a day to add Torch custom-operation
registration, retain the compressed scale allocations in `MarlinExperts`, and
validate graph capture plus TP2. This estimate assumes the mixed checkpoint
loads correctly in the selected vLLM engine; compatibility problems can extend
it. The component C ABI is not that integration.

Budget another 30–60 minutes for a small matched compression-off/on inference
comparison once both engines load, plus checkpoint transfer and startup time.
Pin identical weights, activations, KV settings, routing, speculation policy,
prompts and runtime. Measure retained allocation, startup peak, prefill/decode
latency, tokens/sec, prompt NLL and next-token KL. Treat kernel bit equality as
the prerequisite, not a substitute for those model checks.

The full 183.43 GB AutoRound checkpoint was downloaded on Spark2 and
mirrored to Spark1, with complete shard SHA-256 verification on both nodes.
The current live NVIDIA-derived NVFP4 checkpoint is not an identical-weight
control. Both Sparks are needed for the full-model test, so the detached runner
waited for complete source copies before scheduling the authorized serving pause.
That run and restoration are complete; the preparation record and results follow.

### Full-model integration and on-device preparation (2026-10-15)

The [installed-engine fixture](../benchmarks/glm53-flash-hf-audit-2026-10-15/full-model-preparation-v1/plugin-aligned.json)
now qualifies the opaque Torch custom operation and compressed scale ownership
against the actual serving image's system Python, vLLM
`0.1.dev20051+g487ecf187` and Torch `2.13.0+cu130`. This differs from the older
component virtual environment's vLLM 0.31.0. Six real-source MoE cases cover
M=1/64/128 with varied/concentrated routing, sixteen resident experts and
288-expert routing metadata. Stock, adapter eager execution, a non-default CUDA
stream and CUDA-graph replay agree bit-for-bit for both
[raw](../benchmarks/glm53-flash-hf-audit-2026-10-15/full-model-preparation-v1/plugin-raw.json)
and compressed adapters. This is an integration correctness check, not a
full-model latency, quality or allocation measurement. Earlier failed fixture
attempts exposed the installed repacker's added permutation argument and an
accidentally full 288-expert fixture exceeding its memory cap; neither failed
attempt is counted as accepted evidence.

The opt-in entry point in `integrations/glm53_mxfp4/engine_plugin/` installs
inside disposable containers. `model_plugin.py` retains stock routing,
activation and expert reduction, replaces eligible E8M0 scale tensors after
stock repacking, and updates their quantization descriptors. Unsupported
whole-tensor exponent ranges remain on stock storage and dispatch. Weight
codes and mixed MXFP8 linears are unchanged. Scratch is reused per CUDA stream;
the full-model report records its actual contribution. This remains a custom
vLLM adapter, not an upstream or Ollama-compatible release.

The [cleanup record](../benchmarks/glm53-flash-hf-audit-2026-10-15/full-model-preparation-v1/cleanup.json)
records removal of exactly 90 duplicate `.safetensors` files, totalling
**99,558,020,240 bytes**, on Spark1. Before removal, all five complete experiment
checkpoints were copied to Spark2 and independently source/destination
SHA-256-verified: the h64/norm/rtn/mse controls and precision-budget combined
candidate. This preserves the stronger H64 and mixed-candidate references.
Reports, logprobs, configs and tokenizer metadata remain on Spark1, with
`WEIGHTS_MOVED.json` pointers to the backups. Canonical restored Qwen/DeepSeek
models and all live GLM files were retained. The metadata archive is also saved
locally, SHA-256 `bcc714559fde77005d5fea761aaefbd4907b9e6c1d9f181681031a4404ce3f6d`.

Preparation used CPU-only, bounded download, copy and source-scale audit jobs.
Two extra prefetch workers fetched the remaining shards in reverse order to
reduce network waiting; the main downloader remains authoritative for every
shard hash. `audit_full_scales.py` screens all 42 base expert layers, both TP2
partitions and both fused scale families before the GPU run. MTP is excluded.
Both nodes now have all 120 source shards verified: **183,429,201,056 bytes**
per node. The [complete scale audit](../benchmarks/glm53-flash-hf-audit-2026-10-15/full-model-preparation-v1/full-scale-audit.json)
covers all **36,288 matrices**, representing 9,512,681,472 scale bytes.
**166 of 168** fused per-rank scale groups fit the aligned layout. Both TP2
partitions of layer 44's `w2` span an aligned exponent boundary (118–128 and
119–128), so those two groups retain the stock exact scales. The resulting
projected saving is **4,718,589,344 bytes (4.39 GiB)** globally, or about
**2.20 GiB per rank**. This source-layout projection is separate from the measured
allocation reduction below.

The controller paused the original GLM service at an idle window, completed all
three TP2 arms, and restored the original service. An independent watchdog
remained armed during the authorized pause. No experiment engine remains running.

The completed detached controller ran on Spark1 as PID 3216062. Its records are:

```text
/home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/full-model-v1/
  full-v4-src/full_model_control.py --revision v4
  full-controller-v4.json
  full-controller-v4.log
```

The frozen code archive is 66,064 bytes, SHA-256
`3f7f545aa039c3e0b922f3fc1a9306cc6e111f9ef947b52e021f0eb799530e08`.
[Preflight hashes](../benchmarks/glm53-flash-hf-audit-2026-10-15/full-model-preparation-v1/preflight.json)
match source metadata, both native libraries and integration code across the
two Sparks. CPU-only validation confirmed actual CLI argument parsing and
discovery of the package entry point. Setuptools builds a temporary copy so
the qualified source stays mounted read-only. Local validation passed Ruff,
strict mypy on `mxwave`, and 361 tests with one skip, including restoration,
failed-preparation, secret-exclusion, trace attribution, unstable-control rejection
and distribution-hash checks.

After source preparation, the controller waited for an idle GLM window, armed
an independent restoration watchdog, and paused the two original serving
containers. The completed measurement used:

- Three arms with identical imported weights: stock Marlin, the raw-scale
  adapter, and the aligned compressed-scale adapter. The raw arm isolates
  adapter overhead from compression overhead; stock shows the total change.
- TP2 W4A16, fixed 4 GiB FP8 KV per rank, 8,192 maximum context, no prefix
  cache and no draft/MTP execution. Required architecture and RoCE fixes are
  retained; old NVFP4 checkpoint loaders, certified-head selectors and other
  optional serving optimizations are excluded consistently from all arms.
- Frozen token-ID prompts, batch sizes 1/4/8 and prompt lengths 256/2048,
  two warmups and five 64-token trials per case. Report end-to-end throughput
  and TTFT, plus actual allocations and compressed-layer coverage from both
  workers.
- Sixteen authored 256-token diagnostic contexts for prompt NLL and complete
  next-token distributions, with a raw repeat control. This is a matched
  compression diagnostic, not standard benchmark PPL or qualification against
  BF16. The frozen repeat-relative tolerance lacked a prerequisite for a stable
  control; that analysis defect is corrected and disclosed below.
- Separate warmed PyTorch profiles of short decode and longer concurrent
  prefill in the raw/compressed arms on both ranks. Profiled requests do not
  contribute to throughput medians. Preserve kernel symbols, communication,
  attention and Marlin durations; overlapping kernel sums are not wall time.

The watchdog restores the same original container IDs and configuration
digests if the controller dies or the outage deadline expires. Mounts are
sorted before hashing: repeated real Docker inspections returned different
mount listing orders without configuration changes, and that behavior now
has a regression test. Normal and
failed GPU runs also restore worker then head and require GLM health 200.
GPU profiles were collected separately from timing; copying and CPU-only
comparison occurred after restoration. Results are under `run-v4/results/`.
Laptop locking or SSH disconnection did not own these
processes. The 20-case aligned component confirmation remains incomplete;
its interrupted attempts have not been relabelled as successes by this work.

### Completed whole-model pilot and trace findings

The [reviewed report](../benchmarks/glm53-flash-hf-audit-2026-10-15/full-model-v4/reviewed-summary-v1.json)
separates successful execution from qualification. All three arms completed,
and both ranks report every eligible base expert layer. **166 scale groups**
use exact compressed storage; the two layer-44 `w2` groups retain stock storage
and dispatch. Every compressed scale tensor has a GPU byte-equality proof.

| Measured allocation, per rank | Bytes |
| --- | ---: |
| Stock engine | 92,610,497,536 |
| Raw adapter | 92,648,246,272 |
| Compressed adapter | 90,301,575,680 |
| Raw → compressed saving | **2,346,670,592 (2.186 GiB)** |

The global saving is **4,693,341,184 bytes (4.371 GiB)**, or **2.53%** of the
raw control's allocated CUDA memory. Both adapters retain 37,748,736 scratch
bytes per rank. Actual allocation savings are slightly below scale-payload
savings; the report keeps both measurements. This is measured in the imported
MXFP4 pilot, not the restored NVIDIA-derived NVFP4 deployment.

Against the stock engine, the saving is **4,617,843,712 bytes (4.301 GiB), 2.49%**
globally. Both adapters carry 36 MiB of scratch per rank; comparison against
stock includes this extra cost, while comparison against raw cancels it.

All original throughput medians are retained, including cold compilation:

| Prompt tokens | Batch | Stock tok/s | Raw adapter tok/s | Compressed tok/s |
| --- | ---: | ---: | ---: | ---: |
| 256 | 1 | 19.56 | 19.95 | 19.66 |
| 256 | 4 | 54.41 | 58.36 | **24.69** |
| 256 | 8 | 90.47 | 93.03 | 96.66 |
| 2048 | 1 | 15.28 | 15.68 | 15.35 |
| 2048 | 4 | 30.21 | 30.04 | 29.71 |
| 2048 | 8 | 35.60 | 35.89 | 36.37 |

**The large batch-four drop overlaps cold JIT compilation.** Timestamped HTTP
logs and the embedded TileLang timestamps identify a five-second compilation of
`mhc_pre_big_fuse_with_norm_tilelang` in each of the three slow compressed
B4/256 trials. The raw arm has the same event in one trial. Compressed trials
without that event achieved 57.00 and 57.47 tok/s. These two samples are
exploratory evidence, not a replacement median or a qualified warmed result.
The separate GPU profiles do not cover those slow trials; attribution uses
the retained JIT/access logs and token-arrival records.

Read-only [installed-source inspection](../benchmarks/glm53-flash-hf-audit-2026-10-15/full-model-v4/mhc-source-inspection-v1.json)
narrows the warmup fix. Token count is already dynamic in this TileLang kernel,
but the wrapper derives a specialized `n_splits` from token count and passes
normalization constants. The installed mHC warmup accepts only DeepSeek V4
model types and classes. Add GLM-aware warmup of the actual split/normalization
variants and retain compile caches. The exact missed specialization keys were
not logged, so blindly making token count dynamic again would not address this
evidence. Subsequent component checks qualify the selected warmup sizes, and
the one-load diagnostics execute the real-layer warmup on both ranks. The
effect on warmed request throughput has not been measured.

The [both-rank trace analysis](../benchmarks/glm53-flash-hf-audit-2026-10-15/full-model-v4/trace-analysis-v1.json)
matches fifteen B1 decode steps per rank by exact shape/context annotation and
complete coverage of 84 routed expert GEMMs per step:

| Median GPU duration | Raw | Compressed |
| --- | ---: | ---: |
| Routed expert GEMMs, rank 0 | 10.951 ms | 10.702 ms |
| Routed expert GEMMs, rank 1 | 11.059 ms | 10.732 ms |
| Full compute annotation, rank 0 | 43.829 ms | 45.081 ms |
| Full compute annotation, rank 1 | 43.821 ms | 45.065 ms |

Expert-kernel medians are about 2–3% lower, while whole-step annotation medians
are about 2.9% higher. This does not establish an end-to-end speedup. Inputs
have matching shapes but unqualified numerical behavior, and these are
sequential profiled runs, not interleaved identical-activation microbenchmarks.
Unchanged rank-0 generic Marlin linears take about 9.55 ms per matched raw
decode step; other matrix kernels, including BF16 GEMVs, take about 15.46 ms.
These are larger remaining targets than the approximately 0.25 ms expert
difference. Names alone do not attribute every GEMV to an individual module.

Across the combined short-decode/long-prefill raw rank-0 profile, routed expert
GEMMs account for 30.87% of summed kernel time and unchanged generic Marlin
linears for 25.83%. The raw capture has 36 complete expert forwards versus 37
in the compressed capture, so raw aggregate totals are not matched overhead
measurements. Kernel sums can overlap; neither these shares nor the annotation
durations prove DRAM-bandwidth saturation.

**Quality is unqualified.** Stock output already degenerates into special-token
loops; diagnostic mean NLL is about 12.82. Repeating the raw arm produces mean
next-token KL **0.05751**, with changed greedy tokens. Raw → compressed KL is
also 0.05751; all sixteen compressed next-distribution hashes match the later
raw repeat, which does not repair the unstable control. A separate
[chat/prefix diagnostic](../benchmarks/glm53-flash-hf-audit-2026-10-15/full-model-v4/chat-diagnostic-review-v1.json)
also produces gibberish with the proper chat template and with the required
`[gMASK], <sop>` prefix. Prompt format alone does not explain this run.
Checkpoint/runtime compatibility, mixed MXFP8 dispatch and architecture
overlays remain possible causes; none is isolated here.

The original automatic report incorrectly marked quality as passed by allowing
five times an already large repeat divergence. The reviewed analyzer first
requires a stable repeat and now reports `unqualified_unstable_control`.
The additional repeat limits were introduced retrospectively to reject this
run, not to qualify it under a new protocol. A control sanity check should have
stopped the expensive comparison earlier. Original reports remain unchanged
inside the raw evidence archive; the reviewed report explicitly supersedes
the original automatic flag.

The [restoration record](../benchmarks/glm53-flash-hf-audit-2026-10-15/full-model-v4/restoration-v4.json)
confirms the same original container IDs, images and configuration hashes,
worker restored before head, and health 200. An additional read-only health
check after analysis returned 200. The
[evidence index](../benchmarks/glm53-flash-hf-audit-2026-10-15/full-model-v4/evidence-index-v1.json)
pins all 88 files in the 77,260,800-byte raw archive, SHA-256
`6ce4409f9e783cce836b91edf541deea02f4c420152a658ea7d4ad530adc0172`.
Full traces and probability rows stay outside Git, on Spark1 and in a downloaded
backup. The additional JIT/access-log evidence is independently hash-bound.

### Next work, in order

1. **Repair the uncompressed source/runtime control first.** Use proper chat
   and frozen teacher-forced inputs, require sensible output and stable repeats,
   and inspect generic MXFP8 linears against a CPU reference before rerunning a
   full compressed model. Do not change quantization to hide an execution defect.
2. **Prewarm mHC specializations and detect timed compilations.** Qualify the
   warmup at component level, retain the cache, and only then repeat the affected
   B4/256 pair. Reuse the existing source/byte proofs and unrelated measurements;
   another blind three-arm full sweep is not the next step.
3. **Test capacity at the same total allocation.** The measured 2.186 GiB per
   rank could raise this pilot's deliberately fixed 4 GiB KV budget to about
   **6.19 GiB**, roughly 55% more cache budget. Actual token capacity, hybrid
   page rounding, admitted requests and throughput still need measurement.
   This would use freed memory; it would not compress KV tensors.
4. **Screen additional BF16 linears before more scale-kernel tuning.** Start
   with module attribution and a bounded CPU compressibility scan, then reuse
   the existing head/embedding codec only where a fused consumer is justified.
   The decode trace shows substantial unmodified matrix cost. Neither a
   speedup nor another memory-saving percentage is established for those paths.

Scale compression halves one byte per 32 four-bit values: the ideal payload
reduction is only 0.5/17, about **2.94%** of expert code-plus-scale bytes. A large
whole-model speedup from this change alone is not a realistic target. The
measured memory saving is real; its value depends on whether additional cache
capacity or a model-fit boundary matters after the healthy control is restored.

### Follow-up components and one-load diagnostic

The [component results](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/summary.json)
narrow the execution issue without another throughput sweep:

- **MXFP8 linears: 48/48 checks pass.** Dense layer 0 and shared experts at
  layers 3/44 cover both TP2 slices and merged gate/up plus down projections.
  One-hot inputs reconstruct the independently decoded BF16 weight values
  exactly. Random M=1/8/64 outputs have maximum relative L2 error
  **0.00008362** against a CPU FP32 GEMM rounded to BF16; eager repeats and
  tested graph replays match exactly. CUDA allocation peaks at 152,568,320 bytes.
  This tests stock preparation/GEMM, not actual full-model loader coverage.
- **Routed experts: 8/8 CPU-reference checks pass.** Eight real layer-23 experts
  use stock preparation and top-eight routing, both TP2 weight partitions,
  M=1/8/64/512, and routing weights totalling 2.5. Maximum relative L2 error
  is **0.004741**, within the declared 1% component arithmetic gate. M512 checks
  sixteen selected rows; every output row is finite. M64/512 repeat outputs
  are not bit-exact, so this result does not establish model repeatability.
  CUDA allocation peaks at 111,238,656 bytes.
- **mHC warmup: qualified at component level.** After powers-of-two warmup,
  M257 still compiles a new `mhc_pre_big_fuse_with_norm_tilelang` specialization
  and takes **5.474 seconds**. The selector calls the installed split policy
  and chooses 25 sizes covering twelve split counts plus small-token branches.
  Subsequent validation covers **100 boundary/mixed sizes** with the installed
  JIT monitor in error mode and unchanged TileLang cache counts. M257 takes
  **0.227 ms** warm; all validation calls take less than 8.52 ms. Checked outputs
  before/after warmup remain bit-exact. These timings cover two mHC operations,
  not a complete model request. The monitor covers TileLang/Triton/CuTe, not
  every possible DeepGEMM compilation entry. CUDA allocation peaks at
  400,624,640 bytes.

All accepted components ran on Spark2 while GLM remained available, with an
idle guard that cancels only the experiment if serving activity resumes.
The first MXFP8 attempt failed before executing a GEMM under a 3 GiB container
limit; the telemetry-enabled 5 GiB attempt passed. The first mHC harness lacked
the engine configuration required by `CustomOp` construction. The accepted
standalone harness calls the same installed CUDA implementation directly.
Neither failed attempt is counted as accepted evidence or as a model failure.

The tokenizer is also excluded as an obvious mismatch: the imported source and
working deployment have identical `tokenizer.json` SHA-256
`19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d`
and `tokenizer_config.json` SHA-256
`98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc`.
Vocabularies, merges and added tokens match. Reviewed mHC, KDA, RMSNorm, NoPE
and routed-scaling configuration values also match the deployed model.

The second full-model diagnostic completed its loader and warmup checks on Spark1:

```text
/home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/full-model-v1/
  full-controller-diagv2.json
  full-controller-diagv2.log
  run-diagv2/results/model-diagnostic.json
  restoration-diagv2.json
```

Its frozen 85,161-byte source archive has SHA-256
`0a72113571f0e0dd95f0e6cf657e9e52eea6a46a987d67f66356a7915cd7ebcf`.
CPU-only CLI validation and matching code/source/library metadata on both Sparks
pass before the authorized idle serving pause. One stock MXFP4 engine load
supports selected actual-loader checks, GLM-aware warmup through the real layer
methods, and graph/eager/graph checks without reloading weights. Loaded expert
checks cover all 42 base layers, with extra experts at layers 3/23/44; mixed
MXFP8 and BF16 norm/mHC/head/embedding values are also checked. The planned
graph/eager comparison uses three simple chats and two correctly prefixed
teacher-forced contexts per mode, not a standard quality benchmark. No throughput
sweep runs on failed output.

The first attempt, `diagv1`, loaded the model but its checker moved the loaded
tensor to CPU while leaving the prepared reference on CUDA. `torch.equal`
raised a device mismatch before any inference comparison. This is a harness
failure and establishes no model or compression failure. The
[failure record](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/full-diagv1-failure.json)
confirms automatic restoration of both original container IDs and configuration
hashes, with service health 200. Its immutable source and partial results remain
preserved separately.

The fix copies both sides to CPU before exact-byte comparison and handles scalar,
empty and non-contiguous inputs. Four local regression tests pass. Before the
retry, a [bounded CUDA fixture](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/checker-regression-v1.json)
passed all **27 checks**, including CPU/CUDA comparisons in both directions,
integer/BF16/FP32/FP8 storage, reserved FP8 bits, and four real prepared expert
tensors. Deliberate bit flips, signed-zero differences and shape differences
are rejected. CUDA allocation peaks at **19,269,120 bytes**; GLM stayed running.
The full-model controller now requires this passing report bound to the exact
`diagnostic_worker.py` SHA-256
`25c1316857cd75c288ea401e6f0062c29930067795b407497bfea4606959811b`.
This qualifies the checker regression, not full-model loader coverage.

The [second attempt's review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/full-model-diagv2-review.json)
records **658/658 exact selected comparisons per rank, 1,316 total**. Real-layer
mHC warmup covered all 25 sizes in **48.41/48.63 seconds** on the two ranks.
The next RPC failed because it referenced the older runner's
`cudagraph_dispatcher`; the pinned image actually instantiates
`vllm.v1.worker.gpu.model_runner.GPUModelRunner` with `cudagraph_manager`.
No graph/eager inference comparison ran. Both original containers and their
configuration hashes were restored, with health 200. The 166,286-byte retained
archive `diagv2-evidence-v1.tar.gz` has SHA-256
`6beb69eda0cfac3930c10f9990bb6b6b4963104bdf3a7d1de0707eb5bafb4bd2`.

The [mode-switch qualification](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/dispatch-qualification-v1.json)
passes on a tiny live Llama engine in the same
pinned image and requires the actual full-model runner class. It counts actual
full-graph replays in FULL/NONE/FULL phases while retaining captured graphs.
The corrected switch suspends the manager's capture-ready flag, which its
dispatch method actually consults; changing its mode field alone is insufficient
because dispatch candidates are cached. Unsupported interfaces return a failed
acknowledgement rather than throwing an exception that kills the loaded worker.
It observes **3/0/3 actual graph replays**, with identical greedy output IDs,
and repeats all 27 checker regression cases successfully. PyTorch allocation
peaks at 118,921,216 bytes. This is a TP1 dispatch proof, not GLM or TP2 quality
qualification. The corrected full-model attempt, `diagv3`, uses the exact tested
worker SHA-256 `b9d7f20d5cb8f1ddddfcc81ee59eabf1943eff64475f29cb3b20244e31752696`.
Its frozen 88,243-byte source archive has SHA-256
`7d0ef0409b22e2f372ac57235fdc908994c1ef3504fe1b1a00094e0ff74c67ca`.
The client now saves default-mode inference before attempting the first switch.
The retained first-token distributions and prompt NLL measure prefill; generated
chat output also exercises graph versus eager decode. These two scopes must not
be confused when interpreting a difference between modes.

The [corrected diagnostic](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/full-model-diagv3-review.json)
completed all three phases without a harness failure. Both ranks acknowledge
FULL/NONE/FULL dispatch with their four captured graphs retained. The 1,316
selected loaded-tensor comparisons pass again, and real-layer mHC warmup takes
47.75/48.48 seconds. All nine chat checks produce nonsense; graph and eager
phases generate the same strings for each prompt. Switching dispatch to eager
does not repair this stock-model execution. Attention/KDA projection weights
were outside the selected loader comparisons. The later CPU audit below finds
a concrete selector defect; the subsequent diagv4 section records its GPU repair check.
The two retained distributions per phase concern prefill, not graph decode.

The [restored working deployment](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/production-nvfp4-sanity-client-v1.json)
answers **4, Paris, BLUE**, passing 3/3 of the same prompts. Its low reasoning
effort and parser differ from the initial private diagnostic, as detailed below.
It uses a different NVFP4-derived mixed checkpoint and DFlash, so this
is a serving sanity reference, not a paired quantization-quality comparison.
The check uses `local-spark`'s documented authenticated client through its
existing loopback tunnel. No credential is retained in the evidence.

The [restoration record](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/restoration-diagv3.json)
confirms both original IDs/configuration hashes and health 200. The completed
2,034,018-byte `diagv3-evidence-v1.tar.gz` archive has SHA-256
`34a86fa136055b5f52773abca2a8e9c8ac48fbc9fdfe1fb890a493605a26939d`.
No further full-model throughput sweep is justified until this stock control
works. Next work screens additional compression targets independently and
inventories the working checkpoint's actual mixed formats before estimating
any saving for that deployment.

## Fused-attention dispatch defect

The [CPU audit](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/attention-dispatch-v1.json)
uses the pinned image and the same read-only architecture overlay. It calls the
installed `configure_quant_config()` and compressed-tensors selector, and checks
the actual indexed safetensors headers. It loads no model, requests no GPU and
leaves the restored serving containers running.

The checkpoint excludes individual attention projections from quantization.
The runtime loads these into fused linears, but its inherited
`packed_modules_mapping` contains neither of their fused names. Consequently,
the exclusions fail to match and the generic `Linear` MXFP8 group wins:

| Runtime projection | Base layers | Source tensors | Before repair | After repair |
| --- | ---: | --- | --- | --- |
| `in_proj_qkvbfg_a` | 34 | Six BF16 projections, no scales | MXFP8 | Unquantized BF16 |
| `fused_qkv_a_proj` | 11 | Two BF16 projections, no scales | MXFP8 | Unquantized BF16 |

This is a demonstrated configuration defect and the strongest current
explanation for the failed sanity prompts. Both actual constructors pass their
quantization config through. The installed MXFP8 scheme allocates FP8 weights
and E8M0 scales with `torch.empty`; the parameter loader copies source weights
into the destination dtype. These BF16 source projections provide no scales to
initialize the MXFP8 scale parameters. The default loader enables strict loaded
parameter tracking by default only for unquantized models, allowing this
inconsistency to survive loading. This reasoning follows the source path;
the audit does not inspect those scale bytes in a resident GPU model.

The source review binds `default_loader.py` SHA-256
`9c9d54b1b650bf5affc924ebb8b8c73711e187ae7486cadd9f737bb1279cc7b8`,
`compressed_tensors_w8a8_mxfp8.py`
`62ec78c04519fa03391b9b3532488034da1ae70b6f385b802c10cc076a73757e`
and `parameter.py`
`ff6054fbd19ec932c548d562c6f4cc4506383b71cbe411fcfd554c8b1d87b510`.

`integrations/glm53_mxfp4/attention_mapping.py` adds the missing fused aliases
on a copied architecture mapping before model configuration. The private plugin
enables it only with `MXWAVE_GLM53_ATTENTION_MAPPING=1`. It also declares the
indexer's `wk_weights_proj` aliases for completeness; that constructor already
forces unquantized execution, so it is not counted among the 45 affected
projections. The audit changes incorrect selector assignments from **45 to 0**;
eight controls retain their original expert, feed-forward and unfused-attention
schemes. No checkpoint tensors or production files are changed.

Two local regression tests check inheritance isolation, repeated installation
and rejection of conflicting runtime mappings. Ruff, strict core mypy and the
full suite pass (**363 passed, 1 skipped**). The prior expert/norm/head checks
missed fused attention and therefore did not establish complete loader coverage.

The corrected stock GPU diagnostic below completes the loaded-attention gate.
Only completed final-answer sanity and a matched compression comparison can
qualify model quality or a throughput claim.

## Corrected attention load: completed GPU diagnostic

The [diagv4 review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/full-model-diagv4-review.json)
records the repaired stock execution with compression disabled. All 45 fused
attention input modules on each rank are BF16, with selected source rows matching
the expected TP slices. The 452 attention comparisons per rank join the earlier
658 selected expert, side-layer, norm and table comparisons: **2,220 selected
comparisons pass across both GPUs**. This is selected coverage, not a complete
hash of every resident model weight. The source uses zero RoPE dimensions, so
the MLA fused shape is 2048 × 4096 without additional NoPE padding.

| Diagnostic | Before metadata repair, diagv3 | After repair, diagv4 |
| --- | ---: | ---: |
| Graph-before mean teacher-forced NLL | 12.0949 | 0.74704 |
| Eager mean teacher-forced NLL | 12.0893 | 0.74273 |
| Graph-after mean teacher-forced NLL | 12.0618 | 0.74451 |
| Generated behavior | Context-independent gibberish | Coherent, prompt-specific reasoning |

These NLLs cover only two authored diagnostic contexts, not benchmark
perplexity. The repeated graph-before/after next-token distributions have mean
KL **0.00003407** and identical greedy token IDs. Those distributions are
prefill outputs, not graph-decoded token distributions; they characterize
repeatability of this control rather than the error of a compression candidate.
Graph and eager chat calls produce the same strings.

The raw exact-answer score remains **0/9**, with every completion ending
in `finish_reason=length`. The harness requested `enable_thinking=false` and
only 32 tokens. The [actual template evidence](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/diagv4-chat-format-evidence.json)
shows that flag is unused: the template defaults to maximum reasoning effort.
Production instead uses `reasoning_effort=low` and the `glm45` reasoning parser.
The templates themselves are identical, SHA-256
`34d5ee66b12fa6446cdae131c352b8f68cd85369e0e6fda115583805fada3891`.
The remaining score therefore does not establish another weight failure or
successful final answers.

Future harness code now requests low reasoning effort, retains the raw message,
allows 256 output tokens and requires a natural stop before accepting the final
answer. The private engine enables the same reasoning parser and defaults.
Future paired controls opt into the audited attention mapping. No further model
reload was performed solely to repair chat formatting; these harness changes
are locally tested and await the next necessary full-model comparison.

Both original GLM container identities and configuration hashes were restored;
the [restoration record](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/restoration-diagv4.json)
reports health 200. The immutable diagnostic archive contains 2,029,901 bytes,
SHA-256 `d3d2cd9a1c0b9a8870dd05d924c530a70cf56319ad54d392b0d411f72066a1d5`.
It also preserves a brief controller hold made while checking a separate CPU
fixture-preparation error. The resident model/checker source was unchanged;
normal completion and restoration followed.

## Additional BF16 targets: measured CPU samples

The bounded [imported-checkpoint screen](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/bf16-targets-imported-v1.json)
and [working-checkpoint screen](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/bf16-targets-working-v1.json)
sample first/middle/last row regions of every base-text BF16 attention, head,
embedding and router matrix of at least 64 KiB. They exclude MTP and vision from
compression candidates, verify the active safetensors index, and preserve
header/code/sample hashes. Both existing codecs reconstruct every sampled bit
exactly. The CPU-only container finishes both screens in 5.6 seconds, peaks below
197 MB RSS, and leaves serving health at 200.

| Target | Matrices | Candidate source bytes | Measured sample saving | Estimated source bytes saved |
| --- | ---: | ---: | ---: | ---: |
| Imported attention, chunk codec | 405 | 12,106,596,352 | 29.60% | 3,585,459,242 |
| Working attention still in BF16, chunk codec | 123 | 676,069,376 | 29.46% | 198,149,048 |
| Head, either checkpoint, row codec | 1 | 1,268,776,960 | 29.53% | 374,664,400 |
| Embedding, either checkpoint, row codec | 1 | 1,268,776,960 | 29.53% | 374,617,613 |

These are deterministic sample results and per-matrix extrapolations, not a
random statistical estimate, a complete packed artifact or a measured TP2 CUDA
allocation. Sample byte counts include each codec's headers and indexes.
The 226 source matrices feeding the 45 incorrectly selected fused projections
alone total 7,118,782,464 bytes; their row-codec samples extrapolate to about
2,106,824,321 bytes saved. That subset is already included in imported attention.
The 337 imported attention matrices eligible for the existing 512-word row layout
suggest 3,534,635,274 bytes saved; the rest need another linear decode layout.

The head and embedding together suggest **749 MB** saved in either checkpoint.
The row layout matters: the embedding's chunk-wide exponent window saves only
16.20% on these same sampled regions, while per-row windows save 29.53%. Every
row's sign, mantissa and outlying exponent bytes are preserved. Routers suggest
another 29 MB, a small capacity target.

Most large attention projections in the working checkpoint are already FP8.
Its active base attention headers contain 5,715,263,488 FP8 value bytes,
143,998,976 uint8 bytes, 14,762,496 FP32 bytes and 676,140,032 BF16 bytes.
The imported checkpoint's multi-GB BF16 attention opportunity therefore cannot
be applied as an estimate for the serving model.

The working checkpoint also contains 408 superseded tensor entries in original
shards. The active index points to replacement shards; the quant-mix runtime
has its own duplicate filter. The screen follows that index and reports
197,791,409,864 active payload bytes separately from 211,335,866,056 physical
tensor payload bytes across the referenced files. These are file tensor counts,
not resident target allocations or a disk-cleanup recommendation.

The three real-projection GPU components below screen the existing fused kernel
before any new attention kernel or whole-model integration. Head/embedding
integration remains a separate candidate; exact CPU reconstruction alone does
not establish GPU arithmetic or latency.

## BF16 attention: real-projection GPU component

The [component report](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/bf16-attention-kernel-v1.json)
uses the existing pinned ZipServ native library on Spark2/SM121, with the current
GLM image. It reconstructs three complete source projections, applies the
recorded TP partition and compares every decoded GPU byte with the independent
source reconstruction. The [fixture manifest](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/bf16-attention-prepared-v1.json)
binds source, packed artifact, preparation code, library and runtime hashes.
No quantized weights or production modules are changed.

| Projection fixture | Logical BF16 bytes | Packed retained bytes | Saving |
| --- | ---: | ---: | ---: |
| KDA input, layer 0, rank 0 | 103,022,592 | 73,394,904 | 28.76% |
| MLA input, layer 3, replicated | 16,777,216 | 11,882,136 | 29.18% |
| KDA output, layer 0, rank 0 | 33,554,432 | 23,809,768 | 29.04% |

All three decode bit-for-bit, including 32 explicit alignment rows for the KDA
input. The padded output is sliced and made contiguous inside the timed
candidate, including that copy cost. Random BF16 activations cover token counts
1, 8, 32, 128, 512 and 2048, with split-K values 1 and 4. All 36 arithmetic cases
are finite and pass the predeclared relative-L2 component gate of 1%; the largest
measured relative L2 is **0.41196%**. Exact restored weights do not make the
different GEMM accumulation path bit-exact or qualify full-model quality.

| Fixture, graph replay | Dense, 1 token | Compressed, 1 token | Latency change | Compressed/dense latency, 512 / 2048 tokens |
| --- | ---: | ---: | ---: | ---: |
| KDA input | 576.10 µs | 380.78 µs | −33.9% | 4.05× / 4.31× |
| MLA input | 87.46 µs | 50.77 µs | −42.0% | 2.52× / 3.71× |
| KDA output | 200.27 µs | 99.41 µs | −50.4% | 3.02× / 3.59× |

The table selects the faster of the two tested split choices independently for
each shape; the report retains both choices, eager timings and the raw-report
hash/path. Timings use five trials and alternating arm order. Single-token eager
results are also favorable, but other small batches vary by mode. A single
guarded component run is an initial performance screen, not a repeatable
whole-model speed claim. Graph-captured large batches are component experiments;
the full-model prefill path is eager, and its large-batch component results also
regress.

The GPU test completes in 12.40 seconds at a paired-comparison peak allocation
of 1,129,478,144 bytes. That peak deliberately includes dense and compressed
weights and is not a measured net production allocation. The idle controller
completes on its first attempt without interrupting GLM or stopping either
production rank.

**Decision:** continue with a decode-focused component and a separate prefill
path. Using this fused kernel for every shape would regress prefill. A bounded
decode-to-scratch plus stock GEMM path is the next useful prefill comparison;
retaining all dense weights alongside packed weights would erase the intended
memory benefit. Profile these component paths before another full-model pause.
Whole-model retained allocation, final-answer sanity, likelihood drift and
throughput remain the confirming gates.

### Prefill fallback and traces: completed component

The [prefill review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/bf16-attention-prefill-v1.json)
tests the same three packed projections with temporary GPU decompression followed
by the identical stock BF16 GEMM. All decoded weight bytes and all **12 output
tensors** match the dense control bit-for-bit. Scratch contains one complete
projection: 103,284,736, 16,777,216 or 33,554,432 bytes, respectively, and is
transient rather than a persistent dense model.

| Projection, 2048-token graph replay | Dense | Decode + stock GEMM | Overhead |
| --- | ---: | ---: | ---: |
| KDA input | 2.226 ms | 3.089 ms | +38.8% |
| MLA input | 0.367 ms | 0.471 ms | +28.4% |
| KDA output | 0.720 ms | 0.992 ms | +37.8% |

Eager CUDA-event results add approximately 27–39% at that size. The fallback is
much less costly than the fused kernel for large prefill, but remains slower
than dense. At smaller batches its fixed decode cost is relatively larger;
the single-token fused path remains the preferable latency candidate.

Six separate PyTorch CPU/CUDA traces cover one and 2048 tokens for all three
fixtures. Each trace records three calls per path: dense, decode then dense,
and fused split-K=4. The KDA-input 2048-token trace attributes a mean **0.793 ms**
to decompression, **2.269 ms** to stock GEMM pooled across the two dense-GEMM
paths, and **10.319 ms** to the fused matrix kernel, plus **1.080 ms** split-K
reduction and **0.435 ms** padding-output copy. These are profiled component
durations, separate from timing medians. The stock large-batch kernels identify
SM121 matrix-multiplication paths. This evidence locates the larger fused
prefill bottleneck inside its GEMM implementation rather than attributing the
entire regression to the cost of decoding the representation.

The component finishes in 5.13 seconds with a paired-comparison CUDA peak of
1,030,934,016 bytes. Its first guarded attempt passes; both production ranks
remain running. The six trace hashes, native kernel totals and raw report hash
are retained in the review. The 72,311-byte frozen-source/trace evidence archive
on Spark2 has SHA-256
`bdd8cf182a1a1c49e23496201fefde9e48da354c91bcd022dcd23733f72687b0`
and is also available locally for offline analysis.

The following component tests bounded overlap and resource contention directly.
Any fused decode path still needs matched model-level likelihood checks because
its accumulation differs. These component storage results do not establish a
29% total GLM memory saving or a whole-model speedup.

### Overlapped prefill: completed component, no useful net gain

The [completed asynchronous review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/bf16-attention-overlap-v3.json)
tests a dependent three-projection chain: KDA input, KDA output and MLA input.
The first 4096 output columns feed the next matrix. These are real source
weights with an authored activation bridge; they do not constitute a GLM forward
pass. This short paired comparison requires no full-model load or serving pause.

`bf16_prefetch.py` decodes into two fixed scratch slots on an auxiliary CUDA
stream. Ready events make the compute stream wait for each decoded matrix;
consumption events prevent reuse while GEMM still reads a slot. The compute
stream rejoins the decoder before returning, including during graph capture.
The final variant queues current compute before the next decode. Dense, serial
decode and asynchronous decode are compared using the same stock GEMM and
dependent activations.

| 2048-token chain, eager CUDA events | Dense control | Async decode + GEMM | Async overhead vs dense | Async vs serial decode |
| --- | ---: | ---: | ---: | ---: |
| Include first decode | 3.547 ms | 4.733 ms | +33.4% | +0.5% |
| Keep first weight decoded | 3.513 ms | 3.908 ms | +11.3% | +0.4% |

These are distinct, interleaved paired comparisons; each ratio uses its own
paired control. They show no useful net gain over serial decoding. Graph
results are similar at 2048 tokens: about +31.7% and +10.8% against dense.
At 32–128 tokens the async path is about 7–15% slower than serial decoding.
The cached-first case excludes its separately measured 0.985 ms initial decode,
and retains the first complete BF16 matrix throughout the comparison.

All three caller-owned GPU decodes match source hashes. All eight batch/cache
cases pass exact intermediate-output checks for eager execution, graph replay
and a non-default caller stream. Each case also poisons both scratch buffers
and replays the graph twice without an intervening host wait, repeated three
times; every intermediate remains byte-identical. Replay allocation growth is
zero, and slot addresses remain fixed.

The cold two-slot pool uses **206,569,472 bytes (197 MiB)**. The cached-first
fixture uses **170,393,600 bytes (162.5 MiB)** including its retained first
matrix and two slots sized for the remaining projections. Both fit the declared
256 MiB weight-storage budget. Their fixture-specific sizes are not a full-model
pool estimate. The packed fixture alone saves about 44.3 MB, less than either
pool: a complete model would need to amortize that fixed storage across many
more compressed projections. The paired comparison peaks at 1,032,655,872
allocated CUDA bytes; it deliberately keeps the dense control too.

Four preserved PyTorch CPU/CUDA traces expose the limiting behavior. At 2048
tokens with the first matrix cached, about **202 µs (46%)** of decoder duration
overlaps GEMM, but GEMM duration increases from about **3.240 to 3.375 ms** and
decoder duration from **0.379 to 0.437 ms**. Those increases consume the overlap;
profiled kernel span remains essentially unchanged. The cold path overlaps only
about **28 µs (2.3%)** of decoding. These profiled durations are explanatory
evidence, separate from timing medians.

GB10 reports 102,400 shared-memory bytes per SM. At 128 tokens the stock GEMMs
request 98,304–99,328 bytes per block, while decoder blocks request at least
15,408 bytes. These blocks cannot reside together on one SM. At 2048 tokens
the GEMMs request 78,848–87,040 bytes, allowing some combinations to coexist,
but measured contention still erases the net benefit. Different CUDA streams
alone therefore do not hide this decoder cost with the current kernel choices.

The complete v3 run takes 12.24 seconds and passes on its first guarded attempt.
Both original GLM ranks remain running. The evidence also preserves the earlier
reader failure and the attempt that reached our own unchanged 1.5 GiB allocator
cap. Reusing one warmup/capture stream and one caller stream in v3 bounds
per-stream workspace retention and completes below that cap. No correctness or
memory gate was relaxed.

The [evidence manifest](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/bf16overlap-evidence-v1.json)
binds 15 source/report/trace files across all three revisions. Its 275,287-byte
archive is retained on Spark2 and locally, SHA-256
`6f8e1dc3a9c25d3c500934b547708a50c5ea687df23bf917d2aec746e56c3144`.
Local Ruff, strict core mypy and the full CPU test suite pass; five interval
tests guard overlap calculations against nested intervals and double counting.

**Decision:** keep this async prefill variant experimental and do not integrate
it into the serving model. The single-token fused component remains a separate
positive result. A serial prefill fallback can use one scratch slot and simpler
scheduling if capacity savings justify its latency cost. Further overlap work
needs a concrete kernel-resource or decoder change; another whole-model pause
with this unchanged scheduling is not justified.

### Next priority: selective single-token KDA execution

The async result leaves a narrower positive candidate in the original
[single-projection measurements](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/bf16-attention-kernel-v1.json).
At one token, split-K=1 reduces graph-replay latency by **27.3% for KDA input**
and **28.3% for KDA output**. Eager CUDA-event reductions are 22.0% and 6.9%,
respectively. Measured relative output L2 is 0.02263% and 0.00161%, materially
smaller than the corresponding split-K=4 results near 0.27%. These are initial
random-activation component measurements, not full-model quality or speed.
The MLA split-K=1 single-token case regresses and should remain outside that
initial fast-path candidate.

Prioritize the following gates:

1. Build an opt-in BF16 KDA storage adapter restricted to audited runtime
   projections and supported shapes. Use the lower-error split-K=1 kernel for
   one-token execution and one bounded decode buffer with stock GEMM for prefill.
   Include padding, parameter replacement, TP partitions, bias/return contracts
   and actual buffer ownership in compatibility checks.
2. Qualify that adapter on small real-projection/runtime fixtures, including
   non-default streams and graph replay. Capture representative model
   activations during the next necessary corrected-model run; compare likelihood
   drift to the repeatability control. Even split-K=1 is not output-bit-exact.
3. Once those gates pass, perform one matched full-model quality, allocation and
   timing comparison using the corrected attention mapping and final-answer
   protocol. Include prefill cost in request latency. Reject a candidate that
   merely moves the loss into prefill or probability drift.
4. Separately adapt the earlier head/embedding path to GLM. Its deterministic
   CPU samples suggest about 749 MB of source-storage savings; TP2 allocation,
   logits arithmetic, DFlash head calls and complete-model performance remain
   unmeasured.

A native fused prefill redesign that consumes compressed tiles inside GEMM is
another research route, but requires new kernel work and its own arithmetic and
performance qualification. It has lower priority than testing the existing
single-token candidate. The current async scheduling result does not justify
abandoning all compression work, and it does not justify promoting a release
before the remaining model-level gates.

### Selective KDA adapter component gate: passed

The opt-in adapter is implemented in `kda_policy.py`, `kda_adapter.py` and
`kda_runtime.py`. Selection binds the complete source config and index hashes,
then derives 68 runtime prefixes: fused input and output projections in the 34
KDA layers. MLA, expert kernels and the MTP stack remain outside this selector.
It delegates source slicing to the installed vLLM parameter loaders, verifies
complete shard coverage, checks full GPU reconstruction against each loaded
weight hash, then removes the original dense parameter. Unrecognized shapes,
implicit dtype casts, missing or duplicate source parts fail explicitly.

The [qualified v3 fixture](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/kda-adapter-fixture-v3.json)
passes all 12 actual vLLM projection cases: input/output at layers 0, 22 and 44,
each with both TP source partitions. The fixture patches rank queries and
disables the row-parallel collective on one GPU; complete two-node execution is
a separate gate. Independent source reconstruction, exact packed-weight GPU
decoding, parameter release, output/return contracts, non-default streams,
poisoned-pool graph replay and zero replay-allocation growth all pass.
Fullgraph Dynamo tracing also passes the real input module at one and three
tokens using its eager backend; that is not an Inductor performance result.

For one token, the split-K=1 fused output has maximum measured relative L2
**0.02079%**, below the unchanged 0.1% component gate. This is arithmetic drift,
despite exactly reconstructed weight bytes. For larger token counts, decoding
into the shared scratch pool followed by stock BF16 GEMM produces bit-identical
outputs in all tested eager, graph and stream cases.

| One-token actual-adapter timing, summed paired medians | Graph latency ratio | Eager latency ratio |
| --- | ---: | ---: |
| Six KDA input partitions | 0.720 | 0.808 |
| Six KDA output partitions | 0.854 | 1.097 |

Each ratio is candidate/control. All six inputs improve. Output-projection
results vary: two of six graph cases regress, so the earlier isolated output
speedup must not be generalized to every layer. These component timings do not
establish a request-level gain.

The production adapter uses one **206,569,472-byte (197 MiB)** shared pool per
GPU, with two fixed stream slots. Slot ownership is explicit and reported;
more than two inference streams are rejected. The pool stays below the fixed
256 MiB gate. The component comparison peaks at 625,506,304 allocated CUDA
bytes with both representations deliberately present. The run takes 57.65
seconds and never stops GLM. Its
[evidence manifest](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/kdaadapter-evidence-v1.json)
binds the frozen source and full report; the archive is 230,058 bytes, SHA-256
`677b85c7d05dba5ff1336f0fede04b021d299a4cde3932635bd88fdf5e68e4e5`.

The matched corrected-stock versus selective-KDA comparison uses
using the same imported checkpoint, TP2, fixed KV allocation, reasoning parser,
prompt protocol and single-token graph configuration. It checks selected loaded
source bytes and naturally finished final answers before timing; measures three
request cases and 16 likelihood windows; repeats the stock distributions; and
collects both-rank PyTorch traces separately from timing medians. The stock
repeatability gate remains mean KL <= 1e-5 and absolute mean NLL drift <= 1e-4.
An unstable control cannot qualify the candidate. No DFlash or scale-compression
adapter is enabled in either arm, so this comparison isolates the new BF16 KDA
storage/runtime path. Original GLM container identities and configuration
digests are recorded before the idle pause and checked during automatic restore.

### Full-model selective KDA v2: measured gain at one request, batching penalty

The [completed comparison](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/full-model-kda-v2-review.json)
loads the same corrected imported checkpoint in both arms. Both pass all nine
graph/eager/graph final-answer checks and all 1,110 selected source comparisons
per rank. All 68 selected projections per rank retain compressed storage, with
zero surviving dense-parameter references. The run completes in about 40
minutes, independently of the laptop, and restores both original GLM container
IDs and configuration digests with health 200.

| Prompt tokens | Concurrent requests | Stock aggregate tok/s | Compressed aggregate tok/s | Change |
| --- | ---: | ---: | ---: | ---: |
| 256 | 1 | 15.32 | 17.25 | +12.6% |
| 2048 | 1 | 12.88 | 14.31 | +11.1% |
| 256 | 4 | 39.87 | 30.00 | -24.8% |

Five retained trials determine each median; profiling is separate. First-token
latency changes from 373 to 401 ms for the short single request, 1,144 to
1,155 ms for the long single request, and 1,021 to 1,064 ms for batch four.
These measurements use no DFlash and graph capture only at one token; batch
four runs eagerly in both arms. They are not production NVFP4/DFlash results.

Actual retained CUDA allocation across both GPUs falls from 188,834,497,536 to
186,549,022,720 bytes: **2,285,474,816 bytes (2.13 GiB), or 1.21%**. Selected
BF16 weights shrink from 9,287,237,632 to 6,596,544,432 bytes, about 29.0%.
The two shared pools together cost 413,138,944 bytes, explaining why net savings
are smaller than the weight-payload saving. Allocation is unchanged before
versus after the timing/profile work in both arms.

The small capacity fraction and larger single-request speed gain are
compatible: the unchanged MoE expert bank dominates retained storage, while
the KDA projections execute for every token. In the single-token trace, the
dense BF16 linear group occupies about 35–36 ms of the 59 ms GPU forward,
including the unchanged head and other projections. The adapter targets this
frequently read part of the model rather than the full resident expert bank.

Measured candidate mean KL is **7.1263e-5**, compared with **7.9608e-5** for
stock repeated against itself. Mean NLL changes by +0.000201 for the candidate
and +0.001109 for the stock repeat. All 16 greedy next-token IDs agree.
However, the stock repeat exceeds both unchanged stability limits, so the
formal quality verdict is **unqualified: unstable control**. A candidate drift
smaller than this repeat is encouraging evidence, not a passed quality gate.
The full-vocabulary windows are prefill computations and use the exact dense
fallback; they do not directly measure fused single-token arithmetic. The nine
autoregressive chats exercise that path but remain simple behavior checks.

The [offline both-rank trace review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/full-model-kda-trace-review-v2.json)
matches 14 complete single-token shape annotations per rank. Median GPU ranges
fall from about **59.0 to 51.9 ms** in both ranks. The compressed path replaces
most of the dense BF16 linear work with about 21 ms of fused compressed GEMM.
At four tokens it instead issues **68 weight-decoder kernels every forward**,
about 35 ms total, before the stock GEMMs. Five batch-four forward samples per
rank span about 93.0 ms stock and 117.2–117.4 ms compressed. These are grouped
by request/token counts, not exact matched context lengths: asynchronous
request arrivals differ. The decoder count and time nevertheless identify
direct overhead introduced by this adapter. At 256-token prefill the decoder
adds about 37 ms; at 2048-token prefill it adds about 36 ms, a smaller fraction
of the roughly 1.1-second forward.

The [evidence manifest](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/kdamodel-evidence-v1.json)
binds 93 reports, distributions, logs, source snapshots and trace files. The
51,116,668-byte archive is preserved on Spark1 and locally, SHA-256
`0920a9bed5b447fdf9673cb60fc6fbceb89fb5548e5c41404a790d054dfd980c`.
The earlier v1 preparation hit an isolated CPU-container installation permission
error and left GLM running; v2 fixes that preparation and completes both arms.

**Decision:** retain this as an experimental single-request capacity/speed
result. Do not enable it generally for batched or speculative serving. First
screen fused execution at 2–8 tokens, then qualify the actual adapter across
TP slices. Reuse this saved stock arm for the updated-candidate comparison
when source, runtime, settings and protocol hashes still match; label the
noncontemporaneous control explicitly. Larger-prefill decode cost and stronger
fused-decode quality checks remain separate work.

### Small-batch repair: component gates passed

The [native screen](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/kda-small-batch-native-v1.json)
compares dense BF16, fused split-K=1 and the current decode-to-scratch fallback
on the existing source-bound KDA input/output fixtures. At four tokens, eager
input time falls from **1.218 to 0.439 ms** versus that fallback, and output
time from **0.371 to 0.187 ms**. These gains are relative to the compressed
fallback, not dense BF16. Every tested 2/3/4/7/8-token output matches dense
bytes, including basis inputs; single-token arithmetic retains its existing
0.1% relative-L2 gate. GPU peak is 348,248,064 bytes and execution takes 6.88
seconds. Production GLM remains running throughout the guarded test.

The opt-in `MXWAVE_GLM53_KDA_FUSED_MAX_TOKENS=8` path is then tested through
the [actual vLLM adapter](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/kda-small-batch-adapter-v1.json).
All twelve layer/rank/projection cases pass, including **60 small-batch shape
cases** with byte-exact eager, graph, replay and non-default-stream outputs.
Dense-parameter release, complete TP source loading, poisoned scratch reuse,
zero replay allocation growth, return contracts and the fullgraph Dynamo
fixture pass too. The 197 MiB shared pool and 1.5 GiB process allocator cap
are unchanged; paired peak remains 625,506,304 bytes. This qualification takes
124.27 seconds without stopping GLM.

The default remains the previously qualified single-token path. Both permitted
limits, 1 and 8, are explicit; larger prefill still uses bounded scratch and
stock GEMM. At four tokens the expanded adapter's summed graph latency ratios
against dense are 0.946 for inputs and 1.078 for outputs; eager ratios are
1.063 and 1.448. Removing the much larger decoder cost does not establish
parity with dense or a whole-request speedup. The output projection's eager
dispatch/kernel overhead remains visible and needs to be included in the
model result.

The [component evidence manifest](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/kda-small-batch-evidence-v1.json)
binds both frozen sources and full reports. Its archive is 636,092 bytes,
SHA-256 `d9422d75508a96481a784ce2d2618c2bc49cf9e760f3fe3974d9c1dc723eedfb`.
The `kda8modelv1` controller runs only the updated candidate after validating
the saved `kdamodelv2` stock arm's source, libraries, original service identities,
CLI settings and protocol. It reuses that stock report and both-rank traces;
no baseline reload is scheduled. The saved control is explicitly labelled
noncontemporaneous. Loaded-source checks, final-answer checks, timing, retained
allocation, likelihood windows and profiling precede automatic restoration of
the original containers. The completed model result follows; the earlier
unstable-control quality verdict remains binding.

### Updated model: batching loss reduced from 24.8% to 3.5%

The [candidate-only follow-up](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/full-model-kda8-v1-review.json)
completes in 22.77 minutes including restoration. Both original container IDs
and their configuration digests match, health returns 200, and no experiment
retains a GPU. All nine final-answer checks and all 1,110 selected source
comparisons per rank pass. Both ranks report 68 compressed projections, fused
token limit 8, the same bounded pool, and no retained dense parameter.

| Prompt / concurrent requests | Saved stock tok/s | First compressed tok/s | Updated compressed tok/s | Updated vs stock |
| --- | ---: | ---: | ---: | ---: |
| 256 / 1 | 15.32 | 17.25 | 16.39 | +6.9% |
| 2048 / 1 | 12.88 | 14.31 | 13.64 | +5.9% |
| 256 / 4 | 39.87 | 30.00 | 38.46 | -3.5% |

The updated batch-four rate is **28.2% higher than the first compressed
candidate**. Its retained allocation saves **2,293,863,424 bytes (2.14 GiB,
1.21%)** globally. Weight payload and pool sizes match the first candidate;
the small allocation difference is outside those representations. First-token
latencies are 408 ms, 1,199 ms and 1,098 ms for short single, long single and
short batch-four cases. Larger prefill still pays the decoder cost.

These runs are separated in time. The unchanged single-token native path does
not reproduce its first run's exact speed: measured single-request rates fall
about 5% between candidates. The data establishes a large batching recovery;
small remaining differences against the saved stock control need caution.
The result still falls outside a 2% throughput-regression target at batch four.

The [both-rank trace review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/full-model-kda8-trace-review-v2.json)
finds **68 fused compressed GEMMs and zero full-weight decoder kernels** in
four-token decoding. Four exact-shape forward annotations match per rank;
median GPU ranges are 98.3–98.4 ms, versus 93.0 ms stock and roughly
117 ms in the first compressed candidate's count-group samples. Single-token
matched ranges remain faster than stock: about 52.7 versus 59.0 ms.

At four tokens, rank 1's uncovered stream span rises from about **10.9 to
22.7 ms**, while rank 0's communication-kernel time rises from about **8.3 to
17.1 ms**. These observations identify gaps and rank imbalance; they do not
prove a host-dispatch or network cause. The GPU arithmetic groups alone do
not explain the residual slowdown. Next investigate the nested operator
dispatch and token-padding/copy launches in the eager path, and correlate
launch gaps with communication waits. Small-batch CUDA graphs are another
candidate, but need the same graph settings in the dense control. No new
baseline is justified merely to repeat the unchanged current configuration.

Quality remains **unqualified**, with the saved stock repeat still above both
fixed stability limits. Updated candidate mean KL is **1.0900e-4**, versus
**7.9608e-5** for stock repeated against itself; mean NLL changes by
**+0.000641**, versus **+0.001109** for that repeat. All 16 greedy next-token
IDs match. These prefill windows do not isolate fused-decode arithmetic, and
the passed component and chat checks do not replace stronger model evaluation.

The [full evidence manifest](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/kda8model-evidence-v1.json)
binds 91 files in a 48,963,239-byte archive, SHA-256
`854e7d0db05dd61b6ceb6d997fce74ee480a8f9dbf21321e5de9120680d3a6d1`.
The [offline review manifest](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/kda8-review-evidence-v1.json)
also preserves the analysis code and corrected runtime-policy description;
numeric measurements are unchanged. Keep this opt-in and experimental. The
batching repair is measured, and general deployment still requires the
remaining performance and fused-decode quality checks.

## DFlash feasibility — separate follow-up

The running deployment mounts
`derived/incoai/GLM-5.3-Flash-DFlash2-fp8blk` as `/draft`, with seven speculative
tokens at batch size one. Its source is
[Inco's DFlash2 checkpoint](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2).
The deployed local derivative is already FP8 block-128 for its main linears,
with BF16 FC, convolution projections and selector codebooks. Its
[header audit](../benchmarks/glm53-flash-hf-audit-2026-10-15/dflash-header-report.json)
counts 1,377,706,496 payload bytes: 964,689,920 FP8, 412,781,056 BF16 and
235,520 FP32 scale bytes. These FP32 scales are not the base experts' E8M0
scale format.

The [CPU feasibility report](../benchmarks/glm53-flash-hf-audit-2026-10-15/dflash-codec-report.json)
checks every BF16 value, including all rows of both selector codebooks:

| DFlash component | Original bytes | Exact CPU representation | Saving |
| --- | ---: | ---: | ---: |
| Predecessor codebook | 79,298,560 | 56,930,933 | 28.21% |
| Successor codebook | 79,298,560 | 57,037,599 | 28.07% |
| FC projection | 167,772,160 | 118,323,536 | 29.47% |
| All BF16 components combined | 412,781,056 | 293,300,969 | **28.95%** |

The combined saving is **119,480,087 bytes** (119.5 MB), equivalent to 8.67%
of this checkpoint. The codebooks use the existing row-addressable codec with
256-word blocks; their deployed width is 256, rather than the earlier qualified
embedding's width 4096. Every codebook row and all other BF16 chunks reconstruct
bit-for-bit. The projection representation is a bounded CPU feasibility format,
not yet a fused linear adapter or a measured GPU allocation. Host RSS peaked
at 189,411,328 bytes; the probe had no GPU device access.

There is also redundancy in the already-quantized FP8 bytes: 42 deterministic
1 MiB slices from first/last-layer projections shrink from 44,040,192 to
37,022,710 bytes with zlib level 1 (15.93% saving, per-slice 13.82–16.65%).
This is a CPU storage screen, not a random whole-checkpoint estimate or a
usable GPU decode path. Do not extrapolate a throughput gain or count these
bytes as currently achievable runtime savings.

The deployed selector indexes only requested predecessor/successor rows before
its einsum, making it a plausible row-decoder integration. DFlash inherits the
target's shared head/embedding path; those tables are absent from its checkpoint
and must not be counted again as drafter-owned savings. Adapters must also
support the draft head's top-k logits call, not only the target logits path.

Qualification needs graph/compile-compatible selector row decoding, bounded
scratch, identical selector scores and draft token choices, accepted tokens
per verification step, draft latency, end-to-end tokens/sec and retained memory.
Exact decoded weight bits alone do not establish identical floating-point
execution. The BF16 and FP8 CPU results justify a future component, while the
base-expert scale integration remains first because its potential saving is
much larger. That CPU-only probe did not modify or restart the serving drafter
or GLM.


## Combined GLM storage qualification — 16 October 2026

The current candidate composes exact expert-scale packing, KDA-8, an untied
ZipServ BF16 head and MxWave row-addressable BF16 embeddings. It keeps the
imported checkpoint, TP2, fixed KV allocation and graph size one; DFlash and
MTP remain disabled in this experiment. The four-bit expert codes themselves
are unchanged. This is a private opt-in adapter for the pinned vLLM image,
not a supported general-purpose vLLM package.

The vocabulary component is qualified (`vocabfixturev2`). All four complete
TP shards reconstructed to their original source SHA-256 on the GPU. The
installed vLLM constructors and shard loaders were exercised, including stock
TP masks, boundary and invalid global ids, repeated/empty shapes, alternate
streams and CUDA graphs. The two ranks were tested sequentially on Spark2;
collectives remain a full-model gate. Embedding outputs are exact. Head
relative-L2 error at one token is 0.000132–0.000143; the tested 2–128-token
head outputs were exact against the component dense control.

Vocabulary payloads total 2,537,553,920 bytes dense and 1,788,921,568 bytes
retained: 748,632,352 bytes saved (29.50%). This is tensor storage, not yet a
full-model CUDA allocation result. The component peaked at 1,158,170,624 CUDA
bytes with both comparison arms present and completed in 33.04 seconds.
Component paired head timings were 1.45–1.51x faster at one token and
1.08–1.13x at four tokens; these are isolated results, not whole-model speedups.

CPU preparation first encountered a pathological large-chunk NumPy path. A
bounded stack read pointed to `PyArray_Where`; a small-chunk probe was fast.
The replacement uses 128-row chunks, preserves the same format and exact
source binding, and prepared all four shards in 115.84 seconds. The original
packing attempt and its partial artifacts remain preserved. No serving GLM
container was stopped for preparation or component qualification.

The detached `combinedv1` controller passed its CPU and GPU prerequisite gates
and paused idle GLM with its independent watchdog armed. It reuses the immutable
`kdamodelv2` stock throughput/protocol/traces, obtains a fresh short stock
decode control and repeat, then runs the combined candidate. New quality
requests generate four tokens: position zero is prefill, and the other three
are actual autoregressive decode steps using live KV state. Comparisons require
identical prompts and generated prefixes; the existing strict stock-repeat
stability floor is retained. Thus the earlier prefill-only KL is not presented
as a fused-decode qualification. Both original container IDs and configurations
are restored automatically by the controller and an independent watchdog.

The completed `combinedv2` result saved 7.15 GiB of live CUDA allocation
across both nodes (4.06%), improved single-request throughput 7.6–10.0%,
and measured a 2.6% four-request regression. All 64 actual-decode generated
ids match; strict quality remains unqualified because the stock repeat is
unstable at the predeclared floor. The [concise results and profiling review](GLM53_LOSSLESS_RUNTIME_RESULTS.md)
records the full measurements and next steps. Runtime maintenance currently requires the
pinned image/native ABI, source-bound policy and a small set of vLLM selector
and loader hooks. Packaging into one plugin is conditional on the combined
result; benchmark controllers and diagnostics are not part of that runtime.

`combinedv1` initialized but stopped at the ownership gate before candidate
measurements. The pinned `Glm5NextModel` constructs its text embedding without
passing `quant_config`, so the configuration selector reached the head but
could not select the embedding. The original component fixture passed that
argument explicitly and missed this difference. Both original serving
containers were restored with matching identity/configuration digests and
health 200; no combined throughput or quality result exists from this attempt.

The corrected adapter supplies the existing engine CT config only for the
audited text-embedding constructor prefix when that argument is absent. The
`vocabfixturev3` qualification now reproduces the actual omitted-config call.
All four complete GPU shard hashes match their sources, original dense
parameters are released, and the fixture passed in 49.66 seconds. Other
constructors retain their original config behavior. The ownership RPC is now
saved before evaluation, with per-rank counts included in any failure message.

`combinedv2` is a candidate-only retry, using the generic fast-loader flag.
It reuses both the original timing control and the completed `combinedv1`
stock decode/repeat, checking restored identities, source/library digests,
source/chat qualification, exact prompt conditioning and every distribution
file hash before pausing GLM. Its baseline remains sequential rather than
interleaved. The saved actual-decode stock repeat has mean KL 0.0000245179,
above the predeclared 0.00001 stability floor; a small candidate difference
will therefore remain diagnostic rather than a strict quality qualification.


### Loading time: generic mmap versus the existing production slab reader

The `combinedv1` stock arm reported 640.04 seconds for checkpoint loading,
followed by kernel/weight preparation and graph setup. The imported source
has 120 files and 76,060 tensor entries. This is not a Spark hardware lower
bound. The original production GLM container uses `GLM_FAST_LOAD=1`; its
latest logs show 194.5 GiB through the slab reader in 51.4 seconds and total
model loading in 85.49 seconds. The production checkpoint has 35 files and
148,070 tensor entries, so fewer tensors does not explain the difference.

The experimental launch whitelist had excluded `GLM_FAST_LOAD` alongside
checkpoint-specific hooks. The existing fast reader is actually a generic
byte-preserving safetensors optimization: bounded pinned slabs, asynchronous
`preadv` readers, reusable host allocations and explicit page-cache handling
avoid the slow GB10 file-backed host-to-device path. The local controller now
retains this flag and its documented public tuning options. A regression test
checks inheritance while continuing to exclude secret or unrelated settings.
The already-running `combinedv1` snapshots remain unchanged.

A CPU-only compatibility check used the actual production reader, without
copying its implementation into MxWave. All 1,601 tensors from the first,
middle and last imported shards matched stock bytes, dtype, shape and order:
1,576 slab-backed and 25 mmap pass-through tensors. It completed in 9.73
seconds. This is a reader conformance check, not an end-to-end loading-speed
measurement or full-checkpoint proof; pinned H2D was not enabled in that check.

During the planned candidate load, a five-second process/disk sample observed
about 106% of one CPU core, 893 major file faults, about 101 MiB/s of physical
reads attributed to the worker, and 270 MiB/s at the NVMe device with about
14% busy time. These observations support investigating the mmap/copy path
rather than treating the duration as a sequential SSD bandwidth limit.
They do not independently apportion the entire loading time.
