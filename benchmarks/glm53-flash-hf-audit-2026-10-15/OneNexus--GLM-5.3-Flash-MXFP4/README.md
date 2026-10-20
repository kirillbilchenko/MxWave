---
language:
- en
- zh
library_name: transformers
license: mit
pipeline_tag: image-text-to-text
base_model:
- zai-org/GLM-5.3-Flash
tags:
- glm5_next
- quark
- mxfp4
- rocm
- sglang
- speculative-decoding
---

# OneNexus/GLM-5.3-Flash-MXFP4

An MXFP4 GLM-5.3-Flash checkpoint for text and image-to-text serving on AMD
Instinct MI355X (gfx950). On matched text-reasoning samples against
[amd/GLM-5.3-Flash-Quark-MXFP4](https://huggingface.co/amd/GLM-5.3-Flash-Quark-MXFP4),
it scores **+1.0 percentage point on MMLU** and **+3.5 points on GPQA
Diamond**, with **−0.2 point on GSM8K**. These are observed differences on
fixed samples, not statistically established superiority or a claim about
vision, safety, or general agentic quality.

Runtime support: [SGLang PR #39273](https://github.com/sgl-project/sglang/pull/39273).

## Model at a glance

| Property | Value |
| --- | --- |
| Base model | [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash), 320B total / 18B active parameters |
| Weight format | MXFP4 routed experts in most layers; declared precision retained for other tensors |
| Indexed tensor payload | 195,550,624,632 bytes (182.12 GiB) |
| Model context limit | 1,048,576 tokens; the reported AgentX runs used a 262,144-token benchmark cap |
| Tested serving stack | 4× MI355X VF (gfx950), SGLang on ROCm, TP4/EP4, FP8 KV cache, in-checkpoint MTP with EAGLE 5/1/6 |
| Input / output | Text and image → text; image request and cached follow-up tested, video not evaluated |

## Text-reasoning quality

Both checkpoints answered the same fixed questions with one greedy attempt per
question. A capped or unfinished response counts as incorrect.

| Sample | OneNexus correct | AMD correct | Observed accuracy difference |
| --- | ---: | ---: | ---: |
| GSM8K (500 questions) | 490 (98.0%) | 491 (98.2%) | −0.2 pp |
| MMLU (500 questions) | 428 (85.6%) | 423 (84.6%) | **+1.0 pp** |
| GPQA Diamond (198 questions) | 139 (70.2%) | 132 (66.7%) | **+3.5 pp** |

![Matched GSM8K, MMLU, and GPQA Diamond accuracy for OneNexus and AMD, with 95% Wilson intervals and paired disagreement counts.](https://huggingface.co/OneNexus/GLM-5.3-Flash-MXFP4/resolve/main/assets/checkpoint-quality-comparison.png)

MMLU and GPQA results are sensitive to the answer-length limit: MMLU has
43/500 truncated outputs for each checkpoint, and GPQA has 55/198 for
OneNexus versus 61/198 for AMD. The paired exact-test p-values are 0.458 for
MMLU and 0.265 for GPQA; neither difference reaches p < 0.05. The samples
neither prove superiority nor equivalence. See the
[evaluation protocol](https://huggingface.co/OneNexus/GLM-5.3-Flash-MXFP4/blob/main/reproduction/quality-protocol.md)
and [per-question aggregate data](https://huggingface.co/OneNexus/GLM-5.3-Flash-MXFP4/blob/main/assets/checkpoint-quality-comparison.json)
for settings, truncations, intervals, and paired discordances. These are
fixed-subset text-reasoning evaluations, not full benchmark leaderboard scores.

## Serving performance against AMD

The checkpoints were replayed on the same four MI355X VFs with the same
SGLang build, tokenizer and prompt rendering, TP4/EP4 placement, FP8 KV
cache, EAGLE 5/1/6 settings, cache policy, 393-entry AgentX corpus, and
900-second measurement per concurrency level. Each checkpoint used its own
declared target and MTP weight precision.

| Concurrency | OneNexus output tok/s | AMD output tok/s | OneNexus P90 tok/s/user | AMD P90 tok/s/user |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 261.76 | 258.89 | 379.08 | 364.96 |
| 2 | 444.88 | 444.09 | 333.60 | 330.21 |
| 4 | 764.52 | 769.09 | 295.25 | 297.03 |
| 8 | 1,283.08 | 1,278.38 | 241.23 | 237.75 |
| 16 | 1,845.35 | 1,857.38 | 178.88 | 179.39 |
| 32 | 2,279.72 | 2,290.16 | 110.35 | 110.00 |
| 48 | 2,331.22 | 2,340.38 | 74.85 | 73.89 |

Across c1–c48, OneNexus is **0.1% lower** in geometric-mean aggregate
output throughput and **1.0% higher** in geometric-mean P90 per-user output
rate. This is a measured performance-parity result, not a throughput win.
Both runs have zero request errors (9,223 OneNexus and 9,264 AMD requests).
P90 tok/s/user is the 90th percentile of each eligible request's reciprocal
mean inter-token latency, not aggregate throughput. The
[OneNexus](https://huggingface.co/OneNexus/GLM-5.3-Flash-MXFP4/blob/main/assets/checkpoint-agentx-onenexus.json)
and [AMD](https://huggingface.co/OneNexus/GLM-5.3-Flash-MXFP4/blob/main/assets/checkpoint-agentx-amd.json)
summaries contain the full rung-level evidence.

## How the serving path works

### Sparse attention and K-pool selection

GLM-5.3-Flash uses 256-wide sparse-attention projections with no RoPE tail.
The separate K-pool index cache stores 128 FP8 index-key values and one FP32
scale (132 bytes per row), distinct from the 512-wide MLA attention cache.
The native gfx950 path reads and writes the same preshuffled index layout.
It also keeps selection bounded at long context without dropping valid
positions: 512 selected four-token groups cover up to 2,048 history tokens,
and up to three recent unpooled tokens bring the mapped width to 2,051.

![Zero-RoPE sparse attention, compact K-pool index rows, and bounded selection on gfx950.](https://huggingface.co/OneNexus/GLM-5.3-Flash-MXFP4/resolve/main/assets/dsa-kpool-mechanism.png)

### Verification compute and EAGLE

Target verification runs AITER MXFP4 experts, supported gfx950 mHC, and
shape-scoped fused KDA, with reference fallbacks where required. Expert
execution retains the model's SwiGLU gate and clamp semantics. Five steps
of the in-checkpoint MTP head propose tokens, then one target pass verifies
them. Graph replay reduces repeated launch work; KV and Mamba state must
still refer to the same visible committed prefix after rejection or
stop-token truncation. [SGLang PR #39273](https://github.com/sgl-project/sglang/pull/39273)
contains the implementation and correctness tests.

On the c16 AgentX rung, acceptance length is 4.9072 and acceptance rate is
78.14% over 1,595 completed requests, as recorded in the
[serving results](https://huggingface.co/OneNexus/GLM-5.3-Flash-MXFP4/blob/main/assets/mxfp4-agentx-frontier.json).

## Measured optimization ladder: from KDA traces to kernel selection

The c16 serving capture puts the kernel work in context: target verification
contains 535.5 ms of MoE, 112.6 ms of mHC, and 91.4 ms of KDA device-event
residence per rank, averaged across four ranks. MoE is the largest of these;
KDA is a smaller, concrete opportunity to reduce repeated launches and work.

The following **isolated MI350X VF / gfx950 operator study** shows how the
KDA implementation was tuned. It uses synthetic GLM TP4-shaped tensors,
not a checkpoint comparison or a new MI355X serving run. The first example
has eight requests and eight verification tokens per request.

1. **Observe two kernels.** The reference graph launches causal convolution,
   then gating and delta-rule recurrence. Its paired operator time is
   about **38.9 μs**.
2. **Fuse, then inspect the result.** The initial fused kernel removes the
   intermediate launch, but a four-value tile (`BV=4`) recomputes the Q/K
   convolution for each of 32 value tiles. Its trace has one kernel per
   replay, yet paired time rises to **41.7 μs**. Launch count alone is not
   sufficient evidence of an improvement.
3. **Reduce duplicated work.** A 16-value tile (`BV=16`) shares that work
   across more values, reducing the value-tile grid from 32 to 8. The
   trace still has one kernel per replay, now visibly shorter; separately
   measured graph time falls to **28.4 μs**, **27.0% less time** than the
   reference.

![Recorded KDA GPU timelines show the two-kernel reference, slower initial fusion, and tuned fusion; separate paired graph timings quantify the improvement.](https://huggingface.co/OneNexus/GLM-5.3-Flash-MXFP4/resolve/main/assets/kda-trace-optimization.png)

The timelines retain actual event durations and gaps from the first three
of eight captured replays. The timing bars come from a separate, unprofiled
graph microbenchmark: three reference/fused/fused/reference (ABBA) repeats,
with fixture clones excluded. They are operator measurements, not a
percentage to add to the AgentX throughput curve.

### Check the serving shape and keep the fallback

EAGLE 5/1/6 uses six-token verification. The matched six-token sweep below
confirms the direction at seven batch sizes: **31.50 → 23.02 μs at batch 8**
and **37.66 → 33.03 μs at batch 16**. The smaller batch-16 gain shows why
dispatch must depend on shape. Batch 1–2 retain the four-value tile; the
qualified larger batches use the 16-value tile.

![Seven measured six-token KDA shapes, with reference and selected fused operator times and the full range of three paired repeats.](https://huggingface.co/OneNexus/GLM-5.3-Flash-MXFP4/resolve/main/assets/kda-six-token-sweep.png)

The PR reuses the existing fused KDA path only for gfx950 batches 1–16,
six or eight verification tokens, 16 local heads, 128-wide states, BF16
activations, FP32 convolution weights, no convolution bias, and the −5
gate bound. Other ROCm shapes retain the reference path. The same study
checked three changed-input graph replays for every measured shape:
active output and convolution buffers matched exactly, while intermediate
SSM states passed an absolute tolerance of 0.004.

The [measurement data and kernel-event excerpts](https://huggingface.co/OneNexus/GLM-5.3-Flash-MXFP4/blob/main/assets/kda-trace-optimization.json)
retain every ABBA timing, including slower tile choices. The
[study protocol](https://huggingface.co/OneNexus/GLM-5.3-Flash-MXFP4/blob/main/reproduction/kda-trace-study.md)
pins hardware, source hashes, the trace extraction, and the relationship to
the PR kernel. This is a measured kernel-tuning ladder; the complete
checkpoint-level FP8/MXFP4 and B300-estimate comparisons follow below.

## Additional measured tuning choices

**mHC backend selection.** GLM-5.3-Flash mixes four residual streams across
4,096 channels around its attention and MLP blocks. With AITER enabled on
gfx950, the runtime selects its HIP kernels for this geometry. The saved
32-token traces show shorter projection/squared-sum, mixing, and post kernels,
with the same launch count and caller RMSNorm. Paired graph time falls from
**30.60 to 10.01 μs** for pre plus RMSNorm and **9.66 to 2.03 μs** for post.
Both routes pass the same numerical oracle.

**Vision merge as a matrix multiply.** The downsample has kernel size equal
to stride (2×2), no padding, and one group: each patch is an independent
linear projection. The existing `Conv2dLayer` uses unfold + linear for that
case, bypassing MIOpen convolution search. On the model's noncontiguous
`[256, 1024, 2, 2]` merge layout, the warmed trace replaces the
transpose/convolution/add sequence with a layout copy and a bias-enabled
matrix multiply. Fresh-process first-call time falls from **4.501 to
3.307 seconds**; warmed graph time falls from **63.28 to 25.62 μs**.
BF16 output passes the same numerical threshold.

![Paired MI355X operator measurements for mHC backend selection and the vision downsample implementation.](https://huggingface.co/OneNexus/GLM-5.3-Flash-MXFP4/resolve/main/assets/mhc-vision-operator-study.png)

These are isolated operator measurements, not percentages attributed to the
AgentX serving curves. The [study protocol and raw traces](https://huggingface.co/OneNexus/GLM-5.3-Flash-MXFP4/blob/main/reproduction/mhc-vision-operator-study.md)
and [machine-readable results](https://huggingface.co/OneNexus/GLM-5.3-Flash-MXFP4/blob/main/assets/mhc-vision-operator-study.json)
pin the environment, repeats, errors, and source revision.

## Serving frontier versus supplied B300 estimates

The green MI355X curve below is **measured** on four gfx950 VFs with this
checkpoint. The blue B300 curve consists of **supplied estimates** for four
B300 GPUs; we did not reproduce B300 runs or obtain their raw artifacts.
The disclosed corpus, 256K context cap, GPU count, and EAGLE shape align,
but hardware, DSA backend, and serving installations differ. This is a
frontier illustration, not a controlled cross-hardware speedup claim.

![Measured four-MI355X OneNexus frontier versus supplied four-B300 estimates, with throughput and per-user interactivity on separate axes.](https://huggingface.co/OneNexus/GLM-5.3-Flash-MXFP4/resolve/main/assets/final-agentx-frontier.png)

| Concurrency | MI355X output tok/s (measured) | MI355X P90 tok/s/user (measured) | B300 output tok/s (estimate) | B300 P90 tok/s/user (estimate) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 262.19 | 380.35 | 219 | 349.2 |
| 2 | 441.34 | 337.36 | 398 | 298.9 |
| 4 | 768.45 | 294.51 | 695 | 264.6 |
| 8 | 1,269.90 | 238.61 | 964 | 198.3 |
| 16 | 1,831.06 | 176.74 | 1,446 | 148.0 |
| 32 | 2,175.11 | 106.55 | 1,680 | 94.0 |
| 48 | 2,355.83 | 73.01 | 1,721 | 90.3 |

At c48, the measured MI355X arm has 36.9% higher aggregate output rate
but 19.1% lower P90 per-user rate than the supplied B300 estimate. The measured
MI355X c64 point is dominated by its c48 point, so it is shown as a
diagnostic in the figure but not a frontier row or claim. Across eight
MI355X rungs, 11,810 requests completed with zero errors. The
[measured results](https://huggingface.co/OneNexus/GLM-5.3-Flash-MXFP4/blob/main/assets/mxfp4-agentx-frontier.json),
[supplied B300 values](https://huggingface.co/OneNexus/GLM-5.3-Flash-MXFP4/blob/main/assets/b300-agentx-estimates.json),
and [figure renderer](https://huggingface.co/OneNexus/GLM-5.3-Flash-MXFP4/blob/main/reproduction/render_agentx_frontier.py)
are available.

## Serving uplift versus the official FP8 checkpoint

Both curves here were measured with SGLang source
`5fd4387a753cdc88c0c87efb1bcabab1efabfc93` on the same image, GPU set,
tokenizer, TP4/EP4 placement, FP8 KV policy, EAGLE 5/1/6 shape, graph
limit, cache reset, 393-entry AgentX corpus, and 900-second rungs. The
complete checkpoints and their declared MTP precision differ; this does
not isolate quantization-kernel speed. This is a separate matched replay
from the AMD comparison above, so its OneNexus rates differ slightly.

![Official FP8 and OneNexus MXFP4 measured throughput-versus-interactivity frontiers on four MI355X VFs.](https://huggingface.co/OneNexus/GLM-5.3-Flash-MXFP4/resolve/main/assets/fp8-mxfp4-agentx-frontier.png)

| Concurrency | Official FP8 output tok/s | OneNexus MXFP4 output tok/s | Output uplift | Official FP8 P90 tok/s/user | OneNexus MXFP4 P90 tok/s/user | P90 uplift |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 247.67 | 262.19 | +5.9% | 339.50 | 380.35 | +12.0% |
| 2 | 419.27 | 441.34 | +5.3% | 291.18 | 337.36 | +15.9% |
| 4 | 714.35 | 768.45 | +7.6% | 251.81 | 294.51 | +17.0% |
| 8 | 1,099.33 | 1,269.90 | +15.5% | 191.91 | 238.61 | +24.3% |
| 16 | 1,547.01 | 1,831.06 | +18.4% | 145.77 | 176.74 | +21.2% |
| 32 | 1,859.89 | 2,175.11 | +16.9% | 86.37 | 106.55 | +23.4% |
| 48 | 1,986.75 | 2,355.83 | +18.6% | 63.80 | 73.01 | +14.4% |

Over c1–c48, MXFP4 gives **+12.4% geometric-mean** aggregate output rate
and **+18.2% geometric-mean** P90 per-user rate. Both arms finished with
zero request errors (9,734 FP8; 11,810 MXFP4 across all eight measured
rungs). Indexed tensor payloads are 305.78 GiB for FP8 and 182.12 GiB for
MXFP4, a 40.44% checkpoint-file reduction, not a peak-memory comparison.
With the same memory-allocation policy, measured total weights use 77.69
versus 46.99 GB/GPU and device KV capacity rises from 13.97M to 17.32M
tokens. The [FP8](https://huggingface.co/OneNexus/GLM-5.3-Flash-MXFP4/blob/main/assets/fp8-agentx-control-summary.json)
and [MXFP4](https://huggingface.co/OneNexus/GLM-5.3-Flash-MXFP4/blob/main/assets/mxfp4-agentx-frontier.json)
summaries, [matched protocol](https://huggingface.co/OneNexus/GLM-5.3-Flash-MXFP4/blob/main/reproduction/fp8-mxfp4-agentx-protocol.md),
and [figure renderer](https://huggingface.co/OneNexus/GLM-5.3-Flash-MXFP4/blob/main/reproduction/render_fp8_mxfp4_frontier.py)
contain the full per-rung evidence. Dominated c64 observations are visible
in the figure for context but excluded from the reported frontier uplift.

## Reproduction

The [serving recipe](https://huggingface.co/OneNexus/GLM-5.3-Flash-MXFP4/blob/main/reproduction/public-serving-recipe.md)
specifies the checkpoint revision, container, source, tokenizer, and launch
arguments. The AgentX measurements use SGLang source
`5fd4387a753cdc88c0c87efb1bcabab1efabfc93`; the recipe and
[qualification record](https://huggingface.co/OneNexus/GLM-5.3-Flash-MXFP4/blob/main/reproduction/current-source-validation.json)
identify the source used for serving qualification separately. Quality runs
use BF16 KV with graphs and speculation disabled; see the evaluation protocol
for the evaluator, inputs, and request policy. Raw restricted prompts are not
redistributed.

## License

[MIT](https://huggingface.co/OneNexus/GLM-5.3-Flash-MXFP4/blob/main/LICENSE),
with source attribution in [NOTICE](https://huggingface.co/OneNexus/GLM-5.3-Flash-MXFP4/blob/main/NOTICE).
