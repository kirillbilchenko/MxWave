# DFlash C1: checkpoint cost, compression cost and remaining options

**Service update, 20 October 2026, 04:46 UTC:** original production is now
restored with **500,000 context**; both CT trial containers are stopped.
See the [verified service record](GLM53_SERVING_TRIAL.md). The completion-state
descriptions below refer to 19 October and remain historical.

Completed **19 October 2026, 17:27 UTC**; independently verified on Spark/Mac.
The exact previously live candidate is restored on authenticated port 8000,
with **8,192 context**, DFlash, 4 GiB KV per rank and the existing alias.
All diagnostic containers, controller and watchdog are stopped. No new runtime
configuration was promoted. The installed vLLM package reports
**`0.1.dev20051+g487ecf187`**, verified directly in the live container; it is
not labelled `0.32.0`. Release ancestry has not been established.

The short-C1 replacement regression is mostly present before compression is
enabled. Compression reduces decode time on the same CT checkpoint, but adds
prefill latency; for these 64-token answers the two effects nearly cancel.
The traces find no C1 decode reconstruction fallback or compressed-head slowdown.

## Matched end-to-end measurements

Each arm has seven unprofiled trials per prompt length, 64 output tokens,
temperature zero and identical DFlash/common serving settings. Current ModelOpt,
uncompressed CT and compressed CT run sequentially, one load each. Throughput
is the median of trial rates; TTFT and normalized post-first-token TPOT are means.
Small differences are descriptive, with no new statistical speed qualification.

| Prompt tokens | Checkpoint / storage | Output tok/s | TTFT, ms | TPOT, ms |
|---|---|---:|---:|---:|
| 256 | Existing ModelOpt | 48.163 | 492.8 | 13.183 |
| 256 | CT, compression off | 46.118 | 464.7 | 14.619 |
| 256 | CT, compression on | 45.972 | 517.0 | 13.839 |
| 2,048 | Existing ModelOpt | 27.984 | 1,571.9 | 11.320 |
| 2,048 | CT, compression off | 29.023 | 1,390.9 | 12.875 |
| 2,048 | CT, compression on | 29.001 | 1,423.6 | 12.429 |

Isolated compression changes short/long throughput by **−0.32% / −0.08%**,
adds **52.3 / 32.7 ms TTFT**, and reduces normalized decode time by
**5.34% / 3.46%**. Replacement of current ModelOpt changes short/long throughput
by **−4.55% / +3.64%**. The long-prompt replacement gain comes from prefill;
its decode is still slower than current ModelOpt.

The older same-CT result without DFlash (+13.28% short C1) remains valid for
its own setup. It does not establish the speed of this speculative replacement.
DFlash returns token bursts; normalized TPOT is not individual token-arrival time.

## What the traces establish

One separate 256-prompt/64-output profile follows all timing in each arm.
All six rank traces have nine target FULL graph replays, each with eight rows
and one request, and **zero decode weight reconstruction kernels**. A 12-row
fallback therefore does not explain the measured C1 regression.

| Median profiled phase, ms: rank 0 / rank 1 | ModelOpt | CT off | CT on |
|---|---:|---:|---:|
| Target graph GPU span | 91.454 / 91.136 | 102.582 / 102.225 | 98.130 / 98.022 |
| Target-logits kernel sum | 3.009 / 3.131 | 3.164 / 3.208 | 2.547 / 2.511 |
| Draft-proposal kernel sum | 7.548 / 7.382 | 7.923 / 8.049 | 6.817 / 6.878 |

Compression improves both head and draft work against CT off. Candidate eager
prefill reconstructs **68 KDA matrices per rank**, totaling **38.690 / 36.519 ms**.
This is a substantial part of the TTFT penalty; one profile does not attribute
the entire 52.3 ms difference. Kernel sums and nested scopes are not additive
end-to-end latency. About 1.2 ms per trace remains unattributed and is retained.

Checkpoint metadata declares MXFP8 for all 34 input and 34 output KDA
projections in current ModelOpt; the CT source uses BF16 for these matrices.
Expert and other projection formats also differ. The traces show much more
dense BF16 projection work in CT off. This is evidence of a checkpoint-format
cost, not evidence that lossless packing degraded those BF16 weights.

Sampled clocks remain P0 at roughly 2,405–2,515 MHz, with temperatures 48–68°C.
The samples show no throttling explanation, but do not cover every instant.

## Memory and a concrete way to avoid rebuilds

Post-startup allocated CUDA memory across TP2, with equal KV budgets:

| Arm | Allocated GiB |
|---|---:|
| Existing ModelOpt | 181.688551 |
| CT, compression off | 178.362112 |
| CT, compression on | 169.006272 |

The isolated compression saving is **9.355840 GiB**. Replacement saves
**12.682279 GiB**, including checkpoint differences. These are allocated bytes,
not post-request peaks or a sum of CUDA and host memory.

**Keeping KDA dense while retaining expert-scale and vocabulary compression is
a practical untested option.** It replaces the packed KDA representation; it
does not keep an extra dense copy alongside every compressed matrix.

Per rank, KDA weights occupy **4.324707 GiB BF16 versus about 3.072 GiB packed**,
with another **0.192383 GiB** shared reconstruction pool. Replacing packed KDA
with dense KDA would add about **1.253 GiB/rank**, or **1.061 GiB/rank** if the
unused pool is also eliminated. On this storage accounting it retains about
**6.85–7.23 GiB of the 9.36 GiB TP2 saving** and removes all KDA reconstruction
work. Actual allocation/peak and latency must be measured; allocator and graph
storage can change. The existing `MXWAVE_GLM53_KDA_MODE=stock` selection offers
an implementation route while leaving expert/vocabulary compression enabled.

This gives up the fused KDA decode path, so it may trade decode speed for TTFT.
The existing CT-off arm also disables expert/vocabulary compression and is not
a measurement of this selective option. A paired trial must check outputs,
both-rank rebuild counts, startup/peak memory and end-to-end time. No selective
dense-KDA GPU trial has started at this checkpoint.
The [prepared experiment design](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/dflash-c1-v1-next-experiment-design.json)
records the three KDA flag changes, retained expert/vocabulary settings,
startup ownership checks, measurements and restoration requirement. It is a
design record, not a completed CPU preflight or launched controller.

An additional dense cache would consume the full cached weight size and retain
the compressed representation for decode. That is a different, more expensive
memory tradeoff. A fused prefill kernel could retain the full memory saving,
but [earlier measured routes](GLM53_PREFILL_ROUTING.md) do not qualify it:
256 rows miss gates, and larger tested sizes regress. Another rewrite has no
demonstrated benefit yet.

For C8/C16, the previous trial's four-running/twelve-waiting observation still
justifies a separate bounded KV 4→8 GiB/rank capacity test. More KV does not
remove prefill reconstruction. Larger actual batches may expose other verify
widths or miss current graphs; collect those traces before claiming a gain.
The 2,048-row stable-dispatch contract also prevents blindly copying production's
6,919 prefill budget. Neither cache enlargement nor long context is qualified.

## Quality, retention and service state

All 14 timed sequences per pair match across all three arm pairs. This is finite
greedy-output evidence. **Full DFlash-stack KL, PPL, broad tasks, SQNR and long
context were not measured.** Earlier same-CT non-speculative KL passes remain
separate. Repetitive diagnostic prompts favor speculation; the user's real
chat/coding harness remains necessary.

Spark/Mac recomputation agrees on **12,177 numerical fields with zero
difference**. All **75 archive payloads** verify independently on the Mac.
Raw archive `dflash-c1-v1/evidence-v1.tar.gz` is retained under the Spark
`full-model-v1` experiment root and locally at
`/private/tmp/mxwave-dflash-c1-v1/evidence-v1.tar.gz`: **14,052,856 bytes**,
SHA-256 `fc2c40a3e395b03e5201ba51cb65010a54825d4ea6d5ca5439418bea3e492b24`.

Repository evidence includes the
[compact result](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/dflash-c1-v1-compact-review.json),
[independent timing review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/dflash-c1-v1-mac-measurement-review.json),
[profile review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/dflash-c1-v1-mac-profile-review.json.gz),
[archive verification](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/dflash-c1-v1-mac-archive-review.json),
and [retention manifest](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/dflash-c1-v1-repository-retention.json).

Exact candidate container/config identities were restored. Health/authenticated
models return 200, unauthenticated models returns 401, and chat sanity passes.
Original production containers remain unchanged and stopped. The live candidate
still has **8,192 context**, not production's 500,000. See the
[service record and original-production rollback](GLM53_SERVING_TRIAL.md).

## Deployment target clarified after this run

The user wants better end-to-end numbers than the original tuned ModelOpt/DFlash
setup and accepts a modest first-token penalty. No numerical TTFT tolerance
was specified. Earlier gate decisions remain unchanged. More KV is not a C1
fix, and no KV experiment has started. Keeping dense KDA is a lower-priority
latency tradeoff because it can give up fused decode performance.

The subsequent [read-only compatibility audit](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/production-c1-audit-v1-report.json)
verifies that all four production vocabulary TP shards have exactly the same
source bytes as the qualified CT artifacts. A ModelOpt vocabulary adapter could
therefore preserve production's faster MXFP8 KDA path. The vocabulary storage
saving alone is about 0.7 GiB TP2; no C1 speed result exists for this port.
Current CT-specific hooks cannot simply be switched on for ModelOpt. NVFP4
expert-scale compression is a separate kernel/codec project because its scale
format and global-scale handling differ. No new runtime was deployed.

## Subsequent ModelOpt transfer audit

The [CPU inventory and entropy screen](GLM53_MODELOPT_SCALE_AUDIT.md) completes
on 19 October at 17:59 UTC. NVFP4 scale layouts estimate 3.25–3.30 GiB saved per
TP2 rank, about 4.1% of expert weight-plus-scale bytes. A post-Marlin consumer
prototype is now justified; no speed or quality result is available for it.

The head compatibility review corrects the initial loader-port estimate:
production's certified target head requires and retains dense BF16 weights for
exact refinement, alongside specialized target/draft paths. A packed head must
support those paths explicitly. The faster compressed head in this diagnostic
used common settings without those optimizations and does not establish a win
over the original tuned production head. No ModelOpt adapter or GPU experiment
has been started; live service remains unchanged.

## Subsequent NVFP4 component result

The [NVFP4 scale consumer](GLM53_NVFP4_COMPONENT.md) completes on 19 October at
19:03 UTC. It extends compression to the original production checkpoint rather
than converting that checkpoint to the experimental CT/MXFP4 format. The full
288-expert layer at eight rows takes 1.90% / 3.07% less time than installed stock
on the two TP partitions, with exact covered intermediate/final outputs and
passing repeat controls. Eight rows match a C1 DFlash verification shape; this
is not a measured C1 serving gain. Activations/routing are synthetic, and each
partition runs sequentially on one GPU.

Full-checkpoint encoded-scale storage can save 3.250086 / 3.252740 GiB per rank,
including metadata and raw fallback. A ModelOpt serving adapter, actual VRAM
measurement and full-model KL/end-to-end comparison remain outstanding. Preserve
production's MXFP8 KDA and specialized head/draft paths during that integration.
The live CT/DFlash service is unchanged; all component experiments are terminal.

## Projection-gap review, 20 October

The user proposed keeping MXFP4 experts while converting CT KDA to MXFP8,
porting production serving settings and retaining expert-scale packing. A
read-only reanalysis verifies all six retained raw trace hashes and compares
the existing nine eight-row target graph replays per arm/rank. These are
**common-settings ModelOpt traces, not the fully tuned production deployment**.

| CT minus ModelOpt, ms per verification step: rank 0 / rank 1 | Compression off | Compression on |
|---|---:|---:|
| Non-expert linear kernel-time sum, mean | +14.71 / +13.94 | +12.32 / +13.36 |
| Expert GEMM kernel-time sum, mean | −3.36 / −4.21 | −6.12 / −6.22 |
| Target graph GPU span, median | +11.13 / +11.09 | +6.68 / +6.89 |

The non-expert group includes dense BF16, other Marlin linears, compressed BF16
GEMMs and their reductions. It identifies the main category behind the target
decode gap. The compressed CT experts, target head and draft-proposal phases
are faster in these profiles. Kernel-time sums can overlap, and subtracting
them from a median graph span is not a prediction of end-to-end speed.
See the [retained category review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/dflash-c1-v1-projection-gap-review.json).

Source-order attribution narrows this further. The installed KDA forward runs
the merged input projection, two small gate projections, recurrent core,
gated normalization and output projection in that order. Each graph contains
34 recurrent cores. Selecting the third preceding non-expert GEMM and first
following GEMM gives 68 distinct calls, on the recurrent core's stream. This
selection covers **every compressed KDA GEMM** in all 18 candidate rank/graph
ranges. The corresponding stock CT calls are BF16 GEMMs and ModelOpt calls are
Marlin GEMMs. This is checked source-order inference, not module-labelled scopes.

| Mean main-GEMM kernel time per eight-row verification graph, ms: rank 0 / rank 1 | ModelOpt | CT off | CT on |
|---|---:|---:|---:|
| KDA merged input | 8.52 / 8.88 | 16.47 / 16.56 | 15.19 / 16.01 |
| KDA output | 4.38 / 4.36 | 7.21 / 7.30 | 6.79 / 7.17 |
| KDA input + output | 12.90 / 13.24 | 23.67 / 23.85 | 21.98 / 23.17 |
| Remaining non-expert GEMMs | 9.85 / 10.37 | 13.67 / 13.66 | 13.11 / 13.76 |

Thus KDA contributes **9.08 / 9.94 ms** of compressed CT's **12.33 / 13.33 ms**
main-GEMM gap; other linears contribute **3.25 / 3.40 ms**. Auxiliary quantization,
copies and reduction kernels are excluded from this narrower table. KDA is
about three quarters of the main projection gap in this fixture. This supports
prioritizing a KDA experiment, without claiming its whole-model result.
The [attribution summary](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/dflash-c1-v1-kda-attribution-summary.json)
and [all selected calls](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/dflash-c1-v1-kda-attribution-review.json.gz)
retain the per-graph evidence and source hashes.

MXFP8 KDA is a plausible separate quantization experiment: it would eliminate
the BF16 KDA reconstruction route if the runtime uses those MXFP8 weights
directly. It changes the target weights and needs its own CT-before/after KL
comparison. Production's use of MXFP8 does not prove that conversion passes
the unchanged gate, or that it alone closes the full gap. The subsequent
[weight-rounding diagnostic](GLM53_KDA_MXFP8_QUALITY.md) is complete and verified:
sampled PPL improves slightly, while the unchanged KL gate fails. It uses BF16
GEMMs, so actual MXFP8 serving speed remains unmeasured. No checkpoint conversion
or serving promotion has occurred.

Serving settings also require validation. Seven drafts at every concurrency
creates larger verification widths (for example, 24 rows for three requests),
beyond current fused ≤8/exact-16 coverage, and consumes more speculative state.
The 6,919-token prefill budget exceeds the 2,048-row stable-dispatch contract.
Prefix caching, KV capacity and these policies should not be copied as an
unmeasured bundle. The standard CT checkpoint format improves loader portability;
the custom lossless runtime still needs its adapters.

The completed [same-MXFP4 comparison](GLM53_MXFP4_SERVING_V3.md) kept the current
target weights and DFlash fixed, measured compression off/on on varied prompts,
and restored original production afterward. No MXFP4-versus-ModelOpt broad
quality advantage or new serving speed result is established by this trace review.

## Speed hypothesis after the quality screen

The user's 9–10 ms C1 component-saving hypothesis is consistent with the
9.08/9.94 ms main-KDA kernel-sum gap above, provided CT uses the same fast
Marlin path. Subtracting those sums from the graph-span gap suggests a possible
2–3 ms graph advantage, but mixes mean kernel sums with median elapsed spans
and excludes auxiliary costs. It is a hypothesis, not a new speed measurement.

The installed route is MXFP8 **W8A16**, also described in the
[vLLM Marlin API](https://docs.vllm.ai/en/v0.21.0/api/vllm/model_executor/kernels/linear/mxfp8/marlin/).
Removing BF16 reconstruction is plausible if actual MXFP8 weights are consumed
directly; faster FP8 prefill arithmetic is not established by this weight-only
path. First-token parity needs measurement. C8/C16 use different verification
widths and potentially different kernels, so the same absolute 9–10 ms saving
cannot be assumed. Changed target weights can also change DFlash acceptance;
target-step time alone does not predict accepted-token throughput.

Measured compressed KDA payloads are **3.29857/3.29798 GB per rank**, plus a
**0.20657 GB reconstruction pool** each. The 2,321,809,408 weights/rank would
occupy **2.39437 GB** as E4M3 values plus one E8M0 byte per 32 values. Encoded
payload reduction is therefore about **0.904/0.904 GB per rank**. Removing the
old pool could save more, but actual Marlin scale representation, layout and
workspace must be counted before claiming an allocation saving.
See the [arithmetic and source bindings](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/kda-mxfp8-speed-hypothesis-review.json).

A useful next comparison has three arms: current CT with lossless BF16 KDA,
the same CT with actual MXFP8 KDA and unchanged MXFP4 experts, and ModelOpt under
matching serving settings. Keep graphs, draft policy, admission, KV and prefix
cache policy matched; retain six-plus interleaved trials, clock/thermal data,
realized batch widths, graph replays, reconstruction counts and draft acceptance.
Measure TTFT, decode and complete-request throughput on varied prompts at the
requested concurrencies. Broader quality tests would complement, not silently
replace, the retained failed 1e-5 gate. Other linears should remain unchanged so
this comparison isolates KDA. This serving experiment has not been launched.
