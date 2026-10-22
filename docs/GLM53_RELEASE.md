# MxWave GLM portable recipe v0.1.1

An **experimental, versioned recipe** for GLM-5.3 Flash on two DGX Sparks using
the official vLLM 0.32.0 image and an opt-in MxWave plugin. No vLLM source files
are replaced. The runtime depends on pinned internal APIs and SM121 native
libraries; compatibility with other vLLM versions or GPU families is unqualified.

[Download release](https://github.com/kirillbilchenko/MxWave/releases/tag/glm53-v0.1.1)
· [Installation and launch](../deployment/glm53/README.md)
· [Native build sources](../deployment/glm53/native/README.md)

## What is released

- Python wheel with plugin/CLI, exact checkpoint revisions, source contracts,
  full target-shard hashes, metadata and licenses.
- Qualified ARM64/SM121 native binaries with build scripts and source pins.
- Pinned Dockerfile, CPU asset preparation, launch/stop instructions and checksums.
- Results below, with the historical failures retained. No new speed claim.

Public inputs are Intel AutoRound's CT MXFP4 checkpoint and Modal's pretrained
BF16 DFlash drafter. CPU preparation reproduces the exact GPU-tested vocabulary
files and FP8 drafter; no training/private experiment directory is needed. KDA
converts to MXFP8 during loading. Expert-scale/embedding packing is lossless;
**the complete profile is not lossless** because KDA and drafter conversion are
lossy. MxWave contributes the codecs, adapters, diagnostics and serving recipe,
not the AutoRound quantization or pretrained drafter.

The release was checked with a CPU-only clean build of the pinned official image:
four packed vocabulary files and the prepared drafter match their qualified
hashes exactly. Installation, upstream source contract, complete target inventory
and generated serving commands are checked separately. This packaging check does
not repeat or supersede the existing GPU qualification.

## Stock-vLLM recipe: observed results

21 October 2026, two DGX Sparks, TP2; three interleaved paired trials, realistic
short code/chat prompts, 256 outputs. Comparison is against tuned custom production
ModelOpt/Inco, not the same runtime/checkpoint/drafter. CT uses Modal with 5/3 drafts.

| Workload | Production tok/s | Stock CT tok/s | Observed change |
|---|---:|---:|---:|
| Code C1 | 51.04 | 39.95 | −21.72% |
| Chat C1 | 44.73 | 35.11 | −21.51% |
| Code C8 | 83.29 | 95.85 | +15.08% |
| Chat C8 | 78.59 | 95.33 | +21.31% |

C8 numbers are aggregate throughput. C8 mean first token falls from roughly
4.8–5.0 s to 1.64 s; individual completion latency increases roughly 1–2 s as CT
runs eight requests together rather than six plus two queued. Several repeat
controls fail the 2% spread threshold; medians are descriptive, not stable-gain
guarantees. The 300-question speculation-off GSM8K screen scores **289/300
production vs 287/300 CT**, without API errors/truncation.

Loaded allocation: **87.202210 / 87.202180 GiB per CT rank** including drafter,
KV and graphs. Lossless packing saves **6.94113 GiB across TP2** in resident
scale/embedding storage after expert scratch space. This is not a newly measured
whole-service allocation delta against production. Full ranges and retained
evidence are in the dated [handoff](GLM53_HANDOFF.md).

## Frozen live setup: a separate comparison

22 October, same custom production runtime, Inco drafter and serving settings in
both arms: 500k configured context, 32 slots, 6 GiB KV/rank, 6,919 prefill budget,
prefix cache, actual 4 ms admission, original graphs/draft policy.

| Workload | Production median [range] | CT median [range], tok/s |
|---|---:|---:|
| Code C1 | 46.01 [39.48–51.96] | 49.35 [46.68–55.75] |
| Chat C1 | 39.06 [36.83–39.56] | 45.11 [43.77–46.77] |
| Code C8 | 74.35 [71.34–81.73] | 86.29 [85.73–90.79] |
| Chat C8 | 72.89 [72.84–77.20] | 83.71 [81.53–86.42] |

All eight arm/cell repeat controls exceed 2%. Quality: **291/300 production vs
287/300 CT**, speculation off; seven production-only and three CT-only correct
answers. The small sample establishes neither general degradation nor equivalence.
These are different quantized checkpoints, not isolated compression effects.

Whole-service allocation: production **96.838109 GiB/rank**; CT
**89.605322 / 89.605072 GiB**. Saving **14.465825 GiB TP2 (7.47%)**, exactly repeated
on three CT starts and the live handover. This includes quantization/checkpoint
differences; do not add the 6.941 GiB packing number or earlier experiments.
[Full report](GLM53_SAME_RUNTIME_RESULTS.md).

The user explicitly selected this CT setup for live use despite the unchanged
gates. It remains on the original custom image; publishing the stock recipe does
not migrate it. Its configuration, native/overlay inputs and both copies of all
120 target shards are frozen and verified, with original containers retained for
[rollback](GLM53_CT_LIVE.md). Large payloads are hash-verified in place; small
runtime files are copied into a content-addressed snapshot on both nodes and Mac.

## Limits and credit

Strict KL, output identity and speed-repeatability failures remain recorded.
500k is a startup capacity check, not a long-context retrieval qualification.
No full production GSM8K-1,319 result, new PPL/SQNR claim or universal speedup is
introduced. The earlier full CT GSM8K score belongs to its earlier tested profile;
it cannot substitute for this release's paired subset.

Credit Z.AI for GLM, Intel AutoRound for the CT checkpoint, Modal and Inco for
their respective pretrained drafters, vLLM/Marlin and ZipServ for licensed native
components. The release's notices include licenses, pinned revisions and MxWave
modifications. No DeepInfer code is included. The strongest supported publication
is an engineering case study of measured memory savings, runtime-dependent speed,
speculation and determinism—not a universally faster, lossless model announcement.
