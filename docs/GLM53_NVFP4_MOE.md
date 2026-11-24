# Original-production MoE internal repeat diagnostic

Stock diagnostic complete and independently verified, 19 October 2026.
Deterministic ordering now fixes captured prefill; later decode remains open.
The combined MoE/sparse-MLA follow-up is complete and independently verified.
This follows the first observed difference at
layer 3's routed-expert output in [the prior capture](GLM53_NVFP4_REPEAT.md).
There is no full-model compressed-versus-stock quality pass or serving-speed result yet.

The diagnostic retains the original production checkpoint and serving flags,
DFlash, 6 GiB KV/rank, 32 slots, 6,919-token budget and 500,000 configured context.
It uses private API port 8011 while a guarded controller and independent watchdog
restore the exact prior CT/DFlash service (port 8000, 8,192 context) on every exit.

The fixed protocol is three ordinary one-token controls before hook installation,
four identical 256-token prefill captures, 20 fixed-operand replays per operation,
and three controls with cloning disabled afterward. Layers 3–5 record routing
weights/IDs, token alignment, both Marlin GEMMs, expert sum, the routed/shared
outputs and final TP reduction. Early layer outputs and final raw logits remain
captured. Limits are 512 MiB snapshots plus 384 MiB replay fixtures per rank;
weights remain owned by the loaded model. Actual graph paths are retained for
decode; the selected prefill is eager.

Compare selected-expert sets separately from their ordering and compare weights
by expert ID. Unused alignment-buffer tails can differ without changing the
computation; inspect valid lengths before diagnosing those differences. Fixed
routing replay includes alignment, activation and expert sum. Individual GEMM
replays use captured activations/routing and fresh output/lock buffers. TP replay
runs in the same case order on both ranks.

These captures and replays are intrusive diagnostics, not latency measurements.
The unchanged full-model KL gate is 1e-5. A component repeatability result alone
does not qualify stock versus compressed, nor establish a serving improvement.

Evidence directory: `full-model-v1/nvfp4-moe-v1` on Spark, with local preparation
under `/private/tmp/mxwave-nvfp4-moe-v1`. All earlier failures remain retained.

## Stock diagnostic result

All ten API groups complete. Each of eight rank-specific captures retains
291,961,812 bytes, and replay fixtures retain 355.17 MiB/rank. Layer-3 router
logits, selected IDs and routing weights are exact in all three repeat pairs
on both ranks. The first difference is `moe_align_block_size`'s sorted-token
output. Rank 0 changes 1,637–1,660 of the 3,216 valid/padded-prefix slots;
none of its unused tail changes. All 2,048 token/expert assignments remain
present once, and expert blocks/counts match. This is a permutation of valid
work within the same expert, not changed expert selection.

The first GEMM then changes 1–4 BF16 values on the observed layer-3 pairs
(one pair/rank is exact there); the down GEMM can change too. Each GEMM with
captured token order repeats bitwise exactly in all 20 trials, as do each
routing, expert-sum and TP-reduction replay. Across both ranks and layers 3–5,
all 120 individual GEMM repeats/rank are exact. Complete fixed-routing calls
regenerate token alignment and differ on 8/18/19 of 20 trials for layers 3/4/5
on rank 0 and 6/0/16 on rank 1. This separates unstable alignment from an
intrinsically unstable fixed-operand GEMM/reduction.

Ordinary before-hook C1 self-repeat KL is 0.0501774 / 0.00692177. After-cloning
controls are 0.000598264 / 0.00410194. All layouts match. Captured repeat KL
is 0.0245199 / 0.000127003 / 0.0247797. The 1e-5 gate remains failed;
subsequent layers can add variation independently of the first observed change.

All 1,195 archive payloads verify on Mac. The independently recomputed raw
comparisons agree on 2,214 numerical fields, maximum difference 1.07e-15.
GPU replay reports are hash-verified retained measurements; they are not
re-executed on the Mac. The archive is 745,877,226 bytes, SHA-256
`57b012791df11721bc7b2ea8ac7a9541704e76fd65d22664f064fc93eaa5e32a`,
retained on rank 1 and locally under `nvfp4-moe-v1/repeat-evidence.tar.gz`.
The exact prior service was verified healthy before starting the follow-up.

## Deterministic-ordering follow-up

`nvfp4-align-v1` wraps the existing alignment output with an integer-key sort
by expert and flattened token ID. It preserves expert assignments, padding
counts, shapes and the weight/GEMM implementation. It supports this TP-only
configuration and rejects an expert map. It is enabled before graph capture.
Canonical ordering can change floating-point rounding relative to any one
native permutation; it must be used consistently in both comparison arms.

The run tests three captured repeats, 20 fixed-operand replays, stock C1/C8
four-token repeats and interleaved native/stable/native component timings.
The component timing is eager 256-row MoE work, not end-to-end serving or a
decode-speed qualification. Results and remaining limits follow below.

Evidence: [raw review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/nvfp4-moe-v1-repeat-review.json),
[Mac verification](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/nvfp4-moe-v1-mac-archive-review.json),
[restoration](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/nvfp4-moe-v1-final-check.json).

## Alignment follow-up verified result

All twelve API groups complete. Three captured prefills match bitwise in every
recorded tensor on both ranks, including final raw logits. All 840 internal
replays are exact (21 cases × 20 × two ranks). The eager 256-row expert component
adds 0.38–1.77% on the sampled layers/partitions, with all six timing controls
inside 2%. This does not measure decode or end-to-end overhead.

C1 first-token KL is zero in all before/after pairs. Later positions still vary:
before maximum 3.08012e-5, after maximum 5.29038e-4, with equal generated tokens
and matching layouts. C8 layouts differ; first-token maximum KL is 0.0433177 and
one repeat changes conditioning. Keep the failed/unmatched gates. The numerical
review independently verifies all 913 payloads, 419 raw-comparison fields
(maximum difference 3.56e-15) and six timing aggregations. GPU replay reports
are hash-verified measurements, not GPU replays performed on the Mac.
The 338,683,806-byte archive is retained on rank 1 and Mac, SHA-256
`dd333505ff76dabb1137c07baa2464367c177ec68bbca39b8c74e3f5d946e729`.
The exact prior CT/DFlash service was restored before the next test.

Evidence: [alignment review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/nvfp4-align-v1-repeat-review.json),
[Mac verification](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/nvfp4-align-v1-mac-archive-review.json),
[restoration](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/nvfp4-align-v1-final-check.json).

Reviewing the earlier evidence shows the source-pinned sparse-MLA compaction
mitigation was not enabled in the original-production launch. The next bounded
`nvfp4-inputs-v1` protocol combines that previously tested wrapper with stable
MoE alignment, observes actual target operands, and uses controlled C8 admission.
The controlled cohort is a quality diagnostic, never a speed measurement.


## Combined ordering policies: complete and independently verified

`nvfp4-inputs-v1` completes all twelve groups on 19 October at approximately
21:25 UTC. It uses the same original-production checkpoint/settings, installs
both ordering policies before capture, and records actual prepared inputs,
target graph outputs and raw logits. The observer leaves graph dispatch and
arithmetic intact. Snapshot bounds are 256 MiB/rank/group and 32 forwards.
Three C1 controls precede six captured groups (three C1, three C8), followed
by three C1 controls. Each request generates four tokens. C8 submissions are
queued under keep-pause, then resumed; they are numerical diagnostics.

Independent Spark/Mac API recomputation verifies response hashes and finds:

| Repeat set | Mean first-token KL | Mean decode KL, baseline versus repeats 1 / 2 | Layouts match |
|---|---:|---:|---|
| C1 before capture | 0 / 0 | 2.2512e-5 / 1.9068e-5 | Both |
| C1 with capture | 0 / 0 | 3.9443e-5 / 3.9002e-5 | Both |
| C8 controlled admission | 0 / 0 | 2.4696e-4 / 1.6139e-4 | First pair only |
| C1 after capture | 0 / 0 | 9.1127e-6 / 1.0239e-5 | Both |

The frozen gate uses **mean** prefill and decode KL below 1e-5, with matching
layouts and conditioning. Only one after-control passes; this does not qualify
the reference repeat set. All returned tokens and their conditioning match.
C8 first admits six requests, then two: the original 6 GiB KV allocation limits
resident requests even though 32 sequence slots are configured.

All recorded C1 prepared inputs and embeddings match exactly. Each rank/pair
has 32 exact tensors and 16 changed graph/output tensors. The first decode uses
seven rows, and the next uses two. The first auxiliary boundary already changes
14,891 / 15,414 of 28,672 BF16 elements in its two repeat pairs, maximum absolute
change 0.00146484375; both ranks agree. Read-only model-source inspection, with
matching hashes on both nodes and source modification predating the run, maps
that auxiliary result to decoder layer 5's output plus its separate mHC
reconstruction/contraction. This is an observed boundary, not a causal kernel.

For the matched C8 pair, token IDs/positions/embeddings match; physical request
slot mappings differ. Its first auxiliary output changes 39,222 of 73,728
values, with the same maximum absolute difference. The third run changes the
second decode width from 48 to 42, so that pair fails the layout control.

Next compare logical active attention/KDA state after prefill and before first
decode, accounting for request-slot mappings, then capture internal graph
boundaries in layers 0–5. Record both ordering adapters' actual call paths at
capture time too. The present observer does not capture saved states; equal
explicit inputs do not prove equal complete graph state. No compressed
comparison or speed claim follows from this diagnostic.

All 895 retained payloads independently verify on Mac. Recomputations agree on
5,934 numerical fields, maximum difference 1.56e-15. The 195,674,929-byte archive,
SHA-256 `b6780914da56d844147f7173a5d724259ee6a6faf558c73499cd5dcce0117d4b`,
is retained on rank 1 and locally under `nvfp4-inputs-v1/repeat-evidence.tar.gz`.
The source map is a separately retained post-run read-only audit. The original
archive and frozen experiment source are unchanged.

Controller status is **complete_restored**. Exact prior CT/DFlash identities
are healthy on port 8000 with 8,192 configured context; authenticated models
returns 200, unauthenticated models 401. Original production remains stopped
and unchanged. No GPU experiment remains active. Local validation remains
844 passing tests / one skipped, strict core mypy passing, and Ruff passing.

Evidence: [full comparison](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/nvfp4-inputs-v1-repeat-review.json),
[unchanged gate](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/nvfp4-inputs-v1-qualification-review.json),
[Mac verification](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/nvfp4-inputs-v1-mac-archive-review.json),
[restoration](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/nvfp4-inputs-v1-final-check.json),
[auxiliary source map](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/nvfp4-inputs-v1-auxiliary-source-map.json).


Temporary rank-0 payload copies from the alignment and input runs were removed
only after both complete archives verified on rank 1 and Mac. All source binaries
were rehashed before removal: 426 / 414 files, recovering 875,885,436 /
165,485,181 bytes respectively. Raw evidence remains in each rank-1
`retained-combined` directory and Mac `evidence` directory; rank 0 keeps the
manifests and receipts. Post-cleanup health is 200 at 21:32:40 UTC, with about
6.39 GiB free on rank 0. No new model experiment was started.
