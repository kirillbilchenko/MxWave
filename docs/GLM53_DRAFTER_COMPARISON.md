# Drafter comparison after completed speed and GSM8K

<!-- SELECTOR_CALIBRATION_LIVE_BEGIN -->
**20 October 2026, 12:19 UTC: authorized transfer complete; CT capture is starting.**
The source archive and all 17 source/configuration hashes verify on both Sparks.
CPU preflight passes with all 211 frozen runtime bindings per rank. Queue
`1950880`, controller `1953173`, experiment `selector-calibration-capture-v1`.
The controller is loading CT on private port **8018** after pausing original
production; its watchdog and restoration path are active. No weights are trained
or promoted. Production will be restored before CPU training on the Mac.

The [selector pilot](GLM53_SELECTOR_CALIBRATION.md) uses 32 new training prompts,
eight validation and eight sealed test prompts; only the 1,048,576-weight BF16
selector projection may train. Completed work is committed as `edc4c08` and
backed up with verified Git restoration on **spark2**; pilot source is committed
as `d945612`. **This is not a deployment switch.** The four user-specified
[production replacement gates](GLM53_SELECTOR_CALIBRATION.md#production-replacement-gates)
remain open: drafter recovery, production-equivalent tuning/capacity, stable
C8/C16 comparisons, and long-context qualification beyond the 8k trials.
This block supersedes historical live-work statements below.
<!-- SELECTOR_CALIBRATION_LIVE_END -->

<!-- DRAFTER_CONTEXT_LIVE_BEGIN -->
**Paired target/drafter diagnostic complete and independently verified,
20 October 2026. Original production restored at 11:53:12 UTC**, healthy on
port 8000 with 500,000 context and exact original identities/configurations.
No candidate is promoted; no experiment or training remains running.

Both CT MXFP8 and matched ModelOpt complete before/capture/after 32-token code
requests. All six outputs match the retained continuation; each arm's counters
and stream chunks repeat. **All 88 recorded learned/derived drafter tensors per
rank match between arms.** At the same five generated code tokens, CT proposes
the wrong seventh draft token while ModelOpt proposes the correct one. Both
targets' argmax predictions agree on all seven comparable verification rows:
**six drafts accepted for CT, seven for ModelOpt**. The correct token is already
in CT's top-16 list, at rank 3; its selected token has a 3.25 higher conditional
score. Target features and draft hidden states differ despite matched accepted
history and batch shapes.

This supports a mismatch between the drafter and CT's target features. It
identifies one early proposal error; it does not explain the entire 8.61% code
speed gap, uniquely isolate a target layer, or demonstrate a fix. The head
comparison below remains exact, and the first four CT draft states also match
that earlier diagnostic bit for bit on both ranks. A scalar selector-strength
screen does not repair this early miss; its other results are offline hypotheses,
not measured serving gains. The next useful candidate is drafter-only calibration
on separate CT-generated data, followed by held-out acceptance and serving checks.
No target precision change or training has been applied.

All **126 archived payloads** verify; **1,922 numerical fields** agree across
Spark/Mac with maximum absolute difference **2.24e-13** (tolerance 1e-11).
All 16 retained raw-logit/native-sampling checks pass. Repository checks:
**873 passed, 1 optional-dependency skip**, Ruff and strict core mypy pass.
[Verification receipt](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/drafter-context-v1-independent-verification.json)
and [paired tensor review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/drafter-context-v1-mac-review-v2.json).
Local evidence: `/private/tmp/mxwave-drafter-context-v1-final/evidence`; remote:
`full-model-v1/drafter-context-v1`. Queue 1675212 is complete. This checkpoint
supersedes older live-service statements. The earlier 1e-5 full-model KL failure
is unchanged; this diagnostic supplies no new task-quality or speed qualification.
<!-- DRAFTER_CONTEXT_LIVE_END -->


The paired capture retains native FULL graph execution, actual target sampling,
five target feature layers (5, 14, 24, 33, 42), combined context features, draft
hidden states, top-16 candidates and conditional scores. Each arm retains eight
replays per rank, including post-cap lookahead work. The matched anchors are:

| Generated code prefix | CT accepted drafts | ModelOpt accepted drafts | Proposal comparison |
| --- | ---: | ---: | --- |
| 1 token | 2 | 2 | All seven proposals identical |
| 4 tokens | 0 | 0 | All seven proposals identical |
| 5 tokens | 6 | 7 | Only seventh proposal differs |

At the last anchor CT proposes token 5461; ModelOpt proposes 20057. Both targets
choose 20057 at the corresponding common verification row. Their raw target-logit
margins over 5461 are 219 and 213 respectively on rank 0. This is a draft proposal
miss on the recorded common prefix, not a disagreement about the desired next
token at that position. All 88 recorded drafter parameters/derived weights match
per corresponding TP rank, including the context projection and selector. This
is stronger than the earlier checkpoint-file inventory, but is not a proof that
every possible runtime buffer or execution path is identical.

Feature differences already appear at the first recorded target boundary.
They do not identify which quantized projection/expert or preceding operation
contributes most. No individual layer has been restored to higher precision.
Changing precision could also change target behavior and memory requirements;
the evidence does not yet justify that route.

The separate [fixed-state selector screen](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/drafter-selector-fixed-state-screen-v1-review.json)
uses 52 code and 71 explanation states from the verified head archive. Scaling
the selector correction from 1.0 to 0.25 increases summed matching prefixes
197 → 200 on code and 184 → 196 on explanation. None of the tested factors
(0, 0.25, 0.5, 0.75, 1, 1.25, 1.5) fixes the early seventh-token miss. These
fixed-history counts cannot predict end-to-end acceptance: a changed proposal
changes later request states and round boundaries. No runtime setting is changed.

A reasonable next pilot would freeze the target and adapt a limited part of
the drafter on CT's own feature/verified-token pairs. Use separate calibration
prompts and held-out code/chat/math; do not fit the speed prompts or GSM8K test
set. Start by assessing a small context-projector/selector adjustment, then
consider broader fine-tuning only if needed. Preserve the target weights and
verification algorithm, and still check generated outputs and real serving speed.
A successful component fit would not alone establish production equivalence.

Quantized-target adaptation has precedent in the
[Qwen3.5 drafter adaptation report](https://arxiv.org/abs/2607.04244), and
[SpecForge's DFlash2 training documentation](https://sgl-project.github.io/SpecForge/basic_usage/training.html#dflash2)
describes training from captured features and separate selector objectives.
These support the direction, not a ready-made GLM integration or a speed estimate.
Compatibility with this installed DFlash2 checkpoint/runtime and the pilot's
compute/data requirements remain to be audited. No external training code was
copied and no training process has been started.

<!-- DRAFTER_HEAD_LIVE_BEGIN -->
**Drafter head diagnostic complete and independently verified, 20 October 2026.**
Compressed and exact reconstructed heads produce identical results on every tested
C1 draft state: **129 graph replays/rank, 903 proposed tokens, zero changes** to
raw projection bits, top-16 candidates, selector scores or proposals. The 38
retained raw-logit samples cover **20,599,040 elements** across two ranks with zero
bit differences. All six before/shadow/after requests preserve outputs and
speculative counters, also matching the earlier speed-test continuations.
This clears the head on these states; it does not establish identical whole-model
arithmetic or explain every acceptance difference.

All **580 archived payloads** verify; the separate corrected Spark/Mac reviews
match exactly. The original failed literal replay-count control is retained:
125 verified rounds plus four post-response-cap lookahead replays were captured.
Captured positions reconstruct all 389 accepted tokens and all 14 position
counters exactly. No capture was removed. V1's CPU preparation failure and V2's
stop before measurement are retained. Instrumented timings are not speed claims.
[Verification receipt](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/drafter-head-v3-final-independent-verification.json).

**Original production is restored**, healthy on port 8000 with 500,000 context,
exact container/configuration identities verified at **11:22:59 UTC**. No
candidate is promoted. The subsequent paired CT/ModelOpt capture is complete; its findings and
current service state are recorded above. No target precision change or
drafter retraining has been applied.
Local evidence: `/private/tmp/mxwave-drafter-head-v3-final/evidence`; remote:
`full-model-v1/drafter-head-v3`. This block supersedes earlier live-run statements.
<!-- DRAFTER_HEAD_LIVE_END -->

**Common-prefix evidence:** all six trials per mode repeat the same stream
chunk pattern. At generated code offset 5, with identical preceding tokens and
prior batch boundaries, CT accepts six drafts while ModelOpt accepts seven.
The code outputs first differ at index 107. CT stream-derived boundaries also
match the independently captured GPU proposal positions. This rules out later
output divergence as the entire explanation; it does not yet separate changed
draft proposals from changed target verification. Terminal truncated chunks
are excluded from the inference.
[Reproducible audit](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/drafter-common-prefix-v1-review.json).

**CPU source identity check:** the current ModelOpt production checkpoint's BF16
head and embeddings match CT's prepared source bits exactly for all four TP2
shards, independently on both nodes. This rules out a difference in these
checkpoint weights, while leaving their execution methods and target hidden
states to test. It is not a retroactive runtime-tensor identity claim.
[Rank 0 audit](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/vocab-source-identity-v1-r0.json),
[rank 1 audit](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/vocab-source-identity-v1-r1.json).



<!-- DRAFTER_FOLLOWUP_BEGIN -->
**Post-benchmark drafter audit: initial read-only collection complete.** Matched speculative configurations equal: `True`. Audited draft weight files identical: `True`. Source hashes, shared-head methods and per-position acceptance are retained in [the audit receipt](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/drafter-comparison-v2-review.json). The final profile interpretation and the completed head diagnostic are recorded above.
<!-- DRAFTER_FOLLOWUP_END -->

**Next diagnostic inputs prepared.** The
[retained replay inputs](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/drafter-replay-v1-inputs.json)
bind the original code/explanation prompts and all three models' 256-token
continuations to the completed speed evidence. Each continuation repeats in
all six trials within its mode. CT MXFP8 and ModelOpt first diverge at output
index 107 for code and 16 for explanation.

The head check is now complete and exact on the tested states. The completed paired
comparison retains the five target feature boundaries used by DFlash
(layers 5, 14, 24, 33 and 42), the accepted-prefix metadata and actual draft
proposals from CT and ModelOpt. Compare only aligned token histories and batch
shapes before changing precision. Current file identity does not prove that all
effective draft tensors are identical, so retain their runtime hashes too.

**Read-only inventory interpreted, 20 October 2026, 09:37 UTC.** Both restored
production nodes and all retained speed-test launches use the same mounted
`GLM-5.3-Flash-DFlash2-fp8blk` checkpoint path. The current files on both nodes
match exactly: `model.safetensors` is 1,377,719,136 bytes, SHA-256
`6cba1cff8ab4fcd657b89575f5db3debe1f5a39c78bda422953f553dc2357f8b`;
`config.json` SHA-256 is
`15bc842939ff7ca3ccc5aad36d25fff854da05a60086a4f55672d804fc6dbe7b`.
This rules out a difference between the audited checkpoint files. Post-run file
identity does not retroactively prove runtime-tensor identity during all trials.

The matched speed arms also have identical speculative configurations and
draft quantization class. Restored production has additional serving features
such as draft truncation/fused-convolution flags and a larger batch policy;
the recorded matched-test flags are a separate configuration. Those production
settings should not be silently attributed to the matched speed measurements.
Within the matched test, the observed code gap is reproducibly lower acceptance
of longer draft prefixes despite cheaper C1-shaped execution. Shared output
heads, target hidden states and generated histories remain concrete variables
for matched-token replay. The audit does not establish a need for retraining
or identify one of those variables as the cause.

The completed speed experiment confirms a reproducible, prompt-dependent
acceptance difference. Every one of the six trials per model gives the same
C1 counters and the same output tokens within that model:

| C1 prompt, 256 output tokens | Current CT BF16 KDA: rounds / acceptance | CT MXFP8 KDA: rounds / acceptance | ModelOpt: rounds / acceptance |
| --- | ---: | ---: | ---: |
| Diagnostic | 32 / 100% | 32 / 100% | 32 / 100% |
| Code | 56 / 50.77% | 54 / 54.23% | 47 / 63.22% |
| Explanation | 99 / 22.80% | 71 / 37.02% | 77 / 33.02% |

All launches specify the same DFlash checkpoint mount, seven drafts at C1,
7/5/3 batch policy, seed, rejection policy, graph sizes and backend settings.
Checkpoint file identity is now verified; effective runtime-tensor identity
remains a separate question.
The code regression is therefore not explained by different configured draft
lengths. It is not universal: the explanation prompt favors CT MXFP8.

For code, the fraction of rounds accepting at least the first drafted token
is **87.04% CT MXFP8 versus 87.23% ModelOpt**. At least three are accepted in
**59.26% versus 74.47%** of rounds, and all seven in **27.78% versus 40.43%**.
The difference is primarily in longer accepted prefixes. These are aggregate
survival rates, not comparisons of predictions at identical token positions.
Generated histories differ between models; counters may also include a final
drafted block beyond the response cap.

The final profiles contain C1-shaped draft calls with seven drafts and eight
target input rows. Their median GPU kernel spans are **7.06/7.11 ms** for CT
MXFP8 and **7.61/7.73 ms** for ModelOpt (rank 0/1; 11 versus 16 calls per rank).
These traces do not show a slower draft execution path for CT at that shape.
They are profiled component spans with different histories, not unprofiled
API latency or a controlled head ablation.

Both drafters share their target's head and embeddings. CT uses `VocabMethod`
for the compressed head; ModelOpt uses `UnquantizedLinearMethod`. This is a
specific difference to test, not evidence that compression causes the acceptance
gap. Both report `Fp8Config` for the draft model and BF16 for its fused
context-projection weight. The startup field named `draft_context_buffer_dtype`
actually records `_fused_kv_weight.dtype`; it does not establish KV-cache dtype.

[Recomputed counters, shape-specific profiles and source hashes](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/drafter-retained-review-v1-review.json)
retain all results, including concurrent windows. Position survival rates are
reported only for C1, where every call drafts seven tokens.

The updated sequence is:

1. Completed: all nine speed loads, six profiles, restoration and independent
   artifact verification; local review of all 108 speculative-counter windows.
2. Completed: all 1,319 GSM8K questions in each fresh CT arm, with speculation
   off; original production restored and all answers independently regraded.
3. Completed: checkpoint inventory, source vocabulary identity and the paired
   packed/dense head diagnostic. Completed next: paired target-context/verification capture at the same early
   code history and batch shapes. A drafter-only calibration pilot remains untested.

The first audit compares:

- Draft checkpoint paths, configuration, tensor dtypes/shapes and complete
  safetensors file hashes on both nodes.
- Speculative configuration, draft length, seed, quantization and selected
  custom runtime flags in the matched experiment and restored production.
- Target-owned embeddings and output head. Startup records already prove
  each drafter shares these objects with its target; the same draft checkpoint
  therefore does not imply identical effective computation across target models.
- Fused draft context-projection weight types and graph sizes recorded at startup;
  actual KV-cache types must be checked separately.
- Acceptance counters by draft position, preserving labels and raw deltas.
- Final draft/target profiles and output histories.

If these records cannot separate configuration effects from changed target
hidden states and continuations, the next diagnostic is matched-token replay:
hold input token history and batch shape fixed, inspect draft proposals and
target verification, then change one identified component at a time. Compare
compressed and reconstructed CT heads on the exact same draft hidden states
before attributing proposal differences to the head. Retain top-k candidates,
scores and accepted prefix lengths, not just total acceptance.
No new inference experiment is automatically launched by this read-only queue.
Do not assume retraining is required or attribute the difference to compression
before inspecting the evidence. Draft acceptance is not a task-accuracy metric.

The inventory reads only the known draft mounts and selected nonsecret runtime
settings. It does not load a model or import CUDA, and paces weight-file hashing.
Post-run file hashes establish the current artifact identities; they do not
retroactively prove runtime tensors were identical during each speed pass.

Local staging and status:
`/private/tmp/mxwave-drafter-comparison-v2`.
The collector waited for both local `complete.json` verification receipts and
then completed its read-only inventory. Its final phase is
`read_only_audit_complete`; reports are retained in the repository diagnostics.

The v1 wait ended when GSM8K v1 failed CPU preparation. V2 follows the corrected
GSM8K v2 receipt. The local retained-evidence review ran while GSM8K was active;
remote checkpoint hashing followed restoration. Frozen experiment sources
remain unchanged.
