# GLM compression experiment handoff

<!-- DRAFTER_EXPERT_PADDING_BEGIN -->
**20 October 2026, 17:43 UTC: padding replay prepared; GPU result pending.**
`drafter-expert-padding-v1` tests bounded routing padding on the retained
440-versus-448 dispatch, with both real TP weight partitions and both expert
projections. Native replay must first match the retained up/activation/down
outputs; padded stock/packed results must match each other and be bit-identical
on shared rows. Only then run graph-repeat and eight interleaved timing cycles,
including SM clock, temperature and power telemetry. Two launch policies are
retained: default occupancy and one block per SM. Timing controls retain the
2% drift limit. No serving speed claim follows from component timings.

CPU preflight passes all **1,728 checkpoint tensor records**, four frozen
runtime/library hashes and source compilation. Full pytest, ruff and strict
mypy pass. Plan SHA:
`314ea9bc87e04430d8c88067dd824edb891ac02e3eb538534abf112cc5d587d3`.
Production is still original/running at preparation; the bounded controller
will pause only verified idle production and restore it after the replay, with
an independent watchdog. This is not yet a tested fix. Full-model verification
and frozen drafter qualification wait for the component result and overhead
review. The adapted drafter stays **unqualified**, the sealed test stays
**unopened**, and the full 4,000-prompt training run remains outstanding.

Installed Marlin MoE Python source has no batch-invariance implementation;
the modular base defaults to unsupported. The installed global invariant flag
therefore does not establish coverage of this path. Earlier checkpoints below
remain historical.
<!-- DRAFTER_EXPERT_PADDING_END -->

<!-- DRAFTER_EXPERT_REPLAY_BEGIN -->
**20 October 2026, 17:27:12 UTC: direct expert replay complete; original production restored.**
The first changed operation is reproduced using the original layer-4 gate/up
weights and captured inputs/routing, with both TP partitions replayed separately
on the same Spark GPU. **All 24 stock/raw-bridge/packed-bridge outputs are bitwise
identical to their corresponding retained native outputs.** All **480 graph-repeat
checks pass**. The position-46 negative control stays exact. At position 54,
stock and compressed both reproduce the same one changed BF16 element on rank 0
and two on rank 1. Scale bytes reconstruct exactly. GPU fixture work took
**14.074 seconds**, peak CUDA allocation **1.354 GiB**; this is not a serving speed
measurement. Independent NumPy review confirms all retained output comparisons.

**Conclusion: scale compression does not cause this first divergence.** Shared
input vectors, route weights and expert assignments are exact for all seven
shared rows. The changed values belong to expert 252 on rank 0 and expert 284
on rank 1. Source audit shows that the native Stream-K reduction partitions
change with the 440→448 padded dispatch: one affected dot product changes its K
split from 1536/2560 to 3712/384 elements; another changes from no split to
2752/1344. This identifies a concrete reduction-order mechanism; a correction
has not yet been tested. It does not establish invariance of later operations.

Next bounded experiment: remove dispatch-dependent K splitting on this exact
fixture (e.g. bounded padding that makes the tile count divisible by the launch
grid), check stock/packed shared-row equality, then measure overhead. Only after
that passes should the full-model verification and frozen drafter qualification
be rerun. The adapted drafter remains **unqualified**; no promotion, new training
or sealed-test access. Original service is healthy with its original identities
and **500,000 context**; no diagnostic is running.

Replay source `f8023a0` is committed and verified in the Spark2 backup. Plan SHA:
`6610c01cdf60a330ce9763b6ca3711c656bbf43fa3cae9a6e04b341e80037962`.
All **16 replay evidence payloads** are independently verified on Mac and Spark2.
Archive: `drafter-expert-v1/evidence-v1.tar.gz`, **6,803,427 bytes**, SHA-256
`dacf947a5183aef2b6ba03a81d41e7f7fa6be79a5dbd542ab7c504a0005f7b0a`.
Local independent reviews: `out/drafter-expert-v1/review.json`,
`routing-review.json`, `partition-audit.json`. The prior 230-payload component
archive and failed V1 instrumentation attempt remain intact. All older dated
checkpoints below are historical. Independent analysis (seven payloads) is also
verified on both Sparks: `analysis-v1.tar.gz`, SHA-256
`fd43516a62356c3b040f943517ae81c2a21dfb2beca15ef6a326f3a7369b8323`.
Full pytest, ruff and strict mypy pass.
<!-- DRAFTER_EXPERT_REPLAY_END -->

<!-- DRAFTER_COMPONENTS_BEGIN -->
**20 October 2026, 17:11:44 UTC: component V2 complete; original production restored.**
Both drafter arms pass all before/capture/repeat/after output and counter controls.
All 230 archived payloads verify on Mac and Spark2; all eight arm/rank/position repeat
comparisons are bitwise exact for consumed nonworkspace data. Each graph retains
169 component events / 22,405,968 bytes. Unused dispatch-buffer capacity and output
scratch remain in the raw evidence and are excluded from numerical conclusions.

Both ranks identify the first changed shared-row result in **layer 4 gate/up
expert GEMM**. At position 54, seven input rows match and only the eighth draft
token differs. Rank 0 changes one BF16 element in shared row 1 (max 0.0009765625);
rank 1 changes two in row 4 (max 0.000244140625). Earlier observed component
results, including layer 3 expert results, match on shared rows. The position-46
negative control matches on all eight rows. Layer-4 padded dispatch changes
**440 → 448** tokens; arithmetic partition sensitivity is a hypothesis until the
direct stock-versus-packed replay confirms it.

Source audit also corrects feature naming: `aux_layer_5` means the output after
zero-based decoder **5**, not after decoder 4. DFlash layer IDs are translated to
id+1 boundary indices by installed `eagle3_utils.py`. This explains why the first
auxiliary feature contains more propagated changes than the layer-4 endpoint.

Original identities/configurations, health 200 and 500,000 context are verified
restored. Archive: `drafter-components-v2/evidence-v1.tar.gz`, 598,498,100 bytes,
SHA-256 `816c1667bee815a08b969053ea012f57260307547cd75c6486667b4f0273010f`.
V2 source before launch: `0763f38`; launched 16:56:47 UTC. Independent review:
`out/drafter-components-v2/review.json`. Next: isolated layer-4 expert replay using
actual captured operands and checkpoint weights, comparing installed stock,
raw bridge and the same sparse packed-scale library. No drafter qualification,
training, serving speed claim, promotion or sealed-test access.

V1 remains retained unchanged: the observer selected by eight total tokens and
incorrectly conflated one request × eight tokens with two requests × four tokens.
Its strict guard failed before requests; original production was restored.
V2 selects the complete native descriptor and verifies actual graph replay;
all four serving-image CPU preflights passed seven-graph descriptor checks and
211 frozen runtime bindings. V1: 46 payloads, 141,897 bytes, SHA-256
`e581e07ce1492880cf0cc390a5987acbfe785973d2dcd16b22c1564f563c3f7e`,
independently verified on Mac/Spark2. Failure retained as instrumentation evidence.
<!-- DRAFTER_COMPONENTS_END -->

<!-- DRAFTER_VERIFICATION_BEGIN -->
**20 October 2026, 15:59 UTC: V2 trace complete and independently verified; original production restored.**
Both original/adapted arms pass before/capture/repeat/after output and counter
controls, reproducing their prior uninstrumented prefixes. All **80 frames**
(10 per capture × two captures × two ranks × two arms) are retained. Inputs,
logits and captured features repeat bitwise in all four arm/rank comparisons.

Both ranks show the same first difference at output index **21**. All earlier
captured target inputs/features match. In that eight-row verification, inputs
and positions match for the first seven rows; only the last draft token changes
(**21613 → 369**, position 61). Selected pre-forward recurrent inputs at KDA
layers **0 and 4 are bitwise equal**. Shared rows 1–6 already differ at feature
boundary 5; row 0 is exact there but differs at boundary 14. Original logits tie
419/1128 at **27.625**; adapted-arm token 1128 rises to **27.75**, changing the
winner. Full-vocabulary KL for this one decision is **0.00202506**, not a corpus
quality score. This is a reproducible target-verification sensitivity associated
with changed future draft inputs; the first responsible operation is not yet
identified. Do not claim an mHC, attention, expert or cache bug is proven.

Original identities/configurations, health 200 and **500,000 context** are
verified restored at **15:54:30 UTC**. No diagnostic or training is running.
The candidate remains unqualified, and the sealed test remains closed.
Next: capture attention/mHC/MoE boundaries in layers 0–4 on this exact eight-row
case; isolate the first differing operation before more adaptation. If that is
an expert operation, compare original/changed inputs under stock and packed
expert scales to distinguish stock batch sensitivity from our adapter.

All **229 payloads** independently verify on Mac and Spark2. Archive:
`drafter-verification-v2/evidence-v1.tar.gz` (2,673,102,459 bytes), SHA-256
`3bb1ed2e11499982c939001abe57ee2289f08c985cfb904f6576e5bb85a54f52`.
V1's zero-request client-parser failure remains retained (55 verified payloads).
Corrected source is backed up on Spark2 at `cb1bd06`. The independent reviewer,
full findings and handoff are retained in this branch. `analysis-v1.tar.gz`
(14 payloads) is also copied to both Sparks; SHA-256
`32fff60fb8cb3b226c37c500cd96a7e870fdba2a62e2e1a937f23ce405e5d876`.
Full pytest, ruff and strict mypy pass. Packaging is deferred at the user's request.
<!-- DRAFTER_VERIFICATION_END -->

<!-- NATIVE_DRAFTER_DEVELOPMENT_BEGIN -->
**20 October 2026, 15:14 UTC: all 144 native requests complete; original production restored.**
The original/candidate/original comparison completes 48 requests per load, with
six exact output/counter repeats for each of eight development prompts. Original
outputs and counters reproduce after reload; per-prompt throughput drift is
**+0.14% to +0.90%**, passing the 2% control. All 88 effective drafter tensors per
rank match the expected checkpoint, with only the selected FC changed.

The candidate **fails unchanged-output equivalence on seven of eight prompts**.
Descriptive paired throughput changes are **code +5.91%, chat +1.61%, explanation
−4.57%**; corresponding TTFT changes are **+5.03%, +7.62%, +12.09%**. These changed
responses do not establish a quality-preserving speed gain or task-accuracy loss.
The candidate is not promoted. The sealed test remains unopened.

Before the first different token, the four code cases share 547 output tokens:
original/candidate delivery uses **151/150 stream chunks**, repeated in all six
cycles. This post-hoc observation is only a small improvement on identical text;
it is not an internal trace comparison or a replacement qualification gate.
At five covered divergences, candidate tokens are among the reference's top two,
with reference logit margins **0–0.25**. Three cases have identical prior stream
boundaries. Candidate logits and the numerical cause remain unmeasured.

Production is verified healthy at **15:14:02 UTC**, with both original container
identities/configurations and **500,000 context**. All **83 native evidence
payloads** verify on Mac and Spark2. The next investigation is matched-history
verification at the first divergence, before further training. No additional GPU
experiment or training is running. See
[the complete result, telemetry and retained evidence](GLM53_DRAFTER_CONTEXT_PILOT.md).
<!-- NATIVE_DRAFTER_DEVELOPMENT_END -->

<!-- DRAFTER_TRAINING_PREFLIGHT_BEGIN -->
**Completed CPU pilot, 20 October 2026:** 32 updates in **177.386 seconds**;
step 24 improves offline mean accepted prefix from **4.25 to 4.375** on code,
**2.75 to 2.875** on chat, with explanation unchanged. Step 32 fails chat and is
rejected. Independent review reproduces 1,120 token comparisons, 135 acceptance
fields and selection. Export changes only FC; 115 other tensor payloads and the
config are byte-identical. This small offline result is superseded for serving
claims by the failed native equivalence check above. Prior failures are retained.
All 44 feature-capture/control requests and 5,456 payload hashes verify; the
independent audit finds zero differences across 336,281,600 BF16 feature elements.
Earlier dated checkpoints below are historical.
<!-- DRAFTER_TRAINING_PREFLIGHT_END -->

<!-- DRAFTER_CPU_PREFLIGHT_BEGIN -->
**20 October 2026: CPU preparation items 1–4 complete; production uninterrupted.**
The success thresholds are frozen. The real FP8 drafter passes a full
FP8 → BF16 → FP8 round trip with **zero different bytes across all 116 tensors**;
the entire model/config files match the originals. Conversion and verification
took **7.217 seconds** on the Mac with original block scales retained.
The pinned SpecForge DFlash2 class strictly loads all **81 dense parameter
tensors**, with zero missing/unexpected keys or shape mismatches.

**4,000 training / 200 validation / 200 sealed-test prompts** are frozen and
hashed, with no source answers included. Exact/template/token overlap checks pass.
The exact CT tokenizer/template have been applied on CPU; all prompts fit an
8k capture budget. These are public code/chat/QA proxies, not private harness
traffic. Do not inspect the sealed test while tuning.

Production was checked healthy at **13:21:47 UTC**, with its original identity
and **500,000 context**. No GPU experiment, optimizer step or promotion occurred.
Native forward/backward and serving parity remain untested. The next milestone
is the training-forward/native parity and feature-capture contract, followed by
a bounded GPU development/capture window. Full 4,000-prompt feature storage is
about **98.40 GiB before auxiliary data**, so begin with a subset or stream it.
MTP remains an unqualified separate experiment, not a ready fallback.

Ruff and strict core mypy pass; **885 tests pass, one optional dependency skip**.
The frozen prompt/token corpus and reviews are backed up on **spark2**, with
archive and all six prompt/token file hashes independently verified there.

See [CPU preflight, frozen thresholds, mapping and dataset provenance](GLM53_DRAFTER_PREFLIGHT.md).
This checkpoint supersedes earlier statements that the round-trip/mapping/data
work is still pending. Earlier pilot failures and all deployment gates remain.
<!-- DRAFTER_CPU_PREFLIGHT_END -->

<!-- CT_VARIANT_SCOPE_BEGIN -->
**Scope clarification, 20 October 2026: lossless packing and target quantization
are separate changes.**

| Variant | Expert weights | KDA projection weights | Change relative to the source CT checkpoint |
| --- | --- | --- | --- |
| Source CT | MXFP4 | BF16 | Reference CT quantization |
| Original lossless-compressed CT | Same MXFP4 | Same BF16 values | Reversible storage/runtime encoding; no added weight quantization |
| Later CT MXFP8 candidate | Same MXFP4 | BF16 converted to MXFP8 | Additional lossy KDA quantization, plus retained lossless packing elsewhere |
| Production ModelOpt | NVFP4 | MXFP8 | Different quantized checkpoint; not a compression-off CT control |

The source checkpoint files were not rewritten. The later runtime candidate
really does use MXFP8-rounded KDA weights, despite retaining MXFP4 experts.
Calling both variants simply "CT" obscured this distinction.

The earlier [same-checkpoint compression comparison](GLM53_MXFP4_SERVING_V3.md)
already found **all 24 paired token sequences identical**, with matching measured
speculation counters. There is no observed compression-induced acceptance loss
in that fixture. This finite check does not establish universal bitwise model
equivalence, but it must not be omitted when explaining the drafter work.

The later paired-feature diagnostic compares **CT MXFP8 against ModelOpt**,
not the same CT weights with compression off/on. Its feature differences and
proposal miss therefore do not establish that lossless packing caused a mismatch,
or that compression alone requires retraining. Adapting a drafter to a different
quantized target remains a separate hypothesis. Preserve the explicit target
variant in every adaptation protocol and comparison; do not silently make
further quantization changes. The goal remains a working CT setup and a proper
drafter, with measured evidence determining the necessary intervention.
<!-- CT_VARIANT_SCOPE_END -->

**Recommended continuation after the scope clarification:** retain the already
tested CT MXFP8 variant as the performance candidate, and the original BF16-KDA
CT as the lossless-compression reference. This is a recommendation for the next
experiment, not a production promotion or relaxation of the failed KL gate.
Keep the candidate's target weights, quantization, tokenizer and serving policy
fixed during drafter adaptation.

Reverting KDA precision is not an established fix for the code acceptance gap.
The retained matched run needed **56 code verification rounds with CT BF16 KDA,
54 with CT MXFP8 KDA, and 47 with ModelOpt**, for the same 256-token output budget.
CT MXFP8 code throughput was 44.86 tok/s versus 39.24 for CT BF16 KDA and 49.09
for ModelOpt. These are workload-specific results; they neither isolate KDA's
effect at identical histories nor prove a general training benefit.

The next concrete milestone is a small training/export/native-reload compatibility
check for the existing drafter, followed by bounded CT-specific adaptation only
once that path works. Reuse the completed compression-off/on evidence; do not
repeat it merely to re-establish the already-observed matching tokens/counters.
Evaluate a candidate drafter against the unchanged drafter on fresh CT rollouts
before a wider serving sweep. No new run has started at this update.

<!-- CT_DRAFTER_OBJECTIVE_BEGIN -->
**Active objective, reaffirmed by the user on 20 October 2026:**
> I want CT to work, and want to have a proper drafter.

Continue toward a usable CT model with a drafter adapted to the CT target.
The failed small selector-only pilot closes that patch, not this objective.
Drafter adaptation is a required workstream; it is not conditional on first
proving a concurrency gain. This direction supersedes the earlier recommendation
to decide whether to pursue adaptation only after the serving-configuration test.

Next work:

1. Audit training and export compatibility for the exact five-layer FP8 DFlash2
   checkpoint and custom vLLM runtime. Establish a trainable representation,
   CT target-feature capture, parameter mapping and a native reload check before
   committing to a larger run.
2. Prepare a bounded adaptation protocol with representative code/chat data,
   separate validation and a fresh sealed test set. The pilot's test split is
   already open. Keep the CT target fixed while adapting the drafter; determine
   the trainable scope from the audit rather than assuming another projection-only
   fit or full-backbone training will recover the gap.
3. Validate the exported drafter with native kernels and fresh live rollouts.
   Measure acceptance by draft position, verification rounds, first-token latency
   and end-to-end code/chat throughput. Offline ranking improvement alone is
   insufficient. Compare against the unchanged drafter and matched production.
4. Complete production tuning/capacity, repeatable C8/C16 measurements and actual
   long-context qualification before replacing production.

Immediate technical recommendation after reviewing the retained runtime and
training documentation: establish a train/export/reload smoke test using the
existing pretrained DFlash2 drafter. Then assess a limited context-projection
adaptation on CT features, with selector adaptation if justified. This is a
candidate intervention, not an established repair. The previous pilot changed
only the output selector projection and left target-feature processing frozen.
Training the context projection still requires gradients through the frozen
draft backbone; the custom FP8 inference path is not yet a validated training
path. Measure a few training batches and export/reload agreement before giving
a full-run ETA or expanding the trainable scope.

The upstream [drafter model card](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)
identifies the original GLM target; it does not establish training specifically
for our production ModelOpt quantization. The measured acceptance difference
supports target-feature adaptation without proving that all of the code-speed
gap comes from it. [SpecForge's training documentation](https://sgl-project.github.io/SpecForge/basic_usage/training.html#dflash2)
provides the DFlash2 training approach; compatibility with this custom runtime
and checkpoint remains to be checked.

Status at this update: this records the continuation objective and next work;
no new adaptation run has been launched. The previous patch remains unapplied.
Retain the failed pilot and earlier repeat controls. The MXFP8 target's failed
1e-5 KL comparison is unchanged; adapting its drafter does not establish
losslessness. Original production remains the service to restore between trials.
<!-- CT_DRAFTER_OBJECTIVE_END -->

<!-- SELECTOR_CALIBRATION_LIVE_BEGIN -->
**20 October 2026: selector-only calibration pilot complete; held-out gate failed.**
All 52 API requests complete, and both before/capture/after controls repeat output
tokens and speculative counters exactly. All **3,682 archived payloads**, **1,819
native graph frames** and **1,735 verified draft rounds** pass the independent
capture review. Original production was restored at **12:30:49 UTC**, healthy
with its original identities and **500,000 context**. No experiment or training
remains running; no weights or candidate have been promoted.

The bounded 100-step CPU fit selected step 75 using validation only. Mean matching
prefix length improves **2.904 → 2.931** on validation, but declines **2.394 → 2.383**
on the eight separate test prompts (code **2.409 → 2.404**, explanation
**2.348 → 2.320**). These are fixed-state FP32-surrogate draft metrics, not task
accuracy or serving speed. NumPy independently reproduces **108 numerical fields**.
The frozen no-regression test gate remains failed; **the patch is not applied**.

See the [complete calibration result](GLM53_SELECTOR_CALIBRATION.md). This small
32-prompt, projection-only pilot does not rule out broader drafter adaptation;
it demonstrates no code-speed recovery. All four user-specified
[production replacement gates](GLM53_SELECTOR_CALIBRATION.md#production-replacement-gates)
remain open: drafter recovery, production tuning/capacity, stable C8/C16 results,
and long-context qualification. Completed work and pilot source are committed
and backed up on **spark2**. This block supersedes earlier live-work statements.
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


<!-- KDA_MXFP8_SPEED_LIVE_BEGIN -->
**Final checkpoint, 20 October 2026, 09:35 UTC.** Native MXFP8 speed and
full GSM8K are complete and independently verified. Speed: all nine loads,
115 payloads and 7,441 numerical fields verify. GSM8K: both 1,319-question
passes, 2,690 payloads and 2,638 independently regraded answers verify;
2,933 numerical fields match exactly across Spark/Mac.

Fresh CT BF16 KDA scores **1,248/1,319 (94.62%)**; native MXFP8 KDA scores
**1,260/1,319 (95.53%)**. There are 22 gains and 10 losses, net +12 correct.
Both arms have zero API errors or truncations. The earlier **1e-5 KL gate
remains failed**; the higher observed task score does not establish losslessness.

Original production is restored on port 8000 with **500,000 context**, exact
original identities/configurations and healthy status verified at **09:35:00 UTC**.
No candidate is promoted. The read-only drafter inventory is also complete;
no further GPU experiment has started. See [the speed results](GLM53_KDA_MXFP8_SPEED.md),
[full GSM8K results](GLM53_KDA_MXFP8_GSM8K.md), and
[drafter findings](GLM53_DRAFTER_COMPARISON.md). This checkpoint supersedes older
service-state and interim-score statements below.
<!-- KDA_MXFP8_SPEED_LIVE_END -->

**Native MXFP8 serving comparison complete, 20 October 2026, 08:12 UTC.**
All nine loads and 108 timing groups finish. All **115 archived payloads**
verify; **7,441 recomputed fields match exactly** on Spark/Mac.
Six final profiles and all 232 target graph ranges have been reviewed.
Original production was restored at **08:08:20 UTC**, with exact identities and
500,000 context, before the authorized GSM8K follow-up paused it again.

| Case | Current CT lossless BF16 KDA | CT MXFP8 KDA | ModelOpt, matched settings | CT MXFP8 vs ModelOpt | Repeat controls for that comparison |
| --- | ---: | ---: | ---: | ---: | --- |
| C1 diagnostic, 256 output tokens | 66.05 tok/s | 71.57 tok/s | 69.67 tok/s | +2.74% | Pass |
| C1 code | 39.24 tok/s | 44.86 tok/s | 49.09 tok/s | **−8.61%** | Pass |
| C1 explanation | 23.91 tok/s | 35.85 tok/s | 32.37 tok/s | +10.72% | **Fail: ModelOpt repeat** |
| Offered C8 | 64.04 tok/s | 89.14 tok/s | 79.30 tok/s | +12.41% | **Fail: both repeats** |
| Offered C16 | 72.28 tok/s | 95.13 tok/s | 83.95 tok/s | +13.32% | **Fail: candidate repeat** |

These are completed measurements, not a universal production speed qualification.
All modes run at most six resident requests; C8/C16 include queueing. Code is
14.31% faster than current CT, but remains 8.61% below matched-settings ModelOpt.
Mean code first-token time improves 459.8 → 399.3 ms versus current CT and
442.7 → 399.3 ms versus ModelOpt. The earlier code diagnostic shows lower
speculative acceptance requires more rounds; the completed trace review confirms the KDA component saving.

CT MXFP8 saves a further **1.105/1.086 GiB per GPU** versus current CT lossless
BF16 KDA. Its startup allocation is **7.447/7.430 GiB per GPU lower** than ModelOpt
under the same KV budget. That latter comparison includes different checkpoint
formats and is not solely a lossless-compression saving. The prior **1e-5 KL
failure remains**; the completed fresh GSM8K comparison scores 1,248 → 1,260 correct.
See [the complete speed table and evidence](GLM53_KDA_MXFP8_SPEED.md).

**Trace result:** eight-row KDA GEMM sums fall from 22.7–22.8 to 13.0–13.1 ms
on both ranks, matching the predicted roughly 9–10 ms component saving. Native
MXFP8 has zero weight rebuilds across both recorded sessions; current CT has
68 rebuilds in each of seven captured width-12 graphs per rank. All 232 target
graph ranges are attributed. Width-16 full graphs still trail ModelOpt, so this
does not establish faster execution for every batch layout.
[Profile evidence](GLM53_KDA_MXFP8_SPEED.md).

**Full GSM8K complete and verified (`kda-mxfp8-gsm8k-v2`).** Fresh control
**1,248/1,319 (94.617%)**; native MXFP8 **1,260/1,319 (95.527%)**. The paired
change is 22 gains, 10 losses, net +12 / +0.910 percentage points. Exact
McNemar p=0.05010246: a higher observed score, not established general superiority.
Strict-format scores are 1,226 → 1,245. Both have zero errors and truncations.

This C4 workload with speculation off finishes in **37m44s → 33m19s**, with
output throughput **41.90 → 48.18 tok/s (+14.98%)**. These single-pass natural
response timings do not replace the interleaved serving-speed comparison.
All 2,690 payloads and 2,638 independently regraded responses verify; 2,933
fields agree exactly. Original production was restored at 09:35:00 UTC.
[Full results, changed-question IDs, and evidence](GLM53_KDA_MXFP8_GSM8K.md).

V1's CPU filename-collision failure remains retained. V2 completed under the
same prompts and scoring; controller 110454, queue 108092, collector 9362 and
monitor 9363 are finished. The queued drafter inventory also completes, confirming
identical draft checkpoint/config files on both nodes and identical speculative
settings in the matched speed arms. No new GPU experiment or promotion occurred.

**Next work prepared:** code/explanation replay inputs are retained and bound
to all six completed trials per mode. First compare compressed and reconstructed
CT output heads on identical draft hidden states; then, if needed, compare both
targets on fixed token histories and batch shapes. No additional GPU run has
started. See [the focused drafter diagnostic](GLM53_DRAFTER_COMPARISON.md).

<!-- DRAFTER_FOLLOWUP_BEGIN -->
**Post-benchmark drafter audit: initial read-only collection complete.** Matched speculative configurations equal: `True`. Audited draft weight files identical: `True`. Source hashes, shared-head methods and per-position acceptance are retained in [the audit receipt](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/drafter-comparison-v2-review.json). Interpretation with final profiles remains; no new GPU experiment has started.
<!-- DRAFTER_FOLLOWUP_END -->

**KDA rounding measurement complete and independently verified,
20 October 2026, 06:05 UTC.** All 1,688 payloads verify; 5,414 numerical fields
agree on Spark/Mac with maximum difference 7.11e-15.
Exact original production is restored at **05:51:18 UTC**, healthy on port 8000
with 500,000 context. All original KDA weights restore bit-exactly on both ranks.
No checkpoint conversion or serving promotion has occurred.

The [MXFP8 weight-rounding screen](GLM53_KDA_MXFP8_QUALITY.md) finishes all five
passes. It fails the unchanged 1e-5 KL gate: full mean KL is 1.23e-3–1.59e-3
across the declared pairs. The 4,096-token context also fails self-repeat;
keep that failure. Outside it, all 208 positions repeat exactly within each arm
and cross-arm mean KL is 1.22338e-3 in every pair. Short-prompt KL is nonzero
even at the first prediction. Sampled PPL improves 4.1150974 → 4.0857491
(−0.7132% on 32,768 tokens), exactly repeated before/after and within candidate.
This shows changed distributions, not demonstrated broad quality degradation
or equivalence. Both generated functions pass; shared baseline output-format
failures limit the eight-task smoke screen.

This completed rounding run used unchanged BF16 GEMMs to isolate weight error.
Its analysis, archival and independent verification are complete. The separate
native speed experiment and queued GSM8K comparison are described above; no
quality threshold has been relaxed.

<!-- MXFP4_SERVING_V3_STATUS_BEGIN -->
**Lossless MXFP4/DFlash comparison complete and independently verified,
20 October 2026, 05:39 UTC.** All 48 timed requests across A/B/A/B/A/B complete;
all 80 payloads verify, with 1,863 recomputed fields identical on Spark/Mac.
Allocation savings are 9.36 GiB across TP2, about 4.68 GiB/GPU. C1 observed
end-to-end changes: +0.79% on 64-token diagnostic output, +1.89% on 256-token
output, +2.44% on code and +2.77% on explanation. Code fails the stock-repeat
control (2.85% spread against 2%); the other controls pass. Small observed gains
are not a broad production-upgrade claim. TTFT adds 17.6–35.1 ms; normalized
decode time falls 3.12–4.95%. All 24 paired output sequences and speculation
counters match, which is not full-model KL qualification.

All four final-pair profiles show zero decode rebuilds and eleven eight-row
C1 graph replays. Compressed prefill reconstructs 68 KDA matrices/rank, costing
36.47/37.14 ms in reconstruction kernel sums. Varied-content acceptance is
50.77% on code and 22.80% on explanation, identical between arms; repetitive
diagnostics reach 100%. Exact original production was restored at 05:37:55 UTC
before the quality diagnostic paused it again and restored it at 05:51:18 UTC. See the
[complete comparison and evidence](GLM53_MXFP4_SERVING_V3.md).

A read-only review of the existing DFlash C1 traces finds that compressed CT
spends 12.32/13.36 ms more in non-expert linear kernel sums per target graph
than common-settings ModelOpt, while expert GEMMs take 6.12/6.22 ms less.
A follow-up source-order attribution, cross-checked against all 68 compressed
KDA GEMMs in every graph, assigns 9.08/9.94 ms of the main-GEMM gap to KDA,
about three quarters; other linears contribute 3.25/3.40 ms. This is inference
from the installed operation order, not module-labelled scopes. The user
proposed MXFP8 KDA while keeping MXFP4 experts; that would be a separate lossy
quantization change requiring a separate CT-before/after quality evaluation.
No checkpoint conversion has started. The user raised the additional quality-loss risk:
BF16-to-MXFP8 introduces weight error; the separate quality screen above now measures it.
Existing lossless compression preserves the checkpoint weights; its fused
arithmetic still requires numerical qualification. Do not relax the existing
1e-5 gate or treat production's use of MXFP8 as proof that a new CT variant meets
it. The completed lossless serving comparison retained BF16 KDA in both arms.
The separate completed quality diagnostic rounded KDA values in memory only.
The follow-up speed hypothesis is recorded in the projection review: 9–10 ms
is a plausible C1 KDA component saving, not a measured wall-time improvement.
Marlin W8A16 does not establish faster FP8 prefill arithmetic; C8/C16 gains and
changed DFlash acceptance remain unmeasured. The encoded KDA payload estimate
saves about 0.90 GB/rank before runtime scale/workspace accounting. A comparison
of current CT, CT with actual MXFP8 KDA, and common-settings ModelOpt would
separate the effects. That comparison is now running as described above;
other linears remain unchanged.
See [the projection-gap review](GLM53_DFLASH_C1.md#projection-gap-review-20-october).
<!-- MXFP4_SERVING_V3_STATUS_END -->

<!-- PRODUCTION_RESTORE_20261020_BEGIN -->
**Current checkpoint: 20 October 2026, 04:46 UTC / 06:46 Berlin. Original
production is restored.** Both original container identities and configuration
hashes match the saved pretrial record; both CT trial containers are stopped.
The authenticated port-8000 model endpoint returns 200 and advertises
`glm-5.3-flash` with **500,000 context**. This supersedes all earlier live-service
statements below. See the [restoration receipt](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/production-restore-20261020-status.json)
and [local identity review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/production-restore-20261020-local-review.json).
Future experiments must restore this original production service; older NVFP4
controllers expecting the CT trial as their restoration target need updating.

The user has asked why work moved from MXFP4 to NVFP4. The
[three-arm MXFP4/DFlash diagnostic](GLM53_DFLASH_C1.md) already separates much of
the regression: same-CT compression reduces short-C1 normalized decode time by
5.34%, adds 52.3 ms TTFT, and changes 64-token end-to-end throughput by −0.32%.
These seven-trial descriptive results are not a new speed qualification.
All six profiles have zero decode rebuilds; prefill reconstructs 68 KDA
matrices/rank. CT uses BF16 KDA where production uses MXFP8, so its slower
uncompressed backbone is part of the replacement gap. The NVFP4 extension
compresses the original production checkpoint's expert scales while preserving
that backbone. Its 3.16 GiB/GPU allocation saving is verified; serving speed and
full-model numerical qualification remain unresolved. These savings are not
additive across the two alternative checkpoints.

No new GPU experiment or trace implementation was launched this turn.
Before the direction question, the user requested the known consistent-mHC fix
before further internal tracing, speculation-off quality qualification (or
matched draft layouts), and at least six interleaved speed trials per arm with
clock/thermal, acceptance and admission records. Those checks have not run.
Only read-only runtime source inspection was completed. The selective dense-KDA
MXFP4 option remains untested: estimated 6.85–7.23 GiB TP2 saving with
expert/vocabulary compression retained, removing rebuilds but giving up fused
KDA decode. Neither branch currently establishes a production speed upgrade.
<!-- PRODUCTION_RESTORE_20261020_END -->

<!-- NVFP4_MOE_STATUS_BEGIN -->
**Combined-fix diagnostic complete and independently verified, 19 October 2026,
21:32 UTC.** `nvfp4-inputs-v1` completes all twelve groups with stable MoE
ordering and the source-pinned sparse-MLA padding adapter installed before
capture. All **895 payloads** verify locally; Spark/Mac recomputation agrees on
**5,934 numerical fields**, maximum difference 1.56e-15. Exact prior CT/DFlash
container identities are restored and healthy on port 8000 with **8,192 context**;
original production remains stopped and unchanged. No GPU experiment is running.
Verified duplicate rank-0 captures were removed after checking copies on rank 1
and Mac, recovering 1,041,370,617 bytes; about 6.39 GiB is free. Raw archives and
rank-1 `retained-combined` directories remain complete.

All first-token self-repeat KL values are zero, including C8. C1 ordinary
before-controls have mean decode KL 2.2512e-5 / 1.9068e-5; after-controls have
9.1127e-6 / 1.0239e-5. One after-pair passes, but the repeat set remains
unqualified. Captured C1 pairs reach mean decode KL about 3.9e-5. The matched
C8 pair reaches 2.4696e-4; the other pair changes a later verify width. All
returned tokens/conditioning match. The unchanged gate is **mean** prefill and
decode KL below 1e-5 with matching layouts and conditioning.

C1 actual recorded token IDs, positions, masks, embeddings and other prepared
tensors match. Its target graph outputs differ by the first auxiliary boundary
after decoder layer 5 (including mHC reconstruction). Saved attention/recurrent
state and internal graph boundaries are not yet observed here. Next targeted
work: compare logical active states after prefill, then graph boundaries in
layers 0–5 with both policies' actual call paths recorded. The C8 test admits
six requests then two under the original 6 GiB KV allocation; it does not prove
eight simultaneous resident sequences. No compressed comparison or new serving
speed measurement ran, and the quality gate remains unresolved.

The prior `nvfp4-align-v1` evidence also independently verifies: 913 payloads,
419 numerical fields and six timing aggregations. Canonical MoE ordering makes
all captured prefill tensors exact on both ranks; all 840 internal replays are
exact. Eager 256-row expert-layer overhead is 0.38–1.77%, with six passing 2%
timing controls. See [the complete results and evidence](GLM53_NVFP4_MOE.md).
<!-- NVFP4_MOE_STATUS_END -->


<!-- NVFP4_REPEAT_STATUS_BEGIN -->
**First stock difference located and evidence verified, 19 October 2026, 20:47 UTC.** All 20
diagnostic API groups finish. The focused capture first differs at **layer 3's
MoE expert output**, with identical actual gate logits, hidden-state inputs,
shared-expert output, attention and mHC boundaries on both ranks. Each rank
changes just **4 of 1,048,576 BF16 values** there (maximum 0.00390625); final raw-
logit KL is **0.00537795**. All 20 captured raw-logit/API comparisons agree to
numerical precision. Ordinary controls before hook installation also fail,
including one-output-token requests.

The exact routing/GEMM/reduction operation is not isolated yet. Another pair is
exact through layer 3 and first differs at layer 4; later operations may add
variation. No deterministic fix or compressed-versus-stock rerun is claimed.
Full-model qualification stays unresolved at the unchanged 1e-5 gate. Existing
memory/codec results remain valid. Next: observe the internal MoE boundaries
with real operands and replay fixed routing before choosing a fix.

All **4,290 v2 payloads** independently verify on Mac; Spark/Mac agree on
**5,374 numerical fields** (maximum difference 3.56e-15). The 63-payload v1 failure
is retained and verified too. Local ruff/strict mypy pass; 839 tests pass, one
skipped. Controller **complete_restored**: exact prior CT/DFlash containers are
healthy on port 8000 with **8,192 context**; original production remains stopped
and unchanged. No GPU experiment remains running. Temporary rank-0 snapshot
copies were removed only after verifying copies on rank 1 and Mac, recovering
5,445,162,848 bytes and leaving about 6.60 GiB free.

See [the complete result, next diagnostic and evidence](GLM53_NVFP4_REPEAT.md).
Raw captures now live in rank-1 `nvfp4-repeat-v2/retained-combined` and local
`/private/tmp/mxwave-nvfp4-repeat-v2/evidence`; rank 0 keeps manifests/receipts.
Earlier dated entries below remain historical.
<!-- NVFP4_REPEAT_STATUS_END -->

<!-- NVFP4_MODEL_STATUS_BEGIN -->
**Complete and independently verified, 19 October 2026, 20:04 UTC.** The original-production
NVFP4/MXFP8 checkpoint now runs with lossless expert-scale storage. Actual CUDA
allocation saves **3.1639 / 3.1665 GiB per GPU** after graph capture; all 84 scale
families/rank reconstruct exactly and their dense owners are released. This is
an extension to the existing production format, not a checkpoint conversion.

The compressed arm observes 38.77 / 73.62 / 73.93 tok/s at short C1/C8/C16,
versus stock B's 35.92 / 69.97 / 67.81. **Serving speed is not qualified:** all
four stock-A/B controls miss the unchanged 2% limit (23.06%, 2.24%, 9.66%,
10.50% for short C1/C8/C16 and long C1). The observed 5–9% advantage over B is
encouraging but does not establish a repeatable gain.

**Full-model zero-loss quality is also unqualified.** Both stock runs and the
compressed run fail C1/C8 self-repeat KL at 1e-5. C1 layouts match within each
self-repeat, but speculative verify widths differ across arms; C8 layouts also
vary, and one stock-B repeat changes conditioning tokens. Exact storage and the
22 isolated matched-routing checks do not substitute for this failed model gate.

All three arms finish (60 timed groups, 12 full-vocabulary quality groups,
six TP-rank traces). All traces replay graphs and contain zero standalone scale
reconstruction kernels; the profiled generated sequence and normalized kernel
counts match across arms. All 154 retained payloads verify locally; Spark/Mac
recomputations agree on 1,623 numerical fields (maximum difference 2.45e-15).

The exact prior CT/DFlash service is restored on **port 8000 with 8,192 context**;
authenticated models/health return 200 and unauthenticated models returns 401.
Original production containers remain unchanged and stopped. The NVFP4 adapter
was tested only in disposable containers. Local ruff/strict mypy pass;
835 tests pass, one skipped. Next useful work is a fixed-layout reference
repeat diagnostic before more speed tuning or a zero-loss deployment claim.
See [the full result, limits and retained evidence](GLM53_NVFP4_MODEL_TEST.md).
<!-- NVFP4_MODEL_STATUS_END -->

<!-- NVFP4_COMPONENT_STATUS_BEGIN -->
**NVFP4 component prototype complete, 19 October 2026, 19:03 UTC.** Lossless
scale compression now works on the original production checkpoint's NVFP4 expert
format, preserving its weights and arithmetic. The existing MXFP4 implementation
remains available; no ModelOpt serving adapter or deployment switch has occurred.

The selected five-bit byte-offset consumer takes **1.90% / 3.07% less time than
the installed stock operator** for the full 288-expert layer at eight rows on the
two TP partitions. Both controls pass the 2% repeat limit across 32 trials, with
bitwise-exact intermediate/final outputs. Partitions run sequentially on one
SM121 GPU with real layer-23 weights and synthetic activations/routing. The
broader 1/8/16/64/256-row matrix takes 2.06–4.48% less layer time than the matched
rebuilt raw control. These are component results, not C1/C8 serving measurements.

A full-checkpoint CPU count gives **3.250086 / 3.252740 GiB encoded-storage saving
per rank**, including metadata and raw fallbacks. Actual whole-model VRAM,
TTFT, throughput and KL remain unmeasured for this port. Do not add this storage
saving to the earlier CT/MXFP4 result. The next step is a ModelOpt serving adapter
and matched full-model qualification while preserving MXFP8 KDA and head paths.

All 217 evidence payloads independently verify locally, including timing gates,
315 output-hash comparisons, 13 traces without separate scale reconstruction and
full-range storage accounting. Earlier fast-math, reduced-fixture performance,
allocation-cap and unstable-control failures remain retained. Local validation
passes: ruff, strict mypy, 832 tests / one skipped, including 18 codec tests.

All 17 experimental containers and controllers are terminal; the owned awake
lease has finished. Exact live service identities are unchanged and healthy:
CT/DFlash on port 8000 with **8,192 context**, original production still stopped.
No clocks or KV settings changed. See the [complete result and retained evidence](GLM53_NVFP4_COMPONENT.md).
Earlier dated entries below remain historical and are superseded by this result.
<!-- NVFP4_COMPONENT_STATUS_END -->

<!-- MODELOPT_PORT_STATUS_BEGIN -->
**ModelOpt CPU audit complete, 19 October 2026, 17:59 UTC.** All 36,288 target
NVFP4 scale tensors were scanned: **17.71875 GiB total / 8.859375 GiB per rank**
with even TP2 sharding. Global/per-tensor weighted entropy is **3.773 / 3.359
bits per byte**. Concrete palette layouts estimate **3.25–3.30 GiB saved/rank**,
about **4.1% fewer expert weight-plus-scale bytes**. This is source-storage
feasibility, not measured CUDA allocation or throughput. All 1,008 sampled
block-codec roundtrips are exact; 15 archived payload hashes and aggregate
calculations verify locally. The successful CPU scan took 45.15 seconds with
417.77 MiB peak RSS, two cores, no network/GPU. V1/V2 metadata failures remain
retained; V3 separately accounts for extra layer 45 and superseded shard tensors.

A bounded NVFP4 Marlin component prototype is justified. It must reconstruct
actual post-Marlin S0E5M3 bytes and preserve compensating global scales, then
measure complete expert GEMMs with repeated controls. No GPU performance/KL
result exists for this route. Preserve production's checkpoint, MXFP8 KDA and
specialized head/draft paths; no ModelOpt adapter has been installed.

The head route needs more than a loader port: production's certificate requires
and retains contiguous BF16 head weights and reads candidate rows during exact
refinement. The current packed method would fail its guard, and dropping the
module weight alone would leave the dense certificate reference alive. The
prior 2–4 hour estimate was too optimistic for preserving these optimizations.
Vocabulary source compatibility remains proven, with ~0.697 GiB TP2 potential
storage saving; a production C1 gain remains unmeasured.

All audit containers are terminal. Exact live service identities remain unchanged
and healthy: CT candidate on port 8000, **8,192 context**; original production
remains stopped. No KV tuning or GPU experiment started. See the
[complete inventory, entropy result and next test](GLM53_MODELOPT_SCALE_AUDIT.md).
<!-- MODELOPT_PORT_STATUS_END -->

<!-- DEPLOYMENT_OBJECTIVE_BEGIN -->
**Deployment objective clarified, 19 October 2026, 17:43 UTC.** The target is better
end-to-end performance than the original tuned ModelOpt/DFlash production
setup. The user accepts a modest first-token penalty if the overall result
improves; no numerical TTFT tolerance was specified. Keep historical failed
gates unchanged. Same-CT diagnostic gains and memory savings alone do not
establish this deployment win. Current evidence has not demonstrated it.

C1 is the immediate focus. Extra KV addresses concurrency queueing, **not**
the measured C1 regression. No KV experiment has started. Removing prefill
rebuilds by retaining dense KDA is now a lower-priority tradeoff, because it
can sacrifice fused decode speed to improve an accepted TTFT cost.

A read-only compatibility check finds all four production vocabulary TP shards
byte-identical to the qualified CT vocabulary sources. That offers a bounded
vocabulary/head port while preserving ModelOpt's 68 MXFP8 KDA projections and
DFlash setup. Existing artifacts account for about **0.697 GiB TP2** vocabulary
saving; C1 speed improvement is unmeasured and may be small. The current loader
hooks are CT-specific and need a ModelOpt adapter. Expert-scale compression is
more substantial work: NVFP4 E4M3/global scales differ from the current MXFP4
E8M0 consumer. No port or new GPU run has started; serving remains unchanged.
See [compatibility evidence](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/production-c1-audit-v1-report.json).
<!-- DEPLOYMENT_OBJECTIVE_END -->

<!-- DFLASH_C1_INVESTIGATION_BEGIN -->
**DFlash C1 investigation complete and independently verified, 19 October 2026, 17:27 UTC.**
The exact previously live candidate is restored healthy on authenticated port
8000, with DFlash and **8,192 context**. All diagnostic containers/controller/
watchdog are terminal. Installed vLLM reports `0.1.dev20051+g487ecf187`, not a
package labelled `0.32.0`; release ancestry has not been established.

Seven matched trials at each prompt length separate checkpoint replacement from
compression. Short C1 ModelOpt → CT off → CT on: **48.163 → 46.118 → 45.972 tok/s**.
Compression itself changes short/long throughput by **−0.32% / −0.08%**: essentially
parity at 64 output tokens. It reduces normalized decode time by **5.34% / 3.46%**,
but adds **52.3 / 32.7 ms TTFT**. Most short-C1 replacement regression is already
present with compression disabled. Current ModelOpt's KDA uses MXFP8; CT uses BF16.

All six traces prove nine eight-row FULL target replays and **zero decode weight
rebuilds**. The compressed head and draft are faster against CT off. Prefill
still reconstructs 68 KDA matrices/rank for **36.5–38.7 ms**. All 14 timed sequences
match in each pair; full DFlash KL/PPL/tasks/long context remain unqualified.
Isolated compression saves **9.355840 GiB TP2**; replacement saves 12.682279 GiB.
All 75 payloads and 12,177 numerical fields verify independently.

A concrete untested latency option is to leave KDA dense while keeping expert
scales/vocabulary compressed. Replacing packed KDA (rather than keeping both
copies) could remove all KDA rebuilds and retain an estimated **6.85–7.23 GiB TP2**
saving, at the cost of giving up fused KDA decode. Actual speed/memory need a
paired trial. More KV is a separate queueing option; no new GPU experiment has
started. Original production stays stopped; long-context replacement is not
qualified. See [complete findings and evidence](GLM53_DFLASH_C1.md).
<!-- DFLASH_C1_INVESTIGATION_END -->

<!-- SERVING_TRIAL_STATUS_BEGIN -->
**Complete and independently verified, 19 October 2026, 15:21 UTC: compressed candidate is LIVE with DFlash.**
Same authenticated Spark listener `127.0.0.1:8000`, alias `glm-5.3-flash`.
Trial settings: **8,192 context**, 16 sequence slots, 2,048 prefill budget,
4 GiB KV per rank, 10 ms admission; original containers are retained stopped.
The user's harness can use the candidate now. Final auth and streaming checks pass.

Matched current → candidate short C1/C8/C16 throughput:
**48.399 → 45.754 / 74.383 → 75.704 / 74.885 → 75.512 tok/s**
(−5.46% / +1.78% / +0.84%). Long-prompt gains are +4.39% / +7.75% / +8.45%.
The unchanged tuned production reference is faster in all six cells; no
production speed upgrade is demonstrated. Common C16 logs include four running
and twelve waiting requests, so client latency includes substantial queueing.

Post-startup CUDA allocation is **12.690092 GiB lower across TP2** versus the
current checkpoint with equal KV budgets. This includes checkpoint differences;
the earlier isolated compression saving remains 9.367162 GiB.
All 150 timed output sequences match both references. **Full speculative-stack
KL, broad quality and long context remain unqualified.** Repeated diagnostic
prose yields ~99.8% candidate draft acceptance; harness behavior may differ.

All 37 evidence payloads and 11,477 numerical fields independently verify on
Spark/Mac with zero difference. V1's missing FP8-draft compatibility flag,
failed startup and successful rollback remain retained; v2 uses the existing
adapter in both arms, with CPU/source checks. Controller/watchdog completed;
candidate serving containers remain running. No new compression kernel or MTP
work was started. See [full tables, evidence and rollback](GLM53_SERVING_TRIAL.md).
<!-- SERVING_TRIAL_STATUS_END -->

<!-- MOE_BACKEND_STATUS_BEGIN -->
**Native expert screen complete and independently verified, 19 October 2026, 14:31 UTC.**
Do not integrate the tested native configuration. Complete layer time is
**57.28% longer at C8 and 41.81% longer at C16**, failing the prospective
15% time-reduction requirement. Current replay reproduces captured model outputs
bitwise and repeats exactly; native output differs by about 4.1% relative L2
and does not repeat bitwise. This component error is not full-model KL.

Traces attribute about 0.7 ms/layer to native scale reconstruction; native expert
kernels also take longer. Even the profile sums with reconstruction excluded
remain above the current complete path. No simple scale-cache change is supported
as a sufficient fix. This finding covers this backend/configuration and actual
layer, not every possible native implementation.

The original service is restored with exact container/configuration identities
and HTTP 200. All owned experiment jobs are terminal and the awake lease is
released. No configuration is promoted; native full-model KL and memory remain
unqualified. The existing warmed exact16 compression candidate remains the
measured option: about 9.37 GiB TP2 saving, about 13% faster C1, and passing
measured KL. C16 first-token latency remains an unresolved tradeoff.
The 4,096-budget follow-up also failed its mean-latency and memory policy gates.
This bounded expert/latency investigation is closed. BF16 caching, new kernels
and MTP have not started. See [backend evidence](GLM53_MOE_BACKENDS.md) and
[budget evidence](GLM53_PREFILL_BUDGET_4096.md).
<!-- MOE_BACKEND_STATUS_END -->

<!-- PREFILL_BUDGET_STATUS_BEGIN -->
**4,096-budget follow-up complete, 19 October 2026, 14:18 UTC.** The prospective policy gate
**fails**. 42/42 matched numerical comparisons pass;
CUDA saving is 9.358188 GiB across TP2. Mean/p95 TTFT, single-token ITL,
historical capacity allocation and per-trial peaks retain their original gates.
Stock C16 mean TTFT changes **1.855 -> 2.032 s (+176.7 ms)**; median throughput changes **80.996 -> 84.228 tok/s**.
Compressed C16 mean TTFT changes **1.925 -> 2.056 s (+130.7 ms)**; median throughput changes **81.956 -> 85.851 tok/s**.
Failed checks: candidate_C16_TTFT_improves, stock_C16_TTFT_nonregression, candidate_budget_peak_bounded, candidate_capacity_allocation_bounded. No configuration is promoted.
Original service restored healthy; this experiment and its collector/watchdog
are terminal and its owned awake lease is released. The separate native expert screen is also complete; see the current status above. See [complete result](GLM53_PREFILL_BUDGET_4096.md).
<!-- PREFILL_BUDGET_STATUS_END -->

**Latency follow-up authorized and in preparation, 19 October 2026, 09:17 UTC.** Test scheduler budgets
2,048 versus 2,080 at the same 10 ms admission window, with qualified warmed
exact16 compression, C1 controls, matched same-budget KL and both-rank profiles.
The target is the 23-row trailing prefill observed in the completed C16 run.
Weights remain compressed; no BF16 cache or new compression kernel. Both arms
need a bounded integer-dispatch shape extension and startup warmup through 2,080.
Worker capacity is 2,080 in both interleaved conditions; steady allocation and
per-trial peaks will be checked, including the historical 2,048-capacity baseline.
Original service remains healthy; no GPU experiment has started. MTP remains deferred.
Working evidence: `/private/tmp/mxwave-prefill-budget-v1`.
The completed admission evidence and original failed latency gates below remain valid.

**Completed and independently verified, 19 October 2026, 09:01 UTC.** The 10 ms admission
policy **passes its prospective gate**. Compressed short-C16 median TTFT at
10 ms is **1.694 s**, versus stock
**1.383 s**: a **+311.5 ms** gap.
The hoped-for gap of tens of milliseconds is not achieved under the frozen headline metric;
**the original short-C16 TTFT criterion remains unmet**. This headline is the
median of twelve trial medians. Supplementary statistics over all 192 requests
per arm at 10 ms give a **+57.4 ms mean** gap and **+65.9 ms pooled
median** gap. These describe the same complete data; they do not replace the
frozen headline metric or its decision.
Within the compressed arm, moving 3 -> 10 ms changes C16 TTFT by
**-20.56%** and throughput by
**+0.82%**. The C1 window delay is
**+8.9 ms**.

All 94 numerical comparisons pass at the unchanged KL <=1e-5 / absolute mean NLL <=1e-4 limits.
Timed output pairs match: **True** across
all **1188** retained comparisons. CUDA allocation
saving is **9.367162 GiB across TP2**. These measurements
cover short C1/C16, not every prompt length or concurrency.

The qualified warmed exact16 route remains the compression candidate; the
input-FP32 variant remains rejected. Original service is restored healthy and
all experiment/retention jobs are finished. No configuration is promoted.
The user-requested final compression experiment is complete; further compression
kernel/throughput tuning stops. MTP has not started.

See the [full admission result](GLM53_C16_ADMISSION.md) for every timing cell,
policy decision, trace counts, quality limits and evidence hashes. Sources are
frozen at 211 files with all 207 parent files unchanged. The earlier draft
admission modules are now completed, validated and measured.

Completed FP32 payloads relocated before this run remain verified on the worker
under `ROOT/retained-c16-admission-v1/<original-directory>/<relative-path>`.
Canonical archives, model inputs and sources were preserved. The storage receipt
is retained in the preparation manifest; do not redo cleanup. New head results
use hardlinks; profiler traces use copies after a preserved hardlink-permission
failure and successful archival recovery. Local working evidence is
`/private/tmp/mxwave-c16-admission-v1`; both hosts retain the final archive.

Older checkpoint statements below are historical and superseded where they
describe paused/running work or an unlaunched admission follow-up.

Snapshot: **19 October 2026, 07:43 UTC**. Input-only FP32 model analysis and retention are complete.

**Decision: retain the preceding warmed exact16 route as the numerically
qualified experimental fallback.** The new `fp32inputmodelv1` fails C16:
mean decode KL **1.601802484e-5 > 1e-5**, reproduced in all ten C16 cross-arm
comparisons. C1/C8 pass at **5.390468235e-6 / 3.791330145e-6**. All mean-NLL
checks pass. Both arms self-repeat exactly, actual grouping matches and all
300 timed output sequences match. This is a reproducible cross-arm difference;
matching sampled tokens is not equality of probability distributions.

**Verified on Spark and Mac:** 1,634 payloads / 229 sources / 12 worker traces;
all decisions agree, maximum float64 difference 1.5543e-15. All 20 quality
captures, 46 comparisons and 72 timing groups are complete. Ten numerical
comparisons fail. Original speed/TTFT gates and all prior failures remain.
No configuration is promoted. See the [complete result](GLM53_FP32_INPUT_MODEL.md),
[final findings](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/fp32-input-model-v1-final-findings.json), and
[retention manifest](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/fp32-input-model-v1-final-retention.json).

Final six-trial throughput changes, short/long prompts: C1 **+14.78/+12.24%**,
C8 **+3.13/-1.47%**, C16 **+0.84/-2.03%**. Short TTFT is C1
**0.391 -> 0.432 s (+41 ms)** and C16 **2.079 -> 2.211 s (+132 ms)**;
long-C16 adds 519 ms. CUDA saving stays **9.367162 GiB across TP2**. CPU RSS
is 242.18 MiB higher before quality / 250.52 MiB higher after ordinary timing;
post-profile RSS is unsuitable for a CPU-saving claim.

Trace analysis verifies 34 FP32 inputs + 34 original outputs + 34 reducers,
graph replay and zero decode reconstruction on both candidate ranks. Mixed
KDA kernel sums fall to about **20.36-20.95 ms**, versus ~22 ms previously;
expert work still takes ~60-61 ms at C8 / ~97-98 ms at C16. No recognized JIT
event overlaps timing and caches stay unchanged. All 640 current stock output
rows equal previous stock bitwise; only 290/640 candidate rows equal previous
candidate, under matched inputs/grouping. The first divergent layer is not
localized; no implementation-bug diagnosis is established.

**Operational state:** original GLM restored with exact identities/configuration,
health 200 rechecked ~07:39 UTC. Controller 1258389, collector 1258388, watchdog
1262165 and recovery 1893667 have exited; Mac retention finished successfully.
The original failed watcher is retained, not restarted. Owned awake lease
75605 and completed SSH transport 41123 are released. No GPU experiment or
retention watcher is running. The broader goal remains paused after “but wait”;
“let's see the outcome” authorized observation/review of this completed run.

The initial verifier/collector failed on an expected-grid error: split 2 needs
**12,608 * 2 / 64 = 394**, not 197. The separate offline recovery changes only
that checker expectation. Original failures and all 229 frozen experiment
sources/gates are preserved. Seven negative fixtures pass; the worktree reviewer
fix passes 14 focused tests and targeted Ruff. No model rerun occurred.

**Next scope:** the user ended compression-kernel optimization and accepts
measured concurrency parity. Proposed final compression follow-up: **3 versus
10 ms admission**, using the qualified warmed exact16 fallback in the compressed
arm, with short-C1 controls; then MTP. Neither has started. Three local modules
(`c16_admission_protocol.py`, `measure_c16_admission.py`,
`analyze_c16_admission.py`) are unfinished/unvalidated, absent from the 229-file
freeze and not launchable. The earlier 781-test/Ruff/mypy pass applies to the
completed frozen preparation, not those drafts. No launcher/source freeze exists
for the follow-up. Finish and review only when that phase resumes.

TTFT remains open: all previous/current timing already uses 3 ms. Current
short-C16 first cohorts are stock 4/6/1/7/3/5, candidate 7/5/6/7/2/1.
The smaller gap does not prove an FP32 latency gain. The retained fallback's
short-C1/C16 measurements stay **0.384 -> 0.422 s / 1.838 -> 2.280 s**.
Sixteen 256-token prompts need at least two forwards under a 2,048-token budget.
The qualified fallback crosses stock's C1 median arrivals at token 8 short /
4 long; this failed FP32 candidate crosses at 7 / 6. No per-request guarantee.
V6's 256-row fused route already failed its full screen. A 2 GiB/rank BF16
cache would leave ~5.37 GiB TP2 saving arithmetically, but is unimplemented and
unmeasured. Do not claim near-zero TTFT penalty or repeat naive failed overlap.

The historical prefill resource review covers 28 forwards / 1,904
reconstructions. At 2,048 rows, expert blocks use all 102,400 shared-memory
bytes/SM and cannot co-reside with a decoder block; smaller observed rows leave
capacity only, not proof of useful overlap. CPU-memory observer skipped its
stock capture under its original guard and no candidate-only capture was made.
Memory attribution, broader quality and calibrated roofline claims remain open.

Preserve storage relocation: 14 completed directories / 10,457 files /
7,409,394,725 logical bytes are verified on the worker under
`ROOT/retained-fp32-input-model-v1-v3/<original-directory>/<relative-path>`.
Canonical head archives, sources, model inputs and protected fixtures remain.
`ROOT=/home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/full-model-v1`.
Do not redo cleanup or overwrite the frozen KDA parent with editable worktree code.

Local evidence: `/private/tmp/mxwave-fp32-input-model-v1/evidence.tar.gz`;
remote `ROOT/evidence-fp32-input-model-v1-recovery-v2.tar.gz` (465,586,181 bytes).
Archive SHA256: `bfbbafa8f53bdbd2c82bd2e6cc3d6f48b678433b015beb0ff7863bce9ebca8c2`.
Frozen source SHA256: `aecf6a332294e165dc6a60b4c7e3731e87944c41075d2304dbec82b6e8305cdc`.
Plan SHA256: `aa8af27edbf5fe1d3addacdb1775df3ce4eb38b6de553b993aef5f2fd8cf138c`.
Prepared SHA256: `0be6b5f3787e43324c08307e2321fbc44589884ea42b05cac899e3119c7151f5`.
Corrected verifier SHA256: `d89d085a72172e09a6ec6f75c2c563ab0ccac33a9abf73499583197109639827`.
The pre-final live handoff/current-state snapshots are retained byte-for-byte in
the final manifest. Older checkpoints below are historical and superseded by
this snapshot where they describe pending work.

**Completed input-only FP32 integration, 18 October 2026, 22:50 UTC / 19 October 00:50 Berlin:**
`fp32inputv1` independently verifies on Spark and Mac: **434 payloads /
227 sources**. Both GPU jobs exit zero. All **120 retained-output comparisons**
and **66 raw native-fallback comparisons** pass. Four compiled M8/M16 captures,
two compiled M8 -> native M17 -> M8 sequences, and six simultaneous two-stream
fixtures pass. All 192 graph checks retain zero additional allocation after
replay; all 384 stress replays preserve the expected outputs and guards. Each
two-stream fixture owns distinct slots and rejects a third stream.

The input-only adapter uses FP32 split 2 for physical rows **1/4/8/16**, reusing
the existing **206,569,472-byte/rank two-slot pool** without a separate scratch
allocation or resident BF16 weight cache. All output projections and untested
input row counts preserve the qualified path. Compiled capture extra bytes are
**[202240, 403456] / [202240, 403456]**
on ranks zero/one, below the 16 MiB bound that detects a hidden pool clone.
Peak fixture CUDA allocation is **376,649,728 /
376,649,728 bytes**, below 2 GiB/rank.

**This is an integration pass, not a new full-model result.** Matching the
retained FP32 output does not establish equality to stock GEMM output. All
**216 prior exact-word failures** and **two input split-2 timing repeat failures**
remain failed. No new API speed, model KL/NLL or CPU-memory result exists, and
no configuration is promoted. The preceding component timings suggest roughly
1.0-1.4 ms saved per decode forward/rank by extrapolation; API impact remains
unmeasured. The warmed exact16 model remains the measured C1/GPU-capacity candidate.

All **220 speed-parent sources remain unchanged**, with seven explicit additions.
The initial source-freeze refusal is retained: the editable worktree KDA file
differs from the qualified frozen parent, so the parent was preserved and the
exponent helper added explicitly. Local validation passes **770 tests / one GPU
skip, Ruff and strict core mypy**. Ten adapter/lifetime tests also pass against
the actual frozen dependencies without CUDA initialization. The previous
archival extension correction preserves every script byte/hash using `.py.txt`.

Archive: **138,780,977 bytes**, SHA256
`a7b9f74fa1df33108caf61cc65db4c63997bbcf8e9dd378f4890efc65f1b5e80`.
Local: `/private/tmp/mxwave-fp32-input-v1/evidence.tar.gz`; remote:
`ROOT/evidence-fp32-input-v1.tar.gz`. Both reviewers bind the precision and speed
parent archives. Reviews, failures, findings and operational receipts are retained
under `diagnostics-v1/fp32-input-v1-*`.

**Operations complete:** original GLM restored **22:46:46 UTC**;
independently rechecked **22:49:42 UTC**, health 200 with exact original
identities/configurations. Controller **236636**, collector **236635**, watchdog
**238423**, both GPU jobs and Mac retainer **65803** are terminal. Owned awake
lease **65158** was identity checked and released. Do not rerun completed jobs.

**Historical integration-checkpoint plan, now launched above:** prepare fresh
stock versus compressed+input-FP32 full-model quality,
ordinary API timing, memory and trace measurements. Keep native mHC, matched
quality cohorts, stable sparse/expert order, startup warmup, exact16 output,
**KL <=1e-5 / absolute mean NLL <=1e-4**, and the original speed/TTFT gate.
Verify adequate evidence storage before launching; the head is short of the
previous full-model reserve. CPU-memory improvement, concurrent serving gains,
prefill/admission latency, wider qualification and the calibrated roofline remain
open. MTP stays deferred. That model experiment is now launched above.

**Completed FP32 speed diagnostic, 18 October 2026, 22:31 UTC / 19 October 00:31 Berlin:**
`fp32speedv1` independently verifies on Spark and Mac: **246 payloads / 220
source files**, **884,736 CUDA-event samples / 288 conditions** over 12 actual
rank-specific KDA input/output matrices at layers 0/22/44. Both GPU jobs exit
zero. Every before/after output matches the retained precision fixture; weight
bytes, workspace guards, graph/eager agreement and zero retained replay
allocation pass. All 217 parent sources and native libraries remain unchanged.

**The useful next candidate is input-only FP32 split 2, with modest component
gains.** The following medians cover all six matrices across both GPUs, compared
with the existing split-one compressed kernel. Each condition also compares
the same-compiler rebuilt control and tests all three arms' repeatability.

| Physical rows | Input time saved, warm | Input time saved, pressure | Strict timing passes |
|---|---:|---:|---:|
| M1 | 8.22% | 8.05% | 12/12 |
| M4 | 8.40% | 8.22% | 12/12 |
| M8 | 7.94% | 8.39% | 10/12 |
| M16 | 6.72% | 7.25% | 12/12 |

Input split 2 passes **46/48** strict timing conditions. The two M8 pressure
conditions fail unchanged repeat controls; all input split-2 speed comparisons
pass, but those two conditions remain unqualified. Sampled per-projection
savings are about **29-41 us** by group median. Multiplying those medians by 34
KDA layers gives only an **estimated 1.0-1.4 ms per decode forward/rank**; this
is not measured API latency, a model-throughput result or coverage of every layer.
The earlier large C16 reconstruction fallback was already fixed by exact16.

Output split 2 saves about **28-29% warm / 5-6% under pressure** by group median,
but passes only **42/48** strict timing conditions. Split 4 illustrates why a
warm-cache win is insufficient: at M8 its median output time is about **34%
lower warm but 4.16% higher under pressure**. Larger split factors offer no
uniform improvement. No broad FP32 policy is integrated or promoted.

**The frozen broad timing screen fails:** **78/288** conditions fail (40 on
rank zero, 38 on rank one): 70 candidate-speed failures, seven repeat-control
failures and two rebuilt/original-equivalence failures, with overlap. Every raw
sample, both ABBA/BAAB directions and failed decision remains retained. The
previous **216 failed M4/M8 exact-word precision checks also remain failed**.
This speed experiment establishes no new model KL/NLL, serving-speed or
CPU-memory result. The current warmed exact16 model remains the measured
C1/GPU-capacity candidate; its overall serving-speed gate still fails.

Telemetry retains **4,608 boundaries/rank**: SM clocks **2,405-2,535 /
2,398-2,528 MHz**, temperatures **46-73 / 46-74 C**, no sampled throttle flags;
memory clocks are unavailable. Boundary samples cannot exclude transients.
Peak CUDA allocation is **311,241,216 / 311,248,896 bytes**, below 2 GiB/rank.
Graph timing includes runtime input padding and logical output cropping.
Local validation: **762 tests passed, one GPU skip, Ruff and strict core mypy**.

Archive: **13,315,462 bytes**, SHA256
`55dde1aedfa17bdf97cb607a51d318edb7c3ae0202ab3081335657cb8aa360cc`.
Local: `/private/tmp/mxwave-fp32-speed-v1/evidence.tar.gz`; remote:
`ROOT/evidence-fp32-speed-v1.tar.gz`. Its reviewer also requires the immutable
parent precision archive SHA
`e40d606164bbf8ead44a3f932f6f615d1d7b4be70d60a6a17facf0ce571ae94b`.
Native inputs are bound to that archive rather than duplicated. All decisions
match on Spark/Mac. Both reviews, findings, scripts, proofs and retained control
failures are under `diagnostics-v1/fp32-speed-v1-*`.

**Operations complete:** originals restored **22:29:52 UTC**, independently
rechecked **22:30:40 UTC**, health 200 and exact identities/configurations.
Controller **40765**, collector **40764** and watchdog **42417** are terminal;
both GPU containers are stopped, exit zero, no OOM. The exclusive Mac retainer
**60005** completed; do not rerun it. Owned awake lease **56814** was identity
checked and released **22:31:09 UTC**. Its initial receipt lookup was sandbox
denied, then recovered by identifying the exact sole matching process.

Storage recovery removed only **2,230 verified duplicate unpacked files /
1,061,076,481 bytes** on the head after matching every path, size and hash to
the immutable archive. Parent archives remain verified on both Spark hosts and
Mac, with the Mac unpacked evidence also retained. Original fixtures/native
builds/raw GPU outputs remain. Plan evidence space before another full-model run.

**Next:** prepare a bounded **input-only split-2** full-model probe, retaining
native mHC, matched grouping, stable sparse/expert ordering, startup warming,
exact16 output and the unchanged **KL <=1e-5 / absolute mean NLL <=1e-4** limits.
Only route the tested physical row counts 1/4/8/16 through the new kernel;
leave untested counts on the qualified route until independently covered.
Typed scratch capacity, stream ownership, graph replay and allocation
are now verified by the input-only integration checkpoint above. This exploratory model test must retain the component
precision and timing failures; it cannot relabel them as passes. CPU-memory
savings, concurrent serving gains, prefill/admission latency, wider quality
qualification and the calibrated full roofline remain open. MTP stays deferred.

**Completed precision experiment, 18 October 22:12 UTC / 19 October 00:12 Berlin:**
`fp32splitv1` independently verifies on Spark and Mac: **2,227 payloads / 217
sources**, 12 actual input/output KDA matrices at layers 0/22/44 across both rank
shards. Both GPU jobs exit zero. All **576 candidate comparisons** are retained,
including all **216 failed M4/M8 exact-word checks**. No serving policy is changed.

**FP32 partials remove most of the old split-K error, but do not restore exact
stock rounding.** Every tested candidate is finite and below the unchanged
**0.001 relative-L2** bound; the maximum is **0.000265301**. All **144 basis
comparisons** are exact. Random M4/M8 relative error improves by median paired
factors **25.56 / 22.39** over the BF16-partial path. The old BF16 split results
remain in the same evidence, with relative errors about **0.00226-0.00318**.

| Physical rows | Random candidate comparisons | Relative-L2 range vs stock | Failed frozen checks |
|---|---:|---:|---:|
| M1 | 108 | 0 to 2.65301e-4 | 0 (relative-error gate) |
| M4 | 108 | 3.73201e-5 to 2.42813e-4 | 108 (exact words required) |
| M8 | 108 | 6.08115e-5 to 2.18394e-4 | 108 (exact words required) |
| M16 | 108 | 6.97323e-5 to 1.79198e-4 | 0 (relative-error gate) |

The native M4/M8/M16 controls are bitwise equal to stock on the same inputs;
the FP32 candidate changes rounding. M4/M8 differ in **19,093 of 10,803,456
words (about 0.177%)**, with 17,834 differing by one adjacent BF16 value. This
is not a claim that every difference is one ULP or that model quality is unchanged.
The relative-L2 metric is not model KL. All split factors 2/4/8 therefore retain
**failed** broad precision-screen decisions. M1/M16 relative-error passes do not
qualify a new full-model route or override M4/M8 failures.

Source bytes, decoded weights, rebuilt-vs-original control bits, guarded typed
workspaces, undersized-workspace rejection, within-arm repeats, nondefault-stream
graph capture/replay and zero retained replay-allocation growth all pass. Raw
BF16 outputs are independently rehashed and their numerical errors recomputed
on both hosts. Peak CUDA allocation is **313,282,048 / 313,285,120 bytes**.
Telemetry boundaries record **44-47 C** on both ranks, SM clocks **2,405-2,457 /
240-2,431 MHz**, and no sampled throttle flags; memory clocks are unavailable.
This precision-only stage does not measure performance, model KL or CPU savings.

The isolated native API retains the MMA loop, writes partials as FP32 and reduces
once to BF16. Its distinct `mxzip_mm_fp32` entry point requires workspace bytes;
it cannot be loaded as the original GEMM ABI. Both same-compiler native builds
complete in a CPU-only container. The Apache-2.0 external-source license/notices
and all staged hashes are retained. Local checks: **750 tests passed, one GPU
skip, Ruff and strict core mypy**; all 213 parent source bindings are unchanged.

Archive: **641,093,041 bytes**, SHA256
`e40d606164bbf8ead44a3f932f6f615d1d7b4be70d60a6a17facf0ce571ae94b`.
Local archive/evidence: `/private/tmp/mxwave-fp32-split-v1/`; remote archive:
`ROOT/evidence-fp32-split-v1.tar.gz`. Both independent reviews, raw-error findings,
source/native provenance and preparation/final retention inventories are under
`diagnostics-v1/fp32-split-v1-*`. The exclusive Mac retainer has completed; do not
rerun it. The initial CPU disk-reserve failure is preserved. Recovery removed
only the verified duplicate head transfer archive, retaining the identical
worker archive and all unpacked inputs. The first optional local histogram
script failed on a shadowed variable; its source/log and corrected v2 analysis
are retained. The frozen experiment and independent verifiers are unchanged.

**Operations complete:** originals restored at **22:04:17 UTC**, independently
rechecked at **22:10:07 UTC**, health 200 and exact identities/configurations.
Controller 4114884, collector 4114882 and watchdog 4116600 are terminal. Both
GPU containers and the CPU build container are stopped; the latter had no GPU
device requests. Owned awake lease 50327 was identity-checked and released at
**22:11:00 UTC**. Head disk free space is about **1.51 GB decimal**; plan retention
space before another experiment, preserving all unique evidence.

**Historical precision-checkpoint plan, now completed above:** measure FP32 split
performance before deciding whether a full-model quality probe is worthwhile.
Any such probe must retain these exact-word failures and keep the full-model
**KL <=1e-5 / absolute mean NLL <=1e-4** gates. Passing model gates would answer
a different question from reproducing stock GEMM bits; do not relabel this
failed exact-word screen as passed. No configuration is promoted. Keep the
warmed exact16 model as the experimental C1/GPU-capacity candidate. Concurrent
speed, prefill/admission latency, CPU-memory savings, wider quality qualification
and the calibrated full roofline remain open; MTP stays deferred.

**Completed diagnostic, 18 October 21:52 UTC:** `denseoutputv1` is retained and
independently verified on Spark and Mac: **270 payloads / 213 sources**, six
actual rank-specific output matrices, **221,184 CUDA-event samples / 144
conditions**. Both GPU jobs exit zero. Storage bits, guards, within-arm repeats,
eager/graph agreement and graph-allocation checks pass. The arithmetic screen
passes (maximum relative L2 **0.000353804**); fused decode is not generally
bitwise equal to stock GEMM. No full-model experiment was run for this policy.

**The broad dense-only output policy is rejected by its frozen performance
screen.** All six matrices fail warm-cache M1 in both eager and graph execution:
**12 repeatable cost failures**, plus **six separate unstable repeat controls**.
Warm-cache M1 adds **56.22-65.09 us/projection eager / 59.63-68.33 us graph**,
exceeding the 50-us allowance with stable controls. All 18 failed conditions
remain in the evidence; no gate is relaxed.

Graph + 128-MiB pressure adds **19.98-26.14 us** at M1 (one unstable control),
**10.28-14.43 us** at M8 and **11.36-16.23 us** at M16. Prefill saves
**205.15-212.66 us** at M54 (one unstable control), **244.53-252.09 us** at M256
and **263.12-283.74 us** at M2048 per projection. These component measurements
are not API latency or a full-model speed claim. Cold/pressure success does not
override the warm-cache C1 failure. Replacing all output storage would retain a
projected **8.744 GiB** TP2 CUDA saving; this is arithmetic on parent storage,
not a new model-memory measurement. No dense-output policy is integrated.

Boundary telemetry covers **2,304 samples/rank**, SM clocks **2,372-2,535 /
2,327-2,528 MHz**, temperatures **46-73 / 46-72 C**, and no sampled throttle
flags during timing. Initial rank-zero warmup retains a throttle flag and clock
ramp; memory clocks are unavailable. Boundary sampling cannot exclude transients.
Peak CUDA allocation is **427,129,856 / 427,180,032 bytes**, below 2 GiB/rank.

Archive: **163,211,537 bytes**, SHA256
`cfcdfb279dcbc6fc8d7bda41bc914eee7b654d04e3fd6aa07d6e15950de7316b`.
Local archive/evidence: `/private/tmp/mxwave-dense-output-v1/`; remote archive:
`ROOT/evidence-dense-output-v1.tar.gz`. Both independent reviews, findings,
all failed decisions, memory arithmetic and retention inventory are under
`diagnostics-v1/dense-output-v1-*`. The exclusive Mac retainer completed;
do not rerun it. All 210 parent sources remain unchanged; 13 preexisting
worktree differences are retained and excluded. Local validation remains
**744 tests passed, one GPU skip, Ruff and strict core mypy**.

**Operations complete:** original GLM identities/configurations and health 200
were independently rechecked at **21:51:31 UTC**. Controller 3999584, collector
3999583 and watchdog 4001232 have exited; both diagnostic containers are stopped.
Owned awake lease 44081 was identity-checked and released at **21:51:43 UTC**.

**Source audit before the active prototype:** retained licensed native source confirms
that old split-K writes partial accumulators to BF16 before reducing them.
FP32 partial storage might remove that extra rounding, but requires consistent
kernel/API/bridge/workspace changes and unchanged numerical gates. At that checkpoint no kernel
was modified, compiled or run; the active prototype is tracked above. The original BF16 split-K failures remain valid;
this is an untested hypothesis, not a fix result. Audit provenance is retained
under `diagnostics-v1/fp32-split-audit-v1-*`.

Keep the warmed exact16 model as the current experimental C1/GPU-capacity
candidate. Concurrent throughput, prefill/admission latency, CPU-memory savings,
wider quality qualification and the calibrated full roofline remain open.

**Completed diagnostic, 18 October 21:33 UTC:** KDA memory/counter evidence is
retained and independently verified on Spark and Mac: **560 payloads / 210 source
bindings**, 22 operations and 28 NCU kernel records on each physical GPU. All four
timing/profile processes exit zero; exact outputs and guards pass. All **22,528
CUDA-event samples** remain, including **19 / 16 failed 2% repeat controls** out
of 88 conditions per GPU. This diagnostic changes no model or serving policy.
The preceding 207 model sources are unchanged; full local validation remains
**735 tests passed, one GPU skip, Ruff and strict core mypy**.

Graph + 128-MiB-pressure logical payload rates are **197.36 / 198.32 GB/s** for
input reconstruction and **199.70 / 203.36 GB/s** for output reconstruction, versus
**218.54 / 220.37** and **218.44 / 217.55 GB/s** for the corresponding dense copies.
The rank-one input repeat control fails; it is retained, not excluded. These
ratios are descriptive, not paired optimization gates or measured DRAM efficiency.
The counters retain approximately **204 MB input / 66 MB output L2 requests**,
close to the corresponding dense-copy traffic, but about **41.7 / 42.0 times**
as many instructions. Long-scoreboard has the largest decoder stall ratio.
This supports investigating memory waits, not a proven isolated address-dependency
cause. The evidence does not support expecting a large decoder-only rewrite gain.
Avoiding prefill reconstruction is the stronger next hypothesis.

Telemetry retains 1,408 boundaries per rank: temperatures **45-62 / 45-63 C**, no
observed throttle flags, and SM clocks **221-2,535 / 2,372-2,528 MHz**. The head
includes clock ramping; memory clocks are unavailable. Boundary samples cannot
exclude transients or explain away failed controls. Peak CUDA allocation is
**698.16 MiB/rank**, below the 2-GiB bound. Both GPUs use the same layer-zero,
rank-zero input/output fixtures; this is not coverage of every layer or rank shard.

**Offline attribution recovery is complete.** The frozen verifier correctly
retains `diagnostic_complete: false` because the original CSV lacks NVTX names.
The first named export also changed units/precision; its v1 reviewer failure is
retained. Explicit base-unit export proves every kernel ID, metric and unit equal
using exact decimal comparison. The v2 reviewer additionally exposed the tool's
explicit `(int)` template-cast spelling. The v3 reviewer accepts only that exact
known spelling difference and reproduces all 22 labels / 28 records per GPU on
both hosts. Original sources, CSVs, binary reports, failures and all timing gates
remain intact. No GPU experiment was rerun for export/review recovery.

Archive: **98,606,261 bytes**, SHA256
`30ac97161074f0a965a23776518fb2ad4bda12a21b840f5edc8d95e6e1e339b2`.
Local archive/evidence: `/private/tmp/mxwave-kda-memory-v1/`; remote:
`ROOT/evidence-kda-memory-v1.tar.gz`. The separate export supplement SHA256 is
`7299dd5eb58ca9e4e2f44b3317bac5231f71938def77aefb7d10e71a25bc83de`.
Repository `diagnostics-v1/kda-memory-v1-*` retains both independent reviews,
exact attribution proof, counter findings, all 56 L2 arithmetic-intensity points,
telemetry, failed reviewer sources/logs and the final retention inventory.
[Counter findings](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/kda-memory-v1-counter-findings.json)
and [attribution proof](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/kda-memory-v1-attributed-v3-cross-host-verification.json)
are complete; a calibrated full-model roofline is still outstanding.

**Operations finished:** originals restored at **21:25:28 UTC** and independently
rechecked at **21:32:44 UTC**, health 200 and exact identities/configurations.
Controller 3832759, collector 3832758 and watchdog 3834262 have exited; no diagnostic
container is running. The exclusive Mac retainer and all analyses are complete;
do not rerun them. Owned awake lease 11270 was identity-checked and released at
**21:32:56 UTC**. Initial missing-fixture and profiler-capability preparation
failures and all three preparation receipts remain in the archive.

**Follow-up completed:** the proposed dense-output storage screen is recorded
above and fails warm-cache C1. Keep startup warming and exact16 as the current
candidate. No new full-model run or runtime promotion follows this diagnostic.
CPU-memory reduction, wider qualification and the full roofline remain open.

**Verified full-model result, 18 October 21:10 UTC:** startup warming works in
the measured workload. All **1,612 payloads / 207 sources / 12 worker traces**
independently verify on Spark and Mac. All **46 numerical comparisons** pass at
the unchanged KL <=1e-5 / absolute mean NLL <=1e-4 limits; all **300 paired timed
output sequences** match. Native mHC and matched quality grouping remain in both
arms. This is finite numerical qualification, not universal bitwise equivalence.

| Workload, 64 output tokens | Stock tok/s | Compressed tok/s | Change | Median TTFT, stock -> compressed |
|---|---:|---:|---:|---:|
| C1 / 256 prompt tokens | 14.784 | 16.748 | +13.28% | 0.384 -> 0.422 s |
| C1 / 2,048 prompt tokens | 12.518 | 13.999 | +11.82% | 1.183 -> 1.205 s |
| C8 / 256 prompt tokens | 60.757 | 62.564 | +2.97% | 1.169 -> 1.192 s |
| C8 / 2,048 prompt tokens | 31.488 | 31.530 | +0.13% | 6.586 -> 6.662 s |
| C16 / 256 prompt tokens | 80.870 | 81.108 | +0.29% | 1.838 -> 2.280 s |
| C16 / 2,048 prompt tokens | 36.628 | 36.477 | -0.41% | 11.395 -> 11.594 s |

Six trials/case/arm, 256/2,048 prompt tokens and 64 output tokens, two fresh loads
in fixed stock-then-compressed order. **The original overall speed gate still
fails**: C8/C16 miss the 5% throughput target, and short-C1/C16 exceed the TTFT
allowance. All trials, admission differences and historical failures are retained.
CUDA allocation is **9.367162 GiB lower across TP2** at every measured stage.
CPU process RSS is **158.78 MiB higher before quality / 166.83 MiB higher after
ordinary timing / 173.72 MiB higher after profiles** across TP2. These overlapping
Spark counters must not be added. No CPU-memory improvement is demonstrated.

**The warmup fix is confirmed for the retained workload:** all 34 dispatch
variants are prepared at startup; both-rank caches remain unchanged through
quality, all 72 timing groups and all profiles. No recognized JIT log event
intersects a timed request envelope. The earlier short-C8 2.14 s gap does not
recur: the current maximum arrival gap is 0.448 s candidate / 0.666 s control.
This finite observation does not exclude every future or unlogged stall. The
warmup adds 31.3-31.6 s at startup. One-time trim releases about 47.5-48.8 MiB
RSS per worker, but total startup still adds about 76.6-79.9 MiB per worker.

**No observed numerical change from warming:** all 576 full-vocabulary rows in
**each arm** match that arm's previous exact16 run byte-for-byte, including
prompts, conditioning, generated IDs and actual grouping. Old and new raw rows
were independently rehashed. Within-arm C1/C8/C16 repeats have zero KL/NLL;
cross-arm decode KL is **5.804e-6 / 1.194e-6 / 8.919e-7** respectively. The
compressed GEMM still has small nonzero cross-arm rounding differences.
PPL, task accuracy and SQNR are not newly measured by this experiment.

**Latency/trace assessment complete:** all 600 requests and 38,400 actual
single-token arrivals are included. Median last-32-token mean gaps fall from
62.5 to 53.3 ms at short C1, 113.9 to 110.6 ms at short C8 and 159.0 to 154.3 ms
at short C16. Long-prompt prefill offsets much of this concurrent decode gain.
All 56 candidate physical-16 decode ranges and 50 physical-8 ranges replay graphs
with **68 fused KDA calls and zero weight reconstructions**. The controlled full
C16 step spends about **22.0-22.3 ms/rank** in fused KDA; its two projection grids
account for 15.35-15.40 ms and 6.62-6.90 ms. Thus the earlier reconstruction
fallback is gone. Mixed C8 expert GEMMs consume about **61 ms/rank**, and mixed
C16 about **96-99 ms/rank**, versus KDA around 22 ms. Prefill still has 68 weight
reconstructions, **35.8-38.0 ms/rank** in the controlled traces. Kernel sums may
overlap and are not API latency, measured bandwidth or a TP critical path.
The separate mixed profiles do not enforce equal admission across arms.

**Short-C16 TTFT needs a bounded interpretation:** the original median rises
**1.838 -> 2.280 s (+24.04%)** and remains a failed gate. Initial admitted counts
are control **8,4,1,8,8,7**, candidate **4,8,6,5,3,5**. Across all 96 requests per
arm, mean TTFT rises **1.882 -> 1.997 s (+6.11%)**, while mean time after first
until last token falls **10.667 -> 10.497 s**. The mean is a supplementary
arrival decomposition, not a replacement gate. Different admission amplifies
uncertainty in the median; it does not establish that the whole penalty is noise.
Controlled prefill profiles retain a real reconstruction cost. Timing-boundary
samples show control 2450-2496 MHz / 64-77 C and candidate 2437-2496 MHz / 70-81 C;
fixed load order and sampled telemetry cannot exclude transient thermal effects.

**Operations complete:** model measurement finished **20:58:14 UTC**. Both
original service identities/configurations/mounts are exact and running, health
**200**, independently checked again at **21:08:26 UTC**. Original controller
3129756, collector 3129755, watchdog 3134792 and old Mac watcher 82795 have exited.
Recovery process 3726986 also completed. No warmedmodelv1 container is running.
Mac retention and all seven final analyses completed successfully. The preliminary
timing digests and startup objects match the final archive exactly. Do not rerun
an exclusive retainer or poll old exec session 41250. All source/runtime checks
remain the frozen **728 tests passed, one GPU skip, Ruff and strict core mypy**;
subsequent work only analyzes immutable evidence and updates documentation.

**Preserved verification failure and recovery:** the original offline verifier
incorrectly expected HTTP **207**, caused by a broad source-count replacement.
The corrected verifier changes only that health literal to **200**; source count
remains 207. Negative checks reject 207, paused service and changed identities.
Original verifier, failed collector receipt/log and failed Mac watcher status
remain retained. No model measurement, source or numerical/speed gate changed.
Both hosts independently verify the separate recovery archive, with identical
decisions and maximum float64 difference **1.776e-15**.
Archive **466,928,838 bytes**, SHA256:
`0fc867240d04e1770e2970143f48f630551feb35f53b739403c4fb6b8bde64e0`.
Remote: `ROOT/evidence-warmed-model-v1-recovery-v2.tar.gz`;
receipt: `ROOT/warmed-model-v1-recovery-v2-evidence-check.json`.
Local archive/evidence/reviews: `/private/tmp/mxwave-warmed-model-v1/`.
Corrected verifier SHA256:
`e031bd5c75ff5177bd6dda81a4571923911dcd4ac3b934f25b72ca5b12395932`.
Repository diagnostics retain `warmed-model-v1-*` proofs, executed analyses,
all final review JSON and recovery provenance. `final-findings.json` binds the
preliminary observations to the final reports; original failures remain unchanged.
Read the [final findings](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/warmed-model-v1-final-findings.json)
and [independent verification](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/warmed-model-v1-cross-host-verification.json).

**Pragmatic candidate and remaining work:** exact16 input/output plus startup
warming is a useful experimental candidate for C1 and GPU-capacity savings.
C8/C16 are near throughput parity with a prompt-processing cost; the current
measurements do not support a general speed or CPU-memory win. The next useful
speed target is prefill reconstruction/admission under controlled grouping,
not another unchanged decode-fallback test. A proposed policy must preserve
bounded memory and the measured C1 gain, pass a focused component test, and then
receive fresh paired model qualification. Earlier failed prefill kernels and
naive dense caches are not accepted fixes. Keep existing gates and outliers.
The requested roofline follow-up and broader candidate qualification remain
outstanding; MTP is deferred. No further GPU experiment is currently running.

**Awake leases released:** previous owned lease 60808 was verified and retired.
The final-analysis lease 42912 was identity-checked and released at **21:11:21 UTC**
after all evidence and analyses completed. No other system assertion was touched.

**Earlier component, now followed by the full-model result above:** stablewarmv1 passes on both GPUs, independently
verified on Spark/Mac: **699 payloads / 203 source bindings**, identical decisions.
All 34 unchanged deterministic expert-order variants compile in **33.701 / 34.498 s**;
subsequent calls peak at **85.79 / 95.89 microseconds**, below the frozen 100 ms
bound. Integer outputs, metadata and guards are exact. Reuse at all tested token
limits and after heap trimming produces no additional compilation or retained
CUDA allocation. Each fresh process reserves 2 MiB in the CUDA allocator.

The one-time trim reclaims **12.80 / 12.91 MiB RSS** in this scratch component.
First-call RSS growth includes importing vLLM and the worker modules; it is not
an estimate of added memory in an already loaded model. At component completion, model CPU savings and ordinary serving latency/speed
were unmeasured; the full-model result above now supplies those measurements.

At component completion the original service was restored with exact
identities/configs/mounts and health 200, checked at **19:52:21 UTC**. Controller 3010759, collector 3010758 and watchdog
3034127 are terminal. No serving configuration is promoted. Final archive:
**9,070,086 bytes**, SHA256
4b641179fcce13473c17f70e0ebd34f1ed4baa250c67955d6f8a99ae3b5fdddd.
Local archive/review: /private/tmp/mxwave-stable-warmup-v1/.
Repository diagnostics retain the frozen plan, check receipt, operational sources,
independent review and cross-host verification.

**Frozen full-model protocol:** warmedmodelv1 keeps all 200
exact16-parent sources and adds seven files. Both arms use the qualified startup
warmup, one-time heap trim, native mHC and identical deterministic policies.
The prepared protocol retains the original numerical/speed limits, compares
C1/C8/C16 with short/long prompts (72 ordinary timing groups), and profiles mixed
prompt lengths at C8 and C16. All 46 numerical comparisons are retained.
CPU rehearsal rejects mismatched grouping, excessive KL and new compilation.
Full Ruff, strict core mypy and pytest pass (**728 passed, one GPU skip**).
Both-node source/native/actual-image CLI preparation completed successfully;
prepared SHA256:
ab7ed93880a0d2338d13d1b8da736f6b813a40aa9cda4f08d75b3a6128f4431d.
Verified relocation of completed evidence and raw result copies raised head free
space to **9.33 GB** before staging. All canonical archives, full worker backups
and original metadata are retained. Root-owned generated bytecode is left in
place. Initial retention attempts and recovery receipts are preserved; no
checkpoint or runtime source was changed by the storage work.
Source archive SHA256:
dfadc2307510b123bed71974853effc6214d1eef2cdc8cdca3cf330c34d595e1.
Plan SHA256:
822ad583531d2dbaaa88a77a6f50bfd6e9c7dc973303d447f9f07d8cb4cadde5.
Local preparation: /private/tmp/mxwave-warmed-model-v1/.
Offline arrival, resource and per-forward trace analyses are prepared separately
from the frozen runtime. They have consumed the independently verified archive successfully,
retaining every trial and all original gates. CPU RSS and CUDA counters overlap
on Spark and must remain separate.

**Previous completed diagnostic:** latencyprobev1 completes all 30 groups and independently
verifies on Spark/Mac: **271 payloads / 200 source bindings / four worker traces**,
with identical decisions and zero float64 difference. All 106 paired output
sequences and requested splits match. Exact original service identities,
configs/mounts and health 200 are restored; controller/collector/watchdog have
exited. This diagnostic does not replace numerical or ordinary serving-speed gates.

**Practical assessment:** the verified warmed candidate saves 9.367 GiB CUDA
allocation/TP2 and improves C1 throughput 11.82-13.28%. C8/C16 remain near parity.
The identified first-use dispatch compilation is moved to startup without
changing the recorded outputs. Prefill reconstruction and admission remain
latency costs, and process RSS is higher. The original speed gate still fails.
See the current checkpoint above before interpreting the historical records below.

## New trace finding: both arms show first-use compilation in the shared wrapper

**19:33 update:** the candidate head trace corroborates the stock finding.
The two first-use GPU gaps are **1.929241 s / 1.507655 s**; repeats have largest
gaps **0.580 ms / 0.315 ms**. Candidate host elapsed falls **2.419 -> 0.489 s**
for the 1536-row prefill and **1.927 -> 0.397 s** for the 518-row mixed prefill.
The same cache metadata and IR hashes finish writing **1.247 ms / 1.061 ms**
before the gaps end. Both arms therefore identify the same two specializations.
The final archive independently verifies all four worker traces. This supplemental
causal attribution covers the two closed head traces, with both-rank host
metadata/cache snapshots; both result files and head traces now bind by SHA256
to the verified archive. Cache audits remain separate supplemental artifacts.
The stock gaps contain no recorded kernel, memcpy or memset activity; that is a
trace-local statement, not instrumentation of every external GPU process.

The stock six-plus-two replay reproduces the latency anomaly. On its first
1536-row prefill, head host elapsed/CPU time is **2.463/2.223 s**; on repeat it
is **0.458/0.458 s**. The 518-row mixed prefill is **1.920/1.810 s** first,
**0.392/0.392 s** repeated. Both workers show the same host pattern. Sampled
SM clocks are 2411-2515 MHz with throttle reason zero throughout these groups;
maximum sample gaps are about 139 ms. This is sampled evidence, not continuous
proof against every transient throttle event.

The closed stock head trace covers all **52 recorded/profiled forwards**.
Its first MoE operation has owned-kernel gaps of **1.989426 s / 1.539061 s**,
immediately after native expert-token alignment and before `_canonicalize`.
The enclosing `vllm::moe_forward_shared` host operation consumes almost the
entire gap. GPU kernel union time is 0.942 s first / 0.903 s repeated for the
1536-row step, and 0.529 / 0.534 s for the 518-row step.

**The implicated kernel is our existing deterministic expert-order wrapper,
enabled in both reference and compressed arms.** It is not evidence about an
unmodified upstream stock runtime. Cache metadata/IR retained from both ranks
shows five variants each. On the head, the new `(BLOCK=48, CAPACITY=2048)` and
`(BLOCK=16, CAPACITY=1024)` entries finish writing **1.080 ms / 0.982 ms before
the respective GPU gap ends**. This strongly identifies first-use Triton
compilation as the cause of these two profiled stalls. Candidate corroboration
and final independent archive verification are complete. The original C8 slow
trial had no simultaneous trace, so its attribution remains an inference.

A local startup fix is CPU-prepared in `stable_dispatch_warmup.py`: enumerate
all **34** reachable block/capacity specializations across the existing
2..2048-token, blocks 8/16/32/48/64 contract, including block-48 padding to
2064 slots / capacity 4096. Run unchanged kernels on aligned, guarded scratch
metadata, verify exact integer ordering and guards, and repeat each case.
At the 19:33 checkpoint no GPU warmup had run; the bounded component now
passes on both GPUs as recorded above. The latency replay's frozen source tree
is unchanged. Next: measure the loaded-model startup/memory cost and ordinary
serving latency. Keep sorting enabled and preserve all original gates/outliers.
Model CPU-memory reclamation and the fixed-mantissa GPU prototype remain
unmeasured/deferred.

Local evidence: /private/tmp/mxwave-latency-probe-v1/live/.
Stock closed head trace SHA256:
81d5541edc92fe38f06ebf60652c226b6e6c3b8bdb51e06575b83f35c5614b13.
Stock completed result SHA256:
9a9d364adbba905410c01d07af724d9648c828b1f731d37e1e910a0950c517e2.
Both-rank cache audit SHA256:
a1c53c36a4feb4105331a77b4d3dd9dd8c101fd24c65c598c99d0360ac5dfecc.
Retained repository diagnostics: latency-probe-v1-stock-first-use-review.json,
latency-probe-v1-stock-cache-audit.json, the host/profile analysis sources and
stable-dispatch-warmup-v1-local-checks.json. These supplemental artifacts
remain separate from the completed final evidence archive.
The both-arm supplement and candidate cache audit are also retained:
latency-probe-v1-both-arms-first-use-review.json (SHA256
b02c49bff0e8b395f5abdb3c50fa032ebf5a083b0a7843769c31fee664c21758)
and latency-probe-v1-candidate-cache-audit.json (SHA256
b5c8afee5b012b6def10f55fe8ca2b78b4953665a0d4266fe4763a7cd6c21107).
Candidate closed head trace SHA256:
e965edfa428827331fb9252febc6d657d4ebcf0d0ec680e27b5bcaa61a25d5db.
Candidate completed result SHA256:
d2a8a95f9e8d3779f8ba3cf7e9e5c62bbde09d93baa30260a05a2307591dbb43.

Final archive: **40,975,208 bytes**, SHA256
c0b578765ca488e62efd6d31f55973c9ce928f97666e9c37bea787dcd1ca404c.
Local archive/extraction/reviews: /private/tmp/mxwave-latency-probe-v1/.
Repository diagnostics now retain latency-probe-v1-review.json,
latency-probe-v1-cross-host-verification.json and
latency-probe-v1-final-evidence-binding.json. All profiled forwards are covered;
172 M8 decode ranges pass the retained mechanism checks. The frozen diagnostic
review's pending-analysis field is historical; detailed first-use attribution is
the separate head-rank supplement above, not an attribution of every worker trace.

## Latency arrival audit preceding the focused replay

The following records the hypotheses before latencyprobev1. The completed replay
above now identifies first-use compilation in both arms; the exact original
timed outlier still lacks a simultaneous GPU trace.

The user accepts a modest speed/latency cost for compression if it is an
understood tradeoff, and asks that abnormal behavior be investigated. Retain
the original failed speed gates as historical results; do not reinterpret the
5% target as a correctness failure or change its numbers. The fixed-mantissa
GPU prototype is deferred while this latency investigation finishes.

The new offline arrival audit covers **all 36 timing groups / 204 requests /
13,056 single-token arrivals**, verifies source/result hashes against the C8
inventory and recomputes the stored timing fields. All 102 paired sequences
match. No trial is removed. Independent raw recalculation checks the additive
request-time decomposition, TTFT medians and stall counts.

| Ordinary case | Median TTFT, stock -> compressed | Median 64-token group time, stock -> compressed | Interpretation |
|---|---:|---:|---|
| C1, 256-token prompt | 387.56 -> 418.21 ms | 4.283 -> 3.811 s | About 31 ms later first token; faster subsequent tokens more than recover it |
| C8, 256-token prompts | 1.164 -> 1.196 s | 8.352 -> 8.157 s | Small median improvement; one large compressed outlier remains unresolved |
| C8, 2,048-token prompts | 6.489 -> 6.674 s | 16.113 -> 16.212 s | Small net slowdown after repeated prompt-processing work |

In every long-prompt trial in both arms, requests ordered by first-token arrival
have exactly **7/5/4/3/2/1/0/0** inter-token gaps over 250 ms. The ordinary late
stream intervals are modestly faster with compression: medians across trials
of per-request last-32 mean intervals are 103.83 -> 100.44 ms. This is descriptive
client timing, not an isolated GPU decode benchmark or a replacement speed gate.
The common early-gap pattern, staggered first tokens and retained prefill traces
support prompt/decode interference as the main explanation. The independently
profiled ~36 ms reconstruction per prefill forward is consistent with the
additional latency; the exact timed requests were not simultaneously profiled,
so it is not a complete causal attribution.

**Investigate the candidate short-C8 repeat 2 separately.** Its group time is
12.052 s, initial admission is six of eight requests, six first tokens arrive
at about 2.989 s and two at 5.232 s. Six streams then contain a ~2.14 s gap.
Late-stream mean intervals return to about 110 ms, close to other trials.
Stock also has split-admission trials, but neither has this large total delay.
Normal reconstruction cost does not explain these seconds of extra latency.
The retained engine logs do not identify the missing time; endpoint telemetry
does not rule out transient throttling. JIT/shape preparation, host waiting and
GPU work remain hypotheses. A focused both-arm 6+2 admission replay should
retain CPU/GPU forward timelines, actual grouping, clock/thermal samples and
first-versus-repeat behavior before choosing a fix. Keep its instrumentation
separate from ordinary performance measurements.

Including this outlier, mean per-request completion at short-C8 increases
8.408 -> 8.787 s despite the better median group time. This tail cost must be
reported alongside the medians. No new numerical degradation is found: the
previously verified finite C1/C8 numerical gates remain passing, not universally
bitwise inference or broad task-quality proof.

Retained diagnostics: `matched-c8-v1-latency-review.json`,
`matched-c8-v1-latency-analysis.py.txt`, `matched-c8-v1-latency-review-check.json`.
Review SHA256: d92219f2e4f478709f0fccb5e2831984c32d1f4b28bd0cad47ad4a8b5cd4f3fa.
Source SHA256: b2eeb508eee3f5132bc079cad1d99d850eed0fc05581fb2e4cce060ee3f8f075.
All new work in this audit is CPU-only; the frozen C16 run is unchanged.


## Completed exact16 full-model continuation: matchedrow16v1

**Complete and independently verified on Spark/Mac at 19:08 UTC.** Launched
18:16:44 UTC; original service is restored with exact identity/config/mount
checks and health 200. Controller and collector report complete. All **1,581
payloads / 200 source bindings / eight worker traces** verify on both hosts;
all decisions agree (maximum CPU float64 difference 1.776e-15).

All **46 numerical comparisons pass** the original limits at matched C1/C8/C16
grouping, including twelve exact self-repeat comparisons. All 192 timed paired
output sequences match. These finite checks do not establish universal bitwise
inference or broad task quality. Native mHC is unchanged.

| C16 case | Throughput, stock -> compressed | Change | Median TTFT, stock -> compressed |
|---|---:|---:|---:|
| 256-token prompts | 81.142 -> 81.529 tokens/s | +0.48% | 1.623 -> 2.211 s (+36.22%) |
| 2,048-token prompts | 37.035 -> 36.516 tokens/s | -1.40% | 11.265 -> 11.601 s (+2.98%) |

All six trials/case/arm are retained, including slow trials in both arms. The
original speed gate still fails. CUDA allocation saving remains **9.367162 GiB
across TP2**. All 54 retained physical-16 decode ranges across both arms/ranks
replay graphs; candidate ranges each have 68 compressed KDA calls and zero
reconstruction calls. The C16 decode fallback is fixed, while short-C16 TTFT
and the earlier short-C8 outlier still need causal analysis. Do not attribute
the whole first-token penalty to unavoidable compression cost.

Archive: 443,592,022 bytes; SHA256
15feabeb92479432ba548f85137da090b6a7ab5c6f25dacf8f3472559ac4c0d2.
Local retained archive/extraction/reviews: /private/tmp/mxwave-matched-row16-v1/.
Repository diagnostics: matched-row16-v1-review.json and
matched-row16-v1-cross-host-verification.json.

The focused **latencyprobev1 launched at 19:11:49 UTC**. The idle guard cleared,
original GLM was paused at about 19:12:25, and both arms completed all 15 groups.
Restoration completed at **19:33:51 UTC**, ahead of the 20:42:25 deadline;
the post-restoration identity/config/mount check passed with health 200 at
19:33:58. Controller 2712051, collector 2712050 and watchdog 2715977 are terminal.
The Mac verification is complete and retained above.
Both actual CPU CLI checks,
200 source bindings across both nodes, exact original identities/health and
native NVML clock/temperature/pstate preflights pass. Six new helpers plus 194
unchanged C8 parent sources retain actual admission, CPU elapsed/thread time,
GPU traces and bounded 100 ms clock/thermal telemetry for a 6+2 replay and
ordinary C8 controls. Full Ruff, core mypy and pytest pass; CPU protocol rehearsal
and attribution recheck against ten retained GPU ranges pass. This diagnostic
has 30 groups, 106 paired requests, two profile sessions/four worker traces;
it is not a replacement serving-speed or quality gate. It retains the C8 parent
runtime (no exact16 route) to reproduce that outlier. The controller requires
an acknowledged independent watchdog before any pause and bounds the outage
to 90 minutes. No model arithmetic or GPU clock settings are changed.

Prepared SHA256: cf1ec675fb3f6d564539a187dd63a2f33bac49786262b8c1915143a765eec6f3.
Plan SHA256: 42645dd025cfc4bc128085bdcccbdf053c3aab7f0a7651ad0f3f019d5ff052af.
Source archive SHA256: dfe4194d1c7d13eb4f8cb89c72877719d81e11877fa418ae2710f6c839ee6d4a.
Collector SHA256: a8d192eb3073dadf72e02ee6eb07dddda91a4b3f7246e25107051177be052eb3.
Verifier SHA256: b67178259184c339a9e6445a5e15b6c6f12da7caf7fe14e0a8db9b776b54bf7c.
Local preparation and completed Mac retention: /private/tmp/mxwave-latency-probe-v1/.
retain_and_review.py has completed; do not rerun its exclusive archive creation.
Source/plan/CPU checks are
retained in diagnostics-v1/latency-probe-v1-*. Fixed-mantissa GPU work stays deferred.

**C16 arrival audit is complete:** all 24 trials / 384 requests / 24,576 actual
single-token arrivals are retained and independently recalculated. Initial
short-case admissions are stock [5,8,8,8,3,8] and compressed [7,5,4,8,3,4].
Both five-request starts have unusually late first tokens (~2.79/~2.85 s).
This distribution can affect the median across 16 requests; it does not prove
that admission explains the entire 36.22% TTFT difference. Stock short repeat 0
takes 16.299 s and compressed repeat 1 takes 14.442 s; neither is removed.
Median group completion is 12.620 -> 12.560 s, while mean request completion
(including both outliers) is 13.092 -> 12.708 s. These alternate summaries do
not replace the original gate. Last-32 mean intervals, summarized by trial
medians, are short 157.85 -> 152.74 ms and long 136.85 -> 132.21 ms.
Every long trial in both arms has the same TTFT-ordered >250 ms gap counts:
[16,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0]. These client-side gaps are not pure
GPU time. The shared pattern supports continuing the focused admission replay.
Retained diagnostics: matched-row16-v1-latency-{analysis.py.txt,review.json,review-check.json}.

The offline attribution of all eight C16 profiles / 112 forwards also matches
the independent verifier's kernel counts, graph launches and duration sums.
The controlled full-M16 input fused calls sum to 16.215/18.618 ms on ranks 0/1;
outputs sum to 7.244/8.271 ms. Mixed full-M16 candidate KDA medians are
21.935/22.452 ms, with expert GEMM around 97.5 ms. Stock mixed generic BF16
linear sums are 32.804/33.081 ms, but include other dense projections, so they
are not a direct KDA comparison. Mixed grouping differs between arms; these
short profiled windows are not the timed latency outliers or API benchmarks.
Retained diagnostics: matched-row16-v1-profile-{analysis.py.txt,review.json,review-check.json}.

The isolated pair is **matchedrow16v1**. Both real CPU CLI preflights pass,
original health is 200, and all **200 source bindings** match across ranks:
all 194 parent sources unchanged plus six independently authored helpers.
Full Ruff, strict mypy (35 core files) and pytest pass (one skip).
The initial direct pytest executable selected the main checkout's editable
package and failed collection; its original log/receipt are retained. The
correct worktree Python module invocation passes without source changes.

The frozen protocol has 20 quality groups / **46 numerical comparisons** at
C1/C8/C16, with native mHC and stable compaction in both arms and actual matched
grouping required. Original KL <=1e-5 / absolute mean NLL <=1e-4 gates remain.
Only ordinary C16 speed is remeasured: short/long prompts, six trials/case/arm,
24 groups / 384 requests / 24,576 forced output tokens. Both speed cases require
5% throughput gain and TTFT <=1.05 stock. Existing C1 TTFT failure is not solved
or hidden by this narrower intervention. C1/C8 ordinary speed is not remeasured.

Candidate adds only exact16 input/output routing on the component-qualified
native geometry; global fused limit stays eight, resident BF16 cache stays zero,
native kernels are unchanged. C16 uniform and mixed-length profiles must prove
graph replay and no KDA reconstruction on physical-16 decode. Eight worker
traces were retained. Full-model results are reported above.

CPU protocol rehearsal exercises both eight/sixteen-request collectors, all
46 comparisons and 24 timing groups; wrong grouping, numerical drift and equal
speed are rejected. A synthetic verifier check rejects reconstruction and a
missing fused projection. Neither CPU check makes a GPU/model claim.
Native tile metadata was inspected: the unchanged TilingConfig header hash is
181fcb52c2e5c496b00f84eecec57ef0e322406ae6dcd92f825b97c7766461f6.
The sixteen-row dispatcher retains the same output-row tile geometry used by
the verifier; model runtime/library bytes remain pinned to the qualified build.
Controller launch rechecks every staged control asset and source binding.
The Mac retainer at /private/tmp/mxwave-matched-row16-v1/retain_and_review.py
has completed successfully. Do not rerun its exclusive archive creation; use
the retained verification proof and reviews.

Before preparation, 2,193 payloads in two regenerable C8 extraction directories
were checksum-verified against canonical archives retained on both Spark nodes
and independently verified on Mac. Only those duplicate extraction files were
removed; parent run JSON/protocol/source files and checkpoints remain.
Head free space rises 7,495,639,040 -> 8,634,753,024 bytes.
Full manifest: ROOT/retained-c8-extraction-relocation-v1.json.

CPU-preparation PID 2110144; receipt ROOT/matched-row16-v1-preparation.json passes.
Active controller: ROOT/quality-model-controller-matchedrow16v1.json.
Collector receipt: ROOT/matched-row16-v1-evidence-check.json.
Results: ROOT/run-matchedrow16v1/results/.
Expected archive: ROOT/evidence-matched-row16-v1.tar.gz.
Collector SHA256: f9ca04125febb6076ff8bf9c0e56287b5fac0f4545292b70ab7601dd52337923.
Verifier SHA256: 0ff2f2f5fd169c116e2638354616b26ab000117d8d051b76181035fd65f39a25.
Launcher SHA256: e0a73722e3cdc1d2d01f9c84fc7e80fc474cb1498446451782455f301003c780.
Numerical gates pass; speed gates fail, as reported above.
Prepared SHA256: 470730831d2b9321bc1ab4559610c6756ada35ab063d87b5693a98ce28f20dab.
Plan SHA256: dd6c56953e95ba4ded903b0094ac7a720d9f84cbe10e0a4574c27fc980791844.
Source archive SHA256: 94bba5a1653412fc302d8d53389b2fc0f614bcee957bd90d513c31baf9bc5b3c.
Input component archive SHA256:
76b00c670bf2747448afe35ca9e7f011747b5c3d17ac0b6fca9caa930936b34c.
Local preparation: /private/tmp/mxwave-matched-row16-v1/.
Owned bounded Mac awake lease PID 60808, maximum three hours.
Independent idle guard, acknowledged restoration watchdog and exact original
identity/config/mount/health restoration are required before any service pause.
No configuration is promoted; MTP remains deferred.

## CPU-only follow-up prepared while C16 runs

**18:44 UTC:** no additional GPU experiment or model policy change occurred.
The previous goal turn made progress through the C8 launch/storage analyses;
this turn adds an executed CPU allocator probe and an exact representation oracle.

The retained C8 process counters attribute 229,728,256 of the 260,485,120-byte
initial RSS increase to anonymous memory. This is not proof those pages are
free or reclaimable. The vocabulary loader explicitly stages two dense CPU
shards before replacing them; their 1,268,776,960 bytes/rank are also relevant
to startup peak RSS. Both vocabulary and KDA reports show zero surviving dense
parameter references. Summed per-process high-water marks are not a simultaneous
host peak, and these counters must not be added to CUDA allocation.

A separate bounded host process runs only after stock measurement/profile
completion, with controller phase `starting_candidate` and no candidate
measurement JSON (18:40:04 UTC). Under a 384-MiB address-space bound, it allocates
64 MiB of payload plus 256 KiB of live guards. glibc 2.39 `malloc_trim(0)` releases
61,435,904 bytes (58.59 MiB) in 4.32 ms, and every live guard byte remains exact.
The source hash and expected guard digest independently check on Mac.
**This does not demonstrate reclaimable model memory.** Any actual startup-only
cleanup must be applied to both arms before warmup and freshly measured.
Interface reference: [malloc_trim](https://man7.org/linux/man-pages/man3/malloc_trim.3.html).
Remote receipt: ROOT/cpu-memory-v1/probe-head.json; source in the same directory.
Local source/receipt: /private/tmp/mxwave-matched-c8-v1/cpu_allocator_probe.py
and cpu-allocator-probe-head.json.

The new independently authored **fixed-mantissa CPU oracle** gives every weight
a fixed sign/mantissa byte, keeps three exponent bitplanes and stores only an
exponent byte for each exception. It targets the variable common-stream address
in the current decoder; that dependency is not yet proven to limit performance.
All 65,536 BF16 bit patterns pass at bases 0/64/120/248, including signed zeros,
subnormals, infinities and NaN payloads. Seventy-seven tile cases also cover mixed
exception counts and an independently checked coordinate order; five malformed
layouts are rejected. For all 4,097 possible exception counts in a 4,096-word
tile, payload bytes are equal to the current format or 16 bytes smaller with
the same bitplanes/alignment; this comparison excludes metadata.

Actual checkpoint conversion, a GPU decoder/GEMM, counters, timing and model
integration **do not exist yet** for this layout. This branch is deferred while
latency anomalies are investigated. If resumed, prepare source-bound conversion
and an isolated GPU component preserving original K accumulation.
Require full reconstruction/output checks, NCU attribution, interleaved timing
under both cache conditions, representative layers/both ranks and independent
confirmation before full-model integration. Keep the current C16 run unchanged.
Oracle source: /private/tmp/mxwave-matched-c8-v1/fixed_mantissa_oracle.py,
SHA256 ff04bd5ff5c9886c9f02f45e4691dd26f80bd41d9242fda12b2d241999d37696.
Repository diagnostics retain `fixed-mantissa-v1-cpu-oracle-review.json`, its
executed `.py.txt` source, `cpu-memory-v1-probe-head.json`, the probe source and
`matched-c8-v1-cpu-memory-breakdown.json`. These are CPU evidence, not new
full-model numerical or performance qualification.

## Completed fresh native-mHC pair: matchedc8v1

**Complete and independently verified on Spark/Mac.** All 14 quality groups,
33 declared numerical comparisons, 36 ordinary API timing groups and eight worker
traces finish. Exact original container identities/configs/mounts and health 200
were restored at completion before the next continuation. No configuration is promoted.

**C1/C8 compression passes the original finite numerical gate.** All eight
self-repeat comparisons have KL/NLL zero; every cross/profile comparison passes
KL <=1e-5 and absolute mean NLL <=1e-4. Actual forward shapes, row/prompt order
and chunk boundaries match in all 33 comparisons. Across all nine ordinary
cross-arm pairs at each concurrency, C1 mean decode KL is 5.803778255448176e-6
(NLL -3.6509906652337038e-6), C8 KL is 1.194444923172159e-6
(NLL 2.5822470640270547e-6). All 102 timed paired output sequences also match
(6,528 generated timing tokens per arm). Exact stored weights do not imply
bitwise-identical inference: the first C1 pair has 0/16 equal prefill and 0/48
equal decode vocabulary rows; C8 has 16/16 and 46/48.

**The overall speed gate still fails.** Six retained trials per case/arm,
64 output tokens/request, ordinary admission at 3 ms:

| Workload | Stock tok/s | Compressed tok/s | Throughput change | TTFT change |
|---|---:|---:|---:|---:|
| C1 / 256 prompt tokens | 14.943 | 16.794 | +12.38% | +7.91% |
| C8 / 256 | 61.302 | 62.771 | +2.40% | +2.75% |
| C8 / 2,048 | 31.775 | 31.581 | -0.61% | +2.85% |

C8 misses the required 5% throughput gain; C1 TTFT misses the 5% ceiling.
CUDA allocation before profiles falls 175.907657 -> 166.540494 GiB across TP2:
**9.367162 GiB saved**. Source-before reserved CUDA memory falls
176.779297 -> 167.386719 GiB. Host process RSS is not improved: summed initial
RSS is 12.652 -> 12.894 GiB (final 13.816 -> 14.021 GiB). Spark process/container
and CUDA counters overlap; do not add them.

One retained compressed short-C8 trial reaches only 42.484 tok/s and 2.989 s
median TTFT. Admission logs show six of eight prompts collected initially,
rather than the full cohort. Endpoint clocks remain 2,483-2,496 MHz and
temperatures 68-71 C for that trial; its actual forwards were not profiled.
Split admission is an observed lead, not a proven complete explanation.
No trial is dropped. Across timing groups, endpoint clocks are 2,463-2,509 MHz,
temperatures 61-77 C; clocks are not locked and sampling cannot exclude transients.
Both fresh arms run stock then compressed with an explicit time gap.

Native mHC and stable sparse compaction are used in both arms. The consistent
mHC experimental policy is absent. Controlled grouping applies only to quality;
ordinary timing excludes pause/resume, logprob capture and profiling. All 194
source bindings agree across ranks; all 190 parent sources are unchanged.
Both real CPU CLI checks, full Ruff, strict mypy (35 core files) and pytest pass
(one GPU skip). The CPU simulation independently exercises/rejects the declared
protocol; it is not a model result. All profiled forwards are covered; aligned
profiles record six graph launches/rank and mixed-length profiles fifteen.

The archive has **1,168 payloads / 194 source bindings / eight worker traces**,
316,083,034 bytes, SHA256
eb244b63a6aeb58c337f89820de93cead3210ee0d7121a49677fcbb88ac8341b.
Spark/Mac decisions agree; maximum CPU float64 difference is
1.7759773686398805e-15, within independent-recomputation tolerances
1e-12 absolute / 1e-10 relative. Model gates are unchanged.
Every historical failure is retained. This is finite C1/C8 qualification,
not universal zero-inference-error, C16 qualification or release promotion.

**Fresh C8 trace analysis is complete:** four controlled M8 decode ranges/rank
have no weight reconstruction; compressed KDA kernel sums are 21.91-23.88 ms,
routed expert GEMMs 55.99-56.02 ms. Generic remaining BF16 linears add 8.98-9.60 ms;
stock's combined BF16 linear family is about 32.35-32.39 ms. These operator
families do not isolate every stock KDA tensor. Controlled prefill at 2,048 or
23 rows reconstructs 68 matrices/rank in 35.82-36.90 ms per forward.
Kernel sums may overlap and are not API latency, measured bandwidth or a TP
critical path. Fresh profiles are retained separately from the frozen archive.
The supplemental v1 name filter missed native Decompress kernels; the append-only
v2 fixes that field. V1 source/result remain retained; no measurement/gate changes.

**18:35 UTC offline follow-up:** graph-node order and chronological launch order
agree for all selected linears in four controlled C8 decode forwards/rank.
All 240 unchanged linear signatures align and all 68 replacements occupy the
expected input/output slots. Stock input/output median kernel sums are
16.25-16.32 / 7.05-7.16 ms; compressed sums are 15.24-16.57 / 6.67-7.31 ms,
depending on rank. Rank 1 saves about 1.44 ms in the two paired projection
medians; rank 0 is about 0.34 ms slower. This checked ordering supports
projection attribution, but no tensor-pointer annotations were recorded.
These short profiled sums do not replace ordinary API timings or establish
a TP critical path. The initial analyzer incorrectly assumed logical-stream
graph nodes cannot overlap; one 2.720-us overlap is now retained explicitly,
with the original source/failure log preserved. No original measurement changed.

Exact payload accounting also bounds a possible dense-storage tradeoff:
replacing all output packed matrices with BF16 would leave an estimated
8.744 GiB CUDA saving; also retaining the first five input projections would
leave 8.467 GiB. Replacing all KDA weights, and separately removing the unused
pool, would leave 7.246 GiB. These replace packed storage rather than keeping
two copies, but are arithmetic estimates, not implemented or measured policies.
They predict neither CPU RSS nor a speed gain; all-dense KDA would lose the
existing C1 fused benefit. The source audit also rules out merely changing
the native row-tile template: codec strides encode four warp groups and
only full 64-row tiles have aligned payload padding.

New retained artifacts are `matched-c8-v1-linear-alignment-review.json`,
`matched-c8-v1-storage-tradeoff-review.json`, and
`matched-c8-v1-native-layout-audit.json` in the repository diagnostics directory,
with executed analysis sources and the initial failure. They are Mac CPU
analyses of retained/hash-bound evidence, not new GPU runs or cross-host
qualification of another candidate. Native source inspection remains private
at `/private/tmp/mxwave-matched-c8-v1/native-source-inspection.json`.

**Continuation:** the separately frozen exact16 full-model comparison is now
running, as recorded above. It uses the qualified input component with matched
native-mHC quality and profiles required to prove reconstruction removal.
Global fused limit stays eight; no resident BF16 cache is used.
MTP remains deferred.

Remote ROOT:
 /home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/full-model-v1
Controller: ROOT/quality-model-controller-matchedc8v1.json (complete).
Results: ROOT/run-matchedc8v1/results/.
Archive: ROOT/evidence-matched-c8-v1.tar.gz.
Collector receipt: ROOT/matched-c8-v1-evidence-check.json (complete).
Local full evidence: /private/tmp/mxwave-matched-c8-v1/evidence/.
Local proof: /private/tmp/mxwave-matched-c8-v1/cross-host-verification.json.
Repository compact review/proof/supplement:
 benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/matched-c8-v1-*.json.

Source archive SHA256:
c4ae8c4e070d2b91b60338579d66a2ee3f474c86b60d68567f478570d36c951c
Prepared SHA256:
b031f3619810e001bce7052426233bc6269eeed643cf3c4d70864400c7213033
Plan SHA256:
65c9b8c5c02718433036b72fdc7226e14c10a67a96c0e689d9ce4d439d36c7ae
Collector SHA256:
a57a20b580b6652c92d5dd0d68eb1a139315808de756eccb550bfcc0a039bb94
Verifier SHA256:
394dab3d50a7b6e27360da5bc08a9294532b22dbe43c3c076477a8012e50b3da

## Historical decision preceding matchedc8v1: native mHC and matched grouping

The user reviewed the tradeoff and the next comparison now uses **native mHC in
both arms**, together with the already-tested stable sparse compaction policy.
The completed c8tracev2 evidence already proves exact native-mHC C8 stock repeats
when actual grouping matches. Full serving batch invariance is not a prerequisite
for the compression gate. The consistent mHC policy remains an optional,
component-validated experiment; its full-model test is deferred.

**No GPU experiment is active.** Exact original service identities/configs/mounts
and health 200 are restored. No configuration is promoted. MTP remains deferred.

**Then-prospective pair (now completed above):** collect stock and compressed self-repeats at
C1/C8; control C8 grouping only for qualification; independently require matching
actual shapes, row/prompt order and chunk boundaries in comparisons. Retain the
original KL <=1e-5 / absolute mean NLL <=1e-4 gates, all generated IDs and complete
conditioning. Measure ordinary-admission API throughput/TTFT, CPU/GPU memory and
clock/thermal telemetry separately from controlled quality and profiling.
Exact restored weight bytes do not imply bitwise identity of a fused GEMM.
If both arms repeat but cross-arm KL fails, investigate accumulation/rounding
in the compression path. Do not attribute that failure to stock repeat noise.

The corrected mhcbatchv2 component is complete and independently verified on
Spark/Mac: **523 payloads**, all 48 consistent batch-size comparisons and all
144 same-shape repeats pass. All eight identical-native controls pass the
unchanged 2% limit under actual balanced ABBA/BAAB orders. The main output bytes
are retained; repeat raw bytes remain source-bound assertions only.
Archive evidence-mhc-batch-v2-v2.tar.gz, 10,237,118 bytes, SHA256
3019898e9aa9155ae74478349161ed42b6fd803c69e4a5738c07e0a364bc1cac.
Spark/Mac decisions match, maximum CPU float64 difference zero.
The component cost at 258 rows is +0.89/+13.16 microseconds for pre/fused;
at 2048 rows +134.93/+138.67 microseconds. These are component costs, not a
whole-forward slowdown. Both mHC and sparse fixtures include 16-row cases;
full-model C16 qualification is still missing.

The first corrected-fixture archive retained a live controller.log whose bytes
changed after inventory creation. Mac caught that mismatch. The first archive
and inventory remain intact; append-only v2 snapshots after the controller exits
and verifies all final archive members after packing. No GPU rerun, measurement
change, binding replacement or gate relaxation was used. The original v1
restoration-status verifier repair and ABBA-order error also remain explicit.

**C8 speed analysis:** all eight retained short-prompt stock/compressed worker
traces were checksum-verified and analysed by launch correlation on both ranks.
For the compressed 3 ms-window C8 profile's three M8 decode forwards/rank,
median expert-GEMM duration sums are 59.154–59.158 ms, compressed KDA sums
21.755–21.882 ms, other BF16 linears 8.917–8.969 ms. First-to-last owned-kernel
spans are about 106.116 ms. C8 has no pure-decode reconstruction; prefill retains
68–136 reconstructions/rank (about 36 ms for 68). Kernel sums may overlap and are
not request latency or a measured TP critical path. The control load has a time
gap, profiles cover only short prompts, and these observations do not establish
a new speed gain. Target fused C8 KDA efficiency and prefill reconstruction;
the exact16 input fix alone does not solve C8 throughput.

Local proofs: /private/tmp/mxwave-mhc-batch-v2/cross-host-verification-v2.json
and /private/tmp/mxwave-c8-speed-breakdown-v1/review.json.

## Previous adapter GPU follow-up (now complete)

The source-pinned runtime adapter passes four CPU contract tests and an actual
native-class AST/signature/default check without CUDA initialization. The native
kernel module itself cannot import without CUDA because its optional TileLang
binding is absent; that CPU-only failure is retained, and did not pause GLM.

The isolated mhcbatchv2 GPU fixture is launched (controller 1390944, plan SHA256
1470d18c44e75b528d73c6bf820d65c51c2d3a34de10e59a0d287b443cada763).
It exercises the patched native layer forward_cuda and repeats the 144-case /
24-group component protocol, with an idle guard, bounded timeout and restoration
watchdog. Full-model integration has not started.

Source review caught a timing-order bug in the retained v1 fixture: reversing
ABBA produces ABBA again. Its two parity subsets were not distinct ABBA/BAAB
orders. This does not affect retained bitwise correctness comparisons; the
reported v1 component timings are provisional fixed-order observations.
V2 explicitly alternates ABBA/BAAB and independently verifies each trial order.
No old payload, binding or decision is overwritten.

The user requested a direct explanation and improvement of C8 throughput.
Historical C8 is +2.1–2.5% on short prompts and -0.3–-0.7% on long prompts;
C8 already uses fused decode. Both-rank traces retain 68–136 reconstructions per
rank in prefill and zero in pure decode. CPU analysis of the retained C8 device
traces is running to measure the remaining stage/kernel shares before selecting
a speed intervention. Avoid describing the C16 reconstruction fix as solving C8.
MTP remains deferred and no configuration is promoted.

## Latest checkpoint: completed mHC batch-path fixture

**The mHC component experiment is complete and independently verified on Spark/Mac.**
No GPU experiment is active. Both exact original GLM containers/configs/mounts were
restored with health 200. No full-model test of the mHC policy has started.

Following the C8 device traces, the fixture holds four per-request operands fixed
while varying native mHC batches over 1, 8, 16, 23, 258, 516, 522, 774, 1293, 1548,
1550 and 2048 rows, using actual layer 0 and layer 3 mHC weights.

- Native batch-sensitive path: **38/48 cases differ from their own C1 reference**.
- Fixed prenorm splits alone: **22/48 still differ**.
- Fixed splits plus one consistent native post-then-pre composition:
  **all 48 cases bitwise equal their own C1 reference**.
- All same-shape GPU repeat assertions pass. Main comparison output bytes are
  retained; repeat raw bytes are not retained, and repeat assertions are source-bound.
- All eight identical-native timing controls pass the unchanged 2% limit.
  C8 component timing is roughly flat. At 258 rows the consistent pre/fused paths
  cost +1.51/+11.04 microseconds (+5.15%/+14.09%); at 2048 rows they cost
  +136.16/+149.26 microseconds (+18.10%/+10.29%). These are isolated component
  measurements, not API latency or throughput.

The fixture runs 144 cases and 24 timing groups without loading the full model;
peak allocated CUDA memory is 466,215,424 bytes. All **518 payloads** independently
verify on Mac, with identical Spark/Mac decisions and maximum CPU float64
difference zero. Archive: evidence-mhc-batch-v1-v2.tar.gz, 10,229,773 bytes,
SHA256 1ac95dff73978a105fe4ea8677d26c6437d92ec55bae54ef98b193db0e08da75.
Local retained evidence: /private/tmp/mxwave-mhc-batch-v1/evidence-v2/.
Local proof: /private/tmp/mxwave-mhc-batch-v1/cross-host-verification-v2.json.

The original CPU verifier incorrectly expected restoration status "restored";
the restoration helper actually reports "complete". The original failed verifier
and controller log are preserved. Append-only v2 repairs only this CPU status
check, with no GPU rerun, measurement change, source-binding replacement or
relaxed quality gate.

**Important arithmetic distinction:** this is a consistent rounding policy, not
a bitwise-preserving change relative to the original fused C1 implementation.
The retained layer-input C1 tensors differ in 333/16,384 elements at layer 0 and
339/16,384 at layer 3 (relative L2 0.000816/0.000894). Stock and compressed arms
must use the same policy for a compression comparison.

**Next:** source-pin and test the runtime adapter, then run stock full-model C1/C8
self-repeats with stable sparse compaction and consistent mHC. Confirm ordinary
C8 KL <=1e-5, NLL gate, generated IDs, actual batch shapes and graph replay from
profiles. If another source remains, localize it from traces. Only then qualify
a fresh stock/compressed comparison and exact16 integration. Neither the mHC
full-model fix nor a combined faster/zero-loss candidate is yet established.
MTP remains deferred. No configuration is promoted.

## Completed C8 trace investigation, 18 October 15:43 UTC

**c8tracev2 is complete and independently verified on Spark/Mac.**
All 12 quality groups, eight strict self-comparisons and six profiler sessions
finish; 12 worker traces cover the recorded forwards. Exact original
identities/configs/mounts and health 200 are restored. No GPU experiment is active.

**Controlled C8 repeats are bitwise exact**, both unprofiled and profiled:
all 16 prefill / 48 decode full-vocabulary rows match in all four comparisons;
KL = 0 and NLL difference = 0. Actual forward shapes and prompt order match.
Ordinary unprofiled repeats fail KL at 3.21805e−5 / 5.36556e−5; profiled repeats
fail at 4.59890e−5 / 7.00604e−5. All IDs/conditioning match and NLL passes.
Ordinary schedules and row membership differ. No gate is relaxed.

The complete archive has **1,025 payloads / 190 source bindings / 12 traces**,
288,213,015 bytes; SHA256
37f514033e9b39b32b14b24aae6a74aa37270cc05c36a8379b784d3f9b47672a.
Spark/Mac decisions match, with maximum CPU float64 difference 2.2175e−15.
CPU diagnostic tolerances remain separate from the model's KL ≤1e−5 /
absolute mean NLL ≤1e−4 gates. Every prior failure is retained.
Local evidence: /private/tmp/mxwave-c8-trace-v2/evidence/.
Remote terminal receipt: c8-trace-v2-evidence-check.json.

This establishes repeatability under matched C8 grouping. It does not fix
arbitrary continuous-serving batch invariance or qualify compression.
The remaining issue is strongly associated with batch layout. Most drift
(90–98% in the two ordinary unprofiled comparisons) comes from prompts with
unchanged own chunk boundaries; chunking alone is insufficient.

Device traces confirm different GEMM variants/grids and mHC paths across
batch layouts. Native mHC changes prenorm split counts, and fused post/pre
switches from small-row FMA at <=16 rows to DeepGEMM on larger/mixed steps.
Those are candidates, not a proven first numerical divergence. The next
targeted check should hold per-request mHC operands constant while varying the
observed batch shapes, quantify all output differences and timing, and test a
consistent native reduction path in an isolated fixture before model integration.

The fresh C1 stock/compressed comparison remains prepared locally and unlaunched.
The exact16 input/output integration remains unmeasured in a model. MTP remains
deferred and no configuration is promoted.

## Retained first trace-collection failure

c8tracev1 failed at 14:07:40 UTC before its first model forward/profile.
The observer called int() on the native eager descriptor's valid
max_query_len=None. This was an instrumentation failure, not a numerical-model
result. Both worker logs preserve the exact traceback; no quality group or
profile completed. Source/archive/plan and failed receipts remain immutable.

The original service was exactly restored with health 200. All **212 payloads
and 190 bindings independently verify on Spark/Mac**; measurement_complete=false,
profiles=0 and quality_qualified=false. Evidence archive:
cb3466659af83151dadb81d02089d2321096c03670a0c9b2eadc02efae065c5d
(806,541 bytes). Local retained evidence:
 /private/tmp/mxwave-c8-trace-v1/failure-evidence/.
The corrected v2 preserves optional values and adds the real native-API
regression; it changes no model arithmetic or protocol gate.

## Completed parent experiment state

**The broader stock determinism check `sparsequalv2` is complete.** All six
quality groups, eight warmups, 24 unprofiled timing groups and two profile
sessions per rank finished. Controller 4171618 is terminal. Both original
container identities/configs/mounts and health 200 were restored by 13:38 UTC.
Both that parent and the corrected C8 trace child are terminal; MTP is deferred.

**Independent Spark/Mac verification passes for 641 payloads, 186 source
bindings and four worker traces.** The maximum CPU float64 difference is
1.5542e−15, within the diagnostic 1e−12 absolute / 1e−10 relative tolerances.
These CPU tolerances do not change the model's KL/NLL gates.
Archive `evidence-sparse-quality-v2-v3.tar.gz`: 146,233,023 bytes;
SHA256 `706347d0e2d098f10388ff152d35b3a7547dda59229e36a4b8586a4a81c6ff69`.
Local evidence: `/private/tmp/mxwave-sparse-quality-v2/evidence/`.
The original collector, its waiting receipt, the failed v2 CPU verifier and
both append-only analysis repairs remain retained.

**C1 stock self-repeat passes exactly.** Three runs on 16 contexts produce
bitwise equal full-vocabulary rows: all 16 prefill and 48 decode rows match
in each repeat. Both mean KL and mean NLL difference are zero.
This is stock self-repeat, not a stock-versus-compressed quality result.

**C8 stock self-repeat still fails KL ≤ 1e−5.** Its two repeats have mean KL
6.802436007214667e−5 and 5.331761526844531e−5. Mean NLL differences
6.7429699057153e−5 and 6.521238363011078e−5 pass the absolute 1e−4 gate.
Generated IDs and conditioning match. All 16 contexts already differ at the
first prefill vocabulary output; all 48 decode rows also differ.

Cumulative admission logs derive first idle collections of **3, 1 and 8
requests** across the three C8 runs (each prompt is 258 tokens; eight total
2064, exceeding the 2048-token scheduling budget). This is direct evidence
that admission differed, but it does not record actual forward shapes or
prove which kernel causes the remaining drift. Next C8 diagnostic: record
actual prefill shapes and row membership, repeat a fixed C8 cohort, then
capture the first differing operation. Do not describe the compaction
adapter as fixing full-model C8 repeatability.

Mixed-length profiles verify graph replay: on each rank, C8 has 20 pure-decode
forwards / 20 graph launches (actual requests 4 then 8); C16 has 18 / 18,
all with 16 actual requests. The padding policy did not prevent those captures
from replaying. The run has no fresh native-policy full-model control, so
it does not measure full API padding overhead.

The GPU fixture passes **50 cases / 10,000 conversion graph replays** and
50 native-attention graph replays, including heavy padding and mixed counts.
Isolated conversion adds approximately 2.94–3.18 µs/call at C8/C16; every
identical-native 2% repeat control passes. This is component timing, not
measured full-model/API overhead.

A fresh C1 stock-versus-compressed comparison under the same stable policy
is prepared locally, not frozen or launched. It will also retain descriptive
exact16 input/output integration timings. No C8/C16 numerical qualification
or configuration promotion is claimed.

Remote ROOT:
`/home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/full-model-v1`.
Authoritative terminal receipts:
`quality-model-controller-sparsequalv2.json`,
`sparse-quality-v2-evidence-check-v3.json`,
`sparse-quality-v2-independent-review-v3.json`.
The V8 component controller/collector 4024589/4024588 are also terminal.
Do not overlap GPU controllers or overwrite historical launch/verification receipts.

Source archive:
`a5db5fdafedf429c54ae3a6ff872cb25268a464de43bfaaede0057af6715a789`;
prepared:
`d8c37172e7501c4b73e1a5a14fa8aa6ae25f37a2039817a5df1806893cb0c0f8`;
plan:
`dec5241ac1f9587d3293c60c79dd9c5657e94c4ad07763ff35992dbd914e3bff`.
Final CPU verifier v3:
`de0ad758264e6d9ecf9e28af962fd7565f72b6737de0ad5b38479f8f82ae11f6`.

## Completed input-only exact16 rerun: V8

**All unchanged component gates pass.** One real 12576×4096 input matrix,
exactly 16 activation rows, the existing split-K=1 kernel and unchanged native
libraries. Relative-L2 and maximum absolute error are both zero. Eager/graph,
warm/128MiB cache pressure, combined and both-order medians all pass the 2%
reference-repeat and 5% fused-gain gates. Every leg records clocks/temperature;
clocks were observed, not locked. The earlier V7 control failure stays retained.

Fused 435–487µs versus reconstruction+GEMM1.34–1.38ms per matrix; resident
BF16 GEMM alone 463–514µs. Multiplying the isolated fused timings by 34 layers
gives 14.8–16.6ms/rank, **an extrapolation, not an API decode-step measurement**.
This qualifies the exact16 input component for an isolated integration test;
it does not qualify rows9–15, model quality or a global routing threshold.

Spark/Mac independently recompute identical reviews:42 payloads, 19 source
bindings, 24,576 timed events. Archive SHA256:
`76b00c670bf2747448afe35ca9e7f011747b5c3d17ac0b6fca9caa930936b34c`.
Local retained evidence/verifier:
`/private/tmp/mxwave-prefill-decode-v8/`.
Remote source/results:
`prefill-decode-v8/`; source archive
`89e3417edcc4de61aa5d60878f7197527a6d2db68684955611465a64441b5bc7`;
prepared
`9ba8514a37f069f2057124a59dc4559312abbb85d12108b05168f0a955ccac35`.
The first CPU-preparation metadata count18 was repaired append-only to the
actual19 source files before GPU launch; both receipts are preserved.

## Completed MLA internal diagnostic and remaining uncertainty

`mlainternalv1` completes four balanced tracked/private-workspace probes with
five graph calls each, both ranks: 40 graph invocations. All 1,606 payloads and 162
source bindings independently verify on Spark/Mac; diagnostic float64 differences
are at most7.11e−15. Local evidence:
`/private/tmp/mxwave-resume-checks-20261018/`.
Both original identities and health200 were restored before V8.

The first changed recorded operation is `sparse_convert/output`:
25/32 noninitial replay calls change slot order while all 40 selected slot
multisets (including duplicates), valid counts and planned FP8 byte multisets
remain equal. All planned and valid lengths are 259; no slot-zero overread is
observed in those 40 calls. Layer 3 attention relative-L2 changes by approximately
0.00117–0.00268 in the affected calls. Some later hidden outputs change even
when this layer's output repeats; this does **not** prove compaction is the
only numerical source. Restoring the 136 MiB private workspace does not fix it.

Pinned native helper
`vllm/v1/attention/backends/mla/sparse_utils.py`
SHA`cd95caf494f1d2a946ac87abfb4951f738827fe7749aeefd6c0fd83d1cee072d`.
Width 2176 uses 17 racing 128-column tiles; counted power-of-two rows use a
single row tile. The authored adapter pads 2176→4096 with −1, invokes that native
single-row path, then returns the original 2176 contiguous columns. It changes
no native kernel. Installation covers the canonical helper, all loaded vLLM
aliases and future imports; DCP size greater than one is outside this experiment.
Target counted callers and alternate prefill/decode callers were source-audited.

## Completed broader determinism check

The frozen broader plan and evidence are retained in
`/private/tmp/mxwave-sparse-quality-v2/`. The narrower `mlastablev1` was CPU
prepared only and superseded before any GPU launch. All broader fixture,
ordinary-model, mixed-length timing/profile and independent retention checks
are complete. C1 passes stock self-repeat; C8 fails. A focused fresh C1
comparison must measure both probability agreement and speed before the
historical approximately 12% gain can be combined with a zero-drift claim.
C16 timing can assess the exact16 input/output integration, but cannot qualify
C16 model probabilities. Keep the failed stock C8 floor explicit.

## Append-only count metadata correction

The frozen plan mistakenly declares 90 fixture cases / 18,000 replays.
Its unchanged source executes eight retained contexts × five row sizes plus
ten synthetic cases: **50 cases × two modes × 100 replays = 10,000**.
Every intended context and size ran; this is a count error in plan metadata.
The original plan, verifier, collector and launch hashes remain preserved.
The independent CPU verifiers v2/v3 derive counts from the retained fixture
manifest and source; no GPU source, timing gate, numerical threshold or
measurement is changed. The v2 source filter also omitted the registered
plugin TOML from its 186-file inventory check. Its failure and debug receipt
remain retained. Verifier v3 checks every inventoried source path, including
`engine_plugin/pyproject.toml`; all 186 hashes match on both ranks and Mac.

## Exact16 input integration preparation

An independently authored `input_row16_dispatch.py` wrapper is prepared locally
with two passing route-scope tests. It pins the unchanged output16 parent
runtime SHA `f9ed82f5445c61990aa981058b0dcc03d31b4fce363a156c0be4db7bcd3c3753`.
It preserves the global limit of eight and all unmeasured intermediate rows.
No model integration run has launched. The current worktree runtime differs
from the frozen qualified parent; do not substitute it into the next experiment.
The wrapper is intended only for an isolated child of that frozen parent.

## Local verification

After the adapter and alias-coverage test changes, full repository Ruff passes,
strict mypy passes for all 35 core source files, and the full pytest suite
passes with one GPU-only test skipped. Seven focused sparse-contract tests pass.
After adding the exact16 input wrapper and C8 observer, full Ruff and strict
mypy (35 core source files) pass; the full pytest suite passes with one GPU-only
test skipped. Three observer tests cover call passthrough, retained failures
and preservation of real prompt row order in normalized signatures. Independent
trace-correlation attribution also passes a synthetic CPU check. These checks
do not substitute for GPU/model gates.

## Storage and restoration

Storage recovery v2 checksum-verifies 665 inactive per-run CUDA JIT files on
the worker before removing only head duplicates: 7,957,695,179 bytes. Manifest
`027b897877fc7a17e2bb50dfc3caae70afe3d10c95450104469c2aa1685b5bca`.
Models/checkpoints/global caches/current caches remain untouched. Head free was
about 13.0 GB after retention. Source/artifact checks, idle guard, independent
watchdog, exact original container identities and health200 remain mandatory.

## Historical compression checkpoint:18 October12:18 UTC

The following stopped-state record predates the user's explicit resumption.

Read the live controller before acting; counts below are a dated observation.

**Work is paused at the user's explicit request. No further experiments are
running or should be started until the user resumes. The original service is
restored and healthy.**

## Outcome so far

Compression saves **9.367 GiB across TP2**, but the throughput problem is
**not fixed**. The matched C8/C16 comparison is independently verified and
fails both speed gates and numerical qualification. The output-only row16
follow-up recovers part of the C16 loss, while remaining slower than stock.
No configuration is promoted. **MTP is explicitly deferred by the user.**

The current priority is to finish the compression validation and profiles,
restore the original service, retain and independently verify the evidence,
then investigate the unresolved layer-3 MLA numerical divergence.

## Stopped state

**`output16qualv1` is complete and independently verified on Spark and Mac.**
Exact original identities/configs/mounts and health200 were restored at
**12:14:05 UTC** and rechecked at12:17 UTC. All experiment containers are stopped.
Do not replace or recreate the original containers.

| Item | Current observation |
|---|---|
| Controller | Historical PID **3493991**, terminal phase `complete`, `service_paused=false` |
| Actual client work | Validation only; the generic controller phase name includes historical wording. **Zero new timing groups.** |
| Measurement | `complete`; all evidence independently retained and verified |
| Progress | Warmups **24/24**; fresh quality groups **8/8**; profiles **4/4** |
| Collector | Historical PID **3493990**, terminal status `complete` |
| Restoration watchdog | Historical PID **3496648**; restoration completed normally |
| Head disk free | Last observed **5,627,609,088 bytes**, about 5.24 GiB |
| Errors | None reported by this continuation at this snapshot |
| Original API port8000 | **Health200**, exact original containers running; experiment stopped |

The continuation uses exactly the previous output-only decoder. All178 parent
source bindings remain unchanged; two validation helpers bring the binding
count to180. It collects eight fresh full-vocabulary quality groups and four
short profiles per TP rank. All72 previously completed timings are reused.
The model-load/time gap is explicit. A fresh source-after check **cannot repair
the missing source-after check from the original timing load**.

Independent verification confirms both speed gates failed, all24 numerical
pairs failed KL and8 failed NLL, with672 source-scale groups checked and9.367GiB
CUDA allocation saved across TP2. All generated IDs and conditioning match in
the24 quality comparisons; the probability gates still fail.

All**1,293 payloads /180 source bindings /16 worker traces** independently
verify. Cross-host decisions are exact; probability-only float64 reduction
differences are at most2.00e−15 under the1e−12 verification tolerance.
The archive is362,942,944 bytes, SHA:
`d871cbac8d1d91332469c01434ce72549679974d8e1653d0198642fbf8c44dba`.

Device correlation analysis also matches exactly on Spark/Mac. Eight retained
pure-decode ranges across both ranks have10 or15 actual requests using the
padded16 graph. Every range reconstructs**34 matrices**, all inside the graph,
versus68 in the earlier parent profiles. Reconstruction sums are25.68–28.11ms;
these are diagnostic kernel sums, not request latency or a TP critical path.
All decoder events are assigned; unrelated unassigned kernel events remain
explicit. The intended mechanism works, but it only partly recovers C16 speed.

The route changes only the4096×4096 KDA output projection at exactly16 rows.
Input projections and unmeasured eager rows9–15 retain their previous routes.
The global fused limit stays8. Native libraries, precision, two-slot scratch
pool (206,569,472 bytes/rank) and resident BF16 weight-cache size (zero) stay fixed.

## Where to resume

Local worktree:

`/Users/kyrylobilchenko/Documents/Projects/other/mxstream/.worktrees/glm53-flash-mxfp4`

Branch `feat/glm53-flash-mxfp4`. There is substantial existing uncommitted work;
do not reset, clean, commit or push it. GPU sources are isolated from this
worktree; do not substitute pre-existing unqualified MoE edits into a run.

SSH alias `spark` is the existing head. Remote experiment root (`ROOT` below):

`/home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/full-model-v1`

| Purpose | Path below ROOT |
|---|---|
| Live lifecycle | `quality-model-controller-output16qualv1.json` |
| Actual collection progress | `run-output16qualv1/results/validation-high-concurrency.json` |
| Collector / archive receipt | `output16-validation-v1-evidence-check.json` |
| Operational entry point | `CURRENT_LOSSLESS_STATUS.json` |
| Frozen source | `full-output16qualv1-src/` |
| Prepared source/native/protocol identities | `output16-validation-v1-prepared.json` |
| Launch and collector/verifier hashes | `output16-validation-v1-launch.json` |
| Frozen validation plan | `run-output16qualv1/results/validation-plan.json` |
| Original timing report, preserved separately | `run-output16qualv1/results/original-output16-timing.json` |
| Derived cross-load review, once written | `run-output16qualv1/results/quality-model-review.json` |
| Eventual evidence / archive | `output16-validation-v1-evidence/`, `evidence-output16-validation-v1.tar.gz` |

Local continuation toolkit:
`/private/tmp/mxwave-output16-validation-v1/`.
Its `retain_local.py` has finished successfully. `evidence.tar.gz`, extracted
`evidence/`, `remote-evidence-receipt.json`, `mac-independent-review.json` and
`cross-host-verification.json` are present. Graph analyses are retained as
`spark-graph-attribution.json` and `mac-graph-attribution.json` and match exactly.
Do not rerun extraction or first-attempt launch scripts.

Frozen identifiers:

- Source archive: `b6c3ef4a09b666c49bfe63c8f8dcea68d7abadc518ed71bbbea1cb8eb0e4883f`.
- Prepared: `35039b55d34148b2544d1ca6a120219e1d4289e13b479dde2712a9c524456f1b`.
- Plan: `cf68aa8a0630a82592f1b15da59e1965ab80807202aa5fc1bd74bf267721375b`.
- Verifier: `15ace6f39cbd0c66937f70d39a0295959aa5a5afd5bc715bbdbf0cea160fdbd7`.

The original container IDs, image/config/mount identities are authoritative
in the prepared/controller JSON and `full_model_control.ORIGINAL_IDS`:

- Rank0: `8e0426b9a4183b8fdaa2ed2d4047924e9554b2cb37950e8f31bce52589d07e11`.
- Rank1: `fb147a890d8bdd2b475588935a3e1d6d7c60828f894bc980bc7b6e7a40d00224`.

Restoration must restart these exact containers and pass health200 and identity
checks on both nodes. Do not end active GPU work with originals still paused.
Host checks use stdlib `python3 -S`; Torch/CUDA runs only in bounded containers.

## Verified results and retained failure

`c816retryv2`: 1,277 payloads,174 source bindings,144 timing groups,1,272 requests,
81,408 tokens and16 worker traces independently verify. Archive:
`602e25daf5a498a2b6012c41276cb91f41695ccfd2093108a7eaacae0239d66a`.
Eight retained C16 pure-decode ranges across TP2 each rebuild68 KDA matrices
in34–37ms. These kernel sums are not request latency or measured LPDDR traffic.
All24 numerical comparisons fail KL, including self-repeats; four also fail NLL.

`output16v1`: all72 candidate timing groups /636 requests /40,704 tokens finish.
Disk exhaustion then interrupts the sixth numerical group; five complete,
candidate profiles are absent, and even writing the failed report header
fails with ENOSPC. The raw report therefore still says `running`.
**Controller `failed_restored` and client logs are authoritative.** It was not
OOM-killed. Exact originals restored at11:16:07 UTC before the new continuation.

Recovery verifies all1,532 payloads on Spark/Mac. All636 paired timed sequences
match; all17 available complete numerical pairs fail. Recovery archive
304,447,995 bytes, SHA:
`683fcb4bbc7b204d1a1b3a656766336a05efb389e0d17ed7ff0f641a26697ccb`.

| Workload | Output-only TPS, 0 / 3ms | Change versus retained stock |
|---|---:|---:|
| C1 / P256 | 16.959 / 16.809 | +14.00% / +13.39% |
| C4 / P256 | 40.229 / 43.857 | −0.80% / +2.34% |
| C8 / P256 | 60.115 / 63.108 | +3.48% / +3.31% |
| C8 / P2048 | 31.427 / 31.344 | −0.11% / −0.53% |
| C16 / P256 | 71.248 / 71.657 | −10.90% / −10.20% |
| C16 / P2048 | 34.455 / 34.391 | −6.23% / −6.30% |

C16 improves3.4–4.0% on short prompts and1.3–1.5% on long prompts relative
to the compressed parent, but both original speed gates still fail. C1 TTFT
regresses8.7–9.9%. Stock and compressed-parent controls are reused with explicit
time gaps; these are not fresh interleaved loads.

Local verified failed-run evidence:
`/private/tmp/mxwave-output-row16-v1/evidence/`.
The [compact partial review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/output-row16-v1-partial-review.json)
contains independently recomputed timings and all available numerical pairs.
Discrete decisions match exactly across hosts; probability-only float64
reduction difference is1.20e−16, below the1e−12 verification tolerance.

Unchanged model gates: every C8/C16 case TPS ratio≥1.05, C1/C4≥0.98,
all median TTFT ratios≤1.05; KL≤1e−5, absolute mean NLL≤1e−4,
matching conditioning/IDs, exact reconstruction and complete source coverage.
Matching output IDs alone does not qualify the model.

## Next actions after the user resumes

1. Read this handoff and verify the original service remains healthy. There is
   no live run to resume. Do not rerun72 timing groups or already verified
   collection merely because earlier historical records say pending.
2. Use the [completed continuation review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/output16-validation-v1-review.json)
   and retained traces as the current result. Preserve the original failed-run
   archive, missing original timing source-after check and all time gaps.
3. Resolve the MLA numerical cause separately. The actual layer3 internal
   observer is frozen locally at `/private/tmp/mxwave-mla-internal-probe-v1/`
   (162 expected bindings; source archive2747892d...). It has **not been
   uploaded or GPU-run**. It observes query/value BMMs, cache writes, sparse
   slot conversion, actual planned KV bytes and attention output; four balanced
   tracked/private-workspace probes retain all five replays and serve the fifth.
   A passing isolated attention fixture has not resolved the actual-module drift.
   Preparation now requires this continuation restored/retained and at least
   **12GiB free/node** for temporary complete snapshot images.
4. Keep MTP deferred. Roofline figure collection and PNG/SVG links are still
   owed after model work. No new numerical or throughput success is claimed.

Do not repeat unchanged decoder occupancy, N8, split-K or naive-overlap tests:
they already failed their gates. V7's18,432 exact component events qualify only
the square output matrix at16; the input reference-repeat control fails the
unchanged2% timing limit. No blanket/global threshold or9–15-row interpolation.

## Storage, authorization and local checks

Earlier recovery moved only8,950 checksum-verified older generated payloads /
archives (7,593,681,896 bytes) to the existing worker:
`ROOT/storage-recovery-v1/offloaded-payloads/<original-relative-path>`.
Head-side `offload-plan.json`, backup and cleanup receipts retain all paths /
hashes. Two regenerable extracted evidence directories were removed after
canonical archive checks; their archives remain on head, worker and laptop.
Model checkpoints and global caches are untouched. The original service's
derived33-shard checkpoint differs from the audited120-shard source; do not
deduplicate or delete model data.

A read-only inventory found about10.8GB in older per-run compilation caches;
**no cache cleanup has run**. The local cache-backup script is a draft, not an
executed recovery. Complete compression profiles first, then recover additional
space with hash verification before any MLA launch. Do not assume the12GiB
diagnostic budget is currently available.

The user already authorized ongoing source-only revisions/uploads to the same
Spark experiment and nodes. Do not request that permission again. Documentation
and native implementation inspection snapshots remain local. No MTP launch,
no unlicensed upstream implementation copying, no unsolicited external messages.

Latest full pytest passes with one skip; Ruff passes; strict core mypy passes35
files. New independent NumPy MLA analysis has three focused passing tests;
it is local only and adds no GPU-runtime binding. No tests imply GPU qualification.

The experiment's owned sleep-prevention process, `caffeinate` PID50382, was
released after the user's pause. Unrelated power assertions/processes were
left alone. Receipt `/private/tmp/mxwave-experiment-awake.json`.

See [current state](GLM53_CURRENT_STATE.md),
[output-only experiment](GLM53_OUTPUT_ROW16.md) and
[historical handoff](GLM53_LOSSLESS_RUNTIME_HANDOFF.md) for fuller evidence.
