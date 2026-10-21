# CT drafter: verified features and context-projection pilot

**21 October 2026, 12:05 UTC: small Modal training pilot complete; original production restored.**
The user-authorized bounded run completes **64 updates on 64 training prompts** in
**259.902 seconds (4m 20s)**, with 16 development-validation prompts and no skipped
examples. All 84 capture/control requests completed. Original production restored
at **11:55:08 UTC**, with original container identities and **500,000 context**;
a fresh independent health/identity/context check passes at **12:04:14 UTC**.
No experiment/training is running, no full 4,000-prompt run or native candidate
serving trial is queued, no checkpoint is promoted, and sealed test stays closed.

Offline mean matching prefix, **seven proposed tokens**, four fixed CT-history
anchors per validation prompt (this is not native acceptance or end-to-end speed):

| Domain | Validation prompts | Pretrained Modal | Selected step 16 | Change |
| --- | ---: | ---: | ---: | ---: |
| Code | 8 | 4.1875 | 4.3750 | +4.4776% |
| Chat | 4 | 2.4375 | 2.4375 | unchanged |
| Explanation | 4 | 3.5000 | 3.5000 | unchanged |

**The code gain comes from one prompt and one of 32 code anchors.**
`opencoder-evol-56582` changes prefixes `[4,7,1,5]` → `[4,7,7,5]`.
Steps 32 and 64 give the same prefix metrics; the frozen lower-code-CE tie-break
selects step 16. Later unary CE does not improve consistently. Exploratory capping
of these seven-position prefixes at five gives code 3.5625 → 3.6875 (+3.51%);
this is not an actual five-draft forward or a changed checkpoint-selection rule.
This is a modest, localized offline signal, insufficient to justify full training
or claim the ≥8% native code-throughput goal. A matched native five-draft comparison
would still be required before deployment or broader training; known target
output-invariance limits remain open.

All **10,569 capture payloads** independently verify on Mac and Spark2. Capture
archive **3,267,567,485 bytes**, SHA256:
`a42e20dd40deb674ac1cb0be5ee5a8e0b26b22e26ae28316bfff03efee62d663`.
Accepted export: **80 prompts / 17,871 feature rows / 878,726,159 bytes**. A separate
NumPy audit compares **878,395,392 BF16 elements** across both native rank traces,
with zero differences. Independent training review recomputes 204 acceptance
fields, 1,792 proposal-token comparisons, all three checkpoint distances and the
selected step. All original training-model parameters remain unchanged; only the
FP32 master and separately exported BF16 `fc.weight` are adapted.

The native FP8 candidate is exported but **not loaded or qualified**. All **137
other tensor payloads**, all FP8 scales and configuration bytes remain identical;
checkpoint serialized size is unchanged. Selected patch SHA256:
`fe066544b4b5851a8232a0115d6229052f5f740b50d36559ad5b001f4f0f60c9`.
Export SHA256:
`6dca2af520c6df5105eca783a07d1b170ea16ddbcefe31aba6ece54cec741925`.
Context relative L2 is 0.0000913413; 3,910,948 BF16 entries change. Target weights,
compression policies, drafter architecture and frozen backbone/selector do not
change. Native acceptance, speed, quality and GPU memory have not been remeasured.

**Capture acceleration audit:** request execution totals 290.53 seconds, while
first-request start to last-request end is 490.13 seconds. Gaps total **199.60
seconds**, including **166 seconds of programmed idle-confirmation sleeps**. The
next cheap experiment is shorter idle confirmations with the same repeat controls,
before C2/C4 request-keyed collection. Neither faster policy was run. Current
recorder uses one marker and asserts C1; installed vLLM exposes request IDs, slot
maps and row boundaries needed to replace it. Both Sparks form one TP=2 target,
so independent target replicas are not available on these two GPUs. Bulk evidence
transfer is implemented and replaces serial per-file SCP; no measured speedup
factor is claimed for that change.

Local evidence: `out/drafter-modal-pilot-v1`; native candidate:
`out/drafter-modal-pilot-v1/native-step16`. Capture and its independent backup:
`full-model-v1/drafter-modal-pilot-v1` / `drafter-modal-pilot-v1-retained-backup`.
Selected native candidate is independently verified on Spark2 at
`full-model-v1/drafter-modal-pilot-v1-step16` (only FC changed, 137 payloads and
config exact). All three checkpoints and 28 training-evidence payloads also verify
in the Spark2 retained backup: 470,203,322-byte archive, SHA256
`69704329ad9cebc598a37ac115d53637d3900f4d80fe5a79c2e727f2ca662693`.
Full pytest, Ruff and strict core mypy pass. Preserve the earlier failed and neutral
runs. **Recommendation: do not restart full training on this evidence.** Keep this
small candidate for a matched native check; serving-overhead profiling remains
another independent, promising route.

**21 October 2026, 11:53 UTC: Modal pilot capture completes; original production restoring.**
All **84/84 requests** complete, including 64 train and 16 validation captures.
Both before/capture/after controls have exact output tokens and speculative counters.
The trial is stopped; original 500k containers are restarting under the restoration
controller. Data export/independent bit audit and CPU training have not begun yet.
No training benefit or serving gain is claimed. The captured run remains frozen;
source commit `2d6ce3e` is backed up on Spark2. Capture uses C1; the installed API
has request IDs and row boundaries needed for a future C2/C4 recorder, but that
concurrent recorder is not implemented or qualified in this pilot.

**21 October 2026, 11:40 UTC: bounded Modal capture running detached.**
Launcher 3495096 / controller 3495097 started the frozen 84-request capture.
Original production is **temporarily paused**; CT is loading normally. The watchdog
outage deadline is **12:13:14 UTC**, followed by bounded restoration if needed.
No training update yet. CPU continuation waits for successful capture, original
500k restoration, all payload hashes, accepted-prefix export and independent
NumPy verification. Training is capped at 64 updates / 20 minutes. Source commit
`2d6ce3e` is backed up in Spark2 Git. Live controller:
`full-model-v1/drafter-modal-pilot-v1/controller.json`; local continuation:
`out/drafter-modal-pilot-v1/pilot-status.json`. The capture survives laptop
disconnection; the queued CPU pilot currently runs from the Mac. No sealed test
or automatic promotion. This supersedes the preceding prelaunch status.

**21 October 2026, 11:37 UTC: short Modal-specific training pilot authorized; CPU checks pass.**
The user requests a small training run and asks about parallel capture. Full training
remains out of scope. Freeze **64 training / 16 validation prompts**, 128 output tokens,
seven drafts for capture, then at most 64 CPU updates of `fc.weight` only. All other
Modal FP8 tensors and target weights remain fixed; sealed test stays closed. The
planned native comparison, only if offline development improves, must use matched
five-draft widths and record output differences rather than assume target invariance.

The existing 1,088 verified bulk captures use Inco's five feature taps and cannot
train Modal's six-tap projection. New taps are 23/27/31/35/39/43, causal attention,
window 4096. The Modal FP8 → BF16 → FP8 round-trip preserves all 138 tensors and
complete checkpoint/config bytes exactly (10.45 seconds). A historical inspected
Modal fixture passes both-rank feature equality, exact explicit/inference masks,
and finite gradients through the frozen backbone in 9.04 seconds; **zero updates**
in this check. This is pipeline validation, not a training gain.

Fresh capture revision `drafter-modal-pilot-v1`, API 8043, master 29716. Both serving
image CPU checks pass with 211 runtime bindings per rank. Plan SHA256:
`e9f5d8b6ff3711e38c6bc9738ddeea123188dd4a210748a44ccc22bf25807293`.
Original production remains online at this checkpoint; GPU capture is not yet
launched. The existing watchdog bounds capture to 35 minutes plus restoration;
training starts only after accepted-prefix/both-rank verification and original
500k production restoration. CPU training will use 12 threads with a 20-minute
process limit. Estimated capture/load/restoration plus training: about 20–40 minutes,
subject to startup. A native speed test would take additional time.

Capture concurrency cannot safely be increased by only raising the client count:
the recorder has a single marker and asserts one request. A parallel implementation
needs request/slot mappings, per-request accepted histories, mixed-length controls,
and graph-shape checks. The two Sparks form one tensor-parallel target, not two
independent model replicas. This bounded pilot keeps C1 capture; bulk peer transfer
replaces thousands of serial SCP handshakes after restoration. No C2/C4 capture
speedup is claimed or launched.

**21 October 2026: full capture stopped at the user’s request; pretrained test moved forward.**
Capture ends at 1,095 prompts; original production restores healthy at 05:46:55 UTC,
with its original 500,000-token context. Both Sparks independently rehash all
1,088 prompts in 17 completed chunks: 13.10 GB retained per node, zero differences.
Unexported raw files remain. Training stays stopped; the incomplete capture is
not published as a full training dataset. All eight CPU preflights pass; the shorter
BF16/FP8 comparison now starts under its own restoration watchdog. Production
is temporarily paused for this new comparison. [Current details](GLM53_PRETRAINED_DRAFTERS.md).

**21 October 2026, 05:42 UTC: automatic training held; pretrained BF16/FP8 comparison queued.**
The user prefers avoiding adaptation if an existing drafter performs well. Modal’s
six-layer DFlash2 passes strict CPU loading and both installed vLLM helper checks.
A separate FP8 conversion takes 6.606 seconds: 2.78 → 1.62 GB checkpoint bytes,
42 quantized matrices and 54 bit-identical tensors; independent NumPy checks pass.
No acceptance or speed result yet. The GPU capture continues unchanged; the idle
CPU training launcher is stopped before any update. The bounded pretrained
comparison waits for capture verification/restoration. [Details and limits](GLM53_PRETRAINED_DRAFTERS.md).

**21 October 2026, 05:23 UTC: capture continues without restart; CPU training queued at 12 threads.**
Latest capture read: 892/4,200 prompts, 13 paired chunks, two passing repeat controls,
no error. The 12-thread CPU replay preserves all 32 update metrics and four saved
checkpoints exactly. An interleaved step screen falls from 2.483 to 1.805 seconds;
full-corpus runtime remains unmeasured. Continuation v2 waits for complete verified
capture and production restoration. Estimated CPU training is now 3–4 hours;
serving acceptance/speed remain unqualified. [Current handoff](GLM53_HANDOFF.md).

**21 October 2026, 04:53 UTC: full capture continues; 576 prompts independently verified.**
Nine completed batches have identical accepted payloads on both GPUs; NumPy
checks cover 7.05 billion BF16 elements with zero differences. Both repeat controls
so far pass. Production is temporarily paused; CPU training waits for verified
completion and restoration. No new drafter speed gain is established.
[Current handoff](GLM53_HANDOFF.md).

**20 October 2026, 22:37 UTC: the combined ordering fix passes the long-prompt C1 repeat check.**
All six requests match in tokens/counters; all 12 capture comparisons have exact
target features/logits and initial drafter fields. All 430 evidence payloads
independently verify on Mac/Spark2. Decode sorting adds about 0.0155 ms per indexer
call in stable component timings (19/24 controls pass); serving cost is unmeasured.
Original 500,000-context production is restored. Next is the eight-prompt capture
smoke before full 4,000/200 capture/training; sealed test remains closed.
[Evidence, limits and next step](GLM53_HANDOFF.md).

**20 October 2026, 21:37 UTC:** all 12 independently reviewed comparisons on both
GPUs show identical sparse-prefill scoring inputs, consumed scores and selected
tokens, but native top-k changes their order in 1,921–1,922 prompt rows. This
component defect is established; an ordering fix and its full-model effect are
not yet tested. The repeat gate still fails. Original production is restored
with 500,000 context; all 416 evidence payloads verify on Mac/Spark2. No experiment
is running. Full 4,000/200 capture and training remain gated; the old drafter
pilot is speed-neutral and unqualified, and the sealed test remains closed.
[Current evidence and restoration status](GLM53_HANDOFF.md).

**20 October 2026, 20:07 UTC:** our wider expert-padding component check passes
all 36 fixtures through 128 rows. Independent review finds 54/100 native
shared-prefix comparisons differ, versus 0/100 after padding; all 2,880 graph
repeats are exact. Timing controls pass 27/32, with five failures retained.
Stable component overhead varies from near zero to 8.02% with routing, so this
is not a serving-overhead qualification. Production is restored with 500,000
context; the runtime limit stays at 16 rows. Fresh C1 features and full
4,000-prompt drafter training remain next; the sealed test stays closed.
See [the wider result and timing table](GLM53_HANDOFF.md).

**20 October 2026, 19:23:01 UTC:** the eight-prompt, 144-request development
comparison completes with identical output tokens across original/adapted/original
loads. All within-load and original-reload token/counter controls pass, as do
all eight 2% reload timing controls. The step-24 pilot is effectively speed-neutral:
code **+0.111%**, chat **−0.060%**, explanation **−0.329%** end-to-end throughput.
Both arms use the same CT MXFP8 target and expert padding. This does not establish
the ≥8% code goal or padding overhead; the pilot remains unqualified.
Original production is restored and healthy with **500,000 context**.
All 93 evidence payloads independently verify on Mac/Spark2; no experiment is running.
See [the result and TTFT table](GLM53_HANDOFF.md).

**20 October: direction updated at the user's request.** The DeepInfer-inspired
confidence experiment and its trainer hook are removed. It was never enabled
in serving and has no measured CT gain. The audit is retained as history.
Continue our independently developed fixed-width drafter adaptation and Marlin
padding. The bounded-loader replay now passes: all 32 update metrics and all
four saved checkpoints match the original exactly. Wider component GPU checks
now also pass. Fresh target-feature capture precedes full 4,000-prompt training;
C8/C16 full-model and serving-cost qualification remains a separate gate.
Existing gates are unchanged; sealed evaluation remains closed.
See [the current handoff](GLM53_HANDOFF.md).

**Earlier, 20 October 2026, 18:18:32 UTC:** the full-model check passes and original
production is restored with 500,000 context. On one development prompt, the
original/adapted drafter arms have 32 identical output tokens and bit-identical
full-vocabulary decision logits on both ranks: measured KL 0. All request and
captured-feature repeat controls pass; 250 evidence payloads verify on Mac/Spark2.
Both drafters now propose identical inputs in aligned shared-history frames and
use nine draft rounds, so broader prompts must still test differing proposals.
No acceptance or serving-speed gain is established. The component padding fix
passes the original 440/448 stress case; V3 timing controls remain 14/16 with
two failures retained at 2%. C8/C16 coverage and full 4,000-prompt training remain
outstanding. The adapted drafter remains unqualified and the sealed test closed.
See the current
[handoff checkpoint](GLM53_HANDOFF.md). Older dated checkpoints below are historical.

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

Updated **20 October 2026, 15:59 UTC**, after the completed V2 trace and production restoration.

The fixed target is **CT MXFP4 experts plus the previously tested native MXFP8
KDA conversion**. That KDA conversion is lossy; original BF16-KDA CT remains the
lossless reference. No further target quantization changes are made here.
The failed cross-target KL gate and the failed selector-only pilot remain
retained. This experiment does not waive either.

## First-divergence trace: retained V2 result

The corrected client runs one development prompt, `opencoder-evol-15654`, with
32 output tokens, temperature zero, seed 42, seven drafts and native eight-row
verification. Original/adapted loads each execute two warmups and four requests
(before, capture, repeated capture, after). No weights, target arithmetic or
sampled outputs are substituted by the probe. Both arms reproduce their earlier
uninstrumented output prefix and counters in all four requests. All 80 frames
are retained; both captures match inputs/logits/features exactly on both ranks.
The probe can affect latency, so this is **not a speed measurement**.

The first five recorded target forwards match across arms, including all eight
input rows in each decode step. The sixth forward starts at output index 21:

| Observation | Original drafter | Adapted drafter |
| --- | --- | --- |
| Input IDs at positions 54–60 | 911, 419, 15504, 382, 785, 6770, 3561 | Identical |
| Last input, position 61 | 21613 | 369 |
| Actual recurrent starting states, layers 0 and 4 | Reference | Bitwise equal |
| Shared rows 1–6, feature boundary 5 | Reference | 800 / 982 / 333 / 287 / 338 / 461 differing BF16 elements |
| Shared row 0, feature boundary 5 | Reference | Exact |
| Shared row 0, feature boundary 14 | Reference | 1,762 of 4,096 differing elements |
| Token 419 logit for first output | 27.625 | 27.625 |
| Token 1128 logit for first output | 27.625 | 27.75 |
| Selected output | 419 | 1128 |

Both GPU ranks agree on every value above. Across the full 154,880-element
logit row, 141,144 values differ, maximum absolute difference is 0.25, and
forward KL is **0.0020250586**. This is one next-token distribution under
matched accepted history, not a corpus-level KL estimate or task-accuracy
measurement. The unchanged-output gate remains failed.

The selected recurrent cache column is `num_accepted_tokens - 1`, verified
against the **actual imported** installed FLA kernel, not the separate Kimi
implementation. Audited source SHA-256:
`371e7d733e90a7147fbca8dbfe67e62503ba663bd1570424bc0efa22c54f3e2b`.
Only layers 0 and 4 have raw recurrent states. Unused speculative slots are
excluded; convolution-history slices and other layers' caches are not compared.
The second logit copy is the original logits storage after the sampler returns;
processing can allocate another tensor, so this is not an independent capture
of processed logits. The numerical difference already exists before sampling.

The evidence points to target verification responding differently when later
draft inputs change, despite matching accepted history and a fixed row count.
It does not identify the first differing kernel or establish a causal attention
mask defect. Attention, routing/expert arithmetic, mHC and state handling remain
candidates until the first changed component is isolated. The next diagnostic
should retain native graph-written component inputs/outputs for layers 0–4,
check the same before/capture/repeat/after controls, and reuse this exact pair.
If an expert call is first, a stock/packed × original/changed-input fixture will
separate stock arithmetic from the lossless adapter. Do not train further or
relax equality gates on the basis of this one reproducer.

V2 plan SHA-256:
`965dc634a26117c72eebec46571a9763a97967b7140c52796ff23f346245128d`.
Local review: `out/drafter-verification-v2/review.json`; independent reviewer:
`integrations/glm53_mxfp4/review_drafter_verification.py`. The 229-payload archive
is retained on Spark, Mac and Spark2, and all hashes independently verify on
Mac and Spark2. Original production's IDs/configurations, health and 500k
context are restored at 15:54:30 UTC. No new diagnostic is running.

The first attempt (V1) failed before any inference request because the generated
client's metrics regex was over-escaped. Its 55-payload archive remains retained
and verified, with original production restored at 15:35:38 UTC. V2 changes only
the client parser and directory/port identity; regression tests execute the
emitted parser on labeled/unlabeled, busy and missing metrics, and the corrected
parser was also tested against actual production counters before the retry.

## Training result

The run completes all 32 updates in **177.386 seconds**. Step 24 passes the frozen
offline selection; step 32 fails its chat no-regression requirement.

| Validation mean accepted draft prefix | Original | Step 24 | Step 32 |
| --- | ---: | ---: | ---: |
| Code, four prompts | 4.25 | 4.375 | 4.3125 |
| Chat, two prompts | 2.75 | 2.875 | 2.625 |
| Explanation, two prompts | 4.25 | 4.25 | 4.25 |

These represent only 16 code anchors and eight anchors in each other domain.
Code improves by two accepted draft tokens across its 16 anchors. This is a
small development signal, not evidence of the required 8% serving gain.
Code CE at step 24 is 1.61687 versus baseline 1.61515: it does not improve even
though the selected discrete acceptance metric does. No broad quality claim.

The independent review reproduces **1,120 proposal-token comparisons**, **135
acceptance fields**, all checkpoint hashes and the same selected step. It detects
about 2% relative error in the FP32-reduced weight-distance statistic. Separate
NumPy and Torch FP64 reductions agree on **0.0001171596963** for step 24, with
**5,864,036 changed BF16 elements**. Original reported values remain retained;
the corrected distances do not change the frozen 0.01 gate or selection.
Future trainer and native-export distance checks now accumulate in FP64.
The exact executed V3 trainer remains retained beside its report.

Native export preserves the original file layout and all **115 other tensor
payloads**, including FP8 codes and scales, plus the exact config bytes. Only
`fc.weight` changes. Export SHA-256:
`bf67cb8c2a02e3999269294b7d2b6fc63ce52ebd694fecbd8dac56f5971bdf47`.

## Completed native development comparison

All 144 timed requests complete: eight validation prompts, six shuffled cycles
per load, original/candidate/original, plus two excluded warmups per load.
C1, 256 output tokens, seed 42, temperature zero, seven drafts, prefix caching
off and the identical CT MXFP8 target are fixed. This is one candidate load;
it cannot meet the final three-load/held-out serving requirement.
Plan SHA-256: `5dd97e009313b9ec390ad07933c99fe16ce17a2ae9ceae5ad582bba487dbd9ba`.

Ratios below use per-prompt six-trial medians, the geometric mean of the two
original load medians as control, then a geometric mean across prompts. They
are **descriptive results on changed answers**, not qualified acceleration.

| Domain | Prompts | End-to-end tokens/s change | Verification-round change | TTFT change |
| --- | ---: | ---: | ---: | ---: |
| Code | 4 | +5.91% | −8.89% | +5.03% |
| Chat | 2 | +1.61% | −2.24% | +7.62% |
| Explanation | 2 | −4.57% | +3.97% | +12.09% |

All eight prompts repeat tokens and speculative counters exactly after the original reload.
The original reload speed differences stay between +0.14% and +0.90% per prompt.
All three loads repeat their own outputs/counters in all six cycles.
Only `dolly-1769` preserves all 256 tokens across original/candidate: 91 versus
90 draft calls, about +0.65% paired end-to-end throughput. The other seven change
at token indices 21–254. No task-accuracy conclusion is available.

The post-hoc common-prefix comparison finds 547 identical code tokens before
divergence, delivered in 151/150/151 chunks for original/candidate/original.
Those chunk counts repeat across all six cycles. Delivery chunks include the
enclosing final event; they do not prove identical internal verification work.
The much larger full-answer gains cannot be attributed solely to better drafting
on the same output trajectory.

Both ranks retain native `[1, 8]` graph replays: last logged counts are
**4,032 / 3,904 / 4,032** per arm. Counts include warmups and are lower bounds
because the final fewer-than-64 calls may not be logged. No feature copies are
enabled in timed serving. Clock/thermal sampling retains 107/105/106 pairs of
samples across the loads. All observations report P0; median clocks are
2,470–2,489 MHz and temperatures span 49–76 °C. Sampling does not prove an absence
of throttling; no clocks were locked and throttle-reason counters were not read.

The unchanged-output gate fails. Even disregarding it, this development result
misses the code +8%, explanation −2% and TTFT +2% requirements. Do not promote,
relax thresholds, open the sealed test or infer quality loss from changed text.

The first-divergence comparison requested here is now complete; see the V2
result above. The next step is component-level localization on its frozen pair.

## Relationship to established drafter training

The overall capture/train/export/serve workflow follows published feature-based
speculation: [DFlash](https://proceedings.mlr.press/v306/chen26bv.html) conditions
parallel drafts on target features, and [SpecForge's offline preparation](https://sgl-project.github.io/SpecForge/basic_usage/data_preparation.html)
explicitly supports target-generated responses and captured DFlash2 features.
Our **FC-only, 32-training-prompt CPU pilot** is a deliberately restricted local
experiment; it is not a published guarantee that this small adaptation closes
the CT gap. [NVIDIA's DFlash recipe](https://docs.nvidia.com/nemo/automodel/recipes-e2e-examples/dflash-speculative-decoding)
trains the draft stack, feature projection and norms while freezing the target.

[vLLM documents](https://docs.vllm.ai/en/latest/features/speculative_decoding/#lossless-guarantees-of-speculative-decoding)
the distinction between standard rejection sampling's lossless objective and
finite-precision/batch-dependent output differences. That makes numerical
sensitivity a plausible hypothesis here, not an established explanation or
permission to waive our frozen equality gate. Sources checked 20 October 2026.

## Packaging and stock-vLLM compatibility

The current complete setup has **not** been qualified on an unmodified stock
vLLM release. Measured serving uses `0.1.dev20051+g487ecf187`, overlay-mounted GLM
and sampler/cache modules, custom startup hooks, and MxWave CUDA libraries.
A standard checkpoint format alone does not establish whole-stack portability.

| Component | Packaging route | Present limitation |
| --- | --- | --- |
| Adapted FP8 DFlash2 | Existing safetensors/config layout; only FC payload changes | Stock version must support this exact architecture/FP8 path; native validation currently uses the custom image |
| Source CT MXFP4 experts | Existing standard compressed-tensors MXFP4 checkpoint | Model/hardware support and the complete serving configuration still need a stock smoke test |
| Later MXFP8 KDA target | Export the already-tested rounded values and a fully covered native quantization config | Conversion is currently installed at runtime; the unchanged source checkpoint still contains BF16 KDA |
| Extra lossless expert-scale/vocabulary storage | Installable extension containing loader/dispatch adapters and CUDA kernels | Stock vLLM does not recognize the custom coding automatically |
| Current verification probe | Omit from distribution | Temporary development instrumentation, no serving requirement |

The practical distribution target is **stock vLLM plus a versioned MxWave
extension**, model/config files, and the separate adapted drafter checkpoint.
vLLM supports [out-of-tree quantization registration](https://docs.vllm.ai/en/stable/features/quantization/#out-of-tree-quantization-plugins)
and a [plugin entry-point system](https://docs.vllm.ai/en/latest/design/plugin_system/).
This is a supported integration mechanism, not proof that this whole GLM stack
already works without patches. A thin registration package already exists at
`integrations/glm53_mxfp4/engine_plugin/pyproject.toml`; its module payload is
empty and runtime modules are supplied through PYTHONPATH, so it is not a
self-contained distribution wheel yet.

Before advertising a normal `vllm serve` workflow: choose a stock release, audit
which overlay fixes it already contains, port required MxWave behavior into the
extension, emit/verify the MXFP8 KDA checkpoint, package architecture-compatible
kernels, and run loading/numerics/speculation/graphs plus throughput and context
checks in a clean stock environment. This is not just archiving the current
experiment directory. Using only stock-supported stored tensors could remove
the lossless runtime requirement, but would also give up its resident-memory
savings. Completely installation-free support for the custom lossless coding
would require upstream support.

Packaging is a recorded follow-up. The active first-divergence diagnostic and
its frozen numerical gates remain unchanged. Sources checked 20 October 2026.

## Final native evidence

- Controller: `complete_restored`, no error; production restored **15:14:02 UTC**.
- Original container/configuration identities match, health returns 200, and
  `/v1/models` reports `glm-5.3-flash` with **500,000 context**.
- Archive: **1,408,463 bytes / 83 verified payloads**, SHA-256
  `68079c34ef60623f696e97c24588c19d9393d17fa87db98875f9c9dbc8cac6bb`.
- Mac: `out/drafter-context-native-v1/evidence-v1.tar.gz`; original archive remains
  on Spark at the experiment root. Spark2 backup:
  `git-backups/glm53-drafter-context-native-v1-evidence.tar.gz` under the audit root.
  All 83 payload hashes are independently verified on Spark2.
- Committed summaries live under
  `benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/` with prefix
  `drafter-context-native-v1-`: review, diagnostics/source, archive receipt and
  first-divergence review/source. Earlier partial measurements are retained in
  local `out/`; final analysis reads the frozen archive.

## Capture result

| Check | Result |
| --- | --- |
| Captured prompts | 32 training, eight validation |
| Total API requests, including controls | 44 completed |
| Fixed output budget | 128 tokens per request |
| Before/after controls | Both match token sequences and speculative counters exactly |
| Native graph frames | 1,351 per rank |
| Archived payloads | All 5,456 hashes match |
| Accepted exported feature rows | 8,210 |
| Serialized training/validation export | 336,434,650 bytes |
| Rank-to-rank feature comparison | Bitwise equal |
| Separate raw-safetensors NumPy audit | 336,281,600 BF16 elements, zero differences |
| Excluded work | 5,247 rejected rows; 161 post-cap sampled tokens and feature rows |
| Sealed test | Unopened |

Features concatenate target layers **5, 14, 24, 33 and 42**. Each accepted input
token has one BF16 feature row, except the final capped response token. Input IDs
must match the frozen prompt followed by the actual API response. The independent
audit derives accepted intervals from subsequent query positions rather than
trusting sampler counts.

Capture used the original untrained FP8 DFlash2 checkpoint. Instrumented timing
is not a speed measurement.

## First pre-training failure and execution correction

V1 stopped before any optimizer update: four packed proposal blocks versus
individual-block forwards differed by **0.0145083 relative L2**, exceeding the
fixed 0.005 preflight limit. Its source and failure report are retained.

Independent mask checks pass. Zeroing the masked future context leaves the
first packed block bitwise unchanged. The first block differs from isolated
execution by 0.0017001 relative L2; among traced projection outputs, the first
difference is in layer 4's attention output projection (nine elements).
This supports shape-dependent BF16 arithmetic as a cause; it does not establish
every internal operation as identical or explain every later block's difference.

V2 uses **one independent eight-query-row forward per anchor**. Explicit masks
must reproduce the upstream single-block inference masks bitwise. The baseline
must also repeat exactly before training. Both checks pass. The data, losses,
optimizer settings and selection criteria remain unchanged. V1 is preserved;
the failed packed execution is not used for training.

## CPU derivative correction

V2 completed one update, taking **169.925 seconds** for the first step. A native
CPU stack sample locates the delay in PyTorch's BF16 GEMM backward fallback.
The run was stopped and retained after that completed update; it was not used
as the starting point for the next run.

V3 restarts from the original checkpoint. Linear forwards still call the same
BF16 `F.linear`. Their analytical derivatives use FP32 matrix multiplication,
then cast gradients back to the original tensor dtypes. This changes backward
accumulation rounding; exact gradient equality is not claimed. Independent
FP64-formula and gradcheck tests cover input, weight and bias derivatives.
The real first-prompt check preserves its loss and all proposal metrics exactly,
with finite gradients and **2.222 seconds backward**. Gradient L2 is 5.90428 versus
5.90611 in V2. All eight validation prompts reproduce V2's baseline exactly.
This is a CPU training implementation change, not a serving kernel change.

## Bounded training contract

Only the existing **83,886,080-parameter `fc.weight` context projection** changes.
Its master copy is FP32, and each forward/export uses BF16. All original model
parameters remain frozen, including the selector and backbone. The original FP8
codes/scales, target head, embeddings and target model remain fixed.

- One seeded pass: at most 32 updates, one prompt per update.
- Four deterministic anchors per prompt; four separate model forwards.
- Each anchor has seven CT-verified labels. Labels may end at the first EOS;
  post-EOS text is excluded. Short responses are reported and skipped.
- Mean full-vocabulary cross-entropy plus mean selector cross-entropy where the
  CT label is already in the strict unary top 16. No label injection.
- AdamW: four-step warm-up to 1e-5, cosine decay to 1e-6, zero weight decay,
  gradient clipping at L2 1, seed 42.
- Validation at baseline and steps 8, 16, 24 and 32, plus an earlier final step
  if short-response skips reduce the update count.
- Selection requires strictly better code mean accepted-prefix length, at least
  98% of baseline chat and explanation prefix lengths, and weight relative L2
  at most 0.01. Ties prefer lower code CE, then the earlier step.
- No improving candidate means no promotion and no sealed-test access.

Eight validation prompts are a development screen. Dense CPU arithmetic differs
from native FP8 Marlin and serving KV/head adapters. An offline improvement is
not an end-to-end speed result. A selected checkpoint still needs exact native
export/reload, new native validation, and the unchanged sealed serving gates.

The frozen full serving target remains at least **8% higher C1 code throughput**
against the original drafter on the same CT target, with the specified confidence,
repeatability, no-regression and round-count criteria. Production replacement
also requires serving capacity/tuning, stable C8/C16 measurements and long-context
qualification.

## Retention and reproduction

Local capture root: `out/drafter-feature-capture-v1/`.
Raw evidence is under `evidence/`; accepted sequences under `verified-features/`.
The independent report is `numpy-audit.json`.

Remote capture root on Spark:
`/home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/full-model-v1/drafter-feature-capture-v1`.

The original post-restoration finalizer opened one SCP connection per rank-one
file. Only that owned copier was interrupted after production restoration.
The replacement transferred one archive, verified all bytes and resumed the
original identity/manifest checks. `run-prepared-status.json` retains the original
finalizer's `-15` exit; this is the documented transfer interruption, not a failed
GPU capture. Both original and replacement finalizer sources and transfer receipt
are archived.

| Artifact | SHA-256 |
| --- | --- |
| Capture plan | `2e3dfcf8f61eadbbb7ae4092fed661da986daec7ed1d88921f5d771156bd4c7f` |
| Evidence archive, 1,403,252,349 bytes | `cab5f0cea65a88f54318d2dc7386980863382526cad41336731bb795a2f02f2c` |
| Accepted export manifest | `d79452e7dd86e75cdbd710c5afd4ebfa3aed3122633fa5dde8d0d8d56dc9cd62` |
| Frozen V1 pilot protocol | `14570e1e7c347356545241ddcb7e55b08a76a45c3c616752fbdd6e6b8cac626a` |
| Frozen V2 execution protocol | `54cf80b9ad70ec01e72f5838b5ec86499933808eac88247e8066e551e08a4dbf` |
| Frozen V3 CPU derivative protocol | `d5a1f0aca11182d48e32349fcd53fef6ad1a80315c904e4653e1a9f655775720` |

Pilot root: `out/drafter-context-pilot-v1/`.
`run-v1/` retains the failed preflight and executed source.
`run-v2/` retains the stopped slow CPU run and its executed source.
`run-v3/` holds the completed report, executed source and four BF16 checkpoints.
The report records every update, evaluation, source hash and selected candidate.
The full sealed dataset is never an input. The evidence archive is also retained
on Spark2 at the same capture-root path; its full SHA-256 matches Spark and Mac.

The native exporter accepts only a completed, validation-selected run. It patches
only the existing BF16 `fc.weight` payload into a copy of the original FP8 file,
verifies every other tensor and the config byte-for-byte, and rejects absent or
out-of-bounds candidates. It does not reload or promote a serving model.

```bash
PYTHONPATH=out/drafter-preflight-v1/audit-deps ../../.venv/bin/python \
  -m integrations.glm53_mxfp4.train_drafter_context_pilot \
  --dataset out/drafter-feature-capture-v1/verified-features \
  --checkpoint out/drafter-preflight-v1/roundtrip-v1/trainable \
  --checkout out/drafter-preflight-v1/audit-sources/SpecForge \
  --weights out/drafter-context-pilot-v1/weights \
  --protocol benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/drafter-context-pilot-v3-protocol.json \
  --output out/drafter-context-pilot-v1/run-v3
```

Output directories must be new; never overwrite a retained run.

Source verification: Ruff and strict core mypy pass; **906 tests pass, one
optional dependency skip**. The full native comparison still requires the actual
serving-image preflight and successful runtime checks.
