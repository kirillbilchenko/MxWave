# CT drafter: verified features and context-projection pilot

<!-- DRAFTER_COMPONENTS_BEGIN -->
**20 October 2026, 16:45 UTC: paired component capture launched after both-rank CPU preflight.**
The controller has paused production and is loading the original-drafter arm.
Launch PID 4002532 uses source checkpoint `76b625b` and the independent watchdog.
The bounded observer
covers attention projections/outputs, mHC, dense MLPs, expert routing and expert
GEMMs in layers 0–4 of the native eight-row target graph. It retains graph-written
copies only at start positions 46 and 54, with a 64 MiB graph-allocation limit.
Python component wrappers are removed after capture. All prior tokens and
speculative counters must reproduce in before/capture/repeat/after controls.

The frozen plan is `drafter-components-v1` (private port 8023), SHA-256
`afa5478d56f93f6c7035ead4a5fc3d0401088ea6154b2e47c8156a818ffd5fc2`.
The existing 25-minute experiment cutoff and independent restoration watchdog
are retained. No additional training, timing claim, promotion or sealed-test
access is included. Ruff, strict core mypy and the full test suite pass locally.
Both ranks validate all 211 frozen runtime bindings. Next: bounded paired capture, independent
component comparison, then a direct test of the first differing operation.
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
