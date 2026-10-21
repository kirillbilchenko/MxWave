# GLM lossless experiment: current state and next steps

**Latest, 21 October, 19:56 UTC:** the native DFlash2 zero-acceptance failure is
fixed in a bounded GPU diagnostic. Twelve existing BF16 convolution weights were
being loaded as FP8 without scales; correcting only their quantization exclusions
restores **50–67.5% acceptance** on two short prompts. Verification rounds fall
**31 → 8–10**; native graph replay and eager controls agree. Both ranks' retained
scores and eager hidden/logit tensors are finite. The 54 evidence payloads verify
on Mac and Spark2, and exact original 500k production is restored at **19:55:15 UTC**.
Package CLI integration and a new untraced comparison remain pending. No speed or
quality pass follows from these diagnostic timings; earlier target KL/repeat failures
remain. [Complete finding and limits](GLM53_STOCK_DRAFTER_LOADING.md).
Earlier dated entries below are retained as history.

**21 October, 19:35 UTC:** the user prioritizes diagnosing zero draft acceptance.
The long comparison was stopped with all partial data retained; exact original
500k production was restored at **19:35:23 UTC**. Remaining speed repetitions and
both full 1,319-question GSM8K arms are deferred. First short-code C1: production
accepts 195 draft tokens in 60 rounds versus CT's zero in 255 rounds, explaining
most of the 4.66 versus 20.47 second gap. This is not a compression-only comparison.
A bounded native-versus-draft-eager trace is being prepared; no fix is claimed.
See [the live handoff](GLM53_HANDOFF.md).


**21 October: matched selected-CT stock profile runs, but DFlash acceptance fails.**
The 8k / 16-slot / 6 GiB KV / compressed-vocabulary Modal profile allocates
**172.495512 GiB across TP2**, **0.750561 GiB more** than the historical selected
Modal CT profile (171.744951 GiB). This is not a fresh production-memory delta.
All six C1 requests complete with five proposals per call, shared head/embedding
and **0 accepted out of 3,810 proposals**. Stock startup is functional; useful
speculative execution is not qualified. Code/explanation repeats also differ.
V1's observer failure is retained; V2 fixes only that observer. See the
[profile results](GLM53_STOCK_CT_PROFILE.md) and live restoration state in the
[handoff](GLM53_HANDOFF.md). The CPU MTP audit finds 870/872 quantization-selector
mismatches despite complete source weights, blocking an unmodified MTP comparison.
Exact original 500k production is restored at **18:14:28 UTC**; no further GPU
experiment is queued. All 82 retained evidence payloads independently verify on
Mac and Spark2. Local Ruff, core mypy and full pytest pass.

**GLM stock-vLLM checkpoint, 21 October 2026.** The official vLLM 0.32.0 image
plus the installed MxWave plugin now reaches API readiness and generates responses
at 500k context / 32 slots / 6 GiB KV per GPU / prefix caching on. Selecting native
Triton draft attention resolves the cache-capacity blocker without changing vLLM
source. Actual capacity: 640,287 tokens; allocated startup memory: 87.157 GiB per
rank including the drafter, KV and graphs. A fresh matched memory-saving delta has
not been measured.

Current lossless scale/embedding storage saves **6.941 GiB across TP2**, after
the reported expert workspace, according to both-rank byte receipts. This excludes
the separate lossy MXFP8 KDA conversion and is not a matched production allocation
delta. The earlier **9.367 GiB** compression-only result included BF16 KDA and
the output head; the current recipe uses MXFP8 KDA and a dense output head.
The earlier **14.877 GiB** whole-model advantage over ModelOpt also included
quantization/format differences. These are three different measurements, not
evidence that stock vLLM removed the packing benefit. KDA precision was unchanged
during the port. See the memory table at the top of the [handoff](GLM53_HANDOFF.md).
Exact original 500k production is restored at **17:14:45 UTC** after V2.

**Serving qualification remains incomplete.** The corrected 12,449-token V7
check observes 9,216 prefix-hit tokens, confirming actual reuse, but repeated greedy
messages differ both cold-to-warm and warm-to-warm. V6 also differs without a
reusable prefix. Subsequent speculation-off diagnostics also fail self-repeat KL:
V2 cold/cold **0.0323304**, warm/warm **0.0249708**, against the unchanged 1e-5 gate.
All four V2 paired traces first differ inside layer 3's indexer output with earlier
recorded events equal. Its retained last row has the same index multiset in a
different order, followed by different attention output. A causal ordering/replay
probe is next; no fix, stock speed or long-context pass is claimed. V2's 305 archive
payloads and exact restoration are independently verified on Mac. No GPU run is
currently queued. The [handoff](GLM53_HANDOFF.md) retains failures and evidence.
Local checks pass: 1,087 tests / 1 skipped, Ruff and strict core mypy.

**21 October 2026: consolidated outcome and stock-vLLM deployment target.**
The compression mechanism works: it preserves the encoded weight values, reduces
live GPU allocation and improves some measured workloads. The complete CT serving
candidate is not yet qualified as a replacement for tuned production. The current
selection is CT MXFP4 experts + MXFP8 KDA + lossless expert-scale/vocabulary packing
and pretrained Modal FP8 DFlash2, with five drafts at C1/C2 and three at C3–C16.
MXFP8 KDA is a separate lossy quantization step; the whole candidate must not be
described as lossless relative to the original BF16-KDA CT checkpoint.

This work starts from the imported AutoRound compressed-tensors checkpoint. It
does not establish that MxWave generated a new full-model GLM MXFP4 quantization.
Our contributions here are codecs, direct GPU consumers, runtime integration,
profiling, repeatability diagnostics and serving/drafter experiments.

| Question | Verified result | Scope / limitation |
| --- | --- | --- |
| Does lossless compression save memory? | 9.367 GiB across TP2, about 4.68 GiB per GPU / 5.33% | Same CT checkpoint, matched KV allocation; this is not 9.37 GiB per GPU. |
| Does compression alone improve speed? | C1 +11.82–13.28% without speculation; C8/C16 near parity | The early overall gate still failed on latency. With DFlash, later gains were much smaller. |
| Did MXFP8 KDA remove the bottleneck? | Roughly 9–10 ms less KDA kernel time per eight-row verification step; zero reconstruction kernels in the retained MXFP8 traces | Separate lossy change, native Marlin W8A16; not every graph size or API workload becomes faster. |
| Memory versus ModelOpt? | CT MXFP8 allocated 7.447 / 7.430 GiB less per GPU, 14.877 GiB total | Matched settings in the earlier drafter configuration; includes checkpoint-format differences and quantization. Not additive to the 9.367 GiB figure, and not a fresh measurement of the selected Modal profile. |
| Quality? | Full GSM8K 1,248/1,319 (94.62%) → 1,260/1,319 (95.53%) | Same CT before/after MXFP8 KDA, speculation off. The 1e-5 KL gate still fails; one task does not establish general superiority or zero quality loss. |
| Does a pretrained drafter help? | Five drafts: code 44.37, chat 32.80, explanation 34.74 tok/s | +6.48% / +6.91% / +7.72% over bracketed seven-draft medians on eight C1 development prompts. Not a production comparison. |
| Did adaptation justify more training? | No compelling native benefit in the cancelled small pilot | 96 baseline + 32 adapted requests; final reload cancelled. Training stopped, pretrained checkpoint selected. |

The five-draft screen passed its reload timing controls, but all eight output
sequences changed between draft depths. It establishes measured development speed,
not output invariance or task quality. The earlier MXFP8-versus-ModelOpt C1 code
comparison remained 8.61% slower; its larger-concurrency controls failed. Gains
from these different experiments must not be added together.

Repeatability investigations found sparse-MLA ordering, mHC kernel selection and
MoE reduction partitioning could change rounding independently of compression.
Controlled fixes and component fixtures improved repeatability; universal
full-model invariance across batch sizes and draft depths remains unqualified.
The separate NVFP4 scale-compression branch saved about 6.33 GiB across TP2, but
its speed/full-model quality qualification remained unresolved. It is an
alternative branch, not an additional saving in the selected CT configuration.

Evidence: [lossless results](GLM53_LOSSLESS_RUNTIME_HANDOFF.md),
[native MXFP8 speed and memory](GLM53_KDA_MXFP8_SPEED.md),
[full GSM8K](GLM53_KDA_MXFP8_GSM8K.md),
[drafter/depth results](GLM53_PRETRAINED_DRAFTERS.md), and
[NVFP4 branch](GLM53_NVFP4_MODEL_TEST.md).

The latest independent service check retained by the completed work is
21 October 12:55:54 UTC: original ModelOpt/Inco production, 500,000-token context.
The CT profile is selected for its next launch and is not deployed. No training
or GPU benchmark was started for this summary/compatibility audit.

**Easy deployment is now an explicit requirement: pin vLLM 0.32.0 and minimize
the extension, with a reproducible install and launch.** Upstream's
[0.32.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.32.0)
includes GLM-5.3-Flash DFlash2 support and additional MXFP4 backend options.
Its [plugin interface](https://docs.vllm.ai/en/v0.32.0/design/plugin_system/)
provides a route to extend an upstream installation. These are upstream
capabilities, not proof that our complete combination loads or performs correctly.
This audit reads documentation and our integration source; it is not a 0.32.0
build, import test or GPU compatibility qualification.

The local source identifies concrete packaging work:

- `engine_plugin/pyproject.toml` registers `model_plugin:install` but declares
  `py-modules = []`. It is a registration shim relying on external runtime files,
  not a self-contained installable distribution of the full approach.
- `model_plugin.py` wraps internal compressed-tensors preparation and Marlin MoE
  dispatch; `vocab_adapter.py` installs custom compressed vocabulary methods.
  Our packed buffers still need these consumers. Loading expanded standard
  tensors can provide a fallback but forfeits those resident-memory savings.
- `kda_mxfp8_runtime.py` currently converts 68 matrices in memory and pins an
  installed Marlin helper by source hash. The original checkpoint files are
  unchanged. Portable checkpoint/config emission with coverage verification,
  or a packaged and verified load-time conversion, is still required.
- `build_moe.py` adapts pinned Marlin source and requires a compatible CUDA build.
  Rebuild and qualify the extension for the exact upstream Torch/CUDA/SM121 stack;
  retain the source notices already recorded in `CONTRIBUTING.md`.
- Fused-module aliases, pretrained Modal feature taps, FP8 loading, graph capture
  and repeatability hooks must be compared with upstream support individually.
  Remove an overlay only after its native replacement passes the relevant checks.

The practical target is **upstream vLLM plus one versioned extension**, distributed
as a wheel and a ready image based on a pinned official vLLM image. Zero-extension
vLLM cannot be assumed to understand our private packing; that would require
upstream codec/kernel support or expanded tensors. The intended user workflow is
one verified artifact preparation step, one install/pull, and one documented
launch with the saved five-draft config. This workflow has not yet been delivered.

Next work, in order:

1. Establish a clean 0.32.0 CT + pretrained Modal baseline, then package the minimum
   compression hooks and kernels without requiring edits inside the vLLM install.
   Preserve the measured weight rounding and W4A16/W8A16 policies first; do not
   silently substitute a new activation-quantizing backend during migration.
2. Verify fresh installation on both Sparks, checkpoint identities, exact codec
   reconstruction, the 68 KDA rounded-weight hashes, released dense storage,
   actual loaded backend, graph replay and memory with compression on/off.
3. Run the real harness with production-equivalent serving settings and explicit
   KV budgets. Compare against current production separately from the same-CT
   compression toggle. Record TTFT, end-to-end throughput, acceptance, quality,
   actual resident concurrency and repeated timing controls.
4. Qualify mixed-length C8/C16 and progressively longer contexts through the
   required 500k limit before replacement. The existing 8k trials do not cover it.

No new training is needed for this deployment path. New native FP4 backend
experiments, if later chosen, are separate numerical/performance changes.

**21 October 2026, 12:49 UTC: training abandoned at user request; original 500k production healthy.**
CT's selected configuration is **pretrained Modal FP8, five drafts at C1/C2,
three at C3–C16**. The unmodified checkpoint and config verify on both Sparks;
`full-model-v1/ct-serving-current.json` records the choice. Repository vLLM config:
`integrations/glm53_mxfp4/ct_dflash_speculative_config.json`. **This is the CT default
for its next launch, not a production switch.** Live ModelOpt/Inco production keeps
its own existing configuration. No training, capture, benchmark or promotion queued;
sealed test remains closed.

The user-cancelled adaptation comparison stops after **96 baseline + 32 adapted
requests**. The controller restores original identities and 500k context at
**12:46:51 UTC**; an independent authenticated check passes at **12:48:05 UTC**.
All **143 retained payloads** independently verify on Mac and Spark2. Archive
39,362,729 bytes, SHA256 `c07b7222bd792e4d8e1d079653ecbe988cbbdc50fbe140664810772f57434d4b`.
Within each arm, all completed repeats have exact outputs/counters, and all 16
cross-arm output sequences match. Native loaded FC bytes and FP8 configuration
verify on both ranks. The controller labels the cancellation `failed_restored`;
it was an explicit user stop, not a kernel/load failure.

| Native five-draft development bucket | Pretrained accepted/round | Adapted accepted/round | Rounds per full prompt cycle, pretrained → adapted |
| --- | ---: | ---: | ---: |
| Code, 8 prompts | 2.7083 | 2.7151 | 552 → 551 |
| Chat, 4 prompts | 1.7641 | 1.7567 | 373 → 374 |
| Explanation, 4 prompts | 2.1341 | 2.1152 | 328 → 330 |

These partial results add no compelling training benefit. **The final baseline
reload was cancelled**, so no completed speed qualification or generalization
claim is made. The selected checkpoint had used these validation prompts for
selection. The earlier completed depth screen supports the pragmatic change:
five versus seven drafts gained 6.48% code / 6.91% chat / 7.72% explanation domain
median throughput on its eight-prompt C1 development set. Those results belong
to CT + Modal, not production's different Inco drafter; changed-depth outputs
were different, so that screen was not a quality/zero-loss qualification.

Retained evidence: local `out/drafter-modal-native-v1`; Spark
`full-model-v1/drafter-modal-native-v1`; independent Spark2 backup
`full-model-v1/drafter-modal-native-v1-retained-backup`. The cancelled adaptation
checkpoint remains archived and is not selected. Next CT serving work should use
the unmodified five-draft profile; production tuning, concurrency and long-context
qualification remain separate unfinished replacement requirements.


**21 October 2026, 12:46 UTC: user ends adaptation work; pretrained Modal FP8 at five drafts selected for CT.**
The user requests no further training and chooses the draft-length change. The
native comparison was cancelled via its controller at **12:44:11 UTC**, after
**96 original + 32 adapted requests** (two complete adapted cycles). The final
original reload was not run, so this is **not a completed speed qualification**.
All retained partial evidence will be kept. The first cycle showed essentially
flat code acceptance and slightly worse chat/explanation, with all 16 output
sequences equal; this does not justify scaling the pilot.

The CT selection now points to **unmodified pretrained Modal FP8**, SHA256
`45f34ef126aa258d8e16bd605a3a05c36eb3fef3ed0f57aef20191db49f1ef46`.
Native draft policy is **C1:5 / C2:5 / C3–C16:3**. The higher-batch policy is retained;
only the previously seven-draft C1 default changes. Configuration:
`integrations/glm53_mxfp4/ct_dflash_speculative_config.json`. On both Sparks,
`full-model-v1/ct-serving-current.json` records this selection and its pinned
checkpoint; it is a **selected CT configuration, not a production deployment**.
The five-draft original arm already passed native startup and all 96 repeat checks.

Original ModelOpt/Inco **500k production is restoring** through the existing
controller. Its separate drafter/configuration has not been changed. The CT profile
still needs the outstanding production-tuning and long-context qualification before
a full replacement. No further training, capture, GPU experiment or automatic
promotion is queued. Sealed test remains closed. Detached postprocessing only
verifies/restores/backs up the cancelled run; it cannot launch another experiment.


**21 October 2026, 12:43 UTC: first adapted cycle has no useful native gain; provisional only.**
All 16 first-cycle outputs equal original-a. Code accepted tokens per round change
2.70833 → 2.71506; chat 1.76408 → 1.75668; explanation 2.13415 → 2.11515.
Code uses 551 rather than 552 total verification rounds across its eight prompts:
one prompt saves two rounds, another adds one. The offline-improved prompt
`opencoder-evol-56582` remains at 65 native rounds. Chat adds one round and
explanation adds two across their respective four prompts. First-cycle timings
are roughly 0.7–1.2% lower throughput, but repeats/reload control are incomplete:
**no speed conclusion or qualification yet**. This weak signal does not support
scaling training. The frozen run continues, with no additional experiments queued.


**21 October 2026, 12:41 UTC: selected Modal step 16 passes native loading on both ranks.**
Both workers independently verify checkpoint SHA256 `6dca2af520c6df5105eca783a07d1b170ea16ddbcefe31aba6ece54cec741925`,
bit-identical loaded `fc.weight`, six FP8 drafter layers, context width 24,576 and
five native draft tokens. Target graph rows are [4,6,8,12,16] on both ranks.
The adapted arm is measuring its 96 requests; no complete acceptance/speed
comparison yet. No model promotion, further training or sealed test. Original
production is still temporarily paused and will restore after the final original
reload. The detached controller, watchdog and evidence-backup continuation remain active.


**21 October 2026, 12:35 UTC: baseline complete; adapted Modal checkpoint loading.**
Original-a completes all **96/96 requests**, with exact within-arm outputs and
speculative counters. On the expanded 16-prompt development set, baseline domain
medians are **42.433 code / 31.880 chat / 36.011 explanation tok/s**; native accepted
prefix per round is **2.708 / 1.764 / 2.134** at five drafts. These are baseline
numbers only, not a training gain, and should not be directly compared with the
older eight-prompt domain medians. Adapted startup and its 96 requests follow;
original-b reload then checks drift. Original production remains temporarily paused
under the same 13:32:04 UTC outage deadline and automatic restoration controller.


**21 October 2026, 12:23 UTC: first native original arm is measuring.**
CT startup completed normally; the first unmodified Modal FP8 load is executing
its 96 requests at five drafts. Adapted and original-reload arms follow automatically.
No adapted speed or acceptance result yet. Detached postprocessing PID 3835576
waits for original-500k restoration, then verifies the retained evidence and copies
it to Spark2 for an independent payload audit. Local launch audit confirms identical
commands and serving settings across arms. Training/validation have no exact text,
ID or token-hash overlap, but validation selected the checkpoint: generalization
remains unproven. All earlier timing/quality limitations remain in force.


**21 October 2026, 12:17 UTC: native Modal pilot comparison running detached.**
CPU preflight passed on both ranks with 211 runtime bindings. Launcher 3788307 /
controller 3788308 / independent watchdog 3788394 are active. Production is
**temporarily paused** for the original/adapted/original trial; the watchdog outage
deadline is **13:32:04 UTC**, followed by bounded original-500k restoration if needed.
No native result yet. The experiment survives laptop disconnection. Frozen source
commit `3f1b2d7` is verified in the Spark2 backup repository; full pytest, Ruff and
strict core mypy pass. This remains a development-transfer test on the 16 prompts
used for checkpoint selection, not a generalization or sealed-quality claim.


**21 October 2026, 12:16 UTC: native Modal pilot comparison authorized and preparing.**
The selected step-16 drafter will be compared with its unmodified Modal FP8 source
on the same CT target, with **five drafts in every arm**, original/adapted/original
loads, all **16 pilot-validation prompts**, six seed-42 shuffled cycles, and 256
output tokens (**288 measured requests**). Candidate bytes verify on both Sparks;
CPU serving-image preflight is running. No GPU trial has started at this checkpoint.
Private API 8044 / master 29717; independent watchdog bounds outage to 75 minutes,
then restores the original 500k production containers. No deployment or full training.

**These validation prompts selected the checkpoint.** This experiment checks whether
the offline signal transfers to native serving; it does **not** establish generalization
to new prompts. Keep the sealed set closed. A useful native result still needs an
unseen-prompt gate before broader claims. Record all output differences, acceptance,
TTFT, wall time, graph replay evidence and clock/thermal telemetry. Frozen development
criteria: paired geometric-mean code throughput gain at least 8%, chat/explanation
regression no more than 2%, exact within-arm/reloaded-original outputs and counters,
and at most 2% reload timing drift, including each prompt.

Evidence root: `out/drafter-modal-native-v1`; remote `full-model-v1/drafter-modal-native-v1`.
Frozen plan SHA256: `6a03ea86ee3d7c7ddd92c81328a7f491de22959d3e25c53e6e4e1303c9770e18`.
Estimated total serving check: 45–60 minutes including loads/restoration, subject to startup.


**21 October 2026, 12:05 UTC: short Modal pilot complete; production restored.**
64 updates take **4m 20s**. Selected step 16 raises offline code prefix from
4.1875 to 4.375 (+4.48%); chat/explanation prefixes are unchanged. The gain comes
from one of eight code prompts, one anchor, so no native speed benefit is proven.
All 84 requests/control checks and 878 million independent feature-bit comparisons
pass. Native FP8 export changes only `fc.weight` and is not deployed. Original
500k production is healthy; no GPU trial or further training queued, sealed test
closed. [Full evidence, limitations and capture-concurrency audit](GLM53_HANDOFF.md).

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

**21 October 2026, 11:37 UTC: bounded Modal pilot preparing.** The user authorizes a short
64-train/16-validation, FC-only pilot; full training remains held. Both image CPU
checks and Modal round-trip/mask/gradient checks pass. Production is still online;
no GPU capture yet. [Protocol and capture-concurrency limits](GLM53_HANDOFF.md).

**21 October 2026, 08:35 UTC: CT draft-length screen complete and independently verified.**
Original production restores healthy at **08:32:34 UTC**, with its exact original
container/configuration identities and **500,000-token context**. All 240 measured
requests complete; training stays held, sealed test closed, no candidate promoted
and no further GPU experiment queued.

Same CT target and pretrained Modal FP8 drafter, C1, eight development prompts,
six fixed-seed cycles, 256 output tokens. These are end-to-end request medians:

| Native drafts | Code tok/s | Chat tok/s | Explanation tok/s |
| --- | ---: | ---: | ---: |
| 7, initial | 41.73 | 30.68 | 32.27 |
| 3 | 41.06 | 34.91 | 38.62 |
| 5 | 44.37 | 32.80 | 34.74 |
| Off, drafter unloaded | 22.22 | 22.39 | 22.38 |
| 7, final reload | 41.62 | 30.68 | 32.23 |

Against the mean of the two seven-draft domain medians, **five drafts gains
6.48% code, 6.91% chat and 7.72% explanation**. Three drafts changes code by
-1.48%, but gains **13.81% chat and 19.75% explanation**. Five improves all four
code per-prompt medians; the equal-prompt geometric-mean code gain is **5.45%**.
Three's corresponding code mean gains 2.29%, with one code prompt regressing
substantially: the domain median alone does not describe every prompt.

All within-arm outputs/counters repeat exactly. The final seven-draft load
reproduces all baseline outputs/counters, and reload drift **-0.28%/+0.01%/-0.11%**
passes the frozen 2% gate in every domain. **All eight outputs differ when depth
changes to 3, 5 or off.** Changed trajectories can affect timing and acceptance;
this small development screen is not a task-quality, KL, zero-loss or production
replacement qualification. No target weights, compression settings or training
updates change during the comparison. The pretrained drafter is already useful:
five drafts gives about **2x code throughput** versus speculation-off, while three
gives 1.56x chat and 1.73x explanation throughput.

All 152 retained payloads verify on Mac/Spark2. Five-draft code first-token
latency is 266 ms, about 21 ms above the bracketed seven-draft baseline.
Keep training held; profile CT serving overhead and compatible production tuning
before a broader qualification. [Full evidence, limits and next step](GLM53_HANDOFF.md).

**21 October 2026, 07:29 UTC: CT draft-length comparison running.**
All ten CPU preflights pass. Detached controller 1622330 starts CT + Modal FP8
at seven drafts; 3/5/off/7 follow. Production is temporarily paused under
watchdog 1622434; original 500k containers restore automatically. No result yet,
training held and sealed test closed. [Live status and limits](GLM53_HANDOFF.md).

**21 October 2026, 07:26 UTC: CT-only draft-length screen preparing.**
The user redirects the next test from production to CT and authorizes unattended
work. Same CT + Modal FP8 at 7/3/5/off/7 drafts, eight development prompts and six
cycles per load. CPU checks run on both nodes; production is still online. The
GPU comparison will use the existing independent restoration watchdog. No
training, sealed test or promotion. [Current plan and status](GLM53_HANDOFF.md).

**21 October 2026, 07:16 UTC: current production has similarly modest acceptance.**
Same eight development prompts, C1/256 tokens/seven drafts, two production cycles:
code/chat/explanation accepted-proposal rates **34.82%/23.36%/27.51%**; CT + Modal
FP8 previously measured **42.03%/27.60%/30.21%**. Production remains faster in
these descriptive timings, despite lower aggregate acceptance. Different targets,
outputs, settings and loads prevent causal attribution. All eight production
outputs/counters differ across fixed-seed cycles; CT's six within-load cycles
were exact. No serving changes: original 500k production remains healthy.
The CPU cache audit passes 200 carry checks and narrows native divergence to
at or before layer 4 pre-convolution QKV at position 92; no inspected rollback
metadata error. Follow-up GPU trace not queued; training held, sealed closed.
[Measurements, limits and next trace](GLM53_HANDOFF.md).

**21 October 2026, 06:56 UTC: paired trace reviewed; production restored.**
All eight controls and both GPUs' capture repeats pass; all 406 retained payloads
verify on Mac/Spark2. Target logits first differ at output 38, text at output 48.
Both arms select their own target argmax. The last common observed equal layer 0/4
recurrent-state boundary is position 87; by 94 layer 4 differs while layer 0 remains
exact. This narrows the next cache/arithmetic investigation to the intervening
steps; a specific kernel or compression fault is not proved. Original 500k
production is healthy. No new GPU experiment is queued; training held, sealed
closed. [Result and next targeted check](GLM53_HANDOFF.md).

**21 October 2026, 06:41 UTC: first trace arm complete, second loading.**
All four Inco controls reproduce the frozen 64-token output and matching
speculative counters. Both ranks retain 34 frames each. Modal BF16 is now loading;
paired tensor review and root-cause analysis are pending. Production is temporarily
paused under the restoration watchdog. [Latest evidence](GLM53_HANDOFF.md).

**21 October 2026, 06:35 UTC: speed baseline repeat stopped; targeted trace loads first arm.**
All three first loads complete 48 requests with exact within-arm repeats. Modal
FP8 matches Modal BF16 on all 48 outputs; both differ from Inco on seven of eight
cases. The final Inco reload is deliberately cancelled to prioritize this mismatch;
no speed gain is qualified. All 132 retained timing payloads independently verify.
Original 500k production restores healthy at 06:32:09 UTC. Detached replacement
launcher PID 1180511 passes CPU preflight; controller 1186244 now loads the
first arm under watchdog 1186334. Production is temporarily paused again;
no GPU trace result yet. Read `after-priority-status.json`; old queue PID 987188
is stopped. Training remains stopped and sealed test closed.
[Evidence and current handoff](GLM53_HANDOFF.md).

**21 October 2026, 06:19 UTC: targeted native-decision trace queued.**
Detached Spark PID 987188 waits for the current timing run and verified production
restoration. It then runs one 64-token development case under both drafters with
before/capture/repeat/after controls and a separate 30-minute restoration bound.
Both hosts verify 46 frozen sources; full tests, Ruff and mypy pass. BF16 completes
48/48 requests with exact repeats, code +2.39% versus the first baseline, but
7/8 outputs differ: timing gain remains unqualified. FP8 is starting; baseline
reload follows. Capture/training stopped, sealed test closed.
[Current diagnostic and evidence](GLM53_HANDOFF.md).

**21 October 2026, 06:09 UTC: Modal BF16 serving works, but output identity fails.**
Baseline completes 48/48 requests with exact repeats; Modal BF16 completes 16/48.
First-round code speed is about +3%, but 7/8 outputs differ across drafters,
confounding the speed comparison. The completed Modal second round repeats its
own outputs/counters exactly. No candidate qualification or quality verdict;
FP8 and baseline reload are pending. Target remains CT with MXFP8 KDA. Capture
and training are stopped; production temporarily paused under the restoring
controller. [Evidence and limits](GLM53_HANDOFF.md).

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

**Completed CPU pilot, 20 October 2026:** 32 updates in **177.386 seconds**;
step 24 improves offline mean accepted prefix from **4.25 to 4.375** on code,
**2.75 to 2.875** on chat, with explanation unchanged. Step 32 fails chat and is
rejected. Independent review reproduces 1,120 token comparisons, 135 acceptance
fields and selection. Export changes only FC; 115 other tensor payloads and the
config are byte-identical. This small offline result is superseded for serving
claims by the failed native equivalence check above. Prior failures are retained.
All 44 feature-capture/control requests and 5,456 payload hashes verify; the
independent audit finds zero differences across 336,281,600 BF16 feature elements.

Earlier dated checkpoints below are historical and superseded by the result above.

**Earlier 20 October 2026 CPU preparation complete:** frozen thresholds; exact full-file
FP8/BF16/FP8 round trip; actual SpecForge class strictly loads all 81 dense
parameter tensors; 4,000/200/200 train/validation/sealed-test prompts and token
IDs prepared. Production remains healthy at 500k context, checked 13:21:47 UTC.
No GPU training or native forward parity has run. See
[the preflight report](GLM53_DRAFTER_PREFLIGHT.md) for evidence and remaining work.

**Variant clarification, 20 October 2026:** original lossless CT preserves MXFP4
experts and BF16 KDA values. The later CT MXFP8 candidate additionally quantizes
KDA from BF16 to MXFP8 at runtime; source files remain unchanged. The earlier
[same-CT compression off/on test](GLM53_MXFP4_SERVING_V3.md) matches all 24 paired
token sequences and measured speculation counters. Later CT-versus-ModelOpt
feature differences do not isolate a compression effect. See the
[handoff variant table](GLM53_HANDOFF.md) before interpreting drafter adaptation.

**Current continuation:** preserve the fixed CT MXFP8 target and trace the first
candidate/original verification divergence on identical histories. Training,
export and native checkpoint loading now work; broader training and the sealed
serving test wait for this numerical diagnosis. The user goal remains a CT setup
with a properly adapted drafter. Original CT remains the lossless reference;
the previous MXFP8 cross-target KL failure and production replacement gates stay
unchanged. No new target quantization or production switch is part of this check.

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

**Current service, 20 October 2026, 04:46 UTC:** original production is restored
on authenticated port 8000 with **500,000 context**. Both original identities
and configuration hashes match the saved record, and both CT trial containers
are stopped. The [restoration receipt](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/production-restore-20261020-status.json)
and [local review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/production-restore-20261020-local-review.json)
supersede earlier service-state entries below.

**Why the NVFP4 branch exists:** the [MXFP4/DFlash diagnostic](GLM53_DFLASH_C1.md)
found faster same-checkpoint decode (short C1 time −5.34%) but +52.3 ms TTFT;
64-token end-to-end throughput was essentially unchanged (−0.32%). CT's BF16
KDA backbone also costs more decode time than production's MXFP8 path. NVFP4
scale compression applies the storage technique to the original production
checkpoint while preserving its backbone. It verifies 3.16 GiB/GPU allocation
saving, but has no qualified serving-speed or full-model numerical result.

No new GPU experiment or trace implementation has started. The user is
reviewing that direction. The requested known-mHC test, speculation-off or
matched-layout quality comparison, and six-plus interleaved speed trials with
telemetry remain unrun. Future experiment rollback must target original
production. See the [current handoff](GLM53_HANDOFF.md) for exact pending scope
and the untested selective dense-KDA MXFP4 alternative.

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

Updated **19 October 2026, 07:43 UTC**. All current measurements, independent reviews and
retention are finished. [Handoff](GLM53_HANDOFF.md) / [full result](GLM53_FP32_INPUT_MODEL.md).

**Keep the preceding warmed exact16 candidate. The new input-FP32 route fails
C16 qualification:** KL **1.601802484e-5 > 1e-5**, reproducible in all ten C16
cross-arm comparisons. C1/C8 pass; all within-arm repeats are exact and all 300
timed output pairs match. This is an arithmetic difference between candidates,
not the earlier repeatability noise. Spark/Mac agree after independently checking
1,634 payloads / 229 sources / 12 traces. No configuration is promoted.

The failed new candidate measures C1 +12.24-14.78% throughput, C8/C16 -2.03%
to +3.13%, with the same **9.367162 GiB TP2 CUDA saving**. Short-C1 TTFT rises
**41 ms**, short-C16 **132 ms**; the latter remains admission-confounded.
These latency numbers do not replace the qualified fallback's +38 ms / +442 ms.
Mixed KDA kernel sums improve modestly to ~20-21 ms, while expert work remains
~60-61 ms at C8 / ~97-98 ms at C16. No CPU-memory gain is established.

The original GLM is restored healthy. Controllers, recovery and Mac retention
have finished; the owned awake lease is released. An offline checker correction
accounts for split-2 grid geometry (394 blocks, not 197); all original sources,
failed logs and numerical/performance gates remain preserved. No GPU rerun.

The qualified fallback below still passes all 46 numerical checks, gives C1
+11.82-13.28%, near-parity C8/C16 and the same CUDA saving. This is finite
qualification, not universal zero-loss inference. Its original speed/TTFT gates
remain failed. User scope ends further compression-kernel optimization; next
proposed test is **3 versus 10 ms admission** with the fallback and short-C1
controls, then MTP. Both are unlaunched. Three local admission modules remain
unfinished/unvalidated; the broader goal is still paused pending the next phase.

Historical checkpoints below retain their original evidence and failures;
forward-looking statements in them are superseded by this snapshot.

**18 October 2026, 22:50 UTC integration follow-up:** input-only FP32 split 2 passes
the independently verified GPU integration fixture (**434 payloads / 227
sources**): 120 retained-output checks, 66 native-fallback checks, compiled
captures and six two-stream stress fixtures. The existing pool is reused with
zero retained replay growth; a third stream is rejected. Both GPU jobs, retention
and exact original-service restoration are complete; the awake lease is released.
This adds no full-model quality/speed or CPU-saving result. The 216 prior exact-word
failures and two input timing repeat failures remain failed. Fresh matched model
qualification is next, after evidence-storage preparation. See the [handoff](GLM53_HANDOFF.md).

**18 October 2026, 22:31 UTC speed follow-up:** FP32 timing independently verifies on
Spark/Mac (**246 payloads / 220 sources / 884,736 events**). Input split 2 saves
about **7-8% of component time** (46/48 strict timing conditions pass; two M8
pressure repeat controls fail). Output gains shrink substantially under pressure;
the broad screen fails 78/288 conditions. Prior 216 exact-word failures remain
failed. No new full-model quality/speed or CPU-saving claim is made. GLM is
restored healthy, all jobs/retention complete and the owned awake lease released.
Next is a bounded input-only split-2 model probe under unchanged KL/NLL limits.
See the [handoff](GLM53_HANDOFF.md) for measurements, controls and provenance.

**18 October 22:12 UTC precision follow-up:** FP32 split partials are independently
verified on Spark/Mac (2,227 payloads / 217 sources). All 576 comparisons pass
the 0.001 relative-error bound; all 144 basis cases are exact. However, all 216
random M4/M8 cases fail the retained exact-word gate, despite about 22-26 times
lower relative error than the old BF16-partial route. Source/storage, guards,
repeats and graph checks pass. No speed, model-KL or CPU-saving result is measured,
and no route is promoted. GLM is restored healthy; all jobs and retention are
complete. Next comes paired component timing, then any justified exploratory
model test under the unchanged KL/NLL limits. See the [handoff](GLM53_HANDOFF.md).

**18 October 21:52 UTC follow-up:** dense-output replacement is rejected by its
frozen component screen. Spark/Mac independently verify **270 payloads / 213
sources / 221,184 timing samples**. All six matrices pass storage/execution and
numerical checks, but warm-cache C1 adds **56-68 us per projection**: 12 stable
cost failures plus six separately unstable conditions, all retained. Prefill
saves about **205-284 us/projection** in graph + pressure measurements; these
are not serving-speed results. No dense-only policy is integrated. The warmed
exact16 candidate below remains current. GLM is restored healthy, all jobs and
retention are complete, and the owned awake lease is released. An FP32 split-K
partial-storage idea is source-audited only; no new kernel result exists.
See the [handoff](GLM53_HANDOFF.md) for evidence and limitations.

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

Startup warming moves the identified first-use dispatch compilation out of the
measured requests: all 34 variants remain cached, and no recognized compiler
log event overlaps the 72 timed groups. The prior 2.14 s short-C8 gap is not
reproduced (current candidate maximum 0.448 s). This is finite evidence, not a
guarantee against all stalls. The 31.3-31.6 s startup warmup leaves all 576
recorded full-vocabulary rows per arm bitwise equal to its previous exact16 run.

All candidate physical-8/16 decode ranges replay graphs with 68 fused KDA calls
and zero reconstruction. KDA is about 22 ms/rank; mixed C8/C16 expert GEMMs are
about 61 / 96-99 ms. Prefill still reconstructs 68 matrices in 35.8-38.0 ms/rank.
Short-C16 median TTFT rises 24.04%; differing admission is a confounder, and
mean request TTFT rises 6.11%, but the original failed median gate remains.
Profiles are separate from ordinary timing; kernel sums are not API latency.

The original service is restored with exact identities/configurations and health
200, rechecked at 21:08:26 UTC. Model jobs, recovery and Mac retention have
finished; final latency/resource/trace and both-arm before/after analyses pass.
A separately retained offline verifier correction fixes HTTP 207 to 200 only;
the failed original verifier/logs and all measurements/gates remain preserved.
The archive SHA256 is
`0fc867240d04e1770e2970143f48f630551feb35f53b739403c4fb6b8bde64e0`.
No configuration is promoted. Current PPL, task accuracy and SQNR are unmeasured.
Local runtime checks remain 728 tests passed, one GPU skip, Ruff and strict mypy.

**Next:** keep this as the experimental C1/GPU-capacity candidate. For further
concurrent speed, investigate the measured prefill reconstruction/admission cost
with matched grouping and a bounded component test. CPU-memory savings remain
unproven; broad qualification and the requested roofline follow-up remain open.
MTP stays deferred. No further GPU experiment was running at that 21:10 UTC
checkpoint; the later input-only FP32 model run is active as recorded above.

## Historical matched C1/C8 result

**The fresh stock/compressed C1/C8 pair passes all 33 numerical comparisons**
under the original KL <=1e-5 / absolute mean NLL <=1e-4 gates. Actual grouping
matches in every pair. All eight self-repeats are exact, with native mHC and
stable sparse compaction in both arms. Cross-arm decode KL is 5.804e-6 at C1
and 1.194e-6 at C8. This supports the tested finite compression comparison;
inference is not universally bitwise identical.

**Speed remains incomplete:** C1 +12.38% throughput, short-C8 +2.40%, long-C8
-0.61%; C1 first-token latency +7.91%. Six ordinary trials/case/arm are retained,
including a slow split-admission C8 trial. The overall speed gate fails.
CUDA allocation falls **9.367162 GiB across TP2**; reserved CUDA also falls.
Initial host RSS is slightly higher (12.652 -> 12.894 GiB across ranks).
Do not add host/container and CUDA counters on unified-memory Spark.

All 1,168 payloads, 194 source bindings and eight traces independently verify
on Spark/Mac. Decisions agree; maximum CPU float64 difference is 1.776e-15.
All 102 timed paired output sequences match. Exact original service identities,
configs/mounts and health 200 were restored after that completed pair.
Ruff, strict mypy and full pytest pass (one GPU skip). No configuration is promoted.

Fresh C8 trace analysis is complete: decode has no KDA reconstruction but
spends 21.9-23.9 ms in compressed KDA and about 56 ms in expert GEMM kernel sums;
prefill reconstructs 68 matrices in about 36 ms/forward/rank. These sums are not
API latency or measured bandwidth.

The user accepts an understood compression cost and prioritizes investigating
abnormal latency. A new audit of all 13,056 timed token arrivals finds C1's
31 ms first-token penalty is recovered by faster subsequent tokens. Long-C8
has the same early stall pattern in stock and compressed trials, consistent
with prompt/decode interference plus repeated reconstruction overhead.
The original short-C8 12.052 s outlier had six requests admitted
first, two later, and six streams have a ~2.14 s gap. It raises mean request
completion from 8.408 s stock to 8.787 s compressed despite better median group
time. The focused CPU/GPU replay now reproduces multi-second first-use JIT
stalls in both arms. This strongly supports a fixable startup cause; attribution
of that exact older trial remains an inference because it had no simultaneous
trace. See the [latency audit](GLM53_C8_TRACE.md#latency-cost-and-the-remaining-outlier).

The exact16 input/output full-model follow-up is complete and independently
verified on Spark/Mac: **1,581 payloads / 200 sources / eight traces**. All 46
C1/C8/C16 numerical comparisons pass, twelve self comparisons are exact, and
all 192 timed paired output sequences match. The original service is restored
with exact configuration/identity checks and health 200. C16 short throughput
is **+0.48%**, long throughput **-1.40%**; short TTFT rises **1.623 -> 2.211 s
(+36.22%)**, long TTFT **+2.98%**. The original speed gates fail. Allocation
saving remains 9.367162 GiB/TP2. All retained physical-16 decode ranges replay
graphs, and candidate ranges have 68 compressed KDA calls with zero weight
reconstruction. The fallback is fixed; the first-token penalty remains open.
The C16 arrival audit now covers all 24,576 arrivals: short-case initial batch
sizes differ between arms, and both have a slow five-request start. This makes
admission a concrete confounder, not a proven full explanation. The focused
C8 6+2 replay launched at 19:11:49 UTC after both real CLI/source/telemetry
preflights pass. Both arms, restoration and independent evidence verification
are now complete. It retains CPU/GPU timelines and 100 ms clock/thermal samples.
No configuration is promoted.
Native mHC is retained; the optional consistent policy is not required.
MTP remains deferred.
Read [the completed quality evidence](GLM53_C8_TRACE.md) and the
[compact review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/matched-c8-v1-review.json).

**Historical12:18 UTC compression checkpoint:** The output-only validation continuation is
complete and independently verified:1,293 payloads/180 bindings/16 traces.
Exact original service identities/configs/mounts and health200 restored at
12:14:05 UTC; no experiment containers remain running. No further test is started.

**User priority: finish compression at C8/C16; defer MTP until a working
compression outcome.** The [matched C8/C16 comparison](GLM53_HIGH_CONCURRENCY.md)
is complete and independently verified after fixing its startup scratch
ownership failure. It still **fails both speed and numerical gates**:
C1 gains12.3–12.6%, C8 is near parity, and C16 loses7.5–13.8% throughput.
Memory saving remains9.367GiB/TP2. All24 numerical comparisons fail KL; four
also fail NLL, despite all636 timed paired token sequences matching.

Both-rank profiles identify68 weight reconstructions per retained C16 decode
forward, taking34–37ms. The existing eight-row fused limit causes that path.
A focused exact16-row component check has finished GPU timing. Both fixture
outputs are exact, but only the output matrix passes every timing gate; an
input-matrix reference repeat misses the unchanged2% limit. All45 payloads
and18,432 events independently verify on Spark/Mac, with identical reviews.
No global router change is qualified. The
[output-only exact16 follow-up](GLM53_OUTPUT_ROW16.md) completes all72 timing
groups, but stops during numerical group6 because the head node runs out of
disk space. Five groups complete; no candidate profiles run. ShortC16 improves3.4–4.0% over
the compressed parent, longC16 improves1.3–1.5%; both remain below stock and
the original speed gates fail. Recovery is independently verified on Spark/Mac:
1,532 payloads, all72 timing groups/636 requests/40,704 tokens, and all17
available complete numerical pairs. Every available pair fails the numerical
gate in the original failed run. Fresh validation on a separate load has now
completed all8 groups and4 profile sessions: all24 KL comparisons fail and8
fail NLL. The original timing load's missing source-after check remains explicit.
The input route remains unchanged.
The full-model outcome reuses stock control with
an explicit time gap. All1,277 payloads and sixteen worker traces verify on
Spark/Mac. The original service is exactly restored and healthy after the
output-only follow-up fails; the client is not OOM-killed. Disk recovery
preserves canonical archives before removing regenerable extractions. Another
8,950 older generated payloads were checksum-verified on the existing worker
before relocating7,593,681,896 bytes; model checkpoints remain untouched.
The head now has about5.63GB free after retaining the completed continuation. No configuration
is promoted and throughput is not declared fixed.

## Evidence inventory, including historical experiments

| Area | State | What the evidence supports |
|---|---|---|
| Weight storage | Verified | 9.367 GiB / 5.33% less allocated CUDA memory across TP2 in the completed full-model runs; source reconstruction checks pass. |
| Full GSM8K | Complete | 1,319 questions/arm; numeric 1,246 → 1,244, output TPS 41.77 → 42.08, workload time 37m41s → 37m37s. Roughly equal workload speed. |
| Numerical qualification | Finite C1/C8/C16 compression gates passed | Fresh matched native-mHC C1/C8 pair passes all 33 comparisons; exact16 follow-up passes all 46 C1/C8/C16 comparisons. C1 KL 5.804e-6, C8 1.194e-6 in the C8 pair. Universal serving equivalence is not established; historical failures retained. |
| First numerical divergence | Narrowed | Actual graph captures first differ inside layer 3's inner MLA module, before MoE, with recorded module inputs equal. The earliest recorded change is sparse-index compaction order with equal selected multisets; other sources remain possible. |
| Actual MLA internal probe | Complete, independently verified | 40 graph calls / 1,606 payloads; 25/32 noninitial calls change sparse order with equal multisets. Padding fixture passes; stock C1 repeats become exact, while stock C8 still fails KL and differs already at prefill. |
| Isolated attention fixture | Complete | All six fresh-process cases / thirty graph calls repeat exactly and match eager. The synthetic fixture excludes the full module's projections, cache writes and layer interactions. |
| Matched speed comparison | Complete and independently verified | C1 throughput +10.4–12.4%; C4/C8 −0.07% to +1.84%; TTFT +2.3–5.9%. All 373 payloads, 260 requests and 16,640 tokens verify on Spark/Mac; three analysis JSON files match bytes. |
| Prefill reconstruction | V3 independently verified; no accepted speed improvement | All 36 outer / 13 native payloads and 6,656 event samples verify on Spark/Mac. Both direct-output variants and rebuilt-control parity fail the performance gates. No full-model integration. |
| Exponent-base correctness | CPU audit and V4 GPU regression independently verified | No unsafe bases across 136 KDA shards, two heads or 154,880 embedding rows. Both fixed candidate kernels pass the original base-255 failure and all BF16 bit patterns in eager, separate-stream and graph execution. Both stock control guards refuse unsafe bases. All 34 outer / 13 native V4 payloads verify on Spark/Mac. |
| Small-prefill routing | V5/V6 independently verified | All numerical checks pass across 51,200 retained events. Large gains at small rows; V6 shared component leads [8,24]. Control variance prevents a blanket crossover claim; 257 and 2,048 rows regress. Compressed candidate's eight-row limit unchanged. |
| CUDA graph expansion | Complete and independently verified; bundle fails gates | Exact C2: +8.6–9.0% TPS; graph C8: +1.7–3.4%. All 72 groups / 300 requests / 19,200 tokens and eight worker profiles verify. Short-C8 misses 5% gain, C4 TTFT regresses 34.8%; both numerical repeats fail KL. All 150 paired output sequences match. No configuration promoted; C16 remains separate. |
| Admission / repeated prefill | Complete and independently verified; performance gates pass | 3/10 ms idle-only collection raises short-C4 TPS 6.75%/6.62%; TTFT 1.125 → 0.778/0.782 s. Median large forwards two → one, device reconstructions 136 → 68/rank. All 599 payloads, 108 groups / 450 requests / 28,800 tokens verify; all 300 paired sequence comparisons match. Every numerical comparison still fails KL, including all self-repeats. No promotion. |
| Matched C8/C16 compression | Complete and independently verified; speed/numerical gates fail | C1+12.3–12.6%; short C8+2.2–2.5%, long C8−0.3–0.7%; short C16−13.6–13.8%, long C16−7.5–7.6%. All144 groups/1,272 requests/81,408 tokens,1,277 payloads and sixteen worker traces verify. Stock control reused with time gap. All24 KL comparisons fail; four NLL failures. |
| C16 startup / repeated reconstruction | Startup repaired; speed cause measured | Captures through16 and fixed two-slot ownership pass292 worker observations. Eight retained parent pure-decode ranges across TP2 each reconstruct68 matrices in34–37ms. V7 independently verifies45 payloads/18,432 events: exact outputs, output matrix passes at16, input repeat control fails. Output-only follow-up recovers part of the C16 loss; its execution mechanism still needs profiles. |
| Native MTP compatibility | Deferred by user until compression outcome | Actual-geometry CPU proof verifies complete q/KV source bits with zero padding. The repaired 18-check full-pilot preparer/verifier is frozen and native CPU preflight passes with 174 source bindings. No new MTP GPU controller, acceptance or speed result. |
| New no-drafter control | Complete, independently verified | C1 P256/P2048 medians 16.768/13.961 TPS. Ten trials, 640 timed tokens; all source checks pass. Repeat IDs/conditioning match, but mean decode KL 1.1901962e-5 fails the unchanged 1e-5 floor; mean NLL delta 2.29481e-5 is below 1e-4. |
| Output-only row16 route | Complete, independently verified; gates fail | Original timings preserved; fresh validation archive verifies1,293 payloads/180 bindings/16 traces. ShortC16+3.4–4.0% and longC16+1.3–1.5% over parent, still below stock. All24 KL comparisons fail;8 NLL failures. Eight retained TP2 decode ranges reconstruct34 matrices in25.68–28.11ms. Separate-load/source-after gap remains explicit. |
| Experiment storage | Recovered; additional diagnostic budget unavailable | About5.63GB free after retention. Two regenerable extractions and7.59GB of checksum-backed older generated payloads relocated; worker backups and manifests retained. Checkpoints/global caches untouched. No additional cache cleanup executed. MLA requires12GiB free/node before launch. |
| Exact16 continuation | Complete and independently verified | All 46 numerical comparisons pass; 9.367 GiB CUDA allocation saved/TP2. C16 throughput +0.48% short / -1.40% long; TTFT +36.22% short / +2.98% long. Retained physical-16 decode replays graphs with zero KDA reconstruction. Original speed gates fail. |
| Multi-second latency stalls | Warmup validated in the full model | All 34 variants stay cached through quality/timing/profiles; no recognized JIT event overlaps timed requests, and the prior 2.14 s short-C8 gap does not recur. Startup takes 31.3-31.6 s; all 576 old/new output rows per arm are bitwise equal. |
| Original GLM service | Restored after completed latency replay | Exact identities/configs/mounts and health 200 verified at 19:33:58 UTC. Controllers, collector and watchdog exited; no configuration promoted. |
| Roofline figure | Deferred | Resume after the model work, as requested. No paired roofline collection has started; PNG/SVG links remain outstanding. |

The verified storage result does not imply the same reduction in per-step
memory traffic. Every token uses a subset of stored experts, and compression
also adds decoding work. The latest matched measurement confirms a useful
single-request benefit and near-parity concurrent throughput. A separate C4
profile has about 8% faster decode forwards and 6.6–6.7% slower prefill;
68 explicit KDA reconstructions add about 36.26 ms per profiled prefill.
Slow short-C4 trials correlate with split/mixed admission. Read the
[complete paired result](GLM53_PAIRED_SPEED.md) before interpreting those
profiles as API speed. Quality diagnostics have narrowed a separate problem;
they have not themselves accelerated generation.

The completed admission intervention now directly reduces the repeated
prefill work: separate profiles observe each of 68 KDA weights once per
forward, with two forwards at zero wait and one at either wait window. Device
kernel counts agree on both ranks. The improvement is for finite 64-output
request groups; sustained decode-only speed and continuous active traffic
were not measured. Graph and admission policies have not been combined in a
measured run. Both are serving techniques that can generalize, while this
integration and its measured settings remain GLM/Spark/runtime-specific.

## Matched stock-versus-compressed measurement

The reference and current compressed candidate use the same imported MXFP4
checkpoint, qualified native libraries, TP2 transport, 4 GiB KV/rank and graph
captures **[1,4]**. There is no speculative drafter. C8 stays eager.

- Six workloads: concurrency 1/4/8 × 256/2,048 prompt tokens.
- Five measured groups per workload and arm, exactly 64 generated tokens,
  with two 16-token warmup groups per workload.
- Retain every request, generated ID and streaming arrival. Compare total
  throughput, first-token latency, request latency and client decode cadence.
- One controlled C4 / 256-prompt / four-output-token profile per arm follows
  **all** unprofiled timing. Verify both-rank graph launches and full forwards.
- Source checks before/after each arm; exact original service restoration;
  then independent archive and raw-timing verification on Spark and Mac.

This is a performance measurement, not another numerical qualification or
GSM8K rerun. Arms run sequentially, so small differences retain order/clock
uncertainty. Matching shapes do not prove matching routed activations.

## Next steps

0. **C8/C16 compression:** warmedmodelv1 is complete and independently verified.
   Exact16 decode reconstruction is gone and finite C1/C8/C16 quality passes.
   C1 gains 11.82-13.28%, C8 0.13-2.97%, C16 -0.41 to +0.29%; the original
   overall speed gate fails. All arrivals and all 12 worker traces are analyzed.
   Next target prefill reconstruction/admission with controlled grouping and a
   bounded component test before another model run. Preserve the C1 gain and
   current memory bound; earlier failed kernels and all timing outliers remain.
1. **Repeatability scope:** stable compaction plus matched native-mHC grouping
   passes C1/C8/C16 self-repeats in both arms. Arbitrary batch invariance is a separate
   serving property. The optional consistent-mHC component passes its fixture,
   but is not enabled or required for the compression comparison. Investigate
   new numerical causes only if a newly controlled comparison actually fails.
2. **MTP: explicitly deferred by the user until compression has a working
   outcome.** The [actual zero-rope CPU preflight](GLM53_MTP_PILOT.md) passes.
   The full pilot preparation/verifier is now frozen around 18 checks and zero
   padding, and native CPU preparation passes. No new GPU attempt is started. Keep
   independent source CT policy, the full 1,756-source consumption guard,
   42-layer base scale ownership and stock draft GEMMs. Then measure acceptance,
   verification cost, memory, state rollback and API speed. No MTP gain has
   been measured, and the completed control's KL failure remains.
3. **Serving-policy follow-up:** the [admission test](GLM53_PREFILL_ADMISSION.md)
   passes performance and mechanism gates but fails numerical qualification.
   Three milliseconds gives the primary gain with a smaller wait; ten also
   helps the tested staggered cohort. After numerical work, qualify continuous
   arrivals/longer outputs and matched stock-versus-compressed policies before
   promotion. Long C4 still needs four chunks at the 2,048-token limit.
4. **Graph follow-up:** the [C2/C8 test](GLM53_GRAPH_EXPANSION.md) is complete.
   Exact C2 is a measured throughput lead requiring dedicated C2 probability
   and latency qualification. C8 replay works but gives a modest gain.
   C16's matched max-sequences16 test and KDA reconstruction profiles are now
   complete. Sustained fully active traffic and maximum-context capacity remain
   unqualified. Its startup repair does not resolve the measured speed loss.
5. **Remaining decode work:** target expert GEMMs or compressed KDA only with a
   new profiled mechanism. Decoder occupancy, N8, split-K and naive overlap
   variants already failed their relevant gates; do not rerun unchanged.
   Further expert coding has weak headroom (sampled 3.915/4 bits, 1.65% encoded
   saving). FP8 KV is already on. Lossy FP8 KDA remains separate: approximately
   0.977 / 0.976 GB additional weight saving per GPU before scales, with native
   compute support and numerical qualification still required.
6. After model work, collect the paired roofline evidence and provide the
   requested image links. Do not substitute estimated LPDDR bytes for
   hardware-measured traffic.

## Evidence and continuation

- [Current matched C8/C16 compression protocol and live-state paths](GLM53_HIGH_CONCURRENCY.md)
- [Active output-only exact16 follow-up](GLM53_OUTPUT_ROW16.md)
- [Full GSM8K result](GLM53_GSM8K_FULL_BENCHMARK.md)
- [Latest paired speed, profiles and verified evidence](GLM53_PAIRED_SPEED.md)
- [Completed prefill reconstruction component](GLM53_PREFILL_RECONSTRUCTION.md)
- [Completed small-prefill routing comparison](GLM53_PREFILL_ROUTING.md)
- [Completed C2/C8 graph comparison and verified traces](GLM53_GRAPH_EXPANSION.md)
- [Admission audit and bounded next serving test](GLM53_PREFILL_ADMISSION.md)
- [MTP, routing and FP8 options assessed against actual state](GLM53_SPEED_OPTIONS.md)
- [Native MTP compatibility findings and pilot requirements](GLM53_MTP_PILOT.md)
- [Decode boundary and isolated MLA diagnostics](GLM53_DECODE_STATE_DIAGNOSTIC.md)
- [BF16 kernel measurements and rejected variants](GLM53_BF16_KERNEL_DIAGNOSTIC.md)
- [Detailed continuation handoff](GLM53_LOSSLESS_RUNTIME_HANDOFF.md)

Spark experiment root:
`/home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/full-model-v1`.
Authoritative run state is in `quality-model-controller-pairedspeedv1.json` and
`paired-speed-v1-evidence-check.json`; use these rather than assuming this
timestamped page is live. Frozen source: `full-pairedspeedv1-src`. Local run
notes/scripts: `/private/tmp/mxwave-paired-speed-v1/`.

Latest graph controller: `quality-model-controller-graphexpandv1.json`,
**complete**, original service healthy. Source `full-graphexpandv1-src`,
results `run-graphexpandv1/results`, evidence receipt
`graph-expansion-v1-evidence-check.json`, corrected independent review
`graph-expansion-v1-independent-review-v3.json`. Local evidence and verifier:
`/private/tmp/mxwave-graph-expansion-v1/`. All 466 payloads and eight worker
profiles verify; the bundle's numerical/performance gates fail. Controller
882869, watchdog 890911 and collector 882868 are completed-run IDs.

Latest admission controller: `quality-model-controller-prefilladmissionv1.json`,
**complete**, original service healthy. Source `full-prefilladmissionv1-src`,
results `run-prefilladmissionv1/results`, evidence receipt
`prefill-admission-v1-evidence-check.json`. All 599 payloads and six worker
profiles verify on Spark/Mac; archive SHA256
`d63d6e69c7acda35d5d6f6cb83b38bbb1fea958b2681b1611a24930db64a9da7`.
CPU probability recomputation differs by at most 1.7765e-15; model gates stay
unchanged and all non-probability fields/decisions agree exactly. Local
`/private/tmp/mxwave-prefill-admission-gpu-v1/`. No configuration promoted.

Latest MTP controller: `quality-model-controller-mtpretryv1.json`, **failed_restored**; prepared
`mtp-retry-v1-prepared.json`, frozen assembled source `full-mtpretryv1-src`.
Completed full CPU audit: `mtp-dispatch-v3/`; local continuation:
`/private/tmp/mxwave-mtp-retry-v1/` and `/private/tmp/mxwave-mtp-loader-v3/`.
Controller PID **495392**, restoration watchdog **498189**, untimed-quality
counter observer **506688** are historical completed-run identifiers. All 338
retry payloads verify locally and on Spark; archive SHA256
`90428f9bc31f8e83f35253ec9948afc07352675b6e933ee01403076c10e985e6`.
Corrected geometry source is now frozen/uploaded and passes CPU native binding;
another MTP GPU attempt has not run. Geometry proof progress:
`mtp-geometry-v1-progress.json`, **complete**, 14 payloads; local
`/private/tmp/mxwave-mtp-geometry-v1/`. Archive SHA256
`1d7aa99458c4ccef6144e26f8bc2cf85e4d76f038c92dd10e9330e12cd13b4db`.

Failed first GPU attempt `mtppilotv2` and successful control: 333 payloads verify
on Spark/Mac, independent reviews byte-identical, archive SHA256
`cdfdeb55b386bd714d07707a5ba00e70b5a0ae941ab22c501a6e9194d465cdb2`.
Actual-loader CPU regression `mtp-loader-v3`: nine payloads verified locally,
archive `bce71777d09371d507809d1717fe780bbc925c90a871eb4e4f0b040935014918`.
The retry reuses the measured control across a draft-only loader repair; it
does not claim a fresh control paired in time. Target kernels/weights/precision
and all numerical floors remain unchanged.

Latest completed component state: `prefill-decode-controller-prefilldecodev7.json`;
reports/source in `prefill-decode-v7/`. Local continuation:
`/private/tmp/mxwave-prefill-decode-v7/`. All45 payloads/18,432 events verify
on Spark/Mac; shared16 route fails. Archive SHA256
`ae9d4fd7a8e8173def9654c1ca0b53dc40ff79f1e59340c3d7d8d62711d663f3`.
These controller files take precedence
over this timestamped page. Completed V3: `prefill-decode-v3-evidence.tar.gz`.
CPU audit: `prefill-base-audit-v1/controller.json` and
`prefill-base-audit-v1/results/base-scan.json` on the worker.

Completed run: controller PID **2104128**, collector **2104127**, watchdog
**2108100**; launch **19:53:38 UTC**, restoration **20:22:23 UTC**.
Source inventory SHA256:
`00840627c120063f83d7f2095adfe4bf72206768ba23d9d63a5f13f91ed2e45e`.
Plan SHA256:
`8b339a6c316abe61a4994487e79c11f7822265f31e46539f1b4c9f84e2323adf`.

Current documentation and the preceding optional MLA analysis bundle remain
local: automatic approval review rejected their optional upload to Spark.
Original Spark evidence and Spark CPU verification are already preserved.
The separately approved performance source contains no rejected documents.
