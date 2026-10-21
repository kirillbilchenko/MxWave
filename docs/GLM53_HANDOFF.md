# GLM compression experiment handoff

**21 October 2026, 19:45 UTC: invalid draft computation reproduced; concrete loader mismatch.**
`stock-draft-trace-v1` completes all four requests. Native replay and draft-eager
both accept **0 / 155 proposals per request (0 / 620 total)**. All 80 retained
candidate scores in each observed step are NaN; eager final draft hidden states
and candidate logits are also entirely NaN. Target auxiliary features and their
combined context projection are finite. Proposals repeatedly contain token 15360.
Eight trace files (four requests x two ranks) are retained. This rules out graph
replay alone, and establishes invalid drafter computation rather than ordinary
low acceptance. Production restoration is in progress; consult the V1 controller.

Source and checkpoint audit identifies a specific incompatibility: all twelve
`layers.{0..5}.{attention_conv,mlp_conv}.kernel_projection.weight` tensors are
BF16 [1024,4096], with no scale tensors. The old runtime constructs these linears
with `quant_config=None`; vLLM 0.32 passes the FP8 quantization configuration.
The checkpoint excludes only `model.fc`/`fc`; startup explicitly warns about loading
these BF16 tensors into FP8 parameters. This is the leading cause, not yet a
causally verified fix. The zero-acceptance symptoms are independent of the earlier
stock target-repeat KL failures; no existing numerical gate is cleared.

Preparing V2 with a **separate metadata-corrected drafter directory**: add explicit
BF16 exclusions for local layer IDs and native global IDs 45..50. Original config
and weights remain untouched; the weight file is hard-linked and SHA256-verified.
Both CPU preflights must instantiate all twelve native convolution projections as
BF16, and the GPU observer must confirm dtype/method before generation. Same two
prompts, modes, 32-token budgets, target weights and draft KV dtype; no training,
new quantization, vLLM source replacement or performance claim.

**21 October 2026, 19:38 UTC: short native drafter trace launched.**
Root `stock-draft-trace-v1`, controller **2973511**, independent watchdog **2974159**;
plan SHA256 `a06e81f1e76d47cf4878048b0954e40ee994bd0ebd917c3bc206e18f4a593af5`,
source commit **ea930d3**. Both official-image no-GPU preflights, full local pytest,
Ruff and strict core mypy pass. V3's failing BF16 draft-cache profile is retained;
only the observational worker/client change. Each of two development prompts runs
for 32 output tokens with native draft replay and an explicitly labelled draft-eager
control (`is_profile=True` on the proposal call). Eight boundaries per request are
retained; all timings are diagnostic. The 25-minute lease is a restoration bound.
Production is temporarily paused under exact restoration controls. Consult this
controller for live state; the old deployment comparison is permanently stopped.

**21 October 2026, 19:35 UTC: comparison stopped; diagnose zero acceptance first.**
The user reprioritized identifying the failure before more benchmarking. Controller
2545114 received SIGTERM through its normal restoration path; all partial results
remain in `deployment-compare-v1`. Exact original production was restored at
**19:35:23 UTC**, with both original identity/configuration hashes and authenticated
500,000-token endpoint verified. Controller state is `failed_restored` because of
the intentional interruption. Remaining timing repetitions and both full GSM8K
arms are **deferred, not completed or cancelled**. Do not resume the old controller.

The short-code C1 gap is explained mainly by round count: production accepts 195
draft tokens in 60 rounds; stock CT accepts zero in 255 rounds for the same 256
output tokens. Total request times are 4.662 / 20.472 seconds. After subtracting
TTFT, client-derived time per round is about 70.31 / 78.49 ms (not GPU kernel
timing). Thus 4.25x more rounds explain most of the 4.39x runtime; neither C1 arm
queues. Production's measured proposals are seven per round even at C4/C8/C16;
record this effective behavior separately from its nominal CLI policy.

Preparing `stock-draft-trace-v1`: two existing development prompts, 32 output tokens,
native graph replay versus explicitly labelled draft-only eager execution. Capture
the first eight target/proposal boundaries per request, actual auxiliary feature
health, token IDs, query positions, slots, candidate scores and rejection counts.
This is an observer and graph-isolation diagnostic, not a speed trial or retraining.
Native source review finds matching six target-layer boundaries; numerical
equivalence is not established. V3 already ruled out draft cache precision alone.
No exact cause or fix is claimed. A new independent watchdog and exact restoration
must cover the diagnostic; production is currently restored, preparation is CPU-only.

**21 October 2026, 19:22 UTC: first production pass complete; CT is slower.**
`timing-A-0` completes all 24 cells. `timing-B-0` has completed five cells with
zero API errors and zero unexpected cold-prefix hits. Every completed CT cell
records zero accepted draft tokens. Both controller **2545114** and watchdog
**2545780** remain alive; no stage errors are reported. Full GSM8K has not started;
both speculation-off 1,319-question stages remain scheduled after the six timing pairs.

First-trial code throughput (aggregate output tokens/second; repeats pending):

| Cell | Production | Stock CT | CT difference |
| --- | ---: | ---: | ---: |
| Short C1 | 54.90 | 12.50 | −77.2% |
| Short C4 | 76.23 | 29.51 | −61.3% |
| Short C8 | 83.58 | 42.41 | −49.3% |
| Short C16 | 82.83 | 43.62 | −47.3% |
| Long C1 | 36.38 | 11.63 | −68.0% |

These are current complete-deployment measurements, not compression-only effects
or final medians. Short-C16 CT reaches eight running requests and eight waiting.
Both arms pass the same-salt reuse/fresh-salt miss control; observed reused tokens
are 9,216 for production and 4,608 for CT, exposing an implementation difference
while measured cohorts remain cold. CT rank-0 startup allocates 93,584,163,840 bytes
with the dense BF16 head. No matched production Torch-allocation delta is claimed.
The unchanged run continues; no drafter tuning or promotion occurs between arms.

**21 October 2026, 18:59 UTC: comparison healthy; full GSM8K remains queued.**
The first production timing block has completed **15 / 24 cells**, with **zero
API errors and zero unexpected prefix hits** in those measured cells. The controller,
watchdog and client processes are alive, and all frozen source/input hashes still
match the launch plan. Both full speculation-off GSM8K stages remain in that plan,
after timing. No CT-versus-production speed conclusion or new GSM8K score is
available yet. Continue the existing controller **2545114**; no new run is needed.

**21 October 2026, 18:46 UTC: deployment comparison launched on Spark.**
Root `glm53-package-v1/deployment-compare-v1`, controller **2545114**, independent
watchdog **2545780**. Frozen plan SHA256
`e0dd47afd4ed10062892fda81dba0c28138194fe00a012c6d4d06bbc993f3db6`;
source commit **a166019**. Both no-GPU stock preflights pass. The first stage is
unchanged production, `timing-A-0`. Twelve A/B timing blocks precede full GSM8K
on both targets with speculation disabled. All 1,319 questions per arm are pinned;
no answer-based selection, retries, gate changes or promotion. The ten-hour lease
is a restoration limit, not a predicted duration. Work continues without the laptop.

Both production quality clones are prepared but stopped. Their original image,
mounts, overlays, environment and capacity are verified; only speculation and
private ports differ, plus Docker's equivalent default OOM flag normalization.
The watchdog also terminates an orphaned benchmark client before restoring the
original service. Local Ruff, core mypy and the full suite pass; the added collector,
grading and orphan-client checks pass. The unrelated investigation-proposals document
is untouched. Consult `controller.json` and per-stage `progress.json` for live status;
no completed timing or GSM8K score is claimed at launch.

**21 October 2026, 18:40 UTC: bounded comparison and full GSM8K preparation.**
Latest user priority is a usable stock-vLLM recipe and concrete end-to-end results,
including **all 1,319 GSM8K questions on both production and CT, speculation off**.
Stop adding drafter/MTP/kernel tuning ahead of these deliverables. Preparation is
CPU-only under `glm53-package-v1/deployment-compare-v1`; production remains available.
The frozen sequence is six A/B timing pairs, then separate A/B quality launches.
If timing cannot be collected, retain the blocker and still attempt both full
quality arms. Existing numerical/output-identity gates remain failed; diagnostic
timing and answer accuracy cannot clear them. No promotion.

Production's original authenticated 500k service was exactly restored at
**18:29:47 UTC**, with both original identity/configuration hashes verified.
V3's BF16 draft cache is observed on all six layers, but acceptance remains
**0 / 3,810** across all six requests. Therefore FP8 draft-cache dtype alone does
not explain zero acceptance. Startup allocation is **172.490866 GiB TP2**.
The completed V3 evidence archive has 31 payloads / 3,382,149 bytes, SHA256
`0d2c6ea3610c6f4b303acdbbd1df44ad2ab1c84e4d60c830d6746714029d3cc4`.

Timing uses identical frozen token IDs; production and CT tokenizer JSON, tokenizer
configuration and chat-template file hashes all match. Production has no exposed
prefix-reset route. Use native per-request cache salts in both arms, separately
for warm-up and measurement, and require a same-salt reuse/fresh-salt miss control
before timing. Prefix caching stays enabled. The production quality clone preserves
the original image, overlays, mounts, private environment and capacity; only
speculation and private API/collective ports change. Originals are never edited.
The same existing GSM8K grader is now dependency-free; API failures and truncations
remain in the denominator, with no retries. See
[comparison protocol](GLM53_STOCK_VLLM_BENCHMARK.md) and
`deployment/glm53/compare_control.py`. Consult its controller for actual launch state;
CPU preparation is not a GPU benchmark result.

**21 October 2026, 18:21:32 UTC: draft-cache isolation check launched.**
Both CPU preflights pass; V3's frozen native commands differ from V2 only in
`speculative_config.kv_cache_dtype` (`fp8_e4m3` → `auto`). An untimed observer
also records actual per-layer draft-cache dtype/shape. Controller PID **2320510**,
independent watchdog **2321179**, root `stock-profile-v3`, plan SHA256
`9d4a2ddb760ab8ab898e2458fddc19a1d36cde073994aa66e9057d46900b4c8e`.
The same six prompts and 25-minute lease/exact restoration apply. Production is
temporarily paused; consult this controller for current status. This tests the
cache hypothesis without changing target weights, drafter weights, draft length,
TRITON_ATTN backend, target precision or any quality gate.

**21 October 2026: zero-acceptance follow-up and memory accounting.**
The old selected Modal launch used `speculative_config.kv_cache_dtype=auto` with
FLASH_ATTN; stock V2 uses `fp8_e4m3` with TRITON_ATTN. The old helper named
`GLM_LV_DRAFT_FP8_KV` dequantizes FP8 projection weights into BF16 context-projection
buffers. Earlier text calling it an FP8-cache adapter was imprecise. The native
FP8 draft cache logs default scale 1.0; this is a suspect, not causal proof.
Preparing `stock-profile-v3` to change only the draft cache to `auto`, retain the
six fixed code/chat/explanation requests and inspect actual cache tensor dtypes.
Both CPU preflights, idle admission, the independent watchdog and exact production
restoration must precede/cover the GPU run. No training, MTP launch or promotion.

Independent retained-receipt accounting locates the +0.750561 GiB total allocation:
**+0.149785 GiB after model loading/KDA conversion**, **+0.600776 GiB during later
initialization**. Exact KV/graph/workspace attribution is not available. Both
expert-scale/vocabulary storage and all 68 KDA weight/scale sizes/dtypes/rounded
hashes match the earlier selected CT. Expert scratch is 24 MiB smaller across
TP2, already included in the total. Separate reserved-but-unallocated space grows
0.142017 GiB; do not add it to Torch allocated or call it lost compression.
See `out/glm53-package-v1/stock-profile-memory-breakdown.json`.

**21 October 2026, 18:14:28 UTC: exact original production restored.**
Both original container identities/configuration hashes match and the authenticated
endpoint reports `glm-5.3-flash`, 500,000-token context. `complete_restored` means
the diagnostic and restoration completed, not that speculation or quality passed.
No further GPU experiment is queued. The two profile trials and native MTP CPU
audit are retained in `out/glm53-package-v1/stock-profile-v1-v2-evidence.tar.gz`:
**82 payloads / 4,472,098 bytes**, independently SHA-verified on Mac and Spark2,
archive SHA256 `faa76f88fc593c3c192b9572953dad3668bcaf3d88eec10360830fee26655545`.
The independent Mac review rechecks allocations, all six counter deltas, token-ID
repeat failures, selector mismatches and both restoration receipts. Ruff, core
mypy and the full 1,087-test suite / 1 skipped pass.

**21 October 2026: selected-CT V2 completes all six requests; acceptance fails.**
The native `DFlash2Speculator` and FP8 drafter load with shared compressed head
and embedding, five proposals at C1 and the configured 5/5/3 policy. All six
128-token code/chat/explanation requests complete. Each records 127 draft calls,
635 proposals and **zero accepted tokens**: **0 / 3,810 accepted overall**.
Every per-position counter is zero. Raw elapsed times are 9.933–10.445 seconds;
these are smoke diagnostics, not a qualified speed comparison. Code and
explanation repeats differ; chat repeats match. Existing target numerical
failures remain open. No promotion or MTP GPU run is queued.

The actual native-worker observer measures **172.495512 GiB across TP2** at
startup, **0.750561 GiB above** the previous selected Modal CT profile. Both ranks
confirm whole-module vocabulary sharing. Registered drafter-exclusive tensor
storage is 1,269,951,584 bytes per rank, which excludes KV and graph pools.
Functional generation and retained memory savings do not establish useful
speculative execution. Diagnose feature taps, loaded draft parameters, KV and
proposal/verification alignment before claiming DFlash2 works on stock.

The CPU-only `stock-mtp-audit-v1` also finds a native compatibility blocker:
all 1,756 checkpoint draft tensor headers exist, but **870 / 872** native draft
quantization selectors disagree with the source policy, for both a fresh CT
config and an inherited mapped target config. The stock draft class has no
namespace mapper or fused aliases. Installed loader source still configures the
provided CT object in place. The old namespace/policy isolation repair is relevant,
but must be revalidated for this image and scoped to the draft; do not run an MTP
GPU comparison with the known wrong policy or silently expand scale-compression
coverage to its extra expert layer. Restoration status remains in the controller.

**21 October 2026, 18:06:13 UTC: selected-CT profile V2 launched.**
V1 reaches API readiness and measures **86.248634 / 86.246878 GiB**, or
**172.495512 GiB across TP2** after drafter, KV and graphs. This is **0.750561 GiB
more** than the historical selected Modal profile's 171.744951 GiB; it is not a
new production-memory comparison. V1 then fails before client inference because
our observer reads the legacy `runner.drafter` attribute instead of native
`runner.speculator`. Exact original production is restored at **18:05:35 UTC**.
The failure remains retained.

V2 changes only that observer and adds a CPU regression check using the actual
native runner/speculator/model classes. Both CPU preflights pass. Controller PID
2204242 owns `stock-profile-v2`, plan SHA256
`fcd5f0aa12b935f47771b1de2715eabdfab9b75e8671afe909022f4a5f2aad05`.
The same independent 25-minute watchdog and exact-container restoration apply.
No weights, kernel, gate or serving setting changes. Consult the live controller
for availability; no speculative inference result is claimed at launch.

User now requests **DFlash2 versus native MTP**. This supersedes the earlier MTP
deferral, but does not qualify the old overlays on vLLM 0.32.0. A CPU-only audit
checks the actual stock MTP selector/loader against the CT checkpoint's 1,756
draft tensors before preparing a GPU comparison. Current DFlash2 policy remains
**C1/C2: 5 proposed tokens; C3–C16: 3**. The running diagnostic exercises C1 only;
configured lengths are not accepted-token counts or concurrency qualification.

**21 October 2026, 17:58:20 UTC: selected-CT profile check launched.**
Both CPU preflights pass. Controller PID 2135666 owns `stock-profile-v1`, plan
SHA256 `147b5088945068cee5f18d967bb5a2199b98213222e30b7c21cf4379a6cec492`.
The independent 1,500-second watchdog and exact original-container restoration
are required. The reused compilation caches come from V7; no new image/kernel
build is needed. Consult its controller for actual production availability.
This collects memory and actual drafter counters on the previous selected 8k
Modal configuration; it cannot clear the separate stock numerical failures or
establish a production speed advantage. Local Ruff/core mypy/full pytest pass.

**21 October 2026: preparing a matched selected-CT profile check.**
User asks whether stock vLLM runs CT with its drafter and whether the recipe can
match the previous CT. `stock-profile-v1` uses the same selected Modal FP8 weights,
5/5/3 drafts, 8k context, 16 slots, 4,096 prefill, 6 GiB KV/rank, prefix off and
compressed head plus embedding. The historical selected-Modal startup is
**171.744951 GiB total**, not the earlier Inco-based 170.844 GiB. Native backend and
missing old overlays remain explicit differences. See the frozen
[profile comparison](GLM53_STOCK_CT_PROFILE.md). Six short development requests
check actual speculative execution and memory; no performance or quality gate
is relaxed. Production remains unchanged during CPU preparation; launch requires
both CPU preflights, the independent 25-minute watchdog and exact restoration.

**21 October 2026: memory answer, separated by what was measured.**

| Measurement | Across both GPUs | Per GPU | Meaning |
| --- | ---: | ---: | --- |
| Current stock-vLLM lossless packing | **6.941 GiB** | **3.470 GiB** | Resident expert-scale and embedding bytes saved, after the reported expert workspace; target head is dense. |
| Earlier lossless BF16-KDA configuration | **9.367 GiB** | **about 4.684 GiB** | Matched same-CT allocation reduction; included KDA and output-head compression. |
| Earlier CT MXFP8 versus ModelOpt | **14.877 GiB** | **7.447 / 7.430 GiB** | Whole-model allocation difference, including different quantization and checkpoint formats. |

These figures are not additive. Moving to stock vLLM did not change the selected
KDA precision: the prior MXFP8 weights were retained and their hashes verified.
The current recipe no longer applies lossless BF16 compression to KDA and leaves
the output head dense, so its compression-only accounting covers fewer components.
The full current deployment's advantage over production still needs a fresh matched
allocation measurement; **6.941 GiB is not that total-production delta**.

**21 October 2026, 17:14:45 UTC: V2 complete; exact original production restored.**
All eight frozen short-input requests finish. Independent Mac review verifies all
305 archive payloads, both original container identities/configuration hashes,
the restored authenticated 500k endpoint receipt and all 154,880-value distributions.
Cold/cold KL is **0.0323304**, warm/warm **0.0249708**; all seven comparisons fail
the unchanged 1e-5 gate. Controller `complete_restored` describes collection and
restoration, not a numerical pass. No further GPU experiment is queued.

All four paired traces (cold/warm, ranks 0/1) first differ at event 45,
layer 3 `self_attn.indexer.indexer_op` **output**, with all earlier recorded
events equal. The active output has shape 4,435 x 2,176 int32. Retained last-row
bytes have exactly equal index multisets, including the same 125 negative slots,
but 1,964–2,036 positions differ. Attention output then differs. This supports
an index-order hypothesis for this observed boundary; it does not yet establish
the operation responsible, equality of every row's membership, or a causal fix
for the full-model KL failure. Next: inspect the actual indexer implementation,
capture/replay its ordering and attention consumption, then test canonical order
under identical inputs. Do not blindly apply the old SM90 compaction patch to
the native SM120 backend, change precision, or relax the gate.

V2 archive: `out/glm53-package-v1/stock-numerical-v2-evidence.tar.gz`,
**305 payloads / 6,754,397 bytes**, SHA256
`100f82121a84ce3d72b8d1802e91e7ed2a665052ec64c2c4fce8ce348b010cab`.
Independent review: `stock-numerical-v2-independent-review.json` in the same
directory. V1's 75-payload archive is also independently verified on Mac and
Spark2. The full local checks remain 1,087 passed / 1 skipped, Ruff and core mypy.

**21 October 2026, 17:06:49 UTC: deep layer-three follow-up launched.**
V1 restores exact original 500k production at **17:06:01 UTC**; all original
identity hashes match. V2 then starts under a fresh 1,500-second watchdog,
controller PID 1760972, plan SHA256
`ecafce0f7bdbb9cb6762d6aa3d005abe7b55a943999ab4df444f36a93a0135ee`.
It repeats only the frozen 4,435-token failure, retaining the same eight cold/warm
and traced/untraced controls. The observer now includes every layer-three MLA
submodule, excludes unused rows of shared index buffers, records actual backend
classes and retains last-row bytes for numerical comparison. Both CPU preflights
pass. No kernel, model weight, precision, capacity setting or gate is changed.
Production is temporarily paused; consult `stock-numerical-v2/controller.json`.

**21 October 2026, 17:04 UTC: speculation-off run completes; first divergence localized.**
All 16 fixed-input requests finish. Cold/cold self-repeat KL is **0.0199474** for
4,435 tokens and **0.1555095** for 12,449 tokens; both fail the unchanged 1e-5
gate. Warm/warm and traced controls fail too, although the greedy next token is
the same in every compared pair. These are self-repeat results, not a comparison
of compression against a reference. Production restoration is in progress.

The short cold/cold trace has 138 matching-layout events per rank. The first
differing event on both ranks is **layer 3 `self_attn` output**, with its complete
input hashes identical and all preceding recorded activations equal. Layers 0–2
use dense MLPs; this difference occurs before the first compressed expert layer
executes. Speculation is absent. This localizes the initial discrepancy to the
native MLA path; it does not yet distinguish projection, indexer, compaction,
cache or attention-kernel behavior. Do not label this harmless rounding or claim
the older stable-compaction fix necessarily applies to v0.32. The next targeted
probe observes layer 3's internal boundaries and active index rows.

**21 October 2026, 16:56:51 UTC: stock numerical diagnostic launched.**
Both CPU preflights pass; detached controller PID 1675071 owns `stock-numerical-v1`.
Plan SHA256 `9b0b7aef0d5f0ee59fb4f9e4d9481c8a694b74d326b9bd22a749244da2727419`.
Production is temporarily paused under the independent 1,500-second watchdog.
Both ranks use isolated copies of V7 compilation caches; this is not a cold-boot
timing trial. Check its `controller.json` for actual availability and restoration.
No promotion or final benchmark is authorized by a diagnostic pass alone.

**21 October 2026, 16:56 UTC: user authorizes continued stock qualification.**
Preparing `stock-numerical-v1`: the same v8 stock image, same verified MXFP8 KDA
weights, packed scales/embedding, dense head and 500k / 32 slots / 6 GiB / 6,919 /
prefix-on settings, with speculation explicitly disabled for diagnosis. Frozen
4,435- and 12,449-token inputs get full-vocabulary next-token distributions under
cold/cold, warm/warm and cold/warm controls. Temporary read-only module hooks hash
complete activations in layers 0–5 and last rows in later layers; traced/untraced
controls expose observation effects. This is not a speed or final quality run.
Local checks: 1,087 tests / 1 skipped, Ruff and strict core mypy pass. Production
remains running during CPU preparation; GPU launch must have the independent
1,500-second watchdog and exact-container restoration.

Memory clarification requested by the user: earlier matched ModelOpt allocated
185.720 GiB across TP2 and CT MXFP8 170.844 GiB, a 14.877 GiB total difference.
Current stock CT allocates 174.314 GiB, about 3.47 GiB more than that earlier CT
run, but the runtime, drafter, head policy and capacity settings differ together.
No attribution to vLLM alone is established. The current 6.941 GiB figure counts
only packed-scale/embedding resident bytes, not that total deployment advantage.
Unchanged production has no configured development RPC/worker memory reporter;
a comparable fresh Torch-allocation baseline remains missing. Do not subtract
RSS, driver usage or model-load-only bytes from the candidate startup allocation.

**21 October 2026, 16:48 UTC: V7 closed, exact production restored; memory benefit retained.**
Original container identities/configuration hashes and the authenticated 500k endpoint
are verified restored at **16:44:58 UTC** (`failed_restored`, service not paused).
No further GPU experiment is queued. V6/V7 preserve the repeatability failures;
stock deployment is functional but full serving correctness remains unqualified.

Independent Mac review verifies both-rank startup receipts. Each GPU allocates
**87.157 GiB** after drafter, KV and graphs (87.568 GiB reserved). Expert-scale
storage falls from 4.4297 GiB to 1.1104 GiB per GPU; embedding packing saves
0.1746 / 0.1748 GiB. Subtracting the reported 0.02344 GiB expert workspace gives
**3.47044 / 3.47069 GiB**, or **6.94113 GiB across TP2**, of resident tensor
storage saved by the current lossless packing. This is byte accounting, not a
matched total-allocator or production delta. The target head stays dense.
The separate, lossy KDA conversion lowers allocation by 2.10890 GiB per rank
during loading; do not add these observations to historical production gains.
The earlier 14.877 GiB production difference is not a fresh result for this recipe.

V6/V7 archive: **46 payloads, 213,780 bytes**, independently SHA-verified on Mac
and Spark2; SHA256
`79ab686e11884feb2aa4ab65ec93cd62038f78ad4807c0733227a9ef4c4a5bb8`.
Local archive and review: `out/glm53-package-v1/stock-trials-v6-v7-evidence.tar.gz`
and `stock-trials-v6-v7-independent-review.json`. Both restoration identities,
all packed-scale byte checks, dense-head policy and the 9,216 prefix-hit delta
are independently checked. No speed, KL or 32k/128k result is claimed.

**21 October 2026, 16:43 UTC: V7 confirms real prefix reuse; identity gate fails.**
The corrected 12,449-token fixture completes three 64-token greedy requests.
Prefix-hit counter delta is **9,216 tokens**; API `prompt_tokens_details` remains
null and is now handled correctly. The prompt coverage and observed-reuse checks
pass. Both cold-versus-warm and warm-versus-warm message identity fail, so the
combined smoke result is **failed**, not a qualified prefix-correctness result.
V6 also differed without any reusable prefix. This does not isolate compression,
prefix caching, sparse-MLA ordering, mHC or Stream-K as the cause on this stock image.
No KL was measured in these serving smokes, and output differences alone do not
establish quality degradation.

Stock v0.32 plus the plugin now demonstrably starts, captures graphs and serves
at the requested 500k / 32 slots / 6 GiB / prefix-on settings. Keep the native
Triton drafter-backend setting explicit in the recipe. The capacity blocker is
resolved; stop allocator retries. Numerical repeatability is the remaining startup
qualification blocker. Next diagnostic: fixed-token replay with speculation off,
matched batches and compressed/uncompressed consumers under the same CT MXFP8
weights, then trace the first differing activation. Keep the existing quality
gates and retain both control failures before any timing/GSM8K comparison.

V7 is restoring exact production under its watchdog. No further GPU experiment
is queued. Check its controller until `failed_restored` and authenticated 500k
production health are verified. All V6/V7 evidence will be retained and copied
with both-rank startup memory receipts; 32k/128k probes and benchmark remain undone.

**21 October 2026, 16:37 UTC: corrected V7 prefix check launched.**
V6's exact production restoration is verified at 16:34:17 UTC. V7 uses the same
v8 images and all serving settings, changing only the test client and frozen
real-source fixture. Offline tokenizer count: **12,449**; prompt SHA256
`ea047183899d58965c1e609805c71a61ba5de34f8c2988b7de3cae9a93d40ff7`.
The client handles nullable usage, excludes timestamp counters and requires both
actual prefix reuse and identical repeated messages. The earlier three differing
uncached messages are retained as a failed repeatability control.

Launch 16:36:59 UTC, plan SHA256
`353d7ea251d9cf2f13f1c9778b1c707652f6e2732a2dffd4aff4e5663e24da4e`.
Production is temporarily paused under the same independent watchdog. No model,
backend, capacity, draft policy or quality threshold changed. Local full pytest
passes 1,085 tests / 1 skipped; Ruff passes. CPU cache audit and source through
`bad7911` are hash-verified on Spark2 (212 retained evidence payloads).

**21 October 2026, 16:34 UTC: stock V6 serves; prefix/repeatability gate remains unqualified.**
The engine reaches API readiness with 500k context, native Triton draft attention,
and all requested capacity settings. Graph capture completes in 4 seconds using
0.41 GiB; rank-0 startup receipt records 93,584,163,840 CUDA allocated bytes
(about 87.16 GiB) after model/drafter/cache/graphs. No matched production memory
snapshot was taken, so this does not establish a fresh memory-saving delta.

All three 64-token greedy requests complete, but their reasoning text differs.
The smoke client then crashes on `prompt_tokens_details: null`. Its frozen prompt
is only 4,435 tokens, below the resolved 4,608-token cache block, so this run cannot
qualify prefix reuse. Retain both the actual output-repeat failure and test defects;
do not attribute the differences to prefix caching or compression without controls.
Exact original production restores at **16:34:17 UTC** (`failed_restored`).

The client is corrected for nullable usage, actual prefix counters (excluding
counter-created timestamps), and prompt coverage of at least 9,280 tokens. Its
identity/reuse requirements are unchanged. A longer frozen real-source fixture
is being tokenized on CPU before a V7 retry; no promotion, timing or quality claim.

**21 October 2026, 16:30 UTC: V6 passes actual GPU cache capacity.**
At 16:29:59 stock vLLM reports **640,287 tokens / 1.28x at 500k** with the
unchanged 6 GiB cache budget. Resolved group sizes are
`(4608, 4608, 4608, 4608, 16, 4608)`, scheduler LCM 4608. The native Triton
backend remedy has now passed the real planner, not only the CPU fixture.
Rank-0 model loading logs 123.878 seconds and 81.71 GiB before final cache/graphs;
this is not final serving allocation. JIT warm-up is active. DFlash group warning
persists, prefix smoke and API readiness remain pending. Production is paused
under the V6 restoration watchdog; use its controller for current availability.

**21 October 2026, 16:26 UTC: V6 native-Triton startup trial launched.**
Both v8 images pass CPU installation/CLI checks. `stock-trial-v6` uses the same
500k / 32-slot / 6-GiB / 6,919 / prefix-on / dense-head profile with explicit native
`TRITON_ATTN` draft attention. Launch time 16:26:26 UTC; plan SHA256
`55eb408a5ea450774653babdc04a42b48934b53c1a2429996fe294c57aba652c`.
Production is temporarily paused under the independent 1,500-second watchdog;
the controller restores the exact original containers afterwards. Derived V5
compilation caches were copied into separate V6 directories; this is not a cold
boot timing trial. Check `stock-trial-v6/controller.json` before assuming service
availability. GPU startup/prefix correctness remain pending; no speed claim.

**21 October 2026, 16:24 UTC: cache failure reproduced; native configuration remedy passes CPU planning.**
The read-only audit reproduces V5 exactly using installed v0.32 methods: 13.9561 GiB
required and a 9,472-token maximum with a 6 GiB budget. Default FlashInfer drafter
attention chooses 64-token pages; each claims a 36,022,272-byte shared-pool block.
The drafter reserves 282 blocks, versus 109 for target attention, 24 for KDA state
and one for its indexer tail. This grouping waste explains the failure.

Selecting the **native `TRITON_ATTN` drafter backend** makes its pages 4,608 tokens
and its reservation five blocks. The same stock planner now requires **4.6632 GiB**
for 500k, with all 500k / 32 slots / 6 GiB / 6,919 prefill / prefix-on settings
retained. An Inco-geometry control still needs 12.8826 GiB with stock FlashInfer;
the Modal/Inco size difference alone does not explain the rejection. These are
CPU allocation results, not GPU startup, speed or long-context qualification.
No production allocator implementation is ported and no vLLM source is replaced.

`deployment/glm53/check_cache_capacity.py` retains config/source hashes and the
negative control. Receipt: `out/glm53-package-v1/cache-audit-v1/stock-capacity.json`.
The CLI gains an explicit `--draft-attention-backend TRITON_ATTN`; its default
remains `auto` pending qualification. The separate unidentified-DFlash-group
warning remains unresolved. V6 startup/prefix retry is being prepared under the
existing watchdog; production remains running until its launch. Local full tests,
Ruff and strict core mypy pass; v8 wheel SHA256
`8ede5b783cd9a683cea5bb250edf38ef7a06b2c502586eb2c6061045d7dd149b`.

**21 October 2026, 15:26 UTC: stock capacity gate blocks the requested comparison.**
`stock-trial-v5` passes all GPU components, including the real compressed-embedding
forward and exact TP shard masks. Both models/drafters load, the corrected embedding
path clears the full-model profiling forward, and rank 0 logs 123.294 seconds for
model loading. The engine then rejects **500,000 context with 6 GiB KV per rank**:
its planner requires **13.96 GiB**, has **5.97 GiB usable**, and estimates a maximum
length of **9,472** for this configuration. The null-block reserve explains why
usable cache is below the explicit 6 GiB. No API readiness, prefix reuse, final
KV/graph allocation, speed measurement or GSM8K run completed.

The log also warns that it cannot identify a draft-model KV group for DFlash.
This warning is retained, not established as the cause of the capacity rejection.
The pinned planner accounts for shared-pool group sizes and hybrid-state blocks;
audit those actual specifications against production before proposing a fix.
Do not bypass its guard, reduce context, turn off prefix caching or increase KV
silently: the user explicitly requires reporting this mismatch. No further GPU
retry is queued. Exact original container identities and the authenticated 500k
model endpoint are verified restored at **15:26:07 UTC** (`failed_restored`).
V4/V5 failure and restoration archive: 40 payloads, 384,822 bytes, SHA256
`35e5306d4c37a3d981b0221cd26894d592d5190442b31078f026deb412f4a244`.
Source fix is committed as `8e8cdd1`; all pre-existing failures remain retained.

**Next action: CPU/read-only cache-accounting comparison, before another outage.**
The restored production boot logs report the same explicit 6 GiB budget and
4608-token block size, but a capacity of **723,602 tokens / 1.45x at 500k**.
Stock V5 instead requires 13.96 GiB and estimates 9,472 tokens. These are planner
reports, not successful long-context probes. Compare target/recurrent/drafter
cache specifications, grouping/padding and rollback-state reservations across the
two runtimes; the differing mamba page padding (0.70% versus 2.86%) alone is not
evidence explaining the gap. Check the DFlash group warning as a separate lead.
After resolving or explicitly documenting the configuration difference, repeat
startup, prefix reuse and final memory checks, then 32k/128k probes and the frozen
performance/quality comparison. Production remains running; no GPU retry queued.

**21 October 2026, 15:19 UTC: native loading accelerated; embedding dispatch gap fixed.**
`stock-trial-v4` failed during the first profile forward and restored exact
production at **15:18:05 UTC**. The verified native Run:ai loader completes the
76,060-tensor stream in about 33 seconds on rank 0; total logged model loading
is **126.066 seconds**, versus 794.378 seconds in V3. These are sequential startup
observations, not controlled cold-load benchmarks. API readiness is still pending.

The failure is concrete: v0.32 enables its dense fused embedding lookup for
`UnquantizedEmbeddingMethod` subclasses, bypassing our `embedding()` method and
reading `weight=None` after compression. The packaged method now clears that
dense-only flag during construction. Upstream's masking/reduction and unowned
dense embedding fast path stay intact. The official embedding source is now
hash-bound in the compatibility contract. Both v7 CPU containers pass an actual
upstream forward test with the dense parameter removed, two vocabulary shards,
boundary/repeated/multidimensional IDs, and unchanged dense dispatch. The GPU
component check now also exercises that forward with the real compressed row codec.

Local validation: 1,083 tests / 1 skipped, Ruff and strict core mypy pass.
Wheel SHA256 `a887202f278e57b9bcdfdad1638234eb1d01cad9fc6effd1a05e2272a5272c8e`
(238,249 bytes). `stock-trial-v5` launched at **15:19:33 UTC**, with the
same 500k/32-slot/6-GiB/6,919/prefix-on/dense-head settings and the 3 GiB Run:ai
buffer. The independent watchdog restores exact production after the run;
production is temporarily paused. Plan SHA256:
`f31d441a8b64adedb09ec2450eedad40c268d132f73f4692acdc89ba8d85428d`.
No default kernel change, quality gate change or promotion.

**21 October 2026, 15:12 UTC: full-K independently verified; stock startup retry launched.**
`marlin-full-k-v1` completes in 412.06 seconds. Exact production restoration is
verified at **15:08:02 UTC**. The 45-payload archive is SHA-verified on Mac and
Spark2; independent NumPy review passes 48 native-output comparisons, 96 raw/packed
comparisons and 960 reported graph repeats. Our bounded padding and the upstream
full-K specialization produce **identical bits in all 64 cross-policy comparisons**.
Both fix the shared-row 440-to-448 divergence in both retained TP partitions.
This is component evidence, not full-model or C8/C16 serving qualification.

Full-K versus padding passes the 2% timing drift control in 11/16 pairs. Among
stable packed-scale pairs, full-K latency ranges from 3.87% lower to 1.46% higher;
there is no universal speed win. The separate comparison against native dispatch
passes 23/32 pair controls. All failures and telemetry are retained. Keep the
package default unchanged. Archive SHA256:
`7c95b8c3b58d4967d16a99453b2c997123c839b3b1beda45ec19b853ad197d7f`
(40,481,888 bytes). Local independent review: `out/marlin-full-k-v1/review.json`.
The [measured upstream follow-up](https://github.com/vllm-project/vllm/pull/46639#issuecomment-6319447508)
was posted at 15:13:30 UTC, with component scope and failed timing controls explicit.

Stock Run:ai V3 passes all 1,601 tensors / 7,424,963,848 bytes with a **3 GiB**
read buffer and eight readers; peak process RSS is 4,432,052,224 bytes. Its
5.75-second sample includes hashing and warmed file cache, so it is not a model
boot-speed result. The failed 1/2 GiB attempts remain retained. Both v6 image/CPU
checks pass, including follower `--headless`. `stock-trial-v4` launched at
**15:11:49 UTC**, with the independent 1,500-second outage bound and exact original
container restoration. It explicitly selects the verified 3 GiB loader buffer;
the CLI default remains `auto`. It retains 500k context, 32 slots, 6 GiB KV,
6,919 prefill, prefix caching and a dense target head. No timing or promotion.
Plan SHA256: `4d2513d6f765f878bf8342b6eaf76327704c53eba35d97af4f863ff93c4cbb96`.
Production is temporarily paused for this startup retry; check its controller
before assuming service availability. Full-model stock serving remains unqualified.

**21 October 2026, 15:00 UTC: isolated full-K GPU comparison active; fast-loader failures retained.**
`marlin-full-k-v1` started at 14:58:19 UTC under the retained independent
restoration controller, with a 900-second experiment bound. It compares native,
bounded padding and the #46639 full-K tail specialization on both TP fixtures,
holding row blocks and thread configurations fixed. The first negative-control
numerical checks pass; the changed-proposal case and all timing controls still
need completion. Production is temporarily stopped for this component run.

Stock-loader CPU screen V1: lazy passes 1,601 tensors / 7,424,963,848 bytes from
three complete representative shards, fingerprint inventory SHA256
`bf782d34a78fcf9b41037d91fcd4d4d6a140fb31e32ce09878955ae4fc3e0bcb`.
Eager is OOM-killed at the 8 GiB process cap. Run:ai with a 1 GiB buffer rejects
the 1,268,776,960-byte largest tensor. V2 with 2 GiB is aborted after over 150 s
without a result; memory was about 2.2 GiB and process I/O evidence is retained.
It is unqualified, not selected as a faster loader. Instrumented V3 (3 GiB,
180-second bound, phase logs/stack dumps) waits for full-K restoration.

The corrected wheel includes follower `--headless` and an explicit optional
upstream Run:ai loader configuration; the default remains `auto` until qualification.
Wheel SHA256: `21ea8f76aa28464a0bb17e0688707e82ac42c3932ff2165ddb127b8d445e81c7`
(238,073 bytes). Both-node image/CLI checks are queued after the GPU run so they
do not perturb its timings. Upstream evidence/full-K preparation is committed as
`007d79f`; no package kernel default or quality gate changes.

**21 October 2026, 14:53 UTC: exact production restored after follower launch failure.**
`stock-trial-v3` is `failed_restored` at **14:53:15 UTC**. The original container
identities and authenticated 500k model endpoint are verified. Both candidate
ranks loaded: logged model-loading time 794.378 s on rank 0 / 254.556 s on rank 1,
with 81.71 GiB logged on each before final KV/graph allocation. Rank 1 then hit
`collective_rpc should not be called on follower node`. Our wrapper omitted
upstream's required `--headless` on the follower; this is a launch-recipe defect,
not an upstream bug or a failed compression kernel. The flag and leader/follower
CPU checks are corrected. API/prefix-cache and 500k capacity remain unqualified.

A CPU-only stock-loader screen is queued after restoration, with three complete
representative shards, exact tensor fingerprints and bounded process memory.
It checks native lazy, eager and Run:ai streaming (1 GiB buffer, eight readers).
Timing includes hashing and warmed file cache: it cannot establish cold-start
speed. No production fast-loader implementation is copied into the package.

The isolated full-K raw/packed libraries compile successfully; the first attempt
failed because the production image has no CMake, and the retained retry uses
equivalent explicit NVCC commands. Frozen replay plan SHA256:
`442aad18caae5f09fd5b65750a478ae82ae078ead2bee8f05ae849b3b8159093`.
CPU preflight passes six artifact bindings. GPU replay has not started. Existing
production settings, gates and package kernel defaults remain unchanged.

**21 October 2026, 14:49 UTC: stock trial loading; upstream evidence posted; fast boot gap identified.**
`stock-trial-v3` is running on both official-image-plus-plugin containers under
the independent 25-minute watchdog (deadline 15:00:34 UTC). Production is
temporarily stopped. The component checks pass. Rank 1 finished model loading
at 14:41 with a logged 81.71 GiB allocation; rank 0 is completing the 120-shard
load. These are not final allocation measurements after KV and graphs, and API
readiness/prefix-cache smoke have not passed yet. Capacity flags remain
500k context / 32 slots / 6 GiB KV / 6,919 prefill / prefix on / dense LM head.
Upstream automatically raises the attention block size from requested 2304 to
4608 for hybrid-state page alignment; retain this mismatch in any comparison.

Production's fast boot is the custom `GLM_FAST_LOAD=1` overlay: bounded pinned
slabs, 16 read threads and 1 GiB read-ahead by default. It was omitted from the
stock recipe. The default native safetensors loader reads 120 shards lazily;
rank 0 averages about six seconds/shard, while rank 1 completed much sooner.
The log reports 170.83 GiB of checkpoint files and 25.3 GiB available RAM, with
automatic prefetch disabled on EXT4. Do not force whole-checkpoint prefetch.
Audit stock eager-per-shard / bounded Run:ai loading next, verify tensor bytes
and memory before selecting a faster default. This is a startup gap, not an
inference-speed result or a reason to restore production overlays silently.

[Upstream report and attribution record](GLM53_UPSTREAM_REPORTS.md) now links
the posted GLM/DFlash evidence on #46639 and mHC evidence on existing #59929.
Sparse compaction already has merged #50365 and the v0.32 helper explicitly
handles GLM's 2176 columns; no duplicate report was filed. #60524 concerns dense
WNA16 linears. The isolated #46639 full-K adaptation is staged/hash-verified,
with provenance recorded; no GPU result or serving-default change yet.

**21 October 2026, 14:35 UTC: upstream GPU components pass; explicit serving defaults corrected.**
`gpu-check-v3.json` passes on the official image: exact expert U8/BF16 decode,
raw/packed expert GEMMs with four graph replays at M=1/6/12/16/48, native MXFP8
basis checks and four graph replays at M=4/8/16/48, and exact vocabulary decode/basis.
Peak Torch allocation in this synthetic fixture is 2,830,336 bytes. This validates
component compatibility, not full-model quality, speed or memory savings.

The following `stock-trial-v2` full-model load stopped before weight loading at
the upstream memory-reservation guard: free 110.42 GiB, requested 111.90 GiB from
the omitted/default 0.92 utilization setting. The audited production explicitly
uses **0.85**. That setting is now explicit, along with production's 2,304 cache
block size and disabled FlashInfer autotuning. Context/slots/prefill/KV remain
500k/32/6,919/6 GiB. The trial is restoring exact production.

The stock draft cache is now explicitly `fp8_e4m3`. Upstream `auto` selects the
model dtype; unlike production it has no custom adapter overriding that choice.
This corrects the launch contract and still needs GPU verification. Images
`vllm032-sm121-v5` pass local checks and are undergoing both-node CPU preflight for
fresh `stock-trial-v3`. Wheel SHA256:
`7df60dd0e34e27bc10a8d1d03a5001311976c65acbae9a6e2932a720a2e7d9c1` (237,859 bytes).

[Deployment instructions](../deployment/glm53/README.md) now document the official
image, build/install/export/doctor/launch commands, exact capacity flags and release
limitations. A public asset download/build path is still missing; the current
exporter consumes the existing verified experiment. Do not claim a fresh user can
already reproduce everything from a publicly released bundle.

**21 October 2026, 14:30 UTC: first controlled upstream GPU check found an API gap; production restored.**
`stock-trial-v1` finished `failed_restored` at 14:28:16 UTC. Both original container
identities and the authenticated 500k endpoint are verified. Full-model candidate
servers never started. Expert scale decoding, BF16 fragments and raw-versus-packed
MoE GEMMs/graph replays at M=1/6/12/16/48 passed before the MXFP8 component raised
`AttributeError: layer.workspace`. The old failure report retains only the error;
the completed checks are inferred from its sequential control flow, not a full pass.

Upstream 0.32's repacker no longer creates per-layer Marlin workspaces. The adapter
now passes `None` to the official consumer, which obtains stream/ubatch locks from
vLLM's workspace manager. No upstream files or rounding policies change. GPU checks
now retain completed stages on failure and exercise MXFP8 eager and graph execution
at M=4/8/16/48. Local checks pass: 1,083 tests / 1 skipped, Ruff, strict core mypy.
The first local pytest invocation used the main checkout's installed import path
and failed collection; rerunning with `PYTHONPATH=.` tests this worktree correctly.

Corrected wheel: 237,739 bytes, SHA256
`694de3bbd0ab2c47b248ace66ecf991e13dc02d2575c293a7644083c53416047`.
Images/CPU checks `vllm032-sm121-v4` are rebuilding on both nodes. The next bounded
retry is `stock-trial-v2`, retaining all earlier failures; capacity and prefix-cache
requirements remain unchanged. Stock vLLM full-model serving is still unqualified.

**21 October 2026, 14:24 UTC: user confirms stock vLLM as the default recipe; GPU startup preflight prepared.**
The deployment target is the **official vLLM 0.32.0 ARM64 image plus the installed
MxWave extension**, with no modified vLLM files, old `/overlay` dependencies or
production image requirement. A subsequent comparison with unchanged production
is a deployment comparison: engine and optimization differences must be disclosed.
Do not revert to the custom production runtime just to improve comparability.

The ready image builds on both Sparks. The latest wheel is 237,672 bytes, SHA256
`f6731ce1b28877b1e5b51c4ce416ac83463a78d0025a22750b0a351a11f92d99`.
Both official-image CPU checks pass the entry point, source/asset integrity and
actual native CLI for the original development profile **and** the requested
500k / 32 slots / 6 GiB KV / 6,919 prefill / prefix-cache-on / dense-head profile.
The focused Marlin argument-boundary check now passes, including positional and
keyword routing, unowned-scale passthrough, and rejected incompatible calls.
Source provenance now includes the promoted `model_plugin.py`.

An initial small GPU check beside production failed on its first allocation,
before kernel execution: Spark2 had about 8 GiB host memory available and the
container was capped at 4 GiB. Retain `gpu-check-v1.json`; this is not a measured
kernel correctness failure. The follow-up is prepared with a 12 GiB component
container after a controlled production pause, followed only on success by a TP2
500k-configured model startup and repeated-prefix correctness smoke. Maximum trial
lease is 1,500 seconds, with a separate restoration watchdog and exact original
container identities. No timing/GSM8K or automatic promotion is part of this
startup preflight. Status/plan are under `glm53-package-v1/stock-trial-v1`.

Image IDs for this prepared trial:

- Spark1: `sha256:306173a51992b93a0d878e1bd90be6f6e311e84b1f80ce5c7a40aedcdb1f5641`.
- Spark2: `sha256:53bbcf12ad61a9629eacb4d92c9c4991d237fdade8f6774e2f8555424188bb1d`.
- Plan SHA256: `f68b186a771cd42bc1890d0ffdf1274d4bdaea46a8a133378ae6b10f055fe8d6`.

The user proposes a later full deployment benchmark: C1/C4/C8/C16, realistic
code/chat/explanation prompts around 256/2,048 tokens, 256 outputs, six A/B pairs,
fixed seeds, identical prefix-cache reset/warm-up policy, clocks/thermals/power,
actual running/queued requests and KV usage, per-position acceptance/rounds and
per-rank allocation. Follow with 32k/128k retrieval/latency probes and full 1,319
GSM8K questions on each target with speculation off. Preserve prior gates and all
failed trials; restore production, never promote automatically. Timing must not
start if the required capacity or prefix-cache correctness checks fail.

The live audit corrects the proposed baseline description: production has seven
drafts at C1, five at C2 and three at C4/C8/C16, plus adaptive/truncation hooks;
it is not seven drafts at every concurrency. `GLM_IDLE_COALESCE_MS=4` exists, but
the scheduler-control layer may override it. Its implementation permits only
0–5 ms; a live 10 ms production window is **not established**. Do not silently
change A to satisfy that assumption. Audit retained as `production-settings-audit.json`.
Certified target argmax, custom FP8 draft head, admission/truncation and custom
RoCE/model overlays are absent from the stock recipe. Native upstream DFlash FP8
loading is used and still needs model-level verification. Dense head means only
the embedding is vocabulary-compressed, so previous all-vocabulary memory numbers
cannot be reused for this trial.

Publication framing: verified mechanisms and scoped measurements support an
engineering case study plus reproducible code/assets. Novelty and general
20–50% drafter acceptance are not established by these local development samples.
Retain the failed strict KL result and distinguish original AutoRound quantization,
MxWave's lossless codecs, and the separate lossy MXFP8 KDA conversion. No public
release, article, upstream message or model promotion has been published here.

**21 October 2026, 14:07 UTC / 16:07 Berlin: packaging implementation checkpoint; official vLLM 0.32.0 CPU import check passes.**
The user requests an easily consumable package and asks that all information be
kept in this handoff. No GPU model trial, production stop, promotion or further
training has occurred during packaging. The selected CT profile remains pretrained
Modal FP8 with C1:5 / C2:5 / C3–C16:3. Original ModelOpt/Inco production was last
independently verified at 12:55:54 UTC with 500k context; packaging does not change it.

**Memory answer:** the last matched CT MXFP8-versus-ModelOpt measurement saves
**14.877 GiB across TP2 (7.447 / 7.430 GiB per GPU)**. It includes checkpoint-format
differences, MXFP8 KDA and lossless compression. It predates the selected Modal
drafter and is not a fresh measurement of the complete final serving profile.
The same-CT lossless-only saving is **9.367 GiB total / about 4.68 GiB per GPU**.
Do not add those two figures. The final upstream-0.32.0 setup must be measured
after loading the actual drafter, KV budget and captured graphs.

Implemented, not yet GPU-qualified:

- `mxwave_glm53/` is a real namespaced package included in the MxWave wheel.
  `mxwave-glm53 prepare-bundle`, `doctor`, and `serve --dry-run` provide asset
  export, installation/integrity checks and an explicit two-node launch.
- `pyproject.toml` adds the `glm53` extra pinned to `vllm==0.32.0`, the public
  command, and `vllm.general_plugins` entry point. The plugin imports no Torch
  or vLLM and makes no changes unless explicitly enabled by the launch command.
- `prepare-bundle` copies verified native libraries, all four compressed vocabulary
  shards, KDA rounding references and license notices into a relocatable directory.
  It checks the pinned payload hashes, writes atomically and refuses overwrites.
  The resulting bundle has no dependency on old `/audit`, `/trial`, `/overlay`
  mounts or external experiment Python paths. Model/drafter weights remain separate.
- The existing MxWave runtime is packaged with relative imports and source
  provenance. KDA conversion retains the measured rounding and all 68 checks.
  Startup records actual allocation and requires all 84 expert-scale tensors and
  both vocabulary methods, with dense vocabulary owners released.
- The port accounts for GLM moving to `vllm.models.glm5next.common.model` and
  Marlin MoE dropping old call arguments. Important upstream API source hashes
  are checked before hooks install. No vLLM files are overwritten.
- `deployment/glm53/Dockerfile` layers the wheel onto the pinned official image;
  its `.dockerignore` excludes model/runtime assets from the image build context.
  `deployment/glm53/cpu_check.py` is the reproducible no-GPU import/entrypoint check.
- Existing Marlin and ZipServ native dependency licenses and modification notices
  are included under `mxwave_glm53/licenses`; `CONTRIBUTING.md` records the reuse.
  No DeepInfer implementation is included.

Upstream pin: vLLM **0.32.0**, revision
`5df63a7203cdb9328d824a27b682e77c7b9928be`. Official Linux ARM64 image:
`vllm/vllm-openai@sha256:6ff5ff0ddf3658d2c2d677dac5ca2d4726ce0156b4371166d918181b210bd4bc`.
It is pulled on Spark2, which has ample disk. Spark1 had only about 59 GiB free,
so the isolated build/import work was deliberately placed on Spark2.

Completed checks at this checkpoint:

- Wheel build succeeds. Latest transferred wheel is
  `mxwave-0.1.0-py3-none-any.whl`, 236,754 bytes, SHA256
  `e064840445a0f8ac8ca70920f51cd63346701dea340a14c742940bba7eba1c31`.
- Ten focused tests pass: relocatability, tamper/missing-payload rejection,
  atomic export/no overwrite, path containment, inert plugin discovery, explicit
  five-draft/context/KV arguments, version rejection and unchanged MXFP8 rounding.
- An isolated official-image container on Spark2 installs the initial wheel, exports
  and verifies the real bundle, checks all pinned upstream source hashes,
  discovers the entry point and installs hooks twice successfully. No GPU is
  exposed; network is disabled, with 2 CPUs / 4 GiB memory limits.
  Report versions: vLLM 0.32.0, Torch 2.13.0+cu130, Transformers 5.19.0,
  compressed-tensors 0.17.0. This proves CPU packaging/import compatibility only.
  That first check used the preceding 236,677-byte wheel. The latest wheel above
  adds upstream-contract validation to `doctor`; the updated CPU script also
  parses the generated launch with the real vLLM CLI. These latest changes have
  been transferred but their official-image rerun is still pending.
- Full repository rerun passes: **1,082 passed / 1 skipped**, plus **Ruff passes**
  and **strict `mypy mxwave` passes (35 source files)**. The initial run's only
  failure was the public-command allowlist missing the new CLI; the expected
  public interface was updated. No numerical test was relaxed.
- The no-GPU container logged unavailable CUDA/native-extension warnings.
  Successful CPU imports do not establish that vLLM's GPU extension or our native
  libraries load and execute correctly with the official image.

Artifact locations:

- Mac: `out/glm53-package-v1/wheels/`.
- Both Sparks: `full-model-v1/glm53-package-v1/` holds the wheel/check scripts.
- Spark2: `full-model-v1/glm53-package-v1/runtime/` is the verified portable bundle;
  `cpu-check.json` records the first successful official-image import check.
- CPU container retained for logs: `mxwave-glm53-package-cpu-v1` (Spark2).

Exact working locations and source contracts:

- Active Mac worktree:
  `/Users/kyrylobilchenko/Documents/Projects/other/mxstream/.worktrees/glm53-flash-mxfp4`,
  branch `feat/glm53-flash-mxfp4`. Packaging changes are based on `c67cdf4`;
  implementation, tests and this updated handoff are being saved as a checkpoint.
- On each Spark, `full-model-v1` above means
  `/home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/full-model-v1`.
- [Bundle manifest and fixed model/drafter/native hashes](../mxwave_glm53/bundle.py),
  [reviewed upstream source hashes](../mxwave_glm53/upstream-contract.json),
  [runtime source provenance](../mxwave_glm53/source-provenance.json),
  [license notices](../mxwave_glm53/licenses/THIRD_PARTY_NOTICES.txt),
  [launch command](../mxwave_glm53/cli.py), and
  [CPU reproduction script](../deployment/glm53/cpu_check.py) are the implementation
  sources of truth. The provenance inventory still needs the promoted
  `model_plugin.py` entry before a final release.
- The exported runtime is relocatable; model and drafter checkpoints must be
  available separately on both nodes. The CLI verifies their pinned metadata,
  the drafter payload and presence of every model shard. It does not independently
  hash every full-model shard on each launch. KDA source values, vocabulary values
  and exact expert-scale reconstruction have additional loader checks.
- Default launch is **Linux ARM64 / SM121 / TP2 across two nodes**, API **8046**,
  master port **29719**, context **8,192**, KV **6 GiB per rank**, **16 slots**,
  prefill budget **4,096**, prefix caching **off**, fixed seed **42**, native Modal
  DFlash **5/5/3**, and Marlin linear/MoE backends. Context and KV are explicit
  CLI options; a larger requested value is not evidence of long-context support.
  This is the development profile, not the tuned 500k production configuration.
- `serve --dry-run` checks assets and prints arguments without starting a server.
  `doctor --gpu` checks device availability/capability; it is not an inference
  test. Actual launch emits per-rank KDA/startup/allocation reports. Use a fresh
  reports directory for each launch; retained qualification files are not silently
  overwritten by the KDA installer. API authentication uses standard vLLM settings.
- The old sparse-ordering, mHC, expert-padding and custom drafter optimizations
  are **not ported into this package**. Upstream implementations are used, and
  full-model repeatability, acceptance and performance must therefore be checked
  again. A successful import is not proof that an old overlay is safely redundant.

Remaining before calling the package usable for serving: validate the full native
CLI against 0.32.0, finish the ready-image build and final checks, exercise native
GPU consumers and full-model startup/replay, and record actual final allocations.
Full-model speed/quality and production-equivalent C8/C16/500k qualification remain
separate outstanding gates. This package currently preserves the explicit 8k
development default; it does not silently claim the production 500k qualification.
The old patched-image performance numbers do not establish upstream-0.32.0 speed.

Resume in this order:

1. Complete the source-provenance inventory and add a focused check of the new
   Marlin argument mapping (owned packed scales, ordinary passthrough, and rejected
   incompatible modes). Keep the pinned upstream source guards.
2. Build the ready image on Spark2 from `deployment/glm53/Dockerfile` and the
   final wheel. Re-run the updated `cpu_check.py` without GPU access and retain
   the initial report separately. Confirm the actual generated five-draft CLI
   parses on 0.32.0; record the final wheel/image hashes with that report.
3. Check native GPU loading and exact component outputs, then run a bounded TP2
   model startup/replay trial with original production restoration. No such trial
   is running or queued at this checkpoint. Require 42 expert layers / 84 exact
   scale tensors, 68 converted KDA matrices, both vocabulary methods, no retained
   dense vocabulary ownership and startup reports from both ranks.
4. Measure the selected Modal profile's actual allocations and matched C1
   latency/throughput before publishing a final memory or speed number. Stable
   C8/C16, production settings and 32k→128k→500k checks remain later replacement
   gates. Keep training stopped and the sealed test unopened.

Earlier dated entries below are retained evidence, not instructions to restart
cancelled training or old production-port trials. The whole-project result remains:
lossless packing works; MXFP8 KDA is lossy; the selected drafter improves this CT's
development speed; a faster, production-equivalent upstream-vLLM replacement is
not yet established.

**21 October 2026: whole-project assessment and easy vLLM 0.32.0 setup requested.**
The [consolidated assessment and deployment audit](GLM53_CURRENT_STATE.md) now
records the measured benefits, failed/unqualified gates and concrete packaging
work. Lossless compression saves **9.367 GiB total across TP2** on the same CT;
the separate CT MXFP8 comparison saves **14.877 GiB total versus matched ModelOpt**,
including format/quantization differences. These figures are not additive.
MXFP8 KDA removes the traced reconstruction work but remains lossy and fails the
original 1e-5 KL gate despite the higher observed full-GSM8K score.

Continue with **pretrained Modal FP8, five drafts at C1/C2 and three at C3–C16**;
training remains stopped. The user now explicitly wants easy deployment on
upstream vLLM 0.32.0. Documentation/source audit finds upstream GLM DFlash2 support,
but our existing plugin metadata is only a registration shim; compressed-scale
and vocabulary consumers, pinned Marlin builds and in-memory KDA conversion still
need packaging and version qualification. Target upstream vLLM plus one extension
and a pinned ready image. Do not claim zero-extension compatibility or a completed
0.32.0 port. No GPU experiment, production switch or training was launched here.
Production-equivalent serving, stable concurrency and long-context qualification
remain required. See the linked assessment for the ordered implementation/check plan.

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

First-token medians (code/chat/explanation): seven initial **242/174/197 ms**;
three **261/191/200 ms**; five **266/172/188 ms**; off **236/173/181 ms**;
seven reload **247/180/197 ms**. Five's code first-token increase against the
bracketed baseline is about 21 ms. Native accepted/proposed rates for code are
42.03% at seven, 72.02% at three and 53.07% at five; accepted tokens per round are
2.942, 2.160 and 2.654 respectively. Shorter drafts raise the percentage partly by
removing later proposals; elapsed serving speed, not that percentage alone,
determines the benefit. Median code decode elapsed time per native draft round
is 96.36/74.33/84.95 ms at seven/three/five; it includes host and drafter work.

Startup evidence confirms depths and no loaded drafter at zero, identical 68 KDA
rounding hashes, and captured graph shapes. This run does not add a GPU profile
attributing actual replay/kernel times. Recorded SM-clock medians span
2,463–2,496 MHz across arms/ranks; maximum recorded temperature is 75 C and all
samples report P0. Samples include warmups and brief idle gaps; they do not
independently prove absence of throttling. The retained clock/thermal summaries
and all numerical reviewer fields are **exactly identical on Spark and Mac**.

The target remains CT MXFP4 experts plus the previously measured **MXFP8 KDA**
variant, with exact vocabulary/scales. This run does not turn that later lossy
KDA variant into a lossless claim. Serving settings stay at 8k context, prefix
cache off, 4096 prefill budget, 16 slots and 6 GiB KV per rank. The sealed test is
not opened. Full production tuning, C8/C16 stability and long-context qualification
remain outstanding; the restored production configuration is unchanged.

All **152 retained payloads** verify independently on Mac and Spark2. Archive:
43,445,376 bytes, SHA256
`c88a8ea2c109a718555b07b172476649b3f2864547a30758c2b0dfddca8b301a`.
Plan SHA256 remains
`2ac2443f63ea259d31d4b1802d9284b48c9fd9aea47bf07d599377f254c41f72`.
Raw local evidence: `out/ct-draft-length-v1/retained`; Spark:
`full-model-v1/ct-draft-length-v1`; independent backup on Spark2:
`full-model-v1/ct-draft-length-v1-retained-backup`. Reports:
[measurement review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/ct-draft-length-v1-review.json),
[verification receipt](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/ct-draft-length-v1-independent-verification.json),
[telemetry](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/ct-draft-length-v1-telemetry-review.json)
and [restoration](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/ct-draft-length-v1-restoration.json).
Full pytest, Ruff and strict core mypy pass; the strengthened independent reviewer
also passes all ten focused tests. The earlier failures and raw output differences
remain retained.

**Practical next step:** keep expensive training held. Five drafts is the stronger
conservative code candidate on this screen; three is a chat/explanation candidate.
Profile the current CT serving path before changing production tuning: isolate
target/drafter/head/communication and host time, then test one compatible change
at a time. The CPU audit confirms CT explicitly sets `OMP_NUM_THREADS=4` while
production leaves it unset, and several production optimizations default off in
CT; their impact and compatibility are not yet measured. The earlier component
trace plan for target grouping differences (layers 0–4 around positions 87–94)
remains available, separately from speed tuning. No new GPU run or promotion is
queued after this completed bounded screen.

**21 October 2026, 07:29 UTC: CT-only draft-length comparison running detached.**
All ten CPU preflights pass, with 211 frozen runtime source bindings per node.
Launcher **1622309**, controller **1622330**, watchdog **1622434** start the
7/3/5/off/7 screen; current phase is `starting_s7a`. Original production is
**temporarily paused**. API 8042 and master 29715 are private to the experiment;
original 500k containers and configurations are retained for automatic restoration.
The outage deadline is **08:43:36 UTC**, followed by bounded restoration if needed.
The run survives laptop disconnection. No result yet; training held and sealed
closed. Read `full-model-v1/ct-draft-length-v1/controller.json` and per-arm
`*-measurement.json`; detached launch record is in sibling
`ct-draft-length-v1-launch/launch.json`.

A prelaunch evidence-retention amendment excludes `run-prepared-status.json`
from the immutable archive because the launcher updates it after finalization.
Only `finalize.py` changes; the plan, model/runtime and ten CPU checks remain
unchanged. Both hosts reverify all 45 sources. Original source/manifest copies and
`finalizer-amendment.json` retain this amendment. New finalizer SHA256:
`94f1cd53aad2c804c6a9b34e9a8d84bbdbf373d581697a8311fada36d5d3f6df`.

The prior measurements also show why acceptance is insufficient: median observed
code decode elapsed time per native draft round is about **68.3 ms production
versus 96.6 ms CT + Modal FP8**, despite CT needing fewer rounds (71.5 versus60.5
median rounds per 256 tokens). This ratio includes the whole draft/verify/host
path, not isolated target GPU kernels, and the setups/outputs differ. A read-only
public-environment audit finds many production tuning settings absent from the
CT launch specification; defaults/effective dispatch and their speed impact are
not established. Keep the running comparison unchanged; its purpose is to
measure draft depth on CT, not to mix in serving-policy changes.

**21 October 2026, 07:26 UTC: CT draft-length screen frozen; CPU checks in progress.**
The user redirects the proposed production depth comparison to CT, then authorizes
continued unattended work. No production depth experiment was launched. Use the
same CT target and Modal FP8 checkpoint at **7 → 3 → 5 → off → 7** drafts; the
final 7-draft load checks drift. Same eight development prompts, six seed-42
shuffled cycles per load, C1, 256 output tokens, temperature 0. Retain timing,
TTFT, TPOT, native round/proposal/position counters, all output tokens, graphs at
startup and clock/thermal telemetry. Different outputs are recorded rather than
hidden or assumed equal. No training, sealed-test access or automatic promotion.

Frozen revision `ct-draft-length-v1`, private API **8042**, master **29715**.
Plan SHA256 `2ac2443f63ea259d31d4b1802d9284b48c9fd9aea47bf07d599377f254c41f72`;
45 source payloads verify on Spark/Spark2. Target settings stay at the prior CT
screen: 8k context, prefix caching off, 4096 prefill budget, 16 slots, 6 GiB KV per
rank, the same canonical sparse/mHC/expert policies and lossless vocab/scales.
Depth 0 removes the speculative config and confirms no loaded drafter or native
speculative work. Depths 3/5/7 use the native configured width and assert actual
proposal counts, not post-hoc discarded tokens. The original 500k production
configuration is retained for automatic restoration. Bounded outage is 75 minutes
plus restoration, with the existing independent watchdog; estimated full screen
40–60 minutes. CPU preflight PID **1564948** is running; production remains online.
Ruff, strict core mypy and the full pytest suite pass. No GPU experiment has
started at this checkpoint. Read `full-model-v1/ct-draft-length-v1/prepare-launch.log`,
then `prepared.json` and `controller.json` when they exist.

**21 October 2026, 07:16 UTC: production acceptance screen and CPU cache audit complete.**
Production stays online throughout: same container identity, image, 500,000-token
context and serving settings; final health is 200. Run the same eight frozen
validation prompts twice (16 requests plus two excluded warmups), C1, 256 output
tokens, temperature 0, seed 42 and seven drafts. Counter deltas reconcile per
position and show exactly one completed request between idle boundaries. No
training, sealed-test access or serving change. The diagnostic has finished.

| Setup | Code accepted/proposed | Chat | Explanation |
| --- | ---: | ---: | ---: |
| Current production + Inco FP8 | 34.82% | 23.36% | 27.51% |
| CT MXFP8 + Inco FP8 | 38.81% | 25.60% | 28.16% |
| CT MXFP8 + Modal BF16 | 42.91% | 26.90% | 30.75% |
| CT MXFP8 + Modal FP8 | 42.03% | 27.60% | 30.21% |

Production's two cycle rates are code 35.11%/34.52%, chat 22.43%/24.34%, and
explanation 26.07%/29.08%. Its first code proposal survives 82.24% of rounds,
but all seven survive only 8.38%; the aggregate 34.82% is a different denominator.
Thus modest aggregate acceptance is present in production too, and these prompts
do not establish a CT acceptance deficit. Production is still faster in this
screen: code/chat/explanation 49.09/41.47/41.71 tok/s, versus Modal FP8
41.18/30.70/32.21. These are different targets, trajectories, tuning and loads;
this is not a controlled attribution to compression or a new speed qualification.
**All eight production outputs and counters differ between cycles**, despite the
fixed seed; first token differences occur at zero-based indices 4–104. CT's prior
six within-load cycles repeat exactly. Prefix caching stays enabled on production
and was disabled in CT; its role in these differences is not isolated. Two cycles
and eight reused development prompts do not establish broad quality or speed.

All ten new evidence payloads verify independently on the Mac. Report:
`diagnostics-v1/production-drafter-acceptance-v1-review.json`. Raw evidence on Spark:
`full-model-v1/production-drafter-acceptance-v1`; local:
`out/production-drafter-acceptance-v1/retained`. Archive 433,015 bytes, SHA256
`015aff8346db0b1967fbf9fa46a6d2f22d582d07525668e1a7c52ff12ac1acb3`.

Separately, the saved native trace now passes **200 valid convolution-history
carry checks**, with accepted-count, position, slot and accepted-input metadata
consistent on both captures and ranks. Bind cache semantics to the stopped
runtime's exact convolution/KDA/recurrent sources and frozen width-4 config.
Reconstruct only accepted per-token QKV and recurrent-state writes from the next
snapshot; all 376 logical layer/position repeat comparisons are bit-identical.
Layer 0 QKV/state matches between arms throughout positions 56–102. Layer 4
QKV/state matches through 91, then first differs at **92** on both ranks and
captures. The QKV difference precedes layer 4's convolution/recurrent update;
its later cache difference alone is not a rollback defect. Other-layer cache
issues and arithmetic remain unseparated. Next trace should cover component
boundaries in layers 0–4, particularly layer 1/2 KDA, layer 3 MLA, mHC/experts
and layer 4's input projection around positions 87–94. No follow-up GPU trace
is launched or queued. Reports: `...-cache-review.json` and updated
`...-followup-plan.json`. Ruff, strict core mypy and the full pytest suite pass.

**21 October 2026, 06:56 UTC: paired native trace complete and independently reviewed.**
All eight before/capture/repeat/after requests reproduce their arm's frozen
64-token output and counters. All captured inputs, logits, final-hidden states,
arm-specific auxiliary features and proposals repeat exactly on both GPUs:
17 frames per Inco capture, 20 per Modal capture, **148 frames total**. All
**406 evidence payloads** independently verify on Mac and Spark2. Original
production restores healthy at **06:49:03 UTC**, with unchanged identities and
**500,000 context**. No experiment is running; training is held and sealed test closed.

The first **37 output decisions (indices 0–36)** have identical target logits
and final-hidden states. The first difference is output **38 / index 37**, at
absolute input position 92: Inco processes row1 of a forward beginning at 91,
Modal row0 of one beginning at 92. Both use native eight-row graphs. The token
still agrees (1477); its logits are 33.75/34.0 and decision KL is 7.5371e-8.
At output **48 / index 47**, both arms consume the same eight input IDs and
positions 97–104 but the target scores already differ before sampling:

| Token ID | Inco target logit | Modal target logit |
| --- | ---: | ---: |
| 11 | 26.500 | 25.875 |
| 198 | 26.125 | 26.375 |

Each arm selects its own unique argmax. This boundary therefore reflects changed
target computation/state, rather than different selection from identical logits.
Both ranks agree on the decision evidence. The local decision KL 0.094529 is
**not** a corpus quality metric. Different drafters' auxiliary layers are never
compared as equal layer boundaries.

The supplemental state audit narrows the interval: selected layer 0 and layer 4
recurrent read states match at every common boundary through absolute position 87,
including earlier different batch groupings. At 94 and 97, layer 0 still matches
while layer 4 differs. Layer 4 may inherit an upstream difference; no particular
kernel, cache defect or compression defect is established. Convolution-history
valid slices have not yet been independently compared. Next: audit those slices
and rollback metadata, then trace layers 0–4 around positions 87–94 and replay the
first changed component with equal past inputs/state. The follow-up is specified,
**not launched**; no new production pause is queued. A fix must pass native
output identity and measured serving cost before drafter promotion.

Retained evidence: `diagnostics-v1/drafter-pretrained-verification-v1-review.json`,
`...-batch-boundaries.json`, and `...-followup-plan.json`. Immutable archive:
5,243,463,145 bytes, SHA256
`f549cfe5bdc0f5713ca4d5e2e2515af09e8fc229595fd8c07b9081fdbbf2fa83`;
local `out/drafter-pretrained-verification-v1/retained`; Spark2 sibling
`drafter-pretrained-verification-v1-retained-backup`.

**21 October 2026, 06:44 UTC: preliminary Inco boundary review finds a unique target winner.**
Four selected rank-0 frames independently match their payload hashes: first
output and output index 47, in both captures. At index 47, token **11** has
logit **26.5**, versus **26.125** for token **198** (Modal's previously observed
choice). Inco selects the unique target argmax, with a **0.375** margin; both
captures agree. All selected input, position, full-logit, final-hidden, proposal
and five auxiliary-feature tensors repeat byte for byte in these frames. The
observed original logits buffer also remains unchanged after sampling. This
rules out an exact tie on the Inco side at this boundary, but does not distinguish
changed Modal target scores from a verifier decision. The paired Modal trace,
full evidence review and restoration are pending; no responsible kernel or
quality conclusion is established. Retained report:
`diagnostics-v1/drafter-pretrained-verification-v1-inco-boundary.json`.

**21 October 2026, 06:41 UTC: first native trace arm completes; Modal BF16 is loading.**
Original/Inco arm a1 completes all four 64-token before/capture/repeat/after
requests. Every output matches the frozen native prefix; speculative counters
match across all four controls. Both GPUs retain **34 trace frames each**
(1,383,027,196 bytes per rank). This verifies the API/counter observer controls;
independent tensor-repeat and cross-arm analysis remain pending. Controller is
`starting_b1`, loading Modal BF16 with the same CT target. Production remains
temporarily paused under the active restoration watchdog. No cause of the
original/Modal divergence is established yet. Retained progress:
`diagnostics-v1/drafter-pretrained-verification-v1-a1-progress.json`.

**21 October 2026, 06:35 UTC: final speed repeat stopped; native trace loads first arm.**
The user prioritizes the output-divergence trace over repeating the speed baseline.
At 06:29:04 UTC, stop the superseded trace waiter PID 987188, then signal speed
controller 728927 through its normal restoration handler. No a2 measurements
were taken. All three completed arms retain **48/48 requests** and exact within-arm
token/counter repeats. Original production is verified healthy at **06:32:09 UTC**,
with its original identities and **500,000 context**. Finalization succeeds;
`failed_restored` records the intentional signal-15 stop, not a model failure.
The immutable timing archive independently verifies all **132 payloads** on the
Mac and both Sparks (38,751,310 bytes; SHA256
`e8b760d95499dcc07d4faee67161ed6cdee2f6a59c36c0011c6fe37424042baf`).

Completed Modal FP8 medians: code **41.18 tok/s / 257 ms TTFT**, chat
**30.70 / 184 ms**, explanation **32.21 / 189 ms**. All **48/48** Modal FP8
outputs exactly match the corresponding Modal BF16 outputs. Both Modal variants
match Inco on only **6/48 requests (one of eight cases)**. The FP8 conversion is
therefore not the source of this observed original/Modal mismatch. Without the
reload drift control, and with different trajectories, no speed gain is qualified.
Retained summary: `diagnostics-v1/drafter-pretrained-modal-v1-retained-summary.json`.

Replacement launcher **PID 1180511** starts detached at **06:33:03 UTC**, after
verifying the explicit stop receipt, restoration, retained evidence and all
unchanged frozen trace sources. CPU preflight passes (211 runtime bindings per
rank); controller **1186244** and watchdog **1186334** now load **a1 / Inco FP8**.
Production is temporarily paused again. GPU captures have not completed yet. The launcher accepts only this specific expected cancellation; arbitrary
failures still block it. Both nodes verify its SHA256
`97eb5f6e315f78c72b929efc2d5180817ce8404e8092bf0cba3076d8e5b92fb7`.
Read diagnostic `after-priority-status.json`, then `controller.json`; old
`queue.json` is superseded. The frozen two-arm, eight-request native trace and its
30-minute restoration watchdog are unchanged. All tests, Ruff and strict core
mypy pass. No training restart, sealed-test access or promotion.

**Acceptance interpretation:** the 42.03% Modal FP8 code rate counts all seven
proposals, not first-token agreement. Actual accepted-prefix survival per position
is **80.4%, 58.1%, 47.7%, 39.2%, 29.2%, 22.7%, 16.9%**. The first mismatch truncates
the accepted draft prefix; only 16.9% of code rounds retain all seven proposals.
Chat reaches its first/seventh draft position in 71.6%/5.1% of rounds. Per-position
counters independently sum to total accepted counts in all nine domain/variant
groups. This explains much of the low-looking aggregate; it does not locate why
predictions diverge. BF16/FP8 proximity argues against FP8 being the main observed
cause. CT target mismatch, domain difficulty and runtime differences remain
unseparated; the first-divergence diagnostic addresses the latter. Draft lengths
3/5 are a subsequent measured-cost option after correctness is understood.

**21 October 2026, 06:24 UTC: acceptance remains modest; FP8 is close to Modal BF16.**
Live FP8 progress 37/48 requests. First matched development-round counters,
accepted/proposed draft tokens, original Inco / Modal BF16 / Modal FP8:
code **38.81% / 42.91% / 42.03%**, chat **25.60% / 26.90% / 27.60%**,
explanation **28.16% / 30.75% / 30.21%**. Modal FP8 accepts **2.94 / 1.93 / 2.11**
tokens per seven-token draft round in those domains. These are small-sample
acceptance observations, not a causal gain against Inco's different output
trajectories. BF16 already has modest acceptance; its FP8 conversion is not the
main observed limitation. Target mismatch versus prompt difficulty is unseparated.
The queued identity diagnostic addresses why target decisions differ; resolving
that issue does not itself establish improved acceptance. Retained counters:
`diagnostics-v1/drafter-pretrained-modal-v1-acceptance-progress.json`.

**21 October 2026, 06:22 UTC: FP8 drafter runs; first round matches Modal BF16 outputs.**
`c1` completes 22/48 timed requests. In the first full eight-prompt round,
FP8 and BF16 Modal produce **identical 256-token outputs on all eight cases**;
both match original Inco on only one case. Thus the original/Modal mismatch is
already present before FP8 conversion; this does not yet identify its cause.
First-round medians versus Modal BF16: code **40.81 vs 40.68 tok/s (+0.31%)**,
chat **31.00 vs 29.65 (+4.54%)**, explanation **32.49 vs 31.91 (+1.83%)**.
These are preliminary one-round values; full repeats/reload controls are pending.
The detached native-decision diagnostic remains queued behind the unchanged
timing run. Snapshot SHA256
`9b407d90718716949c2d232212b8e756085bfa8f2c0eef9726fd6a702c465140`;
summary `diagnostics-v1/drafter-pretrained-modal-v1-c1-progress.json`.

**21 October 2026, 06:19 UTC: first-divergence diagnostic implemented and queued on Spark.**
At 06:16:17 UTC, detached queue **PID 987188** starts independently of the Mac.
Both nodes verify all **46 frozen files**. Diagnostic revision
`drafter-pretrained-verification-v1`, API **8041**, master **29714**, plan SHA
`ae1f8d983ada1955fbf94d31ce72ac84e9292ac55ad2909271b555c0150dfe29`.
Current state: `waiting_for_timing_and_restoration`. It requires the existing
timing run to complete, both observed prefixes to repeat six times, successful
finalization and healthy original production before its own preparation/outage.
No concurrent GPU experiment or change to the timed comparison. The diagnostic
uses original/Inco FP8 and Modal BF16, one development case, 64 output tokens,
before/capture/repeated-capture/after: eight requests plus warmups in total.
The passive observer now supports five/six native feature taps, retains actual
target graph identities, pre/post-sampler logits, draft proposals and layers0/4
pre-forward state. Per-arm/rank trace bound 8 GiB; 30-minute restoration watchdog.
The independent reviewer aligns decisions by accepted history and actual input
position; it refuses wrong-prefix comparisons. Full pytest, Ruff and strict core
mypy pass. CPU CLI preflights and GPU trace are pending. Capture/training remain
stopped; sealed set closed. Read `drafter-pretrained-verification-v1/queue.json`.

Modal BF16 now completes all **48 timed requests**, with exact token/counter
repeats in all six trials. Code **40.83 vs 39.88 tok/s (+2.39%)**, chat **29.30 vs
29.56 (-0.90%)**, explanation **31.58 vs 31.20 (+1.22%)** against the first Inco
load. **7/8 output sequences still differ**, so changed trajectories confound
these timing/acceptance comparisons. Modal FP8 is starting; baseline reload is
pending. No candidate speed or quality qualification is claimed. Evidence:
`diagnostics-v1/drafter-pretrained-modal-v1-b1-summary.json` and
`diagnostics-v1/drafter-pretrained-verification-v1-launch.json`.

**First-divergence follow-up prepared locally; no new GPU diagnostic launched.**
Case `opencoder-evol-92378` first differs at output index 47 after an exact shared
47-token prefix. Freeze the observed first 64 tokens and trace native decisions
with within-arm before/capture/repeat/after controls. Inspect logits and verifier
decisions first, then actual cache state and the earliest changed layer if needed.
Existing context instrumentation assumes five auxiliary features; Modal has six,
so it must be adapted before reuse. Current timing comparison is unchanged;
the new diagnostic requires its completion and verified production restoration.
See [diagnostic specification](GLM53_PRETRAINED_DRAFTERS.md#first-divergence-diagnostic-specification).

**21 October 2026, 06:09 UTC: Modal BF16 loads and runs; output identity fails early.**
`b1` has completed 16/48 timed requests. In the first matched eight-prompt round,
code median throughput is 39.45 → 40.68 tok/s (+3.13%), chat +0.31%, explanation
+2.22%; median code verification rounds fall from 65.5 to 60. However, **7/8
outputs differ from the original-drafter baseline**, first diverging at token
indices 47–206. The completed second round repeats Modal tokens and counters
exactly; the full six-round repeat control is still pending. Different generated
trajectories confound speed/acceptance comparisons, so these preliminary numbers
do not qualify the candidate. This is the unquantized BF16 Modal drafter with the
unchanged CT MXFP4-expert/MXFP8-KDA target; Modal FP8 has not yet run. The cause of
cross-arm divergence is unlocalized. Continue the frozen comparison, retain all
failures, and inspect paired verification behavior before any promotion. Capture
and training stay stopped; sealed test closed; restoration watchdog active.
Evidence: `diagnostics-v1/drafter-pretrained-modal-v1-b1-progress.json`, backed by
the retained in-progress measurement SHA256
`d6178e35768dc8544dffed756beeee97efca3bd69508ae7dbfd4ef5e685769ec`.

**21 October 2026, 06:01 UTC: first pretrained-screen baseline completes.**
`a1` (CT MXFP8 target + original Inco FP8 drafter) completes all 48 measured
requests and two excluded warmups. All six repetitions match output tokens and
speculative counters exactly. Median end-to-end throughput: code **39.88 tok/s**,
chat **29.56**, explanation **31.20**; median TTFT **277 / 176 / 197 ms**.
These are first-load baseline results on the eight frozen development prompts,
not candidate gains or a comparison with earlier prompt suites. Controller phase
is now `starting_b1` (Modal BF16); Modal FP8 and the baseline reload follow.
Production remains temporarily paused under the comparison restoration watchdog.
Capture and training remain stopped; sealed test stays closed. Retained measurement
SHA256: `443d95e4b3a1ef826299ae55dc2b7c3263a4e01bcb90511035860f463340f39e`;
summary: `diagnostics-v1/drafter-pretrained-modal-v1-a1-summary.json`.

<!-- CAPTURE_USER_STOP_BEGIN -->
**21 October 2026: user stops the long capture to prioritize pretrained BF16/FP8.**
At **05:44:17 UTC**, verified waiting queue PID 677064 is stopped, then the
capture controller PID 3702964 receives SIGTERM through its restoration handler.
Capture stops after **1,095 prompts** (plus six before/after control requests).
All three completed controls pass output tokens and speculative counters.
Original production restores at **05:46:55 UTC**, healthy with exact original
identities and **500,000 context**. `failed_restored` and the signal-15 error
are expected cancellation records, not a new model failure. Finalizer exits 0
and records `partial-capture.json`; no complete training manifest is published.

Both hosts independently rehash all **1,088 accepted payloads / 17 paired chunks**.
They retain identical **13,097,763,612 bytes** and **319,628 feature rows** each.
Paired NumPy audits cover **13,091,962,880 BF16 elements across both ranks**, zero
differences. All 233 remaining raw files / 157,500,812 bytes per node are left
in place. The seven captured but unexported prompts are not claimed verified.
CPU training remains held; future partial-data use requires its own valid plan.

New launcher **PID 716651** (`launch_after_capture_stop.py`) validates the explicit
stop receipt, expected cancellation, healthy 500k restoration, both complete
partial-data audits and unchanged pretrained source/plan hashes. It then starts
the existing preparation and bounded restoring controller. Current state file:
`full-model-v1/drafter-pretrained-modal-v1/after-stop-status.json`. The old
`queue.json` is stale; its waiter has been stopped. The initial phase is `prepare`;
there is no remaining dependency on completion of the 4,200-prompt capture.
Original/Inco FP8 → Modal BF16 → Modal FP8 → original/Inco FP8, unchanged
eight development prompts and six trials/load, same target and seed. Plan SHA
`52ed3ed2ad97f0de5b05c06342ce04ccc9c866ff82f069d519a6d054a4b8c1e8` remains fixed.
The comparison has its own 60-minute restoration watchdog; no training or
candidate promotion is authorized by this launcher. Sealed test stays closed.

Local evidence: `out/drafter-alternatives-v1/user-stop/`; committed summary
`diagnostics-v1/drafter-capture-user-stop-v1-review.json`. Ruff, strict core mypy
and the full pytest suite pass, including cancellation/restoration guard tests.

**05:48 UTC:** all eight CLI/config CPU preflights pass, with **211 runtime
bindings per rank**. New comparison controller **728927**, restoration watchdog
**728928**; launcher 716651 is `run-prepared`. The controller has begun stopping
original production for the shorter comparison. Thus the 05:46:55 restoration
was verified, and production is temporarily paused again for the new authorized
bounded test. No pretrained inference/acceptance/speed result is available yet.
<!-- CAPTURE_USER_STOP_END -->

<!-- PRETRAINED_DRAFTER_BEGIN -->
**21 October 2026, 05:39 UTC: check pretrained drafters before further adaptation.**
The user requests a cheaper pretrained route and clarifies **FP8**, not FP6.
Automatic CPU training is held: waiting continuation v2 PID **80216** received
SIGTERM at 05:30:51 UTC before any optimizer update; its keep-awake ends with it.
`out/drafter-full-training-v1/continuation-v2/held-for-alternatives.json` is the
current continuation state; the previous `continuation-status.json` is stale.
The original capture controller **3702964** continues unchanged with no error.
Latest read: 1,047 total requests, three completed controls and 16 paired chunks.
Production remains temporarily paused under its original restoration watchdog.

Modal’s pinned pretrained DFlash2 strictly loads all 96 BF16 tensors on CPU.
Both installed images resolve its six causal layers, 4,096 sliding window and
24,576-wide context projection correctly. Its feature taps differ from Inco’s;
the existing capture cannot provide an offline acceptance comparison.
The separately exported block-FP8 version takes 6.606 seconds on CPU and shrinks
checkpoint bytes **2,778,461,544 → 1,621,120,728**. All 54 unquantized tensors
remain exact; 42 backbone matrices are quantized. Independent NumPy validation
checks 6,881,280 sampled rounding decisions with zero differences. This is
lossy drafter conversion, not a target change or a serving-speed result.

Prepared comparison: `out/drafter-alternatives-v1/prepared-fp8/`, revision
`drafter-pretrained-modal-v1`, private API **8040**, master **29713**. Plan SHA
`52ed3ed2ad97f0de5b05c06342ce04ccc9c866ff82f069d519a6d054a4b8c1e8`.
Arms: current Inco FP8 / Modal BF16 / Modal FP8 / current Inco FP8. Eight
development prompts × six repetitions per arm; two excluded warmups; C1,
256 outputs, fixed seven drafts, seed 42, unchanged target/numerical policy.
The queue may start only after complete verified capture and healthy original
restoration; failed capture stops the queue. The new controller has a
60-minute restoring watchdog. No sealed test, auto-training or deployment.
See [pretrained alternatives](GLM53_PRETRAINED_DRAFTERS.md) for measured CPU
evidence, remaining gates and all artifact paths. All required repo checks pass.
At **05:41:53 UTC**, queue PID **677064** is started on Spark and verified in
`waiting_for_verified_capture_and_restoration`. All 44 frozen source files and
all four checkpoint/config files independently hash-verify on **both Sparks**.
Remote run root is `full-model-v1/drafter-pretrained-modal-v1`; read `queue.json`
until the restoring controller starts. Both old CPU PIDs 80216/80217 are absent.
Latest capture read at 05:40:44 UTC: **1,060/4,200 prompts** plus six controls,
no error, 16 paired chunks. Approximately 5–6 hours remain at this pace.
The pretrained screen needs none of this capture; it is waiting to preserve
the originally authorized full dataset as a training fallback.
<!-- PRETRAINED_DRAFTER_END -->

<!-- FULL_CAPTURE_PROGRESS_BEGIN -->
**21 October 2026, 05:23 UTC: capture continues unchanged; queued CPU training accelerated.**
The original controller PID 3702964 and client PID 3763665 remain running under
the original capture plan. Latest read: **892 / 4,200 prompts**, 13 paired chunks,
both completed repeat controls exact, no error. Production remains paused under
its original watchdog. No capture requests or accepted data were restarted.

At the user's request, an isolated Mac audit tested 4/8/12 CPU threads on the
retained training step. Interleaved medians are **2.483 / 2.066 / 1.805 seconds**;
all losses, proposals, clipped gradients, FP32 masters and BF16 export hashes
match. The 12-thread full 32-update replay also passes: baseline, all update
metrics, all validation/selection results and all four checkpoint files are exact.
The replay completes in 108.116 seconds. This is a CPU execution audit, not a
serving-speed result; full-corpus training time remains unmeasured.

The idle Mac continuation was replaced only after this replay and checks passed.
Active continuation: `out/drafter-full-training-v1/continuation-v2/`, **PID 80216**,
keep-awake **PID 80217**, frozen at **12 CPU threads**. It retains the identical
capture plan and training protocol hashes and all completion/restoration gates.
The previous launcher PID 19984 is stopped; its directory and supersession receipt
remain. The new job is `waiting_for_verified_capture`; no optimizer updates have
started. Expect roughly **3–4 hours CPU training** instead of 4–5 if the observed
gain transfers to the full corpus, with about **5.5–6.5 hours capture remaining**.

Capture timing audit: mean request duration **3.360 seconds** plus **two seconds
of fixed client waits per request**, followed by incremental export/audit overhead.
The fixed waits are an optimization opportunity, but the current client has no
qualified resume path and is left running. No concurrency or numerical policy
changes are applied. Source files, timing/replay reviews and the replacement
launch receipt are retained under `diagnostics-v1/drafter-cpu-threads-v1-*` and
`drafter-full-capture-v1-throughput-review.json`. Ruff, strict mypy and the full
pytest suite pass (pytest uses the worktree on `PYTHONPATH`). Sealed test closed.

**21 October 2026, 05:16 UTC: 820 / 4,200 prompts captured (19.5%); training updates have not started.**
The live controller remains `measuring_ct`, with no reported error. Both completed
repeat controls pass exact tokens and speculative counters. Twelve paired receipts
now cover the first 768 prompts; this status read does not replace the independent
576-prompt host verification recorded below. The local CPU continuation remains
`waiting_for_verified_capture`; production is still paused under the restoration
watchdog and must restore before CPU training starts. Sealed test remains closed.

Observed capture pace is approximately 10 prompts/minute, including periodic export
and audit. Estimate **5.5–6.5 hours more capture**, then transfer/final verification
and **4–5 hours CPU training**: roughly **10–12 hours to the first trained candidate**
if the current pace holds. This is an estimate, not a measured training runtime;
native reload, acceptance and serving-speed qualification remain subsequent gates.
Live snapshot: `out/drafter-full-capture-v1/runtime/progress-live-20261021-0515.json`
(actual observation 05:16:26 UTC).

**21 October 2026, 04:53 UTC: capture continues; nine chunks / 576 prompts independently verified on both Sparks.**
Both nodes retain identical accepted exports: 7,052,377,021 bytes per node and
172,101 feature rows. Independent NumPy audits cover **7,049,256,960 BF16 elements
across both ranks, zero differences**. Host verification rehashes all 576 payloads
per node and all paired receipts; the nine paired-receipt hashes match across nodes.
Verified cleanup releases 14,804,804,308 raw staging bytes per node while retaining
accepted exports and raw metadata. About 90 GiB remains free on Spark and 1,020 GiB
on Spark2. Both before/capture/after controls encountered so far pass exact output
tokens and speculative counters. Controller remains `measuring_ct`, no error.

Production remains temporarily paused under the restoration watchdog. The local
CPU continuation is alive in `waiting_for_verified_capture`; it will start only
after complete verification and restored production. At the observed pace, allow
roughly **6–7 more hours capture**, followed by **4–5 hours CPU training**; prompt
mix can change this estimate. Sealed test remains closed. Review:
`benchmarks/.../diagnostics-v1/drafter-full-capture-v1-progress-0576-review.json`.

Adaptation scope: DFlash2's **4096 × 20480 context projection (`fc.weight`),
83,886,080 parameters**, is the only trained weight. It maps the five CT feature
taps into the existing drafter. The resulting checkpoint is specific to this
CT MXFP4-expert / MXFP8-KDA target; its architecture remains DFlash2. Training,
native reload, output identity and acceptance/speed qualification are still pending.
<!-- FULL_CAPTURE_PROGRESS_END -->

<!-- CORRECTED_CAPTURE_BEGIN -->
**21 October 2026, 04:02 UTC: full capture running; first 64-prompt chunk independently verified.**
Both ranks export identical 791,123,554-byte accepted datasets (19,306 feature
rows). Independent NumPy audits compare **790,773,760 BF16 elements in total,
zero differences**. Host checks rehash all 64 exported payloads on each node.
Both rank receipts and the paired receipt agree. Each node releases 1,944 raw
staging files (1,643,602,712 bytes) only after verification; accepted exports and
all raw metadata remain. Later prompts are now capturing. First repeat control
passes exact tokens and counters. No controller error has appeared.

The actual full-run startup independently verifies both qualified ordering
policies, the existing expert policy through 16 target rows and all 39 startup
specializations on both GPUs. The first chunk takes **386.3 seconds (6.44 minutes)**,
including capture/export/audit. Capture estimate is **7–8 hours**, then **4–5 hours
CPU training**; later domain mix can change it. Production remains temporarily
paused during GPU capture and automatically restores before CPU training.

Launch checkpoint `90f05d2` is verified in Spark2's Git backup; bundle SHA
`421860ab67eb7839f9350ea19204da44d805c0c412c85259c27b3b65c99f70f7`.
All 15 frozen continuation files also independently verify on Spark2; archive SHA
`0e73bf33cd72d0d230cdc2b3edf80cce0b078317daf591d9479d3c3a15fc85b8`,
at `drafter-full-capture-v1/continuation-sources-v1.tar.gz`.
First-chunk and startup reviews are retained in `out/drafter-full-capture-v1/runtime/`
and `benchmarks/.../diagnostics-v1/`. The CPU continuation remains waiting for
complete verified capture/restoration. Native acceptance and speed remain unproven;
sealed test closed.


**21 October 2026, 03:49 UTC: fresh smoke passes; full 4,000/200 capture launched.**
All 12 requests complete, including exact before/capture/after tokens and
speculative counters for the 3,973-token train and 2,248-token validation controls.
Original production is restored and healthy with 500,000 context. All **1,203**
archived payloads independently verify on Mac and Spark2. The accepted export
contains 7,468 feature rows (306,018,956 bytes); both GPUs are bit-identical.
An independent NumPy audit compares **305,889,280 BF16 elements with zero
differences**. Replaying the incremental full-capture exporter on both ranks
produces exactly the same eight payload hashes and passes its independent audit.

Smoke plan SHA `ddebc9515d29877255cfcd6aa3432fea6260f734bebafd825777adfaea344405`.
Archive: 892,749,347 bytes, SHA
`53e3df7f3854f5f9e060717038035adfcc2b9a1bc2d31f61930b7d1677fbfd5f`.
Spark2 backup: `full-model-v1/drafter-fresh-capture-smoke-v4-retained-backup/`.
Reviews and receipts are retained in `out/drafter-fresh-capture-smoke-v4/` and
`benchmarks/.../diagnostics-v1/`. Source commits `d943a89` and `db25233` are
verified in Spark2's Git backup.

The frozen full capture is `drafter-full-capture-v1`: **4,000 train / 200 validation**,
12 before/capture/after controls, **4,224 requests**, original fixed-seven drafter,
128 output tokens, same qualified sparse-prefill/decode and expert/mHC policies.
Plan SHA `4d97fd7c23e359c0f76fa95f08bf7da2a5323dddcfd3bccb5b7294058411aacd`,
52 source files; private 8037 / master 29710. Both CPU preflights pass; the full GPU capture is launched.
Sealed test stays closed. Launcher PID 3702963; controller PID 3702964;
restoration watchdog PID 3702965. Original production is temporarily paused.
Run root on both Sparks:
`/home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/full-model-v1/drafter-full-capture-v1`.
Read `controller.json`, `ct-measurement.json` and `ct-client.log` there for progress.
Emergency restoration is the run's `python3 -S -B control.py --restore`;
normal completion/failure automatically restores original 500,000-context service.

Accepted features are exported and independently NumPy-audited every 64 prompts
on each GPU. Raw staging is released only after both payload hashes and history
statistics match; accepted copies, raw metadata and receipts remain on both Sparks.
Estimated accepted data is 41.53 GiB per node; free space before launch is about
97 GiB on Spark and over 1 TiB on Spark2, with a 20 GiB guard. The controller has
a 12-hour client deadline plus startup/restoration allowance and a separate
restoration watchdog.

The conditional CPU continuation is frozen at
`out/drafter-full-training-v1/continuation-v1/` (15 files; running since 03:48 UTC).
Mac PID **19984**, keep-awake PID **19985**; current phase
`waiting_for_verified_capture`. Local machine-readable progress is
`out/drafter-full-training-v1/continuation-v1/continuation-status.json`;
`continuation.log` records transitions and `training.log` will contain CPU updates.
Only after complete capture, restored production and identical verified manifests
on both Sparks does it transfer accepted features to Mac, verify every payload
against the frozen corpus and run up to 4,000 updates of **fc.weight only**.
Protocol SHA `43d2261165f800e9e54e553a000495d84a36777ba73b21d284bee5992c2d11ed`.
It conditionally exports an offline-selected projection into the original native
FP8 checkpoint and verifies result retention on Spark2. It never switches serving.
First full chunk takes 386.3 seconds including export and audit: plan roughly
7–8 hours GPU capture plus 4–5 hours CPU training, subject to later prompt mix. The old pilot remains speed-neutral;
no new native acceptance or serving gain is claimed.
<!-- CORRECTED_CAPTURE_END -->

<!-- LONG_DECODE_BEGIN -->
**20 October 2026, 22:37 UTC: combined sparse-prefill/decode ordering correction passes the long-prompt C1 repeat gate.**
`drafter-canonical-decode-v1` completes all six requests and restores original
production: `complete_restored`, control/finalizer exit 0, no error, original
container identities/configurations, health 200 and **500,000 context**. No GPU
experiment is currently running. All **430 archived payloads** independently
verify on Mac and Spark2; the independent NumPy review passes.

Observed result, original drafter / fixed seven drafts / 3,973 input tokens /
128 output tokens, with per-operation graph observers removed:

- All six before/four-capture/after requests have identical output tokens and
  speculative counters; the unchanged API repeat gate passes.
- All **12 capture-pair comparisons** across both TP ranks have identical full
  prefill features at layers 5/14/24/33/42 and identical accepted-history features.
- Each comparison has 128 shared decision positions and **zero different target
  logit rows**; all initial drafter fields are bit-identical.
- A separate raw-field digest screen verifies all **176 frame payloads**
  (22 × four captures × two ranks). Every numerical tensor field matches across
  each rank's captures. Only `context_slot_mapping_0`, the physical cache-slot
  addresses allocated to different requests, differs.

The correction preserves native selected membership and all -1 masking, then
sorts selected integer pool indices. Canonical prefill remains enabled; decode
sorting applies only to **rows 8 / next_n 1 / k 512**. Other decode shapes retain
native behavior. No model or drafter weights change; no external code is copied.

Both GPUs pass retained-score and mixed-heavy-padding component fixtures:
128 native membership checks, 128 exact canonical repeats and 128 exact canonical
graph replay checks in total. Poisoning unused columns with large values and
NaNs leaves selected membership unchanged. Independent NumPy checks validate
pool counts, uniqueness, causal bounds, sentinel masking and pool-to-token
expansion against retained native outputs.

| Decode component fixture | Stable timing controls, rank 0 / rank 1 | Added ms per call, rank 0 / rank 1 |
| --- | --- | --- |
| Retained native scores | 5/6 / 4/6 | 0.01553 / 0.01559 |
| Mixed heavy padding | 5/6 / 5/6 | 0.01547 / 0.01545 |

These are CUDA-graph component costs: 100 calls per replay, 32 replays per timing
sample, six interleaved trials per fixture with clock/thermal telemetry. **19/24**
repeat controls pass; five failed 2% controls remain retained and excluded.
This does not measure end-to-end serving overhead or demonstrate a drafter speed
improvement. The prior prefill correction's component cost remains separately
recorded below; no full-model C8/C16, cross-drafter or long-context pass is claimed.

**Next:** repeat the eight-prompt fresh-capture smoke under these exact numerical
policies, independently export/audit accepted features on both ranks, then start
the frozen 4,000 training / 200 validation capture and context-projection run
only if that smoke passes. Full capture/training has **not** started. The old
32-update pilot remains speed-neutral and unqualified. The sealed test stays
closed, all quality/performance thresholds unchanged.

Current plan SHA
`a5b4bb6335442edfd71e65148e9eab243f33e20214f69931cdcc3bbbd19e460f`;
42 frozen files, private port 8035, controller launcher PID 2479761.
Archive: **1,916,526,830 bytes**, SHA
`51dc7f1d46e59be73cbc6e444432efd12cc20f72f39b3023c3c377eecbe9d638`.
Spark2: `full-model-v1/drafter-canonical-decode-v1-retained-backup/`.
Reviews, raw-field screen and backup receipt are retained in
`out/drafter-canonical-decode-v1/` and `benchmarks/.../diagnostics-v1/`.
Qualified source checkpoint `e2cbcab` and bundle
`6fb158f2b4456777b3e9cc63a9c97397ceb00d03e75d1e6346d4681583714350`
are verified in Spark2's Git backup.

**Preserved diagnosis:** the prior `drafter-long-decode-v1` verifies all 421
payloads on Mac/Spark2. Every one of 12 comparisons first differs at
`layer3.native_decode_topk#0/output`, with identical **7,952 consumed scores**
and selected membership; unused score columns differ and are excluded. The
per-row bounds are `[993,993,994,994,994,994,995,995]`. Only selected order changes,
in all eight rows. Its observed-graph failure remains recorded; the successful
correction run removes those per-operation observers. Prior archive:
2,048,667,852 bytes, SHA
`0e2530bdd1e6322c2b0af5dfe501b66f1cf88238e7ba4977af5462468e9fc193`.
Prior plan SHA
`6d51d9069e98925f378ad074b6444f97764886650181d2eef00d9a14d23aa578`.
<!-- LONG_DECODE_END -->

<!-- CANONICAL_PREFILL_BEGIN -->
**20 October 2026, 22:07 UTC: prefill correction independently verified; remaining failure starts at first decode.**
`drafter-canonical-prefill-v1` completes all six requests and restores original
production (`complete_restored`, no error, health 200, original 500,000 context).
All 413 evidence payloads independently verify on Mac/Spark2. The fixed native
prefill top-k still selects the same pools, then sorts selected integer indices;
all -1 padding and selected multiplicity are preserved. No weights change.

All 12 comparisons across both TP ranks now have **bit-identical full prefill
feature arrays at layers 5/14/24/33/42, initial target logits, drafter hidden
state and first proposal**. Before the correction every pair differed already
in prefill. Full-model repeatability still fails: all 12 pairs first differ at
the next forward, **decode input position 3,973**, by layer 5 and in target logits,
with identical complete eight-token input IDs/positions. One captured response
diverges at output 80; speculative counters differ. Full training remains gated.

Both GPU component gates pass at 8/16/64/128/3,973 rows: ten fixtures, 320 exact
canonical repeats and 320 exact graph replays, unchanged selected token sets,
invalid masking and causality. Independent NumPy review verifies the retained
integer payloads against the original scores/indices. Timing controls pass
**8/36 overall**, including 7/12 at 3,973 rows; all failed 2% controls are retained.
Stable full-size median added latency is 0.499/0.502 ms per indexer call
(ranks 0/1). This is component timing, not end-to-end serving overhead or a speed
qualification; the correction applies to prefill, not decode.

Review: `out/drafter-canonical-prefill-v1/review.json` and `review-target.json`,
also retained in benchmarks. Archive: **1,952,437,133 bytes**, SHA
`2c2efaf058914d42fda1bb2c15c731c491851e8193752858b71cded1522a942e`.
Plan SHA `8f80040d8a94fc4b89bda62fbedc747a2e2b20d426b582ca9f3ae2d7fdd49521`.
Spark2: `full-model-v1/drafter-canonical-prefill-v1-retained-backup/`.
The first finalizer attempt failed because new safetensors component payloads
were root-readable only. The two files' modes were corrected to 0644 without
content changes; retry succeeded. `finalize-first-failed.log` and original failed
`run-prepared-status.json` remain retained; `archive-receipt.json` records the
successful retry. The source writer now sets 0644 for future evidence.
Source checkpoint `eac094f` and bundle
`e7fc25212503d830d244f9e1300fbac414e5d27b6c0bd07aca22b4fd1ed67726`
are verified in Spark2's Git backup.

**20 October 2026, 22:07 UTC: focused first-decode trace launched after both CPU preflights passed.**
Controller launcher PID 2352907; original production is temporarily paused.
The existing watchdog and controller restore the original 500,000-context service.

`drafter-long-decode-v1`, private port 8034, frozen plan SHA
`6d51d9069e98925f378ad074b6444f97764886650181d2eef00d9a14d23aa578`.
It keeps the prefill correction and existing expert/mHC policies. Bounded graph
copies observe layers 0–4 and native decode top-k scores, sequence bounds and
selected pools; only the first eight-row decode payload is retained. Copies
are present in every control graph and can alter scheduling, so this is
localization evidence, not native runtime or serving-speed qualification.
No new arithmetic change. Prior component fixtures are inherited from verified
evidence, without repeating startup timing trials. The GPU controller retains
automatic restoration and its independent watchdog. No sealed-test access.
Full 4,000/200 capture/training has not started; the 32-update pilot remains
speed-neutral and unqualified. Local ruff, strict mypy and pytest checks pass.
<!-- CANONICAL_PREFILL_END -->

<!-- DRAFTER_SPARSE_PREFILL_BEGIN -->
**20 October 2026, 21:37 UTC: native sparse-prefill top-k changes index order with identical scores and membership.**
`drafter-sparse-prefill-v1` completes all six requests and restores original
production: `complete_restored`, no error, health 200, original identities and
**500,000 context**. Finalization also succeeds. No experiment is currently running.
Independent CPU review verifies all **416 evidence payloads** on Mac; Spark2
independently verifies the complete backup. All **12 capture-pair comparisons**
across both TP ranks establish:

- Layer-3 indexer explicit inputs and consumed MQA scoring operands match exactly.
- All **1,972,098 consumed FP32 scores per pair** are bit-identical, with no
  nonfinite consumed values. Unused score columns are excluded by the native bounds.
- Native pool selection changes order in **1,921–1,922 rows**, first at zero-based
  row **2,051**, while selected multiplicity is identical in every row.
- Expanded token indices show the same order-only changes. Both representations
  have zero duplicates, invalid negative indices or causal/extent violations.
- All initial target-logit comparisons still differ. Later target features also
  differ; this diagnostic does not establish that index ordering explains every
  full-model divergence. Layer-5 first differences can precede row 2,051.

**Conclusion:** order nondeterminism is established in the observed native
prefill top-k component. A correction has **not** been applied or qualified.
Next: canonicalize only the selected pool indices, preserve their exact
multiset and invalid-slot masking, and measure GPU repeatability and overhead.
Then rerun the long-prompt full-model controls without per-operation observation.
If target drift remains, trace the next changed boundary. The observed GPU
copies may alter scheduling; this is neither a serving-speed nor a cross-model
quality pass. The unchanged repeat gate fails and full training stays gated.

Review: `out/drafter-sparse-prefill-v1/review.json`, also retained at
`benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/drafter-sparse-prefill-v1-review.json`.
Archive: **2,262,129,801 bytes**, SHA
`f0f8f18b6b6786da473d3ce4b5a4f4f9040e8da151e4a504d51a5449e0d51efd`.
Plan SHA `c91a61c51d99b3c6e9640dac04257c75f5bf07a278515ed841bb2068b228c5f4`.
Spark2: `full-model-v1/drafter-sparse-prefill-v1-retained-backup/`.
The unchanged target/original drafter uses seven drafts, three short warmups,
and the 3,973-token training prompt in native-before/four-captures/native-after
requests. The passive layer-3 probe retains 177,233,572 bytes per rank using
stream-ordered GPU copies, bounded to 512 MiB; no native arithmetic is replaced.

Previous boundary trace remains retained: all 12 pairs first differ at layer-3
indexer output with equal explicit inputs; 159 boundaries and 10,540,856,360
hashed bytes per prefill/rank. All 414 payloads independently verify on both
machines. Archive SHA
`9c7635b0157e332a2525f0cf0b7d12f91a67ebe32939f931b8b37ae3236c47b5`.
Probe source checkpoint `fe1c618` is committed and verified in Spark2's Git
backup; bundle SHA
`63ced2bbdbd36f7991246edf2dfa857233025f4fd257deaf76fd6fc3b41fb4df`.

Full 4,000-training/200-validation capture and training have **not started**.
The 32-update pilot remains speed-neutral and unqualified; no new speed gain.
The sealed test remains closed. No DeepInfer code or confidence scheduler is
introduced. Local ruff, strict mypy and full pytest pass (one existing skip).
<!-- DRAFTER_SPARSE_PREFILL_END -->

<!-- DRAFTER_FRESH_CAPTURE_BEGIN -->
**20 October 2026, 21:06 UTC: repeated long-prompt target prefill differs before the drafter.**
Full 4,000/200 capture and training remain gated; the sealed test is closed.
`drafter-long-repeat-v1` completes all six long-prompt requests and restores
original production with 500,000 context (`complete_restored`, no error).
Independent Mac review validates all 440 evidence payloads, accepted histories
and all 12 capture pairs across both TP ranks. Every pair has different target
prefill features at the earliest sampled layer, layer 5, and different logits
at all 128 shared decision positions. The initial input IDs, positions, seed
and anchor agree. Layer-5 first differences vary by pair (rows 169, 170 or
1,546); the exact operation is not yet localized.

The four captured runs emit identical 128-token responses despite different
features/logits. The uninstrumented before control diverges from those four
and the after control at output index 80. Repeat gate fails; diagnostic
completion is not training eligibility or a serving-speed/quality pass.
Review: `out/drafter-long-repeat-v1/review.json`, also retained in benchmarks.
Plan SHA `444b42c825545fc27bb7a27b90ddf2aa89db65034e1c1c466ecaac2b40c1ed75`.
Archive: 1,855,265,597 bytes, SHA
`d75c8ccc41df89630c33848e969405e0c9748943b24e4b61f285f467c5bf3ba6`.

**Historical follow-up (now complete/restored; see the latest result above):**
`full-model-v1/drafter-prefill-boundary-v1`, private port 8031, plan SHA
`951680952a525be24449b4bb862609f907a7685cefc4444006d427604629b8be`.
It hashes complete operands/results through layers 0–5 on the same 3,973-row
prefill, retaining native before/after controls and four captures. It excludes
scratch contents and unread routing tails. The controller has restored the original 500,000-context setup. No target/drafter weights or kernel
arithmetic changes. CPU copies synchronize execution and may alter overlap;
a trace that becomes repeatable under observation cannot qualify the native
runtime. Local ruff, strict mypy and full pytest pass (one existing skip).

Earlier fresh capture V3 passes all 39 startup checks on both ranks, then
fails the unchanged repeat gate after six requests/four captured prompts.
The two native controls diverge at output index 44 (84/128 positions differ);
before versus captured text agrees but counters differ. Production restores
at epoch 1792529167.916315. No prompt is discarded to bypass this failure.
Its archive finalizer initially failed its final live-health assertion because
the follow-up diagnostic had already paused production. The failed log is
retained; the retry after restoration succeeds. All 586 evidence payloads
independently verify on Mac and Spark2; the 440-payload long-repeat archive
also independently verifies on both. Spark2 retains each under
`full-model-v1/<run-name>-retained-backup/`. Archive: 480,569,076 bytes, SHA
`1c90c50af8095358fbbed55b54c1b79e19fcd42401c3058235eb577522615322`.
Frozen V3 plan SHA
`7fc8b9b7296fcf10996f2ae37ced4e0bc6452e20cb9b4598155a48cac4cc41e7`.

Retained attempts:
- V1 failed CPU preflight because the boot assertion still required 2,048
  prefill tokens; no production pause.
- V2 passed CPU checks but failed GPU startup at the inherited stable-dispatch
  2,048-row guard; no prompt was captured. Controller restores original
  containers/configurations and 500,000 context (`failed_restored`).
- V3 installs the **existing qualified** 4,096-row integer dispatch extension
  and its 39-variant startup checks. All five helper payloads match historical
  hashes exactly; the startup validator's removed trailing blank line is
  restored only after matching its historical SHA. No new GPU kernel math.

New local work: protocol-driven training schedule (legacy 32-step rates remain
bit-exact), bounded per-rank export with an independent NumPy bit audit,
paired export verification, and raw-staging release only after matching exports
exist on both nodes. The full-capture preparer refuses to run without a verified
fresh smoke manifest and independent audit. It freezes one seed-42 pass over
4,000 training prompts, 200 validation prompts, 200 warmup updates, maximum
4,000 updates, and validation at 0/250/500/1000/2000/3000/4000 (or last eligible
step). Existing selection, 1% weight-distance and serving gates remain fixed.
Full capture uses 64-prompt chunks; accepted features require 41.53 GiB per copy.
The previous 32-update pilot is still speed-neutral and unqualified.
Source checkpoint `03b7bde` is committed and independently verified in Spark2's
bare Git backup. Bundle `glm53-drafter-full-preparation-v1.bundle`, SHA
`3ad88caebb0fce6ff8671ed6192b27e9ea756ecf867d13e236c5b3262a22164c`.
The real retained-data incremental-export replay passes on both ranks: eight
outputs match the old verified exports byte-for-byte, 90,644,480 BF16 elements
are checked independently, and 1,022 rejected rows are excluded. This replay
uses historical features and is a pipeline test only.
Full CPU training vocabulary is ready at `out/drafter-full-training-v1/weights`:
154,880 exact BF16 embedding rows; SHA
`4f7aecd51589ce7879d0ae48a5a1ed25750e4e94d2abe2d14c9526e5ce5641d6`.
All 2,062 old pilot embedding rows independently match their prior bytes;
the unchanged head SHA remains
`aa21c427970f64edd82669db3a8fb46613084e8bc271a3728784a52eb3f25ab4`.
<!-- DRAFTER_FRESH_CAPTURE_END -->

<!-- EXPERT_WIDE_V1_BEGIN -->
**20 October 2026, 20:07:19 UTC: wider expert component check passes; production restored.**
Our existing Marlin routing-padding fix passes all **36 fixtures** at
8/16/32/64/128 target rows, across both TP partitions and repeat/spread expert
routing. Independent NumPy review finds **54/100 native shared-prefix comparisons
differ**, versus **0/100 with padding**. Both GEMMs, activation and weighted route
sum are checked. All **2,880 four-stage graph-repeat checks** are exact;
**96 native-reference checks** and **288 stock/packed comparisons** match exactly.
The actual installed routing block stays eight, and the observed planner stays
K=64/N=128/three blocks per SM. Both projections use quantum nine; appended
routing slots remain masked and whole launch waves are verified.

Timing is a separate gate: **27/32 controls pass**, with all five failures
retained at the original 2% drift limit. Packed-scale component overhead among
stable native-versus-padded pairs is:

| Target rows | Routing stress | Stable pairs | Measured padding overhead |
| --- | --- | ---: | ---: |
| 64 (fixed-seven-draft C8 size) | Repeat | 4/4 | +3.05% to +4.08% |
| 64 | Spread | 1/4 | +0.27% on the sole stable pair |
| 128 (fixed-seven-draft C16 size) | Repeat | 4/4 | +0.07% to +8.02% |
| 128 | Spread | 4/4 | +0.01% to +0.21% |

The spread C8 row is not a general cost estimate: three packed controls fail.
There are 96 clock/thermal samples: SM 2418–2515 MHz, 55–81 C, all recorded P0.
This replay uses synthetic extensions of one real layer's operands; TP
partitions run separately on one GPU. It proves the component result, not
full-model C8/C16 invariance, a serving-speed improvement or serving overhead.
The runtime limit remains **16 rows** until controlled full-model qualification.
No DeepInfer implementation or confidence scheduler is included.

The component run takes **533.67 seconds** and exits successfully. Original
container/configuration identities, health 200 and **500,000 context** verify
restored at 20:07:19 UTC; controller phase `complete_restored`, no error.
The 900-second replay limit and separate 1,200-second restoration allowance
were not exhausted. No experiment/training remains running; sealed test closed.

All **78 evidence payloads** independently verify on Mac and Spark2.
`evidence-v1.tar.gz`: 799,225,880 bytes, SHA
`2ca5d95679e512b6b40e94fe4522c3bb6b7f946af21be118d8151fd112ab0829`.
Frozen plan SHA `4aa286a8ff69f409cc2353a00da13b1507726a76e3b1c8783e4c56ba468aadda`;
seven executable sources, 36 fixture payloads and four retained source frames.
Independent review and summary: `out/drafter-expert-wide-v1/{review,summary}.json`.
The seven-payload analysis archive independently verifies on both Sparks:
`analysis-v1.tar.gz`, 13,946 bytes, SHA
`04ed218cb75c62cde6f47fc858859401aca859e1c4a007e479c1e309bd248ac0`.
Full pytest (one existing skip), ruff and strict mypy pass. Source checkpoint
`911c9f5` is independently verified in Spark2's Git backup; result checkpoint follows.

**Next:** keep the serving policy unchanged, and continue the independent
fixed-width drafter work: freeze the full-run trainer/protocol, capture fresh
features under the qualified C1 target policy, then train on the 4,000 prompts.
C8/C16 full-model identity and serving-cost measurements remain a separate
replacement gate. Do not spend the next cycle chasing the five noisy component
timings or describe the 32-update pilot as a successful adapted drafter.
<!-- EXPERT_WIDE_V1_END -->

<!-- DEEPINFER_TRANSFER_BEGIN -->
**20 October 2026: DeepInfer workstream withdrawn at the user's request.**
The confidence head, CPU scheduler and pilot observer hook have been removed
from active code. They were never trained on real data or enabled in serving;
there is no measured CT speed result for that experiment. The pilot's +0.111%
code result below belongs to our independent fixed-width drafter adaptation,
not confidence scheduling. No DeepInfer/DeepGEMM kernel was copied or installed.
The source audit and provenance remain historical records, not an active port.

Continue our existing Marlin routing-padding fix and context-projection
adaptation. Keep seven fixed draft tokens. The bounded feature loader now reproduces the retained 32-update pilot exactly.
The wider component check now also passes. Next: capture fresh qualified-policy
target features and run the full 4,000-prompt adaptation; C8/C16 full-model
serving qualification remains a separate gate.
The old-policy pilot replay is a loader regression check, not new-policy
training or a speed claim. Preserve the existing quality/performance gates;
the sealed test stays closed. Production remains restored.
See [the archived audit](GLM53_DEEPINFER_TRANSFER.md).
<!-- DEEPINFER_TRANSFER_END -->

<!-- BOUNDED_LOADER_REPLAY_BEGIN -->
**20 October 2026, 19:42 UTC: independent bounded-loader replay passes.**
All **32 updates** reproduce the retained original pilot's losses, gradient
norms and proposal metrics exactly. The baseline and every validation proposal
match. All **four BF16 checkpoint files are byte-identical** to the originals;
step 24 is selected again. Independent NumPy review verifies acceptance and
the already corrected FP64 weight distances. Elapsed CPU run: 183.64 seconds.
This removes the loader-regression prerequisite for full training. It is not
new-policy feature capture, a new adapted candidate or a serving-speed result.

Frozen protocol SHA `119d6da0455407a28a9fa4e098730b4818d280319c86341474032dfb9135ada8`.
Local run/review: `out/drafter-bounded-loader-replay-v1/`.
All **14 evidence payloads** independently verify on Mac and Spark2. Archive
`evidence-v1.tar.gz`, 54,597 bytes, SHA
`7ad60a6548b517bb8c85f9a8c19a7f557ae14550489aa7109795f0f7306a91b7`.
Identical checkpoint payloads are referenced rather than duplicated: all four
original files plus their report were independently rehashed in Spark2's
retained `glm53-drafter-context-pilot-v1-evidence.tar.gz` backup.

Production was never paused; its original service was verified healthy during
the CPU replay. No training remains running and the sealed test stays closed.
Full pytest (one existing skip), ruff and strict mypy pass after the confidence
code/hook removal. Withdrawal/protocol checkpoint `6e4cdd2` is committed and
independently verified in the Spark2 Git backup; the result checkpoint follows.
Next: our C8/C16 padding GPU fixture, then fresh qualified-policy feature
capture and the full 4,000-prompt fixed-width drafter adaptation. Do not restore
the withdrawn confidence experiment or loosen the existing performance gates.
<!-- BOUNDED_LOADER_REPLAY_END -->

<!-- EXPERT_FIXED_DEVELOPMENT_BEGIN -->
**20 October 2026, 19:23:01 UTC: development comparison complete; original production restored.**
All **144 measured requests** complete and all output token sequences match
across original/adapted/original loads on the eight development prompts.
Within-load tokens/counters and original-reload tokens/counters repeat exactly.
**8/8 baseline reload timing controls pass** the unchanged 2% limit; the largest
absolute reload throughput difference is 0.6974%. Both arms use the same CT
MXFP8 target and common expert padding policy.

| Domain | Original drafter tok/s | Step-24 pilot tok/s | Difference | TTFT, original → pilot |
| --- | ---: | ---: | ---: | ---: |
| Code, four prompts | 38.770 | 38.813 | +0.111% | 250.76 → 251.45 ms |
| Chat, two prompts | 29.932 | 29.914 | −0.060% | 174.24 → 182.75 ms |
| Explanation, two prompts | 31.523 | 31.419 | −0.329% | 192.94 → 197.19 ms |

Values are geometric means of per-prompt median end-to-end rates/TTFT;
the original baseline combines both original loads geometrically. TTFT relative
changes are +0.273%, +4.887%, +2.200%, respectively. These are descriptive
development results from one candidate load, not confidence intervals or final
qualification. The small pilot is effectively speed-neutral and does **not**
deliver the ≥8% code target. Keep it unqualified; full training is still required.
This comparison measures drafters on CT; it does not measure the cost of padding
against native execution or make a new CT-versus-production speed claim.

The lightweight runtime counters record at least **3,968 native draft graph
replays per arm/rank**, all six arm/rank groups using descriptor `[1, 8]`.
Startup records verify expert capture coverage on both ranks in all loads.
There are 106 during-measurement resource samples per arm/rank: SM clocks
2405–2515 MHz, temperatures 48–76 C, recorded power states all P0.
This run records output tokens, not full-vocabulary logits; the earlier single
prompt's measured KL 0 remains a separate result.

Original container/configuration identities, health 200 and **500,000 context**
are independently verified restored at 19:23:01 UTC. No experiment/training
remains running. Controller and finalizer return 0. All **93 evidence payloads**
verify independently on Mac and Spark2. Archive `evidence-v1.tar.gz`:
14,307,985 bytes, SHA
`5cf8286390a98047306cedc7eff6120573e34599d077a864bb5f3aba7ce8153f`.
Independent review: `out/expert-fixed-development-v1/review.json`.
The seven-payload independent analysis archive verifies on both Sparks:
`analysis-v1.tar.gz`, 16,028 bytes, SHA
`1026a68eee7f0ab69bb336ac1d09d83a830a009fd199c00b4891b930d70ff0c2`.
Full pytest (one existing skip), ruff and strict mypy pass after the bounded
loader addition. The subsequent confidence-loss normalization correction passes
all 21 relevant confidence/loader/pilot tests and full ruff; no real-data
confidence model has been trained. Work through `983dd43` is verified in the
Spark2 Git backup; the final result checkpoint follows.
Sealed test remains closed. Earlier failed timing/numerical attempts are retained.

**Frozen protocol and preparation:** the run launched at 18:43:35 UTC.
`expert-fixed-development-v1` reuses all eight development prompts (four code,
two chat, two explanation), six seed-42 shuffled cycles and 256 generated tokens.
Loads remain original → adapted step-24 pilot → original, 48 requests per load,
144 measured requests total, with two excluded warmups per load. Both drafters
use the exact expert padding hook verified below on the same CT MXFP8 target.
Target weights and each arm's preselected drafter checkpoint stay unchanged.
The sealed test remains closed.

The timed requests use lightweight draft-graph descriptor counts without
feature/logit copies. Strict within-load token/counter repeats, cross-drafter
output identity and the unchanged 2% baseline reload timing control are retained.
One candidate load remains a development screen, not final speed qualification.
This measures the drafter under a common padding policy; measuring padding
overhead against native execution is separate. C8/C16 policy extension and the
4,000-prompt capture/training run remain outstanding.

Plan SHA `62e8595165e064749d5afe196a73c8ffdb323f78063fdd88c7d4ae3a59e67b3c`;
26 frozen source/plan payloads. Private port 8027; 40-minute experiment cutoff
and independent original-production restoration watchdog. The preparation
verifies all 250 preceding numerical-check evidence payloads and requires
exact decision logits on both ranks. All six serving-image CPU preflights pass,
each verifying 211 runtime bindings. Local full pytest (one existing skip),
ruff and strict mypy pass. Launch source `42718f5` is committed and verified in
the Spark2 backup. The bounded controller completed the temporary production pause,
three diagnostic loads and restoration. Launcher PID 724962. At **19:08:25 UTC**,
the original and adapted arms have completed all 48 requests each, with within-arm
repeat checks passing. All **48 paired output-token sequences are identical**.
Of the eight prompts, draft rounds decrease by one on three, increase by one on
one, and are unchanged on four. This small pilot has no established speed gain.
The final original reload/control subsequently passed, as recorded above.

**Parallel CPU audits (no serving change):** installed Marlin source
`f0046031…` selects routing blocks of eight through 259 target rows for this
288-expert, top-8 BF16 path; C8/C16 with seven drafts correspond to 64/128 rows.
The audited occupancy/cache calculation predicts the same launch tile, and
4,096 routing-count arithmetic checks give whole launch waves with nine-block
padding. Actual GPU planner, shared-row equality and timing checks remain
required before raising the current 16-row runtime limit.
Local evidence: `out/expert-wide-audit-v1/review.json`.
There are now **36 CPU-prepared wider fixtures**, 16,144,080 bytes, from both
TP partitions and both original divergence arms at 8/16/32/64/128 rows.
Original eight-row operands are preserved; appended rows either repeat the
source or spread routes over additional experts. The seven-row shared prefix
is independently checked. This is synthetic component stress, not C8/C16
serving evidence; no wider GPU execution has occurred.
Manifest: `out/expert-wide-audit-v1/fixtures/manifest.json`.

**19:14 UTC:** the final original control is measuring after startup. The new
opt-in bounded feature loader passes CPU tests for exact payloads, one-sample
cache eviction, changed-file rejection and live-autograd input preservation.
Full pytest (one existing skip), ruff and strict mypy pass. Old pilot protocols
retain their original loading behavior; the full-run protocol must explicitly
enable `bounded_feature_loading`. A real pilot replay under the new loader is
still required before full training. Confidence/source-audit commit `2c8e5df`
is independently verified in Spark2's Git backup. The 18-payload licensed source
audit archive verifies on both Sparks, SHA
`c71b930617c2cabfd4fe634247642cd59e8b2de1cf547913e04c9e76ff57954c`.

The frozen training/validation token metadata contains 4,000/200 prompts,
maximum lengths 3,973/2,248; none exceed the 8,064 input budget for 128 outputs.
There are 52/2 prompts beyond the pilot's 1,024-token filter. Preserve them in
the full run. One accepted-feature copy is approximately 41.53 GiB total, so
full capture/export must be incremental and training must load bounded samples.
Pilot updates averaged 3.194 seconds; multiplying by 4,000 gives 3.55 hours,
**excluding capture, longer-context effects and validation**. This is a sizing
extrapolation, not a completion estimate. The sealed token file was not opened.
Local record: `out/expert-wide-audit-v1/full-training-scale-audit.json`.
<!-- EXPERT_FIXED_DEVELOPMENT_END -->

<!-- EXPERT_FIXED_VERIFICATION_BEGIN -->
**20 October 2026, 18:18:32 UTC: full-model numerical check passes; original production restored.**
`expert-fixed-verification-v1` applies default-occupancy expert padding in both
original/adapted drafter arms on the same CT target. It reuses the one development
prompt, 32 generated tokens, fixed seven draft tokens and before/capture/repeat/
after controls. Both arms pass all four controls; all **32 output token IDs are
identical across drafters**. Independent NumPy review verifies **all 32
full-vocabulary decision logits per rank bit-for-bit**, with maximum and mean
KL **0** on both ranks. Captured inputs, logits and auxiliary/final features
repeat exactly in all four arm/rank comparisons. The archive retains 88 frames:
11 per capture × two captures × two arms × two ranks, including 80 observed
full target-graph frames. Component captures at positions 46 and 54 are present
in every arm/rank/repeat. Neither target nor drafter weights changed.

**Limit:** the two drafters now propose identical inputs in every aligned frame
with shared accepted history. This full-model run resolves the observed output
divergence but does not itself stress differing future proposals. The earlier
440/448 component replay supplies that specific stress case. Both arms have
9 draft rounds, 63 proposed tokens and 22 accepted tokens; this short run shows
no acceptance improvement. Broader development prompts remain necessary.
Changing the common expert reduction order changes historical native outputs
at indices 21–31 for `a1` and 28–31 for `b1`; all old records remain retained.
This is not a claim of identity to the old native rounding policy.

No serving-speed claim or sealed-test access. Above 16 target rows, expert
execution remains native; C8/C16 coverage is still outstanding. The two V3
component timing failures remain failed at the unchanged 2% limit. No settled
end-to-end overhead or drafter qualification follows from this numerical pass.
Plan SHA: `3277a210f280cd4c68e25bd3225f640de16e43359979284b7c8e0453ef7f19e1`.
All four serving-image CPU preflights pass, verifying 211 runtime bindings per
rank. Launch source `2bcc64b` is verified in the Spark2 backup. The original
containers/configurations, health 200 and **500,000 context** are verified
restored; no diagnostic or training is running. The run launched 18:03:38 UTC.
All **250 evidence payloads** independently verify on Mac and Spark2. Archive:
`expert-fixed-verification-v1/evidence-v1.tar.gz`, **633,231,767 bytes**, SHA:
`4f858ed07d7ed21f771f7c9bb6916f82ad9e709fc46e09c9446f150f630ff0ef`.
Independent review: `out/expert-fixed-verification-v1/review-v2.json` (adds
explicit graph-frame and proposal-equality checks to the retained first review).
The seven-payload independent analysis archive verifies on both Sparks:
`analysis-v1.tar.gz`, 23,670 bytes, SHA
`1b6d79241dec78bee1bf13e5aff5cf15d177b55ba2bcd6e659ba336bb82d25c8`.
Full pytest (one existing skip), ruff and strict mypy pass.

**Agreed next gates:** rerun the frozen
multi-prompt development comparison with the same target reduction policy in
both arms. Extend coverage beyond 16 target rows before claiming C8/C16
independence: audit the actual routing block sizes and tile choices, then replay
shared rows with varied dispatch sizes. Merely raising the row limit is not
validation. Measure end-to-end padding cost separately at C1/C8/C16 with repeat
controls and clock/thermal telemetry. The 4,000-prompt feature capture/training
run remains outstanding; the current candidate is only the 32-update pilot.
Keep the sealed test closed throughout development and training, then evaluate
it once the candidate and runtime policy are frozen. One C1 prompt cannot
establish drafter-independent outputs for every prompt or serving layout.
<!-- EXPERT_FIXED_VERIFICATION_END -->

<!-- DRAFTER_EXPERT_PADDING_BEGIN -->
**20 October 2026: V2/V3 padding numerical gates pass; overhead remains partly qualified.**
`drafter-expert-padding-v2` completes the exact 440-versus-448 fixture for both
real TP partitions. Both default-occupancy and one-block-per-SM padding make
all seven shared rows bit-identical through up, activation, down and route
summation. Stock and packed scales match under each policy. Native replay
matches retained up/activation/down outputs; all 960 graph-repeat assertions
pass. GPU fixture duration is 64.416 seconds, peak allocation 1.901 GiB.
Independent archived NumPy review confirms all comparisons: 48 native-output
matches and 96 stock/packed matches. All 25 evidence payloads verify on Mac and
Spark2. Archive SHA:
`5a0e7c939cdb9181aaaf3c0f151abc3f321c7ef25cfdea310e553c92532321b2`.

The observed component overhead is a few percent, but **14/32** paired timing
controls exceed the unchanged 2% first/last-half drift limit. Do not claim a
settled overhead or serving speed. `drafter-expert-padding-v3` therefore retains
native and default-occupancy padding, adds 30 seconds warm-up per fixture and
16 interleaved cycles of 200 replays, plus SM/memory clocks and thermals.
V3 completes in **322.796 seconds**, again with exact shared rows and **640**
exact graph repeats. Independent NumPy review passes 48 native-output and 64
stock/packed comparisons. **14/16** timing comparisons meet the 2% drift limit:
all eight stock comparisons pass; two packed comparisons fail at **2.266%**
and **2.180%** drift. Passing paired component overheads range **1.14–2.84%**
for stock and **0.54–3.51%** for packed. These are component results, not serving
slowdowns; no blanket stable-overhead claim. SM clocks span 2457–2515 MHz,
temperature 61–74 C, all recorded power states P0. Memory clocks report N/A.
Default occupancy preserves the native launch configuration, not the fastest
noisy sample. V3 plan SHA:
`6a4e8d5040da614c1a739085d932d057dd0d01446d0ca0d4ed5c161936922e5d`.
Source commits through `148cfc9` are verified in the Spark2 backup. V3 archive:
23 payloads, 25,636,091 bytes, SHA
`92a50ab8eda94dea786083e465f251925da2b4e5178ae97254a74a2a0b44081a`;
all payloads independently verified on Mac and Spark2. The five-payload
independent analysis archive is verified on both Sparks:
`83d43694ec29c2a6a5b2fdde855249f9f6f07d407fb7dd9df351f01f076ace90`.

V1 failed before its numerical gate because the replay called an activation
symbol absent from `vllm._custom_ops`. Installed MoE source uses
`torch.ops._C.silu_and_mul`; V2 corrects that call. V1 is preserved as a harness
failure, with original service verified restored. A separate
GPU-free API import could not load the CUDA extension and is also retained;
it supplies no numerical evidence. The installed original-container API check
then passes with zero allocated CUDA bytes and no CUDA context initialized.
All three component attempts restore original production through the independent
watchdog/controller. V1's 18-payload failure archive verifies on Mac/Spark2:
`70f83f453d388cbfcdb0cfd1f691190a4aa4c17fcca1805772a532cd93dac176`.
The fix is component-proven only. The adapted drafter stays **unqualified**, the sealed test stays
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
unchanged 2% timing limit. No blanket/global threshold or9–15-row interpolation.

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
