# Pretrained drafter screen before further training

**21 October 2026, 07:26 UTC: CT-only draft-length screen preparing.**
The user redirects the next test from production to CT and authorizes unattended
work. Same CT + Modal FP8 at 7/3/5/off/7 drafts, eight development prompts and six
cycles per load. CPU checks run on both nodes; production is still online. The
GPU comparison will use the existing independent restoration watchdog. No
training, sealed test or promotion. [Current plan and status](GLM53_HANDOFF.md).

**21 October 2026, 07:16 UTC: production comparison does not show a CT acceptance deficit.**
A live, unchanged-production screen uses the same eight development prompts,
C1, 256 output tokens, seed 42, temperature 0 and seven drafts, twice. Results:

| Setup | Code acceptance | Chat acceptance | Explanation acceptance |
| --- | ---: | ---: | ---: |
| Current production + Inco FP8 | 34.82% | 23.36% | 27.51% |
| CT + Inco FP8 | 38.81% | 25.60% | 28.16% |
| CT + Modal BF16 | 42.91% | 26.90% | 30.75% |
| CT + Modal FP8 | 42.03% | 27.60% | 30.21% |

Acceptance means accepted draft tokens / all proposed tokens. Production accepts
2.44 code tokens per seven-token draft round; first-position agreement is 82.24%.
All eight production outputs and counters differ between cycles despite fixed
seed; CT's six within-load repeats were exact. The target, output trajectories,
prefix caching and other serving tuning differ, so this is a descriptive setup
comparison. Production still records 49.09/41.47/41.71 tok/s for code/chat/explanation;
no causal speed or quality claim. The run finishes healthy without restarting or
changing production. Report: `diagnostics-v1/production-drafter-acceptance-v1-review.json`.

The saved CT trace also now confirms matching accepted layer0/4 QKV and state
through position 91, with layer4 QKV first different at 92, before its recurrent
update. All 200 inspected convolution-history carry checks pass. Next is a
component trace in layers0–4; no follow-up GPU run is queued. Training stays held.
[Full evidence and limits](GLM53_HANDOFF.md).

**21 October 2026, 06:56 UTC: native-decision trace completes and independently verifies.**
Both GPUs reproduce the same finding: target logits/hidden states remain exact
for the first 37 output decisions, then differ at output 38; generated tokens
first differ at 48. Each arm follows its own target argmax. All within-arm controls
and tensor repeats pass. Layer0/4 recurrent read states match through the common
position 87 boundary; layer 4 differs by 94 while layer 0 remains exact. The next
check targets state/rollback and arithmetic between those boundaries. This does
not establish a responsible kernel or a speed improvement. Production is restored
with 500k context; all 406 artifacts are verified on Mac/Spark2. No follow-up GPU
run is queued. [Full result and limits](GLM53_HANDOFF.md).

**21 October 2026, 06:41 UTC: Inco trace complete; Modal BF16 loading.**
All four Inco before/capture/repeat/after controls reproduce the frozen output and
speculative counters; both ranks retain 34 frames each. Paired trace analysis is
pending. See [the latest handoff](GLM53_HANDOFF.md).

Earlier speed-screen checkpoint, **21 October 2026, 06:35 UTC**. The user prioritizes the native-decision
trace over the final speed baseline reload. The reload is deliberately stopped
before any a2 measurement; its controller restores healthy original production
at **06:32:09 UTC**, with unchanged identities and **500,000 context**.
All three completed first loads have 48 timed requests and exact token/counter
repeats. All 132 retained timing payloads independently verify on Mac and both Sparks.

| Drafter on the unchanged CT target | Code tok/s / TTFT | Chat tok/s / TTFT | Explanation tok/s / TTFT |
| --- | ---: | ---: | ---: |
| Original Inco FP8 | 39.88 / 277 ms | 29.56 / 176 ms | 31.20 / 197 ms |
| Modal BF16 | 40.83 / 251 ms | 29.30 / 184 ms | 31.58 / 197 ms |
| Modal FP8 | 41.18 / 257 ms | 30.70 / 184 ms | 32.21 / 189 ms |

**Modal FP8 and BF16 outputs agree on all 48 matched requests.** Each matches
Inco on only 6/48 (one of eight development cases); differences first occur at
output indices 47–206. The observed mismatch predates FP8 conversion. Its cause
is not identified, and changed output trajectories confound original/Modal
speed comparisons. The baseline reload drift control is intentionally absent;
these medians do not qualify a speed gain or establish a quality regression.
See `diagnostics-v1/drafter-pretrained-modal-v1-retained-summary.json`.

At 06:35 UTC, detached native-trace launcher **1180511** passed CPU preflight.
Controller **1186244** loaded the first/Inco arm under watchdog **1186334**,
temporarily pausing production again. That run is now complete and restored. The launcher verifies
the specific user-stop receipt, successful restoration/finalization and saved
measurements before proceeding with the unchanged frozen diagnostic. Read
`after-priority-status.json`, then `controller.json`; old waiter PID 987188 and
its `queue.json` are superseded. GPU capture and independent review are now complete; see the result above. Training remains
stopped and the sealed test closed. The long capture remains at 1,095 prompts,
with 1,088 verified exports in 17 completed chunks (13,097,763,612 bytes per node).
The unexported files remain; this incomplete dataset is not published for training.

## Verified CPU results

The public [Modal checkpoint](https://huggingface.co/modal-labs/GLM-5.3-Flash-DFlash)
is pinned at `dae6d319510889a7f1ac5242d5b7fb2d5d95eb05`. Despite its repository
name, its actual architecture is **DFlash2DraftModel**: six layers, causal
attention, 4,096 sliding window, and target taps **23/27/31/35/39/43**.
Our current Inco drafter uses five layers and taps **5/14/24/33/42**.
Existing captured features therefore cannot evaluate Modal's proposals offline.

All **96 BF16 tensors / 1,389,225,472 parameters** strictly load in the pinned
SpecForge DFlash2 class, with no missing or unexpected tensors. The actual
installed vLLM configuration helpers pass independently on both Sparks:
the native registry contains DFlash2, attention resolves to six causal
4,096-window layers, translated auxiliary IDs are 24/28/32/36/40/44, and
the context projection accepts 24,576 features. This is CPU compatibility
evidence, not native GPU loading or measured acceptance.

An independently authored converter creates a separate **block-128 E4M3FN FP8**
candidate in the format already used by our current drafter. It quantizes
42 attention/MLP matrices and preserves the other 54 tensors exactly, including
the context projection, selector, convolutions and norms. The CT target,
verification policy, feature capture and training protocol are unchanged.

| Checkpoint | File bytes | Decimal GB | Training |
| --- | ---: | ---: | --- |
| Modal pretrained BF16 | 2,778,461,544 | 2.78 | None |
| Derived Modal FP8 | 1,621,120,728 | 1.62 | None |

Conversion and serialized-payload verification take **6.606 seconds on CPU**.
This is a **41.7% checkpoint-file reduction**, not a runtime memory or speed
measurement. Only the large backbone projections are quantized. The per-matrix
weight relative L2 error is approximately 2.53–2.77%; it is not model KL.
Independent NumPy validation checks all 54 copied tensors byte for byte,
all FP8 tensors for finite codes/positive scales, and **6,881,280 sampled
FP8 rounding decisions**, with zero differences. It is a sampled rounding audit,
not an independent re-encoding of every FP8 weight.

BF16 weights SHA256:
`3860c574465b7c7523896a22a3cc83dc4b9c14e019a6755061de2b8a4b061539`.
FP8 weights SHA256:
`45f34ef126aa258d8e16bd605a3a05c36eb3fef3ed0f57aef20191db49f1ef46`.

## Bounded serving comparison

### First-divergence diagnostic specification

The diagnostic is **complete and independently reviewed**; the frozen specification follows. Use development case
`opencoder-evol-92378` (56 prompt tokens): the first 47 output tokens agree,
and output index 47 differs. Retain the observed 64-token prefixes for both arms
and require before/capture/repeated-capture/after controls to reproduce them.
Observe actual native decoding from the original prompt; adding the generated
prefix to a new prefill request would not reproduce the recurrent cache history.

Capture target logits before/after sampling, proposals, acceptance counts, actual
input positions/graph shapes and recurrent cache/rollback metadata around that
decision. Equal target logits with different chosen tokens directs investigation
to tie handling or verification; different logits with equal accepted history
requires comparing actual state and layer boundaries. Replay the first changed
component with equal shared inputs/state and changed future proposals. A separate
speculation-off reference can help, but its different row/cache layout is not
by itself proof that the verifier is wrong.

The passive context probe now follows each arm's configured five/six auxiliary
layers; tests verify unchanged native outputs and independent retained copies.
The reviewer aligns the actual sampled row and validates its causal prefix.
All repository checks pass. Detached replacement launcher **1180511** starts
at 06:33:03 UTC after the speed repeat is intentionally stopped and original
production restored. The specific cancellation receipt is required; an arbitrary
failed comparison cannot trigger the diagnostic. Both Sparks verify the unchanged
46 frozen files; plan SHA256
`ae1f8d983ada1955fbf94d31ce72ac84e9292ac55ad2909271b555c0150dfe29`.
The separate two-arm trace uses API **8041**, before/capture/repeated-capture/after
controls, and a **30-minute restoration watchdog**. It runs without the laptop.
Read `after-priority-status.json`, then `controller.json`. No training, sealed
qualification or promotion follows this launch. Old `queue.json` is superseded.
Specification and source measurement hashes:
`diagnostics-v1/drafter-pretrained-first-divergence-plan.json` (SHA256
`df23ef9849faced7e4221e9bb308b76e866846c78184d198b3c539712888f43c`).

The prepared `drafter-pretrained-modal-v1` comparison uses the same CT MXFP8
target in all four loads: **current Inco FP8 → Modal BF16 → Modal FP8 → current
Inco FP8**. Eight frozen development prompts cover four code, two chat and two
explanation cases, with six shuffled repetitions and two excluded warmups per
load. C1, 256 output tokens, temperature zero, seed 42, seven draft tokens,
prefix caching off, 4,096 prefill budget and 6 GiB KV per rank are fixed.

Measure end-to-end tokens/s, TTFT, verification rounds, accepted drafts,
output identity, graph replay counts and clock/thermal/resource telemetry.
Both candidate checkpoints retain their own configuration; only backbone
precision differs between Modal BF16 and its FP8 derivative. Startup verifies
checkpoint hashes and the actual replicated context weight against its source.
No feature-copy instrumentation remains in the timed run.

The original full-capture waiter was stopped. A replacement launcher accepts
only the explicit user-stop receipt, expected signal-15 cancellation, healthy
original restoration and independently matching retained chunks. Arbitrary
capture failures or changed source/plan hashes still prevent launch. The existing restoring
controller bounds the new outage to 60 minutes and restores the original
500,000-context service on success or failure. Allow approximately 30–50 minutes
for this screen after capture/restoration, subject to compilation and warmup.
One load per candidate is a development screen: it cannot establish a final
speed claim. Baseline reload drift above 2%, output divergence, or failed repeats
prevent a positive conclusion. The sealed set stays closed.

If either pretrained version shows a useful gain, qualify it on broader held-out
prompts before deciding to abandon adaptation. The existing final serving gates
are unchanged. There is no automatic training restart or deployment in this queue.
Native MTP remains another unqualified alternative; no MTP GPU run is queued here.

Sources, downloaded models, frozen launches and reviews are under
`out/drafter-alternatives-v1/`. Committed numerical evidence is under
`benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/drafter-pretrained-v1-*`.
Ruff, strict core mypy and the full pytest suite pass. Queue tests verify that
unfinished, unverified and failed captures cannot trigger a serving switch.

The earlier FP6 suggestion was corrected by the user to FP8. The installed
MXFP6 CUDA backend is emulation (`dequant + F.linear`), so no FP6 kernel work
or conversion is part of this experiment. FP8 conversion does not adapt the
drafter to CT; only the serving comparison can determine whether training can
be avoided.

Queue launch is verified at **05:41:53 UTC**, remote PID **677064**, phase
`waiting_for_verified_capture_and_restoration`. All 44 frozen files and four
checkpoint/config files hash-verify independently on both Sparks. The run root
is `full-model-v1/drafter-pretrained-modal-v1`; read `queue.json`, then
`controller.json` and the per-arm measurement files. The two candidate model
directories are siblings suffixed `-modal-checkpoint` and `-modal_fp8-checkpoint`.

**User-stop continuation:** the active launcher is PID **716651**, with state
`after-stop-status.json`, initially `prepare`. The old PID 677064 and its
`queue.json` are superseded. `after-stop-readiness.json` binds restoration and
partial-data preservation to the unchanged benchmark plan. All required checks
pass; there is no pretrained acceptance or speed result at this checkpoint.

All **eight CPU preflights pass**, with 211 runtime bindings per rank. At
05:48 UTC the new controller **728927** and watchdog **728928** start the
bounded comparison; production is temporarily paused again. Read
`controller.json` for arm progress and `after-stop-status.json` for the launcher.
