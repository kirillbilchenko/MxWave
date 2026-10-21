# Pretrained drafter screen before further training

Updated **21 October 2026, 06:19 UTC**. The user prefers checking a pretrained
FP8 drafter before paying for adaptation. Automatic CPU training is held:
the idle continuation v2 PID 80216 was stopped before any optimizer update.
Its frozen files and status remain; `held-for-alternatives.json` records the
supersession. At the user’s request, GPU capture stopped at 1,095 prompts. Its handler restored
original production healthy at 05:46:55 UTC with 500,000 context. All 1,088
accepted exports in 17 completed chunks independently rehash on both Sparks
(13,097,763,612 bytes per node); unexported raw files remain. The incomplete
dataset is not published for full training.

**Current result:** Modal BF16 completes all 48 timed requests, with exact
token/counter repeats across all six trials. Code median throughput is
**40.83 vs 39.88 tok/s (+2.39%)**, chat **29.30 vs 29.56 (-0.90%)**, explanation
**31.58 vs 31.20 (+1.22%)**, compared with the first baseline load. **7/8 output sequences differ
from the Inco baseline**, first at indices 47–206. This fails output identity;
different trajectories confound performance/acceptance comparisons. The cause
is not identified. Modal FP8 and the baseline reload remain pending. No
positive candidate conclusion, quality degradation claim, or promotion follows
from these incomplete results. The frozen run continues under its watchdog.

The first original-Inco-FP8 baseline load completes **48 measured requests**
plus two excluded warmups. All six repetitions have identical output tokens and
speculative counters. Median end-to-end throughput is **39.88 / 29.56 / 31.20
tok/s** for code/chat/explanation; median TTFT is **277 / 176 / 197 ms**.
These numbers establish this development suite's first baseline; the baseline
reload remains pending. Production is temporarily paused
under the comparison restoration watchdog. Measurement SHA256:
`443d95e4b3a1ef826299ae55dc2b7c3263a4e01bcb90511035860f463340f39e`.

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

The diagnostic is implemented and queued; its GPU run has **not started**. Use development case
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
All repository checks pass. Detached queue PID **987188** on Spark waits for the
timing comparison, six repeat controls, finalization and healthy restoration.
It then CPU-preflights and runs the separate two-arm trace on API **8041**, with
a **30-minute restoration watchdog**. It needs no laptop connection. Both Sparks
verify all 46 frozen files; plan SHA256
`ae1f8d983ada1955fbf94d31ce72ac84e9292ac55ad2909271b555c0150dfe29`.
The frozen timing run remains untouched. Read the diagnostic root's `queue.json`,
then `controller.json`; no training, sealed test or promotion follows this queue.
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
