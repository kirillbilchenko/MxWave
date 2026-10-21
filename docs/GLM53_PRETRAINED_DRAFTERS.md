# Pretrained drafter screen before further training

Updated **21 October 2026, 05:42 UTC**. The user prefers checking a pretrained
FP8 drafter before paying for adaptation. Automatic CPU training is held:
the idle continuation v2 PID 80216 was stopped before any optimizer update.
Its frozen files and status remain; `held-for-alternatives.json` records the
supersession. The original GPU capture continues with its existing watchdog,
verification and restoration policy. No capture data was restarted or discarded.

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

The queue must wait for complete verified capture and restored production.
It must not start if capture fails or its plan changes. The existing restoring
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
