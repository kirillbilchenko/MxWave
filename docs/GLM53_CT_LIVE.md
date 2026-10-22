# CT serving handover, 22 October 2026

**Status: prepared for the user's explicitly requested CT deployment.** Both
rank CPU import/codec checks pass. Original production has not yet been stopped.
This is an operational handover, not another benchmark or a change to the frozen
acceptance criteria. The user accepts the measured speed and quality tradeoff.

The profile is the successful final comparison's **B arm**: AutoRound CT MXFP4
experts, MXFP8 KDA, lossless expert-scale and embedding packing, dense BF16 output
head, and the same five-layer **Inco FP8 DFlash2** used by production. This uses
production's existing custom image and overlays. It does **not** establish the
performance of the separate stock-vLLM package; stock compatibility remains a
requirement for that distributable recipe.

Preserved settings: 500,000 configured context, 32 sequence slots, 6 GiB KV per
GPU, 6,919 prefill budget, prefix caching, actual 4 ms admission window, original
graph settings and the original Inco draft policy (7 at C1, 5 at C2, 3 at
C3–C27, with the original larger-batch exceptions). No KV increase, projection
hook, drafter training or new kernel optimization accompanies this handover.

The new containers clone the exact stopped `timing-B-2` containers. The only
serving changes are port 8051 → 8000, separate writable report directories, and
disabled development endpoints. Original containers/configurations are preserved
for rollback. The startup watchdog restores them if loading or generation fails,
or startup exceeds its 25-minute bound. After verified handover, the watchdog
exits and CT remains serving independently of the laptop.

Deployment directory on both Sparks:

```text
/home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/full-model-v1/glm53-package-v1/ct-live-20261022
```

Rollback to the exact original production service:

```bash
ssh spark 'python3 -S /home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/full-model-v1/glm53-package-v1/ct-live-20261022/control_ct_live.py --root /home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/full-model-v1/glm53-package-v1/ct-live-20261022 --rollback'
```

Read `controller.json` for live state, `startup.json` for both actual loaded
profiles/allocations, and `smoke.json` for the authenticated generation check.
These are independent from the immutable benchmark evidence. No upload on
Spark2 is stopped or modified.

The [final comparison](GLM53_SAME_RUNTIME_RESULTS.md) remains unchanged:
14.465825 GiB lower allocation across TP2, higher observed speed medians with
failed 2% repeat-spread controls, and 287/300 versus 291/300 GSM8K. Production
and CT are different quantized checkpoints. Lossless packing preserves the CT
values it encodes; it does not make CT's weights identical to ModelOpt's, and
MXFP8 KDA is a lossy conversion. The small quality subset does not establish a
general quality loss or equivalence. No long-context quality claim follows from
successfully starting the 500k configuration.

Preparation caught two bookkeeping issues before any outage: the comparison's
container identity normalizes Docker bind ordering/defaults, and its coordinator
preparation record was only on Spark1. The handover reproduces that normalization
without dropping any identity field and stages the source record on both nodes.
Both rank CPU checks then pass. A final parentheses-only lint fix was re-frozen
before launch, preserving the initial preparation receipt. No GPU trial was
repeated to address these preparation issues.
