# Final same-production-runtime comparison

**22 October, 09:34:54 UTC: final comparison complete; exact production restored.**
All six speed passes and both 300-question speculation-off GSM8K arms complete.
Production is healthy at 500k; both original identities and all 20 stopped trial
containers are independently verified. No promotion or further GPU run is queued.
Both Sparks remain on. [Final report](GLM53_SAME_RUNTIME_RESULTS.md).

CT's observed median gains are **+7.26% code C1, +15.49% chat C1, +16.07% code C8,
+14.85% chat C8**; memory allocation is **14.465825 GiB lower across TP2**, repeated
on all three loads. Every repeat-spread control exceeds 2%, so these percentages
are not qualified stable speedups. GSM8K is **291/300 production vs 287/300 CT**,
zero errors/truncations; seven production-only and three CT-only correct answers.
Both frozen speed-repeatability and observed-quality criteria fail: **retain
production, CT opt-in**, without claiming a general accuracy gap from 300 cases.

The user's current Spark2 rsync starts 09:07:58 UTC, after speed measurement ends
09:01:59; it overlaps both quality arms. No evidence attributes the answer gap to
uploading. The quality-only continuation fixes a missing staged scorer, not a
model/kernel issue; no timings/questions are retried and the deadline is unchanged.
Prefix hit/miss probes pass, but first-trial same-prefix one-token identity fails
for both arms. Cache reuse must not be described as numerical correctness.
This is the production-custom-runtime comparison; stock-plugin compatibility
remains the deployment requirement, and no KV/fast-path change was applied.

**22 October, 09:10 UTC: quality-only continuation is running detached.**
The isolated quality-client import/grading check and both-arm/rank CPU startup
checks pass. Controller **1234052**, independent watchdog **1234695**, stage
`gsm8k-A-0` loading. The absolute deadline is still **10:27:14 UTC**. No timing is
repeated; no GSM8K question has been retried. Production is paused again only for
the two missing quality passes, with exact restoration afterward. Quality results
remain pending. Source bundle `glm53-quality-continuation.bundle` is retained on
both nodes, SHA256 `a59fed12ffb1b2933924b4cd0e7d5d37be476e8d3f3c83ab7201c721b0a2d299`.

**22 October, 09:09 UTC: speed complete; quality-only continuation preparing detached.**
All six timing stages finish without API errors. V3 then stops before sending any
GSM8K question because its staged benchmark imports a missing `numeric_scoring.py`.
This is a second launcher staging error, not a model failure. Exact original 500k
production is restored at **09:07:26 UTC**, independently reverified on both ranks,
and all owned trial containers are stopped. V3 evidence stays unchanged.

Final three-trial medians (production / CT): code C1 **46.01 / 49.35 (+7.3%)**;
chat C1 **39.06 / 45.11 (+15.5%)**; code C8 **74.35 / 86.29 (+16.1%)**;
chat C8 **72.89 / 83.71 (+14.8%)**. Every arm/cell exceeds the frozen 2% spread
control; these observed gains do **not** pass speed qualification. Code C1 ranges
39.48–51.96 / 46.68–55.75 tok/s. All trials are retained, with no timing rerun.

Source **`2fde099`** stages the unchanged original numeric scorer and adds an
isolated CPU import/request/scoring check before any outage. Regression tests
reproduce the missing dependency and prove that an attempted question prevents
continuation. Full tests (one skip), Ruff and core mypy pass.

The already-authorized, unfinished quality scope continues in fresh root
`glm53-package-v1/same-runtime-final-quality-v1`, detached preparer **1230449**.
It schedules only A/B speculation-off GSM8K (same 300 questions, grading and
settings), requires the completed timings and exact prior restoration, and refuses
any quality-question retry. It preserves the original absolute watchdog deadline
**10:27:14 UTC** instead of starting a new two-hour lease. No gate, model, kernel,
overlay or serving setting changes. At this checkpoint CPU preparation is running;
quality results remain pending. No promotion or further experiment is queued.

**22 October, 08:27 UTC: corrected test is running detached on Spark1.**
All eight CPU checks pass (both arms, timing/quality, both ranks), including the
actual sparse-palette packer/native-library contract for CT. Controller **800461**
and independent watchdog **801127** are active; stage **`timing-A-0` is loading**.
Root: `glm53-package-v1/same-runtime-final-v3`. Production is now paused under the
unchanged two-hour restoration lease. The laptop is not required. CT GPU startup
and all benchmark results are still pending. The frozen eight-stage scope and
stop-on-error policy are unchanged; exact original production is restored afterward.

**22 October, 08:26 UTC: corrected comparison preparation running detached.**
The user asks to rerun the authorized final test. Source **`1c1f60a`** restores the
existing `palette_model_worker` registration and `MXWAVE_PALETTE_MODEL=1`; no
production overlay, serving setting, workload, decision rule or gate changes.
A new CPU check binds the actual registered packer to the loaded native library's
`sparse_palette` encoding and records source/binary hashes before any outage.
Regression tests reject the missing flag, wrong decoder and unregistered packer.
Full repository tests pass (one skip), plus Ruff and strict core mypy.

New root: `glm53-package-v1/same-runtime-final-v3`, detached preparer **792772**.
It validates both arms/modes/ranks before launching the same eight stages with a
two-hour independent restoration watchdog. At this checkpoint preparation is
running; GPU timing has not yet been confirmed. Failed v1/v2 evidence is retained.
The original v2 stop remains valid history below; its launcher bug is now fixed.

**Outcome: stopped on launcher error; production restored at 06:51:39 UTC, 22 October.**
Only A0's four timing cells completed. B0 failed during loading on both ranks with
`Native library and scale encoding differ`: the new launcher omitted the existing
`palette_model_worker` registration and `MXWAVE_PALETTE_MODEL=1`, leaving an
aligned-nibble packer paired with a sparse-palette native decoder. CPU checks
covered imports and flags, but did not check this binding. This was an error in
the comparison launcher; full overlay compatibility remains untested.

No candidate timings or GSM8K results exist from this run. The controller stopped
the remaining stages and restored exact original 500k production in 9.69 minutes
total. Follow-up verification confirms all original identities and all 16 trial
containers stopped. No automatic retry, workaround or promotion occurred.
The protocol below remains the frozen intent, not a claim that it completed.

Authorized 22 October 2026. One bounded session; no automatic promotion or further
tuning. User requests detached operation while commuting. Preparation is CPU-only;
the GPU controller starts only if both-rank, both-arm CPU checks pass. Any startup
or collection failure stops the remaining stages and restores exact production.

A and B inherit the actual original image, all overlays, private environment,
Inco drafter and serving flags. Only B's checkpoint and required existing CT
adapters differ: MXFP4 experts, MXFP8 KDA conversion, lossless expert-scale and
embedding packing, dense BF16 head. Both have identical observation hooks and
private API/collective ports. No overlay source is changed.

The frozen settings come from live Docker inspection, not the earlier narrative:
500k context, 32 slots, 6 GiB KV/rank, 6,919 batched-token budget, prefix caching,
original graph sizes, and the complete Inco draft policy (C1 seven, C2 five,
C3–C27 three, with original larger-batch entries retained). The live admission
flag is **4 ms**, so both arms preserve 4 ms; it is not silently changed to 10.

Retain hook activation reports as well as flags. In particular, CT's existing
post-load KDA conversion uses a direct Marlin helper, while some production dense
optimizations wrap the native kernel class and scan weights earlier during load.
Those dispatches may therefore differ even with identical enabled overlays.
This is a same-configuration deployment test, not proof of identical kernel paths
or an isolated lossless-compression comparison. No bridge/overlay fix is authorized
inside this run. A startup incompatibility stops it.

Eight stages: A0/B0/A1/B1/A2/B2 timing, then A/B quality. Timing has C1/C8 short
code/chat, one warmup per cell, 256 output tokens, fixed seeds and matched cold
prefix-cache salts. The frozen 300 GSM8K cases and grading are reused; speculation
is off in both quality arms. No production noise-floor rerun or long-context probe
is added. All output IDs, errors, timing outliers, native speculation counters,
queue/KV samples, per-rank allocation and clock/thermal/power telemetry are retained.

Decision rule, frozen before results:

- Positive deployment screen: CT median throughput is at least 97% of A in **both**
  C1 domains and exceeds A in **both** C8 domains; the existing 2% repeat-spread
  control passes for each arm/cell; CT correct answers are at least A's on the same
  300 questions, with no API errors or truncation. This is an observed quality
  screen, not statistical equivalence or a new KL pass.
- More than 5% C1 deficit in either domain: retain production; CT remains opt-in.
- A 3–5% deficit, unstable controls, mixed speed cells, or a lower quality-screen
  score: inconclusive/unfavorable; retain production and stop. No automatic reruns.
- Overlay/startup failure: stop and retain exact evidence. No fallback settings.

The independent watchdog has a **7,200-second outage lease**. Exact original
containers are restored and verified afterward, including 500k API readiness.
Both Sparks remain on. Logs and progress live on Spark1 under
`full-model-v1/glm53-package-v1/same-runtime-final-v2`; each rank retains its own
startup/CPU logs. The previous C1 diagnostic was committed as `fe13966` and backed
up on both nodes before this new scope.

The first preparation stopped without an outage: historical launch metadata was
present only on the coordinator. V2 copies the required adapter environment as a
frozen run input to both ranks. The protocol and serving settings are unchanged;
the failed v1 remains retained. Detached v2 preparer PID: 376651.
