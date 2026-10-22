# Final same-production-runtime comparison

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
