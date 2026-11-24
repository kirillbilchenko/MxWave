# KDA fused-kernel routing crossover

**18 October 2026, 13:08 UTC: input-only exact16 V8 passes every unchanged
component gate; independently verified on Spark/Mac.** Relative-L2=0,
42 payloads/19 bindings/24,576 events. Fused435–487µs, reconstruct+GEMM
1.34–1.38ms, resident BF16 GEMM463–514µs. All eager/graph, warm/pressure and
both-order controls are within2%; each leg retains clock/thermal telemetry.
No clocks are locked. V7's earlier control failure remains unchanged.
Archive `76b00c670bf2747448afe35ca9e7f011747b5c3d17ac0b6fca9caa930936b34c`.
The exact16 input component can proceed to isolated model integration;
rows9–15, a global limit and model quality remain unqualified. Original service
is restored and healthy.


**17 October 2026, 22:07 UTC: V5 and bracketed V6 are complete and independently
verified on Spark/Mac.** Original service restored healthy at **22:03:41 UTC**.
No serving threshold, model weight or qualified library changes.

## Completed measurements

Both processes finish all planned cases with exact decoded weights and stock
outputs, unchanged guards, and fused arithmetic below the existing 1% relative-L2
gate. V5 retains **20,480 events** across forty matrix/row cases. Small-row
fused execution is substantially faster, but the repeated unchanged control
exceeds 2% in many conditions, leaving no shared eligible row count. These
failed gates remain part of the result.

V6 prospectively refines the timing method: every target gets its own before/
after reference in eight balanced ABBA/BAAB blocks; the graph control replays
the **identical reference graph**, rather than a separate capture of the same
callable. Each of four legs retains four events. A fresh activation seed and
ten fixed row counts include 257/2,048-row negative controls. All numerical,
2% control and 5% gain gates remain unchanged. It retains **30,720 events**.

| Rows | V6 input fused change | V6 output fused change | Pass all conditions on both matrices? |
|---|---:|---:|---|
| 8 | −65.78% | −48.59% | Yes; already inside the current fused route |
| 12 | −65.14% | −57.05% | No; output control variance |
| 16 | −62.58% | −59.36% | No; control variance |
| 24 | −64.68% | −57.22% | **Yes; component lead** |
| 32 | −65.44% | −59.66% | No; output control variance |
| 64 | −58.55% | −56.13% | No; control variance |
| 128 | −34.78% | −50.60% | No; control variance |
| 256 | −2.35% | −9.24% | No; control and/or performance gates |
| 257 | +11.79% | +8.45% | No; fused regression |
| 2,048 | +218.92% | +163.33% | No; fused regression |

Displayed changes are combined paired eager/warm ratios; negative is faster.
Eligibility uses **all** eager/graph, warm/pressure and both-order medians,
not only this column. V6 per-matrix eligible sets are input [8,12,24,32] and
output [8,24]; their intersection is **[8,24]**. Peak CUDA allocation is
917,537,280 bytes in V5 and 878,473,728 in V6. The largest observed relative-L2
error is approximately 0.383% in V5 and 0.3785% in V6, at small output-projection
shapes. Exact restored weights do not imply identical fused GEMM arithmetic.

The crossover region is around 256 rows, with a clear padding/performance
change at 257. These data do not qualify a blanket threshold through 256,
and 24 rows still needs a real-model qualification before serving. Do not
repeat the unchanged sweep to seek a passing control. A C1 MTP pilot can keep
the existing eight-row limit: shallow verification fits it. That is the next
larger performance experiment after native CT loader/alias compatibility.

## Independent evidence

- V5: 35 outer / 13 native payloads, every event recomputed, byte-identical
  Spark/Mac review. Archive SHA256:
  `5927b45ce9e7033e9d0508aaf5ebc5455997e17679eba7bfd50f2a7df096f04c`.
- V6: 36 outer / 13 native payloads, every ABBA/BAAB leg recomputed,
  byte-identical Spark/Mac review. Archive SHA256:
  `3945d1dee2681246a00504027bbc617ac5da312750996eff43b10987dd3f1f9f`.
- [V5 review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/prefill-routing-v5-review.json)
  and [V6 review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/prefill-routing-v6-review.json).
- Both exact original identities/configurations and HTTP health 200 are
  verified. V5 restoration: 21:55:59 UTC; V6: 22:03:41 UTC.
- Local validation: **594 tests passed, one GPU test skipped**, full Ruff and
  strict core mypy passed. Full-model numerical qualification remains open.

## Question and fixed comparison

The current adapter uses the qualified fused split-K=1 operation only up to
eight flattened activation rows. Above that it reconstructs the complete KDA
weight and calls stock BF16 GEMM. One profiled full prefill rebuilds 68 matrices
for about 36.26 ms/rank. Speculative verification can also cross the eight-row
limit. Earlier large-prefill tracing used split-K=4; it does not establish
the crossover for the current split-K=1 path.

Two existing real fixtures are tested: layer-zero/rank-zero input
12,576 × 4,096 (12,608 padded rows), and output 4,096 × 4,096. Reconstruct
the CPU checkpoint slices and compare their hashes before using the prepared
packed artifacts. Decode must recover exact padded bytes and preserve guards.

Four arms share the same activation values:

1. Qualified decoder into caller-owned BF16 storage, then stock GEMM.
2. The identical decode-plus-GEMM callable again, as a repetition control.
3. Stock GEMM on the original resident BF16 matrix.
4. The existing qualified compressed GEMM with split-K=1, including its
   normal token padding and contiguous logical output.

Row counts: **1, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128, 256, 257, 259, 512,
513, 514, 769, 1,024, 2,048**. Each uses eager CUDA-event measurements and
graph replay separately, warm and 128 MiB cache-pressure conditions, eight
alternating-order trials and four samples per arm/trial. Pressure runs before
each event pair. Retain all 20,480 event samples if the full sweep completes.

The stock outputs must match source-GEMM bytes. Fused output must be finite
and pass the existing real-projection relative-L2 threshold **strictly below
1%**. Fused eager and captured outputs must match each other exactly.
Numerical failure remains a failed case; it cannot be traded for speed.

A tested row count is a component lead only if the repeated stock control is
within 2% and fused execution is at least 5% faster for both execution modes,
both cache conditions, both order medians and the combined paired median.
Report each matrix separately and their intersection. A lead does not qualify
unmeasured intermediate row counts, real model activations or serving routing.
No automatic threshold update is allowed.

## Bounds and evidence

Each GPU process has a 180-second timeout and six GiB host RAM limit; timing
has a 1.5 GiB CUDA allocator cap. An idle-only controller and independent
watchdog restore both original containers with exact configuration checks.
The decoder smoke reuses the immutable GPU-qualified V4 build; the routing
arms use the unchanged qualified library. No complete model is allocated.

- V5 controller PID **2975102**, watchdog **2976658** (completed).
- Source: `ROOT/prefill-decode-v5/src`, 16 files.
- Completed state: `ROOT/prefill-decode-controller-prefilldecodev5.json`.
- Worker result: `ROOT/prefill-decode-v5/results/routing.json`.
- Local continuation: `/private/tmp/mxwave-prefill-decode-v5/`.
- Source archive SHA256: `7f45a621945ade246f2b8a10b3dfe773b9d3ec17e5eb808d3bd658d39f260f3b`.
- Plan SHA256: `d98dd6b0692b9154e0422eec2c1456e010b02579f05543bba35e5221cc848ba6`.

V6 controller **3024920**, watchdog **3026544** (completed); source/results
under `ROOT/prefill-decode-v6/`, local `/private/tmp/mxwave-prefill-decode-v6/`.
Its 17-file source archive SHA256 is
`1d29724e09e817bc95d9204b31f4882eb9208cc94c6ad84012da8f7d168fd0ac`;
plan SHA256 is `1314dbfb66da266a1a34415bc977d32db61f75577f12bae15ef989e64f68a757`.
