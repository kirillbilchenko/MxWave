# KDA prefill reconstruction experiment

**19 October 2026, 05:48 UTC: historical full-model trace follow-up, offline only.**
The [retained review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/warmed-model-v1-prefill-windows-v1-review.json.gz)
covers all six candidate warmed-model traces, 28 prefill forwards, 1,904
reconstructions and 1,876 windows between consecutive reconstructions. All
28 forwards contain 34 input and 34 output reconstructions; trace hashes and
launch ownership/counts match the retained independent review. Both ranks
cover physical rows 54, 224, 525, 792, 997 and 2,048. No GPU experiment is run
by this analysis, and it uses no result from the active FP32 model comparison.

The trace device properties report **102,400 shared-memory bytes/SM**. This
review uses actual allocated bytes/block, including allocator overhead, for
every observed pair. Expert work between KDA reconstructions precedes the
next input projection. At 2,048 rows its blocks allocate **102,400 bytes**;
the decoder needs another **21,248-21,888 bytes**, so these blocks cannot
reside on the same SM. At observed 54-997 rows, expert blocks instead allocate
**33,792 or 50,688 bytes**, which leaves shared-memory capacity for a decoder.
This smaller-row case is a distinct hypothesis from the previously failed
three-projection GEMM overlap; it is not a measured speedup or an integration
decision. Registers, occupancy, dependency/buffer lifetime, bandwidth and
scheduling still need checks. Blocks can also execute on different SMs, so
same-SM incompatibility is not a global impossibility of overlap.

Across these mixed/controlled profiles, total decoder duration is about
**35.8-39.2 ms/forward/rank**; the 2,048-row median is about **36.48 ms** on
each rank. Kernel duration sums are not API TTFT or a critical-path estimate.
The previous short-C1 TTFT **0.384 -> 0.422 s** remains a measured regression;
similar cost magnitudes alone do not prove its cause. Short-C16 also has
different admission grouping. No original gate is changed. Preserve the failed
naive overlap and decoder-occupancy experiments below; any different overlap
test must be bounded and justified by buffer lifetime and actual kernel pairs.

The [retention receipt](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/fp32-input-model-v1-offline-preparation-retention.json)
binds the compressed report and analyzer bytes to historical archive
`0fc867240d04e1770e2970143f48f630551feb35f53b739403c4fb6b8bde64e0`.
CPU validation covers nested interval unions and allocated-memory precedence;
the new latency/profile helpers also reproduce all 72 historical timing groups
and 12 profiles exactly. All candidate prefill windows, including zero-work
windows if any, remain in the report; no favorable subset is selected.

**17 October 2026, 21:45 UTC: V3 performance evidence and V4 mask regression
independently verify on Spark/Mac. Original service restored healthy after
V4 at 21:43:57 UTC. No new speed improvement is accepted.**
This is a bounded component test, with an idle-only pause and independent
restoration watchdog. It does not load a model or change the serving adapter.

## Why this is the next target

The [verified paired C4 profiles](GLM53_PAIRED_SPEED.md) contain 68 explicit
candidate KDA weight reconstructions, totaling **36.259 / 36.263 ms** on
ranks 0/1 per profiled prefill. Prefill spans increase **6.6–6.7%**, while
decode spans decrease about **8%**. First-token latency is higher in all six
ordinary API workloads, and concurrent throughput remains close to parity.
Split/mixed admission occurs in the slower C4 groups. These are observed
costs and associations; kernel category sums do not partition wall time.

Earlier fused-prefill and naive overlap attempts were slower. This experiment
instead changes the standalone integer decoder used before the same stock
BF16 GEMM. It removes a **9,216-byte shared output tile**, the final block
barrier and shared-to-global output copy. One variant preserves the original
pair-to-lane mapping; the other writes complete rows with coalesced stores
and reuses bitmap loads. These are source-level mechanism descriptions;
The measured result below passes correctness and fails the performance gates.

## Measured V3 result

All three synthetic fixtures pass the qualified source-byte check and all
three decoder variants, including guards, non-default streams and graph
replays. Both real KDA matrices reconstruct exactly, and every coupled stock
GEMM output matches bytes at all eleven token shapes. Peak allocated CUDA
storage is **966,822,400 bytes**, within the fixed 1.5 GiB cap.

Neither candidate meets the required 5% pure-decoder improvement or all-case
2% coupled nonregression gate. Rebuilt-control parity also fails, so small
coupled differences are inconclusive. The controller correctly skips the
conditional confirmation run; no serving adapter or model-level result changes.

| Matrix / cache | Qualified median, µs | Rebuilt stock paired change | Direct pair paired change | Coalesced rows paired change |
|---|---:|---:|---:|---:|
| KDA input / warm | 842.30 | −0.17% | +0.16% | +2.06% |
| KDA input / pressure | 856.61 | +5.75% | +8.12% | +1.52% |
| KDA output / warm | 281.72 | −0.32% | +1.77% | +0.18% |
| KDA output / pressure | 280.58 | −0.20% | +4.04% | −0.03% |

Changes are combined medians of paired trial ratios; positive means slower.
Both execution-order medians remain in the raw report and in the frozen gate.
All **6,656 CUDA-event samples**, four pure measurements and 44 coupled
measurements are retained. No slow case or order is removed.

Actual resource queries show input occupancy capacity **4 → 8 blocks/SM**
for both variants, and output **6 → 12 / 9** for direct/coalesced stores.
Dynamic shared storage falls by exactly 9,216 bytes per block. This improvement
in resource capacity does not produce a measured reconstruction speedup.
The result does not establish which hardware resource is limiting execution;
logical weight bytes and cache pressure are not measured LPDDR traffic.

For each retained arm, logical packed reads plus padded BF16 writes divided
by its median event time give the following decimal GB/s. No hardware traffic
counter or matched copy-bandwidth baseline is available; these payload rates
do not establish 70–85% of measured peak LPDDR bandwidth.

| Matrix / cache | Qualified GB/s | Rebuilt stock GB/s | Direct pair GB/s | Coalesced rows GB/s |
|---|---:|---:|---:|---:|
| Input / warm | 209.76 | 209.58 | 209.28 | 206.02 |
| Input / pressure | 206.25 | 194.23 | 191.04 | 204.90 |
| Output / warm | 203.62 | 203.88 | 200.32 | 202.96 |
| Output / pressure | 204.45 | 204.23 | 196.57 | 204.46 |

The exact logical payload is 176,679,640 bytes for input (73,394,904 packed
plus 103,284,736 padded output), and 57,364,200 bytes for output (23,809,768
packed plus 33,554,432 output). Both original failed attempts remain retained.
The V3 archive has 36 outer payloads and the unchanged 13-file native build;
all 6,656 event samples recompute, with byte-identical independent reviews
on Spark and Mac. Archive SHA256:
`004472e31ba216790982307a5cf13152fd981432786f293c313994090ab630cb`.

The evidence and exponent-base correctness work below are complete. Next
measure the existing fused
compressed GEMM against reconstruction plus stock GEMM for small prefills.
Test matched batch admission after that crossover measurement. Keep these
unchanged decoder variants out of serving; another full-model load is not
justified by this component.

## Exponent-base fix and checkpoint-wide audit

The candidate header now reconstructs the exponent with
`((exponent + code) & 255u) << 7`. This prevents unsigned base wrap from
leaking into the BF16 sign bit. Candidates accept bases 0–255; mode 0 and
qualified KDA/head writer or loader paths reject bases above 248 before
calling the unchanged external decoder. Local boundary tests pass. The new
header is not the frozen V1/V3 library: V4 uses a separate build and proves
the corrected candidates on actual base-255 fixtures.

The CPU-only `prefill-base-audit-v1` completed in **51.77 seconds**, with
838,758,400 bytes peak host RSS, a two GiB limit and no CUDA/device access.
It recomputes all 136 KDA TP shards and
requires their hashes to match actual previously loaded weights, then calls
the qualified packer's exported CPU exponent analyzer. Both compressed head
metadata records and both embedding base arrays are included. The embedding
codec uses `base + code - 1`, so its safe maximum is 249; the ZipServ field
uses `base + code`, so its safe maximum is 248. No full model is allocated
and serving stays healthy. All 68 input shards use base 115; 66 output
shards use 115 and two use 116. Both head bases are 115. The 154,880 embedding
rows have bases 105–116. **No unsafe target or embedding row is present.**
All KDA logical hashes match actual previously loaded weights. The twelve
evidence payloads independently verify locally; archive SHA256:
`f051a014a7358aff26b4311b347cd598d6a1b3280ca4075dc6d1ca304a1817a8`.
Thus the base bug is real but is not exercised by this checkpoint. The
fresh CPU oracle reproduces 243 errors in the original fixture and 99,200 in a
512 × 512 fixture containing all 65,536 BF16 words, and recovers every source
byte with the mask.

## V4 fixed-header GPU regression

The separate corrected SM121 library passes three safe-base controls and
both base-255 fixtures. All thirteen mode/case combinations pass eager and
non-default-stream byte checks, six graph replays each, unchanged destination
guards and zero replay allocation growth. Peak allocated CUDA memory is
**2,003,456 bytes**. The unsafe stock control is refused by both its Python
wrapper and the C ABI; the external qualified decoder is never launched
with unsafe metadata. Its library remains unchanged.

The controller runs correctness only, with no model load or timing stage.
Both original configurations and HTTP health 200 are verified after restoration
at **21:43:57 UTC**. All 34 outer and 13 native payloads independently verify
on Spark/Mac, with byte-identical review JSON. Local checks: **589 passed,
one GPU test skipped**, full Ruff pass, strict core mypy pass (35 files).
This closes the candidate reconstruction defect; it does not qualify complete
model numerics, change the serving adapter or demonstrate a speed gain.

- Header SHA256: `d2ce4515f7b8f08346b6ff60d079e5f1f1d2285595769569ff6f183045463200`.
- Separate library SHA256: `b408ee25be91520c8da132f921b0a0419fdd3dfc8f53baeb9a7562ca04c7328e`.
- Evidence archive SHA256: `faa7a0f9554971b0422378f7c68fc64b9689cc75eacb641095b26eb13da61f54`.
- [Compact independently verified review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/prefill-decode-v4-review.json).

## Frozen component contract

- Three synthetic matrices cover zero values, raw BF16 words including
  non-finite exceptions, and random BF16 values. Require exact decoded bytes,
  destination guards, non-default streams, six graph replays per variant
  and zero replay allocation growth.
  V3 retains every original 16,384-word raw pattern inside a 256 × 256
  common-exponent-dominated fixture, and requires an independent CPU inverse
  before CUDA initialization. The base-255 regression remains separate.
- Two existing real fixtures: layer-zero/rank-zero KDA input
  **12,576 × 4,096** (padded to 12,608 rows) and output **4,096 × 4,096**.
  Independently reconstruct the original checkpoint slices and verify their
  logical/padded hashes before timing. No complete model is allocated.
- Four arms use the same destination: the currently qualified library,
  a rebuilt unchanged decoder, direct pair stores, and coalesced row stores.
- Pure reconstruction plus reconstruction followed by the unchanged stock
  GEMM at **16, 128, 256, 257, 259, 512, 513, 514, 769, 1,024 and 2,048**
  activation rows. This includes shapes observed in mixed-admission counters.
- Eight trials retain both execution orders, warm replay and 128 MiB cache
  pressure outside the event pair. Keep every raw event sample. Pressure is
  a controlled perturbation, not measured LPDDR traffic.
- The rebuilt control must remain within **2%** of the qualified decoder.
  A candidate needs at least **5% faster pure reconstruction** for both real
  matrices and no coupled case more than **2% slower**. All gates apply to
  both order medians and the combined paired median. Require a separate
  confirmation process, reversed initial order and a new activation seed.
- Each GPU process has a 180-second limit and 6 GiB container RAM limit;
  CUDA allocator caps are 1 GiB for smoke and 1.5 GiB for real timing.
  The independent watchdog enforces restoration after the bounded pause.

Passing this component would justify an opt-in integration and matched
model-level measurement. It would not establish a token-throughput gain or
resolve historical full-MLA numerical qualification failures.

## Preparation and preserved failures

The separate SM121 library compiled successfully with NVCC 13.0.88. The first
unpaused smoke attempt, `prefilldecodev1`, failed at `torch.cuda.set_device(0)`
with an out-of-memory error before any fixture or decoder kernel ran. Its
empty-case report and logs remain preserved. The original service was left
running. No correctness or speed decision can be drawn from that attempt.

For `prefilldecodev2`, the 11-file source snapshot and plan match on both nodes.
A bounded CPU-only container passed imports and all CLI checks, reporting
`cuda_initialized: false`. The local full test suite and Ruff pass; strict
core mypy passes all 35 files. Automatic approval review initially rejected
the source upload. The user then explicitly approved that exact source-only
payload, and its unchanged archive was uploaded successfully. This does not
authorize the separate, previously rejected optional document/report upload.

V2 stopped in smoke: the qualified decoder failed the original raw-word
fixture, before real timing. The zero fixture passed all three variants,
including exact streams, six graph replays and guards. Original identities
were restored with health 200 at **21:03:35 UTC**. Independent Spark/Mac
verification checks all **36 outer payloads and 13 native-build payloads**;
the two review JSON files match bytes. No timing or candidate win was accepted.

The independent CPU inverse reproduces **243 mismatched words** on the same
raw source pattern. A common exponent range beginning at zero makes the
stored unsigned base wrap to **255**; the existing reconstruction adds the
code before shifting, causing a sign-bit error. Applying an eight-bit mask
to the reconstructed exponent recovers all original source bytes on CPU.
The packed values retain the information. This is an existing metadata/decoder
edge case; the GPU failure report did not retain its mismatched output words.
It does not establish corruption of the real KDA fixtures, whose base is 115,
or explain the separate full-MLA repeat variability.

V3 keeps the original native library and all exact-byte/performance gates.
It rejects wrapped bases before invoking the experimental decoder and uses
the larger fixture to test the original raw words with a representable base.
CPU preflight verifies the original tail hash, every supported source word
and the masked inverse of the separate retained underflow regression.
The 12-file snapshot matches both nodes; CPU checks use **540.7 MB peak RSS**
and initialize no CUDA. The local full suite, Ruff and strict core mypy pass.
The user explicitly approved ongoing source-only revisions for this decoder
experiment on the existing Spark nodes after the automatic review rejections.

## Evidence and continuation

Local frozen source and plan: `/private/tmp/mxwave-prefill-decode-v3/`.
Spark ROOT: the `full-model-v1` root recorded in the
[current-state page](GLM53_CURRENT_STATE.md).

- Controller: `ROOT/prefill-decode-controller-prefilldecodev3.json`;
  PID **2710212**, launch **21:13:45 UTC**.
- Component source, plan, prepared hashes, logs and reports:
  `ROOT/prefill-decode-v3/`.
- Native build and the preserved first smoke attempt:
  `ROOT/prefill-decode-v1/`.
- Library SHA256:
  `90abfb73007f2dab8950ad50f04408cb7a0556dd41396df467c5818b41e69329`.
- Header SHA256:
  `6cd0676cac86084e4e0b7c01f5f49e49c007a4aa478b4755e113388284cdc715`.
- Frozen source archive SHA256:
  `346bef5c67615e886f364ec95742e18dbaff35bd5d123b99cb369eac3c5c4aa8`.
- Plan SHA256:
  `f90c9084162c7b5e455e7a8933cd3a490d3140a74d10799ac857a4025c3ec18e`.
- Preserved V2 evidence archive SHA256:
  `1cb52a5c84ad7ae7c7b7213353efdb752543dbdb04418db31d5d8bd32cf242f4`.
- V2 Spark/Mac independent review SHA256:
  `11d8e67191a24e60cec80cc42b7b6c7b7ef8c07ddc131e23084d2b899883c06e`.

After restoration, the frozen independent verifier must recompute every
decision from retained events, validate source/native bindings, and verify
the original container identities. Update this page with the actual outcome
before claiming a speed improvement. Roofline export remains deferred.
