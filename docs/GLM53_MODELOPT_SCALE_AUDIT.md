# ModelOpt lossless8: CPU inventory and NVFP4 scale audit

**Subsequent GPU component validation completed at 19:03 UTC.** The selected
lossless codec saves 3.250086 / 3.252740 GiB in full-checkpoint scale storage
accounting and lowers tested eight-row expert-layer time by 1.90% / 3.07% against
installed stock, with bitwise-exact covered outputs. Actual model VRAM, serving
speed and full-model quality remain unmeasured. See the [component follow-up](GLM53_NVFP4_COMPONENT.md).
The CPU audit below is preserved as its earlier dated checkpoint.

**Complete, 19 October 2026, 17:59 UTC.** The CPU screen supports a bounded
NVFP4 Marlin component prototype. Source scales occupy **17.71875 GiB total**;
the practical layouts below estimate **3.25–3.30 GiB saved per TP2 rank**.
That removes about **4.1% of expert weight-plus-scale bytes**. GPU allocation,
traffic, latency, throughput and full-model quality are not measured by this audit.

The serving containers were unchanged and healthy after the audit. The existing
CT candidate remains on port 8000 with **8,192 context**; original production
remains stopped. No ModelOpt adapter or GPU experiment was started. The objective
remains better end-to-end performance than the original tuned ModelOpt/DFlash
deployment, preserving its checkpoint and specialized inference paths.

## Inventory of the actual checkpoint

Source: `/home/kirya/local-spark/models/glm-5.3-flash/derived/glm-quant-mix/lossless8`.
Configuration declares ModelOpt mixed precision and NVFP4 group size 16.
The audit follows all **148,070 active index entries** across 35 referenced shards.
The 45-layer target has 42 routed expert layers, 288 experts per layer and three
projections per expert: **36,288 scale tensors**, each 524,288 bytes.

| Target tensor group | Source dtype / format | Total GiB | Even TP2 estimate, GiB/rank |
|---|---|---:|---:|
| Expert weights | U8, two packed NVFP4 values/byte | 141.750000 | 70.875000 |
| Expert block scales | F8_E4M3, one scale/16 weights | 17.718750 | 8.859375 |
| Expert auxiliary scales | F32 | 0.000270 | — |
| LM head | BF16 | 1.181641 | 0.590820 |
| Embeddings | BF16 | 1.181641 | 0.590820 |
| Quantized projection weights | Packed NVFP4, MXFP8, block FP8 | 6.518066 | — |
| Projection scales/auxiliary | F8_E4M3, U8, F32 | 0.160969 | — |
| Other tensors | BF16, F32 | 1.851985 | — |

These are checkpoint payload bytes, not CUDA allocations. Even TP2 estimates
divide shardable source storage by two; actual layout, replication, alignment
and loader staging need runtime measurement. Expert scales are exactly **11.11%**
of expert weight-plus-scale bytes. The user's initial 4.5–5 GiB/rank scale
estimate understated this checkpoint's source scale storage.

Projection weights break down into 226,492,416 bytes packed NVFP4,
4,607,967,232 bytes MXFP8 and 2,164,260,864 bytes block FP8. Source-byte inventory
does not imply an attractive codec for those groups. KDA/MXFP8 compression is
outside the proposed next component test.

Two disk-only categories are kept separate: active index entries for exported
layer 45 contain **14,865,185,408 bytes (13.844283 GiB)** outside the 45-layer
target; referenced original shards also retain **408 superseded BF16 tensors
(13,544,456,192 bytes)** whose active index entries point elsewhere. Neither is
counted in target expert-scale savings. The complete dtype/algorithm inventory
is [retained here](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/nvfp4-scale-audit-v3-inventory.json).

## Entropy and concrete storage layouts

Every target scale byte was scanned, with one SHA-256 per tensor. Global
zero-order entropy is **3.772516 bits/byte**; size-weighted per-tensor entropy
is **3.358676 bits/byte**. These are information bounds, excluding indexing and
decoder costs. Individual tensors contain 22–67 distinct byte values:
35,718 need five-bit indices, 569 need six bits and one needs seven bits.

| Layout | Stored bytes, total | Estimated GiB saved/rank | Scale-byte reduction | Expert weight+scale reduction |
|---|---:|---:|---:|---:|
| Fixed per-tensor indices plus palette | 11,929,281,199 | 3.3044 | 37.30% | 4.14% |
| Four-bit palette plus block escapes | 12,041,307,456 | 3.2522 | 36.71% | 4.08% |
| Two-bit palette plus block escapes | 16,260,706,881 | 1.2874 | 14.53% | 1.61% |
| Five-bit indices or raw, with estimated TP metadata | 12,006,365,568 | 3.2685 | 36.89% | 4.10% |

Fixed indices use `ceil(log2(distinct))` bits plus one byte per palette entry;
that row excludes runtime descriptors/alignment. Escape layouts have 32-scale
blocks, a four-byte exception offset per block, literal one-byte escapes, and
3/15-byte palettes for two/four-bit selectors. Their size counts cover all scales;
codec roundtrips cover the stratified samples below. TP palette duplication,
descriptors and runtime alignment remain additional costs for these rows.

The last row is a prospective simpler layout: encode the 35,718 tensors that fit
32 symbols with five-bit selectors, leave the other 570 raw, budget a 32-byte
palette per encoded TP shard and a 16-byte descriptor for every TP shard.
It is a storage calculation, not a materialized or timed GPU implementation.
The five-bit design avoids variable-length escapes but introduces bit extraction
and palette lookups; the GPU test must decide whether that trade pays off.

504 stratified samples cover every expert layer and projection, expert IDs
0/95/191/287, and three 16-KiB windows per tensor (start/middle/end):
**23.625 MiB total**. All **1,008** two/four-bit block-codec roundtrips and both
zlib levels are byte-exact. Zlib retains 51.77% at level 1 and 49.63% at level 6;
those are sampled CPU results, not a proposed GEMM decoder. Eight local
edge/random codec checks also pass.

## Runtime boundaries and the next component test

Installed `nvfp4_marlin_process_scales` converts source E4M3 scales into Marlin's
special **S0E5M3 bytes**, after permutation and power-of-two rescaling with
compensating global scales. Its returned storage is viewed as `float8_e4m3fn`,
but the bit interpretation is different. The existing MXFP4 consumer assumes
E8M0/group 32 and cannot be reused unchanged for NVFP4/group 16.

The next useful experiment is therefore a component prototype against the
unchanged installed NVFP4 Marlin path:

1. Bind the exact post-Marlin scale bytes, TP partition, global scales and source
   hashes. Verify palette feasibility after packing and exact reconstruction.
2. Decode scales inside the existing consumer while preserving weight codes,
   global scales, routing and arithmetic. Include decode scratch and metadata
   in memory accounting; rebuilding the complete scale tensor each step would
   not establish the proposed traffic saving.
3. Compare complete expert GEMM outputs and time at observed C1/C8/C16 shapes
   plus representative prefill shapes. Use interleaved repeated controls and
   record clocks/thermals, graph replay, scratch and actual GPU allocation.
4. Only after the component passes, compare the full unchanged production
   ModelOpt/DFlash configuration with compression off/on, retain the 1e-5 KL
   gate with matched qualification batches, and measure real end-to-end latency.

The 4.1% figure is an expert storage-byte reduction. It does not establish a
4.1% speedup, bandwidth saturation at all concurrency levels, or fewer total
decode bytes. Routing locality, cache reuse, activation/compute work and decoder
overhead can materially change the result. PPL, task accuracy, SQNR, KL,
throughput and TTFT remain unmeasured for this NVFP4 port.

## Vocabulary port correction

All four production vocabulary TP source shards match the earlier CT vocabulary
bytes, with about **0.697218 GiB TP2** potential storage saving. However, original
production enables certified target argmax and an FP8 draft-head path. Its
certificate builder requires an unquantized method and contiguous BF16 head,
retains a dense `State.W` reference, and reads those rows during exact refinement.

Replacing the quantization method would fail that guard; clearing only
`lm_head.weight` would leave the dense certificate reference alive. A useful port
must explicitly support compressed candidate-row refinement and preserve the
draft/target behavior. The earlier compressed-head speed result used common
settings without these production optimizations and does not prove a win over
the specialized production head. The earlier **2–4 hour** loader-port estimate
was too optimistic for this integration. No such adapter was implemented.

[Read-only compatibility findings and source hashes](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/modelopt-vocab-v1-compatibility.json)
record this correction without changing the earlier audit artifacts. NVFP4
expert scales now have the stronger CPU feasibility evidence.

## Execution, retained failures and evidence

The successful v3 scan took **45.15 seconds**, with peak RSS **417.77 MiB**, in a
container limited to two CPU cores and 2 GiB RAM, no network and no GPU devices.
It exited zero. V1 stopped before scanning on the extra layer; v2 stopped before
scanning on superseded shard entries. Both failed source versions and logs are
retained. V3 follows the active index and accounts for both exceptions explicitly.
All three containers are terminal. Final container identity checks and health
HTTP 200 confirm that serving was unchanged.

The [complete evidence archive](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/nvfp4-scale-audit-v3-evidence.tar.gz)
contains 15 payloads, including the auditor, per-tensor hashes/counts, samples,
inventory, launch limits, failed attempts and final service check. Archive size:
**3,502,373 bytes**, SHA-256:
`83a197b9adb3eea005dedf365532250ad127651d05c1d4854b28b8158d6ba141`.

[Mac review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/nvfp4-scale-audit-v3-mac-review.json)
verifies all payload hashes and independently recomputes coverage, inventory sums,
global entropy from the retained histogram, layer-histogram reconciliation,
weighted reduction of recorded tensor entropies, layout bytes and sample totals.
It does not reread the raw checkpoint on Mac or independently reconstruct each
per-tensor histogram. [Repository retention hashes](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/nvfp4-scale-audit-v3-retention.json)
bind the copied reports and compatibility review. No model quality or GPU speed
claim follows from these CPU checks.
