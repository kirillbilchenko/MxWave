# NVFP4 lossless scale consumer: component validation

<!-- NVFP4_RESULT_BEGIN -->
**Complete and independently reviewed, 19 October 2026, 19:03 UTC.** The selected
lossless byte-offset consumer passes bitwise component checks against installed
stock. At eight rows, the complete expert layer takes **1.90% / 3.07% less time**
on the two TP partitions. Full-checkpoint encoded-storage accounting gives
**3.250086 / 3.252740 GiB saved per rank**, including codec metadata.
This supports a serving adapter trial; no ModelOpt adapter has been installed,
no whole-model speed or KL result exists for this port, and serving is unchanged.

| Eight-row full expert layer, SM121 | Installed stock, µs | Compressed, µs | Time reduction | Raw-repeat gap |
|---|---:|---:|---:|---:|
| TP rank 0 | 1,773.917 | 1,740.155 | 1.903% | 1.193% |
| TP rank 1 | 1,755.222 | 1,701.288 | 3.073% | 0.464% |

These are medians from 32 interleaved trials, 60 graph replays per arm/trial.
Both partitions have 25/32 paired trials faster than installed stock. The rebuilt
raw control is within 0.16% of installed stock. Eight rows resemble C1 DFlash
verification geometry; these numbers are **not end-to-end C1 serving throughput**.

The broader matrix compares against the matched rebuilt raw control:

| Rows | Rank 0 layer-time reduction | Rank 1 layer-time reduction |
|---|---:|---:|
| 1 | 3.234% | 2.056% |
| 8 | 2.612% (accepted repeat) | 3.248% |
| 16 | 3.339% | 4.237% |
| 64 | 3.889% | 4.149% |
| 256 | 3.504% | 4.480% |

All listed whole-layer controls pass the unchanged 2% repeat limit. Rank 0's
original eight-row cell failed that control; its unqualified speed result remains
retained and is replaced in this table only by the separately recorded repeat.
<!-- NVFP4_RESULT_END -->

## Purpose and scope

The original tuned production checkpoint already uses NVFP4 experts and MXFP8
KDA. This experiment extends the lossless storage technique to that checkpoint,
preserving its weights and arithmetic. The existing MXFP4 implementation remains
available. Changing quantization format is not required to use compression.

The source is the actual ModelOpt `lossless8` checkpoint bound by the
[CPU audit](GLM53_MODELOPT_SCALE_AUDIT.md). The component prepares real layer-23
expert weights with installed vLLM repacking, S0E5M3 scale processing and global
scale compensation. It exercises both TP2 partitions **sequentially on one
SM121 GPU**. Inputs are seeded synthetic BF16 activations and top-eight routing;
there is no inter-GPU reduction or whole-model inference measurement here.

Each cell compares gate/up GEMM, down GEMM and the complete
GEMM–SiLU–GEMM–sum layer against installed stock. Raw and compressed native
libraries share the same tile plan, weight codes, global scales and reduction
policy. The routing block is fixed at eight rows. This matches installed stock's
selection for the complete 288-expert layer at the tested row counts; the
16-expert fixture is a reduced stress test, not a serving concurrency benchmark.

Rows 1/8/16/64/256 cover small decode, speculative verification and short
prefill shapes. A row count is not client concurrency. Full-model latency,
TTFT, throughput, DFlash acceptance, KL, PPL, SQNR and task accuracy are unmeasured
for this port. Component equality does not replace that qualification.

## Consumers and exactness

All variants encode post-Marlin byte storage, not floating-point approximations.
The first uses five-bit palette indices and direct shared-memory expansion. The
second pipelines low/high bit-plane reads asynchronously, expanding a ready
stage before its existing barrier. The selected offset variant replaces palette
lookup with exact integer byte offsets when a part's span is at most 32 values.
Other expert/projection parts retain their raw bytes. None changes weight codes,
global scales or GEMM accumulation order.

The reduced CPU range check found 95/96 sampled TP projection slices eligible
for offsets. The GPU checks validate the actual post-Marlin bytes, including
mixed raw fallback and graph replay. Buffers include low/high planes, palettes,
raw fallback and pointer/shape descriptors. Allocator rounding, whole-model
loader peaks and freed CUDA allocation require a production adapter measurement.

The measured layer contains no separate reconstruction launch. Runtime sources
are [documented here](../integrations/glm53_nvfp4/README.md); the written upstream
license and adaptations are recorded in [CONTRIBUTING](../CONTRIBUTING.md).

## Full-checkpoint storage accounting

The selected codec's CPU audit re-read all 36,288 source scale tensors
(19,025,362,944 bytes), checked their earlier per-tensor hashes and verified
positive normal E4M3 source values. It completed in 23.77 seconds. Each TP rank
contains 36,288 projection parts after splitting. The existing exact power-of-two
Marlin transformation preserves the byte span for these eligible source values;
the GPU fixtures separately verify eligibility and reconstruction on actual
post-Marlin bytes. This is a storage count, not a full-model GPU allocation test.

| TP rank | Source bytes | Retained bytes, including metadata | Raw fallback parts | Saving, GiB |
|---|---:|---:|---:|---:|
| 0 | 9,512,681,472 | 6,022,928,448 | 765 / 36,288 | 3.250086 |
| 1 | 9,512,681,472 | 6,020,078,560 | 736 / 36,288 | 3.252740 |

The total is **6.502826 GiB across TP2**. Counts include palettes, 32-byte part
pointer tables and 24-byte family descriptors. CUDA allocator rounding, loader
peaks, release of original scale buffers and whole-model residency remain to be
measured during integration. These savings belong to the original NVFP4 path
and must not be added to the earlier CT/MXFP4 checkpoint's savings.

## Timing protocol and interpretation

Each ordinary cell uses 15 trials and 60 CUDA graph replays per arm/trial.
Raw/candidate/repeat order alternates with repeat/candidate/raw. Every arm owns
its output/scratch buffers. Both GEMMs and the whole layer are timed separately.
The raw repeat must differ by no more than 2% in median time; a failed control
is retained and its timing is not accepted. Clock, temperature and power samples
are retained around trials; the focused repeat additionally samples each arm.
No clocks are changed. Timing runs are stopped if live serving activity changes.

The final installed-operator cross-check uses 32 trials and shared output/native
scratch buffers across raw/candidate/repeat; the installed arm shares outputs
and retains its operator-owned internal temporary. Independent snapshots and
per-arm checks before/after timing prevent shared buffers from hiding a mismatch.
Telemetry is sampled before warmup and after timing for each arm, with 20 warmup
graph replays after the telemetry call. The 2% control limit stays unchanged.

The first direct-palette consumer regressed larger rows in the 16-expert fixture.
Asynchronous loads reduced that overhead, and offset decoding reduced it further.
The full-layer run is necessary because the reduced fixture repeatedly reuses
a much smaller set of expert weights. Its larger-row regressions do not establish
the performance of the complete 288-expert layer.

## Retained failures and corrections

- Attempt 1 stopped during preparation: the byte-comparison helper did not flatten
  scalar global scales before viewing them as bytes. The helper was corrected.
- Attempts 2/3 found the same two tiny down-projection differences in raw and
  compressed controls; candidate versus raw was exact and final outputs matched.
  Maximum error was `1.4260876923799515e-8`. Removing `--use_fast_math` restored
  bitwise installed-stock parity. This is consistent with flushing tiny values;
  individual compiler subflags were not separately ablated. The failed checks
  remain retained and were not accepted under a looser tolerance.
- Attempts 4/5/6 complete the direct/asynchronous/offset reduced-fixture screens.
- Attempt 7 hit the experiment's private CUDA allocator cap while the installed
  scale-factor helper constructed a large boolean-index temporary. It stopped
  before timing; the serving containers stayed healthy. Attempt 8 computes the
  exact maximum of positive finite E4M3 source bytes, invokes the same installed
  factor oracle on that maximum, and stages one expert's scales at a time.
  This bounds preparation without changing the factor or scale arithmetic.
- The original rank-0, eight-row full-layer timing failed its repeat control
  (3.42% difference). Its correctness passed; its speed number remains unqualified
  and is retained separately from the focused repeat.
- Attempt 9 repeats that cell with 32 trials and passes: raw-repeat difference
  0.287%, candidate layer time 2.612% lower. Attempt 10 adds the installed stock
  operator, which is within 0.4% of the rebuilt raw control, but both repeat
  controls miss the gate (2.07% / 2.25%). Those speed results are not qualified.
  Attempt 11 removes output/native-scratch placement differences across arms
  and repeats 32 trials; both control gaps pass (1.193% / 0.464%). This changes
  the measurement setup without relaxing the numerical or timing gates.

All GPU experiments run in separate containers beside idle serving, with explicit
memory/CPU limits and deadlines. The existing CT candidate remains the live
8,192-context service; original production remains stopped. This document does
not authorize or claim a serving replacement.

<!-- NVFP4_EVIDENCE_BEGIN -->
## Evidence and final state

The [retained archive](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/nvfp4-component-v1-evidence-v2.tar.gz)
contains **217 payloads**, 1,721,720 compressed bytes; SHA-256
`3dbe4c720619a8645b3f701020af2070772bc983897ae0893058fd5b7442520f`.
It preserves every attempt, failed controls, frozen harnesses, staged licensed
sources/build flags, telemetry, full CPU range records and service checks.

The [independent local review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/nvfp4-component-v1-mac-review.json)
verifies all 217 payload hashes, recomputes timings and stability gates, checks
315 retained output-hash comparisons and reproduces the storage counts.
Thirteen traces each contain eight graph launches and 32 GPU kernels: four
raw and four candidate layer replays, four kernels per replay, with no separate
scale reconstruction. The review does not rerun GPU arithmetic on the Mac.
See the [retention record](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/nvfp4-component-v1-retention.json)
for artifact and repository-source hashes.

Local validation: `ruff check .` and strict `mypy mxwave` pass; the complete
suite passes **832 tests, with one skipped**, including 18 codec tests. Tests
use `python -m pytest` from this worktree; the initial console-entry invocation
resolved the editable root checkout and failed collection, retained in the review.

All **17 experimental containers are terminal**. The final check confirms both
original live candidate container identities unchanged and HTTP health 200.
The CT/DFlash candidate still serves port 8000 with **8,192 context**; original
production containers remain stopped. No service restart, clock change or KV
tuning occurred. Controllers and the owned awake lease have finished.

The next useful step is a ModelOpt serving adapter that preserves production's
NVFP4 weights, MXFP8 KDA and specialized target/draft head paths, followed by
matched full-model quality and end-to-end latency/throughput checks. Component
exactness and storage feasibility are established for the tested scope; that
deployment comparison remains outstanding.
<!-- NVFP4_EVIDENCE_END -->
