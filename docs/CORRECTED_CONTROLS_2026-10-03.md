# Corrected quantization controls on DGX Spark

## Execution identity

This is a new experiment with `mxfp4-rne-v2`, on branch
`chore/research-memory-cleanup` plus uncommitted fixes. Existing published
weights and historical benchmark JSON are preserved. These observations do not
promote or publish a replacement model.

- Hardware: NVIDIA GB10 / SM121, DGX Spark.
- Runtime: installed official vLLM 0.29.0 image, local image ID
  `sha256:785fd756b4ae1cacda7612cb859dbf227330e8ced9a6fdea42e9379128bebce0`.
- Torch: `2.13.0+cu130`; CUDA 13.0.
- BF16 source: `Qwen/Qwen3.8-27B` at
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Spark artifact root:
  `/home/kirya/local-spark/experiments/mxwave-corrected-controls-2026-10-03`.
- Retained source snapshots: `runtime` for chunk timing and `runtime-quality-v1`
for the quality experiment. Benchmark JSON and the frozen protocol contain
content digests of the executed files; HEAD does not identify uncommitted code.
- The fixed six-model comparison is complete. A later isolated
  `runtime-emission-v1` snapshot measures production output writes as well.
  The experiment started on October 3 and documentation was finalized on
  October 4, 2026. Published checkpoints remain unchanged.

## Chunk budget measurements

Two complete sweeps measured real layer-0 `down_proj` and `gate_proj` matrices
using captured H64 block Hessians, clip depths 4 and 8, one warmup and three
timed repetitions per setting. The second sweep reversed projection order,
clip-depth order, and budget order. Packed and scale SHA-256 values agree across
all budgets and both sweeps.

Median seconds, with the range across the two sweep orders:

| Matrix / clip depth | 1,048,576 elements | 4,194,304 | 8,388,608 | 33,554,432 |
|---|---:|---:|---:|---:|
| down_proj / 4 | 0.457–0.473 | 0.660–0.661 | 0.708–0.711 | 0.717–0.718 |
| down_proj / 8 | 0.672–0.676 | 0.964–0.972 | 1.053–1.056 | 1.072–1.073 |
| gate_proj / 4 | 0.451–0.456 | 0.659–0.662 | 0.691–0.699 | 0.699–0.700 |
| gate_proj / 8 | 0.651–0.677 | 0.978–0.980 | 1.024–1.035 | 1.025–1.038 |

The smallest budget uses 60 rows for width 17408 and 204 rows for width 5120;
the largest permits the original 1024-row limit. At clip depth 8, down_proj
peak CUDA allocation was about 81 MiB at the smallest budget and 832 MiB at
1024 rows. Increasing the budget did not improve these matrix timings.
The default stays at 1,048,576, shared by both engines and CLI parsers.

Scope: slice reads, host-to-device transfers, finite checks and quantization.
Output copies to CPU and disk writes are outside the timed region. These are
matrix measurements, not inference throughput or complete checkpoint timings.
Different widths, objectives and GPUs can have different optima.

Evidence:
[first sweep](../benchmarks/qwen3.8-27b-corrected-chunk-performance-2026-10-03.json),
[reverse sweep](../benchmarks/qwen3.8-27b-corrected-chunk-performance-reversed-2026-10-03.json).

### Production output writes included

Two further sweeps used the production row reader and incremental safetensors
writer. Timings include GPU-to-CPU output copies, disk writes, flush and fsync,
with the same warmup, repetitions and reversed-order check. Median seconds,
with the range across orders:

| Matrix / clip depth | 1,048,576 elements | 4,194,304 | 8,388,608 | 33,554,432 |
|---|---:|---:|---:|---:|
| down_proj / 4 | 0.744–0.779 | 0.915–0.942 | 1.022–1.054 | 1.197–1.210 |
| down_proj / 8 | 0.990–0.996 | 1.261–1.269 | 1.388–1.404 | 1.582–1.691 |
| gate_proj / 4 | 0.749–0.769 | 0.892–0.913 | 0.962–0.980 | 0.966–0.984 |
| gate_proj / 8 | 0.966–0.982 | 1.228–1.243 | 1.314–1.330 | 1.329–1.336 |

The default remained fastest in every tested case with output writes included.
All 16 packed/scales hashes match both orders and the earlier benchmark mode.
The executed source digest is
`39b6b305bd81a4ce6be31521ff54e13d1e1bc83894ce919dfe3e9d6f200b0a27`.
These measurements exclude source/shard hashes, rename journals, passthrough
tensors and SQNR checks. They do not compare complete checkpoint times across
budgets or establish the best budget for every GPU and matrix shape.

Evidence:
[forward emission sweep](../benchmarks/qwen3.8-27b-corrected-controls-2026-10-03/chunk-emission-forward.json),
[reverse emission sweep](../benchmarks/qwen3.8-27b-corrected-controls-2026-10-03/chunk-emission-reverse.json).

## Frozen quality protocol

The six fixed controls are corrected RTN, unweighted MSE, corrected norm
fallback MSE, H64, BF16, and the existing AMD AWQ MXFP4 reference. Corrected
conversions use the same 400-module policy; the three MSE variants use percentile
99.5 and clip depth 4. H64 retains the original 64x512 Pile-validation artifact,
including absolute damping and no sink-token exclusion.

The pinned WikiText-2 raw **validation** split is used:
`Salesforce/wikitext@b08601e04326c79dfdd32d625aee71d232d685c3`.
Before any candidate quality measurement, `validation/protocol.json` froze:

- 128 full PPL windows of 4096 characters;
- 64 different KL windows, with prefixes at 64/128/192/256/320/384/448/512
  tokens: 512 exact full-vocabulary next-token positions;
- remaining windows reserved and unscored;
- exact text/token identities and dataset Parquet/corpus hashes;
- an exclusion audit against the supplied historical WikiText test corpus,
  its windows and its 512-token prefixes;
- fixed stock vLLM/Marlin settings, no MTP or prefix caching, and graph capture
  sizes 1/2/4;
- window-level paired bootstrap uncertainty. All eight prefixes in a KL
  window travel together in a resample.

PPL and KL use disjoint windows. The exclusion audit does not certify unknown
earlier use or semantic duplicates. No tuning, precision-budget selection, or
model promotion occurs in this comparison.

All H64 results here belong to the **unpublished corrected checkpoint**, with
checkpoint identity
`fb093892699b24a64fdd99963d31c5f233cf54202c3fc4d257cba9ee86ce789d`.
The published Hub H64 checkpoint retains legacy rounding and its historical
test-split scores. It was not measured on these validation windows, so this
comparison does not isolate the effect of changing its rounding.

After this fixed comparison, norm-proxy weighting was made opt-in with
`--gamma-proxy`. The control converter explicitly sets `gamma_proxy` for every
case; the measured plain-MSE and norm controls retain their recorded settings.
Source digests below identify the actual measured snapshots, before that
default change and subsequent integration with newer `main` work.

The protocol SHA-256, computed from the **uncompressed original JSON bytes**, is
`f15823f74400a6a95059d54eb9d3f81456c5d29e01ba5ae219a20c80c6a84289`;
the executed numerical source SHA-256 is
`3d4343089e18f8beb2dc4d867ee0493e79a559d7e28829d55fc64f40dae48dfe`.
The retained calibration file SHA-256 is
`5f7f6380d2ffd9731c0e4463d89be808c23a2b7101f5d3544eb4124f51c160de`.
Its metadata records 32,768 tokens, absolute damping `1e-6`, no prefix-token
exclusion and Hessian coverage of all 400 targets. The norm fallback applies to
272 targets with an associated preceding norm; the other 128 use unweighted MSE.

### Protocol storage and verification

The frozen v1 protocol is archived as `validation/protocol.json.gz`. Gzip
compression preserves the original 3,819,226 bytes exactly; the archive is
328,577 bytes. It has not been reserialized or converted to a new schema.
Every existing measurement and comparison still refers to the same
uncompressed SHA-256 above. Verify it from the repository root:

```bash
gzip -dc \
  benchmarks/qwen3.8-27b-corrected-controls-2026-10-03/validation/protocol.json.gz \
  | shasum -a 256
```

`collect` and `compare` accept either plain JSON or gzip and always compute
`protocol_sha256` from the decompressed JSON bytes, rather than the gzip
container. Replaying historical collection also requires the recorded source
snapshot; compression does not bypass the existing source-identity check.

Future `prepare` runs write a v2 `protocol.json` manifest with configuration,
character start/end offsets, token ranges, prefix lengths and per-window/prefix
hashes. Token IDs live in `protocol-tokens.safetensors`, referenced by its own
SHA-256 in the manifest. Each PPL or KL window is stored once; KL contexts
reference a window and prefix length. The evaluator verifies the sidecar,
window and prefix hashes before reconstructing token inputs. The manifest hash
therefore binds the token file as well. Existing frozen v1 protocols and their
results retain their original format and identities.

## Completed quality comparison

Every control scored the same 121,284 PPL tokens. KL uses 512 full-vocabulary
positions from 64 other windows. Positive PPL change means worse than BF16.

| Control | Prompt PPL ↓ | Change vs BF16 | Mean forward KL ↓ | Top-1 agreement / 512 |
|---|---:|---:|---:|---:|
| BF16 | 7.831635 | reference | reference | 512 |
| Corrected RTN | 8.059076 | +2.904% | 0.059035 | 455 |
| Unweighted MSE | 8.034145 | +2.586% | 0.062363 | 461 |
| Corrected norm fallback MSE | 8.040323 | +2.665% | 0.074429 | 456 |
| H64 scale-only | **7.954455** | +1.568% | **0.054833** | 461 |
| AMD Quark-AWQ MXFP4 | 7.993200 | +2.063% | 0.060078 | 465 |

Paired PPL differences, using 20,000 window bootstrap resamples:

| Comparison | Relative PPL change | Nominal paired 95% interval |
|---|---:|---:|
| H64 vs AMD | −0.485% | [−0.823%, −0.151%] |
| H64 vs BF16 | +1.568% | [+1.044%, +2.058%] |
| H64 vs RTN | −1.298% | [−1.540%, −1.079%] |
| H64 vs unweighted MSE | −0.992% | [−1.155%, −0.830%] |
| Unweighted MSE vs RTN | −0.309% | [−0.512%, −0.124%] |
| Corrected norm fallback vs unweighted MSE | +0.077% | [−0.015%, +0.167%] |

The fresh controls support a modest benefit from calibrated scale selection.
Unweighted MSE gives a smaller PPL improvement over RTN and a worse KL.
Correcting the norm bug did not make the proxy a useful default recommendation
for this model: its PPL did not improve MSE, and its mean KL increased by
0.012066, with interval [0.000686, 0.032866]. These negative results are retained.

H64's mean KL is 8.73% below AMD. The paired absolute difference is −0.005245,
with interval [−0.010634, −0.000144], resampling the 64 windows with all eight
prefixes together. AMD has higher top-1 agreement, 465 vs 461. These intervals
are nominal and are not adjusted for multiple comparisons. The PPL protocol
uses independent character windows and context resets; its absolute values are
not literature-compatible sliding-window WikiText PPL. These measurements do
not establish broad task superiority or replace evaluation on reserved data.

## Decode throughput

Each setting uses a 512-token input and 128 forced output tokens, one warmup and
three timed repetitions. Rates are medians, include prefill, and count aggregate
output at concurrency 4. Stock vLLM uses Marlin for MXFP4, BF16 activations and
KV cache, no MTP and no prefix cache.

| Control | Output tokens/s, concurrency 1 | Output tokens/s, concurrency 4 |
|---|---:|---:|
| BF16 | 4.65 | 17.55 |
| Corrected RTN | 12.98 | 44.18 |
| Unweighted MSE | 12.96 | 44.03 |
| Corrected norm fallback MSE | 12.81 | 43.38 |
| H64 scale-only | 12.77 | 43.49 |
| AMD Quark-AWQ MXFP4 | 12.53 | 42.32 |

Small differences among MXFP4 variants are observations from this short,
sequential run. No significance claim is made for those throughput differences.
Source/model load and compilation are outside the timed requests.

## Conversion resources and reconstruction

Complete conversions include source hashes, shard hashes, fsync, and sampled
SQNR verification. All four conversions emitted 18 verified shards.

| Control | Wall seconds | Peak CUDA allocation, MiB | Peak process RSS, GiB | Mean unweighted sampled SQNR, dB |
|---|---:|---:|---:|---:|
| Corrected RTN | 1353.20* | 39.3 | 1.05 | 18.911 |
| Unweighted MSE | 236.48 | 45.7 | 1.29 | 18.958 |
| Corrected norm fallback MSE | 254.14 | 45.8 | 1.30 | 18.956 |
| H64 scale-only | 255.46 | 79.8 | 1.44 | 18.821 |

SQNR samples the first 16 rows of each of the 400 targets, averaging tensor
scores rather than all tensor elements. H64's calibration-weighted sampled
SQNR is 19.141 dB; it is a different objective from the unweighted column.
The lower ordinary SQNR alongside better PPL is consistent with allocating
scale choices according to calibrated input sensitivity. This is not an
ablation of full GPTQ, AWQ channel scaling or rotations.

*RTN timing includes a prolonged GPU-to-CPU copy delay and is not a clean
throughput baseline.* Its completed shard files were all rehashed before reuse.

The corrected RTN conversion completed all 18 shards with full coverage and
resource measurements. Its peak allocated CUDA memory was 39.3 MiB and peak
process RSS was 1.05 GiB. Wall time was 1353.20 seconds (22.6 minutes), including
a prolonged GPU-to-CPU copy delay. It is not a clean throughput baseline.

Stack samples placed that delay in the shard writer's GPU-to-CPU copy. The
24-GiB container allowed swapping, and process pages were observed swapped
out. That observation is not proof of the delay's cause. The job completed
successfully just before the diagnostic stop reached it. Its completed files
were subsequently verified against every recorded shard hash and retained.

The other conversions used a 96-GiB cgroup limit with swapping disabled;
evaluations used 108 GiB with swapping disabled. Each job had a 30-minute timeout
and a 16-GiB host-available-memory stop gate. The serial runner pauses the idle
Ollama model and restores it after success or an exception. Other serving
profiles and all model payloads are preserved.

Both the full quality runner and the later emission sweeps completed and
restored the Ollama model container. Local integrity checks recomputed all
report hashes, conversion-manifest hashes, PPL from total NLL/token counts, and
cross-budget/order packed/scales equality.

Evidence:
[comparison and paired intervals](../benchmarks/qwen3.8-27b-corrected-controls-2026-10-03/quality-comparison.json),
[frozen protocol (gzip)](../benchmarks/qwen3.8-27b-corrected-controls-2026-10-03/validation/protocol.json.gz),
[individual measurements](../benchmarks/qwen3.8-27b-corrected-controls-2026-10-03/measurements),
[conversion manifests and source identities](../benchmarks/qwen3.8-27b-corrected-controls-2026-10-03/checkpoints).
Full-vocabulary log-probability safetensors remain on Spark, with their SHA-256
values recorded in the individual reports. GPTQ, AWQ integration and MR-GPTQ
remain separate method-development work; no new quantization method or model
promotion is claimed.

## Local verification and branch state

The measured source snapshot passed Ruff, strict mypy on all 29 `mxwave` source
files, 225 tests and `git diff --check`. Changes are packaged on
`fix/quantization-correctness-and-controls` in separate `fix:` code and `eval:`
control-record commits. After integration with current `main`, Ruff, strict
mypy on 31 source files, all 251 tests and `git diff --check` passed again,
including legacy/gzip/compact protocol comparisons and hash-chain checks.
