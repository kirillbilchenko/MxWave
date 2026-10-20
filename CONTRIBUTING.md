# Contributing to MxWave

## Provenance note

MxWave is an independent clean-room implementation. Its numerical behavior is
derived from the public OCP MX Formats specification and published quantization
research, including AWQ, GPTQ, and QuaRot. Its serialized output follows the
public `compressed-tensors` contract used by vLLM.

The qstream project was consulted only as an ecosystem reference and comparison
target. No source code or prose was copied or adapted from qstream. Historical
experiments that were rejected and removed from the production package remain
documented under `docs/` so their results are not misrepresented as features.

Do not copy implementation code from qstream or any other repository without a
written license grant. Any permitted adaptation must be identified here with
its source, license, files, and nature of the changes. Independent
reimplementations should cite the public specification or paper they follow.

Primary references:

- [OCP Microscaling Formats specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [AWQ](https://arxiv.org/abs/2306.00978)
- [GPTQ](https://arxiv.org/abs/2210.17323)
- [QuaRot](https://arxiv.org/abs/2404.00456)

GGUF compatibility reference:

- `ggml-org/llama.cpp`, MIT license, commit
  `6f767fe960c3b97cf37fac4626c86400561ca1e4` (`b11232`), specifically the
  public `block_mxfp4` serialization contract in `conversion/base.py` and the
  Qwen3.5 grouped-to-tiled linear-attention V-head ordering in
  `conversion/qwen.py`. `mxwave/gguf.py` independently implements those two
  compatibility transforms behind MxWave's typed validation API; it does not
  incorporate qstream code.

Kolibri GGUF compatibility and external runtime:

- `mxwave/kolibri_gguf.py` independently implements the public Kolibri tensor
  mapping and block-FP8 dequantization, reusing MxWave's lossless MXFP4 repacker.
  GGUF metadata/tokenizer serialization delegates to the MIT-licensed
  `ggml-org/llama.cpp` Python converter at
  `edd6e2bbdad5930899a93db8fa73c3b61c7b9bcc`.
- The experimental external Kolibri runtime patch is pinned to
  `Eliasfpv28/Kolibri-1-Q3_K_S-GGUF` revision
  `04f6e403e0db21f5ecb9d1c142f69204a43fe1e7`,
  `runtime-source/kolibri1-runtime.patch`, SHA256
  `2e629b80a55880dd8b1e40bc3bda51c1b9b7b38985439b47c3753a30eecd7cb7`.
  Its written grant licenses new additions under Apache-2.0 and llama.cpp-derived
  portions under MIT; `THIRD_PARTY_NOTICES.txt` and both supplied licenses must
  accompany a redistributed runtime. The patch is an external build dependency,
  not bundled MxWave code. Architecture reference: Apache-2.0 Aleph Alpha
  inference commit `049a6a7bd2405b27d6d280d256bd3d585191c7ae`.

Grouped Marlin MoE scale experiment:

- `integrations/glm53_mxfp4/build_moe.py` stages and adapts the Apache-2.0
  vLLM grouped Marlin kernel and host dispatch at revision
  `db9527a46873454610df6dbedf79a36d6bf1a7f6`. Exact source hashes and the
  written upstream license are bound by `moe_sources.json`; original source
  notices and `LICENSE` remain in the external build directory. No upstream
  source tree is vendored into MxWave.
- The adaptation changes expert scale-pointer advancement and scale loads to
  MxWave's exact fixed records, while retaining the grouped GEMM, routing
  metadata and reduction arithmetic. Experimental variants also expand records
  once per shared-memory stage before the existing barrier, or use byte
  permutation for register decoding; neither adds a decode launch.
  The later nibble layout stores exact exponent offsets under a validated
  tensor-wide base and reconstructs the stock BF16 scale fragments directly.
  Its aligned-base variant uses disjoint bit fields instead of addition;
  unsupported exponent ranges are rejected rather than clipped. These layouts
  change storage and scale decoding, not the upstream quantized weight values.
  The later palette prototype selects three frequent U8 scale values per
  tensor, with two-bit records and direct exact-vector exceptions. Its Marlin
  consumer also suppresses an unused terminal register prefetch before
  interpreting a sparse exception. `integrations/scale_tiles.py` and
  `moe_sparse_scales.cuh` are independently authored; no rANS/DietGPU
  implementation is incorporated. The external source license and hashes
  remain bound for these variants as well.
  Follow-ups snapshot read-only palette metadata and offer an explicit
  consecutive-palette consumer; preparation validates that policy before
  launch. The full-model diagnostic hashes every replaced expert-scale
  group before packing and checks complete GPU reconstruction before and
  after inference. These harnesses are independently authored and do not
  change the source checkpoint or production serving configuration.
  The shared-stage palette experiment expands exact U8 scales into the
  original 16-byte shared-memory slots after the issuing thread's async-copy
  wait and before the existing barrier. Consumers use the stock scale
  conversion path; the source matrix/reduction arithmetic is retained.
  This pipeline adaptation and the single-pause component controller are
  independently authored. They use the same pinned upstream license/source
  provenance and remain subject to GPU exactness and latency gates.
  The opt-in nibble-stage follow-up validates the complete source range,
  including exceptions, before expanding records into the existing aligned
  four-bit shared layout. Upper descriptor bits carry the verified base and
  policy; unsupported ranges retain exact U8 staging. Its policy, packer
  helpers and tests are independently authored and preserve FP4 weight codes
  and upstream matrix/reduction arithmetic.
  A controlled static variant selects the U8 consumer from the verified CPU
  metadata once during binding. Its nibble GEMM omits the device layout branch;
  the existing exact U8 library owns unsupported ranges, including through
  stream execution and graph capture. Decoder qualification retains all ranges.
  A small C ABI replaces the Torch host
  wrapper for an isolated component comparison. MxWave's decoder and bridge
  retain Apache-2.0 notices. This is an experiment, not a supported engine fork.

BF16 attention component:

- `integrations/glm53_mxfp4/build_bf16_fp32_split.py` stages a separate
  FP32 partial-sum diagnostic under the same written Apache-2.0 ZipServ grant
  and revision recorded below. Hash-bound external sources and their notices
  remain outside this repository; the original license is retained in both
  build arms. The adaptation preserves the MMA loop, writes split accumulators
  as FP32, and uses an independently authored FP32 reducer with one final BF16
  conversion. A distinct C symbol validates an explicit workspace byte count.
  This is an isolated precision experiment, not a serving-runtime change.
- `integrations/glm53_mxfp4/build_bf16_n8.py` stages a separate experimental
  adaptation of the same written Apache-2.0 ZipServ grant/revision below.
  It binds every external source file by SHA-256 and preserves upstream
  notices and `LICENSE` in both build directories; no external source or
  binary is vendored. The N8 variant uses eight activation rows for its
  shared buffers and an `ldmatrix.x2` B-fragment load, retaining the N16+
  path, compressed representation, decoded weight values and MMA order.
  The matched control uses identical compiler flags. An independently
  authored resource-query ABI reports CUDA occupancy limits for both arms.
  Separate mechanism variants isolate storage and load width; a specialized
  input-row permutation schedules exception-heavy tiles first while retaining
  each tile's complete accumulation. All staged external variants preserve
  the same upstream grant and notices; none is a default runtime change.
- `integrations/glm53_mxfp4/screen_mxfp4_entropy.py` is an independently
  authored CPU rANS feasibility screen using the public integer-state
  equations. It copies no DietGPU or paper implementation. It measures
  checkpoint samples and stream overhead, not a production artifact format
  or GPU decoder. See the kernel diagnostic for the related paper and limits.
- `integrations/glm53_mxfp4/prepare_bf16_attention.py` and
  `diagnose_bf16_attention.py` call the existing external ZipServ C ABI bridge
  from the lossless-storage experiment. That build uses `xxyux/ZipServ` revision
  `f69cea959005222bed64fc847bc3fcdcca32a904` under its written Apache-2.0 grant.
  The external CUDA source, original notices and license remain in the Spark
  build directory; no ZipServ source or native binary is vendored here.
  MxWave's Python bridge and benchmark helpers are reused by hash. The component
  preserves the existing SM121 build and launch choices; it adds source-bound
  GLM fixtures, exact GPU decoding checks and paired timing.
- `bf16_prefetch.py` independently implements a bounded two-slot CUDA-stream
  scheduler around that same native decode ABI and stock PyTorch BF16 GEMM.
  Its readiness/consumption events, graph capture harness, optional cached first
  matrix and trace-interval analysis are MxWave code. It does not change or
  incorporate the external ZipServ kernel implementation.
- `prefill_decode.cuh` is an independently authored integer-only reconstruction
  experiment for the same licensed triple-bitmap layout. It calls hash-bound
  external copy helpers and an unchanged-decoder control under the Apache-2.0
  ZipServ grant above. `build_prefill_decode.py` stages the external source and
  original LICENSE outside the repository, then appends the authored header.
  Direct output stores and row-coalesced traversal preserve the packed format;
  no external source or native binary is vendored here. Its Python bridge,
  exact-byte smoke, timing harness, restoration controller and independent
  event verifier are MxWave code. No serving default changes in this component.
- `mtp_mapping.py` independently translates this checkpoint's explicit CT
  selector namespace and copies fused aliases for the pinned native vLLM MTP
  class. `audit_mtp_dispatch.py` exercises the installed Apache-2.0 vLLM
  selector on CPU before/after this metadata-only repair, checking that base
  policy is unchanged. No upstream source is copied or vendored, and no
  checkpoint value, native kernel or production configuration is changed.
- `kda_policy.py`, `kda_adapter.py` and `kda_runtime.py` add an opt-in,
  source-bound GLM KDA adapter around that qualified external ABI. They delegate
  parameter creation and TP source slicing to the installed Apache-2.0 vLLM
  linear classes, and use an independently authored ownership/stream bridge.
  No vLLM linear-loader or ZipServ kernel implementation is copied into these
  files. Native binaries remain external dependencies, bound by SHA-256 with
  their license retained beside them. Single-token fused arithmetic is measured
  separately from exact reconstruction of the stored BF16 weight bytes.
- `vocab_policy.py`, `vocab_adapter.py` and `vocab_runtime.py` reuse MxWave's
  row-addressable BF16 codec from the lossless-storage worktree and the same
  externally licensed, hash-bound ZipServ head ABI. The row codec and TP2
  adapter are MxWave code; no external implementation is copied. Installed
  vLLM methods still own vocabulary partition loading, masking, reduction and
  logits gathering. The pinned development loader needs a selective loading
  context wrapper for these two CPU-staged vocabulary parameters. The combined
  worker exposes lazy reconstructed views to the original source checker, so
  compressing a parameter does not silently drop its integrity checks.
  The pinned GLM text-embedding constructor omits `quant_config`; an exact-prefix
  constructor bridge supplies the existing engine config at that one call.
  Component qualification exercises this omitted-config call as well as the
  explicitly configured head.

- The best-candidate quality helpers (`quality_suite.py`, `quality_worker.py`,
  `measure_quality_model.py`, `analyze_quality_model.py`, `run_quality_model.py`)
  are independently authored. They use pinned public WikiText-2/GSM8K test
  data and existing source-check/codec interfaces; no external evaluation
  implementation is copied. Dataset provenance, revisions, sampling policy
  and private artifact hashes are recorded in
  `docs/GLM53_BEST_CANDIDATE_QUALITY.md`. Raw corpus data, generated responses
  and native dependencies are retained on Spark outside the repository.

- The reference-repeat diagnostic helpers (repeat_worker.py, measure_repeat_model.py,
  analyze_repeat_model.py and run_repeat_model.py) are independently authored.
  They use public PyTorch hooks/dispatch APIs and invoke the installed vLLM
  operations with captured runtime inputs; no upstream kernel implementation
  is copied. Private activation fixtures retain source/code bindings and hashes.

- The stable-dispatch experiment, controlled projection probe and saved-trace
  analysis are independently authored. They use Triton's public sorting
  primitive and the installed vLLM repacking/dispatch APIs; no producer or
  kernel implementation was copied. Original runtime and source provenance
  remain pinned in the experiment docs. Trace analysis counts interval unions
  across contained device streams and retains its interpretation limits.

- The practical-repeat client/analyzer reuse the original authored task scorer
  and isolated code sandbox. They retain five fixed rounds, compare both
  directions of within-arm task changes and preserve the preceding strict
  failure; no external evaluation implementation or dataset is vendored.

- The single-dispatch KDA prototype independently wraps the existing pinned
  BF16 bridge C ABI. It changes host dispatcher/schema handling only, retains
  the qualified prefill fallback and does not copy or modify a native kernel.
  Its component gate requires current-path output bits, streams, graphs,
  compilation and measured latency before any integration.
  The paired-graph follow-up uses independently authored warmed-event timing,
  identical controls, saved PyTorch GPU activities and the existing restoration
  watchdog. It copies no profiler, timing or native kernel implementation.
- The eager-only KDA policy, idle worker selector and single-load model harness
  are independently authored around that same hash-pinned ABI. Public PyTorch
  capture/compiler checks keep the original operation for those modes. Raw CUDA
  topology inspection uses public CUDA Python APIs; no upstream implementation
  is copied. Both model modes retain the same weights, pools and graph cache.

## Setup

```bash
pip install -e ".[dev]"
```

## Quality gates

```bash
ruff check .            # lint
mypy mxwave             # type check
pytest                  # unit tests
```

All must pass before a PR.

## Adding a quantization feature

1. Implement the primitive in the appropriate module (`rotate.py`,
   `verify.py`, `core.py`, etc.).
2. Add a focused unit test in `tests/`.
3. Update the README roadmap and the benchmark table if it changes quality.

## Adding a runtime adapter

1. Implement `matches(config)` and `build(config, tensor_names)` under
   `mxwave/adapters/`; discovery must use configuration plus checkpoint headers,
   never tensor payloads.
2. Register the adapter in `mxwave/runtime_adapters.py`.
3. Describe runtime-fused linear groups in their packing order and reject missing
   members so precision-aware features cannot silently split a fused operation.
4. Add synthetic conformance tests for matching, graph construction, fused-group
   membership, incomplete checkpoints, and auxiliary decoder stacks when relevant.
5. Record an end-to-end model smoke test before describing the adapter as verified.

## Benchmarking on the DGX Spark

Benchmarks belong in the README table (same model, same `mxfp4-pack-quantized`
format). Record: model, hardware (SM100/SM121), PPL, task accuracies, SQNR,
throughput. Prefer deterministic metrics (e.g. `eval_ppl.py`-style perplexity)
over noisy small-sample accuracies.

For API likelihood comparisons, use `scripts/evaluate_api_perplexity.py` and
pin the corpus revision plus file hash. Compare identical window hashes, token
hashes, token counts, serving backend, and concurrency. Report paired
uncertainty over windows and describe the window/context-reset protocol; do not
present API prompt perplexity as directly comparable to a literature PPL that
uses different token windows or stride.

## License

The GLM graph-boundary diagnostic is independently authored. It wraps existing
layer methods only during initial capture, records graph-written operand/result
copies, and calls stock operations without adapting their arithmetic. Its
semaphore control, bounded byte snapshots, raw-value decoders and analysis are
MxWave code; inspected runtime sources remain private evidence outside the repo.
The isolated MLA follow-up calls the hash-bound installed vLLM/FlashInfer APIs
with independently generated queries/cache data and explicit private-workspace
snapshots. It adapts no upstream implementation or kernel arithmetic.

The controlled-cohort GLM diagnostic is independently authored. It calls the
pinned vLLM server's existing pause/keep/resume and collective-RPC interfaces,
observes existing dispatch counters and changes graph selection between idle
request groups. It adds no adapted upstream kernel or quantization code.

The GLM native MTP loader bridge and geometry checks are independently authored.
They inspect the installed Apache-2.0 vLLM loader API, rebuild a separate CT
configuration from checkpoint metadata, and call the unchanged native loader.
No external loader implementation is copied, vendored or edited. The admission
audit analyzes existing profiler IDs and counters without changing native code.

The bounded admission adapter and CUDA graph policy fixture are independently
authored. They call the pinned runtime's input-drain/message-handler and graph
candidate-builder/dispatch APIs; native implementation snapshots remain private
evidence. The graph helper rebuilds descriptors on private configuration copies
and leaves native graph replay and model arithmetic unchanged. No upstream
implementation is copied or vendored.

The GLM sparse-index multiset analysis, power-of-two padding adapter, native
attention sentinel fixture and exact16 input routing wrapper are independently
authored. They call pinned installed runtime APIs without copying or changing
upstream kernel implementations. The input wrapper refuses an unpinned parent
runtime and keeps the existing <=8 route and unmeasured intermediate rows.
Observed native implementation snapshots remain private evidence.

The C8 forward observer and profile attribution tools are independently
authored. They wrap existing runner/prepare calls to record CPU request metadata
and correlate profiler launch IDs. They copy no GPU tensors and adapt no native
implementation; controlled admission uses existing disposable-service APIs.

Apache-2.0. By contributing you agree to license your contribution under the
same terms.

Consistent native mHC dispatch experiment:

- integrations/glm53_mxfp4/consistent_mhc.py and its worker are independently
  authored wrappers around installed native vLLM APIs. They source-pin native
  dispatch and kernel modules, bound the tested geometry, and compose the
  existing native post/pre operations with a fixed native prenorm split count.
  No native kernel implementation is copied into the repository. The arithmetic
  rounding policy differs from native fused C1 and requires the same policy in
  both comparison arms; full-model qualification is pending.

The isolated SM121 native expert screen uses unchanged FlashInfer source from
commit `f64b4ba8762ded12a4f4b9a2092974c9a395e7ca`, under its retained Apache-2.0
license. Its 58-file external source inventory includes the license, SM12x
MXFP8/MXFP4 routing and FC1/FC2 modules, and their dependencies. Source files and
original notices remain in the private experiment archive, outside this repo;
no native kernel is adapted or vendored. The independently authored harness
calls those APIs, arranges exact checkpoint rows/scales into their required
layout, and measures complete routed layers against the installed Marlin path.
The source inventories, commit and archive hashes are recorded in
`docs/GLM53_MOE_BACKENDS.md` and the experiment retention records. The new
`integrations/glm53_mxfp4/moe_capture.py` observer is independently authored and
captures actual inputs and outputs through installed vLLM APIs after timing.

NVFP4 scale component prototype (19 October 2026):

- `integrations/glm53_nvfp4/build.py` stages the same hash-pinned Apache-2.0
  vLLM grouped Marlin sources and written license identified by
  `integrations/glm53_mxfp4/moe_sources.json`. It selects the existing NVFP4
  group-16/S0E5M3 specialization and retains its GEMM/global-scale/reduction
  arithmetic. The candidate changes scale fetching to an independently authored
  five-bit palette decoder with exact raw fallback. The raw control retains
  upstream scale loads. Notices and LICENSE stay in the external build tree;
  no upstream source tree is vendored. `codec.py`, `scale_decode.cuh`, the narrow
  C ABI bridge and component harness are independently authored experimental
  plumbing. This is not a production ModelOpt adapter.
  Follow-ups pipeline the two bit planes asynchronously through the existing
  stage barriers, and replace eligible palettes with exact five-bit byte offsets.
  Unsupported byte spans use raw fallback; the stock conversion and accumulation
  remain unchanged. The standard-math build and retained stock-output checks
  qualify the covered component shapes, not a full inference deployment.

The NVFP4 serving follow-up also adjusts the pinned Marlin host launcher's dynamic
shared-memory budget to account for the decoder's static palette, retaining the
original tile and reduction policy. The adapter calls the installed production
Marlin tuning API without copying that overlay's implementation. The new loader,
worker, bootstrap and ownership checks are independently authored; the earlier
component build remains retained separately from the fixed library.

The drafter head and target-context probes are independently authored observers
around installed vLLM/DFlash2 APIs. They pin source versions, preserve native graph
outputs and retain tensors for an independently authored CPU analysis. No vLLM
kernel or external training implementation is copied or adapted into the repo.
