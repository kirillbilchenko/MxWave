# Original-production NVFP4 reference-repeat diagnostic

**Complete and independently verified, 19 October 2026, 20:47 UTC.** The exact prior
CT/DFlash service is restored healthy; all diagnostic containers are stopped. This run locates an observed first
numerical divergence inside the uncompressed production setup's routed-expert
operation. It does not yet identify the internal routing/GEMM/reduction kernel
or establish a fix. The compressed adapter is not enabled in this diagnostic.

## What the captures establish

The original ModelOpt checkpoint, production optimization flags, DFlash, prefix
caching, 6 GiB KV per rank, 32 slots and 6,919-token budget are retained. The
selected forward is an eager 256-token C1 prefill; decode graphs stay enabled.
Twenty API groups include one/four-output-token ordinary repeats, coarse and
focused GPU captures, repeats after a 256/64 C16 workload, and ordinary repeats
after all instrumentation. All captured first-token distributions cover the
full 154,880-token vocabulary.

The focused pair has byte-identical embedding/model inputs, outputs of layers
0–2, layer 3 attention/mHC outputs, MoE hidden-state inputs, actual gate logits and shared-
expert output. The **first differing recorded tensor is
`layer-3/mlp/experts/output`**, on both TP ranks. On each rank it changes **4 of
1,048,576 BF16 elements**, maximum absolute difference **0.00390625**, RMS
**6.0316e-6**. The final raw logits change 153,335/154,880 values and have KL
**0.00537795**. Subsequent layers may independently introduce more variation;
the trace does not establish that the first four elements alone cause all of it.

The production router-dedup hook first passes a `[256,4096]` placeholder through
the external gate call. The capture also observes the actual internal gate call
and its `[256,288]` FP32 logits; those actual logits are byte-identical. Equal
router logits do not establish equal selected-expert order, sorted-token order,
GEMM outputs or reduction order. Those internal boundaries are the next target.

Raw logits agree with their corresponding API distributions: all twenty
rank-specific captures have normalized raw-to-API KL below **8e-16**, with maximum normalized
logprob difference below **1.9e-6** from reported FP32 logprobs. The measured
repeat divergence is present in model logits, not introduced solely by API
serialization or distribution reconstruction.

| Captured pair, rank 0 | First differing boundary | Differing elements there | Raw first-token KL |
|---|---|---:|---:|
| One-token coarse repeat | Layer 3 output | 1 / 1,048,576 | 0.00920439 |
| Layer 3 focused repeat | Layer 3 expert output | 4 / 1,048,576 | 0.00537795 |
| Four-token coarse repeat | Layer 3 output | 2 / 1,048,576 | 0.00200615 |
| Four-token repeat after C16 | Layer 3 output | 3 / 1,048,576 | 0.04791726 |
| Focused repeat after another C16 | Layer 4 output | 2 / 1,048,576 | 0.00344990 |

The last pair is exact throughout captured layer 3 and first differs at layer 4.
This prevents treating layer 3 as a uniquely faulty layer or assuming the first
drift must occur there on every repeat. Layer 4 internals were not captured.
The failure also occurs without a C16 predecessor, so the data do not establish
request-history contamination as the cause.

Ordinary one-token controls before capture have KL **0.00017304 / 0.00054363**
against the first repeat; the after-capture pair has **0.01080618**. Their layouts
match. Ordinary four-token controls also fail prefill repeatability; one final
pair additionally changes speculative forward layouts. The before-controls run
before snapshot-hook installation. Afterward, GPU cloning is disabled but
forwarding wrappers remain installed. Capture therefore does not create the
only observed failure. It remains an intrusive observer, and none of these
requests qualifies performance or supplies an end-to-end compression gain.

## Limits and next test

Keep the **1e-5** quality gate and all earlier failures. The existing exact scale
storage and isolated matched-routing consumer results remain valid, while
full-model zero-added-loss qualification remains unresolved. No production
kernel fix or compressed-versus-stock numerical rerun was performed here.

The next bounded experiment should use the retained actual layer-3 operands and
observe the selected expert IDs/weights, padded token ordering, both Marlin GEMM
outputs, shared-output combination and TP reduction. Replaying fixed routing
and operands can distinguish routing/order variation from the arithmetic
consumer. Choose a deterministic fix only after that separation, apply the
same policy to both arms, and rerun stock self-repeat before comparison.

## Attempts and evidence

`nvfp4-repeat-v1` retains a disk-space prelaunch stop and six completed ordinary
controls. Its first capture missed the installed runtime's precomputed embedding
path (`input_ids=None`) and collected zero tensors. The controller restored the
exact prior CT service. V2 fixes only this probe selection and adds a regression
test. Local ruff/strict mypy pass; **839 tests pass, one skipped**.

All **63 v1 payloads** verify on Mac; Spark/Mac recomputation agrees on 20 numeric
fields with maximum difference 1.56e-15. V1 archive SHA-256:
`2833f6d02e01836987395bb5d128d89cb673705ed432db114325f84f53d1f72e`.
It is retained on rank 1 under `full-model-v1/nvfp4-repeat-v1/repeat-evidence.tar.gz`
and locally at `/private/tmp/mxwave-nvfp4-repeat-v1/repeat-evidence.tar.gz`.

All **4,290 v2 payloads** independently verify on Mac. Spark/Mac recomputation
agrees on **5,374 numerical fields**, with maximum difference 3.5528e-15; all
categorical results and frozen source/client/plan bindings match.

Archive: **3,491,898,037 bytes**, SHA-256
`41660a22f39fc34655956c739d7ea76fc07bdd65b4db278cb2609a1a7887e73e`.
It is retained on rank 1 at
`full-model-v1/nvfp4-repeat-v2/repeat-evidence.tar.gz` and locally at
`/private/tmp/mxwave-nvfp4-repeat-v2/repeat-evidence.tar.gz`. Byte-identical tensor
payloads use archive hardlinks; every manifest path and payload hash verifies.

V2 raw snapshots are available in rank-1 `nvfp4-repeat-v2/retained-combined` and
local `/private/tmp/mxwave-nvfp4-repeat-v2/evidence`. Rank-0 temporary binaries
were removed after checking both retained destinations, recovering
5,445,162,848 bytes. Their manifests and relocation receipt remain in place.
Each coarse capture owns 483,567,104 bytes per rank; focused captures own
635,940,056 bytes, below the
768 MiB GPU snapshot cap. CPU inspection and file writes happen after the
request completes. Restoration checks confirm exact prior live/original
container identities, health/authenticated models 200 and unauthenticated
models 401. The live service
retains its **8,192-token context**; the original production containers remain
unchanged and stopped. No model, KV setting or kernel fix was promoted.

Repository evidence: [full numerical review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/nvfp4-repeat-v2-repeat-review.json),
[Mac verification](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/nvfp4-repeat-v2-mac-archive-review.json),
[restoration](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/nvfp4-repeat-v2-final-check.json),
and [snapshot relocation](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/nvfp4-repeat-v2-payload-relocation.json).

The disk-space stop was resolved by relocating 37 owned experiment archives
(6,414,709,938 bytes) to rank 1, checking SHA-256 before removing each rank-0 copy.
Receipts remain beside the former paths and in the
[relocation manifest](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/nvfp4-repeat-v1-archive-relocation.json).
All evidence remains available; source/checkpoint files were not relocated.
