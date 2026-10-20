# Native MTP: compatibility and C1 pilot

**18 October 08:24 UTC: MTP deferred by the user until compression has a working
outcome.** The repaired `mtppilotv3` preparer/verifier is frozen around eighteen
source checks and actual zero-rope geometry. Native CPU CLI/config/loader
preparation passes with 174 source bindings; original service stays healthy.
No `quality-model-controller-mtppilotv3.json` was created and no new MTP GPU
timing, acceptance or inference was run. Source and CPU readiness are retained
for later; current work is the [matched C8/C16 compression test](GLM53_HIGH_CONCURRENCY.md).

**18 October 07:52 UTC: actual-geometry CPU preflight passed and independently verified.**
After admission restored the exact original service, `mtp-geometry-v1` loaded
both real q/KV source matrices through native `Glm5NextMTP.load_weights`, then
through the loader bridge, into the declared **2,048 × 4,096** fused destination.
All source BF16 bits survive both paths; **zero padding rows** are required.
Four raw artifacts / 48 MiB and all fourteen archive payloads verify on Spark
and Mac, with byte-identical independent reviews. Original service stays healthy.

This tests native naming/shard orchestration with bounded destination storage
and stubbed expert mapping. Only **two projection comparisons** run here.
The corrected worker is frozen/bound but not instantiated; its next live source
check should have **18** comparisons, including sixteen routed/shared checks.
No CUDA, full model construction, expert/shared comparison, inference, acceptance
or throughput is measured by this fixture. The subsequent full-pilot verifier
and preparation repair is complete and CPU-validated; the old nineteen-check /
64-row success branch remains historical evidence. Further MTP testing is deferred.

[Actual-geometry CPU review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/mtp-geometry-v1-review.json).
Archive `evidence-mtp-geometry-v1.tar.gz`, 40,031,996 bytes, SHA256
`1d7aa99458c4ccef6144e26f8bc2cf85e4d76f038c92dd10e9330e12cd13b4db`.
Source archive SHA256 `775f4cf1125a1c262fb28395cf297c16cf09716223b253fa1c45710d8204c23c`.
Completed CPU waiter PID 1475744; progress `ROOT/mtp-geometry-v1-progress.json`;
local source/evidence `/private/tmp/mxwave-mtp-geometry-v1/`.

**18 October 2026, 05:57 UTC: the corrected CT loader reaches GPU readiness;
the retry stops at an incorrect observer assertion before client inference.**
Exact original service identities are restored healthy at **05:50:54 UTC**.
All 338 retry payloads independently verify on Spark and Mac, with byte-identical
reviews. There is no MTP acceptance or speed result. The imported checkpoint contains
one draft layer (45), and the pinned image contains `Glm5NextMTP` plus KDA
accepted-token state/rollback hooks. Those facts establish available code and
weights, not qualified speculative inference.

## Actual loader findings and current retry

The first full-model attempt, `mtppilotv2`, completed its compressed no-drafter
control, then failed **before MTP inference** on both ranks:
`KeyError: model.layers.45.mtp_block.mlp.experts.routed_experts.w2_weight_packed`.
The actual runner uses `model_runner.speculator` and the newer
`vllm.v1.worker.gpu.spec_decode.eagle.utils.load_eagle_model` helper. Earlier
CPU sharing tests exercised the legacy proposer, missing this path.

The newer helper passes the target's **already-mapped CT object** into draft
construction. Consequently, correct draft class metadata alone is insufficient:
its FP4 experts fall through to the default MXFP8 scheme. Independently authored
`mtp_draft_loader.py` makes a separate CT object from the unchanged checkpoint
config for this one draft call. The target CT object's serialized bytes must
remain unchanged.

The earlier claimed padding gap does **not apply to this checkpoint**. The
generic base loader supports a 64-row NoPE case, but this source config declares
`qk_rope_head_dim=0`, `q_lora_rank=1536`, `kv_lora_rank=512` and hidden width
4,096. Its fused projection is **2,048 × 4,096**, with no added rope rows.
The optional adapter's guard correctly skipped padding, but the observer wrongly
required one padding event and later assumed 2,112 fused rows. That observer
error caused this retry's failure. Local checks now derive dimensions from
the loaded model config and cover both zero/nonzero-rope geometries.

The loader also requires all **1,756** draft source tensors and every
declared nonvocabulary parameter before serving; only the vocabulary modules
replaced by native sharing may remain unloaded.

`mtp-loader-v3` exercises the **actual newer helper**, with a substituted model
factory, plus the native `load_weights` function for one real CPU BF16 source
projection in bounded stand-in parameter storage. All 872 selectors pass after
isolation (870 fail when inherited), target policy stays unchanged, native whole
module/index-buffer sharing passes. Its projection subtest uses real source
bits in an **artificial 64-rope-row destination**, preserving those bits and
zero extension; it did not verify the actual zero-rope model geometry.
No CUDA, full draft construction, argmax or
inference is tested. Nine payloads verify locally; archive SHA256
`bce71777d09371d507809d1717fe780bbc925c90a871eb4e4f0b040935014918`.
[CPU review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/mtp-loader-v3-review.json).

The completed control retains ten trials / 640 timed tokens: **16.768 TPS** at
P256 and **13.961 TPS** at P2048. Its 48 real decode comparisons have matching
IDs/conditioning, but mean KL **1.1901962e-5** fails the fixed **1e-5** floor;
mean NLL delta **2.29481e-5** passes the fixed **1e-4** floor. Preserve this
failure. All 333 attempt payloads independently verify on Spark/Mac, reviews
byte-identical; originals restored exactly and healthy at 23:05:49 UTC.
[Attempt review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/mtp-pilot-v2-review.json).

Completed retry `mtpretryv1` loads **one** corrected MTP arm and reuses that complete
control. Target weights, compute sources, precision, transport and row limit
stay unchanged; the control is not freshly paired in time. Historical controller
PID **495392**, watchdog **498189**, untimed metrics observer **506688**. Source:
`ROOT/full-mtpretryv1-src` (173 files); prepared
`ROOT/mtp-retry-v1-prepared.json`; local `/private/tmp/mxwave-mtp-retry-v1/`.
Model loading completes and the isolated service becomes ready, but the
untimed storage report aborts before draft source-byte checks or warmup. No
draft comparison or timing passed. The corrected local observer expects 18
byte comparisons for this checkpoint: 16 routed/shared plus q and KV, with
no empty zero-rope comparison. It is now bound in the passing CPU preflight;
another full-model GPU attempt has not run.
[Retry review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/mtp-retry-v1-review.json),
archive SHA256 `90428f9bc31f8e83f35253ec9948afc07352675b6e933ee01403076c10e985e6`.
The next preflight must exercise actual declared geometry, then require the
same source checks and unchanged numerical gates before any speed claim.
The sections below retain the earlier preparation history.

## Confirmed selector issue

Checkpoint CT targets and BF16 exclusions use explicit names under
`model.language_model.layers.45`. Native MTP creates its text-only modules
under `model.layers.45`; it also needs fused projection aliases. The target
model's existing name mapping and aliases do not by themselves configure the
separate draft class.

`mtp-dispatch-v1` runs the **installed vLLM CT resolver** before/after a draft-only
metadata repair. Of fourteen representative selectors, twelve initially differ
from the checkpoint source policy. After repair, all fourteen agree:

| Tested modules | Source/after policy | Count |
|---|---|---:|
| Fused MLA input, q/kv output projections, attention output, fused indexer, router | BF16 / unquantized | 6 |
| Shared-expert fused gate/up and down | MXFP8 | 2 |
| Experts 0 and 287, all three projections | MXFP4 | 6 |

The independently authored `mtp_mapping.py` copies fused aliases into an
independent draft mapping and translates the checkpoint namespace. The base
mapper object and policy remain unchanged, checked by native selector calls
and an inheritance/isolation unit test. No parameter, checkpoint config,
native kernel or production process is changed.

This is a **representative selector audit**, not all 864 MTP expert selectors,
a weight-loading test, or numerical qualification. It uses only a 2 × 2 meta
Linear: no model allocation or CUDA initialization. The two GiB CPU container
exits zero, and original service health stays 200. Expected CPU-only import
warnings are retained in its log.

Eight evidence payloads and all before/after records verify locally. Archive
SHA256: `b4485a3f19c6a79f567c87b3b338e56ff41b3e26647d620337cb9b1e0a40ddc6`.
[Compact review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/mtp-dispatch-v1-review.json).
Source/results: `ROOT/mtp-dispatch-v1/`; local continuation:
`/private/tmp/mxwave-mtp-dispatch-v1/`. Read-only installed-source inspection
is separately retained under `ROOT/mtp-eligibility-v1/`.

## Complete CPU follow-up

`mtp-dispatch-v3` extends the installed resolver check to **872 selectors**:
all 864 expert projections, six BF16 selectors and two shared MXFP8 selectors.
There are **870 source-policy mismatches before repair and zero afterward**.
The base model's mapper and policy remain unchanged. The repair also maps
the root `lm_head` BF16 exclusion to the native draft head prefix before the
native proposer replaces that module with the target head.

An independent stdlib read checks all **1,756 source tensor headers** from
four shards, including packed/scales dtypes, complete expert coverage and
scale shapes. Native embedding/head sharing passes a CPU fixture whose
target modules have no dense `.weight`; MTP index-reuse selection also passes.
This fixture does not execute local argmax or inference. The first expanded
V2 attempt stopped at missing proposer configuration in the fixture; its
failed log is retained, and the corrected V3 exits zero in about 21.7 seconds.

All twelve V3 payloads verify locally, including independently reread source
headers. No CUDA initialization, model allocation or original-service pause.
Archive SHA256:
`ccec92787cf7071ae9ab3d630cd769918a52836bf2b2db53a272efd6cd4cae9a`.
[Compact verification](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/mtp-dispatch-v3-review.json).
Source/results: `ROOT/mtp-dispatch-v3/`; local evidence:
`/private/tmp/mxwave-mtp-dispatch-v3/`.

## Work before any model pause

1. Start from the **frozen paired-speed source**, available locally under
   `/private/tmp/mxwave-paired-speed-v1/extracted-rank0/source` and remotely
   as `ROOT/full-pairedspeedv1-src`. Do not copy the whole current integration
   directory: its MoE runtime contains unqualified changes.
2. Actual CLI checks for both arms and native speculative configuration now pass.
   The `Glm5NextMTPModel` architecture alias resolves to
   `vllm.models.glm5next.nvidia.mtp.Glm5NextMTP`, with inherited CT/BF16,
   TP2, FP8 KV and the capped 8,192 context. Local argmax reduction is enabled.
   Actual loaded values still require the GPU checks.
3. Keep the draft's scale storage and GEMMs **stock**. The current scale hook would otherwise
   process an additional loaded MXFP4 expert layer, expanding its qualified
   42-layer scope. Add an explicit layer-45 bypass with a checked one-draft
   receipt; require the base scale/KDA/vocabulary audits to remain unchanged.
   The existing qualified TP-only 288-expert canonical producer stays active
   for both base and draft. The owner prefix is captured at native quant-method
   selection because the old adapter's stored labels are all `unknown`.
4. The immutable two-arm controller/client and raw-evidence analysis are written;
   finish CPU preflight before running them. Keep
   idle admission, bounded process/runtime
   limits and exact original-container restoration with an independent watchdog.

## Proposed first GPU comparison

Use the current compressed candidate as the no-drafter control, versus the same
candidate with native **MTP depth two**. Retain the existing KDA fused limit of
eight rows: C1 verification at shallow depth fits it. This avoids conflating
the pilot with the unresolved broad routing threshold. Do not change FP4 codes,
FP8 KV settings, transport, precision or the base adapter's qualification scope.

Compare C1 at 256 and 2,048 prompt tokens, using fixed generated-token limits,
all retained trials and the same request protocol as the paired-speed baseline.
Record ordinary API TPS, TTFT, inter-token cadence, generated IDs, memory, draft
counts, accepted tokens by draft position and actual verification row counts.
Collect draft/verification timings after unprofiled timing. Acceptance alone
is not the speed result.

Frozen source-only overlay: `ROOT/mtp-pilot-v2-overlay/`, source assembled from
the previous frozen snapshot under `ROOT/full-mtppilotv2-src` on both nodes.
Only the conditional plugin entry hook changes in existing qualified source;
native libraries, MoE runtime and scale codec stay hash-identical. The proposed
native configuration uses local argmax reduction for draft token selection.
Successful CPU proof: `ROOT/mtp-pilot-v2-cpu-preflight.json`; prepared inputs:
`ROOT/mtp-pilot-v2-prepared.json`. Earlier alias-assertion and occupied CPU
container-name failures remain preserved; the fresh CPU configuration container
exits zero. No service pause occurs during preparation.

Active model controller: `ROOT/quality-model-controller-mtppilotv2.json`, PID
**3328686**; independent restoration watchdog **3332671**. The controller
revalidates source/CLI and waits for idle before stopping the originals.
Actual target graph metadata rounds the requested capture size one to three
rows for depth-two verification; the requested maximum four still covers it.
Runtime snapshots and the separate profile must confirm actual execution.
The original source archive and test-plan SHA256 are
`e5b0d776934888738fa453d564a21c08fe6bfee979b5854a4cdd35947c6c9538` and
`005c8ac50373d6eaa27b1344707c0666766506db246ed056db1df8464178f7f7`.

Keep numerical/conditioning controls and the existing KL/NLL floors, with
reference-repeat controls beside the candidate. Preserve all discrepancies;
the open MLA cause is not resolved by this pilot. Do not claim 1.5–2×, complete
inference equivalence or release readiness before measuring them. Broader
concurrency/depth sweeps follow only if the C1 pilot supplies a qualified lead.
