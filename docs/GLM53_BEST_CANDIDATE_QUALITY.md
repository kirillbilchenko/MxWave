# GLM best-candidate quality evaluation

**Latest follow-up: [stable dispatch and whole-model results](GLM53_STABLE_DISPATCH_EXPERIMENT.md).**
The new matched run passes the unchanged numerical gates, saves the same
9.37 GiB, and improves single-request throughput 11.4–11.9%. All four sampled
PPL runs have identical float32 scores. Overall strict qualification still
fails because two output-format passes are lost; both score 30/34. Both tools
are now measured and pass. Original GLM is restored and all 959 new archive
payloads are verified. The qualitybestv2 experiment below is historical;
its measurements, original numerical failure and missing tool cases are unchanged.

**The paired evaluation is complete: 9.37 GiB / 5.33% less allocated CUDA
memory across TP2, with no sampled PPL or task-score degradation. Strict
numerical qualification remains failed because both reference and candidate
repeats exceed the unchanged stability floors.** The first observable
repeat difference is in prefill around the MLP, with compression disabled
as well. Both original GLM containers are independently verified restored,
with matching identities/configuration, health 200 and no active experiment
containers at **09:22:22 UTC**. Independent evidence and score verification pass.
No throughput sweep was repeated and no production defaults were changed.

## Results

| Measurement | Reference | U8 combined candidate |
|---|---:|---:|
| CUDA allocation after inference, both nodes | 175.86583 GiB | 166.49867 GiB |
| WikiText sampled PPL, first / repeat | 4.30444 / 4.30203 | 4.29993 / 4.30106 |
| GSM8K fixed test sample | 22/24 | 22/24 |
| Exact text and retrieval | 4/4 | 4/4 |
| Strict JSON | 1/2 | 1/2 |
| Python functional tests | 2/2 | 2/2 |
| Python semantics plus required bare-code format | 1/2 | 1/2 |
| All completed inference cases, strict scoring | 28/32 | 28/32 |
| Named tool cases | 2 API rejections, unmeasured | 2 API rejections, unmeasured |

The allocation saving is **10,057,913,856 bytes**, or about 4.68 GiB per node.
Configured KV allocation stays fixed at 4 GiB per rank. This is additional
runtime saving over the already quantized checkpoint. It does not compress
the FP4 weight-code payload or change KV precision. Every source-scale group
passes the independent audit before and after inference. All selected
loaded-weight checks and nine basic graph/eager chats pass in each arm.

The same two math questions fail in both arms; no reference task pass is lost
and no failed task is newly passed. Both strict-format failures are Markdown
fences: one JSON response and `slugify`. Functional code scores remain separate.
The 24-question math sample and 32 sampled text windows support this bounded
comparison, not broad benchmark accuracy or an improvement claim. PPL is
slightly lower in this sample, but only two repeats are available.

| Actual-decode comparison | Graph mean KL | Eager mean KL |
|---|---:|---:|
| Reference repeat | 0.00003083884 | 0.00003115398 |
| Reference to candidate | 0.00002978811 | 0.00002680417 |
| Candidate repeat | 0.00002727819 | 0.00003349726 |

All generated IDs match for all six comparisons, 64 IDs per comparison,
including 48 real KV decode steps. Full logprob arrays are not bit-identical.
The reference/candidate differences are comparable to the observed repeat
variation, but the strict `1e-5` repeat-KL floor is not passed. Teacher-forced
NLL repeat deltas are -0.0005608541 for the reference and +0.0002632200 for
the candidate, both beyond the `1e-4` floor. Candidate versus first reference
NLL is -0.0010493332; the corresponding sampled PPL change is -0.10488%.
No quality threshold was relaxed. Exact weight storage and end-to-end
bitwise arithmetic identity remain distinct claims.

## Run history and isolation

The first `qualitybestv1` attempt was launched detached on Spark1 at
**08:39:41 UTC, 17 October 2026**. Preparation verified 125 identical files
on both nodes. Its first source-report RPC failed serialization because
`torch.__version__` is a `TorchVersion` subclass that this vLLM serializer
rejects. No quality scores were produced. The client was stopped promptly
and its controller restores the original service. Failed evidence is retained.
The corrected `qualitybestv2` source uses plain RPC primitives and includes
an actual msgspec serialization preflight; both frozen nodes' 111 Python/TOML
files match. The retry launched at **08:49:26 UTC**, passed both CPU
preflights and paused idle GLM at 08:50 UTC. Both arms are now collected. The failed
attempt restored both original identities/configuration and health 200;
its independent evidence receipt is `quality-best-evidence-receipt-v1.json`,
archive SHA256 `dc8e5dcc9d5e6b525dcbecb6d5c565c0cb324016cfdd3b87097517aa960bd830`.

The controller checks the real plugin and CLI in a
CPU-only container before pausing idle GLM, runs both arms without an
intermediate restoration, then restores the same original containers.
An independent watchdog and evidence collector run on Spark1; neither needs
the laptop connection. This batch collects quality and allocation, with no
new throughput sweep.

## Comparison

Both arms load `INCModel3/GLM-5.3-Flash-MXFP4-Mixed-CT-AutoRound`, revision
`35d1b0d5cb4a0db075fd2749c897e970dc15bfa5`, using the same pinned vLLM image,
attention aliases, loader, TP2 transport, BF16 activations, Marlin backends
and 4 GiB FP8 KV allocation per rank. The reference leaves expert scales,
KDA projections, head and embeddings uncompressed at runtime. This reference
is already quantized MXFP4; it is not the different production ModelOpt
NVFP4 model, or an original BF16 accuracy baseline.

The candidate combines exact expert-scale palette storage with U8 shared
staging, selectively compressed BF16 KDA projections (fused up to eight
tokens), compressed BF16 head and row-addressable embeddings, and the lighter
PyTorch dispatcher. FP4 weight codes and the KV representation are unchanged.
The v9 library SHA256 is
`9f5d96dcb0762d7e5ed5436ca296f70d590e73797d40522e4d8c7d6f3902f5f0`.
The rejected v10/v11 nibble consumers are excluded. The runtime and codec
files match the two successful v9 component qualification reports, including
their comparison manifest and candidate library digest.

## Frozen measurements and gates

- Hash all 168 actual TP2 expert-scale groups before and after inference and
  compare them with the independent full source audit. Retain the existing
  selected weight/attention checks and graph/eager/graph chat checks.
- Compare full-vocabulary distributions on the existing 16 diagnostic windows,
  including three real KV decode steps per window. Collect reference repeat,
  candidate and candidate repeat in graph and eager modes during each load.
  These authored windows are numerical diagnostics, not standard perplexity.
- Keep the strict reference-repeat floor: **mean decode KL <= 1e-5 and
  absolute mean NLL difference <= 1e-4**. Candidate limits remain
  `max(1e-7, 5 * reference-repeat KL)` and
  `max(1e-5, 5 * abs(reference-repeat NLL difference))`.
  An unstable reference cannot qualify an identical candidate. Default graph
  dispatch must pass; a favorable eager diagnostic cannot replace it.
- Retain all token logprobs for 32 disjoint WikiText-2 test windows, scoring
  1,024 tokens each (**32,768 scored tokens per arm per repeat**). Recompute
  token-weighted NLL/PPL independently from the hash-bound arrays. Absolute
  reference/candidate repeat NLL drift must each stay <= 1e-4; the candidate
  absolute NLL difference limit is `max(1e-5, 5 * abs(reference-repeat delta))`.
- Score 24 fixed sampled GSM8K test questions and 10 authored practical tasks:
  two JSON contracts, two exact-text contracts, two executable Python
  functions, two retrieval prompts and two forced tool calls. Use natural
  termination; truncated reasoning cannot pass. Generated code is executed
  in a bounded CPU sandbox with no network or host write access. Required
  output format is scored separately from semantics.
- Require that no task passed by the reference fails in the candidate.
  Record individual losses/gains rather than hiding them in an aggregate.
  API failures are missing measurements, not matching model answers.
  This small task sample does not establish full benchmark accuracy.

Both frozen arms specify `--tool-call-parser glm47`. The interval task explicitly
requires a list of two-element lists, repairing the preceding ambiguous return
contract. The tool launch still omits `--enable-auto-tool-choice`: the pinned
parser manager returns no tool parser without that flag, even for a named
tool request. The two reference tool requests return HTTP 400 before inference.
A CPU-only test in the exact image confirms parser absent with the flag false,
and `Glm47MoeModelToolParser` available with it true. The local runner now includes
both flags and checks actual parser availability. **This repair is for the next
launch; it does not alter the frozen v2 engines.** Tool calls in this run remain
unmeasured. The independent review requires all inference requests to complete,
so matching API failures cannot qualify as matching model quality.

If repeat KL fails, collect two eager output-fingerprint runs across decoder,
attention and MLP modules during the same model load. These hooks copy hashes
to CPU and synchronize execution; they can help locate divergence but alter
execution and cannot qualify the normal graph path. A diagnostic hook failure
is retained without discarding the rest of the quality suite, after confirming
all hooks have been removed.

## Repeat fingerprints

Two intrusive eager runs per arm retain 540 events per rank. The first
differing output is the same on both ranks for each comparison:

| Comparison | First event | Module | Differing events |
|---|---:|---|---:|
| Reference repeat | 13 | `language_model.model.layers.4.mlp` | 500/540 |
| Candidate repeat | 16 | `language_model.model.layers.5.mlp` | 493/540 |
| Reference to candidate | 13 | `language_model.model.layers.4.mlp` | 527/540 |

The first differing tensor shape is `[258,4096]`, already a **prefill**
difference before one-token decode. Graph dispatch alone does not explain
it: normal eager repeats also exceed the KL floor. The hook records output
hashes, not MLP inputs or numerical magnitudes, so it does not establish
whether the source is routing, expert/shared-expert GEMM, preceding mHC/norm,
or even whether the first bit difference has numerical significance.
CPU hash copies synchronize execution and can perturb the run. The trace
narrows the next experiment; it does not prove a particular faulty kernel.

## Public data and reproducibility

[WikiText](https://huggingface.co/datasets/Salesforce/wikitext) is pinned to
revision `b08601e04326c79dfdd32d625aee71d232d685c3`, subset `wikitext-2-raw-v1`,
test Parquet SHA256
`5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91`.
Join every text row, including empty rows, with two newline characters.
The resulting UTF-8 text is 1,296,370 bytes, SHA256
`696cca6b65a171b0a358a4be6732cdfdf2dd6164a32e20fd70e3c13fc4dfae83`.
Tokenize with the loaded checkpoint tokenizer, without special tokens, and
select 32 evenly spaced disjoint 1,025-token windows, scoring all but the
first token. Freeze the IDs before candidate inference. This is sampled-window
PPL under the recorded policy, not full-corpus WikiText PPL.

[OpenAI GSM8K](https://github.com/openai/grade-school-math) is pinned to commit
`3101c7d5072418e28b9008a6636bde82a006892c`, raw `test.jsonl` SHA256
`3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`.
Choose 24 sorted test indices using Python RNG seed `20261017`; retain the
indices, prompts and expected numeric answers in the frozen protocol.
Require a final numeric answer, accepting the explicit `####` final-answer
marker, and never extract an arbitrary number from unfinished reasoning.

Original dataset cards, file URLs, revisions, hashes and conversion policy
remain in the private evidence. Corpus content and model completions are not
added to Git. The WikiText card records CC-BY-SA 3.0/GFDL provenance.
The Parquet reader was installed only into an isolated CPU helper directory;
no production environment was modified. Evaluation code is independently
authored; no dataset repository code is copied.

## Spark locations

`ROOT=/home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/full-model-v1`

| Item | Under ROOT |
|---|---|
| Controller state | `quality-model-controller-qualitybestv2.json` |
| Controller log | `quality-best-controller-v2.log` |
| Frozen source, both nodes | `full-qualitybestv2-src/` |
| Qualified library and proofs, both nodes | `quality-best-v1-native/` |
| Dataset artifacts and manifest, both nodes | `quality-best-v1-data/` |
| Partial and final measurements | `run-qualitybestv2/results/` |
| Final analysis | `run-qualitybestv2/results/quality-model-review.json` |
| Restoration proof | `restoration-qualitybestv2.json` |
| Independent final identity/health check | `qualitybestv2-final-service-check.json` |
| Evidence archive and receipt | `quality-best-evidence-v2.tar.gz`, `quality-best-evidence-receipt-v2.json` |

The batch has startup, per-arm and overall deadlines. The watchdog restores
the exact original container identities on a dead controller or expired
outage deadline. Failed preparation and partial measurements are retained.
Never mark the candidate qualified merely because the controller completes;
the review records failed quality gates as measured results.

## Evidence and verification

The [compact independently verified review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/best-u8-full-model-quality-v2-review.json)
retains aggregate distributions, per-case scores, source audit/native bindings,
parser proof and verification hashes. Its SHA256 is
`a10bd61cc588f053f5e631122f3a738eb2812d06556eab7e9add6c456e8a9e0d`.
All full step records, float arrays, prompts/completions, the frozen integration source,
logs and restoration checks remain in the private evidence archive:

- `ROOT/quality-best-evidence-v2.tar.gz`, **157,459,727 bytes**,
  SHA256 `0a25c69d9c8ea39dae86c9b61d9e6683369fb6d3778acb2cb160b1a8c52aee14`.
- All **826 manifest payloads** match their size and SHA256 locally. Every
  original review JSON input hash also matches. The native library matches
  the pinned v9 digest.
- Independent CPU verification compares all actual scale groups with the
  source audit in both arms before/after inference: **672 group checks**.
  The audit SHA256 is
  `a83ec4b65d4520ead1e63f99ddd8cb1e10424664afb65f2763f9fe0c25cee927`.
- Decode metrics are also recomputed with `numpy.logaddexp.reduce`, separately
  from the original max/exp/sum normalization. Held-out losses and every task
  score are recomputed. Code execution remains the original hash-bound
  sandbox result; generated code is not executed on the laptop.
- Cross-platform review differences are at most about `2e-15`; the review
  recomputation tolerance is `1e-12`, independent normalization tolerance
  `1e-10`. These are calculation checks, not changes to model quality gates.
  All shared acceptance flags agree with the frozen review.

The local analyzer adds an explicit completed-inference requirement, so API
errors cannot qualify as matching task quality. This keeps the already-failed
v2 qualification false and does not change its frozen engines or metrics.
The collector initially wrote v2's receipt to the v1 receipt filename. Both
immutable archives were intact and the original v1 receipt was retained in
v2's stage; the metadata filenames are corrected and recorded in
`ROOT/quality-evidence-receipt-repair-v2.json`.

## Next targeted repeatability check

The paired evaluation and independent verification above are complete.
A single-reference diagnostic, repeatpolicyv1, launched detached on Spark1
at **09:43:14 UTC, 17 October 2026**. Its 115 Python/TOML files match on both
nodes; exact-image CPU/plugin/CLI validation passed before the idle pause.
The independent restoration watchdog acknowledged before GLM was paused.
The diagnostic is complete: strict deterministic mode still drifts, while
360 fixed-input expert GEMM replays are bit-exact. Consumed token-dispatch
order varies with matching logical routes and tiny first floating differences.
Both original containers are independently verified restored healthy.
The next controlled ablation swaps only retained dispatch permutations
in a bounded single projection; original qualification remains failed.
See the
[reference diagnostic](GLM53_REFERENCE_REPEAT_DIAGNOSTIC.md) for paths and limits.

The diagnostic uses one targeted stock/reference load, with the same source,
transport and attention fixes. The new `repeat_worker.py` instruments runtime
inputs and components before startup; the old output hashes cannot reconstruct
input activations. The following decision sequence remains in force:

1. Keep the existing first diagnostic prompt: 256 frozen tokens plus
   `[154822,154824]`, giving 258 prefill tokens. Record layer 4 MLP input bytes
   as well as outputs on two repeats, per rank. Record numerical differences
   as well as bitwise hashes. Include the preceding mHC
   and normalization boundaries. Use bounded scratch and a small retained
   activation fixture under `/traces`, rather than retaining every layer.
2. If the input differs, move the boundary upstream until the first differing
   operation has identical inputs. If it matches, record router logits,
   selected expert IDs/weights, expert outputs, shared-expert output and their
   final combination. The current MLP output hash alone cannot distinguish
   these causes.
3. Replay that one operation with the retained fixed inputs and unchanged
   weights, comparing every output byte. Keep uninstrumented and synchronized
   repeats separate. Existing component tests use synthetic routing; their
   exact outputs do not prove full-model router repeatability.
4. Investigate the operation that actually varies. Router variation calls
   for its GEMM/top-k path; fixed routing with changed expert values calls
   for expert GEMM/reduction; matching components with a changed sum calls
   for the final reduction or synchronization. Use Nsight only after that
   boundary is identified, to inspect its launches and ordering.
5. After a fix, first rerun the small strict repeat window with the unchanged
   floors. A stable reference is required before another full loss/task
   qualification. No throughput sweep is needed to locate this variation.

Future tool tests must use both `--tool-call-parser glm47` and
`--enable-auto-tool-choice`. The CPU parser proof is
`ROOT/quality-tool-parser-cpu-proof-v2.json`; the local launcher now tests this
actual parser registration before any production pause. Fixing those flags
does not address the separate numerical repeat failure.

### Interpretation of the proposed follow-up

The current reference-repeat failure already occurs **within one loaded
process**. Comparing new containers is not required to establish it; per-load
algorithm selection remains a secondary hypothesis if fixed-input checks
cannot locate the varying operation.

The diagnostic pins CUBLAS_WORKSPACE_CONFIG=:4096:8 **before** CUDA starts,
then compares five uninstrumented eager repeats under the default and strict
PyTorch deterministic policies in that same load. Both policies share the
new workspace setting, so this does not isolate workspace from the old launch.
Custom kernels may not implement PyTorch's policy, and strict mode may reject
unsupported operations. A stable run would be a useful lead, not proof that
the old drift is harmless reduction-order noise. PyTorch itself documents
that the switch alone is insufficient to guarantee application reproducibility:
[deterministic algorithms](https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html).

No original floor changes. A future statistical non-inferiority protocol
would be a separately declared contract, preserving the failed frozen v2
result. Five repeats alone do not establish harmlessness, broad task parity,
or independence of per-token samples. The missing tool measurements also
remain a requirement; stabilizing the reference does not automatically
qualify the candidate.

Larger held-out and task samples become useful after repeatability is
understood. Prompt/format repairs would be separate labeled cases: the two
existing Markdown-fence failures stay failures in the frozen scoring.
U8 shared staging means staging **scale data in GPU shared memory**; it does
not mean quantizing or compressing the model's shared experts. Remaining
norm/router/mHC tensors need a byte inventory before predicting another
material saving. KV changes require a separate precision/quality experiment.
The existing imported-source inventory already counts only 99,090,432 BF16
router bytes; sampled lossless packing predicts about 29.3 MB saved per
checkpoint copy. Another 71,524,352 BF16 bytes are classified as other.
TP replication changes runtime totals, but these families are not another
multi-GiB opportunity on their own. This is a source inventory/sample estimate,
not a new measured runtime saving.

A BF16-origin run answers the separate question of the imported MXFP4
checkpoint's quantization loss. The audited BF16 tensor payload is
642,646,653,816 bytes (about 643 GB), beyond the two Sparks' combined usable
memory, so this is not a cheap additional in-memory arm on this setup.
It is not needed to isolate the extra lossless adapter's effect against
identical quantized source weights.
