# CT drafter adaptation: CPU preparation

**20 October 2026: requested CPU preparation is complete. No production pause,
GPU inference, optimizer step or drafter promotion occurred.** Original production
was checked healthy at **13:21:47 UTC**, retaining its original container identity
and **500,000-token context**.

The experiment target is explicitly **CT MXFP4 experts + the already-tested native
MXFP8 KDA conversion**. It is a lossy target variant relative to source CT.
Original CT with BF16 KDA remains the lossless-compression reference. This
preparation makes no further target quantization change and does not relax the
failed 1e-5 cross-target KL gate.

## Frozen success contract

The [machine-readable thresholds](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/drafter-adaptation-v2-thresholds.json)
were written before training or collecting new model outputs. SHA-256:
`b71b5b53f0fa3fc3ba4d94dd030e3b45a0736231b32b647e9a4b2ea5f1e1d87b`.

- C1 code: at least **8% higher end-to-end output tok/s**, including first-token
  latency, against the unchanged drafter on the same fixed CT target.
- Code: at least **7.407% fewer verification rounds**, corresponding to 50/54,
  measured as paired ratios on the new test prompts.
- Chat and explanation: at most **2% throughput or TTFT regression** in each
  domain. The throughput non-regression confidence bound must also pass.
- Repeat controls, confidence intervals, exact greedy token agreement, no API
  errors and no unexpected truncations are required; failed controls stay failed.
- The already-seen code prompt's **54 → at most 50 rounds** is a separate
  diagnostic anchor. It is not a sealed test and its absolute count does not
  apply to every unrelated code prompt.

The contract fixes seven drafts, 256 output tokens, seed 42, temperature zero,
C1, disabled prefix caching, matched warmed-model/cold-prefix state, three
interleaved loads and six trials per prompt per arm. Aggregate paired per-prompt
ratios by domain; bootstrap prompts, not correlated draft blocks. The full sealed
serving evaluation is a later campaign, not a short CPU check.

Choose one candidate on validation, then open the test once. Do not change the
draft policy, dataset, thresholds or target to turn a failed result into a pass.
Natural-stop behavior and broader quality remain separate from this fixed-length
speed proxy. Production replacement still requires serving/capacity tuning,
stable C8/C16 results and actual long-context qualification.

## FP8 round trip: exact pass

The Mac reads the existing 1,377,719,136-byte FP8 drafter checkpoint from Spark.
Its SHA-256 matches the earlier inventory:
`6cba1cff8ab4fcd657b89575f5db3debe1f5a39c78bda422953f553dc2357f8b`.

An independently authored, bounded-memory converter decodes all **35 FP8
matrices / 964,689,920 encoded weights** to BF16 with the original FP32
128 × 128 block scales. It writes a 2,342,169,768-byte trainable safetensors file,
then reads that file and recomputes every FP8 code using those retained scales.
The other 46 BF16 tensors and 35 scale tensors preserve their original bytes.

| Check | Result |
| --- | --- |
| All 116 source/export tensor payloads | Zero different bytes |
| Full serialized model file | Original/export SHA-256 identical |
| Config file | Byte-identical |
| CPU conversion, export and hashing | 7.217 seconds |
| BF16 training file SHA-256 | `29a84776f54b4e24378e934adae6f1e869dc6fe4fea74396440e2fb764e2db82` |

The exporter preserves the original safetensors header, so the full-file check
is stronger than merely ignoring serialization differences. FP8 payloads are
recomputed from the written BF16 file; they are not copied around the conversion.
Unit checks cover all finite E4M3FN encodings, signed zero, non-power-of-two
scales, invalid scales and overflow rejection.

**Scope:** this is an untrained checkpoint round trip. It does not prove BF16
training arithmetic equals native FP8 Marlin arithmetic or that changed FP8
backbone weights can be safely re-quantized. The proposed context-first update
changes an existing BF16 matrix, allowing the original FP8 backbone codes/scales
to remain fixed on export.

[Round-trip evidence](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/drafter-preflight-v1-roundtrip-review.json).

## SpecForge mapping: actual strict CPU load passes

Audited upstream revision:
[`03f3682ea852b73488d62dab4cd539033cc60e3e`](https://github.com/sgl-project/SpecForge/tree/03f3682ea852b73488d62dab4cd539033cc60e3e).
The separate checkout remains unmodified. Its actual `DFlash2DraftModel` is
constructed on meta storage, then all **81 dense parameter tensors /
1,171,080,448 parameters** are strictly assigned from the BF16 checkpoint on CPU.
There are **zero missing or unexpected keys**, no shape mismatches and no
remaining meta tensors. Every named parameter maps directly, without renaming.

| Checkpoint component | SpecForge mapping | Remaining native-serving concern |
| --- | --- | --- |
| Five layers; 35 attention/MLP matrices | Same names/shapes after FP8 decode | FP8 Marlin arithmetic; native QKV/gate-up packing |
| `fc.weight`, 4,096 × 20,480 | Same context projection, 83,886,080 parameters | Correct target-feature boundaries, order and normalization |
| Hidden/final/attention norms | Same names/shapes | Native precision/order parity |
| Ten grouped convolutions | Matching base kernels and kernel projections | Native convolution arithmetic |
| Candidate selector | Matching projection and both codebooks | Native top-16 scores/path selection |
| 35 FP32 block-scale tensors | Retained export metadata, outside dense model | Do not silently recalibrate |
| Target head and embeddings | Intentionally absent from draft model | Must use the fixed CT target's actual shared modules |

Five sliding-attention layers, 2,048 sliding window, block size eight,
selector rank 256/top-16 and target layers `[5, 14, 24, 33, 42]` match.

The audit uses Transformers **5.12.1**, matching the pinned SpecForge dependency,
with the Mac's existing Torch **2.14.0**. SpecForge pins Torch **2.13.0** for
training. Its aggregate draft-package imports are bypassed to avoid loading
unrelated Eagle/MTP accelerator dependencies; the DFlash implementation itself
is imported unchanged. This is **not validation of the full CLI/GPU stack**.

Remaining engineering is concrete:

1. Pin an isolated GPU training environment and assert the chosen trainable
   parameter set. Context-only training still backpropagates through the frozen
   backbone; no backward/optimizer run has been performed.
2. Export native CT context-feature sequences and aligned token/position masks
   to the training data contract. The earlier selector-only snapshots do not
   contain these sequences.
3. Verify the untrained training forward pass against retained/native drafter
   behavior, then validate an exported update in the native runtime.
4. Keep FP8 draft KV/head adapters and target arithmetic explicit during the
   acceptance comparison.

These results remove a parameter-layout/serialization obstacle. They do not
establish a training-duration estimate or a recovered speed gap.

[Per-parameter mapping and limitations](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/drafter-preflight-v1-specforge-review.json).

## Frozen public prompt datasets

| Split | Code | Chat | General explanation/QA | Total |
| --- | ---: | ---: | ---: | ---: |
| Train | 2,400 | 800 | 800 | 4,000 |
| Validation | 120 | 40 | 40 | 200 |
| Sealed test | 120 | 40 | 40 | 200 |

Code prompts come from
[OpenCoder's evol-instruct subset](https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage2),
revision `7d28f40d579edd7c24402d17d0c7639f991e6f8d`, MIT license.
Chat/QA prompts come from
[Databricks Dolly](https://huggingface.co/datasets/databricks/databricks-dolly-15k),
revision `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`, CC BY-SA 3.0,
copyright Databricks and applicable Wikipedia contributors. Attribution and
source cards are retained. **Source answers are omitted**; future supervision
must come from the fixed CT target.

These are public, predominantly single-turn proxies, not the user's private
harness traffic. The explanation bucket contains general QA, not exclusively
technical explanations. The public dataset previews are excluded. Hash-ordered
allocation reserves held-out groups first and keeps exact/simple-template
instruction groups disjoint. No held-out content or model outputs were displayed
or used to tune a model. Sealing is a process boundary, not encryption.

The independent check finds no exact/template cross-split overlap, and no
test-versus-train/validation pair with word-five-gram Jaccard similarity at least
0.8. This does not establish absence of all semantic paraphrases or pretraining
contamination. The existing pilot's prompts and named benchmark references are
filtered; all three distinct retained speed-prompt token sequences are absent.

Prompt files:

- Train SHA-256: `f897f39cadbf2d04c960265151781bc0e56215d2c8eb0c4b3f0725cde3baca8b`.
- Validation SHA-256: `fe1041a07c26cad4caee9713801ef1aee973cc5ecb7b5e3171b6203e757bb5c5`.
- Sealed test SHA-256: `4cf86830c6087b59ccc89dc7e86b48a4d7d4f399e4c7027751064abc5b065fbd`.

The exact CT tokenizer and external chat template are copied read-only and
hash-verified. Mac tokenization uses thinking disabled and an assistant generation
prefix. All 4,400 prompts fit the 8k capture budget with 512 output tokens;
maximum input is 3,973 tokens. Tokenized files are separately hashed, with zero
cross-split token overlap. Native tokenizer replay remains a pre-capture check.

Full training input plus 512 generated tokens per prompt would require about
**98.40 GiB for the five BF16 feature streams alone**, before auxiliary data and
validation. Spark now reports roughly 155 GB free, replacing the earlier 3.7 GB
observation. Capture should start with a bounded subset and/or stream to the Mac;
an overnight estimate is not established by this CPU work.

Retained preparation failures: the first dataset assembly rejected an overly
restrictive 80-character lower bound before writing splits; the revised 20-character
bound admits short chat questions. The first tokenizer check rejected its output
because a newer Transformers default returned dictionary keys instead of token
IDs. Explicit `return_dict=False` plus integer validation repairs that issue.
Frozen prompt files were unchanged by the tokenizer correction.

[Dataset manifest](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/drafter-preflight-v1-dataset-manifest.json),
[independent split review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/drafter-preflight-v1-dataset-independent-review.json),
and [tokenized review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/drafter-preflight-v1-tokenized-review.json).

## Next GPU work and preservation

No GPU phase is launched by this preparation. Draft-length 3/5/7 development
baselines may be useful before training, with the actual widths and timing
measured. **MTP is not yet a qualified ready-made replacement**: the earlier
[MTP work](GLM53_MTP_PILOT.md) has CPU loader repairs but no completed acceptance
or speed result. Keep it separately bounded so it cannot consume the feature
capture window without a result.

Local artifacts live in `out/drafter-preflight-v1/`; valid prompts are under
`dataset-v2b/` and valid token files under `dataset-v2b/tokenized-v2/`.
Do not use the failed `dataset-v2/` or `dataset-v2b/tokenized/` preparations.
The original, BF16 and re-exported drafter files are under `original/` and
`roundtrip-v1/`. The source checkpoint already exists on both Sparks; CPU
derived files are reproducible from the pinned source and converter.

[Integrity record](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/drafter-preflight-v1-integrity.json)
binds the executed source files, reports and threshold contract.

Ruff and strict core mypy pass; **885 tests pass with one optional-dependency
skip**. The frozen public corpus and reports are backed up on **spark2** at:

```
/home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/git-backups/glm53-drafter-preflight-v1-data.tar.gz
```

Archive size: 2,461,260 bytes. SHA-256:
`298492d0d06f82d24bdceddbe1117bf9a56d49ff66501a801aca87fc7d6923fe`.
Spark2 independently verifies the archive and all six prompt/token file hashes.
[Backup receipt](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/drafter-preflight-v1-spark2-data-receipt.json).
