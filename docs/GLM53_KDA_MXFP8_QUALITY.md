# KDA MXFP8: measure additional quantization error first

**Follow-up complete and verified, 20 October 2026, 09:35 UTC.** The separate
[native speed experiment](GLM53_KDA_MXFP8_SPEED.md) and
[full GSM8K comparison](GLM53_KDA_MXFP8_GSM8K.md) are complete. Fresh CT BF16 KDA
scores **1,248/1,319 (94.62%)**; native MXFP8 KDA scores **1,260/1,319 (95.53%)**,
with 22 gains and 10 losses. Both have zero API errors and truncations. All
2,638 answers are independently regraded; 2,690 payloads and 2,933 numerical
fields verify exactly. Production is restored with its original 500,000 context.
This is a higher observed task score in one pass, not general superiority or
losslessness. The historical rounding study's failed **1e-5 KL gate remains**.

**20 October 2026, 06:05 UTC: complete and independently verified.**
All **1,688 payloads** verify locally. Spark/Mac recomputation agrees on
**5,414 numerical fields**, maximum difference **7.11e-15**.
All five passes finish. Exact original
production is restored at **05:51:18 UTC**, with authenticated port 8000 healthy
and its 500,000-context configuration retained. All 68 original KDA matrices
restore bit-exactly on each rank. No candidate configuration is promoted.

The declared MXFP8 rounding recipe **does not meet the unchanged 1e-5 KL
gate**. Sampled perplexity improves slightly, so this screen does not establish
a quality regression either. These are different questions: distributional
equivalence fails; broad task-quality equivalence remains unproven.

| Measurement | Original BF16 KDA | MXFP8-rounded KDA, BF16 execution |
|---|---:|---:|
| Sampled PPL, 32,768 tokens/pass | 4.1150974 in all three passes | 4.0857491 in both passes |
| Mean held-out NLL | 1.4146625 | 1.4075051 |
| Relative sampled PPL change | — | −0.7132% |
| Mean KL across all 216 positions | Reference controls fail on the 4,096-token context | 1.2641e-3 / 1.5891e-3 / 1.2326e-3 across the three declared pairs |
| Mean KL on 208 positions outside that long context | All self-repeats exactly zero | 1.22338e-3, identical across all three pairs |
| Eight short prompts: mean KL over 64 positions | All self-repeats exactly zero | 2.75066e-3 |
| Eight short prompts: mean first-token KL | Zero on self-repeats | 1.47761e-3 |
| Next-token argmax changes | Zero on self-repeats | 2/216; one in the long context, one on a short prompt |
| Actual MXFP8 serving speed | Not tested in this rounding study | Measured separately in the linked speed experiment |

The paired-window bootstrap 95% interval for candidate-minus-reference NLL is
**[−0.01409, −0.0007523]**, using the frozen 32-window sample. This is not a
whole-corpus or broad task-quality guarantee. Original and rounded held-out
log probabilities each repeat exactly within their respective arms.

The full reference-repeat controls remain **failed**, not discarded:
A1/A2 mean KL is 2.23248e-4; A1/restored-A3 is 1.69270e-5; B1/B2 is
2.54716e-5. All nonidentical repeat distributions belong to the single
4,096-token context, including its first prediction. Its repeatability issue
remains unresolved despite matching recorded input IDs, positions and batch
sizes: two 2,048-token prefill chunks followed by single-token decode in every
pass. The 208-position breakdown above diagnoses the rounding
effect in the exact-repeat portion; it does not replace or pass the full gate.
No reference KL is subtracted from candidate KL.

For short natural answers, six of eight returned texts match between original
and rounded weights. JSON wording and `slugify` code/formatting change. Both
Python functions pass the bounded sandbox in all three natural-answer passes.
The other six strict-output checks pass only **1/6 in each arm**: extra text
and duplicated answers already occur in the original baseline. Including code
correctness and requested formatting gives **2/8 original, 3/8 rounded,
2/8 restored**; the sole increase is removal of a Markdown fence. These small
counts cannot support a general task-accuracy claim. The restored original
returns exactly the same eight visible texts as the initial original.

Weight SQNR is **42.352/42.362 dB** on ranks 0/1; relative L2 error is
**0.7628%/0.7619%**. Independent analytic CPU checks match all retained samples
from 68 real matrices/rank (1,114,112 sampled elements/rank).

**This rounding study does not measure native MXFP8 speed.** The same BF16
GEMMs execute both arms to isolate weight-rounding error. The subsequently
completed [native serving comparison](GLM53_KDA_MXFP8_SPEED.md) measures that
execution path separately; its full GSM8K quality comparison is in progress.
The rounding results below do not establish native-kernel task accuracy.

This is a disposable, reversible **weight-rounding diagnostic**. The existing
CT MXFP4 checkpoint remains unchanged on disk. The engine backs up the 68
configuration-bound BF16 KDA TP2 matrices in RAM, rounds their values to MXFP8,
and stores the decoded values back into the same BF16 allocations. Native BF16
GEMMs, model settings, graph addresses and other weights stay the same.
Original KDA bytes must restore exactly after the candidate measurement.

The declared recipe is E4M3 round-to-nearest-even with one E8M0 power-of-two
scale per 32 input columns. Scales round upward from `max_abs / 448`, with an
explicit zero-block rule. This independently written implementation follows
the public [OCP MX format](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
and [NVIDIA's MXFP8 scaling description](https://nvidia.github.io/TransformerEngine/features/low_precision_training/mxfp8/mxfp8.html).
It is one declared recipe, not a claim about every possible MXFP8 quantizer.
Independent analytic-codebook tests cover ties, zero, scale boundaries and
TP-aligned partitioning. Both ends of every real matrix are checked against
that CPU reference during conversion.

## Frozen measurement

- Five passes on the same model load: original A1/A2, rounded B1/B2, restored A3.
  Keep all results, including failures. Compare original repeats before/after,
  rounded repeats, and each declared cross-arm pair.
- C1, speculation off, prefix caching off, identical tokenizer, KV allocation,
  kernel policy and input ordering. No concurrent user traffic on the private
  diagnostic port 8013.
- 27 contexts: 16 retained 258-token diagnostics, two code/chat prompts,
  eight short practical prompts, and one 4,096-token held-out context.
- Eight prediction positions per context, retaining raw full-vocabulary logits
  before sampling. A1 generates the continuation. Later passes force those
  exact token IDs after saving their unmodified logits. Actual input IDs and
  positions are retained; incomplete prefill chunks cannot count as decode
  predictions. Forced API probabilities are never used for KL. Fixed eight-token diagnostics
  can extend beyond a natural stop; natural-answer tasks are scored separately.
- Report mean, p95 and maximum reference-to-candidate KL, next-token argmax
  changes, and separate prefill/decode, short-prompt and longer-context results.
  The historical mean KL gate remains **1e-5**; do not assume it passes or fails.
  Reference-repeat noise is reported separately, not subtracted from candidate KL.
- 32 frozen 1,024-scored-token WikiText windows per pass: **32,768 tokens**.
  Retain raw prompt log probabilities and recompute token-weighted NLL and
  sampled perplexity. Report paired window uncertainty. These are sampled
  windows, not standard full-corpus perplexity.
- Eight natural short-answer tasks in A1/B1/A3: JSON, exact text, two simple
  arithmetic questions and two Python functions. Check correctness separately
  from wording and formatting. Execute generated code only in a bounded,
  network-isolated CPU sandbox. This small sample is not a full task benchmark.

Identical visible answers do not prove identical probabilities. Changed text
does not by itself prove worse quality. This experiment estimates **additional
KDA weight quantization error relative to the already-quantized CT checkpoint**.
It does not measure the checkpoint's existing MXFP4 error versus an original
BF16 model.

## Limits and recovery

No native MXFP8 inference kernel or activation quantization is introduced.
Passing this screen would justify testing the actual MXFP8 execution path; it
would not qualify that path or prove a speed/memory gain. Capturing logits
synchronizes GPU work to CPU, so these request times cannot support a serving
speed claim. C8/C16 and context beyond the 8,192-token diagnostic configuration
remain outside scope.

Both-rank preflight checks bind 211 existing runtime files and eight new
diagnostic/protocol files. A detached controller and independent watchdog use
a one-hour outage bound and restore the exact original production containers,
checking the authenticated 500,000-context model endpoint. The controller
refuses to start before `mxfp4-serving-v3` completes its restoration.

Local staging: `/private/tmp/mxwave-kda-rounding-v1`.
Remote directory:
`/home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/full-model-v1/kda-rounding-v1`.
Read `controller.json` and `measurement.json` for execution status.
Controller PID **2777519**, watchdog PID **2777520**, postprocessor PID **2682678**
and local collector PID **6139** have completed. All raw logits, matrix samples,
task sandbox results, source bindings and restoration records are retained.
The local archive and recomputation are in
`/private/tmp/mxwave-kda-rounding-v1-final`. Archive SHA256:
`50cd1d98769e02346b429f3fe21cbdc83412504ae204c6909b0d84438e2682a6`.

Local validation: **852 tests passed, one optional `gguf` test skipped**; Ruff
and strict mypy pass. Restoration mocks and eight new codec/capture checks pass.

Retained results:

- [Independently recomputed quality report](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/kda-rounding-v1-review.json)
- [Archive and numerical verification](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/kda-rounding-v1-independent-verification.json)
- [Subgroup, batch-shape and task breakdown](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/kda-rounding-v1-supplemental-review.json)
- [Independent real-matrix sample checks](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/kda-rounding-v1-samples-review.json)
- [Original-production restoration](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/kda-rounding-v1-restoration.json)
- [Archive receipt](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/kda-rounding-v1-archive-receipt.json)

Retained preparation evidence:

- [Frozen protocol](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/kda-rounding-v1-plan.json)
- [CPU preflight](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/kda-rounding-v1-preflight-review.json)
- [Runtime diagnostic source](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/kda-rounding-v1-source.tar.gz)
- [Frozen analyzer](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/kda-rounding-v1-analysis.py.txt)
