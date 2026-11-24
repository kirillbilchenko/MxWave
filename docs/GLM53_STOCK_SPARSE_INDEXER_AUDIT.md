# Stock vLLM sparse-indexer source and evidence audit

Snapshot: **21 October 2026, 20:28 UTC**. Read-only CPU/source review; no new GPU
experiment, kernel change, benchmark change or quality qualification.

The ordering mechanism is confirmed in the pinned source and agrees with the
retained measurements. Cutoff-tie membership changes are possible in the kernel,
but remain **unmeasured in our current stock 0.32 workload**. Related upstream
reports already exist. Do not describe this as an unreported discovery or infer
membership changes solely from the magnitude of full-model KL.

## Exact source and dispatch

The inspected image is `mxwave-glm53:vllm032-v9`, rank-0 image ID
`b65fb51e35c061f73d01e4bd5e7f07526fe1fe2848e3f0eec99e49eaf7dadb1a`.
Upstream revision: `5df63a7203cdb9328d824a27b682e77c7b9928be`.
Both Python source files below match the installed image byte for byte. CUDA source
was fetched at that revision; it was not present as source in the installed wheel.

| File | SHA256 |
| --- | --- |
| `vllm/models/glm5next/nvidia/sparse_indexer.py` | `cf659b6fdcfa1057ef6a61c3d31d302d2dd15dfea6f3c96743697664d87e3e44` |
| `vllm/model_executor/layers/indexer_topk.py` | `4dc91f83043c659304e0f0808bd9082574f919361b0b12f7c87fef5e261469bb` |
| `csrc/sampler.cu` | `2e8cf659932bf3caba373dc0affd6ef0fe4b6aee6a163aa83cebc405048a3a90` |

Long prefill calls `top_k_per_row_prefill` before expanding selected pools and
appending the incomplete tail. In `sampler.cu`, above-threshold output slots use
atomic allocation. Exact cutoff ties can also compete for the remaining slots.
Even the earlier insertion-sort branch breaks equal-score ties by temporary
buffer position, populated atomically, rather than original pool index. Short
rows return their positions in order and fill unused slots with `-1`.
[Pinned CUDA source](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/csrc/sampler.cu).

The compatibility startup log records `sparse_indexer_topk_backend='auto'`.
For GB10/SM121 and the supported selection sizes, the inspected dispatcher
resolves to **persistent**, with any lower-level fallback still needing a GPU
trace to identify. Cooperative excludes SM12x; DeepSelect is opt-in and restricted
to SM10x. Its unsorted flags are therefore not evidence about this run's decode.
An unsorted contract also does not by itself prove nondeterministic execution.
[Pinned dispatcher](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/model_executor/layers/indexer_topk.py).

## What our retained evidence establishes

| Evidence | Observation | Limit |
| --- | --- | --- |
| Stock 0.32 `stock-numerical-v2` | All four rank/cold-warm comparisons first differ at layer-3 indexer output. Last-row multisets match; 1,964–2,036 positions change order. | Whole-tensor hashes plus last-row bytes were retained, not every row's indices or internal score matrix. All-row membership cannot be recovered from those hashes. |
| Older runtime `drafter-sparse-prefill-v1` | Twelve pairs across both ranks: all 1,972,098 consumed FP32 scores per pair match; 1,921–1,922 rows change order; **zero rows change selected multiplicity**. | Different runtime, 3,973-token prompt; does not rule out ties on the current profile or longer prompts. |
| Older runtime canonical prefill/decode runs | Canonical pool ordering removed the observed prefill drift, then the bounded combined policy passed the six-request C1 repeat fixture. | Does not qualify stock 0.32, C8/C16 or long-context serving. |

The current cold/cold KL is **0.03233036**, warm/warm **0.02497079**, both failed
against the unchanged `1e-5` gate. This localizes a numerical problem but does
not identify membership changes. Equal outer indexer inputs are weaker evidence
than comparing the consumed internal scores. FP8 keys likewise do not establish
the frequency of exact ties in the accumulated FP32 scores; measure that frequency.

Local reviews:

- `out/glm53-package-v1/stock-numerical-v2-independent-review.json`
- [Older all-row review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/drafter-sparse-prefill-v1-review.json)
- [Canonical decode review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/drafter-canonical-decode-v1-review.json)

## Existing upstream work

[PR #55122](https://github.com/vllm-project/vllm/pull/55122) is open and proposes
deterministic persistent top-k. Its discussion also covers per-row decode.
A [22 September field report](https://github.com/vllm-project/vllm/pull/55122#issuecomment-5769499616)
already describes real GLM-5.3 Flash cutoff ties on TP2 GB10, plus prefill and
decode membership reproducers. These are another contributor's measurements,
not validation of our current workload. The PR modifies persistent top-k; it
does not automatically repair GLM's separate prefill selector. Check this thread
before filing a new report, and contribute distinct evidence rather than a duplicate.

[PR #50365](https://github.com/vllm-project/vllm/pull/50365), merged, changes
sparse index remapping/compaction, not the selection kernel discussed here.
[Draft PR #57815](https://github.com/vllm-project/vllm/pull/57815) explicitly
addresses unsupported batch-invariance guarantees in sparse-indexer dispatch.
The inspected GLM file and shared dispatcher do not implement a deterministic
flag branch. `VLLM_BATCH_INVARIANT=1` is not a verified remedy for this profile;
no flag-on GPU run was made in this audit.

## Bounded follow-up, after the frozen comparison

1. Capture the actual backend, consumed scores and valid row bounds at the first
   differing indexer call. Replay identical inputs. Compare ordered indices and
   sorted **multisets** for every active row, preserving duplicate counts, padding,
   pool expansion and causal bounds. Exclude unconsumed score storage.
2. Count the valid scores above and equal to each row's exact kth cutoff. A
   membership-ambiguous cutoff requires more equal entries than remaining slots.
   Save full data for the first changed set and verify exchanged candidates are
   tied; do not assume any membership difference is a tie rather than a bug.
3. Separate native selection, canonical ordering of the already-selected pools,
   and deterministic selection by score then pool position. Sorting selected
   indices cannot recover tied candidates already discarded. Do not perturb
   scores with an epsilon or use ordinary `torch.topk` as a stable-tie oracle.
4. Only after component verification, test full-model repeat KL and serving
   overhead with both prefill and decode covered, then mixed batches and graphs.
   Synthetic all-equal/cutoff-tied, unique-score and heavily padded cases belong
   in the component check. No performance benefit or low cost is assumed.

Source receipts and the upstream comment are retained under
`out/glm53-package-v1/sparse-source-audit-v1/`. No third-party implementation was
adapted. The frozen `deployment-compare-v2` continues with its existing images,
inputs and gates. At **20:28 UTC**, A0 has completed **5/24 timing cells** and is
running `code-long-c4`; both full GSM8K arms remain scheduled after timing.
