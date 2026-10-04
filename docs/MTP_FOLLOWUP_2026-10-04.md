# Corrected H64: MTP diagnostics and matched-target context lengths

Status: complete. All six GPU jobs and the CPU comparison exited successfully.
The experiment containers have stopped, and the original Ollama service is restored.

This follows the [fast pilot](FAST_PILOTS_2026-10-04.md), which measured a
1.60x median paired decode speedup with native two-token MTP but failed exact
token parity on three of twelve prompts. The corrected, unpublished H64 and
the BF16 reference checkpoints are unchanged. This follow-up investigates
those differences, adds a fresh scored task sample and compares context
lengths ending at the same target token. It does not enable production MTP.

## Results and decision

- The three original graph-profile output differences reproduce exactly. At
  each first divergence, MTP changes the target scores from a 0.125 logit
  preference to an exact tie. Both selected tokens are target argmaxes. No
  non-argmax selection appears in 4,608 traced tokens across all four profiles.
  This supports a numerical/execution-path explanation; the specific kernel
  cause remains unestablished. Eager execution also fails exact parity.
- All 32 paired fresh task requests have identical output tokens with MTP off
  and on. Corrected scoring gives 8/8 Python tasks and 1/8 arithmetic tasks
  in both modes. These small, thinking-disabled tasks do not replace GSM8K or
  demonstrate general quality equivalence.
- Code responses have a 1.87× median paired total speedup. Short arithmetic
  responses are slower with MTP, and TTFT increases in both categories.
- On 24 new books with matched targets, mean KL is 0.037718, 0.037700 and
  0.036664 at 512/2k/8k. The paired 8k-minus-512 interval includes zero. This
  sample provides no evidence of increasing quantization divergence with
  history length; it does not establish long-context task quality.

MTP remains a promising option for longer responses. The exact-parity gate
still fails, so these results do not qualify it for the existing production
configuration. No quantization recipe or checkpoint was changed.

## MTP diagnostics

The three reused prompts (`chat-0`, `chat-1`, `prose-2`) each run twice under
four configurations: MTP off/on with the prior CUDA-graph profile, and MTP
off/on with `enforce_eager=True`. The latter disables compilation as well as
CUDA graphs; it does not isolate graph replay alone.

Each generation returns the target model's top-eight raw log probabilities
at every emitted token. At the first mismatch the histories are identical,
so comparing the two selected tokens' probability margins can identify a
rank flip. The difference between two log probabilities equals the difference
between their logits. Every traced selected token is checked against the
returned top probabilities. This can detect a selected token that is not
the target argmax; it does not prove correctness for all sampler paths.

The report also records whether instrumentation reproduces each original
sequence. A separate one-token request uses the original common history as
its input in every profile. That request rebuilds the history during prefill;
it is a diagnostic of a different execution path, not a replacement for the
sequential trace at the actual first divergence.

vLLM describes floating-point precision and batching as possible causes of
differences under speculation. The experiment does not assume either is the
cause here. [vLLM 0.29 lossless guarantees](https://docs.vllm.ai/en/v0.29.0/features/speculative_decoding/#lossless-guarantees-of-speculative-decoding).

### Observations

The graph traces reproduce every original mode-specific output, 6/6 for each
mode. Both repetitions give the following first differences; indices are
zero-based:

| Prompt | Graphs: first differing token | Eager: first differing token | Graph target margin, off → on |
|---|---:|---:|---:|
| `chat-0` | 176 | 55 | 0.125 → 0.0 |
| `chat-1` | 76 | exact parity | 0.125 → 0.0 |
| `prose-2` | 102 | 45 | 0.125 → 0.0 |

The margin compares the two tokens selected by the respective modes at their
shared history. With MTP on, both have the same returned target log probability.
Across 1,152 traced tokens per profile, all selected tokens equal a returned
target maximum. This is evidence of differing target scores/ties, with no
observed greedy-selection error; it does not prove sampler correctness beyond
these traces or identify which numerical operation changed the scores.

Eager changes the generations and still diverges on two prompts. It is not an
exact-parity fix. The one-token requests rebuilt from each original common
history do agree between MTP off/on within each profile, although graphs and
eager do not always agree with each other. Rebuilding the state during prefill
therefore does not reproduce or isolate the sequential divergence mechanism.

## Fresh task and latency screen

The protocol contains eight authored arithmetic tasks and eight Python
function tasks. Answers and unit cases were fixed before inference. These
are 16 diagnostic tasks, not a public benchmark or an estimate of population
accuracy. They are separate from the earlier unscored serving prompts.

Arithmetic requests return only a number or reduced fraction; scoring checks
the exact stripped answer. Python requests specify a function contract and
input immutability. Scoring checks all frozen cases, including empty inputs,
duplicates, negative values and boundary cases where appropriate. Fenced
Python is accepted by the grader. Only function definitions without imports,
decorators or private attribute access are supported. Code runs in a separate
process with a two-second CPU limit, four-second wall timeout and, on Linux,
256-MiB address-space limit. The enclosing comparison container has no network
or model mount. This is a bounded test runner, not a general code benchmark.

Both modes use the normal graph profile, greedy decoding, thinking disabled,
normal EOS, at most 384 output tokens and concurrency one. Two repetitions
measure timing and within-mode consistency; they do not create 32 independent
quality samples. The arithmetic requests give a short-output latency check,
while the code requests exercise longer decoding.

Task timing has no logprob collection. TTFT is the first nonempty token group
in the colocated AsyncLLM stream; total latency ends at the last token group.
TPOT accounts for grouped MTP outputs and is omitted for a single token group.
These times exclude HTTP, proxy and network latency. Diagnostic trace timings
are not used as serving results because probability collection adds work.

### Observations

All 32 output-token pairs match. Scores count eight distinct tasks in each
category; parity and timing count their two repetitions:

| Task sample | Pass count, off / on | Exact token pairs | Median TTFT, off → on | Median total, off → on | Median paired total speedup |
|---|---:|---:|---:|---:|---:|
| Python functions | 8/8 / 8/8 | 16/16 | 120.10 → 182.60 ms | 4.9018 → 2.6229 s | 1.8725× |
| Arithmetic, final answer only | 1/8 / 1/8 | 16/16 | 119.94 → 205.58 ms | 0.3504 → 0.4311 s | 0.8128× |

Median paired decode speedup is 1.9612× on code and 1.0077× on arithmetic.
Draft acceptance is 710/736 (96.47%) on code and 38/64 (59.38%) on arithmetic.
The pooled sample's paired total speedup is 1.4492×, but it depends on this
particular mixture of short and long answers. Speedups are medians of paired
request ratios, not ratios of the reported median latencies. Two repetitions
and concurrency one do not establish production latency distributions.

The poor arithmetic result applies to thinking-disabled, final-answer-only
generation. Both modes return the same wrong answers, and no BF16 task
baseline was run. It cannot be attributed to quantization or MTP from this
experiment. The code score likewise covers only these authored contracts.

### Scorer correction

The frozen scorer omitted the standard builtin `isinstance`. The generated
solution for `fresh-code-4` used it correctly, but the isolated worker raised
`NameError`, yielding an original code score of 7/8 in each mode. The exact
original [comparison](../benchmarks/qwen3.8-27b-mtp-followup-2026-10-04/comparison.json)
is preserved with that result; it has SHA-256
`fdb062fd54e2c3308377133386560075019d3ab3cef4fc48d62139d6729a3386`.

Commit `9c21b85` adds that builtin and a regression test. A separate CPU-only
[rescore](../scripts/rescore_mtp_followup_tasks.py) applies the correction to
the exact archived generations. It changes no prompts, answers, unit cases,
timings or acceptance counters and performs no inference. The
[corrected scores](../benchmarks/qwen3.8-27b-mtp-followup-2026-10-04/corrected-task-scores.json)
bind the original comparison, protocol and raw-report hashes, the frozen
collection source, the updated scoring source, and Python 3.14.6. The corrected
code score is 8/8 in both modes. Original collection/comparison ran on Spark
with the unmodified frozen source and Python 3.12; the rescore ran locally.

## Matched-target long context

Selection uses the pinned PG-19 validation manifest and the first 24 eligible
books after excluding all twelve books in the pilot. Three further books were
too short for the required positions and are recorded in the protocol.

Each book contributes two source-token positions, 12,288 and 16,384. At each
position the same next token is predicted from the preceding 512, 2,048 and
8,192 tokens. There are 48 distinct targets and 144 distributions per model.
The sidecar stores one token vector per book; contexts refer to its suffix
ranges instead of repeating token IDs.

The BF16/H64 comparison uses MTP off, with the same BF16 activations and KV
settings as the prior controls. Bootstrap uncertainty resamples whole books,
averaging the two target positions within a book before resampling, with
20,000 iterations and seed 20261004. Intervals are nominal 95% and unadjusted.

Using the same targets removes the prior token-position confound. Shorter
contexts also omit information, so the comparison cannot isolate recurrent
state accumulation from all other effects of history length. Two positions
per book are still sparse, and this is neither long-context perplexity nor
task accuracy. The 89 reserved final WikiText windows remain untouched.

### Observations

| Preceding tokens | Mean forward KL from BF16 | Nominal book-bootstrap 95% interval | Top-1 agreement / 48 |
|---|---:|---:|---:|
| 512 | 0.037718 | [0.025897, 0.054848] | 43 |
| 2,048 | 0.037700 | [0.027462, 0.050641] | 43 |
| 8,192 | 0.036664 | [0.026030, 0.049385] | 44 |

Paired 8k-minus-512 mean KL is **−0.001053 nats**, with nominal 95% interval
**[−0.009941, +0.007150]**. Thus this screen does not confirm the upward trend
seen in the first pilot, which used different token positions at each length.
That pilot and this follow-up use different books and sampling designs, so
their absolute means are not directly comparable. There are only two target
positions per book; the result neither proves absence of long-context error
nor measures long-context perplexity or task success.

## Frozen provenance and execution

| Artifact | SHA-256 |
|---|---|
| Executed source | `90fcda9bef91a1d3f508a14035dc799de896a1826a993a468f0f8b1689855e1f` |
| Initial protocol | `f103f7de85130fc8e74e123f8e10b8719725e3251f2141b33fd4ba88f8426754` |
| Clean retry protocol | `75f70a2470f7a372108e79be2f54d5f084d3df7eeffd364ab36cd8a3bdf2de8c` |
| Shared token sidecar | `5fca1a5b68e2b6506c225d64ff564c0c5debff8bb584bce9b072526baccd6869` |

The first upload added macOS AppleDouble metadata files. Those files affected
the source inventory hash, although every actual Python source byte matched
local. Preflight caught this before model inference. A clean retry uses
`tar --no-mac-metadata --no-xattrs`, preserves the first frozen protocol as
its parent and reuses exactly the same token bytes, tasks and engine profiles.
The original attempt's source inventory hash is
`2a390f10a6d0321a8fa791a3a93f75e09fd817d901f6ad3c0f9c0b1116c79a15`.

All four H64 configurations passed startup validation before loading weights.
The actual checkpoint shards, config and index are rehashed before measurement
and checked against the corrected controls. The stock vLLM 0.29.0 image is
`sha256:785fd756b4ae1cacda7612cb859dbf227330e8ced9a6fdea42e9379128bebce0`.

Spark roots:

```text
/home/kirya/local-spark/experiments/mxwave-mtp-followup-2026-10-04
/home/kirya/local-spark/experiments/mxwave-mtp-followup-2026-10-04-v2
```

The detached runner (`runner.pid`, `runner.log`) runs six GPU jobs serially,
then a CPU comparison. Each GPU job has a 108-GiB no-swap limit, 30-minute
timeout and a 16-GiB host available-memory floor. The runner uses an exclusive
GPU-work lock and restores the original Ollama model container in `finally`.
Frozen validation and Python runtime mounts are read-only during inference.

GPU jobs ran from 15:43:31 to 16:32:33 UTC on 2026-10-04; the CPU comparison
finished at 16:32:41 UTC. All seven jobs exited 0 without OOM. At 16:40:25 UTC,
there were no running follow-up containers, both original Ollama model/proxy
containers were running, and `ollama list` returned successfully. No further
GPU runs are scheduled for this experiment. Local Ruff, strict mypy and all
265 tests pass.

Source: [collector](../scripts/evaluate_mtp_followup.py),
[runner](../scripts/run_mtp_followup.py), [task scorer](../scripts/mtp_task_screen.py).
Frozen inputs: [protocol](../benchmarks/qwen3.8-27b-mtp-followup-2026-10-04/validation/protocol.json).
The initial protocol is retained as exact gzip; its hash covers uncompressed
JSON bytes. Full-vocabulary matrices remain on Spark.

The six raw reports are retained under
[measurements](../benchmarks/qwen3.8-27b-mtp-followup-2026-10-04/measurements)
as exact gzip archives; every uncompressed SHA-256 was checked against the
original comparison before archiving. The
[run identity](../benchmarks/qwen3.8-27b-mtp-followup-2026-10-04/run-identity.json)
also records the source, checkpoint and image identities. Large probability
matrices are not committed; their hashes are in the retained measurement
reports and were verified by the Spark comparison.

The frozen collection source is commit `5b3333c`, whose source digest matches
the protocol. Current scoring includes the separate `9c21b85` correction;
using it to collect again would produce a different source identity. To replay
only corrected scoring on the retained reports from the current branch:

```bash
python scripts/rescore_mtp_followup_tasks.py \
  --protocol benchmarks/qwen3.8-27b-mtp-followup-2026-10-04/validation/protocol.json \
  --measurements benchmarks/qwen3.8-27b-mtp-followup-2026-10-04/measurements \
  --original-comparison benchmarks/qwen3.8-27b-mtp-followup-2026-10-04/comparison.json \
  --output /tmp/mxwave-mtp-task-rescore.json
```

The rescorer refuses to overwrite an existing output. Its provenance records
the scoring interpreter and source separately from the frozen collection.
