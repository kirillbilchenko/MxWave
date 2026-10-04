# Corrected H64: MTP diagnostics and matched-target context lengths

Status: the frozen Spark experiment is running. Results are pending.

This follows the [fast pilot](FAST_PILOTS_2026-10-04.md), which measured a
1.60x median paired decode speedup with native two-token MTP but failed exact
token parity on three of twelve prompts. The corrected, unpublished H64 and
the BF16 reference checkpoints are unchanged. This follow-up investigates
those differences, adds a fresh scored task sample and compares context
lengths ending at the same target token. It does not enable production MTP.

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

Source: [collector](../scripts/evaluate_mtp_followup.py),
[runner](../scripts/run_mtp_followup.py), [task scorer](../scripts/mtp_task_screen.py).
Frozen inputs: [protocol](../benchmarks/qwen3.8-27b-mtp-followup-2026-10-04/validation/protocol.json).
The initial protocol is retained as exact gzip; its hash covers uncompressed
JSON bytes. Full-vocabulary matrices remain on Spark.
