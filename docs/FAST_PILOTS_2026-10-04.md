# Corrected H64: fast quality and serving pilots (2026-10-04)

This development screen reuses the corrected, unpublished `mxfp4-rne-v2`
H64 checkpoint from the [fixed controls](CORRECTED_CONTROLS_2026-10-03.md).
It adds a breakdown of retained short-context KL, native MTP off/on serving
measurements, and BF16/H64 next-token KL at 512, 2,048 and 8,192 tokens.
It does not change quantization, calibration, checkpoints, published scores,
or the Hugging Face artifact. The 89 final WikiText validation windows remain
reserved.

All three measurement jobs and the paired comparison completed successfully.
The original Ollama model container was restored and its backend responded to
`ollama list`. Successful collection does not mean the strict MTP parity gate
passed: it failed, as recorded below.

## Retained KL by prefix

The retained 512-position full-vocabulary distributions were rehashed against
their original measurement reports before analysis. No inference was needed.
The breakdown reproduces the original overall means and top-1 counts exactly.

| Prefix tokens | H64 forward KL | AMD forward KL | H64 minus AMD |
|---:|---:|---:|---:|
| 64 | 0.052572 | 0.047476 | +0.005096 |
| 128 | 0.048518 | 0.064164 | −0.015646 |
| 192 | 0.094562 | 0.116395 | −0.021833 |
| 256 | 0.034106 | 0.038515 | −0.004410 |
| 320 | 0.046565 | 0.042821 | +0.003743 |
| 384 | 0.038270 | 0.044451 | −0.006182 |
| 448 | 0.068359 | 0.073356 | −0.004997 |
| 512 | 0.055710 | 0.053443 | +0.002267 |

H64's aggregate advantage does not hold at every prefix. Most individual
H64-minus-AMD intervals include zero. Its 512-minus-64 KL change is +0.003138
nats, with nominal paired 95% interval [−0.024536, +0.032164]. This small
short-context sample establishes neither a rising length trend nor consistent
superiority at each length. The prefix breakdown is exploratory analysis of
already observed data. Intervals are not corrected for multiple comparisons.

The uncertainty unit is the original window: the same sampled window indices
are used for every model and prefix, with 20,000 bootstrap iterations and seed
20261004. The full record also includes RTN, unweighted MSE and norm fallback.

Evidence: [KL prefix analysis](../benchmarks/qwen3.8-27b-fast-pilots-2026-10-04/kl-prefix-analysis.json).

## MTP serving result

Two-token native MTP increased output throughput from 13.176 to 21.273 tok/s
on these capped, concurrency-one prompts. The median paired decode speedup
was 1.598×. The measured gain supports further serving qualification; it is
not a universal speed estimate and does not establish a DRAM bandwidth ceiling.

| Workload | Output tok/s, off | Output tok/s, MTP=2 | Paired median decode speedup | Draft acceptance | Exact token parity, requests |
|---|---:|---:|---:|---:|---:|
| All | 13.18 | 21.27 | 1.60× | 73.55% | 27/36 |
| Chat | 13.19 | 20.05 | 1.53× | 65.93% | 3/9 |
| Math | 13.14 | 22.67 | 1.82× | 82.11% | 9/9 |
| Code | 13.18 | 24.13 | 1.90× | 90.73% | 9/9 |
| Prose | 13.19 | 19.01 | 1.46× | 60.15% | 6/9 |

The timed requests proposed 5,592 draft tokens and accepted 4,113. Mean
acceptance length was 2.471 tokens per verification step, including the bonus
token. These rates exclude warmups; the archived global backend counters also
include warmups. Overall median TPOT fell from 75.65 to 47.33 ms, while median
TTFT increased from 106.58 to 212.46 ms. The speed benefit is in continued
decoding; first-token latency was worse in this profile.

### Strict parity failed

Nine of twelve distinct prompts matched exactly; the three timing repetitions
were internally deterministic in each mode. The three mismatched prompts are:

| Prompt | First unequal token, zero-based | Observation |
|---|---:|---|
| `chat-0` | 176 | The continuation names Documents/Pictures folders instead of critical folders |
| `chat-1` | 76 | The continuation changes the cache-warmup wording and later deployment advice |
| `prose-2` | 102 | The continuation begins “could be” instead of “might be” |

Both modes emitted 192 tokens and finished at the length cap on each mismatch.
This is stable divergence between modes, not variation among timing repeats.
It blocks the predeclared exact-token deployment parity gate. The decoded
inspection does not establish semantic equivalence or a quality regression.

vLLM documents that floating-point precision and batch geometry can produce
different outputs with speculation enabled. That is a possible explanation,
not a diagnosed cause for this run. The pilot has no target-logit margin trace
at the first unequal token and does not isolate the verifier or kernels.
[vLLM 0.29 lossless guarantees](https://docs.vllm.ai/en/v0.29.0/features/speculative_decoding/#lossless-guarantees-of-speculative-decoding).

The next useful serving check is a bounded investigation of those three
contexts and the target-logit differences. This result does not authorize
switching production serving or declaring MTP quality-neutral. BF16/AMD MTP,
sampling temperatures above zero and larger concurrency were not measured.

## Long-context result

| Prefix tokens | Mean H64 forward KL from BF16 | Nominal 95% interval | Top-1 agreement |
|---:|---:|---:|---:|
| 512 | 0.022531 | [0.012439, 0.033939] | 11/12 |
| 2,048 | 0.041915 | [0.028567, 0.056758] | 11/12 |
| 8,192 | 0.066802 | [0.025124, 0.123792] | 10/12 |

The observed mean rises with length. The paired 8k-minus-512 change is
+0.044270 nats, nominal 95% interval [−0.002502, +0.106557]. This screen is
inconclusive: it does not establish equivalent long-context quality, and it
does not establish a population length effect. One book contributes an 8k KL
of 0.323539, so the mean and interval are sensitive to a small sample.

Each prefix ends at a different token position in its book. A follow-up can
score several positions and add contexts ending at the same position with
different history lengths to distinguish token-position effects from length.
No additional context experiment or recipe selection was performed here.

Evidence: [paired comparison](../benchmarks/qwen3.8-27b-fast-pilots-2026-10-04/pilot-comparison.json).

## Frozen new workload

Long-context inputs are from `deepmind/pg19` validation at revision
`4d28bd77e66947ad3835cf78ed7aaeb4dd87ad8b`. Selection uses the first 12 books
in the pinned manifest with at least 9,216 tokens in the first 200,000 characters.
The first 1,024 tokens are skipped, then the next 8,192 are retained once.
Each book contributes one next-token distribution at each prefix length.

This is 12 independent books and 36 next-token positions, not a long-context
perplexity or task benchmark. Differences across lengths can reflect which
token position is scored; they do not isolate accumulated state error.
Intervals resample whole books, carrying the three prefixes together.
The book corpus differs from the WikiText controls; frozen token hashes were
checked against the earlier control inputs.

Serving inputs are 12 authored prompts: three each for chat, math, code and
prose. Qwen's chat template is applied with thinking disabled. Each request is
greedy, capped at 192 output tokens, and stops normally at EOS. There are three
repetitions, concurrency one, and a 32-token warmup per category.

Native MTP compares zero versus two draft tokens on the same corrected H64.
The acceptance rate comes from per-request backend proposed/accepted counters;
the configured flag alone is not evidence of successful speculation.
Exact emitted token sequences are compared for every prompt and repetition.
Any mismatch prevents using this pilot as a deployment parity gate.

Timing is taken from the colocated `AsyncLLM` output stream on the Spark.
TTFT is the arrival of the first nonempty token group. TPOT divides the time
from first to last token group by the number of tokens arriving after the first
group. This handles multiple tokens arriving together under MTP. Aggregate
output rate includes prefill and excludes idle time between requests; it does
not include HTTP, authentication or proxy overhead. Repeated prompts are
timing repetitions, not additional independent quality samples.

The math and code prompts exercise workload-dependent acceptance; no accuracy
score is inferred from these generations. Full GSM8K/code evaluation remains
separate.

## Runtime and identities

Runtime is the existing stock vLLM 0.29.0 image:

```text
sha256:785fd756b4ae1cacda7612cb859dbf227330e8ced9a6fdea42e9379128bebce0
```

Both models use BF16 activations and KV, chunked prefill, no prefix cache,
maximum context 8,448, one request at a time, and a 4,096-token prefill budget.
H64 uses Marlin. BF16/H64 GPU memory utilization is 0.65/0.45 respectively.
The frozen protocol records the complete engine profiles.

All actual checkpoint shards, config and index were rehashed before the run
and checked against the earlier corrected controls:

| Checkpoint | SHA-256 identity |
|---|---|
| BF16 | `3a18b4c7bcb5e34151daeec5d1747d0baffb4d28feb2b71ee653a06eb70c5147` |
| Corrected H64 | `fb093892699b24a64fdd99963d31c5f233cf54202c3fc4d257cba9ee86ce789d` |

The initial runtime configuration was rejected before loading or measurement:
request-level speculative metrics had been enabled on the MTP-off profile.
The corrected profile enables those metrics only for MTP-on. The retry retains
exactly the same text/token inputs and records the original protocol digest
as its parent; no candidate output informed the amendment.

The first retry completed the 36 H64 forward positions but failed when saving
the matrix because the output directory had not been created. The launcher
now creates it before starting collection and can resume failed containers.
This changes only launcher setup; the frozen collector, runtime profiles and
protocol digest remain unchanged. No quality values were persisted or inspected
from that failed attempt. Original run identity bytes are retained; resumption
records a separate launcher digest and parent identity hash.

| Artifact | SHA-256 |
|---|---|
| Original protocol | `328b5836a33ad5228549393a701b9a85ef432a5ea415ac37c7df6235fc5d51ca` |
| Retry protocol | `af931c85a40b7fd3068334b5b2d141e45877b64adc8880feb5b5a2dd9ef051e8` |
| Shared token sidecar | `44f6bc37df6d415d0aa8d221f958df1b7503ca289e56b0c0382f6f5730cf8e8d` |
| Retry collector source | `e9041461c80582beb9f33f60f8e4369c3d5542c07d6581ae9af2d18aff3f68eb` |

Spark directories:

```text
/home/kirya/local-spark/experiments/mxwave-fast-pilots-2026-10-04
/home/kirya/local-spark/experiments/mxwave-fast-pilots-2026-10-04-v2
```

The runner isolates each job in a container with a 30-minute timeout,
108-GiB memory ceiling, no container swap, and a 16-GiB host memory stop gate.
GPU jobs run serially. It restores the original Ollama model container after
success, exception, timeout, or a handled termination signal.

## Reproduction

Preparation and collection use
[evaluate_fast_pilots.py](../scripts/evaluate_fast_pilots.py).
The serial Spark launcher is
[run_fast_pilots.py](../scripts/run_fast_pilots.py).
The retained-data analyzer is
[analyze_kl_prefixes.py](../scripts/analyze_kl_prefixes.py).

Keep each immutable protocol beside its `protocol-tokens.safetensors` sidecar.
Its hash covers the JSON bytes; the manifest separately binds the sidecar hash,
per-book token ranges and every KL prefix. The serving prompts carry their own
token hashes. Collection refuses a changed executed-source digest.

The three raw measurement reports are archived as exact `*.json.gz` files
under `measurements/`. Their recorded SHA-256 values refer to the decompressed
JSON bytes. Compression was verified byte-for-byte, and all three hashes match
the paired comparison. Full-vocabulary matrices remain on Spark; their hashes
were verified by the comparison before computing KL. The original protocol
is retained as `validation/initial-protocol.json.gz`, with the same
uncompressed parent digest. This keeps the complete evidence without adding
tens of thousands of expanded trace lines to the repository.

The [parity diagnostics](../benchmarks/qwen3.8-27b-fast-pilots-2026-10-04/mtp-parity-diagnostics.json)
bind both raw report hashes and retain decoded mismatch examples and
within-mode repeat determinism.

```bash
python scripts/analyze_kl_prefixes.py \
  --protocol /path/to/corrected-controls/validation/protocol.json.gz \
  --measurements /path/to/corrected-controls/measurements \
  --output /path/to/new-prefix-analysis.json

python scripts/run_fast_pilots.py \
  --spark-root /home/kirya/local-spark \
  --experiment-root /path/to/frozen-pilot \
  --pause-serving-container local-spark-qwen3-8-27b-mxwave-ollama-model-1
```

For an interrupted run, add `--resume`. The launcher rehashes checkpoints and
rejects a changed protocol or runtime. It verifies the existing container's
image and command, resumes failed jobs, and reuses successful jobs; the final
comparison verifies all measurement and distribution identities.

No upload, model replacement, quantizer change, or deployment is part of this
screen. Larger serving concurrency, prefix caching, final held-out quality and
task accuracy require separate experiments.
