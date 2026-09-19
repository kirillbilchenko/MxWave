# Qwen3.8-27B measured precision-budget experiment

## Outcome

This is the first bounded MxWave precision-allocation experiment that passed its
predeclared selection and untouched-holdout gates. Starting from the H64 MXFP4
checkpoint, it promotes only 12 of 400 quantized tensors to channel-wise FP8:

- four early sequence-output projections in layers 8-11;
- eight late MLP input projections in layers 51-54.

The result is a valid quality/size Pareto point and has completed its full
qualification. It improves full-corpus perplexity by a small but statistically
credible amount, scored 0.455 strict accuracy points above H64 on the paired full
GSM8K run, adds 375.6 MiB, and reduced throughput by 2.17-2.28% on the final
vLLM 0.29.0 runtime. The GSM8K interval crosses zero, so the checkpoint is
qualified as an optional quality-oriented variant rather than a replacement for
the published H64 default.

| Checkpoint | Directory bytes | Precision mix | WikiText-2 PPL |
|---|---:|---|---:|
| BF16 reference | - | BF16 | 7.980864117 |
| MxWave H64 | 19,856,330,600 | 400 MXFP4 tensors | 8.119444612 |
| Precision-budget candidate | 20,250,144,780 | 388 MXFP4 + 12 FP8 tensors | **8.111578398** |
| AMD Quark-AWQ MXFP4 | - | MXFP4 | 8.187543970 |

On the paired 316-window corpus, the candidate was:

- 0.096881% lower PPL than H64, with a 100,000-sample paired bootstrap
  interval of `[-0.129653%, -0.063734%]`;
- 0.927819% lower PPL than AMD Quark-AWQ MXFP4;
- 1.637846% higher PPL than BF16.

The PPL effect is real but small. Whether that trade is useful depends on the
task-level evaluations below, not PPL alone.

## Fixed selection protocol

The screen was registered before candidate evaluation:

- 12 candidates: four semantic tensor families across early, middle, and late
  four-layer windows;
- two disjoint 16-context selection splits;
- eligibility required lower mean forward KL than H64 on both splits;
- at most four buckets and 1 GiB of additional tensor data;
- a final, untouched 96-context gate split into two 48-context halves;
- final promotion required lower mean KL on both halves and no more than one
  lost BF16 top-1 agreement.

Only two candidates were eligible:

| Selected bucket | FP8 tensors | Premium bytes | Selection KL delta vs H64 |
|---|---:|---:|---:|
| `sequence-output-early-l08-11` | 4 | 59,064,320 | -0.000997133 nats |
| `mlp-input-late-l51-54` | 8 | 334,790,656 | -0.001760091 nats |

The combined premium is 393,854,976 tensor bytes (375.6 MiB). The final
checkpoint is 20,226,686,160 manifest-accounted bytes with a 2.747x global
compression ratio; its full directory is 20,250,144,780 bytes.

## Untouched forward-KL gate

The combined candidate passed the untouched 96-context gate:

| Metric | H64 | Candidate |
|---|---:|---:|
| Mean forward KL to BF16 | 0.047117476 | **0.044941898** |
| BF16 top-1 agreement | 89/96 | 89/96 |

The candidate-minus-H64 mean KL was `-0.002175578` nats: `-0.000789743`
on the first half and `-0.003561412` on the second. Context wins were tied
48 to 48. The paired bootstrap interval `[-0.005939469, +0.000523011]`
crossed zero, so the holdout establishes the predeclared directional gate but
does not independently establish a non-zero mean effect. The larger 316-window
PPL comparison provides the stronger statistical evidence.

## Controlled serving results

The fixed serving workload used random 512-token inputs, exactly 128 generated
tokens, ignored EOS, seed `20260919`, temperature zero, and no prefix caching.
Concurrency 1 used 16 prompts; concurrency 4 used 32 prompts. Every run
completed with zero failures.

### Original day-zero runtime

Runtime image:
`vllm/vllm-openai:qwen38-flash-next@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8`

Installed stack: vLLM `0.1.dev20073+g8e685d198`, Torch `2.13.0+cu130`, and
FlashInfer `0.6.17`.

| Model | Concurrency | Output tok/s | Median TTFT | Median TPOT |
|---|---:|---:|---:|---:|
| H64 | 1 | 12.523648 | 529.65 ms | 76.30 ms |
| Candidate | 1 | 12.206125 | 543.69 ms | 78.27 ms |
| H64 | 4 | 23.985347 | 11,550.11 ms | 77.15 ms |
| Candidate | 4 | 23.376993 | 11,853.98 ms | 79.20 ms |

The candidate was 2.54% slower than H64 at both tested concurrency levels.
This is consistent with the GB10 runtime using weight-only FP8 Marlin fallback
for the promoted tensors rather than a native FP8 path.

### Official vLLM 0.29.0 runtime

The candidate also passed an isolated test with the official arm64 image:

```text
vllm/vllm-openai@sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1
```

Installed stack: vLLM `0.29.0`, Torch `2.13.0+cu130`, FlashInfer `0.6.18`,
Transformers `5.16.1`, and compressed-tensors `0.17.0`.

The fresh-cache cold start loaded weights in 139.04 seconds, used 18.5 GiB for
the model, spent 23.28 seconds in `torch.compile`, and spent 144.88 seconds in
engine profiling, cache creation, and warmup. The API became healthy after
approximately 5.5 minutes.

The old `VLLM_TEST_FORCE_FP8_MARLIN=1` compatibility switch has been removed
from vLLM 0.29.0 and must not be used. The runtime selected MXFP4 Marlin and the
FP8 Marlin fallback automatically.

Quality was stable across the runtime change. On the identical first 64
WikiText windows, token hashes and scored-token counts matched exactly:

| Runtime | PPL64 |
|---|---:|
| Day-zero build | 8.161232897 |
| vLLM 0.29.0 | 8.160493801 |

The relative PPL delta was `-0.009056%`, which is operationally negligible.

| vLLM 0.29.0 run | Output tok/s | Median TTFT | Median TPOT |
|---|---:|---:|---:|
| Concurrency 1 | 12.212793 | 545.88 ms | 78.23 ms |
| Concurrency 4, run 1 | 41.229636 | 2,176.36 ms | 80.67 ms |
| Concurrency 4, repeat | 41.250015 | 2,175.20 ms | 80.60 ms |

Single-stream speed was unchanged (`+0.055%`). Concurrency-4 throughput was
76.37-76.46% higher than the day-zero runtime and reproduced within 0.05%; its
median TTFT was 81.64% lower. Per-token decode latency was 1.86% higher, so the
large aggregate gain comes from substantially better concurrent scheduling and
batch execution, not faster single-stream decoding.

The final side-by-side qualification used the same official vLLM 0.29.0 image.
H64 produced 12.484 output tokens/s at concurrency 1 and 42.193 at concurrency
4. The candidate produced 12.213 at concurrency 1 and 41.230-41.250 at
concurrency 4: regressions of 2.17% and 2.24-2.28%, inside the predeclared 3%
budget.

## Full deterministic qualification

The final task gate used all 1,319 GSM8K test examples with the same custom
`gsm8k_nothink` definition, five-shot prompts, greedy decoding, temperature zero,
1,024 maximum generation tokens, and four concurrent requests. Both checkpoints
ran with MTP disabled, prefix caching disabled, and the same vLLM 0.29.0 image.

| Metric | H64 | Candidate | Candidate - H64 |
|---|---:|---:|---:|
| Strict match | 1,208/1,319 (91.5845%) | **1,214/1,319 (92.0394%)** | **+0.4549 points** |
| Flexible extract | 1,208/1,319 (91.5845%) | **1,215/1,319 (92.1152%)** | **+0.5307 points** |

On strict scoring, the candidate alone was correct on 25 examples and H64 alone
on 19. Exact McNemar was `p=0.4514`; the paired 100,000-resample interval was
`[-0.5307, +1.4405]` accuracy points. On flexible scoring, the corresponding
counts were 23 and 16, `p=0.3368`, with interval
`[-0.3791, +1.4405]`. Both directions favor the candidate, but neither task
difference is statistically conclusive.

The deterministic serving smoke covered short completion, strict JSON, tool use,
and Python code. All 12 repeated checks passed with no transport failures. The
candidate therefore passed the task non-inferiority, runtime, and critical-smoke
gates. Combined with the paired perplexity result, this supports publication as
an optional quality-oriented checkpoint. It does not support claiming a
statistically proven GSM8K improvement or replacing H64 as the smaller, faster
default.

## External context as of 2026-09-19

Mixed precision is an established and active direction, so the broad idea is
not unique:

- [NVIDIA Qwen3.8-27B NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4)
  uses FP8 attention plus NVFP4 MLP and LM-head tensors, calibrated with 2,048
  samples and Local Hessian.
- [Unsloth Qwen3.8-27B NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)
  is another broad FP8-attention/NVFP4-MLP PTQ checkpoint.
- [QUASAR Qwen3.8-27B NVFP4](https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4)
  uses quantization-aware recovery training rather than pure PTQ.
- [AMD Quark-AWQ MXFP4](https://huggingface.co/amd/Qwen3.8-27B-Quark-AWQ-MXFP4)
  is the all-MXFP4 PTQ baseline used in our paired comparison.

MxWave's differentiator is narrower: allocate a strict byte budget using
measured end-to-end sensitivity, require agreement across disjoint selection
splits, and verify the combined allocation on untouched data while emitting a
standard vLLM compressed-tensors checkpoint. The candidate promotes only 12
tensors rather than assigning an entire architectural family to FP8.

A recent independent NVFP4 comparison reports WikiText PPL of 8.131 for
Unsloth and 8.139 for NVIDIA, versus 7.993 BF16. Our 8.1116 result is promising,
but those values use a different 323-window serving protocol and must not be
treated as a direct ranking. A same-harness comparison is required.

## Remaining work

1. Publish the candidate with an explicit optional/experimental label, the exact
   tradeoffs above, and a link to the frozen MxWave implementation.
2. Run a code benchmark or long-context qualification only if the model card will
   make claims about those capabilities; neither is required for the measured
   perplexity and GSM8K claims recorded here.
3. Validate the same frozen method on one supported non-Qwen architecture in a
   separate bounded experiment. That is the next test of generality; more Qwen
   bucket tuning would reuse the holdout and is not justified.
4. Use a same-harness public mixed-precision comparison before making a ranking
   claim against NVFP4 or QAT releases.

## Reproducibility

Original experiment snapshot:
`158668ccba0678beff2e91da484f8604e1ce702c`. The cleaned, baseline-agnostic
reproduction implementation is commit
`4d5b585981422cf72b60d7464b49151b28280a32`; it excludes the unsuccessful
operator-response and teacher-KL probes from the research branch.

Spark run root:

```text
$HOME/local-spark/experiments/precision-budget-screen-158668c
```

Important hashes:

```text
d408658ed8bfd9384975a11288da84594ce5efbcd10abaf543a1c6310a83af7a  plan.json
8a073f5768fa96eff7c3554ef82b87745a84b09e0d0e02e35f312d401d2b5b88  selection.json
f89f5e98afbaaec5004108e773cf0c62be61322c2f71753104dce3466333c86f  final.json
ed3ed368221813a1c5bc9ab7dcc150506e2d254172446f1f493bba6923983263  combined-model/mxwave-manifest.json
d636d05374fc411e8558268a73d5818b1496280a21f3f027d0e1ee6431d3b050  precision-budget-158668c-4k316.json
76675a6cd67294a7bac7368e479a6a0382d16b745a2242123212dae014b99779  precision-budget-v029-4k64.json
```

The compact machine-readable result is
`benchmarks/qwen3.8-27b-precision-budget.json`. Large checkpoint and logprob
artifacts remain on Spark and are not committed. The end-to-end serving and task
qualification is recorded in
`benchmarks/qwen3.8-27b-precision-budget-qualification.json`.

Final qualification hashes:

```text
aa4fab87c4fe98b36238f5e32e4115b43d1dbdc52343c652500725bbf2ed9c26  paired-full-1319.json
d4bcabbabfe3367b96b22fb076509c9b2f2bd5aa58de15d2a47353be5f726830  h64-full-results.json
0262f92cd7453bee42c047c95f095f39db3ebf6e636108a1ce56a854d49e2c91  candidate-full-results.json
22eb587ca1347657958719f3b0a17a4b162180c79ee35c6e080407fd7eb6a942  h64-full-samples.jsonl
d8acd840f91d82b54e2b2e0efc06fb1e6785edba4d644bb7735643a3fa373d63  candidate-full-samples.jsonl
```

The model-independent operational procedure is in the
[precision-budget runbook](PRECISION_BUDGET_RUNBOOK.md).
