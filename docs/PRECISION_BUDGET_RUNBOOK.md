# Precision-budget runbook

This runbook turns an existing MxWave MXFP4 checkpoint into a bounded mixed-precision
candidate. It is an opt-in quality-recovery workflow, not part of the default quantization
path. The workflow promotes a small number of measured, fusion-safe runtime groups to
channel-wise FP8 while retaining MXFP4 everywhere else.

The important output is not merely a checkpoint. A complete run produces a frozen plan,
selection report, untouched-holdout report, deployable manifest, paired quality results,
runtime measurements, and a binary promotion decision.

## Supported architecture contract

Precision planning operates on MxWave's runtime-operation graph rather than model-name
strings. The built-in verified adapters are:

- Qwen3.5-text, including the hybrid layout used by Qwen3.8-27B;
- dense Llama.

Adapters validate every required checkpoint weight and fail closed for incomplete or unknown
layouts. A new architecture must receive a runtime adapter and synthetic adapter tests before
using this runbook. Do not bypass adapter validation by hand-selecting tensor-name suffixes.

## Inputs and invariants

Freeze these inputs before looking at candidate results:

1. the source-precision donor checkpoint and revision;
2. the completed MxWave MXFP4 baseline and manifest;
3. one token-ID context manifest shared by every model;
4. the inference image digest, linear backend, seeds, and decoding configuration;
5. selection, holdout, byte, throughput, and task non-inferiority gates.

The donor and MXFP4 checkpoint must describe the same model. The baseline must already pass
coverage, reconstruction, load, and deterministic perplexity checks. Precision budgeting is
not a repair path for a malformed baseline.

Use separate data for selection and final validation. Never add candidates after examining
the untouched holdout. If no pre-registered candidate passes, record a clean negative result
and stop.

## Reference protocol

The measured Qwen3.8-27B experiment used the following bounded protocol:

| Stage | Fixed setting |
|---|---|
| Candidate plan | 4 semantic families × 3 depth bands |
| Window width | 4 decoder layers |
| Candidate cap | 12 buckets |
| Selection | two disjoint 16-context splits |
| Combined holdout | 96 untouched contexts, checked as two halves |
| Selected bucket cap | 4 |
| Added tensor-data cap | 1 GiB |
| Runtime regression cap | 3% |
| Task non-inferiority margin | 1 percentage point |

These are safe defaults for a first experiment. Record any change before collecting candidate
outputs, and do not compare results from different protocols as if they were paired.

## 1. Create the frozen plan

Install the development environment and choose explicit paths:

```bash
pip install -e ".[dev]"

export MXFP4_MODEL=/models/model-mxwave-h64
export DENSE_DONOR=/models/model-bf16
export RUN_ROOT=/runs/model-precision-budget
mkdir -p "$RUN_ROOT"
```

Create the plan. Planning reads configuration and safetensors headers; it does not load the
full checkpoints into memory.

```bash
mxwave-precision-budget plan \
  --primary-model "$MXFP4_MODEL" \
  --dense-donor "$DENSE_DONOR" \
  --bands 3 \
  --layers-per-bucket 4 \
  --max-candidates 12 \
  --max-premium-bytes 1073741824 \
  --output "$RUN_ROOT/plan.json"
```

Review `plan.json` before continuing. It records the semantic family, layer window, runtime
groups, checkpoint modules, byte premium, and a canonical plan hash for every candidate.

## 2. Freeze contexts and baseline distributions

Prepare one immutable context manifest. The corpus and source revision must be recorded next
to the run artifacts.

```bash
python scripts/evaluate_next_token_kl.py prepare \
  --model "$DENSE_DONOR" \
  --corpus /data/frozen-evaluation-corpus.txt \
  --num-contexts 128 \
  --context-tokens 512 \
  --chunk-characters 4096 \
  --output "$RUN_ROOT/contexts.json"
```

Collect BF16 reference and MXFP4 baseline distributions with the same runtime and contexts:

```bash
python scripts/evaluate_next_token_kl.py collect \
  --model "$DENSE_DONOR" \
  --model-label bf16-reference \
  --contexts "$RUN_ROOT/contexts.json" \
  --output "$RUN_ROOT/reference-logprobs.safetensors" \
  --runtime-image vllm/vllm-openai:v0.29.0 \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.65

python scripts/evaluate_next_token_kl.py collect \
  --model "$MXFP4_MODEL" \
  --model-label mxfp4-baseline \
  --contexts "$RUN_ROOT/contexts.json" \
  --output "$RUN_ROOT/baseline-logprobs.safetensors" \
  --runtime-image vllm/vllm-openai:v0.29.0 \
  --linear-backend marlin \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.65
```

The collectors embed the context-manifest hash. The evaluator rejects mismatched contexts or
row counts rather than silently comparing different prompts.

## 3. Run the bounded screen

Run the complete screen on the GPU host. The runner composes and evaluates one bucket at a
time, removes the temporary model after each measurement, selects under the frozen byte cap,
then composes and evaluates the combined candidate.

```bash
python scripts/run_precision_budget_screen.py \
  --plan "$RUN_ROOT/plan.json" \
  --run-root "$RUN_ROOT" \
  --contexts "$RUN_ROOT/contexts.json" \
  --reference-logprobs "$RUN_ROOT/reference-logprobs.safetensors" \
  --baseline-logprobs "$RUN_ROOT/baseline-logprobs.safetensors" \
  --evaluation-script scripts/evaluate_next_token_kl.py \
  --selection-script scripts/evaluate_precision_budget.py \
  --runtime-image vllm/vllm-openai:v0.29.0 \
  --linear-backend marlin \
  --screen-contexts 32 \
  --split-size 16 \
  --max-selected-buckets 4 \
  --max-premium-bytes 1073741824
```

Long runs should be launched inside a detached session or a supervised job on the GPU host.
`state.json` is the progress record. A successful run leaves these important artifacts:

- `selection.json`: every single-bucket result and the deterministic selection;
- `combined-model/`: the deployable mixed-precision checkpoint;
- `combined-logprobs.safetensors`: frozen outputs for the combined candidate;
- `final.json`: untouched-holdout decision;
- `state.json`: terminal run state.

The runner streams primary checkpoint shards and materializes only the selected donor matrix
being converted. Composition does not require both complete models to be resident in memory.

## 4. Interpret the selection and holdout gates

A bucket is eligible only when candidate-minus-baseline forward KL is negative on both fixed
selection splits. Eligible non-overlapping buckets are ranked by their smaller-split gain per
the frozen byte budget.

The combined candidate passes the untouched gate only when:

- mean forward-KL delta is negative on both holdout halves; and
- BF16 top-1 agreement loses no more than one context versus the MXFP4 baseline.

The bootstrap interval is reported as uncertainty, not as a replacement for the predeclared
binary gate. A holdout failure ends the experiment. Do not retune the bucket set against the
holdout.

## 5. Whole-model qualification

A passed holdout admits the candidate to end-to-end qualification; it does not promote it by
itself. Serve baseline and candidate one at a time with identical settings and no unrelated
traffic. At minimum record:

1. deterministic paired perplexity over a corpus large enough to expose small changes;
2. one task tied to intended use, evaluated on identical examples and decoding;
3. deterministic instruction, structured-output, tool-use, and code smoke cases;
4. concurrency-1 and intended-concurrency throughput;
5. cold load, peak memory, request failures, restarts, and OOM events.

For task comparisons, retain per-example outputs and report paired candidate-only and
baseline-only wins. Report a paired interval and a paired test such as exact McNemar for a
binary score. Do not infer superiority from two independently rounded headline accuracies.

Use binary release gates. The Qwen reference decision required:

- no more than 1 point of task regression;
- no more than 3% throughput regression;
- zero critical deterministic-smoke regressions;
- a credible end-to-end fidelity gain.

## 6. Promotion outcomes

Use one of three terminal outcomes:

- `rejected`: a frozen gate failed;
- `qualified-optional`: quality improved or trended positively within the resource budget,
  but the tradeoff does not justify replacing the default;
- `qualified-default`: evidence is strong enough to replace the baseline for its declared use.

Keep the baseline available until the candidate passes a production soak. A candidate that is
larger or slower should normally remain optional unless its user-visible improvement is clear.

## 7. Reproducibility checklist

Before publication, record:

- source model repository and immutable revision;
- MxWave commit and adapter version;
- hashes of plan, selection, final report, configuration, manifest, and model index;
- context manifest, corpus identity, task definition, harness version, and seeds;
- runtime image name and digest, backend, context limit, and concurrency;
- exact model sizes and promoted tensor names;
- paired point estimates, intervals, failed requests, and gate decisions;
- known unsupported runtimes and untested capabilities.

The model card must distinguish a statistically credible deterministic metric from a positive
but inconclusive task trend. Link the machine-readable result instead of presenting only a
rounded table.

## Reference result

The first complete run of this protocol is documented in
[Qwen3.8-27B measured precision-budget experiment](QWEN3_8_27B_PRECISION_BUDGET.md).
Its compact records are
[`qwen3.8-27b-precision-budget.json`](../benchmarks/qwen3.8-27b-precision-budget.json)
and
[`qwen3.8-27b-precision-budget-qualification.json`](../benchmarks/qwen3.8-27b-precision-budget-qualification.json).
