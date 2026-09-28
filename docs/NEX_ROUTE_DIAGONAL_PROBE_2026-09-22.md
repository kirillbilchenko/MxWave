# Nex route-aware diagonal-RMS probe — bounded pre-registration (2026-09-22)

Status: pre-registered; no real-model result has been inspected and no result is
claimed in this document.

## Product question

Can route-conditioned activation statistics improve the BF16-teacher fidelity
of Nex-N2.5-mini's standard MXFP4 routed experts without changing checkpoint
size, checkpoint format, stock-vLLM kernels, or serving precision?

This is a quality-only experiment. It cannot make the model smaller or faster.
Its purpose is to test whether MxWave's generic activation-weighted scale search
becomes useful for sparse experts when the model adapter supplies the right
conditional inputs.

## Motivation and one hypothesis

The raw RTN expert checkpoint is the baseline. The later unweighted-MSE
checkpoint improved paired perplexity but made mean terminal
`KL(BF16 || candidate)` 10.01% worse than RTN, with both fixed halves moving in
the wrong direction. That is evidence of objective mismatch, not permission to
sweep more unweighted scale-search settings.

The frozen hypothesis is narrower:

> For a routed expert, weighting each input channel by its route-conditioned
> second moment will choose better legal MXFP4 block scales than RTN.

For expert `e`, routed token `t`, normalized router coefficient `r_t,e`, and
actual expert input `x_t`, define `a_t,e = r_t,e^2` and

```text
d_e,j = sum_t a_t,e * x_t,j^2 / sum_t a_t,e
```

Scale selection minimizes

```text
sum_rows sum_j d_e,j * (W_q[row,j] - W[row,j])^2
```

over the existing legal MXFP4 exponent candidates. In MxWave's existing
`gamma` interface, `gamma_e,j = sqrt(d_e,j)`, so `gamma^2` supplies the diagonal
weight.

This is **route-aware diagonal RMS**, not a block Hessian, GPTQ, or an
end-to-end objective. It retains no off-diagonal channel correlations, performs
no inverse-Hessian error compensation, and does not optimize teacher KL during
quantization. Teacher KL is held out and used only to accept or reject the
candidate.

The reusable part is model-independent: weighted second-moment accumulation,
effective-sample-size checks, diagonal weighted scale selection, and RTN
fallback. The Qwen3.5-MoE adapter is responsible only for exposing route IDs,
normalized route weights, and the real operator inputs. A positive Nex result
would therefore validate one adapter plus a universal core; it would not by
itself establish architecture-general MoE support.

## Frozen identities

| Item | Identity |
|---|---|
| BF16 source | `nex-agi/Nex-N2.5-mini` |
| Source revision | `87420286149d9cce9bd46cd335ef9bda33c37c1b` |
| Baseline | MxWave routed-expert RTN |
| RTN checkpoint SHA-256 | `eadbf27940485f0c9200e3189a8db03dee0a5380f274a80d182a63ca44abdcd5` |
| RTN config SHA-256 | `8ea02e392011595c2394a26862e51966608abab24fa2b750ce798b4ce67f6351` |
| RTN index SHA-256 | `69d4da6c13645e868454a7e22537e3f2ceef8883cf63817d67e2239145ad293c` |
| RTN manifest SHA-256 | `91a1a1c5ead3ba7a42fa0800aaa559b960010650bb8bd2010203f1c30fd2fb6c` |
| Candidate format | unchanged `mxfp4-pack-quantized`, weight-only W4A16 |
| Runtime | official `vllm/vllm-openai:v0.29.0` |
| Runtime digest | `sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1` |
| Hardware | DGX Spark GB10 / SM121 |
| Bootstrap | 100,000 paired resamples, seed `20260922` |

Before the first statistics pass, the run record must bind the exact calibration
corpus bytes, tokenizer assets, generated token IDs, implementation commit, and
container digest. A hash mismatch fails closed. Paths are not identities.

## Frozen scope

- Decoder layers: `0`, `13`, `26`, and `39` only.
- Source banks changed per layer: routed `gate_up_proj` and `down_proj` only;
  eight source banks in total.
- Logical matrices: every eligible routed expert's gate, up, and down matrix in
  those four layers.
- All other routed-expert matrices remain byte-identical to RTN.
- Routers, shared experts, attention/GDN, embeddings, norms, head, vision, and
  configuration assets remain byte-identical to the RTN checkpoint.
- Scale search remains percentile `99.5`, clip depth `4`, including the fixed
  no-clipping candidate. There is no percentile, depth, layer, or objective
  sweep.
- There are exactly two calibration replicas and therefore exactly two
  candidate patches.

Gate and up use the hidden rows actually routed to the expert. Down uses the
actual BF16 post-activation product consumed by that expert's down projection:

```text
z = activation(gate(x)) * up(x)
```

It must not reuse the layer input statistic or an approximate shared-expert
activation. Contributions use the squared **normalized** top-k router
coefficient. The collector records raw routed-occurrence counts as diagnostics,
but counts do not replace route weights.

## Calibration replicas, coverage, and fallback

Both replicas use deterministic 512-token sequences from the same pinned
calibration corpus:

| Replica | Initial sequences | Conditional one-time extension |
|---|---:|---:|
| A | offset `0`, `16 x 512` | offset `0`, `32 x 512` total |
| B | offset `32`, `16 x 512` | offset `32`, `32 x 512` total |

The ranges remain disjoint even after extension. For `a_t,e = r_t,e^2`, the
per-expert effective sample size is

```text
ESS_e = (sum_t a_t,e)^2 / sum_t a_t,e^2
```

An expert is eligible only when `ESS_e >= 64` and its gate/up and down
statistics are finite with the expected dimensions. Otherwise gate, up, and
down all fall back to exact RTN for that expert; partial eligibility is not
allowed.

For each selected layer, define eligible route energy as

```text
sum_{eligible e,t} r_t,e^2 / sum_{all e,t} r_t,e^2
```

If any selected layer is below 95% eligible route energy after the initial 16
sequences, extend that entire replica once to 32 sequences. Do not extend only a
favorable layer or expert. After 32 sequences, apply RTN fallback to everything
still ineligible and continue while reporting per-layer coverage. There is no
third collection, lower ESS threshold, pooled-expert statistic, or borrowed
statistic from the other replica.

## Frozen held-out teacher-KL data

The primary metric uses WikiText-2 test text split into the same 4,096-character
non-empty chunks and tokenized to at most 512 Nex tokens. It captures one
terminal full-vocabulary distribution per context.

The earlier 64-chunk PPL set consumed chunk indices `0` through `63`; the
earlier exact-KL set consumed every fifth index from `0` through `315`. This
probe excludes both sets. A separate all-chunk PPL replication may already have
measured observed-token likelihood on these chunks, but it contains no
route-aware candidate and no full teacher distribution; it is not an input to
candidate construction or this gate. From the 201 chunks outside the two older
suites, 32 positions are selected deterministically across the full range and
interleaved into two disjoint splits:

```text
split A: 64, 81, 97, 113, 129, 146, 161, 177,
         193, 209, 226, 242, 258, 274, 291, 307

split B: 72, 88, 104, 121, 137, 153, 169, 186,
         202, 218, 233, 249, 266, 282, 298, 314
```

The context generator must store chunk index, text hash, token IDs, token hash,
token count, corpus hash, tokenizer identity, and manifest hash before candidate
collection. The two manifests must be disjoint by both chunk index and token
hash. Candidate scale selection must never read BF16, RTN, or candidate
distributions from these contexts.

Each of the two calibration-replica candidates is evaluated on both held-out
splits. This produces four fixed cells: `A/A`, `A/B`, `B/A`, and `B/B`, where
the first letter is the calibration replica and the second is the held-out
split.

## Primary metric and paired analysis

For each context, compute full-vocabulary forward divergence from the same BF16
teacher:

```text
KL_candidate = D_KL(P_BF16 || P_candidate)
KL_RTN       = D_KL(P_BF16 || P_RTN)
delta        = KL_candidate - KL_RTN
```

Negative `delta` favors route-aware diagonal RMS.

Report all per-context values, each of the four cell means, wins/losses/ties,
teacher-top-1 agreement, and a pooled paired bootstrap. The bootstrap is
stratified by held-out split: resample 16 contexts with replacement inside each
split, carry both calibration-replica deltas for every sampled context, then
average all 64 paired observations. This treats context as the resampling unit
and does not pretend the two replica observations for one context are
independent.

## Frozen pass/fail rule

The probe passes only if **all** conditions hold:

1. Each of the four calibration-replica/held-out-split cells has strictly
   negative mean `candidate - RTN` forward KL. A tie fails.
2. The pooled paired-bootstrap 95% interval for mean `candidate - RTN` forward
   KL has a strictly negative upper bound.
3. Across all 64 replica-context observations, at most one RTN teacher-top-1
   match becomes a candidate mismatch. Candidate-only gains do not cancel more
   than one such loss.
4. Both candidates retain the exact standard MXFP4 tensor shapes, dtypes,
   quantization coverage, target/ignore policy, and tensor-data byte count of
   the RTN checkpoint: `22,902,614,752` bytes.
5. Every tensor outside the eight selected source banks is byte-identical to
   RTN. Ineligible experts inside those banks reconstruct the exact RTN packed
   values and scales.
6. Both candidates load in the pinned stock vLLM image, select the same
   `MarlinExperts` W4A16 path with both Marlin backends forced, and pass the
   existing text and vision smoke requests without fallback or request error.

These requirements are conjunctive. A favorable pooled mean cannot rescue one
reversed cell, and a local reconstruction improvement is not part of the pass
rule.

## Resource and stopping boundary

- Run long jobs detached on Spark so SSH loss does not terminate them.
- At most two 16-sequence calibration passes plus one conditional 16-sequence
  extension per replica.
- Keep hidden states on CPU between layers and only one decoder layer plus the
  bounded expert work unit on the accelerator.
- Patch only the eight selected source banks onto RTN. Do not run a calibrated
  40-layer conversion, task suite, long-context suite, or throughput sweep
  before this gate passes.
- Hard stop after six aggregate GPU-hours or eight wall-clock hours. A timeout
  is `incomplete`, not a pass and not permission to reduce the gate.
- If coverage, identity, numerical, format, runtime, or either quality gate
  fails, record the failure and stop. Do not change ESS, scale-search settings,
  layers, route weighting, calibration offsets, or held-out contexts.

A pass permits one separately pre-registered full route-aware conversion and
whole-model qualification. It does not itself justify publication, a general
quality claim, or a claim that diagonal RMS is better than block Hessian.

## Required immutable record

Before deleting temporary Spark artifacts, retain:

- source, RTN, runtime, corpus, tokenizer, token-manifest, implementation, and
  candidate hashes;
- per-replica routed counts, `sum(r^2)`, `sum(r^4)`, ESS, eligibility, and
  per-layer eligible-energy fractions before and after any extension;
- gate/up and real down-input statistic hashes and finite/shape validation;
- selected bank/tensor change counts, exact RTN fallback checks, tensor bytes,
  structural-verifier output, and kernel-selection logs;
- raw BF16, RTN, and both candidate log-probability artifacts plus all
  per-context metrics and bootstrap settings; and
- wall time, peak process RSS, peak accelerator allocation, commands, exit
  codes, and timeout state.

After completion, add a dated human-readable result, a compact machine-readable
summary, and the roadmap tracker/decision row before deleting large temporary
artifacts. This pre-registration remains result-free until those measurements
exist.
