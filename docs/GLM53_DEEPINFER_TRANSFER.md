# DeepInfer reference for CT drafter confidence and wider batch invariance

Updated 20 October 2026. This adds a separate experimental workstream. The
fixed-seven-draft development comparison has completed with its frozen sources,
weights, prompts and gates: output identity and all eight timing controls pass,
but the small pilot is effectively speed-neutral. Production is restored.
The sealed test remains closed.

## Pinned upstream evidence

[DeepInfer](https://github.com/deepseek-ai/DeepInfer/tree/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01)
is Apache-2.0. Its pinned
[DeepGEMM submodule](https://github.com/deepseek-ai/DeepGEMM/tree/1fc9c66975ee322d628b3288534ca0e9ce12a64d)
is MIT, not Apache-2.0. Both grants are retained in the external audit checkout;
reviewed source hashes are in `integrations/glm53_mxfp4/deepinfer_reference.json`.
`CONTRIBUTING.md` records the references and independently authored changes.
No upstream implementation is vendored into MxWave.

| Mechanism | Upstream evidence | Transfer to this project |
| --- | --- | --- |
| Confidence head | `models/mtp/dflash.py` projects draft hidden state plus predecessor Markov embedding to one logit, with positional temperature | Use GLM DFlash2 hidden state plus its predecessor codebook; train our own weights |
| Verification budget | `csrc/engine/mtp_varlen.hpp` combines historical prefix probabilities with measured graph cost | Measure this target's complete step costs, then choose among qualified widths |
| Per-request allocation | `schedule_mtp_varlen.py` ranks cumulative acceptance probabilities with stable ties | Allocate contiguous draft prefixes, retaining one base query row per request |
| mHC | `mhc_ops.py` fixes CUDA splits at 16; DeepGEMM `apis/mega_mhc.hpp` fixes splits in deterministic mode | Keep split/composition independent of batch size; do not copy the number 16 into a different kernel |
| Gate | DeepGEMM `heuristics/mega_gate.hpp` disables K splitting in deterministic mode | Inspect routing projection reductions as well as expert GEMMs |
| Expert combine | DeepGEMM `sm100_fp8_fp4_mega_moe.cuh` traverses full K and combines shared/routed contributions in a defined order | Validate both GEMMs and the activation/weighted route sum, including shared rows across layouts |

DeepInfer enables deterministic behavior in DeepGEMM and TileKernels, and also
passes a batch-invariance option to attention. This is broader than one expert
padding fix. TileKernels is pinned as version 2.1.0 by DeepInfer; its underlying
kernels have not been audited here. The inspected DeepGEMM code uses SM100
primitives and expert-parallel communication, whereas our path is Marlin W4A16
on SM121 with TP2. It is not a drop-in runtime replacement.

The upstream deployment Q&A estimates +4–7% from variable verification near
its production operating point, but shows −1.5% at its lowest listed
concurrency. Those results use a different model and hardware layout; they
are not a speed prediction for CT.
[Source](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/doc/inference_deployment_qa.md)

## Confidence experiment implemented so far

`drafter_confidence.py` contains a zero-initialized, bias-free FP32 linear head.
For this DFlash2 checkpoint, hidden size 4,096 plus predecessor rank 256 gives
**4,352 trainable parameters / 17,408 weight bytes**. This excludes optimizer
state and execution overhead. Its inputs are detached; confidence loss cannot
update the backbone, selector or context projection.

`Pilot.confidence_observer` is disabled by default. A separately frozen
experiment can attach `ConfidenceObserver`, which trains through its own
optimizer on training prompts and records validation metrics without updates.
It receives the predecessor selected by the draft path, not an unavailable
future target token. Existing adaptation losses and proposal selection remain
unchanged. Real-data confidence training and native export/reload are pending.

The first version targets **greedy decoding only**. A target-sequence token is
a valid label while all earlier draft tokens match. Include the first rejected
token, then censor later positions; otherwise the head would learn from target
continuations conditioned on a different history. Stochastic acceptance needs
actual verifier acceptance probabilities and a separate validation protocol.
The masked loss uses the fixed number of offered slots as its denominator.
Dividing each block by its observed prefix length would overweight early
failures and bias the probability estimate.

The full-run data path also has an opt-in `bounded_feature_loading` policy.
`drafter_feature_store.py` verifies file hashes and loads one feature sample
at a time; anchor selection reads only token IDs. Cache eviction leaves tensors
owned by an active autograd graph alive until backward completes. CPU tests
cover exact values, eviction, changed-file rejection and backward correctness.
The old pilot protocol stays unchanged; a real pilot replay under the bounded
policy remains required before the 4,000-prompt run.

`confidence_verification.py` is a CPU oracle, not an installed scheduler:

- Choose the total query budget using confidence at least two decisions old.
  Current later-position confidence cannot choose its own total budget.
  This follows the upstream scheduler's explicit dependency constraint.
- Allocate that fixed budget using current prefix probabilities and stable
  request/position ties. New requests without drafts have empty probability
  rows; zero confidence is not treated as an availability flag.
- Use only widths with measured complete-step costs under the actual
  graph/padding policy. No guessed interpolation, no extrapolation and no
  assumed savings from removing logical rows that still use the same graph.
- All emitted draft tokens still require target verification. Confidence is
  an efficiency estimate, never permission to skip correctness checks.

Runtime request-ID remapping, confidence-age tracking, EOS/budget constraints,
native head export/reload and device-only scheduling remain integration work.
The CPU oracle rejects impossible budgets rather than inventing a serving
fallback. DFlash2 features also differ from DeepInfer's pre-normalization
features, so its trained confidence weights cannot be reused.

CPU tests check censored labels, gradient isolation, validation freezing,
checkpoint reload identity, invalid inputs, historical-budget enforcement and
prefix allocation against an exhaustive optimizer. They do not establish
model calibration, GPU execution, graph replay or speed.

## Frozen qualification plan before enabling serving

1. **Complete:** the existing fixed-width development comparison and independent
   review. All 144 requests and eight timing controls pass; the pilot's measured
   code gain is only 0.111%. Earlier failed attempts remain retained.
2. Train the head alongside adaptation using training data only. Keep the
   serving target/padding policy recorded with each feature dataset. Old pilot
   features under native reduction are not a substitute for new-policy capture.
3. On development data, report conditional BCE/Brier, prefix Brier, predicted
   versus observed accepted length and per-position/domain reliability. Compare
   with a position-only empirical acceptance baseline. Fit any temperature on
   a designated calibration subset, then report on a distinct development
   subset. Do not tune on the sealed test.
4. Profile fixed lengths 0/3/5/7, or another explicitly frozen supported set,
   at C1/C8/C16 with realistic mixed prompts. Bind costs to active requests,
   context lengths, graph widths and target policy. Include head, scheduler,
   sampler and CPU/GPU handoff costs; collect clocks and thermals.
5. Compare fixed-7, cost-only static length and confidence scheduling under
   identical target policy. Preserve the established ≥8% code end-to-end goal,
   ≤50 code rounds/256 tokens, ≤2% chat/explanation regressions and unchanged
   repeat controls. Scheduling may trade more rounds for cheaper rounds, but
   it does not silently waive the existing round-count gate.
6. Only after candidate, calibration, width set and runtime freeze: sealed
   evaluation. Greedy output identity must pass before a serving speed claim.

## C8/C16 extension informed by the audit

Keep the current runtime limit at 16 target rows until GPU evidence passes.
The prepared 8/16/32/64/128-row fixtures cover both TP partitions and two routing
patterns, but remain CPU-only evidence.

For every width, record actual routing block size, K/N tile, blocks per SM,
grid and padded dispatch count. Whole launch waves should remove Stream-K's
dispatch-dependent K partitions; confirm this in the actual planner, not only
in the padding formula. Check shared rows across widths/arms, stock versus
packed scales, masked dummy routes, up/activation/down and weighted route sum.
Then test graph capture/replay and full-model self-repeat under mixed layouts.
Keep mHC composition fixed and examine gate/attention reductions if divergence
remains. Measure native versus padded serving overhead separately from drafter
adaptation. The DeepInfer audit does not qualify any new Marlin width.
