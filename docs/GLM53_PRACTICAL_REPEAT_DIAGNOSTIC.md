# Practical-task repeatability diagnostic

**Complete; all 340 responses independently regraded.** behaviorrepeatv1 ran
17 October 2026 from 11:16:59 UTC. Both original GLM container identities and
configuration digests were restored, health returned 200, and independent
verification finished at **11:56:51 UTC**. No model download, quantization,
speed sweep or PPL comparison was repeated in this diagnostic.

| Metric | Reference, five rounds | Candidate, five rounds |
|---|---|---|
| Strict passes / 34 | 30, 30, 31, 29, 32 | 31, 29, 30, 30, 30 |
| Total strict passes / 170 attempts | 152 | 150 |
| Math passes / 24 | 21, 22, 22, 20, 23 | 22, 21, 22, 22, 21 |
| Total math passes / 120 attempts | 108 | 108 |
| Functional code checks | 10/10 | 10/10 |
| Actual tool calls | 10/10 | 10/10 |

Both arms vary on the same four pass cases: slugify, gsm8k_0361, gsm8k_0978
and gsm8k_1133. Every case passing all five reference rounds also passes all
five candidate rounds. The two-pass aggregate strict gap comes entirely from
slugify's bare-code formatting (4/5 reference, 2/5 candidate); the function
works in every round. Math changes cancel in aggregate, which is not proof
of identical individual answers. These are repeated temporal trials of 34
questions, not 170 independent questions.

This demonstrates reference task variability at the previously lost cases.
It does **not** qualify the adapter or amend the failed stablequalityv1 strict
contract. All rounds and losses remain. The [compact review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/practical-task-repeat-v1-review.json)
includes every per-case count and both directions of every within-arm pair.
The [separate dispatch component](GLM53_KDA_DIRECT_DISPATCH.md) subsequently
finished; its performance confirmation failed and it remains unintegrated.

## Analysis recovery and independent verification

Inference and all isolated code sandboxes finished successfully. The frozen
controller then hit a CPU analysis compatibility error: its older imported
task scorer did not provide inference_completed. The original controller
state remains failed_restored with this error. Both arm clients exited zero;
no inference was rerun and the frozen 116-source-file inventory was untouched.

An append-only analysis-repair-v2.py derives completion from each raw API
response and cross-checks the newer helper when present. The repaired review
has SHA256 e4e5669d1cd6fd4daf213cc5b01ef166e75c36e6e978af941009cc96ef72ea18;
repair code is ef9f09e8818e83d7e089c931e1296b2bf2983992f968883607e290105ab4de10.
A local regression test covers an HTTP-successful format failure versus an
API error using the legacy helper schema.

The two immutable raw archives contain **297 verified payloads**. Their SHA256
digests are 1b343b841c98524103a49eef52d48d3a970912ef158f092a48d774c53624e8c8
and 934777bd9772854394ce9efc7bc375b3cee077de4f0a46f8e6a413d08b6f2b79.
The archive snapshot preceded repair completion; the three repair/report
payloads are therefore in a separately verified, archive-bound supplement,
SHA256 f4c32eb7a7b3d5541ed29a57c16d28a7ece05714bfd8854933147fa0726ad089.
Both originals are preserved. Independent review checks all **672 source-scale
comparisons in CPU-audit order**, both 116-file source inventories, all 340
raw API grades, generated code hashes/ASTs and isolated sandbox receipts.

## Why this measurement

The completed [stable-dispatch experiment](GLM53_STABLE_DISPATCH_EXPERIMENT.md)
passes the unchanged numerical gates and has identical held-out scores.
Its strict 30/34 task totals conceal two lost format passes and two gained
math passes. One reference/candidate task round cannot establish whether these
format outcomes are stable under repetitions of the same loaded reference.
That missing observation is the purpose of this batch. The earlier strict
failure remains unchanged regardless of its outcome.

## Frozen scope

* Same exact 34 cases, prompts, grader, seed 42 and request parameters as
  stablequalityv1. The earlier quality-protocol.json is copied byte-for-byte.
* **Five predeclared rounds per arm**, 170 attempted tasks per arm. Run all five
  without score-dependent stopping or best-run selection. Keep the first
  round; warmup is engine initialization/capture, with no task round dropped.
* One reference load followed by one qualified-v9 candidate load. No original
  service restoration between arms. Independent restoration at the end/error
  remains mandatory. Both ranks share the same stable dispatch/workspace
  controls, BF16 activations, TP2 transport, fixed 4 GiB FP8 KV/rank and batch-one
  CUDA graph configuration as the completed experiment.
* Full source-scale checks before/after each arm compare all 84 groups/rank
  against the independent 168-group checkpoint audit. Native v9 digest is
  checked. All generated code is scored afterward in isolated CPU sandboxes.
  A pending sandbox score is not a code failure.
* Report every round, both directions of all ten within-arm pairs, per-case
  pass counts, distinct output hashes and API completion. Raw answers remain
  private on Spark. Equal totals must not hide changed task outcomes.
* This is a diagnostic, not a new qualification contract. It does not change
  KL/NLL floors, relax the no-reference-pass-lost rule, or convert the previous
  strict failure into a pass. Temporal repetitions are not independent new
  benchmark questions; no broad accuracy or confidence claim follows from them.

The same frozen 116-file Python/TOML inventory is on both nodes, SHA256
e369dd3a494d09e4be961e779a56c43a0412ed6f32926c791e2126a7ed7b2df7.
The three-file controller/client/analyzer overlay is SHA256
e3da97b670abf422f50017cbdc2ed055abcbb72ecc4ea76c47cb50bec51f482e.
Qualified runtime/codec files and the v9 native library retain their earlier
hashes. No frozen earlier experiment directory is modified.

## Paths and monitoring

On the head, reached with ssh spark:

    ROOT=/home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/full-model-v1

Read these files using host stdlib Python; do not import host Torch:

    ROOT/CURRENT_LOSSLESS_STATUS.json
    ROOT/quality-model-controller-behaviorrepeatv1.json
    ROOT/behaviorrepeatv1-controller.log
    ROOT/behaviorrepeatv1-prepared.json
    ROOT/behaviorrepeatv1-launch-receipt.json
    ROOT/run-behaviorrepeatv1/results/task-repeat-plan.json
    ROOT/run-behaviorrepeatv1/results/{control,candidate}-quality-measurement.json
    ROOT/run-behaviorrepeatv1/results/{control,candidate}-r{0,1,2,3,4}/real-work.json
    ROOT/run-behaviorrepeatv1/results/quality-model-review.json
    ROOT/watchdog-behaviorrepeatv1.json
    ROOT/restoration-behaviorrepeatv1.json
    ROOT/behavior-repeat-v1-collection.json
    ROOT/behavior-repeat-independent-archive-check-v1.json

The source root is ROOT/full-behaviorrepeatv1-src. The final
archives are ROOT/behavior-repeat-evidence-v1-rank{0,1}.tar.gz on their respective
nodes. Collector and verifier are independent detached processes with finite
deadlines; they verify restoration and all archive payloads. Each measurement
client has a 40-minute limit; engine startup has a 20-minute limit. The original
restoration watchdog/outage deadline remains 8,400 seconds after pause.

The recovered review and supplement are ROOT/run-behaviorrepeatv1/results/
quality-model-review.json, ROOT/behavior-repeat-analysis-recovery-v2.json and
ROOT/behavior-repeat-analysis-recovery-evidence-v2.tar.gz. The independent Mac
review is retained privately in /private/tmp/mxwave-behavior-repeat-v1/.
The same original GLM container identities/configurations are retained.

## Interpretation before seeing results

If reference repeats lose their own task passes, the frozen practical task
rule has a reproducibility problem at those cases. That observation does not
prove every adapter difference harmless. If reference is stable while candidate
formats vary or systematically fail, isolate the candidate's BF16/MoE arithmetic
on identical saved inputs before another performance change. Preserve all
rounds and the original failure in either case. Use this evidence to design a
prospective semantic/format reliability protocol, not to select a convenient run.
