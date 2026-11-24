# Matched graph-four quality evaluation

**Complete with failed reference repeatability; originals restored.**
The 14:59:41 UTC launch passed CPU preflight and completed four reference
decode legs. Comparisons against the first reference gave KL
0.000006310838, 0.000005206705 and **0.000011791085**, the last exceeding the
unchanged 0.000010000 limit. All generated ids and conditioning prefixes
matched; the last absolute mean NLL difference was 0.000020123395, below
0.000100000. The planned stop occurred before candidate loading, timing,
expanded held-out windows, practical tasks or profiles. These unrun phases
remain unmeasured, not passed. No inference retry or gate amendment.

Both original container configurations are restored with health 200. The
independent collector and local archive check verify **587 payloads** (443
rank 0, 144 rank 1). The source-bound independent verifier reconstructs all
**256 float32 distribution rows from raw API responses**, rechecks 16
both-rank cohort proofs and 168 source-scale groups, and reproduces all three
reported KL/NLL comparisons. The candidate was never loaded.

All sixteen prefill distributions are byte-identical across all four
reference legs, with prefill KL exactly zero. None of the later decode
distribution rows is byte-identical to its first-leg counterpart. The
remaining difference therefore appears during autoregressive decode. The
largest step KL in the failed comparison is 0.000212047394 at context 4,
decode position 1; this does not identify the first internal operator that
diverges. The exact decode kernel/state cause remains unlocalized.

The compact independently verified result is
`benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/graph-quality-full-model-v1-review.json`
(SHA256 `08e66d805d842df958b08d6b58a2728b80f993d652a195ae29e41bd4957486dc`).
The next user-requested [full GSM8K comparison](GLM53_GSM8K_FULL_BENCHMARK.md)
collects actual answer outcomes while keeping this failed qualification intact.
The previous controlled diagnostic measured graphs versus eager execution on
the same compressed candidate. It reached 40.66–41.03 aggregate tok/s with
four requests, without a speculative drafter. The missing comparison is the
raw-runtime reference using those same graph settings. Earlier reference
numbers (14.99 tok/s at C1 and 38.77 aggregate at C4, prompt 256) used graph
captures [1], so C4 decoded eagerly.

This prospective batch uses one reference load followed by one candidate
load, both with captures [1,4], canonical expert dispatch, BF16 activations,
4 GiB FP8 KV per rank, and the same pinned imported MXFP4 checkpoint. It
retains the original KDA operation and qualified v9 native library. The rejected
direct Python wrapper is not installed. FP4 codes and KV storage are unchanged.

## Frozen inputs and gates

The CPU-only freeze succeeded while original GLM remained healthy. Offline
tokenization matches every old API-tokenized held-out window and the complete
corpus token count. Dataset payload hashes and revisions are unchanged.

- Expanded protocol SHA256:
  `da86bee2df23f2796415ee1e6fbe7c7b111f04767f5d1ad5c0dbf8f6650b43d0`.
- Five fixed full-vocabulary decode repeats per arm: sixteen original contexts,
  four prompts per controlled group, four generated tokens with real KV.
  Both ranks must prove each actual prefill shape; a split group stops the run
  without a retry. Repeat comparisons use the first repeat, never a selected
  favorable repeat. Cross-arm comparisons match repeat indices.
- Original decode limits: mean KL <= 1e-5, absolute mean NLL difference <= 1e-4,
  every generated id and decode conditioning prefix must match. An early
  numerical failure stops the longer held-out and task phases.
- Three held-out repeats per arm, each containing 100 non-overlapping windows
  and 102,400 scored tokens. Within-arm mean NLL differences must be <= 1e-4;
  each paired cross-arm difference must be <= 1e-5. This fixed cross-arm limit
  is at least as strict as the earlier adaptive limit.
- 110 strict practical cases per arm: preserve all 34 original cases and add
  76 distinct GSM8K questions from the pinned test source, sample seed 20261018.
  The resulting set has 100 math questions plus ten practical cases. Existing
  graders, prompts for old cases, natural stops, code-format contracts and
  code sandbox checks remain unchanged. No reference-passed case may be lost.
  Actual tool calls use both `glm47` and `--enable-auto-tool-choice`.
- Normal HTTP timing: C1 and C4 at prompt 256, two warmups and five retained
  64-token trials per case per arm. No scheduler pause or profiler in timing.
- A separate controlled C4 PyTorch trace per rank per arm follows all scoring.
  Profile overhead cannot contribute to throughput medians.

These repeats estimate runtime variability; they are not five independent
benchmark samples. The held-out windows are sampled PPL, not full-corpus PPL.
The reference already contains the source MXFP4 quantization; its error versus
BF16 is outside this comparison. Batches two and three are outside this gate.
All previous failed qualifications remain unchanged, even if this prospective
configuration passes. Exact storage reconstruction is distinct from end-to-end
floating-point equality.

## Execution and recovery

Head alias: `spark`. Both ranks use:

```text
ROOT=/home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/full-model-v1
Frozen source: ROOT/full-graphqualityv1-src
Controller: ROOT/quality-model-controller-graphqualityv1.json
Results: ROOT/run-graphqualityv1/results
Prepared inputs: ROOT/graph-quality-v1-prepared.json
CPU preflight: ROOT/graph-quality-v1-cpu-preflight.json
Restoration: ROOT/restoration-graphqualityv1.json
```

Bindings at launch:

```text
135 Python/TOML source files
inventory: 15c49f201bd46fc2f9f530581330a6df44d2700782ccbfa1befc4f53e0eacaf3
prepared: 018a5b5eb87688e3b5e9aeec5efccbde7431f981f66d6868531b9d2f27dd7b4a
CPU preflight: 5a19bc26632281d697a2e027d40d3fbadc125cb951c7474495867a4131d216fa
independent verifier: 263557ade0e62e231f7c64e0354000dc539314b712715cf2b0554fb6aaed23d9
```

Source assembly starts from the independently verified `cohortdispatchv2`
snapshot, overlaying only the paired-run controller, observer, client,
protocol generator and CPU analyzer. In particular, do not copy the current
worktree's alternate `moe_runtime.py` over the qualified v9 runtime. Native,
codec, source, dataset, CLI and message serialization checks run before pausing
GLM. The prospective protocol is immutable before either model load.

After an idle window, stop only the two recorded original GLM containers.
An independent restoration watchdog must acknowledge readiness first. One
final restoration verifies original ids and configuration digests, plus HTTP
health. No service restart between the two experiment arms. Each quality
client has a 90-minute deadline; the total outage watchdog is four hours.
All compute runs in bounded containers; no host PyTorch process is used.

Monitoring from the laptop:

```bash
ssh spark 'python3 -S -m json.tool /home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/full-model-v1/quality-model-controller-graphqualityv1.json'
```

The raw response distributions, every held-out token score, every practical
response, both-rank source hashes, graphs and separate traces must be retained
and independently verified. Final quality, memory and speed numbers remain
pending this evidence; this document does not assert a new qualification.
