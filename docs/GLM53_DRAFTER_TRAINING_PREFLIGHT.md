# CT drafter: training graph and bounded feature capture

Updated **20 October 2026, 14:18 UTC**. The bounded capture is complete and independently
verified; original production was restored at **14:03:27 UTC** with 500k context.
All 44 requests completed, both output/counter controls pass, and all 5,456
payload hashes match. The accepted export passes a separate NumPy bit audit.
The CPU context-projection pilot is complete with a selected development checkpoint; see the
[capture result and current pilot](GLM53_DRAFTER_CONTEXT_PILOT.md).
The graph and capture protocol below describe the completed preceding stages.

The target remains **CT MXFP4 experts + native MXFP8 KDA**. Source CT with BF16
KDA remains the lossless reference. There is no additional target quantization
change or relaxation of the failed cross-target 1e-5 KL gate.
The [frozen adaptation thresholds](GLM53_DRAFTER_PREFLIGHT.md) remain unchanged.

## Real-input CPU graph check

The pinned, unmodified SpecForge DFlash2 implementation loads the actual converted
checkpoint. Its nonpersistent rotary buffers require materialization after meta
construction: they are absent from the checkpoint state dictionary. The loader
recreates that parameter-free module through its upstream constructor and rejects
any remaining meta buffer.

Inputs are the complete 171-token context from the retained native CT diagnostic,
plus the actual CT embedding rows and vocabulary head. This is an old observed
fixture, not new training or held-out data.

| Check | Result |
| --- | --- |
| Dense CPU forward repeat | Bitwise exact |
| Initial seven-token proposal | All seven match native |
| Context projection relative L2 versus native | 0.0002650 |
| Initial final draft hidden relative L2 versus native | 0.0264810 |
| Gradient through frozen backbone to context projection | Finite; 83,886,079 of 83,886,080 entries nonzero |
| Disposable BF16 update and reload | Exact reload; 1,789,159 BF16 elements changed |
| Original 81 checkpoint parameter tensors | All byte-identical before/after |
| Corrected CPU check | 36.884 seconds; backward about 30.29 seconds |

Only an FP32 master copy of `fc.weight` participates in AdamW (one step, 1e-5,
zero weight decay); the forward uses its BF16 cast. Original parameters are
frozen. MSE against recorded native draft hidden states exercises autograd and
serialization only. **It is not the target supervision or adaptation recipe.**
The loss rises from 0.00217547 to 0.00261060 after the rounded step. The saved
tensor is disposable and must not be deployed or scored as a trained candidate.

V1's FP32 aggregate cosine exceeded one because of reduction precision. V2
retains V1, computes comparison statistics in FP64 and adds stage timings.
Gradient/update counts reproduce exactly. Corrected projection cosine:
0.999999965.

## Eight native histories

Reconstruction of accepted context through all eight saved frames gives
**56/56 identical proposed tokens and 8/8 identical full draft blocks**.
Histories are the actual CT histories; dense predictions are not fed back.
Final hidden relative L2 spans **0.02318–0.02819**. Some top-16 candidates differ
despite matching selected paths. Native arithmetic parity is therefore not
established; BF16 CPU training and native FP8 Marlin/KV/head/fused operations
remain distinct. Eight short, observed frames are not a general acceptance or
speed result. Replay takes 6.718 seconds.

The sequence builder checks frame order, contiguous absolute positions, previous
bonus-token anchors, accepted sampled prefixes, rejection counts and next-query
positions. For the retained 32-token API response it recovers 202 feature rows
(171 prompt + 31 generated inputs), excludes 13 rejected rows, and excludes 12
sampled tokens/feature rows beyond the API cap. Training export must bind to the
exact frozen prompt and capped API output first.

## Bounded capture

Selection takes the first eligible examples in the frozen order, with at most
1,024 prompt tokens, opening only training and validation files.

| Split | Code | Chat | Explanation/QA | Total |
| --- | ---: | ---: | ---: | ---: |
| Training | 20 | 6 | 6 | 32 |
| Validation | 4 | 2 | 2 | 8 |

Selected inputs total 3,130 tokens, maximum 451. Each request generates 128 fixed
output tokens. There are 40 captures plus four before/after controls:
**44 requests total**. Controls require identical outputs and speculative
counters. The original untrained FP8 drafter stays loaded; the disposable CPU
update is absent from the launch.

The passive probe records all five feature streams, sampling/rejection decisions,
positions and native FULL draft graph outputs on both ranks. Accepted feature
storage is approximately 337,920,000 bytes per rank before auxiliary, rejected
and post-cap records. This is a bounded capture-format pilot, not the full corpus.

Both serving images reproduce all 40 frozen prompt token sequences on CPU.
Original container identities and 211 bindings per rank match. The private-port
**8019** trial uses the existing independent restoration watchdog and a
**2,100-second experiment deadline**; restoration adds its own warm-up time.
Busy production prevents launch. Completion, failure or deadline restores the
original containers and authenticated model identity with **500,000 context**.

No instrumented timing is a speed claim. Native reload of a trained update,
validation-selected improvement, sealed evaluation, serving capacity tuning and
long-context qualification remain future gates.

## Artifacts

Local root: `out/drafter-training-preflight-v1/`.
`cpu-v1/` preserves the first metric limitation; `cpu-v2/` contains the corrected
report and disposable tensor. `replay-v1.json` holds eight comparisons.
`source-rows/` holds hashed CT head and selected embeddings.
`capture-source/` and `capture-prepare.json` freeze the new capture.

Remote:
`/home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/full-model-v1/drafter-feature-capture-v1`.
Plan SHA-256:
`2e3dfcf8f61eadbbb7ae4092fed661da986daec7ed1d88921f5d771156bd4c7f`.
CT head SHA-256:
`aa21c427970f64edd82669db3a8fb46613084e8bc271a3728784a52eb3f25ab4`.
Source checkpoint files and production weights remain unmodified.

Validation: Ruff and strict mypy pass; **891 tests pass, one optional dependency
skip**. Six new sequence tests cover accepted history, rejected/post-cap
exclusion, missing frames and corrupted position/token/anchor/rejection metadata.
