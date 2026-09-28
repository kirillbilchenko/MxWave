# Nex-N2.5-mini route-aware diagonal-RMS probe result — 2026-09-22

## Decision

The pre-registered probe **failed**. Route-conditioned diagonal-RMS weighting
did not improve Nex-N2.5-mini's routed-expert MXFP4 checkpoint consistently
against the routed-expert RTN baseline.

Only one of the four fixed calibration-replica/held-out-split cells had a
strictly negative mean `candidate - RTN` BF16-forward KL. The pooled mean delta
was `+0.0095027905` nats, or `+7.9733%` relative to RTN, and its stratified
paired-bootstrap 95% interval was `[-0.0393049352, +0.0724281566]`. The interval
crossed zero and its upper bound was positive. Teacher-top-1 preservation passed
its guardrail with zero RTN-match losses and two candidate-only gains, but that
cannot rescue the failed KL conditions.

Checkpoint identity and exact-RTN fallback conditions passed. Under the frozen
stop rule, the prescribed text/vision smoke, task suite, full 40-layer
route-aware conversion, long-context evaluation, and throughput work were not
run. The six-condition probe is therefore false because conditions 1 and 2
failed; condition 6 is recorded as `not_run_due_to_frozen_stop`, not as a
runtime failure.

This result rejects this exact four-layer, two-replica, ESS-64,
route-coefficient-squared diagonal-RMS hypothesis. It does not show that all
route-aware calibration is ineffective, and it does not undo the bounded MoE
adapter/emitter/runtime engineering result. Neither route-aware candidate is a
publication candidate.

The frozen pre-registration is
[`NEX_ROUTE_DIAGONAL_PROBE_2026-09-22.md`](NEX_ROUTE_DIAGONAL_PROBE_2026-09-22.md).
The complete per-context machine-readable gate report is
[`nex-n2.5-mini-route-diagonal-gate-2026-09-22.json`](../benchmarks/nex-n2.5-mini-route-diagonal-gate-2026-09-22.json),
with SHA-256
`3c079a0e6e2bdaafcec2686fc70d75a0589a46fd4843921f04d5230bea402aaf`.
The compact result is
[`nex-n2.5-mini-route-diagonal-probe-2026-09-22.json`](../benchmarks/nex-n2.5-mini-route-diagonal-probe-2026-09-22.json).

## Product and claim boundary

This was a quality-only experiment. Both candidates retained the same standard
`mxfp4-pack-quantized` format, `22,902,614,752` tensor-data bytes, and Marlin
W4A16 serving representation as RTN. The method could not make the model
smaller or faster.

The primary comparison was route-aware diagonal RMS versus routed-expert RTN.
It was not a direct held-out comparison against the earlier unweighted-MSE
candidate, public NVFP4, or BF16 as a deployable checkpoint. BF16 supplied the
teacher distribution. A positive result would only have admitted a separately
pre-registered full conversion and qualification; it would not have justified
publication or established architecture-general MoE support.

## Frozen method and scope

- Source: `nex-agi/Nex-N2.5-mini` at
  `87420286149d9cce9bd46cd335ef9bda33c37c1b`.
- Baseline: the routed-expert RTN MXFP4 checkpoint.
- Selected decoder layers: `0`, `13`, `26`, and `39`.
- Selected source banks: routed `gate_up_proj` and `down_proj` in each selected
  layer, eight banks total.
- Scale search: percentile `99.5`, clip depth `4`, with the no-clipping
  candidate retained.
- Eligibility: per-expert effective sample size `ESS >= 64`; gate, up, and down
  fall back together when unsupported.
- Route weight: square of the normalized top-k router coefficient.
- Gate/up statistic: actual hidden rows routed to each expert.
- Down statistic: the actual BF16 post-activation product
  `activation(gate(x)) * up(x)` consumed by `down_proj`.
- Calibration replicas: offset `0` for A and offset `32` for B, with
  deterministic 512-token sequences.
- Runtime: official `vllm/vllm-openai:v0.29.0` at digest
  `sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1`.
- Hardware: DGX Spark GB10 / SM121.

The implementation chose four evenly spaced inclusive endpoint layers from the
40-layer model, which resolved to the frozen set `0,13,26,39`. This is a generic
selection rule in the calibration command, not a post-result layer choice.

This is route-aware **diagonal RMS** through the existing `gamma^2` objective.
It is not a block Hessian, GPTQ, AWQ, full-loss Hessian, or end-to-end-KL
optimizer.

## Calibration and coverage

Both initial 16-sequence replicas triggered their one permitted extension to 32
sequences: replica A had layers 26 and 39 below the 95% eligible route-energy
threshold; replica B had layers 13, 26, and 39 below it. No threshold was
lowered and there was no third collection.

| Replica | Offset | Final sequences | Eligible experts | Aggregate route-energy coverage | Layer 0 / 13 / 26 / 39 coverage |
|---|---:|---:|---:|---:|---|
| A | 0 | 32 | 848 / 1,024 | 0.9842039193 | 0.9971776 / 0.9857978 / 0.9784999 / 0.9753320 |
| B | 32 | 32 | 835 / 1,024 | 0.9824127022 | 0.9988821 / 0.9783174 / 0.9776526 / 0.9751416 |

Replica A produced 2,544 calibrated logical matrices and 528 exact-RTN
fallback matrices. Replica B produced 2,505 calibrated matrices and 567
fallback matrices. Each supported expert contributes gate, up, and down.

Recorded calibration resource peaks were:

| Pass | Elapsed seconds | Peak process RSS bytes | Peak accelerator allocation bytes |
|---|---:|---:|---:|
| A16 | 49.6795 | 5,636,091,904 | 1,835,343,360 |
| A32 | 63.7204 | 5,705,433,088 | 1,835,343,360 |
| B16 | 55.9775 | 5,634,564,096 | 1,835,343,360 |
| B32 | 67.8961 | 5,704,953,856 | 1,835,343,360 |

Calibration identities:

| Item | Replica A | Replica B |
|---|---|---|
| Token-ID SHA-256 | `d94862ca54413721a36db35ec7897cd948121465aff2b26104557cedc8167bce` | `4177e5bff9019eb9f5e3d7b004dda0261c9ec65474f784e4175216fd5d1f8a7c` |
| Artifact SHA-256 | `caab329b6901a67d3df9c87fade0abb74b0f8b95ff39dfd6ffefab0639852d1a` | `e114100b59f79359ddf053ba0ff7eaf2daad50696d517d3ad511406c6caf06c9` |
| Overlay-report SHA-256 | `a621e2805660f2d757791bfce5a0888c2723ec1e0b282abc3b46bd42eb2bc915` | `fefce843a3ef011586a93da73e17803f8ff09255d003fcc892e77e65b45491cc` |
| Output-integrity SHA-256 | `2fb91daea0563bfab9c0bd6581b5805f04ef91b5c166e385edd99d976fe3bb7d` | `555e59da6aa3cccecc919fa3fc086a180a69d88852d9013dd56fc830326971cf` |

## Held-out protocol

The evaluation used 32 previously untouched WikiText-2 contexts, each at most
512 Nex tokens, and captured one terminal full-vocabulary distribution over all
248,320 output IDs per context. The 32 contexts are the independent clusters;
the 64 replica-context values are not 64 independent contexts.

Split A used chunk indices:

```text
64, 81, 97, 113, 129, 146, 161, 177,
193, 209, 226, 242, 258, 274, 291, 307
```

Split B used chunk indices:

```text
72, 88, 104, 121, 137, 153, 169, 186,
202, 218, 233, 249, 266, 282, 298, 314
```

The metric was full-vocabulary forward divergence:

```text
delta = D_KL(P_BF16 || P_candidate) - D_KL(P_BF16 || P_RTN)
```

Negative delta favors route-aware calibration. The bootstrap used 100,000
resamples with seed `20260922`, resampling 16 context indices within each split
and carrying both calibration replicas for each sampled context.

Context identities:

- test-corpus SHA-256:
  `696cca6b65a171b0a358a4be6732cdfdf2dd6164a32e20fd70e3c13fc4dfae83`;
- raw preparation manifest:
  `d07346969238bb2e18410c2e460ced5321d2a44d669aaced31c8cd2feae72600`;
- combined context manifest:
  `d05a14341e5ed8e0050da7166f277c21a112570d7e5e116bed3568928eb64bc4`;
- split A manifest:
  `328ace1941421b251a59ac73a564cf632113aa5831818cef825f9e8518ed9666`;
- split B manifest:
  `78328c0d6357d4c13cf98d8b0a048669a3079c795f9aae396ad3d93314a9b3a1`;
- selected-context SHA-256:
  `bc4febabf2cff4de0d8eb57f18b403ab473d78d4926e9edfd9aadf2015f81d5f`.

## Exact-distribution result

Three cells regressed and one improved:

| Calibration / held-out | RTN mean KL | Candidate mean KL | Candidate - RTN | Relative delta | Candidate wins / RTN wins / ties | Cell gate |
|---|---:|---:|---:|---:|---:|---|
| A/A | 0.1394276473 | 0.1619071574 | +0.0224795102 | +16.1227% | 7 / 9 / 0 | fail |
| A/B | 0.0989366415 | 0.0681325996 | -0.0308040419 | -31.1351% | 7 / 9 / 0 | pass |
| B/A | 0.1394276473 | 0.1849373415 | +0.0455096942 | +32.6404% | 4 / 12 / 0 | fail |
| B/B | 0.0989366415 | 0.0997626411 | +0.0008259996 | +0.8349% | 6 / 10 / 0 | fail |

Across all 64 replica-context observations:

- RTN mean forward KL: `0.1191821444` nats;
- route-aware mean forward KL: `0.1286849349` nats;
- route-aware minus RTN: `+0.0095027905` nats (`+7.9733%`);
- candidate-lower / RTN-lower / ties: `24 / 40 / 0`;
- stratified cluster-bootstrap 95% interval:
  `[-0.0393049352, +0.0724281566]`.

The favorable A/B result did not reproduce in the other calibration replica or
the other held-out split. The pre-registered rule deliberately forbids a pooled
or isolated favorable result from rescuing reversed cells.

Teacher-top-1 transitions across the 64 observations were:

| Transition | Count |
|---|---:|
| Both RTN and candidate match teacher | 56 |
| RTN match becomes candidate mismatch | 0 |
| RTN mismatch becomes candidate match | 2 |
| Neither matches teacher | 6 |

The top-1 guardrail passed, but distribution fidelity did not.

## Six-condition gate

| Condition | Result | Evidence |
|---|---|---|
| 1. All four cell means strictly negative | **fail** | Only A/B was negative; A/A, B/A, and B/B were positive. |
| 2. Bootstrap upper 95% bound strictly negative | **fail** | Upper bound was `+0.0724281566`. |
| 3. At most one RTN teacher-top-1 match lost | pass | Zero losses; gains do not offset losses. |
| 4. Exact format/layout/coverage/byte identity | pass | Both overlays retained the RTN config, index, layout, dtypes, target/ignore policy, and `22,902,614,752` tensor bytes. |
| 5. Non-selected and fallback tensors exact RTN | pass | Non-selected files were byte-identical/hardlinked; all unsupported selected matrices passed exact packed-value and scale checks. |
| 6. Prescribed text and vision smokes | `not_run_due_to_frozen_stop` | Quality conditions 1-2 failed. Exact-KL collection did load both candidates with forced Marlin backends, but that is not a substitute for the prescribed smoke gate. |
| **Overall** | **fail** | The rule is conjunctive; conditions 1 and 2 failed. |

## Structural identity

Both candidate overlays retained:

- config SHA-256:
  `8ea02e392011595c2394a26862e51966608abab24fa2b750ce798b4ce67f6351`;
- index SHA-256:
  `69d4da6c13645e868454a7e22537e3f2ceef8883cf63817d67e2239145ad293c`;
- tensor-data bytes: `22,902,614,752`;
- selected expert bank shards: `8`;
- standard stock-vLLM `mxfp4-pack-quantized` W4A16 representation.

The overlay reports verified that the baseline tree remained unchanged, every
non-selected file was byte-identical, ineligible selected matrices exactly
matched RTN, config and index bytes were identical, tensor layout and byte count
were unchanged, and the stock format was unchanged.

## Raw distribution artifacts

| Role | Log-probability artifact SHA-256 | Checkpoint identity in collector metadata | Backends | Collection seconds |
|---|---|---|---|---:|
| BF16 | `964ee8a9f5318cd67cdec4e28aced924c181f6f29a7024e2964cef916f9ea3fe` | `79ca2088682301f4c37e460fa2bd06739275041436074e0ed5a2ad1f803706a0` | auto / auto | 389.864820 |
| RTN | `149f054a9b1e9182457a20354f9d89479e066dc2e3651beec7bab07432b781d3` | `78d675492db34c8c17954490be9d29b957a032680ce25b110b732f1e0d9f1af8` | marlin / marlin | 274.475596 |
| Route A | `df6397db68d9419e3281aa152f5dda5cbff40da462b9f44497846bc10504c1ab` | `2fb91daea0563bfab9c0bd6581b5805f04ef91b5c166e385edd99d976fe3bb7d` | marlin / marlin | 286.588610 |
| Route B | `4d1d449f5bc4f6a77fc96b570b5007d2183f239718b7198d3ad999eae1669097` | `555e59da6aa3cccecc919fa3fc086a180a69d88852d9013dd56fc830326971cf` | marlin / marlin | 283.577467 |

Every artifact is a finite, normalized float32 tensor of shape
`[32, 248320]`. The BF16 collector's `checkpoint_sha256` field intentionally
binds the source index, not an aggregate hash of all BF16 shards.

The RTN identifiers are intentionally named rather than conflated:

- structural checkpoint fingerprint from the original qualification:
  `eadbf27940485f0c9200e3189a8db03dee0a5380f274a80d182a63ca44abdcd5`;
- aggregate identity supplied to this collection:
  `78d675492db34c8c17954490be9d29b957a032680ce25b110b732f1e0d9f1af8`;
- manifest SHA-256:
  `91a1a1c5ead3ba7a42fa0800aaa559b960010650bb8bd2010203f1c30fd2fb6c`;
- run-file SHA-256:
  `0802cb78507aa2553395cbcc7e411c195d0148c31ffb6a2224248ceec244a711`;
- integrity-ledger-file SHA-256:
  `efa7f73b5f9a00f92c9db5ef3520ca1aa17850b0d9661853e5f47e6419fdba4b`.

## Provenance amendments and execution status

The first calibration outputs are superseded and were not used in this result.
Their reproducibility record was malformed and referred to a post-run
compatibility edit rather than the exact executed calibration CLI bytes. The
v2 plan preserved that prior record under SHA-256
`2404187b664d11db8ad0fdb23e46f5ad44afeb6aadfb88d35d4f33edf22a3475`,
froze the source again, reran both calibration replicas, and admitted only the
v2 artifacts listed above.

The working branch was based on commit `309cfe6`, but the experiment code was
not represented by that commit alone. Exact executed bytes were bound by:

- frozen-source file-manifest SHA-256:
  `d02b8fe64ced84797ac1068b6e1ff16b08864eb302a443b64ba8e6535a417e9c`;
- v2 pre-run plan SHA-256:
  `d5b1c29adc0a5fcd01cb961f2a06a4b664ce991591712c934ffc8b08210c5ebf`;
- calibration CLI SHA-256:
  `7d9e7f3f11287aeab98c810ce36bd25518309b08b14169e88d492056018ccf40`;
- v2 calibration runner SHA-256:
  `ad7aff68097722b776ff27fcc39d935fde8b21bdb9a9208abf4093aa3e5cb9a0`;
- frozen overlay builder SHA-256:
  `438f55c7f52de01dc3ec4b993189ba410260c3646fc161c838cfcdf3e96b4500`;
- held-out context evaluator SHA-256:
  `c5e5d2f3b6e3479890906e3db44442f7e8e44def7fb503830968373761f2b0ed`;
- exact-KL gate analyzer SHA-256:
  `25b655af7b4e1646affae2248c976e0d2cbaea62293b539ca99d787e28a70392`.

This `base commit + frozen byte manifest` record is the honest implementation
identity. It is a disclosed deviation from the pre-registration's preference
for one implementation commit; reporting `309cfe6` alone would be incorrect.

All four BF16/RTN/route-A/route-B collectors completed with status `0` and
wrote complete artifacts. The outer exact-KL driver ended with status `123`
only because its final host-side hash step could not read root-owned
safetensors. Permissions were repaired with `chmod a+r` inside a container;
the artifact bytes did not change. A new collection manifest was then written,
whose SHA-256 is
`948da503c949f1bc8037595f6f2c2a4187ab9cea0886644731e0b60b59fb1669`.
The original status `123` and the permission-only postcondition repair are both
retained. This must not be described either as a clean driver exit or as a
model/evaluation failure.

The executed exact-KL runner had SHA-256
`08e6fd84ec7d71dc721f3cfb249a77ce61a3a82072f1a83d7f9cc530bb5ecc57`.
The local runner was patched afterward to fix output permissions, so its later
hash is not the executed identity.

## Stop and retained value

The experiment stopped at the frozen quality gate. It did not run post-hoc
PPL, tasks, long-context tests, throughput, candidate smoke requests, another
calibration sample, a lower ESS threshold, alternate scale-search settings, or
a full 40-layer calibrated conversion.

The retained value is narrower and engineering-oriented:

- actual normalized route coefficients and actual expert inputs were captured
  in a bounded layer-streaming path;
- the real post-activation `down_proj` input was calibrated rather than
  approximated from the layer input;
- two independent calibration replicas produced structurally valid standard
  MXFP4 overlays with exact fallback;
- the fail-closed paired evaluation exposed split instability before a costly
  full conversion or publication attempt.

The scientific conclusion is conservative: this route-conditioned diagonal
statistic was not a reliable scale-selection signal for the frozen Nex scope.
It should not be swept or promoted under the same hypothesis.

## Retained evidence

- Local authoritative gate report:
  `benchmarks/nex-n2.5-mini-route-diagonal-gate-2026-09-22.json`.
- Local compact summary:
  `benchmarks/nex-n2.5-mini-route-diagonal-probe-2026-09-22.json`.
- Local 43-file evidence checksum manifest:
  `benchmarks/nex-n2.5-mini-route-diagonal-evidence-2026-09-22.sha256`, with
  SHA-256
  `d9f30f803687ae8bb55a57e56abf947b3702525aff2a02112556c3ce4af02229`.
- Spark experiment root:
  `/home/kirya/local-spark/experiments/nex-n2.5-moe-preflight-20260922`.
- Route evidence root:
  `/home/kirya/local-spark/experiments/nex-n2.5-moe-preflight-20260922/route-diagonal`.
- Final calibration artifacts:
  `route-diagonal/results-v2/replica-a-32.safetensors` and
  `route-diagonal/results-v2/replica-b-32.safetensors`.
- Candidate overlays: `route-diagonal/candidate-a-v2` and
  `route-diagonal/candidate-b-v2`.
- Held-out evidence:
  `route-diagonal/heldout-kl-20260922`.

No model was uploaded and no publication claim is made.
