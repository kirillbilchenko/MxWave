# CT drafter: verified features and context-projection pilot

**Native comparison live, 20 October 2026, 14:35 UTC:** CPU preflight passes for all three
loads on both Sparks; all 211 bindings per rank match. Controller phase
`starting_a1`: original production is temporarily paused and the original-drafter
CT baseline is loading on private port 8020. Wrapper PID **2969601** and independent
restoration watchdog PID **2969609** are live. The 40-minute experiment deadline
is followed by automatic restoration of the original 500k-context production.
No native candidate measurement exists yet. CPU work is committed as `f4e0b2f`
and verified on Spark2, including all 47 pilot archive payloads/checkpoints.

Updated **20 October 2026**. Feature capture is complete. Original production was
restored at **14:03:27 UTC**, with its original container identities, configuration
and **500,000-token context limit**. The bounded CPU pilot is complete. Step 24 is selected and exported for native
development testing; its native serving comparison is now loading the first baseline.

The fixed target is **CT MXFP4 experts plus the previously tested native MXFP8
KDA conversion**. That KDA conversion is lossy; original BF16-KDA CT remains the
lossless reference. No further target quantization changes are made here.
The failed cross-target KL gate and the failed selector-only pilot remain
retained. This experiment does not waive either.

## Training result and next native check

The run completes all 32 updates in **177.386 seconds**. Step 24 passes the frozen
offline selection; step 32 fails its chat no-regression requirement.

| Validation mean accepted draft prefix | Original | Step 24 | Step 32 |
| --- | ---: | ---: | ---: |
| Code, four prompts | 4.25 | 4.375 | 4.3125 |
| Chat, two prompts | 2.75 | 2.875 | 2.625 |
| Explanation, two prompts | 4.25 | 4.25 | 4.25 |

These represent only 16 code anchors and eight anchors in each other domain.
Code improves by two accepted draft tokens across its 16 anchors. This is a
small development signal, not evidence of the required 8% serving gain.
Code CE at step 24 is 1.61687 versus baseline 1.61515: it does not improve even
though the selected discrete acceptance metric does. No broad quality claim.

The independent review reproduces **1,120 proposal-token comparisons**, **135
acceptance fields**, all checkpoint hashes and the same selected step. It detects
about 2% relative error in the FP32-reduced weight-distance statistic. Separate
NumPy and Torch FP64 reductions agree on **0.0001171596963** for step 24, with
**5,864,036 changed BF16 elements**. Original reported values remain retained;
the corrected distances do not change the frozen 0.01 gate or selection.
Future trainer and native-export distance checks now accumulate in FP64.
The exact executed V3 trainer remains retained beside its report.

Native export preserves the original file layout and all **115 other tensor
payloads**, including FP8 codes and scales, plus the exact config bytes. Only
`fc.weight` changes. Export SHA-256:
`bf67cb8c2a02e3999269294b7d2b6fc63ce52ebd694fecbd8dac56f5971bdf47`.

A bounded original/candidate/original development run is now live on private port
8020: the same eight validation prompts, six shuffled repeat cycles per load,
two excluded warmups, C1, 256 fixed outputs, seed 42, temperature zero and seven
drafts. Both ranks must match all 88 retained effective drafter tensors except
FC. Runtime graph descriptors are counted; no feature tensor copying is enabled.
The original production restoration watchdog is retained, with a 40-minute
experiment deadline plus restoration warm-up. Repeated baseline loads diagnose
load drift; the single candidate load cannot satisfy final speed qualification.
Plan SHA-256: `5dd97e009313b9ec390ad07933c99fe16ce17a2ae9ceae5ad582bba487dbd9ba`.

## Capture result

| Check | Result |
| --- | --- |
| Captured prompts | 32 training, eight validation |
| Total API requests, including controls | 44 completed |
| Fixed output budget | 128 tokens per request |
| Before/after controls | Both match token sequences and speculative counters exactly |
| Native graph frames | 1,351 per rank |
| Archived payloads | All 5,456 hashes match |
| Accepted exported feature rows | 8,210 |
| Serialized training/validation export | 336,434,650 bytes |
| Rank-to-rank feature comparison | Bitwise equal |
| Separate raw-safetensors NumPy audit | 336,281,600 BF16 elements, zero differences |
| Excluded work | 5,247 rejected rows; 161 post-cap sampled tokens and feature rows |
| Sealed test | Unopened |

Features concatenate target layers **5, 14, 24, 33 and 42**. Each accepted input
token has one BF16 feature row, except the final capped response token. Input IDs
must match the frozen prompt followed by the actual API response. The independent
audit derives accepted intervals from subsequent query positions rather than
trusting sampler counts.

Capture used the original untrained FP8 DFlash2 checkpoint. Instrumented timing
is not a speed measurement.

## First pre-training failure and execution correction

V1 stopped before any optimizer update: four packed proposal blocks versus
individual-block forwards differed by **0.0145083 relative L2**, exceeding the
fixed 0.005 preflight limit. Its source and failure report are retained.

Independent mask checks pass. Zeroing the masked future context leaves the
first packed block bitwise unchanged. The first block differs from isolated
execution by 0.0017001 relative L2; among traced projection outputs, the first
difference is in layer 4's attention output projection (nine elements).
This supports shape-dependent BF16 arithmetic as a cause; it does not establish
every internal operation as identical or explain every later block's difference.

V2 uses **one independent eight-query-row forward per anchor**. Explicit masks
must reproduce the upstream single-block inference masks bitwise. The baseline
must also repeat exactly before training. Both checks pass. The data, losses,
optimizer settings and selection criteria remain unchanged. V1 is preserved;
the failed packed execution is not used for training.

## CPU derivative correction

V2 completed one update, taking **169.925 seconds** for the first step. A native
CPU stack sample locates the delay in PyTorch's BF16 GEMM backward fallback.
The run was stopped and retained after that completed update; it was not used
as the starting point for the next run.

V3 restarts from the original checkpoint. Linear forwards still call the same
BF16 `F.linear`. Their analytical derivatives use FP32 matrix multiplication,
then cast gradients back to the original tensor dtypes. This changes backward
accumulation rounding; exact gradient equality is not claimed. Independent
FP64-formula and gradcheck tests cover input, weight and bias derivatives.
The real first-prompt check preserves its loss and all proposal metrics exactly,
with finite gradients and **2.222 seconds backward**. Gradient L2 is 5.90428 versus
5.90611 in V2. All eight validation prompts reproduce V2's baseline exactly.
This is a CPU training implementation change, not a serving kernel change.

## Bounded training contract

Only the existing **83,886,080-parameter `fc.weight` context projection** changes.
Its master copy is FP32, and each forward/export uses BF16. All original model
parameters remain frozen, including the selector and backbone. The original FP8
codes/scales, target head, embeddings and target model remain fixed.

- One seeded pass: at most 32 updates, one prompt per update.
- Four deterministic anchors per prompt; four separate model forwards.
- Each anchor has seven CT-verified labels. Labels may end at the first EOS;
  post-EOS text is excluded. Short responses are reported and skipped.
- Mean full-vocabulary cross-entropy plus mean selector cross-entropy where the
  CT label is already in the strict unary top 16. No label injection.
- AdamW: four-step warm-up to 1e-5, cosine decay to 1e-6, zero weight decay,
  gradient clipping at L2 1, seed 42.
- Validation at baseline and steps 8, 16, 24 and 32, plus an earlier final step
  if short-response skips reduce the update count.
- Selection requires strictly better code mean accepted-prefix length, at least
  98% of baseline chat and explanation prefix lengths, and weight relative L2
  at most 0.01. Ties prefer lower code CE, then the earlier step.
- No improving candidate means no promotion and no sealed-test access.

Eight validation prompts are a development screen. Dense CPU arithmetic differs
from native FP8 Marlin and serving KV/head adapters. An offline improvement is
not an end-to-end speed result. A selected checkpoint still needs exact native
export/reload, new native validation, and the unchanged sealed serving gates.

The frozen full serving target remains at least **8% higher C1 code throughput**
against the original drafter on the same CT target, with the specified confidence,
repeatability, no-regression and round-count criteria. Production replacement
also requires serving capacity/tuning, stable C8/C16 measurements and long-context
qualification.

## Retention and reproduction

Local capture root: `out/drafter-feature-capture-v1/`.
Raw evidence is under `evidence/`; accepted sequences under `verified-features/`.
The independent report is `numpy-audit.json`.

Remote capture root on Spark:
`/home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/full-model-v1/drafter-feature-capture-v1`.

The original post-restoration finalizer opened one SCP connection per rank-one
file. Only that owned copier was interrupted after production restoration.
The replacement transferred one archive, verified all bytes and resumed the
original identity/manifest checks. `run-prepared-status.json` retains the original
finalizer's `-15` exit; this is the documented transfer interruption, not a failed
GPU capture. Both original and replacement finalizer sources and transfer receipt
are archived.

| Artifact | SHA-256 |
| --- | --- |
| Capture plan | `2e3dfcf8f61eadbbb7ae4092fed661da986daec7ed1d88921f5d771156bd4c7f` |
| Evidence archive, 1,403,252,349 bytes | `cab5f0cea65a88f54318d2dc7386980863382526cad41336731bb795a2f02f2c` |
| Accepted export manifest | `d79452e7dd86e75cdbd710c5afd4ebfa3aed3122633fa5dde8d0d8d56dc9cd62` |
| Frozen V1 pilot protocol | `14570e1e7c347356545241ddcb7e55b08a76a45c3c616752fbdd6e6b8cac626a` |
| Frozen V2 execution protocol | `54cf80b9ad70ec01e72f5838b5ec86499933808eac88247e8066e551e08a4dbf` |
| Frozen V3 CPU derivative protocol | `d5a1f0aca11182d48e32349fcd53fef6ad1a80315c904e4653e1a9f655775720` |

Pilot root: `out/drafter-context-pilot-v1/`.
`run-v1/` retains the failed preflight and executed source.
`run-v2/` retains the stopped slow CPU run and its executed source.
`run-v3/` holds the completed report, executed source and four BF16 checkpoints.
The report records every update, evaluation, source hash and selected candidate.
The full sealed dataset is never an input. The evidence archive is also retained
on Spark2 at the same capture-root path; its full SHA-256 matches Spark and Mac.

The native exporter accepts only a completed, validation-selected run. It patches
only the existing BF16 `fc.weight` payload into a copy of the original FP8 file,
verifies every other tensor and the config byte-for-byte, and rejects absent or
out-of-bounds candidates. It does not reload or promote a serving model.

```bash
PYTHONPATH=out/drafter-preflight-v1/audit-deps ../../.venv/bin/python \
  -m integrations.glm53_mxfp4.train_drafter_context_pilot \
  --dataset out/drafter-feature-capture-v1/verified-features \
  --checkpoint out/drafter-preflight-v1/roundtrip-v1/trainable \
  --checkout out/drafter-preflight-v1/audit-sources/SpecForge \
  --weights out/drafter-context-pilot-v1/weights \
  --protocol benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/drafter-context-pilot-v3-protocol.json \
  --output out/drafter-context-pilot-v1/run-v3
```

Output directories must be new; never overwrite a retained run.

Source verification: Ruff and strict core mypy pass; **906 tests pass, one
optional dependency skip**. The full native comparison still requires the actual
serving-image preflight and successful runtime checks.
