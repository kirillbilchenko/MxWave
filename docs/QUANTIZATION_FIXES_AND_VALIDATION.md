# Quantization corrections and fresh validation

## Current behavior and published artifacts

The current mathematical revision is `mxfp4-rne-v2`. It corrects exact E2M1
midpoints to round to the even neighbour and evaluates Qwen3.5's norm proxy as
`abs(1 + stored_weight)`. Both corrections can change newly converted weights.
The norm fallback requires explicit `--gamma-proxy` and is used only when MSE
runs have no real calibration artifact. Uncalibrated MSE is unweighted by default;
H64 uses captured block Hessians and bypasses the proxy. H64 can still change because
of corrected rounding, including a different winning scale when weighted
errors differ at ties.

Run identities and manifests record the revision. Resume refuses older
identities instead of combining shards produced by different mathematics.
The emitted packed uint8 weights, biased E8M0 scales, and compressed-tensors
configuration retain their format.

The published H64 model and checked-in benchmark JSON are historical artifacts.
Their payloads and scores were not changed by these source fixes. Exact replay
of their bytes requires release commit `0179e131544f807ecdb6d04e7e914f181dd21c9f`.
Corrected-code chunk measurements and the six-model whole-model PPL, KL,
and decode comparison have completed on Spark. Do not transfer historical quality
claims to a new conversion. The dated experiment record is
[corrected controls on Spark](CORRECTED_CONTROLS_2026-10-03.md).

## Engineering corrections

| Area | Current contract |
|---|---|
| Norm fallback | Off by default; explicit opt-in uses the effective Qwen multiplier and records its offset |
| Rounding | Ties-to-even at all seven positive and negative E2M1 midpoints |
| Coverage | Explicit exclusions, exact Qwen target names, and a rejection of fused banks in the dense path |
| Chunk residency | Both row and element limits; MSE evaluates candidates sequentially |
| Concurrency | One exclusive writer per output directory, acquired before stale-shard cleanup |
| Source identity | Full source-shard, config, and index hashes in both engines |
| Resume/durability | Shared SHA-256 ledger, durable pending-shard journal, and synced renames |
| Expert emission | Direct chunk writes; unit cap checked before payload loading |
| Input configuration | Malformed JSON and invalid object types raise errors |
| Calibration attention | Missing helpers fail; eager attention requires an explicit causal mask |
| Calibration sinks | Optional `--skip-first-tokens`, applied to every window and recorded in provenance |
| Hessian damping | Configurable absolute damping, or per-block relative damping |
| Composed metrics | Coverage uses the remaining expected modules; missing measurements remain missing |
| Rotation primitives | Correct output-coordinate orientation, Haar QR signs, and 32-wide blocks supporting width 5120 |

`--tensor-chunk-max-elements` defaults to 1,048,576 in both converters. Rows
are bounded by `min(row_limit, element_limit // input_width)`; a row larger
than the element budget is rejected during planning. This bounds the input
size, while allocator, device workspace, calibration tensors, and cache use
remain additional. The expert host cap bounds planned output payload per unit;
production emission streams the payload rather than retaining a full shard.
The persistent `.mxwave-output.lock` file is bookkeeping, not a checkpoint
tensor. Kernel locks are released when a process exits; the file stays in place
to avoid inode races. A lock alone is treated as an empty output directory.
The lock uses POSIX `fcntl`, matching the Linux Spark target.

Before a complete shard is renamed, its hash and run identity are durably
recorded in `mxwave-shard-pending.json`. Resume verifies that intent against the
renamed file, completes the integrity-ledger update, and clears the journal.
A crash before rename regenerates the incomplete shard. Files without a
matching ledger record or journal remain untrusted. Source timestamps are no
longer part of dense run identity: unchanged bytes can resume after a timestamp
change, and changed bytes are rejected even if size and timestamps are preserved.

`--hessian-damp` was already configurable. The historical default remains an
absolute `1e-6`; `--hessian-damp-mode relative --hessian-damp 0.01` instead adds
1% of each block's mean diagonal to that block's diagonal. The mode is recorded
in calibration artifacts and manifests. Relative damping has not received a
quality ablation; the current H64 control uses its historical calibration.

Rotation remains experimental and disabled for checkpoint emission. A valid
production implementation must preserve the full model function and satisfy
its inference runtime's transform contract. Full GPTQ error compensation,
AWQ channel scaling, and MR-GPTQ integration remain separate research work.

## Why historical evaluation needs qualification

The 128 KL contexts are prefixes of a subset of the 316 WikiText test windows
used for the headline PPL. KL scores one next-token position per context.
Top-1 agreement was 120/128 for H64 and 117/128 for AMD. The confidence interval
for the KL difference crossed zero.

Precision-budget selection used the first 32 contexts, and its 96-context
confirmation used the remaining contexts. These were disjoint within that
experiment but had already been evaluated in earlier screens. Earlier recipe
probes also used the WikiText test corpus. Paired bootstrap intervals do not
account for those adaptive decisions. Historical scores remain useful paired
observations; they are not evidence from a globally untouched holdout.

The historical `abs(weight)` fallback artifact is not a valid ablation of the
corrected norm proxy. The new fixed validation comparison includes corrected
RTN, unweighted MSE, corrected norm fallback, H64, BF16, and AMD. It confirms a
modest H64 benefit, while the corrected norm fallback does not improve plain
MSE on these data. The remaining reserved windows have not been scored.

## Fresh comparison protocol

1. Freeze the source checkpoint, current code commit, runtime image/backend,
   tokenizer, candidate settings, and evaluation protocol before collecting results.
2. Keep calibration data separate from recipe-selection validation data.
   Reserve final evaluation data before any recipe or precision-budget choices.
   Previously inspected WikiText test windows cannot become a new final holdout.
3. Convert the same target modules with RTN, unweighted MSE, corrected norm-proxy
   MSE, and H64. Include BF16 and the same AMD reference. Hold scale percentile
   and clip depth fixed for the three MSE variants, with identical calibration
   sequences for calibrated variants.
4. Use validation data for choices. Freeze the selected recipe and candidate
   allocation, then evaluate the reserved final corpus once. Record negative
   results as well as positive ones.
5. Report paired full-sequence PPL, adequate KL positions, reconstruction SQNR,
   task checks, peak memory, and controlled throughput on the same hardware.
   Preserve per-window/token identities, checkpoint hashes, and uncertainty.

Example controls using the 400-module policy and H64's historical clip depth 4:

```bash
mxwave-quantize --model-dir /models/float --output-dir /models/rtn-v2 \
  --policy qwen3.8-27b-compatible --method rtn --no-gamma-proxy

mxwave-quantize --model-dir /models/float --output-dir /models/mse-v2 \
  --policy qwen3.8-27b-compatible --method mse --no-gamma-proxy \
  --scale-percentile 99.5 --mse-clip-depth 4

mxwave-quantize --model-dir /models/float --output-dir /models/proxy-v2 \
  --policy qwen3.8-27b-compatible --method mse --gamma-proxy \
  --scale-percentile 99.5 --mse-clip-depth 4

mxwave-quantize --model-dir /models/float --output-dir /models/h64-v2 \
  --policy qwen3.8-27b-compatible --method mse \
  --scale-percentile 99.5 --mse-clip-depth 4 \
  --activation-stats /data/h64.safetensors --calibration-objective block-hessian
```

Keep sink-token exclusion disabled for the first corrected H64 comparison so
only rounding changes relative to its historical calibration. Test
`mxwave-calibrate --skip-first-tokens 1` as a separately declared ablation;
excluded tokens still propagate through the decoder but do not enter statistics.

Prepare contexts from a fresh split and audit against prior reports:

```bash
python scripts/evaluate_next_token_kl.py prepare \
  --model /models/float --corpus /data/fresh-validation.txt \
  --dataset-name DATASET --dataset-revision IMMUTABLE_REVISION \
  --dataset-split validation --num-contexts 128 \
  --exclude-evaluation /runs/old/contexts.json \
  --exclude-evaluation /runs/old/bf16-ppl.json \
  --output /runs/new/validation-contexts.json
```

The exclusion audit rejects identical corpus bytes even with changed window
boundaries, and exact window/text or token hashes across different corpora.
It records the excluded report hashes and refuses to overwrite a frozen context
manifest. Supply every relevant prior evaluation report; this audit cannot
certify unknown earlier use, dataset-wide independence, or semantic duplicates.
Precision-budget evaluation also checks that selection and confirmation context
hashes do not overlap and binds the final partition to its frozen selection
report. Dataset provenance documents declared splits; it does not prove their
contents were unseen.
