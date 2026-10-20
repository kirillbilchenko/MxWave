# Bounded CT drafter calibration pilot

Prepared 20 October 2026. **Capture and training have not started.** The original
production service remains running. The automatic approval reviewer blocked the
49,313-byte source archive transfer to `spark` and `spark2`; explicit approval of
those destinations is pending. No data was transferred by the rejected command.

The previous paired diagnostic found an early CT proposal error despite both
targets agreeing on the correct next token. All recorded drafter weights matched.
The compressed vocabulary head was exact on the tested states. This supports
adapting the drafter to CT, without establishing that adaptation will recover the
whole measured 8.61% code-speed gap.

## Smallest useful intervention

Train only `model.candidate_selector.hidden_projection.weight`: a
256 × 4,096 BF16 matrix with **1,048,576 parameters**, occupying 2 MiB at serving.
The target, FP8 draft backbone, context projection, vocabulary head, embeddings,
and predecessor/successor codebooks remain frozen. Updating the existing matrix
requires no additional inference matrix or FP8 checkpoint conversion.

This pilot can repair some ranking errors when the correct token is already in
the strict top-16 candidate list. It cannot create missing candidates or directly
correct the backbone's feature representation. A failure does not rule out broader
context-projection or backbone adaptation.

The protocol follows the distinction between covered selector labels and
self-conditioned path metrics described in the
[SpecForge DFlash2 training documentation](https://sgl-project.github.io/SpecForge/basic_usage/training.html#dflash2).
Our trainer is independently authored tensor algebra; no external implementation
is copied or installed. It is not a SpecForge export or full DFlash2 retraining.

## Frozen pilot protocol

- 48 new authored prompts: **32 training, eight validation, eight sealed test**.
  Code and technical explanation appear in every split. Earlier speed prompts
  and GSM8K test questions are not training data.
- Native CT MXFP8 configuration, one serial request at a time, seed 42,
  temperature zero, seven draft positions, 128 output tokens.
- 52 API requests: one capture for every prompt, plus before/after controls for
  two prompts. Both tokens and speculative counters must repeat for the controls.
- Record native FULL-graph hidden states, candidates, scores, proposals, anchors,
  and absolute positions on rank zero. Do not store all five target feature layers.
- Use the reviewed restoration controller, private port **8018**, a 35-minute
  outage bound and a 2 GiB free-disk floor. Restore original production before
  CPU training on the Mac.
- At most **100 AdamW steps**, batch 16, learning rate 1e-5, seed 42. Train only
  covered gold labels. Cap the relative projection update norm at 1%.
- Select among BF16-rounded snapshots every 25 steps using validation prompts,
  requiring improvement overall and no regression in either domain. Open the
  sealed test once after selection, with the same domain conditions.
- Export only a separate 2 MiB parameter patch. Do not modify the original
  checkpoint or production configuration.

Labels come from each recorded CT continuation, aligned to the absolute draft
positions. Incomplete response-tail blocks are counted and excluded. Every
stored native selector walk must reconstruct its actual proposal exactly.

The training math uses an FP32 surrogate. Its baseline proposal differences from
the recorded native BF16 path are reported. The evaluation reuses baseline
histories, so even a successful offline screen is **not live acceptance or a
speed result**. Native-kernel validation and held-out end-to-end requests would
be the next gate before considering a broader experiment.

## Storage and preservation

Spark had about 3.7 GB free at preparation. The selected capture avoids the
40,960 bytes/token required by five complete BF16 feature streams. It stores one
copy of the 153.25 MiB selector weights plus bounded draft frames, instead of
duplicating those tensors across ranks. No old evidence is deleted.

All completed GLM work through the paired diagnostic is committed as
`edc4c08d8c3b4294dbb9263b4fe2b174befbf8c6` on
`feat/glm53-flash-mxfp4`. The complete-history Git bundle is copied to **spark2**:

```
/home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/git-backups/glm53-edc4c08.bundle
```

Bundle SHA-256:
`8332a4a40508f15960f86bad312f5e5ecace5a997d3e5e15f669c7d3f58af0dc`.
Local and remote hashes match. A new bare repository was restored from the
bundle and passed `git fsck --full` and `git bundle verify`.
Nothing was pushed to GitHub.

The new pilot is separate work after that checkpoint. Its source archive is
`/private/tmp/mxwave-selector-calibration-capture-v1-src.tar.gz`, SHA-256
`cfcf59d5f2f6083d4174def52d0f18cbbdfee370334e70adfb17ef71fc850daf`.
It contains 17 source/configuration files plus their manifest, no model weights,
credentials, or captured user conversations.
