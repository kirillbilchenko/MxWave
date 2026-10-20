# Bounded CT drafter calibration pilot

Completed 20 October 2026. **The smallest selector-only pilot fails its sealed-test
screen. No patch is applied and no serving-speed improvement is claimed.** Original
production was restored at **12:30:49 UTC**, healthy on port 8000 with its original
identities and 500,000-token context configuration. No experiment or training remains
running. The 52-request capture, CPU fit, independent review and archival are complete.

The previous paired diagnostic found an early CT proposal error despite both
targets agreeing on the correct next token. All recorded drafter weights matched.
The compressed vocabulary head was exact on the tested states. This supports
adapting the drafter to CT, without establishing that adaptation will recover the
whole measured 8.61% code-speed gap.

## Completed result

| Fixed-state metric: mean consecutive matching draft tokens, out of seven | Original projection | Selected BF16 projection |
| --- | ---: | ---: |
| Validation, eight prompts | 2.903799 | 2.931239 |
| Sealed test, eight prompts | 2.393875 | 2.383249 |
| Test code prompts, six | 2.409049 | 2.404405 |
| Test explanation prompts, two | 2.348352 | 2.319780 |

Step 75 was selected using validation only, before opening the test split. The
validation gain did not carry to the test prompts. Both test domains declined,
so the unchanged improvement/no-regression gate fails. The sample is small; this
is not evidence of statistically established degradation or a general failure of
drafter adaptation. Target weights and inference precision were never changed.

The selected projection changes 859,908 of 1,048,576 BF16 elements, with relative
L2 update norm 0.00883388 (0.8834%, below the 1% limit). Its separate artifact is
retained for reproducibility, with no native runtime installation or live speed test.
Patch SHA-256: `3ee756d92385449decb5d94f4b13527c7b1af18acc68b71204c3e0f47cb882d1`.

The capture has **1,819 native graph replays**, reconstructing all **1,735 verified
draft rounds** and their seven per-position acceptance counters. All 84 post-cap
replays are retained. Complete seven-token training-format blocks: **1,665**
(1,109 training, 261 validation, 295 test); 154 incomplete tail frames are counted
and excluded from that format. Both native before/capture/after controls pass.
The selector's three source tensors match both ranks and the earlier paired audit.

All **3,682 archive payloads** verify. NumPy, independently of the PyTorch trainer,
recomputes **108 numerical fields** from the saved BF16 patch and captured states.
The original FP32 surrogate differs from native proposals on 21/1,827 validation
and 8/2,065 test draft positions. Both training arms use the same surrogate; this
rounding caveat is retained and does not convert the failed offline gate into a pass.

There is still ranking headroom in this fixture: at the native path's first test
error, 237 blocks have the correct token in the top-16 candidates and 38 do not.
This is a fixed-state diagnostic, not an achievable speed estimate. The small
projection-only update did not learn a reliably better ranking on held-out data.
A larger fresh corpus or broader adaptation remains untested; do not tune further
against this already-opened test split and call it sealed.

[Independent verification](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/selector-calibration-fit-v1-independent-verification.json),
[training review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/selector-calibration-fit-v1-review.json),
and [NumPy review](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/selector-calibration-fit-v1-numpy-review.json).

Persistent local capture: `out/selector-calibration-capture-v1/evidence`;
fit: `out/selector-calibration-fit-v1`. Spark retains
`full-model-v1/selector-calibration-capture-v1` and the original archive.
Archive: **224,025,100 bytes**, SHA-256
`7e8dad6d4a9c9c1163ee862fe8327630ca065c2d3892a83cc34a41143b048201`.

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

The pilot and capture verification are also committed as `d8abe78` and backed up
on spark2 as `glm53-d8abe78.bundle` in the same directory. Bundle SHA-256
`5b95d4c226f87b7aa2f9f08d007369552fd00c2a1fe895a323de714e7ba17652`
matches locally/remotely and `git bundle verify` passes.

The frozen capture source archive is
`/private/tmp/mxwave-selector-calibration-capture-v1-src.tar.gz`, SHA-256
`cfcf59d5f2f6083d4174def52d0f18cbbdfee370334e70adfb17ef71fc850daf`.
It contains 17 source/configuration files plus their manifest, no model weights,
credentials, or captured user conversations.

The failed fit artifact and reviews are backed up on **spark2** at
`/home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/git-backups/glm53-selector-calibration-fit-v1.tar.gz`.
Its 1,673,462-byte archive has matching local/remote SHA-256
`b04b7f9c641a1b84bc3600af2901d56eafc4e720caaec29510d69de45cb6d8bf`.
The final source/results Git bundle is `glm53-selector-result.bundle` in the same
backup directory. Restore the branch with `git clone -b feat/glm53-flash-mxfp4
/path/to/glm53-selector-result.bundle mxwave-restored`.

Repository validation: **879 passed, one optional-dependency skip**, Ruff and
strict core mypy pass. A final live check confirms original production healthy
with 500,000 context; no trial container is running.

## Production replacement gates

The user explicitly reaffirmed all four outstanding requirements on 20 October.
They are deployment gates, not claims satisfied by the selector pilot:

1. **Drafter calibration:** validate any selected projection with native kernels
   and separate live rollouts. Measure code/explanation acceptance and end-to-end
   latency with unchanged target precision. A better offline prefix score alone
   does not close the code gap.
2. **CT serving configuration and capacity:** match production's prefix caching,
   prefill budget (6,919), 32 sequence slots, long-context configuration and
   relevant custom adapters. Measure a KV allocation that uses the freed memory,
   actual resident requests and queueing under load; do not assume capacity scales
   linearly with the KV budget. The earlier offered C8/C16 cases admitted at most
   six resident requests. Keep settings and workload identical across comparison
   arms when attributing performance to a model or compression change.
3. **Repeatable concurrency results:** retain earlier failed controls and run
   interleaved C8/C16 comparisons with clocks/thermal telemetry, matched prompts,
   seeds, output policy, speculative configuration and cache state. Report actual
   concurrency, queueing and repeat controls. Current failed-repeat gains remain
   unconfirmed.
4. **Long-context qualification:** the candidate trials used an 8,192-token limit;
   production is configured for 500,000. Test actual increasing context lengths,
   retrieval/quality, first-token latency, memory and concurrent-request capacity
   up to the required serving range. Changing the configured limit alone is not
   qualification. Preserve original production and its 500k configuration until
   a candidate is qualified for the intended use.

No quality threshold has been relaxed. The previously failed 1e-5 cross-model KL
comparison remains failed; the practical quality evidence and any lossless claim
must remain separate. No production replacement is authorized by a diagnostic pass.
