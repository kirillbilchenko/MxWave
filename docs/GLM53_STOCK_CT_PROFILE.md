# Selected CT profile on stock vLLM

**21 October, 20:15 UTC:** the correction is now packaged as `prepare-drafter`,
with pinned config/weight validation in `serve`. Both official-image CPU preflights
and real-checkpoint preparation pass. The fresh production-setting comparison
is prepared, including a compatibility stage before timing and both full GSM8K
arms. Consult the [handoff](GLM53_HANDOFF.md) for launch/restoration status.

**Latest, 21 October 2026, 19:56 UTC: zero acceptance has a verified loading fix.**
Native 0.32 incorrectly selected twelve existing BF16 drafter convolution matrices
for FP8 loading without scales. A separate config adds explicit BF16 exclusions;
weights remain byte-identical and vLLM source is unchanged. The two-prompt, 32-token
V2 diagnostic restores **27/40 and 25/50 accepted proposals**, with **8 and 10 rounds**
instead of **31**. Native graph replay and draft-eager agree on counters and outputs.
Both ranks' 64 observed score/context frames and 32 eager hidden/logit frames are
finite. Exact original 500k production was restored at **19:55:15 UTC**.

This fixes the observed invalid drafter computation, not the separate target KL
or output-repeat failures. Package CLI integration, corrected-profile memory and
untraced timing, 500k/prefix-on compatibility and full GSM8K are still pending.
All prior failures and allocations below are historical measurements before this
correction. [Diagnosis, reproducibility and archived evidence](GLM53_STOCK_DRAFTER_LOADING.md).

**V3 cache isolation result:** actual BF16 cache on all six drafter layers still gives
**0 accepted / 3,810 proposed tokens** across the same six requests. Startup allocation
is 172.490866 GiB TP2. FP8 draft cache alone is not the cause. Exact original production
was restored at 18:29:47 UTC. Evidence: `stock-profile-v3-evidence.tar.gz`, 31 payloads,
SHA256 `0d2c6ea3610c6f4b303acdbbd1df44ad2ab1c84e4d60c830d6746714029d3cc4`.


Prepared 21 October 2026. This check matches the **last selected Modal CT profile**,
not the older Inco-based CT/ModelOpt comparison that saved 14.877 GiB. Its purpose
is to establish the complete CT model with its drafter and compressed vocabulary
on official vLLM 0.32.0, and measure actual allocation after initialization.

V1 launched **17:58:20 UTC**, controller PID 2135666, plan SHA256
`147b5088945068cee5f18d967bb5a2199b98213222e30b7c21cf4379a6cec492`.
Startup succeeds with **172.495512 GiB across TP2**, **0.750561 GiB more** than
the historical selected Modal profile. The read-only observer then fails on an
obsolete runner attribute before any client generation. Exact original production
is restored at **18:05:35 UTC**; retain V1's failure and valid startup measurements.

V2 launches **18:06:13 UTC**, controller PID 2204242, plan SHA256
`fcd5f0aa12b935f47771b1de2715eabdfab9b75e8671afe909022f4a5f2aad05`.
The observer now uses native `runner.speculator.model`, with its complete CPU
contract exercised against actual native class instances on both nodes. Nothing
in model computation changes. Check V2's `controller.json` for live restoration
status. All six requests subsequently complete with **zero accepted proposals**.
Exact original 500k production is restored at **18:14:28 UTC**, with both container
identities/configuration hashes unchanged. The combined V1/V2/MTP-CPU archive has
82 payloads independently verified on Mac and Spark2; archive SHA256
`faa76f88fc593c3c192b9572953dad3668bcaf3d88eec10360830fee26655545`.
Local review: `out/glm53-package-v1/stock-profile-v1-v2-independent-review.json`.

## Collected results

| Check | Result |
| --- | --- |
| Allocation after drafter, KV and graphs | 86.248634 / 86.246878 GiB; 172.495512 GiB total |
| Difference from historical selected Modal CT | +0.750561 GiB total (+0.44%); not a matched production delta |
| Native proposer / model | `DFlash2Speculator` / `DFlash2Qwen3ForCausalLM` |
| Shared target/draft vocabulary | Head and embedding shared by object identity on both ranks |
| C1 proposal length | Exactly five: 635 proposals / 127 calls per request |
| Accepted tokens | **0 / 3,810 overall**, zero at all five positions |
| Completed work | Six requests, 128 output tokens each; code, chat, explanation twice each |
| Repeat identity | Code differs; chat matches; explanation differs |
| Raw request times | 9.933–10.445 seconds; no controlled throughput claim |

The retained stage receipts locate the extra **0.750561 GiB**: **0.149785 GiB**
is already present immediately after KDA conversion/model loading; **0.600776 GiB**
appears during later initialization. Individual KV layouts, CUDA graphs and
runtime workspaces are not separately measured, so this is a stage breakdown,
not attribution to one subsystem. Both configured KV budgets remain 6 GiB/rank.
Expert-scale and vocabulary retained bytes are identical on both ranks. All 68
KDA weight/scale shapes, dtypes, storage sizes and rounded hashes match the old
receipts. Expert scratch actually falls by **24 MiB total**; this is already
included in the allocation difference. Reserved-but-unallocated allocator space
increases separately by **0.142017 GiB**, excluded from the 0.750561 GiB figure.
Independent arithmetic: `out/glm53-package-v1/stock-profile-memory-breakdown.json`.

This is a failed speculative-execution diagnostic despite successful generation.
The same pinned drafter previously had useful acceptance in the old runtime;
zero across all positions requires a native-path investigation, not retraining
or declaring five drafts ineffective. Inspect actual auxiliary features, draft
parameter loading, KV storage and proposal/verification alignment. The earlier
native dense-head V7 diagnostic also retained zero acceptance, so the compressed
head alone is not established as the cause. The separate target KL failure remains.

| Setting | Previous selected CT | Stock profile check |
| --- | --- | --- |
| Target | CT MXFP4 experts + MXFP8 KDA | Same checkpoint and verified rounded KDA values |
| Lossless packing | Expert scales, embedding, output head | Same three components |
| Drafter | Unmodified Modal FP8 DFlash2 | Same pinned weights |
| Draft tokens | C1:5 / C2:5 / C3–C16:3 | Same native policy |
| Context / slots | 8,192 / 16 | Same |
| KV / prefill budget | 6 GiB per rank / 4,096 tokens | Same |
| GPU utilization / block size | 0.90 / 2,304 | Same requested values |
| Prefix cache | Off | Off |
| Requested graph sizes | 1, 2, 4, 8, 16 | Same; actual resolved sizes must be retained |
| Target and draft vocabulary | Shared compressed head and embedding | Must verify sharing after startup |

Explicit differences: old patched runtime versus official vLLM 0.32.0; native
Triton draft attention/FP8 cache versus the old FLASH_ATTN/`auto` draft cache; old sparse
ordering, expert dispatch, RoCE and admission overlays are absent. Run:ai loads
weights on stock; loader time is not a speed comparison. This checks how close
the complete deployment is, without claiming an isolated vLLM-version effect.

**Draft-cache correction:** the old selected Modal plan and launch log explicitly
set `speculative_config.kv_cache_dtype=auto`. Its `GLM_LV_DRAFT_FP8_KV` helper
builds BF16 context-projection buffers from FP8 weight values and scales; that
name does not establish an FP8 cache. The earlier description conflated FP8
projection weights with KV storage. Stock V2 explicitly uses `fp8_e4m3` and logs
default cache scales of 1.0. This is a concrete configuration difference, not yet
a proven cause of zero acceptance. V3 retains V2's six frozen requests, weights,
backend, policy and graph settings, changing only the draft cache to `auto`,
with an untimed report of actual per-layer cache dtype and shape.

The historical baseline is the unmodified `original_a` arm of
`full-model-v1/drafter-modal-native-v1`, independently retained on Mac. After
drafter, KV and graph startup it allocated **85.888963 / 85.855989 GiB** on ranks
0/1, **171.744951 GiB total**. These are the selected Modal profile's figures,
not the earlier 170.844 GiB Inco-based CT startup allocation.

The frozen diagnostic uses one previously used development prompt each for code,
chat and explanation, two repeats each, 128 output tokens, temperature zero and
seed 42. It retains actual speculative counters, per-position acceptance, all
responses and before/after worker allocations. Elapsed request times are raw
smoke diagnostics, not a qualified throughput comparison. Neither sealed data
nor training is involved. Existing numerical gates remain unchanged; the stock
500k profile's repeatability failures are retained and not cleared by this run.

Operational root: `full-model-v1/glm53-package-v1/stock-profile-v1`. CPU validation
precedes any GPU pause. A detached controller and independent 1,500-second
watchdog restore the exact original 500k production containers. This 8k check is
private, never a production replacement and never evidence of 500k correctness.
