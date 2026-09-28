# Recent second-architecture candidates — 2026-09-22

## Decision

The next-model track must not be dense-only. MxWave should evaluate recent MoE
models as first-class candidates because routed experts test an important and
currently unsupported part of the quantization contract.

There is no single current model that is simultaneously new, uncrowded,
stock-vLLM-safe, operationally cheap, and a strong proof of architectural
generality. Use a two-rung plan instead:

1. implement and validate generic fused-MoE support on a recent
   Qwen3.5-MoE-family checkpoint; then
2. qualify that machinery on a non-Qwen family that passes the stock-vLLM
   kernel gate before making a general MoE support claim.

This is a candidate audit, not authorization to download, quantize, publish, or
run a long Spark job.

Update: Nex-N2.5-mini was selected for the bounded Phase-0 MoE preflight on
`research/nex-n2-5-moe-preflight`. The primary target is MxWave MXFP4; the
existing NVFP4 checkpoint is an external comparator, not a second backend in
the same implementation scope.

## Freshness rule

Prefer models released or materially revised in July–September 2026. An older
model is not a preferred target merely because its adapter would be easier.
Mixtral, DeepSeek-V2-Lite, and similar mature fallback models are therefore out
of scope for the primary experiment.

Repository metadata and the quantization landscape change quickly. Re-check
the linked source card, exact revision, license, vLLM support, and derivative
model tree immediately before starting work.

## Shortlist

| Role | Candidate | Why it is useful | Why it is not sufficient by itself | Conservative effort |
|---|---|---|---|---|
| MoE implementation acceptance | [Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) | August 2026, 35B total / about 3B active, 256 routed experts, popular, BF16 source, and supported through the Qwen3.5-MoE path in stock vLLM 0.29 | It is a Qwen3.5-MoE derivative, so it proves fused-expert support but not architectural independence. Official NVFP4/FP8 and community MXFP4 artifacts make publication crowded | 1–2 day preflight; roughly 2–4 weeks for robust support and qualification |
| Newest Qwen-MoE validation | [Nex-N2.5-mini](https://huggingface.co/nex-agi/Nex-N2.5-mini) | Released 2026-09-08, Apache-2.0, BF16 source, about 70.2 GB, 256 experts with top-8 routing, multimodal/MTP, and a stock-vLLM usage path | Same Qwen3.5-MoE family; a measured [NVFP4 derivative](https://huggingface.co/primitive-ai/Nex-N2.5-mini-NVFP4) already reports 23.91 GB and near-BF16 task quality. Use as a fresh regression target, not the flagship claim | Little extra adapter work after Ornith/Qwen-MoE support; full qualification still costs days |
| Conditional cross-family flagship | [NVIDIA Nemotron 3.5 Lightning 30B-A3B BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16) | Released 2026-08-11; 30B / 3B active; combines Mamba-2, MoE, and attention. Its routed experts are ordinary per-expert 2D matrices, and NVIDIA documents single-DGX-Spark deployment | Its expert width is 1856: legal for group-32 quantization but not 128-aligned. A [documented Marlin MXFP4-MoE failure](https://github.com/vllm-project/vllm/issues/38022) makes stock-runtime viability an unresolved gate. Recurrent calibration is also hard. The official [mixed NVFP4/FP8 model](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4) uses a different format/runtime path. OpenMDW-1.1 needs review | Do not estimate the full project until a one-day GB10 kernel probe passes; then roughly 4–8 weeks |
| Novel watchlist | [K2-Horizon-MoVA-36B-A4B](https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B) | Updated 2026-09-21, Apache-2.0, 36B / 4B active, and combines sparse FFN experts with Mixture-of-Values attention. It is the least duplicated technical target in this list | Uses custom code and a newly landed runtime path; the recommended serving shape is not yet a boring single-Spark stock-vLLM path. It would require two sparse systems, not one | 1–2 day watch/preflight only; 6–10+ weeks if the runtime stabilizes |
| Recent dense fallback | [Muse Glimmer 30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) | Released August 2026, Apache-2.0, dense and non-Qwen, and therefore a cleaner architecture-portability test than another Qwen model | The quantization space is already saturated: official/community GGUF and MLX, qstream MXFP4, and multiple NVFP4 variants exist. It does not exercise expert routing | Roughly 2–4 weeks including adapter and full capability-preserving evaluation |

### Not primary candidates

- [Gemma 4 26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B) is a
  useful later adversarial test and has a documented stock-vLLM recipe, but the
  source was released in March 2026, its narrow expert geometry is sensitive to
  4-bit quantization, and official NVFP4 already exists.
- [Edge0-35B-A3B-preview](https://huggingface.co/Edge0/Edge0-35B-A3B-preview)
  is very recent but ships as a recovered 4-bit MLX-oriented artifact rather
  than a clean BF16 source for MxWave. It is not an admissible source model.
- Xing4.0 remains rejected from the earlier investigation because of its custom
  runtime dependency and weak independent quality evidence.
- Large 100B+ MoEs are not a first single-Spark target: output weights may fit,
  but BF16 teacher evaluation, source storage, and per-layer residency become
  the limiting experiment rather than the quantization method.

## What MoE support actually requires

This is not just another name-mapping adapter. MxWave is currently structurally
matrix-oriented:

- `quantize_mxfp4()` rejects tensors that are not two-dimensional;
- the planner, row reader, output specification, and byte estimator assume
  `[out, in]` weights;
- `all-linear` intentionally selects only two-dimensional tensors;
- streaming calibration supports only `qwen3_5_text`; and
- the runtime IR has no router, shared-expert, routed-expert, or fused expert
  bank semantics.

Recent Qwen3.5-MoE checkpoints store the routed expert bank as fused tensors
such as `experts.gate_up_proj` and `experts.down_proj`, while Nemotron and K2
expose individual expert matrices. Nemotron ingestion is consequently closer
to MxWave's current 2D path, but its Mamba-2 streaming calibration adapter is
substantially harder and its 1856-wide experts may not be executable through
the Marlin MXFP4 MoE path. A real implementation therefore needs:

1. a logical tensor view for ordinary 2D matrices, per-expert matrices, and
   fused `[experts, out, in]` banks;
2. role-aware policy rules that keep routers and routing gates in higher
   precision by default;
3. bounded expert- and row-chunk streaming, never whole-model residency;
4. routing-aware calibration statistics, including observation counts and a
   fail-closed policy for unobserved or under-observed experts;
5. layout-preserving packed output compatible with vLLM's FusedMoE loader; and
6. verification that every routed/shared expert is covered and that the
   intended MXFP4 MoE kernel is actually selected.

The shared calibration statistic should not silently be copied to every
expert. A defensible default accumulates an expert's input second moment only
from tokens routed to that expert, optionally weighted by the router
probability, and records the effective sample count.

## Bounded preflight

Before implementing an adapter or downloading another full source model, spend
at most two working days on this gate:

1. pin the exact source revision and inspect every safetensors header;
2. classify routed experts, shared experts, routers, dense projections,
   multimodal weights, and MTP weights, with an exact expected-count contract;
3. prove that each expert dimension is legal for MXFP4 and for the selected
   backend's tile alignment, then calculate the largest per-expert and
   per-layer resident buffers;
4. inspect a known-good vLLM 0.29 MoE checkpoint or its official online-MXFP4
   path to freeze the emitted key/layout contract;
5. pack and load a tiny synthetic fused-expert checkpoint on Spark; and
6. confirm from logs that a supported FusedMoE MXFP4 backend is selected on
   GB10/SM121 without a custom runtime.

Stop if the target needs remote code at serving time, if vLLM cannot consume
the standard checkpoint, if the layout requires full-model materialization, or
if the BF16 teacher cannot be evaluated within the agreed Spark budget.
Loading successfully is insufficient: the preflight must execute at least one
real MoE forward because the known Marlin alignment failure appears on the
first request rather than during checkpoint load.

## Promotion gate

Landing generic MoE infrastructure does not automatically justify publishing a
model. Compare the same BF16 revision against a plain RTN/MXFP4 control, the
MxWave calibrated checkpoint, and the strongest public 4-bit artifact using:

- paired NLL/perplexity on identical tokens;
- full-distribution forward/reverse KL on held-out contexts;
- at least one reasoning and one knowledge or agentic task;
- route coverage and per-expert observation counts;
- capability-normalized size, including vision and MTP when the source has
  them;
- stock-vLLM load, actual dense and MoE kernel selection, peak memory, TTFT,
  decode, and concurrent throughput; and
- long-context checks appropriate to the source model.

The first Qwen3.5-MoE result may be valuable as an engineering milestone even
if it does not beat existing NVFP4 artifacts. A public quality/generality claim
requires the non-Qwen rung and a measured Pareto improvement, not merely a new
checkpoint.
