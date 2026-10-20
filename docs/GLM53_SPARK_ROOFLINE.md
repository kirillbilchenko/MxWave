# GLM operation efficiency on the two Sparks

**18 October 2026: a calibrated full-model roofline is still missing.** We have source-bound
operation timings, occupancy and instruction diagnostics for selected components.
We cannot yet give a defensible percentage of memory-bandwidth or compute efficiency
for the full model. The completed GSM8K comparison remains 41.77 / 42.08 output
tok/s and 9.367 GiB less allocation; allocation alone does not measure traffic.

## Completed KDA memory diagnostic, 18 October

**21:33 UTC:** 560 payloads and 210 source bindings independently verify on Spark
and Mac. Both GPUs complete 22 operations / 28 NCU kernel records. All exact
output and guard checks pass. The archive, offline export recovery and unchanged
failed repeat controls are documented in the [handoff](GLM53_HANDOFF.md).

Graph replay with 128-MiB cache pressure gives these descriptive logical rates:

| Operation | Rank 0 GB/s | Rank 1 GB/s | 2% repeat controls |
|---|---:|---:|---|
| Input reconstruction | 197.36 | 198.32 | Rank 0 passes; rank 1 fails |
| Input dense copy | 218.54 | 220.37 | Both pass |
| Output reconstruction | 199.70 | 203.36 | Both pass |
| Output dense copy | 218.44 | 217.55 | Both pass |

All 22,528 timing samples and 35 failed controls remain. These are separate
operation measurements, not a paired optimization comparison. Cold NCU replay
records approximately 204 MB / 66 MB L2 request bytes for input/output decoders,
versus 207 MB / 67 MB for dense copies. Instructions rise about 41.7 / 42.0 times;
active warps are about 33% / 49%, versus 85-88% for copies. Long-scoreboard is the
largest decoder stall ratio; that ratio is not a percentage of elapsed time.
This supports a memory-wait hypothesis without proving a particular instruction
is the bottleneck or establishing a hardware bandwidth ceiling.

The [56 per-kernel points](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/kda-memory-v1-L2-points.csv)
retain actual NCU duration, L2 request bytes, useful FLOPs (2MNK), arithmetic
intensity, instructions and activity counters. Reconstruction/copy kernels have
zero useful matrix FLOPs; their integer work remains in instruction counters.
[Independent findings](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/kda-memory-v1-counter-findings.json)
include both-rank thermal/clock boundaries and every control failure. Memory-clock
queries are unavailable, and the head includes initial clock ramping. No throttle
flag was sampled; transient effects are not excluded.

This completes a source-bound KDA/copy counter diagnostic. It does **not** complete
the full-model roofline: LPDDR bytes, calibrated arithmetic/L2 ceilings, other
operation families and the TP critical timeline remain missing. Avoiding repeated
prefill reconstruction is now a stronger candidate than expecting a large gain
from an unchanged-format decoder rewrite; its model speed benefit is unmeasured.

## Existing operation timings

The later graph-four capture contains three complete M4 decode forwards per rank
on the same resident compressed candidate. This is an earlier diagnostic, separate
from the completed full GSM8K run. Medians in milliseconds:

| Operation | Spark 1 / rank 0 | Spark 2 / rank 1 |
|---|---:|---:|
| GPU forward span | 78.826 | 78.713 |
| Routed-expert GEMMs, 84 calls | 32.057 | 31.679 |
| Compressed BF16 KDA projections, 68 calls | 23.947 | 21.609 |
| Other BF16 linear kernels | 9.469 | 8.922 |
| Other Marlin linears | 4.445 | 4.206 |
| Communication kernels, including peer waiting | 1.797 | 6.053 |
| KDA recurrent/state kernels | 1.721 | 1.616 |
| mHC kernels | 1.687 | 1.524 |
| Canonical expert ordering | 0.401 | 0.435 |
| Embedding lookup | 0.009 | 0.008 |
| Uncovered forward span across all streams | 1.740 | 1.660 |

These are independent medians and kernel-duration sums: they can overlap and must
not be added into a wall-time partition. Do not add both ranks' times. The head is
outside the annotated model forward; an older 61-forward Nsight Systems capture
separately measured about 2.39 ms/head/rank. It is not part of this 78.8 ms table.
The KDA projection duration includes useful matrix multiplication and decoding;
it does not directly measure the compression overhead.

[Reproducible timing and counter inventory](../benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/roofline-readiness-v1.json)
binds these values to the existing trace review and trace hashes. The
[controlled graph experiment](GLM53_CONTROLLED_BATCH_EXPERIMENT.md),
[Nsight Systems report](GLM53_NSYS_MODEL_DIAGNOSTIC.md) and
[component Nsight Compute report](GLM53_BF16_KERNEL_DIAGNOSTIC.md) retain their
different sample sizes, execution modes and numerical limits.

## What the current profiler exposes

A read-only query of **Nsight Compute 2025.3.1**, using its offline `gb20b`
chip definitions, finds L2 byte/sector counters, tensor-pipeline activity, SM
throughput, instructions and duration. It defines none of `dram__bytes_read.sum`,
`dram__bytes_write.sum`, `dram__bytes.sum` or the requested DRAM-throughput metric.
The 93,373,481-byte definition export and its SHA256 are retained on Spark:

```text
ROOT=/home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/full-model-v1
ROOT/roofline-chip-metrics-v1.txt
ROOT/roofline-counter-availability-v1.json
/opt/nvidia/nsight-compute/2025.3.1/ncu --query-metrics --chips gb20b --query-metrics-mode all
```

This is an installed-tool limitation, not proof that no future tool or platform
counter can measure LPDDR traffic. The host device query returned a driver
`Unknown Error`; offline definitions succeeded. No GPU workload was launched and
GLM remained running. Actual counter access previously worked in the isolated
`SYS_ADMIN` profiling container. Successful definitions do not guarantee a metric
will collect on-device.

L2 request bytes, L2 miss sectors and compute-memory pipeline utilization must
remain distinct from measured LPDDR traffic. An L2 roofline is possible after
collecting its counters; a DRAM roofline needs an independently validated traffic
source, or must be explicitly labelled an analytical memory model.

## Bounded next measurement

1. Reuse the completed source/native bindings and existing trace windows. Do not
   rerun GSM8K or the full accuracy battery to obtain roofline data. Resolve the
   active reference-repeatability diagnostic separately; profiler serialization
   must not be used as evidence of normal-runtime determinism.
2. First qualify a tiny counter fixture in a bounded container on each Spark.
   Request duration, L2 read/write sectors or bytes, hit rate, tensor activity,
   instructions, occupancy and stall/launch statistics. Preserve unavailable
   metrics explicitly. Inspect hierarchical tensor-roofline section support.
   Use cold and warm cache regimes separately; counter replay timings do not
   replace unprofiled serving latency.
3. Measure sustained GPU memory-read/copy ceilings on allocations larger than
   L2 and representative compute ceilings for the arithmetic actually executed.
   Keep CPU/network load, clocks and temperature in the record. The advertised
   sparse FP4 AI peak is not a calibrated ceiling for our Marlin/BF16 operations.
   NVIDIA lists **273 GB/s per Spark**; measure the achieved local ceiling before
   reporting utilization. Two-node TP2 needs per-rank ceilings plus transport;
   summing nominal bandwidth does not create one shared memory channel.
4. Collect representative source-bound kernel windows for both raw and compressed
   storage, with matched shapes, routing, cache regime and graph mode. Separate
   prefill from decode and M=1 from M=4. Start with expert gate/up and down,
   KDA input and output, other Marlin/BF16 linears, vocabulary head, recurrent
   state updates, then mHC/routing/embedding. Prefer bounded existing component
   fixtures; use a short whole-model capture to validate their representativeness.
5. For each kernel report duration, useful matrix FLOPs (FMA=2), L2 bytes,
   FLOPs/L2-byte and achieved FLOP/s. Derive FLOPs from actual shapes/MMA geometry;
   expert coverage must follow selected experts and padded work, not all resident
   weights. Report integer/exponent codec work separately. Compare each point to
   the appropriate measured arithmetic and L2 ceilings. If LPDDR traffic remains
   unavailable, plot an explicitly modelled payload/memory curve separately.
6. Place communication and host gaps on the two-rank critical timeline. GPU
   collective duration includes peer waiting; it is not physical link utilization.
   Record collective payloads/transport counters before estimating network headroom.
   Weight optimization priorities by exposed critical-path time and preserve the
   existing byte-exact reconstruction and model-quality gates.

The resulting artifact should contain per-operation measured points and bounds,
per-rank ceilings, an explicit unavailable-metric list, and a model-step timeline.
Until then, the honest conclusion is that MoE and KDA dominate elapsed GPU work;
the amount of recoverable bandwidth or compute headroom remains unmeasured.

Method and hardware references:
[NVIDIA roofline profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline-charts)
and [DGX Spark hardware specifications](https://docs.nvidia.com/dgx/dgx-spark/hardware.html).
