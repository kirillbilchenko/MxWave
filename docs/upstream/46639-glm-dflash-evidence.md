Adding a GLM-5.3 Flash / DFlash case that gives this PR an end-to-end motivation: changing a future draft token changed a shared-history target result and ultimately a greedy output token. We traced the first changed shared-row operation to Marlin MoE's dispatch-dependent K partition.

**Environment and scope**

- DGX Spark / GB10 (SM121, 48 SMs), TP2; GLM-5.3 Flash compressed-tensors MXFP4 experts, group size 32, BF16 activations.
- Historical serving build: `vllm 0.1.dev20051+g487ecf187`, PyTorch `2.13.0+cu130`. This is evidence from that build, not a claim that we have tested this PR or current main.
- The target checkpoint is `INCModel3/GLM-5.3-Flash-MXFP4-Mixed-CT-AutoRound`. The same failure occurs with native scales and our losslessly packed expert scales.

**Captured 440 → 448 replay**

An eight-row DFlash verification call has identical first seven target-input rows, positions, routing weights and expert assignments in both arms. Only the eighth, future proposal differs. This changes the expert-padded dispatch from 440 to 448 entries.

The first changed shared-row result is layer 4's gate/up MoE GEMM:

| Setting | Value |
|---|---|
| M / N / K | 8 / 2048 / 4096 |
| Experts / top-k | 288 / 8 |
| MoE block size | 8 |
| Thread K / N / blocks per SM | 64 / 128 / 3 |
| Launch grid | 144 CTAs |
| Global MN tiles | 880 → 896 |
| Reduction | Non-atomic, FP32 reduction enabled |

For one affected dot product, the K partition changes from `[0,1536), [1536,4096)` to `[0,3712), [3712,4096)`. Another changes from an unsplit 4096-element reduction to a 2752/1344 split.

The shared-row output differences are small but reproducible: one BF16 element on rank 0 (maximum absolute difference `0.0009765625`), two on rank 1 (`0.000244140625`). Native Marlin, our uncompressed C-ABI bridge, and our packed-scale bridge reproduce their respective captured outputs exactly: **24/24 comparisons and 480/480 fixed-layout graph-repeat checks pass**. A separate captured position is an exact negative control. This isolates layout dependence rather than random repeat noise or scale compression.

Downstream, a target-logit tie at 27.625 becomes 27.625 versus 27.75 and changes the greedy winner. Thus this can affect speculative-decoding output identity even when the accepted history is the same.

Our existing workaround pads the routed tile count to full launch-grid waves, with fixed launch configuration. It restores shared-row bit identity on this captured fixture. That is component evidence, not a general full-model invariance claim. The fixture uses captured operands; the geometry above is not presented as a standalone random-data reproducer.

We are now evaluating this PR's `use_full_k` approach against that padding workaround. We will keep the MoE row block and thread configuration fixed as well, and report correctness and overhead separately. No result for `use_full_k` is claimed yet. Thanks for addressing this upstream; the speculative-decoding consequence makes this particularly useful for our GLM deployment.
