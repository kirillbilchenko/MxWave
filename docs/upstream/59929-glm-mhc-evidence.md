Additional motivation from GLM-5.3 Flash on GB10/SM121: we independently traced batch-layout differences to the same two mHC choices addressed here. Posting this on the existing PR rather than opening a duplicate report.

Our historical fixture uses GLM layer-0/layer-3 mHC parameters, synthetic seeded BF16 inputs, hidden size 4096 and hc multiplier 4. It moves identical anchor rows among batches of 1, 8, 16, 23, 258, 516, 522, 774, 1293, 1548, 1550 and 2048 rows. Fixed-shape repeats are exact.

- Native dispatch: 38 failed component batch-invariance cases.
- Fixed prenorm split count only: 22 failures remain.
- Fixed split count plus a consistent separate post/pre composition: zero failures in this fixture.

The remaining differences after fixing splits came from switching between the small-row fused arithmetic and the larger-row composition. This matters for comparing speculative drafters because proposal/admission changes alter target verification layouts even with shared token histories.

This is historical component evidence, not a test of this PR or a full-model quality claim. Source SHA256 pins for that fixture are `c8ce81539c779436eb2b9fbba84738f3114bf3ac0a149f2b606f7d38bd1067ce` (`tilelang.py`) and `03aeb3f7c0eabfe8391006ff80f8d4cfedba7cf3d0a168e0dee2d678d3e3bd24` (`tilelang_kernels.py`). A consistent policy also changes some default C1 rounding, so we keep its full-model quality and serving-overhead checks separate.

The shared-helper changes here are relevant to GLM too; GLM's model call sites should also be checked for the consistent post/pre path as this work lands. We have not yet qualified GLM against this PR. Related MoE verification evidence is on #46639: https://github.com/vllm-project/vllm/pull/46639#issuecomment-6318857951.
