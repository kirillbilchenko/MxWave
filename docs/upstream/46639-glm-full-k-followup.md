Follow-up on the captured GLM/DFlash case: the full-K policy fixes the 440-to-448 shared-row divergence in our isolated replay.

We adapted the two tail-partition changes from this PR at `21abea6c75c6b36c278c1ae87d50a2f4a57a0e40` into our pinned raw-scale and packed-scale test libraries. This is a specialization of the policy, not a test of the complete PR or current main. Both policies use the same explicit MoE row block, thread tiles and occupancy. The full-K path uses the original routing metadata without our extra padding.

On GB10/SM121, for both retained TP partitions (replayed separately on one GPU), the two-projection fixture passes:

- shared-row bit identity across the changed future proposal, including the previously divergent 440/448 dispatch;
- 96/96 raw-scale versus losslessly packed-scale output comparisons;
- 960/960 fixed-layout CUDA graph-repeat comparisons;
- 64/64 bitwise comparisons of full-K against our bounded-padding workaround, covering gate/up, activation, down and expert summation;
- 48/48 native replay comparisons against the original captured outputs, retaining the original divergence as a control.

An independent CPU review verified the archived payload hashes and output bits. The captured negative-control position also passes. This supports replacing padding as a correctness mechanism in this fixture; it does not yet establish whole-model invariance or coverage of larger concurrent dispatches.

Timing is less conclusive. We retained interleaved CUDA graph trials and clock/temperature telemetry, with the existing 2% first-half/second-half drift gate. Only 11/16 direct full-K-versus-padding timing pairs pass that control. Among stable packed-scale pairs, full-K latency ranges from 3.87% lower to 1.46% higher. We are retaining the failed controls and making no universal or end-to-end speed claim. No serving default has changed.

Thanks again: this removes the need for routing padding in the tested case. Keeping row-block and thread selection fixed remains necessary in our stack; full-K addresses the reduction-partition part of the problem.
