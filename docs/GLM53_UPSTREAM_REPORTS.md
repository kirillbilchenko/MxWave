# GLM upstream evidence and article attribution

Updated 21 October 2026, 15:12 UTC.

The user authorized submitting the retained numerical evidence upstream. No
duplicate issue was opened. Exact public comment bodies are retained under
`docs/upstream/`; private prompts, host details and checkpoint tensors are omitted.

| Mechanism | Existing upstream work | Our action and scope |
|---|---|---|
| Marlin MoE Stream-K | [PR #46639](https://github.com/vllm-project/vllm/pull/46639), open at audit | [Posted GLM/DFlash evidence](https://github.com/vllm-project/vllm/pull/46639#issuecomment-6318857951): exact captured 440/448 replay, native/raw/packed controls, downstream greedy-token change. |
| mHC split and composition | [PR #59929](https://github.com/vllm-project/vllm/pull/59929), open at audit | [Posted GLM component evidence](https://github.com/vllm-project/vllm/pull/59929#issuecomment-6318916233): fixing split alone leaves failures; consistent post/pre clears the retained fixture. GLM call-site coverage remains to be checked. |
| Sparse MLA index ordering | [PR #50365](https://github.com/vllm-project/vllm/pull/50365), merged | No new issue: the inspected v0.32.0 helper explicitly pads counted GLM 2176-entry rows to one program. Historical fixture findings are retained; current-version GPU confirmation is still pending. |
| Dense WNA16 batch invariance | [PR #60524](https://github.com/vllm-project/vllm/pull/60524), merged | Credit related upstream work, but distinguish its compressed-tensors dense-linear scope from the MoE full-K policy in #46639. |

## Full-K comparison

`build_moe.py --full-k` specializes the two tail-partition changes from #46639
head `21abea6c75c6b36c278c1ae87d50a2f4a57a0e40` into the existing isolated raw and
packed-scale libraries. Provenance is recorded in `CONTRIBUTING.md` and each build
manifest. It requires explicit thread tiles and occupancy; row-block selection
remains caller-owned. It does not import the complete PR or change the package's
serving default. Compilation and the isolated GPU replay pass. Independent NumPy
review verifies all 45 archived payloads, 48 native-output matches, 96 raw/packed
matches and 960 reported graph repeats. Full-K and bounded padding agree bit for
bit in all 64 retained output comparisons; both fix the shared-row 440-to-448
divergence on both TP fixtures.

The retained two-projection replay compares `full_k`, native and bounded padding
using fixed row blocks and thread configurations. Eleven of sixteen direct
full-K/padding timing pairs pass the existing 2% drift gate. In stable packed-scale
pairs, full-K latency ranges from 3.87% lower to 1.46% higher. Keep all failed
controls; do not claim a universal performance win or change the package default.
The run takes 412.06 seconds; exact production restoration is verified at
15:08:02 UTC. Archive SHA256:
`7c95b8c3b58d4967d16a99453b2c997123c839b3b1beda45ec19b853ad197d7f`.
Both partitions run separately on one GPU. This is not distributed/full-model
serving qualification or a standalone random-data reproducer.
The [measured follow-up](https://github.com/vllm-project/vllm/pull/46639#issuecomment-6319447508)
was posted at 15:13:30 UTC; its exact body is retained in
`docs/upstream/46639-glm-full-k-followup.md`.

## Publication framing

Credit the upstream reports and fixes above. The strongest supported contribution
is the GLM end-to-end diagnosis: a different future DFlash proposal changes padded
expert dispatch and target rounding, which can flip a greedy decision on a shared
history. Our retained fixtures also separate MoE, mHC and sparse-MLA effects from
lossless scale compression. Do not claim discovery of Stream-K nondeterminism,
or that mHC/sparse ordering were previously unreported.

Keep component correction, whole-model invariance, practical task quality and
serving performance separate. The final MXFP8-KDA CT profile is lossy; upstream
package compatibility and deployment speed are separate claims. Historical
custom-runtime measurements are not results for stock vLLM 0.32.0.
