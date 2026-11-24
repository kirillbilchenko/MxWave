# Sparse MLA compaction ordering: report draft

Updated 18 October 2026. Prepared locally; no external issue or message has
been sent. Findings apply to the pinned vLLM-based runtime below. Whether the
same implementation is present in an upstream release has not been established.

## Observed behavior

With fixed request IDs, block table and 2176-column token-index rows, the
counted mapping helper can return the same selected physical-slot multiset
in different prefix orders across CUDA graph replays. Duplicate multiplicity,
valid counts and invalid tails remain correct. The prefix order changes the
native attention result even when query and selected cache bytes are equal.

The affected helper is
`vllm.v1.attention.backends.mla.sparse_utils.triton_convert_req_index_to_global_index`,
source SHA256
`cd95caf494f1d2a946ac87abfb4951f738827fe7749aeefd6c0fd83d1cee072d`.
It is called with `return_valid_counts=True` by the target
`FLASHINFER_MLA_SPARSE_SM90` backend. The non-power-of-two width uses 17
128-column tiles and an atomic prefix allocator. The counted power-of-two
path assigns one tile to each row and avoids allocation between row tiles.

This was observed with SM121, image
`sha256:39a03677234ea74e35544d301bd14716c1ae9695ad8d91b16e2c76545e26e8d2`.
The target SM90 backend source SHA256 is
`4449ea25921dcae1ec7988581ff26a6ff5a9e5d6d4a0ac4a68a0017ea108f136`.

## Retained evidence

The raw-reference model diagnostic retains 40 graph calls across both ranks.
In 25 of 32 noninitial calls, the earliest changed recorded operation is sparse
conversion output order. All 40 selected slot multisets and planned FP8 byte
multisets match; all valid/planned lengths are 259, with no observed slot-zero
overread. Restoring the private workspace does not remove the order changes.
The affected layer's attention relative-L2 changes by about 0.00117–0.00268.

Some final hidden states also change when the observed layer repeats, so this
does not prove compaction is the only numerical source. The intrusive diagnostic
does not measure ordinary-serving KL or throughput.

The broader actual-GPU fixture executes 50 cases × two modes × 100 graph
replays, including rows 1/2/4/8/16 and mixed valid counts
0/1/17/129/259/1024/2048/2176. At uniform count 259:

| Rows | Native distinct outputs in 100 replays | Padded path |
|---|---:|---:|
| 1 | 25 | 1 |
| 8 | 100 | 1 |
| 16 | 100 | 1 |

All stable outputs match the CPU oracle's original-column order exactly.
The fixture checks selected membership, duplicate multiplicity, counts and
invalid tails. Zero/one-entry rows remain valid controls.

## Candidate mitigation

An independently authored wrapper appends −1 columns to width 4096, invokes
the unchanged native counted mapping helper, then trims its result back to a
contiguous 2176-column tensor. This preserves original input-column order and
adds no valid entries. It does not sort physical slots or modify the native
kernel. Installation covers the canonical helper and its loaded aliases so
future imports and alternate counted prefill/decode callers inherit the wrapper.
This experiment covers DCP size one.

Native attention sentinel tests use per-row lengths ranging from 1 to 2176,
with invalid tails clamped to slot zero containing value 448. Every selected
valid value is one. All 25 graph replays each at C8 and C16 return exact ones,
showing the planned prefix excludes the invalid tail in those cases.

Balanced graph timing retains identical-native controls and clock/thermal
telemetry. Isolated conversion adds about 2.94–3.18 µs per call at C8/C16,
including mixed counts; native 2.64–2.83 µs versus padded 5.59–5.99 µs.
This is not a measurement of full-model padding overhead.

## Independently verified ordinary-model checks

All three ordinary C1 runs complete. Both repeat comparisons have bitwise
equal full-vocabulary outputs: 16 prefill and 48 decode rows, KL = 0 and
NLL difference = 0. This is stock self-repeat, not compression qualification.

C8 repeats still fail KL at 6.802436007214667e−5 and 5.331761526844531e−5.
NLL passes and generated IDs/conditioning match, but every context differs
already at its first prefill vocabulary output. The first idle admission
collections contain 3, 1 and 8 requests across runs. Different prefill batching
is a concrete lead; actual forward shapes and the responsible operation have
not been established.

All 641 payloads, 186 source bindings and four worker profiles independently
verify on Spark/Mac. Mixed-length C8/C16 profiles confirm graph replay on both
ranks. The original service is restored with exact identities and health 200.
Evidence archive SHA256:
`706347d0e2d098f10388ff152d35b3a7547dda59229e36a4b8586a4a81c6ff69`.

Keep KL ≤ 1e−5 and absolute mean NLL difference ≤ 1e−4. Prior failures remain
retained. Stock-versus-compressed must use the same deterministic policy in
both arms; that fresh C1 comparison has not launched.

The complete authored fixture and frozen source are retained locally at
`/private/tmp/mxwave-sparse-quality-v2/`; see the
current handoff for live status, source hashes, evidence
receipts, the explicit case-count metadata correction and restoration status.
The fixture itself is reproducible without model weights using its synthetic
cases. Its real-model operands and native implementation snapshots remain
private evidence.

Before filing, confirm the upstream implementation/version and appropriate
owner. Do not claim the remaining stock C8 failure is fixed. No upstream kernel
implementation is copied into this repository.

A standalone synthetic reproducer was prepared using the same synthetic
construction as the completed fixture. It was CPU syntax checked but not
separately run on a GPU. The experiment scripts for this study are not included
in this branch.
