# MxWave end-to-end qualification

`mxwave-qualify` is the single decision boundary between an experimental
checkpoint and a model that may be described as qualified. It does not silently
convert missing measurements into passes.

## Ownership boundary

MxWave owns:

- whole-checkpoint structural verification;
- immutable input and report hashes;
- fixed perplexity and divergence evidence contracts;
- deterministic serving-result import;
- preregistered gates and the final decision.

The platform runner owns:

- starting and stopping the pinned stock-vLLM runtime;
- collecting logs that prove the selected kernel;
- fresh-runtime-cache and warm startup measurements;
- colocated streaming TTFT and decode measurements;
- concurrency and hardware telemetry;
- MTP proposed/accepted-token counters.

For DGX Spark, use `../local-spark`: it already has tested lifecycle,
streaming-SSE, telemetry, performance, long-context, coding, tool-use, and
repetition suites. Do not copy that implementation into MxWave. Export its
measurements using the versioned runtime-evidence contract below.

## Command

Install the repository and copy the templates:

```bash
pip install -e ".[dev,calibrate]"
cp templates/qualification-spec.json /results/qualification-spec.json
cp templates/runtime-evidence.json /results/runtime-evidence.json
```

Replace every `REPLACE_WITH_...` value before running. The template is
deliberately invalid until the checkpoint identity, frozen protocol hashes, and
candidate-specific limits are supplied; there is no universal PPL or KL limit.

Run structural verification by itself:

```bash
mxwave-qualify inspect --model-dir /models/candidate --hash-shards
```

Run the complete frozen contract:

```bash
mxwave-qualify run \
  --spec /results/qualification-spec.json \
  --output-dir /results/qualification-run
```

Exit codes are deliberately distinct:

- `0`: every required artifact/capability is present and every gate passed;
- `2`: the candidate was rejected or the evidence remains incomplete;
- `1`: the specification or execution was invalid.

`--resume` is accepted only when the existing report has the same specification
hash. An existing evidence artifact is reused only after its format, complete
schema, checkpoint binding, and protocol binding validate. Stale bound evidence
is regenerated when it has a hook.

## Structural contract

The verifier reads safetensors headers without materializing tensor payloads. It
requires:

- `config.json`, the safetensors index, and `mxwave-manifest.json`;
- an exact index-to-shard key mapping with no duplicate or unsafe shard paths;
- valid safetensors sizes and byte offsets;
- identical target/ignore lists in the config and manifest;
- canonical compressed-tensors method, status, MXFP4 group-32 W4A4 scheme,
  and channel-FP8 scheme for mixed checkpoints;
- complete `uint8` weight/scale pairs for every MXFP4 target;
- equal output rows and `packed_columns == scale_columns * 16`;
- no raw floating-point weight alongside a targeted packed weight;
- exact partition of every emitted weight module into MXFP4, FP8, or ignore,
  with no uncovered raw weights, nonexistent ignores, or orphan scales;
- contiguous, non-overlapping safetensors offsets whose byte lengths agree
  with tensor shapes and dtypes;
- matching index tensor-data bytes and manifest shard-file bytes.

`inspect` can omit full hashing for a quick structural check. A qualification
run requires `model.hash_shards: true`. Its checkpoint fingerprint includes the
config, index, manifest, every complete shard, and every tokenizer, processor,
chat-template, generation-config, or remote-code asset listed in the manifest.
The verifier repeats the fingerprint after evidence collection and makes a
mid-run checkpoint change incomplete rather than qualified.

## Evidence hooks

An evidence entry may contain a bounded `hook`:

```json
{
  "id": "perplexity",
  "kind": "perplexity",
  "path": "/results/candidate-ppl.json",
  "hook": {
    "argv": [
      "python",
      "scripts/evaluate_api_perplexity.py",
      "--input",
      "/data/wikitext.txt",
      "--output",
      "{artifact}",
      "--model",
      "candidate",
      "--api-key-file",
      "/run/secrets/llm_api_key",
      "--checkpoint-sha256",
      "{checkpoint_sha256}"
    ],
    "timeout_seconds": 7200,
    "inherit_environment": ["LLM_API_BASE"]
  }
}
```

Hooks use an argument vector with `shell=False`; `{model_dir}`, `{output_dir}`,
`{artifact}`, `{checkpoint_sha256}`, `{config_sha256}`, `{index_sha256}`, and
`{manifest_sha256}` are replaced only when they occupy a complete argument.
Before a hook runs, any artifact at its destination is removed, so an exit-zero
hook that writes nothing cannot expose stale evidence. API keys belong in files
or runtime secret stores, never in the specification. Stdout and stderr are
captured in hashed per-hook logs, including failed-hook exit details.

Supported evidence formats are:

| Kind | Required format |
|---|---|
| Perplexity | `mxwave-api-perplexity-v1` |
| Distribution | `mxwave-next-token-divergence-v1` |
| Serving | `mxwave-serving-qualification-collection-v1` or comparison v1 |
| Runtime | `mxwave-runtime-evidence-v1` |

Exact full-vocabulary divergence must be collected on the accelerator host.
An ordinary OpenAI API does not reliably expose every vocabulary logit.

Every evidence entry must include bindings. At minimum it must bind one report
field to the freshly computed `/checkpoint_sha256` and freeze the report's
top-level `/protocol_sha256` to a literal 64-hex digest:

```json
"bindings": [
  {
    "pointer": "/checkpoint_sha256",
    "structure_pointer": "/checkpoint_sha256"
  },
  {
    "pointer": "/protocol_sha256",
    "value": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  }
]
```

For a multi-candidate divergence report, point the checkpoint binding at the
candidate metadata instead, for example
`/candidates/candidate/metadata/checkpoint_sha256`. The literal protocol hash
must represent the actual frozen inputs and scoring procedure: PPL includes the
corpus and ordered tokenized windows; divergence includes the context manifest,
BF16 reference, metric and bootstrap settings; serving uses the case manifest;
runtime uses the frozen workload/telemetry plan. A caller-supplied model label
is not an identity.

## Runtime-evidence contract

Start from [`templates/runtime-evidence.json`](../templates/runtime-evidence.json).
Capabilities are inferred from measurements rather than trusted from a declared
list:

- stock-vLLM load requires a digest-pinned image, `stock: true`, a passed load,
  and a recorded kernel/backend;
- startup is called **fresh-runtime-cache startup**, not absolute cold load,
  unless the host page cache is also controlled;
- TTFT and decode require positive streaming measurements;
- concurrency requires at least one sample above concurrency one;
- MTP requires positive proposed-token counters, bounded accepted-token
  counters, and deterministic token parity;
- publishable latency requires `client_location: on-device`; a laptop tunnel is
  explicitly not colocated.

The normalized runtime report must carry the same `checkpoint_sha256` and its
frozen runtime `protocol_sha256`. Zero-valued template fields and placeholder
image digests deliberately fail capability checks. Preserve raw platform
reports alongside the normalized evidence rather than replacing them.

## Gates and completeness

Gates address evidence through RFC-6901-style JSON pointers:

```json
{
  "id": "ppl-limit",
  "evidence": "perplexity",
  "pointer": "/perplexity",
  "operator": "<=",
  "threshold": 8.2
}
```

Supported operators are `<`, `<=`, `>`, `>=`, `==`, and `!=`. Complex paired
statistics should be computed by their versioned evaluator and gated on the
resulting scalar or boolean; the qualifier must not recompute them using a
different resampling protocol.

A valid measured value that fails a gate produces `rejected`. A missing,
schema-invalid, stale, or mis-bound report produces `incomplete`, including
when a gate cannot be evaluated. Only complete passing evidence can produce
`qualified-default` or `qualified-optional`. Empty evidence/gate sets,
structural-only contracts, and contracts without a gated quality artifact are
invalid specifications; use `inspect` for structure-only checks.

## Current limitations

- The existing exact-divergence collector measures one terminal position per
  context. It satisfies `distribution_divergence`, not
  `multi_position_divergence`.
- The existing serving comparison proves output parity but not that MTP was
  actually exercised. MTP acceptance requires backend proposed/accepted-token
  counters in runtime evidence.
- The qualifier imports platform measurements; it does not duplicate
  `local-spark` lifecycle or telemetry.
- A real GPU/vLLM run remains an opt-in integration test. CPU CI validates the
  schema, structural checker, hooks, gates, and incomplete/failure semantics.
