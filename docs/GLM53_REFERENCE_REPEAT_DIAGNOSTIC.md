# GLM reference repeat diagnostic

Status: **complete, 17 October 2026**. Strict deterministic mode does not
stabilize reference repeats. All 360 fixed-input expert GEMM replays are
bit-exact; consumed dispatch order varies with identical logical routing.
Both original GLM containers are restored healthy with matching identities,
independently checked at 09:56:58 UTC. Original quality gates and the
completed qualitybestv2 evidence remain unchanged.

The purpose is to identify where the uncompressed runtime reference first
varies on identical requests. Its repeats already fail within one loaded
process. This is not a new candidate speed or broad accuracy evaluation.

## Results and interpretation

Follow-up: the controlled permutation-only projection test and 92-case GPU
canonicalization qualification pass. The matched whole-model run is in
progress; see [stable dispatch](GLM53_STABLE_DISPATCH_EXPERIMENT.md).

| Probe | Result |
| --- | --- |
| Five default-policy eager runs, first two contexts | Four repeat KL values: 1.2477e-5 to 2.0242e-5 |
| Five strict-policy eager runs, same contexts | Four repeat KL values: 1.9916e-5 to 2.5397e-5 |
| Frozen 1e-5 repeat-KL floor | All eight comparisons fail |
| Generated IDs and decode conditioning | Match in all eight comparisons |
| Source scales before/after | All 336 actual group checks match |
| Selected loaded-weight/attention checks | 658 records/rank pass |
| Bounded captures | Three/rank, 96 events and 406,792,968 snapshot bytes/capture |
| Fixed-input GEMM replay | 360/360 outputs bit-exact under both policies |
| Actual GEMM settings | Atomic add off; FP32 reduction on |

Independent CPU review verifies all 80 retained logprob arrays and recomputes
the eight decode comparisons using numpy.logaddexp.reduce. Both private
activation archives pass every payload size/hash check: 822 head files and
727 worker files. No original numerical floor changes.

In the captured layers 3–5, MLP inputs, router logits and actual selected
expert IDs/weights match between requests. Shared-expert outputs also match.
The first changed runtime input is sorted_token_ids, the permutation used
to group token/expert pairs into GEMM blocks. Separate consumed-prefix
verification covers 24 GEMM-input comparisons: every one has a different
order, while all 2,064 valid pairs occur exactly once and retain the same
actual routed expert. This is not a hash difference in unused buffer tails
or a different logical top-k choice.

At the first changed floating output in each of the four per-rank repeat
comparisons, exactly one BF16 value differs out of 4,227,072 or 8,454,144.
Maximum absolute differences range from 5.96046e-8 to 3.81470e-6. The first
changed projection varies across comparisons: layer 3 gate/up, layer 4 down
or layer 5 gate/up. Local floating changes can be rounded away before the
MLP output; a changed final TP sum can originate on the other rank.

**Leading hypothesis:** dispatch-order-dependent numerical rounding.
Fixed-order GEMMs are repeatable in these checks, and all other captured
runtime inputs match at the first changed GEMM output. This remains an
observational lead. Clones perturb scheduling, only three layers are
captured, and neither harmlessness nor a complete explanation of final
KL is established. The strict policy probe does not support declaring
ordinary execution deterministic or revising its quality gate.

The next controlled test holds activations, weights, scales, routing
weights, padding, locks, output buffer and kernel parameters fixed while
swapping only two retained consumed token permutations. It must first
reproduce the original captured output with independently reconstructed
source weights. Then compare canonical per-expert ordering, checking every
token/expert membership and every output byte. Start with a single real
layer-4 down projection on Spark2 under the idle guard; another full-model
load is unnecessary for this ablation. A successful component leads to a
bounded GPU implementation and unchanged-floor reference repeat check.
The candidate still needs its matched numerical and tool-case qualification.

## Evidence and timeline

Launch: 09:43:14 UTC; reference ready: 09:48:21; measurements complete before
the one restoration at 09:51:34; original GLM health 200: 09:54:11.

The compact repository review is
benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/reference-repeat-v1-review.json,
SHA256 f9e60308dda7e33d82e297f57d333930c99ba67c68a313c3a968bc26658106f7.
Full raw activations remain on Spark.

Private archive digests:

- Rank 0: 1,009,482,633 bytes,
  1b6d69c6e34c77d254a155cb9d2c7fcd8884b81c679fdc0f32790565cf74ff8e.
- Rank 1: 984,032,156 bytes,
  c86a56fd5c468ec2f43b3a65c108a1135bb98f3ef926bfb2ee7bd0030cc8b4f5.
- Small review/decode bundle: 22,412,191 bytes,
  560175b3c80a57eee755cd3c757699d4bcf5a61890f405539d6a5cfa56edbab9.

Consumed-order proofs are under each rank's traces/dispatch-order-review.json.
Their standalone stdlib analyzer is ROOT/analyze-repeat-dispatch-order-v1.py.
The final independent service proof is ROOT/reference-repeat-final-service-v1.json.

## Protocol

One TP2 reference load uses the same pinned MXFP4 source, image, transport,
attention mapping and 4 GiB FP8 KV allocation per rank as the completed
quality comparison. Expert scales, KDA projections, head and embeddings
stay uncompressed. Both tool-parser flags are present, but tool cases are
not rerun in this diagnostic.

CUBLAS_WORKSPACE_CONFIG=:4096:8 is set before process startup. Five
uninstrumented eager repeats of the first two frozen 258-token contexts,
four generated tokens per context, run with the default PyTorch
deterministic policy. All full-vocabulary logprobs and conditioning IDs
are retained. The same five-repeat window runs with strict
torch.use_deterministic_algorithms(True). This may reject unsupported
operations; custom kernels may ignore the policy. Both sets share the
workspace setting, so any difference from v2 is not attributed to that
setting alone.

Between these two windows, three instrumented runs of the first context
capture layers 3–5, covering the original layer-4 first-difference lead.
Captures include decoder/attention/mHC inputs and outputs, MLP inputs,
router logits and actual selected IDs/weights, shared-expert components,
expert GEMMs and the final reduction. Each producing stream clones
runtime values; explicit CPU synchronization and exact-byte serialization happen
after inference and hook removal. Allocation itself may synchronize.
This is still intrusive and changes
GPU scheduling. Maximum snapshots: 512 MiB/rank/capture; extra fixed-input
GEMM fixtures: 128 MiB/rank. Overflow marks incomplete evidence without
raising inside model forward.

After each capture, the six actual expert GEMMs are replayed five times
under each deterministic policy with fixed captured activations, routing,
padding and locks and unchanged loaded weights. Replays omit TP
collectives and use fresh outputs and lock copies. Exact bytes and numerical
error are compared against the captured output. The prior policy is restored.
This can distinguish varying GEMM arithmetic from variation entering it;
it does not simulate the whole MoE or prove the rest of the model stable.

All actual source-scale groups are checked before and after; the existing
selected loaded-weight/attention checks run before inference. CPU analysis
checks every retained tensor hash, shape and dtype before reporting the
first changed boundary and numerical magnitudes. No quality floor changes,
candidate qualification or throughput conclusion is permitted from this run.

## Code and validation

The experiment scripts (not included in this branch) were repeat_worker.py (bounded snapshots and
fixed-input replay), measure_repeat_model.py (source checks and policy
windows), analyze_repeat_model.py (independent NumPy byte review) and
run_repeat_model.py (controller, idle guard and independent watchdog).
Tests in tests/test_glm53_repeat_diagnostic.py cover workspace-reuse
independence, BF16 bytes, bounds, corruption/path rejection and stock launch.

Before freezing: ruff check ., strict mypy mxwave (35 files), and
python -m pytest pass: **482 passed, 1 GPU test skipped locally**.
Both nodes' 115 Python/TOML files match, inventory digest
dd87b785cb817808f04cbc0a30c670f771e93d3e94c3f45cf1d8991ed4200a78.
The overlay archive SHA256 is
1b357973bb438e7966feda30a6636d0808c93687bd6710357f6d464c52d6a062.
Exact-image CPU preflight imports and validates the actual worker extension,
parser registration, RPC metadata serialization and vLLM CLI before pausing GLM.

## Spark locations

On either node, ROOT is:

    /home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/full-model-v1

Frozen source and per-rank artifacts:

    ROOT/full-repeatpolicyv1-src/
    ROOT/run-repeatpolicyv1/stock/traces/
    ROOT/reference-repeat-evidence-v1-rank{0,1}.tar.gz
    ROOT/reference-repeat-evidence-receipt-v1-rank{0,1}.json

On the head:

    ROOT/repeat-model-controller-repeatpolicyv1.json
    ROOT/repeatpolicyv1-controller.log
    ROOT/run-repeatpolicyv1/results/reference-repeat-measurement.json
    ROOT/watchdog-repeatpolicyv1.json
    ROOT/restoration-repeatpolicyv1.json
    ROOT/reference-repeat-collection-v1.json
    ROOT/repeatpolicyv1-collector.log
    ROOT/collect-repeatpolicyv1.py

Controller PID: 1460550; independent collector PID: 1460549.
The watchdog acknowledgement records its own PID. All processes are
detached from SSH and independent of the laptop. The controller uses
the existing exclusive experiment lock, waits for an idle window, checks
original identities, arms the watchdog, then pauses both original containers.
It restores those same containers once after the diagnostic or on failure.
The watchdog enforces an outage deadline independently of the controller.
CPU activation analysis and archiving happen after restoration.

The collector retains partial evidence if a phase fails. Snapshot archives
remain private; raw activations and responses do not belong in Git.
No automatic retry, production patch, changed precision or relaxed gate
is authorized by a failed diagnostic.
