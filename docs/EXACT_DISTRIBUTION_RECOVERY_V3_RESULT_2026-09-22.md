# Exact-distribution recovery v3 result

Status: completed successfully; rejected at Selection A.

Date: 2026-09-22

Implementation: `research/exact-distribution-recovery-v3` at
`18fb7808bc4625c63d34f283610afbda63d1e935`

Machine-readable summary:
`benchmarks/qwen3.8-27b-exact-distribution-recovery-v3.json`

## Question

Can a second legal-E2M1 recovery pass, trained against complete BF16 teacher
distributions, improve the packed recovery-v2 checkpoint without changing its
MXFP4 scales, size, format, or stock-vLLM runtime?

## Result

The detached Spark run exited zero and evaluated all 12 pre-registered
snapshot/trust candidates. No candidate satisfied the Selection-A requirement
to improve mean observed-token NLL, mean exact forward KL, per-context KL p95,
and teacher top-1 agreement simultaneously.

The strongest mean-KL candidate was `step-240-trust-1p0`:

| Metric | Recovery-v2 baseline | Candidate | Change |
|---|---:|---:|---:|
| Mean forward KL, nats | 0.0280434972 | 0.0280161529 | -0.0975% |
| Per-context KL p95, nats | 0.0589161178 | 0.0587111868 | -0.3478% |
| Mean observed-token NLL | 1.6895834774 | 1.6891264070 | -0.0271% |
| Teacher top-1 agreement | 93.6335% | 93.5231% | -0.1104 points |

The candidate lost 9 teacher top-1 matches over 8,152 Selection-A positions.
Nine of the 12 candidates reconstructed the packed baseline exactly; the other
changed candidates produced only tiny trade-offs rather than a qualifying
quality improvement.

Training itself was numerically healthy: mean group KL fell from 0.058237 over
the first 16 processed sequences to 0.038591 over the last 16. The failure was
therefore generalization to the frozen held-out selection objective, not an
execution or optimization failure.

## Stopping decision

Per the pre-registration, Hidden B was not read, no packable adapter was
written, and confirmation, packing, PPL, and task evaluation were skipped.
This closes unchanged-size exact-distribution recovery under frozen MXFP4
scales for this Qwen3.8-27B checkpoint. It does not invalidate the earlier
recovery-v2 mechanism, whose modest PPL gain remains recorded separately.

## Resources and cleanup

- teacher cache: 400 samples and 81,520 positions;
- teacher-cache payload: 40,487,181,120 bytes;
- selection/training elapsed time: 1,824.35 seconds;
- peak process RSS: 51,282,907,136 bytes;
- peak accelerator allocation: 59,498,771,968 bytes.

After this durable summary was prepared, the three experiment directories and
seven stopped containers were intentionally deleted from Spark, reclaiming
about 38 GiB. The shared vLLM image was retained because it was not specific to
this experiment.
