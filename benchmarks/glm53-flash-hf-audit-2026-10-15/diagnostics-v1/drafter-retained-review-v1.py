"""Review completed speculative-decoding evidence locally during GSM8K."""

from __future__ import annotations

import hashlib
import json
import statistics
from itertools import pairwise
from pathlib import Path

ROOT = Path("/private/tmp/mxwave-kda-mxfp8-speed-v1-final")
OUT = Path("/private/tmp/mxwave-drafter-retained-review-v1")


def digest(path: Path) -> str:
    """Bind each derived result to its retained input."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def counters(text: str) -> dict[str, float]:
    """Preserve metric names and labels when reading cumulative counters."""
    result = {}
    for line in text.splitlines():
        if line.startswith("vllm:spec_decode_"):
            name, value = line.rsplit(" ", 1)
            if name.split("{", 1)[0].endswith("_total"):
                result[name] = float(value)
    return result


def main() -> None:
    """Recompute all windows and summarize fixed-width C1 acceptance."""
    audit_path = Path("/private/tmp/mxwave-drafter-retained-speed-review.json")
    audit = json.loads(audit_path.read_bytes())
    sources = {str(audit_path): digest(audit_path), str(Path(__file__)): digest(Path(__file__))}
    rows = []
    c1 = {}
    for arm, block in audit["loads"].items():
        path = ROOT / "evidence" / f"{arm}-measurement.json"
        sources[str(path)] = digest(path)
        raw = json.loads(path.read_bytes())
        assert len(raw["trials"]) == len(block["trials"]) == 12
        for trial, recorded in zip(raw["trials"], block["trials"], strict=True):
            before = counters(trial["metrics_before"])
            after = counters(trial["metrics_after"])
            delta = {k: v - before.get(k, 0) for k, v in after.items()}
            assert delta == recorded["counter_deltas_with_labels"]
            totals = {k.split("{", 1)[0]: v for k, v in delta.items() if "_per_pos_" not in k}
            rounds = totals["vllm:spec_decode_num_drafts_total"]
            drafted = totals["vllm:spec_decode_num_draft_tokens_total"]
            accepted = totals["vllm:spec_decode_num_accepted_tokens_total"]
            positions = [recorded["accepted_by_position"][str(i)] for i in range(7)]
            assert sum(positions) == accepted
            assert all(a >= b for a, b in pairwise(positions))
            row = {
                "arm": arm, "mode": block["mode"], "case": trial["case"],
                "trial": trial["trial"], "batch": trial["batch"], "rounds": rounds,
                "drafted": drafted, "accepted": accepted, "accepted_by_position": positions,
                "acceptance_pct": 100 * accepted / drafted,
                "accepted_per_round": accepted / rounds,
                "output_token_sha256": recorded["output_token_sha256"],
            }
            if trial["batch"] == 1:
                assert drafted == 7 * rounds
                request = trial["requests"][0]
                row.update({
                    "position_survival_pct": [100 * v / rounds for v in positions],
                    "conditional_acceptance_pct": [
                        100 * v / previous if previous else None
                        for v, previous in zip(positions, [rounds, *positions[:-1]], strict=True)
                    ],
                    "decode_seconds": request["latency_seconds"] - request["ttft_seconds"],
                    "ttft_seconds": request["ttft_seconds"],
                })
                row["decode_wall_ms_per_round_proxy"] = row["decode_seconds"] * 1000 / rounds
                c1.setdefault(block["mode"], {}).setdefault(trial["case"], []).append(row)
            rows.append(row)
    assert len(rows) == 108
    summarized = {}
    for mode, cases in c1.items():
        summarized[mode] = {}
        for case, trials in cases.items():
            assert len(trials) == 6
            output_variants = len({tuple(t["output_token_sha256"]) for t in trials})
            fields = (
                "rounds", "drafted", "accepted", "accepted_by_position", "acceptance_pct",
                "accepted_per_round", "position_survival_pct", "conditional_acceptance_pct",
            )
            variants = {
                json.dumps({k: t[k] for k in fields}, sort_keys=True) for t in trials
            }
            summarized[mode][case] = {
                "trials": 6, "output_variants": output_variants,
                "counter_variants": len(variants),
                "counter_observations": [json.loads(v) for v in sorted(variants)],
                "median_decode_seconds": statistics.median(t["decode_seconds"] for t in trials),
                "median_ttft_seconds": statistics.median(t["ttft_seconds"] for t in trials),
                "median_decode_wall_ms_per_round_proxy": statistics.median(
                    t["decode_wall_ms_per_round_proxy"] for t in trials
                ),
            }
    profile_path = ROOT / "profiles-review.json"
    sources[str(profile_path)] = digest(profile_path)
    profiles = json.loads(profile_path.read_bytes())
    profile_rows = {}
    selection = {
        "num_reqs": 1, "num_tokens": 8, "num_tokens_after_padding": 8, "num_draft_tokens": 7,
    }
    for name, profile in profiles.items():
        calls = [
            c for c in profile["calls"]
            if c["label"] == "draft_propose" and c["metadata"] == selection
        ]
        assert calls
        profile_rows[name] = {
            "trace_sha256": profile["sha256"], "selection": selection,
            "calls": len(calls),
            "median_kernel_duration_sum_ms": statistics.median(
                c["kernel_duration_sum_us"] for c in calls
            ) / 1000,
            "median_kernel_span_ms": statistics.median(c["kernel_span_us"] for c in calls) / 1000,
        }
    report = {
        "status": "local_retained_evidence_review_complete",
        "source_sha256": sources,
        "windows_recomputed": len(rows),
        "matched_speculative_configs_equal": audit["all_matched_speculative_configs_equal"],
        "shared_target_heads_and_embeddings": audit["draft_contexts_share_target_head_and_embedding"],
        "C1": summarized, "C1_shape_draft_profiles": profile_rows, "windows": rows,
        "limits": [
            "No new inference, remote weight reads, or active GSM8K changes.",
            "C1 position survival divides accepted prefix counts by rounds with seven draft tokens.",
            "Concurrent windows have mixed draft lengths; no position survival rate is inferred.",
            "Counters may include a final drafted block beyond the fixed response-token cap.",
            "Profiles group equal shapes, not equal input histories; profiled spans are not API time.",
            "The startup field draft_context_buffer_dtype is _fused_kv_weight.dtype, not KV-cache dtype.",
            "Same mounted path does not prove identical runtime draft tensors; file audit is queued.",
            "Acceptance is not task accuracy; differing target histories confound causal attribution.",
        ],
        "next_diagnostic": [
            "After GSM8K, finish queued checkpoint and runtime inventory.",
            "Replay identical token histories and batch shapes across targets with seven drafts.",
            "Retain target hidden states, draft proposals/top-k scores, verification and prefix length.",
            "Compare compressed and reconstructed CT heads on the same draft hidden states.",
            "Use component swaps only after identifying a specific discrepancy; retain both outcomes.",
        ],
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "retained-audit.json").write_bytes(audit_path.read_bytes())
    (OUT / "review.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "status": report["status"], "windows": len(rows),
        "code": {m: v["code_256"]["counter_observations"] for m, v in summarized.items()},
    }))


if __name__ == "__main__":
    main()
