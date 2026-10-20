"""Validate the diagnostic streams and actual six-plus-two forward membership."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from itertools import pairwise
from pathlib import Path
from typing import Any

from c8_forward_trace import forward_signature


def prompt_hash(ids: list[int]) -> str:
    """Use the same explicit token serialization as the native batch observer."""
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def split_membership(events: list[dict[str, Any]], prompts: list[list[int]]) -> dict[str, Any]:
    """Prove the first six and later two from actual scheduled prompt identities."""
    if len(prompts) != 8 or not events:
        raise ValueError("Split membership requires eight prompts and observed forwards")
    first_hashes, late_hashes = (
        set(map(prompt_hash, prompts[:6])),
        set(map(prompt_hash, prompts[6:])),
    )
    if len(first_hashes | late_hashes) != 8:
        raise ValueError("Diagnostic prompt identities must be distinct")
    first = events[0]
    rows = first["prepared"]["row_order"]
    first_ok = (
        len(rows) == 6
        and {r["prompt_sha256"] for r in rows} == first_hashes
        and all(r["scheduled_tokens"] == r["prompt_tokens"] == 256 for r in rows)
        and first["prepared"]["has_prefill"]
        and first["prepared"]["num_tokens"] == 1536
    )
    late = next(
        (
            e
            for e in events
            if any(r["prompt_sha256"] in late_hashes for r in e["prepared"]["row_order"])
        ),
        None,
    )
    late_ok = False
    if late is not None:
        rows = late["prepared"]["row_order"]
        new = [r for r in rows if r["prompt_sha256"] in late_hashes]
        old = [r for r in rows if r["prompt_sha256"] in first_hashes]
        late_ok = (
            len(rows) == 8
            and {r["prompt_sha256"] for r in new} == late_hashes
            and len(new) == 2
            and all(r["scheduled_tokens"] == 256 for r in new)
            and len(old) == 6
            and all(r["scheduled_tokens"] == 1 for r in old)
            and late["prepared"]["has_prefill"]
            and late["prepared"]["num_tokens"] == 518
        )
    return {
        "first_six_confirmed": first_ok,
        "two_join_six_decode_confirmed": late_ok,
        "confirmed": first_ok and late_ok,
        "first_sequence": first["sequence"],
        "late_entry_sequence": None if late is None else late["sequence"],
    }


def review(root: Path) -> dict[str, Any]:
    """Retain all outcomes, including a requested split that the scheduler did not realize."""
    plan_path = root / "latency-probe-plan.json"
    plan_raw = plan_path.read_bytes()
    plan = json.loads(plan_raw)
    protocol_raw = (root / "protocol.json").read_bytes()
    assert hashlib.sha256(protocol_raw).hexdigest() == plan["source_protocol_sha256"]
    protocol = json.loads(protocol_raw)
    audit = json.loads((root / "source-audit.json").read_bytes())
    assert len(audit["cases"]) == 168
    prompts = {(v["batch"], v["prompt_tokens"]): v["prompts"] for v in protocol["timing"]}
    summaries, paired = [], {}
    for arm in ("control", "candidate"):
        data = json.loads((root / f"{arm}-latency-probe.json").read_bytes())
        assert data["status"] == "complete" and len(data["groups"]) == len(plan["groups"]) == 15
        assert data["plan_sha256"] == hashlib.sha256(plan_raw).hexdigest()
        assert data["source_sha256"] == plan["source_sha256"]["measure_latency_probe.py"]
        assert len(data["source_before"]) == len(data["source_after"]) == 2
        for stage in ("source_before", "source_after"):
            assert {v["tp_rank"] for v in data[stage]} == {0, 1}
            for source in data[stage]:
                assert source["status"] == "passed" and source["candidate"] == (arm == "candidate")
                expected = Counter(
                    (tuple(v["shape"]), v["raw_sha256"])
                    for v in audit["cases"]
                    if v["rank"] == source["tp_rank"]
                )
                actual = Counter(
                    (tuple(v["shape"]), v["sha256"]) for v in source["source_scale_hashes"]
                )
                assert expected == actual
                if arm == "candidate":
                    assert source["native_sha256"] == (
                        "9f5d96dcb0762d7e5ed5436ca296f70d590e73797d40522e4d8c7d6f3902f5f0"
                    )
        assert all(
            v["native_mhc_unchanged"]
            for key in ("native_mhc_policy", "native_mhc_policy_after")
            for v in data[key]
        )
        assert len(data["profile_sessions"]) == 1
        assert data["profile_sessions"][0]["status"] == "complete"
        warmup_shapes: set[int] = set()
        for item, spec in zip(data["groups"], plan["groups"], strict=True):
            assert all(item[k] == v for k, v in spec.items()) and item["status"] == "complete"
            measured = item["measurement"]
            outputs = measured["requests"]
            assert len(outputs) == spec["concurrency"]
            all_gaps = []
            for index, output in enumerate(outputs):
                assert len(output["token_ids"]) == spec["generated_tokens"]
                assert sum(a["tokens"] for a in output["arrivals"]) == spec["generated_tokens"]
                assert output["usage"]["prompt_tokens"] == spec["prompt_tokens"]
                assert output["usage"]["completion_tokens"] == spec["generated_tokens"]
                arrivals = output["arrivals"]
                start, end = (
                    output["request_start_monotonic_ns"],
                    output["request_end_monotonic_ns"],
                )
                assert start <= arrivals[0]["monotonic_ns"] <= arrivals[-1]["monotonic_ns"] <= end
                assert [a["monotonic_ns"] for a in arrivals] == sorted(
                    a["monotonic_ns"] for a in arrivals
                )
                assert math.isclose(output["elapsed_seconds"], (end - start) / 1e9, abs_tol=1e-12)
                for arrival in arrivals:
                    assert math.isclose(
                        arrival["seconds"], (arrival["monotonic_ns"] - start) / 1e9, abs_tol=1e-12
                    )
                all_gaps.extend(
                    (b["monotonic_ns"] - a["monotonic_ns"]) / 1e9 for a, b in pairwise(arrivals)
                )
                paired[arm, spec["name"], index] = output["token_ids"]
            workers = sorted(item["worker_trace"], key=lambda w: w["rank"])
            assert [w["rank"] for w in workers] == [0, 1]
            assert all(
                not w["enabled"]
                and not w["GPU_tensors_copied"]
                and not w["CUDA_synchronization_added"]
                for w in workers
            )
            signatures = [[forward_signature(e) for e in w["events"]] for w in workers]
            assert signatures[0] == signatures[1]
            scheduled: Counter[str] = Counter()
            for event in workers[0]["events"]:
                batch = event["prepared"]
                rows = batch["row_order"]
                assert len(rows) == batch["num_reqs"]
                assert 0 < batch["num_tokens"] <= 2048
                assert batch["num_tokens_after_padding"] >= batch["num_tokens"]
                assert sum(r["scheduled_tokens"] for r in rows) == batch["num_tokens"]
                for row in rows:
                    scheduled[row["prompt_sha256"]] += row["scheduled_tokens"]
            expected_tokens = Counter(
                {
                    prompt_hash(ids): len(ids) + spec["generated_tokens"] - 1
                    for ids in prompts[spec["concurrency"], spec["prompt_tokens"]]
                }
            )
            assert scheduled == expected_tokens, "Missing or duplicated model work in observer"
            stats = []
            for worker in workers:
                events, telemetry = worker["events"], worker["telemetry"]
                assert events and all(e["status"] == "complete" for e in events)
                assert telemetry["thread_stopped"]
                assert not telemetry["settings_changed"]
                assert all(
                    e["host_elapsed_ns"] >= 0 and e["host_thread_cpu_ns"] >= 0 for e in events
                )
                samples = telemetry["samples"]
                assert samples and all(
                    s["sm_MHz"] > 0 and 0 < s["temperature_C"] < 100 for s in samples
                )
                stats.append(
                    {
                        "rank": worker["rank"],
                        "forwards": len(events),
                        "telemetry_errors": telemetry["errors"],
                        "telemetry_limit_reached": telemetry["limit_reached"],
                        "samples": len(samples),
                        "clock_MHz_range": [
                            min(s["sm_MHz"] for s in samples),
                            max(s["sm_MHz"] for s in samples),
                        ],
                        "temperature_C_range": [
                            min(s["temperature_C"] for s in samples),
                            max(s["temperature_C"] for s in samples),
                        ],
                        "maximum_sample_gap_seconds": max(
                            (
                                (b["sample_start_monotonic_ns"] - a["sample_start_monotonic_ns"])
                                / 1e9
                                for a, b in pairwise(samples)
                            ),
                            default=None,
                        ),
                        "prefill_forwards": [
                            {
                                "sequence": e["sequence"],
                                "prepared": e["prepared"],
                                "host_elapsed_ms": e["host_elapsed_ns"] / 1e6,
                                "host_thread_cpu_ms": e["host_thread_cpu_ns"] / 1e6,
                                "host_prepare_ms": e["host_prepare_elapsed_ns"] / 1e6,
                            }
                            for e in events
                            if e["prepared"]["has_prefill"]
                        ],
                    }
                )
            result = {
                "arm": arm,
                **spec,
                "maximum_client_arrival_gap_seconds": max(all_gaps),
                "median_client_arrival_gap_seconds": statistics.median(all_gaps),
                "client_arrival_chunks_all_single_token": all(
                    a["tokens"] == 1 for output in outputs for a in output["arrivals"]
                ),
                "workers": stats,
            }
            if spec["admission"] == "split":
                result["actual_split"] = split_membership(workers[0]["events"], prompts[8, 256])
                result["1536_row_prefill_seen_in_warmup"] = 1536 in warmup_shapes
                anchor = measured["resume_return_monotonic_ns"]
                result["seconds_after_resume_to_last_arrival"] = (
                    max(o["arrivals"][-1]["monotonic_ns"] for o in outputs) - anchor
                ) / 1e9
                result["first_token_seconds_after_resume"] = [
                    (o["arrivals"][0]["monotonic_ns"] - anchor) / 1e9 for o in outputs
                ]
            if spec["stage"] == "warmup":
                warmup_shapes.update(
                    e["prepared"]["num_tokens"]
                    for e in workers[0]["events"]
                    if e["prepared"]["has_prefill"]
                )
            summaries.append(result)
    comparisons = [
        {
            "group": name,
            "request_index": index,
            "tokens_equal": ids == paired["candidate", name, index],
        }
        for (arm, name, index), ids in paired.items()
        if arm == "control"
    ]
    assert len(comparisons) == 106
    return {
        "status": "complete",
        "quality_qualified": False,
        "diagnostic_not_serving_speed": True,
        "groups": summaries,
        "paired_outputs": comparisons,
        "all_paired_token_sequences_equal": all(v["tokens_equal"] for v in comparisons),
        "all_requested_splits_observed": all(
            s["actual_split"]["confirmed"] for s in summaries if s["admission"] == "split"
        ),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "GPU_trace_analysis_pending": True,
        "limits": [
            (
                "Controlled queue delay and profiling change latency; "
                "these are not ordinary speed gates."
            ),
            "Client-arrival gaps are not isolated GPU kernel duration.",
            "Host elapsed calls include asynchronous submission; they are not GPU completion.",
            (
                "Only executing-thread CPU time is counted; "
                "other compiler/helper threads are excluded."
            ),
            (
                "The first observed controlled split may have had its shape warmed earlier; "
                "report that fact."
            ),
            (
                "Profiling precedes unprofiled diagnostic repeats; "
                "retain that order and observer effects."
            ),
            "Clock samples may miss shorter events; sample gaps and failures are retained.",
        ],
    }


def main() -> None:
    """Write the controller-compatible review without promoting the model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    result = review(args.root)
    with (args.root / "quality-model-review.json").open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("groups", "paired_outputs")}))


if __name__ == "__main__":
    main()
