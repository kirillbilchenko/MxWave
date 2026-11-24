"""Profile C8 quality repeats and retain exact row membership before testing a fixed cohort."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from admission_quality import collect_concurrent
from benchmark_full_model import request, write
from c8_forward_trace import forward_signature
from check_full_model import rpc
from decode_quality import compare
from measure_high_concurrency import admission, configure, snapshot
from measure_quality_model import source_check
from quality_suite import BASE


def controlled_request(body: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Queue eight original prompts on this disposable service and resume in all cases."""
    if BASE != "http://127.0.0.1:18784" or len(body["prompt"]) != 8:
        raise ValueError("Keep-pause is restricted to this isolated eight-request diagnostic")
    receipt: dict[str, Any] = {"started_epoch": time.time(), "grace_seconds": 1.0}
    paused = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        try:
            receipt["pause"] = request(BASE, "/pause?mode=keep&clear_cache=false", {})
            paused = True
            assert request(BASE, "/is_paused")["is_paused"]
            future = pool.submit(request, BASE, "/v1/completions", body, timeout=300)
            time.sleep(1.0)
            if future.done():
                future.result()
                raise RuntimeError("A controlled completion returned while scheduler was paused")
            assert request(BASE, "/is_paused")["is_paused"]
        finally:
            if paused:
                receipt["resume"] = request(BASE, "/resume", {})
        result = future.result()
    receipt["finished_epoch"] = time.time()
    return result, receipt


def collect_controlled(
    contexts: list[list[int]], root: Path, label: str, vocabulary: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Keep all sixteen original contexts; prove actual grouping from separate worker records."""
    path = root / f"{label}-decode-quality.json"
    assert not path.exists() and len(contexts) == 16
    report: dict[str, Any] = {
        "status": "running",
        "label": label,
        "vocabulary": vocabulary,
        "generated_tokens": 4,
        "concurrency": 8,
        "contexts": [],
        "scope": "Eight prompts queued under keep-pause; native 2048-token scheduler unchanged",
    }
    receipts: list[dict[str, Any]] = []
    write(path, report)
    for offset in (0, 8):
        prompts = contexts[offset : offset + 8]
        response, receipt = controlled_request(
            {
                "model": "mxwave-glm53-pilot",
                "prompt": prompts,
                "max_tokens": 4,
                "min_tokens": 4,
                "temperature": 0,
                "seed": 42,
                "ignore_eos": True,
                "return_token_ids": True,
                "return_tokens_as_token_ids": True,
                "logprobs": vocabulary,
            }
        )
        receipts.append(receipt)
        write(root / f"{label}-admission-proofs.json", receipts)
        choices = sorted(response["choices"], key=lambda row: row["index"])
        assert [c["index"] for c in choices] == list(range(8))
        for i, choice in enumerate(choices):
            tokens, tops = choice["token_ids"], choice["logprobs"]["top_logprobs"]
            assert len(tokens) == len(tops) == 4
            context = {
                "index": offset + i,
                "prompt_sha256": hashlib.sha256(
                    json.dumps(prompts[i], separators=(",", ":")).encode()
                ).hexdigest(),
                "token_ids": tokens,
                "steps": [],
            }
            for position, top in enumerate(tops):
                row = np.full(vocabulary, -np.inf, dtype="<f4")
                for key, value in top.items():
                    assert key.startswith("token_id:") and not math.isnan(value)
                    token = int(key.removeprefix("token_id:"))
                    assert 0 <= token < vocabulary
                    row[token] = value
                assert len(top) == vocabulary and np.isfinite(row).any()
                file = root / f"{label}-ctx{offset + i:02d}-step{position}.f32"
                assert not file.exists()
                file.write_bytes(row.tobytes())
                context["steps"].append(
                    {
                        "position": position,
                        "kind": "prefill" if position == 0 else "decode",
                        "conditioning_token_ids": tokens[:position],
                        "file": file.name,
                        "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                    }
                )
            report["contexts"].append(context)
        write(path, report)
    report["status"] = "complete"
    write(path, report)
    return report, receipts


def passed(value: dict[str, Any]) -> bool:
    """Use the existing complete-conditioning KL and absolute-NLL gates without relaxation."""
    return bool(
        value["decode_steps"] == value["expected_decode_steps"] == 48
        and value["all_decode_conditioning_matches"]
        and value["all_generated_ids_match"]
        and value["mean_decode_kl"] is not None
        and value["mean_decode_kl"] <= 1e-5
        and abs(value["mean_decode_nll_difference"]) <= 1e-4
    )


def main() -> None:
    """Measure ordinary repeats before profiling; keep controlled results explicitly diagnostic."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--vocabulary", required=True, type=int)
    args = parser.parse_args()
    root = args.root
    plan = json.loads((root / "c8-trace-plan.json").read_bytes())
    raw = (root / "protocol.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == plan["source_protocol_sha256"]
    protocol = json.loads(raw)
    contexts = [protocol.get("quality_prefix_token_ids", []) + x for x in protocol["quality"]]
    assert len(contexts) == 16 and all(len(x) == 258 for x in contexts)
    path = root / "c8-trace-review.json"
    assert not path.exists()
    report: dict[str, Any] = {
        "status": "running",
        "started_epoch": time.time(),
        "plan_sha256": hashlib.sha256((root / "c8-trace-plan.json").read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "groups": [],
        "comparisons": [],
        "quality_qualified": False,
        "stock_only_no_compression": True,
        "prior_failures_preserved": True,
        "throughput_benchmark_run": False,
        "model_or_scheduler_arithmetic_changed": False,
        "MTP_deferred_by_user": True,
    }

    def save(phase: str) -> None:
        report.update(phase=phase, updated_epoch=time.time())
        write(path, report)

    try:
        save("source_before")
        audit = json.loads((root / "source-audit.json").read_bytes())
        report["source_before"] = source_check("control", audit)
        report["loaded_checks"] = rpc(BASE, "mxwave_checkpoint_check")
        assert len(report["loaded_checks"]) == 2
        assert all(x["status"] == "passed" for x in report["loaded_checks"])
        report["sparse_policy"] = rpc(BASE, "mxwave_stable_sparse_policy")
        assert len(report["sparse_policy"]) == 2
        assert all(
            x["patched_aliases"] and x["actual_decode_context_parallel_size"] == 1
            for x in report["sparse_policy"]
        )
        report["mhc_warmup"] = rpc(BASE, "mxwave_mhc_warmup", [2048])
        report["graphs"] = rpc(BASE, "mxwave_set_eager", [False])
        report["snapshot_before"] = snapshot()
        # Finish all unprofiled repeats before the first profiler session.
        scenarios = [
            ("ordinary", False),
            ("controlled", False),
            ("ordinary", True),
            ("controlled", True),
        ]
        assert plan["scenarios"] == [
            {"admission": mode, "profiled": profiled} for mode, profiled in scenarios
        ]
        for mode, profiled in scenarios:
            configure(3)
            prefix = f"{mode}-{'profiled' if profiled else 'unprofiled'}"
            for repeat in range(3):
                label = f"{prefix}-r{repeat}"
                group: dict[str, Any] = {
                    "label": label,
                    "admission": mode,
                    "profiled": profiled,
                    "repeat": repeat,
                    "started_epoch": time.time(),
                }
                report["groups"].append(group)
                save(label)
                group["trace_armed"] = rpc(BASE, "mxwave_c8_trace", [True, label])
                started = False
                try:
                    if profiled:
                        group["profile_start"] = request(BASE, "/start_profile", {})
                        started = True
                    if mode == "ordinary":
                        collected = collect_concurrent(
                            BASE, contexts, root, label, args.vocabulary, concurrency=8
                        )
                    else:
                        collected, group["controlled_admission_proofs"] = collect_controlled(
                            contexts, root, label, args.vocabulary
                        )
                    group["status"] = "complete"
                finally:
                    if started:
                        group["profile_stop"] = request(
                            BASE, "/stop_profile", {}, timeout=240
                        )
                    group["worker_trace"] = rpc(BASE, "mxwave_c8_trace", [False, label])
                    group["admission_report"] = admission("mxwave_admission_report")
                    group["finished_epoch"] = time.time()
                    write(root / f"{label}-forward-traces.json", group["worker_trace"])
                    save(label + "-retained")
                signatures = [
                    [forward_signature(e) for e in worker["events"]]
                    for worker in group["worker_trace"]
                ]
                assert len(signatures) == 2 and signatures[0] == signatures[1]
                group["TP_forward_signatures_match"] = True
                if repeat:
                    first_label = f"{prefix}-r0"
                    prior = json.loads((root / f"{first_label}-decode-quality.json").read_bytes())
                    value = compare(prior, collected, root)
                    first = next(g for g in report["groups"] if g["label"] == first_label)
                    first_sig = [
                        forward_signature(e) for e in first["worker_trace"][0]["events"]
                    ]
                    report["comparisons"].append(
                        {
                            "reference": first_label,
                            "candidate": label,
                            **value,
                            "passed": passed(value),
                            "actual_batch_shape_and_prompt_order_equal": first_sig == signatures[0],
                        }
                    )
                save(label + "-reviewed")
        report["source_after"] = source_check("control", audit)
        report["snapshot_after"] = snapshot()
        report.update(status="complete", finished_epoch=time.time())
        save("complete")
    except BaseException as exc:
        report.update(status="failed", error=repr(exc), finished_epoch=time.time())
        save("failed")
        raise


if __name__ == "__main__":
    main()
