"""Review frozen quality evidence without weakening the reference stability gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from analyze_palette_model import bitwise, numerical_gates
from benchmark_full_model import write
from decode_quality import compare


def heldout_values(root: Path, label: str, protocol: dict[str, Any]) -> dict[str, Any]:
    """Recompute token-weighted loss from hash-bound, aligned teacher-forced logprobs."""
    report = json.loads((root / (label + "-heldout.json")).read_bytes())
    windows = protocol["wiki_windows"]
    if report["status"] != "complete" or len(report["windows"]) != len(windows):
        raise ValueError("Incomplete held-out evidence")
    total, count = 0.0, 0
    for index, (row, ids) in enumerate(zip(report["windows"], windows, strict=True)):
        expected = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
        if row["index"] != index or row["prompt_sha256"] != expected:
            raise ValueError("Held-out token conditioning changed")
        if Path(row["file"]).name != row["file"]:
            raise ValueError("Held-out artifact must be local to its result directory")
        raw = (root / row["file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("Held-out logprob bytes changed")
        values = np.frombuffer(raw, dtype="<f4").astype(np.float64)
        if len(values) != len(ids) - 1 or len(values) != row["tokens"]:
            raise ValueError("Held-out scored-token count changed")
        if not np.isfinite(values).all() or np.any(values > 1e-5):
            raise ValueError("Invalid held-out logprobs")
        total -= float(values.sum())
        count += len(values)
    if not count or count != protocol["wiki_scored_tokens"]:
        raise ValueError("Wrong held-out token coverage")
    return {
        "scored_tokens": count,
        "mean_nll": total / count,
        "perplexity": math.exp(total / count),
    }


def task_scores(root: Path, label: str, protocol: dict[str, Any]) -> list[dict[str, Any]]:
    """Require the frozen task set, finished sandbox scoring and strict code output format."""
    report = json.loads((root / label / "real-work.json").read_bytes())
    if report["status"] != "complete":
        raise ValueError("Practical-task scoring is unfinished")
    expected = {c["name"]: c for c in protocol["tasks"]}
    if len(report["cases"]) != len(expected) or {r["name"] for r in report["cases"]} != set(
        expected
    ):
        raise ValueError("Practical task coverage changed")
    rows = []
    for row in report["cases"]:
        case = expected[row["name"]]
        if row["prompt_sha256"] != hashlib.sha256(case["prompt"].encode()).hexdigest():
            raise ValueError("Practical task prompt changed")
        if row.get("sandbox_pending") or row.get("passed") is None:
            raise ValueError("Code sandbox scoring is incomplete")
        passed = bool(row["passed"] and (case["kind"] != "code" or row.get("format_passed", False)))
        rows.append(
            {
                "name": row["name"],
                "kind": case["kind"],
                "passed": passed,
                "finish_reason": row.get("finish_reason"),
                "semantic_passed": row["passed"],
                "inference_completed": isinstance(row.get("response"), dict),
                "format_passed": row.get("format_passed"),
                "answer_sha256": hashlib.sha256(row.get("final_answer", "").encode()).hexdigest(),
                "error": row.get("error"),
            }
        )
    return rows


def trace_review(root: Path, label: str) -> dict[str, Any] | None:
    """Locate the first differing eager output per rank without treating intrusive hashes as proof."""
    path = root / (label + "-repeat-trace/repeat-fingerprints.json")
    if not path.exists():
        return None
    report = json.loads(path.read_bytes())
    result: dict[str, Any] = {
        "status": report["status"],
        "intrusive_cpu_sync": True,
        "first_difference_by_rank": {},
    }
    if report["status"] != "complete" or len(report["runs"]) != 2:
        return result
    left, right = [{r["rank"]: r["events"] for r in run["workers"]} for run in report["runs"]]
    for rank in (0, 1):
        if len(left[rank]) != len(right[rank]):
            result["first_difference_by_rank"][rank] = {"error": "Different event counts"}
            continue
        different = next(
            (
                {
                    "event": i,
                    "module": a["module"],
                    "other_module": b["module"],
                    "left": a["outputs"],
                    "right": b["outputs"],
                }
                for i, (a, b) in enumerate(zip(left[rank], right[rank], strict=True))
                if a != b
            ),
            None,
        )
        result["first_difference_by_rank"][rank] = different
    return result


def review(root: Path) -> dict[str, Any]:
    """Keep exact storage, numerical stability, held-out loss and practical outcomes separate."""
    protocol = json.loads((root / "quality-protocol.json").read_bytes())
    protocol_sha = hashlib.sha256((root / "quality-protocol.json").read_bytes()).hexdigest()
    modes, raw_equal = {}, {}
    for mode in ("graph", "eager"):
        reports = {
            label: json.loads((root / (label + "-" + mode + "-decode-quality.json")).read_bytes())
            for label in ("control", "candidate")
        }
        repeats = {
            label: json.loads(
                (root / (label + "-" + mode + "-repeat-decode-quality.json")).read_bytes()
            )
            for label in ("control", "candidate")
        }
        aa = compare(reports["control"], repeats["control"], root)
        ab = compare(reports["control"], reports["candidate"], root)
        bb = compare(reports["candidate"], repeats["candidate"], root)
        stable, numerical = numerical_gates(aa, ab)
        candidate_stable, _ = numerical_gates(bb, bb)
        modes[mode] = {
            "control_repeat_stable": stable,
            "candidate_repeat_stable": candidate_stable,
            "numerical_gate_passed": numerical and candidate_stable,
            "control_repeat": aa,
            "candidate": ab,
            "candidate_repeat": bb,
        }
        raw_equal[mode] = {
            "control_repeat": bitwise(reports["control"], repeats["control"]),
            "candidate": bitwise(reports["control"], reports["candidate"]),
            "candidate_repeat": bitwise(reports["candidate"], repeats["candidate"]),
        }
    losses = {
        label: heldout_values(root, label, protocol)
        for label in ("control", "control-repeat", "candidate", "candidate-repeat")
    }
    aa_nll = losses["control-repeat"]["mean_nll"] - losses["control"]["mean_nll"]
    ab_nll = losses["candidate"]["mean_nll"] - losses["control"]["mean_nll"]
    bb_nll = losses["candidate-repeat"]["mean_nll"] - losses["candidate"]["mean_nll"]
    limit = max(1e-5, 5 * abs(aa_nll))
    heldout_passed = abs(aa_nll) <= 1e-4 and abs(bb_nll) <= 1e-4 and abs(ab_nll) <= limit
    tasks = {label: task_scores(root, label, protocol) for label in ("control", "candidate")}
    candidate = {row["name"]: row for row in tasks["candidate"]}
    lost = [
        r["name"] for r in tasks["control"] if r["passed"] and not candidate[r["name"]]["passed"]
    ]
    gained = [
        r["name"] for r in tasks["control"] if not r["passed"] and candidate[r["name"]]["passed"]
    ]
    inference_failures = {
        label: [r["name"] for r in rows if not r["inference_completed"]]
        for label, rows in tasks.items()
    }
    practical_valid = not any(inference_failures.values())
    categories = {
        label: {
            kind: {
                "passed": sum(r["passed"] for r in rows if r["kind"] == kind),
                "total": sum(r["kind"] == kind for r in rows),
            }
            for kind in sorted({r["kind"] for r in rows})
        }
        for label, rows in tasks.items()
    }
    measurements = {
        label: json.loads((root / (label + "-quality-measurement.json")).read_bytes())
        for label in ("control", "candidate")
    }
    storage, allocations = True, {}
    before_sources: dict[str, Any] = {}
    for label, measurement in measurements.items():
        if (
            measurement["status"] != "collected_pending_code_sandbox"
            or measurement["quality_protocol_sha256"] != protocol_sha
        ):
            raise ValueError("Incomplete measurement or changed protocol")
        storage &= bool(measurement["source_diagnostic"]["all_source_checks"])
        for phase in ("source_before", "source_after"):
            rows = measurement[phase]
            if len(rows) != 2 or {r["tp_rank"] for r in rows} != {0, 1}:
                raise ValueError("Incomplete TP2 source coverage")
            for row in rows:
                storage &= row["status"] == "passed" and row["scale_groups"] == 84
                hashes = Counter(
                    (tuple(c["shape"]), c["sha256"]) for c in row["source_scale_hashes"]
                )
                key = str(row["tp_rank"])
                if label == "control" and phase == "source_before":
                    before_sources[key] = hashes
                else:
                    storage &= before_sources[key] == hashes
        allocations[label] = sum(
            r["allocation"]["cuda_allocated_bytes"] for r in measurement["source_after"]
        )
    qualified = bool(
        storage
        and modes["graph"]["numerical_gate_passed"]
        and heldout_passed
        and practical_valid
        and not lost
    )
    return {
        "status": "complete",
        "quality_qualified": qualified,
        "scale_storage_lossless": bool(storage),
        "default_dispatch": "graph",
        "decode_modes": modes,
        "full_logprob_arrays_bitwise_equal": raw_equal,
        "end_to_end_bitwise_losslessness_demonstrated": bool(
            storage and all(all(value.values()) for value in raw_equal.values())
        ),
        "heldout": {
            "measurements": losses,
            "control_repeat_nll_delta": aa_nll,
            "candidate_nll_delta": ab_nll,
            "candidate_repeat_nll_delta": bb_nll,
            "candidate_perplexity_change_percent": 100 * math.expm1(ab_nll),
            "absolute_candidate_nll_limit": limit,
            "gate_passed": heldout_passed,
        },
        "practical": {
            "scores": tasks,
            "categories": categories,
            "reference_passed_candidate_failed": lost,
            "reference_failed_candidate_passed": gained,
            "no_reference_pass_lost": not lost,
            "all_inference_requests_completed": practical_valid,
            "inference_failures": inference_failures,
        },
        "allocated_cuda_bytes_after_inference": allocations,
        "combined_cuda_saving_bytes": allocations["control"] - allocations["candidate"],
        "repeat_fingerprints": {label: trace_review(root, label) for label in measurements},
        "quality_protocol_sha256": protocol_sha,
        "limits": protocol["limits"]
        + [
            "Strict numerical repeat floors remain KL <= 1e-5 and absolute NLL <= 1e-4.",
            "Unstable reference repeats prevent qualification even if practical tasks pass.",
            "No fresh speed benchmark was run; allocation includes configured KV and scratch.",
            "Hooks in optional eager fingerprints synchronize the GPU and can alter drift.",
        ],
    }


def main() -> None:
    """Write a compact review; failed quality gates are results, not collection failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = review(args.root)
    result["analysis_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["inputs"] = {
        str(p.relative_to(args.root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in args.root.rglob("*.json")
        if p.name != "quality-model-review.json"
    }
    write(args.root / "quality-model-review.json", result)


if __name__ == "__main__":
    main()
