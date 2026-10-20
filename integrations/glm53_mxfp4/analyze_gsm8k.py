"""Regrade every paired full-test GSM8K response without amending earlier qualification failures."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from benchmark_full_model import write
from gsm8k_benchmark import body, score, summarize


def review(root: Path) -> dict[str, Any]:
    """Require all 2,638 raw responses and report paired accuracy, strict format and memory."""
    protocol_path = root / "gsm8k-protocol.json"
    protocol = json.loads(protocol_path.read_bytes())
    arms, summaries, allocations = {}, {}, {}
    audit = json.loads((root / "source-audit.json").read_bytes())
    for label in ("control", "candidate"):
        report = json.loads((root / (label + "-gsm8k.json")).read_bytes())
        if (report["status"] != "complete" or len(report["cases"]) != 1319
                or report["protocol_sha256"] != hashlib.sha256(protocol_path.read_bytes()).hexdigest()
                or [row["source_index"] for row in report["cases"]] != list(range(1319))):
            raise ValueError("Incomplete or changed full-test benchmark evidence")
        for case, compact in zip(protocol["cases"], report["cases"], strict=True):
            if Path(compact["file"]).name != compact["file"]:
                raise ValueError("Case evidence must be local to the result directory")
            raw = (root / label / "gsm8k" / compact["file"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != compact["sha256"]:
                raise ValueError("Raw GSM8K answer changed")
            row = json.loads(raw)
            if any(value != row.get(key) for key, value in compact.items()
                   if key not in ("file", "sha256")):
                raise ValueError("Compact score differs from the hash-bound raw case")
            if (row["source_index"] != case["source_index"] or row["expected"] != case["expected"]
                    or row["prompt_sha256"] != hashlib.sha256(case["prompt"].encode()).hexdigest()
                    or row["request_sha256"] != hashlib.sha256(
                        json.dumps(body(case, protocol), sort_keys=True).encode()).hexdigest()):
                raise ValueError("Question, prompt, answer or generation settings changed")
            if not row["api_error"]:
                recalculated = score(row["response"], case["expected"])
                if any(row[key] != value for key, value in recalculated.items()):
                    raise ValueError("Recorded answer score disagrees with raw response")
            elif row["answer_exact_match"] or row["strict_format_match"]:
                raise ValueError("API failures must count as failed questions")
        summaries[label] = summarize(report["cases"])
        if summaries[label] != report["summary"]:
            raise ValueError("Recorded accuracy disagrees with retained raw responses")
        allocations[label] = sum(row["allocation"]["cuda_allocated_bytes"]
                                 for row in report["source_after"])
        for phase in ("source_before", "source_after"):
            if len(report[phase]) != 2 or {r["tp_rank"] for r in report[phase]} != {0, 1}:
                raise ValueError("Incomplete source rank coverage")
            for row in report[phase]:
                expected = Counter((tuple(c["shape"]), c["raw_sha256"])
                                   for c in audit["cases"] if c["rank"] == row["tp_rank"])
                actual = Counter((tuple(c["shape"]), c["sha256"])
                                 for c in row["source_scale_hashes"])
                if (row["status"] != "passed" or actual != expected
                        or row["candidate"] != (label == "candidate")):
                    raise ValueError("Loaded model source changed")
        arms[label] = report
    differences = []
    for left, right in zip(arms["control"]["cases"], arms["candidate"]["cases"], strict=True):
        if (left["answer_exact_match"] != right["answer_exact_match"]
                or left["strict_format_match"] != right["strict_format_match"]):
            differences.append({"source_index": left["source_index"], "control": {
                key: left[key] for key in ("answer_exact_match", "strict_format_match")},
                "candidate": {key: right[key] for key in (
                    "answer_exact_match", "strict_format_match")}})
    lost = sum(row["control"]["answer_exact_match"]
               and not row["candidate"]["answer_exact_match"] for row in differences)
    gained = sum(not row["control"]["answer_exact_match"]
                 and row["candidate"]["answer_exact_match"] for row in differences)
    discordant = lost + gained
    p = min(1., 2 * sum(math.comb(discordant, k) for k in range(min(lost, gained) + 1))
            / (2 ** discordant)) if discordant else 1.
    return {"status": "complete", "quality_qualified": False, "qualification_attempted": False,
        "reference_repeatability_blocker_preserved": True,
        "previous_failed_qualifications_preserved": True, "questions_per_arm": 1319,
        "fewshot": 0, "concurrency": 4, "speculative_decoding": False,
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "scores": summaries, "accuracy_change_percentage_points": 100 * (
            summaries["candidate"]["accuracy"] - summaries["control"]["accuracy"]),
        "reference_correct_candidate_wrong": lost, "reference_wrong_candidate_correct": gained,
        "discordant_answer_pairs": discordant, "mcnemar_exact_two_sided_p": p,
        "question_differences": differences,
        "allocated_cuda_bytes_after_inference": allocations,
        "combined_cuda_saving_bytes": allocations["control"] - allocations["candidate"],
        "benchmark_wall_seconds": {label: row["benchmark_finished_epoch"]
            - row["benchmark_started_epoch"] for label, row in arms.items()},
        "limits": protocol["limits"] + [
            "Wilson bounds describe question-level sampling uncertainty, not repeat-run noise.",
            "Matched-pair score significance does not prove exact arithmetic or causal attribution."]}


def main() -> None:
    """Write complete benchmark numbers even when accuracy is lower or strict gates remain open."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = review(args.root)
    result["analysis_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write(args.root / "quality-model-review.json", result)


if __name__ == "__main__":
    main()
