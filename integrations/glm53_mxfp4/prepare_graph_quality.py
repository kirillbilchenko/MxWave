"""Freeze an expanded graph-four quality protocol without starting a GPU engine."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from itertools import pairwise
from pathlib import Path
from typing import Any


def expand_suite(
    legacy: dict[str, Any], token_ids: list[int], gsm: list[dict[str, str]],
) -> dict[str, Any]:
    """Preserve all old tasks and tokenizer evidence while adding a fixed larger test sample."""
    if len(token_ids) != legacy["wiki_total_tokens"] or len(token_ids) < 100 * 1025:
        raise ValueError("Expanded corpus length must match the pinned tokenizer's old evidence")
    for offset, window in zip(legacy["wiki_offsets"], legacy["wiki_windows"], strict=True):
        if token_ids[offset:offset + len(window)] != window:
            raise ValueError("Offline tokenizer differs from the previously observed API tokens")
    old = legacy["tasks"]
    math = [row for row in old if row["kind"] == "math"]
    indices = {row["source_index"] for row in math}
    if len(old) != 34 or len(math) != 24:
        raise ValueError("Require the original complete 34-case / 24-math protocol")
    if len(indices) != 24:
        raise ValueError("Legacy source questions must be unique")
    for row in math:
        expected = gsm[row["source_index"]]["answer"].rsplit("####", 1)[1].strip()
        if expected != row["expected"]:
            raise ValueError("Legacy answer does not match the pinned source question")
    remaining = sorted(set(range(len(gsm))) - indices)
    added = sorted(random.Random(20261018).sample(remaining, 76))
    tasks = copy.deepcopy(old)
    tasks.extend({"name": f"gsm8k_{i:04d}", "kind": "math", "source_index": i,
                  "prompt": gsm[i]["question"]
                  + "\nSolve the problem. End your final answer with #### "
                  "followed by the numeric answer.",
                  "expected": gsm[i]["answer"].rsplit("####", 1)[1].strip()}
                 for i in added)
    offsets = [i * (len(token_ids) - 1025) // 99 for i in range(100)]
    if any(right - left < 1025 for left, right in pairwise(offsets)):
        raise ValueError("Expanded held-out windows must not overlap")
    suite = copy.deepcopy(legacy)
    suite.update(version=2, wiki_windows=[token_ids[i:i + 1025] for i in offsets],
                 wiki_offsets=offsets, wiki_scored_tokens=100 * 1024, tasks=tasks,
                 sample_seed=20261018, legacy_task_names=[row["name"] for row in old],
                 added_gsm8k_source_indices=added, heldout_repeats=3,
                 graph_four_numerical_repeats=5,
                 practical_gate="No task passed by the reference may fail in the candidate",
                 heldout_gate={"absolute_mean_nll": 1e-5,
                               "reference_repeat_max_absolute_mean_nll": 1e-4},
                 previous_failed_qualifications_preserved=True,
                 limits=["100 sampled non-overlapping WikiText windows; not full-corpus PPL.",
                         "100 fixed GSM8K questions plus ten practical cases; not full GSM8K.",
                         "Temporal repeats estimate runtime variability, not independent tasks.",
                         "Same imported MXFP4 source; source quantization error is not measured."])
    return suite


def main() -> None:
    """Tokenize on CPU, prove legacy token agreement, then freeze both arms' inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Never overwrite the first expanded protocol")
    legacy = json.loads(args.legacy.read_bytes())
    manifest = json.loads((args.data / "dataset-manifest.json").read_bytes())
    if manifest != legacy["datasets"]:
        raise ValueError("Pinned dataset manifest changed")
    raw = (args.data / "wikitext-test.txt").read_bytes()
    questions = (args.data / "gsm8k-test.jsonl").read_bytes()
    if (hashlib.sha256(raw).hexdigest() != manifest["text_sha256"]
            or hashlib.sha256(questions).hexdigest()
            != manifest["files"]["gsm8k-test.jsonl"]["sha256"]):
        raise ValueError("Pinned test dataset payload changed")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    ids = tokenizer.encode(raw.decode(), add_special_tokens=False)
    suite = expand_suite(legacy, ids, [json.loads(line) for line in questions.splitlines()])
    suite["legacy_protocol_sha256"] = hashlib.sha256(args.legacy.read_bytes()).hexdigest()
    suite["offline_tokenizer_legacy_windows_exact"] = True
    args.output.write_text(json.dumps(suite, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "frozen", "windows": len(suite["wiki_windows"]),
                      "scored_tokens_per_repeat": suite["wiki_scored_tokens"],
                      "heldout_repeats": 3, "tasks": len(suite["tasks"]),
                      "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
