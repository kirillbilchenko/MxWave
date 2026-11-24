"""Freeze public prompt-only adaptation splits without displaying held-out content."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from .drafter_fp8_roundtrip import digest
from .selector_calibration_protocol import cases

SOURCE_HASHES = {
    "opencoder-train-00000-of-00001.parquet":
        "e59784400ca4ba258a84c2bc04d88048cc7b6a26a6cd97692f287ba03a1a1fa7",
    "dolly-databricks-dolly-15k.jsonl":
        "2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec",
}
QUOTAS = {
    "train": {"code": 2400, "chat": 800, "explanation": 800},
    "validation": {"code": 120, "chat": 40, "explanation": 40},
    "test": {"code": 120, "chat": 40, "explanation": 40},
}


def normalized(text: str) -> str:
    """Canonicalize spacing/case for duplicate and prior-fixture exclusion."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def group_key(text: str) -> str:
    """Group exact instructions and simple numeric/literal variants before splitting."""
    value = normalized(text)
    value = re.sub(r"(['\"])[^'\"\n]{0,120}\1", "<literal>", value)
    value = re.sub(r"\b\d+(?:\.\d+)?\b", "<number>", value)
    return hashlib.sha256(value.encode()).hexdigest()


def allocate(
    candidates: list[dict[str, Any]], quotas: dict[str, dict[str, int]],
) -> dict[str, list[dict[str, Any]]]:
    """Select unique instruction groups by a frozen hash order, reserving test first."""
    ordered = sorted(candidates, key=lambda row: hashlib.sha256(
        ("mxwave-ct-drafter-v2-seed42:" + row["prompt_sha256"]).encode()
    ).digest())
    splits: dict[str, list[dict[str, Any]]] = {name: [] for name in quotas}
    selected_groups: set[str] = set()
    used_prompts: set[str] = set()
    counts: dict[str, Counter[str]] = {name: Counter() for name in quotas}
    for row in ordered:
        group = row["group_sha256"]
        if group in selected_groups or row["prompt_sha256"] in used_prompts:
            continue
        for name in ("test", "validation", "train"):
            domain = row["domain"]
            if counts[name][domain] >= quotas[name][domain]:
                continue
            splits[name].append(row)
            counts[name][domain] += 1
            selected_groups.add(group)
            used_prompts.add(row["prompt_sha256"])
            break
    for name, expected in quotas.items():
        if dict(counts[name]) != expected:
            raise ValueError(f"Insufficient distinct prompts for {name}: {dict(counts[name])}")
    return splits


def prepare(sources: Path, output: Path, protocol: Path) -> dict[str, Any]:
    """Select prompt-only public examples; source answers never become training labels."""
    import pyarrow.parquet as pq

    for name, expected in SOURCE_HASHES.items():
        if digest(sources / name) != expected:
            raise ValueError(f"Source hash mismatch: {name}")
    output.mkdir(parents=True, exist_ok=False)
    prior = {normalized(row["text"]) for row in cases()}
    candidates: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()

    def add(source: str, index: int, instruction: str, context: str, domain: str,
            category: str, license_name: str) -> None:
        # Exclude the web dataset previews and short/oversized input proxies.
        if index < 512:
            excluded["preview_prefix"] += 1
            return
        text = instruction.strip() + ("\n\n" + context.strip() if context.strip() else "")
        if not 20 <= len(text) <= 12000:
            excluded["character_bounds"] += 1
            return
        canon = normalized(text)
        if canon in prior:
            excluded["prior_pilot"] += 1
            return
        if re.search(r"\b(gsm8k|humaneval|mbpp|apps benchmark|livecodebench)\b", canon):
            excluded["named_benchmark"] += 1
            return
        prompt_hash = hashlib.sha256(canon.encode()).hexdigest()
        candidates.append({
            "id": f"{source}-{index}", "source": source, "source_row": index,
            "domain": domain, "source_category": category, "license": license_name,
            "messages": [{"role": "user", "content": text}],
            "prompt_sha256": prompt_hash, "group_sha256": group_key(instruction),
        })

    parquet = pq.ParquetFile(sources / "opencoder-train-00000-of-00001.parquet")
    index = 0
    for batch in parquet.iter_batches(columns=["instruction", "tag"], batch_size=2048):
        for row in batch.to_pylist():
            add("opencoder-evol", index, row["instruction"], "", "code", row["tag"], "MIT")
            index += 1
    with (sources / "dolly-databricks-dolly-15k.jsonl").open() as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            category = row["category"]
            if category in {"creative_writing", "brainstorming", "summarization"}:
                domain = "chat"
            elif category in {"open_qa", "general_qa"}:
                domain = "explanation"
            else:
                excluded["unused_dolly_category"] += 1
                continue
            add("dolly", index, row["instruction"], row["context"], domain, category, "CC-BY-SA-3.0")
    splits = allocate(candidates, QUOTAS)
    files = []
    for split, rows in splits.items():
        path = output / f"{split}.jsonl"
        with path.open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        files.append({
            "split": split, "path": str(path), "sha256": digest(path),
            "bytes": path.stat().st_size, "count": len(rows),
            "domains": dict(Counter(row["domain"] for row in rows)),
            "source_categories": dict(Counter(str(row["source_category"]) for row in rows)),
            "character_lengths": {
                "min": min(len(row["messages"][0]["content"]) for row in rows),
                "max": max(len(row["messages"][0]["content"]) for row in rows),
            },
        })
    manifest = {
        "schema": "mxwave-ct-drafter-prompts-v2", "source_hashes": SOURCE_HASHES,
        "source_downloads": json.loads((sources / "dataset-downloads.json").read_text()),
        "protocol_sha256": digest(protocol), "builder_sha256": digest(Path(__file__)),
        "eligible_candidates": len(candidates), "excluded": dict(excluded),
        "files": files, "model_inference_performed": False, "training_performed": False,
        "test_content_displayed": False, "source_answers_included": False,
        "split_policy": "Seeded hash order, unique normalized instruction groups; test then validation then train.",
        "limitations": [
            "Public workload proxies, not private harness traffic; predominantly single-turn.",
            "Dolly explanation bucket is general QA, not a verified technical-only benchmark.",
            "Exact/simple-template grouping is not a proof against semantic paraphrase overlap.",
            "Named benchmark and prior-pilot filters do not establish unknown pretraining-data independence.",
            "Source prompt bytes frozen; tokenizer IDs and actual feature capture remain to be prepared.",
            "Sealed means withheld from tuning; files are not encrypted and the builder processes them.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    """Build frozen splits and print only non-content manifest information."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.sources, args.output, args.protocol), indent=2))


if __name__ == "__main__":
    main()
