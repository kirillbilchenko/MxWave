"""Verify frozen splits and tokenize them on CPU without displaying prompt content."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .drafter_fp8_roundtrip import digest
from .prepare_drafter_dataset import normalized

TOKENIZER_HASHES = {
    "tokenizer.json": "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d",
    "tokenizer_config.json": "98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc",
    "chat_template.jinja": "34d5ee66b12fa6446cdae131c352b8f68cd85369e0e6fda115583805fada3891",
    "config.json": "194d290ebb0f19aedc60d24ea9e236e33c86616e460f4a108a842f6fd5b8b198",
}


def token_digest(ids: list[int]) -> str:
    """Identify an exact sequence of token IDs independent of whitespace serialization."""
    if not ids or any(type(value) is not int or value < 0 for value in ids):
        raise ValueError("Expected nonempty integer token IDs, not tokenizer dictionary keys")
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def verify(dataset: Path, tokenizer: Path, evidence: Path, output: Path) -> dict[str, Any]:
    """Check disjointness, tokenizer identity, lengths and prior speed-prompt exclusion."""
    from transformers import AutoTokenizer

    for name, expected in TOKENIZER_HASHES.items():
        if digest(tokenizer / name) != expected:
            raise ValueError(f"Tokenizer hash mismatch: {name}")
    tok = AutoTokenizer.from_pretrained(tokenizer, local_files_only=True, trust_remote_code=False)
    tok.chat_template = (tokenizer / "chat_template.jinja").read_text()
    manifest = json.loads((dataset / "manifest.json").read_text())
    prior: set[str] = set()
    for name in ("mxfp4-serving-v3-plan.json", "kda-mxfp8-speed-v1-plan.json"):
        for case in json.loads((evidence / name).read_text())["cases"]:
            prior.update(token_digest(ids) for ids in case["prompts"])
    sets: dict[str, set[str]] = defaultdict(set)
    groups: dict[str, set[str]] = defaultdict(set)
    token_sets: dict[str, set[str]] = defaultdict(set)
    output.mkdir(exist_ok=False)
    files = []
    train_token_count = 0
    for record in manifest["files"]:
        path = Path(record["path"])
        if digest(path) != record["sha256"]:
            raise ValueError(f"Prompt file hash mismatch: {path.name}")
        split = record["split"]
        lengths = []
        dst = output / f"{split}.jsonl"
        with path.open() as src, dst.open("x") as stream:
            for line in src:
                row = json.loads(line)
                text = row["messages"][0]["content"]
                expected = hashlib.sha256(normalized(text).encode()).hexdigest()
                if expected != row["prompt_sha256"]:
                    raise ValueError("Prompt normalization digest mismatch")
                ids = list(tok.apply_chat_template(
                    row["messages"], tokenize=True, add_generation_prompt=True,
                    enable_thinking=False, return_dict=False,
                ))
                if len(ids) + 512 > 8192:
                    raise ValueError("Prompt exceeds the frozen 8k capture budget")
                token_hash = token_digest(ids)
                if token_hash in prior:
                    raise ValueError("Old speed prompt appears in the new corpus")
                sets[split].add(expected)
                groups[split].add(row["group_sha256"])
                token_sets[split].add(token_hash)
                lengths.append(len(ids))
                stream.write(json.dumps({
                    "id": row["id"], "prompt_sha256": expected,
                    "prompt_ids": ids, "token_ids_sha256": token_hash,
                }, separators=(",", ":")) + "\n")
        if len(lengths) != record["count"] or len(sets[split]) != len(lengths):
            raise ValueError("Prompt count/uniqueness mismatch")
        if split == "train":
            train_token_count = sum(lengths)
        files.append({
            "split": split, "path": str(dst), "sha256": digest(dst), "count": len(lengths),
            "input_tokens": sum(lengths), "min_input_tokens": min(lengths),
            "max_input_tokens": max(lengths),
        })
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if sets[a] & sets[b] or groups[a] & groups[b] or token_sets[a] & token_sets[b]:
            raise ValueError("Cross-split overlap")
    report = {
        "schema": "mxwave-drafter-tokenized-splits-v1", "tokenizer_hashes": TOKENIZER_HASHES,
        "manifest_sha256": digest(dataset / "manifest.json"), "source_sha256": digest(Path(__file__)),
        "files": files, "enable_thinking": False, "add_generation_prompt": True,
        "legacy_speed_token_sequences_checked": len(prior), "legacy_speed_token_overlap": 0,
        "cross_split_prompt_group_token_overlap": 0, "prompt_content_displayed": False,
        "features_per_token_bytes": 5 * 4096 * 2,
        "train_feature_bytes_if_all_input_plus_512_output_tokens":
            (train_token_count + 4000 * 512) * 5 * 4096 * 2,
        "limits": [
            "Tokenization on Mac; replay equality with the installed serving tokenizer still to verify.",
            "No CT continuations, target features, model outputs or training labels generated.",
            "Feature storage estimate excludes indexes, logits, embeddings, tensor headers and validation data.",
        ],
    }
    (output / "review.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    """Print only hashes, counts and length statistics from the frozen corpus."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.dataset, args.tokenizer, args.evidence, args.output), indent=2))


if __name__ == "__main__":
    main()
