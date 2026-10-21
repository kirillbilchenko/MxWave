"""Freeze a bounded CT feature capture from training/validation splits only."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from .drafter_fp8_roundtrip import digest
from .prepare_selector_calibration import _MEASURE

QUOTAS = {
    "train": {"code": 20, "chat": 6, "explanation": 6},
    "validation": {"code": 4, "chat": 2, "explanation": 2},
}

_TOKENIZER = '''"""Check frozen prompt IDs in the actual serving image without GPU."""
import hashlib
import json
from pathlib import Path

def verify():
    from transformers import AutoTokenizer
    plan = json.loads(Path("/trial/plan.json").read_bytes())
    root = Path("/model")
    for name, sha in plan["tokenizer_hashes"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == sha, name
    tok = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    tok.chat_template = (root / "chat_template.jinja").read_text()
    for row in plan["calibration_cases"]:
        ids = tok.apply_chat_template(row["messages"], tokenize=True,
            add_generation_prompt=True, enable_thinking=False, return_dict=False)
        assert list(ids) == row["prompt_ids"], row["id"]
    print(json.dumps({"native_image_tokenizer_replay_pass": True,
                      "prompts": len(plan["calibration_cases"])}))
'''


def select_cases(
    dataset: Path, quotas: dict[str, dict[str, int]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Select a deterministic bounded subset without opening the sealed test file."""
    manifest = json.loads((dataset / "manifest.json").read_bytes())
    token_manifest = json.loads((dataset / "tokenized-v2/review.json").read_bytes())
    hashes = {}
    selected = []
    quotas = QUOTAS if quotas is None else quotas
    if set(quotas) != {"train", "validation"} or any(
        set(quota) != {"code", "chat", "explanation"}
        or any(type(count) is not int or count <= 0 for count in quota.values())
        for quota in quotas.values()
    ):
        raise ValueError("Only positive train/validation domain quotas are permitted")
    for split, quota in quotas.items():
        source = dataset / f"{split}.jsonl"
        tokens = dataset / "tokenized-v2" / f"{split}.jsonl"
        expected = next(r["sha256"] for r in manifest["files"] if r["split"] == split)
        token_expected = next(r["sha256"] for r in token_manifest["files"] if r["split"] == split)
        if digest(source) != expected or digest(tokens) != token_expected:
            raise ValueError(f"Frozen dataset changed: {split}")
        hashes[f"{split}_prompts"] = expected
        hashes[f"{split}_tokens"] = token_expected
        token_rows = {r["id"]: r for line in tokens.read_text().splitlines()
                      for r in [json.loads(line)]}
        counts: Counter[str] = Counter()
        for line in source.read_text().splitlines():
            row = json.loads(line)
            token_row = token_rows[row["id"]]
            if row["prompt_sha256"] != token_row["prompt_sha256"]:
                raise ValueError("Token/prompt identity mismatch")
            domain = row["domain"]
            if counts[domain] >= quota[domain] or len(token_row["prompt_ids"]) > 1024:
                continue
            selected.append({
                "id": row["id"], "split": split, "domain": domain, "messages": row["messages"],
                "prompt_sha256": row["prompt_sha256"],
                "token_ids_sha256": token_row["token_ids_sha256"],
                "prompt_ids": token_row["prompt_ids"],
            })
            counts[domain] += 1
        if dict(counts) != quota:
            raise ValueError(f"Insufficient bounded examples in {split}")
    return selected, hashes


def prepare(template: Path, dataset: Path, output: Path) -> dict[str, Any]:
    """Write immutable sources using the previously verified restoration controller."""
    selected, hashes = select_cases(dataset)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    for name in (
        "trial_boot.py", "trial_worker.py", "trial_plugin.py", "revision.json",
        "rounding-reference.json", "kda_mxfp8_runtime.py", "kda_mxfp8_roundtrip.py",
        "measure_serving.py", "prepare.py", "control.py", "finalize.py", "launch_followup.py",
    ):
        shutil.copy2(template / name, output / name)
    for name in ("drafter_head_probe.py", "drafter_context_probe.py"):
        shutil.copy2(Path(__file__).parent / name, output / name)
    plan = json.loads((template / "plan.json").read_bytes())
    for key in ("cases", "profile_case_indices"):
        plan.pop(key, None)
    tokenizer = json.loads((dataset / "tokenized-v2/review.json").read_bytes())["tokenizer_hashes"]
    plan.update(
        revision="drafter-feature-capture-v1", arms=["ct"], modes={"ct": "ct_mxfp8"},
        purpose="Bounded full-context training/validation feature capture for fixed CT target.",
        private_port=8019, requests=44, output_tokens=128, max_frames_per_case=160,
        full_logits_frames=0, target_feature_layers=[5, 14, 24, 33, 42],
        client_deadline_seconds=1200, outage_seconds=2100,
        calibration_cases=selected, control_cases=[selected[0]["id"], selected[-1]["id"]],
        frozen_dataset_hashes=hashes, tokenizer_hashes=tokenizer,
        prompt_limit=1024, quotas=QUOTAS, source_sha256=digest(Path(__file__)),
        notes=[
            "32 training / 8 validation prompts; sealed test file is never opened.",
            "First examples in frozen order per domain with <=1024 prompt tokens.",
            "Short bounded capture only; long contexts remain a separate qualification gate.",
            "All five target streams, token positions and rejection decisions on both ranks.",
            "Accepted input prefixes only become training features; rejected/post-cap rows excluded.",
            "No trained weights are loaded; target and original drafter remain fixed.",
            "Before/capture/after token and speculative-counter controls for two prompts.",
            "Native FULL draft graph replay required; no instrumented speed claim.",
            "Original 500k-context production restored on completion, timeout or failure.",
        ],
    )
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    for name in ("prepare.py", "control.py"):
        p = output / name
        p.write_text(p.read_text().replace("8017", "8019").replace("29690", "29692")
                     .replace("mxwave-drafter-context-v1", "mxwave-drafter-feature-capture-v1")
                     .replace("measure_context.py", "measure_features.py"))
    p = output / "control.py"
    p.write_text(p.read_text().replace("> 2 * 1024**3", "> 20 * 1024**3"))
    p = output / "launch_followup.py"
    p.write_text(p.read_text().replace('"drafter-head-v3"', '"selector-calibration-capture-v1"'))
    (output / "measure_features.py").write_text(
        _MEASURE.replace("8018", "8019").replace('"capture_cases": 48', '"capture_cases": 40'),
    )
    (output / "verify_capture_tokenizer.py").write_text(_TOKENIZER)
    p = output / "trial_boot.py"
    text = p.read_text()
    anchor = '        print(\n            json.dumps(\n                {\n                    "status": "validated",'
    if text.count(anchor) != 1:
        raise ValueError("Unexpected CPU validation template")
    text = text.replace(anchor, "        import verify_capture_tokenizer\n"
                        "        verify_capture_tokenizer.verify()\n" + anchor)
    p.write_text(text)
    manifest = {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()}
    (output / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for p in output.glob("*.py"):
        compile(p.read_bytes(), str(p), "exec")
    return {
        "source_files": len(manifest), "plan_sha256": digest(output / "plan.json"),
        "cases": len(selected), "requests": 44, "quotas": QUOTAS,
        "input_tokens": sum(len(r["prompt_ids"]) for r in selected),
        "max_input_tokens": max(len(r["prompt_ids"]) for r in selected),
        "feature_bytes_per_rank_before_auxiliary":
            sum(len(r["prompt_ids"]) + 128 for r in selected) * 40960,
        "frozen_dataset_hashes": hashes, "sealed_test_opened": False,
    }


def main() -> None:
    """Prepare the bounded capture; do not contact or alter the live service."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("template", "dataset", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.template, args.dataset, args.output), indent=2))


if __name__ == "__main__":
    main()
