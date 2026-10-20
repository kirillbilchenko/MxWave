"""Freeze fresh train/validation capture under the qualified C1 expert policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from .drafter_fp8_roundtrip import digest
from .prepare_expert_development import _RUN


def frozen_cases(dataset: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Read every frozen training/validation prompt, including long inputs, but no test file."""
    manifest = json.loads((dataset / "manifest.json").read_bytes())
    tokens = json.loads((dataset / "tokenized-v2/review.json").read_bytes())
    rows, hashes = [], {}
    for split, expected_count in (("train", 4000), ("validation", 200)):
        source = dataset / f"{split}.jsonl"
        token_path = dataset / "tokenized-v2" / f"{split}.jsonl"
        for label, path, index in (("prompts", source, manifest), ("tokens", token_path, tokens)):
            expected = next(r["sha256"] for r in index["files"] if r["split"] == split)
            if digest(path) != expected:
                raise ValueError(f"Frozen {split} {label} changed")
            hashes[f"{split}_{label}"] = expected
        prompts = [json.loads(line) for line in source.read_text().splitlines()]
        token_rows = [json.loads(line) for line in token_path.read_text().splitlines()]
        by_id = {r["id"]: r for r in token_rows}
        if (len(prompts) != expected_count or len(token_rows) != expected_count
                or len(by_id) != expected_count
                or len({r["id"] for r in prompts}) != expected_count):
            raise ValueError("Frozen split has missing or duplicate examples")
        for row in prompts:
            token_row = by_id[row["id"]]
            if row["prompt_sha256"] != token_row["prompt_sha256"]:
                raise ValueError("Prompt/token identity differs")
            if not 1 <= len(token_row["prompt_ids"]) <= 4096:
                raise ValueError("Prompt exceeds the single-prefill capture budget")
            rows.append({
                "id": row["id"], "split": split, "domain": row["domain"],
                "messages": row["messages"], "prompt_sha256": row["prompt_sha256"],
                "token_ids_sha256": token_row["token_ids_sha256"],
                "prompt_ids": token_row["prompt_ids"],
            })
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("IDs overlap across splits")
    return rows, hashes


def smoke_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select first examples of each domain plus the longest prompt in each split."""
    selected = []
    for split in ("train", "validation"):
        part = [r for r in rows if r["split"] == split]
        for domain in ("code", "chat", "explanation"):
            selected.append(next(r for r in part if r["domain"] == domain))
        longest = max(part, key=lambda r: len(r["prompt_ids"]))
        if longest not in selected:
            selected.append(longest)
    return selected


def prepare(template: Path, dataset: Path, verified: Path, output: Path) -> dict[str, Any]:
    """Prepare a small fresh-policy capture before committing to full-corpus collection."""
    source_manifest = json.loads((template / "source-manifest.json").read_bytes())
    if not all(digest(template / n) == sha for n, sha in source_manifest.items()):
        raise ValueError("Historical capture sources changed")
    verification = json.loads((verified / "controller.json").read_bytes())
    if verification["phase"] != "complete_restored" or verification["service_paused"]:
        raise ValueError("Qualified C1 run must be complete and restored")
    all_rows, hashes = frozen_cases(dataset)
    selected = smoke_cases(all_rows)
    output.mkdir(parents=True, exist_ok=False)
    for name in source_manifest:
        shutil.copyfile(template / name, output / name)
    source = Path(__file__).parent
    budget_proof = Path("benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/"
                        "prefill-budget-v3-source-freeze.json")
    budget_sources = json.loads(budget_proof.read_bytes())["new_sources"]
    budget_hashes = {}
    for name in ("budget_dispatch.py", "budget_dispatch_warmup.py", "budget_worker.py",
                 "budget_startup_preparation.py", "stable_dispatch_gpu.py"):
        expected = ("5749762758ac8cb821d8e3df6b781fe22c6414153fa1872f1c972f2ecb44c1eb"
                    if name == "stable_dispatch_gpu.py" else budget_sources[name])
        payload = (source / name).read_bytes()
        # The committed startup validator lost one trailing blank line during cleanup.
        if hashlib.sha256(payload).hexdigest() != expected and hashlib.sha256(payload + b"\n").hexdigest() == expected:
            payload += b"\n"
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError("Previously qualified 4096-token dispatch source changed")
        (output / name).write_bytes(payload)
        budget_hashes[name] = expected
    policy_sources = {}
    for module in ("expert_reduction_runtime", "expert_reduction_policy",
                   "expert_reduction_padding_gpu"):
        path = source / f"{module}.py"
        if digest(path) != digest(verified / path.name):
            raise ValueError("Qualified expert policy changed")
        shutil.copyfile(path, output / path.name)
        policy_sources[path.name] = digest(path)
    previous = json.loads((verified / "plan.json").read_bytes())
    plan = json.loads((template / "plan.json").read_bytes())
    controls = [max((r for r in selected if r["split"] == split),
                    key=lambda r: len(r["prompt_ids"]))["id"]
                for split in ("train", "validation")]
    quotas = {split: dict(Counter(r["domain"] for r in selected if r["split"] == split))
              for split in ("train", "validation")}
    plan.update(
        revision="drafter-fresh-capture-smoke-v3", private_port=8028,
        purpose="Fresh-policy capture/export preflight including both longest frozen prompts.",
        calibration_cases=selected, control_cases=controls, quotas=quotas,
        requests=len(selected) + 2 * len(controls), prompt_limit=4096,
        prefill_budget=4096, kv_bytes_per_rank=6 * 1024**3,
        client_deadline_seconds=900, outage_seconds=2700,
        frozen_dataset_hashes=hashes, source_sha256=digest(Path(__file__)),
        expert_reduction_policy=previous["expert_reduction_policy"],
        expert_reduction_source_sha256=policy_sources,
        prefill_dispatch_source_sha256=budget_hashes,
        prefill_dispatch_qualification_sha256=digest(budget_proof),
        qualified_policy_plan_sha256=digest(verified / "plan.json"),
        full_corpus={
            "train": 4000, "validation": 200, "sealed_test_opened": False,
            "accepted_feature_bytes": sum(len(r["prompt_ids"]) + 127 for r in all_rows) * 40960,
            "capture_started": False,
        },
        notes=[
            "First prompt in each domain and longest prompt in each split; no sealed test access.",
            "Same qualified <=16-row expert policy; larger prefills retain native reduction.",
            "4096 prefill budget retains all prompt features in one initial frame, including long inputs.",
            "Original drafter, fixed seven drafts, 128 response tokens; both ranks must agree bitwise.",
            "Longest train/validation prompts receive before/capture/after output and counter controls.",
            "This capture is a pipeline check, not a training or speed result.",
            "Independent watchdog restores original 500k production on completion, failure or deadline.",
        ],
    )
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    for name in ("prepare.py", "control.py", "measure_features.py"):
        path = output / name
        text = path.read_text().replace("8019", "8028").replace("29692", "29701")
        path.write_text(text.replace("mxwave-drafter-feature-capture-v1",
                                     "mxwave-drafter-fresh-capture-smoke-v3"))
    path = output / "trial_boot.py"
    text = path.read_text()
    anchor = "        assert args.max_model_len == 8192 and args.max_num_batched_tokens == 2048\n"
    assert text.count(anchor) == 1
    path.write_text(text.replace(anchor, (
        '        capture_plan = json.loads(pathlib.Path("/trial/plan.json").read_bytes())\n'
        '        assert args.max_model_len == 8192 and args.max_num_batched_tokens == capture_plan["prefill_budget"]\n'
    )))
    path = output / "trial_worker.py"
    text = path.read_text()
    assert "from warmed_model_worker import WarmedModelWorker" in text
    path.write_text(text.replace("from warmed_model_worker import WarmedModelWorker",
                                 "from budget_worker import BudgetWorker")
                   .replace("class TrialWorker(WarmedModelWorker)", "class TrialWorker(BudgetWorker)"))
    path = output / "prepare.py"
    text = path.read_text()
    anchor = '            command[command.index("--port") + 1] = "8028"\n'
    assert text.count(anchor) == 1
    path.write_text(text.replace(anchor, anchor + (
        '            command[command.index("--max-num-batched-tokens") + 1] = str(plan["prefill_budget"])\n'
        '            command[command.index("--kv-cache-memory-bytes") + 1] = str(plan["kv_bytes_per_rank"])\n'
    )))
    path = output / "measure_features.py"
    path.write_text(path.read_text().replace('"capture_cases": 40',
                                             '"capture_cases": len(plan["calibration_cases"])'))
    path = output / "trial_plugin.py"
    text = path.read_text()
    anchor = "def install():\n"
    assert text.count(anchor) == 1
    text = text.replace(anchor, anchor + "    import budget_dispatch\n"
                        "    budget_dispatch.install()\n")
    anchor = "        import drafter_head_probe\n"
    assert text.count(anchor) == 1
    text = text.replace(anchor, "        import expert_reduction_runtime\n"
                        "        expert_reduction_runtime.install()\n" + anchor)
    anchor = '        report["trial"] = worker.mxwave_trial_snapshot()\n'
    assert text.count(anchor) == 1
    path.write_text(text.replace(anchor, anchor + (
        "        import expert_reduction_runtime\n"
        '        report["expert_reduction"] = expert_reduction_runtime.snapshot()\n'
        '        assert report["expert_reduction"]["installed"]\n'
        '        shapes = {tuple(x["geometry"][:3]) for x in report["expert_reduction"]["dispatches"]}\n'
        "        assert {(8, 2048, 4096), (64, 4096, 1024)} <= shapes\n"
    )))
    (output / "run-prepared.py").write_text(_RUN)
    for path in output.glob("*.py"):
        compile(path.read_bytes(), str(path), "exec")
    manifest = {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()}
    (output / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "plan_sha256": digest(output / "plan.json"), "cases": len(selected),
        "requests": plan["requests"], "quotas": quotas,
        "max_prompt_tokens": max(len(r["prompt_ids"]) for r in selected),
        "full_corpus": plan["full_corpus"], "sealed_test_opened": False,
    }


def main() -> None:
    """Prepare locally before contacting the GPU service."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("template", "dataset", "verified", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.template, args.dataset, args.verified, args.output), indent=2))


if __name__ == "__main__":
    main()
