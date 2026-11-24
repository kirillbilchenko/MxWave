"""Train one selector projection on frozen states; offline metrics are not serving gains."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from .selector_capture_analysis import analyze


def read_frame(path: Path) -> dict[str, torch.Tensor]:
    """Verify retained bytes and restore original BF16 tensor representations."""
    metadata = json.loads(path.with_suffix(".json").read_bytes())
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["payload_sha256"]:
        raise ValueError(f"Frame digest mismatch: {path.name}")
    return {
        name: value.view(torch.bfloat16)
        if metadata["tensor_dtypes"][name] == "torch.bfloat16" else value
        for name, value in load_file(str(path)).items()
    }


def teacher_scores(
    hidden: torch.Tensor, projection: torch.Tensor, predecessor: torch.Tensor,
    successor: torch.Tensor, candidate_ids: torch.Tensor, unary: torch.Tensor,
    gold: torch.Tensor, anchors: torch.Tensor,
) -> torch.Tensor:
    """Score each candidate using the recorded target's preceding token."""
    previous = torch.cat((anchors[:, None], gold[:, :-1]), dim=1)
    conditioning = F.linear(hidden.float(), projection) * predecessor[previous]
    return unary.float() + (conditioning[:, :, None, :] * successor[candidate_ids]).sum(-1)


def covered_loss(scores: torch.Tensor, candidate_ids: torch.Tensor, gold: torch.Tensor) -> Any:
    """Optimize only strict top-k-covered labels; never insert the gold candidate."""
    matches = candidate_ids == gold[..., None]
    covered = matches.any(-1)
    if not covered.any():
        raise ValueError("Batch has no covered target labels")
    return F.cross_entropy(scores[covered], matches.int().argmax(-1)[covered])


@torch.no_grad()
def greedy_tokens(
    hidden: torch.Tensor, projection: torch.Tensor, predecessor: torch.Tensor,
    successor: torch.Tensor, candidate_ids: torch.Tensor, unary: torch.Tensor,
    anchors: torch.Tensor,
) -> torch.Tensor:
    """Follow the selector's own choices, including mistakes, on fixed draft states."""
    features = F.linear(hidden.float(), projection)
    previous = anchors
    outputs = []
    for step in range(candidate_ids.shape[1]):
        ids = candidate_ids[:, step]
        context = predecessor[previous] * features[:, step]
        score = unary[:, step].float() + (context[:, None, :] * successor[ids]).sum(-1)
        choice = score.argmax(-1)
        previous = ids.gather(1, choice[:, None]).squeeze(1)
        outputs.append(previous)
    return torch.stack(outputs, dim=1)


def prefix_lengths(predicted: torch.Tensor, gold: torch.Tensor) -> torch.Tensor:
    """Count only consecutive matching draft tokens before the first error."""
    return (predicted == gold).long().cumprod(-1).sum(-1)


def build_dataset(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join graph positions to actual continuations, excluding incomplete tail blocks."""
    plan = json.loads((root / "plan.json").read_bytes())
    measurement = json.loads((root / "ct-measurement.json").read_bytes())
    if measurement["status"] != "complete":
        raise ValueError("Capture did not complete")
    captures = {x["case"]: x for x in measurement["cases"] if x["policy"] == "capture"}
    blocks = []
    excluded = 0
    for case in plan["calibration_cases"]:
        row = captures[case["id"]]
        prompt_length = len(case["prompt_ids"])
        tokens = row["result"]["token_ids"]
        paths = sorted((root / "ct-r0").glob(f"calibration-{case['id']}-*.safetensors"))
        if not paths:
            raise ValueError(f"No native graph frames for {case['id']}")
        for path in paths:
            frame = read_frame(path)
            ids = frame["ids"].reshape(7, 16).long()
            native_scores = frame["scores"].reshape(7, 16, 16).float()
            predecessor_index = 0
            expected = []
            for step in range(7):
                predecessor_index = int(native_scores[step, predecessor_index].argmax())
                expected.append(int(ids[step, predecessor_index]))
            if expected != frame["proposal"].reshape(-1).tolist():
                raise ValueError(f"Native selector walk mismatch: {path.name}")
            positions = frame["sample_positions"].reshape(-1).long()
            first = int(positions[0])
            if positions.tolist() != list(range(first, first + 7)):
                raise ValueError(f"Noncontiguous speculative positions: {path.name}")
            start = first - prompt_length
            if start < 1:
                raise ValueError(f"Invalid anchor: {path.name}")
            if start + 7 > len(tokens):
                excluded += 1
                continue
            if int(frame["anchors"].item()) != tokens[start - 1]:
                raise ValueError(f"Anchor/history mismatch: {path.name}")
            blocks.append({
                "id": case["id"], "split": case["split"], "domain": case["domain"],
                "hidden": frame["hidden"].reshape(7, -1).float(),
                "ids": ids, "unary": frame["values"].reshape(7, 16).float(),
                "anchor": frame["anchors"].reshape(()).long(),
                "gold": torch.tensor(tokens[start:start + 7]),
                "native_proposal": frame["proposal"].reshape(7).long(),
            })
    return blocks, {"complete_blocks": len(blocks), "incomplete_tail_frames": excluded}


def batch(rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Stack retained fields without reading any target weights."""
    return {
        key: torch.stack([row[key] for row in rows])
        for key in ("hidden", "ids", "unary", "anchor", "gold", "native_proposal")
    }


@torch.no_grad()
def evaluate(
    rows: list[dict[str, Any]], projection: torch.Tensor, weights: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Report per-prompt fixed-state metrics so long responses do not dominate selection."""
    prompts: dict[str, dict[str, Any]] = {}
    for offset in range(0, len(rows), 32):
        chunk = rows[offset:offset + 32]
        data = batch(chunk)
        predicted = greedy_tokens(
            data["hidden"], projection, weights["predecessor"], weights["successor"],
            data["ids"], data["unary"], data["anchor"],
        )
        lengths = prefix_lengths(predicted, data["gold"]).tolist()
        native_lengths = prefix_lengths(data["native_proposal"], data["gold"]).tolist()
        covered = (data["ids"] == data["gold"][..., None]).any(-1)
        oracle_lengths = covered.long().cumprod(-1).sum(-1).tolist()
        for index, row in enumerate(chunk):
            info = prompts.setdefault(row["id"], {
                "domain": row["domain"], "prefix_sum": 0, "native_prefix_sum": 0,
                "blocks": 0, "proposal_tokens_different_from_native": 0,
                "topk_oracle_prefix_sum": 0, "native_first_ranking_errors": 0,
                "native_first_coverage_errors": 0,
            })
            info["prefix_sum"] += lengths[index]
            info["native_prefix_sum"] += native_lengths[index]
            info["topk_oracle_prefix_sum"] += oracle_lengths[index]
            first_error = native_lengths[index]
            if first_error < data["gold"].shape[1]:
                kind = "ranking" if covered[index, first_error] else "coverage"
                info[f"native_first_{kind}_errors"] += 1
            info["blocks"] += 1
            info["proposal_tokens_different_from_native"] += int(
                (predicted[index] != data["native_proposal"][index]).sum()
            )
    if not prompts:
        raise ValueError("Empty evaluation split")
    for info in prompts.values():
        info["mean_prefix"] = info["prefix_sum"] / info["blocks"]
        info["native_mean_prefix"] = info["native_prefix_sum"] / info["blocks"]
        info["topk_oracle_mean_prefix"] = info["topk_oracle_prefix_sum"] / info["blocks"]
    result: dict[str, Any] = {"prompts": prompts, "scope": "fixed-state FP32 surrogate"}
    for domain in ("all", "code", "explanation"):
        selected = [p for p in prompts.values() if domain == "all" or p["domain"] == domain]
        if selected:
            result[domain] = sum(p["mean_prefix"] for p in selected) / len(selected)
    return result


def train(root: Path, output: Path, steps: int = 100) -> dict[str, Any]:
    """Fit only the existing projection, select on validation, then open the test split once."""
    if output.exists():
        raise FileExistsError(output)
    capture_review = analyze(root)
    output.mkdir(parents=True)
    (output / "capture-review.json").write_text(json.dumps(capture_review, indent=2) + "\n")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    weights = {
        name: value.float()
        for name, value in load_file(str(root / "ct-r0/selector-original.safetensors")).items()
    }
    blocks, coverage = build_dataset(root)
    splits = {name: [x for x in blocks if x["split"] == name]
              for name in ("train", "validation", "test")}
    original = weights["projection"].clone()
    projection = torch.nn.Parameter(original.clone())
    optimizer = torch.optim.AdamW([projection], lr=1e-5, weight_decay=0)
    baseline = evaluate(splits["validation"], original, weights)
    selected = original.clone()
    selected_step = 0
    best = baseline["all"]
    history = []
    rng = torch.Generator().manual_seed(42)
    for step in range(1, steps + 1):
        indices = torch.randint(len(splits["train"]), (16,), generator=rng).tolist()
        data = batch([splits["train"][index] for index in indices])
        scores = teacher_scores(
            data["hidden"], projection, weights["predecessor"], weights["successor"],
            data["ids"], data["unary"], data["gold"], data["anchor"],
        )
        loss = covered_loss(scores, data["ids"], data["gold"])
        if not torch.isfinite(loss):
            raise ValueError(f"Nonfinite training loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([projection], 1.0)
        optimizer.step()
        with torch.no_grad():
            delta = projection - original
            limit = original.norm() * 0.01
            if delta.norm() > limit:
                projection.copy_(original + delta * (limit / delta.norm()))
        if step % 25 == 0 or step == steps:
            # Evaluate the actual BF16-exported parameter values, with FP32 arithmetic.
            candidate = projection.detach().to(torch.bfloat16).float()
            review = evaluate(splits["validation"], candidate, weights)
            history.append({"step": step, "training_loss": float(loss.detach()),
                            "validation": review})
            if (review["all"] > best and review["code"] >= baseline["code"]
                    and review["explanation"] >= baseline["explanation"]):
                selected, selected_step, best = candidate.clone(), step, review["all"]
            print(json.dumps({"step": step, "validation_mean_prefix": review["all"],
                              "selected_step": selected_step}), flush=True)
    test_before = evaluate(splits["test"], original, weights)
    test_after = evaluate(splits["test"], selected, weights)
    passed = (
        selected_step > 0 and test_after["all"] > test_before["all"]
        and test_after["code"] >= test_before["code"]
        and test_after["explanation"] >= test_before["explanation"]
    )
    patch = output / "selector-projection.safetensors"
    save_file(
        {"model.candidate_selector.hidden_projection.weight": selected.to(torch.bfloat16)},
        str(patch),
    )
    report = {
        "schema": "mxwave-selector-calibration-pilot-v1", "seed": 42,
        "capture_plan_sha256": hashlib.sha256((root / "plan.json").read_bytes()).hexdigest(),
        "capture_manifest_sha256": hashlib.sha256(
            (root / "evidence-manifest.json").read_bytes()
        ).hexdigest(),
        "trainer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "capture_analyzer_sha256": hashlib.sha256(
            Path(__file__).with_name("selector_capture_analysis.py").read_bytes()
        ).hexdigest(),
        "capture_review_sha256": hashlib.sha256(
            (output / "capture-review.json").read_bytes()
        ).hexdigest(),
        "trainable_parameters": original.numel(), "steps": steps,
        "torch_version": torch.__version__, "device": "cpu", "cpu_threads": 4,
        "learning_rate": 1e-5, "relative_update_norm_limit": 0.01,
        "relative_selected_update_norm": float((selected - original).norm() / original.norm()),
        "changed_bf16_elements": int(
            (selected.to(torch.bfloat16) != original.to(torch.bfloat16)).sum()
        ),
        "source_projection_sha256": hashlib.sha256(
            original.to(torch.bfloat16).view(torch.uint8).numpy().tobytes()
        ).hexdigest(),
        "coverage": coverage, "validation_baseline": baseline, "history": history,
        "selected_step": selected_step, "test_before": test_before, "test_after": test_after,
        "offline_screen_passed": passed,
        "serving_qualified": False, "target_weights_changed": False,
        "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
        "limitations": [
            "Small authored pilot corpus, not representative general-quality evidence.",
            "FP32 surrogate arithmetic differs from native BF16 kernels.",
            "Fixed states are conditioned on baseline histories, not candidate rollouts.",
            "A pass only permits native-kernel and live held-out validation; no speed claim.",
            "No checkpoint is promoted or production configuration modified.",
        ],
    }
    (output / "review.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    """Run the bounded, CPU-only calibration experiment."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.steps <= 100:
        parser.error("Pilot must stay within 1–100 optimizer steps")
    report = train(args.capture, args.output, args.steps)
    print(json.dumps({
        key: report[key] for key in
        ("selected_step", "offline_screen_passed", "serving_qualified")
    }))


if __name__ == "__main__":
    main()
