"""Run a bounded context-only DFlash2 pilot on verified CT feature sequences."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from .audit_drafter_specforge import load_model
from .cpu_drafter_linear import install as install_cpu_derivatives
from .cpu_drafter_linear import linear as cpu_linear
from .drafter_feature_store import VerifiedFeatureStore
from .drafter_forward_preflight import tensor_digest
from .drafter_fp8_roundtrip import digest
from .drafter_training_schedule import TrainingSchedule


def anchors_for(
    ids: list[int], prompt_length: int, eos: list[int], *, seed_key: str, validation: bool,
) -> list[int]:
    """Choose up to four complete blocks, including EOS but excluding any post-EOS text."""
    stop = next((i + 1 for i in range(prompt_length, len(ids)) if ids[i] in eos), len(ids))
    eligible = list(range(prompt_length, stop - 7))
    count = min(4, len(eligible))
    if not count:
        return []
    if validation:
        if count == 1:
            return eligible[:1]
        return [eligible[round(i * (len(eligible) - 1) / (count - 1))] for i in range(count)]
    seed = int.from_bytes(hashlib.sha256(seed_key.encode()).digest()[:8], "little")
    return sorted(random.Random(seed).sample(eligible, count))


def grouped_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Give each prompt equal weight within its domain."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["domain"]].append(row)
    return {
        domain: {
            "prompts": len(group),
            **{key: sum(r[key] for r in group) / len(group)
               for key in ("mean_prefix", "unary_ce", "coverage")},
        }
        for domain, group in sorted(groups.items())
    }


def select_checkpoint(
    baseline: dict[str, Any], evaluations: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Apply the frozen offline development screen without changing serving thresholds."""
    if set(baseline) != {"code", "chat", "explanation"}:
        return None
    allowed = []
    for row in evaluations:
        domains = row["domains"]
        if set(domains) != set(baseline) or row["relative_l2"] > 0.01:
            continue
        if domains["code"]["mean_prefix"] <= baseline["code"]["mean_prefix"]:
            continue
        if any(domains[d]["mean_prefix"] < 0.98 * baseline[d]["mean_prefix"]
               for d in ("chat", "explanation")):
            continue
        allowed.append(row)
    return min(allowed, key=lambda r: (
        -r["domains"]["code"]["mean_prefix"], r["domains"]["code"]["unary_ce"], r["step"],
    )) if allowed else None


class Pilot:
    """Keep target/backbone fixed and differentiate only an FP32 context-weight master."""

    def __init__(
        self, dataset: Path, checkpoint: Path, checkout: Path, weights: Path,
        protocol: dict[str, Any],
    ) -> None:
        self.protocol = protocol
        if protocol.get("execution_policy") != "single_block_forwards":
            raise ValueError("Expected the frozen single-block execution protocol")
        manifest = json.loads((dataset / "manifest.json").read_bytes())
        if (manifest["status"] != "verified" or not manifest["all_rank_features_bitwise_equal"]
                or manifest["plan_sha256"] != protocol["capture_plan_sha256"]):
            raise ValueError("Verified capture does not match the frozen protocol")
        if digest(checkpoint / "model.safetensors") != protocol["dense_checkpoint_sha256"]:
            raise ValueError("Wrong dense drafter checkpoint")
        weight_record = json.loads((weights / "training-rows.json").read_bytes())
        for name in ("training-embeddings.bf16", "target-head.bf16"):
            if digest(weights / name) != weight_record[name]["sha256"]:
                raise ValueError(f"Target weight integrity failure: {name}")
        self.embedding_ids = weight_record["training-embeddings.bf16"]["rows"]
        self.id_map = {token: index for index, token in enumerate(self.embedding_ids)}
        self.embeddings = torch.from_file(
            str(weights / "training-embeddings.bf16"), shared=False,
            size=len(self.embedding_ids) * 4096, dtype=torch.bfloat16,
        ).reshape(-1, 4096)
        self.head = torch.from_file(
            str(weights / "target-head.bf16"), shared=False,
            size=154880 * 4096, dtype=torch.bfloat16,
        ).reshape(154880, 4096)
        self.model, revision = load_model(checkout, checkpoint)
        if revision != protocol["specforge_revision"]:
            raise ValueError("Wrong upstream implementation")
        self.model.eval().requires_grad_(False)
        self.fp32_derivatives = protocol.get("linear_derivatives") == "fp32_accumulate_cast_back"
        if self.fp32_derivatives:
            derivative_source = Path(__file__).with_name("cpu_drafter_linear.py")
            if digest(derivative_source) != protocol["derivative_source_sha256"]:
                raise ValueError("Derivative implementation differs from the frozen protocol")
            self.overridden_linears = install_cpu_derivatives(self.model)
        self.mask_builder = importlib.import_module(
            "specforge.modeling.draft.dflash_mask",
        ).create_dflash_sdpa_mask
        self.samples: list[dict[str, Any]] = []
        self.feature_store = (
            VerifiedFeatureStore(dataset) if protocol.get("bounded_feature_loading", False) else None
        )
        self.skipped = []
        for row in manifest["rows"]:
            if row["split"] not in ("train", "validation"):
                raise ValueError("Test data is forbidden in this pilot")
            path = dataset / row["path"]
            if self.feature_store is None:
                if digest(path) != row["sha256"]:
                    raise ValueError(f"Dataset file changed: {row['id']}")
                tensors = load_file(str(path))
                ids = tensors["input_ids"].tolist()
            else:
                ids = self.feature_store.input_ids(row["path"], row["sha256"]).tolist()
            anchors = anchors_for(
                ids, row["prompt_tokens"], protocol["data"]["eos_token_ids"],
                seed_key=f"42:{row['split']}:{row['id']}",
                validation=row["split"] == "validation",
            )
            if not anchors:
                self.skipped.append({"id": row["id"], "split": row["split"],
                                     "domain": row["domain"], "reason": "short_natural_response"})
                continue
            sample = {**row, "anchors": anchors}
            if self.feature_store is None:
                sample["tensors"] = tensors
            self.samples.append(sample)
        self.master = self.model.fc.weight.detach().float().clone().requires_grad_(True)
        self.original_hashes = {n: tensor_digest(p) for n, p in self.model.named_parameters()}
        self.inputs = {
            "feature_manifest_sha256": digest(dataset / "manifest.json"),
            "target_weight_record": weight_record, "specforge_revision": revision,
            "selected_anchors": [{"id": s["id"], "split": s["split"], "domain": s["domain"],
                                  "anchors": s["anchors"]} for s in self.samples],
            "skipped": self.skipped,
        }

    def features(self, sample: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Load one verified sample when the new protocol explicitly enables bounded residency."""
        if self.feature_store is None:
            return sample["tensors"]
        return self.feature_store.get(sample["path"], sample["sha256"])

    def forward(self, sample: dict[str, Any], anchors: list[int] | None = None) -> torch.Tensor:
        """Execute one proposal block per forward to retain the verified C1 shape policy."""
        anchors = sample["anchors"] if anchors is None else anchors
        return torch.cat([self.forward_blocks(sample, [anchor]) for anchor in anchors])

    def forward_blocks(
        self, sample: dict[str, Any], anchors: list[int], *, automatic_mask: bool = False,
    ) -> torch.Tensor:
        """Build checked masks; multi-block execution is retained only for diagnostics."""
        anchors = sample["anchors"] if anchors is None else anchors
        data = self.features(sample)
        end = max(anchors)
        context = data["target_features"][:end].unsqueeze(0)
        anchor_tensor = torch.tensor([anchors], dtype=torch.long)
        query_positions = anchor_tensor[:, :, None] + torch.arange(8)[None, None, :]
        mask = self.mask_builder(
            anchor_positions=anchor_tensor,
            block_keep_mask=torch.ones_like(anchor_tensor, dtype=torch.bool),
            S=end, block_size=8, device=torch.device("cpu"), sliding_window=2048,
            is_causal=False,
        )
        # Check every mask element against the independent interval/block contract.
        for i, anchor in enumerate(anchors):
            for offset in range(8):
                permitted = set(range(max(0, anchor + offset - 2047), anchor))
                permitted.update(range(end + i * 8, end + (i + 1) * 8))
                actual = set(mask[0, 0, i * 8 + offset].nonzero().flatten().tolist())
                if actual != permitted:
                    raise ValueError("Future-context or cross-block attention leak")
        query_ids = torch.full((len(anchors), 8), 154856, dtype=torch.long)
        query_ids[:, 0] = data["input_ids"][torch.tensor(anchors)]
        lookup = torch.tensor([self.id_map[t] for t in query_ids.flatten().tolist()])
        kwargs = {
            "position_ids": torch.cat([torch.arange(end).unsqueeze(0),
                                       query_positions.flatten(1)], dim=1),
            "noise_embedding": self.embeddings[lookup].reshape(1, -1, 4096),
            "target_hidden": context, "attention_mask": {"sliding_attention": mask},
            "use_cache": False,
        }
        if automatic_mask:
            if len(anchors) != 1:
                raise ValueError("Automatic inference masks require exactly one block")
            kwargs.pop("attention_mask")
        return torch.func.functional_call(
            self.model, {"fc.weight": self.master.to(torch.bfloat16)}, (), kwargs,
        ).reshape(len(anchors), 8, 4096)[:, 1:]

    def objective(self, sample: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute full-vocabulary and strict-covered-selector CE against verified CT tokens."""
        hidden = self.forward(sample)
        anchors = torch.tensor(sample["anchors"])
        ids = self.features(sample)["input_ids"]
        positions = anchors[:, None] + torch.arange(1, 8)[None, :]
        labels = ids[positions]
        predecessors = ids[positions - 1]
        project = cpu_linear if self.fp32_derivatives else F.linear
        logits = project(hidden, self.head).float()
        unary_ce = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
        top_values, top_ids = logits.topk(16, dim=-1)
        matches = top_ids == labels[..., None]
        covered = matches.any(dim=-1)
        selector_logits = self.model.candidate_selector.score_candidates(
            candidate_ids=top_ids, unary_logits=top_values, hidden_states=hidden,
            predecessor_ids=predecessors,
        )
        ce = F.cross_entropy(
            selector_logits.flatten(0, 1), matches.long().argmax(-1).flatten(),
            reduction="none",
        ).reshape_as(covered)
        selector_ce = (ce * covered).sum() / covered.sum().clamp_min(1)
        with torch.no_grad():
            proposal = self.model.candidate_selector.greedy_path(
                candidate_ids=top_ids, unary_logits=top_values, hidden_states=hidden,
                anchor_token_ids=ids[anchors],
            )
            correct = proposal == labels
            prefixes = correct.long().cumprod(-1).sum(-1)
            metrics = {
                "id": sample["id"], "domain": sample["domain"],
                "anchors": sample["anchors"], "mean_prefix": float(prefixes.float().mean()),
                "prefixes": prefixes.tolist(), "unary_ce": float(unary_ce.detach()),
                "selector_ce": float(selector_ce.detach()), "coverage": float(covered.float().mean()),
                "per_position_agreement": correct.float().mean(0).tolist(),
                "proposal": proposal.tolist(),
            }
        return unary_ce + selector_ce, metrics

    @torch.no_grad()
    def evaluate(self) -> dict[str, Any]:
        """Measure the same validation anchors at the BF16 export precision."""
        rows = [self.objective(sample)[1] for sample in self.samples
                if sample["split"] == "validation"]
        return {"rows": rows, "domains": grouped_metrics(rows)}


def train(
    dataset: Path, checkpoint: Path, checkout: Path, weights: Path,
    protocol_path: Path, output: Path,
) -> dict[str, Any]:
    """Run at most one seeded pass with the frozen protocol's budget and preserve every result."""
    output.mkdir(exist_ok=False)
    protocol = json.loads(protocol_path.read_bytes())
    report: dict[str, Any] = {
        "schema": "mxwave-drafter-context-pilot-v1-result", "status": "initializing",
        "protocol_sha256": digest(protocol_path), "trainer_sha256": digest(Path(__file__)),
        "production_changed": False, "sealed_test_opened": False, "steps": [], "evaluations": [],
    }
    started = time.monotonic()

    def write() -> None:
        report["elapsed_seconds"] = time.monotonic() - started
        tmp = output / "report.incomplete"
        tmp.write_text(json.dumps(report, indent=2) + "\n")
        tmp.replace(output / "report.json")
        print(json.dumps({
            "status": report["status"], "steps": len(report["steps"]),
            "elapsed_seconds": report["elapsed_seconds"],
        }), flush=True)

    try:
        schedule = TrainingSchedule.from_protocol(protocol)
        torch.set_num_threads(4)
        torch.manual_seed(42)
        pilot = Pilot(dataset, checkpoint, checkout, weights, protocol)
        report["inputs"] = pilot.inputs
        manifest = json.loads((dataset / "manifest.json").read_bytes())
        for split in ("train", "validation"):
            if sum(row["split"] == split for row in manifest["rows"]) != protocol["data"][
                f"{split}_prompts"
            ]:
                raise ValueError(f"Frozen {split} dataset count differs from the protocol")
        train_samples = [s for s in pilot.samples if s["split"] == "train"]
        random.Random(42).shuffle(train_samples)
        if not train_samples:
            raise ValueError("No eligible training prompts")
        report["planned_updates"] = min(schedule.maximum_steps, len(train_samples))
        report["eligible_training_prompts"] = len(train_samples)
        with torch.no_grad():
            sample = train_samples[0]
            isolated = pilot.forward(sample)
            automatic = torch.cat([
                pilot.forward_blocks(sample, [a], automatic_mask=True) for a in sample["anchors"]
            ])
            report["isolated_explicit_vs_inference_mask_bitwise_equal"] = torch.equal(
                isolated, automatic,
            )
            if not torch.equal(isolated, automatic):
                raise ValueError("Explicit single-block masks differ from inference masks")
            report["execution_policy"] = "One independent eight-row forward per anchor."
        baseline = pilot.evaluate()
        if baseline != pilot.evaluate():
            raise ValueError("Validation baseline is not exactly repeatable")
        report["baseline"] = baseline
        report["baseline_repeat_exact"] = True
        if "baseline_reference" in protocol:
            reference = Path(protocol["baseline_reference"]["path"])
            if digest(reference) != protocol["baseline_reference"]["sha256"]:
                raise ValueError("Baseline reference hash differs")
            report["baseline_vs_original_derivatives_exact"] = (
                baseline == json.loads(reference.read_bytes())
            )
            if not report["baseline_vs_original_derivatives_exact"]:
                raise ValueError("Forward baseline changed with derivative implementation")
            report["fp32_derivative_linear_modules"] = pilot.overridden_linears
        if set(baseline["domains"]) != {"code", "chat", "explanation"}:
            raise ValueError("Validation lacks an eligible domain")
        write()
        optimizer = torch.optim.AdamW([pilot.master], lr=schedule.peak_rate, weight_decay=0)
        for step, sample in enumerate(train_samples[:schedule.maximum_steps], 1):
            rate = schedule.learning_rate(step)
            optimizer.param_groups[0]["lr"] = rate
            optimizer.zero_grad(set_to_none=True)
            step_start = time.monotonic()
            loss, metrics = pilot.objective(sample)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            if pilot.master.grad is None or not torch.isfinite(pilot.master.grad).all():
                raise ValueError("Non-finite or missing context gradient")
            grad_norm = torch.nn.utils.clip_grad_norm_([pilot.master], 1.0)
            if any(p.grad is not None for p in pilot.model.parameters()):
                raise ValueError("A frozen parameter received a gradient")
            optimizer.step()
            report["steps"].append({
                "step": step, "id": sample["id"], "domain": sample["domain"],
                "loss_before_update": float(loss.detach()), "learning_rate": rate,
                "gradient_l2_before_clip": float(grad_norm), "metrics_before_update": metrics,
                "seconds": time.monotonic() - step_start,
            })
            report["status"] = "training"
            if schedule.evaluate_after(step, len(train_samples)):
                rounded = pilot.master.detach().to(torch.bfloat16)
                relative = float(
                    (rounded.float() - pilot.model.fc.weight.float()).norm(dtype=torch.float64)
                    / pilot.model.fc.weight.float().norm(dtype=torch.float64)
                )
                if relative > 0.01:
                    raise ValueError("Context weight change exceeded the frozen limit")
                evaluated = pilot.evaluate()
                path = output / f"context-step-{step:03d}.safetensors"
                save_file({"fc.weight": rounded}, str(path))
                if not torch.equal(load_file(str(path))["fc.weight"], rounded):
                    raise ValueError("Saved BF16 candidate reload differs")
                report["evaluations"].append({
                    "step": step, "relative_l2": relative, **evaluated,
                    "path": path.name, "sha256": digest(path),
                })
            write()
        if {n: tensor_digest(p) for n, p in pilot.model.named_parameters()} != pilot.original_hashes:
            raise ValueError("An original drafter tensor changed")
        selected = select_checkpoint(baseline["domains"], report["evaluations"])
        report.update(
            status="complete", selected=None if selected is None else {
                k: selected[k] for k in ("step", "path", "sha256", "relative_l2", "domains")
            },
            original_drafter_parameters_unchanged=True,
            offline_screen_pass=selected is not None, native_reload_performed=False,
            speed_gain_proven=False,
        )
        write()
        return report
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        write()
        raise


def main() -> None:
    """Run the fixed CPU pilot without modifying serving checkpoints or opening test data."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "checkpoint", "checkout", "weights", "protocol", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    train(args.dataset, args.checkpoint, args.checkout, args.weights, args.protocol, args.output)


if __name__ == "__main__":
    main()
