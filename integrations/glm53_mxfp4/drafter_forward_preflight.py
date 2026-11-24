"""Exercise the dense training graph on a retained native CT draft fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from .audit_drafter_specforge import load_model
from .drafter_fp8_roundtrip import digest


def tensor_digest(value: torch.Tensor) -> str:
    """Hash exact contiguous tensor bytes independently of floating-point conversion."""
    return hashlib.sha256(value.detach().contiguous().view(torch.uint8).numpy()).hexdigest()


def difference(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float | int]:
    """Report numerical differences without assigning an arbitrary parity threshold."""
    a, b = actual.detach().double(), reference.detach().double()
    delta = a - b
    return {
        "elements": a.numel(),
        "different_elements": int((a != b).sum()),
        "relative_l2": float(delta.norm() / b.norm().clamp_min(1e-20)),
        "max_absolute": float(delta.abs().max()),
        "cosine": float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0)),
    }


def run(
    checkout: Path, checkpoint: Path, fixture: Path, source_rows: Path, output: Path,
) -> dict[str, Any]:
    """Check real-input inference and one disposable context-only optimizer step."""
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = time.monotonic()
    report: dict[str, Any] = {
        "schema": "mxwave-drafter-training-preflight-v1", "status": "running",
        "production_changed": False, "candidate_for_deployment": False,
        "fixture_role": "Previously observed diagnostic; never training or held-out data.",
    }

    def record(stage: str) -> None:
        report["stage"] = stage
        report["elapsed_seconds"] = time.monotonic() - started
        (output / "review.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"stage": stage, "elapsed_seconds": report["elapsed_seconds"]}),
              flush=True)

    try:
        torch.manual_seed(42)
        torch.set_num_threads(4)
        metadata = json.loads(fixture.with_suffix(".json").read_bytes())
        if digest(fixture) != metadata["payload_sha256"]:
            raise ValueError("Native fixture integrity failure")
        if metadata["index"] != 0 or metadata["target_num_draft_tokens"] != 0:
            raise ValueError("Expected initial full-prefill context without history reconstruction")
        tensors = load_file(str(fixture))
        for name, dtype in metadata["tensor_dtypes"].items():
            if dtype == "torch.bfloat16":
                tensors[name] = tensors[name].view(torch.bfloat16)
        sources = json.loads((source_rows / "source-rows.json").read_bytes())
        for name in ("query-embeddings.bf16", "target-head.bf16"):
            if digest(source_rows / name) != sources[name]["sha256"]:
                raise ValueError(f"Source weight integrity failure: {name}")
        embeddings = torch.from_file(
            str(source_rows / "query-embeddings.bf16"), shared=False,
            size=2 * 4096, dtype=torch.bfloat16,
        ).reshape(2, 4096)
        embedding_rows = sources["query-embeddings.bf16"]["rows"]
        indices = torch.tensor([embedding_rows.index(t) for t in tensors["draft_query_ids"]])
        noise = embeddings[indices].unsqueeze(0)
        head = torch.from_file(
            str(source_rows / "target-head.bf16"), shared=False,
            size=154880 * 4096, dtype=torch.bfloat16,
        ).reshape(154880, 4096)
        model, revision = load_model(checkout, checkpoint)
        model.eval()
        model.requires_grad_(False)
        context = torch.cat(
            [tensors[f"aux_layer_{i}"] for i in model.target_layer_ids], dim=-1,
        ).unsqueeze(0)
        positions = torch.cat(
            [tensors["target_positions"], tensors["draft_query_positions"]],
        ).unsqueeze(0)
        if not torch.equal(positions[0], torch.arange(positions.numel())):
            raise ValueError("Fixture has incomplete or shifted positions")
        report.update(
            source_sha256=digest(Path(__file__)), specforge_revision=revision,
            torch_version=str(torch.__version__),
            fixture_sha256=digest(fixture), source_rows=sources,
            checkpoint_sha256=digest(checkpoint / "model.safetensors"),
            context_tokens=context.shape[1], query_tokens=noise.shape[1],
            frozen_parameter_tensors=len(list(model.parameters())),
        )
        original_hashes = {n: tensor_digest(p) for n, p in model.named_parameters()}
        record("loaded_real_fixture")
        kwargs = {
            "position_ids": positions, "noise_embedding": noise,
            "target_hidden": context, "use_cache": False,
        }
        with torch.no_grad():
            projected = model.fc(context)[0]
            report["context_projection_vs_native"] = difference(
                projected, tensors["combined_target_hidden"],
            )
            before = model(**kwargs)
            repeated = model(**kwargs)
            report["cpu_forward_bitwise_repeat"] = torch.equal(before, repeated)
            if not report["cpu_forward_bitwise_repeat"] or not torch.isfinite(before).all():
                raise ValueError("Training forward failed repeat/finite checks")
            report["draft_hidden_vs_native"] = difference(before[0, 1:], tensors["draft_hidden"])
            logits = F.linear(before[:, 1:], head).float()
            values, ids = logits.topk(model.candidate_selector.top_k, dim=-1)
            proposal = model.candidate_selector.greedy_path(
                candidate_ids=ids, unary_logits=values, hidden_states=before[:, 1:],
                anchor_token_ids=tensors["anchor_before_propose"].reshape(1),
            )
            report["cpu_proposal"] = proposal.tolist()
            report["native_proposal"] = tensors["draft_proposal"].tolist()
            report["proposal_equal"] = torch.equal(proposal, tensors["draft_proposal"])
            report["top16_set_overlap"] = [
                len(set(a) & set(b)) for a, b in zip(
                    ids[0].tolist(), tensors["draft_ids"].tolist(), strict=True,
                )
            ]
        save_file(
            {"draft_hidden": before.detach().contiguous(), "top_ids": ids, "top_values": values},
            str(output / "untrained-forward.safetensors"),
        )
        record("untrained_forward_compared")

        # This disposable MSE loss tests autograd through the entire frozen backbone.
        # It is not CT supervision, a training recipe, or a promotable trained model.
        master = model.fc.weight.detach().float().clone().requires_grad_(True)
        optimizer = torch.optim.AdamW([master], lr=1e-5, weight_decay=0)
        prediction = torch.func.functional_call(
            model, {"fc.weight": master.to(torch.bfloat16)}, (), kwargs,
        )
        loss = F.mse_loss(prediction[0, 1:].float(), tensors["draft_hidden"].float())
        report["diagnostic_mse_before"] = float(loss.detach())
        record("backward_start")
        loss.backward()
        grad = master.grad
        if grad is None or not torch.isfinite(grad).all() or not grad.count_nonzero():
            raise ValueError("Context master gradient is missing, non-finite or zero")
        report["gradient"] = {
            "parameters": master.numel(), "dtype": str(grad.dtype),
            "nonzero_elements": int(grad.count_nonzero()),
            "l2": float(grad.norm()), "max_absolute": float(grad.abs().max()),
            "all_finite": True,
        }
        if any(p.grad is not None for p in model.parameters()):
            raise ValueError("Frozen checkpoint parameter acquired a gradient")
        record("backward_complete")
        optimizer.step()
        rounded = master.detach().to(torch.bfloat16)
        changed = int((rounded != model.fc.weight).sum())
        if changed == 0 or not torch.isfinite(rounded).all():
            raise ValueError("Optimizer update did not survive BF16 export")
        report["optimizer"] = {
            "name": "AdamW", "learning_rate": 1e-5, "weight_decay": 0,
            "steps": 1, "changed_bf16_elements": changed,
            "relative_l2_bf16": float(
                (rounded.float() - model.fc.weight.float()).norm()
                / model.fc.weight.float().norm()
            ),
        }
        save_file({"fc.weight": rounded}, str(output / "disposable-fc-step.safetensors"))
        reloaded = load_file(str(output / "disposable-fc-step.safetensors"))["fc.weight"]
        if not torch.equal(rounded, reloaded):
            raise ValueError("Disposable update reload differs")
        record("optimizer_and_reload_complete")
        with torch.no_grad():
            after = torch.func.functional_call(model, {"fc.weight": reloaded}, (), kwargs)
            report["diagnostic_mse_after"] = float(
                F.mse_loss(after[0, 1:].float(), tensors["draft_hidden"].float())
            )
        final_hashes = {n: tensor_digest(p) for n, p in model.named_parameters()}
        if original_hashes != final_hashes:
            raise ValueError("Frozen model changed")
        report.update(
            status="complete", training_graph_check_pass=True,
            all_original_model_parameters_unchanged=True,
            updated_bf16_tensor_reload_exact=True,
            native_arithmetic_parity_established=False,
            limits=[
                "One short observed diagnostic, not new training/validation/test data.",
                "Dense CPU BF16 arithmetic differs from native FP8 Marlin and KV/head adapters.",
                "MSE to native draft hidden states is only an autograd smoke objective.",
                "The disposable fc update must never be promoted or scored on sealed data.",
                "No actual training convergence, acceptance, speed or quality result.",
            ],
        )
        record("complete")
        return report
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        record("failed")
        raise


def main() -> None:
    """Run the isolated real-fixture forward/backward check."""
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("checkout", "checkpoint", "fixture", "source-rows", "output"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.checkout, args.checkpoint, args.fixture, args.source_rows, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
