"""Replay retained native contexts through the unmodified upstream dense drafter."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .audit_drafter_specforge import load_model
from .drafter_feature_sequence import AcceptedHistory, load_frame
from .drafter_forward_preflight import difference
from .drafter_fp8_roundtrip import digest


def run(
    checkout: Path, checkpoint: Path, fixtures: Path, source_rows: Path, output: Path,
) -> dict[str, Any]:
    """Compare proposals at matched real histories, without fitting or opening new data."""
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    started = time.monotonic()
    sources = json.loads((source_rows / "replay-rows.json").read_bytes())
    for name in ("replay-embeddings.bf16", "target-head.bf16"):
        if digest(source_rows / name) != sources[name]["sha256"]:
            raise ValueError(f"Weight integrity failure: {name}")
    row_ids = sources["replay-embeddings.bf16"]["rows"]
    embedding = torch.from_file(
        str(source_rows / "replay-embeddings.bf16"), shared=False,
        size=len(row_ids) * 4096, dtype=torch.bfloat16,
    ).reshape(len(row_ids), 4096)
    head = torch.from_file(
        str(source_rows / "target-head.bf16"), shared=False,
        size=154880 * 4096, dtype=torch.bfloat16,
    ).reshape(154880, 4096)
    model, revision = load_model(checkout, checkpoint)
    model.eval().requires_grad_(False)
    history = AcceptedHistory()
    rows = []
    with torch.inference_mode():
        for path in sorted(fixtures.glob("context-code_32-*-r0.safetensors")):
            tensors, metadata = load_frame(path)
            history.append(tensors, metadata)
            context = history.context().unsqueeze(0)
            noise = embedding[torch.tensor(
                [row_ids.index(i) for i in tensors["draft_query_ids"].tolist()],
            )].unsqueeze(0)
            hidden = model(
                position_ids=torch.arange(context.shape[1] + 8).unsqueeze(0),
                noise_embedding=noise, target_hidden=context, use_cache=False,
            )[:, 1:]
            logits = F.linear(hidden, head).float()
            values, ids = logits.topk(16, dim=-1)
            proposal = model.candidate_selector.greedy_path(
                candidate_ids=ids, unary_logits=values, hidden_states=hidden,
                anchor_token_ids=tensors["anchor_before_propose"].reshape(1),
            )
            rows.append({
                "index": metadata["index"], "fixture_sha256": digest(path),
                "context_tokens": context.shape[1],
                "native_proposal": tensors["draft_proposal"].tolist(),
                "dense_proposal": proposal.tolist(),
                "matching_proposal_slots": int((proposal == tensors["draft_proposal"]).sum()),
                "hidden_difference": difference(hidden[0], tensors["draft_hidden"]),
                "top16_set_overlap": [
                    len(set(a) & set(b)) for a, b in zip(
                        ids[0].tolist(), tensors["draft_ids"].tolist(), strict=True,
                    )
                ],
            })
    if len(rows) != 8:
        raise ValueError("Expected all eight retained native frames")
    measurement = json.loads((fixtures.parent / "ct-measurement.json").read_bytes())
    response = next(r["result"]["token_ids"] for r in measurement["cases"]
                    if r["policy"] == "capture")
    sequence = history.validate_response(history.tokens[:history.prompt_length].tolist(), response)
    report = {
        "schema": "mxwave-drafter-native-dense-replay-v1", "specforge_revision": revision,
        "source_sha256": digest(Path(__file__)), "sequence_builder_sha256":
            digest(Path(__file__).with_name("drafter_feature_sequence.py")),
        "checkpoint_sha256": digest(checkpoint / "model.safetensors"), "source_rows": sources,
        "elapsed_seconds": time.monotonic() - started, "rows": rows, "sequence": sequence,
        "matching_slots": sum(r["matching_proposal_slots"] for r in rows), "total_slots": 56,
        "whole_blocks_equal": sum(r["matching_proposal_slots"] == 7 for r in rows),
        "native_arithmetic_parity_established": False,
        "scope": "Old observed diagnostic only; fixed real histories; no fitting or new held-out data.",
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    """Run the eight-frame retained-history comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkout", "checkpoint", "fixtures", "source-rows", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.checkout, args.checkpoint, args.fixtures,
                         args.source_rows, args.output), indent=2))


if __name__ == "__main__":
    main()
