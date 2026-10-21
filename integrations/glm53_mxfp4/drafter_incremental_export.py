"""Export and independently check one capture chunk per rank before releasing staging files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import save_file

from .audit_drafter_feature_export_numpy import arrays
from .drafter_feature_sequence import LAYERS, AcceptedHistory, load_frame
from .drafter_fp8_roundtrip import digest


def audit_rank_export(paths: list[Path], exported: Path, expected_ids: list[int]) -> int:
    """Derive accepted intervals from positions and compare every feature bit using NumPy."""
    target = arrays(exported)
    if (target["input_ids"].tolist() != expected_ids
            or target["positions"].tolist() != list(range(len(expected_ids)))
            or target["target_features"].shape != (len(expected_ids) - 1, 20480)):
        raise ValueError("Exported IDs, positions or feature geometry differ")
    written, compared = 0, 0
    for path in paths:
        src = arrays(path)
        start = int(src["target_positions"][0])
        end = int(src["draft_query_positions"][0])
        valid = min(end, len(expected_ids) - 1) - start
        if valid <= 0:
            continue
        if (start != written or valid > len(src["target_input_ids"])
                or src["target_positions"][:valid].tolist() != list(range(start, start + valid))
                or src["target_input_ids"][:valid].tolist() != expected_ids[start:start + valid]):
            raise ValueError("Position-derived accepted history has a gap, overlap or token mismatch")
        for column, layer in enumerate(LAYERS):
            selected = target["target_features"][
                start:start + valid, column * 4096:(column + 1) * 4096,
            ]
            if not np.array_equal(selected, src[f"aux_layer_{layer}"][:valid]):
                raise ValueError("Exported feature bits differ from native accepted rows")
            compared += selected.size
        written += valid
    if written != len(expected_ids) - 1:
        raise ValueError("Missing accepted feature rows")
    return compared


def export_case(
    root: Path, case: dict[str, Any], response: list[int], rank: int, output: Path,
) -> dict[str, Any]:
    """Export one rank's accepted prefix and independently check it before returning a receipt."""
    identifier = case["id"]
    if (rank not in (0, 1) or case["split"] not in ("train", "validation")
            or not identifier or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                                     for c in identifier)):
        raise ValueError("Invalid capture identity or sealed split")
    path = output / f"{case['split']}-{identifier}.safetensors"
    if path.exists():
        raise FileExistsError(path)
    paths = sorted((root / f"ct-r{rank}").glob(f"context-{identifier}-*-r{rank}.safetensors"))
    if not paths:
        raise ValueError("Missing native feature frames")
    history = AcceptedHistory()
    frames = []
    for frame in paths:
        tensors, metadata = load_frame(frame)
        if (metadata["case"] != identifier or metadata["rank"] != rank
                or metadata["draft_graph"] != [1, 8]):
            raise ValueError("Frame identity or native draft graph differs")
        history.append(tensors, metadata)
        frames.append({
            "path": str(frame.relative_to(root)), "bytes": frame.stat().st_size,
            "sha256": metadata["payload_sha256"],
            "metadata_sha256": digest(frame.with_suffix(".json")),
        })
    stats = history.validate_response(case["prompt_ids"], response)
    expected_ids = case["prompt_ids"] + response
    ids = torch.tensor(expected_ids, dtype=torch.int64)
    output.mkdir(parents=True, exist_ok=True)
    save_file({
        "input_ids": ids, "positions": torch.arange(ids.numel()),
        "target_features": history.context()[:stats["feature_rows"]].contiguous(),
        "generated_token_mask": torch.arange(ids.numel()) >= len(case["prompt_ids"]),
    }, str(path))
    # The container writes as root; host verification and training must read the payload.
    path.chmod(0o644)
    del history, tensors
    compared = audit_rank_export(paths, path, expected_ids)
    return {
        "id": identifier, "split": case["split"], "domain": case["domain"],
        "prompt_sha256": case["prompt_sha256"], "token_ids_sha256": case["token_ids_sha256"],
        **stats, "path": path.name, "bytes": path.stat().st_size, "sha256": digest(path),
        "numpy_bf16_elements_compared": compared, "different_elements": 0,
        "raw_frames": frames,
    }


def export_chunk(root: Path, rank: int, start: int, stop: int) -> dict[str, Any]:
    """Check one frozen contiguous chunk while leaving the service and raw staging intact."""
    plan = json.loads((root / "plan.json").read_bytes())
    cases = plan["calibration_cases"]
    if rank not in (0, 1) or not 0 <= start < stop <= len(cases) or stop - start > 64:
        raise ValueError("Invalid rank or bounded chunk range")
    receipt = root / f"export-{start:04d}-{stop:04d}-r{rank}.json"
    if receipt.exists():
        raise FileExistsError(receipt)
    measured = json.loads((root / "ct-measurement.json").read_bytes())
    rows = []
    for case in cases[start:stop]:
        matches = [r for r in measured["cases"] if r["case"] == case["id"]
                   and r["policy"] == "capture"]
        if len(matches) != 1 or len(matches[0]["result"]["token_ids"]) != plan["output_tokens"]:
            raise ValueError("Missing or duplicate completed capture response")
        rows.append(export_case(root, case, matches[0]["result"]["token_ids"], rank,
                                root / "accepted"))
    report = {
        "status": "rank_verified", "rank": rank, "start": start, "stop": stop,
        "plan_sha256": digest(root / "plan.json"), "rows": rows,
        "source_sha256": digest(Path(__file__)), "sealed_test_opened": False,
        "production_restored": False, "raw_staging_deleted": False,
    }
    receipt.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    """Export one explicitly bounded rank chunk without deleting source evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = export_chunk(args.root, args.rank, args.start, args.stop)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))


if __name__ == "__main__":
    main()
