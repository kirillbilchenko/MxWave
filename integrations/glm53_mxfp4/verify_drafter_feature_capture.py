"""Verify restored capture evidence and export accepted CT supervision sequences."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from .drafter_feature_sequence import AcceptedHistory, load_frame
from .drafter_fp8_roundtrip import digest


def verify(evidence: Path, output: Path) -> dict[str, Any]:
    """Validate hashes, native controls and both ranks before exporting train/validation data."""
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((evidence / "evidence-manifest.json").read_bytes())
    for row in manifest["files"]:
        path = evidence / row["path"]
        if not path.resolve().is_relative_to(evidence.resolve()):
            raise ValueError("Manifest path escapes evidence directory")
        if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError(f"Evidence integrity failure: {row['path']}")
    state = json.loads((evidence / "controller.json").read_bytes())
    restoration = state.get("restoration", {})
    if (state["phase"] != "complete_restored" or state["service_paused"]
            or not restoration.get("originals_running") or not restoration.get("health_200")):
        raise ValueError("Successful capture and production restoration are required")
    if not any(m["max_model_len"] == 500000 and m["id"] == "glm-5.3-flash"
               for m in restoration["models"]["data"]):
        raise ValueError("Restored service identity/context does not match")
    plan = json.loads((evidence / "plan.json").read_bytes())
    prepared = json.loads((evidence / "prepared.json").read_bytes())
    if digest(evidence / "plan.json") != prepared["plan_sha256"]:
        raise ValueError("Prepared plan changed")
    if len(plan["calibration_cases"]) != 40 or plan["requests"] != 44:
        raise ValueError("Unexpected bounded capture plan")
    measurement = json.loads((evidence / "ct-measurement.json").read_bytes())
    if measurement["status"] != "complete" or len(measurement["cases"]) != 44:
        raise ValueError("Capture requests are incomplete")
    controls = []
    for case in plan["control_cases"]:
        rows = [r for r in measurement["cases"] if r["case"] == case]
        if [r["policy"] for r in rows] != ["before", "capture", "after"]:
            raise ValueError("Missing before/capture/after control")
        equal = all(r["result"]["token_ids"] == rows[0]["result"]["token_ids"]
                    and r["counter_delta"] == rows[0]["counter_delta"] for r in rows)
        if not equal:
            raise ValueError(f"Passive capture changed outputs/counters: {case}")
        controls.append({"case": case, "outputs_and_counters_equal": True})
    output.mkdir()
    rows = []
    for case in plan["calibration_cases"]:
        if case["split"] not in ("train", "validation"):
            raise ValueError("Sealed test must not be captured")
        measured = [r for r in measurement["cases"]
                    if r["case"] == case["id"] and r["policy"] == "capture"]
        if len(measured) != 1:
            raise ValueError("Missing or repeated capture request")
        response = measured[0]["result"]["token_ids"]
        if len(response) != plan["output_tokens"]:
            raise ValueError("Unexpected output length")
        histories = []
        for rank in (0, 1):
            history = AcceptedHistory()
            paths = sorted((evidence / f"ct-r{rank}").glob(
                f"context-{case['id']}-*-r{rank}.safetensors",
            ))
            if not paths:
                raise ValueError(f"Missing feature frames: {case['id']} rank {rank}")
            for path in paths:
                tensors, metadata = load_frame(path)
                if (metadata["case"] != case["id"] or metadata["rank"] != rank
                        or metadata["draft_graph"] != [1, 8]):
                    raise ValueError("Frame identity or native graph mismatch")
                history.append(tensors, metadata)
            sequence = history.validate_response(case["prompt_ids"], response)
            histories.append((history, sequence))
        first, stats = histories[0]
        second, second_stats = histories[1]
        n = stats["feature_rows"]
        features = first.context()[:n].contiguous()
        if stats != second_stats or not torch.equal(features, second.context()[:n]):
            raise ValueError(f"Rank target features differ: {case['id']}")
        ids = torch.tensor(case["prompt_ids"] + response, dtype=torch.int64)
        if not torch.equal(first.tokens[:n], second.tokens[:n]):
            raise ValueError("Rank input sequences differ")
        path = output / f"{case['split']}-{case['id']}.safetensors"
        save_file({
            "input_ids": ids,
            "positions": torch.arange(ids.numel()),
            "target_features": features,
            "generated_token_mask": torch.arange(ids.numel()) >= len(case["prompt_ids"]),
        }, str(path))
        rows.append({
            "id": case["id"], "split": case["split"], "domain": case["domain"],
            "prompt_sha256": case["prompt_sha256"],
            "token_ids_sha256": case["token_ids_sha256"],
            **stats, "rank_features_bitwise_equal": True,
            "path": path.name, "bytes": path.stat().st_size, "sha256": digest(path),
        })
    counts = Counter((r["split"], r["domain"]) for r in rows)
    if counts != Counter({
        (split, domain): count for split, domains in plan["quotas"].items()
        for domain, count in domains.items()
    }):
        raise ValueError("Exported split/domain counts differ from the frozen plan")
    report = {
        "schema": "mxwave-ct-drafter-feature-dataset-v1", "status": "verified",
        "verifier_sha256": digest(Path(__file__)),
        "sequence_builder_sha256": digest(Path(__file__).with_name("drafter_feature_sequence.py")),
        "evidence_manifest_sha256": digest(evidence / "evidence-manifest.json"),
        "plan_sha256": digest(evidence / "plan.json"),
        "verified_evidence_payloads": len(manifest["files"]),
        "frozen_dataset_hashes": plan["frozen_dataset_hashes"],
        "controls": controls, "rows": rows,
        "total_feature_rows": sum(r["feature_rows"] for r in rows),
        "total_bytes": sum(r["bytes"] for r in rows),
        "all_rank_features_bitwise_equal": True, "sealed_test_opened": False,
        "production_restored": True, "training_performed": False,
        "feature_contract": {
            "target": "CT MXFP4 experts + previously qualified native MXFP8 KDA rounding",
            "layers": [5, 14, 24, 33, 42], "dtype": "BF16",
            "input_ids": "Frozen prompt followed by exactly 128 verified CT response tokens.",
            "target_features": "One row per input token except the final response token; five layers concatenated.",
            "generated_token_mask": "Identifies response token positions, not a finalized training loss mask.",
            "rejected_and_post_cap_work": "Excluded before export.",
        },
        "limits": [
            "Bounded short-prompt public-proxy subset, not the full corpus or long-context screen.",
            "Fixed-length ignore-EOS outputs may include post-EOS tokens; trainer must apply its explicit mask policy.",
            "No training benefit, quality or speed claim from instrumented feature capture.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    """Verify a completed restored capture and materialize accepted sequences."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = verify(args.evidence, args.output)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
