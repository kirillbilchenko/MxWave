"""Independently audit exported feature bits using positions and NumPy raw-file views."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

from .drafter_fp8_roundtrip import digest


def arrays(path: Path) -> dict[str, np.ndarray]:
    """Read supported safetensors fields directly, treating BF16 exclusively as raw bits."""
    with path.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(size))
    types = {"I64": "<i8", "I32": "<i4", "I16": "<u2", "BF16": "<u2",
             "F32": "<f4", "BOOL": "?"}
    return {
        name: np.memmap(path, mode="r", dtype=types[row["dtype"]],
                        offset=8 + size + row["data_offsets"][0], shape=tuple(row["shape"]))
        for name, row in header.items() if name != "__metadata__"
    }


def audit(evidence: Path, exported: Path, output: Path) -> dict[str, Any]:
    """Derive valid row counts from successive draft positions, independent of the exporter."""
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((exported / "manifest.json").read_bytes())
    plan = json.loads((evidence / "plan.json").read_bytes())
    measured = json.loads((evidence / "ct-measurement.json").read_bytes())
    cases = {r["id"]: r for r in plan["calibration_cases"]}
    responses = {r["case"]: r["result"]["token_ids"] for r in measured["cases"]
                 if r["policy"] == "capture"}
    checked = 0
    for row in manifest["rows"]:
        dst = exported / row["path"]
        if digest(dst) != row["sha256"]:
            raise ValueError("Export changed after verification")
        target = arrays(dst)
        expected_ids = cases[row["id"]]["prompt_ids"] + responses[row["id"]]
        if target["input_ids"].tolist() != expected_ids:
            raise ValueError("Exported token sequence differs from the actual API response")
        for rank in (0, 1):
            written = 0
            for path in sorted((evidence / f"ct-r{rank}").glob(
                f"context-{row['id']}-*-r{rank}.safetensors",
            )):
                src = arrays(path)
                start = int(src["target_positions"][0])
                end = int(src["draft_query_positions"][0])
                valid = min(end, len(expected_ids) - 1) - start
                if valid <= 0:
                    continue
                if start != written or valid > len(src["target_input_ids"]):
                    raise ValueError("Position-derived context contains a gap or overlap")
                if src["target_positions"][:valid].tolist() != list(range(start, start + valid)):
                    raise ValueError("Native target positions are not contiguous")
                if src["target_input_ids"][:valid].tolist() != expected_ids[start:start + valid]:
                    raise ValueError("Native input prefix differs from actual API history")
                for column, layer in enumerate((5, 14, 24, 33, 42)):
                    selected = target["target_features"][
                        start:start + valid, column * 4096:(column + 1) * 4096,
                    ]
                    if not np.array_equal(selected, src[f"aux_layer_{layer}"][:valid]):
                        raise ValueError("Exported feature bits differ from native accepted rows")
                    checked += selected.size
                written += valid
            if written != len(expected_ids) - 1:
                raise ValueError("Missing position-derived feature rows")
    report = {
        "schema": "mxwave-drafter-feature-export-numpy-audit-v1", "status": "verified",
        "prompts": len(manifest["rows"]), "ranks": 2, "bf16_elements_compared": checked,
        "different_elements": 0, "export_manifest_sha256": digest(exported / "manifest.json"),
        "source_sha256": digest(Path(__file__)),
        "method": "Raw safetensors + NumPy; accepted intervals derived from next-query positions, not sampler counts.",
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    """Audit a completed feature export without loading PyTorch tensors or training data code."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("evidence", "exported", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.evidence, args.exported, args.output), indent=2))


if __name__ == "__main__":
    main()
