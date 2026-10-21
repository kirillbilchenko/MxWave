"""Export only a validation-selected BF16 context tensor into the native FP8 file."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from .drafter_fp8_roundtrip import SOURCE_SHA256, digest, raw, read_header
from .train_drafter_context_pilot import select_checkpoint


def patch_native(
    source: Path, patch: Path, output: Path, *, expected_sha256: str = SOURCE_SHA256,
) -> dict[str, Any]:
    """Replace one existing BF16 payload and verify every other serialized tensor."""
    original = source / "model.safetensors"
    if digest(original) != expected_sha256:
        raise ValueError("Original native checkpoint hash differs")
    prefix, header = read_header(original)
    descriptor = header["fc.weight"]
    if descriptor["dtype"] != "BF16":
        raise ValueError("Native context projection is not BF16")
    with safe_open(patch, framework="pt", device="cpu") as tensors:
        if list(tensors.keys()) != ["fc.weight"]:
            raise ValueError("Only the context projection may be exported")
        value = tensors.get_tensor("fc.weight")
    if value.dtype != torch.bfloat16 or list(value.shape) != descriptor["shape"]:
        raise ValueError("Context projection shape or dtype differs")
    if not torch.isfinite(value).all():
        raise ValueError("Non-finite context projection")
    with safe_open(original, framework="pt", device="cpu") as tensors:
        reference = tensors.get_tensor("fc.weight")
        relative_l2 = float((value.float() - reference.float()).norm(dtype=torch.float64)
                            / reference.float().norm(dtype=torch.float64).clamp_min(1e-20))
    if not 0 < relative_l2 <= 0.01:
        raise ValueError("Context update is empty or exceeds the fixed limit")
    output.mkdir(parents=True, exist_ok=False)
    destination = output / "model.safetensors"
    shutil.copyfile(original, destination)
    start, end = descriptor["data_offsets"]
    payload = raw(value)
    if len(payload) != end - start:
        raise ValueError("Serialized context payload length differs")
    with destination.open("r+b") as stream:
        stream.seek(len(prefix) + start)
        stream.write(payload)
    rows = []
    with original.open("rb") as before, destination.open("rb") as after:
        if before.read(len(prefix)) != after.read(len(prefix)):
            raise ValueError("Native serialization header changed")
        for name, desc in header.items():
            if name == "__metadata__":
                continue
            start, end = desc["data_offsets"]
            before.seek(len(prefix) + start)
            after.seek(len(prefix) + start)
            old_hash, new_hash = hashlib.sha256(), hashlib.sha256()
            remaining = end - start
            while remaining:
                count = min(1024 * 1024, remaining)
                old_hash.update(before.read(count))
                new_hash.update(after.read(count))
                remaining -= count
            changed = old_hash.digest() != new_hash.digest()
            if changed != (name == "fc.weight"):
                raise ValueError(f"Unexpected tensor change: {name}")
            rows.append({"name": name, "changed": changed,
                         "original_sha256": old_hash.hexdigest(),
                         "exported_sha256": new_hash.hexdigest()})
    shutil.copyfile(source / "config.json", output / "config.json")
    return {
        "status": "verified", "original_sha256": expected_sha256,
        "patch_sha256": digest(patch), "exported_sha256": digest(destination),
        "config_bytes_identical": digest(source / "config.json") == digest(output / "config.json"),
        "only_changed_tensor": "fc.weight", "relative_l2": relative_l2,
        "tensor_count": len(rows), "tensors": rows, "native_reload_performed": False,
    }


def export_selected(
    source: Path, run: Path, output: Path, *, expected_sha256: str = SOURCE_SHA256,
) -> dict[str, Any]:
    """Require the completed frozen offline screen before writing a native candidate."""
    report = json.loads((run / "report.json").read_bytes())
    if (report["status"] != "complete" or not report.get("offline_screen_pass")
            or not report.get("original_drafter_parameters_unchanged")):
        raise ValueError("No completed, eligible context candidate")
    selected = select_checkpoint(report["baseline"]["domains"], report["evaluations"])
    if selected is None or report["selected"]["sha256"] != selected["sha256"]:
        raise ValueError("Reported selection does not reproduce")
    if Path(selected["path"]).name != selected["path"]:
        raise ValueError("Candidate path must be local to its run")
    patch = run / selected["path"]
    if digest(patch) != selected["sha256"]:
        raise ValueError("Selected checkpoint hash differs")
    result = patch_native(source, patch, output, expected_sha256=expected_sha256)
    result.update(
        source_code_sha256=digest(Path(__file__)), run_report_sha256=digest(run / "report.json"),
        protocol_sha256=report["protocol_sha256"], selected_step=selected["step"],
        sealed_test_opened=False, speed_gain_proven=False,
    )
    (output / "export-review.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    """Export a development candidate without altering source or serving checkpoints."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "run", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = export_selected(args.source, args.run, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "tensors"}, indent=2))


if __name__ == "__main__":
    main()
