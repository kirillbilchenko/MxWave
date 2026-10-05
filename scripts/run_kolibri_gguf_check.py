"""Measure the native GGUF against the same frozen tokens and official FP8 reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file


def sha(path: Path) -> str:
    """Hash a complete artifact with bounded memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    """Run the pinned C++ harness, preserve full distributions, and apply the frozen screen."""
    from evaluate_kolibri_release import _check, _token_sha, compare

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    model = root / "gguf/Kolibri-1-MXFP4.gguf"
    conversion = json.loads(model.with_suffix(".json").read_text())
    if (
        conversion["status"] != "complete"
        or conversion["verified_source_payloads"] != 116303
        or conversion["verified_gguf_tensors"] != 903
        or sha(model) != conversion["output_sha256"]
    ):
        raise ValueError("Full GGUF conversion or its independent verification is incomplete")
    protocol_path = root / "validation/protocol.json"
    protocol_sha = sha(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    for item in protocol["windows"] + protocol["behavior_cases"]:
        if _token_sha(item["token_ids"]) != item["token_ids_sha256"]:
            raise ValueError("Frozen protocol token IDs changed")
    view = root / "gguf/qualification"
    (view / "measurements").mkdir(parents=True, exist_ok=True)
    for name, target in {
        "validation": root / "validation",
        "measurements/fp8.json": root / "measurements/fp8.json",
        "measurements/fp8.safetensors": root / "measurements/fp8.safetensors",
    }.items():
        path = view / name
        if not path.exists():
            path.symlink_to(target)
    output = view / "measurements/mse.json"
    if output.exists():
        raise FileExistsError("GGUF measurements already exist")
    started = time.monotonic()
    harness = root / "gguf/build/bin/kolibri-metrics"
    subprocess.run([str(harness), str(model), str(protocol_path), str(output), "9216"], check=True)
    report = json.loads(output.read_text())
    if report["vocabulary"] != 128000 or len(report["windows"]) != 48:
        raise ValueError("GGUF measurement dimensions do not match the frozen reference")
    raw = np.fromfile(str(output) + ".logprobs.f32", dtype="<f4")
    if raw.size != 48 * 128000 or not np.isfinite(raw).all():
        raise ValueError("GGUF full-vocabulary distributions are incomplete or nonfinite")
    save_file(
        {"logprobs": torch.from_numpy(raw.reshape(48, 128000))},
        str(output.with_suffix(".safetensors")),
        metadata={"protocol_sha256": protocol_sha},
    )
    for row, case in zip(report["behavior"], protocol["behavior_cases"], strict=True):
        if row["id"] != case["id"]:
            raise ValueError("Behavior check order differs from the frozen protocol")
        row["passed"] = _check(case, row["text"])
    report.update(
        protocol_sha256=protocol_sha,
        gguf_sha256=conversion["output_sha256"],
        checkpoint_manifest_sha256=conversion["source_manifest_sha256"],
        runtime="patched llama.cpp CUDA SM121; native MXFP4 experts / BF16 backbone",
        harness_binary_sha256=sha(harness),
        seconds=time.monotonic() - started,
    )
    output.write_text(json.dumps(report, indent=2) + "\n")
    compare(argparse.Namespace(root=view))
    quality = json.loads((view / "quality-comparison.json").read_text())
    quality["candidate"] = "softer RMS native MXFP4 GGUF; FP8 backbone dequantized to BF16"
    quality["gguf_sha256"] = conversion["output_sha256"]
    (view / "quality-comparison.json").write_text(json.dumps(quality, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": quality["status"],
                "ppl": quality["domains"]["all"]["candidate_ppl"],
                "mean_next_token_kl": quality["mean_next_token_kl"],
                "behavior_passed": all(row["passed"] for row in report["behavior"]),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
