"""Export the measured compact soft-RMS candidate without requantizing any weights."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import time
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path

import torch
from assemble_kolibri_mixed import _inventory, _payload_hash
from kolibri_soft_rms_trial import composed_weights
from safetensors import safe_open

from mxwave.incremental_safetensors import IncrementalSafeTensorsWriter, source_data_start
from mxwave.shard import ShardFile, discover_shards, shard_tensor_info
from mxwave.verify import verify_config_coverage


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.incomplete")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _weights(baseline: Path) -> Iterator[tuple[str, torch.Tensor]]:
    for shard in discover_shards(baseline)[0]:
        with safe_open(str(shard.path), framework="pt", device="cpu") as reader:
            for name in sorted(reader.keys()):
                yield name, reader.get_tensor(name)


def export(specification: Path, output: Path) -> dict:
    """Emit resumable shards and verify every payload against the measured trial hashes."""
    started = time.monotonic()
    spec_raw = specification.read_bytes()
    spec = json.loads(spec_raw)
    conversion = json.loads(specification.with_name("conversion-result.json").read_text())
    if (
        conversion["status"] != "complete"
        or conversion["trial_specification_sha256"] != hashlib.sha256(spec_raw).hexdigest()
    ):
        raise ValueError("Soft-RMS conversion does not identify this complete trial")
    baseline = Path(spec["baseline"])
    parent = json.loads((baseline / "mxwave-manifest.json").read_text())
    config = json.loads((baseline / "config.json").read_text())
    inventory = _inventory(baseline)
    expected = spec["effective_payload_sha256"]
    if (
        set(inventory) != set(expected)
        or config["quantization_config"]["format"] != "mixed-precision"
    ):
        raise ValueError("Soft-RMS export requires the complete mixed parent inventory")
    modules = {
        name.removesuffix(".weight_packed")
        if name.endswith(".weight_packed")
        else name.removesuffix(".weight")
        for name in inventory
        if name.endswith((".weight", ".weight_packed"))
    }
    qconfig = config["quantization_config"]
    targets = [target for group in qconfig["config_groups"].values() for target in group["targets"]]
    if verify_config_coverage(targets, qconfig["ignore"], sorted(modules)):
        raise ValueError("Soft-RMS export configuration leaves uncovered modules")
    identity = {
        "trial_specification_sha256": hashlib.sha256(spec_raw).hexdigest(),
        "baseline_manifest_sha256": _sha(baseline / "mxwave-manifest.json"),
        "export_source_sha256": _sha(Path(__file__)),
        "patch_loader_source_sha256": _sha(Path(__file__).with_name("kolibri_soft_rms_trial.py")),
    }
    if identity["baseline_manifest_sha256"] != spec["baseline_manifest_sha256"]:
        raise ValueError("Soft-RMS parent manifest changed")
    ledger_path = output / ".mxwave-soft-export-state.json"
    if output.exists():
        if (output / "mxwave-manifest.json").exists() or not ledger_path.exists():
            raise FileExistsError(f"Refusing to overwrite complete or foreign output: {output}")
        ledger = json.loads(ledger_path.read_text())
        if ledger["identity"] != identity:
            raise ValueError("Soft-RMS export resume identity changed")
    else:
        required = sum(row["bytes"] for row in parent["shard_integrity"]["per_shard"].values())
        if shutil.disk_usage(output.parent).free < required + 3 * 1024**3:
            raise OSError("Soft-RMS export requires the complete checkpoint plus a 3 GiB reserve")
        output.mkdir()
        ledger = {"identity": identity, "shards": {}}
        _write(ledger_path, ledger)
    shards = discover_shards(baseline)[0]
    groups = {shard.path.name: shard_tensor_info(shard) for shard in shards}
    changed = {
        target.removesuffix(".weight") + suffix
        for target in spec["patches"]
        for suffix in (".weight_packed", ".weight_scale")
    }
    current = None
    stack = None
    writer = None
    original = None
    temporary = None
    completed = 0

    def finish() -> None:
        nonlocal completed
        if current is None:
            return
        destination = output / current
        if writer is not None:
            writer.finish()
            assert stack is not None and temporary is not None
            stack.close()
            infos = shard_tensor_info(ShardFile(temporary, {}))
            if set(infos) != set(groups[current]):
                raise ValueError("Exported shard inventory differs from its source")
            with temporary.open("rb") as stream:
                start = source_data_start(stream, temporary)
                for name, info in infos.items():
                    if _payload_hash(stream, start, info) != expected[name]:
                        raise ValueError(
                            f"Exported payload differs from measured candidate: {name}"
                        )
            digest = _sha(temporary)
            temporary.replace(destination)
            ledger["shards"][current] = {"bytes": destination.stat().st_size, "sha256": digest}
            _write(ledger_path, ledger)
        completed += 1
        print(
            f"Exported and verified soft-RMS shard {completed}/{len(shards)}: {current}", flush=True
        )

    try:
        for name, tensor in composed_weights(_weights(baseline), specification):
            filename = inventory[name][0].name
            if filename != current:
                finish()
                current = filename
                writer = None
                if filename in ledger["shards"]:
                    row = ledger["shards"][filename]
                    destination = output / filename
                    if (
                        destination.stat().st_size != row["bytes"]
                        or _sha(destination) != row["sha256"]
                    ):
                        raise ValueError(f"Previously exported shard changed: {filename}")
                    continue
                stack = ExitStack()
                original = stack.enter_context((baseline / filename).open("rb"))
                temporary = output / f".{filename}.incomplete"
                stream = stack.enter_context(temporary.open("w+b"))
                writer = IncrementalSafeTensorsWriter(
                    stream,
                    {key: (info.shape, info.dtype) for key, info in groups[filename].items()},
                )
            if writer is None:
                continue
            if name in changed:
                writer.write_u8_chunk(name, tensor)
            else:
                assert original is not None
                writer.copy_tensor_payload(
                    name,
                    original,
                    baseline / filename,
                    inventory[name][1],
                    source_payload_start=source_data_start(original, baseline / filename),
                )
        finish()
    finally:
        if stack is not None:
            stack.close()
    if set(ledger["shards"]) != set(groups) or specification.read_bytes() != spec_raw:
        raise ValueError("Incomplete or changed soft-RMS export")
    for name in (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "chat_template.jinja",
        "LICENSE",
        "NOTICE",
    ):
        if (baseline / name).is_file():
            shutil.copyfile(baseline / name, output / name)
    manifest = copy.deepcopy(parent)
    manifest.pop("expert_manifest_sha256", None)
    manifest["unweighted_expert_manifest_sha256"] = parent.get("expert_manifest_sha256")
    manifest.update(identity)
    manifest["activation_calibration"] = {
        "objective": "softened routed RMS",
        "recipe": spec["recipe"],
        "calibration_file_sha256": spec["calibration_file_sha256"],
        "counts_sha256": spec["counts_sha256"],
        "covered_targets": conversion["covered_targets"],
        "changed_targets": len(spec["patches"]),
    }
    manifest["expert_weights"] = (
        "Measured soft-RMS MXFP4 E2M1 expert payloads exported from compact patches"
    )
    manifest["payload_sha256"] = dict(expected)
    manifest["payload_verification"] = (
        "Every physical payload matches the measured compact soft-RMS trial"
    )
    manifest["shard_integrity"] = {"per_shard": ledger["shards"]}
    manifest["config_coverage_gaps"] = []
    manifest["quality_qualification"] = (
        "Frozen compact trial measured; standalone serving qualification pending"
    )
    _write(output / "mxwave-manifest.json", manifest)
    result = {
        "status": "complete",
        "seconds": time.monotonic() - started,
        "shards": len(shards),
        "verified_payloads": len(expected),
        "weight_files_bytes": sum(row["bytes"] for row in ledger["shards"].values()),
        "manifest_sha256": _sha(output / "mxwave-manifest.json"),
        **identity,
        "scope": "Exact export of measured payloads; no requantization",
    }
    _write(output.parent / f"{output.name}-export.json", result)
    print(json.dumps(result), flush=True)
    return result


def main() -> None:
    """Export the frozen trial into a separate standalone compressed-tensors checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.specification, args.output)


if __name__ == "__main__":
    main()
