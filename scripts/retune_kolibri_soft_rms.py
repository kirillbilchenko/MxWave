"""Emit only changed MXFP4 blocks for a bounded, confidence-shrunk RMS trial."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import shutil
import time
from contextlib import ExitStack
from functools import partial
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from mxwave.calibration import _validate_statistic, open_calibration_data
from mxwave.checkpoint import DEFAULT_TENSOR_CHUNK_MAX_ELEMENTS, iter_quantized_row_chunks
from mxwave.core import dequant_mxfp4, quantize_mxfp4
from mxwave.engine import QuantizeConfig, plan_model
from mxwave.routed_calibration import soften_routed_rms
from mxwave.shard import discover_shards


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.incomplete")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def _rows(reader, name: str, start: int, stop: int):
    return reader.get_slice(name)[start:stop]


def retune(
    source: Path,
    baseline: Path,
    statistics: Path,
    counts_path: Path,
    output: Path,
    *,
    device: str = "cuda",
    strength: float = 0.25,
    pseudo_count: float = 128.0,
    max_variance_ratio: float = 4.0,
    patch_limit_bytes: int = 6 * 1024**3,
) -> dict:
    """Quantize from the BF16 master but retain only differences from the verified parent."""
    started = time.monotonic()
    if patch_limit_bytes <= 0:
        raise ValueError("Patch storage limit must be positive")
    source, baseline, statistics, counts_path = (
        path.resolve() for path in (source, baseline, statistics, counts_path)
    )
    parent_raw = (baseline / "mxwave-manifest.json").read_bytes()
    parent = json.loads(parent_raw)
    if (
        json.loads((baseline / "config.json").read_text())["quantization_config"]["format"]
        != "mixed-precision"
    ):
        raise ValueError("Soft RMS requires the verified mixed parent")
    plan = plan_model(
        QuantizeConfig(
            model_dir=source,
            output_dir=output,
            policy="kolibri1-routed-experts",
            method="mse",
            mse_clip_depth=4,
            device=device,
            verbose=False,
        )
    )
    inputs = {item.info.name: item for item in plan.tensors if item.quantized}
    source_shards = {shard.path.name: shard.path for shard in plan.shards}
    with safe_open(str(statistics), framework="pt") as reader:
        statistic_keys = reader.keys()
        covered = {key.removeprefix("rms::") for key in statistic_keys if key.startswith("rms::")}
    if not covered or not covered.issubset(inputs):
        raise ValueError("RMS artifact contains no applicable targets or foreign targets")
    calibration = open_calibration_data(
        statistics,
        "rms",
        {name: inputs[name].info.shape[1] for name in covered},
        expected_policy="kolibri1-routed-experts",
        expected_source_repository=parent["source_repository"],
        expected_source_revision=parent["source_revision"],
    )
    counts = json.loads(counts_path.read_bytes())
    config = json.loads((baseline / "config.json").read_text())
    observations = {row["layer"]: row["routed_observations"] for row in counts["layers"]}
    if (
        counts["status"] != "passed"
        or counts["file_sha256"] != calibration.file_sha256
        or set(observations) != set(range(config["num_hidden_layers"]))
        or any(
            len(row) != config["num_experts"]
            or any(not isinstance(value, int) or value < 0 for value in row)
            for row in observations.values()
        )
    ):
        raise ValueError("Route counts are not bound to the frozen calibration and model geometry")
    # Validate the recipe before any output directory is created.
    soften_routed_rms(
        torch.ones(1),
        strength=strength,
        observations=1,
        pseudo_count=pseudo_count,
        max_variance_ratio=max_variance_ratio,
    )
    identity = {
        "statistics_file": str(statistics),
        "counts_file": str(counts_path),
        "baseline_manifest_sha256": hashlib.sha256(parent_raw).hexdigest(),
        "calibration_file_sha256": calibration.file_sha256,
        "counts_sha256": _sha(counts_path),
        "source_config_sha256": _sha(source / "config.json"),
        "retune_source_sha256": _sha(Path(__file__)),
        "gamma_source_sha256": _sha(Path(soften_routed_rms.__code__.co_filename)),
        "source_shard_fingerprints": {
            name: {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for name, path in source_shards.items()
        },
        "recipe": {
            "strength": strength,
            "pseudo_count": pseudo_count,
            "max_variance_ratio": max_variance_ratio,
            "mse_clip_depth": 4,
            "scale_percentile": 99.5,
            "row_limit": 1024,
            "element_limit": DEFAULT_TENSOR_CHUNK_MAX_ELEMENTS,
        },
    }
    state_path = output / "retune-state.json"
    if output.exists():
        if (output / "trial.json").is_file() or not state_path.is_file():
            raise FileExistsError(f"Refusing to overwrite a complete/foreign trial: {output}")
        state = json.loads(state_path.read_text())
        if state["identity"] != identity:
            raise ValueError("Soft RMS resume identity changed")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(output.parent).free < 3 * 1024**3:
            raise OSError("Soft RMS requires a 3 GiB free-space reserve")
        output.mkdir()
        state = {"identity": identity, "shards": {}, "started_at": time.time()}
        _write(state_path, state)
    is_cuda = torch.device(device).type == "cuda"
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    parent_shards = discover_shards(baseline)[0]
    for number, shard in enumerate(parent_shards, 1):
        destination = output / f"delta-{number:05d}.safetensors"
        if shard.path.name in state["shards"]:
            previous = state["shards"][shard.path.name]
            if (
                _sha(destination) != previous["sha256"]
                or destination.stat().st_size != previous["bytes"]
            ):
                raise ValueError("Completed patch shard changed")
            print(f"Verified resumed patch {number}/{len(parent_shards)}", flush=True)
            continue
        patches, targets, source_hashes, sample_metrics = {}, {}, {}, {}
        with ExitStack() as stack:
            gamma_reader = stack.enter_context(safe_open(str(statistics), framework="pt"))
            parent_reader = stack.enter_context(
                safe_open(str(shard.path), framework="pt", device=device)
            )
            names = set(parent_reader.keys())
            selected = [
                name.removesuffix(".weight_packed") + ".weight"
                for name in sorted(names)
                if name.endswith(".weight_packed")
                and name.removesuffix(".weight_packed") + ".weight" in covered
            ]
            readers = {
                filename: stack.enter_context(
                    safe_open(str(source_shards[filename]), framework="pt", device=device)
                )
                for filename in {inputs[name].shard_name for name in selected}
            }
            calibration.verify_unchanged()
            for target in selected:
                parts = target.split(".")
                count = observations[int(parts[2])][int(parts[5])]
                if count == 0:
                    raise ValueError(f"An RMS target has no recorded routes: {target}")
                rows, columns = inputs[target].info.shape
                packed_name = target.removesuffix(".weight") + ".weight_packed"
                scale_name = target.removesuffix(".weight") + ".weight_scale"
                gamma = gamma_reader.get_tensor(f"rms::{target}")
                _validate_statistic(target, "rms", gamma, columns)
                gamma = soften_routed_rms(
                    gamma.to(device),
                    strength=strength,
                    observations=count,
                    pseudo_count=pseudo_count,
                    max_variance_ratio=max_variance_ratio,
                )
                source_reader = readers[inputs[target].shard_name]
                digests = {
                    key: hashlib.sha256()
                    for key in ("packed", "scales", "source", "old_packed", "old_scales")
                }
                changed_indices, changed_packed, changed_scales = [], [], []

                def load(
                    start: int,
                    stop: int,
                    source_reader=source_reader,
                    target=target,
                    digests=digests,
                ):
                    value = _rows(source_reader, target, start, stop)
                    digests["source"].update(_bytes(value))
                    return value

                for row_start, row_stop, packed, scales in iter_quantized_row_chunks(
                    load,
                    partial(quantize_mxfp4, gamma=gamma, method="mse", mse_clip_depth=4),
                    rows=rows,
                    columns=columns,
                    max_rows=1024,
                    max_elements=DEFAULT_TENSOR_CHUNK_MAX_ELEMENTS,
                    name=target,
                ):
                    old_packed = _rows(parent_reader, packed_name, row_start, row_stop)
                    old_scales = _rows(parent_reader, scale_name, row_start, row_stop)
                    blocks, old_blocks = packed.reshape(-1, 16), old_packed.reshape(-1, 16)
                    mask = (scales.reshape(-1) != old_scales.reshape(-1)) | (
                        blocks != old_blocks
                    ).any(dim=1)
                    indices = mask.nonzero().reshape(-1)
                    if indices.numel():
                        changed_indices.append((indices + row_start * (columns // 32)).cpu())
                        changed_packed.append(blocks[indices].cpu())
                        changed_scales.append(scales.reshape(-1)[indices].cpu())
                    digests["packed"].update(_bytes(packed))
                    digests["scales"].update(_bytes(scales))
                    digests["old_packed"].update(_bytes(old_packed))
                    digests["old_scales"].update(_bytes(old_scales))
                    if row_start == 0:
                        n = min(16, row_stop)
                        weight = _rows(source_reader, target, 0, n).float()
                        old = dequant_mxfp4(old_packed[:n], old_scales[:n], (n, columns))
                        new = dequant_mxfp4(packed[:n], scales[:n], (n, columns))
                        before = float(((weight - old).square() * gamma.square()).sum())
                        after = float(((weight - new).square() * gamma.square()).sum())
                        if after > before * (1 + 1e-5) + 1e-12:
                            raise ValueError(f"Softened objective regressed: {target}")
                        sample_metrics[target] = {
                            "baseline_blended_error": before,
                            "soft_blended_error": after,
                            "baseline_unweighted_error": float((weight - old).square().sum()),
                            "soft_unweighted_error": float((weight - new).square().sum()),
                        }
                        del weight, old, new
                    del old_packed, old_scales, blocks, old_blocks, packed, scales
                for key, label in (("old_packed", packed_name), ("old_scales", scale_name)):
                    if digests[key].hexdigest() != parent["payload_sha256"][label]:
                        raise ValueError(f"Parent payload changed: {label}")
                source_hashes[target] = digests["source"].hexdigest()
                if changed_indices:
                    prefix = f"patch::{target}::"
                    patches[prefix + "indices"] = torch.cat(changed_indices).contiguous()
                    patches[prefix + "packed"] = torch.cat(changed_packed).contiguous()
                    patches[prefix + "scales"] = torch.cat(changed_scales).contiguous()
                    targets[target] = {
                        "file": str(destination),
                        "prefix": prefix,
                        "rows": rows,
                        "columns": columns,
                        "changed_blocks": patches[prefix + "indices"].numel(),
                        "packed_sha256": digests["packed"].hexdigest(),
                        "scale_sha256": digests["scales"].hexdigest(),
                    }
                elif (
                    digests["packed"].hexdigest() != parent["payload_sha256"][packed_name]
                    or digests["scales"].hexdigest() != parent["payload_sha256"][scale_name]
                ):
                    raise ValueError("A changed payload has no patch")
                del gamma
        temporary = destination.with_name(f".{destination.name}.incomplete")
        projected = sum(row["bytes"] for row in state["shards"].values()) + sum(
            value.numel() * value.element_size() for value in patches.values()
        )
        if projected > patch_limit_bytes or shutil.disk_usage(output).free < 3 * 1024**3:
            raise OSError("Patch trial exceeds storage budget or free-space reserve")
        save_file(
            patches, str(temporary), metadata={"parent": identity["baseline_manifest_sha256"]}
        )
        if (
            sum(row["bytes"] for row in state["shards"].values()) + temporary.stat().st_size
            > patch_limit_bytes
        ):
            temporary.unlink()
            raise OSError("Patch files exceed storage budget including headers")
        temporary.replace(destination)
        state["shards"][shard.path.name] = {
            "sha256": _sha(destination),
            "bytes": destination.stat().st_size,
            "targets": targets,
            "source_tensor_sha256": source_hashes,
            "sample_metrics": sample_metrics,
        }
        _write(state_path, state)
        print(
            f"Soft RMS shard {number}/{len(parent_shards)}: {len(targets)} changed projections, "
            f"{destination.stat().st_size / 1024**2:.1f} MiB patch",
            flush=True,
        )
    calibration.verify_unchanged(verify_sha256=True)
    effective = dict(parent["payload_sha256"])
    target_patches = {
        name: row for shard in state["shards"].values() for name, row in shard["targets"].items()
    }
    for target, row in target_patches.items():
        effective[target.removesuffix(".weight") + ".weight_packed"] = row["packed_sha256"]
        effective[target.removesuffix(".weight") + ".weight_scale"] = row["scale_sha256"]
    specification = {
        "format": "mxwave-kolibri-soft-rms-patch-v1",
        "baseline": str(baseline),
        **identity,
        "patches": target_patches,
        "effective_payload_sha256": effective,
        "patch_files": {
            str(output / f"delta-{number:05d}.safetensors"): {
                "sha256": state["shards"][shard.path.name]["sha256"],
                "bytes": state["shards"][shard.path.name]["bytes"],
            }
            for number, shard in enumerate(parent_shards, 1)
        },
        "scope": "Compact runtime patch; standalone weights have the parent's layout and size",
    }
    _write(output / "trial.json", specification)
    result = {
        "status": "complete",
        "seconds": time.monotonic() - started,
        "seconds_including_resume": time.time() - state["started_at"],
        "shards": len(state["shards"]),
        "covered_targets": len(covered),
        "changed_targets": len(target_patches),
        "changed_blocks": sum(row["changed_blocks"] for row in target_patches.values()),
        "patch_bytes": sum(row["bytes"] for row in state["shards"].values()),
        "standalone_weight_files_bytes": sum(
            row["bytes"] for row in parent["shard_integrity"]["per_shard"].values()
        ),
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if is_cuda else 0,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if is_cuda else 0,
        "trial_specification_sha256": _sha(output / "trial.json"),
        "recipe": identity["recipe"],
        "quality_qualification": "pending",
        "published": False,
    }
    _write(output / "conversion-result.json", result)
    print(json.dumps(result), flush=True)
    return result


def main() -> None:
    """Create bounded delta shards for one fully fixed softened RMS recipe."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "baseline", "statistics", "counts", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--strength", type=float, default=0.25)
    parser.add_argument("--pseudo-count", type=float, default=128.0)
    parser.add_argument("--max-variance-ratio", type=float, default=4.0)
    args = parser.parse_args()
    retune(
        args.source,
        args.baseline,
        args.statistics,
        args.counts,
        args.output,
        device=args.device,
        strength=args.strength,
        pseudo_count=args.pseudo_count,
        max_variance_ratio=args.max_variance_ratio,
    )


if __name__ == "__main__":
    main()
