"""Install release assets or reproduce their packed tensors from public checkpoints."""

from __future__ import annotations

import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from .bundle import MODEL_FILES, PROFILE, VLLM_VERSION, digest, verify_bundle, verify_files


def checkpoint_contract() -> dict[str, Any]:
    """Return the pinned public checkpoint origins and complete target shard hashes."""
    return json.loads(Path(__file__).with_name("checkpoints.json").read_bytes())


def verify_model(model: Path) -> dict[str, Any]:
    """Verify every target shard once before serving, with bounded hashing memory."""
    verify_files(model, MODEL_FILES)
    contract = checkpoint_contract()["model"]
    shards = {s["name"]: s["sha256"] for s in contract["shards"]}
    verify_files(model, shards)
    return {"status": "passed", "repository": contract["repository"],
            "revision": contract["revision"], "shards_verified": len(shards)}


def install_bundle(archive: Path, output: Path, expected_sha256: str) -> dict[str, Any]:
    """Install a verified release archive atomically, rejecting links and traversal."""
    if output.exists():
        raise FileExistsError(f"Preserve existing bundle: {output}")
    if digest(archive) != expected_sha256:
        raise ValueError("Release archive SHA256 mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".mxwave-install-", dir=output.parent) as tmp:
        staged = Path(tmp) / "runtime"
        staged.mkdir()
        with tarfile.open(archive, "r:*") as stream:
            seen: set[str] = set()
            total = 0
            for member in stream:
                name = PurePosixPath(member.name)
                if (name.is_absolute() or ".." in name.parts or not name.parts
                        or str(name) in seen or len(seen) >= 64
                        or not (member.isfile() or member.isdir()) or member.sparse):
                    raise ValueError(f"Unsafe release archive member: {member.name}")
                seen.add(str(name))
                total += member.size
                if member.size > 1024**3 or total > 3 * 1024**3:
                    raise ValueError("Release archive exceeds the bounded runtime inventory")
                target = staged.joinpath(*name.parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                source = stream.extractfile(member)
                if source is None:
                    raise ValueError("Missing archive member payload")
                with source, target.open("xb") as dest:
                    shutil.copyfileobj(source, dest, length=1024**2)
        result = verify_bundle(staged)
        actual = {str(p.relative_to(staged)) for p in staged.rglob("*") if p.is_file()}
        if actual != set(result["files"]) | {"bundle.json"}:
            raise ValueError("Unexpected files in release archive")
        if output.exists():
            raise FileExistsError(f"Preserve existing bundle: {output}")
        staged.rename(output)
    return {"status": "passed", "profile": PROFILE, "archive_sha256": expected_sha256,
            "files_verified": len(result["files"])}


def prepare_pretrained_drafter(source: Path, output: Path) -> dict[str, Any]:
    """Reproduce the measured FP8 drafter and apply its config-only vLLM correction."""
    from .drafter import prepare_drafter
    from .drafter_quantize import convert

    if output.exists():
        raise FileExistsError(f"Preserve existing drafter: {output}")
    contract = checkpoint_contract()["drafter_bf16"]
    verify_files(source, contract["files"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".mxwave-fp8-", dir=output.parent) as tmp:
        converted = Path(tmp) / "fp8"
        review = convert(source, converted, contract["files"]["model.safetensors"])
        for name in ("LICENSE", "README.md"):
            if (source / name).is_file():
                shutil.copyfile(source / name, converted / name)
        receipt = prepare_drafter(converted, output)
    receipt.update(source_repository=contract["repository"], source_revision=contract["revision"],
                   source_precision="BF16", backbone_conversion="lossy block FP8, 128x128",
                   quantized_tensors=review["quantized_tensors"], training_performed=False)
    (output / "mxwave-drafter.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def build_bundle(model: Path, native: Path, output: Path) -> dict[str, Any]:
    """Reproduce the pinned vocabulary artifacts using CPU packing and supplied native binaries."""
    import hashlib

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    from mxwave.lossless_bf16_rows import pack_rows, write_to

    from ._runtime import head
    from .bundle import PINNED_ASSETS, expected_assets

    if output.exists():
        raise FileExistsError(f"Preserve existing bundle: {output}")
    verify_files(model, MODEL_FILES)
    index = json.loads((model / "model.safetensors.index.json").read_bytes())["weight_map"]
    output.parent.mkdir(parents=True, exist_ok=True)
    package = Path(__file__).parent
    with tempfile.TemporaryDirectory(prefix=".mxwave-build-", dir=output.parent) as tmp:
        staged = Path(tmp) / "runtime"
        (staged / "native").mkdir(parents=True)
        (staged / "vocab").mkdir()
        for name in PINNED_ASSETS:
            if name.startswith("native/"):
                shutil.copyfile(native / Path(name).name, staged / name)
        shutil.copyfile(package / "metadata/rounding-reference.json", staged / "rounding-reference.json")
        shutil.copyfile(package / "metadata/vocab-prepared.json", staged / "vocab/prepared.json")
        shutil.copytree(package / "licenses", staged / "licenses")
        expected = expected_assets(staged)
        prepared = json.loads((staged / "vocab/prepared.json").read_bytes())
        old_library = os.environ.get("MXWAVE_ZIPSERV_LIBRARY")
        os.environ["MXWAVE_ZIPSERV_LIBRARY"] = str(staged / "native/libmxwave_kda_zipserv.so")
        head._library.cache_clear()
        torch.set_num_threads(2)
        try:
            for row in prepared["cases"]:
                rows, width = row["shape"]
                shard = index[row["source_key"]]
                if Path(shard).name != shard:
                    raise ValueError("Require a local checkpoint shard filename")
                with safe_open(model / shard, framework="pt", device="cpu") as handle:
                    weight = handle.get_slice(row["source_key"])[
                        row["rank"] * rows:(row["rank"] + 1) * rows].contiguous()
                if weight.dtype != torch.bfloat16 or list(weight.shape) != [rows, width]:
                    raise ValueError("Vocabulary source shape or dtype changed")
                source_sha = hashlib.sha256(weight.view(torch.uint8).numpy()).hexdigest()
                if source_sha != row["source_sha256"]:
                    raise ValueError("Vocabulary source values differ from the audited checkpoint")
                path = staged / "vocab" / row["artifact"]
                if row["kind"] == "head":
                    packed = head.compress(weight)
                    save_file(dict(zip(head.BUFFER_NAMES, packed.buffers, strict=True)), path)
                else:
                    packed = pack_rows(weight.view(torch.uint16).numpy(),
                                       source_sha256=source_sha, chunk_rows=128)
                    write_to(packed, path)
                if digest(path) != row["artifact_sha256"]:
                    raise ValueError("Rebuilt vocabulary artifact differs from the GPU-tested bytes")
                del packed, weight
        finally:
            head._library.cache_clear()
            if old_library is None:
                os.environ.pop("MXWAVE_ZIPSERV_LIBRARY", None)
            else:
                os.environ["MXWAVE_ZIPSERV_LIBRARY"] = old_library
        manifest = {"schema": 1, "profile": PROFILE, "vllm": VLLM_VERSION,
                    "platform": "linux-aarch64", "compute_capability": [12, 1],
                    "tensor_parallel_size": 2, "files": expected,
                    "quality": "MXFP8 KDA is lossy; original 1e-5 KL gate failed",
                    "qualification": "GPU screen complete; quality/repeatability gates not passed"}
        (staged / "bundle.json").write_text(json.dumps(manifest, indent=2) + "\n")
        verify_bundle(staged)
        staged.rename(output)
    return manifest
