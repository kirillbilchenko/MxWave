"""Relocatable, hash-verified runtime assets; model weights remain separate."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

PROFILE = "glm53-ct-mxfp8-modal5-sm121-tp2-v1"
VLLM_VERSION = "0.32.0"
OFFICIAL_ARM64_IMAGE = (
    "vllm/vllm-openai@sha256:"
    "6ff5ff0ddf3658d2c2d677dac5ca2d4726ce0156b4371166d918181b210bd4bc"
)
PINNED_ASSETS = {
    "native/libmxwave_moe_raw.so":
        "48d9dafefed0d259134f23e7f91a30de40443d0cbb493b337667e6ce9a9d69d2",
    "native/libmxwave_moe_sparse.so":
        "9f5d96dcb0762d7e5ed5436ca296f70d590e73797d40522e4d8c7d6f3902f5f0",
    "native/libmxwave_kda_zipserv.so":
        "6d51630cd075a8c0dcbb957594341fb21e42d4b4634bb1c92735275d81a70f62",
    "rounding-reference.json":
        "b4c04d97bb24743bceba619ff67f26689837d17e81d4ca4329677bfc119839d7",
    "vocab/prepared.json":
        "06731dc5cb5d00fb5edcede73038006b4689f35fc1195a0dbc8bae5557588b6d",
}
MODEL_FILES = {
    "config.json": "194d290ebb0f19aedc60d24ea9e236e33c86616e460f4a108a842f6fd5b8b198",
    "model.safetensors.index.json":
        "6d1364a27eef78675390422b1b5230601817fb7460ce816951b2cd515934c85d",
}
DRAFTER_FILES = {
    "config.json": "c54c63c667baca52604519c5ac0d0d053210e3a3231dfb626d4758b1b6f8655e",
    "model.safetensors": "45f34ef126aa258d8e16bd605a3a05c36eb3fef3ed0f57aef20191db49f1ef46",
}


def digest(path: Path) -> str:
    """Hash a file without reading its complete payload into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def local_file(root: Path, relative: str) -> Path:
    """Reject absolute paths, traversal and symlinks escaping the bundle."""
    name = Path(relative)
    if name.is_absolute() or not name.parts or any(x in ("..", ".") for x in name.parts):
        raise ValueError(f"Invalid bundle-relative path: {relative}")
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing or nonlocal bundle file: {relative}")
    return path


def verify_files(root: Path, files: dict[str, str]) -> None:
    """Require every named payload to match its expected SHA256."""
    for name, expected in files.items():
        if digest(local_file(root, name)) != expected:
            raise ValueError(f"SHA256 mismatch: {name}")


def expected_assets(root: Path) -> dict[str, str]:
    """Bind all four vocabulary shards through the pinned preparation manifest."""
    verify_files(root, PINNED_ASSETS)
    prepared = json.loads((root / "vocab/prepared.json").read_bytes())
    cases = prepared["cases"]
    if (len(cases) != 4 or {(r["kind"], r["rank"]) for r in cases}
            != {(kind, rank) for kind in ("head", "embedding") for rank in (0, 1)}):
        raise ValueError("Require both head and embedding shards on both TP ranks")
    result = dict(PINNED_ASSETS)
    for license_file in Path(__file__).with_name("licenses").iterdir():
        if license_file.is_file():
            result["licenses/" + license_file.name] = digest(license_file)
    for row in cases:
        name = row["artifact"]
        if Path(name).name != name:
            raise ValueError("Vocabulary artifact must be a filename")
        result["vocab/" + name] = row["artifact_sha256"]
    return result


def verify_bundle(root: Path) -> dict[str, Any]:
    """Verify a complete bundle before loading native code or creating a server."""
    manifest = json.loads(local_file(root, "bundle.json").read_bytes())
    if (manifest.get("schema") != 1 or manifest.get("profile") != PROFILE
            or manifest.get("vllm") != VLLM_VERSION):
        raise ValueError("Unsupported runtime bundle profile or vLLM version")
    expected = expected_assets(root)
    if manifest.get("files") != expected:
        raise ValueError("Bundle inventory differs from the pinned runtime assets")
    verify_files(root, expected)
    return manifest


def prepare_bundle(experiment: Path, output: Path) -> dict[str, Any]:
    """Export verified assets once; the result has no experiment-directory dependencies."""
    if output.exists():
        raise FileExistsError(f"Preserve existing bundle: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    mapping = {
        "native/libmxwave_moe_raw.so": "native/libmxwave_moe_raw.so",
        "native/libmxwave_moe_sparse.so": "quality-best-v1-native/libmxwave_moe_sparse.so",
        "native/libmxwave_kda_zipserv.so": "native/libmxwave_kda_zipserv.so",
        "rounding-reference.json": "ct-draft-length-v1/rounding-reference.json",
        "vocab/prepared.json": "vocab-v2/prepared.json",
    }
    with tempfile.TemporaryDirectory(prefix=".mxwave-bundle-", dir=output.parent) as tmp:
        staged = Path(tmp) / "bundle"
        staged.mkdir()
        for target, source in mapping.items():
            dest = staged / target
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(experiment / source, dest)
        expected = expected_assets(staged)
        for name in expected.keys() - mapping.keys():
            dest = staged / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            source = (Path(__file__).with_name("licenses") / Path(name).name
                      if name.startswith("licenses/") else
                      experiment / "vocab-v2" / Path(name).name)
            shutil.copyfile(source, dest)
        manifest = {
            "schema": 1, "profile": PROFILE, "vllm": VLLM_VERSION,
            "platform": "linux-aarch64", "compute_capability": [12, 1],
            "tensor_parallel_size": 2, "files": expected,
            "quality": "MXFP8 KDA is lossy; original 1e-5 KL gate failed",
            "qualification": "upstream_0.32_gpu_qualification_pending",
        }
        (staged / "bundle.json").write_text(json.dumps(manifest, indent=2) + "\n")
        verify_bundle(staged)
        staged.rename(output)
    return manifest
