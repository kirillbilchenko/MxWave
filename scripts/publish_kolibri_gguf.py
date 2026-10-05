"""Publish the measured eight-bit-activation GGUF as an explicitly experimental release."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

REPOSITORY = "kirillbilchenko/Kolibri-1-MXFP4-GGUF-MxWave"
MODEL_SHA = "4e41cf583374a1ed22c4ec921e92a07f91bb4a54209a11a42396d7bd43ac6f81"
MANIFEST_SHA = "4d7b37059099d1306c30e1d20723d339e7b3121525ec541bfcd3e6688cb05a11"
PROTOCOL_SHA = "9561c8d0be435b53152d6d91536d06a670a33e70c7ed606c87779702f84ebc22"
PATCH_SHA = "2e629b80a55880dd8b1e40bc3bda51c1b9b7b38985439b47c3753a30eecd7cb7"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(root: Path) -> dict[str, Any]:
    """Bind the exact artifact, executed source, tokenizer checks, and measured profile."""
    gguf = root / "gguf"
    receipt = _read(gguf / "Kolibri-1-MXFP4.json")
    quality_path = gguf / "qualification-q8/quality-comparison.json"
    quality = _read(quality_path)
    measurement = _read(gguf / "qualification-q8/measurements/mse.json")
    protocol = root / "validation/protocol.json"
    model = gguf / "Kolibri-1-MXFP4.gguf"
    if (
        receipt["status"] != "complete"
        or receipt["verified_source_payloads"] != 116303
        or receipt["verified_gguf_tensors"] != 903
        or receipt["source_manifest_sha256"] != MANIFEST_SHA
        or receipt["output_sha256"] != MODEL_SHA
        or model.stat().st_size != receipt["output_bytes"]
        or _sha(model) != MODEL_SHA
        or _sha(protocol) != PROTOCOL_SHA
        or quality["protocol_sha256"] != PROTOCOL_SHA
        or quality["gguf_sha256"] != MODEL_SHA
        or quality["runtime_mmq_precision"] != "q8"
        or measurement["runtime_mmq_precision"] != "q8"
        or quality["gate"] != _read(protocol)["quality_gate"]
        or not math.isfinite(quality["mean_next_token_kl"])
    ):
        raise ValueError("Release identity or frozen measurement profile is inconsistent")
    if len(quality["behavior"]["mse"]) != 6 or not all(
        row["passed"] for row in quality["behavior"]["mse"]
    ):
        raise ValueError("The selected GGUF did not pass every behavior check")
    gate = quality["gate"]
    if quality["domains"]["all"]["delta_mean_nll"] > gate["max_mean_delta_nll"] or any(
        quality["domains"][domain]["delta_mean_nll"] > gate["max_domain_delta_nll"]
        for domain in ("en", "de", "code")
    ):
        raise ValueError("Likelihood degradation exceeds the frozen experimental release bounds")
    if _sha(gguf / "build/bin/kolibri-metrics") != measurement["harness_binary_sha256"]:
        raise ValueError("Measured harness binary changed")
    tokenizer = _read(gguf / "tokenizer-verification.json")
    if tokenizer["status"] != "passed" or len(tokenizer["cases"]) != 14:
        raise ValueError("Tokenizer interoperability verification is incomplete")
    source = _read(gguf / "source-provenance.json")
    if _sha(gguf / "mxwave-source.tar.gz") != source["archive_sha256"]:
        raise ValueError("Executed-source archive changed")
    if _sha(gguf / "runtime-source/runtime-kolibri1-runtime.patch") != PATCH_SHA:
        raise ValueError("Pinned external runtime patch changed")
    card = (gguf / "MODEL_CARD.md").read_text()
    if not all(text in card for text in ("experimental", "0.061844", "0.030", "MMQ_PREC=q8")):
        raise ValueError("Model card must disclose the measured missed target and required setting")
    output = gguf / "release"
    output.mkdir(exist_ok=True)
    target = output / model.name
    if target.exists():
        if not target.samefile(model):
            raise ValueError("Release folder contains a foreign model artifact")
    else:
        os.link(model, target)
    mapping = {
        "README.md": gguf / "MODEL_CARD.md",
        "LICENSE": root / "checkpoint-soft-rms/LICENSE",
        "NOTICE": root / "checkpoint-soft-rms/NOTICE",
        "conversion.json": gguf / "Kolibri-1-MXFP4.json",
        "mxwave-source.tar.gz": gguf / "mxwave-source.tar.gz",
        "source-provenance.json": gguf / "source-provenance.json",
        "validation/protocol.json": protocol,
        "validation/quality-q8.json": quality_path,
        "validation/measurements-q8.json": gguf / "qualification-q8/measurements/mse.json",
        "validation/quality-default-q4.json": gguf / "qualification/quality-comparison.json",
        "validation/tokenizer-verification.json": gguf / "tokenizer-verification.json",
        "runtime-source/README.md": gguf / "RUNTIME_README.md",
        "runtime-source/provenance.json": gguf / "runtime-provenance.json",
    }
    for name in (
        "kolibri1-runtime.patch",
        "LICENSE-MIT.txt",
        "LICENSE-Apache-2.0.txt",
        "THIRD_PARTY_NOTICES.txt",
    ):
        mapping["runtime-source/" + name] = gguf / "runtime-source" / ("runtime-" + name)
    for name, path in mapping.items():
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    qualification = {
        "experimental": True,
        "strict_screen_status": quality["status"],
        "runtime_mmq_precision": "q8",
        "model_sha256": MODEL_SHA,
        "source_manifest_sha256": MANIFEST_SHA,
        "protocol_sha256": PROTOCOL_SHA,
        "quality_sha256": _sha(quality_path),
        "source_archive_sha256": source["archive_sha256"],
        "source_revision": source["revision"],
        "mean_next_token_kl": quality["mean_next_token_kl"],
        "ppl": quality["domains"]["all"]["candidate_ppl"],
    }
    _write(output / "qualification.json", qualification)
    names = {model.name, "qualification.json", *mapping}
    files: dict[str, Any] = {}
    for name in sorted(names):
        path = output / name
        digest = MODEL_SHA if name == model.name else _sha(path)
        blob = None
        if path.stat().st_size < 32 * 1024**2:
            raw = path.read_bytes()
            blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        files[name] = {"bytes": path.stat().st_size, "sha256": digest, "git_blob_sha1": blob}
    prepared = {"status": "prepared", "repository": REPOSITORY, **qualification, "files": files}
    _write(gguf / "publication-prepared.json", prepared)
    return prepared


def _verify(api: Any, prepared: dict[str, Any], *, private: bool) -> str:
    info = api.model_info(REPOSITORY, files_metadata=True)
    if info.private != private:
        raise ValueError("Unexpected repository visibility")
    remote = {item.rfilename: item for item in info.siblings}
    if set(prepared["files"]) - set(remote):
        raise ValueError("Repository is missing prepared GGUF release files")
    if set(remote) - set(prepared["files"]) - {".gitattributes"}:
        raise ValueError("Repository contains files outside the prepared GGUF release")
    for name, expected in prepared["files"].items():
        item = remote[name]
        digest = item.lfs.sha256 if item.lfs is not None else item.blob_id
        target = expected["sha256"] if item.lfs is not None else expected["git_blob_sha1"]
        if item.size != expected["bytes"] or digest != target:
            raise ValueError(f"Remote file differs from the prepared release: {name}")
    return str(info.sha)


def release(root: Path) -> dict[str, Any]:
    """Upload privately, verify every file, publish, then verify anonymously again."""
    from huggingface_hub import HfApi

    gguf = root / "gguf"
    prepared = _read(gguf / "publication-prepared.json")
    if (
        prepared.get("repository") != REPOSITORY
        or prepared.get("model_sha256") != MODEL_SHA
        or prepared.get("source_manifest_sha256") != MANIFEST_SHA
        or prepared.get("protocol_sha256") != PROTOCOL_SHA
        or prepared.get("runtime_mmq_precision") != "q8"
        or prepared.get("experimental") is not True
    ):
        raise ValueError("Prepared release differs from the selected experimental GGUF")
    api = HfApi()
    if api.whoami()["name"] != REPOSITORY.split("/")[0]:
        raise ValueError("Authenticated account differs from the release namespace")
    intent = gguf / "upload-intent.json"
    identity = {
        "repository": REPOSITORY,
        "prepared_sha256": _sha(gguf / "publication-prepared.json"),
    }
    if intent.exists():
        old = _read(intent)
        if any(old[key] != value for key, value in identity.items()):
            raise ValueError("Existing upload belongs to a different prepared release")
    elif api.repo_exists(REPOSITORY):
        raise ValueError("Existing repository has no matching local upload intent")
    else:
        _write(intent, {**identity, "status": "planned"})
    already_public = api.repo_exists(REPOSITORY) and not api.model_info(REPOSITORY).private
    if already_public:
        revision = _verify(HfApi(token=False), prepared, private=False)
    else:
        api.create_repo(REPOSITORY, repo_type="model", private=True, exist_ok=True)
        _write(intent, {**identity, "status": "uploading-private"})
        api.upload_large_folder(
            REPOSITORY,
            repo_type="model",
            folder_path=gguf / "release",
            allow_patterns=sorted(prepared["files"]),
            ignore_patterns=[".cache/**", "*.incomplete"],
            num_workers=4,
        )
        revision = _verify(api, prepared, private=True)
        _write(intent, {**identity, "status": "verified-private", "revision": revision})
        if api.model_info(REPOSITORY).sha != revision:
            raise ValueError("Repository changed after private verification")
        api.update_repo_settings(REPOSITORY, private=False)
    if _verify(HfApi(token=False), prepared, private=False) != revision:
        raise ValueError("Public revision differs from the verified private revision")
    result = {
        **identity,
        "status": "published",
        "revision": revision,
        "url": "https://huggingface.co/" + REPOSITORY,
        "model_sha256": MODEL_SHA,
        "experimental": True,
        "strict_screen_status": prepared["strict_screen_status"],
        "verified_files": len(prepared["files"]),
        "runtime_mmq_precision": "q8",
    }
    _write(gguf / "publication-result.json", result)
    return result


def main() -> None:
    """Run one explicit preparation or upload/publication phase."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "release"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = {"prepare": prepare, "release": release}[args.phase](args.root.resolve())
    print(json.dumps({key: value for key, value in result.items() if key != "files"}), flush=True)


if __name__ == "__main__":
    main()
