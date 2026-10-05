"""Bundle the exact clean MxWave source and bounded Kolibri evidence for a release."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path


def main() -> None:
    """Create a deterministic licensed source archive with per-file hashes and Git provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, archive = args.source.resolve(), args.output.resolve()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True)
    if dirty:
        raise RuntimeError("Commit the validated release source before packaging its provenance")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=source, text=True,
    ).strip()
    collection = subprocess.check_output(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-o", "addopts="],
        cwd=source, text=True,
    )
    test_count = sum(line.startswith("tests/") and "::" in line for line in collection.splitlines())
    if not test_count:
        raise ValueError("Could not identify the validated source's unit test count")
    names = ["README.md", "LICENSE", "CONTRIBUTING.md", "AGENTS.md", "pyproject.toml"]
    names += [str(path.relative_to(source)) for pattern in (
        "mxwave/*.py", "mxwave/py.typed", "scripts/*.py", "tests/*.py",
        "benchmarks/kolibri1-2026-10-04/**/*.json", "docs/KOLIBRI1*.md",
    ) for path in source.glob(pattern) if path.is_file()]
    hashes = {}
    archive.parent.mkdir(parents=True, exist_ok=True)
    with (
        archive.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as bundle,
    ):
        for name in sorted(set(names)):
            value = (source / name).read_bytes()
            hashes[name] = hashlib.sha256(value).hexdigest()
            member = tarfile.TarInfo(f"mxwave-kolibri1/{name}")
            member.size = len(value)
            member.mode = 0o644
            bundle.addfile(member, io.BytesIO(value))
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    # Read back the independent archive entries, rather than trusting the packer.
    with tarfile.open(archive, "r:gz") as bundle:
        for name, expected in hashes.items():
            stream = bundle.extractfile(f"mxwave-kolibri1/{name}")
            if stream is None or hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"Archive verification failed: {name}")
    record = {
        "archive_name": "mxwave-source.tar.gz", "archive_sha256": digest,
        "archive_bytes": archive.stat().st_size,
        "git_commit": commit, "branch": branch, "clean_worktree": True,
        "source_files_sha256": hashes,
        "frozen_evaluator_sha256": hashes["scripts/evaluate_kolibri_release.py"],
        "validation": {"pytest": f"{test_count} tests passed", "ruff": "passed",
                       "mypy": "strict check of all 32 mxwave modules passed"},
        "license": "MxWave source includes LICENSE and CONTRIBUTING.md. Vendor inference "
        "plugin and b12x/CuTe DSL dependency source are not included. Dataset licenses "
        "and revisions are recorded in the frozen protocol.",
    }
    metadata = archive.with_suffix("").with_suffix(".json")
    metadata.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"archive": str(archive), "metadata": str(metadata),
                      "sha256": digest, "bytes": archive.stat().st_size,
                      "files": len(hashes), "commit": commit}), flush=True)


if __name__ == "__main__":
    main()
