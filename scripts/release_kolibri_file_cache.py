"""Advise release of clean Kolibri tensor-file cache without modifying checkpoint files."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def release_checkpoint_cache(root: Path) -> dict:
    """Release experiment-local clean cache on Linux and record unchanged file metadata."""
    root = root.resolve()
    paths = sorted({p.resolve() for p in root.rglob("*.safetensors") if p.is_file()})
    if any(not path.is_relative_to(root) for path in paths):
        raise ValueError("Cache release is restricted to experiment-local tensor files")
    before = Path("/proc/meminfo").read_text()
    records = []
    for path in paths:
        original = path.stat()
        with path.open("rb") as stream:
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        after = path.stat()
        if (original.st_size, original.st_mtime_ns, original.st_ctime_ns) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError(f"Checkpoint metadata changed during cache advice: {path}")
        records.append({"file": str(path.relative_to(root)), "bytes": original.st_size})
    result = {
        "status": "completed",
        "operation": "POSIX_FADV_DONTNEED on read-only checkpoint descriptors",
        "files_changed_or_removed": 0,
        "files_advised": len(paths),
        "files": records,
        "meminfo_before": before,
        "meminfo_after": Path("/proc/meminfo").read_text(),
    }
    (root / "fp8-cache-release.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    """Run the cache preflight inside the Linux diagnostic container."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = release_checkpoint_cache(args.root)
    print(json.dumps({key: value for key, value in result.items() if key != "files"}), flush=True)


if __name__ == "__main__":
    main()
