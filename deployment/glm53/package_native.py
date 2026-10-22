"""Assemble the pinned native release archive with notices and build sources."""

from __future__ import annotations

import argparse
import gzip
import io
import json
import tarfile
from pathlib import Path

from mxwave_glm53.bundle import PINNED_ASSETS, digest, verify_files


def package(native: Path, output: Path, source: Path) -> None:
    """Create a reproducible archive only from the three qualified native binaries."""
    binaries = {Path(k).name: v for k, v in PINNED_ASSETS.items() if k.startswith("native/")}
    verify_files(native, binaries)
    if output.exists():
        raise FileExistsError(output)
    files = {"native/" + name: native / name for name in binaries}
    for p in (source / "mxwave_glm53/licenses").iterdir():
        if p.is_file():
            files["licenses/" + p.name] = p
    for name in ("build_moe.py", "moe_bridge.cu", "moe_scale_decode.cuh", "moe_sources.json"):
        relative = "integrations/glm53_mxfp4/" + name
        files["source/" + relative] = source / relative
    for relative in ("deployment/glm53/native/README.md",
                     "deployment/glm53/native/zipserv/build_native.py",
                     "deployment/glm53/native/zipserv/native/bridge.cu", "CONTRIBUTING.md",
                     "LICENSE"):
        files["source/" + relative] = source / relative
    hashes = {name: digest(path) for name, path in files.items()}
    manifest = json.dumps({"schema": 1, "release": "glm53-v0.1.1",
                           "platform": "linux-aarch64-sm121", "sha256": hashes},
                          indent=2, sort_keys=True).encode() + b"\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.open("xb") as dest,
          gzip.GzipFile(filename="", fileobj=dest, mode="wb", mtime=0) as gz,
          tarfile.open(fileobj=gz, mode="w") as archive):
        for name in sorted(files):
            info = tarfile.TarInfo(name)
            info.size = files[name].stat().st_size
            info.mode = 0o644
            with files[name].open("rb") as payload:
                archive.addfile(info, payload)
        info = tarfile.TarInfo("native-manifest.json")
        info.size = len(manifest)
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(manifest))
    print(json.dumps({"archive": output.name, "sha256": digest(output),
                      "bytes": output.stat().st_size, "files": len(files) + 1}))


def main() -> None:
    """Build a release archive from a verified native directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package(args.native, args.output, Path(__file__).resolve().parents[2])


if __name__ == "__main__":
    main()
