"""Build a separate decoder experiment from the pinned licensed external control source."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from build_bf16_n8 import BRIDGE_SHA, SOURCE_SHA

CONTROL_API_SHA = "51d24ec5f8fea750bd5cd683ebbb3bb4ec792d3a22d126f63fdd2e0351c43851"


def main() -> None:
    """Preserve source notices and build provenance without changing the qualified library."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve the first component build")
    expected = {**SOURCE_SHA, "L_API.cu": CONTROL_API_SHA}
    originals = {name: (args.source / name).read_bytes() for name in expected}
    for name, data in originals.items():
        if hashlib.sha256(data).hexdigest() != expected[name]:
            raise ValueError(f"The licensed control source changed: {name}")
    bridge = args.bridge.read_bytes()
    if hashlib.sha256(bridge).hexdigest() != BRIDGE_SHA:
        raise ValueError("Qualified C ABI bridge changed")
    header = Path(__file__).with_name("prefill_decode.cuh").read_bytes()
    args.output.mkdir(parents=True)
    stage = args.output / "source"
    stage.mkdir()
    for name, data in originals.items():
        if name == "L_API.cu":
            data += b'\n#include "prefill_decode.cuh"\n'
        (stage / name).write_bytes(data)
    (stage / "prefill_decode.cuh").write_bytes(header)
    (args.output / "bridge.cu").write_bytes(bridge)
    library = args.output / "libmxwave_prefill_decode.so"
    command = ["nvcc", "-std=c++17", "-O3", "--shared", "-Xcompiler=-fPIC",
        "-gencode=arch=compute_121,code=sm_121", "-I", str(stage), str(stage / "L_API.cu"),
        str(args.output / "bridge.cu"), "-o", str(library)]
    report = {"status": "building", "upstream_revision":
        "f69cea959005222bed64fc847bc3fcdcca32a904", "source_sha256": expected,
        "baseline": "Prior N8 control source; standalone decoder and GEMM unchanged",
        "bridge_sha256": BRIDGE_SHA, "header_sha256": hashlib.sha256(header).hexdigest(),
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "variants": {"0": "stock", "1": "direct_pair", "2": "row_coalesced"},
        "command": command, "nvcc": subprocess.check_output(["nvcc", "--version"], text=True)}
    manifest = args.output / "native-build.json"
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    subprocess.run(command, check=True)
    report.update(status="built_pending_gpu_qualification", library=library.name,
                  library_sha256=hashlib.sha256(library.read_bytes()).hexdigest())
    report["staged_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in stage.iterdir()}
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "library_sha256": report["library_sha256"]}))


if __name__ == "__main__":
    main()
