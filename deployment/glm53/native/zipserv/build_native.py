"""Build a Torch-ABI-independent bridge against a pinned external ZipServ checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

REVISION = "f69cea959005222bed64fc847bc3fcdcca32a904"


def main() -> None:
    """Stage permitted external adaptations, retain licenses, and invoke nvcc."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--arch", default="121", choices=["80", "86", "89", "90", "100", "121"])
    parser.add_argument(
        "--n128-warps",
        type=int,
        choices=[4, 8],
        help="Experimental 128-token tile with 4 or 8 warps; omitted uses upstream tiling",
    )
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    # Require a real pinned checkout, not an unverifiable source directory.
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise ValueError(f"Expected ZipServ {REVISION}, got {revision}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / "patched-zipserv"
    stage.mkdir(exist_ok=True)
    provenance: dict[str, str] = {}
    for path in (source / "csrc").iterdir():
        if path.suffix not in (".cu", ".cuh", ".h"):
            continue
        content = path.read_bytes()
        provenance[path.name] = hashlib.sha256(content).hexdigest()
        if path.name == "L_API.cu":
            if content.count(b"static int SHMEM_SZ") != 4:
                raise ValueError("Unexpected ZipServ shared-memory declarations")
            content = content.replace(b"static int SHMEM_SZ", b"int SHMEM_SZ")
            if args.n128_warps is not None:
                # Reuse the licensed upstream launch block, changing its tile
                # parameters only. Keep all other batch sizes on their stock path.
                text = content.decode()
                newline = "\r\n" if "\r\n" in text else "\n"
                start = text.index("        case 64:")
                end = text.index("    " + newline + "    cudaError_t Error", start)
                launch = text[start:end]
                if (
                    launch.count("case 128:") != 1
                    or launch.count("TilingConfigBF16TripleBitmap<4, 1, 4>") != 10
                ):
                    raise ValueError("Unexpected upstream large-batch launch block")
                body = launch[launch.index("            {") : launch.rindex("    }")]
                tile = "<4, 1, 8>" if args.n128_warps == 4 else "<4, 2, 4>"
                body = body.replace("<4, 1, 4>", tile)
                wide = "        case 128:" + newline + body
                stock = launch.replace("        case 128:" + newline, "")
                text = text[:start] + wide + stock + text[end:]
                content = text.encode()
        (stage / path.name).write_bytes(content)
    helper = source / "kernel_benchmark/utils.h"
    content = helper.read_bytes()
    provenance["utils.h"] = hashlib.sha256(content).hexdigest()
    content = content.replace(b"#include <cusparse_v2.h>\r\n", b"")
    content = content.replace(b"#include <cusparse_v2.h>\n", b"")
    (stage / "utils.h").write_bytes(content)
    (stage / "LICENSE").write_bytes((source / "LICENSE").read_bytes())
    bridge = Path(__file__).parent / "native/bridge.cu"
    command = [
        "nvcc",
        "-std=c++17",
        "-O3",
        "--shared",
        "-Xcompiler=-fPIC",
        f"-gencode=arch=compute_{args.arch},code=sm_{args.arch}",
        "-I",
        str(stage),
        str(stage / "L_API.cu"),
        str(bridge),
        "-o",
        str(output),
    ]
    subprocess.run(command, check=True)
    (output.parent / "native-build.json").write_text(
        json.dumps(
            {
                "zipserv_revision": revision,
                "arch": args.arch,
                "command": command,
                "upstream_sha256": provenance,
                "bridge_sha256": hashlib.sha256(bridge.read_bytes()).hexdigest(),
                "library_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "changes": ["per-call shared-memory size", "remove unused legacy cuSPARSE include"],
                "experimental_n128_warps": args.n128_warps,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
