"""Build an external, hash-bound ZipServ N8 shared-memory candidate and matched control."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

SOURCE_SHA = {
    "utils.h": "0522460334d63c0548e11b2941b00e0b254cc0a11d87250fbedc3c038dc81830",
    "Reduction_Kernel.cuh": "4f649091c288eecf9464a01bf4a0d711b4d001b2ca347cf3ccba073bc9e7fabf",
    "MMA_PTX.cuh": "73fa705c9c96ed5093494b1c66569fbb1cccbdcdb2d516abdbe5086b8acd7ff2",
    "L_API.cu": "6e80669ba4cdae2a284f17712abac918a42d6e71071b4bb01cc3abf8b03b1353",
    "MatMulUtilities.cuh": "c324b1ab193ddef76f785804cfae6ff60185efedbd82f339d81def5e30015632",
    "L_Kernel.cuh": "0a08e93c88d0d2eeac3be0e8ed222b6384dbb608ca45d923be8cd761fa166239",
    "TilingConfig.h": "181fcb52c2e5c496b00f84eecec57ef0e322406ae6dcd92f825b97c7766461f6",
    "AsyncCopy_PTX.cuh": "0facc996c67607a7b6701e5f0939d9397094b0e015dfd7a5c1daa4bdbecc3a4c",
    "LICENSE": "1eb85fc97224598dad1852b5d6483bbcf0aa8608790dcc657a5a2a761ae9c8c6",
}
BRIDGE_SHA = "273fc868b14a2a358728cb4764626bde67133adb0a51a426d434bc90512ac0e9"
VARIANTS = {
    "control": (False, False, False),
    "compact": (True, True, False),
    "narrow": (False, True, False),
    "compact_x4": (True, False, False),
    "tailfirst": (False, False, True),
    "compact_tailfirst": (True, True, True),
}

# Original diagnostic ABI, appended identically to the control and candidate.
# It queries resource limits; no inference call uses this function.
QUERY = r'''
extern "C" int mxwave_n8_resources(int common, int rare, int* result) {
    using Config = TilingConfigBF16TripleBitmap<4, 1, 1, 1>;
    const int tokens = MXWAVE_COMPACT_N8 ? Config::TILE_N2 : Config::TILE_N;
    const int input = common * 2 + rare * 4 + tokens * TILE_K * 2 * 2
        + Config::TILE_BITMAP_M_V3 * Config::TILE_BITMAP_K_V3 * 8 * 6;
    const int output = (Config::TILE_M + PADDING_SHARED_MEM_FOR_C) * Config::TILE_N * 4;
    const int dynamic_bytes = input > output ? input : output;
    cudaFuncAttributes attrs{};
    auto status = cudaFuncGetAttributes(&attrs, BF16TripleBitmap_MM_Kernel<Config>);
    if (status != cudaSuccess) return int(status);
    int blocks = 0;
    status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, BF16TripleBitmap_MM_Kernel<Config>, Config::BLOCK_THREADS, dynamic_bytes);
    if (status != cudaSuccess) return int(status);
    result[0] = dynamic_bytes; result[1] = attrs.numRegs;
    result[2] = int(attrs.sharedSizeBytes); result[3] = blocks;
    result[4] = Config::BLOCK_THREADS; result[5] = MXWAVE_COMPACT_N8;
    return 0;
}
'''


def replace_once(content: bytes, before: bytes, after: bytes) -> bytes:
    """Fail closed if a pinned adaptation point is missing or ambiguous."""
    if content.count(before) != 1:
        raise ValueError(f"Ambiguous native adaptation point: {before[:90]!r}")
    return content.replace(before, after)


def patch_kernel(content: bytes) -> bytes:
    """Compact only the BF16 MMA activation buffers; leave decoder and arithmetic untouched."""
    first = content.index(b"__global__ void BF16TripleBitmap_MM_Kernel(")
    last = content.index(b"__device__ __forceinline__ void VectorizedWriteToGlobalMemory", first)
    body = content[first:last]
    for before, after in (
        (b"(TILE_K * TilingConfig::TILE_N * 2)", b"(TILE_K * TilingConfig::TILE_N2 * 2)"),
        (b"((tile_id_k + 1) % 2) * (TILE_K * TilingConfig::TILE_N)",
         b"((tile_id_k + 1) % 2) * (TILE_K * TilingConfig::TILE_N2)"),
        (b"((tile_id_k) % 2) * (TILE_K * TilingConfig::TILE_N)",
         b"((tile_id_k) % 2) * (TILE_K * TilingConfig::TILE_N2)"),
    ):
        body = replace_once(body, before, after)
    return content[:first] + body + content[last:]


def patch_mma(content: bytes, *, narrow: bool = True) -> bytes:
    """Load the two B fragments actually consumed by N8, with all lane addresses in bounds."""
    first = content.index(b"void B_FragLoadFromSharedToRegisters_BF16(")
    last = content.index(b"MMA_FP16_M16N8K16", first)
    body = content[first:last]
    body = replace_once(body, b"if (lane_id > 15)", b"if (lane_id > 15 && !N8)")
    if not narrow:
        # Upper lanes of x4 now address valid duplicates. Only the unchanged
        # lower two fragments are consumed by N8; N16+ keeps its original addresses.
        return content[:first] + body + content[last:]
    # Preserve the original x4 block verbatim for every non-N8 instantiation.
    begin = body.index(b'        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16')
    end = body.index(b'                     : "r"(smem_local_ptr));', begin)
    end += len(b'                     : "r"(smem_local_ptr));')
    original = body[begin:end]
    newline = b"\r\n" if b"\r\n" in body else b"\n"
    authored = newline.join([
        b"        if constexpr (N8) {",
        b'            asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\\n"',
        b'                         : "=r"(Registers[i][0]), "=r"(Registers[i][1])',
        b'                         : "r"(smem_local_ptr));',
        b"            Registers[i][2] = 0; Registers[i][3] = 0;",
        b"        } else {", original, b"        }",
    ])
    body = body[:begin] + authored + body[end:]
    return content[:first] + body + content[last:]


def input_row_tile(logical: int, rows: int, width: int, tokens: int, split: int) -> int:
    """Reference the specialized tail-first permutation without changing a row's K order."""
    if rows == 12608 and width == 4096 and 0 < tokens <= 8 and split == 1:
        return logical + 192 if logical < 5 else logical - 5
    return logical


def patch_schedule(content: bytes) -> bytes:
    """Launch the five exception-heavy input tiles first; leave every other geometry unchanged."""
    first = content.index(b"__global__ void BF16TripleBitmap_MM_Kernel(")
    last = content.index(b"__device__ __forceinline__ void VectorizedWriteToGlobalMemory", first)
    before = b"    const int y           = blockIdx.y % (M_Global / TilingConfig::TILE_M);"
    newline = b"\r\n" if b"\r\n" in content else b"\n"
    after = newline.join([
        b"    const int logical_y = blockIdx.y % (M_Global / TilingConfig::TILE_M);",
        b"    const int y = (TilingConfig::N8 && M_Global == 12608 && K_Global == 4096",
        b"                   && Split_K == 1)",
        b"        ? (logical_y < 5 ? logical_y + 192 : logical_y - 5) : logical_y;",
    ])
    return content[:first] + replace_once(content[first:last], before, after) + content[last:]


def patch_api(content: bytes) -> bytes:
    """Reduce the N8 launch allocation by the same 2 KiB removed from the buffer layout."""
    first, last = content.index(b"        case 8:"), content.index(b"        case 16:")
    body = replace_once(content[first:last],
        b"TilingConfigBF16TripleBitmap<4, 1, 1, 1>::TILE_N * TILE_K",
        b"TilingConfigBF16TripleBitmap<4, 1, 1, 1>::TILE_N2 * TILE_K")
    return content[:first] + body + content[last:]


def main() -> None:
    """Compile both arms with identical nvcc flags, preserving external licenses and source hashes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=["control", "compact"])
    args = parser.parse_args()
    if args.variants[0] != "control" or len(set(args.variants)) != len(args.variants):
        raise ValueError("Build one matched control first and distinct candidates")
    manifest = args.output / "native-build.json"
    if manifest.exists():
        raise FileExistsError("Preserve previous build provenance")
    originals = {name: (args.source / name).read_bytes() for name in SOURCE_SHA}
    for name, content in originals.items():
        if hashlib.sha256(content).hexdigest() != SOURCE_SHA[name]:
            raise ValueError(f"Pinned external source changed: {name}")
    bridge = args.bridge.read_bytes()
    if hashlib.sha256(bridge).hexdigest() != BRIDGE_SHA:
        raise ValueError("The qualified C ABI bridge changed")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "bridge.cu").write_bytes(bridge)
    report: dict[str, Any] = {"status": "building", "source_sha256": SOURCE_SHA,
        "bridge_sha256": BRIDGE_SHA,
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "nvcc": subprocess.check_output(["nvcc", "--version"], text=True), "arms": []}
    for arm in args.variants:
        compact, narrow, tail_first = VARIANTS[arm]
        stage = args.output / arm
        stage.mkdir()
        hashes = {}
        for name, content in originals.items():
            if name == "L_Kernel.cuh":
                if compact:
                    content = patch_kernel(content)
                if tail_first:
                    content = patch_schedule(content)
            elif name == "MMA_PTX.cuh" and (compact or narrow):
                content = patch_mma(content, narrow=narrow)
            elif name == "L_API.cu":
                if compact:
                    content = patch_api(content)
            if name == "L_API.cu":
                content += ("\n#define MXWAVE_COMPACT_N8 " + str(int(compact)) + "\n"
                            + QUERY).encode()
            (stage / name).write_bytes(content)
            hashes[name] = hashlib.sha256(content).hexdigest()
        output = args.output / ("libmxwave_n8_" + arm + ".so")
        command = ["nvcc", "-std=c++17", "-O3", "--shared", "-Xcompiler=-fPIC",
            "-gencode=arch=compute_121,code=sm_121", "-I", str(stage), str(stage / "L_API.cu"),
            str(args.output / "bridge.cu"), "-o", str(output)]
        subprocess.run(command, check=True)
        report["arms"].append({"arm": arm, "compact": compact, "narrow": narrow,
            "tail_first": tail_first, "source_sha256": hashes, "command": command,
            "library": output.name, "library_sha256": hashlib.sha256(output.read_bytes()).hexdigest()})
        manifest.write_text(json.dumps(report, indent=2) + "\n")
    report["status"] = "built_pending_gpu_qualification"
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "libraries": report["arms"]}))


if __name__ == "__main__":
    main()
