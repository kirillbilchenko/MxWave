"""Stage licensed native sources externally for an isolated FP32 partial-sum experiment."""

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
RETAINED_API_SHA = "51d24ec5f8fea750bd5cd683ebbb3bb4ec792d3a22d126f63fdd2e0351c43851"
BRIDGE_SHA = "273fc868b14a2a358728cb4764626bde67133adb0a51a426d434bc90512ac0e9"

# Independently authored reducer; the external MMA loop remains unchanged.
REDUCER = r'''
// MxWave FP32 partial-sum diagnostic, Apache-2.0.
__global__ void MxWaveReduceFP32(__nv_bfloat16* output, const float* partial,
                               int rows, int tokens, int splits) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = rows * tokens;
    if (index >= count) return;
    float value = 0.0f;
    for (int split = 0; split < splits; ++split)
        value += partial[split * count + index];
    output[index] = __float2bfloat16_rn(value);
}
'''


def replace_once(content: bytes, before: bytes, after: bytes) -> bytes:
    """Reject missing or ambiguous native adaptation points."""
    if content.count(before) != 1:
        raise ValueError(f"Ambiguous native adaptation point: {before[:80]!r}")
    return content.replace(before, after)


def original_sources(source: Path) -> dict[str, bytes]:
    """Accept only the original sources or the exact retained resource-query append."""
    result = {}
    for name, expected in SOURCE_SHA.items():
        data = (source / name).read_bytes()
        if name == "L_API.cu" and hashlib.sha256(data).hexdigest() == RETAINED_API_SHA:
            data = data[:data.index(b"\n#define MXWAVE_COMPACT_N8 0\n")]
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Pinned external source changed: {name}")
        result[name] = data
    return result


def patch_kernel(data: bytes) -> bytes:
    """Keep every MMA instruction and index; store each split's accumulator as FP32."""
    first = data.index(b"__global__ void BF16TripleBitmap_MM_Kernel(")
    last = data.index(b"__device__ __forceinline__ void VectorizedWriteToGlobalMemory", first)
    body = data[first:last]
    for before, after in (
        (b"BF16TripleBitmap_MM_Kernel(", b"BF16TripleBitmap_MM_FP32_Kernel("),
        (b"__nv_bfloat16* Output,", b"float* Output,"),
        (b"__nv_bfloat16* OutputPTR = Output", b"float* OutputPTR = Output"),
        (b"__float2bfloat16_rn((*(smem_CFrag + i))[j])", b"(*(smem_CFrag + i))[j]"),
    ):
        body = replace_once(body, before, after)
    return data[:first] + body + data[last:]


def patch_api(data: bytes) -> bytes:
    """Use a typed FP32 workspace and one final BF16 conversion in the separate API."""
    first = data.index(b"cudaError_t BF16TripleBitmap_MM_API(")
    last = data.index(b"cudaError_t BF16TripleBitmap_Decompress_API(", first)
    body = data[first:last]
    newline = b"\r\n" if b"\r\n" in body else b"\n"
    pointer = newline.join([b"    __nv_bfloat16* OutputPTR;", b"    if (Split_K == 1)",
                           b"        OutputPTR = C;", b"    else",
                           b"        OutputPTR = Reduction_Workspace;"])
    body = replace_once(body, pointer, b"    float* OutputPTR = Reduction_Workspace;")
    for before, after in (
        (b"BF16TripleBitmap_MM_API(", b"BF16TripleBitmap_MM_FP32_API("),
        (b"__nv_bfloat16* Reduction_Workspace,", b"float* Reduction_Workspace,"),
        (b"dim3 BlockDim(WARP_SIZE, 1, 1);", b"dim3 BlockDim(256, 1, 1);"),
        (b"SplitK_Reduction_BF16<<<", b"MxWaveReduceFP32<<<"),
    ):
        body = replace_once(body, before, after)
    if body.count(b"BF16TripleBitmap_MM_Kernel<") != 10:
        raise ValueError("Unexpected set of native launch/resource references")
    body = body.replace(b"BF16TripleBitmap_MM_Kernel<", b"BF16TripleBitmap_MM_FP32_Kernel<")
    return data[:first] + REDUCER.encode() + body + data[last:]


def patch_bridge(data: bytes) -> bytes:
    """Expose a new symbol with explicit workspace bytes; reject old ABI use and bad shapes."""
    data = data.replace(b"BF16TripleBitmap_MM_API", b"BF16TripleBitmap_MM_FP32_API")
    data = replace_once(data, b"int, int, int, __nv_bfloat16*, int);",
                        b"int, int, int, float*, int);")
    data = replace_once(data, b'extern "C" int mxzip_mm(', b'extern "C" int mxzip_mm_fp32(')
    data = replace_once(data, b"void* workspace, int split_k) {", b"void* workspace, int split_k,"
                        b" uint64_t workspace_bytes) {\n"
                        b"    if ((split_k != 2 && split_k != 4 && split_k != 8) ||\n"
                        b"        (n != 8 && n != 16) || (m != 4096 && m != 12608) ||\n"
                        b"        k != 4096 || !x || !y || !workspace ||\n"
                        b"        reinterpret_cast<uintptr_t>(workspace) % alignof(float) ||\n"
                        b"        workspace_bytes < uint64_t(m) * n * split_k * sizeof(float)) {\n"
                        b'        last_error = "Invalid FP32 split diagnostic shape/workspace";\n'
                        b"        return int(cudaErrorInvalidValue);\n"
                        b"    }")
    return replace_once(data, b"static_cast<__nv_bfloat16*>(workspace), split_k);",
                        b"static_cast<float*>(workspace), split_k);")


def main() -> None:
    """Build a same-compiler control and isolated candidate without requesting a GPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    originals = original_sources(args.source)
    bridge = args.bridge.read_bytes()
    if hashlib.sha256(bridge).hexdigest() != BRIDGE_SHA:
        raise ValueError("The qualified bridge changed")
    args.output.mkdir()
    report: dict[str, Any] = {"status": "building", "source_sha256": SOURCE_SHA,
        "bridge_sha256": BRIDGE_SHA, "builder_sha256": hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest(), "GPU_requested": False,
        "nvcc": subprocess.check_output(["nvcc", "--version"], text=True), "arms": []}
    manifest = args.output / "native-build.json"
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    try:
        for arm in ("control", "fp32"):
            stage = args.output / arm
            stage.mkdir()
            hashes = {}
            for name, data in originals.items():
                if arm == "fp32":
                    if name == "L_Kernel.cuh":
                        data = patch_kernel(data)
                    elif name == "L_API.cu":
                        data = patch_api(data)
                (stage / name).write_bytes(data)
                hashes[name] = hashlib.sha256(data).hexdigest()
            arm_bridge = patch_bridge(bridge) if arm == "fp32" else bridge
            (stage / "bridge.cu").write_bytes(arm_bridge)
            hashes["bridge.cu"] = hashlib.sha256(arm_bridge).hexdigest()
            output = args.output / f"libmxwave_split_{arm}.so"
            argv = ["nvcc", "-std=c++17", "-O3", "--shared", "-Xcompiler=-fPIC",
                "-gencode=arch=compute_121,code=sm_121", "-I", str(stage),
                str(stage / "L_API.cu"), str(stage / "bridge.cu"), "-o", str(output)]
            with (args.output / f"{arm}-compile.log").open("xb") as log:
                subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=240)
            report["arms"].append({"arm": arm, "command": argv, "source_sha256": hashes,
                "library": output.name, "library_sha256": hashlib.sha256(
                    output.read_bytes()).hexdigest()})
            manifest.write_text(json.dumps(report, indent=2) + "\n")
        report["status"] = "built_pending_gpu_qualification"
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        manifest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "arms": len(report["arms"])}))


if __name__ == "__main__":
    main()
