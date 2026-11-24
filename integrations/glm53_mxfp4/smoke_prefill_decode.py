"""Qualify decoder bits, address bounds and captured writes before any timing fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
from prefill_decode import MODES, NativeDecoder
from prefill_pack_oracle import synthetic_matrices, unpack_words


def verify_words(output: torch.Tensor, expected: torch.Tensor, arena: torch.Tensor) -> None:
    """Require complete byte equality and untouched prefix/suffix guard words."""
    if (not torch.equal(output.view(torch.uint8), expected.view(torch.uint8))
            or not bool(torch.all(arena[:64] == 19))
            or not bool(torch.all(arena[-64:] == 19))):
        raise RuntimeError("A decoder changed words or crossed its destination bounds")


def main() -> None:
    """Test all decoder modes on synthetic words without loading model weights."""
    from mxwave_zipserv import runtime

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Preserve first smoke-test evidence")
    report: dict[str, Any] = {"status": "running", "model_weights_loaded": False, "cases": [],
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "qualified_library_sha256": hashlib.sha256(
            Path(os.environ["MXWAVE_ZIPSERV_LIBRARY"]).read_bytes()).hexdigest()}

    def save() -> None:
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    try:
        torch.set_num_threads(1)
        torch.manual_seed(9417)
        matrices = synthetic_matrices()
        cpu_packed = []
        report["packing_preflight"] = []
        for cpu in matrices:
            packed = runtime.compress(cpu)
            raw = tuple(buffer.numpy().tobytes() for buffer in packed.buffers)
            source = cpu.view(torch.uint8).numpy().tobytes()
            if (not 0 <= packed.start_exp <= 248
                    or unpack_words(raw, packed.m, packed.k, packed.start_exp) != source):
                raise RuntimeError("CPU reconstruction/base gate failed before CUDA initialization")
            report["packing_preflight"].append({"shape": list(cpu.shape), "bits_exact": True,
                "source_sha256": hashlib.sha256(source).hexdigest(),
                "metadata": [packed.max_common, packed.max_rare, packed.start_exp]})
            cpu_packed.append(packed)
        save()
        torch.cuda.set_device(0)
        props = torch.cuda.get_device_properties(0)
        if (props.major, props.minor) != (12, 1):
            raise ValueError("Require SM121 for this component")
        torch.cuda.set_per_process_memory_fraction(1024**3 / props.total_memory)
        native = NativeDecoder(args.build)
        stream = torch.cuda.Stream()
        for index, (cpu, host) in enumerate(zip(matrices, cpu_packed, strict=True)):
            packed = host.to("cuda")
            expected = cpu.cuda()
            case: dict[str, Any] = {"index": index, "shape": list(cpu.shape),
                "source_sha256": hashlib.sha256(cpu.view(torch.uint8).numpy()).hexdigest(),
                "metadata": [packed.max_common, packed.max_rare, packed.start_exp], "modes": [],
                "qualified_bits_exact": False}
            report["cases"].append(case)
            baseline = runtime.decompress(packed)
            if not torch.equal(baseline.view(torch.uint8), expected.view(torch.uint8)):
                actual = baseline.cpu().view(torch.uint16).flatten().to(torch.int64)
                wanted = cpu.view(torch.uint16).flatten().to(torch.int64)
                bad = torch.nonzero(actual != wanted).flatten().tolist()
                case["qualified_mismatch"] = {"words": len(bad), "examples": [
                    {"index": i, "source": int(wanted[i]), "decoded": int(actual[i])}
                    for i in bad[:16]]}
                args.report.with_suffix(".source.bin").write_bytes(
                    cpu.view(torch.uint8).numpy().tobytes())
                args.report.with_suffix(".qualified.bin").write_bytes(
                    baseline.cpu().view(torch.uint8).numpy().tobytes())
                raise RuntimeError("The qualified decoder fails the synthetic source-byte check")
            case["qualified_bits_exact"] = True
            arena = torch.full((cpu.numel() + 128,), 19., device="cuda", dtype=torch.bfloat16)
            output = arena[64:-64].view(cpu.shape)

            for mode in MODES:
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    native.decode(mode, packed, output)
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                verify_words(output, expected, arena)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    native.decode(mode, packed, output)
                allocated = torch.cuda.memory_allocated()
                for _ in range(3):
                    output.fill_(-3)
                    graph.replay()
                    graph.replay()
                    torch.cuda.synchronize()
                    verify_words(output, expected, arena)
                if torch.cuda.memory_allocated() != allocated:
                    raise RuntimeError("Decoder graph replay grew CUDA allocation")
                case["modes"].append({"mode": mode, "bits_exact": True,
                    "nondefault_stream_exact": True, "graph_replays_exact": 6,
                    "guards_unchanged": True, "replay_allocation_growth": 0,
                    "resources": native.resources(mode, packed)})
                save()
                del graph
            del packed, expected, baseline, arena, output
        report.update(status="passed", peak_cuda_bytes=torch.cuda.max_memory_allocated())
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        save()
    print(json.dumps({"status": report["status"], "cases": len(report["cases"])}))


if __name__ == "__main__":
    main()
