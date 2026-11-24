"""Prove the candidate exponent mask on wrapped bases and every BF16 word on SM121."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from prefill_decode import MODES, NativeDecoder
from prefill_pack_oracle import synthetic_matrices, unpack_words
from smoke_prefill_decode import verify_words


def matrices() -> list[torch.Tensor]:
    """Retain the original failure and force base 255 beside all 65,536 BF16 words."""
    original = ((torch.arange(16384, dtype=torch.int64) * 37) & 65535).to(
        torch.uint16).view(torch.bfloat16).reshape(128, 128)
    index = torch.arange(196608, dtype=torch.int64)
    frequent = ((index % 7) << 7) | ((index >> 3) & 127) | ((index & 1) << 15)
    exhaustive = torch.cat([frequent, torch.arange(65536, dtype=torch.int64)]).to(
        torch.uint16).view(torch.bfloat16).reshape(512, 512)
    return [*synthetic_matrices(), original, exhaustive]


def refuse_control(native: NativeDecoder, packed: Any, output: torch.Tensor) -> dict[str, Any]:
    """Check both Python and native control guards without invoking the unsafe decoder."""
    try:
        native.decode("stock", packed, output)
    except ValueError:
        pass
    else:
        raise RuntimeError("Python control accepted a wrapped exponent base")
    output.fill_(-3)
    code = native.library.mxwave_prefill_decode(
        0, torch.cuda.current_stream().cuda_stream, *(t.data_ptr() for t in packed.buffers),
        packed.max_common, packed.max_rare, packed.start_exp, output.data_ptr(),
        packed.m, packed.k)
    torch.cuda.synchronize()
    if code != 1 or not bool(torch.all(output == -3)):
        raise RuntimeError("Native control must return cudaErrorInvalidValue without writing")
    return {"python_rejected": True, "native_error": code, "output_unchanged": True}


def main() -> None:
    """Require CPU inversion, exact GPU bits, safe streams/graphs and unchanged guards."""
    from mxwave_zipserv import runtime

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Preserve the first wrapped-base qualification")
    report: dict[str, Any] = {"status": "running", "model_weights_loaded": False,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": [], "packing_preflight": []}

    def save() -> None:
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    try:
        torch.set_num_threads(1)
        torch.manual_seed(9417)
        weights = matrices()
        hosts = []
        for index, cpu in enumerate(weights):
            host = runtime.compress(cpu)
            raw = tuple(buffer.numpy().tobytes() for buffer in host.buffers)
            source = cpu.view(torch.uint8).numpy().tobytes()
            if unpack_words(raw, host.m, host.k, host.start_exp, wrap_exponent=True) != source:
                raise RuntimeError("Masked CPU oracle differs from original source words")
            if index >= 3 and host.start_exp != 255:
                raise RuntimeError("Wrapped regression must actually exercise exponent base 255")
            unmasked = unpack_words(raw, host.m, host.k, host.start_exp)
            differences = sum(unmasked[n:n + 2] != source[n:n + 2]
                              for n in range(0, len(source), 2))
            if (index >= 3) != bool(differences):
                raise RuntimeError("The fixture did not reproduce the original mask defect")
            report["packing_preflight"].append({"shape": list(cpu.shape), "base": host.start_exp,
                "source_sha256": hashlib.sha256(source).hexdigest(), "cpu_masked_exact": True,
                "unmasked_mismatches": differences})
            hosts.append(host)
        assert not torch.cuda.is_initialized()
        report["cpu_preflight_without_cuda"] = True
        save()
        if args.cpu_only:
            report["status"] = "cpu_verified"
            print(json.dumps({"status": report["status"], "cases": len(hosts),
                              "cuda_initialized": False}))
            return
        torch.cuda.set_device(0)
        props = torch.cuda.get_device_properties(0)
        if (props.major, props.minor) != (12, 1):
            raise ValueError("Require SM121")
        torch.cuda.set_per_process_memory_fraction(1024**3 / props.total_memory)
        native = NativeDecoder(args.build)
        report["native_manifest"] = native.manifest
        stream = torch.cuda.Stream()
        for index, (cpu, host) in enumerate(zip(weights, hosts, strict=True)):
            packed, expected = host.to("cuda"), cpu.cuda()
            arena = torch.full((cpu.numel() + 128,), 19., dtype=torch.bfloat16, device="cuda")
            output = arena[64:-64].view(cpu.shape)
            wrapped = packed.start_exp > 248
            case: dict[str, Any] = {**report["packing_preflight"][index], "modes": [],
                "qualified_skipped_for_wrapped_base": wrapped}
            report["cases"].append(case)
            if wrapped:
                case["control_guards"] = refuse_control(native, packed, output)
            else:
                baseline = runtime.decompress(packed)
                if not torch.equal(baseline.view(torch.uint8), expected.view(torch.uint8)):
                    raise RuntimeError("Safe-base qualified decoder differs from source words")
                case["qualified_bits_exact"] = True
                del baseline
            for mode in MODES:
                if wrapped and mode == "stock":
                    continue
                output.fill_(-3)
                native.decode(mode, packed, output)
                torch.cuda.synchronize()
                verify_words(output, expected, arena)
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    output.fill_(-3)
                    native.decode(mode, packed, output)
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                verify_words(output, expected, arena)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    native.decode(mode, packed, output)
                allocated = torch.cuda.memory_allocated()
                for _ in range(6):
                    output.fill_(-3)
                    graph.replay()
                    torch.cuda.synchronize()
                    verify_words(output, expected, arena)
                if torch.cuda.memory_allocated() != allocated:
                    raise RuntimeError("Wrapped decoder replay grew CUDA allocation")
                case["modes"].append({"mode": mode, "bits_exact": True, "eager_exact": True,
                    "nondefault_stream_exact": True, "graph_replays_exact": 6,
                    "guards_unchanged": True, "replay_allocation_growth": 0})
                save()
                del graph
            del packed, expected, arena, output
        report.update(status="passed", peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                      wrapped_bases_passed=2, full_bf16_word_space_tested=True)
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        save()
    print(json.dumps({"status": report["status"], "wrapped_bases_passed": 2,
                      "peak_cuda_bytes": report["peak_cuda_bytes"]}))


if __name__ == "__main__":
    main()
