"""Bounded real-projection test of the existing fused ZipServ BF16 kernel on Spark."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from prepare_bf16_attention import LIBRARY_SHA, RUNTIME_SHA, rebuild_weight, tensor_digest
from safetensors.torch import load_file


def write(path: Path, report: dict[str, Any]) -> None:
    """Preserve completed cases if the guard interrupts a later benchmark case."""
    temporary = path.with_suffix(".incomplete")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main() -> None:
    """Require exact GPU decode, then compare dense and fused GEMMs in both orders."""
    from benchmark_bf16_head import _graph, _paired
    from mxwave_zipserv import runtime

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--fixtures", type=Path,
                        default=Path("/audit/bf16-attention-v1/artifacts-v2"))
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Preserve previous GPU component reports")
    prepared = json.loads((args.fixtures / "prepared.json").read_bytes())
    if prepared["status"] != "packed_gpu_verification_pending" or len(prepared["cases"]) != 3:
        raise ValueError("Finish the three CPU fixtures first")
    library = Path(os.environ["MXWAVE_ZIPSERV_LIBRARY"])
    if (hashlib.sha256(library.read_bytes()).hexdigest() != LIBRARY_SHA
            or hashlib.sha256(Path(runtime.__file__).read_bytes()).hexdigest() != RUNTIME_SHA):
        raise ValueError("The qualified native library or runtime changed")
    for file, expected in (("config.json", prepared["source_config_sha256"]),
                           ("model.safetensors.index.json", prepared["source_index_sha256"])):
        if hashlib.sha256((args.source / file).read_bytes()).hexdigest() != expected:
            raise ValueError("Fixture source metadata changed")
    if hashlib.sha256(Path(__file__).with_name("prepare_bf16_attention.py").read_bytes()
                      ).hexdigest() != prepared["code_sha256"]:
        raise ValueError("The prepared fixture reconstruction code changed")
    index = json.loads((args.source / "model.safetensors.index.json").read_bytes())["weight_map"]
    torch.set_num_threads(2)
    torch.manual_seed(719)
    device = torch.device("cuda", 0)
    torch.cuda.set_per_process_memory_fraction(
        1.5 * 1024**3 / torch.cuda.get_device_properties(device).total_memory, device)
    started = time.monotonic()
    result: dict[str, Any] = {"status": "running", "cases": [],
        "library_sha256": LIBRARY_SHA, "runtime_sha256": RUNTIME_SHA,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "prepared_sha256": hashlib.sha256((args.fixtures / "prepared.json").read_bytes()).hexdigest(),
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(device),
        "gates": {"gpu_weight_bytes_exact": True, "relative_l2_maximum": .01},
        "limits": ["Three real projection fixtures; not full-model throughput or quality.",
                   "Random BF16 activations, not captured GLM activations.",
                   "Graph replay and eager CUDA-event component timings, alternating arm order.",
                   "CUDA graphs are also measured at prefill sizes; full-model prefill is eager.",
                   "Paired benchmark deliberately retains both dense and packed weights."]}
    write(args.report, result)
    try:
        for case in prepared["cases"]:
            artifact = args.fixtures / case["artifact"]
            if hashlib.sha256(artifact.read_bytes()).hexdigest() != case["artifact_sha256"]:
                raise ValueError("Packed attention artifact changed")
            cpu = rebuild_weight(args.source, index, case)
            if tensor_digest(cpu) != case["logical_weight_sha256"]:
                raise ValueError("Source projection reconstruction changed")
            dense = cpu.cuda()
            del cpu
            buffers = load_file(artifact)
            packed = runtime.PackedWeight(
                tuple(buffers[name].cuda() for name in runtime.BUFFER_NAMES),
                *case["native_metadata"])
            del buffers
            restored = runtime.decompress(packed).cpu()
            exact = tensor_digest(restored) == case["padded_weight_sha256"]
            del restored
            entry: dict[str, Any] = {"name": case["name"], "logical_shape": case["logical_shape"],
                "native_shape": case["native_shape"], "raw_bytes": case["raw_bytes"],
                "retained_bytes": packed.storage_bytes, "saving_fraction": case["saving_fraction"],
                "gpu_decode_bits_exact": exact, "measurements": []}
            result["cases"].append(entry)
            write(args.report, result)
            if not exact:
                raise RuntimeError("Native GPU decoder changed BF16 weight bytes")
            rows, width = case["logical_shape"]
            for tokens in (1, 8, 32, 128, 512, 2048):
                x = torch.randn((tokens, width), dtype=torch.bfloat16, device=device) * .1

                def baseline(value: torch.Tensor = x, weight: torch.Tensor = dense) -> torch.Tensor:
                    return torch.nn.functional.linear(value, weight)

                for split_k in (1, 4):
                    def candidate(value: torch.Tensor = x, storage: Any = packed,
                                  split: int = split_k, count: int = rows) -> torch.Tensor:
                        return runtime.linear(value, storage, split_k=split)[..., :count].contiguous()

                    dense_graph, reference = _graph(baseline)
                    packed_graph, actual = _graph(candidate)
                    error = actual.float() - reference.float()
                    finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
                    relative = (float(error.norm() / reference.float().norm().clamp_min(1e-30))
                                if finite else None)
                    record = {"tokens": tokens, "split_k": split_k, "finite": finite,
                              "relative_l2": relative,
                              "max_absolute_error": float(error.abs().max()) if finite else None,
                              "arithmetic_passed": relative is not None and relative < .01,
                              "graph_replay": _paired(dense_graph.replay, packed_graph.replay,
                                  5, 10, interleaved=True),
                              "eager_cuda_events": _paired(baseline, candidate,
                                  5, 5, interleaved=True)}
                    entry["measurements"].append(record)
                    write(args.report, result)
                    del dense_graph, packed_graph, reference, actual, error
                    gc.collect()
                    torch.cuda.empty_cache()
                del x, baseline, candidate
            del dense, packed
            gc.collect()
            torch.cuda.empty_cache()
        passed = all(case["gpu_decode_bits_exact"]
                     and all(item["arithmetic_passed"] for item in case["measurements"])
                     for case in result["cases"])
        result.update(status="passed" if passed else "failed", passed=passed,
                      seconds=time.monotonic() - started,
                      cuda_peak_bytes=torch.cuda.max_memory_allocated())
        write(args.report, result)
        print(json.dumps({"status": result["status"], "seconds": result["seconds"],
                          "cuda_peak_bytes": result["cuda_peak_bytes"]}))
    except BaseException as exc:
        result.update(status="failed", passed=False, error=repr(exc))
        write(args.report, result)
        raise


if __name__ == "__main__":
    main()
