"""Measure qualified fused split-K=1 versus reconstructed stock GEMM across row counts."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from benchmark_prefill_decode import capture, measure, ratio_passes
from prepare_bf16_attention import LIBRARY_SHA, RUNTIME_SHA, rebuild_weight, tensor_digest
from safetensors.torch import load_file
from zipserv_contract import require_qualified_exponent_base

TOKENS = (1, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128, 256, 257, 259, 512, 513, 514, 769, 1024, 2048)
CONFIRM_TOKENS = (8, 12, 16, 24, 32, 64, 128, 256, 257, 2048)
FOCUS_TOKENS = (8, 16, 257)
ARMS = ("qualified", "stock", "raw_dense", "fused_split1")


def digest(path: Path) -> str:
    """Bind input artifacts and source code without importing CUDA outside the GPU process."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arithmetic(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    """Apply the existing real-projection relative-L2 gate without loosening its tolerance."""
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
    if not finite:
        return {"finite": False, "relative_l2": None, "max_absolute_error": None,
                "passed": False}
    difference = actual.float() - reference.float()
    relative = float(difference.norm() / reference.float().norm().clamp_min(1e-30))
    return {"finite": True, "relative_l2": relative,
            "max_absolute_error": float(difference.abs().max()), "passed": relative < .01}


def eligible(row: dict[str, Any]) -> bool:
    """Require arithmetic, repeat-control parity and faster fused execution in every condition."""
    return bool(row["arithmetic"]["passed"] and row["stock_outputs_bits_exact"]
                and row["fused_eager_graph_bits_exact"]
                and all(ratio_passes(value, "stock", .98, 1.02)
                        and ratio_passes(value, "fused_split1", 0, .95)
                        for group in row["timing"].values() for value in group.values()))


def main() -> None:
    """Preserve all cases/orders; report measured leads without changing the serving router."""
    from mxwave_zipserv import runtime

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Preserve the first complete routing sweep")
    plan = json.loads(args.plan.read_bytes())
    blocked = plan.get("paired_blocks", False)
    queued = plan.get("queued_blocks", False)
    if queued and not blocked:
        raise ValueError("Queued timing requires the bracketed identical-reference control")
    if queued and (plan["iterations_per_leg"] != 8
                   or plan["untimed_primer_calls_per_leg"] != 1
                   or plan["activation_seed"] != 16721):
        raise ValueError("Queued timing sample count, primer or fresh seed changed")
    tokens_sweep = FOCUS_TOKENS if queued else CONFIRM_TOKENS if blocked else TOKENS
    if plan["tokens"] != list(tokens_sweep) or plan["routing_arms"] != list(ARMS):
        raise ValueError("Routing shapes or arm order differ from the frozen plan")
    if digest(Path(os.environ["MXWAVE_ZIPSERV_LIBRARY"])) != LIBRARY_SHA:
        raise ValueError("Require the unchanged qualified native library")
    if digest(Path(runtime.__file__)) != RUNTIME_SHA:
        raise ValueError("Require the unchanged qualified Python runtime")
    if json.loads(args.smoke.read_bytes())["status"] != "passed":
        raise ValueError("Finish the exact decoder smoke before the routing sweep")
    fixtures = args.root / "bf16-attention-v1/artifacts-v2"
    prepared = json.loads((fixtures / "prepared.json").read_bytes())
    if prepared["code_sha256"] != digest(Path(__file__).with_name("prepare_bf16_attention.py")):
        raise ValueError("Fixture reconstruction source changed")
    source = args.root / "source"
    for name, key in (("config.json", "source_config_sha256"),
                      ("model.safetensors.index.json", "source_index_sha256")):
        if digest(source / name) != prepared[key]:
            raise ValueError("Audited checkpoint metadata changed")
    cases = [row for row in prepared["cases"] if row["name"].startswith("kda-")]
    if [row["name"] for row in cases] != plan["cases"]:
        raise ValueError("Require both real KDA matrix shapes")
    for row in cases:
        require_qualified_exponent_base(row["native_metadata"][4], where=row["name"])
    torch.set_num_threads(1)
    torch.manual_seed(16719 + int(blocked) + int(queued))
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    if (props.major, props.minor) != (12, 1):
        raise ValueError("Require SM121")
    torch.cuda.set_per_process_memory_fraction(1.5 * 1024**3 / props.total_memory)
    index = json.loads((source / "model.safetensors.index.json").read_bytes())["weight_map"]
    report: dict[str, Any] = {"status": "running", "cases": [], "tokens": list(tokens_sweep),
        "arms": list(ARMS), "full_model_loaded": False, "qualification_attempted": False,
        "serving_adapter_changed": False, "code_sha256": digest(Path(__file__)),
        "plan_sha256": digest(args.plan), "smoke_sha256": digest(args.smoke),
        "library_sha256": LIBRARY_SHA, "runtime_sha256": RUNTIME_SHA,
        "fixture_manifest_sha256": digest(fixtures / "prepared.json"),
        "paired_blocks": blocked, "same_reference_graph_control": blocked,
        "queued_blocks": queued,
        "started_epoch": time.time(), "torch": torch.__version__,
        "limits": ["Two layer-zero/rank-zero matrices and random BF16 activations.",
            "No model throughput, MTP execution, routing change or numerical release qualification.",
            "Eager CUDA events and graph replay are separate component measurements.",
            "Cache pressure is outside event pairs; no hardware LPDDR traffic is measured.",
            "Stock repetition is the identical decode-plus-GEMM callable captured separately.",
            "Measured eligible row counts do not qualify unmeasured intermediate row counts."]}

    def save() -> None:
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    try:
        pressure = torch.zeros(32 * 1024**2, device="cuda", dtype=torch.int32)
        stream = torch.cuda.Stream()
        for case in cases:
            artifact = fixtures / case["artifact"]
            if artifact.name != case["artifact"] or digest(artifact) != case["artifact_sha256"]:
                raise ValueError("Compressed fixture changed")
            cpu = rebuild_weight(source, index, case)
            if tensor_digest(cpu) != case["logical_weight_sha256"]:
                raise ValueError("Actual source reconstruction changed")
            rows, width = cpu.shape
            dense = cpu.cuda()
            del cpu
            buffers = load_file(artifact)
            packed = runtime.PackedWeight(tuple(buffers[name].cuda() for name in runtime.BUFFER_NAMES),
                                          *case["native_metadata"])
            del buffers
            arena = torch.full((packed.m * packed.k + 128,), 19., device="cuda", dtype=torch.bfloat16)
            destination = arena[64:-64].view(packed.m, packed.k)

            def decode(storage: Any = packed, output: torch.Tensor = destination) -> torch.Tensor:
                runtime._check(runtime._library().mxzip_decode(
                    torch.cuda.current_stream().cuda_stream,
                    *(buffer.data_ptr() for buffer in storage.buffers), storage.max_common,
                    storage.max_rare, storage.start_exp, output.data_ptr(), storage.m, storage.k))
                return output

            def check_weight(output: torch.Tensor = destination, guards: torch.Tensor = arena,
                             wanted: str = case["padded_weight_sha256"]) -> None:
                torch.cuda.synchronize()
                if (tensor_digest(output.cpu()) != wanted
                        or not bool(torch.all(guards[:64] == 19))
                        or not bool(torch.all(guards[-64:] == 19))):
                    raise RuntimeError("The unchanged decoder changed weight bits or destination guards")

            decode()
            check_weight()
            entry: dict[str, Any] = {"name": case["name"], "logical_shape": [rows, width],
                "padded_shape": [packed.m, packed.k], "packed_bytes": packed.storage_bytes,
                "source_weight_sha256": case["logical_weight_sha256"],
                "artifact_sha256": case["artifact_sha256"], "weight_bits_exact": True,
                "measurements": []}
            report["cases"].append(entry)
            save()
            for tokens in tokens_sweep:
                x = torch.randn((tokens, width), dtype=torch.bfloat16, device="cuda") * .1

                def reconstructed(value: torch.Tensor = x, rebuild: Any = decode,
                                  count: int = rows) -> torch.Tensor:
                    return torch.nn.functional.linear(value, rebuild()[:count])

                def raw(value: torch.Tensor = x, weight: torch.Tensor = dense) -> torch.Tensor:
                    return torch.nn.functional.linear(value, weight)

                def fused(value: torch.Tensor = x, storage: Any = packed,
                          count: int = rows) -> torch.Tensor:
                    return runtime.linear(value, storage, split_k=1)[..., :count].contiguous()

                operations: dict[str, Callable[[], torch.Tensor]] = {
                    "qualified": reconstructed, "stock": reconstructed,
                    "raw_dense": raw, "fused_split1": fused}
                reference = raw()
                eager_fused = fused()
                error = arithmetic(eager_fused, reference)
                graphs, outputs = {}, {}
                for name, operation in operations.items():
                    if blocked and name == "stock":
                        graphs[name], outputs[name] = graphs["qualified"], outputs["qualified"]
                    else:
                        graphs[name], outputs[name] = capture(operation, stream)
                if blocked and graphs["qualified"] is not graphs["stock"]:
                    raise RuntimeError("Bracketed control must replay the identical reference graph")
                stock_exact = all(torch.equal(outputs[name].view(torch.uint8),
                                               reference.view(torch.uint8)) for name in ARMS[:3])
                fused_same = bool(torch.equal(eager_fused.view(torch.uint8),
                                              outputs["fused_split1"].view(torch.uint8)))
                if not stock_exact or not fused_same:
                    raise RuntimeError(f"Execution/control byte consistency failed at M={tokens}")
                row: dict[str, Any] = {"tokens": tokens, "native_padded_tokens":
                    runtime.padded_tokens(tokens), "arithmetic": error,
                    "stock_outputs_bits_exact": stock_exact, "fused_eager_graph_bits_exact": fused_same,
                    "timing": {}}
                # Preserve the same balanced arm order for eager and captured calls.
                for execution, calls in (("eager", {name: SimpleNamespace(replay=op)
                        for name, op in operations.items()}), ("graph", graphs)):
                    if blocked:
                        from prefill_routing_blocks import measure as measure_blocks

                        if queued:
                            from prefill_routing_queued_blocks import measure as measure_blocks

                            calls["stock"] = calls["qualified"]

                        row["timing"][execution] = {cache: measure_blocks(calls, pressure,
                            evict=cache == "pressure") for cache in ("warm", "pressure")}
                    else:
                        row["timing"][execution] = {cache: measure(calls, pressure, trials=8,
                            iterations=4, evict=cache == "pressure") for cache in ("warm", "pressure")}
                for graph in graphs.values():
                    graph.replay()
                torch.cuda.synchronize()
                if any(not torch.equal(outputs[name].view(torch.uint8), reference.view(torch.uint8))
                       for name in ARMS[:3]):
                    raise RuntimeError("Stock output changed after retained timing")
                if not torch.equal(outputs["fused_split1"].view(torch.uint8),
                                   eager_fused.view(torch.uint8)):
                    raise RuntimeError("Fused output changed after retained timing")
                row["component_eligible"] = eligible(row)
                entry["measurements"].append(row)
                save()
                del operations, operation, graphs, outputs, graph, calls, x, reference, eager_fused
                gc.collect()
                torch.cuda.empty_cache()
            check_weight()
            entry["destination_guards_unchanged"] = True
            del dense, packed, arena, destination, decode, reconstructed, raw, fused, check_weight
            gc.collect()
            torch.cuda.empty_cache()
        arithmetic_passed = all(row["arithmetic"]["passed"] for case in report["cases"]
                                for row in case["measurements"])
        shared = [tokens for tokens in tokens_sweep if all(case["measurements"][tokens_sweep.index(tokens)][
            "component_eligible"] for case in report["cases"])]
        report.update(status="complete", correctness_passed=arithmetic_passed,
            component_eligible_tested_rows=shared, serving_routing_qualified=False,
            peak_cuda_bytes=torch.cuda.max_memory_allocated(), finished_epoch=time.time())
    except BaseException as exc:
        report.update(status="failed", error=repr(exc), finished_epoch=time.time())
        raise
    finally:
        save()
    print(json.dumps({key: report[key] for key in ("status", "correctness_passed",
        "component_eligible_tested_rows", "peak_cuda_bytes")}))


if __name__ == "__main__":
    main()
