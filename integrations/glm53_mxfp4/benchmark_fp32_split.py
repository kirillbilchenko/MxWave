"""Check FP32 split partials against retained stock, qualified and rebuilt BF16 controls."""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from benchmark_kda_memory import digest, telemetry
from fp32_split_probe import assess, bf16_metrics, protocol, workspace_bytes
from prepare_bf16_attention import LIBRARY_SHA, RUNTIME_SHA, rebuild_weight, tensor_digest
from safetensors.torch import load_file


def raw(tensor: torch.Tensor) -> bytes:
    """Retain contiguous BF16 words, including their original signs and payloads."""
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def load_control(runtime: Any, path: Path) -> Any:
    """Own a distinct handle to the rebuilt control without changing registered operators."""
    previous = os.environ["MXWAVE_ZIPSERV_LIBRARY"]
    try:
        os.environ["MXWAVE_ZIPSERV_LIBRARY"] = str(path)
        runtime._library.cache_clear()
        return runtime._library()
    finally:
        os.environ["MXWAVE_ZIPSERV_LIBRARY"] = previous
        runtime._library.cache_clear()


class Operation:
    """Own fixed padded inputs, guarded output and correctly typed partial workspace."""

    def __init__(self, library: Any, packed: Any, x: torch.Tensor, rows: int,
                 split: int, *, fp32: bool) -> None:
        from mxwave_zipserv.runtime import padded_tokens

        self.library, self.packed, self.rows = library, packed, rows
        self.tokens, self.split, self.fp32 = x.shape[0], split, fp32
        n = padded_tokens(self.tokens)
        self.x = torch.zeros((n, packed.k), dtype=torch.bfloat16, device="cuda")
        self.x[:self.tokens].copy_(x)
        self.output_arena = torch.full((n * packed.m + 256,), 13., device="cuda",
                                       dtype=torch.bfloat16)
        self.output = self.output_arena[128:-128].view(n, packed.m)
        elements = n * packed.m * split if split > 1 else 0
        if fp32 and elements * 4 != workspace_bytes(packed.m, n, split):
            raise ValueError("FP32 workspace element count mismatch")
        self.scratch_arena = torch.full((elements + 256,), 17., device="cuda",
            dtype=torch.float32 if fp32 else torch.bfloat16)
        self.scratch = self.scratch_arena[128:-128]
        self.arguments = [*(t.data_ptr() for t in packed.buffers), packed.max_common,
            packed.max_rare, packed.start_exp, self.x.data_ptr(), self.output.data_ptr(),
            packed.m, n, packed.k, self.scratch.data_ptr(), split]

    def __call__(self) -> torch.Tensor:
        """Launch on the current stream without allocation or hidden workspace conversion."""
        fn = self.library.mxzip_mm_fp32 if self.fp32 else self.library.mxzip_mm
        args = [torch.cuda.current_stream().cuda_stream, *self.arguments]
        if self.fp32:
            args.append(self.scratch.numel() * self.scratch.element_size())
        status = fn(*args)
        if status:
            raise RuntimeError(f"Native component failed: {status}")
        return self.output[:self.tokens, :self.rows]

    def guards(self) -> bool:
        """Check both ends of output and partial workspace after eager and graph execution."""
        return all(bool((a[:128] == value).all() and (a[-128:] == value).all())
                   for a, value in ((self.output_arena, 13.), (self.scratch_arena, 17.)))

    def reject_short_workspace(self) -> bool:
        """Require the distinct candidate ABI to reject undersized allocations before launch."""
        if not self.fp32:
            raise ValueError("Only the FP32 ABI accepts explicit byte capacity")
        return all(self.library.mxzip_mm_fp32(torch.cuda.current_stream().cuda_stream,
            *self.arguments, size) != 0
            for size in (0, self.scratch.numel() * self.scratch.element_size() - 1))


def main() -> None:
    """Run all declared precision cases, retaining failed numerics without claiming speed."""
    from mxwave_zipserv import runtime

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Preserve the first precision attempt")
    plan = json.loads(args.plan.read_bytes())
    p = protocol()
    if plan["protocol"] != p:
        raise ValueError("Precision protocol changed")
    if (digest(Path(os.environ["MXWAVE_ZIPSERV_LIBRARY"])) != LIBRARY_SHA
            or digest(Path(runtime.__file__)) != RUNTIME_SHA):
        raise ValueError("Qualified control/runtime changed")
    build_path = Path(plan["native_build_path"])
    if digest(build_path) != plan["native_build_sha256"]:
        raise ValueError("Native build changed")
    build = json.loads(build_path.read_bytes())
    libraries = {"original": runtime._library()}
    for arm in build["arms"]:
        path = build_path.parent / arm["library"]
        if digest(path) != arm["library_sha256"]:
            raise ValueError("Native library changed")
        if arm["arm"] == "control":
            libraries["control"] = load_control(runtime, path)
        else:
            library = ctypes.CDLL(str(path))
            if hasattr(library, "mxzip_mm"):
                raise ValueError("Candidate must not impersonate the original GEMM ABI")
            library.mxzip_mm_fp32.argtypes = [*libraries["original"].mxzip_mm.argtypes,
                                              ctypes.c_uint64]
            library.mxzip_mm_fp32.restype = ctypes.c_int
            libraries["fp32"] = library
    torch.set_num_threads(1)
    props = torch.cuda.get_device_properties(0)
    if (props.major, props.minor) != (12, 1):
        raise ValueError("Require SM121")
    torch.cuda.set_per_process_memory_fraction(p["CUDA_allocation_cap_bytes"] / props.total_memory)
    report: dict[str, Any] = {"status": "running", "rank": args.rank, "protocol": p,
        "source_sha256": digest(Path(__file__)), "plan_sha256": digest(args.plan),
        "native_build_sha256": digest(build_path), "qualified_library_sha256": LIBRARY_SHA,
        "runtime_sha256": RUNTIME_SHA, "cases": [], "started_epoch": time.time(),
        "device": props.name, "model_loaded": False, "initial_telemetry": telemetry()}

    def save() -> None:
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def retain(name: str, value: torch.Tensor) -> dict[str, Any]:
        data = raw(value)
        path = args.report.parent / (name + ".bf16")
        with path.open("xb") as handle:
            handle.write(data)
        return {"path": path.name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                "shape": list(value.shape)}

    save()
    try:
        source = Path("/audit/source")
        index = json.loads((source / "model.safetensors.index.json").read_bytes())["weight_map"]
        stream = torch.cuda.Stream()
        for case in (c for c in plan["selected_cases"] if c["rank"] == args.rank):
            artifact = Path(case["artifact_path"])
            if digest(artifact) != case["artifact_sha256"]:
                raise ValueError("Packed source artifact changed")
            cpu = rebuild_weight(source, index, case)
            if tensor_digest(cpu) != case["logical_weight_sha256"]:
                raise ValueError("Source checkpoint shard changed")
            dense = cpu.cuda()
            del cpu
            buffers = load_file(artifact)
            packed = runtime.PackedWeight(tuple(buffers[n].cuda() for n in runtime.BUFFER_NAMES),
                                          *case["native_metadata"])
            hashes = [hashlib.sha256(t.numpy().tobytes()).hexdigest()
                      for t in (buffers[n] for n in runtime.BUFFER_NAMES)]
            del buffers
            decoded = torch.empty((packed.m, packed.k), device="cuda", dtype=torch.bfloat16)
            runtime._check(libraries["original"].mxzip_decode(
                torch.cuda.current_stream().cuda_stream,
                *(t.data_ptr() for t in packed.buffers), packed.max_common, packed.max_rare,
                packed.start_exp, decoded.data_ptr(), packed.m, packed.k))
            exact = tensor_digest(decoded.cpu()) == case["padded_weight_sha256"]
            if not exact:
                raise ValueError("Decoded source bytes changed")
            del decoded
            entry = {"name": case["name"], "source_bits_exact": True,
                "decoder_bits_exact": exact, "artifact_sha256": case["artifact_sha256"],
                "source_weight_sha256": case["logical_weight_sha256"], "checks": []}
            report["cases"].append(entry)
            for tokens in p["tokens"]:
                for activation in p["activation_cases"]:
                    if activation == "basis":
                        x = torch.zeros((tokens, packed.k), dtype=torch.bfloat16, device="cuda")
                        for i in range(tokens):
                            x[i, (0, 511, 512, 1023, 1024, 2047, 2048, 4095)[i % 8]] = 1.
                    else:
                        torch.manual_seed(int(activation))
                        x = torch.randn((tokens, packed.k), dtype=torch.bfloat16, device="cuda") * .1
                    name = f"{case['name']}-m{tokens}-{activation}"
                    stock = torch.nn.functional.linear(x, dense)
                    original = Operation(libraries["original"], packed, x, dense.shape[0], 1,
                                         fp32=False)
                    rebuilt = Operation(libraries["control"], packed, x, dense.shape[0], 1,
                                        fp32=False)
                    first = raw(original())
                    rebuilt_exact = raw(rebuilt()) == first
                    exact_required = tokens in p["stock_bits_required_at_tokens"] or activation == "basis"
                    baseline_error = bf16_metrics(first, raw(stock), exact=exact_required)
                    evidence = {"activation": retain(name + "-x", x),
                        "stock": retain(name + "-stock", stock),
                        "original": retain(name + "-original", original.output[:tokens, :dense.shape[0]]),
                        "rebuilt": retain(name + "-rebuilt", rebuilt.output[:tokens, :dense.shape[0]])}
                    for split in p["splits"]:
                        legacy = Operation(libraries["control"], packed, x, dense.shape[0], split,
                                           fp32=False)
                        candidate = Operation(libraries["fp32"], packed, x, dense.shape[0], split,
                                              fp32=True)
                        old, new = legacy(), candidate()
                        before = raw(new)
                        invalid = candidate.reject_short_workspace()
                        within = all(raw(candidate()) == before for _ in range(3))
                        stream.wait_stream(torch.cuda.current_stream())
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=stream):
                            candidate()
                        graph.replay()
                        torch.cuda.synchronize()
                        graph_exact = raw(candidate.output[:tokens, :dense.shape[0]]) == before
                        allocated = torch.cuda.memory_allocated()
                        for _ in range(p["graph_replays"]):
                            graph.replay()
                        torch.cuda.synchronize()
                        retained = torch.cuda.memory_allocated() - allocated
                        graph_exact &= raw(candidate.output[:tokens, :dense.shape[0]]) == before
                        row = {"tokens": tokens, "activation": activation, "split": split,
                            "original_stock_comparison": baseline_error,
                            "stock_comparison": bf16_metrics(before, raw(stock), exact=exact_required),
                            "parent_comparison": bf16_metrics(before, first, exact=False),
                            "legacy_stock_comparison": bf16_metrics(raw(old), raw(stock), exact=exact_required),
                            "rebuilt_matches_original": rebuilt_exact,
                            "within_arm_repeats_exact": within, "graph_matches_eager": graph_exact,
                            "graph_retained_bytes": retained,
                            "invalid_workspace_rejected": invalid,
                            "workspace_bytes": candidate.scratch.numel() * 4,
                            "guards_unchanged": all(op.guards() for op in
                                (candidate, legacy, original, rebuilt)),
                            "artifacts": {**evidence,
                                "candidate": retain(name + f"-s{split}-fp32", new),
                                "legacy": retain(name + f"-s{split}-bf16", old)},
                            "telemetry": telemetry()}
                        entry["checks"].append(row)
                        del graph, candidate, legacy, old, new
                    del original, rebuilt, stock, x
                    save()
            entry["packed_buffers_unchanged"] = hashes == [
                hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest() for t in packed.buffers]
            del packed, dense
            gc.collect()
            torch.cuda.empty_cache()
            save()
        report.update(status="complete", CUDA_peak_bytes=torch.cuda.max_memory_allocated(),
                      finished_epoch=time.time())
        report["assessment"] = assess(report)
    except BaseException as exc:
        report.update(status="failed", error=repr(exc), finished_epoch=time.time())
        raise
    finally:
        save()
    print(json.dumps(report["assessment"]))


if __name__ == "__main__":
    main()
