"""Inspect complete KDA capture dependencies and native addresses without changing kernels."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from functools import partial
from pathlib import Path
from typing import Any

import torch
from benchmark_full_model import write
from benchmark_kda_direct_dispatch import paired_graph_time, require_equal
from kda_direct_dispatch import linear
from kda_runtime import kda_linear, validate_runtime
from prepare_bf16_attention import tensor_digest
from safetensors.torch import load_file


def topology_signature(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare node metadata and dependency edges independently of CUDA handles/enumeration."""
    labels = {node["index"]: json.dumps({k: node[k] for k in
              ("node_type", "kernel_name", "grid", "block", "shared_memory")}, sort_keys=True)
              for node in nodes}
    if len(labels) != len(nodes) or any(
        parent not in labels for node in nodes for parent in node["dependencies"]
    ):
        raise ValueError("Invalid graph topology indices")
    return {"unique_node_labels": len(set(labels.values())) == len(nodes),
            "nodes": sorted(labels.values()),
            "edges": sorted([labels[parent], labels[node["index"]]]
                            for node in nodes for parent in node["dependencies"])}


def checked(result: tuple[Any, ...]) -> tuple[Any, ...]:
    """Check a CUDA Python API return without relying on tools-ID/driver-version extensions."""
    if int(result[0]) != 0:
        raise RuntimeError(f"CUDA graph introspection returned {result[0]}")
    return result[1:]


def topology(graph: Any, path: Path) -> dict[str, Any]:
    """Read kernel, copy, event and empty nodes plus edges from a retained raw CUDA graph."""
    from cuda.bindings import driver, runtime

    raw = graph.raw_cuda_graph()
    _, count = checked(runtime.cudaGraphGetNodes(raw, numNodes=0))
    handles, count = checked(runtime.cudaGraphGetNodes(raw, numNodes=count))
    indices = {int(handle): index for index, handle in enumerate(handles[:count])}
    nodes = []
    for index, handle in enumerate(handles[:count]):
        kind, = checked(runtime.cudaGraphNodeGetType(handle))
        node: dict[str, Any] = {"index": index, "node_type": int(kind), "kernel_name": None,
                                "grid": None, "block": None, "shared_memory": None,
                                "dependencies": []}
        if kind == runtime.cudaGraphNodeType.cudaGraphNodeTypeKernel:
            parameters, = checked(driver.cuGraphKernelNodeGetParams(
                driver.CUgraphNode(init_value=int(handle))))
            name, = checked(driver.cuFuncGetName(parameters.func))
            node.update(kernel_name=name.decode() if isinstance(name, bytes) else str(name),
                        grid=[parameters.gridDimX, parameters.gridDimY, parameters.gridDimZ],
                        block=[parameters.blockDimX, parameters.blockDimY, parameters.blockDimZ],
                        shared_memory=parameters.sharedMemBytes)
        nodes.append(node)
    _, _, _, count = checked(runtime.cudaGraphGetEdges(raw, numEdges=0))
    if count:
        sources, targets, _, count = checked(runtime.cudaGraphGetEdges(raw, numEdges=count))
        for source, target in zip(sources[:count], targets[:count], strict=True):
            nodes[indices[int(target)]]["dependencies"].append(indices[int(source)])
    checked(runtime.cudaGraphDebugDotPrint(
        raw, str(path).encode(), runtime.cudaGraphDebugDotFlags.cudaGraphDebugDotFlagsVerbose))
    return {"nodes": nodes, "signature": topology_signature(nodes),
            "dot": {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}


def inspect_captures(
    operations: dict[str, Any], expected: torch.Tensor, stream: torch.cuda.Stream,
    directory: Path, prefix: str, *, shared_pool: bool,
) -> dict[str, Any]:
    """Log the actual native ABI pointers during capture, then restore it before replay timing."""
    from mxwave_zipserv import runtime

    library = runtime._library()
    native = library.mxzip_mm
    calls: list[list[int]] = []

    def observe(*args: Any) -> Any:
        calls.append([int(value) for value in args])
        return native(*args)

    pool = torch.cuda.graph_pool_handle() if shared_pool else None
    graphs, outputs, records = {}, {}, {}
    for label, operation in operations.items():
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(20):
                operation()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        calls.clear()
        library.mxzip_mm = observe
        try:
            with torch.cuda.graph(graph, stream=stream, pool=pool):
                output = operation()
        finally:
            library.mxzip_mm = native
        if len(calls) != 1 or len(calls[0]) != 18:
            raise RuntimeError("Expected one unchanged mxzip_mm ABI call during each capture")
        graph.instantiate()
        graph.replay()
        require_equal(output, expected)
        records[label] = {"native_call_arguments": list(calls[0]),
                          "logical_output_ptr": output.data_ptr(),
                          "topology": topology(graph, directory / f"{prefix}-{label}.dot")}
        graphs[label] = graph
        if not shared_pool or label == "direct":
            outputs[label] = output
        del output
        torch.cuda.synchronize()
        gc.collect()
    signatures = [record["topology"]["signature"] for record in records.values()]
    # Argument zero is the identical capture stream; all other native args include actual pointers.
    same_native = all(record["native_call_arguments"] == records["direct"]["native_call_arguments"]
                      for record in records.values())
    same_output = all(record["logical_output_ptr"] == records["direct"]["logical_output_ptr"]
                      for record in records.values())
    row: dict[str, Any] = {"shared_pool": shared_pool, "captures": records,
                           "all_topology_signatures_match": all(s == signatures[0] for s in signatures),
                           "all_native_arguments_match": same_native,
                           "all_logical_output_pointers_match": same_output,
                           "capture_output_bits_exact": True}
    if not shared_pool or (same_native and same_output):
        for label, graph in graphs.items():
            graph.replay()
            require_equal(outputs["direct"] if shared_pool else outputs[label], expected)
        row["replay_output_bits_exact"] = True
        row["paired_timings"] = paired_graph_time(graphs)
    else:
        row["limits"] = ["Shared pool did not produce identical addresses; no address-controlled timing claim"]
    del graphs, outputs, records, graph, operation
    gc.collect()
    torch.cuda.empty_cache()
    return row


def main() -> None:
    """Save both independent-pool and address-controlled attempts, not a new qualification."""
    from mxwave_zipserv import runtime

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Preserve the first topology diagnostic")
    validate_runtime()
    prepared_path = args.fixtures / "prepared.json"
    prepared = json.loads(prepared_path.read_bytes())
    cases = [case for case in prepared["cases"] if case["name"].startswith("kda-")]
    if len(cases) != 2:
        raise ValueError("Require the same two source-bound KDA projections")
    for name, expected in (("config.json", prepared["source_config_sha256"]),
                           ("model.safetensors.index.json", prepared["source_index_sha256"])):
        if hashlib.sha256((args.source / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Source metadata changed")
    torch.set_num_threads(1)
    torch.manual_seed(20261017)
    torch.cuda.set_per_process_memory_fraction(
        1.5 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    report: dict[str, Any] = {"status": "running", "diagnostic_only": True, "cases": [],
        "previous_v1_v2_performance_qualified_remains": False, "started_epoch": time.time(),
        "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()
            ).hexdigest() for name in ("diagnose_kda_graph_topology.py", "kda_direct_dispatch.py",
                                      "kda_runtime.py", "benchmark_kda_direct_dispatch.py")},
        "limits": ["Diagnostic only; neither failed performance qualification is amended.",
                   "Retained raw graphs allow inspecting non-kernel dependencies missing from activity traces.",
                   "A shared-pool address control is interpretable only if all recorded addresses match.",
                   "No modified native kernel, model load or serving adapter integration."]}
    write(args.report, report)
    try:
        for case in cases:
            artifact = args.fixtures / case["artifact"]
            if hashlib.sha256(artifact.read_bytes()).hexdigest() != case["artifact_sha256"]:
                raise ValueError("Source-bound packed fixture changed")
            loaded = load_file(artifact)
            packed = runtime.PackedWeight(tuple(loaded[name].cuda() for name in runtime.BUFFER_NAMES),
                                          *case["native_metadata"])
            del loaded
            restored = runtime.decompress(packed).cpu()
            if tensor_digest(restored) != case["padded_weight_sha256"]:
                raise ValueError("GPU reconstructed weight bytes changed")
            del restored
            rows, width = case["logical_shape"]
            pool = torch.empty((2, packed.m * packed.k), dtype=torch.bfloat16, device="cuda")
            x = torch.randn((4, width), dtype=torch.bfloat16, device="cuda") * .1
            buffers = list(packed.buffers)
            metadata = (packed.m, packed.k, packed.max_common, packed.max_rare, packed.start_exp, rows)
            reference = partial(kda_linear, x, buffers, pool, *metadata)
            operations = {"reference_a": reference, "reference_b": reference,
                          "direct": partial(linear, x, buffers, pool, *metadata)}
            expected = reference()
            require_equal(operations["direct"](), expected)
            stream = torch.cuda.Stream()
            row: dict[str, Any] = {"fixture": case["name"], "stages": []}
            report["cases"].append(row)
            for shared in (False, True):
                label = case["name"] + ("-shared" if shared else "-independent")
                row["stages"].append(inspect_captures(
                    operations, expected, stream, args.report.parent, label, shared_pool=shared))
                write(args.report, report)
            del operations, expected, reference, x, pool, packed, buffers, stream
            gc.collect()
            torch.cuda.empty_cache()
        report.update(status="complete", correctness_passed=True,
                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      finished_epoch=time.time())
        write(args.report, report)
    except BaseException as exc:
        report.update(status="failed", error=repr(exc), finished_epoch=time.time())
        write(args.report, report)
        raise


if __name__ == "__main__":
    main()
