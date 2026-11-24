"""Build CPU-only wider routing fixtures from the retained first expert divergence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .audit_drafter_feature_export_numpy import arrays
from .drafter_fp8_roundtrip import digest


def recover_topk(
    tokens: np.ndarray, experts: np.ndarray, count: int, *, rows: int, topk: int = 8,
    block: int = 8, num_experts: int = 288,
) -> np.ndarray:
    """Recover each real route exactly once, rejecting missing, duplicated or invalid entries."""
    routes = rows * topk
    if (min(rows, topk, block, num_experts) < 1 or count < 1 or count % block
            or tokens.ndim != 1 or experts.ndim != 1
            or tokens.dtype != np.int32 or experts.dtype != np.int32
            or count > len(tokens) or count // block > len(experts)):
        raise ValueError("Invalid routing metadata geometry")
    consumed = tokens[:count]
    real = consumed < routes
    if np.any(consumed < 0) or np.any(consumed[~real] != routes):
        raise ValueError("Invalid token sentinel")
    route_ids = consumed[real]
    if not np.array_equal(np.sort(route_ids), np.arange(routes)):
        raise ValueError("Every real route must occur exactly once")
    route_experts = np.repeat(experts[:count // block], block)[real]
    if np.any(route_experts < 0) or np.any(route_experts >= num_experts):
        raise ValueError("Real route has an invalid expert")
    result = np.empty(routes, dtype=np.int32)
    result[route_ids] = route_experts
    return result.reshape(rows, topk)


def expand(
    x: np.ndarray, ids: np.ndarray, weights: np.ndarray, rows: int, pattern: str,
) -> dict[str, np.ndarray]:
    """Preserve the original eight rows and vary only appended expert assignments."""
    if (x.shape != (8, 4096) or x.dtype != np.uint16
            or ids.shape != (8, 8) or ids.dtype != np.int32
            or weights.shape != (8, 8) or weights.dtype != np.float32
            or rows not in (8, 16, 32, 64, 128) or pattern not in ("repeat", "spread")):
        raise ValueError("Expected the exact eight-row BF16 expert fixture and bounded extension")
    if not np.isfinite((x.astype(np.uint32) << 16).view(np.float32)).all():
        raise ValueError("Non-finite source input")
    if not np.isfinite(weights).all() or np.any(ids < 0) or np.any(ids >= 288):
        raise ValueError("Invalid source routes")
    index = np.arange(rows) % 8
    routed = ids[index].copy()
    if pattern == "spread":
        # A whole-group rotation preserves each row's eight distinct expert IDs.
        routed = ((routed + (np.arange(rows) // 8 * 37)[:, None]) % 288).astype(np.int32)
    result = {"input_bf16_bits": x[index].copy(), "topk_ids": routed,
              "topk_weights": weights[index].copy()}
    for name, reference in (("input_bf16_bits", x), ("topk_ids", ids),
                            ("topk_weights", weights)):
        assert result[name][:8].tobytes() == reference.tobytes()
    return result


def prepare(evidence: Path, output: Path) -> dict[str, Any]:
    """Export synthetic batch-layout stress inputs while retaining their real source provenance."""
    output.mkdir(parents=True, exist_ok=False)
    sources, files = [], []
    original = {}
    for rank in (0, 1):
        for arm in ("a1", "b1"):
            path = evidence / f"{arm}-r{rank}" / (
                f"context-opencoder-evol-15654-capture1-0005-r{rank}.safetensors"
            )
            meta = json.loads(path.with_suffix(".json").read_bytes())
            assert digest(path) == meta["payload_sha256"]
            tensors = arrays(path)
            assert int(tensors["target_positions"][0]) == 54
            events = {e["name"]: e for e in meta["components"]["events"]}
            event = events["layer4.moe_gemm#0/input"]
            values = {leaf["path"]: tensors[leaf["payload_key"]] for leaf in event["tensors"]}
            args = event["parameters"]
            assert (args["size_m"], args["size_n"], args["size_k"],
                    args["top_k"], args["moe_block_size"]) == (8, 2048, 4096, 8, 8)
            count = int(values["/13"][0])
            ids = recover_topk(values["/11"], values["/12"], count, rows=8)
            original[arm, rank] = (values["/0"], ids, values["/14"])
            sources.append({"arm": arm, "rank": rank, "path": str(path),
                            "sha256": digest(path), "metadata_sha256": digest(
                                path.with_suffix(".json")), "original_dispatch_tokens": count})
            for rows in (8, 16, 32, 64, 128):
                for pattern in (("repeat",) if rows == 8 else ("repeat", "spread")):
                    data = expand(*original[arm, rank], rows, pattern)
                    target = output / f"r{rank}-{arm}-m{rows}-{pattern}.npz"
                    np.savez(target, **data)
                    files.append({
                        "path": target.name, "bytes": target.stat().st_size,
                        "sha256": digest(target), "rank": rank, "arm": arm,
                        "rows": rows, "pattern": pattern,
                        "unique_experts": int(np.unique(data["topk_ids"]).size),
                    })
    for rank in (0, 1):
        a, b = original["a1", rank], original["b1", rank]
        assert all(x[:7].tobytes() == y[:7].tobytes() for x, y in zip(a, b, strict=True))
        assert a[0][7].tobytes() != b[0][7].tobytes()
    report = {
        "status": "cpu_prepared", "source_sha256": digest(Path(__file__)),
        "sources": sources, "files": files, "shared_prefix_rows": 7,
        "first_eight_rows_unchanged": True, "gpu_executed": False,
        "limits": [
            "Appended rows are deterministic synthetic stress inputs built from real captured operands.",
            "This is an expert-component batch-layout fixture, not full-model C8/C16 qualification.",
            "GPU planner, routing alignment, masked padding, equality and timing checks remain required.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    """Prepare wider component fixtures without using any GPU or sealed data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.evidence, args.output)
    print(json.dumps({"status": result["status"], "fixtures": len(result["files"]),
                      "bytes": sum(f["bytes"] for f in result["files"]),
                      "gpu_executed": False}, indent=2))


if __name__ == "__main__":
    main()
