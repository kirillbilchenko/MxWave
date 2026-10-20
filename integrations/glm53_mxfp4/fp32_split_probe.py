"""Frozen precision gates for separate FP32 split-K component qualification."""

from __future__ import annotations

import array
import math
import struct
import sys
from typing import Any


def protocol() -> dict[str, Any]:
    """Retain historical numerical gates before spending time on performance qualification."""
    return {"revision": "fp32splitv1", "ranks": [0, 1], "layers": [0, 22, 44],
        "kinds": ["input", "output"], "tokens": [1, 4, 8, 16], "splits": [2, 4, 8],
        "activation_seeds": [787, 797, 1909], "activation_cases": ["787", "797", "1909", "basis"],
        "relative_l2_limit": .001, "stock_bits_required_at_tokens": [4, 8],
        "basis_bits_required": True, "rebuilt_control_bits_required": True,
        "graph_replays": 8, "CUDA_allocation_cap_bytes": 2 * 1024**3,
        "component_timeout_seconds": 900, "container_memory_bytes": 8 * 1024**3,
        "model_loaded": False, "timing_qualification_in_this_stage": False,
        "automatic_promotion_allowed": False}


def workspace_bytes(rows: int, tokens: int, splits: int) -> int:
    """Bound the prototype workspace to its explicitly supported padded geometries."""
    if rows not in (4096, 12608) or tokens not in (8, 16) or splits not in (2, 4, 8):
        raise ValueError("Unsupported FP32 diagnostic geometry")
    return rows * tokens * splits * 4


def bf16_metrics(actual: bytes, expected: bytes, *, exact: bool) -> dict[str, Any]:
    """Recompute errors directly from retained little-endian BF16 bytes on either host."""
    if not actual or len(actual) != len(expected) or len(actual) % 2:
        raise ValueError("Require equal nonempty BF16 arrays")
    words = []
    for data in (actual, expected):
        values = array.array("H")
        values.frombytes(data)
        if sys.byteorder != "little":
            values.byteswap()
        words.append(values)
    errors, squares = [], []
    finite = True
    for a, b in zip(*words, strict=True):
        af = struct.unpack("<f", struct.pack("<I", a << 16))[0]
        bf = struct.unpack("<f", struct.pack("<I", b << 16))[0]
        finite &= math.isfinite(af) and math.isfinite(bf)
        errors.append((af - bf) ** 2)
        squares.append(bf * bf)
    relative = math.sqrt(math.fsum(errors) / max(math.fsum(squares), 1e-60)) if finite else None
    bits = actual == expected
    return {"finite": finite, "bits_exact": bits, "relative_l2": relative,
        "different_words": sum(a != b for a, b in zip(*words, strict=True)),
        "requires_bits_exact": exact,
        "passed": bool(finite and (bits if exact else relative is not None and relative <= .001))}


def assess(report: dict[str, Any]) -> dict[str, Any]:
    """Reject missing fixtures/checks and preserve failures independently for every split."""
    p = protocol()
    rank = report["rank"]
    if report["status"] != "complete" or report["protocol"] != p or rank not in p["ranks"]:
        raise ValueError("Incomplete or altered precision protocol")
    expected = [f"kda-{kind}-l{layer}-r{rank}" for layer in p["layers"] for kind in p["kinds"]]
    if [c["name"] for c in report["cases"]] != expected:
        raise ValueError("Missing source-bound fixtures")
    results = {str(s): {"checks": 0, "failed_checks": [], "passed": True} for s in p["splits"]}
    for case in report["cases"]:
        if not all(case[k] for k in ("source_bits_exact", "decoder_bits_exact",
                                      "packed_buffers_unchanged")):
            raise ValueError("Source storage changed")
        checks = case["checks"]
        coverage = [(n, a, s) for n in p["tokens"] for a in p["activation_cases"]
                    for s in p["splits"]]
        if [(r["tokens"], r["activation"], r["split"]) for r in checks] != coverage:
            raise ValueError("Missing precision cases")
        for row in checks:
            for field in ("rebuilt_matches_original", "within_arm_repeats_exact",
                          "graph_matches_eager", "guards_unchanged", "invalid_workspace_rejected"):
                if not row[field]:
                    raise ValueError(f"Execution invariant failed: {field}")
            if row["graph_retained_bytes"] != 0:
                raise ValueError("Graph replay retained new allocation")
            error = row["stock_comparison"]
            exact = row["tokens"] in p["stock_bits_required_at_tokens"] or row["activation"] == "basis"
            passed = bool(error["finite"] and (error["bits_exact"] if exact else
                error["relative_l2"] is not None and error["relative_l2"] <= p["relative_l2_limit"]))
            if error["requires_bits_exact"] != exact or error["passed"] != passed:
                raise ValueError("Numerical gate altered")
            item = results[str(row["split"])]
            item["checks"] += 1
            item["passed"] &= passed
            if not passed:
                item["failed_checks"].append({"name": case["name"], "tokens": row["tokens"],
                    "activation": row["activation"], "relative_l2": error["relative_l2"],
                    "different_words": error["different_words"]})
    return {"diagnostic_complete": True, "rank": rank, "splits": results,
        "any_split_passed": any(r["passed"] for r in results.values()),
        "performance_qualified": False, "model_quality_qualified": False}
