"""Independently check the direct expert replay against the original component tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .audit_drafter_feature_export_numpy import arrays
from .drafter_fp8_roundtrip import digest


def review(root: Path, components: Path, output: Path) -> dict[str, Any]:
    """Require restoration and exact stock/raw/packed matches to the retained native operation."""
    assert not output.exists()
    manifest = json.loads((root / "evidence-manifest.json").read_bytes())
    for row in manifest["files"]:
        p = root / row["path"]
        assert p.stat().st_size == row["bytes"] and digest(p) == row["sha256"]
    state = json.loads((root / "controller.json").read_bytes())
    prepared = json.loads((root / "prepared.json").read_bytes())
    assert state["phase"] == "complete_restored" and not state["service_paused"]
    restored = state["restoration"]
    assert restored["health_200"] and restored["originals_running"]
    assert restored["live"] == prepared["live"] and restored["originals"] == prepared["originals"]
    assert any(x["max_model_len"] == 500000 for x in restored["models"]["data"])
    report = json.loads((root / "result/report.json").read_bytes())
    assert report["status"] == "complete"
    assert report["source_sha256"] == digest(root / "replay_drafter_expert.py")
    assert report["plan_sha256"] == digest(root / "plan.json") == prepared["plan_sha256"]
    assert report["outputs_sha256"] == digest(root / "result/outputs.safetensors")
    results = arrays(root / "result/outputs.safetensors")
    comparisons, matches, repeat_checks = [], 0, 0
    source_proofs = []
    for rank in (0, 1):
        r = report["ranks"][rank]
        assert r["rank"] == rank and r["scale_bytes_exact"]
        assert len(r["cases"]) == 12 and all(x["retained_match"]["all_bits_exact"] for x in r["cases"])
        assert len(r["repeat_checks"]) == 2
        for check in r["repeat_checks"]:
            assert check["comparisons"] == 120 and check["all_bitwise_exact"]
            repeat_checks += check["comparisons"]
        proof = json.loads((root / f"result/weight-sources-r{rank}.json").read_bytes())
        assert len(proof) == 1152
        assert len({x["name"] for x in proof}) == 1152
        source_proofs.append(proof)
        for position in (46, 54):
            number = 4 if position == 46 else 5
            for arm in ("a1", "b1"):
                path = components / f"{arm}-r{rank}" / (
                    f"context-opencoder-evol-15654-capture1-{number:04d}-r{rank}.safetensors"
                )
                meta = json.loads(path.with_suffix(".json").read_bytes())
                assert digest(path) == meta["payload_sha256"]
                source = arrays(path)
                event = next(x for x in meta["components"]["events"]
                             if x["name"] == "layer4.moe_gemm#0/output")
                expected = source[event["tensors"][0]["payload_key"]]
                prefix = f"r{rank}_{position}_{arm}"
                assert results[prefix + "_retained"].tobytes() == expected.tobytes()
                for mode in ("stock", "raw", "packed"):
                    assert results[prefix + "_" + mode].tobytes() == expected.tobytes()
                    matches += 1
            for mode in ("stock", "raw", "packed"):
                a, b = (results[f"r{rank}_{position}_{arm}_{mode}"].reshape(8, 8, 2048)
                        for arm in ("a1", "b1"))
                shared = 8 if position == 46 else 7
                count = np.count_nonzero(a != b, axis=(1, 2)).tolist()
                expected_counts = ([0] * 8 if position == 46 else
                                   [0, 1, 0, 0, 0, 0, 0] if rank == 0 else
                                   [0, 0, 0, 0, 2, 0, 0])
                assert count[:shared] == expected_counts
                comparisons.append({"rank": rank, "position": position, "mode": mode,
                                    "shared_rows": shared, "different_elements_per_row": count})
    assert source_proofs[0] == source_proofs[1], "TP partitions must read identical checkpoint bytes"
    result = {
        "status": "verified", "payloads_verified": len(manifest["files"]),
        "source_sha256": digest(Path(__file__)), "native_output_matches": matches,
        "reported_exact_graph_repeat_checks": repeat_checks,
        "actual_component_seconds": report["finished_epoch"] - report["started_epoch"],
        "comparisons": comparisons,
        "conclusion": "Stock, raw bridge and compressed bridge exactly reproduce the same shared-row sensitivity.",
        "limits": [
            "One layer-4 gate/up GEMM, two TP partitions replayed separately on one GPU.",
            "This excludes scale compression as the cause of this observed first difference.",
            "It does not prove every later operation invariant or qualify the adapted drafter.",
            "The precise reduction-partition mechanism and a correction remain to be tested.",
        ],
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    """Review the archived replay without PyTorch or a running service."""
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "components", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(review(args.root, args.components, args.output), indent=2))


if __name__ == "__main__":
    main()
