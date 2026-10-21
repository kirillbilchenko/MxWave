"""Check retained evidence integrity and descriptive comparison boundaries."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from integrations.glm53_mxfp4.measure_ct_draft_lengths import summarize_counters
from integrations.glm53_mxfp4.mtp_metrics import delta
from integrations.glm53_mxfp4.review_ct_draft_lengths import digest, review


def evidence(root: Path) -> None:
    """Create a small complete retained run with an intentional cross-depth text difference."""
    def write(name: str, data: object) -> None:
        (root / name).write_text(json.dumps(data))

    depths = {"s7a": 7, "s3": 3, "s5": 5, "s0": 0, "s7b": 7}
    cases = [{"id": f"case-{i}", "domain": ("code", "chat", "explanation")[i % 3],
              "prompt_ids": [10, 11]} for i in range(8)]
    write("plan.json", {"revision": "ct-draft-length-v1", "sealed_test_opened": False,
                       "depths": depths, "arms": list(depths), "calibration_cases": cases,
                       "speed_control_relative_tolerance": .02})
    write("prepared.json", {"plan_sha256": digest(root / "plan.json")})
    write("controller.json", {"phase": "complete_restored", "service_paused": False,
        "restoration": {"health_200": True, "originals_running": True, "models": {
            "data": [{"id": "glm-5.3-flash", "max_model_len": 500000}]}}})
    for arm, depth in depths.items():
        after = ""
        if depth:
            counts = {"num_drafts": 10, "num_draft_tokens": depth * 10,
                      "num_accepted_tokens": depth}
            after = "\n".join(f"vllm:spec_decode_{k}_total  {v}" for k, v in counts.items())
            after += "\n" + "\n".join(
                f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{position="{i}"}} 1'
                for i in range(depth))
        counters = delta("", after)
        rows = [{"case": c["id"], "domain": c["domain"], "trial": trial,
            "result": {"token_ids": [depth] * 256,
                       "usage": {"completion_tokens": 256, "prompt_tokens": 2},
                       "latency_seconds": 256 / (20 + depth), "ttft_seconds": .1,
                       "tpot_seconds": 1 / (20 + depth)},
            "metrics_before": "", "metrics_after": after, "counter_delta": counters,
            "speculation": summarize_counters(counters, depth)}
            for trial in range(6) for c in cases]
        write(f"{arm}-measurement.json", {"status": "complete", "depth": depth, "cases": rows})
        write(f"{arm}-startup.json", [{"rank": rank, "status": "complete",
            "context_probe": {"depth": depth, "drafter_loaded": bool(depth)},
            "trial": {"physical_graph_token_counts": [1, 4, 6, 8], "native_mxfp8_kda": {
                "all_68_rounding_hashes_match_quality_run": True,
                "layers": list(range(68)), "original_BF16_parameters_alive": 0}}}
            for rank in (0, 1)])
    write("evidence-manifest.json", {"files": [{"path": p.name, "bytes": p.stat().st_size,
                                              "sha256": digest(p)} for p in root.iterdir()]})


def test_reports_text_changes_without_calling_them_quality_loss(tmp_path: Path) -> None:
    evidence(tmp_path)
    result = review(tmp_path, tmp_path / "review.json")
    assert all(c["within_2_percent"] for c in result["reload_controls"].values())
    assert all(a["all_repeats_exact"] for a in result["arms"].values())
    assert result["comparisons"]["code"]["s7a"]["tps_vs_no_speculation"] == pytest.approx(.35)
    changed = [r for r in result["output_comparisons"] if r["arm"] == "s3"]
    assert all(not r["tokens_exact"] and r["first_different_output_index"] == 0 for r in changed)
    assert not result["automatic_promotion"]


def test_tampering_blocks_review(tmp_path: Path) -> None:
    evidence(tmp_path)
    (tmp_path / "s3-measurement.json").write_text("{}")
    with pytest.raises(AssertionError):
        review(tmp_path, tmp_path / "review.json")
    assert not (tmp_path / "review.json").exists()


def test_unbound_measurement_blocks_review(tmp_path: Path) -> None:
    evidence(tmp_path)
    path = tmp_path / "evidence-manifest.json"
    manifest = json.loads(path.read_bytes())
    manifest["files"] = [r for r in manifest["files"] if r["path"] != "s3-measurement.json"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(AssertionError, match="Unbound evidence"):
        review(tmp_path, tmp_path / "review.json")
    assert not (tmp_path / "review.json").exists()
