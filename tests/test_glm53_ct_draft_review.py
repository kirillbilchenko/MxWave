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


def native_evidence(root: Path, *, reload_slow: bool = False) -> None:
    """Build native pilot evidence with a speed gain and intentional output differences."""
    evidence(root)
    plan = json.loads((root / "plan.json").read_bytes())
    cases = plan["calibration_cases"]
    cases += [{**c, "id": c["id"] + "-extra"} for c in cases]
    arms = ["original_a", "adapted", "original_b"]
    plan.update(revision="drafter-modal-native-v1", depths=dict.fromkeys(arms, 5), arms=arms,
                trials=6, requests=96, selected_step=16,
                variants={a: "modal_fp8_step16" if a == "adapted" else "modal_fp8" for a in arms},
                drafter_files={"modal_fp8": {"model.safetensors": "original"},
                               "modal_fp8_step16": {"model.safetensors": "adapted"}},
                development_thresholds={"code_geomean_tps_gain": .08,
                                        "maximum_chat_explanation_tps_regression": .02})
    (root / "plan.json").write_text(json.dumps(plan))
    (root / "measure_features.py").write_text("# frozen native client\n")
    (root / "prepared.json").write_text(json.dumps({"plan_sha256": digest(root / "plan.json")}))
    for arm in arms:
        data = json.loads((root / "s5-measurement.json").read_bytes())
        data["cases"] += [{**r, "case": r["case"] + "-extra"} for r in data["cases"]]
        data.update(plan_sha256=digest(root / "plan.json"),
                    source_sha256=digest(root / "measure_features.py"))
        for row in data["cases"]:
            # Deep-copy before mutation: avoid altering another prompt's result dictionary.
            row["result"] = dict(row["result"])
            if arm == "adapted":
                row["result"]["token_ids"] = [6] * 256
                row["result"]["latency_seconds"] /= 1.1
            if reload_slow and arm == "original_b" and row["case"] == "case-0":
                row["result"]["latency_seconds"] *= 1.2
        (root / f"{arm}-measurement.json").write_text(json.dumps(data))
        startup = json.loads((root / "s5-startup.json").read_bytes())
        for item in startup:
            variant = plan["variants"][arm]
            item["context_probe"].update(checkpoint_files_verified=True, fc_native_bitwise_equal=True,
                variant=variant, checkpoint_files=plan["drafter_files"][variant],
                quant_config="Fp8Config", draft_layers=6, context_width=24576, causal=True,
                aux_layers=[24, 28, 32, 36, 40, 44])
        (root / f"{arm}-startup.json").write_text(json.dumps(startup))
    (root / "evidence-manifest.json").unlink()
    (root / "evidence-manifest.json").write_text(json.dumps({"files": [
        {"path": p.name, "bytes": p.stat().st_size, "sha256": digest(p)} for p in root.iterdir()]}))


@pytest.mark.parametrize("reload_slow", [False, True])
def test_native_development_gate_keeps_prompt_drift_and_generalization_limits(
    tmp_path: Path, reload_slow: bool,
) -> None:
    native_evidence(tmp_path, reload_slow=reload_slow)
    result = review(tmp_path, tmp_path / "review.json")
    gate = result["development_gate"]
    assert gate["code_gain_pass"] and gate["original_reload_exact"]
    assert gate["passed"] is (not reload_slow)
    assert not gate["generalization_proven"] and not result["automatic_promotion"]
    assert result["comparisons"]["code"]["adapted"]["paired_geomean_tps_gain"] > .08
    assert all(not row["tokens_exact"] for row in result["output_comparisons"]
               if row["arm"] == "adapted")
