"""Independently verify retained MTP source bindings, timing and full-vocabulary distributions."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    """Hash one evidence payload without retaining unrelated model weights."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric_values(text: str) -> dict[str, float]:
    """Parse relevant native metric series independently of the measurement parser."""
    rows = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.startswith("vllm:spec_decode_"):
            continue
        parts = line.rsplit(None, 1)
        if len(parts) != 2:
            continue
        name, raw = parts
        head, separator, labels = name.partition("{")
        head = head.removesuffix("_total")
        if head not in {"vllm:spec_decode_" + suffix for suffix in (
            "num_drafts", "num_draft_tokens", "num_accepted_tokens", "num_accepted_tokens_per_pos")}:
            continue
        key = head + (separator + labels if separator else "")
        value = float(raw)
        assert key not in rows and math.isfinite(value) and value >= 0
        rows[key] = value
    return rows


def numerical(root: Path, left: str, right: str) -> dict:
    """Recompute live-decode KL/NLL with explicit row normalization and source-byte checks."""
    a, b = [json.loads((root / f"{name}-decode-quality.json").read_bytes()) for name in (left, right)]
    assert a["status"] == b["status"] == "complete"
    assert a["vocabulary"] == b["vocabulary"] and a["generated_tokens"] == b["generated_tokens"] == 4
    assert len(a["contexts"]) == len(b["contexts"]) == 16
    ids_equal, decoded, expected, kls, nlls = True, 0, 0, [], []
    for x, y in zip(a["contexts"], b["contexts"], strict=True):
        assert x["index"] == y["index"] and x["prompt_sha256"] == y["prompt_sha256"]
        ids_equal &= x["token_ids"] == y["token_ids"]
        for i, j in zip(x["steps"], y["steps"], strict=True):
            assert i["position"] == j["position"] and i["kind"] == j["kind"]
            arrays = []
            for item in (i, j):
                file = root / item["file"]
                assert file.name == item["file"] and digest(file) == item["sha256"]
                value = np.fromfile(file, dtype="<f4").astype(np.float64)
                assert len(value) == a["vocabulary"] and not np.isnan(value).any()
                maximum = float(np.max(value))
                arrays.append(value - maximum - math.log(float(np.exp(value - maximum).sum())))
            if i["kind"] != "decode":
                continue
            expected += 1
            if i["conditioning_token_ids"] != j["conditioning_token_ids"]:
                continue
            decoded += 1
            p, q = arrays
            supported = np.isfinite(p)
            assert np.isfinite(q[supported]).all()
            kls.append(max(0., float(np.sum(np.exp(p[supported]) * (p[supported] - q[supported])))))
            token = x["token_ids"][i["position"]]
            nlls.append(float(p[token] - q[token]))
    kl = math.fsum(kls) / len(kls) if kls else None
    nll = math.fsum(nlls) / len(nlls) if nlls else None
    passed = (decoded == expected == 48 and ids_equal and kl is not None and nll is not None
              and kl <= 1e-5 and abs(nll) <= 1e-4)
    return {"reference": left, "candidate": right, "decode_steps": decoded,
        "expected_decode_steps": expected, "all_generated_ids_match": ids_equal,
        "mean_decode_kl": kl, "mean_decode_nll_difference": nll, "passed": passed}



def validate_draft_check(item: dict, depth: int, source_hashes: dict[str, str]) -> None:
    """Require the actual zero-rope source geometry and independently pinned BF16 matrices."""
    if item["status"] != "passed" or item["draft_loaded"] != bool(depth):
        raise ValueError("Draft check status differs from its declared arm")
    records = item["records"]
    if not depth:
        if records:
            raise ValueError("No-drafter control contains draft checks")
        return
    geometry = item["projection_geometry"]
    if ({k: geometry[k] for k in ("q_rank", "kv_rank", "rope_rows", "hidden_size")} !=
            {"q_rank": 1536, "kv_rank": 512, "rope_rows": 0, "hidden_size": 4096}
            or type(geometry["nope"]) is not bool or len(records) != 18):
        raise ValueError("Actual zero-rope draft requires eighteen checks, without zero extension")
    names = [r["name"] for r in records]
    if len(set(names)) != 18 or any(not r["bits_exact"] or
            r["loaded_sha256"] != r["source_reference_sha256"] for r in records):
        raise ValueError("Loaded draft source records must be unique and bit-exact")
    routed = {f"expert{expert}.{name}" for expert in (0, 127, 287)
              for name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale")}
    if {n for n in names if n.startswith("expert")} != routed:
        raise ValueError("Missing routed expert coverage")
    for suffix, shape in (("q", [1536, 4096]), ("kv_source", [512, 4096])):
        rows = [r for r in records if r["name"].endswith(
            ".self_attn.fused_qkv_a_proj." + suffix)]
        if (len(rows) != 1 or rows[0]["shape"] != shape
                or rows[0]["dtype"] != "torch.bfloat16"
                or rows[0]["source_reference_sha256"] != source_hashes[suffix]):
            raise ValueError("Fused projection differs from the actual full CPU source matrix")
    shared = [n for n in names if ".mlp.shared_experts." in n]
    for suffix in ("gate_up_proj.weight", "gate_up_proj.weight_scale",
                   "down_proj.weight", "down_proj.weight_scale"):
        if sum(n.endswith(".mlp.shared_experts." + suffix) for n in shared) != 1:
            raise ValueError("Missing shared expert projection coverage")


def validate_loader_snapshot(item: dict, depth: int) -> None:
    """Require actual loader isolation, full source consumption and zero padding events."""
    storage = item["storage"]
    if item["depth"] != depth or storage["base_scale_layers"] != 42:
        raise ValueError("MTP/base source ownership changed")
    if len(storage["draft_modules"]) != int(bool(depth)):
        raise ValueError("Native draft ownership differs from the declared arm")
    bridge = storage["draft_loader"]
    if (bridge["status"] != "passed" or bridge["bf16_nope_padding"]
            or len(bridge["calls"]) != int(bool(depth))
            or len(bridge["weight_loads"]) != int(bool(depth))):
        raise ValueError("Zero-rope model needs no padding event and exactly one native draft load")
    if depth:
        call = bridge["calls"][0]
        load = bridge["weight_loads"][0]
        if (not call["target_policy_unchanged"] or not call["separate_quant_config"]
                or call["source_config_sha256"] !=
                "194d290ebb0f19aedc60d24ea9e236e33c86616e460f4a108a842f6fd5b8b198"
                or load["source_tensors"] != 1756
                or not load["all_declared_nonvocabulary_parameters_loaded"]
                or not load["full_source_count_required"]):
            raise ValueError("Native MTP loading did not satisfy the full source/isolation contract")

def main() -> None:
    """Verify either a restored failed attempt or every retained measurement in a completed pilot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    inventory = json.loads((root / "evidence-inventory.json").read_bytes())
    for item in inventory["files"]:
        rel = Path(item["path"])
        assert not rel.is_absolute() and ".." not in rel.parts
        path = root / rel
        assert path.stat().st_size == item["bytes"] and digest(path) == item["sha256"]
    controller = json.loads((root / "controller.json").read_bytes())
    restored = json.loads((root / "restoration.json").read_bytes())
    assert controller["phase"] in ("complete", "failed_restored") and not controller["service_paused"]
    assert restored["status"] == "complete" and restored["health"] == 200
    actual = sorted(restored["containers"], key=lambda r: r["rank"])
    assert [{k: v for k, v in item.items() if k != "rank"} for item in actual] == controller["originals"]
    prepared = json.loads((root / "prepared.json").read_bytes())
    bindings = prepared["bindings"]
    parent = json.loads((root / "parent-prepared.json").read_bytes())
    assert len(parent["bindings"][0]["code"]) == 160
    assert len(bindings[0]["code"]) == 174
    changed = {name for name, sha in parent["bindings"][0]["code"].items()
               if bindings[0]["code"].get(name) != sha}
    assert changed == {"quality_worker.py"}
    plan = json.loads((root / "results/mtp-pilot-plan.json").read_bytes())
    assert plan["fresh_control"] and not plan["reuse_previous_timing_control"]
    assert plan["loaded_draft_byte_checks_per_rank"] == 18 and plan["required_padding_events"] == 0
    assert plan["complete_draft_source_tensors"] == 1756
    geometry_proof = json.loads((root / "native-geometry-CPU-proof.json").read_bytes())
    assert digest(root / "native-geometry-CPU-proof.json") == plan["actual_geometry_CPU_proof_sha256"]
    assert geometry_proof["status"] == "passed" and not geometry_proof["cuda_initialized"]
    for name, relative in (("bridge", "mtp_draft_loader.py"), ("audit", "audit_mtp_projection_geometry.py")):
        assert geometry_proof["code_sha256"][name] == bindings[0]["code"][relative]
    source_hashes = {"q": geometry_proof["source_tensors"][0]["bf16_sha256"],
                     "kv_source": geometry_proof["source_tensors"][1]["bf16_sha256"]}
    geometry_plan = json.loads((root / "results/native-geometry-CPU-plan.json").read_bytes())
    assert geometry_plan["source_sha256"]["mtp_pilot_worker.py"] == bindings[0]["code"][
        "mtp_pilot_worker.py"]
    loader_proof = json.loads((root / "results/native-mtp-loader.json").read_bytes())
    assert loader_proof["status"] == "passed" and not loader_proof["cuda_initialized"]
    assert loader_proof["selectors"] == 872 and loader_proof["mismatches_isolated"] == 0
    assert loader_proof["code_sha256"]["loader"] == bindings[0]["code"]["mtp_draft_loader.py"]
    assert loader_proof["fixed_target_policy_byte_pickle_unchanged"]
    for name, sha in prepared["protocol_files_sha256"].items():
        assert digest(root / "results" / name) == sha
    assert digest(root / "source-overlay.tar.gz") == prepared["source_archive_sha256"]
    assert bindings[0] == bindings[1]
    for name, sha in bindings[0]["code"].items():
        assert digest(root / "source" / name) == sha
    result = {"status": "passed", "verified_payloads": len(inventory["files"]),
        "source_files": len(bindings[0]["code"]), "controller_phase": controller["phase"],
        "original_identities_restored": True, "original_health": 200, "quality_qualified": False,
        "fresh_control": True, "all_parent_arithmetic_sources_unchanged": True,
        "changed_parent_hook": sorted(changed), "verifier_sha256": digest(Path(__file__)),
        "native_geometry_CPU_proof_sha256": digest(root / "native-geometry-CPU-proof.json")}
    if controller["phase"] != "complete":
        result.update(measurement_completed=False, retained_error=controller.get("error"))
        reports = root / "results"
        control_file = reports / "control-mtp.json"
        if control_file.exists():
            control = json.loads(control_file.read_bytes())
            if control["status"] == "complete":
                assert len(control["trials"]) == 10 and len(control["warmups"]) == 4
                values = []
                for length in (256, 2048):
                    rows = sorted((r for r in control["trials"] if r["prompt_tokens"] == length),
                                  key=lambda r: r["trial"])
                    assert [r["trial"] for r in rows] == list(range(5))
                    rates = []
                    for row in rows:
                        measured = row["measurement"]
                        assert len(measured["requests"]) == 1
                        assert len(measured["requests"][0]["token_ids"]) == 64
                        rate = 64 / measured["wall_seconds"]
                        assert math.isclose(rate, measured["aggregate_tokens_per_second"])
                        rates.append(rate)
                    values.append({"prompt_tokens": length, "median_tps": statistics.median(rates)})
                quality = numerical(reports, "control-a", "control-b")
                result.update(control_timed_requests=10, control_timed_tokens=640,
                    control_workloads=values, control_repeat={
                        k: round(v, 12) if isinstance(v, float) else v for k, v in quality.items()})
    else:
        reports = root / "results"
        summary = json.loads((reports / "quality-model-review.json").read_bytes())
        assert summary["status"] == "complete" and not summary["quality_qualified"]
        arms = {name: json.loads((reports / f"{name}-mtp.json").read_bytes())
                for name in ("control", "candidate")}
        totals = {"timed_requests": 0, "timed_tokens": 0, "warmup_requests": 0, "warmup_tokens": 0}
        workloads = []
        for length in (256, 2048):
            measurements = {}
            for name, arm in arms.items():
                assert arm["status"] == "complete" and len(arm["trials"]) == 10
                rows = sorted([r for r in arm["trials"] if r["prompt_tokens"] == length],
                              key=lambda r: r["trial"])
                assert [r["trial"] for r in rows] == list(range(5))
                tps, ttft, ids = [], [], []
                for row in rows:
                    value = row["measurement"]
                    assert len(value["requests"]) == 1
                    req = value["requests"][0]
                    assert len(req["token_ids"]) == 64 and sum(a["tokens"] for a in req["arrivals"]) == 64
                    assert 0 < req["ttft_seconds"] <= req["elapsed_seconds"] <= value["wall_seconds"]
                    assert req["arrivals"][0]["seconds"] == req["ttft_seconds"]
                    assert req["usage"]["completion_tokens"] == 64
                    assert req["usage"]["prompt_tokens"] == length
                    rate = 64 / value["wall_seconds"]
                    assert math.isclose(rate, value["aggregate_tokens_per_second"], rel_tol=1e-12)
                    tps.append(rate); ttft.append(req["ttft_seconds"]); ids.append(req["token_ids"])
                    totals["timed_requests"] += 1; totals["timed_tokens"] += 64
                measurements[name] = {"tps": statistics.median(tps),
                    "ttft": statistics.median(ttft), "ids": ids}
            before, after = measurements["control"], measurements["candidate"]
            changes = {"prompt_tokens": length, "control_tps": before["tps"],
                "mtp_tps": after["tps"], "tps_change_percent": 100 * (after["tps"] / before["tps"] - 1),
                "ttft_change_percent": 100 * (after["ttft"] / before["ttft"] - 1),
                "all_timed_ids_match": before["ids"] == after["ids"]}
            declared = next(r for r in summary["workloads"] if r["prompt_tokens"] == length)
            for field in ("tps_change_percent", "ttft_change_percent"):
                assert math.isclose(changes[field], declared[field], abs_tol=1e-10)
            assert changes["all_timed_ids_match"] == declared["all_timed_ids_match"]
            workloads.append(changes)
        counter_deltas = {}
        source_groups = 0
        source_audit = json.loads((reports / "source-audit.json").read_bytes())
        for name, arm in arms.items():
            assert len(arm["warmups"]) == 4
            assert {(w["prompt_tokens"], w["index"]) for w in arm["warmups"]} == {
                (length, index) for length in (256, 2048) for index in (0, 1)}
            for warm in arm["warmups"]:
                assert len(warm["measurement"]["requests"]) == 1
                assert len(warm["measurement"]["requests"][0]["token_ids"]) == 16
                totals["warmup_requests"] += 1; totals["warmup_tokens"] += 16
            values = [metric_values((reports / f"{name}-metrics-{phase}-timing.txt").read_text())
                      for phase in ("before", "after")]
            assert set(values[0]) <= set(values[1])
            counter_deltas[name] = {k: v - values[0].get(k, 0.) for k, v in values[1].items()}
            assert all(v >= 0 for v in counter_deltas[name].values())
            assert counter_deltas[name] == arm["timed_speculative_counter_delta"]
            for field in ("source_before", "source_after", "base_loaded_checks", "draft_loaded_checks"):
                assert len(arm[field]) == 2 and all(r["status"] == "passed" for r in arm[field])
                assert {r.get("tp_rank", r.get("rank")) for r in arm[field]} == {0, 1}
            for field in ("source_before", "source_after"):
                for row in arm[field]:
                    expected = Counter((tuple(case["shape"]), case["raw_sha256"])
                        for case in source_audit["cases"] if case["rank"] == row["tp_rank"])
                    actual = Counter((tuple(case["shape"]), case["sha256"])
                                     for case in row["source_scale_hashes"])
                    assert row["candidate"] and expected == actual
                    source_groups += sum(actual.values())
            for item in arm["draft_loaded_checks"]:
                validate_draft_check(item, 0 if name == "control" else 2, source_hashes)
            for phase in ("snapshot_before", "snapshot_after"):
                for item in arm[phase]:
                    validate_loader_snapshot(item, 0 if name == "control" else 2)
        comparisons = [numerical(reports, left, right) for left, right in (
            ("control-a", "control-b"), ("candidate-a", "candidate-b"),
            ("control-a", "candidate-a"), ("control-b", "candidate-b"))]
        for observed, declared in zip(comparisons, summary["numerical_comparisons"], strict=True):
            assert observed["passed"] == declared["passed"]
            for field in ("mean_decode_kl", "mean_decode_nll_difference"):
                if observed[field] is None:assert declared[field] is None
                else:assert math.isclose(observed[field], declared[field], abs_tol=2e-12)
        assert summary["pilot_numerical_passed"] == (
            all(c["passed"] for c in comparisons) and all(w["all_timed_ids_match"] for w in workloads))
        assert source_groups == 672
        distribution_files = [r["path"] for r in inventory["files"]
                              if r["path"].startswith("results/") and r["path"].endswith(".f32")]
        assert len(distribution_files) == 256
        assert any(key.split("{")[0] == "vllm:spec_decode_num_drafts" and value > 0
                   for key, value in counter_deltas["candidate"].items())
        result.update(measurement_completed=True, **totals, workloads=workloads,
            source_scale_groups_verified=source_groups,
            full_vocabulary_distribution_files=len(distribution_files),
            numerical_comparisons=[{k: round(v, 12) if isinstance(v, float) else v
                                    for k, v in item.items()} for item in comparisons],
            counter_deltas=counter_deltas, pilot_numerical_passed=summary["pilot_numerical_passed"])
    profiles = []
    for row in inventory["files"]:
        relative = row["path"]
        if (not relative.startswith("traces/") or not Path(relative).name.startswith("dp0_pp0_tp")
                or not relative.endswith((".pt.trace.json.gz", ".pt.trace.json"))):
            continue
        path = root / relative
        with (gzip.open(path, "rt") if relative.endswith(".gz") else path.open()) as handle:
            events = json.load(handle)["traceEvents"]
        kernels = [e for e in events if e.get("cat") == "kernel"]
        rebuilds = [e for e in kernels if "BF16TripleBitmap_Decompress_Kernel<" in e.get("name", "")]
        profiles.append({"path": relative, "sha256": row["sha256"],
            "kernel_events": len(kernels), "gpu_kernel_duration_sum_us": sum(e["dur"] for e in kernels),
            "bf16_reconstruction_kernels": len(rebuilds),
            "bf16_reconstruction_duration_sum_us": sum(e["dur"] for e in rebuilds),
            "cuda_graph_launches": sum(e.get("cat") == "cuda_runtime" and
                e.get("name", "").startswith("cudaGraphLaunch") for e in events)})
        del events, kernels, rebuilds
    result.update(worker_profiles=profiles, all_four_worker_profiles_present=len(profiles) == 4,
        profile_duration_sums_are_not_end_to_end_or_cross_rank_critical_path=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("workloads", "numerical_comparisons", "counter_deltas",
                                   "worker_profiles")}))


if __name__ == "__main__":
    main()
