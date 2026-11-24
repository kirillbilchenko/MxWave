"""Screen live production acceptance on the frozen CT development prompts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import re
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any


def metrics(text: str) -> dict[str, float]:
    """Read counters and activity gauges without collecting credentials or prompts."""
    result = {}
    for line in text.splitlines():
        if line.startswith("vllm:") and not line.startswith("vllm:spec_decode_", 0):
            keep = any(line.startswith("vllm:" + name) for name in (
                "request_success_total", "num_requests_running", "num_requests_waiting{",
            ))
        else:
            keep = line.startswith("vllm:spec_decode_") and "_total{" in line
        if keep:
            key, value = line.rsplit(" ", 1)
            result[key] = float(value)
    return result


def total(values: dict[str, float], name: str) -> float:
    """Sum a named metric across its labeled series."""
    return sum(v for k, v in values.items() if k.split("{", 1)[0] == "vllm:" + name)


def acceptance(values: dict[str, float]) -> dict[str, Any]:
    """Distinguish accepted/proposed rate from first-position prefix survival."""
    rounds = total(values, "spec_decode_num_drafts_total")
    proposed = total(values, "spec_decode_num_draft_tokens_total")
    accepted = total(values, "spec_decode_num_accepted_tokens_total")
    positions = {int(re.search(r'position="(\d+)"', k)[1]): v
                 for k, v in values.items()
                 if k.startswith("vllm:spec_decode_num_accepted_tokens_per_pos_total{")}
    if rounds <= 0 or proposed != 7 * rounds or not 0 <= accepted <= proposed:
        raise ValueError("Expected seven proposals per native C1 draft round")
    if set(positions) != set(range(7)) or sum(positions.values()) != accepted:
        raise ValueError("Per-position counters do not reconcile with accepted tokens")
    if any(not 0 <= v <= rounds for v in positions.values()):
        raise ValueError("Invalid per-position count")
    if any(positions[i] < positions[i + 1] for i in range(6)):
        raise ValueError("Accepted prefix counts must be nonincreasing")
    return {"draft_rounds": rounds, "proposed_tokens": proposed, "accepted_tokens": accepted,
            "acceptance_rate": accepted / proposed, "accepted_per_round": accepted / rounds,
            "prefix_survival": [positions[i] / rounds for i in range(7)]}


def main() -> None:
    """Send ordinary requests to the unchanged service; retain raw counter evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reference = args.reference
    manifest = json.loads((reference / "source-manifest.json").read_bytes())
    for name in ("plan.json", "measure_serving.py"):
        data = (reference / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == manifest[name]
        (args.output / name).write_bytes(data)
    plan = json.loads((reference / "plan.json").read_bytes())
    assert plan["revision"] == "drafter-pretrained-modal-v1"
    assert not plan["sealed_test_opened"] and len(plan["calibration_cases"]) == 8
    assert all(c["split"] == "validation" for c in plan["calibration_cases"])
    spec = importlib.util.spec_from_file_location("frozen_client", args.output / "measure_serving.py")
    assert spec is not None and spec.loader is not None
    client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)
    client.BASE = "http://127.0.0.1:8000"
    container = "local-spark-glm53-tp2-long500k-20261013-r0"

    def identity() -> dict[str, Any]:
        info = json.loads(subprocess.check_output(["docker", "inspect", container]))[0]
        cmd = info["Config"]["Cmd"]
        flags = ("--speculative-config", "--max-model-len", "--max-num-seqs",
                 "--max-num-batched-tokens")
        return {"container_id": info["Id"], "image": info["Image"],
                "started_at": info["State"]["StartedAt"], "running": info["State"]["Running"],
                "args": {f: cmd[cmd.index(f) + 1] for f in flags if f in cmd},
                "prefix_caching": "--enable-prefix-caching" in cmd}

    def idle_snapshot() -> tuple[str, dict[str, float]]:
        raw = client.request("/metrics").decode()
        values = metrics(raw)
        active = [v for k, v in values.items() if k.startswith(
            ("vllm:num_requests_running{", "vllm:num_requests_waiting{"))]
        assert active and not any(active), "Production is busy; stop this diagnostic"
        return raw, values

    report: dict[str, Any] = {
        "status": "running", "started_epoch": time.time(), "identity_before": identity(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference_plan_sha256": manifest["plan.json"], "cases": [], "warmups": [],
        "sealed_test_opened": False, "serving_configuration_changed": False,
        "speed_qualified": False,
        "limits": ["Eight development prompts, two cycles; not a broad quality benchmark.",
                   "Production and CT have different targets, outputs and serving tuning.",
                   "Acceptance comparison describes complete setups, not compression alone."],
    }
    destination = args.output / "measurement.json"
    client.write(destination, report)
    try:
        report["models"] = json.loads(client.request("/v1/models"))
        assert any(m["max_model_len"] == 500000 for m in report["models"]["data"])
        for _ in range(2):
            idle_snapshot()
            report["warmups"].append(client.stream(plan["warmup_case"]["prompt_ids"], 256))
        order = list(plan["calibration_cases"])
        rng = random.Random(42)
        for trial in range(2):
            rng.shuffle(order)
            for case in order:
                before_raw, before = idle_snapshot()
                result = client.stream(case["prompt_ids"], 256)
                for _ in range(40):
                    after_raw, after = idle_snapshot()
                    count = total(after, "request_success_total") - total(
                        before, "request_success_total")
                    if count:
                        break
                    time.sleep(0.25)
                assert count == 1, "Another request or delayed counters contaminate this sample"
                assert set(before) <= set(after), "Metric series disappeared"
                delta = {k: v - before.get(k, 0) for k, v in after.items()}
                assert all(v >= 0 for v in delta.values()), "Counter reset"
                row = {"case": case["id"], "domain": case["domain"], "trial": trial,
                       "result": result, "counter_delta": delta,
                       "acceptance": acceptance(delta),
                       "metrics_before": before_raw, "metrics_after": after_raw}
                report["cases"].append(row)
                client.write(destination, report)
                print(json.dumps({"completed": len(report["cases"]), "case": case["id"],
                                  "rate": row["acceptance"]["acceptance_rate"]}), flush=True)
        report["summary"] = {}
        for domain in ("code", "chat", "explanation"):
            rows = [r for r in report["cases"] if r["domain"] == domain]
            merged = {k: sum(r["counter_delta"].get(k, 0) for r in rows)
                      for r in rows for k in r["counter_delta"]}
            report["summary"][domain] = {
                **acceptance(merged), "requests": len(rows),
                "median_tps": statistics.median(256 / r["result"]["latency_seconds"] for r in rows),
                "median_ttft": statistics.median(r["result"]["ttft_seconds"] for r in rows),
            }
        report["repeats"] = [{"case": c["id"],
            "tokens_exact": (rows := [r for r in report["cases"] if r["case"] == c["id"]])[0][
                "result"]["token_ids"] == rows[1]["result"]["token_ids"],
            "counters_exact": rows[0]["counter_delta"] == rows[1]["counter_delta"],
        } for c in plan["calibration_cases"]]
        report["identity_after"] = identity()
        assert report["identity_after"] == report["identity_before"]
        client.request("/health")
        report.update(status="complete", completed_epoch=time.time(), health_200=True)
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        client.write(destination, report)


if __name__ == "__main__":
    main()
