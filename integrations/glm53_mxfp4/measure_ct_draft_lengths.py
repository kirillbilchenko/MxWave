"""Measure native CT draft depths without requiring outputs to match across depths."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import re
import statistics
import time
from pathlib import Path
from typing import Any

from .mtp_metrics import delta


def summarize_counters(values: dict[str, float], depth: int) -> dict[str, Any]:
    """Validate native proposal counts, including the truly speculation-free control."""
    def total(name: str) -> float:
        return sum(v for k, v in values.items() if k.split("{", 1)[0] == "vllm:spec_decode_" + name)

    rounds, proposed, accepted = (total(n) for n in (
        "num_drafts", "num_draft_tokens", "num_accepted_tokens"))
    if not depth:
        if any(values.values()):
            raise ValueError("Speculation-off arm reported speculative work")
        return {"draft_rounds": 0, "proposed": 0, "accepted": 0,
                "acceptance_rate": None, "accepted_per_round": None, "prefix_survival": []}
    if rounds <= 0 or proposed != rounds * depth or not 0 <= accepted <= proposed:
        raise ValueError("Actual native proposal depth differs from declared arm")
    positions = {}
    for key, value in values.items():
        if key.startswith("vllm:spec_decode_num_accepted_tokens_per_pos{"):
            match = re.search(r'position="(\d+)"', key)
            if match is None:
                raise ValueError("Missing position label")
            position = int(match[1])
            positions[position] = positions.get(position, 0.) + value
    if (set(positions) != set(range(depth)) or sum(positions.values()) != accepted
            or any(not 0 <= v <= rounds for v in positions.values())
            or any(positions[i] < positions[i + 1] for i in range(depth - 1))):
        raise ValueError("Prefix-survival counters do not reconcile")
    return {"draft_rounds": rounds, "proposed": proposed, "accepted": accepted,
            "acceptance_rate": accepted / proposed, "accepted_per_round": accepted / rounds,
            "prefix_survival": [positions[i] / rounds for i in range(depth)]}


def main() -> None:
    """Retain six shuffled cycles, exact-repeat checks, timing and raw metric snapshots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    args = parser.parse_args()
    root = Path(__file__).parent
    plan = json.loads((root / "plan.json").read_bytes())
    depth = plan["depths"][args.arm]
    assert plan["revision"] == "ct-draft-length-v1" and not plan["sealed_test_opened"]
    assert len(plan["calibration_cases"]) == 8 and plan["trials"] == 6
    spec = importlib.util.spec_from_file_location("frozen_client", root / "measure_serving.py")
    assert spec is not None and spec.loader is not None
    client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)
    client.BASE = "http://127.0.0.1:" + str(plan["private_port"])

    def snapshot() -> str:
        raw = client.request("/metrics").decode()
        active = [float(line.rsplit(" ", 1)[1]) for line in raw.splitlines()
                  if re.match(r"vllm:num_requests_(running|waiting)(\{| )", line)]
        assert active and not any(active), "Unexpected concurrent request"
        return raw

    path = root / f"{args.arm}-measurement.json"
    assert not path.exists()
    report: dict[str, Any] = {
        "status": "running", "arm": args.arm, "depth": depth, "started_epoch": time.time(),
        "plan_sha256": hashlib.sha256((root / "plan.json").read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": [], "warmups": [], "sealed_test_opened": False, "speed_qualified": False,
    }
    client.write(path, report)
    try:
        for _ in range(2):
            snapshot()
            report["warmups"].append(client.stream(plan["warmup_case"]["prompt_ids"], 256))
        order = list(plan["calibration_cases"])
        rng = random.Random(42)
        for trial in range(plan["trials"]):
            rng.shuffle(order)
            for case in order:
                assert case["split"] == "validation"
                before = snapshot()
                result = client.stream(case["prompt_ids"], 256)
                after = snapshot()
                counters = delta(before, after)
                row = {"case": case["id"], "domain": case["domain"], "trial": trial,
                       "result": result, "counter_delta": counters,
                       "speculation": summarize_counters(counters, depth),
                       "metrics_before": before, "metrics_after": after}
                previous = next((r for r in report["cases"] if r["case"] == case["id"]), None)
                row["repeat_tokens_exact"] = (previous is None or
                    previous["result"]["token_ids"] == result["token_ids"])
                row["repeat_counters_exact"] = previous is None or previous["counter_delta"] == counters
                report["cases"].append(row)
                client.write(path, report)
            print(json.dumps({"arm": args.arm, "depth": depth, "trial": trial,
                              "completed": len(report["cases"])}), flush=True)
        report["summary"] = {}
        for domain in ("code", "chat", "explanation"):
            rows = [r for r in report["cases"] if r["domain"] == domain]
            counts = {k: sum(r["counter_delta"].get(k, 0) for r in rows)
                      for r in rows for k in r["counter_delta"]}
            report["summary"][domain] = {
                "requests": len(rows), **summarize_counters(counts, depth),
                "median_tps": statistics.median(256 / r["result"]["latency_seconds"] for r in rows),
                "median_ttft": statistics.median(r["result"]["ttft_seconds"] for r in rows),
                "median_tpot": statistics.median(r["result"]["tpot_seconds"] for r in rows),
            }
        report.update(status="complete", completed_epoch=time.time(),
                      repeat_tokens_and_counters_exact=all(
                          r["repeat_tokens_exact"] and r["repeat_counters_exact"]
                          for r in report["cases"]))
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        client.write(path, report)


if __name__ == "__main__":
    main()
