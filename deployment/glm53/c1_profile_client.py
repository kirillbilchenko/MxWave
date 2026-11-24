"""Collect bounded C1 controls and native traces without tuning either deployment."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from compare_client import Client, cache_probe, cohort
from control_stock_trial import write


def collect(root: Path, arm: str, client: Client) -> None:
    """Keep warmups, unprofiled controls and profiler requests explicitly separate."""
    plan = json.loads((root / "workloads.json").read_bytes())
    output = root / f"profile-{arm}"
    output.mkdir()
    models = json.loads(client.request("/v1/models"))
    if not any(m.get("max_model_len") == 500000 for m in models["data"]):
        raise RuntimeError("Preserve the 500k deployment settings")
    state: dict[str, Any] = {"arm": arm, "status": "running", "models": models,
                             "quality_qualified": False, "speed_qualified": False,
                             "completed_cells": []}
    write(output / "progress.json", state)
    probe = cache_probe(client, plan["cache_probe_ids"], plan["salt"] + "-probe")
    write(output / "cache-probe.json", probe)
    if not probe["passed"]:
        raise RuntimeError("Prefix-cache control failed; preserve settings and stop")
    for cell in plan["cells"]:
        state.update(cell=cell["name"], phase="warmup")
        write(output / "progress.json", state)
        row: dict[str, Any] = {"cell": cell, "arm": arm, "controls_before": [],
                              "controls_after": [], "profiled_request_is_timing": False}
        path = output / (cell["name"] + ".json")
        salt = plan["salt"] + "-" + cell["name"]

        def request(label: str, *, cell: dict[str, Any] = cell,
                    salt: str = salt) -> dict[str, Any]:
            result = cohort(client, cell["prompts"], 256, salt=salt + "-" + label)
            if result["api_errors"] or result["output_tokens"] != 256:
                write(output / (cell["name"] + "-failed-request.json"), result)
                raise RuntimeError("C1 request failed; keep partial evidence")
            hits = sum(v for k, v in result["counters_delta"].items()
                       if k.startswith("vllm:prefix_cache_hits_total"))
            if hits:
                write(output / (cell["name"] + "-failed-request.json"), result)
                raise RuntimeError("Expected a cold salted prefix")
            return result

        row["warmup"] = request("warmup")
        for index in range(3):
            state["phase"] = "unprofiled_before"
            write(output / "progress.json", state)
            row["controls_before"].append(request(f"before-{index}"))
            write(path, row)
        state["phase"] = "native_profile"
        write(output / "progress.json", state)
        row["profile_start_epoch"] = time.time()
        row["profile_start_response"] = client.request("/start_profile", {}).decode()
        write(path, row)
        try:
            row["profiled_request"] = request("profile")
        finally:
            row["profile_stop_response"] = client.request(
                "/stop_profile", {}, timeout=300).decode()
            row["profile_stop_epoch"] = time.time()
            write(path, row)
        state["phase"] = "unprofiled_after"
        write(output / "progress.json", state)
        for index in range(2):
            row["controls_after"].append(request(f"after-{index}"))
            write(path, row)
        state["completed_cells"].append(cell["name"])
        write(output / "progress.json", state)
    state.update(status="complete", phase="complete", finished_epoch=time.time())
    write(output / "progress.json", state)


def main() -> None:
    """Match the existing restoration controller's client interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=("A", "B"))
    parser.add_argument("--port", type=int)
    parser.add_argument("--mode", choices=("timing", "summary"), required=True)
    parser.add_argument("--trial", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "summary":
        return
    if args.arm is None or args.port is None:
        parser.error("Collection requires arm and port")
    collect(args.root, args.arm, Client(args.port))


if __name__ == "__main__":
    main()
